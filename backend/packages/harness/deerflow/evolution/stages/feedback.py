"""Feedback stage: gate verdict, close-out, memory write, promotion.

:class:`FeedbackSynthesizerImpl` implements the loop's
``FeedbackSynthesizer`` protocol:

1. Reads the significance gate (spark ``significance_gate``; skipped when
   execution itself failed) and calls the summarize-profile LLM (static
   versioned templates) for the decision + reason + observations +
   hypothesis evaluation.
2. Closes out the trial governor first (spark ``log_outcome``), then the
   KB (``experiment_commit``, plus ``failure_record`` on failures and
   ``assumption_record`` for newly stated assumptions).
3. Writes one compact turn summary to agent memory.
4. Promotes the belief (spark ``memory_promote``) only when the gate passed
   and the summary confidence meets ``min_confidence_to_promote``; a
   ``pending_human`` answer is accepted without escalation — nothing in
   this path prompts, queues, or waits for approval.

An execution error always yields ``decision=False`` with the error captured
verbatim, mirroring the loop's fallback chain.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from deerflow.evolution.config import EvolutionConfig
from deerflow.evolution.errors import EvolutionError
from deerflow.evolution.knowledge_adapter import KnowledgeAdapter
from deerflow.evolution.loop import (
    COMMIT_SEQUENCE_BASE,
    FAILURE_SEQUENCE_BASE,
    OUTCOME_SEQUENCE_BASE,
    PROMOTE_SEQUENCE_BASE,
    StageContext,
    derive_key,
)
from deerflow.evolution.memory_adapter import MemoryAdapter
from deerflow.evolution.models import (
    CodedExperiment,
    ExperimentFeedback,
    ExperimentOutcome,
    ExperimentSpec,
    Hypothesis,
    HypothesisFeedback,
)
from deerflow.evolution.prompts import SUMMARIZE_SYSTEM, SUMMARIZE_USER
from deerflow.evolution.spark_client import SparkClient
from deerflow.evolution.stages.llm import ChatModelFactory, parse_json_object
from deerflow.evolution.stages.state import IterationState, format_lineage
from deerflow.knowledge.write_api import ASSUMPTION_CATEGORIES

__all__ = [
    "FeedbackSynthesizerImpl",
    "failure_class_for",
]

_FAILURE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("data", "data"),
    ("statistical", "statistical"),
    ("code", "code"),
    ("sandbox", "execution"),
    ("execution", "execution"),
    ("timeout", "execution"),
    ("hypothesis", "hypothesis"),
    ("error", "code"),
)


def failure_class_for(error_type: str | None, error_text: str | None = None) -> str:
    """Map an execution error to a KB failure class (default ``execution``)."""
    haystack = f"{error_type or ''} {error_text or ''}".lower()
    for keyword, failure_class in _FAILURE_KEYWORDS:
        if keyword in haystack:
            return failure_class
    return "execution"


def _confidence_value(payload: Mapping[str, Any]) -> float:
    """Extract the summary confidence in [0, 1] (0.0 when absent/invalid)."""
    confidence = payload.get("confidence", 0.0)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return 0.0
    return min(1.0, max(0.0, float(confidence)))


def _optional_text(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise EvolutionError(f"The summarize model field {key!r} must be a string or null.")
    stripped = value.strip()
    return stripped or None


class FeedbackSynthesizerImpl:
    """Loop ``FeedbackSynthesizer`` bound to spark + KB + memory + profiles.

    Args:
        knowledge: KB binding for commit/failure/assumption writes.
        spark: Trial-governance client for gate/log_outcome/promote.
        memory: Agent-memory binding for the turn summary.
        models: Role-addressed chat model factory.
        config: Evolution config (promotion threshold, family default).
        state: Shared per-iteration sidecar (episode/experiment ids).
    """

    def __init__(
        self,
        *,
        knowledge: KnowledgeAdapter,
        spark: SparkClient,
        memory: MemoryAdapter,
        models: ChatModelFactory,
        config: EvolutionConfig,
        state: IterationState | None = None,
    ) -> None:
        self._knowledge = knowledge
        self._spark = spark
        self._memory = memory
        self._models = models
        self._config = config
        self._state = state or IterationState()

    async def synthesize(
        self,
        *,
        hypothesis: Hypothesis,
        spec: ExperimentSpec,
        coded: CodedExperiment | None,
        outcome: ExperimentOutcome,
        ctx: StageContext,
    ) -> ExperimentFeedback:
        """Gate, summarize, and close out one executed experiment."""
        _ = (spec, coded)
        episode_id = self._state.require_episode_id(ctx.run_id, ctx.iteration)
        experiment_id = self._state.require_experiment_id(ctx.run_id, ctx.iteration)
        if outcome.failed:
            gate_text = f"gate skipped: execution failed ({outcome.error_type or 'unknown error'})"
            gate_passed = False
        else:
            verdict = await self._spark.significance_gate(episode_id)
            gate_passed = verdict.passed
            gate_text = f"status={verdict.status} pass={verdict.passed} sr={verdict.sr} dsr={verdict.dsr} p={verdict.p_value} M_family={verdict.m_family} blockers={','.join(verdict.promotion_blockers) or 'none'}"
        user_prompt = SUMMARIZE_USER.format(
            hypothesis=hypothesis.hypothesis,
            gate_verdict=gate_text,
            metrics=json.dumps(outcome.metrics, sort_keys=True, default=str),
            error=outcome.error or "None",
            lineage=format_lineage(ctx.trace),
        )
        raw = await self._models.generate(role="summarize", system=SUMMARIZE_SYSTEM, user=user_prompt)
        payload = parse_json_object(raw, role="summarize")
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise EvolutionError("The summarize model must return a non-empty 'reason' string.")
        decision = bool(payload.get("decision", False))
        if outcome.failed:
            decision = False
        confidence = _confidence_value(payload)
        kb_outcome = "failure" if outcome.failed else ("success" if decision else "inconclusive")
        failure_class = failure_class_for(outcome.error_type, outcome.error) if outcome.failed else None
        # Ordering: governor first, KB second — close-out mirrors registration.
        log_key = derive_key(
            ctx.run_id,
            OUTCOME_SEQUENCE_BASE,
            ctx.iteration,
            {"episode_id": episode_id, "outcome": kb_outcome, "metrics": outcome.metrics},
        )
        await self._spark.log_outcome(
            episode_id=episode_id,
            outcome=kb_outcome,
            metrics=dict(outcome.metrics),
            failure_reason=outcome.error[:1000] if outcome.failed and outcome.error else None,
            reward=1.0 if decision else 0.0,
            client_id=log_key,
        )
        if outcome.failed:
            assert failure_class is not None
            failure_key = derive_key(
                ctx.run_id,
                FAILURE_SEQUENCE_BASE,
                ctx.iteration,
                {"experiment_id": experiment_id, "failure_class": failure_class, "error": outcome.error},
            )
            notes = f"{outcome.error_type or 'error'}: {outcome.error or ''}".strip()[:4000]
            await self._knowledge.record_failure(
                experiment_id=experiment_id,
                failure_class=failure_class,
                notes=notes,
                idempotency_key=failure_key,
            )
        commit_key = derive_key(
            ctx.run_id,
            COMMIT_SEQUENCE_BASE,
            ctx.iteration,
            {
                "experiment_id": experiment_id,
                "outcome": kb_outcome,
                "metrics": outcome.metrics,
                "artifacts": list(outcome.artifacts),
            },
        )
        await self._knowledge.commit_experiment(
            experiment_id=experiment_id,
            metrics=dict(outcome.metrics),
            outcome=kb_outcome,
            failure_class=failure_class,
            result_artifacts=list(outcome.artifacts),
            idempotency_key=commit_key,
        )
        await self._record_assumptions(payload, experiment_id=experiment_id, ctx=ctx)
        await self._memory.record_turn(
            thread_id=ctx.run_id,
            user_text=f"evolution run {ctx.run_id} iteration {ctx.iteration}: {hypothesis.hypothesis}",
            assistant_text=f"decision={decision} gate_passed={gate_passed} outcome={kb_outcome} confidence={confidence:.2f} reason={reason.strip()}",
            trace_id=ctx.step_id,
        )
        if MemoryAdapter.should_promote(
            gate_passed=gate_passed and not outcome.failed,
            confidence=confidence,
            min_confidence=self._config.min_confidence_to_promote,
        ):
            promote_key = derive_key(
                ctx.run_id,
                PROMOTE_SEQUENCE_BASE,
                ctx.iteration,
                {"episode_ids": [episode_id], "claim": hypothesis.hypothesis, "confidence": confidence},
            )
            scope = self._state.get_scope(ctx.run_id, ctx.iteration) or self._config.resolve_default_family()
            await self._spark.memory_promote(
                episode_ids=[episode_id],
                claim=hypothesis.hypothesis,
                confidence=confidence,
                scope=scope,
                rationale=reason.strip()[:2000],
                client_id=promote_key,
            )
        return HypothesisFeedback(
            decision=decision,
            reason=reason.strip(),
            observations=_optional_text(payload, "observations"),
            hypothesis_evaluation=_optional_text(payload, "hypothesis_evaluation"),
            new_hypothesis=_optional_text(payload, "new_hypothesis"),
            acceptable=decision,
        )

    async def _record_assumptions(self, payload: Mapping[str, Any], *, experiment_id: str, ctx: StageContext) -> None:
        """Record well-formed stated assumptions; skip malformed entries."""
        assumptions = payload.get("assumptions", [])
        if not isinstance(assumptions, list):
            return None
        index = 0
        for entry in assumptions:
            if not isinstance(entry, Mapping):
                continue
            statement = entry.get("statement")
            category = entry.get("category")
            if not isinstance(statement, str) or not statement.strip():
                continue
            if category not in ASSUMPTION_CATEGORIES:
                continue
            key = derive_key(
                ctx.run_id,
                FAILURE_SEQUENCE_BASE,
                ctx.iteration,
                {"assumption_record": {"index": index, "experiment_id": experiment_id, "statement": statement.strip(), "category": category}},
            )
            await self._knowledge.record_assumption(
                statement=statement.strip(),
                category=category,
                experiment_id=experiment_id,
                idempotency_key=key,
            )
            index += 1
        return None

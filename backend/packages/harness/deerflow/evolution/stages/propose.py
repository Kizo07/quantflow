"""Propose stage: hypothesis generation with C6 validation.

:class:`HypothesisGenerator` implements the loop's ``Proposer`` protocol:

1. Builds the proposal context: KB retrieval packet
   (:meth:`KnowledgeAdapter.build_proposal_context`), agent-memory recall
   (:meth:`MemoryAdapter.recall_context`), spark ``memory_recall`` beliefs,
   and spark ``suggest_next`` underexplored-region guidance, plus the trace
   lineage.
2. Calls the propose-profile LLM (static versioned templates from
   :mod:`deerflow.evolution.prompts`) for exactly one hypothesis.
3. Validates the C6 field set locally (mechanism, expected direction,
   horizon, falsifier, scope, predicted IC/sigma; DAG + adjustment set iff
   causal) — the same rule the governor enforces — so malformed proposals
   fail before any write.
4. Runs the advisory ``check_novelty`` read, then ``hypothesis_propose``
   with the derived idempotency key.

The spark ``hypothesis_id`` is recorded in the shared
:class:`IterationState` for the design stage; the returned
:class:`Hypothesis` also carries the id and packet refs in
``concise_knowledge`` for lineage readability.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from deerflow.evolution.config import EvolutionConfig
from deerflow.evolution.errors import EvolutionError
from deerflow.evolution.knowledge_adapter import KnowledgeAdapter
from deerflow.evolution.loop import PROPOSE_SEQUENCE_BASE, StageContext, derive_key
from deerflow.evolution.memory_adapter import MemoryAdapter
from deerflow.evolution.models import Hypothesis
from deerflow.evolution.prompts import PROPOSE_SYSTEM, PROPOSE_USER
from deerflow.evolution.spark_client import SparkClient
from deerflow.evolution.stages.llm import ChatModelFactory, parse_json_object
from deerflow.evolution.stages.state import IterationState, format_lineage

__all__ = [
    "C6_HYPOTHESIS_TYPES",
    "C6Payload",
    "ProposalContext",
    "HypothesisGenerator",
    "validate_c6",
]

logger = logging.getLogger(__name__)

C6_HYPOTHESIS_TYPES = frozenset({"causal", "descriptive", "execution", "portfolio", "predictive", "risk"})


@dataclass(frozen=True)
class C6Payload:
    """Validated C6 hypothesis fields (governor field rule)."""

    mechanism: str
    expected_direction: str
    horizon: str
    falsifier: str
    scope: str
    predicted_ic: float
    predicted_sigma: float
    hypothesis_type: str
    dag: dict[str, Any] | None = None
    adjustment_set: tuple[str, ...] | None = None

    def to_canonical(self) -> dict[str, Any]:
        """Return the canonical payload bound to the propose idempotency key."""
        payload: dict[str, Any] = {
            "mechanism": self.mechanism,
            "expected_direction": self.expected_direction,
            "horizon": self.horizon,
            "falsifier": self.falsifier,
            "scope": self.scope,
            "predicted_ic": self.predicted_ic,
            "predicted_sigma": self.predicted_sigma,
            "hypothesis_type": self.hypothesis_type,
        }
        if self.dag is not None:
            payload["dag"] = self.dag
        if self.adjustment_set is not None:
            payload["adjustment_set"] = list(self.adjustment_set)
        return payload


@dataclass(frozen=True)
class ProposalContext:
    """The assembled inputs behind one proposal (observability sidecar)."""

    topic: str
    packet_text: str
    memory_text: str
    governor_guidance: str
    lineage: str
    packet_refs: tuple[str, ...] = ()
    novelty: float | None = None


def _require_text(payload: Mapping[str, Any], key: str, missing: list[str]) -> str | None:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        missing.append(key)
        return None
    return value.strip()


def _require_number(payload: Mapping[str, Any], key: str, missing: list[str]) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        missing.append(key)
        return None
    return float(value)


def validate_c6(payload: Mapping[str, Any]) -> C6Payload:
    """Validate the C6 field set; raise ``ValueError`` naming gaps.

    Core fields are required for every hypothesis type; ``dag`` plus
    ``adjustment_set`` are additionally required iff the type is causal.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"C6 validation needs a mapping, got {type(payload).__name__}.")
    missing: list[str] = []
    mechanism = _require_text(payload, "mechanism", missing)
    expected_direction = _require_text(payload, "expected_direction", missing)
    horizon = _require_text(payload, "horizon", missing)
    falsifier = _require_text(payload, "falsifier", missing)
    scope = _require_text(payload, "scope", missing)
    predicted_ic = _require_number(payload, "predicted_ic", missing)
    predicted_sigma = _require_number(payload, "predicted_sigma", missing)
    hypothesis_type_raw = payload.get("hypothesis_type")
    hypothesis_type = hypothesis_type_raw.strip() if isinstance(hypothesis_type_raw, str) else ""
    if hypothesis_type not in C6_HYPOTHESIS_TYPES:
        raise ValueError(f"Unknown hypothesis_type {hypothesis_type_raw!r}; expected one of {sorted(C6_HYPOTHESIS_TYPES)}.")
    dag: dict[str, Any] | None = None
    adjustment_set: tuple[str, ...] | None = None
    if hypothesis_type == "causal":
        raw_dag = payload.get("dag")
        if not isinstance(raw_dag, Mapping) or not raw_dag:
            missing.append("dag")
        else:
            dag = dict(raw_dag)
        raw_adjustment = payload.get("adjustment_set")
        if not isinstance(raw_adjustment, (list, tuple)) or not all(isinstance(item, str) and item.strip() for item in raw_adjustment) or not raw_adjustment:
            missing.append("adjustment_set")
        else:
            adjustment_set = tuple(item.strip() for item in raw_adjustment)
    if missing:
        raise ValueError("hypothesis refused (C6): missing required fields: " + ", ".join(sorted(set(missing))))
    assert mechanism is not None and expected_direction is not None
    assert horizon is not None and falsifier is not None and scope is not None
    assert predicted_ic is not None and predicted_sigma is not None
    return C6Payload(
        mechanism=mechanism,
        expected_direction=expected_direction,
        horizon=horizon,
        falsifier=falsifier,
        scope=scope,
        predicted_ic=predicted_ic,
        predicted_sigma=predicted_sigma,
        hypothesis_type=hypothesis_type,
        dag=dag,
        adjustment_set=adjustment_set,
    )


class HypothesisGenerator:
    """Loop ``Proposer`` bound to KB + spark + memory + model profiles.

    Args:
        knowledge: KB retrieval binding for proposal context.
        spark: Trial-governance client.
        memory: Agent-memory binding for proposal context.
        models: Role-addressed chat model factory.
        config: Evolution config (family/universe defaults).
        state: Shared per-iteration sidecar (fresh instance when omitted;
            pass one shared instance across stages in production).
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
        self._last_context: ProposalContext | None = None

    @property
    def last_context(self) -> ProposalContext | None:
        """The context assembled for the most recent proposal, if any."""
        return self._last_context

    async def propose(self, ctx: StageContext) -> Hypothesis:
        """Propose, validate, and register the next hypothesis."""
        family = self._config.resolve_default_family()
        universe = self._config.resolve_default_universe()
        parents = ctx.trace.get_parent_steps()
        if parents and parents[-1].hypothesis is not None:
            topic = parents[-1].hypothesis.hypothesis
        else:
            topic = f"{family} / {universe} run start"
        packet = await self._knowledge.build_proposal_context(topic)
        memory_text = await self._memory.recall_context(topic)
        beliefs = await self._spark.memory_recall(topic)
        suggestions = await self._spark.suggest_next()
        guidance_lines = [f"- belief (conf={belief.effective_confidence}): {belief.claim}" for belief in beliefs]
        for suggestion in suggestions.suggestions:
            label = suggestion.get("family") or suggestion.get("label") or json.dumps(suggestion, sort_keys=True, default=str)
            guidance_lines.append(f"- underexplored: {label}")
        governor_guidance = "\n".join(guidance_lines) if guidance_lines else "(no governor guidance)"
        lineage = format_lineage(ctx.trace)
        user_prompt = PROPOSE_USER.format(
            family=family,
            universe=universe,
            iteration=ctx.iteration,
            knowledge_packet=packet.text or "(no retrieved knowledge)",
            memory_context=memory_text or "(no agent memory)",
            governor_guidance=governor_guidance,
            lineage=lineage,
        )
        raw = await self._models.generate(role="propose", system=PROPOSE_SYSTEM, user=user_prompt)
        payload = parse_json_object(raw, role="propose")
        hypothesis_text = payload.get("hypothesis")
        reason_text = payload.get("reason")
        if not isinstance(hypothesis_text, str) or not hypothesis_text.strip():
            raise EvolutionError("The propose model must return a non-empty 'hypothesis' string.")
        if not isinstance(reason_text, str) or not reason_text.strip():
            raise EvolutionError("The propose model must return a non-empty 'reason' string.")
        c6 = validate_c6(payload)
        try:
            novelty = await self._spark.check_novelty(
                hypothesis_text.strip(),
                family=family,
                predicted_ic=c6.predicted_ic,
                predicted_sigma=c6.predicted_sigma,
            )
            novelty_value = novelty.novelty
        except Exception:
            logger.debug("check_novelty advisory read failed; proposing anyway.", exc_info=True)
            novelty_value = None
        parent_ids = [self._state.get_hypothesis_id(ctx.run_id, step.iteration) for step in parents]
        parent_hypothesis_ids = [value for value in parent_ids if value]
        client_id = derive_key(ctx.run_id, PROPOSE_SEQUENCE_BASE, ctx.iteration, c6.to_canonical())
        result = await self._spark.propose(
            idea=hypothesis_text.strip(),
            family=family,
            universe=universe,
            horizon=c6.horizon,
            scope=c6.scope,
            mechanism=c6.mechanism,
            expected_direction=c6.expected_direction,
            falsifier=c6.falsifier,
            predicted_ic=c6.predicted_ic,
            predicted_sigma=c6.predicted_sigma,
            hypothesis_type=c6.hypothesis_type,
            dag=c6.dag,
            adjustment_set=list(c6.adjustment_set) if c6.adjustment_set is not None else None,
            parent_hypothesis_ids=parent_hypothesis_ids or None,
            client_id=client_id,
        )
        self._state.record_proposal(ctx.run_id, ctx.iteration, hypothesis_id=result.hypothesis_id, scope=c6.scope)
        self._last_context = ProposalContext(
            topic=topic,
            packet_text=packet.text,
            memory_text=memory_text,
            governor_guidance=governor_guidance,
            lineage=lineage,
            packet_refs=packet.refs,
            novelty=novelty_value,
        )
        concise_bits = [f"hypothesis_id={result.hypothesis_id}", f"type={c6.hypothesis_type}", f"scope={c6.scope}"]
        if novelty_value is not None:
            concise_bits.append(f"N={novelty_value:.3f}")
        if packet.refs:
            concise_bits.append("packet_refs=" + ",".join(packet.refs[:8]))
        return Hypothesis(
            hypothesis=hypothesis_text.strip(),
            reason=reason_text.strip(),
            concise_knowledge="; ".join(concise_bits),
        )

"""Binds the evolution loop to the Knowledge Plane.

This adapter owns the loop's Knowledge Plane surface:

* Writes: ``experiment_begin`` / ``experiment_commit`` / ``failure_record``
  / ``assumption_record`` over a caller-supplied
  :class:`deerflow.knowledge.write_api.ExperimentStore`.
* Pre-experiment recheck: ``experiment_search`` (family equivalents) plus
  the execution-hash exact-rerun lookup, before any compute.
* Proposal context: the Phase-2 retrieval path (``intent_from_text`` →
  ``plan_retrieval`` → ``execute_retrieval`` → ``build_context_packet``)
  over caller-supplied channel stores, rendered section by section from
  the packet object.

The adapter performs no I/O itself: every store is caller-supplied, and the
synchronous store calls run in worker threads so the async loop never
blocks. It invents no identity: content hashes come from the experiment
spec (KB hashing semantics), and idempotency keys arrive from the caller
(derived per iteration by the stages).

Ordering (plan contract): spark ``register_hypothesis`` precedes
``experiment_begin`` (design), and spark ``log_outcome`` precedes
``experiment_commit`` (feedback) — the stages enforce this; the adapter only
provides the KB half.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from deerflow.evolution.errors import DuplicateRunError, EvolutionError
from deerflow.evolution.models import ExperimentSpec
from deerflow.knowledge.hashing import execution_hash, experiment_family_hash, normalize_text
from deerflow.knowledge.retrieval.packet import SECTION_TITLES, ContextPacket, build_context_packet
from deerflow.knowledge.retrieval.planner import (
    FailureSearchStore,
    LexicalSearchStore,
    RetrievalPlan,
    StructuredLookupStore,
    VectorSearchStore,
    execute_retrieval,
    plan_retrieval,
)
from deerflow.knowledge.search import ExperimentSearchStore, experiment_search
from deerflow.knowledge.write_api import (
    AssumptionRecord,
    DuplicateExecutionError,
    ExperimentRecord,
    ExperimentStore,
    FailureRecord,
    assumption_record,
    experiment_begin,
    experiment_commit,
    failure_record,
)

__all__ = [
    "RetrievalChannels",
    "ProposalPacket",
    "RecheckResult",
    "KnowledgeAdapter",
]


@dataclass(frozen=True)
class RetrievalChannels:
    """Caller-supplied retrieval channel stores for proposal context."""

    structured: StructuredLookupStore
    lexical: LexicalSearchStore
    vector: VectorSearchStore
    failures: FailureSearchStore


@dataclass(frozen=True)
class ProposalPacket:
    """Rendered retrieval packet plus the evidence refs behind it."""

    text: str
    refs: tuple[str, ...] = ()
    intent_topic: str = ""


@dataclass(frozen=True)
class RecheckResult:
    """Outcome of the pre-experiment recheck for one spec."""

    execution_hash: str
    family_hash: str
    exact_match_id: str | None = None
    family_match_ids: tuple[str, ...] = ()


def _require_uuid(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a UUID string, got {type(value).__name__}.")
    try:
        return str(uuid.UUID(value.strip()))
    except (ValueError, AttributeError):
        raise ValueError(f"{name} must be a valid UUID, got {value!r}.") from None


def _partition_artifact_refs(refs: list[str] | None) -> tuple[list[str], list[str]]:
    """Split artifact refs into (KB artifact UUIDs, sandbox path refs).

    The KB stores only UUID artifact ids; harvested sandbox paths (e.g.
    ``outputs/metrics.json``) are path refs the commit carries separately.
    """
    artifact_ids: list[str] = []
    path_refs: list[str] = []
    for ref in refs or []:
        try:
            artifact_ids.append(str(uuid.UUID(str(ref).strip())))
        except (ValueError, AttributeError):
            path_refs.append(ref)
    return artifact_ids, path_refs


@dataclass(frozen=True)
class _BeginEnvelopes:
    """Spec reduced to exactly the envelopes ``experiment_begin`` hashes."""

    hypothesis: str
    methodology: dict[str, Any]
    datasets: list[dict[str, Any]]
    code: dict[str, Any]
    parameters: dict[str, Any]
    environment: dict[str, Any]


def _split_spec(spec: ExperimentSpec) -> _BeginEnvelopes:
    """Reduce *spec* to the KB write envelopes (single normalization point).

    The KB hashes ``design = {hypothesis, methodology}`` (normalized text,
    no extra keys) and ``data = {datasets: [...]}``; the recheck must hash
    the same envelopes or exact-rerun detection silently misses.
    """
    design = spec.design
    hypothesis_text = design.get("hypothesis")
    if not isinstance(hypothesis_text, str) or not hypothesis_text.strip():
        raise EvolutionError("Cannot begin an experiment whose design carries no hypothesis text.")
    methodology = design.get("methodology", {})
    if not isinstance(methodology, Mapping):
        raise EvolutionError("Cannot begin an experiment whose design methodology is not a mapping.")
    datasets = spec.data.get("datasets", [])
    if not isinstance(datasets, list):
        raise EvolutionError("Cannot begin an experiment whose data.datasets is not a list.")
    return _BeginEnvelopes(
        hypothesis=hypothesis_text,
        methodology=dict(methodology),
        datasets=list(datasets),
        code=dict(spec.code),
        parameters=dict(spec.parameters),
        environment=dict(spec.environment),
    )


def _kb_hashes(envelopes: _BeginEnvelopes) -> tuple[str, str]:
    """Return (execution_hash, family_hash) exactly as the KB computes them."""
    design = {"hypothesis": normalize_text(envelopes.hypothesis), "methodology": envelopes.methodology}
    return (
        execution_hash(
            design=design,
            code=envelopes.code,
            data={"datasets": envelopes.datasets},
            parameters=envelopes.parameters,
            environment=envelopes.environment,
        ),
        experiment_family_hash(design),
    )


class KnowledgeAdapter:
    """Knowledge Plane binding for the evolution loop stages.

    Args:
        store: Write boundary for experiments/failures/assumptions.
        project_id: Owning research project UUID for new experiments.
        search_store: Read boundary for the design recheck. When None, the
            recheck serves exact-rerun lookups from *store* and reports no
            family equivalents.
        retrieval: Channel stores for proposal-context retrieval. When None,
            :meth:`build_proposal_context` returns an empty packet (the
            propose stage still runs on memory + governor guidance).
    """

    def __init__(
        self,
        store: ExperimentStore,
        *,
        project_id: str,
        search_store: ExperimentSearchStore | None = None,
        retrieval: RetrievalChannels | None = None,
    ) -> None:
        self._store = store
        self._project_id = _require_uuid("project_id", project_id)
        self._search_store = search_store
        self._retrieval = retrieval

    @property
    def project_id(self) -> str:
        """The owning research project UUID for new experiments."""
        return self._project_id

    async def recheck(self, spec: ExperimentSpec, *, family_limit: int = 10) -> RecheckResult:
        """Run the pre-experiment recheck for *spec* (pure reads).

        Returns the exact-rerun match (if any) plus family-equivalent ids.
        Raises :class:`DuplicateRunError` when the exact computation already
        ran, so the caller links to the prior run instead of rerunning it.

        Hashes are computed over the KB write envelopes (see
        :func:`_split_spec`), not over the raw spec mapping, so the lookup
        matches what ``experiment_begin`` stored.
        """
        execution_hash_value, family_hash_value = _kb_hashes(_split_spec(spec))
        exact = await asyncio.to_thread(self._store.find_experiment_by_execution_hash, execution_hash_value)
        if exact is not None:
            raise DuplicateRunError(execution_hash_value, exact.id)
        family_ids: tuple[str, ...] = ()
        if self._search_store is not None:
            page = await asyncio.to_thread(
                experiment_search,
                self._search_store,
                family_hash=family_hash_value,
                limit=family_limit,
            )
            family_ids = tuple(record.id for record in page.experiments)
        return RecheckResult(
            execution_hash=execution_hash_value,
            family_hash=family_hash_value,
            exact_match_id=None,
            family_match_ids=family_ids,
        )

    async def begin_experiment(
        self,
        spec: ExperimentSpec,
        *,
        run_id: str,
        parent_experiment_id: str | None = None,
        status: str = "running",
        idempotency_key: str | None = None,
    ) -> ExperimentRecord:
        """Register a new experiment; replay-safe on *idempotency_key*.

        Raises :class:`DuplicateRunError` (mapped from the KB error) when
        the exact computation already ran.
        """
        envelopes = _split_spec(spec)
        try:
            return await asyncio.to_thread(
                experiment_begin,
                self._store,
                project_id=self._project_id,
                hypothesis=envelopes.hypothesis,
                methodology=envelopes.methodology,
                parameters=envelopes.parameters,
                datasets=envelopes.datasets,
                code=envelopes.code,
                environment=envelopes.environment,
                created_by_run_id=run_id,
                parent_experiment_id=parent_experiment_id,
                status=status,
                idempotency_key=idempotency_key,
            )
        except DuplicateExecutionError as exc:
            raise DuplicateRunError(exc.execution_hash, exc.existing_experiment_id) from exc

    async def commit_experiment(
        self,
        *,
        experiment_id: str,
        metrics: Mapping[str, Any],
        outcome: str,
        failure_class: str | None = None,
        result_artifacts: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> ExperimentRecord:
        """Commit terminal results; replay-safe on *idempotency_key*.

        Non-UUID artifact refs (harvested sandbox paths) cannot be stored
        as KB result artifacts, so they ride into the commit on
        ``metrics["sandbox_artifacts"]`` instead of failing validation.
        """
        artifact_ids, path_refs = _partition_artifact_refs(result_artifacts)
        merged_metrics = dict(metrics)
        if path_refs:
            existing = merged_metrics.get("sandbox_artifacts")
            if isinstance(existing, list):
                merged_metrics["sandbox_artifacts"] = [*existing, *path_refs]
            else:
                merged_metrics["sandbox_artifacts"] = list(path_refs)
        return await asyncio.to_thread(
            experiment_commit,
            self._store,
            experiment_id=experiment_id,
            metrics=merged_metrics,
            outcome=outcome,
            failure_class=failure_class,
            result_artifacts=artifact_ids,
            idempotency_key=idempotency_key,
        )

    async def record_failure(
        self,
        *,
        experiment_id: str,
        failure_class: str,
        notes: str = "",
        idempotency_key: str | None = None,
    ) -> FailureRecord:
        """Record a failure detail; replay-safe on *idempotency_key*."""
        return await asyncio.to_thread(
            failure_record,
            self._store,
            experiment_id=experiment_id,
            failure_class=failure_class,
            notes=notes,
            idempotency_key=idempotency_key,
        )

    async def record_assumption(
        self,
        *,
        statement: str,
        category: str,
        experiment_id: str | None = None,
        sensitivity: str = "unknown",
        tested: bool = False,
        status: str = "active",
        idempotency_key: str | None = None,
    ) -> AssumptionRecord:
        """Record an assumption; replay-safe on *idempotency_key*."""
        return await asyncio.to_thread(
            assumption_record,
            self._store,
            statement=statement,
            category=category,
            experiment_id=experiment_id,
            sensitivity=sensitivity,
            tested=tested,
            status=status,
            idempotency_key=idempotency_key,
        )

    async def build_proposal_context(self, topic: str, *, query_family_hash: str | None = None) -> ProposalPacket:
        """Build the retrieval packet for a proposal topic.

        Runs ``intent_from_text`` → ``plan_retrieval`` → ``execute_retrieval``
        → ``build_context_packet`` and renders the packet to text. The vector
        channel runs without a query embedding (lexical/structured/failure
        channels carry the recall). Returns an empty packet when no
        retrieval channels are bound.
        """
        if not isinstance(topic, str) or not topic.strip():
            raise EvolutionError("Proposal context needs a non-empty topic.")
        if self._retrieval is None:
            return ProposalPacket(text="", refs=(), intent_topic=topic.strip())
        # Imported lazily: the bootstrap intent helper pulls the langchain
        # tool dependency, which must never load at evolution import time.
        from deerflow.knowledge.tools.bootstrap import intent_from_text

        channels = self._retrieval
        intent = await asyncio.to_thread(intent_from_text, topic)
        # The bootstrap intent is a sibling type of the planner intent with
        # the same field shape; the planner accepts its mapping form.
        plan: RetrievalPlan = await asyncio.to_thread(plan_retrieval, intent.to_dict())
        result = await asyncio.to_thread(
            execute_retrieval,
            plan,
            query_text=topic,
            structured=channels.structured,
            lexical=channels.lexical,
            vector=channels.vector,
            failures=channels.failures,
            query_family_hash=query_family_hash,
        )
        packet = await asyncio.to_thread(build_context_packet, result.fusion)
        text = _render_packet(packet, intent.topic)
        refs = tuple(candidate.candidate.id for candidate in result.top if candidate.candidate.id)
        return ProposalPacket(text=text, refs=refs, intent_topic=intent.topic)


def _render_packet(packet: ContextPacket, topic: str) -> str:
    """Render one packet as model-facing text (one block per section)."""
    lines = ["RESEARCH CONTEXT", f"topic: {topic}", ""]
    for section in packet.sections:
        title = SECTION_TITLES.get(section.name, section.name)
        lines.append(f"{title}:")
        if not section.items:
            lines.append("  - (none retrieved)")
        for item in section.items:
            lines.append(f"  {item.to_text()}")
        if section.truncated:
            lines.append("  (section truncated to budget)")
    for warning in packet.warnings:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)

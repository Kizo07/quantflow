"""Unit tests for deerflow.evolution.knowledge_adapter (offline).

In-memory fakes stand in for the KB stores; the real write_api / search /
retrieval functions run against them, so these tests prove the adapter's
binding (not just its delegation).
"""

import uuid

import pytest

from deerflow.evolution.errors import DuplicateRunError, EvolutionError
from deerflow.evolution.knowledge_adapter import KnowledgeAdapter, RetrievalChannels
from deerflow.evolution.models import ExperimentSpec
from deerflow.knowledge.hashing import execution_hash, experiment_family_hash, normalize_text
from deerflow.knowledge.retrieval.planner import Candidate
from deerflow.knowledge.write_api import (
    AssumptionRecord,
    ExperimentRecord,
    FailureRecord,
)

PROJECT_ID = "22222222-2222-2222-2222-222222222222"
RUN_ID = "33333333-3333-3333-3333-333333333333"


def _spec(hypothesis: str = "momentum persists", seed: int = 0) -> ExperimentSpec:
    return ExperimentSpec(
        design={"hypothesis": hypothesis, "methodology": {"universe": "sp500"}},
        code={"artifact": "code-v1"},
        data={},
        parameters={"seed": seed},
        environment={"image": "qf:v1"},
    )


class InMemoryExperimentStore:
    """Minimal ExperimentStore fake (write_api boundary)."""

    def __init__(self) -> None:
        self.records: dict[str, ExperimentRecord] = {}
        self.failures: dict[str, FailureRecord] = {}
        self.assumptions: dict[str, AssumptionRecord] = {}

    def find_experiment_by_idempotency_key(self, idempotency_key: str):
        return next((r for r in self.records.values() if r.idempotency_key == idempotency_key), None)

    def find_experiment_by_commit_key(self, idempotency_key: str):
        return next((r for r in self.records.values() if r.commit_idempotency_key == idempotency_key), None)

    def find_experiment_by_execution_hash(self, execution_hash: str):
        return next((r for r in self.records.values() if r.execution_hash == execution_hash), None)

    def get_experiment(self, experiment_id: str):
        return self.records.get(experiment_id)

    def insert_experiment(self, record: ExperimentRecord) -> None:
        self.records[record.id] = record

    def update_experiment(self, record: ExperimentRecord) -> None:
        self.records[record.id] = record

    def find_failure_by_idempotency_key(self, idempotency_key: str):
        return next((r for r in self.failures.values() if r.idempotency_key == idempotency_key), None)

    def insert_failure(self, record: FailureRecord) -> None:
        self.failures[record.id] = record

    def find_assumption_by_idempotency_key(self, idempotency_key: str):
        return next((r for r in self.assumptions.values() if r.idempotency_key == idempotency_key), None)

    def insert_assumption(self, record: AssumptionRecord) -> None:
        self.assumptions[record.id] = record


class InMemorySearchStore:
    """Minimal ExperimentSearchStore fake backed by one experiment list."""

    def __init__(self, experiments: list[ExperimentRecord] | None = None):
        self.experiments = list(experiments or [])

    def find_by_execution_hash(self, execution_hash: str):
        return next((r for r in self.experiments if r.execution_hash == execution_hash), None)

    def find_by_family_hash(self, family_hash: str, *, limit: int, offset: int):
        return [r for r in self.experiments if r.family_hash == family_hash][offset : offset + limit]

    def search_experiments(self, filters, *, limit: int, offset: int):
        matched = [r for r in self.experiments if filters.family_hash is None or r.family_hash == filters.family_hash]
        return matched[offset : offset + limit]


class ChannelFake:
    """Retrieval channel fake serving fixed candidates per channel."""

    def __init__(self, candidates: list[Candidate] | None = None):
        self.candidates = list(candidates or [])

    def structured_lookup(self, scope, *, kinds, limit):
        return [c for c in self.candidates if c.kind in kinds][:limit]

    def lexical_search(self, query_text, scope, *, kinds, limit):
        _ = (query_text, scope)
        return [c for c in self.candidates if c.kind in kinds][:limit]

    def vector_search(self, query_embedding, scope, *, kinds, limit):
        _ = (query_embedding, scope)
        return [c for c in self.candidates if c.kind in kinds][:limit]

    def search_failures(self, query_text, scope, *, limit):
        _ = (query_text, scope)
        return [c for c in self.candidates if c.kind == "failure"][:limit]


def _finding(candidate_id: str, text: str) -> Candidate:
    return Candidate(id=candidate_id, kind="finding", title=f"Finding {candidate_id}", text=text, status="validated")


def _failure_candidate(candidate_id: str, text: str) -> Candidate:
    return Candidate(id=candidate_id, kind="failure", title=f"Failure {candidate_id}", text=text, status="recorded")


def _adapter(store: InMemoryExperimentStore | None = None, **kwargs) -> KnowledgeAdapter:
    return KnowledgeAdapter(store or InMemoryExperimentStore(), project_id=PROJECT_ID, **kwargs)


@pytest.mark.asyncio
async def test_recheck_clean_spec_reports_kb_aligned_hashes_and_no_matches():
    adapter = _adapter(search_store=InMemorySearchStore())
    spec = _spec()
    result = await adapter.recheck(spec)
    design = {"hypothesis": normalize_text("momentum persists"), "methodology": {"universe": "sp500"}}
    assert result.execution_hash == execution_hash(
        design=design,
        code={"artifact": "code-v1"},
        data={"datasets": []},
        parameters={"seed": 0},
        environment={"image": "qf:v1"},
    )
    assert result.family_hash == experiment_family_hash(design)
    assert result.exact_match_id is None
    assert result.family_match_ids == ()


@pytest.mark.asyncio
async def test_recheck_exact_rerun_raises_duplicate_run():
    store = InMemoryExperimentStore()
    adapter = _adapter(store)
    spec = _spec()
    record = await adapter.begin_experiment(spec, run_id=RUN_ID, idempotency_key="begin-key-1")
    with pytest.raises(DuplicateRunError) as exc_info:
        await adapter.recheck(spec)
    assert exc_info.value.execution_hash == record.execution_hash
    assert exc_info.value.existing_experiment_id == record.id


@pytest.mark.asyncio
async def test_recheck_lists_family_equivalents():
    store = InMemoryExperimentStore()
    adapter = _adapter(store)
    first = await adapter.begin_experiment(_spec(seed=1), run_id=RUN_ID, idempotency_key="begin-key-1")
    adapter_with_search = KnowledgeAdapter(store, project_id=PROJECT_ID, search_store=InMemorySearchStore([first]))
    result = await adapter_with_search.recheck(_spec(seed=2))
    assert result.exact_match_id is None
    assert result.family_match_ids == (first.id,)


@pytest.mark.asyncio
async def test_begin_replays_on_same_key_and_payload():
    adapter = _adapter()
    spec = _spec()
    first = await adapter.begin_experiment(spec, run_id=RUN_ID, idempotency_key="begin-key-1")
    second = await adapter.begin_experiment(spec, run_id=RUN_ID, idempotency_key="begin-key-1")
    assert first.id == second.id
    assert first.status == "running"
    assert first.created_by_run_id == RUN_ID


@pytest.mark.asyncio
async def test_begin_rejects_spec_without_hypothesis_text():
    adapter = _adapter()
    with pytest.raises(EvolutionError, match="hypothesis"):
        await adapter.begin_experiment(ExperimentSpec(design={"methodology": {}}), run_id=RUN_ID)


@pytest.mark.asyncio
async def test_commit_failure_record_round_trip():
    adapter = _adapter()
    record = await adapter.begin_experiment(_spec(), run_id=RUN_ID, idempotency_key="begin-key-1")
    failure = await adapter.record_failure(experiment_id=record.id, failure_class="execution", notes="boom", idempotency_key="fail-key-1")
    assert failure.experiment_id == record.id
    committed = await adapter.commit_experiment(
        experiment_id=record.id,
        metrics={"sr": 0.1},
        outcome="failure",
        failure_class="execution",
        idempotency_key="commit-key-1",
    )
    assert committed.status == "failed"
    assert committed.outcome == "failure"
    replayed = await adapter.commit_experiment(
        experiment_id=record.id,
        metrics={"sr": 0.1},
        outcome="failure",
        failure_class="execution",
        idempotency_key="commit-key-1",
    )
    assert replayed.id == committed.id


@pytest.mark.asyncio
async def test_record_assumption_round_trip():
    adapter = _adapter()
    record = await adapter.begin_experiment(_spec(), run_id=RUN_ID, idempotency_key="begin-key-1")
    assumption = await adapter.record_assumption(
        statement="spreads stay positive",
        category="market",
        experiment_id=record.id,
        idempotency_key="assume-key-1",
    )
    assert assumption.experiment_id == record.id
    assert assumption.category == "market"


@pytest.mark.asyncio
async def test_build_proposal_context_without_retrieval_is_empty():
    adapter = _adapter()
    packet = await adapter.build_proposal_context("momentum in us equities")
    assert packet.text == ""
    assert packet.refs == ()
    assert packet.intent_topic == "momentum in us equities"


@pytest.mark.asyncio
async def test_build_proposal_context_runs_full_retrieval_path():
    channels = RetrievalChannels(
        structured=ChannelFake([_finding("f-1", "momentum persists over 21 days")]),
        lexical=ChannelFake([_finding("f-2", "reversal at short horizons")]),
        vector=ChannelFake([]),
        failures=ChannelFake([_failure_candidate("fl-1", "crowded momentum crashed")]),
    )
    adapter = _adapter(retrieval=channels)
    packet = await adapter.build_proposal_context("momentum in us equities")
    assert "RESEARCH CONTEXT" in packet.text
    assert "momentum" in packet.text.lower()
    assert set(packet.refs) >= {"f-1", "f-2", "fl-1"}
    assert packet.intent_topic


@pytest.mark.asyncio
async def test_build_proposal_context_rejects_empty_topic():
    adapter = _adapter()
    with pytest.raises(EvolutionError, match="topic"):
        await adapter.build_proposal_context("   ")


@pytest.mark.asyncio
async def test_commit_carries_harvested_path_refs_in_metrics():
    """Harvested sandbox paths must not fail the KB commit (UUID validation).

    Non-UUID artifact refs (e.g. ``outputs/metrics.json`` harvested by the
    sandbox executor) ride into the commit via ``metrics["sandbox_artifacts"]``
    while real artifact UUIDs land in ``result_artifacts``.
    """
    adapter = _adapter()
    record = await adapter.begin_experiment(_spec(), run_id=RUN_ID, idempotency_key="begin-key-paths")
    artifact_id = str(uuid.uuid4())
    committed = await adapter.commit_experiment(
        experiment_id=record.id,
        metrics={"sr": 1.5},
        outcome="success",
        result_artifacts=["outputs/metrics.json", artifact_id],
        idempotency_key="commit-key-paths",
    )
    assert committed.result_artifacts == [artifact_id]
    assert committed.metrics["sandbox_artifacts"] == ["outputs/metrics.json"]
    assert committed.metrics["sr"] == 1.5

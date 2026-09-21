"""Tests for the Phase 2 PG retrieval bindings (integration).

Covers :mod:`deerflow.knowledge.pg_retrieval` against a file-backed SQLite
database (same query code paths as PostgreSQL; SQLite is the dev/test
dialect per the schema conventions):

* row-to-candidate mapping (stable kinds, scope/validity derivation,
  failure markers);
* the four channel stores (structured kinds filter + newest-first order,
  lexical portable ranking, vector brute-force cosine, failure-set
  restriction) including hard-filter screening and limit trimming;
* ``SQLExperimentLookupStore.find_by_id``;
* the embedding plane binding (text source, vector store, backfill
  round-trip, search-document refresh);
* the ``KnowledgeRetrievalBackend`` service adapter (kind mapping,
  structured filters, ACL, status post-filter, pagination, id lookup);
* PostgreSQL SQL compilation for the FTS / pgvector / reindex statements
  (no server required) and the ``open_pg_backends`` DSN helper smoke test.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import sessionmaker

import deerflow.knowledge.schema as kb_schema
from deerflow.knowledge import pg_retrieval as pg
from deerflow.knowledge.embeddings import (
    EMBEDDING_DIM,
    FAKE_MODEL_ID,
    DeterministicEmbeddingProvider,
    backfill_findings_embeddings,
    render_embeddable_text,
)
from deerflow.knowledge.retrieval.planner import (
    FailureSearchStore,
    LexicalSearchStore,
    ScopeFilter,
    StructuredLookupStore,
    VectorSearchStore,
)
from deerflow.knowledge.schema.experiments import AssumptionRow, ExperimentRow
from deerflow.knowledge.schema.findings import FindingRow
from deerflow.knowledge.tools.lookup import KnowledgeBackends, knowledge_get, knowledge_search
from deerflow.knowledge.write_api import KnowledgeError, KnowledgeValidationError
from deerflow.persistence.base import Base

KB_TABLES: tuple[str, ...] = (
    "research_project",
    "agent_run",
    "finding",
    "experiment",
    "experiment_dataset",
    "experiment_artifact",
    "assumption",
)

FAMILY_A = "aa" * 32
FAMILY_B = "bb" * 32
EXEC_A = "a1" * 32
EXEC_B = "b2" * 32
EXEC_C = "c3" * 32


def _engine(tmp_path: Path, name: str = "kb_retrieval.db") -> sa.Engine:
    engine = sa.create_engine(f"sqlite:///{(tmp_path / name).as_posix()}")
    for table in KB_TABLES:
        Base.metadata.tables[table].create(engine, checkfirst=True)
    return engine


@pytest.fixture
def store(tmp_path: Path) -> Iterator[dict]:
    """Build the seeded database; return ids + session factory + rows."""
    engine = _engine(tmp_path)
    factory = sessionmaker(engine, expire_on_commit=False)
    project_id = uuid.uuid4()
    run_id = uuid.uuid4()
    finding_id = uuid.uuid4()
    failure_finding_id = uuid.uuid4()
    experiment_id = uuid.uuid4()
    failure_experiment_id = uuid.uuid4()
    assumption_id = uuid.uuid4()
    with factory() as session:
        session.add(kb_schema.ResearchProjectRow(id=project_id, name="momentum", visibility_scope={}))
        session.add(kb_schema.AgentRunRow(id=run_id, project_id=project_id, agent_type="research", task="retrieval binding", status="running"))
        session.add(
            FindingRow(
                id=finding_id,
                project_id=project_id,
                canonical_key="empirical:momentum-persists-net-costs",
                finding_type="empirical",
                statement="Cross-sectional momentum persists net of transaction costs on liquid United States stocks.",
                scope={"asset_class": "equity", "market": "US", "universe": "sp500-pit", "horizon": "6m", "frequency": "daily"},
                status="candidate",
                confidence={"overall_tier": "moderate"},
                effective_from=datetime(2015, 1, 1, tzinfo=UTC),
                effective_to=None,
                recorded_at=datetime(2026, 3, 10, tzinfo=UTC),
                created_by_run_id=run_id,
            )
        )
        session.add(
            FindingRow(
                id=failure_finding_id,
                project_id=project_id,
                canonical_key="failure:borrow-costs-short-leg",
                finding_type="failure",
                statement="Short-leg borrow costs erase reversal strategy profits in hard-to-borrow names.",
                scope={"asset_class": "equity", "market": "US", "universe": "sp500-pit", "horizon": "5d", "frequency": "daily"},
                status="candidate",
                confidence={"overall_tier": "moderate"},
                effective_from=datetime(2015, 1, 1, tzinfo=UTC),
                effective_to=None,
                recorded_at=datetime(2026, 3, 11, tzinfo=UTC),
                created_by_run_id=run_id,
            )
        )
        session.add(
            ExperimentRow(
                id=experiment_id,
                project_id=project_id,
                created_by_run_id=run_id,
                experiment_family_hash=FAMILY_A,
                execution_hash=EXEC_A,
                hypothesis="Cross-sectional momentum 126-day backtest",
                methodology={
                    "asset_class": "equity",
                    "market": "US",
                    "universe": "sp500-pit",
                    "horizon": "6m",
                    "frequency": "daily",
                    "sample_period": ["2015-01-01", "2026-09-19"],
                },
                parameters={"signal_name": "mom_126d", "top_n": 20, "eval_retrieval_text": "quintile long-short momentum backtest sharpe turnover"},
                metrics={"sharpe": 0.82},
                outcome="success",
                failure_class=None,
                status="completed",
                started_at=datetime(2026, 1, 10, tzinfo=UTC),
                completed_at=datetime(2026, 1, 11, tzinfo=UTC),
            )
        )
        session.add(
            ExperimentRow(
                id=failure_experiment_id,
                project_id=project_id,
                created_by_run_id=run_id,
                experiment_family_hash=FAMILY_B,
                execution_hash=EXEC_B,
                hypothesis="Short-horizon reversal backtest with borrow costs",
                methodology={
                    "asset_class": "equity",
                    "market": "US",
                    "universe": "sp500-pit",
                    "horizon": "5d",
                    "frequency": "daily",
                    "sample_period": ["2015-01-01", "2026-09-19"],
                },
                parameters={"signal_name": "reversal_5d", "eval_retrieval_text": "borrow costs short leg reversal drawdown"},
                metrics={"sharpe": -0.4},
                outcome="failure",
                failure_class="execution",
                status="completed",
                started_at=datetime(2026, 2, 10, tzinfo=UTC),
                completed_at=datetime(2026, 2, 11, tzinfo=UTC),
            )
        )
        session.add(
            AssumptionRow(
                id=assumption_id,
                experiment_id=None,
                statement="Closing auction volume absorbs the rebalanced notional without material slippage.",
                category="execution",
                sensitivity="high",
                tested=False,
                status="active",
            )
        )
        session.commit()
    try:
        yield {
            "factory": factory,
            "engine": engine,
            "project_id": project_id,
            "run_id": run_id,
            "finding_id": finding_id,
            "failure_finding_id": failure_finding_id,
            "experiment_id": experiment_id,
            "failure_experiment_id": failure_experiment_id,
            "assumption_id": assumption_id,
        }
    finally:
        engine.dispose()


def _scope(**overrides) -> ScopeFilter:
    kwargs = {
        "asset_class": "equity",
        "markets": ("US",),
        "universe": "sp500-pit",
        "horizon": "6m",
        "frequency": "daily",
        "valid_from": "2015-01-01",
        "valid_to": "2026-09-19",
    }
    kwargs.update(overrides)
    return ScopeFilter(**kwargs)


class TestRowMapping:
    def test_finding_mapping(self, store: dict) -> None:
        with store["factory"]() as session:
            row = session.get(FindingRow, store["finding_id"])
            assert row is not None
            candidate = pg.finding_row_to_candidate(row)
        assert candidate.id == str(store["finding_id"])
        assert candidate.kind == "finding"
        assert candidate.title == "empirical:momentum-persists-net-costs"
        assert "momentum persists" in candidate.text
        assert candidate.scope["universe"] == "sp500-pit"
        assert candidate.status == "candidate"
        assert candidate.finding_type == "empirical"
        assert candidate.project_id == str(store["project_id"])
        assert candidate.valid_from is not None and "2015-01-01" in candidate.valid_from
        assert candidate.valid_to is None
        assert candidate.recorded_at != ""
        assert candidate.evidence == ()

    def test_failure_kinds_are_stable(self, store: dict) -> None:
        with store["factory"]() as session:
            failure_finding = session.get(FindingRow, store["failure_finding_id"])
            failed_experiment = session.get(ExperimentRow, store["failure_experiment_id"])
            ok_experiment = session.get(ExperimentRow, store["experiment_id"])
            assert failure_finding is not None and failed_experiment is not None and ok_experiment is not None
            assert pg.finding_row_to_candidate(failure_finding).kind == "failure"
            assert pg.experiment_row_to_candidate(failed_experiment).kind == "failure"
            assert pg.experiment_row_to_candidate(ok_experiment).kind == "experiment"

    def test_experiment_scope_and_validity_derive_from_methodology(self, store: dict) -> None:
        with store["factory"]() as session:
            row = session.get(ExperimentRow, store["experiment_id"])
            assert row is not None
            candidate = pg.experiment_row_to_candidate(row)
        assert candidate.scope == {"asset_class": "equity", "market": "US", "universe": "sp500-pit", "horizon": "6m", "frequency": "daily"}
        assert candidate.valid_from == "2015-01-01"
        assert candidate.valid_to == "2026-09-19"
        assert candidate.family_hash == FAMILY_A
        assert candidate.metadata["outcome"] == "success"
        assert "mom_126d" in candidate.text  # parameters surface stays matchable
        assert candidate.project_id == str(store["project_id"])

    def test_failed_experiment_carries_failure_markers(self, store: dict) -> None:
        with store["factory"]() as session:
            row = session.get(ExperimentRow, store["failure_experiment_id"])
            assert row is not None
            candidate = pg.experiment_row_to_candidate(row)
        assert candidate.metadata["outcome"] == "failure"
        assert candidate.metadata["failure_class"] == "execution"
        assert "borrow costs" in candidate.text

    def test_assumption_mapping_is_scope_neutral(self, store: dict) -> None:
        with store["factory"]() as session:
            row = session.get(AssumptionRow, store["assumption_id"])
            assert row is not None
            candidate = pg.assumption_row_to_candidate(row)
        assert candidate.kind == "assumption"
        assert candidate.scope == {}
        assert candidate.project_id is None
        assert candidate.metadata["category"] == "execution"
        assert candidate.metadata["tested"] is False
        assert "slippage" in candidate.text


class TestChannelProtocols:
    def test_stores_satisfy_the_planner_protocols(self, store: dict) -> None:
        factory = store["factory"]
        assert isinstance(pg.SQLStructuredLookupStore(factory), StructuredLookupStore)
        assert isinstance(pg.SQLLexicalSearchStore(factory), LexicalSearchStore)
        assert isinstance(pg.SQLVectorSearchStore(factory), VectorSearchStore)
        assert isinstance(pg.SQLFailureSearchStore(factory), FailureSearchStore)


class TestStructuredStore:
    def test_kinds_filter_and_newest_first(self, store: dict) -> None:
        backend = pg.SQLStructuredLookupStore(store["factory"])
        # No scope declared: every row is compatible; kind "failure" covers
        # failed experiments and failure findings alike, newest-first.
        found = backend.structured_lookup(ScopeFilter(), kinds=["experiment", "failure"], limit=10)
        assert [item.id for item in found] == [
            str(store["failure_finding_id"]),  # recorded 2026-03-11
            str(store["failure_experiment_id"]),  # started 2026-02-10
            str(store["experiment_id"]),  # started 2026-01-10
        ]
        assert [item.kind for item in found] == ["failure", "failure", "experiment"]
        experiments_only = backend.structured_lookup(ScopeFilter(), kinds=["experiment"], limit=10)
        assert [item.id for item in experiments_only] == [str(store["experiment_id"])]

    def test_findings_and_assumptions(self, store: dict) -> None:
        backend = pg.SQLStructuredLookupStore(store["factory"])
        found = backend.structured_lookup(ScopeFilter(), kinds=["finding", "assumption"], limit=10)
        by_id = {item.id: item for item in found}
        assert by_id[str(store["finding_id"])].kind == "finding"
        assert by_id[str(store["assumption_id"])].kind == "assumption"
        assert str(store["failure_finding_id"]) not in by_id  # kind "failure" not requested

    def test_scope_screening_matches_fusion(self, store: dict) -> None:
        backend = pg.SQLStructuredLookupStore(store["factory"])
        scoped = backend.structured_lookup(_scope(), kinds=["finding", "experiment", "failure", "assumption"], limit=10)
        assert {item.id for item in scoped} == {
            str(store["finding_id"]),
            str(store["failure_finding_id"]),
            str(store["experiment_id"]),
            str(store["failure_experiment_id"]),
            str(store["assumption_id"]),
        }
        mismatched = backend.structured_lookup(_scope(asset_class="credit"), kinds=["finding", "experiment", "failure", "assumption"], limit=10)
        # Only the scope-neutral assumption survives an asset-class mismatch.
        assert [item.id for item in mismatched] == [str(store["assumption_id"])]
        disjoint = backend.structured_lookup(_scope(valid_from="1990-01-01", valid_to="1990-12-31"), kinds=["finding", "experiment"], limit=10)
        assert disjoint == []

    def test_limit_trims(self, store: dict) -> None:
        backend = pg.SQLStructuredLookupStore(store["factory"])
        found = backend.structured_lookup(ScopeFilter(), kinds=["finding", "experiment", "failure", "assumption"], limit=2)
        assert len(found) == 2

    def test_scope_match_count_counts_declared_fields(self) -> None:
        scope = _scope()
        assert pg.scope_match_count({"asset_class": "Equity", "universe": "sp500-pit", "horizon": "6m", "frequency": "daily", "market": "US"}, scope) == 5
        assert pg.scope_match_count({"asset_class": "equity", "universe": "sp500-pit", "horizon": "5d", "frequency": "daily", "market": "US"}, scope) == 4
        assert pg.scope_match_count({}, scope) == 0
        assert pg.scope_match_count({"asset_class": "equity"}, ScopeFilter()) == 0  # empty filter: no signal

    def test_scope_match_count_ignores_placeholders(self) -> None:
        # "mixed"/"n/a" mean unspecified: placeholder agreement scores 0,
        # and placeholders never count as mismatches either.
        assert pg.scope_match_count({"horizon": "mixed"}, ScopeFilter(horizon="mixed")) == 0
        assert pg.scope_match_count({"universe": "n/a"}, ScopeFilter(universe="sp500-pit")) == 0
        full = {"asset_class": "equity", "universe": "sp500-pit", "frequency": "daily", "market": "US"}
        assert pg.scope_match_count({**full, "horizon": "mixed"}, _scope(horizon="mixed")) == 4
        assert pg.scope_match_count({**full, "horizon": "63d"}, _scope(horizon="mixed")) == 4

    def test_ranks_by_scope_match_strength(self, store: dict) -> None:
        backend = pg.SQLStructuredLookupStore(store["factory"])
        # finding + experiment match 5 scope fields (6m horizon); both
        # failures match 4 (5d horizon). Strength beats recency: the older
        # experiment outranks the newer failure_finding.
        found = backend.structured_lookup(_scope(), kinds=["finding", "experiment", "failure"], limit=10)
        assert [item.id for item in found] == [
            str(store["finding_id"]),  # 5 matches, recorded 2026-03-10
            str(store["experiment_id"]),  # 5 matches, started 2026-01-10
            str(store["failure_finding_id"]),  # 4 matches, recorded 2026-03-11
            str(store["failure_experiment_id"]),  # 4 matches, started 2026-02-10
        ]

    def test_prunes_zero_match_rows_but_keeps_scope_neutral(self, store: dict) -> None:
        # The filter constrains universe/horizon/frequency/markets (no asset
        # class, so no hard filter fires) and nothing in the fixture matches
        # any of them: every scoped row scores 0 and drops, while the
        # scope-empty assumption stays (declaring nothing is neutral).
        backend = pg.SQLStructuredLookupStore(store["factory"])
        alien = ScopeFilter(markets=("JP",), universe="nikkei", horizon="99d", frequency="hourly")
        found = backend.structured_lookup(alien, kinds=["finding", "experiment", "failure", "assumption"], limit=10)
        assert [item.id for item in found] == [str(store["assumption_id"])]
        # An empty filter constrains nothing: no pruning, recency order.
        found = backend.structured_lookup(ScopeFilter(), kinds=["finding", "experiment", "failure", "assumption"], limit=10)
        assert len(found) == 5


class TestLexicalStore:
    def test_portable_ranking_prefers_token_overlap(self, store: dict) -> None:
        backend = pg.SQLLexicalSearchStore(store["factory"])
        found = backend.lexical_search("momentum backtest sharpe", ScopeFilter(), kinds=["finding", "experiment", "failure", "assumption"], limit=10)
        assert found, "expected lexical matches for fixture vocabulary"
        assert found[0].id == str(store["experiment_id"])
        assert all(item.id != str(store["assumption_id"]) for item in found)  # zero overlap excluded

    def test_zero_overlap_returns_nothing(self, store: dict) -> None:
        backend = pg.SQLLexicalSearchStore(store["factory"])
        assert backend.lexical_search("zyzzyva quokka", ScopeFilter(), kinds=["finding", "experiment", "failure", "assumption"], limit=10) == []

    def test_kinds_and_scope_apply(self, store: dict) -> None:
        backend = pg.SQLLexicalSearchStore(store["factory"])
        found = backend.lexical_search("borrow costs", ScopeFilter(), kinds=["failure"], limit=10)
        assert {item.id for item in found} == {str(store["failure_finding_id"]), str(store["failure_experiment_id"])}
        scoped_out = backend.lexical_search("borrow costs", _scope(asset_class="credit"), kinds=["failure"], limit=10)
        assert scoped_out == []


class TestVectorStore:
    def test_portable_cosine_finds_nearest(self, store: dict) -> None:
        provider = DeterministicEmbeddingProvider()
        texts = pg.SQLFindingTextSource(store["factory"])
        vectors = pg.SQLEmbeddingVectorStore(store["factory"], model_id=provider.model_id)
        result = backfill_findings_embeddings(provider, texts, vectors, [str(store["finding_id"]), str(store["failure_finding_id"])])
        assert result.upserted == 2 and not result.failures
        backend = pg.SQLVectorSearchStore(store["factory"])
        query_text = texts.get_finding_text(str(store["finding_id"]))
        assert query_text is not None
        [query_vector] = provider.embed_batch([query_text])
        found = backend.vector_search(query_vector, ScopeFilter(), kinds=["finding", "failure"], limit=10)
        assert [item.id for item in found] == [str(store["finding_id"]), str(store["failure_finding_id"])]

    def test_unembedded_rows_never_match(self, store: dict) -> None:
        backend = pg.SQLVectorSearchStore(store["factory"])
        query_vector = DeterministicEmbeddingProvider().embed_one("momentum")
        assert backend.vector_search(query_vector, ScopeFilter(), kinds=["finding", "failure"], limit=10) == []

    def test_dimension_mismatch_is_rejected(self, store: dict) -> None:
        backend = pg.SQLVectorSearchStore(store["factory"])
        with pytest.raises(KnowledgeValidationError):
            backend.vector_search([0.1, 0.2], ScopeFilter(), kinds=["finding"], limit=10)

    def test_non_finding_kinds_match_nothing(self, store: dict) -> None:
        backend = pg.SQLVectorSearchStore(store["factory"])
        query_vector = DeterministicEmbeddingProvider().embed_one("momentum")
        assert backend.vector_search(query_vector, ScopeFilter(), kinds=["experiment", "assumption"], limit=10) == []


class TestFailureStore:
    def test_only_failures_ranked_by_relevance(self, store: dict) -> None:
        backend = pg.SQLFailureSearchStore(store["factory"])
        found = backend.search_failures("borrow costs short leg", ScopeFilter(), limit=10)
        assert {item.id for item in found} == {str(store["failure_finding_id"]), str(store["failure_experiment_id"])}
        assert all(item.kind == "failure" for item in found)

    def test_scope_screening_applies(self, store: dict) -> None:
        backend = pg.SQLFailureSearchStore(store["factory"])
        assert backend.search_failures("borrow costs", _scope(asset_class="credit"), limit=10) == []


class TestExperimentLookup:
    def test_find_by_id_hit_and_miss(self, store: dict) -> None:
        backend = pg.SQLExperimentLookupStore(store["factory"])
        record = backend.find_by_id(str(store["experiment_id"]))
        assert record is not None
        assert record.hypothesis == "Cross-sectional momentum 126-day backtest"
        assert record.family_hash == FAMILY_A
        assert record.execution_hash == EXEC_A
        assert backend.find_by_id(str(uuid.uuid4())) is None

    def test_find_by_id_rejects_non_uuid(self, store: dict) -> None:
        backend = pg.SQLExperimentLookupStore(store["factory"])
        with pytest.raises(KnowledgeValidationError):
            backend.find_by_id("not-a-uuid")


class TestEmbeddingPlane:
    def test_text_source_round_trip(self, store: dict) -> None:
        texts = pg.SQLFindingTextSource(store["factory"])
        expected = render_embeddable_text(
            title="empirical:momentum-persists-net-costs",
            body="Cross-sectional momentum persists net of transaction costs on liquid United States stocks.",
        )
        assert texts.get_finding_text(str(store["finding_id"])) == expected
        assert texts.get_finding_text(str(uuid.uuid4())) is None
        with pytest.raises(KnowledgeValidationError):
            texts.get_finding_text("nope")

    def test_vector_store_round_trip_and_model_report(self, store: dict) -> None:
        provider = DeterministicEmbeddingProvider()
        vectors = pg.SQLEmbeddingVectorStore(store["factory"], model_id=provider.model_id)
        finding_id = str(store["finding_id"])
        assert vectors.get_embedding_model(finding_id) is None
        [vector] = provider.embed_batch(["momentum"])
        vectors.upsert_embedding(finding_id, vector, model_id=provider.model_id)
        assert vectors.get_embedding_model(finding_id) == provider.model_id
        assert vectors.get_embedding_model(str(uuid.uuid4())) is None

    def test_vector_store_refuses_bad_writes(self, store: dict) -> None:
        vectors = pg.SQLEmbeddingVectorStore(store["factory"], model_id=FAKE_MODEL_ID)
        finding_id = str(store["finding_id"])
        with pytest.raises(KnowledgeValidationError):
            vectors.upsert_embedding(finding_id, [0.1, 0.2], model_id=FAKE_MODEL_ID)
        [vector] = DeterministicEmbeddingProvider().embed_batch(["momentum"])
        with pytest.raises(KnowledgeValidationError):
            vectors.upsert_embedding(finding_id, vector, model_id="other-model/v1")
        with pytest.raises(KnowledgeError):
            vectors.upsert_embedding(str(uuid.uuid4()), vector, model_id=FAKE_MODEL_ID)
        with pytest.raises(KnowledgeValidationError):
            vectors.upsert_embedding("nope", vector, model_id=FAKE_MODEL_ID)

    def test_backfill_is_idempotent(self, store: dict) -> None:
        provider = DeterministicEmbeddingProvider()
        texts = pg.SQLFindingTextSource(store["factory"])
        vectors = pg.SQLEmbeddingVectorStore(store["factory"], model_id=provider.model_id)
        ids = [str(store["finding_id"])]
        first = backfill_findings_embeddings(provider, texts, vectors, ids)
        assert (first.upserted, first.skipped_up_to_date) == (1, 0)
        second = backfill_findings_embeddings(provider, texts, vectors, ids)
        assert (second.upserted, second.skipped_up_to_date) == (0, 1)

    def test_refresh_search_documents_sqlite(self, store: dict) -> None:
        assert pg.refresh_finding_search_documents(store["factory"]) == 2
        with store["factory"]() as session:
            row = session.get(FindingRow, store["finding_id"])
            assert row is not None
            assert "momentum-persists-net-costs" in (row.search_document or "")
            assert "momentum persists" in (row.search_document or "")
        assert pg.refresh_finding_search_documents(store["factory"], [str(store["finding_id"])]) == 1
        with pytest.raises(KnowledgeValidationError):
            pg.refresh_finding_search_documents(store["factory"], ["nope"])


class TestRetrievalService:
    def test_search_fuses_channels(self, store: dict) -> None:
        service = pg.KnowledgeRetrievalService(store["factory"])
        page = service.search("momentum backtest", kinds=("finding", "experiment", "failure", "assumption"), filters={}, limit=10, offset=0)
        assert page.total and page.total >= 3
        top_ids = {doc.id for doc in page.documents[:3]}
        assert {str(store["experiment_id"]), str(store["finding_id"])} <= top_ids
        assert all(doc.payload["channels"] for doc in page.documents), "expected channel provenance on fused hits"

    def test_search_failure_kind(self, store: dict) -> None:
        service = pg.KnowledgeRetrievalService(store["factory"])
        page = service.search("borrow costs", kinds=("failure",), filters={}, limit=10, offset=0)
        assert {doc.id for doc in page.documents} == {str(store["failure_finding_id"]), str(store["failure_experiment_id"])}
        assert all(doc.kind == "failure" for doc in page.documents)

    def test_search_kinds_exclude_failure_channel_leak(self, store: dict) -> None:
        # "borrow costs" matches the failure rows lexically and the failure
        # channel runs regardless of kinds — but kinds=("finding",) must
        # still return findings only.
        service = pg.KnowledgeRetrievalService(store["factory"])
        page = service.search("borrow costs short leg", kinds=("finding",), filters={}, limit=10, offset=0)
        assert page.total >= 1
        assert all(doc.kind == "finding" for doc in page.documents)
        assert str(store["failure_finding_id"]) not in {doc.id for doc in page.documents}

    def test_search_filters_and_pagination(self, store: dict) -> None:
        service = pg.KnowledgeRetrievalService(store["factory"])
        scoped = service.search("momentum", kinds=("finding", "experiment"), filters={"asset_class": "credit"}, limit=10, offset=0)
        assert scoped.total == 0
        validated = service.search("momentum", kinds=("finding",), filters={"status": "validated"}, limit=10, offset=0)
        assert validated.total == 0  # Phase 2 seeds candidates only
        candidates = service.search("momentum", kinds=("finding",), filters={"status": "candidate"}, limit=10, offset=0)
        assert candidates.total == 1
        first = service.search("momentum", kinds=("finding", "experiment", "failure", "assumption"), filters={}, limit=1, offset=0)
        second = service.search("momentum", kinds=("finding", "experiment", "failure", "assumption"), filters={}, limit=1, offset=1)
        assert first.total == second.total and first.total and first.documents[0].id != second.documents[0].id

    def test_search_project_acl(self, store: dict) -> None:
        service = pg.KnowledgeRetrievalService(store["factory"])
        visible = service.search("momentum", kinds=("finding", "experiment"), filters={"project_id": str(store["project_id"])}, limit=10, offset=0)
        assert visible.total and visible.total >= 1
        hidden = service.search("momentum", kinds=("finding", "experiment"), filters={"project_id": str(uuid.uuid4())}, limit=10, offset=0)
        assert hidden.total == 0

    def test_search_artifact_kind_is_empty(self, store: dict) -> None:
        service = pg.KnowledgeRetrievalService(store["factory"])
        page = service.search("momentum", kinds=("artifact",), filters={}, limit=10, offset=0)
        assert page.documents == [] and page.total == 0

    def test_search_with_test_fake_provider_runs_vector(self, store: dict) -> None:
        provider = DeterministicEmbeddingProvider()
        texts = pg.SQLFindingTextSource(store["factory"])
        vectors = pg.SQLEmbeddingVectorStore(store["factory"], model_id=provider.model_id)
        backfill_findings_embeddings(provider, texts, vectors, [str(store["finding_id"]), str(store["failure_finding_id"])])
        service = pg.KnowledgeRetrievalService(store["factory"], embedding_provider=provider)
        page = service.search("momentum backtest", kinds=("finding", "experiment"), filters={}, limit=10, offset=0)
        assert page.total and page.total >= 1

    def test_search_without_provider_skips_vector(self, store: dict) -> None:
        service = pg.KnowledgeRetrievalService(store["factory"])
        assert service.embedding_provider is None
        page = service.search("momentum", kinds=("finding", "experiment"), filters={}, limit=10, offset=0)
        assert page.total and page.total >= 1

    def test_tool_functions_run_over_the_service(self, store: dict) -> None:
        service = pg.KnowledgeRetrievalService(store["factory"])
        payload = knowledge_search("borrow costs", kinds="failure", backend=service)
        assert payload["count"] == 2
        fetched = knowledge_get([str(store["experiment_id"]), str(uuid.uuid4())], backend=service)
        assert len(fetched["documents"]) == 1
        assert fetched["errors"] == [{"id": fetched["errors"][0]["id"], "error": "not_found"}]

    def test_get_documents_order_and_evidence(self, store: dict) -> None:
        service = pg.KnowledgeRetrievalService(store["factory"])
        ids = [str(store["assumption_id"]), "not-a-uuid", str(store["finding_id"]), str(store["experiment_id"]), str(uuid.uuid4())]
        documents = service.get_documents(ids, include_evidence=True)
        assert [doc.id if doc is not None else None for doc in documents] == [ids[0], None, ids[2], ids[3], None]
        assert documents[0] is not None and documents[0].kind == "assumption"
        assert documents[2] is not None and documents[2].payload["canonical_key"] == "empirical:momentum-persists-net-costs"
        assert documents[3] is not None and documents[3].payload["record"]["execution_hash"] == EXEC_A


class TestPostgresCompilation:
    def test_fts_statements_render_operators(self) -> None:
        dialect = postgresql.dialect()
        finding_sql = str(pg._finding_fts_statement("momentum costs").compile(dialect=dialect))
        assert "@@" in finding_sql and "plainto_tsquery" in finding_sql and "ts_rank_cd" in finding_sql
        assert "search_document" in finding_sql
        experiment_sql = str(pg._experiment_fts_statement("momentum").compile(dialect=dialect))
        assert "@@" in experiment_sql and "to_tsvector" in experiment_sql and "parameters" in experiment_sql
        assumption_sql = str(pg._assumption_fts_statement("slippage").compile(dialect=dialect))
        assert "@@" in assumption_sql and "to_tsvector" in assumption_sql

    def test_vector_statement_renders_pgvector_operator(self) -> None:
        dialect = postgresql.dialect()
        sql = str(pg._finding_vector_statement("[0.1,0.2]").compile(dialect=dialect))
        assert "<=>" in sql and "VECTOR(768)" in sql and "embedding IS NOT NULL" in sql

    def test_reindex_update_renders_tsvector(self) -> None:
        dialect = postgresql.dialect()
        sql = str(pg._finding_reindex_update([uuid.uuid4()]).compile(dialect=dialect))
        assert "to_tsvector" in sql and "search_document" in sql


class TestOpenBackends:
    def test_open_pg_backends_sqlite(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path, "open_backends.db")
        engine.dispose()
        dsn = f"sqlite:///{(tmp_path / 'open_backends.db').as_posix()}"
        backends = pg.open_pg_backends(dsn)
        assert isinstance(backends, KnowledgeBackends)
        assert isinstance(backends.retrieval, pg.KnowledgeRetrievalService)
        assert isinstance(backends.experiments, pg.SQLExperimentLookupStore)
        assert backends.artifacts is None
        assert backends.retrieval.embedding_provider is None  # no silent test-fake

    def test_open_pg_backends_with_fake_provider(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path, "open_backends_fake.db")
        engine.dispose()
        dsn = f"sqlite:///{(tmp_path / 'open_backends_fake.db').as_posix()}"
        backends = pg.open_pg_backends(dsn, embedding_model="test-fake/v1")
        assert backends.retrieval is not None
        assert backends.retrieval.embedding_provider is not None
        assert backends.retrieval.embedding_provider.model_id == FAKE_MODEL_ID
        assert backends.retrieval.embedding_provider.dimension == EMBEDDING_DIM

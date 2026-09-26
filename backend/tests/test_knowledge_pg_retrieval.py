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
  (no server required) and the ``open_pg_backends`` DSN helper smoke test;
* the startup binding (``bind_knowledge_backends_from_config``: DSN +
  embedding model from ``KnowledgeConfig``) and a read-only pass over the
  real shared KB file (hash-pinned: the file must be bit-identical after).
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

import deerflow.knowledge.schema as kb_schema
from deerflow.knowledge import embeddings as embeddings_mod
from deerflow.knowledge import pg_retrieval as pg
from deerflow.knowledge.config import (
    KnowledgeConfig,
    get_knowledge_config,
    set_knowledge_config,
)
from deerflow.knowledge.embeddings import (
    EMBEDDING_DIM,
    FAKE_MODEL_ID,
    DeterministicEmbeddingProvider,
    EmbeddingProviderError,
    backfill_experiments_embeddings,
    backfill_findings_embeddings,
    render_embeddable_text,
)
from deerflow.knowledge.retrieval.planner import (
    EDGE_TYPES,
    FailureSearchStore,
    LexicalSearchStore,
    RelationalExpansionStore,
    ScopeFilter,
    StructuredLookupStore,
    VectorSearchStore,
    execute_retrieval,
    plan_retrieval,
)
from deerflow.knowledge.schema.experiments import AssumptionRow, ExperimentRow
from deerflow.knowledge.schema.findings import FindingRow
from deerflow.knowledge.tools.lookup import (
    KnowledgeBackends,
    get_knowledge_backends,
    ledger_get,
    ledger_search,
    reset_knowledge_backends,
)
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

    def test_portable_cosine_finds_nearest_experiment(self, store: dict) -> None:
        provider = DeterministicEmbeddingProvider()
        finding_texts = pg.SQLFindingTextSource(store["factory"])
        finding_vectors = pg.SQLEmbeddingVectorStore(store["factory"], model_id=provider.model_id)
        backfill_findings_embeddings(provider, finding_texts, finding_vectors, [str(store["finding_id"])])
        experiment_texts = pg.SQLExperimentTextSource(store["factory"])
        experiment_vectors = pg.SQLExperimentEmbeddingVectorStore(store["factory"], model_id=provider.model_id)
        experiment_text = experiment_texts.get_experiment_text(str(store["experiment_id"]))
        assert experiment_text is not None
        [experiment_vector] = provider.embed_batch([experiment_text])
        experiment_vectors.upsert_embedding(str(store["experiment_id"]), experiment_vector, model_id=provider.model_id)
        backend = pg.SQLVectorSearchStore(store["factory"])
        found = backend.vector_search(experiment_vector, ScopeFilter(), kinds=["finding", "experiment"], limit=10)
        assert [item.id for item in found] == [str(store["experiment_id"]), str(store["finding_id"])]
        experiments_only = backend.vector_search(experiment_vector, ScopeFilter(), kinds=["experiment"], limit=10)
        assert [item.id for item in experiments_only] == [str(store["experiment_id"])]

    def test_failure_kind_draws_from_both_tables(self, store: dict) -> None:
        provider = DeterministicEmbeddingProvider()
        finding_texts = pg.SQLFindingTextSource(store["factory"])
        finding_vectors = pg.SQLEmbeddingVectorStore(store["factory"], model_id=provider.model_id)
        backfill_findings_embeddings(provider, finding_texts, finding_vectors, [str(store["finding_id"]), str(store["failure_finding_id"])])
        experiment_texts = pg.SQLExperimentTextSource(store["factory"])
        experiment_vectors = pg.SQLExperimentEmbeddingVectorStore(store["factory"], model_id=provider.model_id)
        for key in ("experiment_id", "failure_experiment_id"):
            text = experiment_texts.get_experiment_text(str(store[key]))
            assert text is not None
            [vector] = provider.embed_batch([text])
            experiment_vectors.upsert_embedding(str(store[key]), vector, model_id=provider.model_id)
        backend = pg.SQLVectorSearchStore(store["factory"])
        query_text = experiment_texts.get_experiment_text(str(store["failure_experiment_id"]))
        assert query_text is not None
        [query_vector] = provider.embed_batch([query_text])
        found = backend.vector_search(query_vector, ScopeFilter(), kinds=["failure"], limit=10)
        assert [item.id for item in found] == [str(store["failure_experiment_id"]), str(store["failure_finding_id"])]
        assert all(item.kind == "failure" for item in found)

    def test_unembedded_experiments_never_match(self, store: dict) -> None:
        provider = DeterministicEmbeddingProvider()
        finding_texts = pg.SQLFindingTextSource(store["factory"])
        finding_vectors = pg.SQLEmbeddingVectorStore(store["factory"], model_id=provider.model_id)
        backfill_findings_embeddings(provider, finding_texts, finding_vectors, [str(store["finding_id"])])
        backend = pg.SQLVectorSearchStore(store["factory"])
        experiment_text = pg.SQLExperimentTextSource(store["factory"]).get_experiment_text(str(store["experiment_id"]))
        assert experiment_text is not None
        [query_vector] = provider.embed_batch([experiment_text])
        # The NULL experiment row stays invisible even for its own text;
        # the embedded finding still matches the shared query.
        assert backend.vector_search(query_vector, ScopeFilter(), kinds=["experiment"], limit=10) == []
        found = backend.vector_search(query_vector, ScopeFilter(), kinds=["finding", "experiment"], limit=10)
        assert [item.id for item in found] == [str(store["finding_id"])]

    def test_scope_screening_applies_to_experiments(self, store: dict) -> None:
        provider = DeterministicEmbeddingProvider()
        experiment_texts = pg.SQLExperimentTextSource(store["factory"])
        experiment_vectors = pg.SQLExperimentEmbeddingVectorStore(store["factory"], model_id=provider.model_id)
        text = experiment_texts.get_experiment_text(str(store["experiment_id"]))
        assert text is not None
        [query_vector] = provider.embed_batch([text])
        experiment_vectors.upsert_embedding(str(store["experiment_id"]), query_vector, model_id=provider.model_id)
        backend = pg.SQLVectorSearchStore(store["factory"])
        assert backend.vector_search(query_vector, _scope(), kinds=["experiment"], limit=10) != []
        assert backend.vector_search(query_vector, _scope(asset_class="credit"), kinds=["experiment"], limit=10) == []

    def test_malformed_experiment_embedding_is_skipped(self, store: dict, caplog: pytest.LogCaptureFixture) -> None:
        provider = DeterministicEmbeddingProvider()
        experiment_texts = pg.SQLExperimentTextSource(store["factory"])
        experiment_vectors = pg.SQLExperimentEmbeddingVectorStore(store["factory"], model_id=provider.model_id)
        text = experiment_texts.get_experiment_text(str(store["failure_experiment_id"]))
        assert text is not None
        [query_vector] = provider.embed_batch([text])
        experiment_vectors.upsert_embedding(str(store["failure_experiment_id"]), query_vector, model_id=provider.model_id)
        with store["factory"]() as session:
            row = session.get(ExperimentRow, store["experiment_id"])
            assert row is not None
            row.embedding = "not-a-vector"  # type: ignore[assignment]
            session.commit()
        backend = pg.SQLVectorSearchStore(store["factory"])
        with caplog.at_level("WARNING", logger="deerflow.knowledge.pg_retrieval"):
            found = backend.vector_search(query_vector, ScopeFilter(), kinds=["experiment", "failure"], limit=10)
        assert [item.id for item in found] == [str(store["failure_experiment_id"])]
        assert "malformed stored embedding" in caplog.text

    def test_limit_trims_merged_tables(self, store: dict) -> None:
        provider = DeterministicEmbeddingProvider()
        finding_texts = pg.SQLFindingTextSource(store["factory"])
        finding_vectors = pg.SQLEmbeddingVectorStore(store["factory"], model_id=provider.model_id)
        backfill_findings_embeddings(provider, finding_texts, finding_vectors, [str(store["finding_id"]), str(store["failure_finding_id"])])
        experiment_texts = pg.SQLExperimentTextSource(store["factory"])
        experiment_vectors = pg.SQLExperimentEmbeddingVectorStore(store["factory"], model_id=provider.model_id)
        for key in ("experiment_id", "failure_experiment_id"):
            text = experiment_texts.get_experiment_text(str(store[key]))
            assert text is not None
            [vector] = provider.embed_batch([text])
            experiment_vectors.upsert_embedding(str(store[key]), vector, model_id=provider.model_id)
        backend = pg.SQLVectorSearchStore(store["factory"])
        query_vector = provider.embed_one("momentum backtest sharpe")
        assert len(backend.vector_search(query_vector, ScopeFilter(), kinds=["finding", "experiment", "failure"], limit=2)) == 2
        assert len(backend.vector_search(query_vector, ScopeFilter(), kinds=["finding", "experiment", "failure"], limit=10)) == 4


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

    def test_experiment_text_source_round_trip(self, store: dict) -> None:
        texts = pg.SQLExperimentTextSource(store["factory"])
        expected = render_embeddable_text(
            title="Cross-sectional momentum 126-day backtest",
            body="Cross-sectional momentum 126-day backtest quintile long-short momentum backtest sharpe turnover mom_126d success",
        )
        assert texts.get_experiment_text(str(store["experiment_id"])) == expected
        assert texts.get_experiment_text(str(uuid.uuid4())) is None
        with pytest.raises(KnowledgeValidationError):
            texts.get_experiment_text("nope")

    def test_experiment_vector_store_round_trip_and_model_report(self, store: dict) -> None:
        provider = DeterministicEmbeddingProvider()
        vectors = pg.SQLExperimentEmbeddingVectorStore(store["factory"], model_id=provider.model_id)
        experiment_id = str(store["experiment_id"])
        assert vectors.get_embedding_model(experiment_id) is None
        [vector] = provider.embed_batch(["momentum"])
        vectors.upsert_embedding(experiment_id, vector, model_id=provider.model_id)
        assert vectors.get_embedding_model(experiment_id) == provider.model_id
        assert vectors.get_embedding_model(str(uuid.uuid4())) is None

    def test_experiment_vector_store_refuses_bad_writes(self, store: dict) -> None:
        vectors = pg.SQLExperimentEmbeddingVectorStore(store["factory"], model_id=FAKE_MODEL_ID)
        experiment_id = str(store["experiment_id"])
        with pytest.raises(KnowledgeValidationError):
            vectors.upsert_embedding(experiment_id, [0.1, 0.2], model_id=FAKE_MODEL_ID)
        [vector] = DeterministicEmbeddingProvider().embed_batch(["momentum"])
        with pytest.raises(KnowledgeValidationError):
            vectors.upsert_embedding(experiment_id, vector, model_id="other-model/v1")
        with pytest.raises(KnowledgeError):
            vectors.upsert_embedding(str(uuid.uuid4()), vector, model_id=FAKE_MODEL_ID)
        with pytest.raises(KnowledgeValidationError):
            vectors.upsert_embedding("nope", vector, model_id=FAKE_MODEL_ID)
        with pytest.raises(KnowledgeValidationError):
            vectors.get_embedding_model("nope")
        with pytest.raises(KnowledgeValidationError):
            pg.SQLExperimentEmbeddingVectorStore(store["factory"], model_id="  ")

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

    def test_search_with_test_fake_provider_covers_experiments(self, store: dict) -> None:
        provider = DeterministicEmbeddingProvider()
        backfill_findings_embeddings(
            provider,
            pg.SQLFindingTextSource(store["factory"]),
            pg.SQLEmbeddingVectorStore(store["factory"], model_id=provider.model_id),
            [str(store["finding_id"]), str(store["failure_finding_id"])],
        )
        experiment_texts = pg.SQLExperimentTextSource(store["factory"])
        experiment_vectors = pg.SQLExperimentEmbeddingVectorStore(store["factory"], model_id=provider.model_id)
        for key in ("experiment_id", "failure_experiment_id"):
            text = experiment_texts.get_experiment_text(str(store[key]))
            assert text is not None
            [vector] = provider.embed_batch([text])
            experiment_vectors.upsert_embedding(str(store[key]), vector, model_id=provider.model_id)
        service = pg.KnowledgeRetrievalService(store["factory"], embedding_provider=provider)
        page = service.search("momentum backtest", kinds=("finding", "experiment"), filters={}, limit=10, offset=0)
        by_id = {doc.id: doc for doc in page.documents}
        assert str(store["experiment_id"]) in by_id
        assert "vector" in by_id[str(store["experiment_id"])].payload["channels"]
        assert "vector" in by_id[str(store["finding_id"])].payload["channels"]

    def test_search_without_provider_skips_vector(self, store: dict) -> None:
        service = pg.KnowledgeRetrievalService(store["factory"])
        assert service.embedding_provider is None
        page = service.search("momentum", kinds=("finding", "experiment"), filters={}, limit=10, offset=0)
        assert page.total and page.total >= 1

    def test_tool_functions_run_over_the_service(self, store: dict) -> None:
        service = pg.KnowledgeRetrievalService(store["factory"])
        payload = ledger_search("borrow costs", kinds="failure", backend=service)
        assert payload["count"] == 2
        fetched = ledger_get([str(store["experiment_id"]), str(uuid.uuid4())], backend=service)
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

    def test_experiment_vector_statement_renders_pgvector_operator(self) -> None:
        dialect = postgresql.dialect()
        sql = str(pg._experiment_vector_statement("[0.1,0.2]").compile(dialect=dialect))
        assert "<=>" in sql and "VECTOR(768)" in sql and "embedding IS NOT NULL" in sql
        assert "experiment" in sql and "started_at" in sql

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


class TestBindFromConfig:
    @pytest.fixture(autouse=True)
    def _clean_state(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        """Isolate the config singleton, backend registry, and env overrides."""
        monkeypatch.delenv("DEER_FLOW_KNOWLEDGE_ENABLED", raising=False)
        monkeypatch.delenv("DEER_FLOW_KNOWLEDGE_DSN", raising=False)
        monkeypatch.delenv("DEER_FLOW_KNOWLEDGE_EMBEDDING_MODEL", raising=False)
        previous = get_knowledge_config()
        try:
            yield
        finally:
            reset_knowledge_backends()
            set_knowledge_config(previous)

    def test_disabled_leaves_registry_untouched(self) -> None:
        set_knowledge_config(KnowledgeConfig(enabled=False, database_dsn="sqlite:////tmp/never.db"))
        assert pg.bind_knowledge_backends_from_config() is None
        assert get_knowledge_backends() == KnowledgeBackends()

    def test_missing_dsn_leaves_registry_untouched(self) -> None:
        set_knowledge_config(KnowledgeConfig())
        assert pg.bind_knowledge_backends_from_config() is None
        assert get_knowledge_backends() == KnowledgeBackends()

    def test_binds_sqlite_without_vector(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path, "bind_no_vector.db")
        engine.dispose()
        set_knowledge_config(KnowledgeConfig(database_dsn=f"sqlite:///{(tmp_path / 'bind_no_vector.db').as_posix()}"))
        bound = pg.bind_knowledge_backends_from_config()
        assert bound is not None
        assert isinstance(bound.retrieval, pg.KnowledgeRetrievalService)
        assert isinstance(bound.experiments, pg.SQLExperimentLookupStore)
        assert bound.retrieval.embedding_provider is None  # no silent test-fake
        assert get_knowledge_backends().retrieval is bound.retrieval
        assert get_knowledge_backends().experiments is bound.experiments

    def test_binds_with_fake_model_and_runs_two_table_vector(self, store: dict, tmp_path: Path) -> None:
        set_knowledge_config(KnowledgeConfig(database_dsn=f"sqlite:///{(tmp_path / 'kb_retrieval.db').as_posix()}", embedding_model="test-fake/v1"))
        bound = pg.bind_knowledge_backends_from_config()
        assert bound is not None and bound.retrieval is not None
        assert bound.retrieval.embedding_provider is not None
        assert bound.retrieval.embedding_provider.model_id == FAKE_MODEL_ID
        provider = DeterministicEmbeddingProvider()
        backfill_findings_embeddings(
            provider,
            pg.SQLFindingTextSource(store["factory"]),
            pg.SQLEmbeddingVectorStore(store["factory"], model_id=provider.model_id),
            [str(store["finding_id"]), str(store["failure_finding_id"])],
        )
        backfill_experiments_embeddings(
            provider,
            pg.SQLExperimentTextSource(store["factory"]),
            pg.SQLExperimentEmbeddingVectorStore(store["factory"], model_id=provider.model_id),
            [str(store["experiment_id"]), str(store["failure_experiment_id"])],
        )
        page = bound.retrieval.search("momentum backtest", kinds=("finding", "experiment"), filters={}, limit=10, offset=0)
        by_id = {doc.id: doc for doc in page.documents}
        assert str(store["finding_id"]) in by_id
        assert str(store["experiment_id"]) in by_id
        assert "vector" in by_id[str(store["finding_id"])].payload["channels"]
        assert "vector" in by_id[str(store["experiment_id"])].payload["channels"]

    def test_embedding_model_env_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _engine(tmp_path, "bind_env_model.db")
        engine.dispose()
        monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_EMBEDDING_MODEL", "test-fake/v1")
        set_knowledge_config(KnowledgeConfig(database_dsn=f"sqlite:///{(tmp_path / 'bind_env_model.db').as_posix()}"))
        bound = pg.bind_knowledge_backends_from_config()
        assert bound is not None and bound.retrieval is not None
        assert bound.retrieval.embedding_provider is not None
        assert bound.retrieval.embedding_provider.model_id == FAKE_MODEL_ID

    def test_existing_factory_not_overwritten(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path, "bind_custom_factory.db")
        engine.dispose()
        sentinel = DeterministicEmbeddingProvider(model_id="custom-model/v1")

        def factory():
            return sentinel

        embeddings_mod.register_provider_factory("custom-model/v1", factory)
        try:
            set_knowledge_config(KnowledgeConfig(database_dsn=f"sqlite:///{(tmp_path / 'bind_custom_factory.db').as_posix()}", embedding_model="custom-model/v1"))
            bound = pg.bind_knowledge_backends_from_config()
            assert bound is not None and bound.retrieval is not None
            assert bound.retrieval.embedding_provider is sentinel
            assert embeddings_mod._provider_factories["custom-model/v1"] is factory
        finally:
            embeddings_mod._provider_factories.pop("custom-model/v1", None)

    def test_unloadable_model_degrades_without_vector(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        engine = _engine(tmp_path, "bind_bad_model.db")
        engine.dispose()

        class _Unavailable:
            def __init__(self, model_id: str = "missing-model-xyz") -> None:
                raise EmbeddingProviderError(f"checkpoint {model_id!r} unavailable")

        monkeypatch.setattr("deerflow.knowledge.providers_st.SentenceTransformerEmbeddingProvider", _Unavailable)
        set_knowledge_config(KnowledgeConfig(database_dsn=f"sqlite:///{(tmp_path / 'bind_bad_model.db').as_posix()}", embedding_model="missing-model-xyz"))
        try:
            with caplog.at_level("WARNING", logger="deerflow.knowledge.pg_retrieval"):
                bound = pg.bind_knowledge_backends_from_config()
        finally:
            embeddings_mod._provider_factories.pop("missing-model-xyz", None)
        assert bound is not None and bound.retrieval is not None
        assert bound.retrieval.embedding_provider is None
        assert "without the vector channel" in caplog.text
        page = bound.retrieval.search("momentum", kinds=("finding",), filters={}, limit=5, offset=0)
        assert page.total == 0  # empty seed DB; lexical still runs, nothing matches


#: Shared research KB. Tests below open it read-only only (``mode=ro`` URI);
#: any write would fail the hash pin — and sqlite itself refuses writes.
REAL_KB_PATH = Path("/home/fire/Documents/Audit/kb/quantflow_kb.db")


def _file_fingerprint(path: Path) -> tuple[str, int]:
    """Return (sha256, mtime_ns) for a read-only tamper check."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest(), os.stat(path).st_mtime_ns


class TestRealKbReadOnly:
    def test_finding_channels_over_real_kb_leave_file_untouched(self) -> None:
        if not REAL_KB_PATH.exists():
            pytest.skip(f"shared KB not present: {REAL_KB_PATH}")
        before = _file_fingerprint(REAL_KB_PATH)
        # Raw engine, NOT the process-wide sync-engine cache: its connect
        # listener runs PRAGMA journal_mode=WAL, which a read-only
        # connection cannot take. The ro URI makes writes impossible.
        engine = sa.create_engine(f"sqlite:///file:{REAL_KB_PATH.as_posix()}?mode=ro&uri=true")
        try:
            factory = sessionmaker(engine, expire_on_commit=False)
            with factory() as session:
                finding_count = session.execute(sa.text("SELECT COUNT(*) FROM finding")).scalar()
                assert finding_count is not None and finding_count > 0
            plan = plan_retrieval({"topic": "momentum", "needed_memory": ["validated_findings"]}, kinds=("finding",), relational=False)
            assert "vector" in plan.channels
            [query_embedding] = DeterministicEmbeddingProvider().embed_batch(["momentum factor persistence"])
            result = execute_retrieval(
                plan,
                query_text="momentum",
                structured=pg.SQLStructuredLookupStore(factory),
                lexical=pg.SQLLexicalSearchStore(factory),
                vector=pg.SQLVectorSearchStore(factory),
                failures=pg.SQLFailureSearchStore(factory),
                query_embedding=query_embedding,
            )
            assert result.channel_counts["structured"] > 0
            assert result.channel_counts["lexical"] > 0
            assert result.channel_counts["vector"] > 0
            assert "vector channel skipped" not in " ".join(result.warnings)
            with factory() as session, pytest.raises(OperationalError):
                session.execute(sa.text("CREATE TABLE _must_not_exist (x)"))
                session.commit()
        finally:
            engine.dispose()
        assert _file_fingerprint(REAL_KB_PATH) == before


LINKED_FAMILY_PARENT = "c0" * 32
LINKED_FAMILY_CHILD = "c1" * 32
LINKED_FAMILY_REPLICA = "c2" * 32
LINKED_FAMILY_OTHER = "c3" * 32
LINKED_FAMILY_LONELY = "c4" * 32
LINKED_FAMILY_RESCUE = "c5" * 32


def _linked_methodology() -> dict:
    return {
        "asset_class": "equity",
        "market": "US",
        "universe": "sp500-pit",
        "horizon": "6m",
        "frequency": "daily",
        "sample_period": ["2015-01-01", "2026-09-19"],
    }


@pytest.fixture
def linked(tmp_path: Path) -> Iterator[dict]:
    """Throwaway DB with a linked experiment/finding/assumption graph.

    Graph (all edge kinds exercised):

    * ``parent`` — root experiment (family P): child + rescue derive from
      it, replica replicates it, sibling shares its family, other shares
      its dataset version + artifact, one assumption applies to it;
    * ``child`` — derived_from parent, outcome failure (experiment-with-a
      failure neighbor);
    * ``rescue`` — derived_from parent, but declares only horizon/frequency
      scope and lexically misses the service-test query, so pass 1 cannot
      return it and only the relational channel surfaces it;
    * ``lonely`` — uses a dataset version nothing else touches (no
      neighbors at all);
    * ``finding_new`` supersedes ``finding_old``;
    * one linked assumption (parent) and one orphan (no experiment).
    """
    engine = sa.create_engine(f"sqlite:///{(tmp_path / 'kb_relational.db').as_posix()}")
    for table in KB_TABLES + ("artifact", "dataset_version"):
        Base.metadata.tables[table].create(engine, checkfirst=True)
    factory = sessionmaker(engine, expire_on_commit=False)
    project_id = uuid.uuid4()
    run_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    child_id = uuid.uuid4()
    replica_id = uuid.uuid4()
    sibling_id = uuid.uuid4()
    other_id = uuid.uuid4()
    lonely_id = uuid.uuid4()
    rescue_id = uuid.uuid4()
    old_id = uuid.uuid4()
    new_id = uuid.uuid4()
    linked_assumption_id = uuid.uuid4()
    orphan_assumption_id = uuid.uuid4()
    shared_dataset_id = uuid.uuid4()
    solo_dataset_id = uuid.uuid4()
    shared_artifact_id = uuid.uuid4()
    with factory() as session:
        session.add(kb_schema.ResearchProjectRow(id=project_id, name="relational", visibility_scope={}))
        session.add(kb_schema.AgentRunRow(id=run_id, project_id=project_id, agent_type="research", task="relational expansion", status="running"))

        def _experiment(
            key: uuid.UUID,
            hypothesis: str,
            family: str,
            execution: str,
            started: datetime,
            *,
            parent: uuid.UUID | None = None,
            replicated: uuid.UUID | None = None,
            outcome: str = "success",
            failure_class: str | None = None,
            methodology: dict | None = None,
        ) -> None:
            session.add(
                ExperimentRow(
                    id=key,
                    project_id=project_id,
                    created_by_run_id=run_id,
                    experiment_family_hash=family,
                    execution_hash=execution,
                    hypothesis=hypothesis,
                    methodology=methodology if methodology is not None else _linked_methodology(),
                    parameters={},
                    metrics=None,
                    outcome=outcome,
                    failure_class=failure_class,
                    status="completed",
                    parent_experiment_id=parent,
                    replicated_experiment_id=replicated,
                    started_at=started,
                    completed_at=started,
                )
            )

        _experiment(parent_id, "Cross-sectional momentum 126-day backtest", LINKED_FAMILY_PARENT, "e0" * 32, datetime(2026, 1, 10, tzinfo=UTC))
        _experiment(
            child_id,
            "Momentum turnover stress test",
            LINKED_FAMILY_CHILD,
            "e1" * 32,
            datetime(2026, 2, 10, tzinfo=UTC),
            parent=parent_id,
            outcome="failure",
            failure_class="execution",
        )
        _experiment(replica_id, "Momentum 126-day independent replication", LINKED_FAMILY_REPLICA, "e2" * 32, datetime(2026, 3, 10, tzinfo=UTC), replicated=parent_id)
        _experiment(sibling_id, "Sector-neutral momentum variant backtest", LINKED_FAMILY_PARENT, "e3" * 32, datetime(2026, 4, 10, tzinfo=UTC))
        _experiment(other_id, "Corporate bond liquidity provision study", LINKED_FAMILY_OTHER, "e4" * 32, datetime(2026, 5, 10, tzinfo=UTC))
        _experiment(lonely_id, "Unlinked volatility regime study", LINKED_FAMILY_LONELY, "e5" * 32, datetime(2026, 6, 10, tzinfo=UTC))
        _experiment(
            rescue_id,
            "Overnight reversal microstructure examination",
            LINKED_FAMILY_RESCUE,
            "e6" * 32,
            datetime(2026, 7, 10, tzinfo=UTC),
            parent=parent_id,
            methodology={"horizon": "6m", "frequency": "daily"},
        )
        session.add(
            FindingRow(
                id=old_id,
                project_id=project_id,
                canonical_key="empirical:relational-old-claim",
                finding_type="empirical",
                statement="Older claim about momentum persistence.",
                scope={"asset_class": "equity", "market": "US", "universe": "sp500-pit", "horizon": "6m", "frequency": "daily"},
                status="candidate",
                confidence={},
                recorded_at=datetime(2026, 3, 1, tzinfo=UTC),
                created_by_run_id=run_id,
            )
        )
        session.add(
            FindingRow(
                id=new_id,
                project_id=project_id,
                canonical_key="empirical:relational-new-claim",
                finding_type="empirical",
                statement="Revised claim about momentum persistence.",
                scope={"asset_class": "equity", "market": "US", "universe": "sp500-pit", "horizon": "6m", "frequency": "daily"},
                status="candidate",
                confidence={},
                recorded_at=datetime(2026, 4, 1, tzinfo=UTC),
                created_by_run_id=run_id,
                supersedes_id=old_id,
            )
        )
        session.add(
            AssumptionRow(
                id=linked_assumption_id,
                experiment_id=parent_id,
                statement="Closing auction volume absorbs the rebalanced notional.",
                category="execution",
                sensitivity="high",
                tested=False,
                status="active",
            )
        )
        session.add(
            AssumptionRow(
                id=orphan_assumption_id,
                experiment_id=None,
                statement="Borrow is available at the quoted rate.",
                category="market",
                sensitivity="medium",
                tested=False,
                status="active",
            )
        )
        session.add(kb_schema.DatasetVersionRow(id=shared_dataset_id, dataset_key="test:shared:features", provider="test"))
        session.add(kb_schema.DatasetVersionRow(id=solo_dataset_id, dataset_key="test:solo:features", provider="test"))
        session.add(
            kb_schema.ArtifactRow(
                id=shared_artifact_id,
                sha256="ab" * 32,
                kind="result",
                storage_uri="artifact://test/shared",
                created_by_run_id=run_id,
                artifact_metadata={},
            )
        )
        session.add(kb_schema.ExperimentDatasetRow(experiment_id=parent_id, dataset_version_id=shared_dataset_id, role="features"))
        session.add(kb_schema.ExperimentDatasetRow(experiment_id=other_id, dataset_version_id=shared_dataset_id, role="features"))
        session.add(kb_schema.ExperimentDatasetRow(experiment_id=lonely_id, dataset_version_id=solo_dataset_id, role="features"))
        session.add(kb_schema.ExperimentArtifactRow(experiment_id=parent_id, artifact_id=shared_artifact_id, role="result"))
        session.add(kb_schema.ExperimentArtifactRow(experiment_id=other_id, artifact_id=shared_artifact_id, role="result"))
        session.commit()
    try:
        yield {
            "factory": factory,
            "engine": engine,
            "parent_id": parent_id,
            "child_id": child_id,
            "replica_id": replica_id,
            "sibling_id": sibling_id,
            "other_id": other_id,
            "lonely_id": lonely_id,
            "rescue_id": rescue_id,
            "old_id": old_id,
            "new_id": new_id,
            "linked_assumption_id": linked_assumption_id,
            "orphan_assumption_id": orphan_assumption_id,
        }
    finally:
        engine.dispose()


class TestRelationalExpansion:
    def test_satisfies_the_planner_protocol(self, linked: dict) -> None:
        assert isinstance(pg.SQLRelationalExpansionStore(linked["factory"]), RelationalExpansionStore)

    def test_parent_seed_reaches_every_mapped_edge_kind(self, linked: dict) -> None:
        backend = pg.SQLRelationalExpansionStore(linked["factory"])
        found = backend.expand_neighbors([str(linked["parent_id"])], _scope(), edge_types=list(EDGE_TYPES), limit=50)
        # child + rescue (derived_from), replica (replicates), sibling
        # (related_to), other (uses, via both the shared dataset and the
        # shared artifact — deduped to one hit), assumption (applies_to).
        # The failed child is included: expansion has no kinds filter.
        assert [item.id for item in found].count(str(linked["other_id"])) == 1
        assert {item.id for item in found} == {
            str(linked["child_id"]),
            str(linked["rescue_id"]),
            str(linked["replica_id"]),
            str(linked["sibling_id"]),
            str(linked["other_id"]),
            str(linked["linked_assumption_id"]),
        }
        assert {item.id: item.kind for item in found}[str(linked["child_id"])] == "failure"

    def test_finding_supersedes_edges_run_both_directions(self, linked: dict) -> None:
        backend = pg.SQLRelationalExpansionStore(linked["factory"])
        forward = backend.expand_neighbors([str(linked["new_id"])], _scope(), edge_types=["supersedes"], limit=10)
        assert [item.id for item in forward] == [str(linked["old_id"])]
        backward = backend.expand_neighbors([str(linked["old_id"])], _scope(), edge_types=["supersedes"], limit=10)
        assert [item.id for item in backward] == [str(linked["new_id"])]

    def test_assumption_seed_reaches_its_experiment(self, linked: dict) -> None:
        backend = pg.SQLRelationalExpansionStore(linked["factory"])
        found = backend.expand_neighbors([str(linked["linked_assumption_id"])], _scope(), edge_types=["applies_to"], limit=10)
        assert [item.id for item in found] == [str(linked["parent_id"])]
        assert backend.expand_neighbors([str(linked["orphan_assumption_id"])], _scope(), edge_types=list(EDGE_TYPES), limit=10) == []

    def test_edge_types_filter_the_traversal(self, linked: dict) -> None:
        backend = pg.SQLRelationalExpansionStore(linked["factory"])
        seed = [str(linked["parent_id"])]
        derived = backend.expand_neighbors(seed, _scope(), edge_types=["derived_from"], limit=50)
        assert {item.id for item in derived} == {str(linked["child_id"]), str(linked["rescue_id"])}
        uses = backend.expand_neighbors(seed, _scope(), edge_types=["uses"], limit=50)
        assert [item.id for item in uses] == [str(linked["other_id"])]
        # supports/contradicts have no backing relation: accepted, empty.
        assert backend.expand_neighbors(seed, _scope(), edge_types=["supports", "contradicts"], limit=50) == []
        assert backend.expand_neighbors(seed, _scope(), edge_types=[], limit=50) == []

    def test_unknown_edge_types_are_rejected(self, linked: dict) -> None:
        backend = pg.SQLRelationalExpansionStore(linked["factory"])
        with pytest.raises(KnowledgeValidationError):
            backend.expand_neighbors([str(linked["parent_id"])], _scope(), edge_types=["cites"], limit=10)
        with pytest.raises(KnowledgeValidationError):
            backend.expand_neighbors([str(linked["parent_id"])], _scope(), edge_types="derived_from", limit=10)  # type: ignore[arg-type]

    def test_single_pass_bound_excludes_two_hop_neighbors(self, linked: dict) -> None:
        backend = pg.SQLRelationalExpansionStore(linked["factory"])
        # The child reaches only its parent: the sibling, replica, other,
        # and assumption are two hops away and must stay out.
        found = backend.expand_neighbors([str(linked["child_id"])], _scope(), edge_types=list(EDGE_TYPES), limit=50)
        assert [item.id for item in found] == [str(linked["parent_id"])]
        found = backend.expand_neighbors([str(linked["sibling_id"])], _scope(), edge_types=list(EDGE_TYPES), limit=50)
        assert [item.id for item in found] == [str(linked["parent_id"])]

    def test_empty_graph_and_unknown_seeds_return_nothing(self, linked: dict) -> None:
        backend = pg.SQLRelationalExpansionStore(linked["factory"])
        assert backend.expand_neighbors([str(linked["lonely_id"])], _scope(), edge_types=list(EDGE_TYPES), limit=50) == []
        assert backend.expand_neighbors([str(uuid.uuid4())], _scope(), edge_types=list(EDGE_TYPES), limit=50) == []
        assert backend.expand_neighbors(["not-a-uuid"], _scope(), edge_types=list(EDGE_TYPES), limit=50) == []
        assert backend.expand_neighbors([], _scope(), edge_types=list(EDGE_TYPES), limit=50) == []

    def test_seeds_are_excluded_from_results(self, linked: dict) -> None:
        backend = pg.SQLRelationalExpansionStore(linked["factory"])
        found = backend.expand_neighbors([str(linked["parent_id"]), str(linked["child_id"])], _scope(), edge_types=list(EDGE_TYPES), limit=50)
        ids = {item.id for item in found}
        assert str(linked["parent_id"]) not in ids
        assert str(linked["child_id"]) not in ids
        assert ids == {
            str(linked["rescue_id"]),
            str(linked["replica_id"]),
            str(linked["sibling_id"]),
            str(linked["other_id"]),
            str(linked["linked_assumption_id"]),
        }

    def test_scope_screening_matches_fusion(self, linked: dict) -> None:
        backend = pg.SQLRelationalExpansionStore(linked["factory"])
        # Under an asset-class mismatch only scope-neutral neighbors survive:
        # the assumption (no scope at all) and the rescue row (declares
        # horizon/frequency only, no asset class).
        found = backend.expand_neighbors([str(linked["parent_id"])], _scope(asset_class="credit"), edge_types=list(EDGE_TYPES), limit=50)
        assert {item.id for item in found} == {str(linked["linked_assumption_id"]), str(linked["rescue_id"])}

    def test_limit_trims(self, linked: dict) -> None:
        backend = pg.SQLRelationalExpansionStore(linked["factory"])
        found = backend.expand_neighbors([str(linked["parent_id"])], _scope(), edge_types=list(EDGE_TYPES), limit=2)
        assert len(found) == 2

    def test_seed_order_ranks_first(self, linked: dict) -> None:
        backend = pg.SQLRelationalExpansionStore(linked["factory"])
        found = backend.expand_neighbors([str(linked["new_id"]), str(linked["child_id"])], _scope(), edge_types=list(EDGE_TYPES), limit=50)
        # The finding neighbor (seed 0) outranks the experiment neighbor
        # (seed 1) regardless of recency.
        assert [item.id for item in found] == [str(linked["old_id"]), str(linked["parent_id"])]

    def test_service_search_surfaces_expansion_only_neighbors(self, linked: dict) -> None:
        service = pg.KnowledgeRetrievalService(linked["factory"])
        page = service.search("momentum 126-day backtest", kinds=("experiment",), filters={"market": "US"}, limit=10, offset=0)
        by_id = {doc.id: doc for doc in page.documents}
        # The rescue row is invisible to pass 1 (structured prunes it: it
        # declares scope but no market; lexical scores it 0), yet it is
        # scope-compatible at fusion — so only the relational channel
        # surfaces it.
        assert str(linked["parent_id"]) in by_id
        assert str(linked["rescue_id"]) in by_id
        assert by_id[str(linked["rescue_id"])].payload["channels"] == ["relational"]

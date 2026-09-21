"""Tests for the Phase 2 integration wiring (no stubs, all seams connected).

Covers the integration-owned seams between the Phase 2 builders and the
harness (mirroring the Phase 1 wiring test style):

* package surface: ``FindingRow`` re-exported from
  ``deerflow.knowledge.schema`` and registered as ``KnowledgeFindingRow``
  in ``deerflow.persistence.models``; ``retrieval``/``tools``/``embeddings``
  lazily wired on ``deerflow.knowledge`` without dragging ``deerflow.config``
  or langchain into the import;
* run start: ``RuntimeFeatures.knowledge`` (default off, opt-in assembly)
  and the lead-agent ``build_middlewares`` chain (bootstrap middleware
  always present, right after ``MemoryMiddleware``);
* bootstrap parity: the middleware injects a context packet through the
  real PG-bound retrieval service (SQLite dialect) with zero fakes, and
  degrades to no-injection when the plane is disabled or unbound.
"""

from __future__ import annotations

import subprocess
import sys
import types
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

import deerflow.knowledge as knowledge_pkg
import deerflow.knowledge.schema as kb_schema
from deerflow.knowledge import pg_retrieval as pg
from deerflow.knowledge.config import KnowledgeConfig, get_knowledge_config, set_knowledge_config
from deerflow.knowledge.schema.experiments import ExperimentRow
from deerflow.knowledge.schema.findings import FindingRow
from deerflow.knowledge.tools.lookup import bind_knowledge_backends, reset_knowledge_backends
from deerflow.knowledge.tools.middleware import (
    KNOWLEDGE_BOOTSTRAP_MARKER,
    KNOWLEDGE_BOOTSTRAP_MESSAGE_ID,
    KnowledgeBootstrapMiddleware,
)
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


@pytest.fixture
def clean_registry():
    """Restore backend bindings and knowledge config after each test."""
    previous = get_knowledge_config()
    try:
        yield
    finally:
        reset_knowledge_backends()
        set_knowledge_config(previous)


def _make_runtime(**context):
    return types.SimpleNamespace(context=dict(context), server_info=None)


class TestPackageWiring:
    def test_schema_reexports_finding(self) -> None:
        assert kb_schema.FindingRow is FindingRow
        assert FindingRow.__tablename__ == "finding"
        assert kb_schema.FINDING_TYPES == ("empirical", "methodological", "data_quality", "failure", "prior")
        assert kb_schema.FINDING_STATUSES_PHASE2 == ("candidate",)
        assert kb_schema.EMBEDDING_DIMENSIONS == 768
        for name in ("FindingRow", "FINDING_TYPES", "FINDING_STATUSES", "EMBEDDING_DIMENSIONS", "EmbeddingVector", "tsvector"):
            assert name in kb_schema.__all__

    def test_persistence_models_register_finding_alias(self) -> None:
        from deerflow.persistence.models import KnowledgeFindingRow

        assert KnowledgeFindingRow is FindingRow
        assert "finding" in Base.metadata.tables

    def test_knowledge_package_lazy_wiring(self) -> None:
        for name in ("retrieval", "tools", "embeddings", "pg_retrieval"):
            assert name in knowledge_pkg.__all__
            assert name in knowledge_pkg._LAZY_SUBMODULES
        assert knowledge_pkg.retrieval.__name__ == "deerflow.knowledge.retrieval"
        assert knowledge_pkg.embeddings.__name__ == "deerflow.knowledge.embeddings"
        assert knowledge_pkg.tools.__name__ == "deerflow.knowledge.tools"
        assert knowledge_pkg.pg_retrieval.__name__ == "deerflow.knowledge.pg_retrieval"

    def test_knowledge_import_stays_light(self) -> None:
        probe = (
            "import sys; import deerflow.knowledge; "
            "heavy = sorted(m for m in sys.modules "
            "if m.split('.')[0] in ('langchain', 'langchain_core', 'langgraph', 'sqlalchemy') "
            "or m == 'deerflow.config' or m.startswith('deerflow.config.')); "
            "print(','.join(heavy))"
        )
        completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=Path(__file__).resolve().parent, timeout=120)
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "", f"heavy modules leaked into deerflow.knowledge import: {completed.stdout.strip()}"


class TestRunStartWiring:
    def test_knowledge_feature_defaults_off(self) -> None:
        from deerflow.agents.features import RuntimeFeatures

        assert RuntimeFeatures().knowledge is False

    def test_factory_assembles_knowledge_middleware(self) -> None:
        from deerflow.agents.factory import _assemble_from_features
        from deerflow.agents.features import RuntimeFeatures

        chain, _ = _assemble_from_features(RuntimeFeatures(memory=False, knowledge=True), name="probe")
        assert KnowledgeBootstrapMiddleware in [type(middleware) for middleware in chain]
        chain_off, _ = _assemble_from_features(RuntimeFeatures(memory=False), name="probe")
        assert KnowledgeBootstrapMiddleware not in [type(middleware) for middleware in chain_off]

    def test_factory_accepts_custom_knowledge_middleware(self) -> None:
        from deerflow.agents.factory import _assemble_from_features
        from deerflow.agents.features import RuntimeFeatures

        custom = KnowledgeBootstrapMiddleware(enabled=False)
        chain, _ = _assemble_from_features(RuntimeFeatures(memory=False, knowledge=custom), name="probe")
        assert custom in chain

    def test_lead_agent_chain_contains_bootstrap_after_memory(self) -> None:
        from deerflow.agents.lead_agent.agent import build_middlewares
        from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware
        from deerflow.config.app_config import AppConfig
        from deerflow.config.sandbox_config import SandboxConfig

        middlewares = build_middlewares(
            config={"configurable": {}},
            model_name="gpt-4o",
            app_config=AppConfig(sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider")),
        )
        kinds = [type(middleware) for middleware in middlewares]
        assert KnowledgeBootstrapMiddleware in kinds
        assert kinds.index(KnowledgeBootstrapMiddleware) == kinds.index(MemoryMiddleware) + 1


@pytest.fixture
def seeded_factory(tmp_path: Path) -> Iterator[dict]:
    """Session factory over a scratch SQLite DB with one finding + experiment."""
    engine = sa.create_engine(f"sqlite:///{(tmp_path / 'kb_wiring.db').as_posix()}")
    for table in KB_TABLES:
        Base.metadata.tables[table].create(engine, checkfirst=True)
    factory = sessionmaker(engine, expire_on_commit=False)
    project_id = uuid.uuid4()
    run_id = uuid.uuid4()
    with factory() as session:
        session.add(kb_schema.ResearchProjectRow(id=project_id, name="momentum", visibility_scope={}))
        session.add(kb_schema.AgentRunRow(id=run_id, project_id=project_id, agent_type="research", task="wiring", status="running"))
        session.add(
            FindingRow(
                id=uuid.uuid4(),
                project_id=project_id,
                canonical_key="empirical:momentum-persists",
                finding_type="empirical",
                statement="Cross-sectional momentum persists net of transaction costs.",
                scope={"asset_class": "equity", "market": "US"},
                status="candidate",
                confidence={"overall_tier": "moderate"},
                created_by_run_id=run_id,
            )
        )
        session.add(
            ExperimentRow(
                id=uuid.uuid4(),
                project_id=project_id,
                created_by_run_id=run_id,
                experiment_family_hash="ab" * 32,
                execution_hash="cd" * 32,
                hypothesis="Cross-sectional momentum 126-day backtest",
                methodology={"asset_class": "equity", "market": "US", "universe": "sp500-pit", "horizon": "6m", "frequency": "daily"},
                parameters={"signal_name": "mom_126d"},
                metrics={"sharpe": 0.8},
                outcome="success",
                status="completed",
                started_at=datetime(2026, 1, 10, tzinfo=UTC),
            )
        )
        session.commit()
    try:
        yield {"factory": factory, "project_id": project_id}
    finally:
        engine.dispose()


class TestBootstrapParity:
    def test_middleware_injects_packet_over_real_pg_backends(self, seeded_factory: dict, clean_registry) -> None:
        from langchain_core.messages import HumanMessage

        set_knowledge_config(KnowledgeConfig(enabled=True))
        factory = seeded_factory["factory"]
        bind_knowledge_backends(
            retrieval=pg.KnowledgeRetrievalService(factory),
            experiments=pg.SQLExperimentLookupStore(factory),
        )
        middleware = KnowledgeBootstrapMiddleware()  # registry-bound stores, config-gated
        state = {"messages": [HumanMessage(content="Study cross-sectional momentum.")]}
        update = middleware.before_agent(state, _make_runtime(thread_id="t-parity"))
        assert update is not None
        [injected] = update["messages"]
        assert injected.id == KNOWLEDGE_BOOTSTRAP_MESSAGE_ID
        assert injected.additional_kwargs[KNOWLEDGE_BOOTSTRAP_MARKER] is True
        assert "momentum 126-day backtest" in injected.content  # closest prior experiments
        assert "channel error" not in injected.content.lower()  # no channel gaps
        rerun = middleware.before_agent({"messages": [*state["messages"], injected]}, _make_runtime(thread_id="t-parity"))
        assert rerun is None  # once per run

    def test_middleware_skips_when_plane_disabled(self, seeded_factory: dict, clean_registry) -> None:
        from langchain_core.messages import HumanMessage

        set_knowledge_config(KnowledgeConfig(enabled=False))
        factory = seeded_factory["factory"]
        bind_knowledge_backends(
            retrieval=pg.KnowledgeRetrievalService(factory),
            experiments=pg.SQLExperimentLookupStore(factory),
        )
        middleware = KnowledgeBootstrapMiddleware()
        state = {"messages": [HumanMessage(content="Study momentum.")]}
        assert middleware.before_agent(state, _make_runtime(thread_id="t-off")) is None

    def test_middleware_skips_when_unbound(self, clean_registry) -> None:
        from langchain_core.messages import HumanMessage

        set_knowledge_config(KnowledgeConfig(enabled=True))
        middleware = KnowledgeBootstrapMiddleware()
        state = {"messages": [HumanMessage(content="Study momentum.")]}
        assert middleware.before_agent(state, _make_runtime(thread_id="t-unbound")) is None

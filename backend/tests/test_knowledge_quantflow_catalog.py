"""Phase 2b catalog/documents conformance for the Quantflow fork provider.

Direct provider conformance (offline, no live credentials) plus Gateway
router integration following ``test_knowledge_provider_router.py`` patterns.

Every test reaches the fork through the ``qp_module`` fixture, which uses
``pytest.importorskip`` so the whole file SKIPS -- honestly, with no
collection errors -- until Phase 2a implements
``deerflow.knowledge.quantflow_provider``.

Contract source: ``backend/docs/KNOWLEDGE.md`` fork section (four fixed
datasets, six read-only operations, FileNotFoundError -> 404 for missing
datasets) and ``deerflow.knowledge.contract`` (``RetrievalCatalogPage``
bounds: page_size 1..100, items <= 100, IDs <= 256 chars).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest
import sqlalchemy as sa
from _router_auth_helpers import make_authed_test_app
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

import deerflow.knowledge.schema as kb_schema
from app.gateway.auth.models import User
from app.gateway.deps import get_config
from app.gateway.routers import knowledge
from deerflow.extensions import ExtensionRegistry, reset_loaded_extensions
from deerflow.knowledge import KnowledgeRequestContext, RetrievalCatalogPage
from deerflow.knowledge.providers import GENERIC_KNOWLEDGE_TOOL
from deerflow.knowledge.schema.experiments import AssumptionRow, ExperimentRow
from deerflow.knowledge.schema.findings import FindingRow
from deerflow.persistence.base import Base

EXPECTED_DATASET_IDS = frozenset({"findings", "experiments", "failures", "assumptions"})

_CATALOG_TABLES: tuple[str, ...] = (
    "research_project",
    "agent_run",
    "finding",
    "experiment",
    "experiment_dataset",
    "experiment_artifact",
    "assumption",
)


@pytest.fixture
def qp_module():
    """Import the Phase 2 fork provider module, skipping until it exists."""
    return pytest.importorskip(
        "deerflow.knowledge.quantflow_provider",
        reason="Phase 2: Quantflow fork provider not implemented yet",
    )


@pytest.fixture
def provider(qp_module, tmp_path: Path):
    """Build the Phase 2 provider over a minimally seeded SQLite store.

    The provider is fail-closed without a session factory (LookupError),
    so documents()/catalog tests bind one row per dataset: enough to prove
    rows flow through paging, small enough to stay fast.
    """
    engine = sa.create_engine(f"sqlite:///{(tmp_path / 'kb_catalog.db').as_posix()}")
    for table in _CATALOG_TABLES:
        Base.metadata.tables[table].create(engine, checkfirst=True)
    factory = sessionmaker(engine, expire_on_commit=False)
    project_id, run_id = uuid.uuid4(), uuid.uuid4()
    now = datetime(2026, 3, 10, tzinfo=UTC)
    with factory() as session:
        session.add(kb_schema.ResearchProjectRow(id=project_id, name="catalog", visibility_scope={}))
        session.add(kb_schema.AgentRunRow(id=run_id, project_id=project_id, agent_type="research", task="catalog", status="running"))
        session.add(
            FindingRow(
                id=uuid.uuid4(),
                project_id=project_id,
                canonical_key="catalog:finding",
                finding_type="empirical",
                statement="Catalog finding.",
                scope={},
                status="candidate",
                confidence={},
                effective_from=datetime(2015, 1, 1, tzinfo=UTC),
                effective_to=None,
                recorded_at=now,
                created_by_run_id=run_id,
            )
        )
        session.add(
            ExperimentRow(
                id=uuid.uuid4(),
                project_id=project_id,
                created_by_run_id=run_id,
                experiment_family_hash="cc" * 32,
                execution_hash="c3" * 32,
                hypothesis="Catalog experiment.",
                outcome="success",
                status="completed",
            )
        )
        session.add(
            FindingRow(
                id=uuid.uuid4(),
                project_id=project_id,
                canonical_key="catalog:failure",
                finding_type="failure",
                statement="Catalog failure.",
                scope={},
                status="candidate",
                confidence={},
                effective_from=datetime(2015, 1, 1, tzinfo=UTC),
                effective_to=None,
                recorded_at=now,
                created_by_run_id=run_id,
            )
        )
        session.add(
            AssumptionRow(
                id=uuid.uuid4(),
                experiment_id=None,
                statement="Catalog assumption.",
                category="execution",
                sensitivity="high",
                tested=False,
                status="active",
            )
        )
        session.commit()
    return qp_module.QuantflowKnowledgeProvider(factory)


@pytest.fixture
def context():
    return KnowledgeRequestContext(user_id="alice", agent_name="lead_agent")


@pytest.fixture(autouse=True)
def no_singleton_provider():
    reset_loaded_extensions()
    yield
    reset_loaded_extensions()


# ---------------------------------------------------------------------------
# Direct provider conformance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_datasets_list_four_fork_kinds(provider, context):
    """catalog() lists exactly the four fork datasets."""
    page = await provider.catalog(context=context, page=1, page_size=100, search="")
    assert isinstance(page, RetrievalCatalogPage)
    assert page.page == 1
    assert page.total == 4, f"fork catalog must list 4 datasets, saw total={page.total}"
    ids = {item.id for item in page.items}
    assert ids == set(EXPECTED_DATASET_IDS), f"dataset IDs {sorted(ids)} != {sorted(EXPECTED_DATASET_IDS)}"
    for item in page.items:
        assert item.id and item.name
        assert len(item.id) <= 256


@pytest.mark.asyncio
async def test_documents_page_shape_per_dataset(provider, context):
    """documents(dataset_id) pages correctly for each fork dataset."""
    catalog = await provider.catalog(context=context, page=1, page_size=100, search="")
    assert catalog.total == 4
    for dataset in catalog.items:
        first = await provider.documents(dataset.id, context=context, page=1, page_size=20, search="")
        assert isinstance(first, RetrievalCatalogPage), f"documents({dataset.id!r})"
        assert first.page == 1
        assert 1 <= first.page_size <= 100
        assert first.total >= 0
        assert len(first.items) <= first.page_size
        assert len(first.items) <= 100
        for doc in first.items:
            assert doc.id and doc.name
            assert len(doc.id) <= 256
        second = await provider.documents(dataset.id, context=context, page=2, page_size=20, search="")
        assert second.page == 2
        assert second.total == first.total
        assert len(second.items) <= 20


@pytest.mark.asyncio
async def test_paging_caps_enforced(provider, context):
    """Paging caps (<=100): over-limit requests never return >100 rows.

    The fork follows the ``search.py`` convention (oversized limits are
    rejected, never silently clamped), but this test accepts either a clean
    validation error or a capped page -- the hard requirement is that no
    call ever yields more than 100 items or a page_size above 100.
    """
    from pydantic import ValidationError

    from deerflow.knowledge.write_api import KnowledgeError

    catalog = await provider.catalog(context=context, page=1, page_size=100, search="")
    assert len(catalog.items) <= 100
    dataset_id = catalog.items[0].id if catalog.items else "findings"

    for coro in (
        provider.catalog(context=context, page=1, page_size=101, search=""),
        provider.documents(dataset_id, context=context, page=1, page_size=101, search=""),
    ):
        try:
            page = await coro
        except (ValueError, KnowledgeError, ValidationError):
            continue
        assert isinstance(page, RetrievalCatalogPage)
        assert page.page_size <= 100
        assert len(page.items) <= 100


@pytest.mark.asyncio
async def test_unknown_dataset_errors_cleanly(provider, context):
    """documents() on an unknown dataset raises FileNotFoundError (router -> 404)."""
    with pytest.raises(FileNotFoundError):
        await provider.documents("no-such-dataset", context=context, page=1, page_size=20, search="")


# ---------------------------------------------------------------------------
# Router integration (test_knowledge_provider_router.py patterns)
# ---------------------------------------------------------------------------


def setup_fork_app(fork_provider):
    """Bind the real fork provider behind the Gateway catalog routes."""
    registry = ExtensionRegistry()
    with registry.attributed_to("test:fork"):
        registry.knowledge_provider("quantflow-pg", fork_provider)
    user = User(email="catalog@example.com", password_hash="x", system_role="user")
    app = make_authed_test_app(user_factory=lambda: user, bind_current_user=True)
    app.state.extensions = registry.build()
    app.include_router(knowledge.router)
    tool = SimpleNamespace(use=GENERIC_KNOWLEDGE_TOOL, model_extra={"provider": "quantflow-pg"})
    config = SimpleNamespace(knowledge_base=SimpleNamespace(enabled=True, scope_selection_enabled=True), get_tool_config=lambda _: tool)
    app.dependency_overrides[get_config] = lambda: config
    return app, user


def test_router_datasets_list_four_fork_kinds(qp_module, provider):
    app, _ = setup_fork_app(provider)
    with TestClient(app) as client:
        response = client.get("/api/knowledge/retrieval-catalog/datasets?agent_name=lead_agent&page_size=100")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 4
    assert {item["id"] for item in body["items"]} == set(EXPECTED_DATASET_IDS)


def test_router_unknown_dataset_returns_404_without_leak(qp_module, provider):
    app, _ = setup_fork_app(provider)
    with TestClient(app) as client:
        response = client.get("/api/knowledge/retrieval-catalog/datasets/no-such-dataset/documents?agent_name=lead_agent")
    assert response.status_code == 404
    assert "secret" not in response.text.lower()
    assert "traceback" not in response.text.lower()


def test_router_paging_cap_rejects_page_size_over_100(qp_module, provider):
    app, _ = setup_fork_app(provider)
    dataset_id = quote("findings", safe="")
    with TestClient(app) as client:
        assert client.get("/api/knowledge/retrieval-catalog/datasets?agent_name=lead_agent&page_size=101").status_code == 422
        capped = client.get(f"/api/knowledge/retrieval-catalog/datasets/{dataset_id}/documents?agent_name=lead_agent&page_size=101")
    assert capped.status_code == 422

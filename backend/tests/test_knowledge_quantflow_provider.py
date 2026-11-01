"""Phase 2a conformance for the Quantflow fork knowledge provider (live).

The module under test, ``deerflow.knowledge.quantflow_provider``, implements
the upstream retrieval protocols over the fork KB tables. Fixtures seed a
file-backed SQLite database (same query code paths as PostgreSQL; SQLite is
the dev/test dialect per the schema conventions), following
``test_knowledge_pg_retrieval.py`` patterns, and construct the provider
against its session factory — no live credentials, fully offline.

Contract source: ``backend/docs/KNOWLEDGE.md`` (read-only knowledge
contract) and ``deerflow.knowledge.contract`` (protocols) /
``deerflow.knowledge.providers`` (resolution, scope enforcement).
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

import deerflow.knowledge.schema as kb_schema
from deerflow.config.tool_config import ToolConfig
from deerflow.knowledge import (
    KnowledgeCapabilities,
    KnowledgeRequestContext,
    RetrievalCatalogPage,
    RetrievalResult,
)
from deerflow.knowledge.providers import (
    GENERIC_KNOWLEDGE_TOOLS,
    get_knowledge_capabilities,
    search_provider,
    validate_provider,
)
from deerflow.knowledge.quantflow_provider import CAPABILITIES, PROVIDER_NAME, QuantflowPgProvider
from deerflow.knowledge.schema.experiments import AssumptionRow, ExperimentRow
from deerflow.knowledge.schema.findings import FindingRow
from deerflow.persistence.base import Base

EXPECTED_OPERATIONS = frozenset({"search", "sources", "catalog", "documents", "dataset_filter", "document_filter"})

EXPECTED_KINDS = ("findings", "experiments", "failures", "assumptions")

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


def _engine(tmp_path: Path, name: str = "kb_quantflow_provider.db") -> sa.Engine:
    engine = sa.create_engine(f"sqlite:///{(tmp_path / name).as_posix()}")
    for table in KB_TABLES:
        Base.metadata.tables[table].create(engine, checkfirst=True)
    return engine


@pytest.fixture
def store(tmp_path: Path) -> Iterator[dict]:
    """Build the seeded database; return ids + session factory."""
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
        session.add(kb_schema.AgentRunRow(id=run_id, project_id=project_id, agent_type="research", task="provider conformance", status="running"))
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


@pytest.fixture
def provider(store):
    """Build the Phase 2a provider against the seeded SQLite store (offline)."""
    return QuantflowPgProvider(store["factory"])


@pytest.fixture
def context():
    return KnowledgeRequestContext(user_id="alice", agent_name="lead_agent")


def test_caps_advertised_equal_read_only_surface(provider):
    """The fork provider advertises exactly the six read-only operations."""
    assert provider.capabilities.operations == EXPECTED_OPERATIONS


def test_provider_passes_registry_validation(provider):
    """The fork provider satisfies validate_provider (registry admission rule)."""
    validate_provider(provider)
    for method in ("search", "search_scoped", "catalog", "documents"):
        assert inspect.iscoroutinefunction(getattr(provider, method)), f"{method} must be async"


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "   ", "\n\t "])
async def test_empty_query_rejected(provider, query):
    """search()/search_scoped() reject empty queries with ValueError (fail closed)."""
    with pytest.raises(ValueError, match="non-empty string"):
        await provider.search(query)


@pytest.mark.asyncio
async def test_search_returns_attributed_hits_with_verbatim_excerpts(provider):
    """search() returns RetrievalResult; attributed hits carry verbatim excerpts."""
    # Query must lexically match the seeded fixture: with an unconstrained
    # scope the structured channel is (correctly) skipped, so only genuine
    # lexical/vector matches return — never recency-ordered noise.
    result = await provider.search("momentum persists net of transaction costs")
    assert isinstance(result, RetrievalResult)
    assert result.hits, "conformance fixture must return at least one hit"
    for hit in result.hits:
        assert isinstance(hit.content, str) and hit.content
        if hit.source is not None:
            assert hit.source.provider
            assert hit.source.title
            # KNOWLEDGE.md: readers require the actual excerpt in tool content;
            # excerpts are verbatim provider evidence, never paraphrases.
            assert hit.source.excerpt in hit.content


@pytest.mark.asyncio
async def test_catalog_paging_shape_and_bounds(provider, context):
    """catalog() returns a bounded RetrievalCatalogPage honoring page/page_size."""
    page = await provider.catalog(context=context, page=1, page_size=20, search="")
    assert isinstance(page, RetrievalCatalogPage)
    assert page.page == 1
    assert 1 <= page.page_size <= 100
    assert page.total >= 0
    assert len(page.items) <= page.page_size
    assert len(page.items) <= 100
    for item in page.items:
        assert item.id and item.name
    second = await provider.catalog(context=context, page=2, page_size=20, search="")
    assert second.page == 2
    assert second.total == page.total


@pytest.mark.asyncio
async def test_documents_cover_expected_knowledge_kinds(provider, context):
    """Catalog datasets cover findings/experiments/failures/assumptions; each lists documents."""
    datasets: dict[str, str] = {}  # dataset_id -> matched kind
    page_no = 1
    while True:
        page = await provider.catalog(context=context, page=page_no, page_size=100, search="")
        for item in page.items:
            haystack = f"{item.id} {item.name}".lower()
            for kind in EXPECTED_KINDS:
                if kind in haystack:
                    datasets.setdefault(item.id, kind)
        if len(page.items) < page.page_size or page_no * page.page_size >= page.total:
            break
        page_no += 1
    missing = [kind for kind in EXPECTED_KINDS if kind not in set(datasets.values())]
    assert not missing, f"catalog datasets missing kinds {missing}; saw {sorted(datasets.values())}"
    for dataset_id, kind in sorted(datasets.items()):
        docs = await provider.documents(dataset_id, context=context, page=1, page_size=20, search="")
        assert isinstance(docs, RetrievalCatalogPage), f"documents({dataset_id!r}) for kind {kind!r}"


@pytest.mark.asyncio
async def test_scope_honoring_selected_datasets_and_disabled(provider, context):
    """Selected dataset_ids constrain retrieval; disabled scope blocks it."""
    catalog = await provider.catalog(context=context, page=1, page_size=100, search="")
    assert catalog.items, "conformance fixture must expose at least one dataset"
    dataset_id = catalog.items[0].id
    selected = {"version": 1, "mode": "selected", "dataset_ids": [dataset_id]}
    result = await search_provider(provider, "status update", scope=selected, user_id="alice", agent_name="lead_agent")
    assert isinstance(result, RetrievalResult)
    with pytest.raises(ValueError, match="disabled"):
        await search_provider(
            provider,
            "status update",
            scope={"version": 1, "mode": "disabled"},
            user_id="alice",
            agent_name="lead_agent",
        )


class _UnscopedFallbackUsed(Exception):
    """Sentinel: raised only if the provider falls back to unscoped search."""


@pytest.mark.asyncio
async def test_inaccessible_selection_fails_whole_request_without_fallback(provider, monkeypatch):
    """A missing/inaccessible selected dataset fails the request; never retries unconstrained."""
    if hasattr(provider, "search"):
        from unittest.mock import AsyncMock

        monkeypatch.setattr(provider, "search", AsyncMock(side_effect=_UnscopedFallbackUsed))
    with pytest.raises(Exception) as exc_info:
        await search_provider(
            provider,
            "status update",
            scope={"version": 1, "mode": "selected", "dataset_ids": ["missing-dataset-id"]},
            user_id="alice",
            agent_name="lead_agent",
        )
    assert not isinstance(exc_info.value, _UnscopedFallbackUsed), "constrained failure must not retry as an unconstrained search"


def _fake_app_config(*tools: ToolConfig) -> SimpleNamespace:
    """Minimal config surface for get_knowledge_capabilities (get_tool_config only)."""
    by_name = {tool.name: tool for tool in tools}
    return SimpleNamespace(get_tool_config=lambda name: by_name.get(name))


def _fake_extensions(*named_caps: tuple[str, KnowledgeCapabilities]) -> SimpleNamespace:
    """Minimal extension snapshot: (source, (name, provider)) pairs with caps only."""
    return SimpleNamespace(knowledge_providers=tuple((f"test:{name}", (name, SimpleNamespace(capabilities=caps))) for name, caps in named_caps))


def test_mixed_provider_identities_collapse_scope_capabilities():
    """Two provider identities strip catalog/documents/filters (providers.py:123-125).

    WHY single-provider deployment retires this risk:
    ``get_knowledge_capabilities`` intersects the per-operation capability
    projections and drops ``catalog``/``documents``/``dataset_filter``/
    ``document_filter`` ONLY when more than one provider identity is
    configured (``providers.py:123-125``) — different registrations share no
    resource-ID namespace, so a mixed deployment can enforce all/disabled
    only. The fork ships exactly one provider (``quantflow-pg``), pinned by
    ``test_knowledge_local_only_guard.py::test_exactly_one_provider_once_quantflow_pg_exists``,
    so the multi-identity branch is unreachable in the deployed
    configuration and ``selected`` scopes keep their full meaning.

    This test simulates a hypothetical two-provider deployment (real fork
    caps + a synthetic second provider) to prove the collapse rule fires,
    plus a single-provider control proving the same projections survive
    when only one identity is present — i.e. there is no collapse path
    with the single fork provider deployed.
    """
    assert PROVIDER_NAME == "quantflow-pg"
    assert CAPABILITIES.operations == EXPECTED_OPERATIONS

    search_tool = ToolConfig(
        name="knowledge_search",
        group="knowledge",
        use=GENERIC_KNOWLEDGE_TOOLS["knowledge_search"],
        provider="quantflow-pg",
    )
    entries_tool = ToolConfig(
        name="knowledge_entries_search",
        group="knowledge",
        use=GENERIC_KNOWLEDGE_TOOLS["knowledge_entries_search"],
        provider="synthetic-entries",
    )
    # A second provider rich enough that the pre-collapse intersection is
    # the full six operations: entries_search projects to search plus the
    # mapped dataset/document filters (providers.operation_capabilities).
    entries_caps = KnowledgeCapabilities(operations=frozenset({"entries_search", "sources", "catalog", "documents", "entries_dataset_filter", "entries_document_filter"}))
    mixed = get_knowledge_capabilities(
        _fake_app_config(search_tool, entries_tool),
        extensions=_fake_extensions(("quantflow-pg", CAPABILITIES), ("synthetic-entries", entries_caps)),
    )
    assert mixed.operations == frozenset({"search", "sources"}), f"mixed identities must strip catalog/documents/filters per providers.py:123-125; got {sorted(mixed.operations)}"

    # Single-provider control: one identity behind both tools keeps everything.
    solo_caps = KnowledgeCapabilities(
        operations=frozenset(
            {
                "search",
                "entries_search",
                "sources",
                "catalog",
                "documents",
                "dataset_filter",
                "document_filter",
                "entries_dataset_filter",
                "entries_document_filter",
            }
        )
    )
    solo_search_tool = ToolConfig(
        name="knowledge_search",
        group="knowledge",
        use=GENERIC_KNOWLEDGE_TOOLS["knowledge_search"],
        provider="solo",
    )
    solo_entries_tool = ToolConfig(
        name="knowledge_entries_search",
        group="knowledge",
        use=GENERIC_KNOWLEDGE_TOOLS["knowledge_entries_search"],
        provider="solo",
    )
    solo = get_knowledge_capabilities(
        _fake_app_config(solo_search_tool, solo_entries_tool),
        extensions=_fake_extensions(("solo", solo_caps)),
    )
    assert solo.operations == EXPECTED_OPERATIONS, f"single identity must retain the full selection surface; got {sorted(solo.operations)}"


@pytest.mark.asyncio
async def test_empty_selected_scope_denies(provider, monkeypatch):
    """A selected scope with an empty dataset list denies; it never widens to all datasets."""
    # Public path: scope validation rejects the empty selection (deny, not results).
    with pytest.raises((ValueError, PermissionError)):
        await search_provider(
            provider,
            "status update",
            scope={"version": 1, "mode": "selected", "dataset_ids": []},
            user_id="alice",
            agent_name="lead_agent",
        )
    # Provider layer itself fails closed even if validation is bypassed.
    monkeypatch.setattr(
        "deerflow.knowledge.providers.validate_scope_capabilities",
        lambda scope, capabilities: {"version": 1, "mode": "selected", "dataset_ids": []},
    )
    with pytest.raises(PermissionError, match="selects no datasets"):
        await provider.search_scoped(
            "status update",
            context=KnowledgeRequestContext(user_id="alice", agent_name="lead_agent", scope={"version": 1, "mode": "selected", "dataset_ids": []}),
        )


@pytest.mark.asyncio
async def test_unknown_dataset_denies_whole_request_without_partial_results(provider):
    """One unknown id in a mixed known+unknown selection fails the whole request (PermissionError, no partial hits)."""
    scope = {"version": 1, "mode": "selected", "dataset_ids": ["findings", "missing-dataset-id"]}
    with pytest.raises(PermissionError, match="unknown dataset"):
        await provider.search_scoped(
            "momentum persists net of transaction costs",
            context=KnowledgeRequestContext(user_id="alice", agent_name="lead_agent", scope=scope),
        )
    with pytest.raises(PermissionError, match="unknown dataset"):
        await search_provider(
            provider,
            "momentum persists net of transaction costs",
            scope=scope,
            user_id="alice",
            agent_name="lead_agent",
        )


@pytest.mark.asyncio
async def test_disabled_scope_refuses_at_provider_layer(provider):
    """search_scoped itself refuses a disabled scope (ValueError, matching the sibling convention)."""
    with pytest.raises(ValueError, match="disabled"):
        await provider.search_scoped(
            "status update",
            context=KnowledgeRequestContext(user_id="alice", agent_name="lead_agent", scope={"version": 1, "mode": "disabled"}),
        )


@pytest.mark.asyncio
async def test_document_filters_restrict_fused_ranking(provider, context):
    """document_filters gate the fused ranking: listed ids pass, unlisted ids return nothing."""

    async def _scoped(query: str, scope: dict) -> RetrievalResult:
        return await provider.search_scoped(
            query,
            context=KnowledgeRequestContext(user_id="alice", agent_name="lead_agent", scope=scope),
        )

    query = "momentum persists net of transaction costs"
    docs = await provider.documents("findings", context=context, page=1, page_size=20, search="")
    assert len(docs.items) == 1, "conformance fixture seeds exactly one non-failure finding"
    doc_id = docs.items[0].id

    unfiltered = await _scoped(query, {"version": 1, "mode": "selected", "dataset_ids": ["findings"]})
    assert unfiltered.hits, "selected findings search must return the seeded finding"
    unfiltered_contents = {hit.content for hit in unfiltered.hits}

    allowed = await _scoped(
        query,
        {"version": 1, "mode": "selected", "dataset_ids": ["findings"], "document_filters": [{"dataset_id": "findings", "document_ids": [doc_id]}]},
    )
    assert allowed.hits, "filter listing the matching doc must keep its hits"
    assert {hit.content for hit in allowed.hits} <= unfiltered_contents, "filtered hits must be a subset of the fused ranking"

    denied = await _scoped(
        query,
        {"version": 1, "mode": "selected", "dataset_ids": ["findings"], "document_filters": [{"dataset_id": "findings", "document_ids": ["0" * 32]}]},
    )
    assert denied.hits == [], "filter listing only an unrelated doc id must return no hits"

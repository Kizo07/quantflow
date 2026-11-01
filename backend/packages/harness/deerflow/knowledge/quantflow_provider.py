"""Fork-owned read-only knowledge provider over the QuantFlow evidence plane.

This module is the Phase 2a upstream seam: :class:`QuantflowPgProvider`
implements the upstream retrieval protocols
(:class:`~deerflow.knowledge.contract.RetrievalProvider`,
:class:`~deerflow.knowledge.contract.ScopedRetrievalProvider`,
:class:`~deerflow.knowledge.contract.RetrievalCatalogProvider`,
:class:`~deerflow.knowledge.contract.RetrievalDocumentsProvider`) against
the fork KB tables, running the Phase 1/2 fork stack (retrieval planner,
RRF fusion, SQL channel stores) and shaping hits through the Phase 1a
evidence adapters (:func:`candidate_to_retrieval_hit
<deerflow.knowledge.pg_retrieval.candidate_to_retrieval_hit>`), so excerpts
stay verbatim provider evidence with content-hash source ids.

Advertised capabilities (exactly the six read-only operations from
``backend/docs/KNOWLEDGE.md`` "Fork extension"):

* ``search`` — fused retrieval over all four datasets.
* ``sources`` — shared v1 evidence envelope, no native sidecar.
* ``catalog`` — paginated listing of the four fixed datasets.
* ``documents`` — paginated documents within one dataset.
* ``dataset_filter`` / ``document_filter`` — ``search_scoped`` restricted
  to the selected datasets' kinds / document ids.

Dataset to kind mapping (KNOWLEDGE.md "Dataset to kind mapping"):

* ``findings`` → planner kind ``finding`` (``finding`` rows except
  ``finding_type == "failure"``).
* ``experiments`` → planner kind ``experiment`` (``experiment`` rows
  except ``outcome == "failure"``).
* ``failures`` → planner kind ``failure`` (both tables' failure rows).
* ``assumptions`` → planner kind ``assumption`` (``assumption`` rows).

Scope rules: ``disabled`` refuses every search; a ``selected`` scope with
an unknown dataset id fails the whole constrained request (never retries
unconstrained); an empty ``selected`` dataset list denies (never widens
to all datasets); known selections constrain the planner kinds (and, when
document filters are present, the fused ranking by document id).

Import-cycle contract: this module stays import-light — only the stdlib
and the pydantic-only ``contract`` at top level. The SQLAlchemy-backed
``pg_retrieval``/``schema`` bindings, ``embeddings``, ``config``,
``providers`` (scope validation), ``extensions`` (registration), and the
persistence engine cache all load lazily inside methods, mirroring
``deerflow.knowledge.providers`` / ``tools``. It is registered in
``deerflow.knowledge.__getattr__``'s lazy map for the same reason.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from deerflow.knowledge.contract import (
    KnowledgeCapabilities,
    KnowledgeRequestContext,
    RetrievalCatalogItem,
    RetrievalCatalogPage,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PROVIDER_NAME",
    "CAPABILITIES",
    "SEARCH_TOP_K",
    "QuantflowPgProvider",
    "QuantflowKnowledgeProvider",
    "dataset_kinds",
    "ensure_quantflow_provider_registered",
]

#: Registration name (``backend/docs/KNOWLEDGE.md`` "Local-only deployment").
PROVIDER_NAME = "quantflow-pg"

#: Exactly the six read-only operations; ``search`` is required by ``validate_provider``.
CAPABILITIES = KnowledgeCapabilities(
    operations=frozenset({"search", "sources", "catalog", "documents", "dataset_filter", "document_filter"}),
)

#: Default fused top-k for provider searches (within the planner ``MAX_TOP_K`` ceiling).
#: Planner top-k / per-channel floor for provider retrieval. Note this does
#: NOT cap ``search()`` output: ``_retrieve`` iterates the full fused
#: candidate list (bounded in practice by channel limits); display callers
#: stay bounded via ``format_result`` (100-hit cap + budgets).
SEARCH_TOP_K = 20

#: Fixed fork datasets → planner kinds (one kind each; ``failures`` spans both tables).
_DATASET_KINDS: dict[str, tuple[str, ...]] = {
    "findings": ("finding",),
    "experiments": ("experiment",),
    "failures": ("failure",),
    "assumptions": ("assumption",),
}

#: Human-readable dataset names (each carries its kind for scope pickers).
_DATASET_NAMES: dict[str, str] = {
    "findings": "Findings",
    "experiments": "Experiments",
    "failures": "Failures",
    "assumptions": "Assumptions",
}


def dataset_kinds() -> dict[str, tuple[str, ...]]:
    """Return the dataset-id → planner-kinds mapping (copy).

    Public accessor so service-layer scope code (`pg_retrieval`) shares
    the single source of truth without importing a private name.
    """
    return dict(_DATASET_KINDS)


_ALL_PLANNER_KINDS: tuple[str, ...] = ("finding", "experiment", "failure", "assumption")

_registry_lock = threading.Lock()


def _scope_declares_nothing(scope: Any) -> bool:
    """Whether a ScopeFilter imposes no constraint (all fields empty).

    An empty filter scores every row 0, so the structured store degrades
    to recency order — "no constraint, no signal" — and its noise rows
    pollute RRF. The provider skips the structured channel in that case
    (Phase 2 exit-gate finding: +0.146 recall@10).
    """
    return not any(
        [
            scope.asset_class,
            scope.markets,
            scope.universe,
            scope.horizon,
            scope.frequency,
            scope.dataset_family,
            scope.valid_from,
            scope.valid_to,
        ]
    )


class _EmptyStructuredStore:
    """StructuredLookupStore that returns no rows (unconstrained scope)."""

    def structured_lookup(self, scope: Any, *, kinds: Sequence[str], limit: int) -> list[Any]:
        return []


class QuantflowPgProvider:
    """Upstream retrieval protocols over the fork PostgreSQL/SQLite KB.

    Args:
        session_factory: Zero-argument SQLAlchemy session factory (one
            session per call, mirroring ``pg_store``). None resolves the
            DSN from :class:`KnowledgeConfig <deerflow.knowledge.config.KnowledgeConfig>`
            lazily on first use through the process-wide sync-engine cache,
            raising :class:`LookupError` when the plane is disabled or has
            no DSN — so credential-free construction never performs I/O.
        embedding_provider: Query-embedding boundary for the vector
            channel. None (default) disables the vector channel; the
            planner skips it with its standard warning.
    """

    capabilities = CAPABILITIES

    def __init__(self, session_factory: Any | None = None, *, embedding_provider: Any | None = None) -> None:
        """Bind the provider; construction performs no I/O."""
        self._session_factory = session_factory
        self._embedding_provider = embedding_provider
        self._lock = threading.Lock()
        self._resolved_factory: Any | None = None
        self._stores: tuple[Any, ...] | None = None

    def _sessions(self) -> Any:
        """Return the session factory, resolving the configured DSN lazily."""
        if self._session_factory is not None:
            return self._session_factory
        with self._lock:
            if self._resolved_factory is not None:
                return self._resolved_factory
            from deerflow.knowledge.config import get_knowledge_config

            config = get_knowledge_config()
            if not config.is_enabled():
                raise LookupError("Knowledge retrieval is not configured")
            dsn = config.get_database_dsn()
            if not dsn:
                raise LookupError("Knowledge retrieval is not configured")
            from deerflow.persistence.agents.sql import get_sync_sessionmaker

            self._resolved_factory = get_sync_sessionmaker(dsn)
            return self._resolved_factory

    def _channels(self) -> tuple[Any, Any, Any, Any, Any]:
        """Return the cached SQL channel stores (structured/lexical/vector/failures/relational)."""
        with self._lock:
            if self._stores is not None:
                return self._stores
            from deerflow.knowledge.pg_retrieval import (
                SQLFailureSearchStore,
                SQLLexicalSearchStore,
                SQLRelationalExpansionStore,
                SQLStructuredLookupStore,
                SQLVectorSearchStore,
            )

            sessions = self._sessions()
            self._stores = (
                SQLStructuredLookupStore(sessions),
                SQLLexicalSearchStore(sessions),
                SQLVectorSearchStore(sessions),
                SQLFailureSearchStore(sessions),
                SQLRelationalExpansionStore(sessions),
            )
            return self._stores

    def _query_embedding(self, query_text: str) -> list[float] | None:
        """Embed the query, degrading to None (vector skipped) on any failure."""
        provider = self._embedding_provider
        if provider is None:
            return None
        from deerflow.knowledge.embeddings import EMBEDDING_DIM, embed_texts

        try:
            vectors = embed_texts(provider, [query_text])
        except Exception:
            logger.warning("QuantflowPgProvider: query embedding failed; running retrieval without the vector channel", exc_info=True)
            return None
        if not vectors or len(vectors[0]) != EMBEDDING_DIM:
            logger.warning(
                "QuantflowPgProvider: provider %r emitted %d-d vectors, expected %d; running retrieval without the vector channel",
                getattr(provider, "model_id", "?"),
                len(vectors[0]) if vectors else 0,
                EMBEDDING_DIM,
            )
            return None
        return vectors[0]

    def _retrieve(
        self,
        query: str,
        *,
        kinds: tuple[str, ...],
        allowed_doc_ids: frozenset[str] | None,
        now: datetime | str | None = None,
    ) -> RetrievalResult:
        """Run fused fork retrieval and shape it as upstream hits.

        ``now`` pins the fusion reference clock (forwarded to
        ``execute_retrieval``); ``None`` keeps wall-clock behavior.
        """
        from deerflow.knowledge.pg_retrieval import candidate_to_retrieval_hit
        from deerflow.knowledge.retrieval.planner import (
            DEFAULT_PER_CHANNEL_LIMIT,
            MAX_PER_CHANNEL_LIMIT,
            MAX_TOP_K,
            ResearchIntent,
            execute_retrieval,
            plan_retrieval,
        )

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        query_text = query.strip()
        structured, lexical, vector, failures, relational = self._channels()
        intent = ResearchIntent(topic=query_text)
        plan = plan_retrieval(
            intent,
            per_channel_limit=min(MAX_PER_CHANNEL_LIMIT, max(DEFAULT_PER_CHANNEL_LIMIT, SEARCH_TOP_K)),
            top_k=min(MAX_TOP_K, SEARCH_TOP_K),
            kinds=list(kinds),
            relational=True,
        )
        if _scope_declares_nothing(plan.scope):
            structured = _EmptyStructuredStore()
        result = execute_retrieval(
            plan,
            query_text=query_text,
            structured=structured,
            lexical=lexical,
            vector=vector,
            failures=failures,
            relational=relational,
            query_embedding=self._query_embedding(query_text),
            now=now,
        )
        # Kinds apply as a post-fusion exact filter: channels that ignore
        # kinds (notably the failure channel) must never leak rows the
        # caller did not ask for into the page.
        wanted_kinds = frozenset(kinds)
        ranked = [item for item in result.fusion.candidates if item.candidate.kind in wanted_kinds]
        if allowed_doc_ids is not None:
            ranked = [item for item in ranked if item.candidate.id in allowed_doc_ids]
        hits = []
        for item in ranked:
            score = item.score if isinstance(item.score, (int, float)) and not isinstance(item.score, bool) and math.isfinite(item.score) else None
            hits.append(candidate_to_retrieval_hit(item.candidate, score=score))
        return RetrievalResult(hits=hits)

    async def search(self, query: str, *, now: datetime | str | None = None) -> RetrievalResult:
        """Return fused retrieval hits across all four fork datasets.

        ``now`` pins the fusion reference clock (exit gates pin it so
        fixture recency never decays); ``None`` keeps wall-clock behavior.
        """
        return self._retrieve(query, kinds=_ALL_PLANNER_KINDS, allowed_doc_ids=None, now=now)

    async def search_scoped(self, query: str, *, context: KnowledgeRequestContext, now: datetime | str | None = None) -> RetrievalResult:
        """Return fused hits constrained by the execution scope.

        ``disabled`` refuses; ``selected`` restricts the planner to the
        selected datasets' kinds (plus a document-id post-filter when
        document filters are present). Unknown dataset ids fail the whole
        constrained request — never a best-effort subset, never an
        unconstrained retry. An empty ``selected`` dataset list denies
        (fail closed), never widens to all datasets.
        """
        from deerflow.knowledge.providers import validate_scope_capabilities

        scope = validate_scope_capabilities(context.scope, self.capabilities) if context.scope is not None else None
        if scope is not None and scope["mode"] == "disabled":
            # ValueError matches the sibling convention (providers.search_provider,
            # RAGFlowKnowledgeProvider); PermissionError is reserved for scope that
            # selects something inaccessible (tools.py maps it to "not accessible").
            raise ValueError("Knowledge search is disabled for this turn")
        kinds = _ALL_PLANNER_KINDS
        allowed_doc_ids: frozenset[str] | None = None
        if scope is not None and scope["mode"] == "selected":
            dataset_ids = list(scope.get("dataset_ids") or [])
            if not dataset_ids:
                # Fail closed: an empty selection denies, never widens to all
                # datasets. Unreachable through validate_scope_capabilities
                # (selected requires >= 1 dataset id), but the provider must
                # not depend on the validation layer for this invariant.
                raise PermissionError("Knowledge scope selects no datasets")
            unknown = [dataset_id for dataset_id in dataset_ids if dataset_id not in _DATASET_KINDS]
            if unknown:
                raise PermissionError(f"Knowledge scope selects unknown dataset(s): {', '.join(sorted(unknown))}")
            kinds = tuple(dict.fromkeys(kind for dataset_id in dataset_ids for kind in _DATASET_KINDS[dataset_id]))
            filters = list(scope.get("document_filters") or [])
            for entry in filters:
                if not isinstance(entry, dict) or entry.get("dataset_id") not in _DATASET_KINDS:
                    raise PermissionError("Knowledge scope selects unknown dataset(s)")
            if filters:
                allowed_doc_ids = frozenset(document_id for entry in filters for document_id in (entry.get("document_ids") or []) if isinstance(document_id, str))
        return self._retrieve(query, kinds=kinds, allowed_doc_ids=allowed_doc_ids, now=now)

    async def catalog(self, *, context: KnowledgeRequestContext, page: int, page_size: int, search: str) -> RetrievalCatalogPage:
        """Return one page of the four fixed fork datasets."""
        needle = search.strip().casefold() if isinstance(search, str) else ""
        items = [RetrievalCatalogItem(id=dataset_id, name=_DATASET_NAMES[dataset_id], selectable=True) for dataset_id in _DATASET_KINDS if not needle or needle in dataset_id.casefold() or needle in _DATASET_NAMES[dataset_id].casefold()]
        start = (page - 1) * page_size
        return RetrievalCatalogPage(items=items[start : start + page_size], page=page, page_size=page_size, total=len(items))

    async def documents(self, dataset_id: str, *, context: KnowledgeRequestContext, page: int, page_size: int, search: str) -> RetrievalCatalogPage:
        """Return one page of documents (KB rows) within one fork dataset.

        Raises:
            FileNotFoundError: When ``dataset_id`` is not one of the four
                fork datasets (unknown ids are rejected, never mapped).
        """
        if dataset_id not in _DATASET_KINDS:
            raise FileNotFoundError(f"Unknown knowledge dataset: {dataset_id!r}")
        from sqlalchemy import or_, select

        from deerflow.knowledge.pg_retrieval import (
            assumption_row_to_candidate,
            experiment_row_to_candidate,
            finding_row_to_candidate,
        )
        from deerflow.knowledge.schema.experiments import AssumptionRow, ExperimentRow
        from deerflow.knowledge.schema.findings import FindingRow

        needle = search.strip().casefold() if isinstance(search, str) else ""
        with self._sessions()() as session:
            if dataset_id == "findings":
                rows = session.scalars(select(FindingRow).where(FindingRow.finding_type != "failure"))
                candidates = [finding_row_to_candidate(row) for row in rows]
            elif dataset_id == "experiments":
                # ``outcome`` is nullable: NULL (undecided) rows are experiments, not failures.
                rows = session.scalars(select(ExperimentRow).where(or_(ExperimentRow.outcome != "failure", ExperimentRow.outcome.is_(None))))
                candidates = [experiment_row_to_candidate(row) for row in rows]
            elif dataset_id == "failures":
                candidates = [finding_row_to_candidate(row) for row in session.scalars(select(FindingRow).where(FindingRow.finding_type == "failure"))]
                candidates.extend(experiment_row_to_candidate(row) for row in session.scalars(select(ExperimentRow).where(ExperimentRow.outcome == "failure")))
            else:
                rows = session.scalars(select(AssumptionRow))
                candidates = [assumption_row_to_candidate(row) for row in rows]
        if needle:
            candidates = [item for item in candidates if needle in f"{item.title} {item.text}".casefold()]
        # Deterministic page order: recency first (unknown moments last), then id.
        ordered = sorted(candidates, key=lambda item: item.id)
        ordered = sorted([item for item in ordered if item.recorded_at], key=lambda item: item.recorded_at, reverse=True) + [item for item in ordered if not item.recorded_at]
        start = (page - 1) * page_size
        items = [RetrievalCatalogItem(id=item.id, name=(item.title.strip() or f"{item.kind}:{item.id}")[:500], selectable=True) for item in ordered[start : start + page_size]]
        return RetrievalCatalogPage(items=items, page=page, page_size=page_size, total=len(ordered))


#: Skeleton-contract alias (Phase 0b conformance named the entry point
#: ``QuantflowKnowledgeProvider``); the canonical name is :class:`QuantflowPgProvider`.
QuantflowKnowledgeProvider = QuantflowPgProvider


def ensure_quantflow_provider_registered(*, provider: QuantflowPgProvider | None = None, source: str = "builtin:quantflow-pg") -> bool:
    """Idempotently register the fork provider in the process-wide extensions.

    The built-in assembly step (called behind the ``feat.knowledge`` gate
    once backends are bound): validates the provider exactly like
    :meth:`ExtensionRegistry.knowledge_provider
    <deerflow.extensions.registry.ExtensionRegistry.knowledge_provider>`
    would, then appends it to the immutable loaded snapshot. A second call
    (or an operator-registered ``quantflow-pg``) is a no-op returning
    False. Callers must skip this while backends are unbound — a provider
    constructed then could only raise :class:`LookupError` at call time.

    Returns:
        True when this call registered the provider, False when the name
        was already present.
    """
    from dataclasses import replace

    from deerflow.extensions import get_loaded_extensions, set_loaded_extensions
    from deerflow.knowledge.providers import validate_provider

    if provider is None:
        from deerflow.knowledge.agent_tools.lookup import get_knowledge_backends

        bound = get_knowledge_backends().retrieval
        provider = QuantflowPgProvider(embedding_provider=getattr(bound, "embedding_provider", None))
    validate_provider(provider)
    with _registry_lock:
        loaded = get_loaded_extensions()
        if any(name == PROVIDER_NAME for _, (name, _) in loaded.knowledge_providers):
            return False
        set_loaded_extensions(replace(loaded, knowledge_providers=(*loaded.knowledge_providers, (source, (PROVIDER_NAME, provider)))))
        return True

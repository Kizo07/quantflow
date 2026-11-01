"""Quantflow Research Knowledge Plane (deer-flow harness integration).

This package implements the KB "evidence commits" path (Phase 1: experiments,
failures, assumptions, content-addressed artifacts, experiment search) on top
of the harness PostgreSQL persistence layer. Later phases add findings,
validation, consolidation, and skills promotion.

Submodules:

* :mod:`deerflow.knowledge.config` — ``KnowledgeConfig`` + process singleton
  (standalone; also registered as the ``knowledge:`` section of ``AppConfig``).
* :mod:`deerflow.knowledge.hashing` — canonical JSON + family/execution hashes.
* :mod:`deerflow.knowledge.write_api` — ``experiment_begin/commit``,
  ``failure_record``, ``assumption_record`` over the ``ExperimentStore``
  protocol (no I/O of its own).
* :mod:`deerflow.knowledge.search` — ``experiment_search`` pre-experiment
  recheck over the ``ExperimentSearchStore`` protocol (no I/O of its own).
* :mod:`deerflow.knowledge.pg_store` — binds ``ExperimentSearchStore`` to the
  real Phase 1 ORM models (``experiment`` + link tables).
* :mod:`deerflow.knowledge.schema` — Phase 1 ORM models (episodic half of the
  KB canonical schema; migration ``0022_knowledge_phase1``).
* :mod:`deerflow.knowledge.artifacts` — SHA-256 content-addressed artifact
  store over pluggable object backends (local FS, MinIO/S3).
* :mod:`deerflow.knowledge.eval` — retrieval-eval fixture + runner (Phase 0;
  stdlib-only, no ``deerflow`` imports).
* :mod:`deerflow.knowledge.retrieval` — hybrid retrieval planner, RRF fusion,
  and fixed-budget context packets (Phase 2; namespace package, no I/O).
* :mod:`deerflow.knowledge.embeddings` — embedding provider interface,
  deterministic test fake, and backfill helper (Phase 2; no I/O).
* :mod:`deerflow.knowledge.agent_tools` — agent lookup tools, run-start
  bootstrap, and bootstrap middleware (Phase 2; namespace package, no
  I/O). Renamed from ``deerflow.knowledge.tools`` by the upstream merge
  (batch B3): upstream's new :mod:`deerflow.knowledge.tools` *module*
  would otherwise shadow this subpackage at import time, so upstream
  keeps the plain ``tools`` name and the Quantflow subpackage moved.
* :mod:`deerflow.knowledge.pg_retrieval` — binds the retrieval, lookup, and
  embedding store protocols to PostgreSQL/SQLite (integration).
* :mod:`deerflow.knowledge.contract` — provider-neutral retrieval
  contracts (upstream; read-only, pydantic-only).
* :mod:`deerflow.knowledge.sources` — evidence budgeting + citation
  helpers over the contracts (upstream; read-only).
* :mod:`deerflow.knowledge.providers` — provider resolution + scope
  enforcement (upstream; imports ``deerflow.config``, lazy).
* :mod:`deerflow.knowledge.tools` — stable agent entry point for
  registered read-only providers (upstream; imports
  ``deerflow.config``/langchain, lazy).

Import-cycle contract: ``deerflow.config.app_config`` imports
``deerflow.knowledge.config``, so importing this package must never pull in
``deerflow.config``. The eagerly imported modules below depend only on the
stdlib, pydantic, and each other — including upstream's ``contract`` and
``sources`` (pydantic-only, safe to import eagerly); the
SQLAlchemy-backed ``pg_store``, ``pg_retrieval`` and ``schema``
submodules — plus ``retrieval``, ``embeddings`` and ``agent_tools``
(``agent_tools`` pulls langchain), and upstream's ``providers`` and
``tools`` (which pull ``deerflow.config``/langchain) — load lazily via
:func:`__getattr__` (explicit ``deerflow.knowledge.pg_store`` /
``deerflow.knowledge.retrieval.planner`` style imports work directly and are
unaffected).
"""

from __future__ import annotations

import importlib
from typing import Any

from deerflow.knowledge import artifacts, config, hashing, search, write_api
from deerflow.knowledge.config import (
    ENV_DSN,
    ENV_ENABLED,
    ENV_S3_BUCKET,
    ENV_S3_ENDPOINT,
    ENV_S3_REGION,
    KnowledgeConfig,
    get_knowledge_config,
    load_knowledge_config_from_dict,
    set_knowledge_config,
)

# Upstream retrieval contracts (union merge batch B3). No top-level name
# collisions with the Quantflow side (verified: disjoint __all__ sets), so
# upstream keeps its plain names. Near-miss (NOT a collision): Quantflow's
# Phase 2 planner type lives at
# ``deerflow.knowledge.retrieval.planner.FusedRetrieval`` while upstream's
# contract model is re-exported here as ``deerflow.knowledge.RetrievalResult``;
# both keep their names in their own modules.
from deerflow.knowledge.contract import (
    EntityRelationsProvider,
    KnowledgeCapabilities,
    KnowledgeEntriesProvider,
    KnowledgeRequestContext,
    KnowledgeSource,
    RetrievalCatalogItem,
    RetrievalCatalogPage,
    RetrievalCatalogProvider,
    RetrievalDocumentsProvider,
    RetrievalHit,
    RetrievalProvider,
    RetrievalResult,
    ScopedRetrievalProvider,
)
from deerflow.knowledge.hashing import (
    EXECUTION_DOMAIN,
    FAMILY_DOMAIN,
    IDEMPOTENCY_DOMAIN,
    canonical_bytes,
    canonical_json,
    execution_hash,
    experiment_family_hash,
    make_idempotency_key,
    normalize,
    normalize_text,
    sha256_hex,
)
from deerflow.knowledge.search import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    ExperimentFilter,
    ExperimentSearchResult,
    ExperimentSearchStore,
    experiment_get_by_execution_hash,
    experiment_search,
)
from deerflow.knowledge.sources import format_result, sources_from_messages
from deerflow.knowledge.write_api import (
    ASSUMPTION_CATEGORIES,
    ASSUMPTION_SENSITIVITIES,
    ASSUMPTION_STATUSES,
    BEGIN_STATUSES,
    DATASET_ROLES,
    EXPERIMENT_STATUSES,
    FAILURE_CLASSES,
    OUTCOME_CLASSIFICATIONS,
    TERMINAL_STATUSES,
    AssumptionRecord,
    DuplicateExecutionError,
    ExperimentNotFoundError,
    ExperimentRecord,
    ExperimentStore,
    FailureRecord,
    IdempotencyConflictError,
    InteropMismatchError,
    KnowledgeError,
    KnowledgeValidationError,
    assumption_record,
    experiment_begin,
    experiment_commit,
    failure_record,
)

__all__ = [
    "artifacts",
    "config",
    "hashing",
    "search",
    "write_api",
    "pg_store",
    "pg_retrieval",
    "schema",
    "retrieval",
    "embeddings",
    "agent_tools",
    "quantflow_provider",
    "ENV_DSN",
    "ENV_ENABLED",
    "ENV_S3_BUCKET",
    "ENV_S3_ENDPOINT",
    "ENV_S3_REGION",
    "KnowledgeConfig",
    "get_knowledge_config",
    "load_knowledge_config_from_dict",
    "set_knowledge_config",
    "EXECUTION_DOMAIN",
    "FAMILY_DOMAIN",
    "IDEMPOTENCY_DOMAIN",
    "canonical_bytes",
    "canonical_json",
    "execution_hash",
    "experiment_family_hash",
    "make_idempotency_key",
    "normalize",
    "normalize_text",
    "sha256_hex",
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "ExperimentFilter",
    "ExperimentSearchResult",
    "ExperimentSearchStore",
    "experiment_get_by_execution_hash",
    "experiment_search",
    "ASSUMPTION_CATEGORIES",
    "ASSUMPTION_SENSITIVITIES",
    "ASSUMPTION_STATUSES",
    "BEGIN_STATUSES",
    "DATASET_ROLES",
    "EXPERIMENT_STATUSES",
    "FAILURE_CLASSES",
    "OUTCOME_CLASSIFICATIONS",
    "TERMINAL_STATUSES",
    "AssumptionRecord",
    "DuplicateExecutionError",
    "ExperimentNotFoundError",
    "ExperimentRecord",
    "ExperimentStore",
    "FailureRecord",
    "IdempotencyConflictError",
    "InteropMismatchError",
    "KnowledgeError",
    "KnowledgeValidationError",
    "assumption_record",
    "experiment_begin",
    "experiment_commit",
    "failure_record",
    # Upstream retrieval-contracts submodules (union merge batch B3).
    "contract",
    "providers",
    "sources",
    "tools",
    # Upstream retrieval contracts (eager; pydantic/stdlib only).
    "EntityRelationsProvider",
    "KnowledgeEntriesProvider",
    "KnowledgeCapabilities",
    "KnowledgeRequestContext",
    "KnowledgeSource",
    "RetrievalCatalogItem",
    "RetrievalCatalogPage",
    "RetrievalCatalogProvider",
    "RetrievalDocumentsProvider",
    "RetrievalHit",
    "RetrievalProvider",
    "RetrievalResult",
    "ScopedRetrievalProvider",
    "format_result",
    "sources_from_messages",
]

_LAZY_SUBMODULES = frozenset({"pg_store", "pg_retrieval", "schema", "retrieval", "embeddings", "agent_tools", "quantflow_provider", "contract", "providers", "sources", "tools"})


def __getattr__(name: str) -> Any:
    """Lazily import SQLAlchemy-backed submodules (keeps this import light)."""
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

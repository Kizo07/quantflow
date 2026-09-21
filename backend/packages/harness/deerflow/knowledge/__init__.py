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

Import-cycle contract: ``deerflow.config.app_config`` imports
``deerflow.knowledge.config``, so importing this package must never pull in
``deerflow.config``. The eagerly imported modules below depend only on the
stdlib, pydantic, and each other; the SQLAlchemy-backed ``pg_store`` and
``schema`` submodules load lazily via :func:`__getattr__` (explicit
``deerflow.knowledge.pg_store`` / ``deerflow.knowledge.schema`` imports work
directly and are unaffected).
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
    "schema",
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
]

_LAZY_SUBMODULES = frozenset({"pg_store", "schema"})


def __getattr__(name: str) -> Any:
    """Lazily import SQLAlchemy-backed submodules (keeps this import light)."""
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

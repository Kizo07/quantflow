"""Experiment search for the Research Knowledge Plane.

Implements the KB pre-experiment recheck ``experiment_search(spec)``: before
running a backtest, agents query the normalized experiment family to find
equivalent prior runs (same ``family_hash``) and exact reruns (same
``execution_hash``), plus scoped filters over status/outcome/failure class.

Storage boundary: reads go through ``ExperimentSearchStore``, a small
``Protocol`` the integration step binds to PostgreSQL (see the Protocol
docstring for the query mapping). This module performs no I/O of its own.

Lookup paths:

* Pure family lookup (``family_hash`` or ``spec`` with no other filters)
  uses ``find_by_family_hash`` — the indexed KB "we already tested this"
  query (``experiment`` by ``experiment_family_hash``).
* Exact-rerun checks use ``find_by_execution_hash`` (unique lookup).
* Anything with additional filters uses ``search_experiments`` with an
  ``ExperimentFilter`` (structured ``WHERE`` + ``LIMIT``/``OFFSET``).
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from deerflow.knowledge.hashing import experiment_family_hash
from deerflow.knowledge.write_api import (
    EXPERIMENT_STATUSES,
    FAILURE_CLASSES,
    OUTCOME_CLASSIFICATIONS,
    ExperimentRecord,
    KnowledgeValidationError,
)

_HASH_RE = re.compile(r"[0-9a-f]{64}")
DEFAULT_LIMIT = 20
MAX_LIMIT = 100

__all__ = [
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "ExperimentFilter",
    "ExperimentSearchResult",
    "ExperimentSearchStore",
    "experiment_search",
    "experiment_get_by_execution_hash",
]


@dataclass(frozen=True)
class ExperimentFilter:
    """Structured filter set for experiment search (all fields AND-combined)."""

    family_hash: str | None = None
    project_id: str | None = None
    status: str | None = None
    outcome: str | None = None
    failure_class: str | None = None
    hypothesis_contains: str | None = None

    def is_family_only(self) -> bool:
        """Return True when the filter is a pure family lookup (indexed path)."""
        return self.family_hash is not None and self.project_id is None and self.status is None and self.outcome is None and self.failure_class is None and self.hypothesis_contains is None


@dataclass(frozen=True)
class ExperimentSearchResult:
    """One page of experiment search results."""

    experiments: list[ExperimentRecord] = field(default_factory=list)
    limit: int = DEFAULT_LIMIT
    offset: int = 0
    family_hash_used: str | None = None


@runtime_checkable
class ExperimentSearchStore(Protocol):
    """Read boundary for experiment search; the integration step binds this to PostgreSQL.

    PG binding sketch (KB canonical schema): all three methods read the
    ``experiment`` table. ``find_by_execution_hash`` is a unique-index point
    lookup on ``execution_hash``; ``find_by_family_hash`` is an index scan on
    ``experiment_family_hash`` ordered by ``started_at`` with ``LIMIT``/``OFFSET``;
    ``search_experiments`` builds a parameterized ``WHERE`` from the non-None
    ``ExperimentFilter`` fields (``hypothesis_contains`` maps to
    ``hypothesis ILIKE '%' || $1 || '%'`` — substring only; FTS ranking is a
    Phase 2 concern) with the same ordering and pagination.
    """

    def find_by_execution_hash(self, execution_hash: str) -> ExperimentRecord | None:
        """Return the experiment with this execution hash, or None (unique lookup)."""
        ...

    def find_by_family_hash(self, family_hash: str, *, limit: int, offset: int) -> list[ExperimentRecord]:
        """Return one page of experiments in this family, oldest first."""
        ...

    def search_experiments(self, filters: ExperimentFilter, *, limit: int, offset: int) -> list[ExperimentRecord]:
        """Return one page of experiments matching all non-None filter fields, oldest first."""
        ...


def _require_hash(name: str, value: Any) -> str:
    """Validate a 64-char lowercase hex digest."""
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise KnowledgeValidationError(f"{name} must be a 64-character lowercase hex digest, got {value!r}.")
    return value


def _require_pagination(limit: int, offset: int, *, max_limit: int) -> tuple[int, int]:
    """Validate limit/offset (oversized limits are rejected, never clamped)."""
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise KnowledgeValidationError(f"limit must be an int, got {type(limit).__name__}.")
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise KnowledgeValidationError(f"offset must be an int, got {type(offset).__name__}.")
    if limit < 1:
        raise KnowledgeValidationError(f"limit must be >= 1, got {limit}.")
    if limit > max_limit:
        raise KnowledgeValidationError(f"limit {limit} exceeds max_limit {max_limit}.")
    if offset < 0:
        raise KnowledgeValidationError(f"offset must be >= 0, got {offset}.")
    return limit, offset


def _optional_enum(name: str, value: Any, allowed: frozenset[str]) -> str | None:
    """Validate an optional enum filter (None passes through)."""
    if value is None:
        return None
    if not isinstance(value, str) or value not in allowed:
        raise KnowledgeValidationError(f"{name} must be one of {sorted(allowed)}, got {value!r}.")
    return value


def experiment_search(
    store: ExperimentSearchStore,
    *,
    family_hash: str | None = None,
    spec: Mapping[str, Any] | None = None,
    project_id: str | None = None,
    status: str | None = None,
    outcome: str | None = None,
    failure_class: str | None = None,
    hypothesis_contains: str | None = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    max_limit: int = MAX_LIMIT,
) -> ExperimentSearchResult:
    """Search experiments by family identity and/or structured filters.

    ``spec`` is a design mapping (``{"hypothesis": ..., "methodology": ...}``)
    hashed with :func:`deerflow.knowledge.hashing.experiment_family_hash`;
    passing both ``spec`` and ``family_hash`` requires them to agree.

    Args:
        store: Read boundary implementation.
        family_hash: Optional 64-char hex family digest to match.
        spec: Optional design mapping; its family hash becomes the filter.
        project_id: Optional owning-project UUID (validated as UUID).
        status: Optional status filter (planned/running/completed/failed/invalidated).
        outcome: Optional outcome filter (success/failure/inconclusive).
        failure_class: Optional failure-class filter.
        hypothesis_contains: Optional case-insensitive substring over hypothesis prose.
        limit: Page size (1..max_limit).
        offset: Zero-based page offset.
        max_limit: Hard ceiling for ``limit`` (bind to
            ``KnowledgeConfig.max_page_size`` at the call site).

    Returns:
        An ``ExperimentSearchResult`` page; ``family_hash_used`` echoes the
        effective family filter (None when no family scoping was applied).

    Raises:
        KnowledgeValidationError: On invalid filters, pagination, a spec that
            is not a mapping / not canonicalizable, or a spec/family_hash
            disagreement.
    """
    validated_limit, validated_offset = _require_pagination(limit, offset, max_limit=max_limit)
    validated_family = _optional_hash("family_hash", family_hash)

    effective_family = validated_family
    if spec is not None:
        if not isinstance(spec, Mapping):
            raise KnowledgeValidationError(f"spec must be a mapping, got {type(spec).__name__}.")
        try:
            spec_family = experiment_family_hash(spec)
        except (TypeError, ValueError) as exc:
            raise KnowledgeValidationError(f"spec must be JSON-canonicalizable: {exc}") from None
        if effective_family is not None and effective_family != spec_family:
            raise KnowledgeValidationError("family_hash and spec disagree: spec hashes to a different family.")
        effective_family = spec_family

    validated_project: str | None = None
    if project_id is not None:
        import uuid as _uuid

        if not isinstance(project_id, str):
            raise KnowledgeValidationError(f"project_id must be a UUID string, got {type(project_id).__name__}.")
        try:
            validated_project = str(_uuid.UUID(project_id.strip()))
        except ValueError:
            raise KnowledgeValidationError(f"project_id must be a valid UUID, got {project_id!r}.") from None

    validated_status = _optional_enum("status", status, EXPERIMENT_STATUSES)
    validated_outcome = _optional_enum("outcome", outcome, OUTCOME_CLASSIFICATIONS)
    validated_failure_class = _optional_enum("failure_class", failure_class, FAILURE_CLASSES)

    validated_substring: str | None = None
    if hypothesis_contains is not None:
        if not isinstance(hypothesis_contains, str) or not hypothesis_contains.strip():
            raise KnowledgeValidationError("hypothesis_contains must be a non-empty string.")
        validated_substring = hypothesis_contains.strip()

    filters = ExperimentFilter(
        family_hash=effective_family,
        project_id=validated_project,
        status=validated_status,
        outcome=validated_outcome,
        failure_class=validated_failure_class,
        hypothesis_contains=validated_substring,
    )
    if filters.is_family_only():
        assert effective_family is not None
        experiments = store.find_by_family_hash(effective_family, limit=validated_limit, offset=validated_offset)
    else:
        experiments = store.search_experiments(filters, limit=validated_limit, offset=validated_offset)
    return ExperimentSearchResult(experiments=list(experiments), limit=validated_limit, offset=validated_offset, family_hash_used=effective_family)


def _optional_hash(name: str, value: Any) -> str | None:
    """Validate an optional hex-digest filter (None passes through)."""
    if value is None:
        return None
    return _require_hash(name, value)


def experiment_get_by_execution_hash(store: ExperimentSearchStore, execution_hash_value: str) -> ExperimentRecord | None:
    """Return the experiment for an exact execution hash, or None (exact-rerun check).

    Raises:
        KnowledgeValidationError: When the hash is not a 64-char lowercase hex digest.
    """
    validated = _require_hash("execution_hash", execution_hash_value)
    return store.find_by_execution_hash(validated)

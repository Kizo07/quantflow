"""Shared column types and enum vocabularies for the Knowledge Plane schema.

Phase 1 covers the episodic half of the KB canonical schema
(``knowledge_base.md`` § "Proposed canonical schema" and
``implementation_plan.md`` Phase 1): research projects, agent runs,
artifacts, dataset versions, experiments, and assumptions.

Dialect notes
-------------
* Primary keys use :class:`sqlalchemy.Uuid` with default native rendering:
  native ``UUID`` on PostgreSQL, ``CHAR(32)`` on SQLite (which has no
  native UUID type), with :class:`uuid.UUID` values on both.
* JSON columns use :func:`jsonb`: native ``JSONB`` on PostgreSQL, plain
  ``JSON`` on SQLite (dev/test). Timestamps use timezone-aware
  ``DateTime`` (``TIMESTAMPTZ`` on PostgreSQL).

The enum vocabularies below mirror the KB contract. Alembic revision
``0022_knowledge_phase1`` repeats these values literally so the
migration stays a self-contained, immutable snapshot; if a vocabulary
ever changes, add a new revision rather than editing history.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# KB contract vocabularies (knowledge_base.md: artifact record, experiment
# object, assumption object, canonical schema).
ARTIFACT_KINDS: tuple[str, ...] = (
    "dataset_snapshot",
    "source",
    "code",
    "notebook",
    "result",
    "log",
    "chart",
    "environment",
)

AGENT_RUN_STATUSES: tuple[str, ...] = (
    "pending",
    "running",
    "completed",
    "failed",
    "interrupted",
)

EXPERIMENT_STATUSES: tuple[str, ...] = (
    "planned",
    "running",
    "completed",
    "failed",
    "invalidated",
)

OUTCOME_CLASSES: tuple[str, ...] = (
    "success",
    "failure",
    "inconclusive",
)

FAILURE_CLASSES: tuple[str, ...] = (
    "data",
    "code",
    "statistical",
    "execution",
    "hypothesis",
)

DATASET_ROLES: tuple[str, ...] = (
    "features",
    "labels",
    "benchmark",
)

ASSUMPTION_CATEGORIES: tuple[str, ...] = (
    "data",
    "market",
    "execution",
    "statistical",
    "modeling",
)

ASSUMPTION_SENSITIVITIES: tuple[str, ...] = (
    "low",
    "medium",
    "high",
    "unknown",
)

ASSUMPTION_STATUSES: tuple[str, ...] = (
    "active",
    "challenged",
    "invalidated",
)


def jsonb() -> sa.types.TypeEngine:
    """Return a JSON column type that is JSONB on PostgreSQL.

    A fresh instance is returned on every call so each column owns its
    type object. Renders as ``JSONB`` under the ``postgresql`` dialect
    and plain ``JSON`` elsewhere (SQLite dev/test databases).
    """
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def uuid_column(*, primary_key: bool = False, nullable: bool = True) -> sa.Uuid:
    """Return the cross-dialect UUID column type used by every KB table."""
    return sa.Uuid()


def new_uuid() -> uuid.UUID:
    """Return a fresh ``uuid4`` for Python-side primary-key defaults."""
    return uuid.uuid4()


def utcnow() -> datetime:
    """Return the current timezone-aware UTC timestamp for column defaults."""
    return datetime.now(UTC)


def check_in(name: str, column: str, values: tuple[str, ...]) -> sa.CheckConstraint:
    """Build a named ``CHECK (column IN (...))`` constraint.

    Args:
        name: Constraint name; must match the name used in migration
            ``0022_knowledge_phase1`` so fresh-DB ``create_all`` and
            migrated databases carry identical constraint names.
        column: Bare column name the membership test applies to.
        values: Allowed string values (single quotes are escaped).

    Returns:
        A SQLAlchemy :class:`CheckConstraint` usable in ``__table_args__``.
    """
    literals = ", ".join("'" + value.replace("'", "''") + "'" for value in values)
    return sa.CheckConstraint(f"{column} IN ({literals})", name=name)

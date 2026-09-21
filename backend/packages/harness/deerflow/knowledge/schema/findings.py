"""Knowledge Plane Phase 2 ORM model: findings (semantic research memory).

``finding`` is the primary semantic-memory object of the KB canonical
schema: a claim about evidence (``knowledge_base.md`` § "Findings should
be claims, not notes" and § "Proposed canonical schema", plus
``implementation_plan.md`` Phase 2 "Shared recall"). Findings are
claims, not notes: each row carries a stable ``canonical_key`` identity,
a typed ``statement``, a structured ``scope`` window, lifecycle
``status``, a multi-dimensional ``confidence`` profile, valid-time
(``effective_from``/``effective_to``) and transaction-time
(``recorded_at``/``retired_at``) bounds, a provenance anchor
(``created_by_run_id``), and an optional supersession edge
(``supersedes_id``). Evidence edges (``finding_evidence``), conflict
sets, and promotion gates land in Phase 3; this table is the durable
claim ledger those structures annotate.

Phase 2 status contract
-----------------------
Only ``candidate`` rows may be written at this phase: the CHECK
constraint enforces ``status IN ('candidate')`` because no validation
pipeline exists yet to justify any other state. The remaining KB states
(``reviewed``, ``validated``, ``disputed``, ``superseded``, ``rejected``)
are defined in :data:`FINDING_STATUSES` for reference, and Phase 3
widens the CHECK via a new revision (never by editing history).

Retrieval columns
-----------------
* ``search_document`` uses :func:`tsvector`: native ``TSVECTOR`` on
  PostgreSQL, plain ``TEXT`` on SQLite (dev/test). It is populated
  asynchronously by indexing workers (NULL until indexed), and
  ``ix_finding_search_document`` renders ``USING gin`` on PostgreSQL
  (plain btree on SQLite). Indexing failure must never corrupt
  canonical state, so both retrieval columns stay nullable.
* ``embedding`` uses :class:`EmbeddingVector` (768 dimensions): native
  ``VECTOR(768)`` (pgvector) on PostgreSQL, plain ``JSON`` holding a
  list of floats on SQLite with exact round-trip. Phase 2 runs exact
  search; the HNSW/IVFFlat approximate index is deferred per the KB.

PostgreSQL requirements
-----------------------
Production PostgreSQL needs the pgvector ``vector`` extension for the
``embedding`` column: ``CREATE EXTENSION IF NOT EXISTS vector``. Both
provisioning paths handle it — alembic revision
``0023_knowledge_findings`` runs it (PostgreSQL-only, before DDL) and
the metadata-level ``before_create`` listener below covers the
fresh-database ``create_all`` path. The ``pgvector`` Python package is
intentionally *not* a dependency of this module (it is absent from the
test environment): DDL renders ``VECTOR(768)`` without it, NULL/JSON
semantics on SQLite are unaffected, and only live vector reads/writes
against PostgreSQL need the package plus the server extension. The
``search_document`` column and its GIN index need no extension
(core PostgreSQL FTS).

Dialect notes (mirroring :mod:`.types`)
---------------------------------------
Primary keys use :class:`sqlalchemy.Uuid` (native ``UUID`` on
PostgreSQL, ``CHAR(32)`` on SQLite). JSON columns use :func:`jsonb`
(``JSONB`` on PostgreSQL, ``JSON`` on SQLite). Timestamps use
timezone-aware ``DateTime`` (``TIMESTAMPTZ`` on PostgreSQL).
Dimensionality (768) is enforced natively by PostgreSQL; the SQLite
JSON fallback stores whatever list it is given.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.knowledge.schema.types import check_in, jsonb, new_uuid, utcnow
from deerflow.persistence.base import Base

# KB contract vocabularies (knowledge_base.md: finding object). The
# migration ``0023_knowledge_findings`` repeats the CHECK-enforced values
# literally so the revision stays a self-contained, immutable snapshot; any
# vocabulary change ships as a new revision, never an edit.
FINDING_TYPES: tuple[str, ...] = (
    "empirical",
    "methodological",
    "data_quality",
    "failure",
    "prior",
)

#: Full finding lifecycle vocabulary. Only ``candidate`` is CHECK-enforced
#: at this phase (see :data:`FINDING_STATUSES_PHASE2`); the rest become
#: writable when Phase 3 lands the validation pipeline and transitions.
FINDING_STATUSES: tuple[str, ...] = (
    "candidate",
    "reviewed",
    "validated",
    "disputed",
    "superseded",
    "rejected",
)

#: Status values permitted by the Phase 2 CHECK constraint.
FINDING_STATUSES_PHASE2: tuple[str, ...] = ("candidate",)

#: Embedding dimensionality. Single source of truth shared by the ORM
#: column, the alembic DDL, and (later) the embedding workers.
EMBEDDING_DIMENSIONS: int = 768


def tsvector() -> sa.types.TypeEngine:
    """Return a full-text-search column type that is TSVECTOR on PostgreSQL.

    A fresh instance is returned on every call so each column owns its
    type object. Renders as ``TSVECTOR`` under the ``postgresql`` dialect
    (core PostgreSQL FTS, no extension required) and plain ``TEXT``
    elsewhere (SQLite dev/test databases), mirroring :func:`jsonb`.
    """
    return sa.Text().with_variant(postgresql.TSVECTOR(), "postgresql")


class NativeVector(sa.types.UserDefinedType):
    """Render a native pgvector ``VECTOR(n)`` column type on PostgreSQL.

    DDL-only rendering vehicle for :class:`EmbeddingVector`: it carries
    no bind/result processing of its own. Values pass through untouched
    to the driver, so live vector reads/writes against PostgreSQL need
    the ``pgvector`` Python package (plus the server-side ``vector``
    extension) registered by the caller. Never instantiated directly —
    use :class:`EmbeddingVector`, which selects this type only on the
    ``postgresql`` dialect.
    """

    cache_ok = True

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        """Bind the vector dimensionality rendered into ``VECTOR(n)`` DDL."""
        self.dimensions = int(dimensions)

    def get_col_spec(self, **kw) -> str:
        """Return the ``VECTOR(n)`` column specification for DDL."""
        return f"VECTOR({self.dimensions})"


class EmbeddingVector(sa.types.TypeDecorator):
    """Portable fixed-dimension embedding column type.

    * PostgreSQL: native ``VECTOR(dimensions)`` (pgvector) via
      :class:`NativeVector`, with values passed through to the driver.
    * Any other dialect (SQLite dev/test): plain ``JSON`` storing the
      vector as a list of floats, with exact round-trip.

    Dimensionality is enforced natively by PostgreSQL; the SQLite
    fallback stores whatever list it is given.
    """

    impl = sa.JSON
    cache_ok = True

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        """Bind the vector dimensionality (default :data:`EMBEDDING_DIMENSIONS`)."""
        super().__init__()
        self.dimensions = int(dimensions)

    def load_dialect_impl(self, dialect):  # noqa: ANN001, ANN202
        """Select native ``VECTOR(n)`` on PostgreSQL, JSON elsewhere."""
        if dialect.name == "postgresql":
            return NativeVector(self.dimensions)
        return super().load_dialect_impl(dialect)


def embedding_vector(dimensions: int = EMBEDDING_DIMENSIONS) -> EmbeddingVector:
    """Return a fresh :class:`EmbeddingVector` column type.

    A fresh instance is returned on every call so each column owns its
    type object, mirroring :func:`jsonb` and :func:`tsvector`.
    """
    return EmbeddingVector(dimensions)


class FindingRow(Base):
    """One research finding: a versioned, evidence-backed claim.

    Identity is the stable ``canonical_key`` (unique: exact duplicates
    collapse to one row). ``supersedes_id`` chains a finding to the prior
    revision it replaces without destroying historical truth; the row it
    points at keeps its own status until Phase 3 transitions manage the
    lifecycle. ``project_id`` is the optional scope anchor into
    ``research_project`` (NULL for cross-project findings).
    """

    __tablename__ = "finding"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(), primary_key=True, default=new_uuid)
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("research_project.id", ondelete="SET NULL"),
        nullable=True,
    )
    canonical_key: Mapped[str] = mapped_column(sa.Text, nullable=False)
    finding_type: Mapped[str] = mapped_column(sa.Text, nullable=False)
    statement: Mapped[str] = mapped_column(sa.Text, nullable=False)
    scope: Mapped[dict] = mapped_column(jsonb(), nullable=False, default=dict, server_default=sa.text("'{}'"))
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, default="candidate")
    confidence: Mapped[dict] = mapped_column(jsonb(), nullable=False, default=dict, server_default=sa.text("'{}'"))
    effective_from: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    effective_to: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=utcnow)
    retired_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    created_by_run_id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("agent_run.id", ondelete="RESTRICT"),
        nullable=False,
    )
    supersedes_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("finding.id", ondelete="SET NULL"),
        nullable=True,
    )
    search_document: Mapped[str | None] = mapped_column(tsvector(), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(embedding_vector(), nullable=True)

    __table_args__ = (
        sa.UniqueConstraint("canonical_key", name="uq_finding_canonical_key"),
        sa.Index("ix_finding_project", "project_id"),
        sa.Index("ix_finding_type", "finding_type"),
        sa.Index("ix_finding_status", "status"),
        sa.Index("ix_finding_created_by_run", "created_by_run_id"),
        sa.Index("ix_finding_supersedes", "supersedes_id"),
        sa.Index("ix_finding_search_document", "search_document", postgresql_using="gin"),
        check_in("ck_finding_type", "finding_type", FINDING_TYPES),
        check_in("ck_finding_status", "status", FINDING_STATUSES_PHASE2),
    )


def _ensure_pgvector_extension(target, bind, **kw) -> None:
    """Create the pgvector ``vector`` extension ahead of ``create_all`` DDL.

    Registered as a metadata-level ``before_create`` listener, so it runs
    once per ``Base.metadata.create_all`` call. It is a strict no-op on
    every dialect except ``postgresql`` (where the ``embedding``
    ``VECTOR(768)`` column requires the extension) and idempotent there
    via ``IF NOT EXISTS``. The alembic upgrade path runs the equivalent
    step in revision ``0023_knowledge_findings``; this listener covers
    the fresh-database ``create_all`` bootstrap path.
    """
    if bind.dialect.name != "postgresql":
        return
    bind.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))


event.listen(Base.metadata, "before_create", _ensure_pgvector_extension)

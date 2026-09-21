"""Knowledge Plane Phase 2 table: findings (semantic research memory).

Revision ID: 0023_knowledge_findings
Revises: 0022_knowledge_phase1
Create Date: 2026-09-21

Creates ``finding`` per ``knowledge_base.md`` § "Findings should be
claims, not notes" / § "Proposed canonical schema" and
``implementation_plan.md`` Phase 2 ("Shared recall"): the durable claim
ledger (canonical key, typed statement, scope, candidate-only status,
confidence profile, valid/transaction time, provenance anchor,
supersession edge) plus the two asynchronously populated retrieval
columns (``search_document`` TSVECTOR, ``embedding`` VECTOR(768)).

Conventions (matching ``0022_knowledge_phase1`` and the migrations AGENTS.md):

* The ``op.create_table`` is guarded by ``inspector.has_table`` so a
  retried upgrade is a safe no-op, and every index is created through
  ``_ensure_index`` so a partially applied run still converges.
* JSON columns use ``JSON().with_variant(JSONB(), "postgresql")``:
  native ``JSONB`` on PostgreSQL, plain ``JSON`` on SQLite.
* UUID columns use ``sa.Uuid()``: native ``UUID`` on
  PostgreSQL, ``CHAR(32)`` on SQLite.
* ``search_document`` uses ``Text().with_variant(TSVECTOR(),
  "postgresql")``: native ``TSVECTOR`` on PostgreSQL (core FTS, no
  extension required), plain ``TEXT`` on SQLite. Its index renders
  ``USING gin`` on PostgreSQL and a plain btree index on SQLite.
* ``embedding`` uses the frozen ``_EmbeddingVector`` snapshot below:
  native ``VECTOR(768)`` on PostgreSQL, plain ``JSON`` on SQLite.
  ``_ensure_vector_extension`` runs ``CREATE EXTENSION IF NOT EXISTS
  vector`` on PostgreSQL before DDL (no-op elsewhere); the downgrade
  deliberately does not drop the extension (shared resource — other
  objects may depend on it).
* Enum CHECK values are repeated literally here (frozen snapshot) and
  must stay identical to ``deerflow.knowledge.schema.findings``; any
  vocabulary change ships as a new revision, never an edit. The status
  CHECK is candidate-only at this phase; Phase 3 widens it.
* This revision is self-contained: the type helpers duplicate (not
  import) the ORM-side definitions so the migration never depends on
  current model code.

PostgreSQL requirements: the pgvector ``vector`` server extension must
be installable by the migrating role (see ``_ensure_vector_extension``).
The ``pgvector`` Python package is not needed for DDL; only live vector
reads/writes against PostgreSQL need it.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0023_knowledge_findings"
down_revision: str | Sequence[str] | None = "0022_knowledge_phase1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _jsonb() -> sa.types.TypeEngine:
    """Fresh JSONB-on-PostgreSQL / JSON-elsewhere column type."""
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def _uuid() -> sa.Uuid:
    """Fresh cross-dialect UUID column type."""
    return sa.Uuid()


def _tsvector() -> sa.types.TypeEngine:
    """Fresh TSVECTOR-on-PostgreSQL / TEXT-elsewhere column type."""
    return sa.Text().with_variant(postgresql.TSVECTOR(), "postgresql")


class _NativeVector(sa.types.UserDefinedType):
    """Frozen snapshot of the native ``VECTOR(n)`` DDL rendering type.

    Must stay identical to ``NativeVector`` in
    ``deerflow.knowledge.schema.findings``.
    """

    cache_ok = True

    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = int(dimensions)

    def get_col_spec(self, **kw) -> str:
        return f"VECTOR({self.dimensions})"


class _EmbeddingVector(sa.types.TypeDecorator):
    """Frozen snapshot of the portable embedding column type.

    Native ``VECTOR(n)`` on PostgreSQL, plain ``JSON`` elsewhere. Must
    stay identical to ``EmbeddingVector`` in
    ``deerflow.knowledge.schema.findings``.
    """

    impl = sa.JSON
    cache_ok = True

    def __init__(self, dimensions: int = 768) -> None:
        super().__init__()
        self.dimensions = int(dimensions)

    def load_dialect_impl(self, dialect):  # noqa: ANN001, ANN202
        if dialect.name == "postgresql":
            return _NativeVector(self.dimensions)
        return super().load_dialect_impl(dialect)


def _embedding_vector(dimensions: int = 768) -> _EmbeddingVector:
    """Fresh portable embedding column type (VECTOR on PG, JSON elsewhere)."""
    return _EmbeddingVector(dimensions)


def _ensure_vector_extension() -> None:
    """Create pgvector's ``vector`` extension on PostgreSQL; no-op elsewhere.

    The ``embedding VECTOR(768)`` column requires the server extension at
    CREATE TABLE time. ``IF NOT EXISTS`` makes the step idempotent, and
    non-PostgreSQL dialects (SQLite dev/test) skip it entirely.
    """
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))


def _ensure_index(name: str, table: str, columns: list[str], *, unique: bool = False, **dialect_kw) -> None:
    """Create index *name* on *table* unless it already exists.

    Extra keyword arguments (e.g. ``postgresql_using="gin"``) pass
    through to the index DDL and are ignored by dialects that do not
    understand them.
    """
    inspector = sa.inspect(op.get_bind())
    if table not in inspector.get_table_names():
        return
    existing = {index["name"] for index in inspector.get_indexes(table)}
    if name not in existing:
        op.create_index(name, table, columns, unique=unique, **dialect_kw)


def _drop_index_if_exists(name: str, table: str) -> None:
    """Drop index *name* on *table* unless it is already gone."""
    inspector = sa.inspect(op.get_bind())
    if table not in inspector.get_table_names():
        return
    existing = {index["name"] for index in inspector.get_indexes(table)}
    if name in existing:
        op.drop_index(name, table_name=table)


def upgrade() -> None:
    _ensure_vector_extension()
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())

    if "finding" not in tables:
        op.create_table(
            "finding",
            sa.Column("id", _uuid(), nullable=False),
            sa.Column("project_id", _uuid(), nullable=True),
            sa.Column("canonical_key", sa.Text(), nullable=False),
            sa.Column("finding_type", sa.Text(), nullable=False),
            sa.Column("statement", sa.Text(), nullable=False),
            sa.Column("scope", _jsonb(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("confidence", _jsonb(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("effective_from", sa.DateTime(timezone=True), nullable=True),
            sa.Column("effective_to", sa.DateTime(timezone=True), nullable=True),
            sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_by_run_id", _uuid(), nullable=False),
            sa.Column("supersedes_id", _uuid(), nullable=True),
            sa.Column("search_document", _tsvector(), nullable=True),
            sa.Column("embedding", _embedding_vector(), nullable=True),
            sa.ForeignKeyConstraint(["project_id"], ["research_project.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["created_by_run_id"], ["agent_run.id"], ondelete="RESTRICT"),
            sa.ForeignKeyConstraint(["supersedes_id"], ["finding.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("canonical_key", name="uq_finding_canonical_key"),
            sa.CheckConstraint("finding_type IN ('empirical', 'methodological', 'data_quality', 'failure', 'prior')", name="ck_finding_type"),
            sa.CheckConstraint("status IN ('candidate')", name="ck_finding_status"),
        )
    _ensure_index("ix_finding_project", "finding", ["project_id"])
    _ensure_index("ix_finding_type", "finding", ["finding_type"])
    _ensure_index("ix_finding_status", "finding", ["status"])
    _ensure_index("ix_finding_created_by_run", "finding", ["created_by_run_id"])
    _ensure_index("ix_finding_supersedes", "finding", ["supersedes_id"])
    _ensure_index("ix_finding_search_document", "finding", ["search_document"], postgresql_using="gin")


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())

    _drop_index_if_exists("ix_finding_search_document", "finding")
    _drop_index_if_exists("ix_finding_supersedes", "finding")
    _drop_index_if_exists("ix_finding_created_by_run", "finding")
    _drop_index_if_exists("ix_finding_status", "finding")
    _drop_index_if_exists("ix_finding_type", "finding")
    _drop_index_if_exists("ix_finding_project", "finding")
    if "finding" in tables:
        op.drop_table("finding")

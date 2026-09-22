"""Knowledge Plane Phase 3 column: experiment embeddings (semantic recall).

Revision ID: 0028_knowledge_experiment_embeddings
Revises: 0027_merge_knowledge_upstream
Create Date: 2026-09-22

Adds the nullable ``experiment.embedding`` retrieval column per the
Phase 3 plan (experiment embeddings): native ``VECTOR(768)``
(pgvector) on PostgreSQL, plain ``JSON`` holding a list of floats on
SQLite with exact round-trip. The column mirrors ``finding.embedding``
(``0023_knowledge_findings``): it is populated asynchronously by
embedding workers (NULL until indexed), runs exact search, and carries
no approximate index.

Conventions (matching ``0023_knowledge_findings`` and the migrations AGENTS.md):

* The column change goes through the idempotent helpers in
  ``migrations/_helpers.py`` (``safe_add_column`` / ``safe_drop_column``)
  so a retried upgrade or downgrade is a safe no-op, including against
  a database where the ``experiment`` table is absent.
* ``embedding`` uses the frozen ``_EmbeddingVector`` snapshot below:
  native ``VECTOR(768)`` on PostgreSQL, plain ``JSON`` on SQLite.
  ``_ensure_vector_extension`` runs ``CREATE EXTENSION IF NOT EXISTS
  vector`` on PostgreSQL before DDL (no-op elsewhere); the downgrade
  deliberately does not drop the extension (shared resource — other
  objects may depend on it).
* This revision is self-contained: the type helpers duplicate (not
  import) the ORM-side definitions so the migration never depends on
  current model code.
* The column is nullable with no server default, so rows written before
  this revision stay valid and old readers may omit it.

PostgreSQL requirements: the pgvector ``vector`` server extension must
be installable by the migrating role (see ``_ensure_vector_extension``).
The ``pgvector`` Python package is not needed for DDL; only live vector
reads/writes against PostgreSQL need it.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0028_knowledge_experiment_embeddings"
down_revision: str | Sequence[str] | None = "0027_merge_knowledge_upstream"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


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
    ALTER TABLE time. ``IF NOT EXISTS`` makes the step idempotent, and
    non-PostgreSQL dialects (SQLite dev/test) skip it entirely.
    """
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))


def upgrade() -> None:
    from deerflow.persistence.migrations._helpers import safe_add_column

    _ensure_vector_extension()
    safe_add_column("experiment", sa.Column("embedding", _embedding_vector(), nullable=True))


def downgrade() -> None:
    from deerflow.persistence.migrations._helpers import safe_drop_column

    safe_drop_column("experiment", "embedding")

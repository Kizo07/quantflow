"""Knowledge Plane Phase 1 ORM models: artifacts and dataset versions.

``artifact`` is the immutable, content-addressed evidence record: every
significant output (dataset snapshot, source, code bundle, notebook,
result, log, chart, environment) is stored once under its SHA-256 digest
and never mutated. A changed file is a new artifact row.

``dataset_version`` separates the logical dataset name (``dataset_key``,
e.g. "crsp-daily-equities") from the exact snapshot used on a run
(``vintage_at`` + ``schema_hash`` + backing ``artifact_id``), so
point-in-time provenance survives dataset refreshes.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.knowledge.schema.types import ARTIFACT_KINDS, check_in, jsonb, new_uuid, utcnow
from deerflow.persistence.base import Base


class ArtifactRow(Base):
    """One immutable content-addressed artifact (evidence)."""

    __tablename__ = "artifact"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(), primary_key=True, default=new_uuid)
    sha256: Mapped[str] = mapped_column(sa.Text, nullable=False)
    kind: Mapped[str] = mapped_column(sa.Text, nullable=False)
    storage_uri: Mapped[str] = mapped_column(sa.Text, nullable=False)
    media_type: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    byte_size: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True)
    created_by_run_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("agent_run.id", ondelete="SET NULL"),
        nullable=True,
    )
    # The DB column is named ``metadata`` per the KB canonical schema; the
    # attribute is renamed because ``metadata`` is reserved on the
    # declarative base (``Base.metadata``).
    artifact_metadata: Mapped[dict] = mapped_column("metadata", jsonb(), nullable=False, default=dict, server_default=sa.text("'{}'"))
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=utcnow)

    __table_args__ = (
        sa.UniqueConstraint("sha256", name="uq_artifact_sha256"),
        sa.Index("ix_artifact_created_by_run", "created_by_run_id"),
        check_in("ck_artifact_kind", "kind", ARTIFACT_KINDS),
    )


class DatasetVersionRow(Base):
    """One exact version/vintage of a logical dataset."""

    __tablename__ = "dataset_version"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(), primary_key=True, default=new_uuid)
    dataset_key: Mapped[str] = mapped_column(sa.Text, nullable=False)
    provider: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    vintage_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    coverage_start: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    coverage_end: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    schema_hash: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("artifact.id", ondelete="SET NULL"),
        nullable=True,
    )
    lineage: Mapped[dict] = mapped_column(jsonb(), nullable=False, default=dict, server_default=sa.text("'{}'"))
    access_metadata: Mapped[dict] = mapped_column(jsonb(), nullable=False, default=dict, server_default=sa.text("'{}'"))
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=utcnow)

    __table_args__ = (
        sa.Index("ix_dataset_version_key", "dataset_key"),
        sa.Index("ix_dataset_version_key_vintage", "dataset_key", "vintage_at"),
        sa.Index("ix_dataset_version_artifact", "artifact_id"),
    )

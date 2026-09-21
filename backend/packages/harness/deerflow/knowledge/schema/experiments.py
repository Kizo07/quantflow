"""Knowledge Plane Phase 1 ORM models: experiments and assumptions.

``experiment`` is the first-class episodic record: hypothesis, method,
parameters, metrics, outcome, and stable pointers to the dataset
versions, code/environment artifacts, and result artifacts that make the
run reproducible. Two signatures separate concerns:

* ``experiment_family_hash`` groups conceptually equivalent research
  designs (replication search).
* ``execution_hash`` (unique) identifies one exact computational
  configuration — same code, data, parameters, environment — so reruns
  are an exact database lookup, never a similarity guess.

``experiment_dataset`` / ``experiment_artifact`` are typed link tables
(role-qualified edges). ``assumption`` records the explicit assumptions
an experiment depends on (category, sensitivity, tested flag, status)
instead of burying them in prose.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.knowledge.schema.types import (
    ASSUMPTION_CATEGORIES,
    ASSUMPTION_SENSITIVITIES,
    ASSUMPTION_STATUSES,
    DATASET_ROLES,
    EXPERIMENT_STATUSES,
    FAILURE_CLASSES,
    OUTCOME_CLASSES,
    check_in,
    jsonb,
    new_uuid,
)
from deerflow.persistence.base import Base


class ExperimentRow(Base):
    """One experiment execution with full reproducibility pointers."""

    __tablename__ = "experiment"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(), primary_key=True, default=new_uuid)
    project_id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("research_project.id", ondelete="RESTRICT"),
        nullable=False,
    )
    created_by_run_id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("agent_run.id", ondelete="RESTRICT"),
        nullable=False,
    )
    experiment_family_hash: Mapped[str] = mapped_column(sa.Text, nullable=False)
    execution_hash: Mapped[str] = mapped_column(sa.Text, nullable=False)
    hypothesis: Mapped[str] = mapped_column(sa.Text, nullable=False)
    methodology: Mapped[dict] = mapped_column(jsonb(), nullable=False, default=dict, server_default=sa.text("'{}'"))
    parameters: Mapped[dict] = mapped_column(jsonb(), nullable=False, default=dict, server_default=sa.text("'{}'"))
    metrics: Mapped[dict | None] = mapped_column(jsonb(), nullable=True)
    code_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("artifact.id", ondelete="SET NULL"),
        nullable=True,
    )
    environment_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("artifact.id", ondelete="SET NULL"),
        nullable=True,
    )
    outcome: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    failure_class: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, default="planned")
    parent_experiment_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("experiment.id", ondelete="SET NULL"),
        nullable=True,
    )
    replicated_experiment_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("experiment.id", ondelete="SET NULL"),
        nullable=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)

    __table_args__ = (
        sa.UniqueConstraint("execution_hash", name="uq_experiment_execution_hash"),
        sa.Index("ix_experiment_family_hash", "experiment_family_hash"),
        sa.Index("ix_experiment_project", "project_id"),
        sa.Index("ix_experiment_created_by_run", "created_by_run_id"),
        sa.Index("ix_experiment_status", "status"),
        check_in("ck_experiment_status", "status", EXPERIMENT_STATUSES),
        check_in("ck_experiment_outcome", "outcome", OUTCOME_CLASSES),
        check_in("ck_experiment_failure_class", "failure_class", FAILURE_CLASSES),
    )


class ExperimentDatasetRow(Base):
    """Role-qualified link between an experiment and a dataset version."""

    __tablename__ = "experiment_dataset"

    experiment_id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("experiment.id", ondelete="CASCADE"),
        primary_key=True,
    )
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("dataset_version.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    role: Mapped[str] = mapped_column(sa.Text, nullable=False, primary_key=True)

    __table_args__ = (
        sa.Index("ix_experiment_dataset_dataset", "dataset_version_id"),
        check_in("ck_experiment_dataset_role", "role", DATASET_ROLES),
    )


class ExperimentArtifactRow(Base):
    """Role-qualified link between an experiment and an artifact.

    ``role`` is free-form (``code``, ``environment``, ``result``, ``log``,
    ``chart``, ``notebook``, ``input``, ...) and intentionally carries no
    CHECK constraint: new producers must be able to add roles without a
    schema migration.
    """

    __tablename__ = "experiment_artifact"

    experiment_id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("experiment.id", ondelete="CASCADE"),
        primary_key=True,
    )
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("artifact.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    role: Mapped[str] = mapped_column(sa.Text, nullable=False, primary_key=True)

    __table_args__ = (sa.Index("ix_experiment_artifact_artifact", "artifact_id"),)


class AssumptionRow(Base):
    """One explicit assumption an experiment depends on."""

    __tablename__ = "assumption"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(), primary_key=True, default=new_uuid)
    experiment_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("experiment.id", ondelete="SET NULL"),
        nullable=True,
    )
    statement: Mapped[str] = mapped_column(sa.Text, nullable=False)
    category: Mapped[str] = mapped_column(sa.Text, nullable=False)
    sensitivity: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    tested: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, default="active")
    evidence_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("artifact.id", ondelete="SET NULL"),
        nullable=True,
    )

    __table_args__ = (
        sa.Index("ix_assumption_experiment", "experiment_id"),
        sa.Index("ix_assumption_status", "status"),
        sa.Index("ix_assumption_category", "category"),
        check_in("ck_assumption_category", "category", ASSUMPTION_CATEGORIES),
        check_in("ck_assumption_sensitivity", "sensitivity", ASSUMPTION_SENSITIVITIES),
        check_in("ck_assumption_status", "status", ASSUMPTION_STATUSES),
    )

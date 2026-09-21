"""Knowledge Plane Phase 1 tables (episodic half of the KB canonical schema).

Revision ID: 0022_knowledge_phase1
Revises: 0019_thread_incarnations
Create Date: 2026-09-20

Creates ``research_project``, ``agent_run``, ``artifact``,
``dataset_version``, ``experiment``, ``experiment_dataset``,
``experiment_artifact``, and ``assumption`` per
``knowledge_base.md`` § "Proposed canonical schema" and
``implementation_plan.md`` Phase 1.

Conventions (matching ``0019_projects`` and the migrations AGENTS.md):

* Each ``op.create_table`` is guarded by ``inspector.has_table`` so a
  retried upgrade is a safe no-op, and every index is created through
  ``_ensure_index`` so a partially applied run still converges.
* JSON columns use ``JSON().with_variant(JSONB(), "postgresql")``:
  native ``JSONB`` on PostgreSQL, plain ``JSON`` on SQLite.
* UUID columns use ``sa.Uuid()``: native ``UUID`` on
  PostgreSQL, ``CHAR(32)`` on SQLite.
* Enum CHECK values are repeated literally here (frozen snapshot) and
  must stay identical to ``deerflow.knowledge.schema.types``; any
  vocabulary change ships as a new revision, never an edit.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0022_knowledge_phase1"
down_revision: str | Sequence[str] | None = "0019_thread_incarnations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _jsonb() -> sa.types.TypeEngine:
    """Fresh JSONB-on-PostgreSQL / JSON-elsewhere column type."""
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def _uuid() -> sa.Uuid:
    """Fresh cross-dialect UUID column type."""
    return sa.Uuid()


def _ensure_index(name: str, table: str, columns: list[str], *, unique: bool = False) -> None:
    """Create index *name* on *table* unless it already exists."""
    inspector = sa.inspect(op.get_bind())
    if table not in inspector.get_table_names():
        return
    existing = {index["name"] for index in inspector.get_indexes(table)}
    if name not in existing:
        op.create_index(name, table, columns, unique=unique)


def _drop_index_if_exists(name: str, table: str) -> None:
    """Drop index *name* on *table* unless it is already gone."""
    inspector = sa.inspect(op.get_bind())
    if table not in inspector.get_table_names():
        return
    existing = {index["name"] for index in inspector.get_indexes(table)}
    if name in existing:
        op.drop_index(name, table_name=table)


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())

    if "research_project" not in tables:
        op.create_table(
            "research_project",
            sa.Column("id", _uuid(), nullable=False),
            sa.Column("parent_project_id", _uuid(), nullable=True),
            sa.Column("name", sa.Text(), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("visibility_scope", _jsonb(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["parent_project_id"], ["research_project.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
        )
    _ensure_index("ix_research_project_parent", "research_project", ["parent_project_id"])

    if "agent_run" not in tables:
        op.create_table(
            "agent_run",
            sa.Column("id", _uuid(), nullable=False),
            sa.Column("project_id", _uuid(), nullable=False),
            sa.Column("parent_run_id", _uuid(), nullable=True),
            sa.Column("agent_type", sa.Text(), nullable=False),
            sa.Column("model_config", _jsonb(), nullable=True),
            sa.Column("task", sa.Text(), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("status", sa.Text(), nullable=False),
            sa.ForeignKeyConstraint(["project_id"], ["research_project.id"], ondelete="RESTRICT"),
            sa.ForeignKeyConstraint(["parent_run_id"], ["agent_run.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
            sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'interrupted')", name="ck_agent_run_status"),
        )
    _ensure_index("ix_agent_run_project", "agent_run", ["project_id"])
    _ensure_index("ix_agent_run_parent", "agent_run", ["parent_run_id"])
    _ensure_index("ix_agent_run_status", "agent_run", ["status"])

    if "artifact" not in tables:
        op.create_table(
            "artifact",
            sa.Column("id", _uuid(), nullable=False),
            sa.Column("sha256", sa.Text(), nullable=False),
            sa.Column("kind", sa.Text(), nullable=False),
            sa.Column("storage_uri", sa.Text(), nullable=False),
            sa.Column("media_type", sa.Text(), nullable=True),
            sa.Column("byte_size", sa.BigInteger(), nullable=True),
            sa.Column("created_by_run_id", _uuid(), nullable=True),
            sa.Column("metadata", _jsonb(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["created_by_run_id"], ["agent_run.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("sha256", name="uq_artifact_sha256"),
            sa.CheckConstraint("kind IN ('dataset_snapshot', 'source', 'code', 'notebook', 'result', 'log', 'chart', 'environment')", name="ck_artifact_kind"),
        )
    _ensure_index("ix_artifact_created_by_run", "artifact", ["created_by_run_id"])

    if "dataset_version" not in tables:
        op.create_table(
            "dataset_version",
            sa.Column("id", _uuid(), nullable=False),
            sa.Column("dataset_key", sa.Text(), nullable=False),
            sa.Column("provider", sa.Text(), nullable=True),
            sa.Column("vintage_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("coverage_start", sa.DateTime(timezone=True), nullable=True),
            sa.Column("coverage_end", sa.DateTime(timezone=True), nullable=True),
            sa.Column("schema_hash", sa.Text(), nullable=True),
            sa.Column("artifact_id", _uuid(), nullable=True),
            sa.Column("lineage", _jsonb(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("access_metadata", _jsonb(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["artifact_id"], ["artifact.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
        )
    _ensure_index("ix_dataset_version_key", "dataset_version", ["dataset_key"])
    _ensure_index("ix_dataset_version_key_vintage", "dataset_version", ["dataset_key", "vintage_at"])
    _ensure_index("ix_dataset_version_artifact", "dataset_version", ["artifact_id"])

    if "experiment" not in tables:
        op.create_table(
            "experiment",
            sa.Column("id", _uuid(), nullable=False),
            sa.Column("project_id", _uuid(), nullable=False),
            sa.Column("created_by_run_id", _uuid(), nullable=False),
            sa.Column("experiment_family_hash", sa.Text(), nullable=False),
            sa.Column("execution_hash", sa.Text(), nullable=False),
            sa.Column("hypothesis", sa.Text(), nullable=False),
            sa.Column("methodology", _jsonb(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("parameters", _jsonb(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column("metrics", _jsonb(), nullable=True),
            sa.Column("code_artifact_id", _uuid(), nullable=True),
            sa.Column("environment_artifact_id", _uuid(), nullable=True),
            sa.Column("outcome", sa.Text(), nullable=True),
            sa.Column("failure_class", sa.Text(), nullable=True),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("parent_experiment_id", _uuid(), nullable=True),
            sa.Column("replicated_experiment_id", _uuid(), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["project_id"], ["research_project.id"], ondelete="RESTRICT"),
            sa.ForeignKeyConstraint(["created_by_run_id"], ["agent_run.id"], ondelete="RESTRICT"),
            sa.ForeignKeyConstraint(["code_artifact_id"], ["artifact.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["environment_artifact_id"], ["artifact.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["parent_experiment_id"], ["experiment.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["replicated_experiment_id"], ["experiment.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("execution_hash", name="uq_experiment_execution_hash"),
            sa.CheckConstraint("status IN ('planned', 'running', 'completed', 'failed', 'invalidated')", name="ck_experiment_status"),
            sa.CheckConstraint("outcome IN ('success', 'failure', 'inconclusive')", name="ck_experiment_outcome"),
            sa.CheckConstraint("failure_class IN ('data', 'code', 'statistical', 'execution', 'hypothesis')", name="ck_experiment_failure_class"),
        )
    _ensure_index("ix_experiment_family_hash", "experiment", ["experiment_family_hash"])
    _ensure_index("ix_experiment_project", "experiment", ["project_id"])
    _ensure_index("ix_experiment_created_by_run", "experiment", ["created_by_run_id"])
    _ensure_index("ix_experiment_status", "experiment", ["status"])

    if "experiment_dataset" not in tables:
        op.create_table(
            "experiment_dataset",
            sa.Column("experiment_id", _uuid(), nullable=False),
            sa.Column("dataset_version_id", _uuid(), nullable=False),
            sa.Column("role", sa.Text(), nullable=False),
            sa.ForeignKeyConstraint(["experiment_id"], ["experiment.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["dataset_version_id"], ["dataset_version.id"], ondelete="RESTRICT"),
            sa.PrimaryKeyConstraint("experiment_id", "dataset_version_id", "role"),
            sa.CheckConstraint("role IN ('features', 'labels', 'benchmark')", name="ck_experiment_dataset_role"),
        )
    _ensure_index("ix_experiment_dataset_dataset", "experiment_dataset", ["dataset_version_id"])

    if "experiment_artifact" not in tables:
        op.create_table(
            "experiment_artifact",
            sa.Column("experiment_id", _uuid(), nullable=False),
            sa.Column("artifact_id", _uuid(), nullable=False),
            sa.Column("role", sa.Text(), nullable=False),
            sa.ForeignKeyConstraint(["experiment_id"], ["experiment.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["artifact_id"], ["artifact.id"], ondelete="RESTRICT"),
            sa.PrimaryKeyConstraint("experiment_id", "artifact_id", "role"),
        )
    _ensure_index("ix_experiment_artifact_artifact", "experiment_artifact", ["artifact_id"])

    if "assumption" not in tables:
        op.create_table(
            "assumption",
            sa.Column("id", _uuid(), nullable=False),
            sa.Column("experiment_id", _uuid(), nullable=True),
            sa.Column("statement", sa.Text(), nullable=False),
            sa.Column("category", sa.Text(), nullable=False),
            sa.Column("sensitivity", sa.Text(), nullable=True),
            sa.Column("tested", sa.Boolean(), nullable=False),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("evidence_artifact_id", _uuid(), nullable=True),
            sa.ForeignKeyConstraint(["experiment_id"], ["experiment.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["evidence_artifact_id"], ["artifact.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
            sa.CheckConstraint("category IN ('data', 'market', 'execution', 'statistical', 'modeling')", name="ck_assumption_category"),
            sa.CheckConstraint("sensitivity IN ('low', 'medium', 'high', 'unknown')", name="ck_assumption_sensitivity"),
            sa.CheckConstraint("status IN ('active', 'challenged', 'invalidated')", name="ck_assumption_status"),
        )
    _ensure_index("ix_assumption_experiment", "assumption", ["experiment_id"])
    _ensure_index("ix_assumption_status", "assumption", ["status"])
    _ensure_index("ix_assumption_category", "assumption", ["category"])


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())

    _drop_index_if_exists("ix_assumption_category", "assumption")
    _drop_index_if_exists("ix_assumption_status", "assumption")
    _drop_index_if_exists("ix_assumption_experiment", "assumption")
    if "assumption" in tables:
        op.drop_table("assumption")

    _drop_index_if_exists("ix_experiment_artifact_artifact", "experiment_artifact")
    if "experiment_artifact" in tables:
        op.drop_table("experiment_artifact")

    _drop_index_if_exists("ix_experiment_dataset_dataset", "experiment_dataset")
    if "experiment_dataset" in tables:
        op.drop_table("experiment_dataset")

    _drop_index_if_exists("ix_experiment_status", "experiment")
    _drop_index_if_exists("ix_experiment_created_by_run", "experiment")
    _drop_index_if_exists("ix_experiment_project", "experiment")
    _drop_index_if_exists("ix_experiment_family_hash", "experiment")
    if "experiment" in tables:
        op.drop_table("experiment")

    _drop_index_if_exists("ix_dataset_version_artifact", "dataset_version")
    _drop_index_if_exists("ix_dataset_version_key_vintage", "dataset_version")
    _drop_index_if_exists("ix_dataset_version_key", "dataset_version")
    if "dataset_version" in tables:
        op.drop_table("dataset_version")

    _drop_index_if_exists("ix_artifact_created_by_run", "artifact")
    if "artifact" in tables:
        op.drop_table("artifact")

    _drop_index_if_exists("ix_agent_run_status", "agent_run")
    _drop_index_if_exists("ix_agent_run_parent", "agent_run")
    _drop_index_if_exists("ix_agent_run_project", "agent_run")
    if "agent_run" in tables:
        op.drop_table("agent_run")

    _drop_index_if_exists("ix_research_project_parent", "research_project")
    if "research_project" in tables:
        op.drop_table("research_project")

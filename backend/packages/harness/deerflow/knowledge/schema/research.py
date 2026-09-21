"""Knowledge Plane Phase 1 ORM models: research projects and agent runs.

``research_project`` is the organizational scope root of the KB canonical
schema. It is deliberately distinct from the harness ``projects`` table
(user-owned chat organization): a research project scopes experiments,
runs, and findings, and nests via ``parent_project_id`` to mirror the
``organization -> domain -> strategy -> project`` visibility ladder.

``agent_run`` records a single agent execution bound to a research
project. It complements — not replaces — the harness ``runs`` table,
which tracks Gateway/LangGraph conversational runs; an agent run is the
provenance anchor (``created_by_run_id``) for every KB row a research
agent writes.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.knowledge.schema.types import AGENT_RUN_STATUSES, check_in, jsonb, new_uuid, utcnow
from deerflow.persistence.base import Base


class ResearchProjectRow(Base):
    """One research project: the scope root for experiments and runs."""

    __tablename__ = "research_project"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(), primary_key=True, default=new_uuid)
    parent_project_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("research_project.id", ondelete="SET NULL"),
        nullable=True,
    )
    name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    description: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    visibility_scope: Mapped[dict] = mapped_column(jsonb(), nullable=False, default=dict, server_default=sa.text("'{}'"))
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=utcnow)

    __table_args__ = (sa.Index("ix_research_project_parent", "parent_project_id"),)


class AgentRunRow(Base):
    """One agent execution bound to a research project."""

    __tablename__ = "agent_run"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(), primary_key=True, default=new_uuid)
    project_id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("research_project.id", ondelete="RESTRICT"),
        nullable=False,
    )
    parent_run_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.Uuid(),
        sa.ForeignKey("agent_run.id", ondelete="SET NULL"),
        nullable=True,
    )
    agent_type: Mapped[str] = mapped_column(sa.Text, nullable=False)
    model_config: Mapped[dict | None] = mapped_column(jsonb(), nullable=True)
    task: Mapped[str] = mapped_column(sa.Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, default="running")

    __table_args__ = (
        sa.Index("ix_agent_run_project", "project_id"),
        sa.Index("ix_agent_run_parent", "parent_run_id"),
        sa.Index("ix_agent_run_status", "status"),
        check_in("ck_agent_run_status", "status", AGENT_RUN_STATUSES),
    )

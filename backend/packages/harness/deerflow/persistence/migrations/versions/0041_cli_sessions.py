"""CLI-session registry + launch presets.

Revision ID: 0041_cli_sessions
Revises: 0040_merge_quantflow_upstream
Create Date: 2026-10-14
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0041_cli_sessions"
down_revision: str | Sequence[str] | None = "0040_merge_quantflow_upstream"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("cli_sessions"):
        op.create_table(
            "cli_sessions",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column("session_ref", sa.String(length=128), nullable=True),
            sa.Column("resumed_from", sa.String(length=64), nullable=True),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("mode", sa.String(length=16), nullable=False),
            sa.Column("model", sa.String(length=128), nullable=True),
            sa.Column("provider", sa.String(length=16), nullable=True),
            sa.Column("preset", sa.String(length=64), nullable=True),
            sa.Column("effort", sa.String(length=16), nullable=True),
            sa.Column("contributor_mode", sa.Boolean(), nullable=True),
            sa.Column("approval_mode", sa.String(length=16), nullable=True),
            sa.Column("sandbox_network", sa.String(length=32), nullable=True),
            sa.Column("trust_workspace", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("disable_approval", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("disable_sandbox", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("yolo", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("workspace", sa.String(length=512), nullable=True),
            sa.Column("prompt", sa.Text(), nullable=True),
            sa.Column("pid", sa.Integer(), nullable=True),
            sa.Column("tmux_session", sa.String(length=64), nullable=True),
            sa.Column("log_path", sa.String(length=1024), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_by", sa.String(length=64), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        with op.batch_alter_table("cli_sessions", schema=None) as batch_op:
            batch_op.create_index("ix_cli_sessions_status", ["status"], unique=False)
            batch_op.create_index("ix_cli_sessions_created_by", ["created_by"], unique=False)
            batch_op.create_index("ix_cli_sessions_owner_created", ["created_by", "created_at"], unique=False)

    if not inspector.has_table("cli_presets"):
        op.create_table(
            "cli_presets",
            sa.Column("name", sa.String(length=64), nullable=False),
            sa.Column("model", sa.String(length=128), nullable=True),
            sa.Column("provider", sa.String(length=16), nullable=True),
            sa.Column("preset", sa.String(length=64), nullable=True),
            sa.Column("effort", sa.String(length=16), nullable=True),
            sa.Column("contributor_mode", sa.Boolean(), nullable=True),
            sa.Column("yolo", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("approval_mode", sa.String(length=16), nullable=True),
            sa.Column("updated_by", sa.String(length=64), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("name"),
        )

    cli_presets = sa.table(
        "cli_presets",
        sa.column("name", sa.String),
        sa.column("model", sa.String),
        sa.column("effort", sa.String),
        sa.column("contributor_mode", sa.Boolean),
        sa.column("yolo", sa.Boolean),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )
    existing = bind.execute(sa.select(sa.func.count()).select_from(cli_presets).where(cli_presets.c.name == "muse-1.3")).scalar()
    if not existing:
        bind.execute(
            cli_presets.insert().values(
                name="muse-1.3",
                model="muse-spark-1.3",
                effort="ultra",
                contributor_mode=True,
                yolo=True,
                updated_at=datetime.now(UTC),
            )
        )


def downgrade() -> None:
    op.drop_table("cli_presets")
    op.drop_table("cli_sessions")

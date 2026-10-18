"""Persistent-bot singleton record.

Revision ID: 0043_persistent_bot
Revises: 0042_cli_audit
Create Date: 2026-10-18
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0043_persistent_bot"
down_revision: str | Sequence[str] | None = "0042_cli_audit"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("persistent_bot"):
        op.create_table(
            "persistent_bot",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.String(length=64), nullable=False),
            sa.Column("agent_name", sa.String(length=128), nullable=False),
            sa.Column("thread_id", sa.String(length=64), nullable=False),
            sa.Column("heartbeat_task_id", sa.String(length=64), nullable=True),
            sa.Column("user_paused", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.CheckConstraint("id = 1", name="ck_persistent_bot_singleton"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("thread_id"),
            sa.UniqueConstraint("heartbeat_task_id"),
        )
        with op.batch_alter_table("persistent_bot", schema=None) as batch_op:
            batch_op.create_index("ix_persistent_bot_user_id", ["user_id"], unique=False)
            batch_op.create_index("ix_persistent_bot_thread_id", ["thread_id"], unique=False)


def downgrade() -> None:
    op.drop_table("persistent_bot")

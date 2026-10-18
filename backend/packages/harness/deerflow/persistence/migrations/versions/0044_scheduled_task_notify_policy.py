"""Per-task proactive-notification policy on scheduled tasks.

Revision ID: 0044_scheduled_task_notify_policy
Revises: 0043_persistent_bot
Create Date: 2026-10-18
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0044_scheduled_task_notify_policy"
down_revision: str | Sequence[str] | None = "0043_persistent_bot"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("scheduled_tasks"):
        return
    existing = {column["name"] for column in inspector.get_columns("scheduled_tasks")}
    with op.batch_alter_table("scheduled_tasks", schema=None) as batch_op:
        if "notify_policy" not in existing:
            batch_op.add_column(sa.Column("notify_policy", sa.String(length=16), nullable=False, server_default="summary"))
        if "notify_quiet" not in existing:
            batch_op.add_column(sa.Column("notify_quiet", sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("scheduled_tasks"):
        return
    existing = {column["name"] for column in inspector.get_columns("scheduled_tasks")}
    with op.batch_alter_table("scheduled_tasks", schema=None) as batch_op:
        if "notify_quiet" in existing:
            batch_op.drop_column("notify_quiet")
        if "notify_policy" in existing:
            batch_op.drop_column("notify_policy")

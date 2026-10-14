"""CLI-harness audit log.

Revision ID: 0042_cli_audit
Revises: 0041_cli_sessions
Create Date: 2026-10-14
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0042_cli_audit"
down_revision: str | Sequence[str] | None = "0041_cli_sessions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("cli_audit"):
        op.create_table(
            "cli_audit",
            sa.Column("id", sa.Integer(), nullable=False, autoincrement=True),
            sa.Column("actor", sa.String(length=64), nullable=False),
            sa.Column("action", sa.String(length=32), nullable=False),
            sa.Column("session_id", sa.String(length=64), nullable=True),
            sa.Column("command", sa.Text(), nullable=True),
            sa.Column("preset", sa.String(length=64), nullable=True),
            sa.Column("preset_version", sa.String(length=64), nullable=True),
            sa.Column("result", sa.String(length=64), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        with op.batch_alter_table("cli_audit", schema=None) as batch_op:
            batch_op.create_index("ix_cli_audit_actor", ["actor"], unique=False)
            batch_op.create_index("ix_cli_audit_action", ["action"], unique=False)
            batch_op.create_index("ix_cli_audit_session_id", ["session_id"], unique=False)


def downgrade() -> None:
    op.drop_table("cli_audit")

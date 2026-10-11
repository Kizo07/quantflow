"""Merge QuantFlow knowledge leg with upstream 2026-10-11 chain.

Revision ID: 0040_merge_quantflow_upstream
Revises: 0029_merge_knowledge_outbox, 0039_user_disabled
Create Date: 2026-10-11

Empty merge revision: the QuantFlow branch (headed at
0029_merge_knowledge_outbox, the 2026-10-02 merge) and upstream
(0029_scheduler_agent_tasks -> ... -> 0039_user_disabled) forked at
0028_parked_attempts. No schema operations here; the two legs are
additive and independent (QuantFlow knowledge tables vs upstream
scheduler/auth/channel/projects revisions).
"""

from __future__ import annotations

from collections.abc import Sequence

revision: str = "0040_merge_quantflow_upstream"
down_revision: str | Sequence[str] | None = (
    "0029_merge_knowledge_outbox",
    "0039_user_disabled",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """No-op: join the two heads (both legs already applied below)."""


def downgrade() -> None:
    """No-op: merging down across both legs is not supported; restore from backup."""

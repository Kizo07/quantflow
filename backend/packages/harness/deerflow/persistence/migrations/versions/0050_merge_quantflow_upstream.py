"""Merge QuantFlow local leg with upstream 2026-11-01 chain.

Revision ID: 0050_merge_quantflow_upstream
Revises: 0044_scheduled_task_notify_policy, 0049_run_inputs
Create Date: 2026-11-01

Empty merge revision: the QuantFlow branch (headed at
0044_scheduled_task_notify_policy, the 0040_merge_quantflow_upstream leg
plus local 0041_cli_sessions -> 0042_cli_audit -> 0043_persistent_bot ->
0044_scheduled_task_notify_policy) and upstream
(0040_scheduled_group_notice -> ... -> 0049_run_inputs) forked at
0039_user_disabled. No schema operations here; the two legs are
additive and independent (QuantFlow CLI/persistent-bot/notify-policy
tables vs upstream group-notice/IM/channel/evidence/skill/OAuth/
extension/run-input revisions).
"""

from __future__ import annotations

from collections.abc import Sequence

revision: str = "0050_merge_quantflow_upstream"
down_revision: str | Sequence[str] | None = (
    "0044_scheduled_task_notify_policy",
    "0049_run_inputs",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """No-op: join the two heads (both legs already applied below)."""


def downgrade() -> None:
    """No-op: merging down across both legs is not supported; restore from backup."""

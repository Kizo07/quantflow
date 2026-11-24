"""Merge QuantFlow local leg with upstream 2026-11-24 chain.

Revision ID: 0061_merge_quantflow_upstream
Revises: 0050_merge_quantflow_upstream, 0060_coordination_run_settings
Create Date: 2026-11-24

Empty merge revision: the QuantFlow leg (0050_merge_quantflow_upstream,
joining the CLI/persistent-bot/notify-policy chain with the upstream
0049_run_inputs chain as of the 2026-11-01 sync) and upstream
(0049_run_producer_fencing -> ... -> 0060_coordination_run_settings)
forked at 0049_run_inputs. No schema operations here; the two legs are
additive and independent (upstream fencing/batch/approvals/project-memory/
shares/coordination/MCP-OAuth revisions vs the already-merged QuantFlow
leg below 0050).
"""

from __future__ import annotations

from collections.abc import Sequence

revision: str = "0061_merge_quantflow_upstream"
down_revision: str | Sequence[str] | None = (
    "0050_merge_quantflow_upstream",
    "0060_coordination_run_settings",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """No-op: join the two heads (both legs already applied below)."""


def downgrade() -> None:
    """No-op: merging down across both legs is not supported; restore from backup."""

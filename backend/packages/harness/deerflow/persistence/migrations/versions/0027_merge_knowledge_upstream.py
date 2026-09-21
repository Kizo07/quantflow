"""Merge QuantFlow knowledge-plane branch with upstream head.

Revision ID: 0027_merge_knowledge_upstream
Revises: 0023_knowledge_findings, 0026_mcp_task_lease_tokens
Create Date: 2026-09-21

Empty merge revision: the QuantFlow branch (0022_knowledge_phase1 ->
0023_knowledge_findings) and upstream (0022_scheduled_occurrence_seq ->
... -> 0026_mcp_task_lease_tokens) forked at 0019_thread_incarnations.
No schema operations here; the two legs are additive and independent.
"""

from __future__ import annotations

from collections.abc import Sequence

revision: str = "0027_merge_knowledge_upstream"
down_revision: str | Sequence[str] | None = (
    "0023_knowledge_findings",
    "0026_mcp_task_lease_tokens",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """No-op: join the two heads (both legs already applied below)."""


def downgrade() -> None:
    """No-op: merging down across both legs is not supported; restore from backup."""

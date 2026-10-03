"""Merge QuantFlow knowledge leg with upstream notification-outbox leg.

Revision ID: 0029_merge_knowledge_outbox
Revises: 0028_parked_attempts, 0028_knowledge_experiment_embeddings
Create Date: 2026-10-02

Empty merge revision: the QuantFlow branch (0027_merge_knowledge_upstream ->
0028_knowledge_experiment_embeddings) and upstream (0027_notification_deliveries
-> 0028_parked_attempts) forked at 0026_mcp_task_lease_tokens. No schema
operations here; the two legs are additive and independent (new outbox table
vs new experiment.embedding column).
"""

from __future__ import annotations

from collections.abc import Sequence

revision: str = "0029_merge_knowledge_outbox"
down_revision: str | Sequence[str] | None = (
    "0028_parked_attempts",
    "0028_knowledge_experiment_embeddings",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """No-op: join the two heads (both legs already applied below)."""


def downgrade() -> None:
    """No-op: merging down across both legs is not supported; restore from backup."""

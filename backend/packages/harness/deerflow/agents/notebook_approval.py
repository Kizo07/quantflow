"""Approval gating for notebook operations (Phase 3 approval wiring).

Agents drive notebooks through MCP tools while a human may watch the same
notebook live in the embedded UI. Every gated operation resolves through
the 2×2 matrix:

- agent-initiated × interactive → ``risk_confirmation`` (the caller raises
  ``ask_clarification`` with that type and waits for the human)
- agent-initiated × autonomous (or any other non-interactive mode) → BLOCKED
  (nothing waits on a human that isn't there)
- human-initiated (own click) × either mode → allowed, audit-logged

Reads are never gated. Agent edits issued while a human watches still count
as agent-initiated — observation is not authorship. Diffs surface through
the embedded UI and the existing draft/baseline flow, not through this
module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from deerflow.agents.interaction_policy import RunInteractionPolicy

logger = logging.getLogger(__name__)


class NotebookInitiator(StrEnum):
    """Who initiated the notebook operation."""

    AGENT = "agent"
    HUMAN = "human"


class NotebookOperation(StrEnum):
    """Notebook operation classes. Only EDIT and RUN are gated."""

    READ = "read"
    EDIT = "edit"
    RUN = "run"


class NotebookApprovalVerdict(StrEnum):
    """Approval outcome for one notebook operation."""

    ALLOWED = "allowed"
    RISK_CONFIRMATION = "risk_confirmation"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class NotebookApprovalDecision:
    """One matrix verdict plus the caller-facing follow-ups."""

    verdict: NotebookApprovalVerdict
    initiator: NotebookInitiator
    operation: NotebookOperation
    interactive: bool

    @property
    def clarification_type(self) -> str | None:
        """The ``ask_clarification`` type the caller must raise, if any."""
        if self.verdict is NotebookApprovalVerdict.RISK_CONFIRMATION:
            return NotebookApprovalVerdict.RISK_CONFIRMATION.value
        return None

    def blocked_result(self) -> dict[str, str]:
        """Structured BLOCKED payload for non-interactive agent callers."""
        return {
            "status": "BLOCKED",
            "operation": self.operation.value,
            "missing_decision": (f"Human approval for agent-initiated notebook {self.operation.value}; no synchronous human is available in this run."),
        }


def decide_notebook_approval(
    *,
    initiator: NotebookInitiator,
    operation: NotebookOperation,
    policy: RunInteractionPolicy,
    notebook_id: str = "",
    user_id: str = "",
) -> NotebookApprovalDecision:
    """Resolve the approval matrix and audit-log the verdict.

    ``initiator`` must be ``AGENT`` for anything an agent run issues —
    including edits made while a human watches the embedded UI.
    """
    if operation is NotebookOperation.READ or initiator is NotebookInitiator.HUMAN:
        verdict = NotebookApprovalVerdict.ALLOWED
    elif policy.allows_clarification:
        verdict = NotebookApprovalVerdict.RISK_CONFIRMATION
    else:
        verdict = NotebookApprovalVerdict.BLOCKED
    logger.info(
        "Notebook approval: verdict=%s initiator=%s operation=%s interactive=%s notebook=%s user=%s",
        verdict.value,
        initiator.value,
        operation.value,
        policy.allows_clarification,
        notebook_id or "-",
        user_id or "-",
    )
    return NotebookApprovalDecision(
        verdict=verdict,
        initiator=initiator,
        operation=operation,
        interactive=policy.allows_clarification,
    )

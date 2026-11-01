"""Notebook approval-matrix tests (Phase 3 approval wiring).

Pins the 2×2 gating matrix: agent-initiated × interactive →
``risk_confirmation``; agent-initiated × non-interactive → BLOCKED;
human-initiated (own click) × either mode → allowed, audit-logged.
Reads are never gated.
"""

from __future__ import annotations

import pytest

from deerflow.agents.interaction_policy import RunInteractionPolicy, resolve_run_interaction_policy
from deerflow.agents.notebook_approval import (
    NotebookApprovalVerdict,
    NotebookInitiator,
    NotebookOperation,
    decide_notebook_approval,
)


@pytest.mark.parametrize("operation", [NotebookOperation.EDIT, NotebookOperation.RUN])
def test_agent_initiated_gated_operation_requires_risk_confirmation_when_interactive(operation):
    decision = decide_notebook_approval(
        initiator=NotebookInitiator.AGENT,
        operation=operation,
        policy=RunInteractionPolicy.interactive(),
    )
    assert decision.verdict is NotebookApprovalVerdict.RISK_CONFIRMATION
    assert decision.clarification_type == "risk_confirmation"


@pytest.mark.parametrize("mode", ["autonomous", "webhook", "scheduled"])
@pytest.mark.parametrize("operation", [NotebookOperation.EDIT, NotebookOperation.RUN])
def test_agent_initiated_gated_operation_is_blocked_without_a_human(mode, operation):
    policy = resolve_run_interaction_policy({"configurable": {"interaction_mode": mode}})
    decision = decide_notebook_approval(
        initiator=NotebookInitiator.AGENT,
        operation=operation,
        policy=policy,
    )
    assert decision.verdict is NotebookApprovalVerdict.BLOCKED
    assert decision.clarification_type is None
    blocked = decision.blocked_result()
    assert blocked["status"] == "BLOCKED"
    assert blocked["operation"] == operation.value
    assert "missing_decision" in blocked and blocked["missing_decision"]


@pytest.mark.parametrize("mode", ["interactive", "autonomous"])
@pytest.mark.parametrize("operation", [NotebookOperation.EDIT, NotebookOperation.RUN])
def test_human_initiated_operation_is_allowed_in_either_mode(mode, operation, caplog):
    policy = resolve_run_interaction_policy({"configurable": {"interaction_mode": mode}})
    with caplog.at_level("INFO", logger="deerflow.agents.notebook_approval"):
        decision = decide_notebook_approval(
            initiator=NotebookInitiator.HUMAN,
            operation=operation,
            policy=policy,
            notebook_id="nb-1",
            user_id="user-1",
        )
    assert decision.verdict is NotebookApprovalVerdict.ALLOWED
    assert decision.clarification_type is None
    assert any("verdict=allowed" in record.message and "nb-1" in record.message for record in caplog.records)


@pytest.mark.parametrize("initiator", [NotebookInitiator.AGENT, NotebookInitiator.HUMAN])
@pytest.mark.parametrize("mode", ["interactive", "autonomous"])
def test_reads_are_never_gated(initiator, mode):
    policy = resolve_run_interaction_policy({"configurable": {"interaction_mode": mode}})
    decision = decide_notebook_approval(initiator=initiator, operation=NotebookOperation.READ, policy=policy)
    assert decision.verdict is NotebookApprovalVerdict.ALLOWED


def test_legacy_non_interactive_flag_blocks_agent_edits():
    """The scheduler's legacy flag resolves to a non-interactive policy."""
    policy = resolve_run_interaction_policy({"configurable": {"non_interactive": True}})
    decision = decide_notebook_approval(
        initiator=NotebookInitiator.AGENT,
        operation=NotebookOperation.EDIT,
        policy=policy,
    )
    assert decision.verdict is NotebookApprovalVerdict.BLOCKED


def test_blocked_decisions_are_audit_logged(caplog):
    policy = resolve_run_interaction_policy({"configurable": {"interaction_mode": "autonomous"}})
    with caplog.at_level("INFO", logger="deerflow.agents.notebook_approval"):
        decide_notebook_approval(
            initiator=NotebookInitiator.AGENT,
            operation=NotebookOperation.RUN,
            policy=policy,
            notebook_id="nb-9",
        )
    assert any("verdict=blocked" in record.message and "nb-9" in record.message for record in caplog.records)

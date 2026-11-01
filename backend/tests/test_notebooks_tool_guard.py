"""Notebook tool-guard integration tests (joint contract D3).

Exercises the REAL tool path — :func:`guarded_notebook_tool_call` with a
recording transport — not just the matrix decision: LLM-supplied approval
keys are stripped, autonomous agent calls BLOCK without touching the tool,
interactive agent calls return RISK_CONFIRMATION until the human-confirmed
retry, and human calls pass through with harness-set keys.
"""

from __future__ import annotations

from typing import Any

import pytest

from deerflow.agents.interaction_policy import RunInteractionPolicy, resolve_run_interaction_policy
from deerflow.agents.notebook_approval import NotebookInitiator, NotebookOperation
from deerflow.agents.notebook_tool_guard import (
    LLM_STRIPPED_NOTEBOOK_ARGS,
    RISK_CONFIRMATION_STATUS,
    guarded_notebook_tool_call,
    strip_llm_notebook_args,
)


class _RecordingTransport:
    """Fake ReportForge MCP transport: records forwarded calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(args))
        return {"status": "ok", "echo": dict(args)}


def test_llm_approval_keys_are_stripped_without_mutation():
    assert LLM_STRIPPED_NOTEBOOK_ARGS == frozenset({"confirmed", "interaction_mode", "mode"})
    args = {"cell_id": "c1", "source": "1+1", "confirmed": True, "interaction_mode": "autonomous", "mode": "x"}
    stripped = strip_llm_notebook_args(args)
    assert stripped == {"cell_id": "c1", "source": "1+1"}
    assert args["confirmed"] is True  # input untouched


@pytest.mark.asyncio()
@pytest.mark.parametrize("mode", ["autonomous", "webhook", "scheduled"])
@pytest.mark.parametrize("operation", [NotebookOperation.EDIT, NotebookOperation.RUN])
async def test_autonomous_agent_call_is_blocked_and_never_invokes_tool(mode, operation):
    """Agent × non-interactive → BLOCKED payload; the tool never runs — even
    when the model smuggles confirmed=True into its args."""
    transport = _RecordingTransport()
    policy = resolve_run_interaction_policy({"configurable": {"interaction_mode": mode}})
    result = await guarded_notebook_tool_call(
        tool_name="reportforge_notebook_run_cell",
        args={"cell_id": "c1", "confirmed": True, "interaction_mode": "interactive"},
        operation=operation,
        initiator=NotebookInitiator.AGENT,
        policy=policy,
        call=transport,
        notebook_id="nb-1",
        user_id="user-1",
    )
    assert result["status"] == "BLOCKED"
    assert result["operation"] == operation.value
    assert result["missing_decision"]
    assert transport.calls == []


@pytest.mark.asyncio()
async def test_blocked_calls_ignore_human_confirmed_and_never_retry():
    """human_confirmed only unlocks the interactive retry — a non-interactive
    run has no human to confirm, so BLOCKED stands and the tool never runs."""
    transport = _RecordingTransport()
    policy = resolve_run_interaction_policy({"configurable": {"interaction_mode": "autonomous"}})
    result = await guarded_notebook_tool_call(
        tool_name="reportforge_notebook_run_cell",
        args={"cell_id": "c1"},
        operation=NotebookOperation.RUN,
        initiator=NotebookInitiator.AGENT,
        policy=policy,
        call=transport,
        human_confirmed=True,
    )
    assert result["status"] == "BLOCKED"
    assert transport.calls == []


@pytest.mark.asyncio()
@pytest.mark.parametrize("operation", [NotebookOperation.EDIT, NotebookOperation.RUN])
async def test_interactive_agent_call_returns_risk_confirmation_first(operation):
    """Agent × interactive without confirmation → RISK_CONFIRMATION directive."""
    transport = _RecordingTransport()
    result = await guarded_notebook_tool_call(
        tool_name="reportforge_notebook_run_cell",
        args={"cell_id": "c1"},
        operation=operation,
        initiator=NotebookInitiator.AGENT,
        policy=RunInteractionPolicy.interactive(),
        call=transport,
        notebook_id="nb-1",
    )
    assert result["status"] == RISK_CONFIRMATION_STATUS == "RISK_CONFIRMATION"
    assert result["clarification_type"] == "risk_confirmation"
    assert result["operation"] == operation.value
    assert "ask_clarification" in result["retry"]
    assert "human_confirmed=True" in result["retry"]
    assert transport.calls == []


@pytest.mark.asyncio()
async def test_interactive_agent_confirmed_retry_forwards_with_harness_keys():
    """The post-approval retry forwards with confirmed=True + resolved mode,
    overriding whatever the model supplied."""
    transport = _RecordingTransport()
    result = await guarded_notebook_tool_call(
        tool_name="reportforge_notebook_run_cell",
        args={"cell_id": "c1", "confirmed": False, "interaction_mode": "autonomous", "mode": "autonomous"},
        operation=NotebookOperation.RUN,
        initiator=NotebookInitiator.AGENT,
        policy=RunInteractionPolicy.interactive(),
        call=transport,
        notebook_id="nb-1",
        human_confirmed=True,
    )
    assert result["status"] == "ok"
    assert transport.calls == [
        {"cell_id": "c1", "confirmed": True, "interaction_mode": "interactive"},
    ]


@pytest.mark.asyncio()
@pytest.mark.parametrize("mode", ["interactive", "autonomous"])
async def test_human_call_is_allowed_in_either_mode_with_harness_keys(mode):
    transport = _RecordingTransport()
    policy = resolve_run_interaction_policy({"configurable": {"interaction_mode": mode}})
    result = await guarded_notebook_tool_call(
        tool_name="reportforge_notebook_run_cell",
        args={"cell_id": "c1", "confirmed": False},
        operation=NotebookOperation.RUN,
        initiator=NotebookInitiator.HUMAN,
        policy=policy,
        call=transport,
    )
    assert result["status"] == "ok"
    assert transport.calls == [{"cell_id": "c1", "confirmed": True, "interaction_mode": mode}]


@pytest.mark.asyncio()
@pytest.mark.parametrize("initiator", [NotebookInitiator.AGENT, NotebookInitiator.HUMAN])
async def test_reads_are_never_gated_through_the_tool_path(initiator):
    transport = _RecordingTransport()
    policy = resolve_run_interaction_policy({"configurable": {"interaction_mode": "autonomous"}})
    result = await guarded_notebook_tool_call(
        tool_name="reportforge_notebook_read",
        args={"cell_id": "c1"},
        operation=NotebookOperation.READ,
        initiator=initiator,
        policy=policy,
        call=transport,
    )
    assert result["status"] == "ok"
    assert len(transport.calls) == 1

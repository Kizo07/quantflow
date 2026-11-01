"""Harness-side guard for ReportForge notebook MCP tool calls (joint contract D3).

Agents drive notebooks through ReportForge MCP tools while a human may watch
the same notebook live in the embedded UI. Every gated (edit/run) call passes
through :func:`guarded_notebook_tool_call`, which enforces the approval matrix
from :mod:`deerflow.agents.notebook_approval`:

- agent-initiated × interactive → ``RISK_CONFIRMATION`` directive: the agent
  must raise ``ask_clarification`` with ``clarification_type="risk_confirmation"``
  and, only after the human approves, retry with ``human_confirmed=True`` —
  the retry is then forwarded with ``confirmed=True``.
- agent-initiated × autonomous (or any other non-interactive mode) → BLOCKED
  payload; the tool is never invoked and must never be retried.
- human-initiated (own click) × either mode → forwarded, audit-logged.

Reads are never gated.

The model MUST NOT pass ``confirmed`` / ``interaction_mode`` / ``mode`` itself:
the guard STRIPS any LLM-supplied values and sets them per run policy, so a
model can never self-approve. ReportForge adds the ``interaction_mode`` param
to its gated tools; the guard always forwards the harness-resolved mode.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from deerflow.agents.interaction_policy import RunInteractionPolicy
from deerflow.agents.notebook_approval import (
    NotebookApprovalVerdict,
    NotebookInitiator,
    NotebookOperation,
    decide_notebook_approval,
)

logger = logging.getLogger(__name__)

#: Tool-arg keys the model must never set: stripped from every call and set
#: per run policy instead. ``mode`` is the legacy alias some prompts use for
#: ``interaction_mode``; both spellings are stripped.
LLM_STRIPPED_NOTEBOOK_ARGS: frozenset[str] = frozenset({"confirmed", "interaction_mode", "mode"})

#: Status of the directive returned for agent×interactive gated calls issued
#: without a prior human confirmation.
RISK_CONFIRMATION_STATUS = "RISK_CONFIRMATION"


def strip_llm_notebook_args(args: Mapping[str, Any]) -> dict[str, Any]:
    """Return *args* without the LLM-supplied approval keys (never mutates)."""
    return {key: value for key, value in dict(args).items() if key not in LLM_STRIPPED_NOTEBOOK_ARGS}


def _forward_args(args: Mapping[str, Any], policy: RunInteractionPolicy) -> dict[str, Any]:
    """Approval keys as the harness sets them: confirmed + resolved mode."""
    forwarded = strip_llm_notebook_args(args)
    forwarded["confirmed"] = True
    forwarded["interaction_mode"] = policy.mode.value
    return forwarded


async def guarded_notebook_tool_call(
    *,
    tool_name: str,
    args: Mapping[str, Any],
    operation: NotebookOperation,
    initiator: NotebookInitiator,
    policy: RunInteractionPolicy,
    call: Callable[[dict[str, Any]], Awaitable[Any]],
    notebook_id: str = "",
    user_id: str = "",
    human_confirmed: bool = False,
) -> Any:
    """Run one notebook MCP tool call through the approval matrix.

    Args:
        tool_name: ReportForge MCP tool name (audit label only).
        args: Model-supplied tool args; approval keys are stripped.
        operation: ``READ`` (never gated) or ``EDIT``/``RUN`` (gated).
        initiator: ``AGENT`` for anything an agent run issues — including
            edits made while a human watches the embedded UI — or ``HUMAN``
            for the human's own clicks.
        policy: Run interaction policy (``human_confirmed`` aside, the only
            authority for the confirmation decision).
        call: Transport invoking the tool with harness-set args.
        human_confirmed: True only on the retry after the human approved the
            ``risk_confirmation`` clarification. Never set from model output.

    Returns:
        The tool result for allowed/confirmed calls; a ``BLOCKED`` payload
        for non-interactive agent calls (never retried); a
        ``RISK_CONFIRMATION`` directive for interactive agent calls awaiting
        human approval (retry with ``human_confirmed=True`` after approval).
    """
    decision = decide_notebook_approval(
        initiator=initiator,
        operation=operation,
        policy=policy,
        notebook_id=notebook_id,
        user_id=user_id,
    )
    if decision.verdict is NotebookApprovalVerdict.ALLOWED:
        return await call(_forward_args(args, policy))
    if decision.verdict is NotebookApprovalVerdict.BLOCKED:
        logger.info(
            "Notebook tool guard: blocked tool=%s operation=%s notebook=%s user=%s (no retry)",
            tool_name,
            operation.value,
            notebook_id or "-",
            user_id or "-",
        )
        return decision.blocked_result()
    if human_confirmed:
        logger.info(
            "Notebook tool guard: confirmed retry tool=%s operation=%s notebook=%s user=%s",
            tool_name,
            operation.value,
            notebook_id or "-",
            user_id or "-",
        )
        return await call(_forward_args(args, policy))
    return {
        "status": RISK_CONFIRMATION_STATUS,
        "clarification_type": NotebookApprovalVerdict.RISK_CONFIRMATION.value,
        "tool": tool_name,
        "operation": operation.value,
        "question": (f"Approve agent-initiated notebook {operation.value}" + (f" on notebook {notebook_id}" if notebook_id else "") + "?"),
        "retry": ("Raise ask_clarification with clarification_type='risk_confirmation' and, only after the human approves, retry this call with human_confirmed=True."),
    }

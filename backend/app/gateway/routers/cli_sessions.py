"""Dashboard API for headless CLI sessions and their launch presets."""

from __future__ import annotations

import asyncio
import json
import shlex
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.gateway import cli_runner
from app.gateway.authz import require_permission
from app.gateway.cli_capabilities import CliCapabilitiesUnavailable, get_cli_capabilities
from app.gateway.deps import get_cli_audit_repo, get_cli_preset_repo, get_cli_session_repo, get_current_user
from app.gateway.sse_headers import sse_response_headers
from deerflow.config.cli_config import CliConfig

router = APIRouter(prefix="/api/cli-sessions", tags=["cli-sessions"])
presets_router = APIRouter(prefix="/api/cli-presets", tags=["cli-presets"])
capabilities_router = APIRouter(prefix="/api/cli-capabilities", tags=["cli-capabilities"])

_MAX_PUBLIC_ERROR_CHARS = 500

# Overridable run-log directory (tests point it at a tmp dir). Production
# resolves `{base_dir}/cli_runs` per request so base-dir overrides apply.
_LOG_DIR_OVERRIDE: Path | None = None


class CliLaunchRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str | None = None
    provider: str | None = None
    preset: str | None = None
    effort: str | None = None
    contributor_mode: bool | None = None
    approval_mode: str | None = None
    sandbox_network: str | None = None
    trust_workspace: bool = False
    disable_approval: bool = False
    disable_sandbox: bool = False
    yolo: bool = False
    workspace: str | None = None
    mode: Literal["research", "coding"] = "coding"


class CliResumeRequest(BaseModel):
    prompt: str | None = None


class CliChatSendRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str | None = None
    provider: str | None = None
    effort: str | None = None
    yolo: bool | None = None


class CliExportRequest(BaseModel):
    redacted: bool = True


class CliPresetPut(BaseModel):
    model: str | None = None
    provider: str | None = None
    preset: str | None = None
    effort: str | None = None
    contributor_mode: bool | None = None
    yolo: bool = False
    approval_mode: str | None = None


def _cli_config() -> CliConfig:
    from app.gateway.deps import get_config

    try:
        config = get_config()
    except (FileNotFoundError, RuntimeError):
        return CliConfig()
    cli = getattr(config, "cli", None)
    return cli if isinstance(cli, CliConfig) else CliConfig()


def _runs_dir() -> Path:
    if _LOG_DIR_OVERRIDE is not None:
        return _LOG_DIR_OVERRIDE
    from deerflow.config.paths import get_paths

    return get_paths().base_dir / "cli_runs"


async def _current_user_id(request: Request) -> str:
    user_id = await get_current_user(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id


def _short_error(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)[:_MAX_PUBLIC_ERROR_CHARS]


def _item(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "session_ref": record.get("session_ref"),
        "resumed_from": record.get("resumed_from"),
        "status": record["status"],
        "mode": record.get("mode", "coding"),
        "model": record.get("model"),
        "provider": record.get("provider"),
        "preset": record.get("preset"),
        "effort": record.get("effort"),
        "contributor_mode": record.get("contributor_mode"),
        "approval_mode": record.get("approval_mode"),
        "yolo": bool(record.get("yolo")),
        "workspace": record.get("workspace"),
        "pid": record.get("pid"),
        "tmux_session": record.get("tmux_session"),
        "error": _short_error(record.get("error")),
        "started_at": record.get("started_at"),
        "ended_at": record.get("ended_at"),
        "created_at": record.get("created_at"),
    }


def _preset_item(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": record["name"],
        "model": record.get("model"),
        "provider": record.get("provider"),
        "preset": record.get("preset"),
        "effort": record.get("effort"),
        "contributor_mode": record.get("contributor_mode"),
        "yolo": bool(record.get("yolo")),
        "approval_mode": record.get("approval_mode"),
        "updated_at": record.get("updated_at"),
    }


def _audit_item(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "actor": record["actor"],
        "action": record["action"],
        "session_id": record.get("session_id"),
        "command": record.get("command"),
        "preset": record.get("preset"),
        "preset_version": record.get("preset_version"),
        "result": record["result"],
        "created_at": record.get("created_at"),
    }


async def _record_audit(
    request: Request,
    *,
    actor: str,
    action: str,
    result: str,
    session_id: str | None = None,
    command: str | None = None,
    preset: str | None = None,
    preset_version: str | None = None,
) -> None:
    """Append one audit row for a launch or control action."""
    await get_cli_audit_repo(request).record(
        actor=actor,
        action=action,
        result=result,
        session_id=session_id,
        command=None if command is None else cli_runner.redact_command_string(command),
        preset=preset,
        preset_version=preset_version,
    )


# Chat layer: payload_type-driven mapping from exec JSONL lines to typed chat
# frames, per the evidenced exec vocabulary (`cli_runs/*.jsonl` mirrors):
# assistant text comes ONLY from `run.output.delta` records — the
# `run.terminal.completed` record repeats the full concatenated text and
# must never surface as chat text (that echo duplicated assistant content).
# Tool/progress rows come from `task.lifecycle.*` records. Anything
# unrecognized yields no typed frame here — the raw line still flows
# through as a compat `data:` frame and is never dropped.
_CLI_CHAT_DELTA_TYPES = frozenset({"run.output.delta"})
_CLI_CHAT_LIFECYCLE_PREFIX = "task.lifecycle."
# Pre-envelope flat shapes (no `payload_type` key) keep their legacy mapping
# so older producers still render.
_CLI_CHAT_MESSAGE_TYPES = frozenset({"message", "text", "delta", "assistant", "output", "content", "chunk"})
_CLI_CHAT_TOOL_TYPES = frozenset({"tool", "tool_call", "tool_result", "tool_use", "progress"})
_CLI_CHAT_TEXT_KEYS = ("delta", "text", "content", "output", "message")

# Backfill walks at most this many `resumed_from` links (newest first);
# longer chats report `truncated: true`.
_CLI_CHAT_CHAIN_CAP = 50


def _lifecycle_tool_frame(payload_type: str, payload: dict[str, Any], *, run_id: str) -> dict[str, Any] | None:
    """Map one `task.lifecycle.*` envelope payload to a tool/progress row.

    Only events carrying displayable content become chat rows: bare state
    transitions (proposed/accepted/scheduled/started/completed) name an
    opaque task id and would render as dozens of empty rows per run, so
    they yield no frame. The raw line still flows through as a compat
    `data:` frame for the run inspector.
    """
    event = payload.get("event")
    event = event if isinstance(event, dict) else {}
    kind = str(event.get("kind") or payload_type.rsplit(".", 1)[-1] or "")
    task_id = str(payload.get("task_id") or event.get("task_id") or "")
    if kind in {"proposed", "accepted", "scheduled", "started", "completed"}:
        return None
    if kind == "side_effect_intent":
        operation = str(event.get("operation") or "")
        if not operation:
            return None
        if operation == "reminder" or operation.startswith("reminder."):
            return None  # internal reminder machinery, never a user-facing tool
        policy = str(event.get("policy_decision") or "")
        detail = "" if policy == "not_applicable" else policy
        return {"run_id": run_id, "name": operation, "status": "intent", "detail": detail}
    if kind == "status":
        message = str(event.get("message") or "")
        if not message:
            return None
        details = event.get("details")
        details = details if isinstance(details, dict) else {}
        facets = details.get("facets")
        facets = facets if isinstance(facets, list) else []
        first = facets[0] if facets and isinstance(facets[0], dict) else {}
        name = str(first.get("operation") or task_id or "")
        if not name:
            return None
        return {"run_id": run_id, "name": name, "status": str(details.get("phase") or "status"), "detail": message}
    if kind == "cancelled":
        reason = str(event.get("reason") or "")
        if not reason or not task_id:
            return None
        return {"run_id": run_id, "name": task_id, "status": "cancelled", "detail": reason}
    detail = str(event.get("message") or event.get("reason") or "")
    if not detail or not task_id:
        return None
    return {"run_id": run_id, "name": task_id, "status": kind or "update", "detail": detail}


def _legacy_chat_frames(obj: dict[str, Any], *, run_id: str) -> list[tuple[str, dict[str, Any]]]:
    """Map a pre-envelope flat line (no `payload_type`) to typed chat frames."""
    kind = str(obj.get("type", "") or "").lower()
    is_tool = kind in _CLI_CHAT_TOOL_TYPES or kind.startswith("tool")
    frames: list[tuple[str, dict[str, Any]]] = []
    tool_obj = obj.get("tool")
    tool_obj = tool_obj if isinstance(tool_obj, dict) else {}
    name = obj.get("name", tool_obj.get("name"))
    status = obj.get("status", tool_obj.get("status"))
    if is_tool or (name is not None and status is not None):
        detail = obj.get("detail", tool_obj.get("detail", obj.get("output", "")))
        frames.append(("tool", {"run_id": run_id, "name": name or "", "status": status or "", "detail": detail or ""}))
        return frames
    delta: str | None = None
    if kind in _CLI_CHAT_MESSAGE_TYPES or kind == "":
        for key in _CLI_CHAT_TEXT_KEYS:
            value = obj.get(key)
            if isinstance(value, str) and value:
                delta = value
                break
    if delta:
        frames.append(("message", {"role": "assistant", "delta": delta, "run_id": run_id}))
    return frames


def _typed_chat_frames(line: str, *, run_id: str) -> list[tuple[str, dict[str, Any]]]:
    """Map one exec JSONL line to typed chat frames (shared by SSE + backfill).

    Envelope lines dispatch on `payload_type`: only `run.output.delta`
    yields assistant text, only `task.lifecycle.*` yields tool rows. The
    terminal record, user-input echo, model/linkage records, and non-JSON
    preamble (`muse:`/`tbh:`) yield no typed frame by construction.
    """
    try:
        obj = json.loads(line)
    except ValueError:
        return []
    if not isinstance(obj, dict):
        return []
    if "payload_type" not in obj:
        return _legacy_chat_frames(obj, run_id=run_id)
    payload_type = str(obj.get("payload_type") or "")
    payload = obj.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    if payload_type in _CLI_CHAT_DELTA_TYPES:
        text = payload.get("text")
        if isinstance(text, str) and text:
            return [("message", {"role": "assistant", "delta": text, "run_id": run_id})]
        return []
    if payload_type.startswith(_CLI_CHAT_LIFECYCLE_PREFIX):
        frame = _lifecycle_tool_frame(payload_type, payload, run_id=run_id)
        return [("tool", frame)] if frame is not None else []
    return []


def _run_stats_for_log(log_path: str | Path | None, *, run_id: str) -> dict[str, Any]:
    """Summarize one run log for the UI summary line: model, terminal state, elapsed, exit.

    `recorded_at` is microseconds; `elapsed_ms` spans first-to-last envelope.
    Missing carriers stay None rather than guessing.
    """
    stats: dict[str, Any] = {
        "run_id": run_id,
        "model": None,
        "provider": None,
        "display_label": None,
        "terminal": None,
        "reason": None,
        "elapsed_ms": None,
        "exit_code": cli_runner.sentinel_exit_code(log_path),
    }
    if not log_path:
        return stats
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return stats
    first_ts: float | None = None
    last_ts: float | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue  # non-JSON preamble carries no stats
        if not isinstance(obj, dict):
            continue
        ts = obj.get("recorded_at")
        if isinstance(ts, (int, float)) and not isinstance(ts, bool):
            first_ts = ts if first_ts is None else min(first_ts, ts)
            last_ts = ts if last_ts is None else max(last_ts, ts)
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        if obj.get("payload_type") == "run.model.configured":
            stats["model"] = payload.get("model_id") or stats["model"]
            stats["provider"] = payload.get("provider_id") or stats["provider"]
            stats["display_label"] = payload.get("display_label") or stats["display_label"]
        elif obj.get("payload_type") == "run.terminal.completed":
            stats["terminal"] = payload.get("terminal")
            stats["reason"] = payload.get("reason")
    if first_ts is not None and last_ts is not None:
        stats["elapsed_ms"] = max(0.0, (last_ts - first_ts) / 1000.0)
    return stats


def _chat_frames_for_line(line: str, *, run_id: str | None) -> list[str]:
    """Raw compat frame plus any typed chat frames for one log line."""
    out = [f"data: {line}\n\n"]
    if run_id is not None:
        for event, payload in _typed_chat_frames(line, run_id=run_id):
            out.append(f"event: {event}\ndata: {json.dumps(payload)}\n\n")
    return out


def _replay_run_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Replay one run's stored prompt + JSONL log into chat messages."""
    messages: list[dict[str, Any]] = []
    run_id = record["id"]
    if record.get("prompt"):
        messages.append({"run_id": run_id, "role": "user", "content": record["prompt"]})
    lines: list[str] = []
    if record.get("log_path"):
        try:
            lines = Path(record["log_path"]).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
    pending: list[str] = []

    def _flush() -> None:
        if pending:
            messages.append({"run_id": run_id, "role": "assistant", "content": "".join(pending)})
            pending.clear()

    for line in lines:
        if not line.strip():
            continue
        for event, payload in _typed_chat_frames(line, run_id=run_id):
            if event == "message":
                pending.append(payload["delta"])
            elif event == "tool":
                _flush()
                messages.append(
                    {
                        "run_id": run_id,
                        "role": "tool",
                        "name": payload["name"],
                        "status": payload["status"],
                        "detail": payload["detail"],
                    }
                )
    _flush()
    return messages


async def _chat_chain(session_repo, leaf_id: str, *, user_id: str) -> tuple[list[dict[str, Any]], bool]:
    """Walk `resumed_from` from the leaf to the root; oldest run first."""
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current: str | None = leaf_id
    truncated = False
    while current is not None and current not in seen:
        if len(chain) >= _CLI_CHAT_CHAIN_CAP:
            truncated = True
            break
        record = await session_repo.get(current, user_id=user_id)
        if record is None:
            break
        chain.append(record)
        seen.add(current)
        current = record.get("resumed_from")
    chain.reverse()
    return chain, truncated


def _preset_version_label(preset: dict[str, Any] | None) -> str:
    """Version stamp for a preset row: stored updated_at, else built-in."""
    updated = (preset or {}).get("updated_at")
    return str(updated) if updated is not None else "built-in"


def _params_from_launch(body: CliLaunchRequest) -> cli_runner.CliLaunchParams:
    return cli_runner.CliLaunchParams(
        prompt=body.prompt,
        model=body.model,
        provider=body.provider,
        preset=body.preset,
        effort=body.effort,
        contributor_mode=body.contributor_mode,
        approval_mode=body.approval_mode,
        sandbox_network=body.sandbox_network,
        trust_workspace=body.trust_workspace,
        disable_approval=body.disable_approval,
        disable_sandbox=body.disable_sandbox,
        yolo=body.yolo,
        workspace=body.workspace,
    )


async def _start_run(
    *,
    request: Request,
    user_id: str,
    params: cli_runner.CliLaunchParams,
    mode: str,
    action: str,
    resumed_from: str | None = None,
) -> dict[str, Any]:
    """Validate, persist, spawn, and audit one CLI run; spawn failures mark the row failed."""
    session_repo = get_cli_session_repo(request)
    preset_repo = get_cli_preset_repo(request)
    preset_name: str | None = None
    preset_version: str | None = None
    if params.model == cli_runner.CLI_MODEL_MUSE_13:
        stored = await cli_runner.load_preset_or_default(preset_repo, cli_runner.CLI_PRESET_MUSE_13)
        params = cli_runner.apply_preset_defaults(params, stored)
        preset_name = (stored or {}).get("name") or cli_runner.CLI_PRESET_MUSE_13
        preset_version = _preset_version_label(stored)
    try:
        params = cli_runner.validate_launch_params(params)
    except cli_runner.CliValidationError as exc:
        await _record_audit(
            request,
            actor=user_id,
            action=action,
            result="rejected",
            preset=preset_name,
            preset_version=preset_version,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    config = _cli_config()
    row_id = uuid.uuid4().hex
    session_uuid = str(uuid.uuid4())
    params = cli_runner.CliLaunchParams(**{**params.__dict__, "session_id": session_uuid})
    command = cli_runner.build_command_string(params, binary=config.binary)
    log_path = _runs_dir() / f"{row_id}.jsonl"
    record = await session_repo.create(
        created_by=user_id,
        mode=mode,
        session_ref=session_uuid,
        resumed_from=resumed_from,
        model=params.model,
        provider=params.provider,
        preset=params.preset,
        effort=params.effort,
        contributor_mode=params.contributor_mode,
        approval_mode=params.approval_mode,
        sandbox_network=params.sandbox_network,
        trust_workspace=params.trust_workspace,
        disable_approval=params.disable_approval,
        disable_sandbox=params.disable_sandbox,
        yolo=params.yolo,
        workspace=params.workspace,
        prompt=params.prompt,
        log_path=str(log_path),
        session_id=row_id,
    )
    try:
        spawn = await cli_runner.spawn_run(
            row_id=row_id,
            command=command,
            log_path=log_path,
            tmux_mode=config.tmux,
            workspace=params.workspace,
            timeout=config.probe_timeout_seconds,
        )
    except (cli_runner.CliSpawnError, OSError) as exc:
        from datetime import UTC, datetime

        await session_repo.update(row_id, user_id=user_id, status="failed", error=str(exc)[:2000], ended_at=datetime.now(UTC))
        await _record_audit(
            request,
            actor=user_id,
            action=action,
            result="failed",
            session_id=row_id,
            command=command,
            preset=preset_name,
            preset_version=preset_version,
        )
        raise HTTPException(status_code=500, detail=f"CLI run failed to start: {exc}") from exc
    updated = await session_repo.update(row_id, user_id=user_id, tmux_session=spawn.tmux_session, pid=spawn.pid)
    await _record_audit(
        request,
        actor=user_id,
        action=action,
        result="started",
        session_id=row_id,
        command=command,
        preset=preset_name,
        preset_version=preset_version,
    )
    return updated or record


@router.get("")
@require_permission("runs", "read")
async def list_cli_sessions(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    status: str | None = None,
    mode: str | None = None,
) -> list[dict[str, Any]]:
    """List the caller's CLI sessions, newest first."""
    user_id = await _current_user_id(request)
    session_repo = get_cli_session_repo(request)
    records = await session_repo.list_by_owner(user_id=user_id, limit=limit, status=status, mode=mode)
    records = await _refresh_finished_rows(session_repo, records, user_id=user_id)
    return [_item(record) for record in records]


@router.post("", status_code=201)
@require_permission("runs", "create")
async def launch_cli_session(body: CliLaunchRequest, request: Request) -> dict[str, Any]:
    """Validate, persist, and spawn one headless CLI run."""
    user_id = await _current_user_id(request)
    record = await _start_run(request=request, user_id=user_id, params=_params_from_launch(body), mode=body.mode, action="launch")
    return _item(record)


@router.get("/{session_id}")
@require_permission("runs", "read")
async def get_cli_session(session_id: str, request: Request) -> dict[str, Any]:
    """Return one CLI session owned by the caller."""
    user_id = await _current_user_id(request)
    session_repo = get_cli_session_repo(request)
    record = await session_repo.get(session_id, user_id=user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="CLI session not found")
    (record,) = await _refresh_finished_rows(session_repo, [record], user_id=user_id)
    return _item(record)


@router.post("/{session_id}/resume", status_code=201)
@require_permission("runs", "create")
async def resume_cli_session(session_id: str, request: Request, body: CliResumeRequest | None = None) -> dict[str, Any]:
    """Headless resume: a new ``muse exec`` run inheriting the prior row's flags, linked via ``resumed_from``."""
    user_id = await _current_user_id(request)
    prior = await get_cli_session_repo(request).get(session_id, user_id=user_id)
    if prior is None:
        raise HTTPException(status_code=404, detail="CLI session not found")
    requested_prompt = body.prompt if body is not None else None
    prompt = requested_prompt if requested_prompt is not None else prior.get("prompt")
    if not prompt:
        raise HTTPException(status_code=422, detail="No prompt to resume with: the prior run stored none and none was supplied")
    params = cli_runner.CliLaunchParams(
        prompt=prompt,
        model=prior.get("model"),
        provider=prior.get("provider"),
        preset=prior.get("preset"),
        effort=prior.get("effort"),
        contributor_mode=prior.get("contributor_mode"),
        approval_mode=prior.get("approval_mode"),
        sandbox_network=prior.get("sandbox_network"),
        trust_workspace=bool(prior.get("trust_workspace")),
        disable_approval=bool(prior.get("disable_approval")),
        disable_sandbox=bool(prior.get("disable_sandbox")),
        yolo=bool(prior.get("yolo")),
        workspace=prior.get("workspace"),
    )
    record = await _start_run(request=request, user_id=user_id, params=params, mode=prior.get("mode", "coding"), action="resume", resumed_from=session_id)
    return _item(record)


@router.get("/{session_id}/messages")
@require_permission("runs", "read")
async def list_cli_chat_messages(session_id: str, request: Request) -> dict[str, Any]:
    """Backfill a chat: replay the `resumed_from` chain's stored logs into messages, oldest first."""
    user_id = await _current_user_id(request)
    session_repo = get_cli_session_repo(request)
    leaf = await session_repo.get(session_id, user_id=user_id)
    if leaf is None:
        raise HTTPException(status_code=404, detail="CLI session not found")
    chain, truncated = await _chat_chain(session_repo, session_id, user_id=user_id)
    messages: list[dict[str, Any]] = []
    stats: dict[str, dict[str, Any]] = {}
    for record in chain:
        messages.extend(_replay_run_messages(record))
        run_stats = _run_stats_for_log(record.get("log_path"), run_id=record["id"])
        if run_stats["model"] is None:
            run_stats["model"] = record.get("model")
        if run_stats["provider"] is None:
            run_stats["provider"] = record.get("provider")
        stats[record["id"]] = run_stats
    return {"id": session_id, "runs": [record["id"] for record in chain], "messages": messages, "truncated": truncated, "stats": stats}


@router.post("/{session_id}/messages", status_code=201)
@require_permission("runs", "create")
async def send_cli_chat_message(session_id: str, body: CliChatSendRequest, request: Request) -> dict[str, Any]:
    """Follow-up send: a headless exec run inheriting the prior row's flags unless overridden, linked via `resumed_from`."""
    user_id = await _current_user_id(request)
    session_repo = get_cli_session_repo(request)
    prior = await session_repo.get(session_id, user_id=user_id)
    if prior is None:
        raise HTTPException(status_code=404, detail="CLI session not found")
    if not (body.prompt or "").strip():
        raise HTTPException(status_code=422, detail="chat prompt must not be empty")
    if prior["status"] == "running":
        exit_code = cli_runner.sentinel_exit_code(prior.get("log_path"))
        if exit_code is not None:
            finished = await session_repo.mark_finished(session_id, user_id=user_id, exit_code=exit_code)
            if finished is not None:
                prior = finished
        else:
            raise HTTPException(status_code=409, detail="chat has an active run; wait for it to finish or stop it first")
    params = cli_runner.CliLaunchParams(
        prompt=body.prompt.strip(),
        model=body.model if body.model is not None else prior.get("model"),
        provider=body.provider if body.provider is not None else prior.get("provider"),
        preset=prior.get("preset"),
        effort=body.effort if body.effort is not None else prior.get("effort"),
        contributor_mode=prior.get("contributor_mode"),
        approval_mode=prior.get("approval_mode"),
        sandbox_network=prior.get("sandbox_network"),
        trust_workspace=bool(prior.get("trust_workspace")),
        disable_approval=bool(prior.get("disable_approval")),
        disable_sandbox=bool(prior.get("disable_sandbox")),
        yolo=body.yolo if body.yolo is not None else bool(prior.get("yolo")),
        workspace=prior.get("workspace"),
    )
    record = await _start_run(request=request, user_id=user_id, params=params, mode=prior.get("mode", "coding"), action="chat_send", resumed_from=session_id)
    return _item(record)


@router.post("/{session_id}/export")
@require_permission("runs", "create")
async def export_cli_session(session_id: str, request: Request, body: CliExportRequest | None = None) -> dict[str, Any]:
    """Export a session's durable CLI log (redacted by default)."""
    user_id = await _current_user_id(request)
    record = await get_cli_session_repo(request).get(session_id, user_id=user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="CLI session not found")
    session_ref = record.get("session_ref")
    if not session_ref:
        raise HTTPException(status_code=409, detail="CLI session has no linked CLI session ref yet")
    redacted = body.redacted if body is not None else True
    config = _cli_config()
    out_path = _runs_dir() / f"{session_id}.export.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    probe_command = shlex.join(cli_runner.build_export_argv(session_ref=session_ref, out_path=out_path, redacted=redacted, binary=config.binary))
    try:
        await cli_runner.export_session(session_ref=session_ref, out_path=out_path, redacted=redacted, binary=config.binary, timeout=config.probe_timeout_seconds)
    except cli_runner.CliProbeError as exc:
        await _record_audit(request, actor=user_id, action="export", result="failed", session_id=session_id, command=probe_command)
        raise HTTPException(status_code=500, detail=f"CLI export failed: {exc}") from exc
    try:
        text = out_path.read_text(encoding="utf-8")
    except OSError as exc:
        await _record_audit(request, actor=user_id, action="export", result="failed", session_id=session_id, command=probe_command)
        raise HTTPException(status_code=500, detail=f"CLI export produced no file: {exc}") from exc
    try:
        document: Any = json.loads(text)
    except ValueError:
        document = text
    await _record_audit(request, actor=user_id, action="export", result="ok", session_id=session_id, command=probe_command)
    return {"id": session_id, "session_ref": session_ref, "redacted": redacted, "document": document}


@router.get("/{session_id}/events")
@require_permission("runs", "read")
async def stream_cli_session_events(session_id: str, request: Request) -> StreamingResponse:
    """Stream a run's JSONL log over SSE until its sentinel file lands."""
    user_id = await _current_user_id(request)
    session_repo = get_cli_session_repo(request)
    record = await session_repo.get(session_id, user_id=user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="CLI session not found")
    config = _cli_config()

    async def _finalize(exit_code: int) -> None:
        await session_repo.mark_finished(session_id, user_id=user_id, exit_code=exit_code)

    return StreamingResponse(
        _event_generator(
            record.get("log_path"),
            poll_seconds=config.events_poll_seconds,
            request=request,
            on_complete=_finalize,
            run_id=session_id,
        ),
        media_type="text/event-stream",
        headers=sse_response_headers(),
    )


async def _refresh_finished_rows(session_repo, records: list[dict[str, Any]], *, user_id: str) -> list[dict[str, Any]]:
    """Finalize running rows whose sentinel file has landed (poll/refresh reconciliation)."""
    refreshed: list[dict[str, Any]] = []
    for record in records:
        if record["status"] == "running" and record.get("log_path"):
            exit_code = cli_runner.sentinel_exit_code(record["log_path"])
            if exit_code is not None:
                finished = await session_repo.mark_finished(record["id"], user_id=user_id, exit_code=exit_code)
                if finished is not None:
                    record = finished
        refreshed.append(record)
    return refreshed


async def _event_generator(log_path: str | None, *, poll_seconds: float, request: Request, on_complete=None, run_id: str | None = None):
    """Yield run-log lines as SSE data frames plus typed chat frames, then a terminal end frame."""
    offset = 0
    polls = 0
    path = Path(log_path) if log_path else None

    async def _drain_and_end(exit_code: int):
        if path is not None and path.exists():
            # One final drain: the sentinel lands after the last write.
            nonlocal offset
            try:
                with open(path, encoding="utf-8", errors="replace") as handle:
                    handle.seek(offset)
                    tail = handle.read()
            except OSError:
                tail = ""
            for line in tail.splitlines():
                if line.strip():
                    for frame in _chat_frames_for_line(line, run_id=run_id):
                        yield frame
        if on_complete is not None:
            try:
                await on_complete(exit_code)
            except Exception:
                pass
        end_stats = _run_stats_for_log(path, run_id=run_id or "")
        end_stats["exit_code"] = exit_code
        yield f"event: end\ndata: {json.dumps({'exit_code': exit_code, 'stats': end_stats})}\n\n"

    while True:
        if await request.is_disconnected():
            return
        if path is not None and path.exists():
            try:
                with open(path, encoding="utf-8", errors="replace") as handle:
                    handle.seek(offset)
                    chunk = handle.read()
                    offset = handle.tell()
            except OSError:
                chunk = ""
            for line in chunk.splitlines():
                if line.strip():
                    for frame in _chat_frames_for_line(line, run_id=run_id):
                        yield frame
            exit_code = cli_runner.sentinel_exit_code(path)
            if exit_code is not None:
                async for frame in _drain_and_end(exit_code):
                    yield frame
                return
        else:
            exit_code = cli_runner.sentinel_exit_code(path) if path is not None else None
            if exit_code is not None:
                async for frame in _drain_and_end(exit_code):
                    yield frame
                return
        polls += 1
        if polls % 40 == 0:
            yield ": ping\n\n"
        await asyncio.sleep(poll_seconds)


@router.delete("/{session_id}")
@require_permission("runs", "cancel")
async def stop_cli_session(session_id: str, request: Request) -> dict[str, Any]:
    """Stop a run: terminal ``stopped`` status first, then process termination."""
    from datetime import UTC, datetime

    user_id = await _current_user_id(request)
    session_repo = get_cli_session_repo(request)
    record = await session_repo.get(session_id, user_id=user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="CLI session not found")
    if record["status"] not in cli_runner.TERMINAL_STATUSES:
        record = await session_repo.update(session_id, user_id=user_id, status="stopped", ended_at=datetime.now(UTC)) or record
    await cli_runner.stop_run(row_id=session_id, tmux_session=record.get("tmux_session"), pid=record.get("pid"))
    await _record_audit(request, actor=user_id, action="stop", result="stopped", session_id=session_id)
    return _item(record)


@router.get("/{session_id}/inspect")
@require_permission("runs", "read")
async def inspect_cli_session(session_id: str, request: Request) -> dict[str, Any]:
    """Render ``trace inspect`` output for a session's on-disk CLI log."""
    user_id = await _current_user_id(request)
    record = await get_cli_session_repo(request).get(session_id, user_id=user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="CLI session not found")
    session_ref = record.get("session_ref")
    if not session_ref:
        raise HTTPException(status_code=409, detail="CLI session has no linked CLI session ref yet")
    match = next((entry for entry in cli_runner.discover_session_logs() if entry["session_id"] == session_ref), None)
    if match is None:
        raise HTTPException(status_code=409, detail="CLI session log is not on disk yet")
    config = _cli_config()
    probe_command = shlex.join(cli_runner.build_inspect_argv(session_log=match["path"], binary=config.binary))
    try:
        output = await cli_runner.inspect_session_log(session_log=match["path"], binary=config.binary, timeout=config.probe_timeout_seconds)
    except cli_runner.CliProbeError as exc:
        await _record_audit(request, actor=user_id, action="inspect", result="failed", session_id=session_id, command=probe_command)
        raise HTTPException(status_code=500, detail=f"CLI inspect failed: {exc}") from exc
    await _record_audit(request, actor=user_id, action="inspect", result="ok", session_id=session_id, command=probe_command)
    return {"id": session_id, "session_ref": session_ref, "session_log": match["path"], "output": output}


@router.get("/{session_id}/audit")
@require_permission("runs", "read")
async def list_cli_session_audit(session_id: str, request: Request, limit: int = Query(default=100, ge=1, le=200)) -> list[dict[str, Any]]:
    """Return the caller's audit rows for one CLI session, oldest first."""
    user_id = await _current_user_id(request)
    record = await get_cli_session_repo(request).get(session_id, user_id=user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="CLI session not found")
    rows = await get_cli_audit_repo(request).list_for_session(session_id, actor=user_id, limit=limit)
    return [_audit_item(row) for row in rows]


@capabilities_router.get("")
@require_permission("runs", "read")
async def get_cli_capabilities_payload(request: Request) -> dict[str, Any]:
    """Return the local CLI's advertised models, effort tiers, and providers."""
    await _current_user_id(request)
    try:
        return get_cli_capabilities()
    except CliCapabilitiesUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"CLI capabilities unavailable: {exc}") from exc


@presets_router.get("")
@require_permission("runs", "read")
async def list_cli_presets(request: Request) -> list[dict[str, Any]]:
    """List named CLI launch presets."""
    await _current_user_id(request)
    return [_preset_item(record) for record in await get_cli_preset_repo(request).list()]


@presets_router.get("/{name}")
@require_permission("runs", "read")
async def get_cli_preset(name: str, request: Request) -> dict[str, Any]:
    """Return one CLI launch preset by name (built-in 1.3 default when unstored)."""
    await _current_user_id(request)
    record = await cli_runner.load_preset_or_default(get_cli_preset_repo(request), name)
    if record is None:
        raise HTTPException(status_code=404, detail="CLI preset not found")
    return _preset_item(record)


@presets_router.put("/{name}")
@require_permission("runs", "create")
async def put_cli_preset(name: str, body: CliPresetPut, request: Request) -> dict[str, Any]:
    """Create or replace a launch preset (1.3-persistence guard enforced)."""
    user_id = await _current_user_id(request)
    try:
        cli_runner.validate_launch_params(
            cli_runner.CliLaunchParams(
                prompt="preset-validation-only",
                model=body.model,
                provider=body.provider,
                preset=body.preset,
                effort=body.effort,
                contributor_mode=body.contributor_mode,
                approval_mode=body.approval_mode,
            )
        )
        cli_runner.check_preset_guard(model=body.model, effort=body.effort, contributor_mode=body.contributor_mode, yolo=body.yolo)
    except cli_runner.CliValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    record = await get_cli_preset_repo(request).upsert(
        name,
        updated_by=user_id,
        model=body.model,
        provider=body.provider,
        preset=body.preset,
        effort=body.effort,
        contributor_mode=body.contributor_mode,
        yolo=body.yolo,
        approval_mode=body.approval_mode,
    )
    await _record_audit(request, actor=user_id, action="preset_put", result="ok", preset=name, preset_version=_preset_version_label(record))
    return _preset_item(record)

"""Chat-layer backend tests: messages backfill, follow-up send, typed SSE frames."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.gateway import cli_runner
from app.gateway.auth_disabled import AUTH_SOURCE_SESSION
from app.gateway.authz import AuthContext
from app.gateway.routers import cli_sessions
from deerflow.config.cli_config import CliConfig
from deerflow.persistence.cli_sessions import CliAuditRepository, CliPresetRepository, CliSessionRepository
from deerflow.persistence.cli_sessions.model import CliAuditRow, CliPresetRow, CliSessionRow

RUN_PERMISSIONS = ["runs:read", "runs:create", "runs:cancel"]


@pytest_asyncio.fixture
async def gateway(monkeypatch, tmp_path):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(CliSessionRow.__table__.create)
            await connection.run_sync(CliPresetRow.__table__.create)
            await connection.run_sync(CliAuditRow.__table__.create)
        sf = async_sessionmaker(engine, expire_on_commit=False)
        session_repo = CliSessionRepository(sf)
        preset_repo = CliPresetRepository(sf)
        audit_repo = CliAuditRepository(sf)
        monkeypatch.setattr(cli_sessions, "_LOG_DIR_OVERRIDE", tmp_path / "cli_runs")
        monkeypatch.setattr(cli_sessions, "_cli_config", lambda: CliConfig(tmux="off"))
        app = FastAPI()
        app.state.cli_session_repo = session_repo
        app.state.cli_preset_repo = preset_repo
        app.state.cli_audit_repo = audit_repo
        app.include_router(cli_sessions.router)

        @app.middleware("http")
        async def authenticate(request, call_next):
            mode = request.headers.get("x-test-auth", "ok")
            user = None if mode == "anonymous" else SimpleNamespace(id="user-1", system_role="user")
            permissions = [] if mode in ("anonymous", "denied") else list(RUN_PERMISSIONS)
            request.state.user = user
            request.state.auth_source = AUTH_SOURCE_SESSION
            request.state.auth = AuthContext(user=user, permissions=permissions)
            return await call_next(request)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield SimpleNamespace(client=client, sf=sf, session_repo=session_repo, preset_repo=preset_repo, audit_repo=audit_repo, tmp_path=tmp_path)
    finally:
        await engine.dispose()


def _spawn_stub(*, path="direct", pid=515001, tmux_session=None):
    async def _spawn(**kwargs):
        log_path = Path(kwargs["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text('{"type":"started"}\n', encoding="utf-8")
        return cli_runner.SpawnResult(path=path, pid=pid, tmux_session=tmux_session)

    return _spawn


async def _seed_run(gateway, session_id, *, prompt=None, resumed_from=None, log_lines=(), status="completed"):
    record = await gateway.session_repo.create(
        created_by="user-1",
        status=status,
        prompt=prompt,
        resumed_from=resumed_from,
        log_path=str(gateway.tmp_path / "cli_runs" / f"{session_id}.jsonl"),
        session_id=session_id,
    )
    log_path = Path(record["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("".join(f"{line}\n" for line in log_lines), encoding="utf-8")
    return record


def _vocab_envelope(payload_type, payload, *, recorded_at=1792031604529169):
    return json.dumps(
        {
            "schema_version": 1,
            "sequence": 1,
            "recorded_at": recorded_at,
            "record_type": "event",
            "payload_type": payload_type,
            "payload": payload,
        }
    )


def _vocab_fixture_lines():
    """Realistic exec log: preamble noise + deltas + terminal echo + lifecycle rows."""
    full = "I'm Muse Code powered by Meta Muse Spark."
    return [
        "muse: workspace root: /home/fire/Documents/deer-flow/backend (cwd default)",
        "muse: workspace trust: trusted source=run-flag",
        "tbh: reasoning effort ultra is not available; using xhigh",
        _vocab_envelope("runtime.command.accepted", {"kind": "command_accepted", "command_kind": "turn.submit"}, recorded_at=1000),
        _vocab_envelope(
            "run.model.configured",
            {"kind": "run_model_configured", "model_id": "muse-spark-1.3", "provider_id": "meta", "display_label": "muse-spark-1.3"},
            recorded_at=2000,
        ),
        _vocab_envelope("turn.input.user", {"kind": "turn_input_user", "prompt": "what model are you?"}, recorded_at=3000),
        _vocab_envelope("run.lifecycle.started", {"kind": "run_started", "prompt": "what model are you?"}, recorded_at=4000),
        _vocab_envelope(
            "task.lifecycle.proposed",
            {"kind": "task_lifecycle", "task_id": "task-1", "event": {"kind": "proposed", "task_id": "task-1", "task_kind": "model.meta.response"}},
            recorded_at=5000,
        ),
        _vocab_envelope(
            "task.lifecycle.side_effect_intent",
            {
                "kind": "task_lifecycle",
                "task_id": "task-1",
                "event": {"kind": "side_effect_intent", "task_id": "task-1", "operation": "model.meta.response", "policy_decision": "not_applicable"},
            },
            recorded_at=6000,
        ),
        _vocab_envelope(
            "task.lifecycle.status",
            {
                "kind": "task_lifecycle",
                "task_id": "task-1",
                "event": {
                    "kind": "status",
                    "task_id": "task-1",
                    "message": "opening meta model stream attempt 1/10",
                    "details": {"phase": "opening_stream", "facets": [{"kind": "external_attempt", "operation": "model.response"}]},
                },
            },
            recorded_at=7000,
        ),
        _vocab_envelope("run.output.delta", {"kind": "run_output_delta", "text": "I'm"}, recorded_at=8000),
        _vocab_envelope("run.output.delta", {"kind": "run_output_delta", "text": " Muse Code powered by Meta Muse Spark."}, recorded_at=9000),
        _vocab_envelope(
            "task.lifecycle.completed",
            {"kind": "task_lifecycle", "task_id": "task-1", "event": {"kind": "completed", "task_id": "task-1"}},
            recorded_at=10000,
        ),
        _vocab_envelope(
            "run.terminal.completed",
            {"kind": "run_terminal", "terminal": "completed", "reason": None, "text": full},
            recorded_at=11000,
        ),
    ]


def test_typed_frames_terminal_echo_and_noise_emit_no_text():
    full = "I'm Muse Code powered by Meta Muse Spark."
    terminal = _vocab_envelope("run.terminal.completed", {"kind": "run_terminal", "terminal": "completed", "text": full})
    assert cli_sessions._typed_chat_frames(terminal, run_id="r1") == []
    assert cli_sessions._typed_chat_frames("muse: workspace trust: trusted source=run-flag", run_id="r1") == []
    assert cli_sessions._typed_chat_frames(_vocab_envelope("run.lifecycle.started", {"kind": "run_started", "prompt": "hi"}), run_id="r1") == []
    first = cli_sessions._typed_chat_frames(_vocab_envelope("run.output.delta", {"kind": "run_output_delta", "text": "I'm"}), run_id="r1")
    assert first == [("message", {"role": "assistant", "delta": "I'm", "run_id": "r1"})]


def test_lifecycle_events_map_to_tool_rows():
    # Bare state transitions carry no displayable content: no chat rows.
    for kind in ("proposed", "accepted", "scheduled", "started", "completed"):
        frames = cli_sessions._typed_chat_frames(
            _vocab_envelope(
                f"task.lifecycle.{kind}",
                {"task_id": "t1", "event": {"kind": kind, "task_id": "t1", "task_kind": "model.meta.response"}},
            ),
            run_id="r1",
        )
        assert frames == []
    # Intents without an operation and statuses without a message are dropped too.
    assert (
        cli_sessions._typed_chat_frames(
            _vocab_envelope("task.lifecycle.side_effect_intent", {"task_id": "t1", "event": {"kind": "side_effect_intent"}}),
            run_id="r1",
        )
        == []
    )
    # Internal reminder machinery never surfaces as a chat row.
    assert (
        cli_sessions._typed_chat_frames(
            _vocab_envelope(
                "task.lifecycle.side_effect_intent",
                {"task_id": "t1", "event": {"kind": "side_effect_intent", "operation": "reminder.child_run"}},
            ),
            run_id="r1",
        )
        == []
    )
    assert (
        cli_sessions._typed_chat_frames(
            _vocab_envelope("task.lifecycle.status", {"task_id": "t1", "event": {"kind": "status"}}),
            run_id="r1",
        )
        == []
    )
    intent = cli_sessions._typed_chat_frames(
        _vocab_envelope(
            "task.lifecycle.side_effect_intent",
            {"task_id": "t1", "event": {"kind": "side_effect_intent", "operation": "shell.exec", "policy_decision": "auto"}},
        ),
        run_id="r1",
    )
    assert intent == [("tool", {"run_id": "r1", "name": "shell.exec", "status": "intent", "detail": "auto"})]
    status = cli_sessions._typed_chat_frames(
        _vocab_envelope(
            "task.lifecycle.status",
            {
                "task_id": "t1",
                "event": {"kind": "status", "message": "opening stream", "details": {"phase": "opening_stream", "facets": [{"operation": "model.response"}]}},
            },
        ),
        run_id="r1",
    )
    assert status == [("tool", {"run_id": "r1", "name": "model.response", "status": "opening_stream", "detail": "opening stream"})]
    cancelled = cli_sessions._typed_chat_frames(
        _vocab_envelope("task.lifecycle.cancelled", {"task_id": "t9", "event": {"kind": "cancelled", "task_id": "t9", "reason": "superseded"}}),
        run_id="r1",
    )
    assert cancelled == [("tool", {"run_id": "r1", "name": "t9", "status": "cancelled", "detail": "superseded"})]


def test_typed_frames_map_known_lines_and_ignore_unknown():
    message = cli_sessions._typed_chat_frames(json.dumps({"type": "message", "delta": "hello"}), run_id="r1")
    assert message == [("message", {"role": "assistant", "delta": "hello", "run_id": "r1"})]
    tool = cli_sessions._typed_chat_frames(json.dumps({"type": "tool_call", "name": "edit", "status": "started"}), run_id="r1")
    assert tool == [("tool", {"run_id": "r1", "name": "edit", "status": "started", "detail": ""})]
    assert cli_sessions._typed_chat_frames(json.dumps({"type": "future_event", "blob": [1, 2]}), run_id="r1") == []
    assert cli_sessions._typed_chat_frames("not json at all", run_id="r1") == []
    assert cli_sessions._typed_chat_frames("[1, 2]", run_id="r1") == []
    envelope = cli_sessions._typed_chat_frames(
        json.dumps({"payload_type": "run.output.delta", "payload": {"kind": "run_output_delta", "text": "echo hi"}}),
        run_id="r1",
    )
    assert envelope == [("message", {"role": "assistant", "delta": "echo hi", "run_id": "r1"})]
    user_input = cli_sessions._typed_chat_frames(
        json.dumps({"payload_type": "turn.input.user", "payload": {"kind": "turn_input_user", "prompt": "hi"}}),
        run_id="r1",
    )
    assert user_input == []


@pytest.mark.asyncio
async def test_messages_unauthenticated_returns_401(gateway):
    assert (await gateway.client.get("/api/cli-sessions/x/messages", headers={"x-test-auth": "anonymous"})).status_code == 401
    response = await gateway.client.post("/api/cli-sessions/x/messages", json={"prompt": "hi"}, headers={"x-test-auth": "anonymous"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_messages_unknown_chat_returns_404(gateway):
    assert (await gateway.client.get("/api/cli-sessions/missing/messages")).status_code == 404
    response = await gateway.client.post("/api/cli-sessions/missing/messages", json={"prompt": "hi"})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_backfill_orders_chain_oldest_first(gateway):
    await _seed_run(
        gateway,
        "chat-root",
        prompt="first question",
        log_lines=[
            json.dumps({"type": "message", "delta": "first "}),
            json.dumps({"type": "message", "delta": "answer"}),
            json.dumps({"type": "tool_call", "name": "search", "status": "ok", "detail": "3 hits"}),
            json.dumps({"type": "future_event", "blob": True}),
        ],
    )
    await _seed_run(
        gateway,
        "chat-leaf",
        prompt="follow-up",
        resumed_from="chat-root",
        log_lines=[json.dumps({"type": "message", "delta": "second answer"})],
    )
    response = await gateway.client.get("/api/cli-sessions/chat-leaf/messages")
    assert response.status_code == 200
    body = response.json()
    assert body["runs"] == ["chat-root", "chat-leaf"]
    assert body["truncated"] is False
    assert body["messages"] == [
        {"run_id": "chat-root", "role": "user", "content": "first question"},
        {"run_id": "chat-root", "role": "assistant", "content": "first answer"},
        {"run_id": "chat-root", "role": "tool", "name": "search", "status": "ok", "detail": "3 hits"},
        {"run_id": "chat-leaf", "role": "user", "content": "follow-up"},
        {"run_id": "chat-leaf", "role": "assistant", "content": "second answer"},
    ]


@pytest.mark.asyncio
async def test_backfill_caps_long_chains(gateway, monkeypatch):
    monkeypatch.setattr(cli_sessions, "_CLI_CHAT_CHAIN_CAP", 2)
    await _seed_run(gateway, "cap-1", prompt="one")
    await _seed_run(gateway, "cap-2", prompt="two", resumed_from="cap-1")
    await _seed_run(gateway, "cap-3", prompt="three", resumed_from="cap-2")
    body = (await gateway.client.get("/api/cli-sessions/cap-3/messages")).json()
    assert body["runs"] == ["cap-2", "cap-3"]
    assert body["truncated"] is True
    assert [message["content"] for message in body["messages"] if message["role"] == "user"] == ["two", "three"]


@pytest.mark.asyncio
async def test_send_rejects_empty_prompt(gateway, monkeypatch):
    monkeypatch.setattr(cli_runner, "spawn_run", _spawn_stub())
    await _seed_run(gateway, "send-base", prompt="base")
    assert (await gateway.client.post("/api/cli-sessions/send-base/messages", json={"prompt": ""})).status_code == 422
    assert (await gateway.client.post("/api/cli-sessions/send-base/messages", json={"prompt": "   "})).status_code == 422


@pytest.mark.asyncio
async def test_send_accepts_yolo_without_confirm(gateway, monkeypatch):
    monkeypatch.setattr(cli_runner, "spawn_run", _spawn_stub())
    await _seed_run(gateway, "send-yolo", prompt="base")
    response = await gateway.client.post("/api/cli-sessions/send-yolo/messages", json={"prompt": "hi", "yolo": True})
    assert response.status_code == 201
    assert response.json()["yolo"] is True


@pytest.mark.asyncio
async def test_send_links_resume_chain_inherits_and_audits(gateway, monkeypatch):
    monkeypatch.setattr(cli_runner, "spawn_run", _spawn_stub())
    prior = await gateway.session_repo.create(
        created_by="user-1",
        status="completed",
        model="muse-spark-1.3",
        effort="low",
        prompt="original",
        log_path=str(gateway.tmp_path / "cli_runs" / "send-prior.jsonl"),
        session_id="send-prior",
    )
    assert prior["status"] == "completed"
    response = await gateway.client.post(
        "/api/cli-sessions/send-prior/messages",
        json={"prompt": "follow-up"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["resumed_from"] == "send-prior" and body["id"] != "send-prior"
    assert body["model"] == "muse-spark-1.3" and body["effort"] == "low"
    stored = await gateway.session_repo.get(body["id"], user_id="user-1")
    assert stored is not None and stored["prompt"] == "follow-up"
    assert stored["mode"] == prior["mode"]
    audits = await gateway.audit_repo.list_for_session(body["id"], actor="user-1")
    assert [row["action"] for row in audits] == ["chat_send"]


@pytest.mark.asyncio
async def test_send_overrides_apply_per_send(gateway, monkeypatch):
    seen = {}

    async def _spawn(**kwargs):
        seen["command"] = kwargs["command"]
        return cli_runner.SpawnResult(path="direct", pid=515002)

    monkeypatch.setattr(cli_runner, "spawn_run", _spawn)
    await _seed_run(gateway, "send-ovr", prompt="base")
    response = await gateway.client.post(
        "/api/cli-sessions/send-ovr/messages",
        json={
            "prompt": "again",
            "model": "muse-spark-1.3",
            "effort": "low",
            "provider": "meta",
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert (body["model"], body["effort"], body["provider"]) == ("muse-spark-1.3", "low", "meta")
    assert "--reasoning-effort" in seen["command"] and "--provider" in seen["command"]


@pytest.mark.asyncio
async def test_send_rejects_active_run_with_409_then_allows_after_finish(gateway, monkeypatch):
    monkeypatch.setattr(cli_runner, "spawn_run", _spawn_stub())
    launched = (await gateway.client.post("/api/cli-sessions", json={"prompt": "long one"})).json()
    assert launched["status"] == "running"
    busy = await gateway.client.post(f"/api/cli-sessions/{launched['id']}/messages", json={"prompt": "too soon"})
    assert busy.status_code == 409
    record = await gateway.session_repo.get(launched["id"], user_id="user-1")
    assert record is not None
    Path(record["log_path"]).with_suffix(".exit").write_text("0", encoding="utf-8")
    follow = await gateway.client.post(f"/api/cli-sessions/{launched['id']}/messages", json={"prompt": "now ok"})
    assert follow.status_code == 201
    assert (await gateway.session_repo.get(launched["id"], user_id="user-1"))["status"] == "completed"


@pytest.mark.asyncio
async def test_sse_emits_typed_frames_beside_raw_passthrough(gateway):
    unknown = json.dumps({"type": "future_event", "blob": [1]})
    await _seed_run(
        gateway,
        "chat-sse",
        prompt="hi",
        log_lines=[
            json.dumps({"type": "message", "delta": "streamed"}),
            json.dumps({"type": "tool_call", "name": "edit", "status": "done"}),
            unknown,
        ],
    )
    log_path = gateway.tmp_path / "cli_runs" / "chat-sse.jsonl"
    log_path.with_suffix(".exit").write_text("0", encoding="utf-8")
    async with gateway.client.stream("GET", "/api/cli-sessions/chat-sse/events") as response:
        assert response.status_code == 200
        body = (await response.aread()).decode()
    assert f"data: {unknown}" in body
    assert "event: message" in body and '"delta": "streamed"' in body and '"run_id": "chat-sse"' in body
    assert "event: tool" in body and '"name": "edit"' in body
    assert "event: end" in body and '"exit_code": 0' in body


@pytest.mark.asyncio
async def test_backfill_vocab_fixture_single_text_plus_tools_and_stats(gateway):
    await _seed_run(gateway, "chat-vocab", prompt="what model are you?", log_lines=_vocab_fixture_lines())
    (gateway.tmp_path / "cli_runs" / "chat-vocab.exit").write_text("0", encoding="utf-8")
    response = await gateway.client.get("/api/cli-sessions/chat-vocab/messages")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "chat-vocab" and body["runs"] == ["chat-vocab"] and body["truncated"] is False
    assert body["messages"] == [
        {"run_id": "chat-vocab", "role": "user", "content": "what model are you?"},
        {"run_id": "chat-vocab", "role": "tool", "name": "model.meta.response", "status": "intent", "detail": ""},
        {
            "run_id": "chat-vocab",
            "role": "tool",
            "name": "model.response",
            "status": "opening_stream",
            "detail": "opening meta model stream attempt 1/10",
        },
        {"run_id": "chat-vocab", "role": "assistant", "content": "I'm Muse Code powered by Meta Muse Spark."},
    ]
    stats = body["stats"]["chat-vocab"]
    assert stats["model"] == "muse-spark-1.3" and stats["provider"] == "meta"
    assert stats["display_label"] == "muse-spark-1.3"
    assert stats["terminal"] == "completed" and stats["reason"] is None
    assert stats["elapsed_ms"] == pytest.approx((11000 - 1000) / 1000.0)
    assert stats["exit_code"] == 0


@pytest.mark.asyncio
async def test_sse_vocab_fixture_no_duplicate_text_end_carries_stats(gateway):
    await _seed_run(gateway, "chat-vocab-sse", prompt="hi", log_lines=_vocab_fixture_lines())
    log_path = gateway.tmp_path / "cli_runs" / "chat-vocab-sse.jsonl"
    log_path.with_suffix(".exit").write_text("0", encoding="utf-8")
    async with gateway.client.stream("GET", "/api/cli-sessions/chat-vocab-sse/events") as response:
        assert response.status_code == 200
        body = (await response.aread()).decode()
    assert body.count('"delta": "I\'m"') == 1
    assert '"delta": "I\'m Muse Code powered by Meta Muse Spark."' not in body
    assert "run.terminal.completed" in body  # raw compat passthrough preserved
    assert '"name": "model.meta.response"' in body and '"status": "intent"' in body
    assert '"status": "proposed"' not in body and '"status": "completed"' not in body.replace('"terminal": "completed"', "")
    assert '"exit_code": 0' in body and '"model": "muse-spark-1.3"' in body and '"terminal": "completed"' in body

"""Phase-3 backend tests: CLI-harness audit trail (H3.1)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
        await preset_repo.upsert(
            "muse-1.3",
            updated_by="seed",
            model="muse-spark-1.3",
            effort="ultra",
            contributor_mode=True,
            yolo=True,
        )
        monkeypatch.setattr(cli_sessions, "_LOG_DIR_OVERRIDE", tmp_path / "cli_runs")
        monkeypatch.setattr(cli_sessions, "_cli_config", lambda: CliConfig(tmux="off"))
        app = FastAPI()
        app.state.cli_session_repo = session_repo
        app.state.cli_preset_repo = preset_repo
        app.state.cli_audit_repo = audit_repo
        app.include_router(cli_sessions.router)
        app.include_router(cli_sessions.presets_router)

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


def _spawn_stub(*, path="direct", pid=424242, tmux_session=None):
    async def _spawn(**kwargs):
        log_path = Path(kwargs["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text('{"type":"started"}\n', encoding="utf-8")
        return cli_runner.SpawnResult(path=path, pid=pid, tmux_session=tmux_session)

    return _spawn


def test_redact_command_string_masks_secrets():
    scrubbed = cli_runner.redact_command_string("muse exec --json password=hunter2 do things api-key: ABC123")
    assert "hunter2" not in scrubbed and "ABC123" not in scrubbed
    assert "password=[redacted]" in scrubbed and "api-key:[redacted]" in scrubbed
    assert "do things" in scrubbed


def test_redact_command_string_masks_bearer():
    scrubbed = cli_runner.redact_command_string("run with bearer abc.def.ghi trailing")
    assert "abc.def.ghi" not in scrubbed
    assert "bearer [redacted]" in scrubbed and "trailing" in scrubbed


def test_redact_command_string_leaves_plain_commands_alone():
    command = "muse exec --json --model muse-spark-1.3 --reasoning-effort low 'hello'"
    assert cli_runner.redact_command_string(command) == command


@pytest.mark.asyncio
async def test_launch_records_audit_row_with_preset(gateway, monkeypatch):
    monkeypatch.setattr(cli_runner, "spawn_run", _spawn_stub())
    launched = await gateway.client.post("/api/cli-sessions", json={"prompt": "hello", "model": "muse-spark-1.3", "yolo": True, "yolo_confirm": "I-CONFIRM-YOLO"})
    assert launched.status_code == 201
    row_id = launched.json()["id"]
    rows = await gateway.audit_repo.list_for_session(row_id, actor="user-1")
    assert len(rows) == 1
    row = rows[0]
    assert row["actor"] == "user-1" and row["action"] == "launch" and row["result"] == "started"
    assert row["preset"] == "muse-1.3" and row["preset_version"]
    assert "muse exec" in row["command"] and row["created_at"] is not None


@pytest.mark.asyncio
async def test_launch_audit_redacts_prompt_secrets(gateway, monkeypatch):
    monkeypatch.setattr(cli_runner, "spawn_run", _spawn_stub())
    launched = await gateway.client.post("/api/cli-sessions", json={"prompt": "deploy with password=hunter2 now"})
    assert launched.status_code == 201
    rows = await gateway.audit_repo.list_for_session(launched.json()["id"], actor="user-1")
    assert len(rows) == 1 and "hunter2" not in rows[0]["command"]
    assert "[redacted]" in rows[0]["command"]


@pytest.mark.asyncio
async def test_resume_stop_export_inspect_record_audit_rows(gateway, monkeypatch):
    monkeypatch.setattr(cli_runner, "spawn_run", _spawn_stub())
    launched = (await gateway.client.post("/api/cli-sessions", json={"prompt": "hi"})).json()

    resumed = await gateway.client.post(f"/api/cli-sessions/{launched['id']}/resume", json={})
    assert resumed.status_code == 201

    async def _export(*, session_ref, out_path, redacted, binary, timeout):
        out_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return out_path

    monkeypatch.setattr(cli_runner, "export_session", _export)
    exported = await gateway.client.post(f"/api/cli-sessions/{launched['id']}/export", json={})
    assert exported.status_code == 200

    monkeypatch.setattr(cli_runner, "discover_session_logs", lambda **kwargs: [{"session_id": launched["session_ref"], "path": "/tmp/x/session.jsonl"}])
    monkeypatch.setattr(cli_runner, "inspect_session_log", AsyncMock(return_value="trace-ok"))
    inspected = await gateway.client.get(f"/api/cli-sessions/{launched['id']}/inspect")
    assert inspected.status_code == 200

    monkeypatch.setattr(cli_runner, "stop_run", AsyncMock())
    stopped = await gateway.client.delete(f"/api/cli-sessions/{launched['id']}")
    assert stopped.status_code == 200

    rows = await gateway.audit_repo.list_for_session(launched["id"], actor="user-1")
    assert [(row["action"], row["result"]) for row in rows] == [("launch", "started"), ("export", "ok"), ("inspect", "ok"), ("stop", "stopped")]
    resume_rows = await gateway.audit_repo.list_for_session(resumed.json()["id"], actor="user-1")
    assert [(row["action"], row["result"]) for row in resume_rows] == [("resume", "started")]


@pytest.mark.asyncio
async def test_audit_endpoint_lists_rows_and_requires_auth(gateway, monkeypatch):
    monkeypatch.setattr(cli_runner, "spawn_run", _spawn_stub())
    launched = (await gateway.client.post("/api/cli-sessions", json={"prompt": "hi"})).json()
    response = await gateway.client.get(f"/api/cli-sessions/{launched['id']}/audit")
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1 and rows[0]["action"] == "launch" and rows[0]["actor"] == "user-1"

    anonymous = await gateway.client.get(f"/api/cli-sessions/{launched['id']}/audit", headers={"x-test-auth": "anonymous"})
    assert anonymous.status_code == 401

    missing = await gateway.client.get("/api/cli-sessions/does-not-exist/audit")
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_audit_rows_are_owner_scoped(gateway, monkeypatch):
    monkeypatch.setattr(cli_runner, "spawn_run", _spawn_stub())
    launched = (await gateway.client.post("/api/cli-sessions", json={"prompt": "hi"})).json()
    await gateway.session_repo.create(created_by="user-2", session_id="other", prompt="x")
    await gateway.audit_repo.record(actor="user-2", action="launch", result="started", session_id=launched["id"])
    rows = await gateway.audit_repo.list_for_session(launched["id"], actor="user-1")
    assert [row["actor"] for row in rows] == ["user-1"]


@pytest.mark.asyncio
async def test_preset_put_records_audit_row(gateway):
    response = await gateway.client.put("/api/cli-presets/muse-1.3", json={"model": "muse-spark-1.3", "effort": "ultra", "contributor_mode": True, "yolo": True})
    assert response.status_code == 200
    rows = await gateway.audit_repo.list_by_actor(actor="user-1", action="preset_put")
    assert len(rows) == 1
    assert rows[0]["preset"] == "muse-1.3" and rows[0]["preset_version"] and rows[0]["result"] == "ok"


@pytest.mark.asyncio
async def test_spawn_failure_records_failed_audit_row(gateway, monkeypatch):
    async def _boom(**kwargs):
        raise cli_runner.CliSpawnError("no tmux here")

    monkeypatch.setattr(cli_runner, "spawn_run", _boom)
    response = await gateway.client.post("/api/cli-sessions", json={"prompt": "hi"})
    assert response.status_code == 500
    rows = await gateway.audit_repo.list_by_actor(actor="user-1", action="launch")
    assert len(rows) == 1 and rows[0]["result"] == "failed" and rows[0]["session_id"]

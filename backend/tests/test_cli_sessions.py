"""Phase-1 backend tests: CLI-session registry, launch, presets, reconcile."""

from __future__ import annotations

import asyncio
import json
import shlex
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


@pytest.mark.asyncio
async def test_unauthenticated_list_returns_401(gateway):
    response = await gateway.client.get("/api/cli-sessions", headers={"x-test-auth": "anonymous"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_unauthenticated_launch_returns_401(gateway):
    response = await gateway.client.post("/api/cli-sessions", json={"prompt": "hi"}, headers={"x-test-auth": "anonymous"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_denied_stop_returns_403(gateway, monkeypatch):
    monkeypatch.setattr(cli_runner, "spawn_run", _spawn_stub())
    launched = await gateway.client.post("/api/cli-sessions", json={"prompt": "hi"})
    assert launched.status_code == 201
    response = await gateway.client.delete(f"/api/cli-sessions/{launched.json()['id']}", headers={"x-test-auth": "denied"})
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_launch_and_list_round_trip(gateway, monkeypatch):
    monkeypatch.setattr(cli_runner, "spawn_run", _spawn_stub())
    launched = await gateway.client.post("/api/cli-sessions", json={"prompt": "hello", "effort": "low"})
    assert launched.status_code == 201
    body = launched.json()
    assert body["status"] == "running" and body["mode"] == "coding" and body["effort"] == "low"
    assert body["session_ref"]
    detail = await gateway.client.get(f"/api/cli-sessions/{body['id']}")
    assert detail.status_code == 200 and detail.json()["id"] == body["id"]
    listed = await gateway.client.get("/api/cli-sessions")
    assert [row["id"] for row in listed.json()] == [body["id"]]


@pytest.mark.asyncio
async def test_launch_is_owner_scoped(gateway, monkeypatch):
    monkeypatch.setattr(cli_runner, "spawn_run", _spawn_stub())
    launched = await gateway.client.post("/api/cli-sessions", json={"prompt": "mine"})
    row_id = launched.json()["id"]
    assert await gateway.session_repo.get(row_id, user_id="user-2") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", "bogus"),
        ("effort", "turbo"),
        ("approval_mode", "sometimes"),
        ("preset", "evil preset!"),
        ("model", "not a model!!"),
        ("sandbox_network", "wide open"),
    ],
)
async def test_command_builder_rejects_unknown_values(gateway, monkeypatch, field, value):
    monkeypatch.setattr(cli_runner, "spawn_run", _spawn_stub())
    response = await gateway.client.post("/api/cli-sessions", json={"prompt": "hi", field: value})
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"provider": "bogus"}, {"effort": "bogus"}, {"approval_mode": "bogus"}, {"preset": "bogus preset!"}, {"model": "bad model"}, {"workspace": "/nonexistent-dir-xyz"}])
async def test_validate_launch_params_rejects_unknown_values(kwargs):
    with pytest.raises(cli_runner.CliValidationError):
        cli_runner.validate_launch_params(cli_runner.CliLaunchParams(prompt="hi", **kwargs))


def test_validate_launch_params_passes_opaque_preset_through():
    """Run presets are an opaque passthrough: any safe token is accepted
    and forwarded to `muse exec --preset`; the CLI owns the namespace."""
    params = cli_runner.validate_launch_params(cli_runner.CliLaunchParams(prompt="hi", preset="my-future-preset"))
    argv = cli_runner.build_exec_argv(params)
    assert "--preset" in argv
    assert argv[argv.index("--preset") + 1] == "my-future-preset"


def test_validate_launch_params_rejects_effort_with_echo():
    with pytest.raises(cli_runner.CliValidationError, match="echo"):
        cli_runner.validate_launch_params(cli_runner.CliLaunchParams(prompt="hi", provider="echo", effort="low"))


def test_command_string_quotes_prompt_safely():
    params = cli_runner.CliLaunchParams(prompt="hello; rm -rf /", effort="low", model="muse-spark-1.3")
    command = cli_runner.build_command_string(params)
    assert command.startswith("muse exec --json ")
    assert shlex.split(command)[-1] == "hello; rm -rf /"


def test_command_has_no_freeform_passthrough():
    params = cli_runner.CliLaunchParams(prompt="hi", yolo=True)
    argv = cli_runner.build_exec_argv(params)
    assert argv[:3] == ["muse", "exec", "--json"]
    assert "--yolo" in argv
    assert not hasattr(params, "extra_args")


@pytest.mark.asyncio
async def test_yolo_launches_without_confirm(gateway, monkeypatch):
    seen = {}

    async def _spawn(**kwargs):
        seen["command"] = kwargs["command"]
        return cli_runner.SpawnResult(path="direct", pid=999001)

    monkeypatch.setattr(cli_runner, "spawn_run", _spawn)
    response = await gateway.client.post("/api/cli-sessions", json={"prompt": "hi", "yolo": True})
    assert response.status_code == 201
    assert response.json()["yolo"] is True
    assert "--yolo" in shlex.split(seen["command"])


@pytest.mark.asyncio
async def test_preset_scoping_rejects_privileged_fields_for_other_models(gateway):
    for payload in (
        {"model": "other-model", "contributor_mode": True},
        {"model": "other-model", "effort": "low"},
        {"model": "other-model", "effort": "ultra"},
        {"model": "other-model", "yolo": True},
        {"model": None, "effort": "low"},
    ):
        response = await gateway.client.put("/api/cli-presets/other", json=payload)
        assert response.status_code == 422, payload


@pytest.mark.asyncio
async def test_preset_scoping_allows_plain_fields_for_other_models(gateway):
    response = await gateway.client.put("/api/cli-presets/other", json={"model": "other-model", "provider": "meta", "approval_mode": "never"})
    assert response.status_code == 200
    assert response.json()["provider"] == "meta"


@pytest.mark.asyncio
async def test_unknown_preset_name_returns_404(gateway):
    assert (await gateway.client.get("/api/cli-presets/nope")).status_code == 404


@pytest.mark.asyncio
async def test_13_preset_falls_back_to_builtin_when_unstored():
    class _EmptyRepo:
        async def get(self, name):
            return None

    fallback = await cli_runner.load_preset_or_default(_EmptyRepo(), "muse-1.3")
    assert fallback["model"] == "muse-spark-1.3" and fallback["effort"] == "ultra"
    assert fallback["contributor_mode"] is True and fallback["yolo"] is True
    assert await cli_runner.load_preset_or_default(_EmptyRepo(), "other") is None


@pytest.mark.asyncio
async def test_preset_scoping_allows_privileged_fields_for_13_key(gateway):
    response = await gateway.client.put(
        "/api/cli-presets/muse-1.3",
        json={"model": "muse-spark-1.3", "effort": "ultra", "contributor_mode": True, "yolo": True},
    )
    assert response.status_code == 200
    fetched = await gateway.client.get("/api/cli-presets/muse-1.3")
    assert fetched.json()["effort"] == "ultra"


@pytest.mark.asyncio
async def test_launch_applies_13_preset_only_on_explicit_model_match(gateway, monkeypatch):
    seen = {}

    async def _spawn(**kwargs):
        seen["command"] = kwargs["command"]
        return cli_runner.SpawnResult(path="direct", pid=999002)

    monkeypatch.setattr(cli_runner, "spawn_run", _spawn)
    matched = await gateway.client.post(
        "/api/cli-sessions",
        json={"prompt": "hi", "model": "muse-spark-1.3"},
    )
    assert matched.status_code == 201
    assert matched.json()["effort"] == "ultra" and matched.json()["contributor_mode"] is True
    assert "--reasoning-effort" in shlex.split(seen["command"])

    omitted = await gateway.client.post("/api/cli-sessions", json={"prompt": "hi"})
    assert omitted.status_code == 201
    assert omitted.json()["effort"] is None and omitted.json()["contributor_mode"] is None and omitted.json()["yolo"] is False

    other = await gateway.client.post("/api/cli-sessions", json={"prompt": "hi", "model": "other-model"})
    assert other.status_code == 201
    assert other.json()["effort"] is None and other.json()["yolo"] is False


@pytest.mark.asyncio
async def test_sse_events_headers_and_terminal_stream(gateway):
    record = await gateway.session_repo.create(
        created_by="user-1",
        status="completed",
        model="muse-spark-1.3",
        log_path=str(gateway.tmp_path / "cli_runs" / "evt1.jsonl"),
        session_id="evt1",
    )
    log_path = Path(record["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text('{"type":"a"}\n{"type":"b"}\n', encoding="utf-8")
    log_path.with_suffix(".exit").write_text("0", encoding="utf-8")
    async with gateway.client.stream("GET", "/api/cli-sessions/evt1/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert "no-transform" in response.headers["cache-control"]
        assert response.headers["x-accel-buffering"] == "no"
        text = await response.aread()
    body = text.decode()
    assert 'data: {"type":"a"}' in body and 'data: {"type":"b"}' in body
    assert "event: end" in body and '"exit_code": 0' in body


@pytest.mark.asyncio
async def test_detail_poll_finalizes_sentinel_finished_rows(gateway):
    record = await gateway.session_repo.create(created_by="user-1", log_path=str(gateway.tmp_path / "cli_runs" / "fin1.jsonl"), session_id="fin1")
    log_path = Path(record["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text('{"type":"done"}\n', encoding="utf-8")
    log_path.with_suffix(".exit").write_text("3", encoding="utf-8")
    detail = await gateway.client.get("/api/cli-sessions/fin1")
    assert detail.status_code == 200
    assert detail.json()["status"] == "failed" and detail.json()["ended_at"] is not None
    assert (await gateway.session_repo.get("fin1", user_id="user-1"))["status"] == "failed"


@pytest.mark.asyncio
async def test_resume_inherits_prior_flags_and_links(gateway, monkeypatch):
    monkeypatch.setattr(cli_runner, "spawn_run", _spawn_stub())
    prior = (
        await gateway.client.post(
            "/api/cli-sessions",
            json={"prompt": "original", "model": "muse-spark-1.3", "effort": "low"},
        )
    ).json()
    resumed = await gateway.client.post(f"/api/cli-sessions/{prior['id']}/resume", json={})
    assert resumed.status_code == 201
    body = resumed.json()
    assert body["resumed_from"] == prior["id"] and body["model"] == "muse-spark-1.3" and body["effort"] == "low"
    assert body["id"] != prior["id"] and body["session_ref"] != prior["session_ref"]


@pytest.mark.asyncio
async def test_resume_without_any_prompt_rejected(gateway, monkeypatch):
    monkeypatch.setattr(cli_runner, "spawn_run", _spawn_stub())
    record = await gateway.session_repo.create(created_by="user-1", session_id="noprompt")
    assert record["prompt"] is None
    response = await gateway.client.post("/api/cli-sessions/noprompt/resume", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_stop_sets_terminal_status_first(gateway, monkeypatch):
    monkeypatch.setattr(cli_runner, "spawn_run", _spawn_stub())
    stop_spy = AsyncMock()
    monkeypatch.setattr(cli_runner, "stop_run", stop_spy)
    launched = (await gateway.client.post("/api/cli-sessions", json={"prompt": "hi"})).json()
    stopped = await gateway.client.delete(f"/api/cli-sessions/{launched['id']}")
    assert stopped.status_code == 200
    assert stopped.json()["status"] == "stopped" and stopped.json()["ended_at"] is not None
    stop_spy.assert_awaited_once()


@pytest.mark.asyncio
async def test_export_requires_linked_session_ref(gateway):
    await gateway.session_repo.create(created_by="user-1", session_id="nolink")
    response = await gateway.client.post("/api/cli-sessions/nolink/export", json={})
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_export_returns_document(gateway, monkeypatch):
    record = await gateway.session_repo.create(created_by="user-1", session_ref="ref-1", session_id="exp1")

    async def _export(*, session_ref, out_path, redacted, binary, timeout):
        assert session_ref == "ref-1" and redacted is True
        out_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return out_path

    monkeypatch.setattr(cli_runner, "export_session", _export)
    response = await gateway.client.post(f"/api/cli-sessions/{record['id']}/export", json={})
    assert response.status_code == 200
    assert response.json()["document"] == {"ok": True}


@pytest.mark.asyncio
async def test_inspect_requires_on_disk_log(gateway, monkeypatch):
    record = await gateway.session_repo.create(created_by="user-1", session_ref="ref-missing", session_id="insp1")
    monkeypatch.setattr(cli_runner, "discover_session_logs", lambda **kwargs: [])
    response = await gateway.client.get(f"/api/cli-sessions/{record['id']}/inspect")
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_boot_reconciliation_marks_orphaned_interrupted(gateway):
    orphan = await gateway.session_repo.create(created_by="user-1", pid=2**30 - 7, session_id="orphan1")
    assert orphan["status"] == "running"
    stopped = await gateway.session_repo.create(created_by="user-1", status="stopped", session_id="term1")
    summary = await cli_runner.reconcile_on_boot(gateway.session_repo)
    assert summary["checked"] >= 1 and "orphan1" in summary["interrupted"]
    assert (await gateway.session_repo.get("orphan1", user_id="user-1"))["status"] == "interrupted"
    terminal = await gateway.session_repo.get("term1", user_id="user-1")
    assert terminal["status"] == "stopped" and terminal["ended_at"] == stopped["ended_at"]


@pytest.mark.asyncio
async def test_boot_reconciliation_finalizes_sentinel_finished_rows(gateway):
    log_path = gateway.tmp_path / "cli_runs" / "bootfin.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text('{"t":1}\n', encoding="utf-8")
    log_path.with_suffix(".exit").write_text("0", encoding="utf-8")
    await gateway.session_repo.create(created_by="user-1", tmux_session="qf-gone", log_path=str(log_path), session_id="bootfin")
    summary = await cli_runner.reconcile_on_boot(gateway.session_repo)
    assert "bootfin" in summary["finished"]
    assert (await gateway.session_repo.get("bootfin", user_id="user-1"))["status"] == "completed"


@pytest.mark.asyncio
async def test_direct_spawn_and_stop_round_trip(gateway):
    log_path = gateway.tmp_path / "cli_runs" / "direct1.jsonl"
    spawn = await cli_runner.spawn_run(row_id="direct1", command="printf '{\"t\":1}\\n'", log_path=log_path, tmux_mode="off")
    assert spawn.path == "direct" and spawn.pid
    for _ in range(100):
        if cli_runner.sentinel_exit_code(log_path) is not None:
            break
        await asyncio.sleep(0.05)
    assert cli_runner.sentinel_exit_code(log_path) == 0
    assert '{"t":1}' in log_path.read_text(encoding="utf-8")
    await cli_runner.stop_run(row_id="direct1", tmux_session=None, pid=spawn.pid)


@pytest.mark.asyncio
async def test_tmux_spawn_reattach_and_boot_rule(gateway):
    if not cli_runner.tmux_available():
        pytest.skip("tmux binary missing")
    name = cli_runner.tmux_session_name("tmuxrt1")
    log_path = gateway.tmp_path / "cli_runs" / "tmuxrt1.jsonl"
    started = asyncio.get_running_loop().time()
    spawn = await cli_runner.spawn_run(row_id="tmuxrt1", command="printf '{\"t\":2}\\n'; sleep 30", log_path=log_path, tmux_mode="on")
    # Spawn must return while the run is still going: awaiting the tmux
    # client's pipes would block until the server exits (run end).
    assert asyncio.get_running_loop().time() - started < 15
    assert spawn.path == "tmux" and spawn.tmux_session == name
    try:
        assert await cli_runner.tmux_has_session(name)
        await gateway.session_repo.create(created_by="user-1", tmux_session=name, log_path=str(log_path), session_id="tmuxrt1")
        summary = await cli_runner.reconcile_on_boot(gateway.session_repo)
        assert "tmuxrt1" in summary["reattached"]
        assert (await gateway.session_repo.get("tmuxrt1", user_id="user-1"))["status"] == "running"
    finally:
        await cli_runner.stop_run(row_id="tmuxrt1", tmux_session=name, pid=None)
    assert not await cli_runner.tmux_has_session(name)
    summary = await cli_runner.reconcile_on_boot(gateway.session_repo)
    assert "tmuxrt1" in summary["interrupted"]


@pytest.mark.asyncio
async def test_tmux_on_without_binary_fails_launch(gateway, monkeypatch):
    monkeypatch.setattr(cli_runner, "tmux_available", lambda: False)
    monkeypatch.setattr(cli_sessions, "_cli_config", lambda: CliConfig(tmux="on"))
    response = await gateway.client.post("/api/cli-sessions", json={"prompt": "hi"})
    assert response.status_code == 500
    listed = await gateway.client.get("/api/cli-sessions")
    assert listed.json()[0]["status"] == "failed"

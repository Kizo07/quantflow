"""Persistent-bot HTTP surface: status, off-switch, heartbeat task lock.

Bare FastAPI app (stub auth) over per-test temp SQLite, mirroring
``test_scheduled_task_router_behavior``'s HTTP fixture. The bot is seeded
through the real ``PersistentBotService``; the live gateway/DB are never
touched.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
import pytest_asyncio
from _router_auth_helpers import make_authed_test_app

from app.gateway.auth.models import User
from app.gateway.auth_disabled import AUTH_SOURCE_SESSION
from app.gateway.routers import persistent_bot as persistent_bot_router
from app.gateway.routers import scheduled_tasks
from deerflow.config.database_config import DatabaseConfig
from deerflow.config.persistent_bot_config import PersistentBotConfig
from deerflow.persistence.engine import close_engine, get_session_factory, init_engine_from_config
from deerflow.persistence.persistent_bot import PersistentBotRepository
from deerflow.persistence.scheduled_task_runs import ScheduledTaskRunRepository
from deerflow.persistence.scheduled_tasks import ScheduledTaskRepository
from deerflow.persistence.thread_meta.sql import ThreadMetaRepository
from deerflow.persistent_bot.service import HEARTBEAT_TASK_ID, PersistentBotService

_OWNER = User(id=UUID("12345678-1234-1234-1234-123456789abc"), email="bot-owner@example.com", password_hash="unused", system_role="user")
OWNER = str(_OWNER.id)


@pytest_asyncio.fixture
async def http(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("DEER_FLOW_HOME", str(home))
    from deerflow.config import paths as paths_module

    monkeypatch.setattr(paths_module, "_paths", None)

    await init_engine_from_config(DatabaseConfig(backend="sqlite", sqlite_dir=str(tmp_path / "db")))
    sf = get_session_factory()
    assert sf is not None
    bot_repo = PersistentBotRepository(sf)
    task_repo = ScheduledTaskRepository(sf)
    thread_store = ThreadMetaRepository(sf)

    from deerflow.persistence.agents.file import FileAgentStore

    config = PersistentBotConfig(enabled=True, owner_user_id=OWNER, heartbeat_interval_seconds=300)
    service = PersistentBotService(config=config, bot_repo=bot_repo, thread_store=thread_store, task_repo=task_repo, agent_store=FileAgentStore())
    seeded = await service.ensure_seeded()
    assert seeded["seeded"] is True

    app = make_authed_test_app(user_factory=lambda: _OWNER, bind_current_user=True)

    @app.middleware("http")
    async def mark_session_source(request, call_next):
        request.state.auth_source = AUTH_SOURCE_SESSION
        return await call_next(request)

    app.state.scheduled_task_repo = task_repo
    app.state.scheduled_task_run_repo = ScheduledTaskRunRepository(sf)
    app.state.persistent_bot_repo = bot_repo
    app.state.persistent_bot_service = service
    app.state.thread_store = thread_store
    app.include_router(persistent_bot_router.router)
    app.include_router(scheduled_tasks.router)
    monkeypatch.setattr(persistent_bot_router, "get_config", lambda: SimpleNamespace(persistent_bot=config))
    monkeypatch.setattr(scheduled_tasks, "get_config", lambda: SimpleNamespace(scheduler=SimpleNamespace()))
    monkeypatch.setattr(scheduled_tasks, "is_scheduler_running", lambda _request: True)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://bot.test") as client:
            yield SimpleNamespace(client=client, task_repo=task_repo, bot_repo=bot_repo, seeded=seeded)
    finally:
        await close_engine()


@pytest.mark.asyncio
async def test_status_pause_resume_round_trip(http):
    response = await http.client.get("/api/persistent-bot")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["enabled"] is True
    assert body["seeded"] is True
    assert body["user_paused"] is False
    assert body["heartbeat_task_id"] == HEARTBEAT_TASK_ID
    assert body["heartbeat_status"] == "enabled"

    response = await http.client.post("/api/persistent-bot/pause")
    assert response.status_code == 200, response.text
    assert response.json()["user_paused"] is True
    task = await http.task_repo.get_internal(HEARTBEAT_TASK_ID)
    assert task["status"] == "paused"

    response = await http.client.post("/api/persistent-bot/resume")
    assert response.status_code == 200, response.text
    assert response.json()["user_paused"] is False
    task = await http.task_repo.get_internal(HEARTBEAT_TASK_ID)
    assert task["status"] == "enabled"


@pytest.mark.asyncio
async def test_heartbeat_hidden_from_task_list_and_locked(http):
    # An ordinary task for contrast.
    await http.task_repo.create(
        task_id="ordinary-1",
        user_id=OWNER,
        thread_id=None,
        context_mode="fresh_thread_per_run",
        assistant_id="lead_agent",
        title="Ordinary",
        prompt="Check things.",
        schedule_type="interval",
        schedule_spec={"every_seconds": 3600},
        timezone="UTC",
        next_run_at=None,
    )

    response = await http.client.get("/api/scheduled-tasks")
    assert response.status_code == 200, response.text
    ids = [task["id"] for task in response.json()]
    assert ids == ["ordinary-1"]

    for method, url in [
        ("patch", f"/api/scheduled-tasks/{HEARTBEAT_TASK_ID}"),
        ("post", f"/api/scheduled-tasks/{HEARTBEAT_TASK_ID}/pause"),
        ("post", f"/api/scheduled-tasks/{HEARTBEAT_TASK_ID}/resume"),
        ("delete", f"/api/scheduled-tasks/{HEARTBEAT_TASK_ID}"),
    ]:
        kwargs = {"json": {"title": "Hijacked"}} if method == "patch" else {}
        response = await getattr(http.client, method)(url, **kwargs)
        assert response.status_code == 403, (method, response.text)
        assert response.json()["detail"]["code"] == "task_managed"

    # The heartbeat survived untouched; the ordinary task still pauses.
    task = await http.task_repo.get_internal(HEARTBEAT_TASK_ID)
    assert task is not None and task["status"] == "enabled"
    response = await http.client.post("/api/scheduled-tasks/ordinary-1/pause")
    assert response.status_code == 200, response.text

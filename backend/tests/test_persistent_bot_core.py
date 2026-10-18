"""Persistent-bot core (P1): config, singleton, seed/repair, arbitration.

All databases are per-test temp SQLite files; the live gateway database is
never touched. The bot agent uses the real ``FileAgentStore`` rooted at a
temp ``DEER_FLOW_HOME``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from deerflow.config.database_config import DatabaseConfig
from deerflow.config.persistent_bot_config import PersistentBotConfig, get_persistent_bot_config
from deerflow.persistence.engine import close_engine, get_session_factory, init_engine_from_config
from deerflow.persistence.persistent_bot import PersistentBotRepository
from deerflow.persistence.persistent_bot.model import PersistentBotRow
from deerflow.persistence.scheduled_task_runs import ScheduledTaskRunRepository
from deerflow.persistence.scheduled_task_runs.model import ScheduledTaskRunRow
from deerflow.persistence.scheduled_tasks import ScheduledTaskRepository
from deerflow.persistence.thread_meta.base import THREAD_ARCHIVED_METADATA_KEY, THREAD_PINNED_METADATA_KEY
from deerflow.persistence.thread_meta.sql import ThreadMetaRepository
from deerflow.persistent_bot.service import HEARTBEAT_TASK_ID, PersistentBotService

pytestmark = pytest.mark.asyncio


@pytest.fixture()
def agent_home(tmp_path, monkeypatch):
    """Root the file agent store at a temp DEER_FLOW_HOME."""
    home = tmp_path / "home"
    monkeypatch.setenv("DEER_FLOW_HOME", str(home))
    from deerflow.config import paths as paths_module

    monkeypatch.setattr(paths_module, "_paths", None)
    return home


def _bot_config(**overrides) -> PersistentBotConfig:
    values: dict = {"enabled": True, "heartbeat_interval_seconds": 300}
    values.update(overrides)
    return PersistentBotConfig(**values)


async def _make_bot_service(tmp_path: Path, config: PersistentBotConfig):
    await init_engine_from_config(DatabaseConfig(backend="sqlite", sqlite_dir=str(tmp_path / "db")))
    session_factory = get_session_factory()
    assert session_factory is not None
    from deerflow.persistence.agents.file import FileAgentStore

    return (
        PersistentBotService(
            config=config,
            bot_repo=PersistentBotRepository(session_factory),
            thread_store=ThreadMetaRepository(session_factory),
            task_repo=ScheduledTaskRepository(session_factory),
            agent_store=FileAgentStore(),
        ),
        session_factory,
    )


# -- config ---------------------------------------------------------------


async def test_persistent_bot_defaults_off_with_interval_floor():
    config = PersistentBotConfig()
    assert config.enabled is False
    assert get_persistent_bot_config().enabled is False
    assert config.heartbeat_interval_seconds >= 300
    with pytest.raises(ValueError):
        PersistentBotConfig(heartbeat_interval_seconds=299)
    assert PersistentBotConfig(heartbeat_interval_seconds=300).heartbeat_interval_seconds == 300
    with pytest.raises(ValueError):
        PersistentBotConfig(agent_name="bad name!")
    with pytest.raises(ValueError):
        PersistentBotConfig(overlap_policy="whatever")  # type: ignore[arg-type]


async def test_persistent_bot_registered_startup_only():
    from deerflow.config.app_config import AppConfig
    from deerflow.config.reload_boundary import STARTUP_ONLY_FIELDS, STARTUP_ONLY_PREFIX

    assert "persistent_bot" in STARTUP_ONLY_FIELDS
    description = AppConfig.model_fields["persistent_bot"].description or ""
    assert description.startswith(STARTUP_ONLY_PREFIX)
    assert AppConfig.model_fields["persistent_bot"].default_factory() == PersistentBotConfig()


# -- singleton ------------------------------------------------------------


async def test_singleton_uniqueness_under_concurrent_ensure(tmp_path):
    await init_engine_from_config(DatabaseConfig(backend="sqlite", sqlite_dir=str(tmp_path / "db")))
    try:
        session_factory = get_session_factory()
        assert session_factory is not None
        repo = PersistentBotRepository(session_factory)
        results = await asyncio.gather(*(repo.ensure(user_id="default", agent_name="persistent-bot", thread_id=f"thread-{i}") for i in range(8)))
        assert sum(created for _, created in results) == 1
        assert len({row["thread_id"] for row, _ in results}) == 1
        async with session_factory() as session:
            count = await session.scalar(sa.text("SELECT COUNT(*) FROM persistent_bot"))
            assert count == 1
            # The check constraint rejects any second singleton id.
            session.add(
                PersistentBotRow(
                    id=2,
                    user_id="default",
                    agent_name="other",
                    thread_id="thread-other",
                    user_paused=False,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )
            with pytest.raises(IntegrityError):
                await session.commit()
    finally:
        await close_engine()


# -- seed / repair ----------------------------------------------------------


async def test_seed_creates_agent_thread_singleton_and_heartbeat(agent_home, tmp_path):
    service, session_factory = await _make_bot_service(tmp_path, _bot_config())
    try:
        seeded = await service.ensure_seeded()
        assert seeded["seeded"] is True
        assert seeded["heartbeat_task_id"] == HEARTBEAT_TASK_ID

        from deerflow.persistence.agents.file import FileAgentStore

        store = FileAgentStore()
        agent = store.get("persistent-bot", user_id="default")
        assert agent.mcp_plugins == []
        assert store.get_soul("persistent-bot", user_id="default")

        thread = await ThreadMetaRepository(session_factory).get(seeded["thread_id"], user_id="default")
        assert thread is not None
        assert thread["assistant_id"] == "persistent-bot"
        assert thread["metadata"][THREAD_PINNED_METADATA_KEY] is True
        assert thread["metadata"][THREAD_ARCHIVED_METADATA_KEY] is False

        bot = await PersistentBotRepository(session_factory).get()
        assert bot is not None
        assert bot["thread_id"] == seeded["thread_id"]
        assert bot["heartbeat_task_id"] == HEARTBEAT_TASK_ID
        assert bot["user_paused"] in (False, 0)

        task = await ScheduledTaskRepository(session_factory).get_internal(HEARTBEAT_TASK_ID)
        assert task is not None
        assert task["context_mode"] == "reuse_thread"
        assert task["thread_id"] == seeded["thread_id"]
        assert task["assistant_id"] == "persistent-bot"
        assert task["goal_objective"] is None
        assert task["stop_condition"] is None
        assert task["max_runs"] is None
        assert task["end_at"] is None
        assert task["overlap_policy"] == "skip"
        assert task["schedule_type"] == "interval"
        assert task["schedule_spec"] == {"every_seconds": 300}
        assert task["status"] == "enabled"
        assert task["next_run_at"] is not None
    finally:
        await close_engine()


async def test_reseed_is_idempotent_and_concurrent_seed_converges(agent_home, tmp_path):
    service, session_factory = await _make_bot_service(tmp_path, _bot_config())
    try:
        first = await service.ensure_seeded()
        second = await service.ensure_seeded()
        assert second["thread_id"] == first["thread_id"]
        assert second["heartbeat_task_id"] == first["heartbeat_task_id"]

        results = await asyncio.gather(*(service.ensure_seeded() for _ in range(4)))
        assert {result["thread_id"] for result in results} == {first["thread_id"]}
        tasks = await ScheduledTaskRepository(session_factory).list_by_user("default")
        assert [task["id"] for task in tasks] == [HEARTBEAT_TASK_ID]
    finally:
        await close_engine()


async def test_repair_after_thread_delete_and_archive(agent_home, tmp_path):
    service, session_factory = await _make_bot_service(tmp_path, _bot_config())
    try:
        thread_store = ThreadMetaRepository(session_factory)
        seeded = await service.ensure_seeded()
        original_thread = seeded["thread_id"]

        # Archive + unpin is repaired in place (same thread).
        await thread_store.update_metadata(
            original_thread,
            {THREAD_ARCHIVED_METADATA_KEY: True, THREAD_PINNED_METADATA_KEY: False},
            user_id="default",
        )
        repaired = await service.ensure_seeded()
        assert repaired["thread_id"] == original_thread
        thread = await thread_store.get(original_thread, user_id="default")
        assert thread["metadata"][THREAD_PINNED_METADATA_KEY] is True
        assert thread["metadata"][THREAD_ARCHIVED_METADATA_KEY] is False

        # Deletion is repaired with a fresh thread and a rebound heartbeat.
        await thread_store.delete(original_thread, user_id="default")
        replaced = await service.ensure_seeded()
        assert replaced["thread_id"] != original_thread
        assert await thread_store.get(replaced["thread_id"], user_id="default") is not None
        bot = await PersistentBotRepository(session_factory).get()
        assert bot["thread_id"] == replaced["thread_id"]
        task = await ScheduledTaskRepository(session_factory).get_internal(HEARTBEAT_TASK_ID)
        assert task["thread_id"] == replaced["thread_id"]
    finally:
        await close_engine()


async def test_off_switch_and_task_pause_are_respected(agent_home, tmp_path):
    service, session_factory = await _make_bot_service(tmp_path, _bot_config())
    try:
        bot_repo = PersistentBotRepository(session_factory)
        task_repo = ScheduledTaskRepository(session_factory)
        thread_store = ThreadMetaRepository(session_factory)
        seeded = await service.ensure_seeded()

        # A user-paused heartbeat task is never re-enabled by repair.
        result = await task_repo.pause_with_queue_cancellation(
            HEARTBEAT_TASK_ID,
            user_id="default",
            error="paused by user",
            now=datetime.now(UTC),
        )
        assert result == "paused"
        await service.ensure_seeded()
        task = await task_repo.get_internal(HEARTBEAT_TASK_ID)
        assert task["status"] == "paused"

        # The explicit off-switch stops all repair, even after deletion.
        await bot_repo.set_user_paused(True)
        await thread_store.delete(seeded["thread_id"], user_id="default")
        await task_repo.delete(HEARTBEAT_TASK_ID, user_id="default")
        outcome = await service.ensure_seeded()
        assert outcome["user_paused"] is True
        assert outcome["seeded"] is False
        assert await thread_store.get(seeded["thread_id"], user_id="default") is None
        assert await task_repo.get_internal(HEARTBEAT_TASK_ID) is None
    finally:
        await close_engine()


async def test_cancelled_heartbeat_becomes_off_switch_and_resume_revives(agent_home, tmp_path):
    service, session_factory = await _make_bot_service(tmp_path, _bot_config())
    try:
        bot_repo = PersistentBotRepository(session_factory)
        task_repo = ScheduledTaskRepository(session_factory)
        await service.ensure_seeded()

        # Cancelling the heartbeat is a user stop decision: repair records
        # the off-switch instead of resurrecting the task.
        await task_repo.update(HEARTBEAT_TASK_ID, user_id="default", updates={"status": "cancelled"})
        await service.ensure_seeded()
        task = await task_repo.get_internal(HEARTBEAT_TASK_ID)
        assert task["status"] == "cancelled"
        bot = await bot_repo.get()
        assert bot["user_paused"] in (True, 1)

        # Only an explicit resume revives it.
        await service.set_user_paused(False)
        task = await task_repo.get_internal(HEARTBEAT_TASK_ID)
        assert task["status"] == "enabled"
        assert task["next_run_at"] is not None
        bot = await bot_repo.get()
        assert bot["user_paused"] in (False, 0)
    finally:
        await close_engine()


async def test_owner_change_retires_previous_heartbeat_and_pin(agent_home, tmp_path):
    service, session_factory = await _make_bot_service(tmp_path, _bot_config())
    try:
        task_repo = ScheduledTaskRepository(session_factory)
        thread_store = ThreadMetaRepository(session_factory)
        seeded = await service.ensure_seeded()
        old_thread = seeded["thread_id"]

        # Re-seed under a new owner (no-auth -> auth migration): the old
        # owner's heartbeat must not keep firing on the orphaned thread.
        from deerflow.persistence.agents.file import FileAgentStore

        new_service = PersistentBotService(
            config=_bot_config(owner_user_id="admin-1"),
            bot_repo=PersistentBotRepository(session_factory),
            thread_store=thread_store,
            task_repo=task_repo,
            agent_store=FileAgentStore(),
        )
        outcome = await new_service.ensure_seeded()
        assert outcome["user_id"] == "admin-1"
        assert outcome["thread_id"] != old_thread

        old_task = await task_repo.get_internal(HEARTBEAT_TASK_ID)
        if old_task is not None and old_task.get("user_id") == "default":
            assert old_task["status"] == "paused"
        else:
            # Stable id adopted by the new owner; no enabled default-owned task remains.
            default_tasks = await task_repo.list_by_user("default")
            assert all(task["status"] != "enabled" for task in default_tasks)
        old = await thread_store.get(old_thread, user_id="default")
        assert old["metadata"][THREAD_PINNED_METADATA_KEY] is False
    finally:
        await close_engine()


async def test_disabled_config_seeds_nothing(agent_home, tmp_path):
    service, session_factory = await _make_bot_service(tmp_path, PersistentBotConfig(enabled=False))
    try:
        outcome = await service.ensure_seeded()
        assert outcome == {"enabled": False, "seeded": False}
        assert await PersistentBotRepository(session_factory).get() is None
        assert await ScheduledTaskRepository(session_factory).list_by_user("default") == []
        from deerflow.persistence.agents.file import FileAgentStore

        assert FileAgentStore().exists("persistent-bot", user_id="default") is False
    finally:
        await close_engine()


# -- arbitration ------------------------------------------------------------


async def _occurrence_statuses(session_factory, task_id: str) -> list[str]:
    async with session_factory() as session:
        rows = await session.execute(select(ScheduledTaskRunRow.status).where(ScheduledTaskRunRow.task_id == task_id).order_by(ScheduledTaskRunRow.created_at, ScheduledTaskRunRow.id))
        return list(rows.scalars())


async def _make_scheduler(tmp_path: Path, *, overlap_policy: str, launch, thread_busy_check=None, arbitrated: bool = True):
    """Real repos + scheduler over a temp DB with one due heartbeat-like task."""
    from app.scheduler.service import ScheduledTaskService

    await init_engine_from_config(DatabaseConfig(backend="sqlite", sqlite_dir=str(tmp_path / "db")))
    session_factory = get_session_factory()
    assert session_factory is not None
    thread_id = "bot-thread-1"
    await ThreadMetaRepository(session_factory).create(thread_id, assistant_id="persistent-bot", user_id="default")
    task_repo = ScheduledTaskRepository(session_factory)
    now = datetime.now(UTC)
    await task_repo.create(
        task_id=HEARTBEAT_TASK_ID,
        user_id="default",
        thread_id=thread_id,
        context_mode="reuse_thread",
        assistant_id="persistent-bot",
        title="Persistent bot heartbeat",
        prompt="heartbeat",
        schedule_type="interval",
        schedule_spec={"every_seconds": 300},
        timezone="UTC",
        next_run_at=now - timedelta(seconds=1),
        overlap_policy=overlap_policy,
    )
    service = ScheduledTaskService(
        task_repo=task_repo,
        task_run_repo=ScheduledTaskRunRepository(session_factory),
        launch_run=launch,
        poll_interval_seconds=1,
        lease_seconds=120,
        max_concurrent_runs=3,
        queue_timeout_seconds=86400,
        own_stop_available=False,
        thread_busy_check=thread_busy_check,
    )
    if arbitrated:
        # Mirrors the gateway seed wiring: busy arbitration applies only to
        # registered tasks (the heartbeat), never to ordinary tasks.
        service.configure_task_arbitration(HEARTBEAT_TASK_ID, stale_drop_seconds=900)
    return service, session_factory, now


async def test_busy_thread_skip_policy_drops_heartbeat_without_launch(tmp_path):
    calls: list[dict] = []

    async def launch(**kwargs):
        calls.append(kwargs)
        return {"run_id": "run-1", "thread_id": kwargs["thread_id"]}

    async def busy(_thread_id: str) -> bool:
        return True

    service, session_factory, now = await _make_scheduler(tmp_path, overlap_policy="skip", launch=launch, thread_busy_check=busy)
    try:
        await service.run_once(now=now)
        assert calls == []
        assert await _occurrence_statuses(session_factory, HEARTBEAT_TASK_ID) == ["skipped"]
    finally:
        await close_engine()


async def test_busy_thread_enqueue_policy_keeps_heartbeat_queued(tmp_path):
    calls: list[dict] = []

    async def launch(**kwargs):
        calls.append(kwargs)
        return {"run_id": "run-1", "thread_id": kwargs["thread_id"]}

    async def busy(_thread_id: str) -> bool:
        return True

    service, session_factory, now = await _make_scheduler(tmp_path, overlap_policy="enqueue", launch=launch, thread_busy_check=busy)
    try:
        await service.run_once(now=now)
        assert calls == []
        assert await _occurrence_statuses(session_factory, HEARTBEAT_TASK_ID) == ["queued"]
    finally:
        await close_engine()


@pytest.mark.parametrize(
    ("overlap_policy", "expected"),
    [("skip", ["skipped"]), ("enqueue", ["queued"])],
)
async def test_launch_conflict_yields_to_user_run(tmp_path, overlap_policy, expected):
    """Concurrent user turn holds the thread; the heartbeat launch 409s."""
    from deerflow.runtime import ConflictError

    calls: list[dict] = []

    async def launch(**kwargs):
        calls.append(kwargs)
        raise ConflictError("Thread bot-thread-1 already has an active run")

    service, session_factory, now = await _make_scheduler(tmp_path, overlap_policy=overlap_policy, launch=launch)
    try:
        await service.run_once(now=now)
        assert len(calls) == 1
        assert await _occurrence_statuses(session_factory, HEARTBEAT_TASK_ID) == expected
    finally:
        await close_engine()


async def test_unregistered_skip_task_keeps_historical_requeue_on_conflict(tmp_path):
    """Zero-change guard: without arbitration registration, a skip-policy
    task that 409s requeues exactly as before the persistent bot existed."""
    from deerflow.runtime import ConflictError

    calls: list[dict] = []

    async def launch(**kwargs):
        calls.append(kwargs)
        raise ConflictError("Thread bot-thread-1 already has an active run")

    service, session_factory, now = await _make_scheduler(tmp_path, overlap_policy="skip", launch=launch, arbitrated=False)
    try:
        await service.run_once(now=now)
        assert len(calls) == 1
        assert await _occurrence_statuses(session_factory, HEARTBEAT_TASK_ID) == ["queued"]
    finally:
        await close_engine()


async def test_stale_queued_heartbeat_is_dropped(tmp_path):
    calls: list[dict] = []

    async def launch(**kwargs):
        calls.append(kwargs)
        return {"run_id": "run-1", "thread_id": kwargs["thread_id"]}

    service, session_factory, now = await _make_scheduler(tmp_path, overlap_policy="enqueue", launch=launch)
    try:
        # Park the due task in the future so run_once only drains the queue.
        task_repo = ScheduledTaskRepository(session_factory)
        await task_repo.update(HEARTBEAT_TASK_ID, user_id="default", updates={"next_run_at": now + timedelta(hours=1)})
        await ScheduledTaskRunRepository(session_factory).create(
            run_record_id="task-run-stale",
            task_id=HEARTBEAT_TASK_ID,
            thread_id="bot-thread-1",
            scheduled_for=now - timedelta(hours=2),
            trigger="scheduled",
            status="queued",
        )
        service.configure_task_arbitration(HEARTBEAT_TASK_ID, stale_drop_seconds=900)
        await service.run_once(now=now)
        assert calls == []
        assert await _occurrence_statuses(session_factory, HEARTBEAT_TASK_ID) == ["skipped"]
    finally:
        await close_engine()


async def test_daily_budget_skips_due_heartbeat(tmp_path):
    calls: list[dict] = []

    async def launch(**kwargs):
        calls.append(kwargs)
        return {"run_id": "run-1", "thread_id": kwargs["thread_id"]}

    service, session_factory, now = await _make_scheduler(tmp_path, overlap_policy="skip", launch=launch)
    try:
        await ScheduledTaskRunRepository(session_factory).create(
            run_record_id="task-run-launched-today",
            task_id=HEARTBEAT_TASK_ID,
            thread_id="bot-thread-1",
            scheduled_for=now - timedelta(hours=1),
            trigger="scheduled",
            status="success",
        )
        service.configure_task_arbitration(HEARTBEAT_TASK_ID, max_runs_per_day=1)
        await service.run_once(now=now)
        assert calls == []
        assert await _occurrence_statuses(session_factory, HEARTBEAT_TASK_ID) == ["success"]
        task = await ScheduledTaskRepository(session_factory).get_internal(HEARTBEAT_TASK_ID)
        assert task["status"] == "enabled"
        assert task["next_run_at"] is not None
    finally:
        await close_engine()


# -- migration --------------------------------------------------------------


async def test_migration_0043_upgrade_downgrade_cycle(tmp_path: Path) -> None:
    from alembic import command as alembic_command
    from alembic.config import Config as AlembicConfig
    from sqlalchemy.ext.asyncio import create_async_engine

    from deerflow.persistence.bootstrap import _MIGRATIONS_DIR, _get_head_revision

    db_path = tmp_path / "persistent-bot-migration.db"
    url = f"sqlite+aiosqlite:///{db_path}"
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    engine = create_async_engine(url)
    try:
        await asyncio.to_thread(alembic_command.upgrade, cfg, "head")
        assert _get_head_revision() == "0044_scheduled_task_notify_policy"
        async with engine.connect() as conn:
            tables = await conn.run_sync(lambda sync: set(sa.inspect(sync).get_table_names()))
            columns = await conn.run_sync(lambda sync: {c["name"] for c in sa.inspect(sync).get_columns("persistent_bot")})
        assert "persistent_bot" in tables
        assert columns == {"id", "user_id", "agent_name", "thread_id", "heartbeat_task_id", "user_paused", "created_at", "updated_at"}

        await asyncio.to_thread(alembic_command.downgrade, cfg, "0042_cli_audit")
        async with engine.connect() as conn:
            tables = await conn.run_sync(lambda sync: set(sa.inspect(sync).get_table_names()))
        assert "persistent_bot" not in tables

        await asyncio.to_thread(alembic_command.upgrade, cfg, "head")
        async with engine.connect() as conn:
            tables = await conn.run_sync(lambda sync: set(sa.inspect(sync).get_table_names()))
        assert "persistent_bot" in tables
    finally:
        await engine.dispose()

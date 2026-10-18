"""Redacted proactive content egress for scheduled/bot runs.

Covers the per-task notify policy (none|summary|content) + quiet mode, the
deployment gate (``channel_connections.proactive_content_enabled``, off by
default), fail-closed bound-identity checks for run-output egress, and
no-duplicate retry through the durable outbox. Everything runs on temp
SQLite with a fake channel; no real IM calls.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from app.scheduler.notification_text import redact_egress_text
from deerflow.persistence.notification_deliveries import NotificationDeliveryRepository, NotificationDeliveryRow

SECRET = "sk-" + "S" * 40
ACCESS_TOKEN = "fake-token-000000000000"
REPLY = f"Weekly digest ready.\nDetails: used {SECRET} and access token: {ACCESS_TOKEN}\nSecond line of findings."
NOW = datetime(2026, 10, 18, 8, tzinfo=UTC)


class FakeChannel:
    def __init__(self, *, running: bool = True) -> None:
        self.sent: list[tuple[str, str]] = []
        self.is_running = running

    async def send_notification(self, *, target, text_markdown):
        self.sent.append((target, text_markdown))


@pytest.fixture
async def world(tmp_path):
    from app.scheduler.service import ScheduledTaskService
    from deerflow.persistence.channel_connections import ChannelConnectionRepository
    from deerflow.persistence.engine import close_engine, get_session_factory, init_engine
    from deerflow.persistence.scheduled_task_runs import ScheduledTaskRunRepository
    from deerflow.persistence.scheduled_tasks import ScheduledTaskRepository

    await init_engine("sqlite", url=f"sqlite+aiosqlite:///{tmp_path / 'egress.db'}", sqlite_dir=str(tmp_path))
    sf = get_session_factory()
    tasks, runs = ScheduledTaskRepository(sf), ScheduledTaskRunRepository(sf)
    deliveries = NotificationDeliveryRepository(sf)
    service = ScheduledTaskService(
        task_repo=tasks,
        task_run_repo=runs,
        launch_run=None,
        poll_interval_seconds=60,
        lease_seconds=30,
        max_concurrent_runs=3,
        connection_repo=ChannelConnectionRepository(sf),
        notification_repo=deliveries,
    )
    try:
        yield {"sf": sf, "tasks": tasks, "runs": runs, "deliveries": deliveries, "service": service}
    finally:
        await close_engine()


async def _bind(world, *, status: str = "connected") -> None:
    from deerflow.persistence.channel_connections.model import ChannelConnectionRow

    async with world["sf"]() as session:
        session.add(ChannelConnectionRow(id="binding", owner_user_id="alice", provider="wecom", status=status, external_account_id="alice-external"))
        await session.commit()


async def _finish_run(world, *, notify_policy: str = "summary", notify_quiet: bool = False) -> None:
    """Create a task + running occurrence and finalize it as success."""
    await world["tasks"].create(
        task_id="task-a",
        user_id="alice",
        thread_id=None,
        context_mode="fresh_thread_per_run",
        assistant_id=None,
        title="Digest",
        prompt="Summarize.",
        schedule_type="interval",
        schedule_spec={"every_seconds": 3600},
        timezone="UTC",
        next_run_at=NOW,
        notify_policy=notify_policy,
        notify_quiet=notify_quiet,
    )
    await world["runs"].create(run_record_id="occurrence-a", task_id="task-a", thread_id="thread-a", scheduled_for=NOW, trigger="scheduled", status="running")
    await world["runs"].update_status("occurrence-a", status="running", run_id="run-a", started_at=NOW)
    assert await world["tasks"].complete_run("task-a", user_id="alice", task_run_id="occurrence-a", run_id="run-a", status="success", error=None, finished_at=NOW) is True


async def _delivery_rows(world) -> list[NotificationDeliveryRow]:
    async with world["sf"]() as session:
        return list((await session.execute(select(NotificationDeliveryRow))).scalars())


def _worker(world, channel, *, content_enabled: bool = False, connections="bound"):
    from app.scheduler.notification_delivery import NotificationDeliveryWorker

    async def resolve_reply(run_id, user_id):
        return REPLY

    async def resolve_connections(owner_user_id):
        if connections == "bound":
            return [{"provider": "wecom", "external_account_id": "alice-external", "status": "connected", "owner_user_id": "alice"}]
        return []

    return NotificationDeliveryWorker(
        delivery_repo=world["deliveries"],
        resolve_channel=lambda provider: channel if provider == "wecom" else None,
        resolve_run_summary=resolve_reply,
        resolve_connections=None if connections is None else resolve_connections,
        content_egress_enabled=content_enabled,
    )


@pytest.mark.anyio
async def test_content_policy_delivers_redacted_run_output(world):
    await _bind(world)
    await _finish_run(world, notify_policy="content")
    channel = FakeChannel()
    await _worker(world, channel, content_enabled=True).run_once(now=datetime.now(UTC))

    assert len(channel.sent) == 1
    target, text = channel.sent[0]
    assert target == "alice-external"
    # The run output itself, multi-line, not the canned template.
    assert "Weekly digest ready." in text
    assert "Second line of findings." in text
    assert "Finished a run." not in text
    assert "Result:" not in text
    # Every secret shape is scrubbed before egress.
    assert SECRET not in text
    assert ACCESS_TOKEN not in text
    assert "[redacted]" in text
    (row,) = await _delivery_rows(world)
    assert row.status == "sent"
    assert row.payload_json["notify_policy"] == "content"


@pytest.mark.anyio
async def test_content_gate_off_falls_back_to_template(world):
    """Default deployment behavior is unchanged: content rows render the template."""
    await _bind(world)
    await _finish_run(world, notify_policy="content")
    channel = FakeChannel()
    await _worker(world, channel, content_enabled=False).run_once(now=datetime.now(UTC))

    assert len(channel.sent) == 1
    text = channel.sent[0][1]
    assert "Finished a run." in text
    assert "Result: Weekly digest ready." in text
    assert "Second line of findings." not in text
    assert SECRET not in text


@pytest.mark.anyio
async def test_summary_policy_uses_template_even_with_gate_on(world):
    await _bind(world)
    await _finish_run(world, notify_policy="summary")
    channel = FakeChannel()
    await _worker(world, channel, content_enabled=True).run_once(now=datetime.now(UTC))

    assert len(channel.sent) == 1
    text = channel.sent[0][1]
    assert "Finished a run." in text
    assert "Result: Weekly digest ready." in text
    assert "Second line of findings." not in text
    # Default-policy payloads keep the exact v2 shape (no notify keys).
    (row,) = await _delivery_rows(world)
    assert "notify_policy" not in row.payload_json
    assert "notify_quiet" not in row.payload_json


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("policy", "quiet"),
    [("none", False), ("content", True), ("summary", True)],
)
async def test_none_policy_and_quiet_mode_enqueue_nothing(world, policy, quiet):
    await _bind(world)
    await _finish_run(world, notify_policy=policy, notify_quiet=quiet)
    assert await _delivery_rows(world) == []
    channel = FakeChannel()
    await _worker(world, channel, content_enabled=True).run_once(now=datetime.now(UTC))
    assert channel.sent == []


@pytest.mark.anyio
async def test_unbound_target_blocks_content_send(world):
    await _bind(world)
    await _finish_run(world, notify_policy="content")
    channel = FakeChannel()
    # The owner disconnected the identity after enqueue.
    await _worker(world, channel, content_enabled=True, connections="unbound").run_once(now=datetime.now(UTC))

    assert channel.sent == []
    (row,) = await _delivery_rows(world)
    assert row.status == "failed"


@pytest.mark.anyio
async def test_content_without_connection_resolver_fails_closed(world):
    await _bind(world)
    await _finish_run(world, notify_policy="content")
    channel = FakeChannel()
    await _worker(world, channel, content_enabled=True, connections=None).run_once(now=datetime.now(UTC))

    assert channel.sent == []
    (row,) = await _delivery_rows(world)
    # Parked (retryable once a resolver exists), never sent unverified.
    assert row.status == "pending"
    assert row.parked_attempts == 1


@pytest.mark.anyio
async def test_channel_down_retry_sends_exactly_once(world):
    await _bind(world)
    await _finish_run(world, notify_policy="content")
    channel = FakeChannel(running=False)
    worker = _worker(world, channel, content_enabled=True)

    await worker.run_once(now=datetime.now(UTC))
    assert channel.sent == []
    (row,) = await _delivery_rows(world)
    assert row.status == "pending"

    channel.is_running = True
    await worker.run_once(now=datetime.now(UTC) + timedelta(seconds=901))
    assert len(channel.sent) == 1
    (row,) = await _delivery_rows(world)
    assert row.status == "sent"

    # A later poll re-claims nothing: the recorded send is not duplicated.
    await worker.run_once(now=datetime.now(UTC) + timedelta(seconds=1800))
    assert len(channel.sent) == 1
    async with world["sf"]() as session:
        assert await session.scalar(select(func.count()).select_from(NotificationDeliveryRow)) == 1


@pytest.mark.anyio
async def test_no_binding_enqueues_nothing(world):
    await _finish_run(world, notify_policy="content")
    assert await _delivery_rows(world) == []


@pytest.mark.anyio
async def test_task_repository_stores_and_validates_notify_fields(world):
    await _finish_run(world, notify_policy="content", notify_quiet=True)
    task = await world["tasks"].get("task-a", user_id="alice")
    assert task["notify_policy"] == "content"
    assert task["notify_quiet"] is True

    updated = await world["tasks"].update("task-a", user_id="alice", updates={"notify_policy": "none", "notify_quiet": False})
    assert updated["notify_policy"] == "none"
    assert updated["notify_quiet"] is False

    with pytest.raises(ValueError, match="notify_policy"):
        await world["tasks"].update("task-a", user_id="alice", updates={"notify_policy": "loud"})
    with pytest.raises(ValueError, match="notify_quiet"):
        await world["tasks"].create(
            task_id="task-b",
            user_id="alice",
            thread_id=None,
            context_mode="fresh_thread_per_run",
            assistant_id=None,
            title="Bad",
            prompt="x",
            schedule_type="interval",
            schedule_spec={"every_seconds": 3600},
            timezone="UTC",
            next_run_at=NOW,
            notify_quiet="yes",  # type: ignore[arg-type]
        )


def test_redact_egress_text_covers_long_form_variants():
    jwt = "eyJ" + "a" * 20 + "." + "b" * 12 + "." + "c" * 12
    samples = [
        f"access token: {ACCESS_TOKEN}",
        "Api Key = keyvalue123456",
        "client secret: shhh-value",
        "Authorization: Bearer abcdefgh12345678",
        f"session {jwt} end",
        "https://example.invalid/cb?token=urlsecret123&ok=1",
        "passphrase: correct-horse",
    ]
    for sample in samples:
        redacted = redact_egress_text(sample)
        assert "[redacted]" in redacted, sample
    assert ACCESS_TOKEN not in redact_egress_text(samples[0])
    assert "keyvalue123456" not in redact_egress_text(samples[1])
    assert "shhh-value" not in redact_egress_text(samples[2])
    assert "abcdefgh12345678" not in redact_egress_text(samples[3])
    assert jwt not in redact_egress_text(samples[4])
    assert "urlsecret123" not in redact_egress_text(samples[5])
    assert "ok=1" in redact_egress_text(samples[5])
    assert "correct-horse" not in redact_egress_text(samples[6])
    # Ordinary prose is untouched.
    assert redact_egress_text("the digest is ready for review") == "the digest is ready for review"


def test_redact_egress_text_covers_quoted_prefixed_and_header_forms():
    # Quoted JSON keys, prefixed/suffixed env names, and bare credential
    # headers are the same protected values under other spellings.
    cases = [
        ('{"api_key": "quotedsecret123"}', "quotedsecret123"),
        ('"access_token": "tok-abc-123xyz"', "tok-abc-123xyz"),
        ("Authorization: qwertyuiop123456", "qwertyuiop123456"),
        ("aws_secret_access_key=fakeawssecret123456", "fakeawssecret123456"),
        ("DB_PASSWORD=hunter2secret", "hunter2secret"),
        ("Cookie: sessionid=abc123def456", "abc123def456"),
        ("MY_API_TOKEN_VALUE=tokvalue98765", "tokvalue98765"),
    ]
    for sample, secret in cases:
        redacted = redact_egress_text(sample)
        assert secret not in redacted, sample
        assert "[redacted]" in redacted, sample
    # Keyword-looking prose without an assignment stays intact.
    assert redact_egress_text("the token count is 42 and secret santa is fine") == "the token count is 42 and secret santa is fine"


def test_bot_and_channel_config_defaults_keep_egress_off():
    from deerflow.config.channel_connections_config import ChannelConnectionsConfig
    from deerflow.config.persistent_bot_config import PersistentBotConfig

    assert ChannelConnectionsConfig().proactive_content_enabled is False
    bot = PersistentBotConfig()
    assert bot.notify_policy == "none"
    assert bot.notify_quiet is False
    with pytest.raises(ValueError):
        PersistentBotConfig(notify_policy="loud")


@pytest.mark.anyio
async def test_migration_0044_upgrade_downgrade_cycle(tmp_path):
    import asyncio

    import sqlalchemy as sa
    from alembic import command as alembic_command
    from alembic.config import Config as AlembicConfig
    from sqlalchemy.ext.asyncio import create_async_engine

    from deerflow.persistence.bootstrap import _MIGRATIONS_DIR, _get_head_revision

    url = f"sqlite+aiosqlite:///{tmp_path / 'notify-policy-migration.db'}"
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    engine = create_async_engine(url)
    try:
        await asyncio.to_thread(alembic_command.upgrade, cfg, "0043_persistent_bot")
        async with engine.connect() as conn:
            columns = await conn.run_sync(lambda sync: {c["name"] for c in sa.inspect(sync).get_columns("scheduled_tasks")})
        assert "notify_policy" not in columns and "notify_quiet" not in columns

        await asyncio.to_thread(alembic_command.upgrade, cfg, "head")
        assert _get_head_revision() == "0044_scheduled_task_notify_policy"
        async with engine.connect() as conn:
            columns = await conn.run_sync(lambda sync: {c["name"] for c in sa.inspect(sync).get_columns("scheduled_tasks")})
        assert {"notify_policy", "notify_quiet"} <= columns

        await asyncio.to_thread(alembic_command.downgrade, cfg, "0043_persistent_bot")
        async with engine.connect() as conn:
            columns = await conn.run_sync(lambda sync: {c["name"] for c in sa.inspect(sync).get_columns("scheduled_tasks")})
        assert "notify_policy" not in columns and "notify_quiet" not in columns

        await asyncio.to_thread(alembic_command.upgrade, cfg, "head")
        async with engine.connect() as conn:
            columns = await conn.run_sync(lambda sync: {c["name"] for c in sa.inspect(sync).get_columns("scheduled_tasks")})
        assert {"notify_policy", "notify_quiet"} <= columns
    finally:
        await engine.dispose()

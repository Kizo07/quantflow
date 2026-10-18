"""Idempotent seed and startup repair for the persistent bot.

When ``persistent_bot.enabled`` is true, :meth:`PersistentBotService.ensure_seeded`
guarantees, in order:

1. the bot custom agent exists with an explicit ``mcp_plugins=[]`` (the bot
   never silently inherits every installed MCP plugin) and a SOUL.md seed
   (an existing SOUL.md is never overwritten);
2. the DB-backed singleton row exists and points at a live, pinned,
   unarchived thread (a deleted thread is replaced);
3. exactly one heartbeat scheduled task exists, bound to that thread with
   ``context_mode="reuse_thread"`` and no goal/stop/run caps.

Repair never fights an explicit user off-switch: a singleton row with
``user_paused`` stops every repair, and a heartbeat task the user paused
stays paused. When the feature is disabled the service does nothing.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from deerflow.config.persistent_bot_config import PersistentBotConfig
from deerflow.persistence.persistent_bot.sql import PersistentBotRepository
from deerflow.persistence.scheduled_tasks.sql import ScheduledTaskRepository
from deerflow.persistence.thread_meta.base import (
    THREAD_ARCHIVED_METADATA_KEY,
    THREAD_PINNED_METADATA_KEY,
    ThreadMetaStore,
)
from deerflow.scheduler.host_notes import RUN_ERROR_PAUSED_WHILE_QUEUED
from deerflow.scheduler.schedules import next_run_at as compute_next_run_at

logger = logging.getLogger(__name__)

#: Stable scheduled-task id of the heartbeat. A fixed id makes re-seeding
#: naturally idempotent; a foreign collision falls back to a random id.
HEARTBEAT_TASK_ID = "persistent-bot-heartbeat"
HEARTBEAT_TASK_TITLE = "Persistent bot heartbeat"
BOT_THREAD_METADATA_KEY = "persistent_bot"

DEFAULT_SOUL = (
    "You are a persistent assistant that lives in this one conversation.\n"
    "Keep replies concise and concrete. On heartbeat check-ins, only raise "
    "items that genuinely need attention; when there is nothing to report, "
    "say so in one short line. Never invent tasks, deadlines, or facts that "
    "are not in this conversation."
)

_TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "cancelled"})


class PersistentBotService:
    def __init__(
        self,
        *,
        config: PersistentBotConfig,
        bot_repo: PersistentBotRepository,
        thread_store: ThreadMetaStore,
        task_repo: ScheduledTaskRepository,
        agent_store: Any | None = None,
    ) -> None:
        self._config = config
        self._bot_repo = bot_repo
        self._thread_store = thread_store
        self._task_repo = task_repo
        # Resolved lazily so a disabled service never touches agent storage.
        self._agent_store = agent_store
        # Set only for the duration of an explicit resume: a terminal
        # heartbeat is otherwise treated as the user off-switch.
        self._resume_revive = False

    async def ensure_seeded(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Seed or repair the bot; a no-op summary when disabled or paused."""
        config = self._config
        if not config.enabled:
            return {"enabled": False, "seeded": False}
        reference = now or datetime.now(UTC)
        bot = await self._bot_repo.get()
        if bot is not None and bot.get("user_paused"):
            logger.info("Persistent bot is paused by the user; skipping seed/repair")
            return {
                "enabled": True,
                "seeded": False,
                "user_paused": True,
                "thread_id": bot.get("thread_id"),
                "heartbeat_task_id": bot.get("heartbeat_task_id"),
                "agent_name": config.agent_name,
                "user_id": config.owner_user_id,
            }
        if bot is not None and bot.get("user_id") != config.owner_user_id:
            await self._retire_previous_identity(bot, now=reference)
        await asyncio.to_thread(self._ensure_agent_sync)
        thread_id = await self._ensure_thread(bot)
        bot = await self._bot_repo.get()
        heartbeat_task_id = await self._ensure_heartbeat(bot, thread_id, now=reference)
        return {
            "enabled": True,
            "seeded": True,
            "user_paused": False,
            "thread_id": thread_id,
            "heartbeat_task_id": heartbeat_task_id,
            "agent_name": config.agent_name,
            "user_id": config.owner_user_id,
        }

    async def _retire_previous_identity(self, bot: dict[str, Any], *, now: datetime) -> None:
        """Stop the previous owner's bot before seeding under a new owner.

        An owner change (e.g. the documented no-auth to auth migration)
        must never leave the old owner's heartbeat enabled on an orphaned
        thread: pause that task (cancelling any queued occurrence) and
        unpin/clear the bot marker on the old thread. Best-effort per step:
        a failure is logged, and the new owner's seed still proceeds.
        """
        old_owner = bot.get("user_id")
        if not isinstance(old_owner, str) or not old_owner:
            return
        logger.warning("Persistent bot owner changed %s -> %s; retiring previous owner's heartbeat and thread pin", old_owner, self._config.owner_user_id)
        task_id = bot.get("heartbeat_task_id") or HEARTBEAT_TASK_ID
        try:
            task = await self._task_repo.get_internal(task_id)
            if task is not None and task.get("user_id") == old_owner and task.get("status") not in _TERMINAL_TASK_STATUSES and task.get("status") != "paused":
                await self._task_repo.pause_with_queue_cancellation(task_id, user_id=old_owner, error=RUN_ERROR_PAUSED_WHILE_QUEUED, now=now)
        except Exception:
            logger.warning("Failed to pause previous owner's heartbeat %s", task_id, exc_info=True)
        old_thread = bot.get("thread_id")
        if isinstance(old_thread, str) and old_thread:
            try:
                await self._thread_store.update_metadata(
                    old_thread,
                    {THREAD_PINNED_METADATA_KEY: False, BOT_THREAD_METADATA_KEY: False},
                    touch=False,
                    user_id=old_owner,
                )
            except Exception:
                logger.warning("Failed to unpin previous owner's bot thread %s", old_thread, exc_info=True)

    async def set_user_paused(self, paused: bool, *, now: datetime | None = None) -> dict[str, Any] | None:
        """Flip the explicit user off-switch.

        Pausing also pauses the heartbeat task (cancelling a queued
        occurrence); unpausing clears the flag and re-runs the seed, which
        resumes a heartbeat task that this pause stopped.
        """
        reference = now or datetime.now(UTC)
        bot = await self._bot_repo.get()
        if bot is None:
            return None
        updated = await self._bot_repo.set_user_paused(paused)
        task_id = bot.get("heartbeat_task_id")
        if isinstance(task_id, str) and task_id:
            if paused:
                await self._task_repo.pause_with_queue_cancellation(
                    task_id,
                    user_id=self._config.owner_user_id,
                    error=RUN_ERROR_PAUSED_WHILE_QUEUED,
                    now=reference,
                )
            else:
                existing = await self._task_repo.get(task_id, user_id=self._config.owner_user_id)
                if existing is not None and existing.get("status") == "paused":
                    await self._task_repo.update(
                        task_id,
                        user_id=self._config.owner_user_id,
                        updates={
                            "status": "enabled",
                            "next_run_at": compute_next_run_at(
                                existing["schedule_type"],
                                existing["schedule_spec"],
                                existing["timezone"],
                                now=reference,
                            ),
                        },
                    )
        if not paused and self._config.enabled:
            self._resume_revive = True
            try:
                await self.ensure_seeded(now=reference)
            finally:
                self._resume_revive = False
            return await self._bot_repo.get()
        return updated

    # -- agent ------------------------------------------------------------

    def _resolved_agent_store(self) -> Any:
        if self._agent_store is None:
            from deerflow.persistence.agents import get_agent_store

            self._agent_store = get_agent_store()
        return self._agent_store

    def _agent_document(self) -> dict[str, Any]:
        config = self._config
        document: dict[str, Any] = {
            "name": config.agent_name,
            "description": "Always-on assistant living in one pinned conversation.",
            # Explicit empty selection: None would inherit every plugin.
            "mcp_plugins": [],
        }
        if config.model is not None:
            document["model"] = config.model
        if config.tool_groups is not None:
            document["tool_groups"] = list(config.tool_groups)
        if config.skills is not None:
            document["skills"] = list(config.skills)
        if config.allowed_subagents is not None:
            document["allowed_subagents"] = list(config.allowed_subagents)
        return document

    def _ensure_agent_sync(self) -> None:
        from deerflow.persistence.agents.base import AgentExistsError

        config = self._config
        store = self._resolved_agent_store()
        name = config.agent_name
        owner = config.owner_user_id
        soul = config.soul_text if config.soul_text is not None else DEFAULT_SOUL
        try:
            existing = store.get(name, user_id=owner)
        except FileNotFoundError:
            existing = None
        if existing is None:
            try:
                store.create(name, self._agent_document(), soul, user_id=owner)
                return
            except AgentExistsError:
                existing = store.get(name, user_id=owner)
        # Repair: the bot must keep an explicit empty MCP plugin selection,
        # and config-pinned fields (model, tool groups, skills, allowed
        # subagents) re-apply on every startup — ``persistent_bot`` config
        # is the source of truth for the bot agent, not create-only.
        desired = self._agent_document()
        drifted = existing.mcp_plugins != []
        for field in ("model", "tool_groups", "skills", "allowed_subagents"):
            if field in desired and getattr(existing, field, None) != desired[field]:
                drifted = True
        if drifted:
            merged = existing.model_dump(exclude_unset=True)
            merged.update(desired)
            merged["name"] = name
            store.update(name, merged, None, user_id=owner)
        if store.get_soul(name, user_id=owner) is None:
            store.update(name, None, soul, user_id=owner)

    # -- thread -----------------------------------------------------------

    def _thread_metadata(self) -> dict[str, Any]:
        return {
            THREAD_PINNED_METADATA_KEY: True,
            THREAD_ARCHIVED_METADATA_KEY: False,
            BOT_THREAD_METADATA_KEY: True,
        }

    async def _repair_thread_metadata(self, thread: dict[str, Any]) -> None:
        metadata = thread.get("metadata") or {}
        desired = self._thread_metadata()
        if all(metadata.get(key) == value for key, value in desired.items()):
            return
        await self._thread_store.update_metadata(
            thread["thread_id"],
            desired,
            touch=False,
            user_id=self._config.owner_user_id,
        )

    async def _create_thread(self) -> str:
        config = self._config
        thread_id = str(uuid.uuid4())
        await self._thread_store.create(
            thread_id,
            assistant_id=config.agent_name,
            user_id=config.owner_user_id,
            display_name=config.thread_title,
            metadata=self._thread_metadata(),
        )
        return thread_id

    async def _ensure_thread(self, bot: dict[str, Any] | None) -> str:
        config = self._config
        if bot is not None:
            recorded = bot.get("thread_id")
            if isinstance(recorded, str) and recorded:
                existing = await self._thread_store.get(recorded, user_id=config.owner_user_id)
                if existing is not None:
                    await self._repair_thread_metadata(existing)
                    if bot.get("user_id") != config.owner_user_id or bot.get("agent_name") != config.agent_name:
                        await self._bot_repo.set_identity(user_id=config.owner_user_id, agent_name=config.agent_name)
                    return recorded
            # Recorded thread was deleted (or is no longer owned here):
            # replace it and repoint the singleton.
            replacement = await self._create_thread()
            await self._bot_repo.set_thread(replacement)
            if bot.get("user_id") != config.owner_user_id or bot.get("agent_name") != config.agent_name:
                await self._bot_repo.set_identity(user_id=config.owner_user_id, agent_name=config.agent_name)
            logger.info("Persistent bot thread replaced: %s", replacement)
            return replacement

        candidate = await self._create_thread()
        row, created = await self._bot_repo.ensure(
            user_id=config.owner_user_id,
            agent_name=config.agent_name,
            thread_id=candidate,
        )
        winner = row["thread_id"]
        if created or winner == candidate:
            return candidate
        # A concurrent seed won the singleton: drop our orphan thread and
        # adopt (and repair) the winner's.
        try:
            await self._thread_store.delete(candidate, user_id=config.owner_user_id)
        except Exception:
            logger.warning("Failed to remove orphaned persistent-bot thread %s", candidate, exc_info=True)
        existing = await self._thread_store.get(winner, user_id=config.owner_user_id)
        if existing is not None:
            await self._repair_thread_metadata(existing)
            return winner
        replacement = await self._create_thread()
        await self._bot_repo.set_thread(replacement)
        return replacement

    # -- heartbeat --------------------------------------------------------

    async def _find_heartbeat(self, bot: dict[str, Any] | None) -> dict[str, Any] | None:
        owner = self._config.owner_user_id
        recorded = bot.get("heartbeat_task_id") if bot else None
        for candidate_id in (recorded, HEARTBEAT_TASK_ID):
            if not isinstance(candidate_id, str) or not candidate_id:
                continue
            task = await self._task_repo.get_internal(candidate_id)
            if task is not None and task.get("user_id") == owner:
                return task
        return None

    async def _ensure_heartbeat(self, bot: dict[str, Any] | None, thread_id: str, *, now: datetime) -> str:
        config = self._config
        owner = config.owner_user_id
        schedule_spec = {"every_seconds": config.heartbeat_interval_seconds}
        existing = await self._find_heartbeat(bot)
        if existing is None:
            stable_taken = await self._task_repo.get_internal(HEARTBEAT_TASK_ID) is not None
            task_id = f"task-{uuid.uuid4().hex}" if stable_taken else HEARTBEAT_TASK_ID
            try:
                await self._task_repo.create(
                    task_id=task_id,
                    user_id=owner,
                    thread_id=thread_id,
                    context_mode="reuse_thread",
                    assistant_id=config.agent_name,
                    title=HEARTBEAT_TASK_TITLE,
                    prompt=config.heartbeat_prompt,
                    schedule_type="interval",
                    schedule_spec=schedule_spec,
                    timezone=config.heartbeat_timezone,
                    next_run_at=compute_next_run_at("interval", schedule_spec, config.heartbeat_timezone, now=now),
                    goal_objective=None,
                    stop_condition=None,
                    max_runs=None,
                    end_at=None,
                    overlap_policy=config.overlap_policy,
                    notify_policy=config.notify_policy,
                    notify_quiet=config.notify_quiet,
                )
            except Exception:
                # A concurrent seed created it first; adopt that row.
                adopted = await self._find_heartbeat(await self._bot_repo.get()) or await self._task_repo.get_internal(task_id)
                if adopted is None or adopted.get("user_id") != owner:
                    raise
                existing = adopted
                task_id = adopted["id"]
            if existing is None:
                await self._bot_repo.set_heartbeat_task(task_id)
                return task_id

        task_id = existing["id"]
        if bot is None or bot.get("heartbeat_task_id") != task_id:
            await self._bot_repo.set_heartbeat_task(task_id)

        desired: dict[str, Any] = {
            "thread_id": thread_id,
            "context_mode": "reuse_thread",
            "assistant_id": config.agent_name,
            "title": HEARTBEAT_TASK_TITLE,
            "prompt": config.heartbeat_prompt,
            "goal_objective": None,
            "stop_condition": None,
            "max_runs": None,
            "end_at": None,
            "overlap_policy": config.overlap_policy,
            "notify_policy": config.notify_policy,
            "notify_quiet": config.notify_quiet,
            "schedule_type": "interval",
            "schedule_spec": schedule_spec,
            "timezone": config.heartbeat_timezone,
        }
        updates = {key: value for key, value in desired.items() if existing.get(key) != value}
        if existing.get("status") == "paused":
            # Respect the pause: keep the binding repairable for a later
            # resume, but never re-enable or reschedule behind the user.
            updates = {key: updates[key] for key in ("thread_id", "context_mode", "assistant_id", "notify_policy", "notify_quiet") if key in updates}
        elif existing.get("status") in _TERMINAL_TASK_STATUSES:
            # A cancelled/completed heartbeat is a user decision to stop the
            # bot, not drift to repair: record the off-switch and leave the
            # task terminal. Resuming goes through set_user_paused(False),
            # which re-runs the seed — and only then is the task revived
            # (the bot row is unpaused by that point; ensure_seeded returns
            # early while paused, so reaching here unpaused after an
            # explicit resume is the one revival path).
            if bot is not None and not bot.get("user_paused") and not self._resume_revive:
                logger.warning("Persistent bot heartbeat %s is %s; treating as user off-switch (user_paused)", task_id, existing.get("status"))
                await self._bot_repo.set_user_paused(True)
                return task_id
            updates["status"] = "enabled"
            updates["next_run_at"] = compute_next_run_at("interval", schedule_spec, config.heartbeat_timezone, now=now)
        else:
            schedule_changed = bool({"schedule_type", "schedule_spec", "timezone"} & updates.keys())
            if schedule_changed or existing.get("next_run_at") is None:
                updates["next_run_at"] = compute_next_run_at("interval", schedule_spec, config.heartbeat_timezone, now=now)
        if updates:
            await self._task_repo.update(task_id, user_id=owner, updates=updates)
        return task_id

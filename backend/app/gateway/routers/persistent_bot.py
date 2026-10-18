"""Persistent-bot status and off-switch API (``/api/persistent-bot``).

The bot is seeded at gateway startup from ``persistent_bot`` config; this
router is the runtime surface over it. ``pause`` is the user's off-switch:
it records ``user_paused`` and pauses the heartbeat task, and it stays
reachable when the feature is later disabled in config so a leftover
heartbeat can always be stopped. ``resume`` requires the feature to be
enabled (a resumed bot needs its seed/arbitration wiring).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.gateway.authz import require_permission
from app.gateway.deps import (
    get_config,
    get_optional_user_from_request,
    get_persistent_bot_repo,
    get_scheduled_task_repo,
    get_thread_store,
)

router = APIRouter(prefix="/api", tags=["persistent-bot"])


def _service(request: Request):
    """The startup-seeded service, or one built on demand from app repos.

    Building on demand keeps pause reachable when ``persistent_bot`` was
    disabled after seeding (no startup service exists in that case).
    """
    service = getattr(request.app.state, "persistent_bot_service", None)
    if service is not None:
        return service
    bot_repo = get_persistent_bot_repo(request)
    if bot_repo is None:
        return None
    from deerflow.persistent_bot.service import PersistentBotService

    return PersistentBotService(
        config=get_config().persistent_bot,
        bot_repo=bot_repo,
        thread_store=get_thread_store(request),
        task_repo=get_scheduled_task_repo(request),
    )


async def _bot_and_caller(request: Request) -> tuple[dict[str, Any] | None, Any]:
    user = await get_optional_user_from_request(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    bot_repo = get_persistent_bot_repo(request)
    bot = await bot_repo.get() if bot_repo is not None else None
    return bot, user


def _ensure_owner(bot: dict[str, Any] | None, user: Any) -> None:
    owner = (bot or {}).get("user_id") or get_config().persistent_bot.owner_user_id
    if str(user.id) != str(owner):
        raise HTTPException(status_code=403, detail="Only the persistent bot's owner can manage it")


async def _status(request: Request, bot: dict[str, Any] | None) -> dict[str, Any]:
    config = get_config().persistent_bot
    heartbeat_status = None
    if bot is not None and bot.get("heartbeat_task_id"):
        task = await get_scheduled_task_repo(request).get_internal(str(bot["heartbeat_task_id"]))
        heartbeat_status = (task or {}).get("status")
    return {
        "enabled": config.enabled,
        "seeded": bot is not None,
        "user_paused": bool((bot or {}).get("user_paused")),
        "agent_name": (bot or {}).get("agent_name") or config.agent_name,
        "thread_id": (bot or {}).get("thread_id"),
        "heartbeat_task_id": (bot or {}).get("heartbeat_task_id"),
        "heartbeat_status": heartbeat_status,
    }


@router.get("/persistent-bot")
@require_permission("threads", "read")
async def get_persistent_bot(request: Request):
    bot, user = await _bot_and_caller(request)
    _ensure_owner(bot, user)
    return await _status(request, bot)


@router.post("/persistent-bot/pause")
@require_permission("threads", "write")
async def pause_persistent_bot(request: Request):
    bot, user = await _bot_and_caller(request)
    _ensure_owner(bot, user)
    if bot is None:
        raise HTTPException(status_code=404, detail="Persistent bot has not been seeded")
    service = _service(request)
    if service is None:
        raise HTTPException(status_code=503, detail="Persistent bot storage not available")
    await service.set_user_paused(True)
    return await _status(request, await get_persistent_bot_repo(request).get())


@router.post("/persistent-bot/resume")
@require_permission("threads", "write")
@require_permission("runs", "create")
async def resume_persistent_bot(request: Request):
    bot, user = await _bot_and_caller(request)
    _ensure_owner(bot, user)
    if bot is None:
        raise HTTPException(status_code=404, detail="Persistent bot has not been seeded")
    if not get_config().persistent_bot.enabled:
        raise HTTPException(status_code=409, detail="persistent_bot.enabled is false; enable it and restart the gateway to resume the bot")
    service = _service(request)
    if service is None:
        raise HTTPException(status_code=503, detail="Persistent bot storage not available")
    await service.set_user_paused(False)
    return await _status(request, await get_persistent_bot_repo(request).get())

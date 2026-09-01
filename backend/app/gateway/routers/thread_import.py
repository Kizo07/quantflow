"""Session import endpoint (Tier 1 Phase C).

Rebuilds an exported session — the JSON shape produced by the frontend's
``formatThreadAsJSON`` — as a new viewable thread. Import content is
**untrusted data**: only ``human``/``ai`` transcript rows with non-blank
text content survive; tool messages, tool calls, reasoning payloads, and
original message ids are dropped so an export file cannot re-inject tool
invocations or forged provenance into the runtime.

Message injection reuses the spike-proven state-update path
(``build_thread_checkpoint_state_mutation_accessor`` + ``Overwrite``),
the same mechanism ``POST /api/threads/{id}/state`` uses.

Route ordering: this router must be included **before** ``threads.router``
so ``POST /api/threads/import`` matches before the ``/{thread_id}`` routes.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.types import Overwrite
from pydantic import BaseModel, Field

from app.gateway.authz import require_permission
from app.gateway.deps import get_checkpointer, get_thread_store
from app.gateway.services import (
    build_thread_checkpoint_state_mutation_accessor,
    reserve_checkpoint_write,
)
from app.gateway.utils import sanitize_log_param
from deerflow.persistence.thread_meta import THREAD_IMPORTED_METADATA_KEY
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.utils.thread_id import resolve_thread_id
from deerflow.utils.time import now_iso

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/threads", tags=["threads"])

# Hard limits: an import file is user-supplied and may be hostile.
_MAX_IMPORT_BODY_BYTES = 10 * 1024 * 1024  # 10 MiB
_MAX_IMPORT_MESSAGES = 2000
_MAX_IMPORT_MESSAGE_CHARS = 500_000
_MAX_IMPORT_TITLE_CHARS = 256

_IMPORTABLE_TYPES = frozenset({"human", "ai"})


class ThreadImportResponse(BaseModel):
    """Response model for a session import."""

    thread_id: str = Field(description="New thread created from the import")
    status: str = Field(default="idle")
    created_at: str = Field(default="")
    updated_at: str = Field(default="")
    metadata: dict = Field(default_factory=dict)
    imported_message_count: int = Field(default=0)


def _sanitize_import_messages(raw: object) -> list[dict[str, str]]:
    """Reduce untrusted export rows to safe human/ai transcript dicts.

    Drops tool messages, tool calls, reasoning, original ids, and rows
    without non-blank text content. Raises 413 for rows over the per-message
    size cap (a single giant row is a payload problem, not a skip case).
    """
    if not isinstance(raw, list):
        raise HTTPException(status_code=400, detail="'messages' must be a list")
    if len(raw) > _MAX_IMPORT_MESSAGES:
        raise HTTPException(
            status_code=413,
            detail=f"Import exceeds the {_MAX_IMPORT_MESSAGES} message limit",
        )

    cleaned: list[dict[str, str]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        msg_type = row.get("type")
        if msg_type not in _IMPORTABLE_TYPES:
            continue
        content = row.get("content")
        if not isinstance(content, str):
            continue
        stripped = content.strip()
        if not stripped:
            continue
        if len(stripped) > _MAX_IMPORT_MESSAGE_CHARS:
            raise HTTPException(
                status_code=413,
                detail="Import contains a message over the per-message size limit",
            )
        # Deliberately no tool_calls / id / reasoning: untrusted data must
        # not re-enter the runtime as tool invocations or forged identity.
        cleaned.append({"type": msg_type, "content": stripped})
    return cleaned


@router.post("/import", response_model=ThreadImportResponse)
@require_permission("threads", "write")
async def import_thread(request: Request) -> ThreadImportResponse:
    """Import an exported session JSON as a new thread."""
    body = await request.body()
    if len(body) > _MAX_IMPORT_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Import payload too large")
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="Import payload is not valid JSON") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Import payload must be a JSON object")
    if "messages" not in payload:
        raise HTTPException(status_code=400, detail="Import payload is missing the 'messages' field")

    messages = _sanitize_import_messages(payload["messages"])
    if not messages:
        raise HTTPException(status_code=400, detail="Import contains no importable messages")

    raw_title = payload.get("title")
    title = raw_title.strip()[:_MAX_IMPORT_TITLE_CHARS] if isinstance(raw_title, str) else ""

    thread_store = get_thread_store(request)
    checkpointer = get_checkpointer(request)
    thread_id = resolve_thread_id(None)
    now = now_iso()
    metadata = {THREAD_IMPORTED_METADATA_KEY: True}

    try:
        await thread_store.create(
            thread_id,
            metadata=metadata,
        )
    except Exception:
        logger.exception("Failed to write thread_meta for import %s", sanitize_log_param(thread_id))
        raise HTTPException(status_code=500, detail="Failed to create thread for import")

    # Empty checkpoint so the state-mutation accessor has a base to write on
    # (mirrors create_thread in the threads router).
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    try:
        ckpt_metadata = {
            "step": -1,
            "source": "input",
            "writes": None,
            "parents": {},
            **metadata,
            "created_at": now,
        }
        await checkpointer.aput(config, empty_checkpoint(), ckpt_metadata, {})
    except Exception:
        logger.exception("Failed to create checkpoint for import %s", sanitize_log_param(thread_id))
        await _cleanup_failed_import(thread_store, thread_id)
        raise HTTPException(status_code=500, detail="Failed to create thread for import")

    # Inject the transcript through the spike-proven state-update path.
    try:
        accessor, read_config = await build_thread_checkpoint_state_mutation_accessor(
            request,
            thread_id=thread_id,
            as_node="session_import",
        )
        async with reserve_checkpoint_write(request, thread_id, user_id=get_effective_user_id()):
            await accessor.aupdate(
                read_config,
                {"messages": Overwrite(messages)},
                as_node="session_import",
            )
    except HTTPException:
        await _cleanup_failed_import(thread_store, thread_id)
        raise
    except Exception:
        logger.exception("Failed to inject messages for import %s", sanitize_log_param(thread_id))
        await _cleanup_failed_import(thread_store, thread_id)
        raise HTTPException(status_code=500, detail="Failed to import messages")

    if title and thread_store is not None:
        try:
            await thread_store.update_display_name(thread_id, title)
        except Exception:
            logger.debug("Failed to set import title for %s (non-fatal)", sanitize_log_param(thread_id))

    logger.info("Thread imported: %s (%d messages)", sanitize_log_param(thread_id), len(messages))
    return ThreadImportResponse(
        thread_id=thread_id,
        status="idle",
        created_at=now,
        updated_at=now,
        metadata=metadata,
        imported_message_count=len(messages),
    )


async def _cleanup_failed_import(thread_store, thread_id: str) -> None:
    """Best-effort removal of the meta record when injection fails."""
    try:
        await thread_store.delete(thread_id)
    except Exception:
        logger.debug("Failed to clean up thread_meta for %s (non-fatal)", sanitize_log_param(thread_id))

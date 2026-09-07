"""Tests for POST /api/threads/import — session import (Tier 1 Phase C).

The endpoint accepts the JSON schema produced by the frontend's
``formatThreadAsJSON`` export and rebuilds it as a viewable thread via the
spike-proven state-update injection path. Import content is untrusted data:
tool messages and tool_calls are dropped, only human/ai transcript rows with
text content survive, and the payload is size-capped.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import yaml
from _router_auth_helpers import make_authed_test_app
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.gateway.auth.models import User
from app.gateway.routers import thread_import, threads
from deerflow.config.app_config import reset_app_config
from deerflow.persistence.thread_meta.memory import THREADS_NS, MemoryThreadMetaStore


@pytest.fixture(autouse=True)
def _config_env(tmp_path, monkeypatch):
    """State-mutation routes read AppConfig on every request via
    ``deps.get_run_context`` and 503 when no config.yaml exists (CI has none,
    the file is gitignored). Point the loader at a minimal test config."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "sandbox": {"use": "deerflow.sandbox.local:LocalSandboxProvider"},
                "models": [{"name": "test-model", "use": "langchain_openai:ChatOpenAI", "model": "gpt-test"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEER_FLOW_CONFIG_PATH", str(config_path))
    reset_app_config()
    yield
    reset_app_config()


# Stable identity across requests: the default stub factory mints a fresh
# UUID per request, which breaks owner-scoped reads (search after import).
_STUB_USER = User(
    email="import-test@example.com",
    password_hash="x",
    system_role="user",
    id=uuid4(),
)


def _stable_user() -> User:
    return _STUB_USER


class _PermissiveThreadMetaStore(MemoryThreadMetaStore):
    async def _get_owned_record(self, thread_id, user_id, method_name):  # type: ignore[override]
        item = await self._store.aget(THREADS_NS, thread_id)
        return dict(item.value) if item is not None else None

    async def check_access(self, thread_id, user_id, *, require_existing=False):  # type: ignore[override]
        item = await self._store.aget(THREADS_NS, thread_id)
        if item is None:
            return not require_existing
        return True


class _ThreadTestRunManager:
    async def list_by_thread(self, _thread_id: str, *, user_id=None, limit: int = 100) -> list:
        return []

    @asynccontextmanager
    async def reserve_thread_operation(self, _thread_id: str, **_kwargs):
        yield


def _make_import_app() -> FastAPI:
    app = make_authed_test_app(user_factory=_stable_user)
    store = InMemoryStore()
    app.state.store = store
    app.state.checkpointer = InMemorySaver()
    app.state.run_manager = _ThreadTestRunManager()
    app.state.run_event_store = SimpleNamespace(find_latest_ai_message_run_ids=AsyncMock(return_value={}))
    app.state.thread_store = _PermissiveThreadMetaStore(store)
    # Import router first: /import must match before /{thread_id} routes.
    app.include_router(thread_import.router)
    app.include_router(threads.router)
    return app


_EXPORT = {
    "title": "Imported research chat",
    "thread_id": "old-thread-id-ignored",
    "created_at": "2026-08-30T10:00:00Z",
    "exported_at": "2026-08-31T10:00:00Z",
    "messages": [
        {"type": "human", "id": "h1", "content": "What is the Sharpe ratio?"},
        {"type": "ai", "id": "a1", "content": "A risk-adjusted return measure."},
        {"type": "tool", "id": "t1", "content": "tool output must not be imported"},
        {"type": "ai", "id": "a2", "content": "", "tool_calls": [{"name": "dangerous"}]},
        {"type": "human", "id": "h2", "content": "   "},
    ],
}


def test_import_creates_thread_with_sanitized_transcript() -> None:
    with TestClient(_make_import_app()) as client:
        response = client.post("/api/threads/import", json=_EXPORT)
        assert response.status_code == 200, response.text
        body = response.json()
        thread_id = body["thread_id"]
        assert thread_id != "old-thread-id-ignored"
        assert body["metadata"].get("deerflow_imported") is True

        state = client.get(f"/api/threads/{thread_id}/state")
        assert state.status_code == 200, state.text
        got = state.json()["values"]["messages"]
        # tool message dropped, tool-call-only AI row dropped, blank row dropped
        assert [m["type"] for m in got] == ["human", "ai"]
        assert got[0]["content"] == "What is the Sharpe ratio?"
        assert got[1]["content"] == "A risk-adjusted return measure."
        # no tool_calls survive an import
        for m in got:
            assert not m.get("tool_calls")


def test_import_sets_display_name_from_title() -> None:
    with TestClient(_make_import_app()) as client:
        response = client.post("/api/threads/import", json=_EXPORT)
        assert response.status_code == 200, response.text
        thread_id = response.json()["thread_id"]
        search = client.post("/api/threads/search", json={})
        assert search.status_code == 200, search.text
        rows = search.json()  # list[ThreadResponse]
        match = [r for r in rows if r.get("thread_id") == thread_id]
        assert match, rows
        assert match[0].get("values", {}).get("title") == "Imported research chat"


def test_import_rejects_missing_messages_field() -> None:
    with TestClient(_make_import_app()) as client:
        response = client.post("/api/threads/import", json={"title": "no messages"})
        assert response.status_code == 400, response.text


def test_import_rejects_invalid_json() -> None:
    with TestClient(_make_import_app()) as client:
        response = client.post(
            "/api/threads/import",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400, response.text


def test_import_rejects_too_many_messages() -> None:
    payload = {
        "title": "oversized",
        "messages": [{"type": "human", "content": f"m{i}"} for i in range(2001)],
    }
    with TestClient(_make_import_app()) as client:
        response = client.post("/api/threads/import", json=payload)
        assert response.status_code == 413, response.text


def test_import_rejects_empty_transcript() -> None:
    with TestClient(_make_import_app()) as client:
        response = client.post(
            "/api/threads/import",
            json={"title": "empty", "messages": [{"type": "tool", "content": "x"}]},
        )
        assert response.status_code == 400, response.text


def test_import_requires_authentication() -> None:
    # Bare app without the stub auth middleware: @require_permission must
    # reject unauthenticated callers.
    app = FastAPI()
    store = InMemoryStore()
    app.state.store = store
    app.state.checkpointer = InMemorySaver()
    app.state.run_manager = _ThreadTestRunManager()
    app.state.run_event_store = SimpleNamespace(find_latest_ai_message_run_ids=AsyncMock(return_value={}))
    app.state.thread_store = _PermissiveThreadMetaStore(store)
    app.include_router(thread_import.router)
    with TestClient(app) as client:
        response = client.post("/api/threads/import", json=_EXPORT)
        assert response.status_code in (401, 403), response.text

"""C1 spike: can imported messages be injected into a fresh thread via POST /state?"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from _router_auth_helpers import make_authed_test_app
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.gateway.routers import threads
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


class _PermissiveThreadMetaStore(MemoryThreadMetaStore):
    async def _get_owned_record(self, thread_id, user_id, method_name):  # type: ignore[override]
        item = await self._store.aget(THREADS_NS, thread_id)
        return dict(item.value) if item is not None else None

    async def check_access(self, thread_id, user_id, *, require_existing=False):  # type: ignore[override]
        item = await self._store.aget(THREADS_NS, thread_id)
        if item is None:
            return not require_existing
        return True

    async def create(self, thread_id, *, assistant_id=None, user_id=None, display_name=None, metadata=None):  # type: ignore[override]
        return await super().create(thread_id, assistant_id=assistant_id, user_id=None, display_name=display_name, metadata=metadata)


class _ThreadTestRunManager:
    async def list_by_thread(self, _thread_id: str, *, user_id=None, limit: int = 100) -> list:
        return []

    @asynccontextmanager
    async def reserve_thread_operation(self, _thread_id: str, **_kwargs):
        yield


def test_import_messages_via_state_update() -> None:
    app = make_authed_test_app()
    store = InMemoryStore()
    checkpointer = InMemorySaver()
    app.state.store = store
    app.state.checkpointer = checkpointer
    app.state.run_manager = _ThreadTestRunManager()
    app.state.run_event_store = SimpleNamespace(find_latest_ai_message_run_ids=AsyncMock(return_value={}))
    app.state.thread_store = _PermissiveThreadMetaStore(store)
    app.include_router(threads.router)

    with TestClient(app) as client:
        created = client.post("/api/threads", json={"metadata": {}})
        assert created.status_code == 200, created.text
        thread_id = created.json()["thread_id"]

        messages = [
            {"type": "human", "content": "What is the Sharpe ratio?"},
            {"type": "ai", "content": "Risk-adjusted return measure."},
        ]

        response = client.post(
            f"/api/threads/{thread_id}/state",
            json={"values": {"messages": messages}},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        got = body["values"]["messages"]
        assert len(got) == 2, got
        assert got[0]["type"] == "human"
        assert got[1]["type"] == "ai"

        # Read it back through the GET path to confirm persistence.
        state = client.get(f"/api/threads/{thread_id}/state")
        assert state.status_code == 200, state.text
        persisted = state.json()["values"]["messages"]
        assert [m["type"] for m in persisted] == ["human", "ai"]

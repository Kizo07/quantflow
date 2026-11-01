"""Unit tests for deerflow.evolution.memory_adapter (offline).

A fake agent memory stands in for the real manager; the adapter never
imports the manager contract.
"""

from typing import Any

import pytest

from deerflow.evolution.memory_adapter import MemoryAdapter


class FakeMemory:
    """AgentMemory fake with scripted context, hits, and a write log."""

    def __init__(self, context: str = "", hits: list[dict[str, Any]] | None = None, searchable: bool = True):
        self.context = context
        self.hits = hits if hits is not None else []
        self.searchable = searchable
        self.queries: list[str] = []
        self.writes: list[tuple[str, list[Any]]] = []

    def get_context(self, user_id, *, agent_name=None, thread_id=None, query=None) -> str:
        _ = (user_id, agent_name, thread_id)
        self.queries.append(query or "")
        return self.context

    def search(self, query, top_k=5, *, user_id=None, agent_name=None, category=None):
        _ = (user_id, agent_name, category)
        if not self.searchable:
            raise NotImplementedError("search not supported")
        return self.hits[:top_k]

    def add(self, thread_id, messages, *, agent_name=None, user_id=None, trace_id=None) -> None:
        _ = (agent_name, user_id, trace_id)
        self.writes.append((thread_id, list(messages)))


@pytest.mark.asyncio
async def test_recall_combines_context_and_search_hits():
    memory = FakeMemory(context="prior: momentum worked", hits=[{"fact": "reversal fails"}, {"text": "costs matter"}])
    adapter = MemoryAdapter(memory, user_id="u-1")
    text = await adapter.recall_context("momentum")
    assert "prior: momentum worked" in text
    assert "reversal fails" in text
    assert "costs matter" in text
    assert memory.queries == ["momentum"]


@pytest.mark.asyncio
async def test_recall_without_search_support_uses_context_only():
    memory = FakeMemory(context="only context", searchable=False)
    adapter = MemoryAdapter(memory)
    assert await adapter.recall_context("topic") == "only context"


@pytest.mark.asyncio
async def test_recall_unbound_is_empty():
    adapter = MemoryAdapter(None)
    assert adapter.bound is False
    assert await adapter.recall_context("anything") == ""


@pytest.mark.asyncio
async def test_record_turn_queues_compact_messages():
    memory = FakeMemory()
    adapter = MemoryAdapter(memory, user_id="u-1", agent_name="evolution-loop")
    await adapter.record_turn(thread_id="run-1", user_text="iteration 0: try momentum", assistant_text="decision=True")
    assert len(memory.writes) == 1
    thread_id, messages = memory.writes[0]
    assert thread_id == "run-1"
    assert messages[0]["role"] == "user"
    assert "momentum" in messages[0]["content"]
    assert messages[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_record_turn_unbound_is_noop():
    adapter = MemoryAdapter(None)
    assert await adapter.record_turn(thread_id="run-1", user_text="u", assistant_text="a") is None


def test_should_promote_truth_table():
    assert MemoryAdapter.should_promote(gate_passed=True, confidence=0.9, min_confidence=0.7) is True
    assert MemoryAdapter.should_promote(gate_passed=True, confidence=0.7, min_confidence=0.7) is True
    assert MemoryAdapter.should_promote(gate_passed=True, confidence=0.69, min_confidence=0.7) is False
    assert MemoryAdapter.should_promote(gate_passed=False, confidence=0.99, min_confidence=0.7) is False
    assert MemoryAdapter.should_promote(gate_passed=True, confidence=None, min_confidence=0.0) is False
    assert MemoryAdapter.should_promote(gate_passed=True, confidence=float("nan"), min_confidence=0.0) is False
    assert MemoryAdapter.should_promote(gate_passed=True, confidence=1.5, min_confidence=0.0) is False

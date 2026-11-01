"""Binds the evolution loop to agent memory.

The loop is a client of agent memory only:

* Pre-proposal recall: ``get_context`` with the hypothesis topic as the
  query hint, plus ``search`` for ranked related memories when the backend
  supports it. The recalled text is injected into hypothesis generation
  alongside the retrieval packet and spark recall.
* Post-feedback write: ``add`` of one compact turn summary per iteration.
* Promotion policy: :meth:`MemoryAdapter.should_promote` gates the spark
  ``memory_promote`` call (gate passed + confidence threshold); spark itself
  runs the contradiction check and may answer ``pending_human``, which the
  loop accepts without escalation.

This module never touches the ``MemoryManager`` contract: it defines the
structural :class:`AgentMemory` protocol (mirroring the tier-1 methods plus
``search``) and the production wiring passes a real manager. The adapter is
also usable unbound (``memory=None``), in which case recall returns "" and
writes are no-ops — the loop still runs on KB + governor context.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AgentMemory",
    "MemoryAdapter",
    "DEFAULT_SEARCH_TOP_K",
]

DEFAULT_SEARCH_TOP_K = 5


@runtime_checkable
class AgentMemory(Protocol):
    """Structural agent-memory boundary (mirrors ``MemoryManager``).

    ``get_context`` returns injection-ready text; ``search`` returns ranked
    fact dicts and raises ``NotImplementedError`` on backends without
    retrieval; ``add`` queues a conversation for memory update.
    """

    def get_context(
        self,
        user_id: str | None,
        *,
        agent_name: str | None = None,
        thread_id: str | None = None,
        query: str | None = None,
    ) -> str:
        """Return injection-ready memory text for the given bucket."""
        ...

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the bucket's memory for facts matching ``query``."""
        ...

    def add(
        self,
        thread_id: str,
        messages: list[Any],
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        """Queue a conversation for memory update."""
        ...


def _fact_text(fact: Any) -> str | None:
    """Extract displayable text from one search hit (None when absent)."""
    if isinstance(fact, str):
        return fact.strip() or None
    if isinstance(fact, dict):
        for key in ("fact", "text", "content", "memory"):
            value = fact.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


class MemoryAdapter:
    """Agent-memory binding for the evolution loop stages.

    Args:
        memory: Bound agent memory, or None to run without memory.
        user_id: Per-user bucket for recall and writes.
        agent_name: Per-agent bucket for recall and writes.
        search_top_k: Ranked memories appended after the injected context.
    """

    def __init__(
        self,
        memory: AgentMemory | None,
        *,
        user_id: str | None = None,
        agent_name: str | None = "evolution-loop",
        search_top_k: int = DEFAULT_SEARCH_TOP_K,
    ) -> None:
        self._memory = memory
        self._user_id = user_id
        self._agent_name = agent_name
        self._search_top_k = search_top_k

    @property
    def bound(self) -> bool:
        """True when an agent memory is bound."""
        return self._memory is not None

    async def recall_context(self, topic: str) -> str:
        """Recall memory context for a proposal topic ("" when unbound).

        Combines ``get_context`` (with *topic* as the query hint) and the
        top-ranked ``search`` hits. A backend without retrieval contributes
        its injected context only; read failures propagate so fail-closed
        backends are honored.
        """
        if self._memory is None:
            return ""
        memory = self._memory
        context = await asyncio.to_thread(
            memory.get_context,
            self._user_id,
            agent_name=self._agent_name,
            query=topic,
        )
        try:
            hits = await asyncio.to_thread(
                memory.search,
                topic,
                self._search_top_k,
                user_id=self._user_id,
                agent_name=self._agent_name,
            )
        except NotImplementedError:
            hits = []
        lines = [context.strip()] if context.strip() else []
        bullets = [text for text in (_fact_text(hit) for hit in hits) if text]
        if bullets:
            lines.append("Related memories:")
            lines.extend(f"- {text}" for text in bullets)
        return "\n".join(lines)

    async def record_turn(
        self,
        *,
        thread_id: str,
        user_text: str,
        assistant_text: str,
        trace_id: str | None = None,
    ) -> None:
        """Queue one compact turn summary (no-op when unbound)."""
        if self._memory is None:
            return None
        messages: list[Any] = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        await asyncio.to_thread(
            self._memory.add,
            thread_id,
            messages,
            agent_name=self._agent_name,
            user_id=self._user_id,
            trace_id=trace_id,
        )

    @staticmethod
    def should_promote(*, gate_passed: bool, confidence: float | None, min_confidence: float) -> bool:
        """Return True when the promotion policy passes.

        Promotion requires the significance gate to have passed and the
        summary confidence to meet *min_confidence* (the loop's
        ``min_confidence_to_promote``). Non-numeric or missing confidence
        never promotes. Contradiction detection stays spark-side.
        """
        if not gate_passed:
            return False
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            return False
        if not (0.0 <= confidence <= 1.0):
            return False
        return confidence >= min_confidence

"""Run-start knowledge bootstrap middleware (Phase 2: shared recall).

:class:`KnowledgeBootstrapMiddleware` runs the compulsory KB
``knowledge_bootstrap`` step before substantive research: on the first
``before_agent`` of a run it converts the latest user message into a
research intent, builds the fixed-budget context packet
(:func:`deerflow.knowledge.tools.bootstrap.knowledge_bootstrap`), and
appends it as a stamped ``SystemMessage`` so every downstream model
call sees prior consensus, experiments, failures, contradictions,
assumptions and skills.

Patterns mirrored from the harness:

* ``memory_middleware.py`` — ``AgentMiddleware`` subclass with
  ``before_agent``/``abefore_agent``, thread id resolution (runtime
  context first, LangGraph configurable fallback), user id via
  ``resolve_runtime_user_id``, config-gated no-op, async path via
  ``asyncio.to_thread``.
* ``dynamic_context_middleware.py`` — run-start injection returning a
  ``{"messages": [...]}`` state update with stable message ids and
  ``additional_kwargs`` markers, once-per-run guard, bounded async
  timeout that degrades to "no new injection" instead of hanging.
* ``agents/memory/tools.py`` — failures map to logged warnings; the run
  is never broken by unavailable knowledge (the agent can still call
  the lookup tools mid-run).

This module is intentionally **not registered anywhere**: the
integration step instantiates the middleware with bound stores and
adds it to the agent middleware chain. With no stores bound (or the
knowledge plane disabled) every hook degrades to ``None``.
"""

import asyncio
import logging
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

from deerflow.knowledge.config import get_knowledge_config
from deerflow.knowledge.tools.bootstrap import (
    DEFAULT_CHANNEL_LIMIT,
    DEFAULT_PACKET_CHARS,
    DEFAULT_PACKET_ITEMS,
    BootstrapStores,
    PacketBudget,
    format_context_packet,
    knowledge_bootstrap,
)
from deerflow.knowledge.tools.lookup import get_knowledge_backends
from deerflow.runtime.user_context import resolve_runtime_user_id

logger = logging.getLogger(__name__)

__all__ = [
    "KNOWLEDGE_BOOTSTRAP_MESSAGE_ID",
    "KNOWLEDGE_BOOTSTRAP_MARKER",
    "INJECT_TIMEOUT_SECONDS",
    "KnowledgeBootstrapMiddlewareState",
    "KnowledgeBootstrapMiddleware",
]

#: Stable id of the injected packet message (lets LangGraph replace, not duplicate).
KNOWLEDGE_BOOTSTRAP_MESSAGE_ID = "knowledge-bootstrap-context"
#: ``additional_kwargs`` marker identifying injected packet messages.
KNOWLEDGE_BOOTSTRAP_MARKER = "knowledge_bootstrap"
#: Async injection timeout; expiry degrades to "no packet this turn".
INJECT_TIMEOUT_SECONDS = 20.0


class KnowledgeBootstrapMiddlewareState(AgentState):
    """Compatible with the ``ThreadState`` schema."""

    pass


class KnowledgeBootstrapMiddleware(AgentMiddleware[KnowledgeBootstrapMiddlewareState]):
    """Inject the research context packet once per run, before the agent acts.

    The middleware derives the research intent from the latest human
    message (or an explicit ``intent_builder``), runs
    :func:`knowledge_bootstrap`, and appends the rendered packet as a
    ``SystemMessage``. Injection happens at most once per run: messages
    already carrying :data:`KNOWLEDGE_BOOTSTRAP_MARKER` suppress
    re-injection on later turns.

    Failure policy: any bootstrap error (unbound stores, retrieval
    outage, invalid intent) logs a warning and returns ``None`` so the
    run proceeds without the packet; the agent retains the
    ``knowledge_*``/``experiment_get``/``artifact_*`` tools for mid-run
    recall.
    """

    state_schema = KnowledgeBootstrapMiddlewareState

    def __init__(
        self,
        *,
        stores: BootstrapStores | None = None,
        agent_name: str | None = None,
        enabled: bool | None = None,
        max_items: int = DEFAULT_PACKET_ITEMS,
        max_chars: int = DEFAULT_PACKET_CHARS,
        channel_limit: int = DEFAULT_CHANNEL_LIMIT,
        intent_builder: Any | None = None,
    ):
        """Initialize the middleware (integration constructs this; nothing auto-registers it).

        Args:
            stores: Interface bundle for packet retrieval. When omitted,
                the process-wide :func:`get_knowledge_backends` bindings
                are used (integration binds real PG there).
            agent_name: Optional agent scope recorded for future
                per-agent recall tuning; currently informational.
            enabled: Master switch override. ``None`` (default) falls back
                to ``KnowledgeConfig.is_enabled()`` at each hook call so a
                config flip applies without a restart.
            max_items: Packet entry cap.
            max_chars: Rendered packet character cap.
            channel_limit: Per-channel retrieval depth.
            intent_builder: Optional callable ``(task_text, *, runtime,
                user_id) -> research_intent`` overriding the default
                "latest human message is the task" derivation. It may
                return a task string, an intent mapping, or a
                ``ResearchIntent``.
        """
        super().__init__()
        self._stores = stores
        self._agent_name = agent_name
        self._enabled = enabled
        self._budget = PacketBudget(max_items=max_items, max_chars=max_chars, channel_limit=channel_limit)
        self._intent_builder = intent_builder

    @property
    def budget(self) -> PacketBudget:
        """The packet budget this middleware injects with."""
        return self._budget

    def release_policy_parameters(self) -> dict[str, object]:
        """Describe behaviour-affecting fields for release-policy collection.

        Implements the middleware self-description contract (duck-typed
        ``ReleasePolicyProvider``): JSON-serializable values only, no
        prompt or packet copies.
        """
        return {
            "enabled": self._enabled,
            "agent_name": self._agent_name,
            "max_items": self._budget.max_items,
            "max_chars": self._budget.max_chars,
            "channel_limit": self._budget.channel_limit,
            "has_explicit_stores": self._stores is not None,
            "has_intent_builder": self._intent_builder is not None,
        }

    def _is_enabled(self) -> bool:
        """Return the effective enabled flag (explicit override, else config)."""
        if self._enabled is not None:
            return self._enabled
        try:
            return get_knowledge_config().is_enabled()
        except Exception:
            logger.warning("KnowledgeBootstrapMiddleware: config read failed; treating knowledge as disabled", exc_info=True)
            return False

    def _resolve_stores(self) -> BootstrapStores | None:
        """Return explicit stores, else registry bindings, else None."""
        if self._stores is not None:
            return self._stores
        try:
            bindings = get_knowledge_backends()
        except Exception:
            logger.warning("KnowledgeBootstrapMiddleware: backend registry read failed", exc_info=True)
            return None
        if bindings.retrieval is None and bindings.experiments is None:
            return None
        return BootstrapStores(retrieval=bindings.retrieval, experiments=bindings.experiments, artifacts=bindings.artifacts)

    @staticmethod
    def _resolve_thread_id(runtime: Runtime) -> str | None:
        """Resolve the thread id (runtime context first, configurable fallback)."""
        context = getattr(runtime, "context", None)
        thread_id = context.get("thread_id") if isinstance(context, dict) else None
        if thread_id is None:
            try:
                config_data = get_config()
            except Exception:
                config_data = {}
            if isinstance(config_data, dict):
                configurable = config_data.get("configurable", {})
                if isinstance(configurable, dict):
                    thread_id = configurable.get("thread_id")
        return str(thread_id) if thread_id else None

    @staticmethod
    def _already_injected(messages: list[BaseMessage]) -> bool:
        """Return True when a packet message is already in history."""
        for message in messages:
            kwargs = getattr(message, "additional_kwargs", None)
            if isinstance(kwargs, dict) and kwargs.get(KNOWLEDGE_BOOTSTRAP_MARKER):
                return True
            if getattr(message, "id", None) == KNOWLEDGE_BOOTSTRAP_MESSAGE_ID:
                return True
        return False

    @staticmethod
    def _latest_human_text(messages: list[BaseMessage]) -> str | None:
        """Return the latest human message text, or None when absent/empty."""
        for message in reversed(messages):
            if not isinstance(message, HumanMessage):
                continue
            content = message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts = [part.get("text", "") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)]
                text = " ".join(part.strip() for part in parts if part.strip()).strip()
                if text:
                    return text
        return None

    def _build_packet_text(self, task_text: str, stores: BootstrapStores, runtime: Runtime, user_id: str) -> str | None:
        """Run bootstrap for ``task_text`` and render the packet (None on failure)."""
        research_intent: Any = task_text
        if self._intent_builder is not None:
            try:
                research_intent = self._intent_builder(task_text, runtime=runtime, user_id=user_id)
            except Exception:
                logger.warning("KnowledgeBootstrapMiddleware: intent_builder failed; falling back to raw task text", exc_info=True)
                research_intent = task_text
        try:
            packet = knowledge_bootstrap(research_intent, stores=stores, budget=self._budget)
        except Exception:
            logger.warning("KnowledgeBootstrapMiddleware: knowledge_bootstrap failed; running without the packet", exc_info=True)
            return None
        try:
            return format_context_packet(packet)
        except Exception:
            logger.warning("KnowledgeBootstrapMiddleware: packet render failed; running without the packet", exc_info=True)
            return None

    def _inject(self, state: KnowledgeBootstrapMiddlewareState, runtime: Runtime) -> dict | None:
        """Compute the run-start injection update (sync; no hook side effects)."""
        if not self._is_enabled():
            return None
        messages = list(state.get("messages", []))
        if not messages:
            return None
        if self._already_injected(messages):
            return None
        stores = self._resolve_stores()
        if stores is None:
            logger.debug("KnowledgeBootstrapMiddleware: no stores bound; skipping injection")
            return None
        task_text = self._latest_human_text(messages)
        if not task_text:
            logger.debug("KnowledgeBootstrapMiddleware: no human message to derive intent from; skipping")
            return None
        thread_id = self._resolve_thread_id(runtime)
        user_id = resolve_runtime_user_id(runtime)
        packet_text = self._build_packet_text(task_text, stores, runtime, user_id)
        if not packet_text:
            return None
        logger.info(
            "KnowledgeBootstrapMiddleware: injecting context packet (thread_id=%r, agent_name=%r, chars=%d)",
            thread_id,
            self._agent_name,
            len(packet_text),
        )
        return {
            "messages": [
                SystemMessage(
                    content=packet_text,
                    id=KNOWLEDGE_BOOTSTRAP_MESSAGE_ID,
                    additional_kwargs={KNOWLEDGE_BOOTSTRAP_MARKER: True, "hide_from_ui": True},
                )
            ]
        }

    @override
    def before_agent(self, state: KnowledgeBootstrapMiddlewareState, runtime: Runtime) -> dict | None:
        """Inject the context packet on the first turn of a run (sync path)."""
        return self._inject(state, runtime)

    @override
    async def abefore_agent(self, state: KnowledgeBootstrapMiddlewareState, runtime: Runtime) -> dict | None:
        """Inject the context packet on the first turn of a run (async path).

        Retrieval does blocking I/O, so injection runs in a worker thread
        with a bounded timeout; expiry degrades to "no packet this turn"
        rather than hanging the run.
        """
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._inject, state, runtime), timeout=INJECT_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.warning("KnowledgeBootstrapMiddleware: injection timed out after %.1fs; running without the packet", INJECT_TIMEOUT_SECONDS)
            return None

"""End-to-end wiring for the native evolution loop.

This module is the composition root: it binds the loop core
(:mod:`deerflow.evolution.loop`) to its production boundaries — the spark
trial governor over MCP transport, the Knowledge Plane stores, agent
memory, LangChain model profiles, and the Quantflow sandbox provider —
and exposes the non-interactive entry points (single-tick and full-run).

Import-cycle contract: this module is loaded lazily (see
:mod:`deerflow.evolution`), and every heavy import (``deerflow.config``,
``deerflow.mcp``, ``deerflow.sandbox``, ``deerflow.models``/langchain)
stays inside the builder call path, so importing the evolution package —
and even this module — never pulls provider SDKs, transport code, or the
application config graph at module load.

LangChain profile resolution happens once at loop construction:
:func:`resolve_model_profiles` maps every role in
:data:`~deerflow.evolution.config.MODEL_ROLES` through
``EvolutionConfig.model_roles`` to a ``config.yaml`` model profile and
fails fast naming the missing role or profile. Standalone loop turns use
``attach_tracing=True`` (the factory default kept by
:class:`~deerflow.evolution.stages.llm.LangChainModelFactory`).

No human-in-the-loop: the entry points take no queues, prompts, or
approval callbacks. A disabled config (``enabled=False``, the default)
makes :meth:`EvolutionRunner.tick`/:meth:`EvolutionRunner.run` return
``None`` without touching any boundary, so schedulers can call
unconditionally.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from deerflow.evolution.config import MODEL_ROLES, EvolutionConfig
from deerflow.evolution.errors import EvolutionError
from deerflow.evolution.loop import EvolutionLoop
from deerflow.evolution.models import EvoStep
from deerflow.evolution.selectors import AncestorSOTASelector, CheckpointSelector, LatestSOTASelector, SOTASelector
from deerflow.evolution.trace import InMemoryTraceStore, TraceStore

__all__ = [
    "SPARK_SERVER_NAME",
    "EVOLUTION_AGENT_NAME",
    "McpSparkTransport",
    "spark_transport_from_task_caller",
    "EvolutionServices",
    "WiredLoop",
    "EvolutionRunner",
    "build_loop",
    "resolve_model_profiles",
    "build_production_loop",
]

SPARK_SERVER_NAME = "spark"
EVOLUTION_AGENT_NAME = "evolution-loop"


class McpSparkTransport:
    """Spark MCP transport over a caller-supplied async tool caller.

    The caller binds this to the configured MCP machinery (e.g.
    :class:`deerflow.mcp.task_tool_caller.McpTaskToolCaller` via
    :func:`spark_transport_from_task_caller`); unit tests bind it to an
    in-memory fake. Decoded results pass through untouched — error
    envelopes and refusal mapping belong to
    :class:`~deerflow.evolution.spark_client.SparkClient`.
    """

    def __init__(self, call_tool: Callable[[str, Mapping[str, Any]], Awaitable[Any]]) -> None:
        self._call_tool = call_tool

    async def call_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> Any:
        """Call one spark MCP tool and return its decoded result."""
        return await self._call_tool(tool_name, dict(arguments))


def spark_transport_from_task_caller(
    caller: Any,
    *,
    user_id: str,
    thread_id: str,
    server_name: str = SPARK_SERVER_NAME,
) -> McpSparkTransport:
    """Bind a task tool caller to the spark server as a spark transport.

    *caller* is duck-typed (``await caller.call_tool(server_name=...,
    tool_name=..., arguments=..., user_id=..., thread_id=...)``) so this
    module never imports the MCP machinery itself.
    """
    if not user_id or not thread_id:
        raise ValueError("spark_transport_from_task_caller requires non-empty user_id and thread_id.")

    async def _call(tool_name: str, arguments: Mapping[str, Any]) -> Any:
        return await caller.call_tool(
            server_name=server_name,
            tool_name=tool_name,
            arguments=dict(arguments),
            user_id=user_id,
            thread_id=thread_id,
        )

    return McpSparkTransport(_call)


@dataclass
class EvolutionServices:
    """Caller-supplied boundaries for one wired evolution loop."""

    spark_transport: Any
    kb_store: Any
    project_id: str
    models: Any
    executor: Any
    memory: Any | None = None
    trace_store: TraceStore | None = None
    search_store: Any | None = None
    retrieval: Any | None = None
    user_id: str | None = None
    agent_name: str = EVOLUTION_AGENT_NAME


@dataclass
class WiredLoop:
    """A loop plus the shared handles its stages were bound to."""

    loop: EvolutionLoop
    state: Any
    spark: Any
    knowledge: Any
    memory: Any
    config: EvolutionConfig
    trace_store: TraceStore


def build_loop(
    config: EvolutionConfig,
    services: EvolutionServices,
    *,
    checkpoint_selector: CheckpointSelector | None = None,
    sota_selector: SOTASelector | None = None,
    clock: Callable[[], float] | None = None,
    span_emitter: Any | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> WiredLoop:
    """Bind every stage to *services* over one shared iteration state.

    Args:
        config: Loop budgets, defaults, and model-role mapping.
        services: Concrete boundaries (spark transport, KB stores, agent
            memory, model factory, sandbox executor, trace store).
        checkpoint_selector: Attach-point rule (default: latest SOTA).
        sota_selector: Winner rule for run summaries (default: ancestor
            SOTA walk, which also sees accepted ancestors).

    Returns:
        The wired loop plus the shared
        :class:`~deerflow.evolution.stages.state.IterationState` and
        adapter handles (for tests and observability sidecars).
    """
    from deerflow.evolution.knowledge_adapter import KnowledgeAdapter
    from deerflow.evolution.memory_adapter import MemoryAdapter
    from deerflow.evolution.spark_client import SparkClient
    from deerflow.evolution.stages.design import ExperimentDesignerImpl
    from deerflow.evolution.stages.execute import CoderImpl, RunnerImpl
    from deerflow.evolution.stages.feedback import FeedbackSynthesizerImpl
    from deerflow.evolution.stages.propose import HypothesisGenerator
    from deerflow.evolution.stages.state import IterationState

    _require_service("spark_transport", services.spark_transport)
    _require_service("kb_store", services.kb_store)
    _require_service("models", services.models)
    _require_service("executor", services.executor)
    if not services.project_id:
        raise ValueError("EvolutionServices.project_id must be a non-empty project UUID.")

    state = IterationState()
    spark = SparkClient(services.spark_transport)
    knowledge = KnowledgeAdapter(
        services.kb_store,
        project_id=services.project_id,
        search_store=services.search_store,
        retrieval=services.retrieval,
    )
    memory = MemoryAdapter(services.memory, user_id=services.user_id, agent_name=services.agent_name)
    trace_store = services.trace_store or InMemoryTraceStore()
    loop_kwargs: dict[str, Any] = {
        "config": config,
        "trace_store": trace_store,
        "checkpoint_selector": checkpoint_selector or LatestSOTASelector(),
        "sota_selector": sota_selector or AncestorSOTASelector(),
        "proposer": HypothesisGenerator(knowledge=knowledge, spark=spark, memory=memory, models=services.models, config=config, state=state),
        "designer": ExperimentDesignerImpl(knowledge=knowledge, spark=spark, models=services.models, config=config, state=state),
        "coder": CoderImpl(services.models),
        "runner": RunnerImpl(spark=spark, executor=services.executor, state=state),
        "synthesizer": FeedbackSynthesizerImpl(knowledge=knowledge, spark=spark, memory=memory, models=services.models, config=config, state=state),
    }
    if clock is not None:
        loop_kwargs["clock"] = clock
    if span_emitter is not None:
        loop_kwargs["span_emitter"] = span_emitter
    if should_stop is not None:
        loop_kwargs["should_stop"] = should_stop
    loop_kwargs["state"] = state
    return WiredLoop(
        loop=EvolutionLoop(**loop_kwargs),
        state=state,
        spark=spark,
        knowledge=knowledge,
        memory=memory,
        config=config,
        trace_store=trace_store,
    )


def _require_service(name: str, value: Any) -> None:
    if value is None:
        raise ValueError(f"EvolutionServices.{name} is required to wire the evolution loop.")


def resolve_model_profiles(config: EvolutionConfig, app_config: Any) -> dict[str, str]:
    """Resolve every model role to a configured model profile name.

    Fails fast: a role with no mapping raises naming the role; a mapped
    name with no ``config.yaml`` model profile raises naming both.
    """
    resolved: dict[str, str] = {}
    for role in MODEL_ROLES:
        profile = config.model_for_role(role)
        if app_config.get_model_config(profile) is None:
            raise ValueError(f"Evolution role {role!r} maps to model profile {profile!r}, which has no entry in config.yaml models.")
        resolved[role] = profile
    return resolved


def build_production_loop(
    *,
    kb_store: Any,
    project_id: str,
    app_config: Any | None = None,
    spark_transport: Any | None = None,
    spark_caller: Any | None = None,
    spark_user_id: str | None = None,
    spark_thread_id: str | None = None,
    spark_server_name: str = SPARK_SERVER_NAME,
    memory: Any | None = None,
    user_id: str | None = None,
    agent_name: str = EVOLUTION_AGENT_NAME,
    trace_store: TraceStore | None = None,
    search_store: Any | None = None,
    retrieval: Any | None = None,
    thinking_enabled: bool = False,
    reasoning_effort: str | None = None,
    command_timeout: float | None = None,
    checkpoint_selector: CheckpointSelector | None = None,
    sota_selector: SOTASelector | None = None,
    clock: Callable[[], float] | None = None,
    span_emitter: Any | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> WiredLoop:
    """Build a loop with production singletons resolved lazily.

    Resolves the app config (``evolution:`` section), validates every
    model role against ``config.yaml`` profiles, builds the LangChain
    model factory, and wraps the sandbox provider singleton in the
    Quantflow executor. Spark arrives as exactly one of *spark_transport*
    (already bound) or *spark_caller* (a task tool caller bound here with
    *spark_user_id*/*spark_thread_id*); the KB store and agent memory
    stay caller-supplied (PG bindings are phase 2).
    """
    from deerflow.config import get_app_config
    from deerflow.evolution.sandboxing import QuantflowSandboxExecutor
    from deerflow.evolution.stages.llm import LangChainModelFactory
    from deerflow.sandbox import get_sandbox_provider

    resolved_app_config = app_config or get_app_config()
    config = resolved_app_config.evolution
    resolve_model_profiles(config, resolved_app_config)
    transport = _resolve_production_transport(
        spark_transport=spark_transport,
        spark_caller=spark_caller,
        spark_user_id=spark_user_id,
        spark_thread_id=spark_thread_id,
        spark_server_name=spark_server_name,
    )
    models = LangChainModelFactory(
        config,
        thinking_enabled=thinking_enabled,
        app_config=resolved_app_config,
        reasoning_effort=reasoning_effort,
    )
    executor = QuantflowSandboxExecutor(
        get_sandbox_provider(),
        command_timeout=command_timeout if command_timeout is not None else config.resolve_sandbox_command_timeout(),
    )
    services = EvolutionServices(
        spark_transport=transport,
        kb_store=kb_store,
        project_id=project_id,
        models=models,
        executor=executor,
        memory=memory,
        trace_store=trace_store,
        search_store=search_store,
        retrieval=retrieval,
        user_id=user_id,
        agent_name=agent_name,
    )
    return build_loop(
        config,
        services,
        checkpoint_selector=checkpoint_selector,
        sota_selector=sota_selector,
        clock=clock,
        span_emitter=span_emitter,
        should_stop=should_stop,
    )


def _resolve_production_transport(
    *,
    spark_transport: Any | None,
    spark_caller: Any | None,
    spark_user_id: str | None,
    spark_thread_id: str | None,
    spark_server_name: str,
) -> Any:
    if (spark_transport is None) == (spark_caller is None):
        raise ValueError("build_production_loop needs exactly one of spark_transport or spark_caller.")
    if spark_transport is not None:
        return spark_transport
    if not spark_user_id or not spark_thread_id:
        raise ValueError("Binding spark_caller requires spark_user_id and spark_thread_id.")
    return spark_transport_from_task_caller(spark_caller, user_id=spark_user_id, thread_id=spark_thread_id, server_name=spark_server_name)


@dataclass
class EvolutionRunner:
    """Non-interactive entry point over a wired loop.

    :meth:`tick` runs exactly one iteration (the scheduled-tick entry);
    :meth:`run` iterates to budget. Both return ``None`` — without
    touching any boundary — when the loop is disabled, so schedulers can
    call unconditionally.
    """

    wired: WiredLoop = field(repr=False)

    @property
    def config(self) -> EvolutionConfig:
        """The wired loop's evolution config."""
        return self.wired.config

    @property
    def loop(self) -> EvolutionLoop:
        """The wired evolution loop."""
        return self.wired.loop

    async def tick(self, run_id: str) -> EvoStep | None:
        """Run one iteration for *run_id* (None when disabled)."""
        if not self.wired.config.is_enabled():
            return None
        return await self.wired.loop.step(run_id)

    async def run(self, run_id: str) -> Any | None:
        """Run *run_id* to budget (None when disabled)."""
        if not self.wired.config.is_enabled():
            return None
        return await self.wired.loop.run(run_id)

    def __post_init__(self) -> None:
        if not isinstance(self.wired, WiredLoop):
            raise EvolutionError(f"EvolutionRunner needs a WiredLoop, got {type(self.wired).__name__}.")

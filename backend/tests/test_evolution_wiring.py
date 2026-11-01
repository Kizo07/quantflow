"""Unit tests for deerflow.evolution.wiring (offline).

Covers the composition root: shared iteration state across stages,
LangChain profile resolution at construction, the MCP transport binding,
production-singleton wiring, and the non-interactive runner entry points.
"""

import json
from typing import Any

import pytest

from deerflow.evolution.config import MODEL_ROLES, EvolutionConfig
from deerflow.evolution.errors import EvolutionError
from deerflow.evolution.loop import StageContext
from deerflow.evolution.models import CodedExperiment
from deerflow.evolution.selectors import AncestorSOTASelector, LatestSOTASelector, RootRestartSelector
from deerflow.evolution.stages.execute import SandboxResult
from deerflow.evolution.trace import InMemoryTraceStore
from deerflow.evolution.wiring import (
    EVOLUTION_AGENT_NAME,
    SPARK_SERVER_NAME,
    EvolutionRunner,
    EvolutionServices,
    McpSparkTransport,
    WiredLoop,
    build_loop,
    build_production_loop,
    resolve_model_profiles,
    spark_transport_from_task_caller,
)
from deerflow.knowledge.write_api import AssumptionRecord, ExperimentRecord, FailureRecord

RUN_ID = "55555555-5555-5555-5555-555555555555"
PROJECT_ID = "66666666-6666-6666-6666-666666666666"

ROLES = {
    "propose": "propose-profile",
    "design": "design-profile",
    "summarize": "summarize-profile",
    "coder": "coder-profile",
}


class ScriptedModels:
    """ChatModelFactory fake: scripted response queue per role."""

    def __init__(self, responses: dict[str, list[str]] | None = None) -> None:
        self.responses = {role: list(items) for role, items in (responses or {}).items()}
        self.calls: list[str] = []

    async def generate(self, *, role: str, system: str, user: str) -> str:
        self.calls.append(role)
        queue = self.responses.get(role, [])
        if not queue:
            raise AssertionError(f"No scripted response for role {role!r}.")
        return queue.pop(0)


class InMemoryExperimentStore:
    """Minimal ExperimentStore fake (write_api boundary)."""

    def __init__(self) -> None:
        self.records: dict[str, ExperimentRecord] = {}
        self.failures: dict[str, FailureRecord] = {}
        self.assumptions: dict[str, AssumptionRecord] = {}

    def find_experiment_by_idempotency_key(self, idempotency_key: str):
        return next((r for r in self.records.values() if r.idempotency_key == idempotency_key), None)

    def find_experiment_by_commit_key(self, idempotency_key: str):
        return next((r for r in self.records.values() if r.commit_idempotency_key == idempotency_key), None)

    def find_experiment_by_execution_hash(self, execution_hash: str):
        return next((r for r in self.records.values() if r.execution_hash == execution_hash), None)

    def get_experiment(self, experiment_id: str):
        return self.records.get(experiment_id)

    def insert_experiment(self, record: ExperimentRecord) -> None:
        self.records[record.id] = record

    def update_experiment(self, record: ExperimentRecord) -> None:
        self.records[record.id] = record

    def find_failure_by_idempotency_key(self, idempotency_key: str):
        return next((r for r in self.failures.values() if r.idempotency_key == idempotency_key), None)

    def insert_failure(self, record: FailureRecord) -> None:
        self.failures[record.id] = record

    def find_assumption_by_idempotency_key(self, idempotency_key: str):
        return next((r for r in self.assumptions.values() if r.idempotency_key == idempotency_key), None)

    def insert_assumption(self, record: AssumptionRecord) -> None:
        self.assumptions[record.id] = record


class CannedTransport:
    """SparkTransport fake returning canned per-tool payloads."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.counter = 0

    async def call_tool(self, tool_name: str, arguments) -> Any:
        self.calls.append((tool_name, dict(arguments)))
        self.counter += 1
        n = self.counter
        if tool_name == "hypothesis_propose":
            return {"hypothesis_id": f"h-{n}", "N": 0.5}
        if tool_name == "register_hypothesis":
            return {"episode_id": f"e-{n}", "protocol_hash": f"ph-{n}", "seed": 11}
        if tool_name == "backtest_run":
            return {"metrics": {"sr": 1.2}, "trial": {"id": f"t-{n}"}, "simulated": True}
        if tool_name == "significance_gate":
            return {"pass": True, "status": "pass", "sr": 1.2, "dsr": 0.9, "p": 0.01, "M_family": 1}
        if tool_name == "log_outcome":
            return {"ok": True}
        if tool_name == "check_novelty":
            return {"N": 0.6}
        if tool_name == "suggest_next":
            return {"suggestions": [{"family": "value"}], "family_selection_probabilities": {"value": 0.7}}
        if tool_name == "memory_recall":
            return {"beliefs": []}
        if tool_name == "memory_promote":
            return {"belief_id": f"b-{n}", "promotion_status": "auto", "confidence": 0.9}
        raise AssertionError(f"Unexpected tool {tool_name!r}.")


class FakeExecutor:
    """SandboxExecutor fake returning a canned result."""

    def __init__(self, stdout: str = "sharpe=1.2") -> None:
        self.result = SandboxResult(stdout=stdout)
        self.calls = 0

    async def run(self, coded: CodedExperiment, ctx: StageContext) -> SandboxResult:
        self.calls += 1
        return self.result


class FakeMemory:
    """AgentMemory fake recording context reads and turn writes."""

    def __init__(self) -> None:
        self.turns: list[dict[str, Any]] = []

    def get_context(self, user_id, *, agent_name=None, thread_id=None, query=None) -> str:
        return "prior: value worked in 2024"

    def search(self, query: str, top_k: int = 5, **kwargs) -> list[dict[str, Any]]:
        return []

    def add(self, thread_id: str, messages: list[Any], **kwargs) -> None:
        self.turns.append({"thread_id": thread_id, "messages": messages, **kwargs})


def _c6_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hypothesis": "21d momentum persists in liquid us equities.",
        "reason": "Slow information diffusion sustains trends.",
        "mechanism": "slow diffusion",
        "expected_direction": "positive",
        "horizon": "21d",
        "falsifier": "IC <= 0 over the next 63 sessions",
        "scope": "us-equities",
        "predicted_ic": 0.05,
        "predicted_sigma": 0.02,
        "hypothesis_type": "predictive",
    }
    payload.update(overrides)
    return payload


def _spec_payload() -> dict[str, Any]:
    return {
        "design": {"hypothesis": "21d momentum persists.", "methodology": {"universe": "sp500"}},
        "code": {"artifact": "code-v1"},
        "data": {},
        "parameters": {"seed": 7},
        "environment": {"image": "qf:v1"},
    }


def _summary_payload() -> dict[str, Any]:
    return {
        "decision": True,
        "reason": "gate passed with margin",
        "observations": "sr stable",
        "hypothesis_evaluation": "confirmed",
        "new_hypothesis": None,
        "confidence": 0.9,
    }


def _wired_models() -> ScriptedModels:
    return ScriptedModels(
        {
            "propose": [json.dumps(_c6_payload())],
            "design": [json.dumps(_spec_payload())],
            "coder": [json.dumps({"code": {"files": {"run.py": "x=1"}, "entrypoint": "python run.py"}, "summary": "v1"})],
            "summarize": [json.dumps(_summary_payload())],
        }
    )


def _services(**overrides: Any) -> EvolutionServices:
    values: dict[str, Any] = {
        "spark_transport": CannedTransport(),
        "kb_store": InMemoryExperimentStore(),
        "project_id": PROJECT_ID,
        "models": _wired_models(),
        "executor": FakeExecutor(),
        "memory": FakeMemory(),
    }
    values.update(overrides)
    return EvolutionServices(**values)


def test_build_loop_shares_one_iteration_state_across_stages():
    wired = build_loop(EvolutionConfig(max_iterations=3), _services())

    assert isinstance(wired, WiredLoop)
    assert wired.loop.proposer._state is wired.state
    assert wired.loop.designer._state is wired.state
    assert wired.loop.runner._state is wired.state
    assert wired.loop.synthesizer._state is wired.state
    assert isinstance(wired.loop.checkpoint_selector, LatestSOTASelector)
    assert isinstance(wired.loop.sota_selector, AncestorSOTASelector)
    assert isinstance(wired.trace_store, InMemoryTraceStore)
    assert wired.memory._agent_name == EVOLUTION_AGENT_NAME


def test_build_loop_honors_selector_clock_and_stop_overrides():
    clock_calls: list[None] = []
    wired = build_loop(
        EvolutionConfig(max_iterations=3),
        _services(),
        checkpoint_selector=RootRestartSelector(),
        clock=lambda: 1.0,
        should_stop=lambda: False,
    )
    assert isinstance(wired.loop.checkpoint_selector, RootRestartSelector)
    assert wired.loop.clock() == 1.0
    assert wired.loop.should_stop is not None and wired.loop.should_stop() is False
    assert clock_calls == []


def test_build_loop_requires_each_service():
    config = EvolutionConfig(max_iterations=1)
    for field in ("spark_transport", "kb_store", "models", "executor"):
        with pytest.raises(ValueError, match=field):
            build_loop(config, _services(**{field: None}))
    with pytest.raises(ValueError, match="project_id"):
        build_loop(config, _services(project_id=""))


@pytest.mark.asyncio
async def test_wired_step_records_shared_lineage_and_memory_turn():
    services = _services()
    wired = build_loop(EvolutionConfig(max_iterations=2, enabled=True), services)

    step = await wired.loop.step(RUN_ID)

    assert step.iteration == 0
    assert step.feedback is not None and step.feedback.decision is True
    assert wired.state.get_hypothesis_id(RUN_ID, 0) is not None
    assert wired.state.get_episode_id(RUN_ID, 0) is not None
    assert wired.state.get_experiment_id(RUN_ID, 0) is not None
    assert len(services.memory.turns) == 1
    tools = [name for name, _ in services.spark_transport.calls]
    assert tools == [
        "memory_recall",
        "suggest_next",
        "check_novelty",
        "hypothesis_propose",
        "register_hypothesis",
        "backtest_run",
        "significance_gate",
        "log_outcome",
        "memory_promote",
    ]


class _AppConfig:
    def __init__(self, models: dict[str, Any], evolution: EvolutionConfig) -> None:
        self._models = models
        self.evolution = evolution

    def get_model_config(self, name: str) -> Any | None:
        return self._models.get(name)


def test_resolve_model_profiles_returns_role_mapping():
    config = EvolutionConfig(model_roles=dict(ROLES))
    app = _AppConfig({name: object() for name in ROLES.values()}, config)
    assert resolve_model_profiles(config, app) == dict(ROLES)


def test_resolve_model_profiles_fails_fast_naming_role_and_profile():
    config = EvolutionConfig(model_roles={"propose": "propose-profile"})
    app = _AppConfig({"propose-profile": object()}, config)
    with pytest.raises(ValueError, match="design"):
        resolve_model_profiles(config, app)
    full = EvolutionConfig(model_roles=dict(ROLES))
    app_missing = _AppConfig({name: object() for name in ROLES.values() if name != "coder-profile"}, full)
    with pytest.raises(ValueError, match="coder-profile"):
        resolve_model_profiles(full, app_missing)


@pytest.mark.asyncio
async def test_mcp_spark_transport_delegates_with_copied_arguments():
    seen: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(tool_name: str, arguments) -> Any:
        seen.append((tool_name, arguments))
        return {"ok": True}

    transport = McpSparkTransport(call_tool)
    assert await transport.call_tool("backtest_run", {"episode_id": "e-1"}) == {"ok": True}
    assert seen == [("backtest_run", {"episode_id": "e-1"})]


@pytest.mark.asyncio
async def test_spark_transport_from_task_caller_binds_server_and_identity():
    calls: list[dict[str, Any]] = []

    class Caller:
        async def call_tool(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return {"ok": True}

    transport = spark_transport_from_task_caller(Caller(), user_id="u-1", thread_id="t-1")
    await transport.call_tool("significance_gate", {"episode_id": "e-1"})

    assert calls == [
        {
            "server_name": SPARK_SERVER_NAME,
            "tool_name": "significance_gate",
            "arguments": {"episode_id": "e-1"},
            "user_id": "u-1",
            "thread_id": "t-1",
        }
    ]
    with pytest.raises(ValueError, match="user_id"):
        spark_transport_from_task_caller(Caller(), user_id="", thread_id="t-1")


def test_build_production_loop_needs_exactly_one_spark_binding():
    config = EvolutionConfig(max_iterations=1, model_roles=dict(ROLES))
    app = _AppConfig({name: object() for name in ROLES.values()}, config)
    with pytest.raises(ValueError, match="exactly one"):
        build_production_loop(kb_store=InMemoryExperimentStore(), project_id=PROJECT_ID, app_config=app)
    with pytest.raises(ValueError, match="exactly one"):
        build_production_loop(
            kb_store=InMemoryExperimentStore(),
            project_id=PROJECT_ID,
            app_config=app,
            spark_transport=CannedTransport(),
            spark_caller=object(),
        )


def test_build_production_loop_resolves_profiles_models_and_sandbox(monkeypatch):
    import deerflow.sandbox

    config = EvolutionConfig(max_iterations=1, model_roles=dict(ROLES), sandbox_command_timeout=45.0)
    app = _AppConfig({name: object() for name in ROLES.values()}, config)
    provider = object()
    monkeypatch.setattr(deerflow.sandbox, "get_sandbox_provider", lambda **kwargs: provider)

    wired = build_production_loop(
        kb_store=InMemoryExperimentStore(),
        project_id=PROJECT_ID,
        app_config=app,
        spark_transport=CannedTransport(),
        memory=FakeMemory(),
        user_id="u-1",
    )

    assert wired.config is config
    assert type(wired.loop.runner._executor).__name__ == "QuantflowSandboxExecutor"
    assert wired.loop.runner._executor._command_timeout == 45.0
    assert wired.loop.runner._executor._provider is provider
    assert set(wired.loop.proposer._models._config.model_roles) == set(MODEL_ROLES)


def test_build_production_loop_fails_fast_on_missing_profile():
    config = EvolutionConfig(max_iterations=1, model_roles={})
    app = _AppConfig({}, config)
    with pytest.raises(ValueError, match="propose"):
        build_production_loop(kb_store=InMemoryExperimentStore(), project_id=PROJECT_ID, app_config=app)


@pytest.mark.asyncio
async def test_runner_tick_returns_none_when_disabled_without_touching_boundaries():
    services = _services()
    wired = build_loop(EvolutionConfig(max_iterations=2, enabled=False), services)
    runner = EvolutionRunner(wired)

    assert await runner.tick(RUN_ID) is None
    assert await runner.run(RUN_ID) is None
    assert services.spark_transport.calls == []
    assert services.models.calls == []
    assert services.executor.calls == 0


@pytest.mark.asyncio
async def test_runner_tick_delegates_when_enabled():
    wired = build_loop(EvolutionConfig(max_iterations=2, enabled=True), _services())
    runner = EvolutionRunner(wired)

    step = await runner.tick(RUN_ID)

    assert step is not None and step.iteration == 0
    assert runner.config.is_enabled() is True
    assert runner.loop is wired.loop


def test_runner_rejects_non_wired_loop():
    with pytest.raises(EvolutionError, match="WiredLoop"):
        EvolutionRunner(object())  # type: ignore[arg-type]


def test_sandbox_timeout_resolves_field_then_env(monkeypatch):
    assert EvolutionConfig().resolve_sandbox_command_timeout() == 300.0
    assert EvolutionConfig(sandbox_command_timeout=0).resolve_sandbox_command_timeout() is None
    monkeypatch.setenv("DEER_FLOW_EVOLUTION_SANDBOX_TIMEOUT", "17.5")
    assert EvolutionConfig().resolve_sandbox_command_timeout() == 17.5
    monkeypatch.setenv("DEER_FLOW_EVOLUTION_SANDBOX_TIMEOUT", "not-a-number")
    with pytest.raises(ValueError, match="SANDBOX_TIMEOUT"):
        EvolutionConfig().resolve_sandbox_command_timeout()


@pytest.mark.asyncio
async def test_evolution_runner_exposes_run_id_validation():
    wired = build_loop(EvolutionConfig(max_iterations=1, enabled=True), _services())
    runner = EvolutionRunner(wired)
    with pytest.raises(ValueError, match="UUID"):
        await runner.tick("not-a-uuid")

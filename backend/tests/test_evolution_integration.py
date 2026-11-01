"""Integration tests for the wired evolution loop (offline, deterministic).

Real stage classes plus real adapters over in-memory boundaries: the KB
path runs the real ``write_api`` hashing and replay semantics, the
governor path runs the real ``SparkClient`` parsing, and the sandbox path
runs the real ``QuantflowSandboxExecutor`` over a fake provider. Only the
MCP transport bytes, the chat models, and the sandbox backend are faked.
"""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from deerflow.evolution.config import EvolutionConfig
from deerflow.evolution.errors import EvolutionError
from deerflow.evolution.loop import StageContext, derive_key
from deerflow.evolution.models import CodedExperiment, ExperimentSpec
from deerflow.evolution.sandboxing import PROVIDER_MANIFEST_FILENAME, QuantflowSandboxExecutor, evolution_workdir
from deerflow.evolution.trace import NEW_ROOT
from deerflow.evolution.wiring import EvolutionRunner, EvolutionServices, build_loop
from deerflow.knowledge.write_api import AssumptionRecord, ExperimentRecord, FailureRecord

RUN_ID = "77777777-7777-7777-7777-777777777777"
PROJECT_ID = "88888888-8888-8888-8888-888888888888"


class ScriptedModels:
    """ChatModelFactory fake: scripted response queue per role."""

    def __init__(self, responses: dict[str, list[str]]) -> None:
        self.responses = {role: list(items) for role, items in responses.items()}

    async def generate(self, *, role: str, system: str, user: str) -> str:
        queue = self.responses.get(role, [])
        if not queue:
            raise AssertionError(f"No scripted response for role {role!r}.")
        return queue.pop(0)


class InMemoryExperimentStore:
    """ExperimentStore fake with an order log for governor-first assertions."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.records: dict[str, ExperimentRecord] = {}
        self.failures: dict[str, FailureRecord] = {}
        self.assumptions: dict[str, AssumptionRecord] = {}
        self.events = events if events is not None else []

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
        self.events.append("kb:begin")

    def update_experiment(self, record: ExperimentRecord) -> None:
        self.records[record.id] = record
        self.events.append("kb:commit")

    def find_failure_by_idempotency_key(self, idempotency_key: str):
        return next((r for r in self.failures.values() if r.idempotency_key == idempotency_key), None)

    def insert_failure(self, record: FailureRecord) -> None:
        self.failures[record.id] = record
        self.events.append("kb:failure")

    def find_assumption_by_idempotency_key(self, idempotency_key: str):
        return next((r for r in self.assumptions.values() if r.idempotency_key == idempotency_key), None)

    def insert_assumption(self, record: AssumptionRecord) -> None:
        self.assumptions[record.id] = record
        self.events.append("kb:assumption")


class GovernorTransport:
    """SparkTransport fake with governed, idempotent semantics.

    Writes replay on ``client_id``; ``backtest_scripts`` is a FIFO of
    per-call overrides (None = default success, Exception = raise);
    ``tombstone_id`` blocks backtests without a matching recorded ack.
    """

    def __init__(self, events: list[str] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.events = events if events is not None else []
        self.backtest_scripts: list[Any] = []
        self.tombstone_id: str | None = None
        self.default_gate: dict[str, Any] | None = None
        self.proposed: list[str] = []
        self._replay: dict[tuple[str, str], Any] = {}
        self._counter = 0

    async def call_tool(self, tool_name: str, arguments) -> Any:
        args = dict(arguments)
        self.calls.append((tool_name, args))
        if tool_name in {"hypothesis_propose", "register_hypothesis", "backtest_run", "log_outcome", "memory_promote"}:
            self.events.append(f"spark:{tool_name}")
        client_id = args.get("client_id")
        if isinstance(client_id, str):
            cached = self._replay.get((tool_name, client_id))
            if cached is not None:
                return cached
        payload = self._dispatch(tool_name, args)
        if isinstance(client_id, str):
            self._replay[(tool_name, client_id)] = payload
        return payload

    def _dispatch(self, tool_name: str, args: dict[str, Any]) -> Any:
        self._counter += 1
        n = self._counter
        if tool_name == "memory_recall":
            return {"beliefs": []}
        if tool_name == "suggest_next":
            return {"suggestions": [{"family": "value"}], "family_selection_probabilities": {"value": 1.0}}
        if tool_name == "check_novelty":
            return {"N": 0.6}
        if tool_name == "hypothesis_propose":
            hypothesis_id = f"h-{n}"
            self.proposed.append(hypothesis_id)
            return {"hypothesis_id": hypothesis_id, "N": 0.5}
        if tool_name == "register_hypothesis":
            return {"episode_id": f"e-{n}", "protocol_hash": f"ph-{n}", "seed": args["seed"]}
        if tool_name == "backtest_run":
            if self.tombstone_id is not None:
                ack = args.get("acknowledge_tombstone") or {}
                if ack.get("tombstone_id") != self.tombstone_id:
                    raise EvolutionError(f'tombstone pre-check blocked dispatch: {{"tombstone_id": "{self.tombstone_id}"}}')
            if self.backtest_scripts:
                scripted = self.backtest_scripts.pop(0)
                if scripted is None:
                    return {"metrics": {"sr": 1.2, "ic": 0.05}, "trial": {"id": f"t-{n}"}, "simulated": False}
                if isinstance(scripted, Exception):
                    raise scripted
                return scripted
            return {"metrics": {"sr": 1.2, "ic": 0.05}, "trial": {"id": f"t-{n}"}, "simulated": False}
        if tool_name == "significance_gate":
            if self.default_gate is not None:
                return dict(self.default_gate)
            return {"pass": True, "status": "pass", "sr": 1.2, "dsr": 0.9, "p": 0.01, "M_family": 1}
        if tool_name == "log_outcome":
            return {"ok": True}
        if tool_name == "memory_promote":
            return {"belief_id": f"b-{n}", "promotion_status": "auto", "confidence": args["confidence"]}
        raise AssertionError(f"Unexpected tool {tool_name!r}.")


class RecordingMemory:
    """AgentMemory fake recording reads and turn writes."""

    def __init__(self) -> None:
        self.turns: list[dict[str, Any]] = []

    def get_context(self, user_id, *, agent_name=None, thread_id=None, query=None) -> str:
        return "prior: value worked in 2024"

    def search(self, query: str, top_k: int = 5, **kwargs) -> list[dict[str, Any]]:
        return [{"fact": "momentum decayed in 2023"}]

    def add(self, thread_id: str, messages: list[Any], **kwargs) -> None:
        self.turns.append({"thread_id": thread_id, "messages": messages, **kwargs})


class FakeSandbox:
    """In-memory Sandbox double recording writes and commands."""

    def __init__(self, stdout: str = "backtest ok") -> None:
        self.files: dict[str, str] = {}
        self.commands: list[dict[str, Any]] = []
        self.stdout = stdout

    def write_file(self, path: str, content: str, append: bool = False) -> None:
        self.files[path] = content

    def execute_command(self, command: str, env=None, timeout=None) -> str:
        self.commands.append({"command": command, "timeout": timeout})
        return self.stdout

    def list_dir(self, path: str, max_depth: int = 2) -> list[str]:
        raise OSError("no outputs dir")


class FakeProvider:
    """Minimal SandboxProvider double over one FakeSandbox."""

    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox
        self.acquired: list[str | None] = []
        self.released: list[str] = []

    def acquire(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        self.acquired.append(thread_id)
        return f"sb-{len(self.acquired)}"

    def get(self, sandbox_id: str):
        return self.sandbox

    def release(self, sandbox_id: str) -> None:
        self.released.append(sandbox_id)


def _models_for(
    ticks: int,
    *,
    decisions: list[bool] | None = None,
    design_params: list[dict[str, Any]] | None = None,
) -> ScriptedModels:
    """Script distinct per-tick payloads (distinct specs avoid reruns)."""
    decisions = decisions if decisions is not None else [True] * ticks
    propose, design, coder, summarize = [], [], [], []
    for tick in range(ticks):
        propose.append(
            json.dumps(
                {
                    "hypothesis": f"tick-{tick}: {21 + tick}d momentum persists in liquid us equities.",
                    "reason": f"Slow information diffusion sustains trends (variant {tick}).",
                    "mechanism": "slow diffusion",
                    "expected_direction": "positive",
                    "horizon": "21d",
                    "falsifier": "IC <= 0 over the next 63 sessions",
                    "scope": "us-equities",
                    "predicted_ic": 0.05,
                    "predicted_sigma": 0.02,
                    "hypothesis_type": "predictive",
                }
            )
        )
        params = {"seed": 7 + tick}
        if design_params is not None and tick < len(design_params):
            params.update(design_params[tick])
        design.append(
            json.dumps(
                {
                    "design": {
                        "hypothesis": f"tick-{tick}: momentum persists.",
                        "methodology": {"universe": "sp500", "variant": tick},
                    },
                    "code": {"artifact": "code-v1"},
                    "data": {},
                    "parameters": params,
                    "environment": {"image": "qf:v1"},
                }
            )
        )
        coder.append(
            json.dumps(
                {
                    "code": {"files": {"run.py": f"print('tick {tick}')"}, "entrypoint": "python run.py"},
                    "summary": f"v{tick}",
                }
            )
        )
        summarize.append(
            json.dumps(
                {
                    "decision": decisions[tick],
                    "reason": f"gate verdict stands for tick {tick}",
                    "observations": "sr stable",
                    "hypothesis_evaluation": "confirmed" if decisions[tick] else "rejected",
                    "new_hypothesis": None,
                    "confidence": 0.9,
                }
            )
        )
    return ScriptedModels({"propose": propose, "design": design, "coder": coder, "summarize": summarize})


def _rig(
    ticks: int,
    *,
    max_iterations: int | None = None,
    transport: GovernorTransport | None = None,
    models: ScriptedModels | None = None,
    events: list[str] | None = None,
) -> SimpleNamespace:
    shared_events: list[str] = events if events is not None else []
    store = InMemoryExperimentStore(shared_events)
    governor = transport or GovernorTransport(shared_events)
    governor.events = shared_events
    memory = RecordingMemory()
    sandbox = FakeSandbox()
    provider = FakeProvider(sandbox)
    config = EvolutionConfig(max_iterations=max_iterations if max_iterations is not None else ticks, enabled=True)
    services = EvolutionServices(
        spark_transport=governor,
        kb_store=store,
        project_id=PROJECT_ID,
        models=models or _models_for(ticks),
        executor=QuantflowSandboxExecutor(provider),
        memory=memory,
    )
    wired = build_loop(config, services)
    return SimpleNamespace(
        config=config,
        store=store,
        memory=memory,
        transport=governor,
        sandbox=sandbox,
        provider=provider,
        wired=wired,
        runner=EvolutionRunner(wired),
        events=shared_events,
    )


def _experiment_ids(rig: SimpleNamespace, ticks: int) -> list[str]:
    return [rig.wired.state.get_experiment_id(RUN_ID, tick) for tick in range(ticks)]


@pytest.mark.asyncio
async def test_three_ticks_record_governed_chain_with_kb_lineage():
    rig = _rig(3)
    steps = [await rig.runner.tick(RUN_ID) for _ in range(3)]

    assert [step.iteration for step in steps] == [0, 1, 2]
    assert all(step.feedback is not None and step.feedback.decision for step in steps)
    trace = rig.wired.trace_store.get_trace(RUN_ID)
    assert trace.dag_parent == [NEW_ROOT, (0,), (1,)]

    experiment_ids = _experiment_ids(rig, 3)
    assert len(rig.store.records) == 3
    records = [rig.store.records[eid] for eid in experiment_ids]
    for record in records:
        assert record.created_by_run_id == RUN_ID
        assert record.project_id == PROJECT_ID
        assert record.outcome == "success"
        assert record.status == "completed"
    assert records[0].parent_experiment_id is None
    assert records[1].parent_experiment_id == records[0].id
    assert records[2].parent_experiment_id == records[1].id

    assert len(rig.memory.turns) == 3
    assert rig.provider.acquired == [RUN_ID, RUN_ID, RUN_ID]
    assert len(rig.provider.released) == 3
    assert steps[0].subject is not None


@pytest.mark.asyncio
async def test_writes_follow_governor_first_ordering():
    rig = _rig(1)
    await rig.runner.tick(RUN_ID)

    assert rig.events == [
        "spark:hypothesis_propose",
        "spark:register_hypothesis",
        "kb:begin",
        "spark:backtest_run",
        "spark:log_outcome",
        "kb:commit",
        "spark:memory_promote",
    ]


@pytest.mark.asyncio
async def test_restart_replays_without_new_writes():
    rig = _rig(2, max_iterations=2)
    first = await rig.runner.run(RUN_ID)
    assert first.iterations_run == 2
    assert first.stopped_reason == "max-iterations"
    records_before = dict(rig.store.records)
    calls_before = len(rig.transport.calls)

    second = await rig.runner.run(RUN_ID)

    assert second.iterations_run == 0
    assert second.steps_recorded == 2
    assert rig.store.records == records_before
    assert len(rig.transport.calls) == calls_before
    assert len(rig.provider.acquired) == 2


@pytest.mark.asyncio
async def test_backtest_failure_closes_out_with_failure_record_and_no_promote():
    transport = GovernorTransport()
    transport.backtest_scripts = [None, RuntimeError("provider exploded")]
    rig = _rig(2, transport=transport, models=_models_for(2, decisions=[True, False]))

    first = await rig.runner.tick(RUN_ID)
    second = await rig.runner.tick(RUN_ID)

    assert first.feedback.decision is True
    assert second.feedback is not None and second.feedback.decision is False
    assert len(rig.store.failures) == 1
    failure = next(iter(rig.store.failures.values()))
    assert failure.experiment_id == rig.wired.state.get_experiment_id(RUN_ID, 1)
    assert "provider exploded" in failure.notes
    failed_record = rig.store.records[failure.experiment_id]
    assert failed_record.outcome == "failure"
    assert failed_record.status == "failed"
    outcomes = [args["outcome"] for name, args in transport.calls if name == "log_outcome"]
    assert outcomes == ["success", "failure"]
    assert [name for name, _ in transport.calls].count("memory_promote") == 1


@pytest.mark.asyncio
async def test_tombstone_block_retries_with_recorded_ack():
    transport = GovernorTransport()
    transport.tombstone_id = "tb-1"
    models = _models_for(1, design_params=[{"tombstone_acknowledgement": {"tombstone_id": "tb-1", "rationale": "narrower universe avoids the dead family"}}])
    rig = _rig(1, transport=transport, models=models, max_iterations=1)

    step = await rig.runner.tick(RUN_ID)

    backtests = [args for name, args in transport.calls if name == "backtest_run"]
    assert len(backtests) == 2
    assert "acknowledge_tombstone" not in backtests[0]
    assert backtests[1]["acknowledge_tombstone"]["tombstone_id"] == "tb-1"
    assert step.feedback is not None and step.feedback.decision is True


@pytest.mark.asyncio
async def test_tombstone_block_without_ack_closes_out_as_failure():
    transport = GovernorTransport()
    transport.tombstone_id = "tb-1"
    rig = _rig(1, transport=transport, models=_models_for(1, decisions=[False]), max_iterations=1)

    step = await rig.runner.tick(RUN_ID)

    assert [name for name, _ in transport.calls].count("backtest_run") == 1
    assert step.feedback is not None and step.feedback.decision is False
    assert len(rig.store.failures) == 1


@pytest.mark.asyncio
async def test_second_tick_threads_parent_hypothesis_and_kb_lineage():
    rig = _rig(2)
    await rig.runner.tick(RUN_ID)
    await rig.runner.tick(RUN_ID)

    proposes = [args for name, args in rig.transport.calls if name == "hypothesis_propose"]
    assert len(proposes) == 2
    assert proposes[0].get("parent_hypothesis_ids") in (None, [])
    assert proposes[1]["parent_hypothesis_ids"] == [rig.transport.proposed[0]]
    assert "decision=True" in rig.wired.loop.proposer.last_context.lineage


@pytest.mark.asyncio
async def test_kb_writes_replay_on_identical_keys():
    rig = _rig(1)
    spec = ExperimentSpec(
        design={"hypothesis": "replay probe", "methodology": {"universe": "sp500"}},
        code={},
        data={},
        parameters={},
        environment={},
    )
    begin_key = derive_key(RUN_ID, 3000, 0, {"probe": "begin"})
    commit_key = derive_key(RUN_ID, 5000, 0, {"probe": "commit"})

    first = await rig.wired.knowledge.begin_experiment(spec, run_id=RUN_ID, idempotency_key=begin_key)
    replay = await rig.wired.knowledge.begin_experiment(spec, run_id=RUN_ID, idempotency_key=begin_key)
    assert replay.id == first.id
    assert len(rig.store.records) == 1

    committed = await rig.wired.knowledge.commit_experiment(experiment_id=first.id, metrics={"sr": 1.0}, outcome="success", idempotency_key=commit_key)
    recommitted = await rig.wired.knowledge.commit_experiment(experiment_id=first.id, metrics={"sr": 1.0}, outcome="success", idempotency_key=commit_key)
    assert recommitted.id == committed.id
    assert recommitted.outcome == "success"


@pytest.mark.asyncio
async def test_gate_rejection_yields_inconclusive_without_promote():
    transport = GovernorTransport()
    transport.default_gate = {"pass": False, "status": "fail", "sr": 0.2, "dsr": -0.1, "p": 0.4, "M_family": 3}
    rig = _rig(1, transport=transport, models=_models_for(1, decisions=[False]), max_iterations=1)

    step = await rig.runner.tick(RUN_ID)

    assert step.feedback is not None and step.feedback.decision is False
    record = rig.store.records[rig.wired.state.get_experiment_id(RUN_ID, 0)]
    assert record.outcome == "inconclusive"
    assert record.status == "completed"
    assert "memory_promote" not in [name for name, _ in transport.calls]


@pytest.mark.asyncio
async def test_run_summary_names_sota_winner():
    rig = _rig(2, max_iterations=2)
    summary = await rig.runner.run(RUN_ID)

    assert summary.stopped_reason == "max-iterations"
    assert summary.steps_recorded == 2
    trace = rig.wired.trace_store.get_trace(RUN_ID)
    assert summary.sota_step_id == trace.hist[-1].step_id


@pytest.mark.asyncio
async def test_sandbox_runs_use_canonical_workdir_and_manifest():
    rig = _rig(1)
    await rig.runner.tick(RUN_ID)

    workdir = evolution_workdir(RUN_ID, 0)
    assert rig.sandbox.commands[0]["command"] == f"cd {workdir} && python run.py"
    assert rig.sandbox.files[f"{workdir}/run.py"] == "print('tick 0')"
    manifest = json.loads(rig.sandbox.files[f"{workdir}/{PROVIDER_MANIFEST_FILENAME}"])
    assert manifest["provider"] == "alpha_engine"
    assert manifest["datasets"] == []
    assert manifest["run_id"] == RUN_ID


@pytest.mark.asyncio
async def test_quantflow_executor_satisfies_runner_protocol():
    rig = _rig(1)
    coded = CodedExperiment(spec=_spec(), code={"files": {"run.py": "x"}, "entrypoint": "python run.py"})
    result = await rig.wired.loop.runner._executor.run(coded, _ctx_for(rig))
    assert result.error is None


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        design={"hypothesis": "probe", "methodology": {}},
        code={},
        data={},
        parameters={},
        environment={},
    )


def _ctx_for(rig: SimpleNamespace) -> StageContext:
    import uuid as uuid_module

    from deerflow.evolution.trace import Trace

    return StageContext(
        run_id=RUN_ID,
        iteration=0,
        step_id=str(uuid_module.uuid4()),
        trace=Trace(),
        config=rig.config,
    )

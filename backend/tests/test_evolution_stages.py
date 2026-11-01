"""Unit + wiring tests for deerflow.evolution.stages (offline).

Each stage runs against scripted fakes for the spark/KB/memory/model
boundaries; the final tests wire the real stages into the real
``EvolutionLoop`` to prove the full propose → design → code → run →
feedback → record pipeline plus governor-first write ordering.
"""

import json
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from deerflow.evolution.config import EvolutionConfig
from deerflow.evolution.errors import DuplicateRunError, EvolutionError
from deerflow.evolution.knowledge_adapter import ProposalPacket, RecheckResult
from deerflow.evolution.loop import EvolutionLoop, StageContext
from deerflow.evolution.models import (
    CodedExperiment,
    EvoStep,
    ExperimentFeedback,
    ExperimentOutcome,
    ExperimentSpec,
    Hypothesis,
    HypothesisFeedback,
)
from deerflow.evolution.selectors import AncestorSOTASelector, LatestSOTASelector
from deerflow.evolution.stages.design import ExperimentDesignerImpl
from deerflow.evolution.stages.execute import CoderImpl, RunnerImpl, SandboxResult
from deerflow.evolution.stages.feedback import FeedbackSynthesizerImpl
from deerflow.evolution.stages.llm import parse_json_object
from deerflow.evolution.stages.propose import C6Payload, HypothesisGenerator, validate_c6
from deerflow.evolution.stages.state import IterationState
from deerflow.evolution.trace import InMemoryTraceStore, Trace

RUN_ID = "44444444-4444-4444-4444-444444444444"
PROJECT_ID = "55555555-5555-5555-5555-555555555555"


def _config(**overrides: Any) -> EvolutionConfig:
    values: dict[str, Any] = {"max_iterations": 5, "min_confidence_to_promote": 0.7}
    values.update(overrides)
    return EvolutionConfig(**values)


def _ctx(iteration: int = 0, trace: Trace | None = None) -> StageContext:
    return StageContext(
        run_id=RUN_ID,
        iteration=iteration,
        step_id=str(uuid.uuid4()),
        trace=trace or Trace(),
        config=_config(),
    )


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


def _spec_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "design": {"hypothesis": "21d momentum persists.", "methodology": {"universe": "sp500"}},
        "code": {"artifact": "code-v1"},
        "data": {},
        "parameters": {"seed": 7},
        "environment": {"image": "qf:v1"},
    }
    payload.update(overrides)
    return payload


class ScriptedModels:
    """ChatModelFactory fake: scripted response queue per role."""

    def __init__(self, responses: dict[str, list[str]] | None = None):
        self.responses = {role: list(items) for role, items in (responses or {}).items()}
        self.calls: list[tuple[str, str, str]] = []

    async def generate(self, *, role: str, system: str, user: str) -> str:
        self.calls.append((role, system, user))
        queue = self.responses.get(role, [])
        if not queue:
            raise AssertionError(f"No scripted response for role {role!r}.")
        return queue.pop(0)


class FakeKnowledge:
    """KnowledgeAdapter fake with call log and scripted recheck."""

    def __init__(self, order: list[str] | None = None):
        self.order = order if order is not None else []
        self.project_id = PROJECT_ID
        self.recheck_error: Exception | None = None
        self.packet = ProposalPacket(text="KB PACKET", refs=("kb-1",), intent_topic="t")
        self.begun: list[dict[str, Any]] = []
        self.committed: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []
        self.assumptions: list[dict[str, Any]] = []
        self._counter = 0

    async def recheck(self, spec: ExperimentSpec, **kwargs):
        _ = kwargs
        if self.recheck_error is not None:
            raise self.recheck_error
        return RecheckResult(execution_hash=spec.execution_hash(), family_hash=spec.family_hash())

    async def begin_experiment(self, spec, **kwargs):
        _ = spec
        self.order.append("kb:begin")
        self._counter += 1
        record = SimpleNamespace(id=f"exp-{self._counter}", **kwargs)
        self.begun.append({"id": record.id, **kwargs})
        return record

    async def commit_experiment(self, **kwargs):
        self.order.append("kb:commit")
        self.committed.append(dict(kwargs))
        return SimpleNamespace(id=kwargs.get("experiment_id"), status="completed")

    async def record_failure(self, **kwargs):
        self.order.append("kb:failure")
        self.failures.append(dict(kwargs))
        return SimpleNamespace(id="fail-1", **kwargs)

    async def record_assumption(self, **kwargs):
        self.order.append("kb:assumption")
        self.assumptions.append(dict(kwargs))
        return SimpleNamespace(id=f"asm-{len(self.assumptions)}", **kwargs)

    async def build_proposal_context(self, topic: str, **kwargs):
        _ = kwargs
        return ProposalPacket(text=self.packet.text, refs=self.packet.refs, intent_topic=topic)


class FakeSpark:
    """SparkClient fake with call log and scripted method results."""

    def __init__(self, order: list[str] | None = None):
        self.order = order if order is not None else []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.scripts: dict[str, Any] = {}
        self.propose_result = {"hypothesis_id": "h-1", "N": 0.6, "S": None}
        self.register_result = {"episode_id": "e-1", "protocol_hash": "ph-1"}
        self.gate_passed = True
        self.promote_status = "auto"

    def _record(self, name: str, kwargs: dict[str, Any]) -> None:
        self.order.append(f"spark:{name}")
        self.calls.append((name, dict(kwargs)))

    async def propose(self, **kwargs):
        self._record("propose", kwargs)
        if "propose" in self.scripts:
            scripted = self.scripts["propose"]
            if isinstance(scripted, Exception):
                raise scripted
            return scripted
        return SimpleNamespace(
            hypothesis_id=self.propose_result["hypothesis_id"],
            novelty=self.propose_result.get("N"),
            surprise=self.propose_result.get("S"),
        )

    async def register(self, **kwargs):
        self._record("register", kwargs)
        return SimpleNamespace(
            episode_id=self.register_result["episode_id"],
            protocol_hash=self.register_result["protocol_hash"],
        )

    async def log_outcome(self, **kwargs):
        self._record("log_outcome", kwargs)
        return {"episode_id": kwargs.get("episode_id"), "status": "ok"}

    async def significance_gate(self, episode_id: str):
        self._record("gate", {"episode_id": episode_id})
        return SimpleNamespace(
            passed=self.gate_passed,
            status="validated" if self.gate_passed else "inconclusive",
            sr=1.5,
            dsr=0.96,
            p_value=0.04,
            m_family=12,
            promotion_blockers=(),
        )

    async def check_novelty(self, idea_or_code: str, **kwargs):
        self._record("novelty", {"idea_or_code": idea_or_code, **kwargs})
        if "check_novelty" in self.scripts:
            scripted = self.scripts["check_novelty"]
            if isinstance(scripted, Exception):
                raise scripted
            return scripted
        return SimpleNamespace(novelty=0.62, surprise=None)

    async def suggest_next(self, **kwargs):
        self._record("suggest", dict(kwargs))
        return SimpleNamespace(suggestions=[{"family": "f-a"}], selection_probabilities={"f-a": 1.0})

    async def memory_recall(self, query: str, **kwargs):
        self._record("recall", {"query": query, **kwargs})
        return (SimpleNamespace(claim="momentum works", effective_confidence=0.8),)

    async def memory_promote(self, **kwargs):
        self._record("promote", kwargs)
        return SimpleNamespace(belief_id="b-1", promotion_status=self.promote_status, contradictions=(), confidence=kwargs.get("confidence"))

    async def backtest_run(self, **kwargs):
        self._record("backtest", kwargs)
        if "backtest_run" in self.scripts:
            scripted = self.scripts["backtest_run"]
            if isinstance(scripted, Exception):
                raise scripted
            if callable(scripted):
                return scripted(kwargs)
            return scripted
        return SimpleNamespace(metrics={"sr": 1.2, "T": 252}, trial={"trial_id": "t-1"}, simulated=True)


class FakeMemory:
    """MemoryAdapter fake with canned recall and a write log."""

    def __init__(self):
        self.context = "MEMORY CONTEXT"
        self.turns: list[dict[str, Any]] = []

    async def recall_context(self, topic: str) -> str:
        _ = topic
        return self.context

    async def record_turn(self, **kwargs):
        self.turns.append(dict(kwargs))


class FakeExecutor:
    """SandboxExecutor fake with a canned result."""

    def __init__(self, result: SandboxResult | None = None):
        self.result = result or SandboxResult(stdout="sharpe=1.2\n")
        self.calls = 0

    async def run(self, coded: CodedExperiment, ctx: StageContext) -> SandboxResult:
        _ = (coded, ctx)
        self.calls += 1
        return self.result


def _stages(models: ScriptedModels, order: list[str], state: IterationState | None = None):
    knowledge = FakeKnowledge(order)
    spark = FakeSpark(order)
    memory = FakeMemory()
    shared = state or IterationState()
    config = _config()
    proposer = HypothesisGenerator(knowledge=knowledge, spark=spark, memory=memory, models=models, config=config, state=shared)
    designer = ExperimentDesignerImpl(knowledge=knowledge, spark=spark, models=models, config=config, state=shared)
    coder = CoderImpl(models)
    runner = RunnerImpl(spark=spark, executor=FakeExecutor(), state=shared)
    synthesizer = FeedbackSynthesizerImpl(knowledge=knowledge, spark=spark, memory=memory, models=models, config=config, state=shared)
    return SimpleNamespace(
        knowledge=knowledge,
        spark=spark,
        memory=memory,
        models=models,
        state=shared,
        config=config,
        proposer=proposer,
        designer=designer,
        coder=coder,
        runner=runner,
        synthesizer=synthesizer,
    )


# -- C6 validation ------------------------------------------------------


def test_validate_c6_accepts_complete_predictive_payload():
    c6 = validate_c6(_c6_payload())
    assert isinstance(c6, C6Payload)
    assert c6.hypothesis_type == "predictive"
    assert c6.to_canonical()["scope"] == "us-equities"


def test_validate_c6_rejects_missing_fields_naming_them():
    with pytest.raises(ValueError, match=r"missing required fields: .*falsifier.*scope"):
        validate_c6(_c6_payload(falsifier="  ", scope=""))


def test_validate_c6_rejects_unknown_hypothesis_type():
    with pytest.raises(ValueError, match="Unknown hypothesis_type"):
        validate_c6(_c6_payload(hypothesis_type="astrology"))


def test_validate_c6_requires_dag_and_adjustment_set_iff_causal():
    with pytest.raises(ValueError, match=r"missing required fields: .*adjustment_set.*dag"):
        validate_c6(_c6_payload(hypothesis_type="causal"))
    causal = validate_c6(_c6_payload(hypothesis_type="causal", dag={"nodes": ["a", "b"]}, adjustment_set=["a"]))
    assert causal.adjustment_set == ("a",)
    assert causal.dag == {"nodes": ["a", "b"]}


def test_parse_json_object_strips_fences_and_rejects_non_objects():
    assert parse_json_object('```json\n{"a": 1}\n```', role="propose") == {"a": 1}
    with pytest.raises(EvolutionError, match="valid JSON"):
        parse_json_object("not json", role="propose")
    with pytest.raises(EvolutionError, match="JSON object"):
        parse_json_object("[1, 2]", role="propose")


# -- propose stage --------------------------------------------------------


@pytest.mark.asyncio
async def test_propose_registers_and_records_hypothesis_id():
    order: list[str] = []
    rig = _stages(ScriptedModels({"propose": [json.dumps(_c6_payload())]}), order)
    hypothesis = await rig.proposer.propose(_ctx())
    assert hypothesis.hypothesis.startswith("21d momentum")
    assert "h-1" in hypothesis.concise_knowledge
    assert rig.state.get_hypothesis_id(RUN_ID, 0) == "h-1"
    assert rig.state.get_scope(RUN_ID, 0) == "us-equities"
    propose_calls = [kwargs for name, kwargs in rig.spark.calls if name == "propose"]
    assert len(propose_calls) == 1
    sent = propose_calls[0]
    assert sent["family"] == "quant-alpha"
    assert sent["universe"] == "sp500"
    assert sent["hypothesis_type"] == "predictive"
    assert len(sent["client_id"]) == 64
    # Context assembly order: recall + suggest before the LLM call.
    assert order[:2] == ["spark:recall", "spark:suggest"]
    assert "spark:propose" in order
    prompt_user = rig.models.calls[0][2]
    assert "KB PACKET" in prompt_user
    assert "MEMORY CONTEXT" in prompt_user
    assert "momentum works" in prompt_user


@pytest.mark.asyncio
async def test_propose_rejects_incomplete_c6_before_any_write():
    order: list[str] = []
    rig = _stages(ScriptedModels({"propose": [json.dumps(_c6_payload(falsifier=""))]}), order)
    with pytest.raises(ValueError, match=r"missing required fields"):
        await rig.proposer.propose(_ctx())
    assert "spark:propose" not in order
    assert rig.state.get_hypothesis_id(RUN_ID, 0) is None


@pytest.mark.asyncio
async def test_propose_rejects_model_output_without_hypothesis_or_reason():
    order: list[str] = []
    rig = _stages(ScriptedModels({"propose": [json.dumps(_c6_payload(hypothesis="  "))]}), order)
    with pytest.raises(EvolutionError, match="hypothesis"):
        await rig.proposer.propose(_ctx())
    assert "spark:propose" not in order


@pytest.mark.asyncio
async def test_propose_treats_novelty_failure_as_advisory():
    order: list[str] = []
    rig = _stages(ScriptedModels({"propose": [json.dumps(_c6_payload())]}), order)
    rig.spark.scripts["check_novelty"] = ConnectionError("archive down")
    hypothesis = await rig.proposer.propose(_ctx())
    assert hypothesis.hypothesis
    assert "spark:propose" in order
    assert rig.proposer.last_context is not None
    assert rig.proposer.last_context.novelty is None


@pytest.mark.asyncio
async def test_propose_forwards_parent_hypothesis_ids_from_lineage():
    order: list[str] = []
    rig = _stages(ScriptedModels({"propose": [json.dumps(_c6_payload())]}), order)
    trace = Trace()
    parent = Hypothesis(hypothesis="parent claim", reason="parent reason")
    trace.record(
        EvoStep(
            step_id=str(uuid.uuid4()),
            run_id=RUN_ID,
            iteration=0,
            subject=ExperimentSpec(design={"hypothesis": "parent claim"}),
            hypothesis=parent,
            feedback=ExperimentFeedback(decision=True, reason="accepted"),
        ),
        loop_id=0,
    )
    trace.set_current_selection((0,))
    rig.state.record_proposal(RUN_ID, 0, hypothesis_id="h-parent", scope="us-equities")
    await rig.proposer.propose(_ctx(iteration=1, trace=trace))
    propose_calls = [kwargs for name, kwargs in rig.spark.calls if name == "propose"]
    assert propose_calls[0]["parent_hypothesis_ids"] == ["h-parent"]
    assert "[0] decision=True parent claim" in rig.proposer.last_context.lineage


# -- design stage ---------------------------------------------------------


@pytest.mark.asyncio
async def test_design_registers_governor_first_then_kb():
    order: list[str] = []
    rig = _stages(ScriptedModels({"design": [json.dumps(_spec_payload())]}), order)
    rig.state.record_proposal(RUN_ID, 0, hypothesis_id="h-1", scope="us-equities")
    spec = await rig.designer.design(Hypothesis(hypothesis="h", reason="r"), _ctx())
    assert spec.design["methodology"] == {"universe": "sp500"}
    assert order == ["spark:register", "kb:begin"]
    assert rig.state.get_episode_id(RUN_ID, 0) == "e-1"
    assert rig.state.get_experiment_id(RUN_ID, 0) == "exp-1"
    register_kwargs = [kwargs for name, kwargs in rig.spark.calls if name == "register"][0]
    assert register_kwargs["hypothesis_id"] == "h-1"
    assert register_kwargs["seed"] == rig.state.get_seed(RUN_ID, 0)
    assert len(register_kwargs["client_id"]) == 64
    assert rig.knowledge.begun[0]["status"] == "running"
    assert len(rig.knowledge.begun[0]["idempotency_key"]) == 64


@pytest.mark.asyncio
async def test_design_rerun_raises_before_any_write():
    order: list[str] = []
    rig = _stages(ScriptedModels({"design": [json.dumps(_spec_payload())]}), order)
    rig.state.record_proposal(RUN_ID, 0, hypothesis_id="h-1", scope="us-equities")
    rig.knowledge.recheck_error = DuplicateRunError("ab" * 32, "exp-prior")
    with pytest.raises(DuplicateRunError) as exc_info:
        await rig.designer.design(Hypothesis(hypothesis="h", reason="r"), _ctx())
    assert exc_info.value.existing_experiment_id == "exp-prior"
    assert order == []


@pytest.mark.asyncio
async def test_design_without_proposal_identity_fails_fast():
    order: list[str] = []
    rig = _stages(ScriptedModels({"design": [json.dumps(_spec_payload())]}), order)
    with pytest.raises(EvolutionError, match="propose stage must run first"):
        await rig.designer.design(Hypothesis(hypothesis="h", reason="r"), _ctx())
    assert order == []


@pytest.mark.asyncio
async def test_design_defaults_missing_hypothesis_text_from_proposal():
    order: list[str] = []
    payload = _spec_payload()
    del payload["design"]["hypothesis"]
    rig = _stages(ScriptedModels({"design": [json.dumps(payload)]}), order)
    rig.state.record_proposal(RUN_ID, 0, hypothesis_id="h-1", scope="us-equities")
    spec = await rig.designer.design(Hypothesis(hypothesis="proposal claim", reason="r"), _ctx())
    assert spec.design["hypothesis"] == "proposal claim"


# -- execute stages ---------------------------------------------------------


def _coded(**parameter_overrides: Any) -> CodedExperiment:
    parameters = {"seed": 7}
    parameters.update(parameter_overrides)
    spec = ExperimentSpec(
        design={"hypothesis": "h", "methodology": {}},
        code={"artifact": "v1"},
        parameters=parameters,
    )
    return CodedExperiment(
        spec=spec,
        code={"files": {"main.py": "print('hi')"}, "entrypoint": "python main.py"},
        code_change_summary="initial",
    )


@pytest.mark.asyncio
async def test_coder_parses_json_code_contract():
    order: list[str] = []
    rig = _stages(
        ScriptedModels({"coder": [json.dumps({"code": {"files": {"run.py": "x=1"}, "entrypoint": "python run.py"}, "summary": "v1"})]}),
        order,
    )
    coded = await rig.coder.implement(ExperimentSpec(design={"hypothesis": "h"}), _ctx())
    assert coded.code["entrypoint"] == "python run.py"
    assert coded.code_change_summary == "v1"


@pytest.mark.asyncio
async def test_coder_falls_back_to_single_file_on_prose():
    order: list[str] = []
    rig = _stages(ScriptedModels({"coder": ["print('just code')"]}), order)
    coded = await rig.coder.implement(ExperimentSpec(design={"hypothesis": "h"}), _ctx())
    assert coded.code["files"] == {"main.py": "print('just code')"}


@pytest.mark.asyncio
async def test_runner_dispatches_backtest_with_state_seed():
    order: list[str] = []
    rig = _stages(ScriptedModels({}), order)
    rig.state.record_design(RUN_ID, 0, episode_id="e-9", experiment_id="exp-9", protocol_hash="ph", seed=1234)
    outcome = await rig.runner.execute(_coded(), _ctx())
    assert outcome.error is None
    assert outcome.metrics["sr"] == 1.2
    assert "sharpe=1.2" in outcome.metrics["sandbox_stdout_tail"]
    assert outcome.metrics["simulated"] is True
    backtest_kwargs = [kwargs for name, kwargs in rig.spark.calls if name == "backtest"][0]
    assert backtest_kwargs["episode_id"] == "e-9"
    assert backtest_kwargs["spec"]["seed"] == 1234
    assert len(backtest_kwargs["client_id"]) == 64


@pytest.mark.asyncio
async def test_runner_retries_tombstone_block_only_with_recorded_ack():
    order: list[str] = []
    rig = _stages(ScriptedModels({}), order)
    rig.state.record_design(RUN_ID, 0, episode_id="e-9", experiment_id="exp-9", protocol_hash="ph", seed=1)
    block = EvolutionError('tombstone pre-check blocked dispatch: cos=0.99. matched tombstone: {"tombstone_id": "tb-1"}')
    calls: list[dict[str, Any]] = []

    def scripted_backtest(kwargs: dict[str, Any]):
        calls.append(kwargs)
        if len(calls) == 1:
            raise block
        return SimpleNamespace(metrics={"sr": 0.8}, trial={}, simulated=False)

    rig.spark.scripts["backtest_run"] = scripted_backtest
    coded = _coded(tombstone_acknowledgement={"tombstone_id": "tb-1", "rationale": "narrower universe avoids the dead family"})
    outcome = await rig.runner.execute(coded, _ctx())
    assert outcome.error is None
    assert outcome.metrics["sr"] == 0.8
    assert len(calls) == 2
    assert calls[1]["acknowledge_tombstone"] == {"tombstone_id": "tb-1", "rationale": "narrower universe avoids the dead family"}


@pytest.mark.asyncio
async def test_runner_tombstone_block_without_ack_becomes_error_outcome():
    order: list[str] = []
    rig = _stages(ScriptedModels({}), order)
    rig.state.record_design(RUN_ID, 0, episode_id="e-9", experiment_id="exp-9", protocol_hash="ph", seed=1)
    rig.spark.scripts["backtest_run"] = EvolutionError('tombstone pre-check blocked dispatch: {"tombstone_id": "tb-1"}')
    outcome = await rig.runner.execute(_coded(), _ctx())
    assert outcome.failed
    assert "tombstone" in (outcome.error or "")


@pytest.mark.asyncio
async def test_runner_sandbox_error_skips_backtest_and_becomes_outcome():
    order: list[str] = []
    rig = _stages(ScriptedModels({}), order)
    rig.state.record_design(RUN_ID, 0, episode_id="e-9", experiment_id="exp-9", protocol_hash="ph", seed=1)
    rig.runner._executor = FakeExecutor(SandboxResult(stdout="", error="exit code 1: KeyError"))
    outcome = await rig.runner.execute(_coded(), _ctx())
    assert outcome.failed
    assert outcome.error_type == "SandboxError"
    assert "backtest" not in [name for name, _ in rig.spark.calls]


@pytest.mark.asyncio
async def test_runner_converts_unexpected_errors_to_outcomes():
    order: list[str] = []
    rig = _stages(ScriptedModels({}), order)
    rig.state.record_design(RUN_ID, 0, episode_id="e-9", experiment_id="exp-9", protocol_hash="ph", seed=1)
    rig.spark.scripts["backtest_run"] = RuntimeError("provider exploded")
    outcome = await rig.runner.execute(_coded(), _ctx())
    assert outcome.failed
    assert outcome.error_type == "RuntimeError"


# -- feedback stage ---------------------------------------------------------


def _summary_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "decision": True,
        "reason": "gate passed with margin",
        "observations": "sr stable across folds",
        "hypothesis_evaluation": "confirmed",
        "new_hypothesis": None,
        "confidence": 0.9,
    }
    payload.update(overrides)
    return payload


def _wired_state() -> IterationState:
    state = IterationState()
    state.record_proposal(RUN_ID, 0, hypothesis_id="h-1", scope="us-equities")
    state.record_design(RUN_ID, 0, episode_id="e-1", experiment_id="exp-1", protocol_hash="ph", seed=11)
    return state


@pytest.mark.asyncio
async def test_feedback_success_path_orders_log_outcome_before_commit_and_promotes():
    order: list[str] = []
    rig = _stages(ScriptedModels({"summarize": [json.dumps(_summary_payload())]}), order, _wired_state())
    outcome = ExperimentOutcome(metrics={"sr": 1.5}, artifacts=())
    feedback = await rig.synthesizer.synthesize(
        hypothesis=Hypothesis(hypothesis="h", reason="r"),
        spec=ExperimentSpec(design={"hypothesis": "h"}),
        coded=None,
        outcome=outcome,
        ctx=_ctx(),
    )
    assert isinstance(feedback, HypothesisFeedback)
    assert feedback.decision is True
    assert feedback.observations == "sr stable across folds"
    assert order == ["spark:gate", "spark:log_outcome", "kb:commit", "spark:promote"]
    log_kwargs = [kwargs for name, kwargs in rig.spark.calls if name == "log_outcome"][0]
    assert log_kwargs["outcome"] == "success"
    assert log_kwargs["episode_id"] == "e-1"
    promote_kwargs = [kwargs for name, kwargs in rig.spark.calls if name == "promote"][0]
    assert promote_kwargs["episode_ids"] == ["e-1"]
    assert promote_kwargs["confidence"] == 0.9
    assert promote_kwargs["scope"] == "us-equities"
    assert rig.memory.turns[0]["thread_id"] == RUN_ID


@pytest.mark.asyncio
async def test_feedback_failure_path_skips_gate_and_promotion():
    order: list[str] = []
    rig = _stages(ScriptedModels({"summarize": [json.dumps(_summary_payload(decision=True))]}), order, _wired_state())
    outcome = ExperimentOutcome(metrics={}, error="KeyError: 'close'", error_type="KeyError")
    feedback = await rig.synthesizer.synthesize(
        hypothesis=Hypothesis(hypothesis="h", reason="r"),
        spec=ExperimentSpec(design={"hypothesis": "h"}),
        coded=None,
        outcome=outcome,
        ctx=_ctx(),
    )
    assert feedback.decision is False
    assert order == ["spark:log_outcome", "kb:failure", "kb:commit"]
    assert rig.knowledge.failures[0]["failure_class"] == "code"
    commit_kwargs = rig.knowledge.committed[0]
    assert commit_kwargs["outcome"] == "failure"
    assert commit_kwargs["failure_class"] == "code"
    log_kwargs = [kwargs for name, kwargs in rig.spark.calls if name == "log_outcome"][0]
    assert log_kwargs["outcome"] == "failure"
    assert "KeyError" in (log_kwargs["failure_reason"] or "")


@pytest.mark.asyncio
async def test_feedback_low_confidence_skips_promotion():
    order: list[str] = []
    rig = _stages(ScriptedModels({"summarize": [json.dumps(_summary_payload(confidence=0.2))]}), order, _wired_state())
    await rig.synthesizer.synthesize(
        hypothesis=Hypothesis(hypothesis="h", reason="r"),
        spec=ExperimentSpec(design={"hypothesis": "h"}),
        coded=None,
        outcome=ExperimentOutcome(metrics={"sr": 1.1}),
        ctx=_ctx(),
    )
    assert "spark:promote" not in order


@pytest.mark.asyncio
async def test_feedback_rejected_model_verdict_logs_inconclusive_without_promote():
    order: list[str] = []
    rig = _stages(ScriptedModels({"summarize": [json.dumps(_summary_payload(decision=False, confidence=0.95))]}), order, _wired_state())
    rig.spark.gate_passed = False
    await rig.synthesizer.synthesize(
        hypothesis=Hypothesis(hypothesis="h", reason="r"),
        spec=ExperimentSpec(design={"hypothesis": "h"}),
        coded=None,
        outcome=ExperimentOutcome(metrics={"sr": 0.1}),
        ctx=_ctx(),
    )
    log_kwargs = [kwargs for name, kwargs in rig.spark.calls if name == "log_outcome"][0]
    assert log_kwargs["outcome"] == "inconclusive"
    assert "spark:promote" not in order


@pytest.mark.asyncio
async def test_feedback_records_well_formed_assumptions_and_skips_malformed():
    order: list[str] = []
    rig = _stages(
        ScriptedModels(
            {
                "summarize": [
                    json.dumps(
                        _summary_payload(
                            confidence=0.1,
                            assumptions=[
                                {"statement": "costs stay flat", "category": "market"},
                                {"statement": "", "category": "market"},
                                {"statement": "bad category", "category": "vibes"},
                                "not-a-mapping",
                            ],
                        )
                    )
                ]
            }
        ),
        order,
        _wired_state(),
    )
    await rig.synthesizer.synthesize(
        hypothesis=Hypothesis(hypothesis="h", reason="r"),
        spec=ExperimentSpec(design={"hypothesis": "h"}),
        coded=None,
        outcome=ExperimentOutcome(metrics={"sr": 1.1}),
        ctx=_ctx(),
    )
    assert len(rig.knowledge.assumptions) == 1
    assert rig.knowledge.assumptions[0]["statement"] == "costs stay flat"


@pytest.mark.asyncio
async def test_feedback_accepts_pending_human_promotion():
    order: list[str] = []
    rig = _stages(ScriptedModels({"summarize": [json.dumps(_summary_payload())]}), order, _wired_state())
    rig.spark.promote_status = "pending_human"
    feedback = await rig.synthesizer.synthesize(
        hypothesis=Hypothesis(hypothesis="h", reason="r"),
        spec=ExperimentSpec(design={"hypothesis": "h"}),
        coded=None,
        outcome=ExperimentOutcome(metrics={"sr": 1.5}),
        ctx=_ctx(),
    )
    assert feedback.decision is True
    assert "spark:promote" in order


@pytest.mark.asyncio
async def test_feedback_without_design_identities_fails_fast():
    order: list[str] = []
    rig = _stages(ScriptedModels({"summarize": [json.dumps(_summary_payload())]}), order)
    with pytest.raises(EvolutionError, match="design stage must run first"):
        await rig.synthesizer.synthesize(
            hypothesis=Hypothesis(hypothesis="h", reason="r"),
            spec=ExperimentSpec(design={"hypothesis": "h"}),
            coded=None,
            outcome=ExperimentOutcome(metrics={}),
            ctx=_ctx(),
        )
    assert order == []


# -- full-loop wiring ---------------------------------------------------------


def _loop_models() -> ScriptedModels:
    return ScriptedModels(
        {
            "propose": [json.dumps(_c6_payload()), json.dumps(_c6_payload(hypothesis="follow-up: 63d momentum decay."))],
            "design": [json.dumps(_spec_payload()), json.dumps(_spec_payload(parameters={"seed": 8}))],
            "coder": [
                json.dumps({"code": {"files": {"run.py": "x=1"}, "entrypoint": "python run.py"}, "summary": "v1"}),
                json.dumps({"code": {"files": {"run.py": "x=2"}, "entrypoint": "python run.py"}, "summary": "v2"}),
            ],
            "summarize": [json.dumps(_summary_payload()), json.dumps(_summary_payload())],
        }
    )


def _wired_loop(rig: SimpleNamespace, state: Any = None) -> EvolutionLoop:
    return EvolutionLoop(
        config=rig.config,
        trace_store=InMemoryTraceStore(),
        checkpoint_selector=LatestSOTASelector(),
        sota_selector=AncestorSOTASelector(),
        proposer=rig.proposer,
        designer=rig.designer,
        coder=rig.coder,
        runner=rig.runner,
        synthesizer=rig.synthesizer,
        state=state,
    )


@pytest.mark.asyncio
async def test_full_loop_step_wires_all_boundaries_in_order():
    order: list[str] = []
    rig = _stages(_loop_models(), order)
    loop = _wired_loop(rig)
    step = await loop.step(RUN_ID)
    assert step.iteration == 0
    assert step.feedback is not None and step.feedback.decision is True
    assert step.hypothesis is not None and step.hypothesis.hypothesis.startswith("21d momentum")
    assert order == [
        "spark:recall",
        "spark:suggest",
        "spark:novelty",
        "spark:propose",
        "spark:register",
        "kb:begin",
        "spark:backtest",
        "spark:gate",
        "spark:log_outcome",
        "kb:commit",
        "spark:promote",
    ]
    assert rig.state.get_hypothesis_id(RUN_ID, 0) == "h-1"
    assert rig.state.get_episode_id(RUN_ID, 0) == "e-1"
    assert rig.state.get_experiment_id(RUN_ID, 0) == "exp-1"
    assert len(rig.memory.turns) == 1


@pytest.mark.asyncio
async def test_recorded_step_carries_sidecar_ids_when_state_attached():
    order: list[str] = []
    rig = _stages(_loop_models(), order)
    loop = _wired_loop(rig, state=rig.state)
    step = await loop.step(RUN_ID)
    assert (step.hypothesis_id, step.episode_id, step.experiment_id) == ("h-1", "e-1", "exp-1")


@pytest.mark.asyncio
async def test_recorded_step_ids_empty_without_state():
    order: list[str] = []
    rig = _stages(_loop_models(), order)
    loop = _wired_loop(rig)
    step = await loop.step(RUN_ID)
    assert (step.hypothesis_id, step.episode_id, step.experiment_id) == (None, None, None)


def test_hydrate_from_trace_restores_records_and_keeps_live_values():
    trace = Trace()
    trace.record(
        EvoStep(
            step_id=str(uuid.uuid4()),
            run_id=RUN_ID,
            iteration=0,
            subject=ExperimentSpec(design={"hypothesis": "h0", "methodology": {}}),
            hypothesis=Hypothesis(hypothesis="h0", reason="r"),
            hypothesis_id="h-1",
            episode_id="e-1",
            experiment_id="exp-1",
        ),
        loop_id=0,
    )
    trace.record(
        EvoStep(
            step_id=str(uuid.uuid4()),
            run_id=RUN_ID,
            iteration=1,
            subject=ExperimentSpec(design={"hypothesis": "h1", "methodology": {}}),
        ),
        loop_id=1,
    )
    state = IterationState()
    assert state.hydrate_from_trace(trace) == 1
    assert state.get_hypothesis_id(RUN_ID, 0) == "h-1"
    assert state.get_episode_id(RUN_ID, 0) == "e-1"
    assert state.get_experiment_id(RUN_ID, 0) == "exp-1"
    assert state.get_experiment_id(RUN_ID, 1) is None
    state.record_design(RUN_ID, 0, episode_id="e-live", experiment_id="exp-live", protocol_hash="ph", seed=1)
    assert state.hydrate_from_trace(trace) == 0
    assert state.get_experiment_id(RUN_ID, 0) == "exp-live"


@pytest.mark.asyncio
async def test_second_tick_threads_parent_lineage():
    order: list[str] = []
    rig = _stages(_loop_models(), order)
    loop = _wired_loop(rig)
    await loop.step(RUN_ID)
    second = await loop.step(RUN_ID)
    assert second.iteration == 1
    propose_calls = [kwargs for name, kwargs in rig.spark.calls if name == "propose"]
    assert len(propose_calls) == 2
    assert propose_calls[1]["parent_hypothesis_ids"] == ["h-1"]
    assert "decision=True" in rig.proposer.last_context.lineage


@pytest.mark.asyncio
async def test_loop_step_converts_stage_crash_to_rejection_without_close_out_gap():
    order: list[str] = []
    models = _loop_models()
    models.responses["design"] = ["not json at all"]
    rig = _stages(models, order)
    loop = _wired_loop(rig)
    step = await loop.step(RUN_ID)
    assert step.feedback is not None and step.feedback.decision is False
    # Propose wrote its hypothesis; nothing downstream ran (no half-registered trial).
    assert order == ["spark:recall", "spark:suggest", "spark:novelty", "spark:propose"]

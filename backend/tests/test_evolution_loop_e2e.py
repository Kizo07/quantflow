"""End-to-end tests for deerflow.evolution.loop (offline, deterministic).

A scripted stub stage-set (canned hypotheses/specs/outcomes/gate verdicts),
the in-memory trace store, and a recording span emitter drive the full
``propose → design → code → run → feedback → record → reselect`` pipeline.
"""

import uuid

import pytest

from deerflow.evolution.config import EvolutionConfig
from deerflow.evolution.errors import BudgetExhaustedError, DuplicateRunError
from deerflow.evolution.loop import EvolutionLoop, Stage, StageContext
from deerflow.evolution.models import (
    CodedExperiment,
    ExperimentFeedback,
    ExperimentOutcome,
    ExperimentSpec,
    Hypothesis,
    HypothesisFeedback,
)
from deerflow.evolution.selectors import AncestorSOTASelector, LatestSOTASelector
from deerflow.evolution.trace import NEW_ROOT, InMemoryTraceStore

RUN_ID = "11111111-1111-1111-1111-111111111111"


class ScriptedStages:
    """Deterministic stub stages with canned per-iteration payloads."""

    def __init__(self, *, fail_at: dict[int, str] | None = None, duplicate_at: int | None = None):
        self.fail_at = fail_at or {}
        self.duplicate_at = duplicate_at
        self.calls: list[tuple[str, int]] = []

    async def propose(self, ctx: StageContext) -> Hypothesis:
        self.calls.append(("propose", ctx.iteration))
        self._maybe_fail(ctx.iteration, "propose")
        return Hypothesis(
            hypothesis=f"Hypothesis {ctx.iteration}: momentum persists.",
            reason="Prior step accepted; continue the lineage.",
        )

    async def design(self, hypothesis: Hypothesis, ctx: StageContext) -> ExperimentSpec:
        self.calls.append(("design", ctx.iteration))
        self._maybe_fail(ctx.iteration, "design")
        if self.duplicate_at == ctx.iteration:
            raise DuplicateRunError("deadbeef" * 8, "existing-experiment-id")
        return ExperimentSpec(
            design={"hypothesis": hypothesis.hypothesis, "methodology": {"universe": "sp500", "iteration": ctx.iteration}},
            code={"artifact": f"code-{ctx.iteration}"},
            data={"datasets": []},
            parameters={"seed": ctx.iteration},
            environment={"image": "qf:v1"},
        )

    async def implement(self, spec: ExperimentSpec, ctx: StageContext) -> CodedExperiment:
        self.calls.append(("code", ctx.iteration))
        self._maybe_fail(ctx.iteration, "code")
        return CodedExperiment(spec=spec, code={"entry": "main.py"}, code_change_summary=f"impl-{ctx.iteration}")

    async def execute(self, coded: CodedExperiment, ctx: StageContext) -> ExperimentOutcome:
        self.calls.append(("run", ctx.iteration))
        self._maybe_fail(ctx.iteration, "run")
        return ExperimentOutcome(metrics={"ic": 0.05, "iteration": ctx.iteration}, artifacts=[])

    async def synthesize(self, *, hypothesis, spec, coded, outcome, ctx: StageContext) -> ExperimentFeedback:
        self.calls.append(("feedback", ctx.iteration))
        self._maybe_fail(ctx.iteration, "feedback")
        accepted = bool(outcome.metrics.get("ic", 0) > 0)
        return HypothesisFeedback(
            decision=accepted,
            reason=f"Gate {'passed' if accepted else 'failed'} for iteration {ctx.iteration}.",
            observations=f"ic={outcome.metrics.get('ic')}",
            acceptable=accepted,
        )

    def _maybe_fail(self, iteration: int, stage: str) -> None:
        if self.fail_at.get(iteration) == stage:
            raise RuntimeError(f"scripted {stage} failure at iteration {iteration}")


class RecordingSpans:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, int]] = []

    def emit(self, *, run_id, iteration, step_id, stage, status, detail=None) -> None:
        self.events.append((stage, status, iteration))


def _loop(stages: ScriptedStages, **overrides) -> tuple[EvolutionLoop, InMemoryTraceStore, RecordingSpans]:
    store = InMemoryTraceStore()
    spans = RecordingSpans()
    config = EvolutionConfig(max_iterations=overrides.pop("max_iterations", 5))
    loop = EvolutionLoop(
        config=config,
        trace_store=store,
        checkpoint_selector=overrides.pop("checkpoint_selector", LatestSOTASelector()),
        sota_selector=overrides.pop("sota_selector", LatestSOTASelector()),
        proposer=stages,
        designer=stages,
        coder=stages,
        runner=stages,
        synthesizer=stages,
        span_emitter=spans,
        **overrides,
    )
    return loop, store, spans


@pytest.mark.asyncio
async def test_full_run_records_linear_chain_with_consistent_parents():
    stages = ScriptedStages()
    loop, store, spans = _loop(stages, max_iterations=3)

    summary = await loop.run(RUN_ID)

    assert summary.run_id == RUN_ID
    assert summary.iterations_run == 3
    assert summary.steps_recorded == 3
    assert summary.stopped_reason == "max-iterations"

    trace = store.get_trace(RUN_ID)
    assert [step.iteration for step in trace.hist] == [0, 1, 2]
    assert trace.dag_parent == [NEW_ROOT, (0,), (1,)]
    for idx, step in enumerate(trace.hist):
        expected = () if idx == 0 else (trace.hist[idx - 1].step_id,)
        assert step.parent_step_ids == expected
        assert step.run_id == RUN_ID
        assert bool(step.feedback) is True

    recorded = [event for event in spans.events if event[0] == Stage.RECORD and event[1] == "ok"]
    assert len(recorded) == 3
    for stage in (Stage.PROPOSE, Stage.DESIGN, Stage.CODE, Stage.RUN, Stage.FEEDBACK):
        assert sum(1 for event in spans.events if event[0] == stage and event[1] == "ok") == 3


@pytest.mark.asyncio
async def test_restart_replays_without_creating_new_steps():
    stages = ScriptedStages()
    loop, store, _ = _loop(stages, max_iterations=4)
    await loop.run(RUN_ID)
    before = [step.step_id for step in store.get_trace(RUN_ID).hist]
    calls_before = list(stages.calls)

    rerun_summary = await loop.run(RUN_ID)
    assert rerun_summary.iterations_run == 0
    assert rerun_summary.steps_recorded == 4
    assert [step.step_id for step in store.get_trace(RUN_ID).hist] == before
    assert stages.calls == calls_before


@pytest.mark.asyncio
async def test_single_step_scheduler_tick_from_non_interactive_context():
    """One step() runs standalone with no prompt/queue/approval dependency."""
    stages = ScriptedStages()
    loop, store, _ = _loop(stages, max_iterations=10)

    first = await loop.step(RUN_ID)
    assert first.iteration == 0
    second = await loop.step(RUN_ID)
    assert second.iteration == 1
    assert len(store.get_trace(RUN_ID)) == 2


@pytest.mark.asyncio
async def test_step_past_budget_raises_and_run_stops_cleanly():
    stages = ScriptedStages()
    loop, store, _ = _loop(stages, max_iterations=1)
    await loop.step(RUN_ID)
    with pytest.raises(BudgetExhaustedError):
        await loop.step(RUN_ID)
    summary = await loop.run(RUN_ID)
    assert summary.stopped_reason == "max-iterations"
    assert summary.iterations_run == 0


@pytest.mark.asyncio
async def test_stage_exception_becomes_rejecting_feedback_and_records():
    stages = ScriptedStages(fail_at={1: "run"})
    loop, store, spans = _loop(stages, max_iterations=3)
    summary = await loop.run(RUN_ID)

    assert summary.steps_recorded == 3
    trace = store.get_trace(RUN_ID)
    failed = trace.hist[1]
    assert bool(failed.feedback) is False
    assert failed.feedback.error_type == "RuntimeError"
    assert "scripted run failure" in failed.feedback.reason
    assert trace.hist[0].accepted is True
    assert trace.hist[2].accepted is True
    error_spans = [event for event in spans.events if event[0] == Stage.RUN and event[1] == "error"]
    assert len(error_spans) == 1


@pytest.mark.asyncio
async def test_duplicate_run_links_instead_of_rerunning():
    stages = ScriptedStages(duplicate_at=1)
    loop, store, _ = _loop(stages, max_iterations=2)
    await loop.run(RUN_ID)

    trace = store.get_trace(RUN_ID)
    linked = trace.hist[1]
    assert bool(linked.feedback) is False
    assert "existing-experiment-id" in linked.feedback.reason
    assert ("run", 1) not in stages.calls


@pytest.mark.asyncio
async def test_propose_failure_records_nothing_and_raises():
    stages = ScriptedStages(fail_at={0: "propose"})
    loop, store, _ = _loop(stages, max_iterations=2)
    with pytest.raises(RuntimeError, match="scripted propose failure"):
        await loop.step(RUN_ID)
    assert store.get_trace(RUN_ID) is None


@pytest.mark.asyncio
async def test_sota_selector_returns_gated_winner():
    stages = ScriptedStages(fail_at={2: "feedback"})
    loop, store, _ = _loop(
        stages,
        max_iterations=3,
        checkpoint_selector=AncestorSOTASelector(),
        sota_selector=AncestorSOTASelector(),
    )
    summary = await loop.run(RUN_ID)
    trace = store.get_trace(RUN_ID)
    assert summary.sota_step_id == trace.hist[1].step_id
    assert trace.dag_parent[2] == (1,)


@pytest.mark.asyncio
async def test_stop_rule_and_wall_clock_end_the_run():
    stages = ScriptedStages()
    loop, _, _ = _loop(stages, max_iterations=10, should_stop=lambda: True)
    summary = await loop.run(RUN_ID)
    assert summary.stopped_reason == "stop-requested"
    assert summary.iterations_run == 0

    readings = iter([100.0, 100.0, 200.0])
    stages2 = ScriptedStages()
    loop2, _, _ = _loop(stages2, max_iterations=10)
    loop2.config.max_wall_clock_seconds = 10.0
    loop2.clock = lambda: next(readings)
    summary2 = await loop2.run(str(uuid.uuid4()))
    assert summary2.stopped_reason == "wall-clock"
    assert summary2.iterations_run == 1


@pytest.mark.asyncio
async def test_run_rejects_non_uuid_run_id():
    stages = ScriptedStages()
    loop, _, _ = _loop(stages)
    with pytest.raises(ValueError):
        await loop.run("not-a-uuid")

"""Async evolution-loop orchestrator: propose → design → execute → feedback → record → reselect.

:class:`EvolutionLoop` owns step sequencing, idempotency-key derivation,
exception→feedback conversion, and span emission. It performs no I/O of its
own: stages arrive as caller-supplied protocols (later components bind them
to spark-mcp, the Knowledge Plane, agent memory, and LangChain model
profiles), persistence arrives as a
:class:`deerflow.evolution.trace.TraceStore`, and lineage branching arrives
as pure selectors.

Execution contract:

* :meth:`EvolutionLoop.step` runs exactly one iteration — the scheduled-tick
  entry point. Non-interactive by construction: no queues, no prompts, no
  approval waits exist anywhere in this path.
* :meth:`EvolutionLoop.run` repeats :meth:`step` until the budget
  (``max_iterations``, wall-clock, or stop rule) exhausts, then returns a
  :class:`RunSummary`.
* Every iteration reselects the attach point first (via the checkpoint
  selector), so standalone ticks and continuous runs share one lineage rule.
  Recording advances the current selection to the new step, so the next
  reselect walks from the frontier unless a selector branches elsewhere.
* A stage failure after a hypothesis exists still records a step with
  ``decision=False`` feedback (mirroring the ``running → coding → exp_gen``
  fallback chain); only a ``propose`` failure records nothing and raises,
  since there is no subject to attach.
* ``DuplicateRunError`` from any post-propose stage becomes ``decision=False``
  feedback that links to the prior run instead of rerunning it.
* Restart replay: already-recorded iterations are never re-run. A crash
  between ``record`` and ``put_trace`` loses at most the in-memory step;
  the retry reuses the same derived keys so downstream writers replay
  instead of duplicating.

Idempotency keys derive deterministically from
:func:`deerflow.knowledge.hashing.make_idempotency_key` as
``SHA256(run_id || sequence || canonical_payload)`` under the
``quantflow.idempotency/v1`` domain, with one sequence base per write
(``1000`` for propose through ``7000`` for promote, plus ``2500`` for
governed backtest dispatch, each ``+ iteration``) per the plan's
data-model mapping. Stages derive their own keys over real payloads;
the loop core keeps no key bundle.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from deerflow.evolution.config import EvolutionConfig
from deerflow.evolution.errors import BudgetExhaustedError, DuplicateRunError
from deerflow.evolution.models import CodedExperiment, EvoStep, ExperimentFeedback, ExperimentOutcome, ExperimentSpec, Hypothesis, HypothesisFeedback
from deerflow.evolution.selectors import CheckpointSelector, SOTASelector
from deerflow.evolution.trace import Trace, TraceStore
from deerflow.knowledge.hashing import make_idempotency_key

if TYPE_CHECKING:
    # Runtime import would cycle (stages.* import this module); the loop
    # only needs the sidecar's getter surface via duck typing.
    from deerflow.evolution.stages.state import IterationState

__all__ = [
    "PROPOSE_SEQUENCE_BASE",
    "REGISTER_SEQUENCE_BASE",
    "BACKTEST_SEQUENCE_BASE",
    "BEGIN_SEQUENCE_BASE",
    "OUTCOME_SEQUENCE_BASE",
    "COMMIT_SEQUENCE_BASE",
    "FAILURE_SEQUENCE_BASE",
    "PROMOTE_SEQUENCE_BASE",
    "Stage",
    "StageContext",
    "RunSummary",
    "SpanEmitter",
    "NullSpanEmitter",
    "Proposer",
    "ExperimentDesigner",
    "Coder",
    "Runner",
    "FeedbackSynthesizer",
    "EvolutionLoop",
    "derive_key",
]

logger = logging.getLogger(__name__)

PROPOSE_SEQUENCE_BASE = 1000
REGISTER_SEQUENCE_BASE = 2000
BACKTEST_SEQUENCE_BASE = 2500
BEGIN_SEQUENCE_BASE = 3000
OUTCOME_SEQUENCE_BASE = 4000
COMMIT_SEQUENCE_BASE = 5000
FAILURE_SEQUENCE_BASE = 6000
PROMOTE_SEQUENCE_BASE = 7000


class Stage:
    """Stage names for span emission and error attribution."""

    PROPOSE = "propose"
    DESIGN = "design"
    CODE = "code"
    RUN = "run"
    FEEDBACK = "feedback"
    RECORD = "record"
    RESELECT = "reselect"


def derive_key(run_id: str, sequence_base: int, iteration: int, payload: Any) -> str:
    """Derive one deterministic idempotency key for a loop write.

    Args:
        run_id: Loop run UUID (non-empty string).
        sequence_base: Per-write base (1000..7000 step 1000, plus 2500
            for backtest dispatch); the sequence is
            ``sequence_base + iteration``.
        iteration: Non-negative loop iteration.
        payload: JSON-native payload the key binds to.

    Returns:
        64-character lowercase hex digest.
    """
    if not isinstance(sequence_base, int) or isinstance(sequence_base, bool) or sequence_base < 0:
        raise ValueError(f"sequence_base must be a non-negative int, got {sequence_base!r}.")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise ValueError(f"iteration must be a non-negative int, got {iteration!r}.")
    return make_idempotency_key(run_id, sequence_base + iteration, payload)


@dataclass(frozen=True)
class StageContext:
    """Read-only context handed to every stage of one iteration."""

    run_id: str
    iteration: int
    step_id: str
    trace: Trace
    config: EvolutionConfig
    queried_knowledge: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunSummary:
    """Outcome of :meth:`EvolutionLoop.run`."""

    run_id: str
    iterations_run: int
    steps_recorded: int
    stopped_reason: str
    sota_step_id: str | None = None


@runtime_checkable
class SpanEmitter(Protocol):
    """Observability boundary for loop/stage span events (no I/O of its own)."""

    def emit(self, *, run_id: str, iteration: int, step_id: str, stage: str, status: str, detail: str | None = None) -> None:
        """Record one span event (``status``: started|ok|error|skipped)."""
        ...


class NullSpanEmitter:
    """Default span emitter: debug-level log lines, no external tracing."""

    def emit(self, *, run_id: str, iteration: int, step_id: str, stage: str, status: str, detail: str | None = None) -> None:
        logger.debug("evolution run=%s iteration=%d step=%s stage=%s status=%s detail=%s", run_id, iteration, step_id, stage, status, detail)


@runtime_checkable
class Proposer(Protocol):
    """Stage 1: propose the next hypothesis from the trace and context."""

    async def propose(self, ctx: StageContext) -> Hypothesis: ...


@runtime_checkable
class ExperimentDesigner(Protocol):
    """Stage 2: convert a hypothesis into an executable experiment spec."""

    async def design(self, hypothesis: Hypothesis, ctx: StageContext) -> ExperimentSpec: ...


@runtime_checkable
class Coder(Protocol):
    """Stage 3: render a spec into runnable sandbox code."""

    async def implement(self, spec: ExperimentSpec, ctx: StageContext) -> CodedExperiment: ...


@runtime_checkable
class Runner(Protocol):
    """Stage 4: execute coded experiments and observe the outcome."""

    async def execute(self, coded: CodedExperiment, ctx: StageContext) -> ExperimentOutcome: ...


@runtime_checkable
class FeedbackSynthesizer(Protocol):
    """Stage 5: turn an observed outcome into accept/reject feedback."""

    async def synthesize(
        self,
        *,
        hypothesis: Hypothesis,
        spec: ExperimentSpec,
        coded: CodedExperiment | None,
        outcome: ExperimentOutcome,
        ctx: StageContext,
    ) -> ExperimentFeedback: ...


@dataclass
class EvolutionLoop:
    """Async runner for the closed evolution loop.

    Args:
        config: Loop budgets, defaults, and model-role mapping.
        trace_store: Persistence boundary for traces.
        checkpoint_selector: Chooses the attach point before each iteration.
        sota_selector: Surfaces the state-of-the-art step for summaries.
        proposer/designer/coder/runner/synthesizer: Stage implementations.
        clock: Injectable monotonic clock (wall-clock budget tests).
        span_emitter: Observability boundary (default: debug logging).
        should_stop: Optional stop-rule callback (scheduler integration);
            returning True ends the run with ``stop-requested``.
        state: Shared per-iteration sidecar (the same instance the stages
            record into). When bound, each tick first rehydrates it from
            the persisted trace (restart recovery) and the recorded step
            carries the tick's cross-system ids; when None, steps record
            with empty ids.
    """

    config: EvolutionConfig
    trace_store: TraceStore
    checkpoint_selector: CheckpointSelector
    sota_selector: SOTASelector
    proposer: Proposer
    designer: ExperimentDesigner
    coder: Coder
    runner: Runner
    synthesizer: FeedbackSynthesizer
    clock: Callable[[], float] = field(default=time.monotonic)
    span_emitter: SpanEmitter = field(default_factory=NullSpanEmitter)
    should_stop: Callable[[], bool] | None = None
    state: IterationState | None = None

    def __post_init__(self) -> None:
        self._deadline: float | None = None

    # -- public API -----------------------------------------------------

    async def step(self, run_id: str) -> EvoStep:
        """Run exactly one iteration for *run_id* (the scheduled-tick entry).

        Returns the recorded step. Raises :class:`BudgetExhaustedError`
        when no budget remains (max iterations, wall-clock deadline, or
        stop rule); the caller treats that as a clean stop, not a failure.
        """
        validated_run_id = self._require_run_id(run_id)
        trace = self.trace_store.get_trace(validated_run_id) or Trace()
        iteration = self._next_iteration(trace)
        self._check_budget(iteration, trace)
        if self.state is not None:
            self.state.hydrate_from_trace(trace)

        step_id = str(uuid.uuid4())
        self._emit(validated_run_id, iteration, step_id, Stage.RESELECT, "started")
        selection = self.checkpoint_selector.select(trace)
        trace.set_current_selection(selection)
        self._emit(validated_run_id, iteration, step_id, Stage.RESELECT, "ok", detail=repr(selection))

        ctx = StageContext(
            run_id=validated_run_id,
            iteration=iteration,
            step_id=step_id,
            trace=trace,
            config=self.config,
        )

        self._emit(validated_run_id, iteration, step_id, Stage.PROPOSE, "started")
        try:
            hypothesis = await self.proposer.propose(ctx)
        except Exception as exc:
            self._emit(validated_run_id, iteration, step_id, Stage.PROPOSE, "error", detail=f"{type(exc).__name__}: {exc}")
            raise
        self._emit(validated_run_id, iteration, step_id, Stage.PROPOSE, "ok")

        try:
            spec = await self._run_stage(ctx, Stage.DESIGN, self.designer.design(hypothesis, ctx))
            coded: CodedExperiment | None = await self._run_stage(ctx, Stage.CODE, self.coder.implement(spec, ctx))
            outcome = await self._run_stage(ctx, Stage.RUN, self.runner.execute(coded, ctx))
            feedback = await self._run_stage(
                ctx,
                Stage.FEEDBACK,
                self.synthesizer.synthesize(hypothesis=hypothesis, spec=spec, coded=coded, outcome=outcome, ctx=ctx),
            )
        except DuplicateRunError as duplicate:
            feedback = HypothesisFeedback(
                decision=False,
                reason=f"Exact rerun of experiment {duplicate.existing_experiment_id}; linked to the prior run instead of executing.",
                acceptable=False,
            )
            spec = self._fallback_spec(hypothesis)
            coded = None
        except Exception as exc:  # noqa: BLE001 - the loop safety net converts every stage failure to feedback
            feedback = self._feedback_from_error(exc)
            spec = self._fallback_spec(hypothesis, exc)
            coded = None

        self._emit(validated_run_id, iteration, step_id, Stage.RECORD, "started")
        state = self.state
        step = EvoStep(
            step_id=step_id,
            run_id=validated_run_id,
            iteration=iteration,
            subject=spec,
            hypothesis=hypothesis,
            queried_knowledge=ctx.queried_knowledge,
            feedback=feedback,
            hypothesis_id=state.get_hypothesis_id(validated_run_id, iteration) if state is not None else None,
            episode_id=state.get_episode_id(validated_run_id, iteration) if state is not None else None,
            experiment_id=state.get_experiment_id(validated_run_id, iteration) if state is not None else None,
        )
        stored = trace.record(step, loop_id=iteration)
        trace.set_current_selection((len(trace) - 1,))
        self.trace_store.put_trace(validated_run_id, trace)
        self._emit(validated_run_id, iteration, step_id, Stage.RECORD, "ok", detail=f"decision={feedback.decision}")
        return stored

    async def run(self, run_id: str) -> RunSummary:
        """Run iterations for *run_id* until the budget exhausts.

        Returns a :class:`RunSummary`; never raises
        :class:`BudgetExhaustedError` itself (that is the clean-stop
        signal caught per step).
        """
        validated_run_id = self._require_run_id(run_id)
        wall_clock = self.config.resolve_max_wall_clock_seconds()
        self._deadline = self.clock() + wall_clock if wall_clock > 0 else None
        try:
            iterations_run = 0
            while True:
                try:
                    await self.step(validated_run_id)
                except BudgetExhaustedError as exhausted:
                    return self._summarize(validated_run_id, iterations_run, exhausted.reason)
                iterations_run += 1
        finally:
            self._deadline = None

    # -- internals ------------------------------------------------------

    @staticmethod
    def _require_run_id(run_id: str) -> str:
        if not isinstance(run_id, str):
            raise ValueError(f"run_id must be a UUID string, got {type(run_id).__name__}.")
        try:
            return str(uuid.UUID(run_id.strip()))
        except (ValueError, AttributeError):
            raise ValueError(f"run_id must be a valid UUID, got {run_id!r}.") from None

    @staticmethod
    def _next_iteration(trace: Trace) -> int:
        if len(trace) == 0:
            return 0
        return max(step.iteration for step in trace.hist) + 1

    def _check_budget(self, iteration: int, trace: Trace) -> None:
        if self.should_stop is not None and self.should_stop():
            raise BudgetExhaustedError(f"Stop requested for run at iteration {iteration}.", reason="stop-requested")
        if self._deadline is not None and self.clock() >= self._deadline:
            raise BudgetExhaustedError(f"Wall-clock budget exhausted at iteration {iteration}.", reason="wall-clock")
        max_iterations = self.config.resolve_max_iterations()
        if iteration >= max_iterations:
            raise BudgetExhaustedError(f"Iteration budget exhausted ({iteration}/{max_iterations}).", reason="max-iterations")
        del trace

    def _summarize(self, run_id: str, iterations_run: int, reason: str) -> RunSummary:
        trace = self.trace_store.get_trace(run_id) or Trace()
        sota = self.sota_selector.select_sota(trace)
        return RunSummary(
            run_id=run_id,
            iterations_run=iterations_run,
            steps_recorded=len(trace),
            stopped_reason=reason,
            sota_step_id=sota.step_id if sota is not None else None,
        )

    async def _run_stage(self, ctx: StageContext, stage: str, awaitable: Any) -> Any:
        self._emit(ctx.run_id, ctx.iteration, ctx.step_id, stage, "started")
        try:
            result = await awaitable
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit(ctx.run_id, ctx.iteration, ctx.step_id, stage, "error", detail=f"{type(exc).__name__}: {exc}")
            raise
        self._emit(ctx.run_id, ctx.iteration, ctx.step_id, stage, "ok")
        return result

    @staticmethod
    def _fallback_spec(hypothesis: Hypothesis, exc: BaseException | None = None) -> ExperimentSpec:
        """Minimal subject preserving a failed iteration's hypothesis in lineage."""
        del exc
        return ExperimentSpec(design={"hypothesis": hypothesis.hypothesis.strip(), "methodology": {}})

    @staticmethod
    def _feedback_from_error(exc: BaseException) -> ExperimentFeedback:
        if isinstance(exc, asyncio.CancelledError):
            raise exc
        base = ExperimentFeedback.from_exception(exc)
        return HypothesisFeedback(
            decision=False,
            reason=base.reason,
            code_change_summary=base.code_change_summary,
            error=base.error,
            error_type=base.error_type,
            acceptable=False,
        )

    def _emit(self, run_id: str, iteration: int, step_id: str, stage: str, status: str, detail: str | None = None) -> None:
        try:
            self.span_emitter.emit(run_id=run_id, iteration=iteration, step_id=step_id, stage=stage, status=status, detail=detail)
        except Exception:  # noqa: BLE001 - observability must never break the loop
            logger.debug("Span emitter failed; continuing.", exc_info=True)

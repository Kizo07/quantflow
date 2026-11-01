"""Execute stages: sandbox coding plus governed backtest dispatch.

* :class:`CoderImpl` implements the loop's ``Coder`` protocol: it renders
  an :class:`ExperimentSpec` into sandbox code via the coder-profile LLM.
* :class:`RunnerImpl` implements the loop's ``Runner`` protocol: it runs
  the code through a :class:`SandboxExecutor` and dispatches quantitative
  evaluation through spark ``backtest_run``.

Tombstone handling (no human in the loop): a tombstone-blocked dispatch
retries exactly once, and only when the spec carries a recorded
``tombstone_acknowledgement`` (``{tombstone_id, rationale}``) whose id
matches the block. Anything else — a block without a recorded ack, a
sandbox failure, a provider error — becomes an
:class:`ExperimentOutcome` with ``error`` set (never raised), so the
feedback stage still closes out the episode (``log_outcome`` +
``experiment_commit`` + ``failure_record``) instead of leaving a
half-registered trial. Only cancellation and duplicate-run signals
propagate.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from deerflow.evolution.errors import DuplicateRunError
from deerflow.evolution.loop import BACKTEST_SEQUENCE_BASE, StageContext, derive_key
from deerflow.evolution.models import CodedExperiment, ExperimentOutcome, ExperimentSpec
from deerflow.evolution.prompts import CODER_SYSTEM
from deerflow.evolution.spark_client import SparkClient
from deerflow.evolution.stages.design import derive_seed
from deerflow.evolution.stages.llm import ChatModelFactory
from deerflow.evolution.stages.state import IterationState

if TYPE_CHECKING:
    from deerflow.sandbox.sandbox_provider import SandboxProvider

__all__ = [
    "SandboxResult",
    "SandboxExecutor",
    "ProviderSandboxExecutor",
    "CoderImpl",
    "RunnerImpl",
    "TOMBSTONE_BLOCK_MARKER",
]

TOMBSTONE_BLOCK_MARKER = "tombstone pre-check blocked dispatch"
_TOMBSTONE_ID_RE = re.compile(r'"tombstone_id"\s*:\s*"([^"]+)"')
_STDOUT_TAIL_CHARS = 2000


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of one sandbox code run."""

    stdout: str = ""
    artifacts: tuple[str, ...] = ()
    error: str | None = None


@runtime_checkable
class SandboxExecutor(Protocol):
    """Code-execution boundary for the runner stage."""

    async def run(self, coded: CodedExperiment, ctx: StageContext) -> SandboxResult:
        """Run coded experiment code and capture its output."""
        ...


class ProviderSandboxExecutor:
    """Default :class:`SandboxExecutor` over a ``SandboxProvider``.

    Writes the coded ``files`` mapping into a fresh provider sandbox and
    runs its ``entrypoint`` command. Blocking provider calls run in worker
    threads; the sandbox is always released.
    """

    def __init__(self, provider: SandboxProvider, *, command_timeout: float | None = 300.0) -> None:
        self._provider = provider
        self._command_timeout = command_timeout

    async def run(self, coded: CodedExperiment, ctx: StageContext) -> SandboxResult:
        """Run the coded files' entrypoint in a provider sandbox."""
        return await asyncio.to_thread(self._run_sync, coded, ctx)

    def _run_sync(self, coded: CodedExperiment, ctx: StageContext) -> SandboxResult:
        files = coded.code.get("files", {})
        entrypoint = coded.code.get("entrypoint", "")
        if not isinstance(files, Mapping) or not files:
            return SandboxResult(stdout="", error="Coded experiment carries no files to run.")
        if not isinstance(entrypoint, str) or not entrypoint.strip():
            return SandboxResult(stdout="", error="Coded experiment carries no entrypoint command.")
        workdir = f"/tmp/evolution-{ctx.run_id}-{ctx.iteration}"
        sandbox_id = self._provider.acquire(thread_id=ctx.run_id)
        try:
            sandbox = self._provider.get(sandbox_id)
            if sandbox is None:
                return SandboxResult(stdout="", error=f"Sandbox {sandbox_id} vanished after acquire.")
            for path, content in files.items():
                if not isinstance(path, str) or not isinstance(content, str):
                    return SandboxResult(stdout="", error=f"Sandbox file {path!r} must map a path to text content.")
                sandbox.write_file(f"{workdir}/{path}", content)
            # Trust boundary: the entrypoint is LLM-generated shell text. It
            # runs only inside the provider sandbox (never on the host), so
            # no host-side sanitization is applied; do not reuse this pattern
            # for host execution.
            output = sandbox.execute_command(f"cd {workdir} && {entrypoint.strip()}", timeout=self._command_timeout)
            return SandboxResult(stdout=output or "")
        finally:
            self._provider.release(sandbox_id)


def _parse_coder_output(raw: str) -> tuple[dict[str, Any], str]:
    """Parse coder output into (code mapping, summary).

    Accepts a JSON object with ``code`` (+ optional ``summary``); anything
    else is kept verbatim as a single ``main.py`` file so a run never fails
    on code formatting alone.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, Mapping) and isinstance(parsed.get("code"), Mapping):
        summary = parsed.get("summary", "")
        return dict(parsed["code"]), summary if isinstance(summary, str) else ""
    return {"files": {"main.py": raw}, "entrypoint": "python main.py"}, ""


class CoderImpl:
    """Loop ``Coder`` rendering specs into sandbox code via the LLM."""

    def __init__(self, models: ChatModelFactory) -> None:
        self._models = models

    async def implement(self, spec: ExperimentSpec, ctx: StageContext) -> CodedExperiment:
        """Render *spec* into runnable sandbox code."""
        _ = ctx
        raw = await self._models.generate(
            role="coder",
            system=CODER_SYSTEM,
            user=json.dumps(spec.to_dict(), sort_keys=True),
        )
        code, summary = _parse_coder_output(raw)
        return CodedExperiment(spec=spec, code=code, code_change_summary=summary)


def _tombstone_id_from_block(exc: BaseException) -> str | None:
    """Extract the blocking tombstone id from a dispatch error, if present."""
    if TOMBSTONE_BLOCK_MARKER not in str(exc):
        return None
    match = _TOMBSTONE_ID_RE.search(str(exc))
    return match.group(1) if match else None


def _recorded_acknowledgement(spec: ExperimentSpec, tombstone_id: str) -> dict[str, Any] | None:
    """Return the spec's recorded ack when it matches the blocking tombstone."""
    ack = spec.parameters.get("tombstone_acknowledgement")
    if not isinstance(ack, Mapping):
        return None
    ack_id = ack.get("tombstone_id")
    rationale = ack.get("rationale")
    if ack_id != tombstone_id or not isinstance(rationale, str) or not rationale.strip():
        return None
    return {"tombstone_id": ack_id, "rationale": rationale.strip()}


class RunnerImpl:
    """Loop ``Runner`` executing code and dispatching governed backtests.

    Args:
        spark: Trial-governance client for ``backtest_run``.
        executor: Code-execution boundary.
        state: Shared per-iteration sidecar (episode id + seed).
    """

    def __init__(self, *, spark: SparkClient, executor: SandboxExecutor, state: IterationState | None = None) -> None:
        self._spark = spark
        self._executor = executor
        self._state = state or IterationState()

    async def execute(self, coded: CodedExperiment, ctx: StageContext) -> ExperimentOutcome:
        """Execute one coded experiment, converting failures to outcomes."""
        try:
            return await self._execute_inner(coded, ctx)
        except asyncio.CancelledError:
            raise
        except DuplicateRunError:
            raise
        except Exception as exc:
            return ExperimentOutcome(
                metrics={},
                artifacts=(),
                error=str(exc)[:2000],
                error_type=type(exc).__name__,
            )

    async def _execute_inner(self, coded: CodedExperiment, ctx: StageContext) -> ExperimentOutcome:
        episode_id = self._state.require_episode_id(ctx.run_id, ctx.iteration)
        seed = self._state.get_seed(ctx.run_id, ctx.iteration)
        if seed is None:
            seed = derive_seed(ctx.run_id, ctx.iteration)
        sandbox_result = await self._executor.run(coded, ctx)
        if sandbox_result.error is not None:
            return ExperimentOutcome(
                metrics={},
                artifacts=sandbox_result.artifacts,
                error=sandbox_result.error[:2000],
                error_type="SandboxError",
            )
        trial_spec: dict[str, Any] = {
            "experiment_spec": coded.spec.to_dict(),
            "seed": seed,
            "sandbox_stdout_tail": sandbox_result.stdout[-_STDOUT_TAIL_CHARS:],
        }
        backtest_key = derive_key(
            ctx.run_id,
            BACKTEST_SEQUENCE_BASE,
            ctx.iteration,
            {"backtest_run": {"episode_id": episode_id, "spec": trial_spec, "seed": seed}},
        )
        try:
            result = await self._spark.backtest_run(episode_id=episode_id, spec=trial_spec, client_id=backtest_key)
        except Exception as exc:
            tombstone_id = _tombstone_id_from_block(exc)
            ack = _recorded_acknowledgement(coded.spec, tombstone_id) if tombstone_id else None
            if ack is None:
                raise
            result = await self._spark.backtest_run(
                episode_id=episode_id,
                spec=trial_spec,
                acknowledge_tombstone=ack,
                client_id=backtest_key,
            )
        metrics = dict(result.metrics)
        metrics.setdefault("simulated", result.simulated)
        metrics["sandbox_stdout_tail"] = sandbox_result.stdout[-_STDOUT_TAIL_CHARS:]
        return ExperimentOutcome(metrics=metrics, artifacts=sandbox_result.artifacts)

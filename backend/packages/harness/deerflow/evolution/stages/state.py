"""Per-iteration coordination state shared by the evolution stages.

The loop core hands each stage a frozen :class:`StageContext` and takes
back only the stage product (hypothesis, spec, outcome, feedback), so
cross-system identities created mid-iteration — the spark
``hypothesis_id``/``episode_id``, the KB ``experiment_id`` — need a home
outside the stage signatures. :class:`IterationState` is that home: one
instance is shared by every stage of a run, keyed by ``(run_id,
iteration)``.

It also carries small shared stage helpers (lineage formatting) that need
no external bindings.

The loop core reads the tick's ids back out of the sidecar when it builds
the recorded ``EvoStep``, and rehydrates the sidecar from the persisted
trace on every tick, so a scheduler restart only loses in-flight ticks —
recorded iterations keep their cross-system linkage.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from deerflow.evolution.errors import EvolutionError
from deerflow.evolution.trace import Trace

__all__ = [
    "IterationRecord",
    "IterationState",
    "format_lineage",
]


@dataclass
class IterationRecord:
    """Cross-system identities for one ``(run_id, iteration)``."""

    hypothesis_id: str | None = None
    c6_scope: str | None = None
    episode_id: str | None = None
    experiment_id: str | None = None
    protocol_hash: str | None = None
    seed: int | None = None


class IterationState:
    """Thread-safe per-iteration sidecar shared by the stages of a run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str], IterationRecord] = {}

    @staticmethod
    def _key(run_id: str, iteration: int) -> tuple[str, str]:
        return (run_id, str(iteration))

    def _record(self, run_id: str, iteration: int) -> IterationRecord:
        with self._lock:
            key = self._key(run_id, iteration)
            record = self._records.get(key)
            if record is None:
                record = IterationRecord()
                self._records[key] = record
            return record

    def record_proposal(self, run_id: str, iteration: int, *, hypothesis_id: str, scope: str) -> None:
        """Store the propose-stage identities for one iteration."""
        record = self._record(run_id, iteration)
        with self._lock:
            record.hypothesis_id = hypothesis_id
            record.c6_scope = scope

    def record_design(
        self,
        run_id: str,
        iteration: int,
        *,
        episode_id: str,
        experiment_id: str,
        protocol_hash: str,
        seed: int,
    ) -> None:
        """Store the design-stage identities for one iteration."""
        record = self._record(run_id, iteration)
        with self._lock:
            record.episode_id = episode_id
            record.experiment_id = experiment_id
            record.protocol_hash = protocol_hash
            record.seed = seed

    def get_hypothesis_id(self, run_id: str, iteration: int) -> str | None:
        """Return the recorded hypothesis id, if any."""
        with self._lock:
            record = self._records.get(self._key(run_id, iteration))
            return record.hypothesis_id if record else None

    def get_scope(self, run_id: str, iteration: int) -> str | None:
        """Return the recorded C6 scope, if any."""
        with self._lock:
            record = self._records.get(self._key(run_id, iteration))
            return record.c6_scope if record else None

    def get_episode_id(self, run_id: str, iteration: int) -> str | None:
        """Return the recorded episode id, if any."""
        with self._lock:
            record = self._records.get(self._key(run_id, iteration))
            return record.episode_id if record else None

    def get_experiment_id(self, run_id: str, iteration: int) -> str | None:
        """Return the recorded experiment id, if any."""
        with self._lock:
            record = self._records.get(self._key(run_id, iteration))
            return record.experiment_id if record else None

    def get_seed(self, run_id: str, iteration: int) -> int | None:
        """Return the recorded trial seed, if any."""
        with self._lock:
            record = self._records.get(self._key(run_id, iteration))
            return record.seed if record else None

    def require_hypothesis_id(self, run_id: str, iteration: int) -> str:
        """Return the hypothesis id, or raise when propose did not record one."""
        value = self.get_hypothesis_id(run_id, iteration)
        if value is None:
            raise EvolutionError(f"No hypothesis_id recorded for run {run_id} iteration {iteration}; the propose stage must run first.")
        return value

    def require_episode_id(self, run_id: str, iteration: int) -> str:
        """Return the episode id, or raise when design did not record one."""
        value = self.get_episode_id(run_id, iteration)
        if value is None:
            raise EvolutionError(f"No episode_id recorded for run {run_id} iteration {iteration}; the design stage must run first.")
        return value

    def require_experiment_id(self, run_id: str, iteration: int) -> str:
        """Return the experiment id, or raise when design did not record one."""
        value = self.get_experiment_id(run_id, iteration)
        if value is None:
            raise EvolutionError(f"No experiment_id recorded for run {run_id} iteration {iteration}; the design stage must run first.")
        return value

    def hydrate_from_trace(self, trace: Trace) -> int:
        """Repopulate records from recorded steps (restart recovery).

        Fills only fields that are still empty, so live in-memory records
        always win over persisted ones. Returns the number of records
        touched. Steps recorded without ids (pre-linkage runs) are
        skipped.
        """
        touched = 0
        for step in trace.hist:
            if step.hypothesis_id is None and step.episode_id is None and step.experiment_id is None:
                continue
            record = self._record(step.run_id, step.iteration)
            with self._lock:
                changed = False
                if record.hypothesis_id is None and step.hypothesis_id is not None:
                    record.hypothesis_id = step.hypothesis_id
                    changed = True
                if record.episode_id is None and step.episode_id is not None:
                    record.episode_id = step.episode_id
                    changed = True
                if record.experiment_id is None and step.experiment_id is not None:
                    record.experiment_id = step.experiment_id
                    changed = True
                if changed:
                    touched += 1
        return touched


def format_lineage(trace: Trace, *, limit: int = 8) -> str:
    """Format the root-to-current parent lineage for stage prompts.

    One line per ancestor: ``[iteration] decision=<bool|pending>
    <hypothesis one-liner>``. Returns ``"(no parent steps)"`` on a fresh
    tree.
    """
    parents = trace.get_parent_steps()
    if not parents:
        return "(no parent steps)"
    lines: list[str] = []
    for step in parents[-limit:]:
        decision = step.feedback.decision if step.feedback is not None else "pending"
        claim = step.hypothesis.hypothesis if step.hypothesis is not None else "(hypothesis not recorded)"
        one_liner = " ".join(claim.split())
        if len(one_liner) > 160:
            one_liner = one_liner[:157] + "..."
        lines.append(f"[{step.iteration}] decision={decision} {one_liner}")
    return "\n".join(lines)

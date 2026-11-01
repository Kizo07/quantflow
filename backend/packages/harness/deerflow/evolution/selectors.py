"""Checkpoint and SOTA selectors for the evolution Trace DAG.

Selectors are pure functions over :class:`deerflow.evolution.trace.Trace`:
they never call LLMs, never touch stores, and never mutate the trace. The
loop applies their output via ``Trace.set_current_selection`` before
recording the next step.

Selection vocabulary (shared with :mod:`deerflow.evolution.trace`):

* ``(idx,)`` — attach the next step to ``hist[idx]``.
* ``SEL_LATEST_SOTA`` ((``-1,``)) — attach to the latest recorded step.
* ``None`` — start a fresh tree (stored as ``NEW_ROOT``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from deerflow.evolution.models import EvoStep
from deerflow.evolution.trace import NEW_ROOT, SEL_LATEST_SOTA, Trace

__all__ = [
    "CheckpointSelector",
    "SOTASelector",
    "LatestSOTASelector",
    "AncestorSOTASelector",
    "RootRestartSelector",
    "FirstAcceptedSelector",
]


@runtime_checkable
class CheckpointSelector(Protocol):
    """Chooses where the next step attaches in the trace."""

    def select(self, trace: Trace) -> tuple[int, ...] | None:
        """Return the selection for the next step (None = new tree)."""
        ...


@runtime_checkable
class SOTASelector(Protocol):
    """Chooses the state-of-the-art step to surface downstream."""

    def select_sota(self, trace: Trace) -> EvoStep | None:
        """Return the SOTA step, or None when the trace has no accepted step."""
        ...


class LatestSOTASelector:
    """Continue from the latest step; start a new tree when empty.

    Deterministic default for single-lineage runs. On a non-empty trace it
    returns ``SEL_LATEST_SOTA`` (resolved at record time); on an empty trace
    it returns None so the first step records as a root.
    """

    def select(self, trace: Trace) -> tuple[int, ...] | None:
        if len(trace) == 0:
            return None
        return SEL_LATEST_SOTA

    def select_sota(self, trace: Trace) -> EvoStep | None:
        return trace.get_sota_experiment()


class AncestorSOTASelector:
    """Continue from the latest accepted ancestor of the current selection.

    Walks the current ancestor chain root→…→current and selects the newest
    accepted step as a concrete ``(idx,)``. Returns None (new tree) when the
    trace is empty or no ancestor was accepted — the lineage is exhausted
    and the next hypothesis should restart from scratch.
    """

    def select(self, trace: Trace) -> tuple[int, ...] | None:
        if len(trace) == 0 or trace.is_new_tree():
            return None
        selection = trace.get_current_selection()
        idx = selection[0]
        if idx == -1:
            idx = len(trace) - 1
        for ancestor in reversed(trace.get_parents(idx)):
            if trace.hist[ancestor].accepted:
                return (ancestor,)
        return None

    def select_sota(self, trace: Trace) -> EvoStep | None:
        return trace.get_sota_experiment()


class RootRestartSelector:
    """Always start a fresh tree (parallel-breadth exploration).

    Returns None unconditionally; the SOTA query still reports the latest
    accepted step on the current chain for observability.
    """

    def select(self, trace: Trace) -> tuple[int, ...] | None:
        del trace
        return None

    def select_sota(self, trace: Trace) -> EvoStep | None:
        return trace.get_sota_experiment()


class FirstAcceptedSelector:
    """Continue from the earliest accepted step (stable-baseline re-branching).

    Returns the first accepted index in record order as ``(idx,)``, or None
    when nothing was accepted yet. Useful for re-branching ablations from a
    fixed baseline instead of the moving frontier.
    """

    def select(self, trace: Trace) -> tuple[int, ...] | None:
        for idx, step in enumerate(trace.hist):
            if step.accepted:
                return (idx,)
        return None

    def select_sota(self, trace: Trace) -> EvoStep | None:
        return trace.get_sota_experiment()


def resolve_selection_for_record(trace: Trace, selection: tuple[int, ...] | None) -> tuple[int, ...]:
    """Normalize a selector output to the stored-selection vocabulary.

    None becomes ``NEW_ROOT``; every other value passes through unchanged
    (``SEL_LATEST_SOTA`` is resolved by :meth:`Trace.record`).
    """
    if selection is None:
        return NEW_ROOT
    return selection

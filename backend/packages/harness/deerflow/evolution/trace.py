"""Append-only Trace DAG for the native Quantflow evolution loop.

A :class:`Trace` holds the ordered step history (``hist``) plus the
``dag_parent`` index tuples that link each step to its parent step(s):

* ``NEW_ROOT`` (``()``) — the step starts a fresh tree (no parents).
* ``(idx,)`` — the step continues from the step at ``hist[idx]``.
* ``SEL_LATEST_SOTA`` ((``-1,``)) — selection sugar meaning "the latest
  recorded step"; resolved to a concrete index at record time and never
  stored in ``dag_parent``.

``hist`` and ``dag_parent`` always grow together via :meth:`Trace.record`,
which derives the stored parent tuple from the current selection (or an
explicit selection) and fills the step's ``parent_step_ids`` so the
index-based and id-based edges can never disagree.

Storage boundary: :class:`TraceStore` is a small ``Protocol`` (same no-I/O
style as ``ExperimentStore``); :class:`InMemoryTraceStore` is the
deterministic in-process binding used by tests and scheduled ticks. A
PostgreSQL binding is explicitly phase 2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol, runtime_checkable

from deerflow.evolution.models import EvoStep

__all__ = [
    "NEW_ROOT",
    "SEL_LATEST_SOTA",
    "Trace",
    "TraceStore",
    "InMemoryTraceStore",
]

NEW_ROOT: tuple[int, ...] = ()
SEL_LATEST_SOTA: tuple[int, ...] = (-1,)


def _validate_selection(selection: tuple[int, ...], *, size: int) -> tuple[int, ...]:
    """Validate a selection tuple against a trace of *size* steps."""
    if not isinstance(selection, tuple):
        raise ValueError(f"selection must be a tuple of ints, got {type(selection).__name__}.")
    for item in selection:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"selection must be a tuple of ints, got {selection!r}.")
    if len(selection) > 1:
        raise ValueError(f"Multi-parent selections are not supported yet, got {selection!r}.")
    if selection == SEL_LATEST_SOTA:
        # Resolved at use time; on an empty trace callers treat it as a root.
        return selection
    if len(selection) == 1:
        idx = selection[0]
        if idx < 0 or idx >= size:
            raise ValueError(f"Selection index {idx} out of range for trace of size {size}.")
    return selection


class Trace:
    """Append-only DAG of evolution steps for one logical lineage.

    The trace is intentionally scenario-free: it stores steps and edges only.
    Selection state (``current_selection``) marks where the next recorded
    step will attach; selectors compute it, the loop applies it.
    """

    def __init__(self, *, current_selection: tuple[int, ...] = SEL_LATEST_SOTA) -> None:
        self.hist: list[EvoStep] = []
        self.dag_parent: list[tuple[int, ...]] = []
        self.idx2loop_id: dict[int, int] = {}
        if not isinstance(current_selection, tuple) or len(current_selection) > 1:
            raise ValueError(f"Initial selection must be a tuple of at most one int, got {current_selection!r}.")
        for item in current_selection:
            if not isinstance(item, int) or isinstance(item, bool):
                raise ValueError(f"Initial selection must be a tuple of at most one int, got {current_selection!r}.")
        # Range is checked at use sites (record/select walks), since a fresh
        # trace is empty by construction and restored traces carry concrete
        # selections from the store.
        self.current_selection: tuple[int, ...] = current_selection

    def __len__(self) -> int:
        return len(self.hist)

    # -- selection ------------------------------------------------------

    def get_current_selection(self) -> tuple[int, ...]:
        return self.current_selection

    def set_current_selection(self, selection: tuple[int, ...] | None) -> None:
        """Set where the next recorded step attaches.

        ``None`` means "start a new tree" and is stored as ``NEW_ROOT``.
        """
        if selection is None:
            self.current_selection = NEW_ROOT
            return
        _validate_selection(selection, size=len(self.hist))
        self.current_selection = selection

    def is_new_tree(self, selection: tuple[int, ...] | None = None) -> bool:
        """True when *selection* (default: current) starts a fresh tree."""
        if selection is None:
            selection = self.current_selection
        return selection == NEW_ROOT or len(self.dag_parent) == 0

    # -- recording (parent/history sync) --------------------------------

    def record(self, step: EvoStep, *, loop_id: int, selection: tuple[int, ...] | None = None) -> EvoStep:
        """Append *step* and its DAG edge atomically; return the stored step.

        The stored parent tuple derives from *selection* (default: the
        current selection): ``NEW_ROOT``/empty trace stores ``NEW_ROOT``,
        ``SEL_LATEST_SOTA`` resolves to the latest index, and ``(idx,)``
        stores ``(idx,)``. The returned step carries the matching
        ``parent_step_ids`` so index edges and id edges stay in sync.
        """
        if not isinstance(step, EvoStep):
            raise ValueError(f"step must be an EvoStep, got {type(step).__name__}.")
        if not isinstance(loop_id, int) or isinstance(loop_id, bool):
            raise ValueError(f"loop_id must be an int, got {loop_id!r}.")
        if selection is None:
            selection = self.current_selection
        _validate_selection(selection, size=len(self.hist))

        if len(self.hist) == 0 or len(selection) == 0:
            parent: tuple[int, ...] = NEW_ROOT
        else:
            idx = selection[0]
            if idx == -1:
                idx = len(self.hist) - 1
            parent = (idx,)

        stored = step
        expected_parents = tuple(self.hist[i].step_id for i in parent)
        if stored.parent_step_ids != expected_parents:
            stored = replace(step, parent_step_ids=expected_parents)
        self.hist.append(stored)
        self.dag_parent.append(parent)
        self.idx2loop_id[len(self.hist) - 1] = loop_id
        return stored

    # -- walks ----------------------------------------------------------

    def get_parents(self, child_idx: int) -> list[int]:
        """Return ancestor indices root→…→parent→node for ``hist[child_idx]``."""
        if not isinstance(child_idx, int) or isinstance(child_idx, bool):
            raise ValueError(f"child_idx must be an int, got {child_idx!r}.")
        if child_idx < 0 or child_idx >= len(self.hist):
            raise ValueError(f"child_idx {child_idx} out of range for trace of size {len(self.hist)}.")
        ancestors: list[int] = []
        current = child_idx
        while True:
            ancestors.insert(0, current)
            parent_tuple = self.dag_parent[current]
            if not parent_tuple or parent_tuple[0] == current:
                break
            current = parent_tuple[0]
        return ancestors

    def is_parent(self, parent_idx: int, child_idx: int) -> bool:
        """True when ``parent_idx`` is an ancestor-or-self of ``child_idx``."""
        return parent_idx in self.get_parents(child_idx)

    def get_parent_steps(self, selection: tuple[int, ...] | None = None) -> list[EvoStep]:
        """Return ancestor steps root→…→current for *selection* (fresh tree → [])."""
        if selection is None:
            selection = self.current_selection
        if self.is_new_tree(selection):
            return []
        _validate_selection(selection, size=len(self.hist))
        idx = selection[0]
        if idx == -1:
            idx = len(self.hist) - 1
        return [self.hist[i] for i in self.get_parents(idx)]

    def get_children(self, parent_idx: int | None = None) -> list[EvoStep]:
        """Return steps whose stored parent is ``parent_idx`` (None → roots)."""
        if parent_idx is not None:
            if not isinstance(parent_idx, int) or isinstance(parent_idx, bool):
                raise ValueError(f"parent_idx must be an int or None, got {parent_idx!r}.")
            if parent_idx < 0 or parent_idx >= len(self.hist):
                raise ValueError(f"parent_idx {parent_idx} out of range for trace of size {len(self.hist)}.")
        target: tuple[int, ...] = (parent_idx,) if parent_idx is not None else NEW_ROOT
        return [self.hist[i] for i, parents in enumerate(self.dag_parent) if parents == target and i < len(self.hist)]

    def get_sota_experiment(self, node_id: int | None = None) -> EvoStep | None:
        """Return the latest accepted step on the ancestor chain of *node_id*.

        Defaults to the current selection; ``-1`` resolves to the latest
        step. Returns None for fresh trees and when no ancestor was accepted.
        """
        if node_id is None:
            selection = self.current_selection
            if self.is_new_tree(selection):
                return None
            node_id = selection[0]
        if not isinstance(node_id, int) or isinstance(node_id, bool):
            raise ValueError(f"node_id must be an int or None, got {node_id!r}.")
        if node_id == -1:
            if not self.hist:
                return None
            node_id = len(self.hist) - 1
        if node_id < 0 or node_id >= len(self.hist):
            raise ValueError(f"node_id {node_id} out of range for trace of size {len(self.hist)}.")
        for idx in reversed(self.get_parents(node_id)):
            if self.hist[idx].accepted:
                return self.hist[idx]
        return None

    def step_index(self, step_id: str) -> int | None:
        """Return the hist index of *step_id*, or None when absent."""
        for idx, step in enumerate(self.hist):
            if step.step_id == step_id:
                return idx
        return None


@runtime_checkable
class TraceStore(Protocol):
    """Storage boundary for traces; the integration step binds this to PostgreSQL.

    PG binding sketch (phase 2): steps map to an ``evolution_step`` table
    keyed by (``run_id``, ``iteration``) UNIQUE with ``step_id`` UNIQUE,
    ``dag_parent`` stored as an int array, and the step payload as JSONB.
    Every method below is one indexed lookup or one row write; the PG
    implementation wraps each loop iteration in a single transaction.
    """

    def get_trace(self, run_id: str) -> Trace | None:
        """Return the trace for *run_id*, or None when the run is unknown."""
        ...

    def put_trace(self, run_id: str, trace: Trace) -> None:
        """Persist (upsert) the full trace for *run_id*."""
        ...

    def list_runs(self) -> list[str]:
        """Return known run ids in first-seen order."""
        ...


class InMemoryTraceStore:
    """Deterministic in-process :class:`TraceStore` (tests, scheduled ticks).

    Stores deep-ish copies on write (steps are frozen, so a structural copy
    of the lists suffices) so callers cannot mutate persisted state through
    a retrieved trace.
    """

    def __init__(self) -> None:
        self._traces: dict[str, Trace] = {}

    def get_trace(self, run_id: str) -> Trace | None:
        stored = self._traces.get(run_id)
        if stored is None:
            return None
        return self._copy(stored)

    def put_trace(self, run_id: str, trace: Trace) -> None:
        if not isinstance(trace, Trace):
            raise ValueError(f"trace must be a Trace, got {type(trace).__name__}.")
        self._traces[run_id] = self._copy(trace)

    def list_runs(self) -> list[str]:
        return list(self._traces)

    @staticmethod
    def _copy(trace: Trace) -> Trace:
        clone = Trace(current_selection=trace.current_selection)
        clone.hist = list(trace.hist)
        clone.dag_parent = [tuple(parents) for parents in trace.dag_parent]
        clone.idx2loop_id = dict(trace.idx2loop_id)
        return clone


def selections_equal(first: Sequence[int] | None, second: Sequence[int] | None) -> bool:
    """Compare two selections (or None) for equality."""
    if first is None or second is None:
        return first is None and second is None
    return tuple(first) == tuple(second)

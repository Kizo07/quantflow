"""Unit tests for deerflow.evolution.selectors (pure functions, offline)."""

import uuid

from deerflow.evolution.models import EvoStep, ExperimentFeedback, ExperimentSpec
from deerflow.evolution.selectors import (
    AncestorSOTASelector,
    FirstAcceptedSelector,
    LatestSOTASelector,
    RootRestartSelector,
    resolve_selection_for_record,
)
from deerflow.evolution.trace import NEW_ROOT, SEL_LATEST_SOTA, Trace

RUN_ID = "11111111-1111-1111-1111-111111111111"


def _step(iteration: int, accepted: bool) -> EvoStep:
    return EvoStep(
        step_id=str(uuid.uuid4()),
        run_id=RUN_ID,
        iteration=iteration,
        subject=ExperimentSpec(design={"hypothesis": f"h{iteration}", "methodology": {}}),
        feedback=ExperimentFeedback(decision=accepted, reason="ok" if accepted else "no"),
    )


def _chain(*decisions: bool) -> Trace:
    trace = Trace()
    for iteration, accepted in enumerate(decisions):
        trace.record(_step(iteration, accepted), loop_id=iteration)
    return trace


def test_selectors_on_empty_trace_start_new_tree():
    trace = Trace()
    assert LatestSOTASelector().select(trace) is None
    assert AncestorSOTASelector().select(trace) is None
    assert RootRestartSelector().select(trace) is None
    assert FirstAcceptedSelector().select(trace) is None
    for selector in (LatestSOTASelector(), AncestorSOTASelector(), RootRestartSelector(), FirstAcceptedSelector()):
        assert selector.select_sota(trace) is None


def test_latest_selector_tracks_frontier():
    trace = _chain(True, False)
    assert LatestSOTASelector().select(trace) == SEL_LATEST_SOTA
    assert LatestSOTASelector().select_sota(trace).step_id == trace.hist[0].step_id


def test_ancestor_selector_skips_rejected_frontier():
    trace = _chain(True, True, False)
    trace.set_current_selection((2,))
    assert AncestorSOTASelector().select(trace) == (1,)
    assert AncestorSOTASelector().select_sota(trace).step_id == trace.hist[1].step_id


def test_ancestor_selector_exhausted_lineage_restarts():
    trace = _chain(False, False)
    trace.set_current_selection((1,))
    assert AncestorSOTASelector().select(trace) is None
    assert AncestorSOTASelector().select_sota(trace) is None


def test_root_restart_always_new_tree_but_reports_sota():
    trace = _chain(True, False)
    assert RootRestartSelector().select(trace) is None
    assert RootRestartSelector().select_sota(trace).step_id == trace.hist[0].step_id


def test_first_accepted_pins_baseline():
    trace = _chain(False, True, True)
    assert FirstAcceptedSelector().select(trace) == (1,)
    rejected = _chain(False, False)
    assert FirstAcceptedSelector().select(rejected) is None


def test_resolve_selection_for_record():
    assert resolve_selection_for_record(Trace(), None) == NEW_ROOT
    assert resolve_selection_for_record(Trace(), SEL_LATEST_SOTA) == SEL_LATEST_SOTA
    assert resolve_selection_for_record(Trace(), (2,)) == (2,)


def test_selectors_are_pure():
    trace = _chain(True, False, True)
    before_hist = list(trace.hist)
    before_parents = list(trace.dag_parent)
    before_selection = trace.get_current_selection()
    selectors = (LatestSOTASelector(), AncestorSOTASelector(), RootRestartSelector(), FirstAcceptedSelector())
    for selector in selectors:
        selector.select(trace)
        selector.select_sota(trace)
    assert trace.hist == before_hist
    assert trace.dag_parent == before_parents
    assert trace.get_current_selection() == before_selection

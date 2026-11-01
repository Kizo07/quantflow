"""Unit tests for deerflow.evolution.trace (Trace DAG + stores, offline)."""

import uuid

import pytest

from deerflow.evolution.models import EvoStep, ExperimentFeedback, ExperimentSpec, Hypothesis
from deerflow.evolution.trace import NEW_ROOT, SEL_LATEST_SOTA, InMemoryTraceStore, Trace

RUN_ID = "11111111-1111-1111-1111-111111111111"


def _step(iteration: int, *, accepted: bool | None = None) -> EvoStep:
    feedback = None
    if accepted is not None:
        feedback = ExperimentFeedback(decision=accepted, reason="ok" if accepted else "no")
    return EvoStep(
        step_id=str(uuid.uuid4()),
        run_id=RUN_ID,
        iteration=iteration,
        subject=ExperimentSpec(design={"hypothesis": f"h{iteration}", "methodology": {}}),
        hypothesis=Hypothesis(hypothesis=f"h{iteration}", reason="r"),
        feedback=feedback,
    )


def test_empty_trace_edges():
    trace = Trace()
    assert len(trace) == 0
    assert trace.dag_parent == []
    assert trace.get_current_selection() == SEL_LATEST_SOTA
    assert trace.is_new_tree() is True
    assert trace.get_parent_steps() == []
    assert trace.get_children() == []
    assert trace.get_sota_experiment() is None
    with pytest.raises(ValueError):
        trace.get_parents(0)
    with pytest.raises(ValueError):
        trace.get_children(0)
    with pytest.raises(ValueError):
        trace.get_sota_experiment(0)


def test_first_record_is_root_and_syncs_parents():
    trace = Trace()
    stored = trace.record(_step(0, accepted=True), loop_id=0)
    assert trace.hist == [stored]
    assert trace.dag_parent == [NEW_ROOT]
    assert stored.parent_step_ids == ()
    assert trace.idx2loop_id == {0: 0}
    assert trace.step_index(stored.step_id) == 0
    assert trace.step_index("missing") is None


def test_sel_latest_sota_resolves_at_record_time():
    trace = Trace()
    first = trace.record(_step(0, accepted=True), loop_id=0)
    second = trace.record(_step(1, accepted=False), loop_id=1)
    assert trace.dag_parent == [NEW_ROOT, (0,)]
    assert second.parent_step_ids == (first.step_id,)
    assert trace.hist[1].step_id == second.step_id


def test_explicit_selection_and_parent_walks():
    trace = Trace()
    steps = [trace.record(_step(i, accepted=(i != 1)), loop_id=i) for i in range(3)]
    assert trace.get_parents(2) == [0, 1, 2]
    assert trace.is_parent(0, 2) is True
    assert trace.is_parent(2, 0) is False

    trace.set_current_selection((0,))
    assert trace.get_parent_steps() == [steps[0]]
    trace.set_current_selection(None)
    assert trace.get_current_selection() == NEW_ROOT
    assert trace.is_new_tree() is True
    assert trace.get_parent_steps() == []

    branch = trace.record(_step(3, accepted=True), loop_id=3)
    assert trace.dag_parent[3] == NEW_ROOT
    assert branch.parent_step_ids == ()
    assert trace.get_children() == [steps[0], branch]
    assert trace.get_children(0) == [steps[1]]
    assert trace.get_children(1) == [steps[2]]
    assert trace.get_children(2) == []


def test_record_rejects_bad_selection_and_inputs():
    trace = Trace()
    trace.record(_step(0), loop_id=0)
    with pytest.raises(ValueError):
        trace.record(_step(1), loop_id=1, selection=(0, 1))
    with pytest.raises(ValueError):
        trace.record(_step(1), loop_id=1, selection=(9,))
    with pytest.raises(ValueError):
        trace.record("nope", loop_id=1)
    with pytest.raises(ValueError):
        trace.record(_step(1), loop_id=True)
    with pytest.raises(ValueError):
        trace.set_current_selection((4,))
    with pytest.raises(ValueError):
        Trace(current_selection=(0, 1))
    with pytest.raises(ValueError):
        Trace(current_selection="0")
    assert Trace(current_selection=(0,)).get_current_selection() == (0,)


def test_get_sota_experiment_walks_current_chain_only():
    trace = Trace()
    trace.record(_step(0, accepted=True), loop_id=0)  # idx 0: accepted root
    trace.record(_step(1, accepted=False), loop_id=1)  # idx 1: rejected child
    trace.set_current_selection(None)
    trace.record(_step(2, accepted=False), loop_id=2)  # idx 2: rejected root
    trace.record(_step(3, accepted=True), loop_id=3)  # idx 3: accepted child of 2

    trace.set_current_selection((1,))
    assert trace.get_sota_experiment().step_id == trace.hist[0].step_id
    trace.set_current_selection((3,))
    assert trace.get_sota_experiment().step_id == trace.hist[3].step_id
    trace.set_current_selection((2,))
    assert trace.get_sota_experiment() is None
    assert trace.get_sota_experiment(node_id=-1).step_id == trace.hist[3].step_id
    trace.set_current_selection(None)
    assert trace.get_sota_experiment() is None


def test_multi_tree_lineage_stays_independent():
    trace = Trace()
    roots = []
    for i in range(2):
        trace.set_current_selection(None)
        roots.append(trace.record(_step(i, accepted=True), loop_id=i))
    assert trace.dag_parent == [NEW_ROOT, NEW_ROOT]
    assert trace.get_parents(1) == [1]
    assert trace.get_sota_experiment(node_id=1).step_id == roots[1].step_id


def test_in_memory_store_round_trip_and_isolation():
    store = InMemoryTraceStore()
    assert store.get_trace(RUN_ID) is None
    assert store.list_runs() == []

    trace = Trace()
    stored = trace.record(_step(0, accepted=True), loop_id=0)
    store.put_trace(RUN_ID, trace)

    loaded = store.get_trace(RUN_ID)
    assert loaded is not trace
    assert loaded.hist == [stored]
    assert loaded.dag_parent == [NEW_ROOT]
    assert store.list_runs() == [RUN_ID]

    loaded.hist.clear()
    assert len(store.get_trace(RUN_ID)) == 1

    with pytest.raises(ValueError):
        store.put_trace(RUN_ID, "nope")

    other_run = "99999999-9999-9999-9999-999999999999"
    store.put_trace(other_run, Trace())
    assert store.list_runs() == [RUN_ID, other_run]

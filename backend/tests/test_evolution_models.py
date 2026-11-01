"""Unit tests for deerflow.evolution.models (frozen dataclasses, offline)."""

import pytest

from deerflow.evolution.models import (
    CodedExperiment,
    EvoStep,
    ExperimentFeedback,
    ExperimentOutcome,
    ExperimentSpec,
    Hypothesis,
    HypothesisFeedback,
)

RUN_ID = "11111111-1111-1111-1111-111111111111"
STEP_ID = "22222222-2222-2222-2222-222222222222"
PARENT_ID = "33333333-3333-3333-3333-333333333333"


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        design={"hypothesis": "Momentum persists at 21d.", "methodology": {"universe": "sp500"}},
        code={"artifact": "abc123"},
        data={"datasets": []},
        parameters={"seed": 7},
        environment={"image": "qf:v1"},
    )


def test_hypothesis_requires_non_empty_core_fields():
    with pytest.raises(ValueError):
        Hypothesis(hypothesis="  ", reason="r")
    with pytest.raises(ValueError):
        Hypothesis(hypothesis="h", reason="")
    hypo = Hypothesis(hypothesis=" h ", reason=" r ")
    assert hypo.hypothesis == "h"
    assert hypo.reason == "r"
    assert "Hypothesis: h" in str(hypo)


def test_hypothesis_round_trip():
    hypo = Hypothesis(
        hypothesis="h",
        reason="r",
        concise_reason="cr",
        concise_observation="co",
        concise_justification="cj",
        concise_knowledge="ck",
    )
    assert Hypothesis.from_dict(hypo.to_dict()) == hypo
    with pytest.raises(ValueError):
        Hypothesis.from_dict("nope")


def test_experiment_spec_hash_identity():
    base = _spec()
    same_design_other_seed = ExperimentSpec(
        design=dict(base.design),
        code=dict(base.code),
        data=dict(base.data),
        parameters={"seed": 8},
        environment=dict(base.environment),
    )
    other_design = ExperimentSpec(
        design={"hypothesis": "Reversal wins at 5d.", "methodology": {"universe": "sp500"}},
        code=dict(base.code),
        data=dict(base.data),
        parameters=dict(base.parameters),
        environment=dict(base.environment),
    )
    assert same_design_other_seed.family_hash() == base.family_hash()
    assert same_design_other_seed.execution_hash() != base.execution_hash()
    assert other_design.family_hash() != base.family_hash()
    assert other_design.execution_hash() != base.execution_hash()


def test_experiment_spec_rejects_non_canonical():
    with pytest.raises(ValueError):
        ExperimentSpec(design={"ok": {b"bytes"}})
    with pytest.raises(ValueError):
        ExperimentSpec(design="not-a-mapping")
    assert ExperimentSpec.from_dict(_spec().to_dict()) == _spec()


def test_coded_experiment_and_outcome():
    coded = CodedExperiment(spec=_spec(), code={"entry": "main.py"}, code_change_summary=" v1 ")
    assert coded.code_change_summary == "v1"
    assert CodedExperiment.from_dict(coded.to_dict()) == coded
    with pytest.raises(ValueError):
        CodedExperiment(spec="nope")

    ok = ExperimentOutcome(metrics={"ic": 0.05}, artifacts=["a1"])
    assert not ok.failed
    assert ExperimentOutcome.from_dict(ok.to_dict()) == ok
    failed = ExperimentOutcome(error="boom", error_type="RuntimeError")
    assert failed.failed


def test_feedback_truthiness_and_rendering():
    accept = ExperimentFeedback(decision=True, reason="gate passed", code_change_summary="s")
    reject = ExperimentFeedback(decision=False, reason="gate failed")
    assert bool(accept) is True
    assert bool(reject) is False
    assert "Decision: True" in str(accept)
    assert "Code Change Summary: s" in str(accept)
    assert "Code Change Summary" not in str(reject)
    with pytest.raises(ValueError):
        ExperimentFeedback(decision="yes", reason="r")
    with pytest.raises(ValueError):
        ExperimentFeedback(decision=True, reason="  ")
    assert ExperimentFeedback.from_dict(accept.to_dict()) == accept


def test_feedback_from_exception():
    feedback = ExperimentFeedback.from_exception(RuntimeError("kaboom"))
    assert bool(feedback) is False
    assert feedback.error_type == "RuntimeError"
    assert "kaboom" in feedback.reason


def test_hypothesis_feedback_fields_and_round_trip():
    feedback = HypothesisFeedback(
        decision=True,
        reason="r",
        observations="o",
        hypothesis_evaluation="e",
        new_hypothesis="n",
        acceptable=True,
    )
    text = str(feedback)
    assert "Observations: o" in text
    assert "Hypothesis Evaluation: e" in text
    assert "New Hypothesis: n" in text
    assert "Overall Acceptable: True" in text
    assert HypothesisFeedback.from_dict(feedback.to_dict()) == feedback
    with pytest.raises(ValueError):
        HypothesisFeedback(decision=True, reason="r", acceptable="yes")


def test_evostep_validation_and_linkage():
    step = EvoStep(
        step_id=STEP_ID,
        run_id=RUN_ID,
        iteration=3,
        subject=_spec(),
        hypothesis=Hypothesis(hypothesis="h", reason="r"),
        queried_knowledge=["k1"],
        feedback=ExperimentFeedback(decision=True, reason="ok"),
        parent_step_ids=[PARENT_ID],
        hypothesis_id="hyp-1",
        episode_id="ep-1",
        experiment_id="exp-1",
    )
    assert step.accepted is True
    assert step.design_text() == "Momentum persists at 21d."
    assert EvoStep.from_dict(step.to_dict()) == step

    pending = EvoStep(step_id=STEP_ID, run_id=RUN_ID, iteration=0, subject=_spec())
    assert pending.accepted is False
    assert pending.feedback is None

    with pytest.raises(ValueError):
        EvoStep(step_id="not-a-uuid", run_id=RUN_ID, iteration=0, subject=_spec())
    with pytest.raises(ValueError):
        EvoStep(step_id=STEP_ID, run_id=RUN_ID, iteration=-1, subject=_spec())
    with pytest.raises(ValueError):
        EvoStep(step_id=STEP_ID, run_id=RUN_ID, iteration=0, subject=_spec(), parent_step_ids=["nope"])
    with pytest.raises(ValueError):
        EvoStep.from_dict({"feedback": {"kind": "mystery"}})


def test_evostep_feedback_kind_round_trip():
    hypo_feedback = HypothesisFeedback(decision=False, reason="no", acceptable=False)
    step = EvoStep(step_id=STEP_ID, run_id=RUN_ID, iteration=0, subject=_spec(), feedback=hypo_feedback)
    restored = EvoStep.from_dict(step.to_dict())
    assert isinstance(restored.feedback, HypothesisFeedback)
    assert restored == step

    plain = EvoStep(step_id=STEP_ID, run_id=RUN_ID, iteration=0, subject=_spec(), feedback=ExperimentFeedback(decision=True, reason="yes"))
    restored_plain = EvoStep.from_dict(plain.to_dict())
    assert type(restored_plain.feedback) is ExperimentFeedback

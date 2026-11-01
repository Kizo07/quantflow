"""Core data model for the native Quantflow evolution loop.

Clean-room definitions in Quantflow style (frozen dataclasses, no I/O):
``Hypothesis`` → ``ExperimentSpec`` → execution → ``ExperimentFeedback`` /
``HypothesisFeedback``, sequenced as ``EvoStep`` records in a
:class:`deerflow.evolution.trace.Trace` DAG.

Content identity reuses :mod:`deerflow.knowledge.hashing` (family/execution
hashes); this module invents no hashing of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from deerflow.knowledge.hashing import execution_hash, experiment_family_hash, normalize, normalize_text

__all__ = [
    "Hypothesis",
    "ExperimentSpec",
    "CodedExperiment",
    "ExperimentOutcome",
    "ExperimentFeedback",
    "HypothesisFeedback",
    "EvoStep",
]


def _require_non_empty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _require_uuid(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a UUID string, got {type(value).__name__}.")
    try:
        return str(uuid.UUID(value.strip()))
    except (ValueError, AttributeError):
        raise ValueError(f"{name} must be a valid UUID, got {value!r}.") from None


def _optional_uuid(name: str, value: Any) -> str | None:
    if value is None:
        return None
    return _require_uuid(name, value)


def _freeze_mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}.")
    try:
        normalized = normalize(dict(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-canonicalizable: {exc}") from None
    assert isinstance(normalized, dict)
    return normalized


def _freeze_str_tuple(name: str, value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be a sequence of strings, got {type(value).__name__}.")
    items = tuple(value)
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f"{name} must contain only strings, got {item!r}.")
    return items


@dataclass(frozen=True)
class Hypothesis:
    """A proposed research belief: claim plus its rationale and concise views.

    The ``concise_*`` fields are short derived summaries used when the full
    text would overflow a stage's context budget.
    """

    hypothesis: str
    reason: str
    concise_reason: str = ""
    concise_observation: str = ""
    concise_justification: str = ""
    concise_knowledge: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "hypothesis", _require_non_empty_str("hypothesis", self.hypothesis))
        object.__setattr__(self, "reason", _require_non_empty_str("reason", self.reason))
        for name in ("concise_reason", "concise_observation", "concise_justification", "concise_knowledge"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string, got {type(value).__name__}.")
            object.__setattr__(self, name, value.strip())

    def __str__(self) -> str:
        return f"Hypothesis: {self.hypothesis}\nReason: {self.reason}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "reason": self.reason,
            "concise_reason": self.concise_reason,
            "concise_observation": self.concise_observation,
            "concise_justification": self.concise_justification,
            "concise_knowledge": self.concise_knowledge,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Hypothesis:
        if not isinstance(data, Mapping):
            raise ValueError(f"Hypothesis.from_dict expects a mapping, got {type(data).__name__}.")
        return cls(
            hypothesis=data.get("hypothesis", ""),
            reason=data.get("reason", ""),
            concise_reason=data.get("concise_reason", ""),
            concise_observation=data.get("concise_observation", ""),
            concise_justification=data.get("concise_justification", ""),
            concise_knowledge=data.get("concise_knowledge", ""),
        )


@dataclass(frozen=True)
class ExperimentSpec:
    """Concrete experiment envelope: design + code + data + params + env.

    ``design`` covers the normalized research design (hypothesis text plus
    methodology); the remaining envelopes pin the exact computation, so
    :meth:`family_hash` groups replications while :meth:`execution_hash`
    identifies exact reruns (KB hashing semantics).
    """

    design: dict[str, Any] = field(default_factory=dict)
    code: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("design", "code", "data", "parameters", "environment"):
            object.__setattr__(self, name, _freeze_mapping(name, getattr(self, name)))

    def family_hash(self) -> str:
        """Hash identifying conceptually equivalent designs (replication group)."""
        return experiment_family_hash(self.design)

    def execution_hash(self) -> str:
        """Hash identifying one exact computation (deduplication key)."""
        return execution_hash(
            design=self.design,
            code=self.code,
            data=self.data,
            parameters=self.parameters,
            environment=self.environment,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "design": normalize(self.design),
            "code": normalize(self.code),
            "data": normalize(self.data),
            "parameters": normalize(self.parameters),
            "environment": normalize(self.environment),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentSpec:
        if not isinstance(data, Mapping):
            raise ValueError(f"ExperimentSpec.from_dict expects a mapping, got {type(data).__name__}.")
        return cls(
            design=data.get("design", {}),
            code=data.get("code", {}),
            data=data.get("data", {}),
            parameters=data.get("parameters", {}),
            environment=data.get("environment", {}),
        )


@dataclass(frozen=True)
class CodedExperiment:
    """An experiment spec rendered into runnable sandbox code."""

    spec: ExperimentSpec
    code: dict[str, Any] = field(default_factory=dict)
    code_change_summary: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ExperimentSpec):
            raise ValueError(f"spec must be an ExperimentSpec, got {type(self.spec).__name__}.")
        object.__setattr__(self, "code", _freeze_mapping("code", self.code))
        if not isinstance(self.code_change_summary, str):
            raise ValueError(f"code_change_summary must be a string, got {type(self.code_change_summary).__name__}.")
        object.__setattr__(self, "code_change_summary", self.code_change_summary.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "code": normalize(self.code),
            "code_change_summary": self.code_change_summary,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CodedExperiment:
        if not isinstance(data, Mapping):
            raise ValueError(f"CodedExperiment.from_dict expects a mapping, got {type(data).__name__}.")
        return cls(
            spec=ExperimentSpec.from_dict(data.get("spec", {})),
            code=data.get("code", {}),
            code_change_summary=data.get("code_change_summary", ""),
        )


@dataclass(frozen=True)
class ExperimentOutcome:
    """Observed result of executing a coded experiment.

    Either ``error`` is None (ran to completion; ``metrics`` carry the
    quantitative result) or ``error`` describes the failure (``metrics`` may
    still carry partial observations).

    ``artifacts`` are artifact refs: KB artifact UUIDs or harvested
    sandbox paths (``outputs/...``). The knowledge adapter partitions
    them at commit time — UUIDs become result artifacts, paths ride in
    ``metrics["sandbox_artifacts"]``.
    """

    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    error: str | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _freeze_mapping("metrics", self.metrics))
        object.__setattr__(self, "artifacts", _freeze_str_tuple("artifacts", self.artifacts))
        if self.error is not None:
            if not isinstance(self.error, str):
                raise ValueError(f"error must be a string or None, got {type(self.error).__name__}.")
            object.__setattr__(self, "error", self.error.strip())
        if self.error_type is not None:
            if not isinstance(self.error_type, str):
                raise ValueError(f"error_type must be a string or None, got {type(self.error_type).__name__}.")
            object.__setattr__(self, "error_type", self.error_type.strip())

    @property
    def failed(self) -> bool:
        """True when execution raised before producing a usable result."""
        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": normalize(self.metrics),
            "artifacts": list(self.artifacts),
            "error": self.error,
            "error_type": self.error_type,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentOutcome:
        if not isinstance(data, Mapping):
            raise ValueError(f"ExperimentOutcome.from_dict expects a mapping, got {type(data).__name__}.")
        return cls(
            metrics=data.get("metrics", {}),
            artifacts=tuple(data.get("artifacts", ())),
            error=data.get("error"),
            error_type=data.get("error_type"),
        )


@dataclass(frozen=True)
class ExperimentFeedback:
    """Verdict on one executed experiment: accept (True) or reject (False)."""

    decision: bool
    reason: str
    code_change_summary: str = ""
    error: str | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, bool):
            raise ValueError(f"decision must be a bool, got {type(self.decision).__name__}.")
        object.__setattr__(self, "reason", _require_non_empty_str("reason", self.reason))
        if not isinstance(self.code_change_summary, str):
            raise ValueError(f"code_change_summary must be a string, got {type(self.code_change_summary).__name__}.")
        object.__setattr__(self, "code_change_summary", self.code_change_summary.strip())
        if self.error is not None and not isinstance(self.error, str):
            raise ValueError(f"error must be a string or None, got {type(self.error).__name__}.")
        if self.error_type is not None and not isinstance(self.error_type, str):
            raise ValueError(f"error_type must be a string or None, got {type(self.error_type).__name__}.")

    def __bool__(self) -> bool:
        return self.decision

    def __str__(self) -> str:
        text = f"Decision: {self.decision}\nReason: {self.reason}"
        if self.code_change_summary:
            text += f"\nCode Change Summary: {self.code_change_summary}"
        return text

    @classmethod
    def from_exception(cls, error: BaseException) -> ExperimentFeedback:
        """Build a rejecting feedback from a stage exception (loop safety net)."""
        return cls(
            decision=False,
            reason=f"The experiment failed with {type(error).__name__}: {error}",
            error=str(error),
            error_type=type(error).__name__,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "code_change_summary": self.code_change_summary,
            "error": self.error,
            "error_type": self.error_type,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentFeedback:
        if not isinstance(data, Mapping):
            raise ValueError(f"ExperimentFeedback.from_dict expects a mapping, got {type(data).__name__}.")
        return cls(
            decision=bool(data.get("decision", False)),
            reason=data.get("reason", ""),
            code_change_summary=data.get("code_change_summary", ""),
            error=data.get("error"),
            error_type=data.get("error_type"),
        )


@dataclass(frozen=True)
class HypothesisFeedback(ExperimentFeedback):
    """Feedback that also evaluates the hypothesis and proposes the next one."""

    observations: str | None = None
    hypothesis_evaluation: str | None = None
    new_hypothesis: str | None = None
    acceptable: bool | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("observations", "hypothesis_evaluation", "new_hypothesis"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be a string or None, got {type(value).__name__}.")
                object.__setattr__(self, name, value.strip())
        if self.acceptable is not None and not isinstance(self.acceptable, bool):
            raise ValueError(f"acceptable must be a bool or None, got {type(self.acceptable).__name__}.")

    def __str__(self) -> str:
        text = super().__str__()
        if self.observations is not None:
            text += f"\nObservations: {self.observations}"
        if self.hypothesis_evaluation is not None:
            text += f"\nHypothesis Evaluation: {self.hypothesis_evaluation}"
        if self.new_hypothesis is not None:
            text += f"\nNew Hypothesis: {self.new_hypothesis}"
        if self.acceptable is not None:
            text += f"\nOverall Acceptable: {self.acceptable}"
        return text

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "observations": self.observations,
                "hypothesis_evaluation": self.hypothesis_evaluation,
                "new_hypothesis": self.new_hypothesis,
                "acceptable": self.acceptable,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HypothesisFeedback:  # type: ignore[override]
        if not isinstance(data, Mapping):
            raise ValueError(f"HypothesisFeedback.from_dict expects a mapping, got {type(data).__name__}.")
        return cls(
            decision=bool(data.get("decision", False)),
            reason=data.get("reason", ""),
            code_change_summary=data.get("code_change_summary", ""),
            error=data.get("error"),
            error_type=data.get("error_type"),
            observations=data.get("observations"),
            hypothesis_evaluation=data.get("hypothesis_evaluation"),
            new_hypothesis=data.get("new_hypothesis"),
            acceptable=data.get("acceptable"),
        )


@dataclass(frozen=True)
class EvoStep:
    """One recorded loop iteration: subject, knowledge refs, and feedback.

    Identity linkage (stable across systems, never by mutable text):

    * ``step_id`` ↔ KB ``ExperimentRecord.id`` (``experiment_id``) ↔ spark
      ``episode_id`` ↔ ``hypothesis_id``. The step carries all three once
      the corresponding stage registers them.
    * DAG edges: the trace's ``dag_parent`` index tuples ↔ KB
      ``parent_experiment_id`` ↔ spark ``parent_hypothesis_ids``.
    """

    step_id: str
    run_id: str
    iteration: int
    subject: ExperimentSpec
    hypothesis: Hypothesis | None = None
    queried_knowledge: tuple[str, ...] = ()
    feedback: ExperimentFeedback | None = None
    parent_step_ids: tuple[str, ...] = ()
    hypothesis_id: str | None = None
    episode_id: str | None = None
    experiment_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _require_uuid("step_id", self.step_id))
        object.__setattr__(self, "run_id", _require_uuid("run_id", self.run_id))
        if not isinstance(self.iteration, int) or isinstance(self.iteration, bool) or self.iteration < 0:
            raise ValueError(f"iteration must be a non-negative int, got {self.iteration!r}.")
        if not isinstance(self.subject, ExperimentSpec):
            raise ValueError(f"subject must be an ExperimentSpec, got {type(self.subject).__name__}.")
        if self.hypothesis is not None and not isinstance(self.hypothesis, Hypothesis):
            raise ValueError(f"hypothesis must be a Hypothesis or None, got {type(self.hypothesis).__name__}.")
        object.__setattr__(self, "queried_knowledge", _freeze_str_tuple("queried_knowledge", self.queried_knowledge))
        if self.feedback is not None and not isinstance(self.feedback, ExperimentFeedback):
            raise ValueError(f"feedback must be an ExperimentFeedback or None, got {type(self.feedback).__name__}.")
        parents = _freeze_str_tuple("parent_step_ids", self.parent_step_ids)
        object.__setattr__(self, "parent_step_ids", tuple(_require_uuid("parent_step_ids[]", item) for item in parents))
        for name in ("hypothesis_id", "episode_id", "experiment_id"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must be a non-empty string or None.")
                object.__setattr__(self, name, value.strip())

    @property
    def accepted(self) -> bool:
        """True when the step has accepting feedback."""
        return self.feedback is not None and bool(self.feedback)

    def design_text(self) -> str:
        """Normalized hypothesis text used for design identity."""
        design = self.subject.design
        text = design.get("hypothesis", "")
        return normalize_text(text) if isinstance(text, str) and text.strip() else ""

    def to_dict(self) -> dict[str, Any]:
        feedback = self.feedback.to_dict() if self.feedback is not None else None
        if feedback is not None:
            feedback["kind"] = "hypothesis" if isinstance(self.feedback, HypothesisFeedback) else "experiment"
        return {
            "step_id": self.step_id,
            "run_id": self.run_id,
            "iteration": self.iteration,
            "subject": self.subject.to_dict(),
            "hypothesis": self.hypothesis.to_dict() if self.hypothesis is not None else None,
            "queried_knowledge": list(self.queried_knowledge),
            "feedback": feedback,
            "parent_step_ids": list(self.parent_step_ids),
            "hypothesis_id": self.hypothesis_id,
            "episode_id": self.episode_id,
            "experiment_id": self.experiment_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvoStep:
        if not isinstance(data, Mapping):
            raise ValueError(f"EvoStep.from_dict expects a mapping, got {type(data).__name__}.")
        raw_hypothesis = data.get("hypothesis")
        raw_feedback = data.get("feedback")
        feedback: ExperimentFeedback | None = None
        if raw_feedback is not None:
            if not isinstance(raw_feedback, Mapping):
                raise ValueError(f"feedback must be a mapping or None, got {type(raw_feedback).__name__}.")
            kind = raw_feedback.get("kind", "experiment")
            if kind == "hypothesis":
                feedback = HypothesisFeedback.from_dict(raw_feedback)
            elif kind == "experiment":
                feedback = ExperimentFeedback.from_dict(raw_feedback)
            else:
                raise ValueError(f"Unknown feedback kind: {kind!r}.")
        return cls(
            step_id=data.get("step_id", ""),
            run_id=data.get("run_id", ""),
            iteration=data.get("iteration", 0),
            subject=ExperimentSpec.from_dict(data.get("subject", {})),
            hypothesis=Hypothesis.from_dict(raw_hypothesis) if raw_hypothesis is not None else None,
            queried_knowledge=tuple(data.get("queried_knowledge", ())),
            feedback=feedback,
            parent_step_ids=tuple(data.get("parent_step_ids", ())),
            hypothesis_id=data.get("hypothesis_id"),
            episode_id=data.get("episode_id"),
            experiment_id=data.get("experiment_id"),
        )

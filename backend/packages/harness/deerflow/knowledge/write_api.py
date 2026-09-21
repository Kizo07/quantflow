"""Synchronous evidence-commit write API for the Research Knowledge Plane.

Implements the KB "evidence commits" path (deterministic, immediate writes):
``experiment_begin`` / ``experiment_commit`` / ``failure_record`` /
``assumption_record``. Knowledge *proposals* (candidate findings, validation,
promotion) are a later phase and intentionally absent.

Storage boundary: every function takes an ``ExperimentStore`` — a small
``Protocol`` the integration step binds to PostgreSQL (see the Protocol
docstrings for the table/column mapping). This module performs no I/O of its
own, holds no connections, and is safe to call from any thread or event loop
as long as the store implementation is.

Semantics:

* Inputs are validated eagerly; every violation raises
  ``KnowledgeValidationError`` before any store call.
* ``experiment_begin`` computes ``family_hash`` (normalized
  hypothesis+methodology) and ``execution_hash`` (design+code+data+params+env)
  via :mod:`deerflow.knowledge.hashing`. Records store normalized JSON-safe
  copies of methodology/parameters/code/environment/datasets.
* Idempotency: when ``idempotency_key`` is supplied, replaying the same key
  with an equal payload returns the original record; replaying it with a
  different payload raises ``IdempotencyConflictError``.
* Exact reruns: ``execution_hash`` is globally unique. Beginning an
  experiment whose execution hash already exists raises
  ``DuplicateExecutionError`` carrying the existing experiment id, so agents
  link to the prior run instead of duplicating it.
* Status machine: ``planned``/``running`` -> ``completed``/``failed`` via
  ``experiment_commit``; ``invalidated`` is terminal and set only by a future
  validation path, never by commit. Terminal records reject further commits
  (except idempotent replay of the original commit key).
* ``failure_record`` stores the failure detail record; it does not itself
  flip experiment status — ``experiment_commit(outcome="failure")`` does.
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from deerflow.knowledge.hashing import canonical_json, execution_hash, experiment_family_hash, normalize, normalize_text

OUTCOME_CLASSIFICATIONS = frozenset({"success", "failure", "inconclusive"})
FAILURE_CLASSES = frozenset({"data", "code", "statistical", "execution", "hypothesis"})
EXPERIMENT_STATUSES = frozenset({"planned", "running", "completed", "failed", "invalidated"})
BEGIN_STATUSES = frozenset({"planned", "running"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "invalidated"})
ASSUMPTION_CATEGORIES = frozenset({"data", "market", "execution", "statistical", "modeling"})
ASSUMPTION_SENSITIVITIES = frozenset({"low", "medium", "high", "unknown"})
ASSUMPTION_STATUSES = frozenset({"active", "challenged", "invalidated"})
DATASET_ROLES = frozenset({"features", "labels", "benchmark"})

__all__ = [
    "OUTCOME_CLASSIFICATIONS",
    "FAILURE_CLASSES",
    "EXPERIMENT_STATUSES",
    "BEGIN_STATUSES",
    "TERMINAL_STATUSES",
    "ASSUMPTION_CATEGORIES",
    "ASSUMPTION_SENSITIVITIES",
    "ASSUMPTION_STATUSES",
    "DATASET_ROLES",
    "KnowledgeError",
    "KnowledgeValidationError",
    "ExperimentNotFoundError",
    "DuplicateExecutionError",
    "IdempotencyConflictError",
    "InteropMismatchError",
    "ExperimentRecord",
    "FailureRecord",
    "AssumptionRecord",
    "ExperimentStore",
    "experiment_begin",
    "experiment_commit",
    "failure_record",
    "assumption_record",
]


class KnowledgeError(Exception):
    """Base class for all Knowledge API errors."""


class KnowledgeValidationError(KnowledgeError):
    """Raised when caller input violates the write contract (bad types, values, transitions)."""


class ExperimentNotFoundError(KnowledgeError):
    """Raised when referencing an experiment id the store does not contain."""

    def __init__(self, experiment_id: str):
        super().__init__(f"Experiment not found: {experiment_id}")
        self.experiment_id = experiment_id


class DuplicateExecutionError(KnowledgeError):
    """Raised when the exact computation (execution_hash) has already run."""

    def __init__(self, execution_hash_value: str, existing_experiment_id: str):
        super().__init__(f"Execution {execution_hash_value} already ran as experiment {existing_experiment_id}; link to the prior run instead of duplicating it.")
        self.execution_hash = execution_hash_value
        self.existing_experiment_id = existing_experiment_id


class IdempotencyConflictError(KnowledgeError):
    """Raised when an idempotency key is reused with a different payload."""

    def __init__(self, idempotency_key: str):
        super().__init__(f"Idempotency key {idempotency_key!r} was already used with a different payload.")
        self.idempotency_key = idempotency_key


class InteropMismatchError(KnowledgeValidationError):
    """Raised when client-echoed hashes disagree with server recomputation.

    The server always recomputes family/execution hashes from the received
    hypothesis/methodology/code/data/parameters/environment and stores its
    own values. When the caller echoes the hashes it computed locally
    (``client_family_hash``/``client_execution_hash``) and they differ, the
    two sides disagree on the normalization spec or the caller hashed
    different inputs (e.g. unresolved dataset refs) — fail loudly instead
    of storing silently divergent identity.
    """

    def __init__(self, which: str, client_value: str, server_value: str):
        super().__init__(f"Client {which} {client_value!r} != server recomputation {server_value!r}; client and server hashed different inputs or diverged from the hashing spec.")
        self.which = which
        self.client_value = client_value
        self.server_value = server_value


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _require_uuid(name: str, value: Any) -> str:
    """Validate a UUID string and return its canonical lowercase form."""
    if not isinstance(value, str):
        raise KnowledgeValidationError(f"{name} must be a UUID string, got {type(value).__name__}.")
    try:
        return str(uuid.UUID(value.strip()))
    except (ValueError, AttributeError):
        raise KnowledgeValidationError(f"{name} must be a valid UUID, got {value!r}.") from None


def _optional_uuid(name: str, value: Any) -> str | None:
    """Validate an optional UUID string (None passes through)."""
    if value is None:
        return None
    return _require_uuid(name, value)


def _require_non_empty_str(name: str, value: Any, *, max_len: int = 100_000) -> str:
    """Validate a required string and return it stripped of outer whitespace."""
    if not isinstance(value, str):
        raise KnowledgeValidationError(f"{name} must be a string, got {type(value).__name__}.")
    stripped = value.strip()
    if not stripped:
        raise KnowledgeValidationError(f"{name} must be a non-empty string.")
    if len(value) > max_len:
        raise KnowledgeValidationError(f"{name} exceeds {max_len} characters ({len(value)}).")
    return stripped


def _optional_str(name: str, value: Any, *, max_len: int = 100_000) -> str | None:
    """Validate an optional string (None passes through)."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise KnowledgeValidationError(f"{name} must be a string or None, got {type(value).__name__}.")
    if len(value) > max_len:
        raise KnowledgeValidationError(f"{name} exceeds {max_len} characters ({len(value)}).")
    return value


def _require_enum(name: str, value: Any, allowed: frozenset[str]) -> str:
    """Validate membership in a closed string vocabulary."""
    if not isinstance(value, str) or value not in allowed:
        raise KnowledgeValidationError(f"{name} must be one of {sorted(allowed)}, got {value!r}.")
    return value


def _optional_enum(name: str, value: Any, allowed: frozenset[str]) -> str | None:
    """Validate an optional enum value (None passes through)."""
    if value is None:
        return None
    return _require_enum(name, value, allowed)


def _require_jsonable(name: str, value: Any) -> Any:
    """Validate that a value canonicalizes, returning its normalized copy."""
    try:
        return normalize(value)
    except (TypeError, ValueError) as exc:
        raise KnowledgeValidationError(f"{name} must be JSON-canonicalizable: {exc}") from None


def _require_mapping(name: str, value: Any) -> dict[str, Any]:
    """Validate a required mapping and return its normalized plain-dict copy."""
    if not isinstance(value, Mapping):
        raise KnowledgeValidationError(f"{name} must be a mapping, got {type(value).__name__}.")
    normalized = _require_jsonable(name, dict(value))
    assert isinstance(normalized, dict)
    return normalized


def _optional_mapping(name: str, value: Any) -> dict[str, Any]:
    """Validate an optional mapping (None becomes {})."""
    if value is None:
        return {}
    return _require_mapping(name, value)


def _require_idempotency_key(value: Any) -> str:
    """Validate a required idempotency key (opaque non-empty string, max 255 chars)."""
    key = _require_non_empty_str("idempotency_key", value, max_len=255)
    if any(ch.isspace() for ch in key):
        raise KnowledgeValidationError("idempotency_key must not contain whitespace.")
    return key


def _optional_idempotency_key(value: Any) -> str | None:
    """Validate an optional idempotency key (None passes through)."""
    if value is None:
        return None
    return _require_idempotency_key(value)


def _validate_datasets(datasets: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Validate dataset links and return normalized copies.

    Each entry must be ``{"dataset_version_id": <uuid>, "role": <features|labels|benchmark>}``.
    """
    if datasets is None:
        return []
    if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes, bytearray)):
        raise KnowledgeValidationError(f"datasets must be a sequence of mappings, got {type(datasets).__name__}.")
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(datasets):
        if not isinstance(entry, Mapping):
            raise KnowledgeValidationError(f"datasets[{index}] must be a mapping, got {type(entry).__name__}.")
        dataset_version_id = _require_uuid(f"datasets[{index}].dataset_version_id", entry.get("dataset_version_id"))
        role = _require_enum(f"datasets[{index}].role", entry.get("role"), DATASET_ROLES)
        normalized.append({"dataset_version_id": dataset_version_id, "role": role})
    return normalized


def _validate_artifacts(artifacts: Sequence[str] | None, name: str) -> list[str]:
    """Validate artifact id lists (UUID strings)."""
    if artifacts is None:
        return []
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes, bytearray)):
        raise KnowledgeValidationError(f"{name} must be a sequence of artifact-id UUID strings, got {type(artifacts).__name__}.")
    return [_require_uuid(f"{name}[{index}]", item) for index, item in enumerate(artifacts)]


@dataclass(frozen=True)
class ExperimentRecord:
    """Immutable snapshot of an experiment row (mirrors the KB ``experiment`` table)."""

    id: str
    project_id: str
    created_by_run_id: str
    hypothesis: str
    methodology: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    code: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    family_hash: str = ""
    execution_hash: str = ""
    status: str = "planned"
    outcome: str | None = None
    failure_class: str | None = None
    metrics: dict[str, Any] | None = None
    result_artifacts: list[str] = field(default_factory=list)
    parent_experiment_id: str | None = None
    replicated_experiment_id: str | None = None
    idempotency_key: str | None = None
    commit_idempotency_key: str | None = None
    started_at: str = ""
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the record."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "created_by_run_id": self.created_by_run_id,
            "hypothesis": self.hypothesis,
            "methodology": normalize(self.methodology),
            "parameters": normalize(self.parameters),
            "datasets": normalize(self.datasets),
            "code": normalize(self.code),
            "environment": normalize(self.environment),
            "family_hash": self.family_hash,
            "execution_hash": self.execution_hash,
            "status": self.status,
            "outcome": self.outcome,
            "failure_class": self.failure_class,
            "metrics": normalize(self.metrics) if self.metrics is not None else None,
            "result_artifacts": list(self.result_artifacts),
            "parent_experiment_id": self.parent_experiment_id,
            "replicated_experiment_id": self.replicated_experiment_id,
            "idempotency_key": self.idempotency_key,
            "commit_idempotency_key": self.commit_idempotency_key,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass(frozen=True)
class FailureRecord:
    """Immutable snapshot of a failure detail record for one experiment."""

    id: str
    experiment_id: str
    failure_class: str
    notes: str = ""
    idempotency_key: str | None = None
    recorded_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the record."""
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "failure_class": self.failure_class,
            "notes": self.notes,
            "idempotency_key": self.idempotency_key,
            "recorded_at": self.recorded_at,
        }


@dataclass(frozen=True)
class AssumptionRecord:
    """Immutable snapshot of an assumption row (mirrors the KB ``assumption`` table)."""

    id: str
    experiment_id: str | None
    statement: str
    category: str
    sensitivity: str = "unknown"
    tested: bool = False
    status: str = "active"
    evidence_artifact_id: str | None = None
    idempotency_key: str | None = None
    recorded_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the record."""
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "statement": self.statement,
            "category": self.category,
            "sensitivity": self.sensitivity,
            "tested": self.tested,
            "status": self.status,
            "evidence_artifact_id": self.evidence_artifact_id,
            "idempotency_key": self.idempotency_key,
            "recorded_at": self.recorded_at,
        }


@runtime_checkable
class ExperimentStore(Protocol):
    """Storage boundary for the write API; the integration step binds this to PostgreSQL.

    PG binding sketch (KB canonical schema): experiments map to the
    ``experiment`` table (``execution_hash`` UNIQUE, ``idempotency_key``
    UNIQUE, ``commit_idempotency_key`` UNIQUE NULLS NOT DISTINCT), failures
    to ``experiment`` outcome columns plus a failure detail row keyed by its
    own idempotency key, assumptions to the ``assumption`` table. Every
    method below is one indexed lookup or one row write; the PG
    implementation wraps each public API call in a single transaction and
    surfaces unique violations as the corresponding typed error.
    """

    def find_experiment_by_idempotency_key(self, idempotency_key: str) -> ExperimentRecord | None:
        """Return the experiment begun under this key, or None. (``experiment`` by ``idempotency_key``.)"""
        ...

    def find_experiment_by_commit_key(self, idempotency_key: str) -> ExperimentRecord | None:
        """Return the experiment committed under this key, or None. (``experiment`` by ``commit_idempotency_key``.)"""
        ...

    def find_experiment_by_execution_hash(self, execution_hash: str) -> ExperimentRecord | None:
        """Return the experiment with this execution hash, or None. (``experiment`` by ``execution_hash``.)"""
        ...

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        """Return the experiment by id, or None. (``experiment`` by primary key.)"""
        ...

    def insert_experiment(self, record: ExperimentRecord) -> None:
        """Persist a new experiment row (plus its ``experiment_dataset`` links)."""
        ...

    def update_experiment(self, record: ExperimentRecord) -> None:
        """Persist a full-row update of an existing experiment (commit path)."""
        ...

    def find_failure_by_idempotency_key(self, idempotency_key: str) -> FailureRecord | None:
        """Return the failure recorded under this key, or None."""
        ...

    def insert_failure(self, record: FailureRecord) -> None:
        """Persist a new failure detail row."""
        ...

    def find_assumption_by_idempotency_key(self, idempotency_key: str) -> AssumptionRecord | None:
        """Return the assumption recorded under this key, or None."""
        ...

    def insert_assumption(self, record: AssumptionRecord) -> None:
        """Persist a new assumption row."""
        ...


def _begin_payload_fingerprint(*, project_id: str, created_by_run_id: str, execution_hash_value: str) -> str:
    """Canonical fingerprint identifying a begin payload for idempotency comparison."""
    return canonical_json({"project_id": project_id, "created_by_run_id": created_by_run_id, "execution_hash": execution_hash_value})


def experiment_begin(
    store: ExperimentStore,
    *,
    project_id: str,
    hypothesis: str,
    methodology: Mapping[str, Any],
    parameters: Mapping[str, Any] | None = None,
    datasets: Sequence[Mapping[str, Any]] | None = None,
    code: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    created_by_run_id: str,
    parent_experiment_id: str | None = None,
    replicated_experiment_id: str | None = None,
    status: str = "planned",
    idempotency_key: str | None = None,
    client_family_hash: str | None = None,
    client_execution_hash: str | None = None,
) -> ExperimentRecord:
    """Begin (register) a new experiment and return its record.

    Computes ``family_hash`` over the normalized ``{"hypothesis",
    "methodology"}`` design and ``execution_hash`` over
    design+code+data+parameters+environment. Raises ``DuplicateExecutionError``
    when the exact computation already ran. When the caller echoes locally
    computed hashes, they are verified against the recomputation and a
    mismatch raises ``InteropMismatchError``.

    Args:
        store: Storage boundary implementation.
        project_id: Owning research project UUID.
        hypothesis: Research hypothesis prose (whitespace-normalized for identity).
        methodology: Methodology mapping (universe, horizon, protocol, cost model, ...).
        parameters: Full parameter mapping including random seeds (default {}).
        datasets: Dataset links, each ``{"dataset_version_id": <uuid>, "role": ...}``.
        code: Code identity mapping (artifact sha256, git commit, ...; default {}).
        environment: Runtime identity mapping (lock hash, versions; default {}).
        created_by_run_id: Agent run UUID registering the experiment.
        parent_experiment_id: Optional parent experiment UUID (follow-ups).
        replicated_experiment_id: Optional experiment UUID this run replicates.
        status: Initial status, ``planned`` or ``running``.
        idempotency_key: Optional opaque key; replays return the original record.

    Returns:
        The persisted ``ExperimentRecord`` (or the original on idempotent replay).

    Raises:
        KnowledgeValidationError: On any invalid input.
        DuplicateExecutionError: When ``execution_hash`` already exists.
        IdempotencyConflictError: When the key was used with a different payload.
    """
    validated_project_id = _require_uuid("project_id", project_id)
    validated_run_id = _require_uuid("created_by_run_id", created_by_run_id)
    validated_hypothesis = _require_non_empty_str("hypothesis", hypothesis)
    validated_methodology = _require_mapping("methodology", methodology)
    validated_parameters = _optional_mapping("parameters", parameters)
    validated_code = _optional_mapping("code", code)
    validated_environment = _optional_mapping("environment", environment)
    validated_datasets = _validate_datasets(datasets)
    validated_parent = _optional_uuid("parent_experiment_id", parent_experiment_id)
    validated_replicated = _optional_uuid("replicated_experiment_id", replicated_experiment_id)
    validated_status = _require_enum("status", status, BEGIN_STATUSES)
    validated_key = _optional_idempotency_key(idempotency_key)

    design = {"hypothesis": normalize_text(validated_hypothesis), "methodology": validated_methodology}
    data_envelope = {"datasets": validated_datasets}
    family_hash_value = experiment_family_hash(design)
    execution_hash_value = execution_hash(design=design, code=validated_code, data=data_envelope, parameters=validated_parameters, environment=validated_environment)

    if client_family_hash is not None and client_family_hash != family_hash_value:
        raise InteropMismatchError("family_hash", client_family_hash, family_hash_value)
    if client_execution_hash is not None and client_execution_hash != execution_hash_value:
        raise InteropMismatchError("execution_hash", client_execution_hash, execution_hash_value)

    fingerprint = _begin_payload_fingerprint(project_id=validated_project_id, created_by_run_id=validated_run_id, execution_hash_value=execution_hash_value)
    if validated_key is not None:
        existing = store.find_experiment_by_idempotency_key(validated_key)
        if existing is not None:
            existing_fingerprint = _begin_payload_fingerprint(project_id=existing.project_id, created_by_run_id=existing.created_by_run_id, execution_hash_value=existing.execution_hash)
            if existing_fingerprint != fingerprint:
                raise IdempotencyConflictError(validated_key)
            return existing

    duplicate = store.find_experiment_by_execution_hash(execution_hash_value)
    if duplicate is not None:
        raise DuplicateExecutionError(execution_hash_value, duplicate.id)

    record = ExperimentRecord(
        id=str(uuid.uuid4()),
        project_id=validated_project_id,
        created_by_run_id=validated_run_id,
        hypothesis=validated_hypothesis,
        methodology=validated_methodology,
        parameters=validated_parameters,
        datasets=validated_datasets,
        code=validated_code,
        environment=validated_environment,
        family_hash=family_hash_value,
        execution_hash=execution_hash_value,
        status=validated_status,
        parent_experiment_id=validated_parent,
        replicated_experiment_id=validated_replicated,
        idempotency_key=validated_key,
        started_at=_utcnow_iso(),
    )
    store.insert_experiment(record)
    return record


def _commit_payload_fingerprint(*, experiment_id: str, outcome: str, failure_class: str | None, metrics: dict[str, Any], result_artifacts: list[str]) -> str:
    """Canonical fingerprint identifying a commit payload for idempotency comparison."""
    return canonical_json({"experiment_id": experiment_id, "outcome": outcome, "failure_class": failure_class, "metrics": metrics, "result_artifacts": result_artifacts})


def experiment_commit(
    store: ExperimentStore,
    *,
    experiment_id: str,
    metrics: Mapping[str, Any],
    outcome: str,
    failure_class: str | None = None,
    result_artifacts: Sequence[str] | None = None,
    idempotency_key: str | None = None,
) -> ExperimentRecord:
    """Commit results for a begun experiment and return the updated record.

    Transitions ``planned``/``running`` to ``completed`` (success/inconclusive)
    or ``failed`` (failure). ``failure_class`` is required if and only if
    ``outcome == "failure"``.

    Args:
        store: Storage boundary implementation.
        experiment_id: Experiment UUID from ``experiment_begin``.
        metrics: Result metrics mapping (JSON-canonicalizable, may be empty).
        outcome: One of ``success`` / ``failure`` / ``inconclusive``.
        failure_class: Required when outcome is ``failure`` (one of ``data`` /
            ``code`` / ``statistical`` / ``execution`` / ``hypothesis``);
            must be None otherwise.
        result_artifacts: Optional artifact-id UUIDs for outputs/logs/figures.
        idempotency_key: Optional opaque key; replays return the committed record.

    Returns:
        The updated terminal ``ExperimentRecord``.

    Raises:
        KnowledgeValidationError: On invalid input or an illegal transition
            (e.g. committing an already-terminal experiment with a fresh key).
        ExperimentNotFoundError: When the experiment id is unknown.
        IdempotencyConflictError: When the key was used with a different payload.
    """
    validated_id = _require_uuid("experiment_id", experiment_id)
    validated_metrics = _require_mapping("metrics", metrics)
    validated_outcome = _require_enum("outcome", outcome, OUTCOME_CLASSIFICATIONS)
    validated_failure_class = _optional_enum("failure_class", failure_class, FAILURE_CLASSES)
    validated_artifacts = _validate_artifacts(result_artifacts, "result_artifacts")
    validated_key = _optional_idempotency_key(idempotency_key)

    if validated_outcome == "failure" and validated_failure_class is None:
        raise KnowledgeValidationError('failure_class is required when outcome is "failure".')
    if validated_outcome != "failure" and validated_failure_class is not None:
        raise KnowledgeValidationError(f'failure_class must be None when outcome is "{validated_outcome}".')

    fingerprint = _commit_payload_fingerprint(experiment_id=validated_id, outcome=validated_outcome, failure_class=validated_failure_class, metrics=validated_metrics, result_artifacts=validated_artifacts)
    if validated_key is not None:
        committed = store.find_experiment_by_commit_key(validated_key)
        if committed is not None:
            committed_fingerprint = _commit_payload_fingerprint(
                experiment_id=committed.id,
                outcome=committed.outcome or "",
                failure_class=committed.failure_class,
                metrics=committed.metrics or {},
                result_artifacts=committed.result_artifacts,
            )
            if committed.id != validated_id or committed_fingerprint != fingerprint:
                raise IdempotencyConflictError(validated_key)
            return committed

    current = store.get_experiment(validated_id)
    if current is None:
        raise ExperimentNotFoundError(validated_id)
    if current.status in TERMINAL_STATUSES:
        raise KnowledgeValidationError(f"Experiment {validated_id} is already terminal (status={current.status!r}); commits are single-shot.")

    updated = replace(
        current,
        status="failed" if validated_outcome == "failure" else "completed",
        outcome=validated_outcome,
        failure_class=validated_failure_class,
        metrics=validated_metrics,
        result_artifacts=validated_artifacts,
        commit_idempotency_key=validated_key,
        completed_at=_utcnow_iso(),
    )
    store.update_experiment(updated)
    return updated


def failure_record(
    store: ExperimentStore,
    *,
    experiment_id: str,
    failure_class: str,
    notes: str = "",
    idempotency_key: str | None = None,
) -> FailureRecord:
    """Record a failure detail for an experiment.

    This stores the failure *detail record*; experiment status itself flips
    only via ``experiment_commit(outcome="failure")``.

    Args:
        store: Storage boundary implementation.
        experiment_id: Experiment UUID the failure belongs to.
        failure_class: One of ``data`` / ``code`` / ``statistical`` /
            ``execution`` / ``hypothesis``.
        notes: Free-text failure notes (may be empty).
        idempotency_key: Optional opaque key; replays return the original record.

    Raises:
        KnowledgeValidationError: On invalid input.
        ExperimentNotFoundError: When the experiment id is unknown.
        IdempotencyConflictError: When the key was used with a different payload.
    """
    validated_id = _require_uuid("experiment_id", experiment_id)
    validated_class = _require_enum("failure_class", failure_class, FAILURE_CLASSES)
    if not isinstance(notes, str):
        raise KnowledgeValidationError(f"notes must be a string, got {type(notes).__name__}.")
    validated_key = _optional_idempotency_key(idempotency_key)

    fingerprint = canonical_json({"experiment_id": validated_id, "failure_class": validated_class, "notes": notes})
    if validated_key is not None:
        existing = store.find_failure_by_idempotency_key(validated_key)
        if existing is not None:
            existing_fingerprint = canonical_json({"experiment_id": existing.experiment_id, "failure_class": existing.failure_class, "notes": existing.notes})
            if existing_fingerprint != fingerprint:
                raise IdempotencyConflictError(validated_key)
            return existing

    if store.get_experiment(validated_id) is None:
        raise ExperimentNotFoundError(validated_id)

    record = FailureRecord(
        id=str(uuid.uuid4()),
        experiment_id=validated_id,
        failure_class=validated_class,
        notes=notes,
        idempotency_key=validated_key,
        recorded_at=_utcnow_iso(),
    )
    store.insert_failure(record)
    return record


def assumption_record(
    store: ExperimentStore,
    *,
    statement: str,
    category: str,
    experiment_id: str | None = None,
    sensitivity: str = "unknown",
    tested: bool = False,
    status: str = "active",
    evidence_artifact_id: str | None = None,
    idempotency_key: str | None = None,
) -> AssumptionRecord:
    """Record an assumption, optionally attached to an experiment.

    Args:
        store: Storage boundary implementation.
        statement: Assumption prose (non-empty).
        category: One of ``data`` / ``market`` / ``execution`` /
            ``statistical`` / ``modeling``.
        experiment_id: Optional experiment UUID this assumption was made under.
        sensitivity: One of ``low`` / ``medium`` / ``high`` / ``unknown``.
        tested: Whether the assumption has been explicitly tested (strict bool).
        status: One of ``active`` / ``challenged`` / ``invalidated``.
        evidence_artifact_id: Optional artifact UUID evidencing the assumption.
        idempotency_key: Optional opaque key; replays return the original record.

    Raises:
        KnowledgeValidationError: On invalid input.
        ExperimentNotFoundError: When ``experiment_id`` is given but unknown.
        IdempotencyConflictError: When the key was used with a different payload.
    """
    validated_statement = _require_non_empty_str("statement", statement)
    validated_category = _require_enum("category", category, ASSUMPTION_CATEGORIES)
    validated_experiment = _optional_uuid("experiment_id", experiment_id)
    validated_sensitivity = _require_enum("sensitivity", sensitivity, ASSUMPTION_SENSITIVITIES)
    if not isinstance(tested, bool):
        raise KnowledgeValidationError(f"tested must be a bool, got {type(tested).__name__}.")
    validated_status = _require_enum("status", status, ASSUMPTION_STATUSES)
    validated_evidence = _optional_uuid("evidence_artifact_id", evidence_artifact_id)
    validated_key = _optional_idempotency_key(idempotency_key)

    fingerprint = canonical_json(
        {
            "statement": validated_statement,
            "category": validated_category,
            "experiment_id": validated_experiment,
            "sensitivity": validated_sensitivity,
            "tested": tested,
            "status": validated_status,
            "evidence_artifact_id": validated_evidence,
        }
    )
    if validated_key is not None:
        existing = store.find_assumption_by_idempotency_key(validated_key)
        if existing is not None:
            existing_fingerprint = canonical_json(
                {
                    "statement": existing.statement,
                    "category": existing.category,
                    "experiment_id": existing.experiment_id,
                    "sensitivity": existing.sensitivity,
                    "tested": existing.tested,
                    "status": existing.status,
                    "evidence_artifact_id": existing.evidence_artifact_id,
                }
            )
            if existing_fingerprint != fingerprint:
                raise IdempotencyConflictError(validated_key)
            return existing

    if validated_experiment is not None and store.get_experiment(validated_experiment) is None:
        raise ExperimentNotFoundError(validated_experiment)

    record = AssumptionRecord(
        id=str(uuid.uuid4()),
        experiment_id=validated_experiment,
        statement=validated_statement,
        category=validated_category,
        sensitivity=validated_sensitivity,
        tested=tested,
        status=validated_status,
        evidence_artifact_id=validated_evidence,
        idempotency_key=validated_key,
        recorded_at=_utcnow_iso(),
    )
    store.insert_assumption(record)
    return record

"""Tests for evolution idempotency-key derivation and KB replay semantics (offline).

Key derivation is pinned here; the replay half exercises the real
``deerflow.knowledge.write_api`` against a small in-memory
``ExperimentStore`` fake to prove the loop's write pattern (same key +
equal payload → replay; same key + different payload → conflict;
same execution hash → link to prior run) holds end to end.
"""

import uuid

import pytest

from deerflow.evolution.loop import (
    BACKTEST_SEQUENCE_BASE,
    BEGIN_SEQUENCE_BASE,
    COMMIT_SEQUENCE_BASE,
    FAILURE_SEQUENCE_BASE,
    OUTCOME_SEQUENCE_BASE,
    PROMOTE_SEQUENCE_BASE,
    PROPOSE_SEQUENCE_BASE,
    REGISTER_SEQUENCE_BASE,
    derive_key,
)
from deerflow.knowledge.hashing import make_idempotency_key
from deerflow.knowledge.write_api import (
    DuplicateExecutionError,
    ExperimentRecord,
    IdempotencyConflictError,
    experiment_begin,
    experiment_commit,
)

RUN_ID = "11111111-1111-1111-1111-111111111111"
PROJECT_ID = "22222222-2222-2222-2222-222222222222"

METHODOLOGY = {"universe": "sp500", "horizon": "21d", "protocol": "cs-momentum"}


class FakeExperimentStore:
    """Minimal in-memory ExperimentStore with KB idempotency semantics."""

    def __init__(self) -> None:
        self._by_id: dict[str, ExperimentRecord] = {}
        self._by_begin_key: dict[str, str] = {}
        self._by_commit_key: dict[str, str] = {}
        self._by_execution: dict[str, str] = {}
        self._failures: dict[str, object] = {}
        self._assumptions: dict[str, object] = {}

    def find_experiment_by_idempotency_key(self, key):
        exp_id = self._by_begin_key.get(key)
        return self._by_id.get(exp_id) if exp_id else None

    def find_experiment_by_commit_key(self, key):
        exp_id = self._by_commit_key.get(key)
        return self._by_id.get(exp_id) if exp_id else None

    def find_experiment_by_execution_hash(self, value):
        exp_id = self._by_execution.get(value)
        return self._by_id.get(exp_id) if exp_id else None

    def get_experiment(self, experiment_id):
        return self._by_id.get(experiment_id)

    def insert_experiment(self, record):
        self._by_id[record.id] = record
        if record.idempotency_key:
            self._by_begin_key[record.idempotency_key] = record.id
        self._by_execution[record.execution_hash] = record.id

    def update_experiment(self, record):
        self._by_id[record.id] = record
        if record.commit_idempotency_key:
            self._by_commit_key[record.commit_idempotency_key] = record.id

    def find_failure_by_idempotency_key(self, key):
        return self._failures.get(key)

    def insert_failure(self, record):
        self._failures[record.idempotency_key] = record

    def find_assumption_by_idempotency_key(self, key):
        return self._assumptions.get(key)

    def insert_assumption(self, record):
        self._assumptions[record.idempotency_key] = record


def test_sequence_bases_match_plan_mapping():
    assert (PROPOSE_SEQUENCE_BASE, REGISTER_SEQUENCE_BASE, BEGIN_SEQUENCE_BASE) == (1000, 2000, 3000)
    assert BACKTEST_SEQUENCE_BASE == 2500
    assert (OUTCOME_SEQUENCE_BASE, COMMIT_SEQUENCE_BASE) == (4000, 5000)
    assert (FAILURE_SEQUENCE_BASE, PROMOTE_SEQUENCE_BASE) == (6000, 7000)


def test_key_derivation_is_deterministic_and_canonical():
    payload = {"b": [1, 2], "a": "x"}
    first = derive_key(RUN_ID, PROPOSE_SEQUENCE_BASE, 3, payload)
    again = derive_key(RUN_ID, PROPOSE_SEQUENCE_BASE, 3, {"a": "x", "b": [1, 2]})
    assert first == again
    assert first == make_idempotency_key(RUN_ID, PROPOSE_SEQUENCE_BASE + 3, payload)
    assert len(first) == 64


def test_any_field_change_changes_the_key():
    base = derive_key(RUN_ID, PROPOSE_SEQUENCE_BASE, 0, {"a": 1})
    assert derive_key(RUN_ID, PROPOSE_SEQUENCE_BASE, 0, {"a": 2}) != base
    assert derive_key(RUN_ID, PROPOSE_SEQUENCE_BASE, 0, {"a": 1, "b": 0}) != base
    assert derive_key(RUN_ID, PROPOSE_SEQUENCE_BASE, 1, {"a": 1}) != base
    assert derive_key(RUN_ID, REGISTER_SEQUENCE_BASE, 0, {"a": 1}) != base
    assert derive_key(str(uuid.uuid4()), PROPOSE_SEQUENCE_BASE, 0, {"a": 1}) != base


def test_each_write_has_a_distinct_sequence_base():
    """Every loop write derives under its own base (same payload, 8 keys)."""
    payload = {"same": "payload"}
    bases = (
        PROPOSE_SEQUENCE_BASE,
        REGISTER_SEQUENCE_BASE,
        BACKTEST_SEQUENCE_BASE,
        BEGIN_SEQUENCE_BASE,
        OUTCOME_SEQUENCE_BASE,
        COMMIT_SEQUENCE_BASE,
        FAILURE_SEQUENCE_BASE,
        PROMOTE_SEQUENCE_BASE,
    )
    keys = {derive_key(RUN_ID, base, 2, payload) for base in bases}
    assert len(keys) == len(bases)


def test_derive_key_rejects_bad_inputs():
    with pytest.raises((TypeError, ValueError)):
        derive_key("", PROPOSE_SEQUENCE_BASE, 0, {})
    with pytest.raises(ValueError):
        derive_key(RUN_ID, PROPOSE_SEQUENCE_BASE, -1, {})
    with pytest.raises(ValueError):
        derive_key(RUN_ID, -5, 0, {})


def test_begin_replay_returns_original_record():
    store = FakeExperimentStore()
    key = derive_key(RUN_ID, BEGIN_SEQUENCE_BASE, 0, {"project_id": PROJECT_ID, "created_by_run_id": RUN_ID, "execution_hash": "ex0"})
    first = experiment_begin(
        store,
        project_id=PROJECT_ID,
        hypothesis="Momentum persists.",
        methodology=METHODOLOGY,
        created_by_run_id=RUN_ID,
        idempotency_key=key,
    )
    replay = experiment_begin(
        store,
        project_id=PROJECT_ID,
        hypothesis="Momentum persists.",
        methodology=METHODOLOGY,
        created_by_run_id=RUN_ID,
        idempotency_key=key,
    )
    assert replay.id == first.id


def test_begin_key_conflict_on_different_payload():
    store = FakeExperimentStore()
    key = derive_key(RUN_ID, BEGIN_SEQUENCE_BASE, 0, {"project_id": PROJECT_ID, "created_by_run_id": RUN_ID, "execution_hash": "ex0"})
    experiment_begin(
        store,
        project_id=PROJECT_ID,
        hypothesis="Momentum persists.",
        methodology=METHODOLOGY,
        created_by_run_id=RUN_ID,
        idempotency_key=key,
    )
    with pytest.raises(IdempotencyConflictError):
        experiment_begin(
            store,
            project_id=PROJECT_ID,
            hypothesis="Reversal wins instead.",
            methodology=METHODOLOGY,
            created_by_run_id=RUN_ID,
            idempotency_key=key,
        )


def test_execution_hash_rerun_links_to_prior_run():
    store = FakeExperimentStore()
    first = experiment_begin(
        store,
        project_id=PROJECT_ID,
        hypothesis="Momentum persists.",
        methodology=METHODOLOGY,
        parameters={"seed": 7},
        created_by_run_id=RUN_ID,
    )
    with pytest.raises(DuplicateExecutionError) as excinfo:
        experiment_begin(
            store,
            project_id=PROJECT_ID,
            hypothesis="Momentum persists.",
            methodology=METHODOLOGY,
            parameters={"seed": 7},
            created_by_run_id=RUN_ID,
        )
    assert excinfo.value.existing_experiment_id == first.id


def test_commit_replay_returns_committed_record():
    store = FakeExperimentStore()
    record = experiment_begin(
        store,
        project_id=PROJECT_ID,
        hypothesis="Momentum persists.",
        methodology=METHODOLOGY,
        created_by_run_id=RUN_ID,
    )
    commit_key = derive_key(RUN_ID, COMMIT_SEQUENCE_BASE, 0, {"experiment_id": record.id, "outcome": "success"})
    first = experiment_commit(store, experiment_id=record.id, outcome="success", metrics={"ic": 0.05}, idempotency_key=commit_key)
    replay = experiment_commit(store, experiment_id=record.id, outcome="success", metrics={"ic": 0.05}, idempotency_key=commit_key)
    assert replay.id == first.id
    assert replay.status == "completed"

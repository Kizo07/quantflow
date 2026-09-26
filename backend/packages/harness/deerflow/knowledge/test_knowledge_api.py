"""Standalone tests for the Research Knowledge Plane write API (Phase 1).

Runs with no database, no config files, and no integration wiring: a fake
in-memory store implements the ``ExperimentStore`` / ``ExperimentSearchStore``
boundaries defined in ``write_api.py`` / ``search.py``.

Run from anywhere (the bootstrap below locates the harness package)::

    python -m pytest backend/packages/harness/deerflow/knowledge/test_knowledge_api.py -q
"""

import sys
import uuid
from pathlib import Path

_HARNESS_DIR = Path(__file__).resolve().parents[2]
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

import pytest  # noqa: E402

from deerflow.knowledge import config as knowledge_config  # noqa: E402
from deerflow.knowledge import (  # noqa: E402
    hashing,  # noqa: E402
    write_api,  # noqa: E402
)
from deerflow.knowledge import search as knowledge_search  # noqa: E402
from deerflow.knowledge.search import ExperimentFilter, ExperimentSearchResult  # noqa: E402
from deerflow.knowledge.write_api import (  # noqa: E402
    AssumptionRecord,
    DuplicateExecutionError,
    ExperimentNotFoundError,
    ExperimentRecord,
    FailureRecord,
    IdempotencyConflictError,
    InteropMismatchError,
    KnowledgeValidationError,
)

PROJECT_ID = "11111111-1111-1111-1111-111111111111"
PROJECT_ID_2 = "22222222-2222-2222-2222-222222222222"
RUN_ID = "33333333-3333-3333-3333-333333333333"
DATASET_ID = "44444444-4444-4444-4444-444444444444"


def make_methodology(**overrides):
    methodology = {
        "universe": "sp500-pit",
        "frequency": "daily",
        "horizon": "21d",
        "sample_period": ["2015-01-01", "2025-12-31"],
        "train_test_protocol": "walk-forward",
        "portfolio_construction": "long-short-decile",
        "transaction_cost_model": {"bps": 10},
        "slippage_model": "none",
        "neutralization": ["sector"],
        "rebalance_rule": "ME",
    }
    methodology.update(overrides)
    return methodology


def make_design():
    return {"hypothesis": "Cross-sectional momentum persists net of costs.", "methodology": make_methodology()}


class FakeKnowledgeStore:
    """In-memory fake implementing both storage boundaries (structural typing)."""

    def __init__(self):
        self.experiments: dict[str, ExperimentRecord] = {}
        self.by_begin_key: dict[str, ExperimentRecord] = {}
        self.by_commit_key: dict[str, ExperimentRecord] = {}
        self.by_execution: dict[str, ExperimentRecord] = {}
        self.failures: dict[str, FailureRecord] = {}
        self.by_failure_key: dict[str, FailureRecord] = {}
        self.assumptions: dict[str, AssumptionRecord] = {}
        self.by_assumption_key: dict[str, AssumptionRecord] = {}
        self.used_find_by_family_hash = 0
        self.used_search_experiments = 0

    # -- ExperimentStore --
    def find_experiment_by_idempotency_key(self, key):
        return self.by_begin_key.get(key)

    def find_experiment_by_commit_key(self, key):
        return self.by_commit_key.get(key)

    def find_experiment_by_execution_hash(self, execution_hash_value):
        return self.by_execution.get(execution_hash_value)

    def get_experiment(self, experiment_id):
        return self.experiments.get(experiment_id)

    def insert_experiment(self, record):
        self.experiments[record.id] = record
        self.by_execution[record.execution_hash] = record
        if record.idempotency_key is not None:
            self.by_begin_key[record.idempotency_key] = record
        if record.commit_idempotency_key is not None:
            self.by_commit_key[record.commit_idempotency_key] = record

    def update_experiment(self, record):
        self.experiments[record.id] = record
        self.by_execution[record.execution_hash] = record
        if record.idempotency_key is not None:
            self.by_begin_key[record.idempotency_key] = record
        if record.commit_idempotency_key is not None:
            self.by_commit_key[record.commit_idempotency_key] = record

    def find_failure_by_idempotency_key(self, key):
        return self.by_failure_key.get(key)

    def insert_failure(self, record):
        self.failures[record.id] = record
        if record.idempotency_key is not None:
            self.by_failure_key[record.idempotency_key] = record

    def find_assumption_by_idempotency_key(self, key):
        return self.by_assumption_key.get(key)

    def insert_assumption(self, record):
        self.assumptions[record.id] = record
        if record.idempotency_key is not None:
            self.by_assumption_key[record.idempotency_key] = record

    # -- ExperimentSearchStore --
    def find_by_execution_hash(self, execution_hash_value):
        return self.by_execution.get(execution_hash_value)

    def find_by_family_hash(self, family_hash_value, *, limit, offset):
        self.used_find_by_family_hash += 1
        matches = [rec for rec in self.experiments.values() if rec.family_hash == family_hash_value]
        matches.sort(key=lambda rec: rec.started_at)
        return matches[offset : offset + limit]

    def search_experiments(self, filters, *, limit, offset):
        self.used_search_experiments += 1
        matches = []
        for rec in self.experiments.values():
            if filters.family_hash is not None and rec.family_hash != filters.family_hash:
                continue
            if filters.project_id is not None and rec.project_id != filters.project_id:
                continue
            if filters.status is not None and rec.status != filters.status:
                continue
            if filters.outcome is not None and rec.outcome != filters.outcome:
                continue
            if filters.failure_class is not None and rec.failure_class != filters.failure_class:
                continue
            if filters.hypothesis_contains is not None and filters.hypothesis_contains.lower() not in rec.hypothesis.lower():
                continue
            matches.append(rec)
        matches.sort(key=lambda rec: rec.started_at)
        return matches[offset : offset + limit]


@pytest.fixture
def store():
    return FakeKnowledgeStore()


@pytest.fixture
def begun(store):
    return write_api.experiment_begin(
        store,
        project_id=PROJECT_ID,
        hypothesis="Cross-sectional momentum persists net of costs.",
        methodology=make_methodology(),
        parameters={"lookback_days": 126, "top_n": 20, "seed": 7},
        datasets=[{"dataset_version_id": DATASET_ID, "role": "features"}],
        code={"git_commit": "abc123", "artifact_sha256": "00" * 32},
        environment={"lock_hash": "deadbeef"},
        created_by_run_id=RUN_ID,
    )


def begin_variant(store, **overrides):
    kwargs = {
        "project_id": PROJECT_ID,
        "hypothesis": "Cross-sectional momentum persists net of costs.",
        "methodology": make_methodology(),
        "parameters": {"lookback_days": 126, "top_n": 20, "seed": 7},
        "datasets": [{"dataset_version_id": DATASET_ID, "role": "features"}],
        "code": {"git_commit": "abc123", "artifact_sha256": "00" * 32},
        "environment": {"lock_hash": "deadbeef"},
        "created_by_run_id": RUN_ID,
    }
    kwargs.update(overrides)
    return write_api.experiment_begin(store, **kwargs)


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------


class TestKnowledgeConfig:
    def test_defaults(self):
        cfg = knowledge_config.KnowledgeConfig()
        assert cfg.is_enabled() is True
        assert cfg.get_database_dsn() is None
        assert cfg.get_object_store_endpoint() is None
        assert cfg.get_object_store_bucket() == "quantflow-knowledge"
        assert cfg.get_object_store_region() == "us-east-1"
        assert cfg.default_page_size == 20
        assert cfg.max_page_size == 100

    def test_explicit_field_beats_env(self, monkeypatch):
        monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_DSN", "postgresql://env/db")
        cfg = knowledge_config.KnowledgeConfig(database_dsn="postgresql://field/db")
        assert cfg.get_database_dsn() == "postgresql://field/db"

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_DSN", "postgresql://env/db")
        monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_S3_ENDPOINT", "http://minio:9000")
        monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_S3_BUCKET", "env-bucket")
        monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_S3_REGION", "eu-west-1")
        cfg = knowledge_config.KnowledgeConfig()
        assert cfg.get_database_dsn() == "postgresql://env/db"
        assert cfg.get_object_store_endpoint() == "http://minio:9000"
        assert cfg.get_object_store_bucket() == "env-bucket"
        assert cfg.get_object_store_region() == "eu-west-1"

    def test_blank_env_counts_as_unset(self, monkeypatch):
        monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_DSN", "   ")
        assert knowledge_config.KnowledgeConfig().get_database_dsn() is None

    def test_embedding_model_defaults_to_none(self, monkeypatch):
        monkeypatch.delenv("DEER_FLOW_KNOWLEDGE_EMBEDDING_MODEL", raising=False)
        assert knowledge_config.KnowledgeConfig().get_embedding_model() is None

    def test_embedding_model_explicit_field_beats_env(self, monkeypatch):
        monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_EMBEDDING_MODEL", "env-model/v1")
        cfg = knowledge_config.KnowledgeConfig(embedding_model="field-model/v1")
        assert cfg.get_embedding_model() == "field-model/v1"

    def test_embedding_model_env_override(self, monkeypatch):
        monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
        assert knowledge_config.KnowledgeConfig().get_embedding_model() == "sentence-transformers/all-mpnet-base-v2"

    def test_embedding_model_blank_counts_as_unset(self, monkeypatch):
        monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_EMBEDDING_MODEL", "   ")
        assert knowledge_config.KnowledgeConfig().get_embedding_model() is None
        assert knowledge_config.KnowledgeConfig(embedding_model="  ").get_embedding_model() is None

    def test_embedding_model_dict_roundtrip(self, monkeypatch):
        monkeypatch.setattr(knowledge_config, "_knowledge_config", knowledge_config.KnowledgeConfig())
        knowledge_config.load_knowledge_config_from_dict({"embedding_model": "mpnet/v1"})
        assert knowledge_config.get_knowledge_config().get_embedding_model() == "mpnet/v1"

    @pytest.mark.parametrize("token,expected", [("1", True), ("true", True), ("YES", True), ("On", True), ("0", False), ("false", False), ("no", False), ("OFF", False)])
    def test_enabled_flag_tokens(self, monkeypatch, token, expected):
        monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_ENABLED", token)
        assert knowledge_config.KnowledgeConfig().is_enabled() is expected
        assert knowledge_config.KnowledgeConfig(enabled=not expected).is_enabled() is expected

    def test_enabled_flag_invalid(self, monkeypatch):
        monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_ENABLED", "maybe")
        with pytest.raises(ValueError, match="DEER_FLOW_KNOWLEDGE_ENABLED"):
            knowledge_config.KnowledgeConfig().is_enabled()

    def test_page_size_bounds_rejected(self):
        with pytest.raises(Exception):
            knowledge_config.KnowledgeConfig(default_page_size=0)
        with pytest.raises(Exception):
            knowledge_config.KnowledgeConfig(max_page_size=0)

    def test_load_from_dict_and_singleton_roundtrip(self, monkeypatch):
        monkeypatch.setattr(knowledge_config, "_knowledge_config", knowledge_config.KnowledgeConfig())
        knowledge_config.load_knowledge_config_from_dict({"database_dsn": "postgresql://x", "unknown_future_key": 1})
        assert knowledge_config.get_knowledge_config().get_database_dsn() == "postgresql://x"
        fresh = knowledge_config.KnowledgeConfig()
        knowledge_config.set_knowledge_config(fresh)
        assert knowledge_config.get_knowledge_config() is fresh


# ---------------------------------------------------------------------------
# hashing.py
# ---------------------------------------------------------------------------


class TestHashing:
    def test_family_hash_deterministic_under_key_reordering(self):
        design_a = {"hypothesis": "H", "methodology": {"b": 1, "a": [1, 2, {"z": 1, "y": 2}]}, "extra": {"k": "v"}}
        design_b = {"extra": {"k": "v"}, "methodology": {"a": [1, 2, {"y": 2, "z": 1}], "b": 1}, "hypothesis": "H"}
        assert hashing.experiment_family_hash(design_a) == hashing.experiment_family_hash(design_b)

    def test_family_stable_when_only_seed_params_change(self):
        design = make_design()
        base_kwargs = {"design": design, "code": {"rev": "a"}, "data": {"ds": ["x"]}, "environment": {"lock": "l"}}
        exec_a = hashing.execution_hash(parameters={"seed": 1}, **base_kwargs)
        exec_b = hashing.execution_hash(parameters={"seed": 2}, **base_kwargs)
        assert exec_a != exec_b
        assert hashing.experiment_family_hash(design) == hashing.experiment_family_hash(dict(design))

    def test_execution_changes_on_every_dimension(self):
        design, code, data, params, env = make_design(), {"rev": "a"}, {"ds": ["x"]}, {"seed": 1}, {"lock": "l"}
        base = hashing.execution_hash(design=design, code=code, data=data, parameters=params, environment=env)
        assert base != hashing.execution_hash(design={"other": 1}, code=code, data=data, parameters=params, environment=env)
        assert base != hashing.execution_hash(design=design, code={"rev": "b"}, data=data, parameters=params, environment=env)
        assert base != hashing.execution_hash(design=design, code=code, data={"ds": ["y"]}, parameters=params, environment=env)
        assert base != hashing.execution_hash(design=design, code=code, data=data, parameters={"seed": 2}, environment=env)
        assert base != hashing.execution_hash(design=design, code=code, data=data, parameters=params, environment={"lock": "m"})

    def test_distinct_designs_distinct_hashes(self):
        hashes = {hashing.experiment_family_hash({"hypothesis": f"H{i}", "methodology": {"k": i}}) for i in range(25)}
        assert len(hashes) == 25

    def test_list_order_is_significant(self):
        assert hashing.canonical_json([1, 2]) != hashing.canonical_json([2, 1])

    def test_tuple_equals_list(self):
        assert hashing.canonical_json((1, [2, 3])) == hashing.canonical_json([1, [2, 3]])

    def test_none_is_significant(self):
        assert hashing.canonical_json({"a": None}) != hashing.canonical_json({})

    def test_unicode_nfc_equivalence(self):
        assert hashing.canonical_json("caf\u00e9") == hashing.canonical_json("cafe\u0301")

    def test_case_sensitivity(self):
        assert hashing.canonical_json("AAPL") != hashing.canonical_json("aapl")

    def test_int_float_distinct(self):
        assert hashing.canonical_json(1) != hashing.canonical_json(1.0)
        assert hashing.canonical_json(True) != hashing.canonical_json(1)

    def test_negative_zero_normalizes(self):
        assert hashing.canonical_json(-0.0) == hashing.canonical_json(0.0)

    def test_canonical_format(self):
        assert hashing.canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
        assert hashing.canonical_json("é") == '"\\u00e9"'

    def test_domain_separation(self):
        payload = {"a": 1}
        assert hashing.experiment_family_hash(payload) != hashing.execution_hash(design=payload, code={}, data={}, parameters={}, environment={})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_floats_rejected(self, bad):
        with pytest.raises(ValueError):
            hashing.canonical_json({"x": bad})

    @pytest.mark.parametrize("bad", [b"bytes", {1, 2}, object(), {1: "non-str-key"}])
    def test_non_json_types_rejected(self, bad):
        with pytest.raises(TypeError):
            hashing.canonical_json(bad)

    def test_non_mapping_design_rejected(self):
        with pytest.raises(TypeError):
            hashing.experiment_family_hash(["not", "a", "mapping"])

    def test_normalize_text(self):
        assert hashing.normalize_text("  multi   space\nhypothesis\t") == "multi space hypothesis"
        with pytest.raises(TypeError):
            hashing.normalize_text(123)

    def test_idempotency_key(self):
        key = hashing.make_idempotency_key(RUN_ID, 3, {"a": 1})
        assert key == hashing.make_idempotency_key(RUN_ID, 3, {"a": 1})
        assert len(key) == 64
        assert key != hashing.make_idempotency_key(RUN_ID, 3, {"a": 2})
        assert key != hashing.make_idempotency_key(RUN_ID, 4, {"a": 1})
        assert key != hashing.make_idempotency_key(PROJECT_ID, 3, {"a": 1})
        with pytest.raises(TypeError):
            hashing.make_idempotency_key("", 0, {})
        with pytest.raises(TypeError):
            hashing.make_idempotency_key(RUN_ID, -1, {})
        with pytest.raises(TypeError):
            hashing.make_idempotency_key(RUN_ID, True, {})


# ---------------------------------------------------------------------------
# write_api.py
# ---------------------------------------------------------------------------


class TestExperimentBegin:
    def test_happy_path(self, store, begun):
        assert isinstance(uuid.UUID(begun.id), uuid.UUID)
        assert begun.project_id == PROJECT_ID
        assert begun.status == "planned"
        assert len(begun.family_hash) == 64
        assert len(begun.execution_hash) == 64
        assert begun.family_hash == hashing.experiment_family_hash(make_design())
        assert begun.started_at
        assert begun.completed_at is None
        assert store.get_experiment(begun.id) == begun
        as_dict = begun.to_dict()
        assert as_dict["id"] == begun.id
        assert as_dict["datasets"] == [{"dataset_version_id": DATASET_ID, "role": "features"}]

    def test_equivalent_design_same_family(self, store, begun):
        other = begin_variant(store, parameters={"lookback_days": 63, "top_n": 50, "seed": 99}, code={"git_commit": "zzz"})
        assert other.family_hash == begun.family_hash
        assert other.execution_hash != begun.execution_hash

    def test_idempotent_replay(self, store):
        first = begin_variant(store, idempotency_key="begin-key-1")
        replay = begin_variant(store, idempotency_key="begin-key-1")
        assert replay == first
        assert len(store.experiments) == 1

    def test_idempotency_conflict(self, store):
        begin_variant(store, idempotency_key="begin-key-2")
        with pytest.raises(IdempotencyConflictError):
            begin_variant(store, idempotency_key="begin-key-2", parameters={"seed": 12345})

    def test_duplicate_execution_rejected(self, store, begun):
        with pytest.raises(DuplicateExecutionError) as exc_info:
            begin_variant(store)
        assert exc_info.value.existing_experiment_id == begun.id
        assert exc_info.value.execution_hash == begun.execution_hash

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"project_id": "not-a-uuid"},
            {"created_by_run_id": "bogus"},
            {"hypothesis": "   "},
            {"hypothesis": 42},
            {"methodology": ["not-a-mapping"]},
            {"methodology": {"x": float("nan")}},
            {"parameters": {"s": {1, 2}}},
            {"status": "completed"},
            {"status": "bogus"},
            {"datasets": [{"role": "features"}]},
            {"datasets": [{"dataset_version_id": DATASET_ID, "role": "bogus"}]},
            {"idempotency_key": "has whitespace"},
        ],
    )
    def test_validation_errors(self, store, kwargs):
        with pytest.raises(KnowledgeValidationError):
            begin_variant(store, **kwargs)

    def test_running_status_and_links(self, store):
        parent = begin_variant(store)
        child = begin_variant(
            store,
            parameters={"seed": 8},
            status="running",
            parent_experiment_id=parent.id,
            replicated_experiment_id=parent.id,
        )
        assert child.status == "running"
        assert child.parent_experiment_id == parent.id
        assert child.replicated_experiment_id == parent.id

    def test_client_hash_echo_match_accepted(self, store, begun):
        design = make_design()
        family = hashing.experiment_family_hash(design)
        execution = hashing.execution_hash(
            design=design,
            code=begun.code,
            data={"datasets": begun.datasets},
            parameters=begun.parameters,
            environment=begun.environment,
        )
        assert family == begun.family_hash
        assert execution == begun.execution_hash
        other = begin_variant(
            store,
            parameters={"lookback_days": 126, "top_n": 20, "seed": 8},
            client_family_hash=family,
            client_execution_hash=hashing.execution_hash(
                design=design,
                code=begun.code,
                data={"datasets": begun.datasets},
                parameters={"lookback_days": 126, "top_n": 20, "seed": 8},
                environment=begun.environment,
            ),
        )
        assert other.family_hash == family

    def test_client_hash_echo_mismatch_rejected(self, store):
        with pytest.raises(InteropMismatchError) as exc_info:
            begin_variant(store, client_family_hash="0" * 64)
        assert exc_info.value.which == "family_hash"
        with pytest.raises(InteropMismatchError) as exc_info:
            begin_variant(store, client_execution_hash="0" * 64)
        assert exc_info.value.which == "execution_hash"


class TestExperimentCommit:
    def test_commit_success(self, store, begun):
        committed = write_api.experiment_commit(store, experiment_id=begun.id, metrics={"sharpe": 1.2}, outcome="success", result_artifacts=[DATASET_ID])
        assert committed.status == "completed"
        assert committed.outcome == "success"
        assert committed.metrics == {"sharpe": 1.2}
        assert committed.completed_at
        assert store.get_experiment(begun.id) == committed

    def test_commit_failure(self, store, begun):
        committed = write_api.experiment_commit(store, experiment_id=begun.id, metrics={"sharpe": -0.5}, outcome="failure", failure_class="statistical")
        assert committed.status == "failed"
        assert committed.failure_class == "statistical"

    def test_commit_unknown_experiment(self, store):
        with pytest.raises(ExperimentNotFoundError):
            write_api.experiment_commit(store, experiment_id=PROJECT_ID, metrics={}, outcome="success")

    def test_commit_terminal_rejected(self, store, begun):
        write_api.experiment_commit(store, experiment_id=begun.id, metrics={}, outcome="success")
        with pytest.raises(KnowledgeValidationError, match="terminal"):
            write_api.experiment_commit(store, experiment_id=begun.id, metrics={}, outcome="success")

    def test_commit_idempotent_replay(self, store, begun):
        first = write_api.experiment_commit(store, experiment_id=begun.id, metrics={"a": 1}, outcome="success", idempotency_key="commit-key-1")
        replay = write_api.experiment_commit(store, experiment_id=begun.id, metrics={"a": 1}, outcome="success", idempotency_key="commit-key-1")
        assert replay == first

    def test_commit_idempotency_conflict(self, store, begun):
        write_api.experiment_commit(store, experiment_id=begun.id, metrics={"a": 1}, outcome="success", idempotency_key="commit-key-2")
        with pytest.raises(IdempotencyConflictError):
            write_api.experiment_commit(store, experiment_id=begun.id, metrics={"a": 2}, outcome="success", idempotency_key="commit-key-2")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"outcome": "bogus"},
            {"outcome": "failure"},
            {"outcome": "success", "failure_class": "data"},
            {"outcome": "inconclusive", "failure_class": "code"},
            {"outcome": "failure", "failure_class": "bogus"},
            {"metrics": ["not-a-mapping"]},
            {"metrics": {"x": float("inf")}},
        ],
    )
    def test_commit_validation_errors(self, store, begun, kwargs):
        params: dict = {"metrics": {}, "outcome": "success"}
        params.update(kwargs)
        with pytest.raises(KnowledgeValidationError):
            write_api.experiment_commit(store, experiment_id=begun.id, **params)


class TestFailureRecord:
    def test_happy_path(self, store, begun):
        failure = write_api.failure_record(store, experiment_id=begun.id, failure_class="execution", notes="OOM at rebalance", idempotency_key="fail-1")
        assert failure.experiment_id == begun.id
        assert failure.failure_class == "execution"
        assert failure.recorded_at
        assert store.get_experiment(begun.id).status == "planned"
        assert failure.to_dict()["id"] == failure.id

    def test_unknown_experiment(self, store):
        with pytest.raises(ExperimentNotFoundError):
            write_api.failure_record(store, experiment_id=PROJECT_ID, failure_class="data")

    def test_bad_class(self, store, begun):
        with pytest.raises(KnowledgeValidationError):
            write_api.failure_record(store, experiment_id=begun.id, failure_class="vibes")

    def test_idempotent_replay_and_conflict(self, store, begun):
        first = write_api.failure_record(store, experiment_id=begun.id, failure_class="code", notes="n", idempotency_key="fail-2")
        assert write_api.failure_record(store, experiment_id=begun.id, failure_class="code", notes="n", idempotency_key="fail-2") == first
        with pytest.raises(IdempotencyConflictError):
            write_api.failure_record(store, experiment_id=begun.id, failure_class="data", notes="n", idempotency_key="fail-2")


class TestAssumptionRecord:
    def test_happy_path_defaults(self, store, begun):
        assumption = write_api.assumption_record(store, experiment_id=begun.id, statement="No survivorship bias.", category="data")
        assert assumption.experiment_id == begun.id
        assert assumption.sensitivity == "unknown"
        assert assumption.tested is False
        assert assumption.status == "active"
        assert assumption.to_dict()["category"] == "data"

    def test_standalone_assumption(self, store):
        assumption = write_api.assumption_record(
            store,
            statement="Costs are 10bps.",
            category="execution",
            sensitivity="high",
            tested=True,
            status="challenged",
            evidence_artifact_id=DATASET_ID,
            idempotency_key="asm-1",
        )
        assert assumption.experiment_id is None
        assert assumption.tested is True

    def test_unknown_experiment(self, store):
        with pytest.raises(ExperimentNotFoundError):
            write_api.assumption_record(store, experiment_id=PROJECT_ID, statement="s", category="data")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"statement": "  ", "category": "data"},
            {"statement": "s", "category": "bogus"},
            {"statement": "s", "category": "data", "sensitivity": "extreme"},
            {"statement": "s", "category": "data", "tested": "yes"},
            {"statement": "s", "category": "data", "status": "bogus"},
            {"statement": "s", "category": "data", "evidence_artifact_id": "nope"},
        ],
    )
    def test_validation_errors(self, store, kwargs):
        with pytest.raises(KnowledgeValidationError):
            write_api.assumption_record(store, **kwargs)

    def test_idempotent_replay_and_conflict(self, store):
        first = write_api.assumption_record(store, statement="s", category="market", idempotency_key="asm-2")
        assert write_api.assumption_record(store, statement="s", category="market", idempotency_key="asm-2") == first
        with pytest.raises(IdempotencyConflictError):
            write_api.assumption_record(store, statement="other", category="market", idempotency_key="asm-2")


# ---------------------------------------------------------------------------
# search.py
# ---------------------------------------------------------------------------


class TestExperimentSearch:
    def _seed_family(self, store):
        first = begin_variant(store, parameters={"seed": 1})
        second = begin_variant(store, parameters={"seed": 2})
        assert first.family_hash == second.family_hash
        return first, second

    def test_search_by_family_hash_finds_equivalent_reruns(self, store):
        first, second = self._seed_family(store)
        result = knowledge_search.experiment_search(store, family_hash=first.family_hash)
        assert isinstance(result, ExperimentSearchResult)
        assert {rec.id for rec in result.experiments} == {first.id, second.id}
        assert result.family_hash_used == first.family_hash
        assert store.used_find_by_family_hash == 1
        assert store.used_search_experiments == 0

    def test_search_by_spec(self, store):
        first, _ = self._seed_family(store)
        result = knowledge_search.experiment_search(store, spec=make_design())
        assert {rec.id for rec in result.experiments} == {rec.id for rec in store.experiments.values()}
        assert result.family_hash_used == first.family_hash

    def test_search_spec_and_hash_must_agree(self, store):
        self._seed_family(store)
        with pytest.raises(KnowledgeValidationError, match="disagree"):
            knowledge_search.experiment_search(store, family_hash="00" * 32, spec=make_design())

    def test_search_filters_and_pagination(self, store):
        first, second = self._seed_family(store)
        other = begin_variant(store, project_id=PROJECT_ID_2, hypothesis="Unrelated value factor study.", methodology=make_methodology(universe="russell2000"))
        write_api.experiment_commit(store, experiment_id=first.id, metrics={}, outcome="success")
        write_api.experiment_commit(store, experiment_id=second.id, metrics={}, outcome="failure", failure_class="data")

        by_status = knowledge_search.experiment_search(store, status="planned")
        assert [rec.id for rec in by_status.experiments] == [other.id]
        assert store.used_search_experiments >= 1

        by_outcome = knowledge_search.experiment_search(store, outcome="failure", failure_class="data")
        assert [rec.id for rec in by_outcome.experiments] == [second.id]

        by_project = knowledge_search.experiment_search(store, project_id=PROJECT_ID_2)
        assert [rec.id for rec in by_project.experiments] == [other.id]

        by_text = knowledge_search.experiment_search(store, hypothesis_contains="VALUE factor")
        assert [rec.id for rec in by_text.experiments] == [other.id]

        page1 = knowledge_search.experiment_search(store, family_hash=first.family_hash, limit=1, offset=0)
        page2 = knowledge_search.experiment_search(store, family_hash=first.family_hash, limit=1, offset=1)
        assert len(page1.experiments) == 1 and len(page2.experiments) == 1
        assert page1.experiments[0].id != page2.experiments[0].id
        assert page1.limit == 1 and page1.offset == 0

    def test_search_no_match_returns_empty_page(self, store):
        result = knowledge_search.experiment_search(store, family_hash="ab" * 32)
        assert result.experiments == []

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"family_hash": "not-hex"},
            {"family_hash": "AB" * 32},
            {"family_hash": "ab"},
            {"spec": ["not-a-mapping"]},
            {"spec": {"x": float("nan")}},
            {"project_id": "bogus"},
            {"status": "bogus"},
            {"outcome": "bogus"},
            {"failure_class": "bogus"},
            {"hypothesis_contains": "   "},
            {"limit": 0},
            {"limit": 101},
            {"offset": -1},
        ],
    )
    def test_search_validation_errors(self, store, kwargs):
        with pytest.raises(KnowledgeValidationError):
            knowledge_search.experiment_search(store, **kwargs)

    def test_get_by_execution_hash(self, store, begun):
        assert knowledge_search.experiment_get_by_execution_hash(store, begun.execution_hash) == begun
        assert knowledge_search.experiment_get_by_execution_hash(store, "ff" * 32) is None
        with pytest.raises(KnowledgeValidationError):
            knowledge_search.experiment_get_by_execution_hash(store, "bogus")

    def test_filter_is_family_only(self):
        assert ExperimentFilter(family_hash="ab" * 32).is_family_only() is True
        assert ExperimentFilter(family_hash="ab" * 32, status="planned").is_family_only() is False
        assert ExperimentFilter().is_family_only() is False

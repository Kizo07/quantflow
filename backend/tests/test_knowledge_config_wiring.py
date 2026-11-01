"""Tests for Knowledge Plane AppConfig registration (integration).

Covers the ``knowledge:`` section wiring in
:mod:`deerflow.config.app_config` (field defaults, YAML round-trip, singleton
refresh) plus the ``deerflow.config`` re-exports and the
``config.example.yaml`` section shape.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
import sqlalchemy as sa
import yaml
from sqlalchemy.orm import sessionmaker

import deerflow.knowledge.schema as kb_schema
from deerflow.config import KnowledgeConfig, get_knowledge_config
from deerflow.config.app_config import AppConfig, reset_app_config
from deerflow.knowledge import pg_retrieval as pg
from deerflow.knowledge.agent_tools.lookup import ArtifactStoreReader, reset_knowledge_backends
from deerflow.knowledge.artifacts import ArtifactStore
from deerflow.knowledge.artifacts.backends import LocalFilesystemBackend
from deerflow.knowledge.config import (
    KnowledgeConfig as DirectKnowledgeConfig,
)
from deerflow.knowledge.config import (
    load_knowledge_config_from_dict,
    set_knowledge_config,
)
from deerflow.knowledge.retrieval.fusion import FusionWeights
from deerflow.knowledge.schema.findings import FindingRow
from deerflow.knowledge.write_api import KnowledgeValidationError
from deerflow.persistence.base import Base


def _write_config_with_sections(path: Path, sections: dict | None = None) -> None:
    config = {
        "sandbox": {"use": "deerflow.sandbox.local:LocalSandboxProvider"},
        "models": [
            {
                "name": "first-model",
                "use": "langchain_openai:ChatOpenAI",
                "model": "gpt-test",
            }
        ],
    }
    if sections:
        config.update(sections)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def _write_extensions_config(path: Path) -> None:
    path.write_text(json.dumps({"mcpServers": {}, "skills": {}}), encoding="utf-8")


@pytest.fixture(autouse=True)
def _clean_singletons():
    yield
    reset_app_config()
    load_knowledge_config_from_dict({})


def _load(tmp_path, monkeypatch, sections: dict | None) -> AppConfig:
    config_path = tmp_path / "config.yaml"
    extensions_path = tmp_path / "extensions_config.json"
    _write_config_with_sections(config_path, sections)
    _write_extensions_config(extensions_path)
    monkeypatch.setenv("DEER_FLOW_EXTENSIONS_CONFIG_PATH", str(extensions_path))
    reset_app_config()
    return AppConfig.from_file(str(config_path))


def test_app_config_knowledge_defaults() -> None:
    config = AppConfig.model_validate({"sandbox": {"use": "deerflow.sandbox.local:LocalSandboxProvider"}})
    assert isinstance(config.knowledge, KnowledgeConfig)
    assert config.knowledge.enabled is True
    assert config.knowledge.get_database_dsn() is None
    assert config.knowledge.get_embedding_model() is None
    assert config.knowledge.get_object_store_bucket() == "quantflow-knowledge"
    assert config.knowledge.default_page_size == 20
    assert config.knowledge.max_page_size == 100
    assert config.knowledge.artifact_store is None
    assert config.knowledge.fusion_weights == {}


def test_from_file_loads_knowledge_section_and_refreshes_singleton(tmp_path, monkeypatch) -> None:
    config = _load(
        tmp_path,
        monkeypatch,
        {
            "knowledge": {
                "enabled": False,
                "database_dsn": "postgresql+psycopg://kb/db",
                "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                "object_store_bucket": "kb-evidence",
                "object_store_region": "eu-west-1",
                "default_page_size": 5,
                "max_page_size": 50,
                "max_payload_bytes": 2048,
            }
        },
    )
    assert config.knowledge.enabled is False
    assert config.knowledge.get_database_dsn() == "postgresql+psycopg://kb/db"
    assert config.knowledge.get_embedding_model() == "sentence-transformers/all-mpnet-base-v2"
    assert config.knowledge.get_object_store_bucket() == "kb-evidence"
    assert config.knowledge.get_object_store_region() == "eu-west-1"
    assert config.knowledge.default_page_size == 5
    assert config.knowledge.max_page_size == 50
    singleton = get_knowledge_config()
    assert singleton.get_database_dsn() == "postgresql+psycopg://kb/db"
    assert singleton.get_embedding_model() == "sentence-transformers/all-mpnet-base-v2"
    assert singleton.get_object_store_bucket() == "kb-evidence"
    assert singleton.max_page_size == 50


def test_missing_section_resets_singleton_to_defaults(tmp_path, monkeypatch) -> None:
    set_knowledge_config(DirectKnowledgeConfig(object_store_bucket="stale-bucket"))
    config = _load(tmp_path, monkeypatch, None)
    assert config.knowledge.get_object_store_bucket() == "quantflow-knowledge"
    assert get_knowledge_config().get_object_store_bucket() == "quantflow-knowledge"


def test_knowledge_enabled_env_override(tmp_path, monkeypatch) -> None:
    config = _load(tmp_path, monkeypatch, {"knowledge": {"enabled": True}})
    assert config.knowledge.is_enabled() is True
    monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_ENABLED", "0")
    assert config.knowledge.is_enabled() is False


def test_config_package_reexports_knowledge_config() -> None:
    assert KnowledgeConfig is DirectKnowledgeConfig
    assert callable(get_knowledge_config)


def test_example_yaml_knowledge_section_validates(monkeypatch) -> None:
    monkeypatch.delenv("DEER_FLOW_KNOWLEDGE_ARTIFACT_STORE", raising=False)
    monkeypatch.delenv("DEER_FLOW_KNOWLEDGE_FUSION_WEIGHTS", raising=False)
    example_path = Path(__file__).resolve().parents[2] / "config.example.yaml"
    config_data = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    assert "knowledge" in config_data
    parsed = DirectKnowledgeConfig(**config_data["knowledge"])
    assert parsed.enabled is True
    assert parsed.database_dsn is None
    assert parsed.embedding_model is None
    assert parsed.default_page_size == 20
    assert parsed.max_page_size == 100
    assert parsed.artifact_store is None
    assert parsed.fusion_weights == {}
    assert parsed.get_artifact_store() is None
    assert parsed.get_fusion_weights() == FusionWeights()


def test_from_file_loads_artifact_store_and_fusion_weights(tmp_path, monkeypatch) -> None:
    config = _load(
        tmp_path,
        monkeypatch,
        {
            "knowledge": {
                "artifact_store": "file:///var/lib/quantflow/artifacts",
                "fusion_weights": {"rrf_k": 5, "failure_boost": 2.0, "near_dup_penalty": 0.25},
            }
        },
    )
    assert config.knowledge.get_artifact_store() == "file:///var/lib/quantflow/artifacts"
    weights = config.knowledge.get_fusion_weights()
    assert weights.rrf_k == 5
    assert weights.failure_boost == 2.0
    assert weights.near_dup_penalty == 0.25
    assert weights.exact_scope_boost == FusionWeights().exact_scope_boost
    singleton = get_knowledge_config()
    assert singleton.get_artifact_store() == "file:///var/lib/quantflow/artifacts"
    assert singleton.get_fusion_weights() == weights


def test_artifact_store_env_override(tmp_path, monkeypatch) -> None:
    config = _load(tmp_path, monkeypatch, {"knowledge": {"artifact_store": "file:///from-file"}})
    assert config.knowledge.get_artifact_store() == "file:///from-file"
    monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_ARTIFACT_STORE", "s3://from-env")
    assert config.knowledge.get_artifact_store() == "file:///from-file"
    fileless = _load(tmp_path, monkeypatch, {"knowledge": {"enabled": True}})
    assert fileless.knowledge.get_artifact_store() == "s3://from-env"


def test_fusion_weights_env_json_override(tmp_path, monkeypatch) -> None:
    config = _load(tmp_path, monkeypatch, {"knowledge": {"fusion_weights": {"rrf_k": 3}}})
    monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_FUSION_WEIGHTS", '{"rrf_k": 7, "failure_boost": 2.5}')
    weights = config.knowledge.get_fusion_weights()
    assert weights.rrf_k == 3
    assert weights.failure_boost == 2.5


def test_fusion_weights_invalid_config_fails_closed(tmp_path, monkeypatch) -> None:
    unknown = _load(tmp_path, monkeypatch, {"knowledge": {"fusion_weights": {"bogus_knob": 1.0}}})
    with pytest.raises(ValueError, match="Unknown fusion_weights keys"):
        unknown.knowledge.get_fusion_weights()
    bad_value = _load(tmp_path, monkeypatch, {"knowledge": {"fusion_weights": {"rrf_k": 0}}})
    with pytest.raises(KnowledgeValidationError):
        bad_value.knowledge.get_fusion_weights()


_KB_TABLES: tuple[str, ...] = (
    "research_project",
    "agent_run",
    "finding",
    "experiment",
    "experiment_dataset",
    "experiment_artifact",
    "assumption",
)


def _seed_single_finding_db(tmp_path: Path) -> str:
    """Create a SQLite KB holding one momentum finding; return its DSN."""
    db_path = tmp_path / "kb_weights.db"
    dsn = f"sqlite:///{db_path.as_posix()}"
    engine = sa.create_engine(dsn)
    for table in _KB_TABLES:
        Base.metadata.tables[table].create(engine, checkfirst=True)
    factory = sessionmaker(engine, expire_on_commit=False)
    project_id = uuid.uuid4()
    run_id = uuid.uuid4()
    with factory() as session:
        session.add(kb_schema.ResearchProjectRow(id=project_id, name="momentum", visibility_scope={}))
        session.add(kb_schema.AgentRunRow(id=run_id, project_id=project_id, agent_type="research", task="weights threading", status="running"))
        session.add(
            FindingRow(
                id=uuid.uuid4(),
                project_id=project_id,
                canonical_key="empirical:momentum-persists-net-costs",
                finding_type="empirical",
                statement="Cross-sectional momentum persists net of transaction costs on liquid United States stocks.",
                scope={"asset_class": "equity", "market": "US", "universe": "sp500-pit", "horizon": "6m", "frequency": "daily"},
                status="candidate",
                confidence={"overall_tier": "moderate"},
                effective_from=datetime(2015, 1, 1, tzinfo=UTC),
                effective_to=None,
                recorded_at=datetime(2026, 3, 10, tzinfo=UTC),
                created_by_run_id=run_id,
            )
        )
        session.commit()
    engine.dispose()
    return dsn


def _single_hit_score(dsn: str, *, weights: FusionWeights | None = None) -> float:
    """Return the fused score of the one finding hit for a momentum query."""
    backends = pg.open_pg_backends(dsn, fusion_weights=weights)
    assert backends.retrieval is not None
    page = backends.retrieval.search("momentum", kinds=("finding",), filters={}, limit=10, offset=0)
    assert page.total == 1
    return page.documents[0].score


def test_configured_rrf_k_reaches_fuse(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DEER_FLOW_KNOWLEDGE_FUSION_WEIGHTS", raising=False)
    dsn = _seed_single_finding_db(tmp_path)
    default_score = _single_hit_score(dsn)
    assert default_score > 0
    set_knowledge_config(DirectKnowledgeConfig(fusion_weights={"rrf_k": 1}))
    configured_score = _single_hit_score(dsn)
    # One candidate at rank 1 in every channel: the fused score scales as
    # (60 + 1) / (1 + 1) while every other modifier cancels out.
    assert configured_score / default_score == pytest.approx(61 / 2)


def test_default_weights_identical_to_fusion_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DEER_FLOW_KNOWLEDGE_FUSION_WEIGHTS", raising=False)
    dsn = _seed_single_finding_db(tmp_path)
    assert _single_hit_score(dsn) == pytest.approx(_single_hit_score(dsn, weights=FusionWeights()))
    service = pg.open_pg_backends(dsn).retrieval
    assert service is not None
    assert service.fusion_weights is None


def test_explicit_service_weights_win_over_config(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DEER_FLOW_KNOWLEDGE_FUSION_WEIGHTS", raising=False)
    dsn = _seed_single_finding_db(tmp_path)
    default_score = _single_hit_score(dsn)
    set_knowledge_config(DirectKnowledgeConfig(fusion_weights={"rrf_k": 5}))
    explicit_score = _single_hit_score(dsn, weights=FusionWeights(rrf_k=1))
    assert explicit_score / default_score == pytest.approx(61 / 2)


def test_open_pg_backends_uri_unset_keeps_current_behavior(tmp_path) -> None:
    dsn = _seed_single_finding_db(tmp_path)
    backends = pg.open_pg_backends(dsn)
    assert backends.retrieval is not None
    assert backends.experiments is not None
    assert backends.artifacts is None


def test_bind_from_config_file_uri_binds_local_reader(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "kb_artifacts.db"
    store_dir = tmp_path / "artifacts"
    _load(
        tmp_path,
        monkeypatch,
        {"knowledge": {"database_dsn": f"sqlite:///{db_path.as_posix()}", "artifact_store": f"file://{store_dir.as_posix()}"}},
    )
    try:
        bound = pg.bind_knowledge_backends_from_config()
        assert bound is not None
        assert isinstance(bound.artifacts, ArtifactStoreReader)
        assert isinstance(bound.artifacts.store.backend, LocalFilesystemBackend)
        record = bound.artifacts.store.put_bytes(b"evidence bytes", kind="result", media_type="text/plain")
        assert bound.artifacts.get_artifact_record(record.uri).sha256 == record.sha256
        window = bound.artifacts.read_artifact_slice(record.uri, offset=0, length=8)
        assert window.data == b"evidence"
    finally:
        reset_knowledge_backends()


def test_bind_from_config_unset_uri_leaves_artifacts_unbound(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "kb_noartifacts.db"
    _load(tmp_path, monkeypatch, {"knowledge": {"database_dsn": f"sqlite:///{db_path.as_posix()}"}})
    try:
        bound = pg.bind_knowledge_backends_from_config()
        assert bound is not None
        assert bound.artifacts is None
    finally:
        reset_knowledge_backends()


def test_bind_from_config_explicit_store_wins_over_uri(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "kb_explicit.db"
    explicit = ArtifactStore(LocalFilesystemBackend((tmp_path / "explicit_artifacts").as_posix()))
    _load(
        tmp_path,
        monkeypatch,
        {"knowledge": {"database_dsn": f"sqlite:///{db_path.as_posix()}", "artifact_store": "file:///uri-ignored"}},
    )
    try:
        bound = pg.bind_knowledge_backends_from_config(artifact_store=explicit)
        assert bound is not None
        assert isinstance(bound.artifacts, ArtifactStoreReader)
        assert bound.artifacts.store is explicit
    finally:
        reset_knowledge_backends()


def test_bind_from_config_invalid_uri_scheme_fails_closed(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "kb_baduri.db"
    _load(
        tmp_path,
        monkeypatch,
        {"knowledge": {"database_dsn": f"sqlite:///{db_path.as_posix()}", "artifact_store": "gopher://kb-evidence"}},
    )
    try:
        with pytest.raises(ValueError, match="unsupported backend spec"):
            pg.bind_knowledge_backends_from_config()
    finally:
        reset_knowledge_backends()

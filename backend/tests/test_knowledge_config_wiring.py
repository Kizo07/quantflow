"""Tests for Knowledge Plane AppConfig registration (integration).

Covers the ``knowledge:`` section wiring in
:mod:`deerflow.config.app_config` (field defaults, YAML round-trip, singleton
refresh) plus the ``deerflow.config`` re-exports and the
``config.example.yaml`` section shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from deerflow.config import KnowledgeConfig, get_knowledge_config
from deerflow.config.app_config import AppConfig, reset_app_config
from deerflow.knowledge.config import (
    KnowledgeConfig as DirectKnowledgeConfig,
)
from deerflow.knowledge.config import (
    load_knowledge_config_from_dict,
    set_knowledge_config,
)


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


def test_example_yaml_knowledge_section_validates() -> None:
    example_path = Path(__file__).resolve().parents[2] / "config.example.yaml"
    config_data = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    assert "knowledge" in config_data
    parsed = DirectKnowledgeConfig(**config_data["knowledge"])
    assert parsed.enabled is True
    assert parsed.database_dsn is None
    assert parsed.embedding_model is None
    assert parsed.default_page_size == 20
    assert parsed.max_page_size == 100

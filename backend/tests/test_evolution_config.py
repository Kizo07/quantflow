"""Unit tests for deerflow.evolution.config (resolvers + AppConfig wiring, offline)."""

import pytest

from deerflow.evolution.config import (
    EvolutionConfig,
    get_evolution_config,
    load_evolution_config_from_dict,
    set_evolution_config,
)


@pytest.fixture(autouse=True)
def _restore_singleton():
    previous = get_evolution_config()
    try:
        yield
    finally:
        set_evolution_config(previous)


def test_defaults_and_model_roles():
    config = EvolutionConfig()
    assert config.is_enabled() is False
    assert config.resolve_max_iterations() == 10
    assert config.resolve_max_wall_clock_seconds() == 0.0
    assert config.resolve_default_family() == "quant-alpha"
    assert config.resolve_default_universe() == "sp500"
    assert config.overrides_for_role("propose") == {}

    config = EvolutionConfig(model_roles={"propose": "research-pro"}, sampling_overrides={"propose": {"temperature": 0.7, "max_tokens": None}})
    assert config.model_for_role("propose") == "research-pro"
    assert config.overrides_for_role("propose") == {"temperature": 0.7}
    with pytest.raises(ValueError, match="design"):
        config.model_for_role("design")
    with pytest.raises(ValueError, match="Unknown evolution model role"):
        config.model_for_role("nope")


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("DEER_FLOW_EVOLUTION_ENABLED", "yes")
    monkeypatch.setenv("DEER_FLOW_EVOLUTION_MAX_ITERATIONS", "4")
    monkeypatch.setenv("DEER_FLOW_EVOLUTION_MAX_WALL_CLOCK_SECONDS", "1.5")
    monkeypatch.setenv("DEER_FLOW_EVOLUTION_DEFAULT_FAMILY", "factors")
    monkeypatch.setenv("DEER_FLOW_EVOLUTION_DEFAULT_UNIVERSE", "csi300")
    config = EvolutionConfig()
    assert config.is_enabled() is True
    assert config.resolve_max_iterations() == 4
    assert config.resolve_max_wall_clock_seconds() == 1.5
    assert config.resolve_default_family() == "factors"
    assert config.resolve_default_universe() == "csi300"

    monkeypatch.setenv("DEER_FLOW_EVOLUTION_MAX_ITERATIONS", "0")
    with pytest.raises(ValueError):
        config.resolve_max_iterations()


def test_singleton_accessors():
    assert isinstance(get_evolution_config(), EvolutionConfig)
    installed = load_evolution_config_from_dict({"max_iterations": 7})
    assert installed.resolve_max_iterations() == 7
    assert get_evolution_config() is installed
    set_evolution_config(None)
    assert isinstance(get_evolution_config(), EvolutionConfig)
    with pytest.raises(ValueError):
        load_evolution_config_from_dict("nope")


def test_app_config_registers_evolution_section():
    from deerflow.config.app_config import AppConfig

    base = {"sandbox": {"use": "deerflow.sandbox.local:LocalSandboxProvider"}}
    config = AppConfig.model_validate({**base, "evolution": {"max_iterations": 3, "model_roles": {"propose": "x"}}})
    assert isinstance(config.evolution, EvolutionConfig)
    assert config.evolution.resolve_max_iterations() == 3
    assert config.evolution.model_for_role("propose") == "x"
    assert AppConfig.model_validate(base).evolution.is_enabled() is False

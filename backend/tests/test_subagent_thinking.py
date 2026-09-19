"""Tests for config-driven subagent thinking.

Covers:
- resolve_subagent_thinking(): explicit on/off, inherit-None default, invalid
  effort levels, thinking on incapable models
- CustomSubagentConfig thinking-field defaults
- registry: custom_agents entries carry the fields into SubagentConfig
"""

from types import SimpleNamespace

import pytest

from deerflow.config.subagents_config import CustomSubagentConfig
from deerflow.subagents.config import SubagentConfig


def _subagent_config(**kwargs):
    base = {"name": "probe", "description": "probe subagent"}
    base.update(kwargs)
    return SubagentConfig(**base)


def _model_config(name="probe-model", *, supports_thinking=False, supports_reasoning_effort=False):
    return SimpleNamespace(
        name=name,
        supports_thinking=supports_thinking,
        supports_reasoning_effort=supports_reasoning_effort,
    )


def _resolve(config, model_config):
    from deerflow.subagents.config import resolve_subagent_thinking

    return resolve_subagent_thinking(config, model_config)


class TestResolveSubagentThinking:
    def test_none_fields_preserve_non_thinking_default(self):
        """Unset fields keep today's exact behavior: non-thinking, no override."""
        thinking, effort = _resolve(
            _subagent_config(),
            _model_config(supports_thinking=True, supports_reasoning_effort=True),
        )
        assert (thinking, effort) == (False, None)

    def test_explicit_thinking_on_with_effort(self):
        thinking, effort = _resolve(
            _subagent_config(thinking_enabled=True, reasoning_effort="high"),
            _model_config(supports_thinking=True, supports_reasoning_effort=True),
        )
        assert (thinking, effort) == (True, "high")

    def test_explicit_thinking_off(self):
        thinking, effort = _resolve(
            _subagent_config(thinking_enabled=False),
            _model_config(supports_thinking=True),
        )
        assert (thinking, effort) == (False, None)

    def test_effort_without_thinking_passes_through(self):
        thinking, effort = _resolve(
            _subagent_config(reasoning_effort="low"),
            _model_config(supports_thinking=True, supports_reasoning_effort=True),
        )
        assert (thinking, effort) == (False, "low")

    def test_invalid_effort_fails_fast(self):
        with pytest.raises(ValueError, match="invalid reasoning_effort"):
            _resolve(
                _subagent_config(reasoning_effort="ultra"),
                _model_config(supports_thinking=True),
            )

    def test_thinking_on_incapable_model_fails_fast(self):
        with pytest.raises(ValueError, match="does not support thinking"):
            _resolve(
                _subagent_config(thinking_enabled=True),
                _model_config(supports_thinking=False),
            )

    def test_unknown_model_skips_capability_check(self):
        """model_config None (profile unknown): resolution succeeds; the factory
        raises the canonical unknown-model error downstream."""
        thinking, effort = _resolve(
            _subagent_config(thinking_enabled=True, reasoning_effort="high"),
            None,
        )
        assert (thinking, effort) == (True, "high")

    def test_effort_on_unsupported_model_passes_through_for_factory_strip(self):
        """No error here — the model factory strips effort overrides for models
        without supports_reasoning_effort."""
        thinking, effort = _resolve(
            _subagent_config(reasoning_effort="high"),
            _model_config(supports_thinking=False, supports_reasoning_effort=False),
        )
        assert (thinking, effort) == (False, "high")


class TestCustomSubagentThinkingFields:
    def test_defaults_are_none(self):
        config = CustomSubagentConfig(description="d", system_prompt="p")
        assert config.thinking_enabled is None
        assert config.reasoning_effort is None

    def test_explicit_values(self):
        config = CustomSubagentConfig(
            description="d",
            system_prompt="p",
            thinking_enabled=True,
            reasoning_effort="high",
        )
        assert config.thinking_enabled is True
        assert config.reasoning_effort == "high"


class TestCustomAgentThinkingWiring:
    def test_custom_agents_entry_carries_thinking_fields(self):
        from deerflow.subagents.registry import _build_custom_subagent_config

        app_config = SimpleNamespace(
            subagents=SimpleNamespace(
                custom_agents={
                    "probe": CustomSubagentConfig(
                        description="d",
                        system_prompt="p",
                        thinking_enabled=True,
                        reasoning_effort="medium",
                    )
                },
            )
        )
        config = _build_custom_subagent_config("probe", app_config=app_config)
        assert config.thinking_enabled is True
        assert config.reasoning_effort == "medium"

    def test_custom_agents_defaults_are_none(self):
        from deerflow.subagents.registry import _build_custom_subagent_config

        app_config = SimpleNamespace(
            subagents=SimpleNamespace(
                custom_agents={"probe": CustomSubagentConfig(description="d", system_prompt="p")},
            )
        )
        config = _build_custom_subagent_config("probe", app_config=app_config)
        assert config.thinking_enabled is None
        assert config.reasoning_effort is None

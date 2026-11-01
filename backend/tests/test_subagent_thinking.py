"""Tests for config-driven subagent thinking.

Covers:
- CustomSubagentConfig thinking-field defaults
- registry: custom_agents entries carry the fields into SubagentConfig

Thinking resolution itself lives in the executor via upstream's native
``deerflow.models.reasoning`` contract (the fork's ``resolve_subagent_thinking``
helper was removed in the 2026-11-01 upstream merge).
"""

from types import SimpleNamespace

from deerflow.config.subagents_config import CustomSubagentConfig


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

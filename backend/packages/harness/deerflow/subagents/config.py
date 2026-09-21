"""Subagent configuration definitions."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig


@dataclass
class SubagentConfig:
    """Configuration for a subagent.

    Attributes:
        name: Unique identifier for the subagent.
        description: When Claude should delegate to this subagent.
        system_prompt: The system prompt that guides the subagent's behavior.
        tools: Optional list of tool names to allow. If None, inherits all tools.
        disallowed_tools: Optional list of tool names to deny.
        skills: Optional list of skill names to make discoverable and activatable.
                If None, all enabled skills are available. If empty, skills are
                disabled for this subagent. Skill bodies and their allowed-tools
                policies take effect only after activation/loading at runtime.
        model: Model to use - 'inherit' uses parent's model.
        max_turns: Maximum agent turns — model call plus the tools it runs —
            before stopping. Built-in agents use the value set here
            (general-purpose=150, bash=60) unless the global
            ``subagents.max_turns`` is set. ``turn_budget.py`` converts this
            into the LangGraph ``recursion_limit`` that buys that many turns
            through the assembled middleware chain; it is not passed through as
            a super-step count.
        timeout_seconds: Bare fallback execution-time cap. For built-in agents the
            effective limit is the global ``subagents.timeout_seconds`` (default
            1800 = 30 min), layered on by the registry; this 900 only applies
            when no differing global value exists.
        thinking_enabled: Extended-thinking mode for this subagent's model.
            None (default) preserves today's behavior: non-thinking.
        reasoning_effort: Named effort override for this subagent's model, one
            of ``SUBAGENT_REASONING_EFFORT_LEVELS``. None (default) means no
            override — the model profile value (if any) stands.
    """

    name: str
    description: str
    system_prompt: str | None = None
    tools: list[str] | None = None
    disallowed_tools: list[str] | None = field(default_factory=lambda: ["task"])
    skills: list[str] | None = None
    model: str = "inherit"
    max_turns: int = 50
    timeout_seconds: int = 900
    thinking_enabled: bool | None = None
    reasoning_effort: str | None = None


SUBAGENT_REASONING_EFFORT_LEVELS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
"""Named reasoning-effort levels accepted by subagent config.

Same vocabulary as the run launcher's thinking control. Whether a level is
accepted by a given provider endpoint varies (e.g. some OpenAI-compatible
endpoints reject ``max``) — this set only rejects typos, not valid names.
"""


def resolve_subagent_thinking(config: SubagentConfig, model_config) -> tuple[bool, str | None]:
    """Resolve ``(thinking_enabled, reasoning_effort)`` for model construction.

    ``None`` fields preserve today's behavior: non-thinking, no effort
    override. Raises ``ValueError`` for an unknown effort level, or for
    thinking on a model without ``supports_thinking``. A ``None``
    ``model_config`` (profile unknown) skips the capability check so the
    model factory can raise its canonical unknown-model error downstream.
    Effort on a model without ``supports_reasoning_effort`` is passed
    through untouched — the factory strips it.
    """
    effort = config.reasoning_effort
    if effort is not None and effort not in SUBAGENT_REASONING_EFFORT_LEVELS:
        raise ValueError(f"Subagent '{config.name}': invalid reasoning_effort {effort!r}; expected one of {sorted(SUBAGENT_REASONING_EFFORT_LEVELS)}.")
    thinking = config.thinking_enabled if config.thinking_enabled is not None else False
    if thinking and model_config is not None and not getattr(model_config, "supports_thinking", False):
        raise ValueError(f"Subagent '{config.name}': thinking_enabled=true but model '{getattr(model_config, 'name', '?')}' does not support thinking.")
    return thinking, effort


def _default_model_name(app_config: "AppConfig") -> str:
    if not app_config.models:
        raise ValueError("No chat models are configured. Please configure at least one model in config.yaml.")
    return app_config.models[0].name


def resolve_subagent_model_name(config: SubagentConfig, parent_model: str | None, *, app_config: "AppConfig | None" = None) -> str:
    """Resolve the effective model name a subagent should use."""
    if config.model != "inherit":
        return config.model

    if parent_model is not None:
        return parent_model

    if app_config is None:
        from deerflow.config import get_app_config

        app_config = get_app_config()
    return _default_model_name(app_config)

"""Configuration for the persistent (always-on) bot.

Disabled by default: with ``enabled: false`` the gateway performs no
persistent-bot work at all (no agent, thread, singleton row, or heartbeat
task is created or repaired).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from deerflow.config._boolean_guards import reject_boolean

DEFAULT_HEARTBEAT_PROMPT = "Heartbeat check-in: review this conversation for anything that needs attention or a follow-up. If there is nothing actionable, reply briefly that there is nothing to report."

# Floor for the heartbeat cadence. A persistent bot must never be able to
# spin a sub-five-minute autonomous loop, whatever the operator configures.
MIN_HEARTBEAT_INTERVAL_SECONDS = 300
MAX_HEARTBEAT_INTERVAL_SECONDS = 30 * 24 * 60 * 60


class PersistentBotConfig(BaseModel):
    """Persistent-bot seed, heartbeat cadence, budget, and tool policy."""

    enabled: bool = Field(
        default=False,
        description=(
            "Whether the gateway seeds and repairs the persistent bot "
            "(dedicated agent, pinned thread, and one heartbeat scheduled "
            "task) at startup. Requires scheduler.enabled=true and a SQL "
            "database backend; startup fails closed when this is true and "
            "the seed cannot complete."
        ),
    )
    agent_name: str = Field(
        default="persistent-bot",
        description="Custom-agent name of the persistent bot (letters, digits, and hyphens).",
    )
    owner_user_id: str = Field(
        default="default",
        description=("User id that owns the bot agent, pinned thread, and heartbeat task. Use the default in no-auth mode; set the operator/admin user id when authentication is enabled."),
    )
    thread_title: str = Field(
        default="Persistent Bot",
        description="Display title for the pinned bot thread.",
    )
    heartbeat_interval_seconds: int = Field(
        default=1800,
        ge=MIN_HEARTBEAT_INTERVAL_SECONDS,
        le=MAX_HEARTBEAT_INTERVAL_SECONDS,
        description=(f"Heartbeat cadence in seconds. The floor is {MIN_HEARTBEAT_INTERVAL_SECONDS}s and cannot be configured lower."),
    )
    heartbeat_timezone: str = Field(
        default="UTC",
        description="IANA timezone stored on the heartbeat scheduled task.",
    )
    heartbeat_prompt: str = Field(
        default=DEFAULT_HEARTBEAT_PROMPT,
        description="Prompt sent to the pinned thread on each heartbeat run.",
    )
    overlap_policy: Literal["skip", "enqueue"] = Field(
        default="skip",
        description=("What a due heartbeat does while the pinned thread is busy: 'skip' drops that occurrence (the next tick is fresh); 'enqueue' keeps one durable queued occurrence and retries it until the stale-drop deadline."),
    )
    queued_stale_drop_seconds: int = Field(
        default=900,
        ge=60,
        le=604800,
        description=("A queued heartbeat older than this many seconds past its scheduled time is dropped as skipped instead of launched late."),
    )
    max_heartbeat_runs_per_day: int = Field(
        default=96,
        ge=0,
        le=10000,
        description=("Per-day (UTC) budget of launched heartbeat runs; 0 means no budget. Occurrences due after the budget is reached are skipped until the next day."),
    )
    model: str | None = Field(
        default=None,
        description="Optional model name pinned on the bot agent; None inherits the default model selection.",
    )
    tool_groups: list[str] | None = Field(
        default=None,
        description="Tool groups for the bot agent; None inherits all, [] selects none.",
    )
    skills: list[str] | None = Field(
        default=None,
        description="Skills for the bot agent; None inherits all enabled, [] selects none.",
    )
    allowed_subagents: list[str] | None = Field(
        default=None,
        description="Subagents the bot agent may invoke; None inherits all enabled, [] selects none.",
    )
    notify_policy: Literal["none", "summary", "content"] = Field(
        default="none",
        description=("Proactive IM notice policy stored on the heartbeat task: 'none' (default) sends nothing, 'summary' sends the canned template, 'content' sends redacted run output (needs proactive_content_enabled)."),
    )
    notify_quiet: bool = Field(
        default=False,
        description="Quiet mode for the heartbeat task: suppresses proactive IM notices entirely, whatever notify_policy says.",
    )
    soul_text: str | None = Field(
        default=None,
        description="Optional SOUL.md seed text for the bot agent; None uses the built-in seed. An existing SOUL.md is never overwritten.",
    )

    @field_validator(
        "heartbeat_interval_seconds",
        "queued_stale_drop_seconds",
        "max_heartbeat_runs_per_day",
        mode="before",
    )
    @classmethod
    def _reject_boolean_integers(cls, value: object, info: ValidationInfo) -> object:
        return reject_boolean(value, info, kind="an integer")

    @field_validator("agent_name")
    @classmethod
    def _validate_agent_name(cls, value: str) -> str:
        from deerflow.config.agents_config import validate_agent_name

        validated = validate_agent_name(value)
        assert validated is not None  # value is a str, never None here
        return validated

    @field_validator("owner_user_id", "thread_title", "heartbeat_timezone", "heartbeat_prompt")
    @classmethod
    def _validate_nonblank(cls, value: str, info: ValidationInfo) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"persistent_bot.{info.field_name} must not be blank")
        return value


_persistent_bot_config: PersistentBotConfig = PersistentBotConfig()


def get_persistent_bot_config() -> PersistentBotConfig:
    """Get the current persistent-bot configuration."""
    return _persistent_bot_config


def set_persistent_bot_config(config: PersistentBotConfig) -> None:
    """Set the persistent-bot configuration."""
    global _persistent_bot_config
    _persistent_bot_config = config


def load_persistent_bot_config_from_dict(config_dict: dict) -> None:
    """Load persistent-bot configuration from a dictionary."""
    global _persistent_bot_config
    _persistent_bot_config = PersistentBotConfig(**config_dict)

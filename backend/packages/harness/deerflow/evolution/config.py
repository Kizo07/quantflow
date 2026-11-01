"""Configuration for the native Quantflow evolution loop.

Mirrors the conventions of ``deerflow.knowledge.config`` (pydantic
``BaseModel`` with documented ``Field`` entries plus resolver methods with
environment-variable overrides) and the singleton accessors of
``deerflow.config.memory_config``.

It intentionally does **not** import ``deerflow.config`` (whose package
``__init__`` pulls the full application config graph). The evolution loop
must stay importable standalone so its unit tests run without a Gateway
config and so the integration step can wire it into ``AppConfig`` as the
``evolution:`` section without creating an import cycle.

Resolution order for every resolvable setting:
    1. Explicit model field value
    2. ``DEER_FLOW_EVOLUTION_*`` environment variable
    3. Built-in default

Environment variables:
    DEER_FLOW_EVOLUTION_ENABLED        "1/true/yes/on" or "0/false/no/off"
    DEER_FLOW_EVOLUTION_MAX_ITERATIONS  Positive int budget per run
    DEER_FLOW_EVOLUTION_MAX_WALL_CLOCK_SECONDS
                                      Positive float budget per run (0 = unlimited)
    DEER_FLOW_EVOLUTION_DEFAULT_FAMILY  Default experiment family label
    DEER_FLOW_EVOLUTION_DEFAULT_UNIVERSE
                                      Default research universe label
    DEER_FLOW_EVOLUTION_SANDBOX_TIMEOUT
                                      Per-command sandbox timeout in seconds (0 = none)

Hot-reload contract: scalar budgets (``max_iterations``,
``max_wall_clock_seconds``) are safe to reload between scheduler ticks;
``model_roles`` resolve once at loop construction (missing profile fails
fast with the model name in the error).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

ENV_ENABLED = "DEER_FLOW_EVOLUTION_ENABLED"
ENV_MAX_ITERATIONS = "DEER_FLOW_EVOLUTION_MAX_ITERATIONS"
ENV_MAX_WALL_CLOCK_SECONDS = "DEER_FLOW_EVOLUTION_MAX_WALL_CLOCK_SECONDS"
ENV_DEFAULT_FAMILY = "DEER_FLOW_EVOLUTION_DEFAULT_FAMILY"
ENV_DEFAULT_UNIVERSE = "DEER_FLOW_EVOLUTION_DEFAULT_UNIVERSE"
ENV_SANDBOX_TIMEOUT = "DEER_FLOW_EVOLUTION_SANDBOX_TIMEOUT"

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})

MODEL_ROLES = ("propose", "design", "summarize", "coder")

__all__ = [
    "ENV_ENABLED",
    "ENV_MAX_ITERATIONS",
    "ENV_MAX_WALL_CLOCK_SECONDS",
    "ENV_DEFAULT_FAMILY",
    "ENV_DEFAULT_UNIVERSE",
    "ENV_SANDBOX_TIMEOUT",
    "MODEL_ROLES",
    "EvolutionConfig",
    "get_evolution_config",
    "set_evolution_config",
    "load_evolution_config_from_dict",
]


def _env_str(name: str) -> str | None:
    """Return a stripped env value, treating missing/blank as unset."""
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_flag(name: str) -> bool | None:
    """Parse a boolean env var; return None when unset."""
    raw = _env_str(name)
    if raw is None:
        return None
    token = raw.lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise ValueError(f"Environment variable {name}={raw!r} is not a recognized boolean (expected one of {sorted(_TRUE_TOKENS | _FALSE_TOKENS)}).")


def _env_int(name: str) -> int | None:
    raw = _env_str(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Environment variable {name}={raw!r} is not an integer.") from None


def _env_float(name: str) -> float | None:
    raw = _env_str(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Environment variable {name}={raw!r} is not a number.") from None


class EvolutionConfig(BaseModel):
    """Configuration for the native evolution loop (phase-1 core scope)."""

    enabled: bool = Field(default=False, description="Master switch for scheduled evolution ticks.")
    max_iterations: int = Field(default=10, ge=1, description="Maximum loop iterations per run (budget).")
    max_wall_clock_seconds: float = Field(default=0.0, ge=0.0, description="Wall-clock budget per run in seconds; 0 means unlimited.")
    default_family: str = Field(default="quant-alpha", description="Default experiment family label for new hypotheses.")
    default_universe: str = Field(default="sp500", description="Default research universe label for new hypotheses.")
    min_confidence_to_promote: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum gate confidence before a belief promotion is allowed.")
    sandbox_command_timeout: float = Field(default=300.0, ge=0.0, description="Per-command wall-clock timeout in seconds for sandbox execution; 0 disables the timeout.")
    model_roles: dict[str, str] = Field(
        default_factory=dict,
        description="Role→profile mapping ({propose, design, summarize, coder} → config.yaml model name), resolved once at loop construction.",
    )
    sampling_overrides: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-role sampling overrides (temperature/max_tokens) layered over profiles; None values never clobber profile values.",
    )

    model_config = {"extra": "ignore"}

    def is_enabled(self) -> bool:
        """Return the effective enabled flag (field, then env)."""
        env = _env_flag(ENV_ENABLED)
        return self.enabled if env is None else env

    def resolve_max_iterations(self) -> int:
        """Return the effective iteration budget (field, then env)."""
        env = _env_int(ENV_MAX_ITERATIONS)
        value = self.max_iterations if env is None else env
        if value < 1:
            raise ValueError(f"max_iterations must be >= 1, got {value}.")
        return value

    def resolve_max_wall_clock_seconds(self) -> float:
        """Return the effective wall-clock budget (field, then env; 0 = unlimited)."""
        env = _env_float(ENV_MAX_WALL_CLOCK_SECONDS)
        value = self.max_wall_clock_seconds if env is None else env
        if value < 0:
            raise ValueError(f"max_wall_clock_seconds must be >= 0, got {value}.")
        return value

    def resolve_default_family(self) -> str:
        """Return the effective default family label (field, then env)."""
        return _env_str(ENV_DEFAULT_FAMILY) or self.default_family

    def resolve_default_universe(self) -> str:
        """Return the effective default universe label (field, then env)."""
        return _env_str(ENV_DEFAULT_UNIVERSE) or self.default_universe

    def resolve_sandbox_command_timeout(self) -> float | None:
        """Return the effective sandbox timeout (field, then env; 0 = None)."""
        env = _env_float(ENV_SANDBOX_TIMEOUT)
        value = self.sandbox_command_timeout if env is None else env
        if value < 0:
            raise ValueError(f"sandbox_command_timeout must be >= 0, got {value}.")
        return value if value > 0 else None

    def model_for_role(self, role: str) -> str:
        """Return the configured model profile name for *role* (fail fast)."""
        if role not in MODEL_ROLES:
            raise ValueError(f"Unknown evolution model role {role!r}; expected one of {list(MODEL_ROLES)}.")
        name = self.model_roles.get(role)
        if not name:
            raise ValueError(f"No model profile configured for evolution role {role!r}; set evolution.model_roles[{role!r}] in config.")
        return name

    def overrides_for_role(self, role: str) -> dict[str, Any]:
        """Return sampling overrides for *role* ({} when unconfigured)."""
        overrides = self.sampling_overrides.get(role, {})
        return {key: value for key, value in overrides.items() if value is not None}


_config_singleton: EvolutionConfig | None = None


def get_evolution_config() -> EvolutionConfig:
    """Return the process-wide evolution config (default when never set)."""
    global _config_singleton
    if _config_singleton is None:
        _config_singleton = EvolutionConfig()
    return _config_singleton


def set_evolution_config(config: EvolutionConfig | None) -> None:
    """Replace (or clear, with None) the process-wide evolution config."""
    global _config_singleton
    _config_singleton = config


def load_evolution_config_from_dict(data: Mapping[str, Any]) -> EvolutionConfig:
    """Build and install an :class:`EvolutionConfig` from a plain mapping."""
    if not isinstance(data, Mapping):
        raise ValueError(f"Evolution config data must be a mapping, got {type(data).__name__}.")
    config = EvolutionConfig.model_validate(dict(data))
    set_evolution_config(config)
    return config

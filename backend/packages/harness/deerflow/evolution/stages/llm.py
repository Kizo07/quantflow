"""LLM boundary for the evolution stages.

Hard rule (plan contract): every LLM call goes through
``deerflow.models.create_chat_model`` (LangChain model profiles/factory).
This module is the single place that rule is implemented: stages take a
:class:`ChatModelFactory` and never touch providers directly.

Role→profile mapping comes from ``EvolutionConfig.model_roles`` (resolved
once, fail fast on a missing profile); per-role sampling overrides layer
over the profile without clobbering unset values; standalone loop turns use
``attach_tracing=True`` so model-level callbacks still emit traces.

The langchain and factory imports stay inside the call path so importing
the stages never pulls provider SDKs at module load.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from deerflow.evolution.config import EvolutionConfig
from deerflow.evolution.errors import EvolutionError

__all__ = [
    "ChatModelFactory",
    "LangChainModelFactory",
    "parse_json_object",
]


@runtime_checkable
class ChatModelFactory(Protocol):
    """Role-addressed text generation boundary for the stages."""

    async def generate(self, *, role: str, system: str, user: str) -> str:
        """Generate one completion for *role* and return its text."""
        ...


def parse_json_object(raw: str, *, role: str) -> dict[str, Any]:
    """Parse one JSON object from model output (fences tolerated).

    Raises :class:`EvolutionError` when the output is not a JSON object.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvolutionError(f"The {role} model did not return valid JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise EvolutionError(f"The {role} model must return a JSON object, got {type(parsed).__name__}.")
    return dict(parsed)


def _content_to_text(content: Any) -> str:
    """Render LangChain message content (str or blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text")
                parts.append(text if isinstance(text, str) else repr(block))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


class LangChainModelFactory:
    """Production :class:`ChatModelFactory` over model profiles.

    Args:
        config: Evolution config (role→profile mapping + overrides).
        thinking_enabled: Enable extended-thinking mode when supported.
        app_config: Explicit application config (falls back to the cached
            global inside the factory when omitted).
        reasoning_effort: Optional per-run reasoning effort, forwarded
            through the factory's reasoning contract.
    """

    def __init__(
        self,
        config: EvolutionConfig,
        *,
        thinking_enabled: bool = False,
        app_config: Any | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self._config = config
        self._thinking_enabled = thinking_enabled
        self._app_config = app_config
        self._reasoning_effort = reasoning_effort

    async def generate(self, *, role: str, system: str, user: str) -> str:
        """Generate one completion via the role's configured profile."""
        from langchain_core.messages import HumanMessage, SystemMessage

        from deerflow.models.factory import create_chat_model

        profile = self._config.model_for_role(role)
        overrides = self._config.overrides_for_role(role)
        extra: dict[str, Any] = {}
        if self._reasoning_effort is not None:
            extra["reasoning_effort"] = self._reasoning_effort
        model = create_chat_model(
            profile,
            self._thinking_enabled,
            app_config=self._app_config,
            attach_tracing=True,
            model_overrides=overrides,
            **extra,
        )
        result = await model.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
        return _content_to_text(result.content)

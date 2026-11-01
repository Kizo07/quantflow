"""Minimal text-completion client boundary for knowledge-plane LLM parsing.

Harness-side only: this module defines the :class:`LLMTextClient` protocol
consumed by the opt-in Muse intent parser plus scripted fakes for unit
tests. It deliberately performs no I/O of its own — no gateway imports,
no subprocess calls here (architectural boundary). The app layer binds a
real implementation (``app.gateway.knowledge_llm_client``); when nothing
is bound the keyword parser is used.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DEFAULT_LLM_TIMEOUT",
    "DEFAULT_LLM_MAX_CHARS",
    "LLMTextClient",
    "FakeLLMTextClient",
]

#: Default per-call timeout (seconds) for text completions.
DEFAULT_LLM_TIMEOUT = 120.0
#: Default cap on returned completion characters (responses are truncated).
DEFAULT_LLM_MAX_CHARS = 4000


@runtime_checkable
class LLMTextClient(Protocol):
    """Minimal text-completion surface for knowledge-plane LLM parsing.

    ``timeout`` bounds one call in seconds; ``max_chars`` caps the
    returned completion text (implementations truncate, never fail, on
    overlong responses). Implementations raise on transport/model
    failures — callers fall back to the deterministic keyword path.
    """

    def complete(self, prompt: str, *, timeout: float = DEFAULT_LLM_TIMEOUT, max_chars: int = DEFAULT_LLM_MAX_CHARS) -> str:
        """Return one completion for ``prompt`` (blocking)."""
        ...

    async def acomplete(self, prompt: str, *, timeout: float = DEFAULT_LLM_TIMEOUT, max_chars: int = DEFAULT_LLM_MAX_CHARS) -> str:
        """Return one completion for ``prompt`` (async)."""
        ...


class FakeLLMTextClient:
    """Scripted :class:`LLMTextClient` for unit tests.

    Args:
        responses: Replay queue; each entry is either a completion
            string (returned verbatim, truncated to ``max_chars``) or a
            :class:`BaseException` instance (raised). When the queue is
            empty ``default_response`` is returned instead.
        default_response: Completion used once ``responses`` is exhausted.

    Every call (sync or async) is recorded in :attr:`calls` as a dict
    with ``prompt``/``timeout``/``max_chars`` keys.
    """

    def __init__(self, responses: Any = (), *, default_response: str = "") -> None:
        if isinstance(responses, (str, BaseException)):
            responses = (responses,)
        self._responses: deque[Any] = deque(responses)
        self._default_response = default_response
        self.calls: list[dict[str, Any]] = []

    def complete(self, prompt: str, *, timeout: float = DEFAULT_LLM_TIMEOUT, max_chars: int = DEFAULT_LLM_MAX_CHARS) -> str:
        """Replay the next scripted response (or raise the scripted error)."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")
        self.calls.append({"prompt": prompt, "timeout": timeout, "max_chars": max_chars})
        if self._responses:
            item = self._responses.popleft()
            if isinstance(item, BaseException):
                raise item
            text = item
        else:
            text = self._default_response
        if not isinstance(text, str):
            raise ValueError(f"scripted response must be a string, got {type(text).__name__}.")
        if max_chars is not None and max_chars >= 0:
            text = text[:max_chars]
        return text

    async def acomplete(self, prompt: str, *, timeout: float = DEFAULT_LLM_TIMEOUT, max_chars: int = DEFAULT_LLM_MAX_CHARS) -> str:
        """Replay the next scripted response (async mirror of :meth:`complete`)."""
        return self.complete(prompt, timeout=timeout, max_chars=max_chars)

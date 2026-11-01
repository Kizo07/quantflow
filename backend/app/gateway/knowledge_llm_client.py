"""App-layer :class:`LLMTextClient` bound to ``muse exec --json``.

Implements the harness ``deerflow.knowledge.llm_text_client`` protocol by
spawning one headless CLI run per completion (provider ``meta``, pinned
model ``muse-spark-1.3``). Argument construction reuses
:mod:`app.gateway.cli_runner` validation (allowlisted provider/model ids,
per-argument quoting); audit strings are scrubbed with
:func:`redact_command_string` so secret-looking ``key=value`` fragments in
the task text can never land in logs. This module persists nothing — the
prompt travels only as a subprocess argv element, and every logged string
is the redacted form.

Timeouts are enforced on both paths (``subprocess.run`` for sync,
:func:`run_probe` for async); every failure raises
:class:`MuseExecError` (or :class:`CliValidationError` for invalid
prompts) so the harness caller can fall back to the keyword parser.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass

from app.gateway import cli_runner
from app.gateway.cli_runner import (
    CLI_MODEL_MUSE_13,
    CliLaunchParams,
    CliProbeError,
    CliValidationError,
)
from deerflow.knowledge.llm_text_client import (
    DEFAULT_LLM_MAX_CHARS,
    DEFAULT_LLM_TIMEOUT,
    LLMTextClient,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CliValidationError",
    "MuseExecError",
    "MuseIntentClient",
    "extract_exec_text",
]

#: Pinned provider for knowledge intent parsing (``muse exec --help`` mode).
INTENT_PROVIDER = "meta"


class MuseExecError(RuntimeError):
    """A ``muse exec`` completion run failed, timed out, or yielded no text."""


def extract_exec_text(stdout: str) -> str:
    """Extract the final text from ``muse exec --json`` JSONL stdout.

    The stream carries ``run.output.delta`` text chunks plus one
    ``run.terminal.completed`` event repeating the full text; the
    terminal payload wins when present, otherwise the deltas are joined.
    A ``run.terminal.failed`` event (or any terminal event whose status
    is not ``completed``) raises :class:`MuseExecError`, as does an
    empty result.
    """
    deltas: list[str] = []
    terminal_text: str | None = None
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        payload_type = event.get("payload_type")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload_type == "run.output.delta":
            text = payload.get("text")
            if isinstance(text, str) and text:
                deltas.append(text)
        elif isinstance(payload_type, str) and payload_type.startswith("run.terminal."):
            status = payload.get("terminal")
            if payload_type != "run.terminal.completed" or status not in (None, "completed"):
                reason = payload.get("reason") or payload.get("text") or payload_type
                raise MuseExecError(f"muse exec run did not complete ({payload_type}): {reason}")
            text = payload.get("text")
            if isinstance(text, str) and text:
                terminal_text = text
    result = terminal_text if terminal_text else "".join(deltas)
    if not result.strip():
        raise MuseExecError("muse exec run produced no output text.")
    return result


@dataclass
class MuseIntentClient(LLMTextClient):
    """Run one ``muse exec --json`` completion per call (blocking or async).

    Attributes:
        model: Pinned model id (default ``muse-spark-1.3``).
        provider: Pinned provider mode (default ``meta``).
        effort: Optional reasoning-effort tier; ``None`` lets the CLI
            apply its default (high).
        binary: CLI binary name for argv construction.
        default_timeout: Per-call timeout in seconds.
        default_max_chars: Cap applied to returned completion text.
    """

    model: str = CLI_MODEL_MUSE_13
    provider: str = INTENT_PROVIDER
    effort: str | None = None
    binary: str = "muse"
    default_timeout: float = DEFAULT_LLM_TIMEOUT
    default_max_chars: int = DEFAULT_LLM_MAX_CHARS

    def _launch_params(self, prompt: str) -> CliLaunchParams:
        """Validate prompt/model/provider through the cli_runner gate."""
        return cli_runner.validate_launch_params(CliLaunchParams(prompt=prompt, model=self.model, provider=self.provider, effort=self.effort))

    def _audit_label(self, params: CliLaunchParams) -> str:
        """Redacted command string for logs (never the raw prompt)."""
        return cli_runner.redact_command_string(cli_runner.build_command_string(params, binary=self.binary))

    @staticmethod
    def _check_limits(timeout: float, max_chars: int) -> None:
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"timeout must be a positive number of seconds, got {timeout!r}.")
        if not isinstance(max_chars, int) or max_chars < 1:
            raise ValueError(f"max_chars must be a positive integer, got {max_chars!r}.")

    def complete(
        self,
        prompt: str,
        *,
        timeout: float = DEFAULT_LLM_TIMEOUT,
        max_chars: int = DEFAULT_LLM_MAX_CHARS,
    ) -> str:
        """Run one blocking ``muse exec`` completion (call from sync code only)."""
        self._check_limits(timeout, max_chars)
        params = self._launch_params(prompt)
        argv = cli_runner.build_exec_argv(params, binary=self.binary)
        logger.debug("knowledge intent exec: %s", self._audit_label(params))
        try:
            finished = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        except FileNotFoundError as exc:
            raise MuseExecError(f"muse binary not found: {self.binary!r}.") from exc
        except subprocess.TimeoutExpired as exc:
            raise MuseExecError(f"muse exec timed out after {timeout}s.") from exc
        if finished.returncode != 0:
            detail = (finished.stderr or "").strip().splitlines()
            tail = detail[-1] if detail else f"exit {finished.returncode}"
            raise MuseExecError(f"muse exec failed (exit {finished.returncode}): {tail[:500]}")
        return extract_exec_text(finished.stdout)[:max_chars]

    async def acomplete(
        self,
        prompt: str,
        *,
        timeout: float = DEFAULT_LLM_TIMEOUT,
        max_chars: int = DEFAULT_LLM_MAX_CHARS,
    ) -> str:
        """Run one ``muse exec`` completion via a non-blocking control probe."""
        self._check_limits(timeout, max_chars)
        params = self._launch_params(prompt)
        argv = cli_runner.build_exec_argv(params, binary=self.binary)
        logger.debug("knowledge intent exec: %s", self._audit_label(params))
        try:
            returncode, stdout, stderr = await cli_runner.run_probe(argv, timeout=timeout)
        except CliProbeError as exc:
            raise MuseExecError(str(exc)) from exc
        if returncode != 0:
            tail = (stderr or "").strip().splitlines()
            detail = tail[-1] if tail else f"exit {returncode}"
            raise MuseExecError(f"muse exec failed (exit {returncode}): {detail[:500]}")
        return extract_exec_text(stdout)[:max_chars]

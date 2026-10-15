"""Launch + supervise headless CLI runs for the cli_sessions registry.

Builds ``muse exec`` command strings from validated parameters only
(allowlisted flags, per-argument :func:`shlex.quote`), spawns them either
inside a named tmux session (``qf-<row id>``) so runs survive a gateway
restart, or as a direct background subprocess. Both paths mirror stdout
to the same per-run JSONL log file; completion is signalled by a
sentinel file holding the exit code.

Short control probes (``tmux has-session``, ``trace inspect``,
``export``) run through :func:`asyncio.create_subprocess_exec` with a
timeout — never through a blocking call on the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Phase-0 pinned values (U0.5): the only model id verified against the
# local CLI surface, and the preset row that may persist privileged
# defaults (contributor mode, top effort, yolo) for it.
CLI_MODEL_MUSE_13 = "muse-spark-1.3"
CLI_PRESET_MUSE_13 = "muse-1.3"
CLI_TOP_EFFORT = "ultra"

# Built-in Muse 1.3 launch defaults. Fresh databases are provisioned via
# create_all + stamp (the 0041 seed insert only runs on the upgrade path),
# so preset reads and launch-time loads fall back to this when no stored
# row exists; PUT persists an overriding row.
DEFAULT_MUSE_13_PRESET: dict[str, Any] = {
    "name": CLI_PRESET_MUSE_13,
    "model": CLI_MODEL_MUSE_13,
    "provider": None,
    "preset": None,
    "effort": CLI_TOP_EFFORT,
    "contributor_mode": True,
    "yolo": True,
    "approval_mode": None,
}


async def load_preset_or_default(preset_repo: Any, name: str) -> dict[str, Any] | None:
    """Return the stored preset row, the built-in 1.3 default, or None."""
    row = await preset_repo.get(name)
    if row is not None:
        return row
    if name == CLI_PRESET_MUSE_13:
        return dict(DEFAULT_MUSE_13_PRESET)
    return None


EFFORT_TIERS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"})
# CLI default when --reasoning-effort is omitted (from `muse exec --help`):
# the runner passes nothing and lets the CLI apply it.
CLI_DEFAULT_EFFORT = "high"
PROVIDERS = frozenset({"echo", "meta", "local"})
APPROVAL_MODES = frozenset({"untrusted", "on-request", "never"})

TERMINAL_STATUSES = frozenset({"completed", "failed", "stopped", "interrupted"})

_MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}")
_SANDBOX_NETWORK_RE = re.compile(r"[a-z][a-z0-9-]{0,31}")
# Run presets are an opaque passthrough: the CLI owns the preset
# namespace (see `muse exec --help` on this host), so the gateway only
# rejects empty/overlong/unsafe tokens and forwards the rest verbatim.
_PRESET_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}")

_SESSION_LOG_GLOB = "session.jsonl"
_DISCOVERY_CAP = 500


class CliValidationError(ValueError):
    """A launch/preset parameter failed fail-closed validation."""


class CliSpawnError(RuntimeError):
    """The run could not be started on the selected path."""


class CliProbeError(RuntimeError):
    """A short control probe failed or timed out."""


@dataclass(frozen=True)
class CliLaunchParams:
    """Validated inputs for one ``muse exec`` launch.

    There is intentionally no free-form extra-args field: every flag the
    runner can emit is a named, validated attribute below.
    """

    prompt: str
    model: str | None = None
    provider: str | None = None
    preset: str | None = None
    effort: str | None = None
    contributor_mode: bool | None = None
    approval_mode: str | None = None
    sandbox_network: str | None = None
    trust_workspace: bool = False
    disable_approval: bool = False
    disable_sandbox: bool = False
    yolo: bool = False
    workspace: str | None = None
    session_id: str | None = None


def validate_launch_params(params: CliLaunchParams) -> CliLaunchParams:
    """Return *params* when every field is allowlisted, else raise."""
    if not params.prompt or not params.prompt.strip():
        raise CliValidationError("prompt must not be empty")
    if params.model is not None and _MODEL_ID_RE.fullmatch(params.model) is None:
        raise CliValidationError(f"unsupported model id: {params.model!r}")
    if params.provider is not None and params.provider not in PROVIDERS:
        raise CliValidationError(f"unsupported provider: {params.provider!r} (expected one of {sorted(PROVIDERS)})")
    if params.preset is not None and _PRESET_NAME_RE.fullmatch(params.preset) is None:
        raise CliValidationError(f"unsupported preset: {params.preset!r}")
    if params.effort is not None and params.effort not in EFFORT_TIERS:
        raise CliValidationError(f"unsupported effort: {params.effort!r} (expected one of {sorted(EFFORT_TIERS)})")
    if params.approval_mode is not None and params.approval_mode not in APPROVAL_MODES:
        raise CliValidationError(f"unsupported approval_mode: {params.approval_mode!r} (expected one of {sorted(APPROVAL_MODES)})")
    if params.sandbox_network is not None and _SANDBOX_NETWORK_RE.fullmatch(params.sandbox_network) is None:
        raise CliValidationError(f"unsupported sandbox_network: {params.sandbox_network!r}")
    if params.workspace is not None and not Path(params.workspace).is_dir():
        raise CliValidationError(f"workspace is not an existing directory: {params.workspace!r}")
    if params.provider == "echo" and params.effort is not None:
        raise CliValidationError("reasoning effort is not supported with the echo provider (CLI refuses the run)")
    return params


def build_exec_argv(params: CliLaunchParams, *, binary: str = "muse") -> list[str]:
    """Build the ``muse exec`` argv from validated params (no shell quoting)."""
    argv = [binary, "exec", "--json"]
    if params.session_id is not None:
        argv += ["--session-id", params.session_id]
    if params.model is not None:
        argv += ["--model", params.model]
    if params.provider is not None:
        argv += ["--provider", params.provider]
    if params.preset is not None:
        argv += ["--preset", params.preset]
    if params.effort is not None:
        argv += ["--reasoning-effort", params.effort]
    if params.contributor_mode is not None:
        argv += ["--contributor-mode", "on" if params.contributor_mode else "off"]
    if params.approval_mode is not None:
        argv += ["--approval-mode", params.approval_mode]
    if params.sandbox_network is not None:
        argv += ["--sandbox-network", params.sandbox_network]
    if params.workspace is not None:
        argv += ["--workspace", params.workspace]
    if params.trust_workspace:
        argv.append("--trust-workspace")
    if params.disable_approval:
        argv.append("--disable-approval")
    if params.disable_sandbox:
        argv.append("--disable-sandbox")
    if params.yolo:
        argv.append("--yolo")
    argv.append(params.prompt)
    return argv


def build_command_string(params: CliLaunchParams, *, binary: str = "muse") -> str:
    """Build the shell command string, quoting every argument."""
    return shlex.join(build_exec_argv(params, binary=binary))


# A prompt pasted into a launch may carry secrets, so audit rows store a
# scrubbed command string. Each pattern keeps its flag/key name and masks
# the value that follows it.
_REDACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(bearer)\s+[^\s'\"]+"),
    re.compile(r"(?i)\b(api[_-]?key|auth[_-]?token|access[_-]?token|secret|password|passwd|pwd|token)\s*([:=])\s*[^\s'\"]+"),
)


def redact_command_string(command: str) -> str:
    """Mask secret-looking values in a command string for audit storage."""
    scrubbed = _REDACT_PATTERNS[0].sub("bearer [redacted]", command)
    return _REDACT_PATTERNS[1].sub(lambda match: f"{match.group(1)}{match.group(2)}[redacted]", scrubbed)


def build_export_argv(*, session_ref: str, out_path: Path, redacted: bool = True, binary: str = "muse") -> list[str]:
    """Build the session-export probe argv."""
    argv = [binary, "export", "--session", session_ref, "--out", str(out_path)]
    if redacted:
        argv.append("--redacted")
    return argv


def build_inspect_argv(*, session_log: str, binary: str = "muse") -> list[str]:
    """Build the trace-inspect probe argv."""
    return [binary, "trace", "inspect", "--session-log", session_log]


def apply_preset_defaults(params: CliLaunchParams, preset: dict[str, Any] | None) -> CliLaunchParams:
    """Fill unset launch fields from a preset row.

    The Muse 1.3 preset applies only when the launch model explicitly
    matches the 1.3 model key. An omitted model does NOT match — no
    preset defaults are applied (fail-closed).
    """
    if preset is None or params.model != CLI_MODEL_MUSE_13:
        return params
    updates: dict[str, Any] = {}
    if params.provider is None and preset.get("provider") is not None:
        updates["provider"] = preset["provider"]
    if params.preset is None and preset.get("preset") is not None:
        updates["preset"] = preset["preset"]
    if params.effort is None and preset.get("effort") is not None:
        updates["effort"] = preset["effort"]
    if params.contributor_mode is None and preset.get("contributor_mode") is not None:
        updates["contributor_mode"] = preset["contributor_mode"]
    if params.approval_mode is None and preset.get("approval_mode") is not None:
        updates["approval_mode"] = preset["approval_mode"]
    if not params.yolo and preset.get("yolo"):
        # The preset only defaults the toggle; the per-launch confirm
        # token is still required by validate_launch_params.
        updates["yolo"] = True
    return replace(params, **updates) if updates else params


def check_preset_guard(*, model: str | None, effort: str | None, contributor_mode: bool | None, yolo: bool) -> None:
    """Reject privileged persistence for models other than Muse 1.3.

    Contributor mode, any explicit reasoning effort, and yolo may be
    persisted only under the 1.3 model key; every other model takes
    those as per-run flags only.
    """
    if model == CLI_MODEL_MUSE_13:
        return
    if contributor_mode is not None:
        raise CliValidationError("contributor_mode may be persisted only for the Muse 1.3 model")
    if effort is not None:
        raise CliValidationError("effort may be persisted only for the Muse 1.3 model")
    if yolo:
        raise CliValidationError("yolo may be persisted only for the Muse 1.3 model")


def tmux_available() -> bool:
    """Whether the tmux binary is present on PATH."""
    return shutil.which("tmux") is not None


def resolve_spawn_path(tmux_mode: str) -> str:
    """Map the ``cli.tmux`` config value to a concrete spawn path."""
    if tmux_mode == "on":
        if not tmux_available():
            raise CliSpawnError("cli.tmux=on but the tmux binary is not available")
        return "tmux"
    if tmux_mode == "off":
        return "direct"
    if tmux_mode == "auto":
        return "tmux" if tmux_available() else "direct"
    raise CliSpawnError(f"invalid cli.tmux value: {tmux_mode!r} (expected auto|on|off)")


async def run_probe(argv: list[str], *, timeout: float) -> tuple[int, str, str]:
    """Run a short control probe, returning (returncode, stdout, stderr)."""
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise CliProbeError(f"probe binary not found: {argv[0]}") from exc
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError as exc:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        raise CliProbeError(f"probe timed out after {timeout}s: {argv[0]}") from exc
    return process.returncode or 0, stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace")


async def tmux_has_session(name: str, *, timeout: float = 10.0) -> bool:
    """Whether a tmux session *name* currently exists."""
    try:
        returncode, _, _ = await run_probe(["tmux", "has-session", "-t", name], timeout=timeout)
    except CliProbeError:
        return False
    return returncode == 0


async def tmux_capture_pane(name: str, *, timeout: float = 10.0) -> str:
    """Capture a tmux session's visible pane (fallback probe)."""
    returncode, stdout, stderr = await run_probe(["tmux", "capture-pane", "-p", "-t", name], timeout=timeout)
    if returncode != 0:
        raise CliProbeError(f"tmux capture-pane failed for {name}: {stderr.strip()}")
    return stdout


def tmux_session_name(row_id: str) -> str:
    return f"qf-{row_id}"


def wrap_with_log_mirror(command: str, *, log_path: Path, sentinel_path: Path) -> str:
    """Wrap *command* so stdout/stderr append to the run log and the exit code lands in the sentinel file."""
    return f"{command} >> {shlex.quote(str(log_path))} 2>&1; code=$?; printf '%s' \"$code\" > {shlex.quote(str(sentinel_path))}; exit $code"


@dataclass
class SpawnResult:
    path: str
    tmux_session: str | None = None
    pid: int | None = None


# Direct-path child handles, keyed by row id. Process-local only: after a
# gateway restart this is empty and boot reconciliation falls back to the
# stored pid's /proc cmdline check (direct rows) or has-session (tmux rows).
_LIVE_PROCESSES: dict[str, asyncio.subprocess.Process] = {}


async def spawn_run(
    *,
    row_id: str,
    command: str,
    log_path: Path,
    tmux_mode: str,
    workspace: str | None = None,
    timeout: float = 30.0,
) -> SpawnResult:
    """Start one CLI run on the tmux or direct path; both mirror to *log_path*."""
    path = resolve_spawn_path(tmux_mode)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path = log_path.with_suffix(".exit")
    if sentinel_path.exists():
        sentinel_path.unlink()
    wrapped = wrap_with_log_mirror(command, log_path=log_path, sentinel_path=sentinel_path)
    cwd = workspace or None
    if path == "tmux":
        name = tmux_session_name(row_id)
        # never capture pipes here: a freshly spawned tmux server
        # inherits the client's stdout/stderr, so awaiting EOF would
        # block until the server itself exits (i.e. until the run ends).
        try:
            process = await asyncio.create_subprocess_exec(
                "tmux",
                "new-session",
                "-d",
                "-s",
                name,
                "sh",
                "-c",
                wrapped,
                cwd=cwd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise CliSpawnError("tmux binary not found") from exc
        try:
            returncode = await asyncio.wait_for(process.wait(), timeout=timeout)
        except TimeoutError as exc:
            raise CliSpawnError(f"tmux new-session timed out for {name}") from exc
        if returncode != 0:
            raise CliSpawnError(f"tmux new-session failed for {name} (exit {returncode})")
        return SpawnResult(path="tmux", tmux_session=name)
    process = await asyncio.create_subprocess_exec(
        "sh",
        "-c",
        wrapped,
        cwd=cwd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    _LIVE_PROCESSES[row_id] = process
    return SpawnResult(path="direct", pid=process.pid)


async def stop_run(*, row_id: str, tmux_session: str | None, pid: int | None, timeout: float = 10.0) -> None:
    """Best-effort termination of a run; never raises."""
    handle = _LIVE_PROCESSES.pop(row_id, None)
    if handle is not None:
        try:
            handle.terminate()
            await asyncio.wait_for(handle.wait(), timeout=timeout)
        except TimeoutError:
            try:
                handle.kill()
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass
        except Exception:
            logger.debug("direct-path stop failed for %s", row_id, exc_info=True)
    if tmux_session:
        try:
            await run_probe(["tmux", "kill-session", "-t", tmux_session], timeout=timeout)
        except CliProbeError:
            logger.debug("tmux kill-session failed for %s", tmux_session, exc_info=True)
    elif pid is not None and handle is None:
        try:
            os.kill(pid, 15)
        except (ProcessLookupError, PermissionError):
            pass
        except Exception:
            logger.debug("pid stop failed for %s", pid, exc_info=True)


def _pid_cmdline_contains(pid: int, marker: str) -> bool:
    """Whether /proc/<pid>/cmdline still mentions *marker* (Linux only)."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return marker.encode() in handle.read()
    except OSError:
        return False


def sentinel_exit_code(log_path: str | Path | None) -> int | None:
    """Read the sentinel exit code for a run log, or None when still running."""
    if not log_path:
        return None
    sentinel = Path(log_path).with_suffix(".exit")
    try:
        text = sentinel.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def discover_session_logs(*, root: Path | None = None) -> list[dict[str, Any]]:
    """List on-disk CLI session logs, newest first (never raises, never invents)."""
    base = root or (Path.home() / ".local" / "share" / "muse" / "sessions")
    try:
        candidates = sorted(base.rglob(_SESSION_LOG_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []
    entries: list[dict[str, Any]] = []
    for path in candidates[:_DISCOVERY_CAP]:
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append(
            {
                "session_id": path.parent.name,
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    return entries


async def export_session(
    *,
    session_ref: str,
    out_path: Path,
    redacted: bool = True,
    binary: str = "muse",
    timeout: float = 60.0,
) -> Path:
    """Export one CLI session's durable log to *out_path* via a control probe."""
    argv = build_export_argv(session_ref=session_ref, out_path=out_path, redacted=redacted, binary=binary)
    returncode, _, stderr = await run_probe(argv, timeout=timeout)
    if returncode != 0:
        raise CliProbeError(f"export failed for {session_ref}: {stderr.strip()}")
    return out_path


async def inspect_session_log(
    *,
    session_log: str,
    binary: str = "muse",
    timeout: float = 60.0,
) -> str:
    """Render ``trace inspect`` output for one on-disk session log."""
    returncode, stdout, stderr = await run_probe(build_inspect_argv(session_log=session_log, binary=binary), timeout=timeout)
    if returncode != 0:
        raise CliProbeError(f"trace inspect failed: {stderr.strip()}")
    return stdout


async def reconcile_on_boot(session_repo: Any, *, probe_timeout: float = 10.0) -> dict[str, Any]:
    """Reattach live runs after a restart; mark process-gone rows interrupted.

    Terminal rows (completed/failed/stopped/interrupted) are never
    rewritten. Rows are never silently dropped.
    """
    running = await session_repo.list_running()
    summary: dict[str, Any] = {"checked": len(running), "reattached": [], "finished": [], "interrupted": []}
    for record in running:
        row_id = record["id"]
        exit_code = sentinel_exit_code(record.get("log_path"))
        if exit_code is not None:
            # The run ended while no poll observed it: finalize from the
            # sentinel rather than crying interrupted.
            await session_repo.mark_finished(row_id, user_id=record["created_by"], exit_code=exit_code)
            summary["finished"].append(row_id)
            continue
        tmux_session = record.get("tmux_session")
        pid = record.get("pid")
        alive = False
        if tmux_session and await tmux_has_session(tmux_session, timeout=probe_timeout):
            alive = True
        elif pid is not None and _pid_cmdline_contains(int(pid), row_id):
            alive = True
        if alive:
            summary["reattached"].append(row_id)
        else:
            await session_repo.mark_interrupted(row_id)
            summary["interrupted"].append(row_id)
    return summary

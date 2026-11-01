"""Quantflow sandbox adaptation for the evolution execute stage.

This module adapts the generic sandbox code/run boundary
(:mod:`deerflow.evolution.stages.execute`) to Quantflow conventions:

* Canonical workdir: ``/mnt/user-data/evolution/<run_id>/iter-<n>`` honors
  the thread-scoped user-data contract instead of ``/tmp`` scratch space,
  so runs work identically on host-backed and bind-mount providers.
* Code contract: the coder's ``code`` mapping is normalized to
  ``{files: {path: text}, entrypoint: command}`` before anything is
  written; malformed code becomes an error outcome (``SandboxError``),
  never a half-written sandbox.
* Provider manifest: every run stages a ``provider.json`` manifest naming
  the quantitative provider (``alpha_engine``) and the datasets the spec
  pins. Quantflow never imports that provider (it is a separate,
  reference-only repository); price/factor computation happens
  governor-side through spark ``backtest_run``, and the manifest exists so
  sandbox code can assert the datasets it was promised.
* Artifact harvest: files the entrypoint leaves under ``outputs/`` are
  collected as outcome artifact refs (capped), so the KB commit links
  the run's result artifacts without the coder naming them upfront.
  Harvested paths are not KB artifact UUIDs; the knowledge adapter
  carries them into the commit on ``metrics["sandbox_artifacts"]``.

:class:`QuantflowSandboxExecutor` is the production
:class:`~deerflow.evolution.stages.execute.SandboxExecutor`; the wiring
step binds it over :func:`deerflow.sandbox.get_sandbox_provider`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from deerflow.evolution.loop import StageContext
from deerflow.evolution.models import ExperimentSpec
from deerflow.evolution.stages.execute import ProviderSandboxExecutor, SandboxResult

__all__ = [
    "CODE_CONTRACT_VERSION",
    "PROVIDER_NAME",
    "OUTPUTS_DIRNAME",
    "PROVIDER_MANIFEST_FILENAME",
    "MAX_HARVESTED_ARTIFACTS",
    "evolution_workdir",
    "normalize_code_mapping",
    "provider_manifest",
    "QuantflowSandboxExecutor",
]

CODE_CONTRACT_VERSION = "evolution-code/v1"
PROVIDER_NAME = "alpha_engine"
OUTPUTS_DIRNAME = "outputs"
PROVIDER_MANIFEST_FILENAME = "provider.json"
MAX_HARVESTED_ARTIFACTS = 32


def evolution_workdir(run_id: str, iteration: int) -> str:
    """Return the canonical sandbox workdir for one iteration."""
    return f"/mnt/user-data/evolution/{run_id}/iter-{iteration}"


def normalize_code_mapping(code: Mapping[str, Any]) -> tuple[dict[str, str], str]:
    """Normalize a coder ``code`` mapping to (files, entrypoint).

    Raises :class:`ValueError` with a precise message when the mapping
    carries no runnable files or no entrypoint command.
    """
    files = code.get("files", {})
    entrypoint = code.get("entrypoint", "")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("Coded experiment carries no files to run.")
    normalized: dict[str, str] = {}
    for path, content in files.items():
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"Sandbox file path {path!r} must be a non-empty string.")
        if not isinstance(content, str):
            raise ValueError(f"Sandbox file {path!r} must map a path to text content.")
        normalized[path] = content
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ValueError("Coded experiment carries no entrypoint command.")
    return normalized, entrypoint.strip()


def provider_manifest(*, run_id: str, iteration: int, spec: ExperimentSpec) -> dict[str, Any]:
    """Build the provider manifest staged alongside the coder files."""
    datasets = spec.data.get("datasets", [])
    return {
        "contract_version": CODE_CONTRACT_VERSION,
        "provider": PROVIDER_NAME,
        "run_id": run_id,
        "iteration": iteration,
        "datasets": list(datasets) if isinstance(datasets, list) else [],
    }


class QuantflowSandboxExecutor(ProviderSandboxExecutor):
    """Provider-backed executor with Quantflow workdir/manifest/artifacts.

    Runs the coded files' entrypoint in the canonical evolution workdir,
    stages ``provider.json`` (unless the coder already emitted one, which
    is kept verbatim), and harvests ``outputs/`` entries as artifacts.
    """

    def _run_sync(self, coded: Any, ctx: StageContext) -> SandboxResult:
        try:
            files, entrypoint = normalize_code_mapping(coded.code)
        except ValueError as exc:
            return SandboxResult(stdout="", error=str(exc))
        workdir = evolution_workdir(ctx.run_id, ctx.iteration)
        manifest = provider_manifest(run_id=ctx.run_id, iteration=ctx.iteration, spec=coded.spec)
        provider = self._provider
        timeout = self._command_timeout
        sandbox_id = provider.acquire(thread_id=ctx.run_id)
        try:
            sandbox = provider.get(sandbox_id)
            if sandbox is None:
                return SandboxResult(stdout="", error=f"Sandbox {sandbox_id} vanished after acquire.")
            for path, content in files.items():
                sandbox.write_file(f"{workdir}/{path}", content)
            if PROVIDER_MANIFEST_FILENAME not in files:
                sandbox.write_file(f"{workdir}/{PROVIDER_MANIFEST_FILENAME}", json.dumps(manifest, indent=2, sort_keys=True))
            # Trust boundary: the entrypoint is LLM-generated shell text. It
            # runs only inside the provider sandbox (never on the host), so
            # no host-side sanitization is applied; do not reuse this pattern
            # for host execution.
            output = sandbox.execute_command(f"cd {workdir} && {entrypoint}", timeout=timeout)
            return SandboxResult(stdout=output or "", artifacts=self._harvest_artifacts(sandbox, workdir))
        finally:
            provider.release(sandbox_id)

    @staticmethod
    def _harvest_artifacts(sandbox: Any, workdir: str) -> tuple[str, ...]:
        """Collect ``outputs/`` entries as artifact refs (best effort).

        Entries are ``outputs/<name>`` path refs, not KB artifact UUIDs;
        the knowledge adapter routes them into the commit's metrics.
        """
        try:
            entries = sandbox.list_dir(f"{workdir}/{OUTPUTS_DIRNAME}")
        except Exception:
            return ()
        if not isinstance(entries, list):
            return ()
        harvested: list[str] = []
        for entry in entries:
            if not isinstance(entry, str) or not entry.strip():
                continue
            name = entry.strip().rstrip("/").rsplit("/", 1)[-1]
            if name:
                harvested.append(f"{OUTPUTS_DIRNAME}/{name}")
            if len(harvested) >= MAX_HARVESTED_ARTIFACTS:
                break
        return tuple(harvested)

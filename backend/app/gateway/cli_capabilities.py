"""Local CLI capability discovery for the dashboard launch form.

Discovery mechanism (verified on this host, 2026-10-13): the ``muse`` CLI
ships no list/enumerate verb for models, and ``muse model-profile show
<anything>`` exits 0 even for bogus ids, so per-id probing cannot verify
anything. The sole enumeration authority is the provider catalog JSON
under ``~/.local/share/muse/model-catalog/*.json``: every advertised
model id comes from a ``visibility == "visible"`` catalog row, and each
model's effort tiers come from that row's ``reasoning_effort_variants``
``tier`` entries (cross-checked against the global tier list in
``muse exec --help``). Hidden rows (``*-contributor`` twins, image
models) are never advertised: contributor mode travels as its own
launch flag.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.gateway import cli_runner

# Canonical tier order from `muse exec --help` (Muse Code 1.4.6).
_CLI_TIER_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")

# Short TTL so model installs/fresh catalogs show up without per-request
# disk storms.
_CLI_CAPABILITIES_TTL_SECONDS = 300.0

# Overridable catalog directory (tests point it at a tmp dir).
_CLI_CATALOG_DIR_OVERRIDE: Path | None = None

_cache_payload: dict[str, Any] | None = None
_cache_expires_at: float = 0.0
_cache_last_good: dict[str, Any] | None = None


class CliCapabilitiesUnavailable(RuntimeError):
    """Raised when no catalog probe ever succeeded (serve 503, never invented data)."""


def _catalog_dir() -> Path:
    if _CLI_CATALOG_DIR_OVERRIDE is not None:
        return _CLI_CATALOG_DIR_OVERRIDE
    return Path.home() / ".local/share/muse" / "model-catalog"


def _reset_cli_capabilities_cache() -> None:
    """Clear cached payloads (tests only)."""
    global _cache_payload, _cache_expires_at, _cache_last_good
    _cache_payload = None
    _cache_expires_at = 0.0
    _cache_last_good = None


def _ordered_tiers(variants: Any) -> list[str]:
    tiers: set[str] = set()
    if isinstance(variants, list):
        for entry in variants:
            tier = entry.get("tier") if isinstance(entry, dict) else None
            if isinstance(tier, str) and tier in cli_runner.EFFORT_TIERS:
                tiers.add(tier)
    return [tier for tier in _CLI_TIER_ORDER if tier in tiers]


def _provider_label(provider_id: str) -> str:
    return {"meta": "Meta"}.get(provider_id, provider_id.replace("_", " ").title() or provider_id)


def _probe_cli_capabilities() -> dict[str, Any]:
    """Read every catalog file and build the capabilities payload; raises on failure."""
    rows: list[dict[str, Any]] = []
    for path in sorted(_catalog_dir().glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        file_rows = document.get("rows") if isinstance(document, dict) else None
        if isinstance(file_rows, list):
            rows.extend(row for row in file_rows if isinstance(row, dict))
    models: list[dict[str, Any]] = []
    default_model: str | None = None
    current_model: str | None = None
    provider_ids: list[str] = []
    for row in rows:
        if row.get("visibility") != "visible":
            continue
        model_id = row.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            continue
        tiers = _ordered_tiers(row.get("reasoning_effort_variants"))
        if not tiers:
            continue
        label = row.get("display_label") or model_id
        models.append(
            {
                "id": model_id,
                "label": label,
                "effortTiers": tiers,
                "defaultEffort": "high" if "high" in tiers else tiers[0],
            }
        )
        if row.get("is_default") and default_model is None:
            default_model = model_id
        if row.get("is_current") and current_model is None:
            current_model = model_id
        provider_id = row.get("provider_id")
        if isinstance(provider_id, str) and provider_id and provider_id not in provider_ids:
            provider_ids.append(provider_id)
    if not models:
        raise CliCapabilitiesUnavailable("no visible models in the local CLI model catalog")
    # echo/local are accepted by launch validation for smoke/tests but are
    # not real-model providers, so they are never advertised here.
    provider_ids = [pid for pid in provider_ids if pid not in {"echo", "local"}]
    if not provider_ids:
        raise CliCapabilitiesUnavailable("no real-model provider in the local CLI model catalog")
    chosen_default = default_model or current_model or models[0]["id"]
    return {
        "models": models,
        "defaultModel": chosen_default,
        "providers": [{"id": pid, "label": _provider_label(pid)} for pid in provider_ids],
        "defaultProvider": provider_ids[0],
    }


def get_cli_capabilities() -> dict[str, Any]:
    """Return the cached payload; on probe failure serve last-good, else raise."""
    global _cache_payload, _cache_expires_at, _cache_last_good
    now = time.monotonic()
    if _cache_payload is not None and now < _cache_expires_at:
        return _cache_payload
    try:
        payload = _probe_cli_capabilities()
    except Exception as exc:
        if _cache_last_good is not None:
            return _cache_last_good
        raise CliCapabilitiesUnavailable(str(exc)) from exc
    _cache_payload = payload
    _cache_last_good = payload
    _cache_expires_at = now + _CLI_CAPABILITIES_TTL_SECONDS
    return payload

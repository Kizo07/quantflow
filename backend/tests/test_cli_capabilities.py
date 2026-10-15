"""Backend tests: GET /api/cli-capabilities shape, auth, cache, fallback."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from app.gateway import cli_capabilities
from app.gateway.auth_disabled import AUTH_SOURCE_SESSION
from app.gateway.authz import AuthContext
from app.gateway.routers import cli_sessions

RUN_PERMISSIONS = ["runs:read", "runs:create", "runs:cancel"]


def _catalog_row(model_id, *, visible=True, tiers=("high", "low", "max", "medium", "minimal", "xhigh"), default=False, current=False):
    return {
        "model_id": model_id,
        "display_label": model_id,
        "visibility": "visible" if visible else "hidden",
        "is_default": default,
        "is_current": current,
        "provider_id": "meta",
        "reasoning_effort_variants": [{"tier": tier, "description": None} for tier in tiers],
    }


def _write_catalog(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "catalog.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")


@pytest_asyncio.fixture
async def gateway(monkeypatch, tmp_path):
    cli_capabilities._reset_cli_capabilities_cache()
    catalog_dir = tmp_path / "model-catalog"
    _write_catalog(
        catalog_dir,
        [
            _catalog_row("muse-wonder-1.0", default=True),
            _catalog_row("muse-spark-1.3", current=True),
            _catalog_row("muse-spark-1.3-contributor", visible=False),
            _catalog_row("muse-image-1.0", visible=False, tiers=()),
        ],
    )
    monkeypatch.setattr(cli_capabilities, "_CLI_CATALOG_DIR_OVERRIDE", catalog_dir)
    app = FastAPI()
    app.include_router(cli_sessions.capabilities_router)

    @app.middleware("http")
    async def authenticate(request, call_next):
        mode = request.headers.get("x-test-auth", "ok")
        user = None if mode == "anonymous" else SimpleNamespace(id="user-1", system_role="user")
        permissions = [] if mode in ("anonymous", "denied") else list(RUN_PERMISSIONS)
        request.state.user = user
        request.state.auth_source = AUTH_SOURCE_SESSION
        request.state.auth = AuthContext(user=user, permissions=permissions)
        return await call_next(request)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield SimpleNamespace(client=client, catalog_dir=catalog_dir)
    cli_capabilities._reset_cli_capabilities_cache()


@pytest.mark.asyncio
async def test_capabilities_shape(gateway):
    response = await gateway.client.get("/api/cli-capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"models", "defaultModel", "providers", "defaultProvider"}
    assert [model["id"] for model in payload["models"]] == ["muse-wonder-1.0", "muse-spark-1.3"]
    first = payload["models"][0]
    assert first["label"] == "muse-wonder-1.0"
    assert first["effortTiers"] == ["minimal", "low", "medium", "high", "xhigh", "max"]
    assert first["defaultEffort"] == "high"
    assert payload["defaultModel"] == "muse-wonder-1.0"
    assert payload["providers"] == [{"id": "meta", "label": "Meta"}]
    assert payload["defaultProvider"] == "meta"


@pytest.mark.asyncio
async def test_capabilities_unauthenticated_returns_401(gateway):
    response = await gateway.client.get("/api/cli-capabilities", headers={"x-test-auth": "anonymous"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_capabilities_cached_within_ttl(gateway):
    first = await gateway.client.get("/api/cli-capabilities")
    assert first.status_code == 200
    # Rewrite the catalog: a cached payload must not change within the TTL.
    _write_catalog(gateway.catalog_dir, [_catalog_row("muse-spark-9.9", default=True)])
    second = await gateway.client.get("/api/cli-capabilities")
    assert second.status_code == 200
    assert second.json() == first.json()


@pytest.mark.asyncio
async def test_capabilities_probe_failure_serves_last_good(gateway, monkeypatch):
    first = await gateway.client.get("/api/cli-capabilities")
    assert first.status_code == 200
    monkeypatch.setattr(cli_capabilities, "_CLI_CAPABILITIES_TTL_SECONDS", 0.0)
    for path in gateway.catalog_dir.glob("*.json"):
        path.unlink()
    response = await gateway.client.get("/api/cli-capabilities")
    assert response.status_code == 200
    assert response.json() == first.json()


@pytest.mark.asyncio
async def test_capabilities_no_cache_and_no_catalog_returns_503(gateway, monkeypatch):
    cli_capabilities._reset_cli_capabilities_cache()
    monkeypatch.setattr(cli_capabilities, "_CLI_CATALOG_DIR_OVERRIDE", gateway.catalog_dir / "missing")
    response = await gateway.client.get("/api/cli-capabilities")
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_capabilities_empty_visible_models_returns_503(gateway):
    cli_capabilities._reset_cli_capabilities_cache()
    _write_catalog(gateway.catalog_dir, [_catalog_row("muse-image-1.0", visible=False, tiers=())])
    response = await gateway.client.get("/api/cli-capabilities")
    assert response.status_code == 503

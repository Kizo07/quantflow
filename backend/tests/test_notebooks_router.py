"""Notebook connector router tests: token, health, and the httpx relay.

The sidecar is faked in-process (ASGI transport) — no sockets. Auth is a
dependency override; the middleware carve-out itself is covered in
``test_notebooks_auth_middleware.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from app.gateway.auth.sidecar_token import (
    SIDECAR_TOKEN_QUERY_PARAM,
    mint_sidecar_token,
    set_sidecar_secret,
    validate_sidecar_token,
)
from app.gateway.auth_disabled import AUTH_SOURCE_SESSION, AUTH_SOURCE_SIDECAR
from app.gateway.deps import get_current_user_from_request
from app.gateway.routers import notebooks

_TEST_SECRET = "test-sidecar-secret-not-for-production"
_USER_ID = str(uuid4())


def _fake_sidecar_app() -> FastAPI:
    sidecar = FastAPI()

    @sidecar.get("/health")
    async def _health():
        return {"component_version": "9.9.9-test", "contract_version": 1}

    @sidecar.get("/events")
    async def _events():
        async def _frames():
            yield b"event: cell\ndata: one\n\n"
            yield b"event: cell\ndata: two\n\n"

        return StreamingResponse(_frames(), media_type="text/event-stream")

    @sidecar.post("/echo")
    async def _echo(request: Request):
        body = await request.body()
        return JSONResponse({"bytes": len(body), "body": body.decode("utf-8")})

    @sidecar.get("/inspect")
    async def _inspect(request: Request):
        return JSONResponse(
            {
                "authorization": request.headers.get("authorization"),
                "cookie": request.headers.get("cookie"),
                "origin": request.headers.get("origin"),
                "connection": request.headers.get("connection"),
                "query": dict(request.query_params),
            }
        )

    @sidecar.get("/boom")
    async def _boom():
        return JSONResponse(status_code=500, content={"detail": "sidecar exploded"})

    async def _asset(request: Request):
        return JSONResponse({"authorization": request.headers.get("authorization")})

    # D1 bundle: the sidecar exempts these from per-request auth too.
    sidecar.add_api_route("/", _asset, methods=["GET"])
    for _name in ("app.js", "bridge.js", "sanitize.js", "styles.css"):
        sidecar.add_api_route(f"/{_name}", _asset, methods=["GET"])

    return sidecar


class _StubNotebookAuthMiddleware(BaseHTTPMiddleware):
    """Mirror of the AuthMiddleware contract for relay tests.

    A request carrying a valid sidecar credential stamps the sidecar source
    (the relay's verbatim-reuse path); anything else stamps the session user
    (the fresh-mint path). The carve-out boundary itself is pinned in
    ``test_notebooks_auth_middleware.py``.
    """

    async def dispatch(self, request, call_next):
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        payload = validate_sidecar_token(token) if token else None
        if payload is not None:
            request.state.auth_source = AUTH_SOURCE_SIDECAR
            request.state.user = SimpleNamespace(id=payload.sub, email="nb@example.com")
        else:
            request.state.auth_source = AUTH_SOURCE_SESSION
            request.state.user = SimpleNamespace(id=_USER_ID, email="nb@example.com")
        return await call_next(request)


@pytest.fixture()
def sidecar_client_factory():
    transport = httpx.ASGITransport(app=_fake_sidecar_app())

    def _factory(timeout=None):
        return httpx.AsyncClient(transport=transport, timeout=timeout)

    return _factory


@pytest.fixture(autouse=True)
def _isolated_sidecar_secret(monkeypatch):
    monkeypatch.setenv("NOTEBOOKS_SIDECAR_JWT_SECRET", _TEST_SECRET)
    set_sidecar_secret(None)
    yield
    set_sidecar_secret(None)


@pytest.fixture()
def app(monkeypatch, sidecar_client_factory):
    monkeypatch.setattr(notebooks, "resolve_sidecar_base_url", lambda: "http://sidecar.test")
    monkeypatch.setattr(notebooks, "build_sidecar_client", sidecar_client_factory)

    application = FastAPI()
    application.add_middleware(_StubNotebookAuthMiddleware)

    async def _fake_user():
        return SimpleNamespace(id=_USER_ID, email="nb@example.com")

    application.dependency_overrides[get_current_user_from_request] = _fake_user
    application.include_router(notebooks.router)
    return application


@pytest.fixture()
def bare_app(monkeypatch, sidecar_client_factory):
    """Router with NO auth middleware: proves fail-closed behavior when the
    auth stamp is absent entirely (never AttributeError/500)."""
    monkeypatch.setattr(notebooks, "resolve_sidecar_base_url", lambda: "http://sidecar.test")
    monkeypatch.setattr(notebooks, "build_sidecar_client", sidecar_client_factory)

    application = FastAPI()

    async def _fake_user():
        return SimpleNamespace(id=_USER_ID, email="nb@example.com")

    application.dependency_overrides[get_current_user_from_request] = _fake_user
    application.include_router(notebooks.router)
    return application


@pytest.fixture()
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def test_token_mints_valid_sidecar_jwt_for_caller(client):
    response = client.get("/api/notebooks/token")
    assert response.status_code == 200
    payload = response.json()
    assert payload["token_type"] == "Bearer"
    assert payload["expires_at"]
    validated = validate_sidecar_token(payload["token"])
    assert validated is not None
    assert validated.sub == _USER_ID


def test_health_aggregates_sidecar_versions(client):
    response = client.get("/api/notebooks/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["bridge_contract_version"] == notebooks.NOTEBOOK_BRIDGE_CONTRACT_VERSION == 1
    assert payload["sidecar"] == {
        "reachable": True,
        "component_version": "9.9.9-test",
        "contract_version": 1,
    }


def test_health_reports_unreachable_when_sidecar_is_down(monkeypatch, app):
    def _raising_factory(timeout=None):
        def _raise(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        return httpx.AsyncClient(transport=httpx.MockTransport(_raise))

    monkeypatch.setattr(notebooks, "build_sidecar_client", _raising_factory)
    with TestClient(app) as test_client:
        response = test_client.get("/api/notebooks/health")
    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "unreachable"
    assert payload["sidecar"] is None
    assert payload["bridge_contract_version"] == 1


def test_health_reports_unreachable_when_portfile_is_missing(monkeypatch, app):
    def _missing() -> str:
        raise RuntimeError("not running")

    monkeypatch.setattr(notebooks, "resolve_sidecar_base_url", _missing)
    with TestClient(app) as test_client:
        response = test_client.get("/api/notebooks/health")
    assert response.status_code == 503
    assert response.json()["status"] == "unreachable"


def test_relay_streams_sse_unbuffered_with_sse_headers(client):
    with client.stream("GET", "/api/notebooks/events") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert response.headers["Cache-Control"] == "no-cache, no-transform"
        assert response.headers["X-Accel-Buffering"] == "no"
        body = b"".join(response.iter_bytes())
    assert body == b"event: cell\ndata: one\n\n" + b"event: cell\ndata: two\n\n"


def test_relay_streams_request_body_for_post(client):
    response = client.post("/api/notebooks/echo", content=b"print(1 + 1)")
    assert response.status_code == 200
    assert response.json() == {"bytes": 12, "body": "print(1 + 1)"}


def test_relay_passes_status_codes_through(client):
    response = client.get("/api/notebooks/boom")
    assert response.status_code == 500
    assert response.json() == {"detail": "sidecar exploded"}


def test_relay_forwards_bearer_strips_cookie_and_hop_by_hop(client):
    token = mint_sidecar_token(_USER_ID)
    response = client.get(
        "/api/notebooks/inspect?notebook=nb1",
        headers={
            "Authorization": f"Bearer {token}",
            "Cookie": "access_token=session-cookie-must-not-cross",
            "Origin": "https://app.example",
            # Distinctive value: httpx adds its own `connection: keep-alive`
            # upstream, so assert the incoming value does not pass through.
            "Connection": "close",
        },
    )
    assert response.status_code == 200
    seen = response.json()
    # Already-validated incoming credentials are reused verbatim upstream.
    assert seen["authorization"] == f"Bearer {token}"
    assert seen["cookie"] is None
    assert seen["origin"] is None
    assert seen["connection"] != "close"
    assert seen["query"] == {"notebook": "nb1"}


def test_relay_mints_fresh_upstream_token_for_session_requests(client):
    """Cookie-authed direct access (no incoming credential) mints upstream."""
    response = client.get("/api/notebooks/inspect")
    assert response.status_code == 200
    seen = response.json()
    assert seen["authorization"] is not None
    assert seen["authorization"].startswith("Bearer ")
    validated = validate_sidecar_token(seen["authorization"].removeprefix("Bearer "))
    assert validated is not None
    assert validated.sub == _USER_ID
    assert seen["cookie"] is None


def test_relay_preserves_notebook_selector_with_encoded_slash(client):
    """?notebook=<id> keeps its exact shape (incl. /) across the relay."""
    response = client.get("/api/notebooks/inspect", params={"notebook": "proj/nb-1"})
    assert response.status_code == 200
    assert response.json()["query"] == {"notebook": "proj/nb-1"}


def test_relay_reuses_validated_incoming_bearer():
    """When the middleware already validated a sidecar Bearer, reuse it verbatim."""
    from starlette.requests import Request as StarletteRequest

    from app.gateway.auth_disabled import AUTH_SOURCE_SIDECAR

    token = mint_sidecar_token(_USER_ID)
    scope = {
        "type": "http",
        "method": "GET",
        "headers": [(b"authorization", f"Bearer {token}".encode())],
    }
    request = StarletteRequest(scope)
    request.state.auth_source = AUTH_SOURCE_SIDECAR
    assert notebooks._upstream_credential(request) == f"Bearer {token}"


def test_relay_mints_fresh_token_for_cookie_authed_requests():
    from starlette.requests import Request as StarletteRequest

    from app.gateway.auth_disabled import AUTH_SOURCE_SESSION

    scope = {"type": "http", "method": "GET", "headers": []}
    request = StarletteRequest(scope)
    request.state.auth_source = AUTH_SOURCE_SESSION
    request.state.user = SimpleNamespace(id=_USER_ID)
    credential = notebooks._upstream_credential(request)
    assert credential.startswith("Bearer ")
    validated = validate_sidecar_token(credential.removeprefix("Bearer "))
    assert validated is not None
    assert validated.sub == _USER_ID


def test_relay_strips_sidecar_token_query_param(client):
    token = mint_sidecar_token(_USER_ID)
    response = client.get(f"/api/notebooks/inspect?{SIDECAR_TOKEN_QUERY_PARAM}={token}&notebook=nb1")
    assert response.status_code == 200
    seen = response.json()
    assert SIDECAR_TOKEN_QUERY_PARAM not in seen["query"]
    assert seen["query"] == {"notebook": "nb1"}


def test_relay_returns_503_when_sidecar_is_down(monkeypatch, app):
    def _raising_factory(timeout=None):
        def _raise(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        return httpx.AsyncClient(transport=httpx.MockTransport(_raise))

    monkeypatch.setattr(notebooks, "build_sidecar_client", _raising_factory)
    with TestClient(app) as test_client:
        response = test_client.get("/api/notebooks/events")
    assert response.status_code == 503


def test_relay_serves_public_assets_without_any_auth_stamp(bare_app):
    """D1: the static UI bundle relays with no Authorization and no identity."""
    with TestClient(bare_app) as test_client:
        for asset in ("", "app.js", "bridge.js", "sanitize.js", "styles.css"):
            response = test_client.get(f"/api/notebooks/{asset}")
            assert response.status_code == 200, asset
            assert response.json()["authorization"] is None


def test_relay_asset_with_token_still_forwards_unauthenticated(client):
    """D1: even when the document carries ?sidecar_token=, assets go up bare."""
    token = mint_sidecar_token(_USER_ID)
    response = client.get(f"/api/notebooks/app.js?{SIDECAR_TOKEN_QUERY_PARAM}={token}")
    assert response.status_code == 200
    assert response.json()["authorization"] is None


def test_relay_rejects_dot_segments_before_upstream_contact(monkeypatch, app):
    """D2: traversal attempts 400 even when the sidecar is entirely missing."""
    monkeypatch.setattr(notebooks, "resolve_sidecar_base_url", _missing_sidecar)
    with TestClient(app) as test_client:
        for path in ("/api/notebooks/%2E%2E/admin", "/api/notebooks/a/%2E%2E/b"):
            response = test_client.get(path)
            assert response.status_code == 400, path
            assert response.json() == {"detail": "Invalid notebook relay path"}


def _missing_sidecar() -> str:
    raise RuntimeError("not running")


@pytest.mark.asyncio()
async def test_relay_rejects_dot_segments_direct_call(monkeypatch):
    """Deterministic containment pin: no routing/transport decoding involved."""
    from starlette.requests import Request as StarletteRequest

    monkeypatch.setattr(notebooks, "resolve_sidecar_base_url", _missing_sidecar)
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/notebooks/../admin",
        "raw_path": b"/api/notebooks/../admin",
        "query_string": b"",
        "headers": [],
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
    }

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = StarletteRequest(scope, _receive)
    response = await notebooks.relay_to_sidecar("../admin", request)
    assert response.status_code == 400


def test_contained_proxy_path_pins_dot_segment_rule():
    assert notebooks._is_contained_proxy_path("api/cells") is True
    assert notebooks._is_contained_proxy_path("") is True
    assert notebooks._is_contained_proxy_path("app.js") is True
    assert notebooks._is_contained_proxy_path("..") is False
    assert notebooks._is_contained_proxy_path("../admin") is False
    assert notebooks._is_contained_proxy_path("a/../b") is False
    assert notebooks._is_contained_proxy_path("a/./b") is False


def test_relay_fails_closed_without_auth_stamp(bare_app):
    """No middleware stamp on a non-asset path → 401, never AttributeError."""
    with TestClient(bare_app) as test_client:
        response = test_client.get("/api/notebooks/inspect")
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}


def test_upstream_credential_fails_closed_without_user():
    from starlette.requests import Request as StarletteRequest

    scope = {"type": "http", "method": "GET", "headers": []}
    request = StarletteRequest(scope)
    with pytest.raises(notebooks.UpstreamCredentialError):
        notebooks._upstream_credential(request)


def test_upstream_credential_mints_for_query_param_gets_without_header():
    """Sidecar source + no header (query-param GET) mints for the resolved user."""
    from starlette.requests import Request as StarletteRequest

    scope = {"type": "http", "method": "GET", "headers": []}
    request = StarletteRequest(scope)
    request.state.auth_source = AUTH_SOURCE_SIDECAR
    request.state.user = SimpleNamespace(id=_USER_ID)
    credential = notebooks._upstream_credential(request)
    assert credential.startswith("Bearer ")
    validated = validate_sidecar_token(credential.removeprefix("Bearer "))
    assert validated is not None
    assert validated.sub == _USER_ID


def test_health_reports_unreachable_on_foreign_shape(monkeypatch, app):
    """A sidecar health payload that is not a JSON object → 503, not 500."""

    def _foreign_factory(timeout=None):
        def _respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=["not", "a", "dict"])

        return httpx.AsyncClient(transport=httpx.MockTransport(_respond))

    monkeypatch.setattr(notebooks, "build_sidecar_client", _foreign_factory)
    with TestClient(app) as test_client:
        response = test_client.get("/api/notebooks/health")
    assert response.status_code == 503
    assert response.json()["status"] == "unreachable"


def test_health_reports_unreachable_on_non_200_sidecar(monkeypatch, app):
    def _sick_factory(timeout=None):
        def _respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"detail": "sick"})

        return httpx.AsyncClient(transport=httpx.MockTransport(_respond))

    monkeypatch.setattr(notebooks, "build_sidecar_client", _sick_factory)
    with TestClient(app) as test_client:
        response = test_client.get("/api/notebooks/health")
    assert response.status_code == 503
    assert "500" in response.json()["hint"]


def test_health_hint_names_shared_path_requirement(monkeypatch, app):
    """Portfile-missing 503s hint at DEER_FLOW_HOME / REPORTFORGE_SIDECAR_URL."""
    monkeypatch.setattr(notebooks, "resolve_sidecar_base_url", _missing_sidecar)
    with TestClient(app) as test_client:
        response = test_client.get("/api/notebooks/health")
    assert response.status_code == 503
    hint = response.json()["hint"]
    assert "DEER_FLOW_HOME" in hint
    assert "REPORTFORGE_SIDECAR_URL" in hint


def test_resolve_sidecar_base_url_explicit_env_wins(monkeypatch):
    monkeypatch.setenv(notebooks.SIDECAR_URL_ENV_VAR, "http://127.0.0.1:9999/")
    assert notebooks.SIDECAR_URL_ENV_VAR == "REPORTFORGE_SIDECAR_URL"
    assert notebooks.resolve_sidecar_base_url() == "http://127.0.0.1:9999"


def test_resolve_sidecar_base_url_shared_portfile_contract(monkeypatch, tmp_path):
    """Pin the shared-path contract: <base_dir>/reportforge-sidecar/port."""
    from deerflow.config import paths as paths_module

    assert notebooks.SIDECAR_PORTFILE_DIRNAME == "reportforge-sidecar"
    assert notebooks.SIDECAR_PORTFILE_NAME == "port"
    monkeypatch.delenv(notebooks.SIDECAR_URL_ENV_VAR, raising=False)
    monkeypatch.setattr(paths_module, "get_paths", lambda: SimpleNamespace(base_dir=tmp_path))

    portfile = tmp_path / "reportforge-sidecar" / "port"
    portfile.parent.mkdir(parents=True)
    portfile.write_text("8123\n", encoding="utf-8")
    assert notebooks.resolve_sidecar_base_url() == "http://127.0.0.1:8123"

    portfile.write_text("http://127.0.0.1:5555/", encoding="utf-8")
    assert notebooks.resolve_sidecar_base_url() == "http://127.0.0.1:5555"


def test_resolve_sidecar_base_url_missing_or_empty_portfile_fails(monkeypatch, tmp_path):
    from deerflow.config import paths as paths_module

    monkeypatch.delenv(notebooks.SIDECAR_URL_ENV_VAR, raising=False)
    monkeypatch.setattr(paths_module, "get_paths", lambda: SimpleNamespace(base_dir=tmp_path))
    with pytest.raises(RuntimeError):
        notebooks.resolve_sidecar_base_url()

    portfile = tmp_path / "reportforge-sidecar" / "port"
    portfile.parent.mkdir(parents=True)
    portfile.write_text("  \n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        notebooks.resolve_sidecar_base_url()

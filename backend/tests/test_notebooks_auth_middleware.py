"""Sidecar-token middleware carve-out tests (Phase 3 connector auth).

Builds a real ``AuthMiddleware`` app (per ``_router_auth_helpers``: auth
boundary tests must not use the permissive stub) with a fake user
provider. Covers the carve-out branch before the PAT branch: Bearer and
GET query-param credentials, /token + /health exclusion, disabled and
unknown users, and the strict no-cookie-fallback rule.
"""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.gateway.auth.jwt import create_access_token
from app.gateway.auth.models import User
from app.gateway.auth.sidecar_token import mint_sidecar_token, set_sidecar_secret
from app.gateway.auth_middleware import AuthMiddleware

_TEST_SECRET = "test-sidecar-secret-not-for-production"


def _identity(request: Request) -> dict[str, str]:
    return {
        "auth_source": request.state.auth_source,
        "user_id": str(request.state.user.id),
    }


def _asset_marker(request: Request) -> dict[str, object]:
    """D1 assets carry no auth stamp: the middleware lets them through bare."""
    return {
        "asset": True,
        "auth_source": getattr(request.state, "auth_source", None),
        "has_user": hasattr(request.state, "user"),
    }


class _FakeProvider:
    def __init__(self, users: list[User]) -> None:
        self._users = {str(u.id): u for u in users}

    async def get_user(self, user_id: str) -> User | None:
        return self._users.get(str(user_id))


@pytest.fixture()
def active_user() -> User:
    return User(id=uuid4(), email="nb@example.com", system_role="user")


@pytest.fixture()
def app(monkeypatch, active_user):
    monkeypatch.setenv("NOTEBOOKS_SIDECAR_JWT_SECRET", _TEST_SECRET)
    monkeypatch.setenv("AUTH_JWT_SECRET", "test-session-secret-not-for-production")
    set_sidecar_secret(None)

    disabled = User(id=uuid4(), email="disabled@example.com", system_role="user", disabled=True)
    provider = _FakeProvider([active_user, disabled])
    monkeypatch.setattr("app.gateway.deps.get_local_provider", lambda: provider)
    monkeypatch.setattr("app.gateway.auth_middleware.is_auth_disabled", lambda: False)

    application = FastAPI()
    application.add_middleware(AuthMiddleware)

    @application.get("/api/notebooks/token")
    async def _token(request: Request):
        return _identity(request)

    @application.get("/api/notebooks/health")
    async def _health(request: Request):
        return _identity(request)

    # D1 asset routes BEFORE the catch-all (Starlette matches in order).
    async def _asset_handler(request: Request):
        return _asset_marker(request)

    for _asset in ("", "app.js", "bridge.js", "sanitize.js", "styles.css"):
        application.add_api_route(f"/api/notebooks/{_asset}", _asset_handler, methods=["GET"])

    @application.api_route("/api/notebooks/{relay_path:path}", methods=["GET", "POST"])
    async def _relay(request: Request):
        return _identity(request)

    @application.get("/api/models")
    async def _models(request: Request):
        return _identity(request)

    application.state.disabled_user_id = str(disabled.id)
    yield application
    set_sidecar_secret(None)


@pytest.fixture()
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def test_relay_bearer_authenticates_with_sidecar_source(client, active_user):
    response = client.get(
        "/api/notebooks/some/notebook",
        headers={"Authorization": f"Bearer {mint_sidecar_token(str(active_user.id))}"},
    )
    assert response.status_code == 200
    assert response.json() == {"auth_source": "sidecar_token", "user_id": str(active_user.id)}


def test_relay_bearer_is_required_per_request(client):
    """No credential at all falls through to the cookie path → 401."""
    response = client.get("/api/notebooks/some/notebook")
    assert response.status_code == 401


def test_present_but_invalid_bearer_never_falls_back_to_cookie(client):
    response = client.get(
        "/api/notebooks/some/notebook",
        headers={"Authorization": "Bearer garbage-token"},
        cookies={"access_token": "whatever-the-cookie-holds"},
    )
    assert response.status_code == 401


def test_non_bearer_authorization_is_rejected(client):
    response = client.get(
        "/api/notebooks/some/notebook",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
    )
    assert response.status_code == 401


def test_expired_sidecar_token_is_rejected(client, active_user):
    token = mint_sidecar_token(str(active_user.id), ttl=timedelta(seconds=-1))
    response = client.get(
        "/api/notebooks/some/notebook",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_session_jwt_as_relay_bearer_is_rejected(client, active_user):
    """Session tokens carry no aud — they must not pass the carve-out."""
    session_token = create_access_token(str(active_user.id))
    response = client.get(
        "/api/notebooks/some/notebook",
        headers={"Authorization": f"Bearer {session_token}"},
    )
    assert response.status_code == 401


def test_unknown_user_is_rejected(client):
    response = client.get(
        "/api/notebooks/some/notebook",
        headers={"Authorization": f"Bearer {mint_sidecar_token(str(uuid4()))}"},
    )
    assert response.status_code == 401


def test_disabled_user_is_rejected(client, app):
    response = client.get(
        "/api/notebooks/some/notebook",
        headers={"Authorization": f"Bearer {mint_sidecar_token(app.state.disabled_user_id)}"},
    )
    assert response.status_code == 401


def test_token_endpoint_is_excluded_from_carve_out(client, active_user):
    """A sidecar Bearer on /token is not relay auth (falls to PAT → 401)."""
    response = client.get(
        "/api/notebooks/token",
        headers={"Authorization": f"Bearer {mint_sidecar_token(str(active_user.id))}"},
    )
    assert response.status_code == 401


def test_health_endpoint_is_excluded_from_carve_out(client, active_user):
    response = client.get(
        "/api/notebooks/health",
        headers={"Authorization": f"Bearer {mint_sidecar_token(str(active_user.id))}"},
    )
    assert response.status_code == 401


def test_sidecar_bearer_is_route_bound_to_notebooks(client, active_user):
    """A sidecar Bearer on any other route is not valid auth (PAT → 401)."""
    response = client.get(
        "/api/models",
        headers={"Authorization": f"Bearer {mint_sidecar_token(str(active_user.id))}"},
    )
    assert response.status_code == 401


def test_get_accepts_sidecar_token_query_param(client, active_user):
    """Relay GETs that cannot set headers authenticate by query."""
    response = client.get(f"/api/notebooks/api/cells?sidecar_token={mint_sidecar_token(str(active_user.id))}")
    assert response.status_code == 200
    assert response.json() == {"auth_source": "sidecar_token", "user_id": str(active_user.id)}


def test_invalid_query_token_never_falls_back_to_cookie(client):
    response = client.get(
        "/api/notebooks/api/cells?sidecar_token=garbage",
        cookies={"access_token": "whatever-the-cookie-holds"},
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}


def test_query_token_is_rejected_for_mutations(client, active_user):
    """State-changing relay requests must carry the header, never the query."""
    response = client.post(f"/api/notebooks/api/cells?sidecar_token={mint_sidecar_token(str(active_user.id))}")
    assert response.status_code == 401


def test_empty_query_token_is_present_but_invalid_no_cookie_fallback(client):
    """``?sidecar_token=`` (empty) triggers the carve-out and hard-401s.

    The ``Invalid token`` detail proves the carve-out answered — the cookie
    path would return a structured ``AuthErrorResponse`` detail instead.
    """
    response = client.get(
        "/api/notebooks/api/cells?sidecar_token=",
        cookies={"access_token": "whatever-the-cookie-holds"},
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}


def test_duplicated_query_token_is_present_but_invalid_no_cookie_fallback(client, active_user):
    """Two ``?sidecar_token=`` values are ambiguous → hard 401, no fallback."""
    first = mint_sidecar_token(str(active_user.id))
    second = mint_sidecar_token(str(active_user.id))
    response = client.get(
        f"/api/notebooks/api/cells?sidecar_token={first}&sidecar_token={second}",
        cookies={"access_token": "whatever-the-cookie-holds"},
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}


def test_empty_query_token_is_rejected_for_mutations_without_cookie_fallback(client):
    response = client.post(
        "/api/notebooks/api/cells?sidecar_token=",
        cookies={"access_token": "whatever-the-cookie-holds"},
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}


def test_absent_query_token_still_falls_through_to_cookie_path(client):
    """Control: with no query key at all, the cookie path answers (garbage
    cookie → structured auth-error detail, not the carve-out verdict)."""
    response = client.get(
        "/api/notebooks/api/cells",
        cookies={"access_token": "whatever-the-cookie-holds"},
    )
    assert response.status_code == 401
    assert response.json() != {"detail": "Invalid token"}


def test_public_asset_set_loads_without_any_credential(client):
    """D1: the full static UI bundle loads with no credential at all."""
    for asset in ("", "app.js", "bridge.js", "sanitize.js", "styles.css"):
        response = client.get(f"/api/notebooks/{asset}")
        assert response.status_code == 200, asset
        assert response.json() == {"asset": True, "auth_source": None, "has_user": False}
    # The bare prefix (no trailing slash) is the same document.
    assert client.get("/api/notebooks").status_code == 200


def test_public_asset_ignores_present_but_invalid_credentials(client):
    """D1: credentials are not consulted on exempt paths — garbage cannot 401."""
    response = client.get("/api/notebooks/app.js?sidecar_token=garbage")
    assert response.status_code == 200


def test_public_asset_exemption_is_exact_match_only(client):
    """Near-miss paths stay authenticated: no prefix/traversal/suffix slip."""
    for near_miss in ("/api/notebooks/app.js.map", "/api/notebooks/app.js/"):
        response = client.get(near_miss)
        assert response.status_code == 401, near_miss
    # A query string on an exempt path is still exempt (matching is on path).
    assert client.get("/api/notebooks/styles.css?v=2").status_code == 200


def test_document_load_with_token_still_succeeds(client, active_user):
    """The frame mounts the document with ?sidecar_token=; assets need nothing."""
    response = client.get(f"/api/notebooks/?sidecar_token={mint_sidecar_token(str(active_user.id))}&notebook=nb-1")
    assert response.status_code == 200


def test_bearer_wins_over_query_param(client, active_user):
    """A present Authorization header decides; the query param is ignored."""
    response = client.get(
        f"/api/notebooks/api/cells?sidecar_token={mint_sidecar_token(str(active_user.id))}",
        headers={"Authorization": "Bearer garbage-token"},
    )
    assert response.status_code == 401

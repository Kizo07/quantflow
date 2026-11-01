"""Sidecar-token contract tests (Phase 3 connector auth).

Pins the cross-repo contract ReportForge re-implements (no shared
package): HS256 JWT, ``{sub: user_id, aud: "reportforge-sidecar"}``,
5-minute TTL, dedicated secret that is NEVER the session secret.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import jwt
import pytest

from app.gateway.auth.errors import TokenError
from app.gateway.auth.jwt import create_access_token, decode_token
from app.gateway.auth.sidecar_token import (
    SIDECAR_AUDIENCE,
    SIDECAR_SECRET_ENV_VAR,
    SIDECAR_TOKEN_TTL,
    get_sidecar_secret,
    mint_sidecar_token,
    set_sidecar_secret,
    validate_sidecar_token,
)

_TEST_SECRET = "test-sidecar-secret-not-for-production"


@pytest.fixture(autouse=True)
def _isolated_sidecar_secret(monkeypatch):
    monkeypatch.setenv(SIDECAR_SECRET_ENV_VAR, _TEST_SECRET)
    set_sidecar_secret(None)
    yield
    set_sidecar_secret(None)


def test_token_claims_contract_is_pinned():
    """sub/aud/exp/iat claims and the 5-minute TTL are exact."""
    assert SIDECAR_AUDIENCE == "reportforge-sidecar"
    assert SIDECAR_TOKEN_TTL == timedelta(minutes=5)

    user_id = str(uuid4())
    token = mint_sidecar_token(user_id)
    payload = validate_sidecar_token(token)
    assert payload is not None
    assert payload.sub == user_id
    assert payload.aud == "reportforge-sidecar"
    assert payload.exp - payload.iat == timedelta(minutes=5)


def test_validate_round_trip():
    payload = validate_sidecar_token(mint_sidecar_token("user-1"))
    assert payload is not None
    assert payload.sub == "user-1"


def test_validate_rejects_garbage():
    assert validate_sidecar_token("not-a-jwt") is None
    assert validate_sidecar_token("") is None


def test_validate_rejects_wrong_secret():
    token = mint_sidecar_token("user-1")
    set_sidecar_secret("a-different-secret")
    try:
        assert validate_sidecar_token(token) is None
    finally:
        set_sidecar_secret(_TEST_SECRET)


def test_validate_rejects_expired_token():
    token = mint_sidecar_token("user-1", ttl=timedelta(seconds=-1))
    assert validate_sidecar_token(token) is None


def test_validate_rejects_wrong_or_missing_audience():
    now = datetime.now(UTC)
    wrong_aud = jwt.encode(
        {"sub": "user-1", "aud": "some-other-audience", "exp": now + timedelta(minutes=5), "iat": now},
        _TEST_SECRET,
        algorithm="HS256",
    )
    assert validate_sidecar_token(wrong_aud) is None
    missing_aud = jwt.encode(
        {"sub": "user-1", "exp": now + timedelta(minutes=5), "iat": now},
        _TEST_SECRET,
        algorithm="HS256",
    )
    assert validate_sidecar_token(missing_aud) is None


def test_session_token_is_not_a_sidecar_token():
    """A session JWT (no aud) never validates as a sidecar token."""
    session_token = create_access_token(str(uuid4()))
    assert validate_sidecar_token(session_token) is None


def test_sidecar_token_is_not_a_session_token():
    """A sidecar JWT never decodes as a session token (secret separation).

    Session validation performs no ``aud`` check, so this holds only
    because the sidecar secret differs from the session secret — reusing
    one secret would let either token replay as the other.
    """
    result = decode_token(mint_sidecar_token(str(uuid4())))
    assert isinstance(result, TokenError)


def test_env_secret_wins_over_file(monkeypatch, tmp_path):
    """An explicit env secret is used without touching the secret file."""
    from deerflow.config import paths as paths_module

    def _boom():
        raise AssertionError("secret file must not be read when the env secret is set")

    monkeypatch.setattr(paths_module, "get_paths", _boom)
    assert get_sidecar_secret() == _TEST_SECRET


def test_secret_persists_to_file_beside_jwt_secret(monkeypatch, tmp_path):
    """Without env, the secret is file-persisted 0600 and stable across resets."""
    import os

    from app.gateway.auth.sidecar_token import SIDECAR_SECRET_FILE
    from deerflow.config import paths as paths_module

    monkeypatch.delenv(SIDECAR_SECRET_ENV_VAR, raising=False)
    monkeypatch.setattr(paths_module, "get_paths", lambda: SimpleNamespace(base_dir=tmp_path))
    set_sidecar_secret(None)
    try:
        first = get_sidecar_secret()
        secret_file = tmp_path / SIDECAR_SECRET_FILE
        assert secret_file.is_file()
        assert os.stat(secret_file).st_mode & 0o777 == 0o600
        assert secret_file.read_text(encoding="utf-8").strip() == first
        set_sidecar_secret(None)
        assert get_sidecar_secret() == first
    finally:
        set_sidecar_secret(None)

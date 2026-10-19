"""Tests for auth error types and typed decode_token."""

from datetime import UTC, datetime, timedelta

import jwt as pyjwt

from app.gateway.auth.config import AuthConfig, set_auth_config
from app.gateway.auth.errors import AuthErrorCode, AuthErrorResponse, TokenError
from app.gateway.auth.jwt import create_access_token, decode_token


def test_auth_error_code_values():
    assert AuthErrorCode.INVALID_CREDENTIALS == "invalid_credentials"
    assert AuthErrorCode.TOKEN_EXPIRED == "token_expired"
    assert AuthErrorCode.NOT_AUTHENTICATED == "not_authenticated"


def test_token_error_values():
    assert TokenError.EXPIRED == "expired"
    assert TokenError.INVALID_SIGNATURE == "invalid_signature"
    assert TokenError.MALFORMED == "malformed"


def test_auth_error_response_serialization():
    err = AuthErrorResponse(
        code=AuthErrorCode.TOKEN_EXPIRED,
        message="Token has expired",
    )
    d = err.model_dump()
    assert d == {"code": "token_expired", "message": "Token has expired"}


def test_auth_error_response_from_dict():
    d = {"code": "invalid_credentials", "message": "Wrong password"}
    err = AuthErrorResponse(**d)
    assert err.code == AuthErrorCode.INVALID_CREDENTIALS


# ── decode_token typed failure tests ──────────────────────────────

_TEST_SECRET = "test-secret-for-jwt-decode-token-tests"


def _setup_config():
    set_auth_config(AuthConfig(jwt_secret=_TEST_SECRET))


def test_decode_token_returns_token_error_on_expired():
    _setup_config()
    expired_payload = {"sub": "user-1", "exp": datetime.now(UTC) - timedelta(hours=1), "iat": datetime.now(UTC)}
    token = pyjwt.encode(expired_payload, _TEST_SECRET, algorithm="HS256")
    result = decode_token(token)
    assert result == TokenError.EXPIRED


def test_decode_token_returns_token_error_on_bad_signature():
    _setup_config()
    payload = {"sub": "user-1", "exp": datetime.now(UTC) + timedelta(hours=1), "iat": datetime.now(UTC)}
    token = pyjwt.encode(payload, "wrong-secret", algorithm="HS256")
    result = decode_token(token)
    assert result == TokenError.INVALID_SIGNATURE


def test_decode_token_returns_token_error_on_malformed():
    _setup_config()
    result = decode_token("not-a-jwt")
    assert result == TokenError.MALFORMED


def test_decode_token_returns_payload_on_valid():
    _setup_config()
    token = create_access_token("user-123")
    result = decode_token(token)
    assert not isinstance(result, TokenError)
    assert result.sub == "user-123"


# ── Concise email validation copy (V-7c) ────────────────────────────


def test_register_rejects_invalid_email_with_concise_copy():
    """Reserved/special-use domains fail with the short signup message."""
    import pytest
    from pydantic import ValidationError

    from app.gateway.routers.auth import RegisterRequest

    with pytest.raises(ValidationError) as exc_info:
        RegisterRequest(email="x@test.local", password="LongEnough123!", remember_me=True)
    error = exc_info.value.errors()[0]
    assert error["msg"] == "Enter a valid email address."
    # The frontend ErrorDetailSchema only accepts the value_error shape.
    assert error["type"] == "value_error"


def test_register_normalizes_valid_email_like_emailstr():
    """Valid addresses keep EmailStr normalization (domain lowercased)."""
    from app.gateway.routers.auth import RegisterRequest

    assert (
        RegisterRequest(email="User@Example.COM", password="LongEnough123!", remember_me=True).email
        == "User@example.com"
    )

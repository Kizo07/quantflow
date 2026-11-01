"""Short-lived sidecar tokens for the ReportForge notebook connector (Phase 3).

The notebook UI lives in the ReportForge repo and is served by
``reportforge serve`` (loopback-only HTTP). QuantFlow embeds it in a
``sandbox``-without-``allow-same-origin`` iframe, which sends no cookies,
so the gateway mints a short-lived JWT that the iframe delivers per
request and the sidecar validates per request on EVERY route.

Contract (pinned in ``backend/tests/test_notebooks_sidecar_token.py``;
ReportForge re-implements validation — there is no shared package):

- HS256 JWT, claims ``{sub: user_id, aud: "reportforge-sidecar"}``,
  5-minute TTL (``exp``/``iat``).
- Dedicated sidecar secret, file-persisted (0600) beside ``.jwt_secret``
  as ``.notebook_sidecar_secret`` so uvicorn workers converge on one
  value. NEVER the session secret: session tokens carry no ``aud``
  check, so reusing that secret would let a sidecar token replay as a
  session cookie (and vice versa).

TTL rationale: the token travels in the frame URL (``?sidecar_token=`` —
navigations and ``<script>``/``<link>`` loads cannot set headers), so its
exposure surface is browser history, shoulder-surfing, and any log line
that escapes the access-log query strip. Five minutes bounds that window
while staying far above the 60-second early-refresh margin and the
slowest iframe mount, so a leaked token is near-useless but legitimate
sessions never churn: the shell renews each token a full margin before
expiry. Shortening below ~2 minutes would risk refresh races on slow
mounts; lengthening past ~15 would widen the URL-leak window for no UX
gain. Gateway access logs omit query strings entirely (see
``AccessLogQueryStripFilter``), and the relay strips the param before
forwarding, so the URL is the token's only resting place.

The ReportForge sidecar must validate against the same secret value —
via the ``NOTEBOOKS_SIDECAR_JWT_SECRET`` env var or the shared secret
file — and enforce the ``aud`` claim.
"""

from __future__ import annotations

import logging
import os
import secrets
from datetime import UTC, datetime, timedelta

import jwt
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SIDECAR_AUDIENCE = "reportforge-sidecar"
SIDECAR_TOKEN_TTL = timedelta(minutes=5)
SIDECAR_SECRET_ENV_VAR = "NOTEBOOKS_SIDECAR_JWT_SECRET"
SIDECAR_SECRET_FILE = ".notebook_sidecar_secret"
# Query param carrying the sidecar JWT for GET document/asset loads from the
# sandboxed iframe (navigations and <script>/<link> fetches cannot set
# headers). Stripped before forwarding; never accepted for mutations.
SIDECAR_TOKEN_QUERY_PARAM = "sidecar_token"


class SidecarTokenPayload(BaseModel):
    """Validated sidecar-token claims."""

    sub: str  # user_id
    aud: str
    exp: datetime
    iat: datetime | None = None


_sidecar_secret: str | None = None


def _load_or_create_sidecar_secret() -> str:
    """Load the persisted sidecar secret, or generate and persist a new one.

    Mirrors ``app.gateway.auth.config._load_or_create_secret``: exclusive
    creation (0600) so processes sharing the runtime home converge on one
    secret instead of each keeping its own.
    """
    from deerflow.config.paths import get_paths
    from deerflow.config.secret_file import read_or_create_secret_file

    secret_file = get_paths().base_dir / SIDECAR_SECRET_FILE
    try:
        return read_or_create_secret_file(secret_file, lambda: secrets.token_urlsafe(32))
    except OSError as exc:
        raise RuntimeError(
            f"Failed to read or persist the notebook sidecar secret at {secret_file}. Set {SIDECAR_SECRET_ENV_VAR} explicitly or fix DEER_FLOW_HOME/base directory permissions so the gateway can keep a stable sidecar secret."
        ) from exc


def get_sidecar_secret() -> str:
    """Return the process-wide sidecar signing secret (env wins, else file)."""
    global _sidecar_secret
    if _sidecar_secret is None:
        env_secret = os.environ.get(SIDECAR_SECRET_ENV_VAR)
        if env_secret:
            _sidecar_secret = env_secret
        else:
            _sidecar_secret = _load_or_create_sidecar_secret()
            os.environ[SIDECAR_SECRET_ENV_VAR] = _sidecar_secret
            logger.warning(
                "⚠ %s is not set — using an auto-generated secret persisted to %s. The ReportForge sidecar must share this value to validate tokens.",
                SIDECAR_SECRET_ENV_VAR,
                SIDECAR_SECRET_FILE,
            )
    return _sidecar_secret


def set_sidecar_secret(secret: str | None) -> None:
    """Override the process-wide sidecar secret (for testing)."""
    global _sidecar_secret
    _sidecar_secret = secret


def mint_sidecar_token(user_id: str, *, ttl: timedelta = SIDECAR_TOKEN_TTL) -> str:
    """Mint a short-lived sidecar JWT for *user_id*."""
    now = datetime.now(UTC)
    payload = {
        "sub": user_id,
        "aud": SIDECAR_AUDIENCE,
        "exp": now + ttl,
        "iat": now,
    }
    return jwt.encode(payload, get_sidecar_secret(), algorithm="HS256")


def validate_sidecar_token(token: str) -> SidecarTokenPayload | None:
    """Validate a sidecar JWT; return its claims, or ``None`` when invalid.

    Every failure mode — wrong secret, expired, malformed, wrong/missing
    audience — collapses to ``None`` so callers cannot serve as an oracle.
    """
    try:
        payload = jwt.decode(
            token,
            get_sidecar_secret(),
            algorithms=["HS256"],
            audience=SIDECAR_AUDIENCE,
        )
        return SidecarTokenPayload(**payload)
    except jwt.PyJWTError:
        return None

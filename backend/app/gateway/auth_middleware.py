"""Global authentication middleware — fail-closed safety net.

Rejects unauthenticated requests to non-public paths with 401. When a
request passes the cookie check, resolves the JWT payload to a real
``User`` object and stamps it into both ``request.state.user`` and the
``deerflow.runtime.user_context`` contextvar so that repository-layer
owner filtering works automatically via the sentinel pattern.

Fine-grained permission checks remain in authz.py decorators.
"""

import logging
from collections.abc import Callable

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from app.gateway.auth.errors import AuthErrorCode, AuthErrorResponse
from app.gateway.auth_disabled import (
    AUTH_SOURCE_AUTH_DISABLED,
    AUTH_SOURCE_INTERNAL,
    AUTH_SOURCE_PAT,
    AUTH_SOURCE_SESSION,
    AUTH_SOURCE_SIDECAR,
    get_auth_disabled_user,
    is_auth_disabled,
)
from app.gateway.authz import AuthContext, resolve_route_permissions
from app.gateway.internal_auth import INTERNAL_AUTH_HEADER_NAME, get_internal_user, is_valid_internal_auth_token
from app.gateway.request_path import get_request_route_path
from deerflow.runtime.user_context import reset_current_user, set_current_user

logger = logging.getLogger(__name__)

# Paths that never require authentication.
_PUBLIC_PATH_PREFIXES: tuple[str, ...] = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1/auth/oauth/",
    "/api/v1/auth/callback/",
    # Inbound webhooks authenticate themselves via provider-specific signatures
    # (e.g. GitHub's X-Hub-Signature-256), not session cookies.
    "/api/webhooks/",
)

# Exact auth paths that are public (login/register/status check).
# /api/v1/auth/me, /api/v1/auth/change-password etc. are NOT public.
_PUBLIC_EXACT_PATHS: frozenset[str] = frozenset(
    {
        "/api/v1/auth/login/local",
        "/api/v1/auth/register",
        "/api/v1/auth/logout",
        "/api/v1/auth/setup-status",
        "/api/v1/auth/initialize",
        "/api/v1/auth/providers",
    }
)


def _is_public(path: str) -> bool:
    stripped = path.rstrip("/")
    if stripped in _PUBLIC_EXACT_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in _PUBLIC_PATH_PREFIXES)


# Gateway-owned connector endpoints stay on the session cookie; every other
# path under /api/notebooks/ is a sidecar relay path the sandboxed iframe
# (which sends no cookies) reaches with a sidecar token instead — except the
# static UI bundle below, which is EXEMPT from per-request auth (joint
# contract D1): the document shell plus its <script>/<link> subresources
# carry no notebook data (data loads via authenticated API calls), and the
# sandboxed frame cannot attach credentials to subresource loads anyway. The
# single ?sidecar_token= on the document navigation authenticates the mount;
# the assets load on that document token alone. ReportForge owns the files;
# the paths are pinned here so shell↔sidecar skew fails closed.
_NOTEBOOK_RELAY_PREFIX = "/api/notebooks/"
_NOTEBOOK_NON_RELAY_PATHS: frozenset[str] = frozenset(
    {
        "/api/notebooks/token",
        "/api/notebooks/health",
    }
)

# D1 public asset set. EXACT match only — no prefix matching, no trailing-
# slash normalization, no traversal resolution. Anything not in this set
# (e.g. /api/notebooks/app.js.map, /api/notebooks/../x) stays on the
# authenticated relay path.
_NOTEBOOK_PUBLIC_ASSET_PATHS: frozenset[str] = frozenset(
    {
        "/api/notebooks",
        "/api/notebooks/",
        "/api/notebooks/app.js",
        "/api/notebooks/bridge.js",
        "/api/notebooks/sanitize.js",
        "/api/notebooks/styles.css",
    }
)


def _is_notebook_public_asset(path: str) -> bool:
    """True only for the exact D1 asset paths (no normalization)."""
    return path in _NOTEBOOK_PUBLIC_ASSET_PATHS


def _is_notebook_relay_path(path: str) -> bool:
    if _is_notebook_public_asset(path):
        return False
    normalized = path.rstrip("/") or "/"
    if normalized in _NOTEBOOK_NON_RELAY_PATHS:
        return False
    return normalized == "/api/notebooks" or normalized.startswith(_NOTEBOOK_RELAY_PREFIX)


def _notebook_query_token_present(request: Request) -> bool:
    """True when the request carries the sidecar-token query key at all.

    Presence (not validity) drives the carve-out: an empty or duplicated
    ``?sidecar_token=`` is a present-but-invalid credential and must hard-401
    without falling back to the session cookie — same strictness as #4849.
    """
    from app.gateway.auth.sidecar_token import SIDECAR_TOKEN_QUERY_PARAM

    return SIDECAR_TOKEN_QUERY_PARAM in request.query_params


def _notebook_query_token(request: Request) -> str | None:
    from app.gateway.auth.sidecar_token import SIDECAR_TOKEN_QUERY_PARAM

    values = request.query_params.getlist(SIDECAR_TOKEN_QUERY_PARAM)
    if len(values) != 1 or not values[0]:
        return None
    return values[0]


class AuthMiddleware(BaseHTTPMiddleware):
    """Strict auth gate: reject requests without a valid session.

    Two-stage check for non-public paths:

    1. Cookie presence — return 401 NOT_AUTHENTICATED if missing
    2. JWT validation via ``get_optional_user_from_request`` — return 401
       TOKEN_INVALID if the token is absent, malformed, expired, or the
       signed user does not exist / is stale

    On success, stamps ``request.state.user`` and the
    ``deerflow.runtime.user_context`` contextvar so that repository-layer
    owner filters work downstream without every route needing a
    ``@require_auth`` decorator. Routes that need per-resource
    authorization (e.g. "user A cannot read user B's thread by guessing
    the URL") should additionally use ``@require_permission(...,
    owner_check=True)`` for explicit enforcement — but authentication
    itself is fully handled here.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        route_path = get_request_route_path(request)
        if _is_public(route_path):
            return await call_next(request)
        if _is_notebook_public_asset(route_path):
            # D1: the static UI bundle is public code — no credential is
            # required, and none is consulted (a garbage ?sidecar_token= on
            # an asset URL cannot 401 a load that needs no auth). The relay
            # forwards these upstream without an Authorization header.
            return await call_next(request)

        internal_user = None
        if is_valid_internal_auth_token(request.headers.get(INTERNAL_AUTH_HEADER_NAME)):
            # Extract the channel owner user ID from the trusted header.
            # When present, the synthetic internal user carries the actual
            # owner identity so that get_effective_user_id() and per-user
            # filesystem paths (custom skills, memory, thread data) resolve
            # to the IM channel user instead of falling back to "default".
            from app.gateway.internal_auth import INTERNAL_OWNER_USER_ID_HEADER_NAME

            owner_user_id = request.headers.get(INTERNAL_OWNER_USER_ID_HEADER_NAME)
            if owner_user_id:
                owner_user_id = owner_user_id.strip()
            internal_user = get_internal_user(owner_user_id=owner_user_id or None)
            if owner_user_id and not is_auth_disabled():
                # The internal principal is synthesized from trusted headers
                # without a users-row lookup, so suspension has no other chance
                # to bite on this surface (#3462 gap 3): enforce it here, the
                # one gate every owner-bound internal HTTP call (IM channel
                # dispatch above all) passes through. Auth-disabled mode is
                # exempt — its owner header carries the synthetic
                # auth-disabled identity and no suspendable account exists.
                # Owner ids without a users row stay allowed (a nonexistent
                # account cannot be suspended), and a store that cannot be
                # read cannot hold a suspension verdict either; run admission
                # re-asserts the check when the row is readable.
                from app.gateway.deps import get_local_provider

                try:
                    owner = await get_local_provider().get_user(owner_user_id)
                except Exception:
                    logger.warning("Internal-owner suspension check skipped: users lookup failed", exc_info=True)
                    owner = None
                if owner is not None and getattr(owner, "disabled", False):
                    return JSONResponse(
                        status_code=401,
                        content={
                            "detail": AuthErrorResponse(
                                code=AuthErrorCode.ACCOUNT_DISABLED,
                                message="Account disabled",
                            ).model_dump()
                        },
                    )

        auth_source = AUTH_SOURCE_SESSION
        access_token = request.cookies.get("access_token")
        authorization = request.headers.get("authorization")
        pat_scopes: frozenset[str] = frozenset()

        # Non-public path: require session cookie
        if internal_user is not None:
            user = internal_user
            auth_source = AUTH_SOURCE_INTERNAL
        elif _is_notebook_relay_path(route_path) and not is_auth_disabled() and (authorization is not None or _notebook_query_token_present(request)):
            # Sidecar-token carve-out (BEFORE the PAT branch): the sandboxed
            # notebook iframe sends no cookies, so its fetches carry the
            # short-lived sidecar JWT minted by GET /api/notebooks/token —
            # as a Bearer header normally, or as ?sidecar_token= on GETs
            # that cannot set headers. The carve-out triggers on credential
            # PRESENCE (header present, or the query key present even when
            # empty/duplicated): present-but-invalid credentials are a hard
            # 401 and never fall back to the session cookie (same strictness
            # as #4849); requests carrying neither credential fall through
            # to the cookie path below.
            from app.gateway.auth.pat import extract_bearer_token
            from app.gateway.auth.sidecar_token import validate_sidecar_token
            from app.gateway.deps import get_local_provider

            raw_token: str | None
            if authorization is not None:
                raw_token = extract_bearer_token(authorization)
            elif request.method == "GET":
                raw_token = _notebook_query_token(request)
            else:
                raw_token = ""
            payload = validate_sidecar_token(raw_token) if raw_token else None
            if payload is None:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
            try:
                user = await get_local_provider().get_user(payload.sub)
            except Exception:
                logger.warning("Sidecar-token users lookup failed", exc_info=True)
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
            if user is None or getattr(user, "disabled", False):
                # Mirror the PAT path: deleted/suspended owners resolve to
                # the same generic 401 so the verdict is not an oracle.
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
            auth_source = AUTH_SOURCE_SIDECAR
        elif authorization is not None and not is_auth_disabled():
            # Bearer (PAT) credential precedence (#4849): a present-but-invalid
            # Authorization header is a hard 401 and never silently falls back
            # to the session cookie. This is also what makes the CSRF
            # middleware's Bearer skip safe — a cross-site attacker cannot ride
            # a victim's cookie by padding the request with a garbage Bearer
            # header, because the request dies here before any route runs.
            # Auth-disabled mode is an operator override of all authentication,
            # so it stays ahead of the Bearer check (a stray Authorization
            # header from a proxy must not 401 an E2E sandbox).
            from app.gateway.auth.pat import authenticate_pat, is_pat_allowed_route

            try:
                user, pat_scopes = await authenticate_pat(request.app, authorization)
            except HTTPException as exc:
                return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
            # Default-deny route boundary (#5041 review P1-1): scopes only
            # constrain @require_permission routes, so any route outside the
            # explicit PAT policy is closed to PAT callers outright — an
            # all-scopes token must not reach undecorated mutation routes.
            if not is_pat_allowed_route(request.method, get_request_route_path(request)):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "PAT credentials are not permitted on this route"},
                )
            auth_source = AUTH_SOURCE_PAT
        elif access_token:
            # Strict JWT validation: reject junk/expired tokens with 401
            # right here instead of silently passing through. This closes
            # the "junk cookie bypass" gap (AUTH_TEST_PLAN test 7.5.8):
            # without this, non-isolation routes like /api/models would
            # accept any cookie-shaped string as authentication.
            #
            # We call the *strict* resolver so that fine-grained error
            # codes (token_expired, token_invalid, user_not_found, …)
            # propagate from AuthErrorCode, not get flattened into one
            # generic code. BaseHTTPMiddleware doesn't let HTTPException
            # bubble up, so we catch and render it as JSONResponse here.
            from app.gateway.deps import get_current_user_from_request

            try:
                user = await get_current_user_from_request(request)
            except HTTPException as exc:
                if not is_auth_disabled():
                    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
                user = get_auth_disabled_user()
                auth_source = AUTH_SOURCE_AUTH_DISABLED
        elif is_auth_disabled():
            user = get_auth_disabled_user()
            auth_source = AUTH_SOURCE_AUTH_DISABLED
        else:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": AuthErrorResponse(
                        code=AuthErrorCode.NOT_AUTHENTICATED,
                        message="Authentication required",
                    ).model_dump()
                },
            )

        # Stamp both request.state.user (for the contextvar pattern)
        # and request.state.auth (so @require_permission's "auth is
        # None" branch short-circuits instead of running the entire
        # JWT-decode + DB-lookup pipeline a second time per request).
        request.state.user = user
        request.state.auth_source = auth_source
        permissions = await resolve_route_permissions(
            user,
            is_internal=auth_source == AUTH_SOURCE_INTERNAL,
        )
        if auth_source == AUTH_SOURCE_PAT:
            # A PAT can only narrow its owning user's permissions: the stored
            # scopes intersect the resolved route permissions, never widen
            # them, and role changes / authorization policy stay authoritative
            # because they were resolved fresh from the owning user above.
            permissions = [permission for permission in permissions if permission in pat_scopes]
        request.state.auth = AuthContext(user=user, permissions=permissions)
        token = set_current_user(user)
        try:
            return await call_next(request)
        finally:
            reset_current_user(token)

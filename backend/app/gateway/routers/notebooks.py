"""QuantFlow connector for the ReportForge notebook sidecar (Phase 3).

Thin gateway surface over ``reportforge serve`` (loopback-only HTTP):

- ``GET /api/notebooks/token`` — mint a short-lived sidecar JWT for the
  cookie-authenticated caller. The sandboxed iframe cannot send cookies,
  so the shell fetches this token and delivers it via ``postMessage``.
- ``GET /api/notebooks/health`` — connector health/status aggregating
  the sidecar's component + contract versions. Sidecar down → 503.
- ``/{proxy_path:path}`` — httpx reverse-proxy relay to the sidecar
  (notebook JSON API + UI bundle), streaming responses unbuffered.

Authentication on the relay paths is handled by ``AuthMiddleware``'s
sidecar-token carve-out (sidecar Bearer header, or ``?sidecar_token=``
for GETs that cannot carry headers) — except the D1 static UI bundle
(``/``, ``/app.js``, ``/bridge.js``, ``/sanitize.js``, ``/styles.css``),
which is EXEMPT from per-request auth on both sides and forwarded
upstream with no ``Authorization`` header. This router only decides
which upstream credential to send: it reuses the already validated
incoming Bearer header when the request arrived with one, and mints a
fresh token for the resolved user otherwise (cookie-authed direct
access, query-param GETs). The session cookie is never forwarded.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlencode

import httpx
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.gateway.auth.sidecar_token import SIDECAR_TOKEN_QUERY_PARAM, mint_sidecar_token
from app.gateway.auth_disabled import AUTH_SOURCE_SIDECAR
from app.gateway.deps import get_current_user_from_request
from app.gateway.sse_headers import sse_response_headers

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notebooks", tags=["notebooks"])

# Versioned postMessage bridge contract (QuantFlow shell ↔ ReportForge UI).
# Owned by ReportForge; pinned here and in frontend tests so shell↔iframe
# skew hard-fails with a banner instead of drifting silently.
NOTEBOOK_BRIDGE_CONTRACT_VERSION = 1

SIDECAR_URL_ENV_VAR = "REPORTFORGE_SIDECAR_URL"
SIDECAR_PORTFILE_NAME = "port"
SIDECAR_PORTFILE_DIRNAME = "reportforge-sidecar"

# D1 static UI bundle as relay-relative proxy paths (mirrors
# ``auth_middleware._NOTEBOOK_PUBLIC_ASSET_PATHS``; the middleware lets
# these through without credentials, so the relay forwards them upstream
# with no Authorization header — the sidecar exempts them too). Exact
# match: "" is the document root (a request to /api/notebooks or
# /api/notebooks/).
_NOTEBOOK_PUBLIC_ASSET_PROXY_PATHS: frozenset[str] = frozenset(
    {
        "",
        "app.js",
        "bridge.js",
        "sanitize.js",
        "styles.css",
    }
)

# Hop-by-hop headers (RFC 9110 §7.6.1) plus gateway-owned headers that must
# never cross the relay boundary in either direction.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_STRIP_REQUEST_HEADERS = _HOP_BY_HOP_HEADERS | frozenset(
    {
        "authorization",  # replaced with the upstream sidecar credential
        "cookie",  # the session cookie must never reach the sidecar
        "host",
        "content-length",  # httpx recomputes for the relayed body
        "origin",  # the sidecar allowlists origins itself; the gateway relays originless
        "referer",
    }
)

# Long cell runs stream output for minutes: bound connection setup but
# never the read side, or the proxy would sever slow streams.
_RELAY_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0)
_HEALTH_TIMEOUT_SECONDS = 5.0

# CORS for the opaque-origin iframe (sandbox without allow-same-origin):
# the UI's module scripts and Bearer-authed API fetches are CORS-mode
# requests from `Origin: null`, so relayed responses must carry ACAO and
# preflights must be answered at the relay. Scoped to this router only —
# never the gateway-wide CORSMiddleware allowlist (which cannot express
# the `null` origin). Safe: every non-D1 relay route still requires the
# short-lived per-user sidecar Bearer [REDACTED] (validated before this router
# runs); ACAO:* alone discloses nothing without that token, and the D1
# bundle it also covers is already public by contract. NOTE: in
# auth-enabled deployments AuthMiddleware 401s credential-less preflights
# before they reach this router — a preflight bypass there is still needed
# for production; auth-disabled E2E passes through.
_NOTEBOOK_CORS_ALLOW_METHODS = "GET, POST, PUT, PATCH, DELETE"
_NOTEBOOK_CORS_ALLOW_HEADERS = "authorization, content-type"
_NOTEBOOK_CORS_MAX_AGE_SECONDS = "600"


def resolve_sidecar_base_url() -> str:
    """Return the sidecar base URL (explicit env wins, else the portfile).

    The sidecar is single-instance via lockfile + portfile; its portfile
    holds either ``http://127.0.0.1:PORT`` or a bare port. Module-level so
    tests can monkeypatch it.

    Shared-path requirement: the portfile lives at
    ``<base_dir>/reportforge-sidecar/port`` where ``base_dir`` derives from
    ``DEER_FLOW_HOME`` (see ``deerflow.config.paths``), so the gateway and
    the sidecar MUST share ``DEER_FLOW_HOME`` — or the gateway MUST set
    ``REPORTFORGE_SIDECAR_URL`` explicitly. A gateway that resolves a
    different runtime home than the sidecar wrote to will report the
    connector unreachable even while the sidecar is healthy.
    """
    explicit = os.environ.get(SIDECAR_URL_ENV_VAR)
    if explicit:
        return explicit.rstrip("/")

    from deerflow.config.paths import get_paths

    portfile = get_paths().base_dir / SIDECAR_PORTFILE_DIRNAME / SIDECAR_PORTFILE_NAME
    try:
        raw = portfile.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Notebook sidecar is not running (unreadable portfile at {portfile})") from exc
    if not raw:
        raise RuntimeError(f"Notebook sidecar is not running (empty portfile at {portfile})")
    if "://" in raw:
        return raw.rstrip("/")
    return f"http://127.0.0.1:{raw}"


def build_sidecar_client(timeout: float | httpx.Timeout | None = None) -> httpx.AsyncClient:
    """Build a one-request sidecar client. Module-level so tests can swap the transport."""
    if timeout is None:
        return httpx.AsyncClient()
    return httpx.AsyncClient(timeout=timeout)


def _strip_token_query_param(query: bytes) -> str:
    """Rebuild the query string without the sidecar-token param (no log leaks)."""
    params = [(key, value) for key, value in parse_qsl(query.decode("latin-1"), keep_blank_values=True) if key != SIDECAR_TOKEN_QUERY_PARAM]
    return urlencode(params)


def _is_contained_proxy_path(proxy_path: str) -> bool:
    """True when *proxy_path* stays under the sidecar base URL (D2 pin).

    The sidecar UI derives its API base from its own location (never an
    absolute ``/api/*`` URL), so every legitimate relay target is a plain
    relative path under the relay prefix. Dot-segments (``.``/``..``) would
    resolve above the base and must never be forwarded — the relay rejects
    them with 400 before any upstream contact.
    """
    return all(segment not in (".", "..") for segment in proxy_path.split("/"))


class UpstreamCredentialError(RuntimeError):
    """No upstream credential could be resolved; the relay must fail closed."""


def _upstream_credential(request: Request) -> str:
    """Return the Bearer credential to send upstream.

    When the request arrived with a sidecar Bearer (already validated by
    the middleware carve-out), reuse it verbatim; otherwise mint a fresh
    token for the resolved user (cookie-authed direct access and
    query-param GETs both land here).

    Fail-closed: a missing identity (no ``request.state.user``, e.g. the
    middleware never ran) raises :class:`UpstreamCredentialError` — never
    ``AttributeError`` — so the relay answers 401 instead of 500.
    """
    if getattr(request.state, "auth_source", None) == AUTH_SOURCE_SIDECAR:
        incoming = request.headers.get("authorization", "")
        if incoming:
            return incoming
        # Query-param GETs carry no header: fall through and mint for the
        # middleware-resolved user below.
    user = getattr(request.state, "user", None)
    user_id = getattr(user, "id", None)
    if user_id is None:
        raise UpstreamCredentialError("no authenticated user to mint an upstream credential for")
    return f"Bearer {mint_sidecar_token(str(user_id))}"


@router.get("/token")
async def mint_notebook_token(user=Depends(get_current_user_from_request)) -> dict[str, str]:
    """Mint a short-lived sidecar JWT for the cookie-authenticated caller."""
    token = mint_sidecar_token(str(user.id))
    from app.gateway.auth.sidecar_token import SIDECAR_TOKEN_TTL

    expires_at = datetime.now(UTC) + SIDECAR_TOKEN_TTL
    return {"token": token, "token_type": "Bearer", "expires_at": expires_at.isoformat()}


@router.get("/health")
async def notebook_connector_health(user=Depends(get_current_user_from_request)) -> Response:
    """Aggregate sidecar component + contract versions; 503 when unreachable."""
    try:
        base_url = resolve_sidecar_base_url()
    except RuntimeError:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unreachable",
                "bridge_contract_version": NOTEBOOK_BRIDGE_CONTRACT_VERSION,
                "sidecar": None,
                "hint": ("Notebook sidecar portfile is missing or unreadable. The gateway and the sidecar must share DEER_FLOW_HOME, or the gateway must set REPORTFORGE_SIDECAR_URL explicitly (see resolve_sidecar_base_url)."),
            },
        )
    # The sidecar validates a token per request on EVERY route, including
    # /health, so the connector mints one for the requesting user.
    upstream_headers = {"Authorization": f"Bearer {mint_sidecar_token(str(user.id))}"}
    try:
        async with build_sidecar_client(timeout=_HEALTH_TIMEOUT_SECONDS) as client:
            upstream = await client.get(f"{base_url}/health", headers=upstream_headers)
            payload = upstream.json()
            if not isinstance(payload, dict):
                # A foreign shape (list, string, …) is not a health report —
                # fail inside the guarded block so it 503s like any other
                # unreadable sidecar instead of AttributeErroring on .get.
                raise ValueError(f"unexpected sidecar health shape: {type(payload).__name__}")
    except (httpx.HTTPError, ValueError):
        logger.warning("Notebook sidecar health check failed", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "status": "unreachable",
                "bridge_contract_version": NOTEBOOK_BRIDGE_CONTRACT_VERSION,
                "sidecar": None,
                "hint": "Notebook sidecar is unreachable at its resolved URL; it may have been culled while idle.",
            },
        )
    if upstream.status_code != 200:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unreachable",
                "bridge_contract_version": NOTEBOOK_BRIDGE_CONTRACT_VERSION,
                "sidecar": None,
                "hint": f"Notebook sidecar health returned HTTP {upstream.status_code}; it may be starting or unhealthy.",
            },
        )
    return JSONResponse(
        content={
            "status": "healthy",
            "bridge_contract_version": NOTEBOOK_BRIDGE_CONTRACT_VERSION,
            "sidecar": {
                "reachable": True,
                "component_version": payload.get("component_version"),
                "contract_version": payload.get("contract_version"),
            },
        }
    )


@router.api_route("/{proxy_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def relay_to_sidecar(proxy_path: str, request: Request) -> Response:
    """Relay one request to the sidecar, streaming the response unbuffered."""
    if not _is_contained_proxy_path(proxy_path):
        # D2: dot-segments would resolve above the sidecar base URL —
        # reject before any upstream contact (and before minting).
        return JSONResponse(status_code=400, content={"detail": "Invalid notebook relay path"})
    if request.method == "OPTIONS":
        # CORS preflight from the opaque-origin iframe: answer at the
        # relay, never upstream (preflights carry no credentials by spec
        # and the sidecar has no OPTIONS routes). Auth-disabled E2E
        # reaches here; see the _NOTEBOOK_CORS_* note for production.
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": _NOTEBOOK_CORS_ALLOW_METHODS,
                "Access-Control-Allow-Headers": _NOTEBOOK_CORS_ALLOW_HEADERS,
                "Access-Control-Max-Age": _NOTEBOOK_CORS_MAX_AGE_SECONDS,
            },
        )
    try:
        base_url = resolve_sidecar_base_url()
    except RuntimeError:
        return JSONResponse(status_code=503, content={"detail": "Notebook sidecar is not running"})
    query = _strip_token_query_param(request.url.query.encode("latin-1"))
    upstream_url = f"{base_url}/{proxy_path}"
    if query:
        upstream_url = f"{upstream_url}?{query}"

    headers = {name: value for name, value in request.headers.items() if name.lower() not in _STRIP_REQUEST_HEADERS}
    if proxy_path in _NOTEBOOK_PUBLIC_ASSET_PROXY_PATHS:
        # D1: the static UI bundle is exempt from per-request auth on both
        # sides — forward with no Authorization header. The middleware let
        # this through without credentials, so no identity is required.
        pass
    else:
        try:
            headers["Authorization"] = _upstream_credential(request)
        except UpstreamCredentialError:
            logger.warning("Notebook relay without a resolvable upstream credential for %s %s", request.method, proxy_path)
            return JSONResponse(status_code=401, content={"detail": "Invalid token"})

    # Request-body streaming for upload-bearing methods; httpx recomputes
    # framing (we stripped content-length above).
    content = request.stream() if request.method in {"POST", "PUT", "PATCH", "DELETE"} else None

    client = build_sidecar_client()
    stream_context = client.stream(
        request.method,
        upstream_url,
        headers=headers,
        content=content,
        timeout=_RELAY_TIMEOUT,
    )
    try:
        # httpx 0.28 gotcha: the response must be entered manually — an
        # `async with` here would close the stream before StreamingResponse
        # below has consumed it. The generator's finally block exits the
        # context and closes the client once the body is fully sent (or the
        # downstream client disconnects).
        upstream = await stream_context.__aenter__()
    except httpx.HTTPError:
        await client.aclose()
        logger.warning("Notebook sidecar relay failed for %s %s", request.method, proxy_path, exc_info=True)
        return JSONResponse(status_code=503, content={"detail": "Notebook sidecar is unreachable"})

    response_headers = {name: value for name, value in upstream.headers.items() if name.lower() not in _HOP_BY_HOP_HEADERS}
    # The opaque-origin iframe reads every relayed response through CORS:
    # stamp ACAO so the browser releases the bytes to the frame (the auth
    # story is unchanged — the sidecar token requirement stands). Upstream
    # keys arrive lowercased via httpx, so compare case-insensitively —
    # a doubled ACAO header ("*, *") would be rejected by browsers.
    if "access-control-allow-origin" not in {name.lower() for name in response_headers}:
        response_headers["Access-Control-Allow-Origin"] = "*"
    # Bytes cross untouched (aiter_raw + preserved content-encoding /
    # content-length): the proxy must never transform the stream.
    if "text/event-stream" in upstream.headers.get("content-type", ""):
        response_headers.update(sse_response_headers())
    status_code = upstream.status_code

    async def _stream_body():
        try:
            async for chunk in upstream.aiter_raw():
                if await request.is_disconnected():
                    break
                yield chunk
        finally:
            await stream_context.__aexit__(None, None, None)
            await client.aclose()

    # media_type stays None so the upstream Content-Type (charset and all)
    # passes through verbatim instead of being re-rendered by Starlette.
    return StreamingResponse(_stream_body(), status_code=status_code, headers=response_headers)

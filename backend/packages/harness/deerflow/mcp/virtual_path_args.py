"""Translate sandbox ``/mnt/user-data`` tool arguments for opted-in stdio MCP servers.

The model only ever sees the sandbox's virtual paths (``/mnt/user-data/uploads/x.pdf``),
but stdio MCP servers run on the host, and the pooled tool wrapper forwards
``arguments`` to them verbatim (only *results* are translated back to virtual
paths). A host-side server that opens files therefore receives a path that does
not exist for it.

A server opts in with ``"translate_virtual_paths": true`` on its
``mcpServers`` entry (``McpServerConfig`` allows extra fields), and the
interceptor is registered through the generic custom-interceptor hook::

    "mcpInterceptors": ["deerflow.mcp.virtual_path_args:build_virtual_path_args_interceptor"]

For an opted-in server, every string argument value (recursively through dicts
and lists) that *is* a virtual path — equal to ``/mnt/user-data`` or starting
with ``/mnt/user-data/`` — is replaced with the current thread's host path via
:meth:`deerflow.config.paths.Paths.resolve_virtual_path`, the same resolver the
built-in file tools use. Free text that merely mentions a path, and dict keys,
are left alone. A path that resolves outside the thread's user-data dir
(``..`` segments, a symlink pointing out) is left untouched, so the server
rejects it instead of receiving a host path it must not see.

Opt-in is per server because some servers give ``/mnt/user-data`` their own
meaning (report-forge maps it to its project tree), and only stdio servers are
eligible: a remote HTTP/SSE server cannot open host paths.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from deerflow.config.extensions_config import ExtensionsConfig
from deerflow.config.paths import VIRTUAL_PATH_PREFIX, get_paths

logger = logging.getLogger(__name__)

OPT_IN_FIELD = "translate_virtual_paths"


def _opted_in_servers(extensions_config: ExtensionsConfig) -> frozenset[str]:
    servers: set[str] = set()
    for server_name, server_config in extensions_config.get_enabled_mcp_servers().items():
        if (server_config.model_extra or {}).get(OPT_IN_FIELD) is not True:
            continue
        if server_config.type != "stdio":
            logger.warning(
                "MCP server '%s' sets %s but uses the '%s' transport; host paths only apply to 'stdio' servers — ignoring",
                server_name,
                OPT_IN_FIELD,
                server_config.type,
            )
            continue
        servers.add(server_name)
    return frozenset(servers)


def _is_virtual_path(value: str) -> bool:
    return value == VIRTUAL_PATH_PREFIX or value.startswith(VIRTUAL_PATH_PREFIX + "/")


def _rewrite(value: Any, resolve: Any) -> Any:
    if isinstance(value, str):
        return resolve(value) if _is_virtual_path(value) else value
    if isinstance(value, list):
        return [_rewrite(item, resolve) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite(item, resolve) for key, item in value.items()}
    return value


def build_virtual_path_args_interceptor() -> Any | None:
    """Build the argument-translation interceptor, or ``None`` if no server opts in.

    Called once per discovered server by ``build_mcp_tool_interceptors``. The
    opt-in set is read from the same on-disk ``extensions_config.json`` that
    discovery reads; a config change re-runs discovery and so rebuilds this.
    """
    try:
        opted_in = _opted_in_servers(ExtensionsConfig.from_file())
    except Exception:
        logger.warning("Could not read extensions config for %s; virtual path translation disabled", OPT_IN_FIELD, exc_info=True)
        return None
    if not opted_in:
        return None

    async def virtual_path_args_interceptor(request: Any, handler: Any) -> Any:
        if request.server_name not in opted_in or not request.args:
            return await handler(request)

        # Same thread/user resolution the pooled stdio wrapper uses for its cwd,
        # TMPDIR and result-path translation, so arguments and results agree.
        from deerflow.mcp.tools import _extract_thread_id
        from deerflow.runtime.user_context import resolve_runtime_user_id

        thread_id = _extract_thread_id(request.runtime)
        user_id = resolve_runtime_user_id(request.runtime)
        paths = get_paths()

        def resolve(virtual_path: str) -> str:
            try:
                return str(paths.resolve_virtual_path(thread_id, virtual_path, user_id=user_id))
            except ValueError:
                logger.warning(
                    "Not translating MCP argument for server '%s': path escapes the thread's user-data dir",
                    request.server_name,
                )
                return virtual_path

        # resolve_virtual_path() touches the filesystem (symlink resolution); keep it off the loop.
        args = await asyncio.to_thread(_rewrite, request.args, resolve)
        if args == request.args:
            return await handler(request)
        return await handler(request.override(args=args))

    return virtual_path_args_interceptor

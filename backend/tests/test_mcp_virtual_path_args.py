"""Integration tests for the opt-in ``/mnt/user-data`` MCP argument interceptor.

The interceptor is loaded the production way: a real ``extensions_config.json``
names it in ``mcpInterceptors`` and resolves it by dotted path, and calls go
through the pooled stdio tool wrapper down to the session's ``call_tool``.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool
from mcp.types import CallToolResult

import deerflow.config.paths as paths_module
from deerflow.config.extensions_config import ExtensionsConfig
from deerflow.config.paths import Paths
from deerflow.mcp.interceptors import build_mcp_tool_interceptors
from deerflow.mcp.tools import _make_session_pool_tool
from deerflow.mcp.virtual_path_args import build_virtual_path_args_interceptor

INTERCEPTOR_PATH = "deerflow.mcp.virtual_path_args:build_virtual_path_args_interceptor"
THREAD_ID = "thread-vp"
USER_ID = "user-vp"


@pytest.fixture
def deer_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("DEER_FLOW_HOME", str(home))
    monkeypatch.setattr(paths_module, "_paths", Paths(home))
    return home


def _write_config(tmp_path, monkeypatch, servers: dict) -> ExtensionsConfig:
    config_file = tmp_path / "extensions_config.json"
    config_file.write_text(json.dumps({"mcpInterceptors": [INTERCEPTOR_PATH], "mcpServers": servers}))
    monkeypatch.setenv("DEER_FLOW_EXTENSIONS_CONFIG_PATH", str(config_file))
    return ExtensionsConfig.from_file(str(config_file))


def _pooled_tool(server_name: str, interceptors: list):
    session = AsyncMock()
    session.call_tool = AsyncMock(return_value=CallToolResult(content=[], isError=False))
    pool = MagicMock()
    pool.get_session = AsyncMock(return_value=session)
    pool.close_session_if_current = AsyncMock()
    original = StructuredTool(
        name=f"{server_name}_parse_document",
        description="test",
        args_schema={"type": "object", "properties": {}},
        coroutine=AsyncMock(),
        response_format="content_and_artifact",
    )
    with patch("deerflow.mcp.tools.get_session_pool", return_value=pool):
        wrapped = _make_session_pool_tool(
            original,
            server_name,
            {"transport": "stdio", "command": "x", "args": []},
            tool_interceptors=interceptors,
        )
    return wrapped, session


def _runtime():
    return SimpleNamespace(context={"thread_id": THREAD_ID, "user_id": USER_ID, "thread_incarnation": None}, config={})


async def _call(server_name: str, interceptors: list, **arguments) -> dict:
    wrapped, session = _pooled_tool(server_name, interceptors)
    await wrapped.coroutine(deerflow_runtime=_runtime(), **arguments)
    session.call_tool.assert_awaited_once()
    return session.call_tool.await_args.args[1]


@pytest.mark.asyncio
async def test_rewrites_virtual_paths_only_for_opted_in_server(tmp_path, monkeypatch, deer_home):
    config = _write_config(
        tmp_path,
        monkeypatch,
        {
            "kizonlp": {"type": "stdio", "command": "x", "translate_virtual_paths": True},
            "reportforge": {"type": "stdio", "command": "y"},
        },
    )
    interceptors = build_mcp_tool_interceptors(config)
    user_data = Paths(deer_home).sandbox_user_data_dir(THREAD_ID, user_id=USER_ID).resolve()

    arguments = {
        "path": "/mnt/user-data/uploads/10-K.pdf",
        "options": {"extra": ["/mnt/user-data/outputs/notes.md", "plain"], "n": 3},
        "root": "/mnt/user-data",
        # Only whole-string path values are rewritten, never free text.
        "text": "summarise /mnt/user-data/uploads/10-K.pdf",
    }

    sent = await _call("kizonlp", interceptors, **arguments)
    assert sent == {
        "path": str(user_data / "uploads" / "10-K.pdf"),
        "options": {"extra": [str(user_data / "outputs" / "notes.md"), "plain"], "n": 3},
        "root": str(user_data),
        "text": "summarise /mnt/user-data/uploads/10-K.pdf",
    }

    # reportforge gives /mnt/user-data its own meaning; it must see the virtual path.
    assert await _call("reportforge", interceptors, **arguments) == arguments


@pytest.mark.asyncio
async def test_paths_escaping_thread_user_data_are_left_untouched(tmp_path, monkeypatch, deer_home):
    _write_config(tmp_path, monkeypatch, {"kizonlp": {"type": "stdio", "command": "x", "translate_virtual_paths": True}})
    uploads = Paths(deer_home).sandbox_uploads_dir(THREAD_ID, user_id=USER_ID)
    uploads.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (uploads / "escape").symlink_to(outside, target_is_directory=True)

    escapes = {
        "dotdot": "/mnt/user-data/uploads/../../../../../etc/passwd",
        "symlink": "/mnt/user-data/uploads/escape/secret.txt",
        "lookalike": "/mnt/user-dataX/uploads/a.pdf",
    }
    assert await _call("kizonlp", [build_virtual_path_args_interceptor()], **escapes) == escapes


def test_builder_skips_without_stdio_opt_in(tmp_path, monkeypatch, deer_home):
    _write_config(
        tmp_path,
        monkeypatch,
        {
            "plain": {"type": "stdio", "command": "x"},
            # Host paths mean nothing to a remote server, so an HTTP opt-in is ignored.
            "remote": {"type": "http", "url": "https://example.test/mcp", "translate_virtual_paths": True},
        },
    )
    assert build_virtual_path_args_interceptor() is None

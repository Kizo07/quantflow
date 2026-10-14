"""Migration tests for 0041_cli_sessions.

Runs the full alembic chain on an empty SQLite database (not
``create_all`` + stamp), then exercises the 0041 downgrade/upgrade cycle.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from sqlalchemy.ext.asyncio import create_async_engine

from deerflow.persistence.bootstrap import _MIGRATIONS_DIR

pytestmark = pytest.mark.asyncio

_SCRIPT_LOCATION = str(_MIGRATIONS_DIR)
_PREVIOUS = "0040_merge_quantflow_upstream"

_EXPECTED_SESSION_COLUMNS = {
    "id",
    "session_ref",
    "resumed_from",
    "status",
    "mode",
    "model",
    "provider",
    "preset",
    "effort",
    "contributor_mode",
    "approval_mode",
    "sandbox_network",
    "trust_workspace",
    "disable_approval",
    "disable_sandbox",
    "yolo",
    "workspace",
    "prompt",
    "pid",
    "tmux_session",
    "log_path",
    "error",
    "started_at",
    "ended_at",
    "created_by",
    "created_at",
    "updated_at",
}

_EXPECTED_PRESET_COLUMNS = {
    "name",
    "model",
    "provider",
    "preset",
    "effort",
    "contributor_mode",
    "yolo",
    "approval_mode",
    "updated_by",
    "updated_at",
}


def _alembic_config(db_url: str) -> AlembicConfig:
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", _SCRIPT_LOCATION)
    cfg.set_main_option("sqlalchemy.url", db_url.replace("%", "%%"))
    return cfg


def _table_names(sync_conn) -> set[str]:
    return set(sa.inspect(sync_conn).get_table_names())


def _column_names(sync_conn, table: str) -> set[str]:
    return {column["name"] for column in sa.inspect(sync_conn).get_columns(table)}


async def _inspect(engine, fn):
    async with engine.connect() as conn:
        return await conn.run_sync(fn)


async def test_cli_sessions_migration_upgrade_downgrade_cycle(tmp_path: Path) -> None:
    db_path = tmp_path / "cli-sessions-migration.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    cfg = _alembic_config(f"sqlite+aiosqlite:///{db_path}")
    try:
        await asyncio.to_thread(alembic_command.upgrade, cfg, "head")

        tables = await _inspect(engine, _table_names)
        assert "cli_sessions" in tables
        assert "cli_presets" in tables
        assert "alembic_version" in tables
        columns = await _inspect(engine, lambda conn: _column_names(conn, "cli_sessions"))
        assert columns == _EXPECTED_SESSION_COLUMNS
        preset_columns = await _inspect(engine, lambda conn: _column_names(conn, "cli_presets"))
        assert preset_columns == _EXPECTED_PRESET_COLUMNS
        indexes = await _inspect(
            engine,
            lambda conn: {idx["name"] for idx in sa.inspect(conn).get_indexes("cli_sessions")},
        )
        assert "ix_cli_sessions_status" in indexes
        assert "ix_cli_sessions_created_by" in indexes
        assert "ix_cli_sessions_owner_created" in indexes

        async with engine.connect() as conn:
            seed = (
                await conn.execute(
                    sa.text("SELECT model, effort, contributor_mode, yolo FROM cli_presets WHERE name = 'muse-1.3'"),
                )
            ).one()
        assert seed[0] == "muse-spark-1.3" and seed[1] == "ultra" and bool(seed[2]) and bool(seed[3])

        await asyncio.to_thread(alembic_command.downgrade, cfg, _PREVIOUS)
        tables_after_down = await _inspect(engine, _table_names)
        assert "cli_sessions" not in tables_after_down
        assert "cli_presets" not in tables_after_down

        await asyncio.to_thread(alembic_command.upgrade, cfg, "head")
        tables_after_up = await _inspect(engine, _table_names)
        assert "cli_sessions" in tables_after_up
        assert "cli_presets" in tables_after_up
    finally:
        await engine.dispose()

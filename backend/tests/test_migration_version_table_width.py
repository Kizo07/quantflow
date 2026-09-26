"""Tests for the wide ``alembic_version`` table (``migrations/_version_table.py``).

Alembic's native version table declares ``version_num VARCHAR(32)``; revision
``0028_knowledge_experiment_embeddings`` (36 chars) overflows it on Postgres,
where the length is enforced (SQLite ignores ``VARCHAR`` lengths, which masked
the bug). ``env.py`` now pre-creates/widens the table to ``TEXT`` before
alembic runs.

* Server-free checks: SQLite is left alone, and the custom definition compiles
  to ``TEXT`` on the Postgres dialect.
* Live checks: against a scratch Postgres database (created and dropped by the
  fixture -- peer socket locally, ``DEERFLOW_TEST_POSTGRES_URL`` when set),
  ``alembic stamp head`` through the repo-blessed in-process config stamps the
  36-char head cleanly with a fresh, legacy-``VARCHAR(32)``, and existing-
  ``TEXT`` version table (the last being the live workaround shape, which must
  stay compatible). Skipped when no Postgres server is reachable.
"""

from __future__ import annotations

import getpass
import os
import uuid

import pytest
import sqlalchemy as sa
from alembic import command as alembic_command
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from deerflow.persistence import bootstrap
from deerflow.persistence.migrations._version_table import (
    VERSION_COLUMN_NAME,
    VERSION_TABLE_NAME,
    ensure_wide_version_table,
    version_table_definition,
)

REVISION_0028 = "0028_knowledge_experiment_embeddings"

_MAINTENANCE_URL: str | None | bool = None  # None=unprobed, False=unreachable


def _maintenance_url() -> str | None:
    """Return a sync maintenance-database URL, or None when PG is unreachable.

    Prefers ``DEERFLOW_TEST_POSTGRES_URL`` (CI service shape,
    ``postgresql://...`` without a driver) and falls back to the local peer
    socket. The result is probed once per session with ``SELECT 1``.
    """
    global _MAINTENANCE_URL
    if _MAINTENANCE_URL is not None:
        return _MAINTENANCE_URL or None
    raw = os.getenv("DEERFLOW_TEST_POSTGRES_URL") or f"postgresql+psycopg://{getpass.getuser()}@/postgres?host=/run/postgresql"
    candidate = str(make_url(raw).set(drivername="postgresql+psycopg"))
    try:
        engine = sa.create_engine(candidate, poolclass=sa.pool.NullPool, connect_args={"connect_timeout": 5})
        try:
            with engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
        finally:
            engine.dispose()
    except Exception:  # noqa: BLE001 -- any connection failure means "no PG here"
        _MAINTENANCE_URL = False
        return None
    _MAINTENANCE_URL = candidate
    return candidate


@pytest.fixture
def scratch_pg():
    """Create a scratch Postgres database; drop it on teardown. Skips when unreachable."""
    maintenance = _maintenance_url()
    if maintenance is None:
        pytest.skip("no reachable PostgreSQL server (peer socket or DEERFLOW_TEST_POSTGRES_URL)")
    db_name = f"df_vertbl_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    maint_engine = sa.create_engine(maintenance, isolation_level="AUTOCOMMIT", poolclass=sa.pool.NullPool)
    try:
        with maint_engine.connect() as conn:
            conn.execute(sa.text(f'CREATE DATABASE "{db_name}"'))
    finally:
        maint_engine.dispose()
    base = make_url(maintenance)
    info = {
        "db_name": db_name,
        "sync_url": str(base.set(database=db_name)),
        "async_url": base.set(drivername="postgresql+asyncpg", database=db_name).render_as_string(hide_password=False),
    }
    yield info
    drop_engine = sa.create_engine(maintenance, isolation_level="AUTOCOMMIT", poolclass=sa.pool.NullPool)
    try:
        with drop_engine.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))
    finally:
        drop_engine.dispose()


def _version_rows(sync_url: str) -> list[str]:
    engine = sa.create_engine(sync_url, poolclass=sa.pool.NullPool)
    try:
        with engine.connect() as conn:
            return list(conn.execute(sa.text(f"SELECT {VERSION_COLUMN_NAME} FROM {VERSION_TABLE_NAME}")).scalars())
    finally:
        engine.dispose()


def _version_column_type(sync_url: str) -> str:
    engine = sa.create_engine(sync_url, poolclass=sa.pool.NullPool)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                sa.text("SELECT data_type FROM information_schema.columns WHERE table_name = :table AND column_name = :column"),
                {"table": VERSION_TABLE_NAME, "column": VERSION_COLUMN_NAME},
            ).one()
            return str(row[0])
    finally:
        engine.dispose()


def _stamp_head(async_url: str) -> str:
    """Stamp head through the repo-blessed in-process config (runs env.py)."""
    head = bootstrap._get_head_revision()
    assert len(head) > 32, f"expected a >32-char head to exercise the overflow, got {head!r}"
    engine = create_async_engine(async_url, poolclass=sa.pool.NullPool)
    try:
        cfg = bootstrap._get_alembic_config(engine)
        alembic_command.stamp(cfg, head)
    finally:
        # Sync disposal: AsyncEngine.dispose() needs a running loop, and this
        # test stays sync so env.py's own asyncio.run() can drive alembic.
        engine.sync_engine.dispose()
    return head


class TestDefinition:
    def test_compiles_to_text_on_postgres(self) -> None:
        ddl = str(sa.schema.CreateTable(version_table_definition()).compile(dialect=postgresql.dialect()))
        assert "TEXT" in ddl
        assert "VARCHAR(32)" not in ddl
        assert "alembic_version_pkc" in ddl
        assert "version_num" in ddl

    def test_sqlite_is_left_alone(self, tmp_path) -> None:
        """SQLite ignores declared lengths, so alembic's native DDL suffices."""
        engine = sa.create_engine(f"sqlite:///{(tmp_path / 'ver.db').as_posix()}")
        try:
            with engine.begin() as conn:
                assert ensure_wide_version_table(conn) == "skipped-non-postgres"
            assert VERSION_TABLE_NAME not in sa.inspect(engine).get_table_names()
        finally:
            engine.dispose()


class TestStampOnPostgres:
    def test_stamp_head_creates_text_version_table_on_fresh_db(self, scratch_pg) -> None:
        head = _stamp_head(scratch_pg["async_url"])
        assert head == REVISION_0028
        assert _version_rows(scratch_pg["sync_url"]) == [head]
        assert _version_column_type(scratch_pg["sync_url"]) == "text"

    def test_stamp_head_widens_legacy_varchar32_table(self, scratch_pg) -> None:
        """A pre-fix narrow table (short stamp) upgrades to TEXT in place."""
        engine = sa.create_engine(scratch_pg["sync_url"], poolclass=sa.pool.NullPool)
        try:
            with engine.begin() as conn:
                conn.execute(sa.text(f"CREATE TABLE {VERSION_TABLE_NAME} ({VERSION_COLUMN_NAME} VARCHAR(32) NOT NULL)"))
                conn.execute(
                    sa.text(f"INSERT INTO {VERSION_TABLE_NAME} ({VERSION_COLUMN_NAME}) VALUES (:rev)"),
                    {"rev": "0001_baseline"},
                )
        finally:
            engine.dispose()
        head = _stamp_head(scratch_pg["async_url"])
        assert _version_column_type(scratch_pg["sync_url"]) == "text"
        assert _version_rows(scratch_pg["sync_url"]) == [head]

    def test_stamp_head_reuses_existing_text_table(self, scratch_pg) -> None:
        """The live workaround shape (pre-created TEXT table) stays compatible."""
        engine = sa.create_engine(scratch_pg["sync_url"], poolclass=sa.pool.NullPool)
        try:
            with engine.begin() as conn:
                conn.execute(sa.text(f"CREATE TABLE {VERSION_TABLE_NAME} ({VERSION_COLUMN_NAME} TEXT NOT NULL)"))
                conn.execute(
                    sa.text(f"INSERT INTO {VERSION_TABLE_NAME} ({VERSION_COLUMN_NAME}) VALUES (:rev)"),
                    {"rev": "0001_baseline"},
                )
        finally:
            engine.dispose()
        head = _stamp_head(scratch_pg["async_url"])
        assert _version_column_type(scratch_pg["sync_url"]) == "text"
        assert _version_rows(scratch_pg["sync_url"]) == [head]

"""Wide ``alembic_version`` table for long revision ids.

Alembic's default version table declares ``version_num VARCHAR(32)``.
Revision ``0028_knowledge_experiment_embeddings`` (36 chars) overflows it on
PostgreSQL, where the length is enforced -- SQLite ignores ``VARCHAR`` lengths,
which is why the bug never surfaced until the knowledge-plane Postgres
migration (stamping/upgrading then failed with ``StringDataRightTruncation``
and rolled back). The live knowledge database worked around it with a manually
pre-created ``TEXT`` version table.

``do_run_migrations`` in ``env.py`` calls :func:`ensure_wide_version_table`
inside its transaction before alembic runs, so every online stamp/upgrade --
Gateway bootstrap, the knowledge-plane chain, or a direct ``alembic`` CLI run
from this directory -- gets a version column that fits:

* fresh database -- create ``alembic_version`` with ``TEXT`` up front, so
  alembic's own ``checkfirst`` create becomes a no-op and the stamp fits;
* legacy ``VARCHAR(n)`` table -- widen the column to ``TEXT`` in place
  (``varchar`` -> ``text`` needs no row rewrite on Postgres);
* existing ``TEXT`` table (e.g. the manually pre-created one) -- untouched,
  fully compatible.

Non-PostgreSQL dialects are left alone: SQLite ignores declared lengths, so
alembic's native DDL is already sufficient there.
"""

from __future__ import annotations

import logging

import sqlalchemy as sa

logger = logging.getLogger(__name__)

VERSION_TABLE_NAME = "alembic_version"
VERSION_COLUMN_NAME = "version_num"
# Alembic names the version-table PK ``{version_table}_pkc`` (see
# ``alembic.ddl.impl.DefaultImpl.version_table_impl``); match it so a table we
# pre-create is indistinguishable from one alembic created itself, width aside.
VERSION_TABLE_PK_NAME = "alembic_version_pkc"


def version_table_definition() -> sa.Table:
    """Return our ``alembic_version`` table definition (``TEXT`` version_num)."""
    return sa.Table(
        VERSION_TABLE_NAME,
        sa.MetaData(),
        sa.Column(VERSION_COLUMN_NAME, sa.Text, nullable=False),
        sa.PrimaryKeyConstraint(VERSION_COLUMN_NAME, name=VERSION_TABLE_PK_NAME),
    )


def ensure_wide_version_table(connection: sa.Connection) -> str:
    """Ensure the version table fits long revision ids; return what happened.

    Returns one of ``"created"`` (fresh table), ``"widened"`` (legacy bounded
    ``VARCHAR`` converted to ``TEXT``), ``"already-wide"`` (``TEXT`` or
    otherwise unbounded -- includes the manually pre-created live table),
    ``"skipped-non-postgres"`` (SQLite and friends ignore declared lengths),
    or ``"unexpected-shape"`` (table exists but has no ``version_num`` column;
    left for alembic to fail loudly on rather than guessed at).
    """
    if connection.dialect.name != "postgresql":
        return "skipped-non-postgres"
    inspector = sa.inspect(connection)
    if VERSION_TABLE_NAME not in inspector.get_table_names():
        version_table_definition().create(connection, checkfirst=True)
        logger.info("alembic: created %s with TEXT version_num (fits long revision ids)", VERSION_TABLE_NAME)
        return "created"
    columns = {column["name"]: column for column in inspector.get_columns(VERSION_TABLE_NAME)}
    column = columns.get(VERSION_COLUMN_NAME)
    if column is None:
        logger.warning("alembic: %s exists without a %s column; leaving it for alembic to report", VERSION_TABLE_NAME, VERSION_COLUMN_NAME)
        return "unexpected-shape"
    column_type = column["type"]
    if isinstance(column_type, sa.String) and column_type.length is not None:
        connection.execute(sa.text(f"ALTER TABLE {VERSION_TABLE_NAME} ALTER COLUMN {VERSION_COLUMN_NAME} TYPE TEXT"))
        logger.info(
            "alembic: widened %s.%s from %s to TEXT (fits long revision ids)",
            VERSION_TABLE_NAME,
            VERSION_COLUMN_NAME,
            column_type,
        )
        return "widened"
    return "already-wide"

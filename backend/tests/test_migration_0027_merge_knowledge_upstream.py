"""Migration tests for 0027_merge_knowledge_upstream.

Empty merge revision joining the QuantFlow knowledge-plane leg
(0023_knowledge_findings) with the upstream leg
(0026_mcp_task_lease_tokens). This file owns the chain-head pin, moved on
from ``test_migration_0026_mcp_task_lease_tokens`` with this revision.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

from deerflow.persistence import bootstrap

migration = importlib.import_module("deerflow.persistence.migrations.versions.0027_merge_knowledge_upstream")

pytestmark = pytest.mark.asyncio

REVISION = "0027_merge_knowledge_upstream"
PARENTS = ("0023_knowledge_findings", "0026_mcp_task_lease_tokens")


async def test_0027_is_the_chain_head():
    assert bootstrap._get_head_revision() == REVISION


def test_0027_joins_both_legs():
    assert migration.revision == REVISION
    assert tuple(migration.down_revision) == PARENTS


def test_single_head_no_branches():
    migrations_dir = Path(migration.__file__).resolve().parent.parent
    config = Config()
    config.set_main_option("script_location", migrations_dir.as_posix())
    script = ScriptDirectory.from_config(config)
    assert script.get_heads() == [REVISION]

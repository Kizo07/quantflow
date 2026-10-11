"""Migration tests for 0040_merge_quantflow_upstream.

Empty merge revision joining the upstream chain (0029_scheduler_agent_tasks
-> ... -> 0039_user_disabled) with the QuantFlow leg
(0029_merge_knowledge_outbox). This file owns the chain-head pin, moved
on from ``test_migration_0029_merge_knowledge_outbox``.
"""

from __future__ import annotations

import importlib

from alembic.config import Config
from alembic.script import ScriptDirectory

migration = importlib.import_module("deerflow.persistence.migrations.versions.0040_merge_quantflow_upstream")

REVISION = "0040_merge_quantflow_upstream"
PARENTS = ("0029_merge_knowledge_outbox", "0039_user_disabled")


def test_0040_joins_both_legs():
    assert migration.revision == REVISION
    assert tuple(migration.down_revision) == PARENTS
    # alembic_version.version_num is VARCHAR(32).
    assert len(REVISION) <= 32


def test_0040_is_the_chain_head():
    from pathlib import Path

    migrations_dir = Path(migration.__file__).resolve().parent.parent
    config = Config()
    config.set_main_option("script_location", migrations_dir.as_posix())
    script = ScriptDirectory.from_config(config)
    assert script.get_heads() == [REVISION]

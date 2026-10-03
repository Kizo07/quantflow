"""Migration tests for 0029_merge_knowledge_outbox.

Empty merge revision joining the upstream notification-outbox leg
(0028_parked_attempts) with the QuantFlow knowledge leg
(0028_knowledge_experiment_embeddings). This file owns the chain-head
pin, moved on from ``test_migration_0028_knowledge_experiment_embeddings``
and ``test_migration_0027_0028_notification_outbox``.
"""

from __future__ import annotations

import importlib

from alembic.config import Config
from alembic.script import ScriptDirectory

migration = importlib.import_module("deerflow.persistence.migrations.versions.0029_merge_knowledge_outbox")

REVISION = "0029_merge_knowledge_outbox"
PARENTS = ("0028_parked_attempts", "0028_knowledge_experiment_embeddings")


def test_0029_joins_both_legs():
    assert migration.revision == REVISION
    assert tuple(migration.down_revision) == PARENTS


def test_0029_is_the_chain_head():
    from pathlib import Path

    migrations_dir = Path(migration.__file__).resolve().parent.parent
    config = Config()
    config.set_main_option("script_location", migrations_dir.as_posix())
    script = ScriptDirectory.from_config(config)
    assert script.get_heads() == [REVISION]

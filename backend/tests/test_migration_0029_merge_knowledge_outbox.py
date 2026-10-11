"""Migration tests for 0029_merge_knowledge_outbox.

Empty merge revision joining the upstream notification-outbox leg
(0028_parked_attempts) with the QuantFlow knowledge leg
(0028_knowledge_experiment_embeddings). The chain-head pin moved on to
``test_migration_0040_merge_quantflow_upstream`` with the 2026-10-11
QuantFlow/upstream merge revision.
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


def test_0029_remains_in_the_single_migration_chain():
    from pathlib import Path

    migrations_dir = Path(migration.__file__).resolve().parent.parent
    config = Config()
    config.set_main_option("script_location", migrations_dir.as_posix())
    script = ScriptDirectory.from_config(config)
    assert script.get_heads() == ["0040_merge_quantflow_upstream"]
    ancestors = {revision.revision for revision in script.walk_revisions()}
    assert REVISION in ancestors

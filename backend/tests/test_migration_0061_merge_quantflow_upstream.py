"""Migration tests for 0061_merge_quantflow_upstream.

Empty merge revision joining the QuantFlow leg
(0050_merge_quantflow_upstream, the 2026-11-01 join of the local
CLI/persistent-bot/notify-policy chain with upstream's 0049_run_inputs
chain) with upstream's 2026-11-24 chain
(0049_run_producer_fencing -> ... -> 0060_coordination_run_settings).
"""

from __future__ import annotations

import importlib
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

from deerflow.persistence import bootstrap

migration = importlib.import_module("deerflow.persistence.migrations.versions.0061_merge_quantflow_upstream")

REVISION = "0061_merge_quantflow_upstream"
PARENTS = ("0050_merge_quantflow_upstream", "0060_coordination_run_settings")


def test_0061_joins_both_legs():
    assert migration.revision == REVISION
    assert tuple(migration.down_revision) == PARENTS
    # alembic_version.version_num is VARCHAR(32).
    assert len(REVISION) <= 32


def test_0061_is_the_single_chain_head():
    assert bootstrap._get_head_revision() == REVISION
    migrations_dir = Path(migration.__file__).resolve().parent.parent
    config = Config()
    config.set_main_option("script_location", migrations_dir.as_posix())
    script = ScriptDirectory.from_config(config)
    assert script.get_heads() == [REVISION]
    assert script.get_revision("0060_coordination_run_settings").down_revision == "0059_coordination_grants"
    assert tuple(script.get_revision("0050_merge_quantflow_upstream").down_revision) == (
        "0044_scheduled_task_notify_policy",
        "0049_run_inputs",
    )

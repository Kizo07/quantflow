"""Migration tests for 0050_merge_quantflow_upstream.

Empty merge revision joining the upstream chain
(0040_scheduled_group_notice -> ... -> 0049_run_inputs) with the
QuantFlow leg (0040_merge_quantflow_upstream -> 0041_cli_sessions ->
0042_cli_audit -> 0043_persistent_bot ->
0044_scheduled_task_notify_policy). This file owns the chain-head pin,
moved on from ``test_migration_0047_mcp_oauth_tokens``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

from deerflow.persistence import bootstrap

migration = importlib.import_module("deerflow.persistence.migrations.versions.0050_merge_quantflow_upstream")

REVISION = "0050_merge_quantflow_upstream"
PARENTS = ("0044_scheduled_task_notify_policy", "0049_run_inputs")


def test_0050_joins_both_legs():
    assert migration.revision == REVISION
    assert tuple(migration.down_revision) == PARENTS
    # alembic_version.version_num is VARCHAR(32).
    assert len(REVISION) <= 32


def test_0050_is_the_single_chain_head():
    assert bootstrap._get_head_revision() == REVISION
    migrations_dir = Path(migration.__file__).resolve().parent.parent
    config = Config()
    config.set_main_option("script_location", migrations_dir.as_posix())
    script = ScriptDirectory.from_config(config)
    assert script.get_heads() == [REVISION]
    assert script.get_revision("0049_run_inputs").down_revision == "0048_extension_execution_leases"
    assert script.get_revision("0044_scheduled_task_notify_policy").down_revision == "0043_persistent_bot"

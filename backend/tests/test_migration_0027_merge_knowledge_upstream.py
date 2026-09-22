"""Migration tests for 0027_merge_knowledge_upstream.

Empty merge revision joining the QuantFlow knowledge-plane leg
(0023_knowledge_findings) with the upstream leg
(0026_mcp_task_lease_tokens). The chain-head pin moved on to
``test_migration_0028_knowledge_experiment_embeddings`` with the
experiment-embeddings revision.
"""

from __future__ import annotations

import importlib

migration = importlib.import_module("deerflow.persistence.migrations.versions.0027_merge_knowledge_upstream")

REVISION = "0027_merge_knowledge_upstream"
PARENTS = ("0023_knowledge_findings", "0026_mcp_task_lease_tokens")


def test_0027_joins_both_legs():
    assert migration.revision == REVISION
    assert tuple(migration.down_revision) == PARENTS

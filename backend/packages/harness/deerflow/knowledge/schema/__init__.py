"""ORM models for the Research Knowledge Plane (KB episodic + semantic rows).

Re-exports the mapped rows (Phase 1 migration ``0022_knowledge_phase1`` plus
the Phase 2 ``finding`` table from ``0023_knowledge_findings``) and the
shared column helpers and enum vocabularies so call sites bind to one
stable import surface::

    from deerflow.knowledge.schema import AssumptionRow, ExperimentRow, FindingRow

Table layout:

* ``research_project`` / ``agent_run`` (:mod:`.research`) — scope roots and
  provenance anchors for every KB row a research agent writes.
* ``artifact`` / ``dataset_version`` (:mod:`.evidence`) — immutable,
  content-addressed evidence and exact dataset vintages.
* ``experiment`` / ``experiment_dataset`` / ``experiment_artifact`` /
  ``assumption`` (:mod:`.experiments`) — first-class experiments with typed
  dataset/artifact links and explicit assumptions.
* ``finding`` (:mod:`.findings`) — versioned, evidence-backed claims with
  the asynchronously populated retrieval columns (``search_document``,
  ``embedding``).
"""

from __future__ import annotations

from deerflow.knowledge.schema.evidence import ArtifactRow, DatasetVersionRow
from deerflow.knowledge.schema.experiments import (
    AssumptionRow,
    ExperimentArtifactRow,
    ExperimentDatasetRow,
    ExperimentRow,
)
from deerflow.knowledge.schema.findings import (
    EMBEDDING_DIMENSIONS,
    FINDING_STATUSES,
    FINDING_STATUSES_PHASE2,
    FINDING_TYPES,
    EmbeddingVector,
    FindingRow,
    NativeVector,
    embedding_vector,
    tsvector,
)
from deerflow.knowledge.schema.research import AgentRunRow, ResearchProjectRow
from deerflow.knowledge.schema.types import (
    AGENT_RUN_STATUSES,
    ARTIFACT_KINDS,
    ASSUMPTION_CATEGORIES,
    ASSUMPTION_SENSITIVITIES,
    ASSUMPTION_STATUSES,
    DATASET_ROLES,
    EXPERIMENT_STATUSES,
    FAILURE_CLASSES,
    OUTCOME_CLASSES,
    check_in,
    jsonb,
    new_uuid,
    utcnow,
    uuid_column,
)

__all__ = [
    "AgentRunRow",
    "ArtifactRow",
    "AssumptionRow",
    "DatasetVersionRow",
    "ExperimentArtifactRow",
    "ExperimentDatasetRow",
    "ExperimentRow",
    "FindingRow",
    "ResearchProjectRow",
    "EMBEDDING_DIMENSIONS",
    "FINDING_STATUSES",
    "FINDING_STATUSES_PHASE2",
    "FINDING_TYPES",
    "EmbeddingVector",
    "NativeVector",
    "embedding_vector",
    "tsvector",
    "AGENT_RUN_STATUSES",
    "ARTIFACT_KINDS",
    "ASSUMPTION_CATEGORIES",
    "ASSUMPTION_SENSITIVITIES",
    "ASSUMPTION_STATUSES",
    "DATASET_ROLES",
    "EXPERIMENT_STATUSES",
    "FAILURE_CLASSES",
    "OUTCOME_CLASSES",
    "check_in",
    "jsonb",
    "new_uuid",
    "utcnow",
    "uuid_column",
]

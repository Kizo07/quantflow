"""SQLAlchemy binding of ``ExperimentSearchStore`` to the Phase 1 ORM models.

:class:`SQLExperimentSearchStore` implements the read boundary defined in
:mod:`deerflow.knowledge.search` over the real ``experiment`` table plus its
``experiment_dataset`` / ``experiment_artifact`` link tables (migration
``0022_knowledge_phase1``). Canonical store is PostgreSQL; SQLite works for
dev/test with identical query semantics.

Row-to-record mapping (``ExperimentRow`` -> ``ExperimentRecord``):

* ``experiment_family_hash`` / ``execution_hash`` / hypothesis / methodology /
  parameters / metrics / status / outcome / failure_class map 1:1.
* ``datasets`` come from ``experiment_dataset`` links as
  ``{"dataset_version_id": <uuid str>, "role": ...}`` (sorted by id, then
  role, for deterministic pages).
* ``code`` / ``environment`` expose the linked artifact ids as
  ``{"artifact_id": <uuid str>}`` (``{}`` when no artifact is linked): the
  Phase 1 schema models code/env identity as ``artifact`` foreign keys, not
  free-form JSON, so the full identity dicts live behind ``artifact`` rows.
* ``result_artifacts`` collects ``experiment_artifact`` links whose role is
  in :data:`RESULT_ARTIFACT_ROLES` (sorted for determinism).
* ``idempotency_key`` / ``commit_idempotency_key`` are always ``None``: the
  Phase 1 schema carries no idempotency columns (the write path binds in a
  later phase, which must extend the schema rather than overload these).
* ``started_at`` renders as ISO-8601 (``""`` when NULL);
  ``completed_at`` renders as ISO-8601 (``None`` when NULL). Naive
  datetimes (SQLite) are interpreted as UTC.

Ordering is oldest-first (``started_at ASC NULLS LAST``, ``id ASC`` tiebreak)
so pages are stable across dialects. ``hypothesis_contains`` is a
case-insensitive substring (``ILIKE`` with ``%``/``_``/``\\`` escaped);
lexical ranking is a Phase 2 concern.

Sessions: the store holds a synchronous session factory and opens one short
session per call, so a single instance is safe to share across threads (each
call gets its own ``Session``). :func:`open_search_store` builds a store from
a DSN, reusing the process-wide sync-engine cache in
:mod:`deerflow.persistence.agents.sql` (same pattern as the agent ``db``
store: one engine/pool per URL per process).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from deerflow.knowledge.schema.experiments import (
    ExperimentArtifactRow,
    ExperimentDatasetRow,
    ExperimentRow,
)
from deerflow.knowledge.search import ExperimentFilter
from deerflow.knowledge.write_api import ExperimentRecord

__all__ = [
    "RESULT_ARTIFACT_ROLES",
    "SQLExperimentSearchStore",
    "open_search_store",
    "row_to_record",
]

#: ``experiment_artifact`` roles that surface as ``ExperimentRecord.result_artifacts``.
RESULT_ARTIFACT_ROLES = frozenset({"result", "log", "chart"})


def _iso(value: datetime | None) -> str | None:
    """Render a timestamp as ISO-8601, interpreting naive values as UTC."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


def _like_escape(value: str) -> str:
    """Escape LIKE metacharacters so the substring match is literal."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def row_to_record(
    row: ExperimentRow,
    *,
    datasets: list[dict[str, str]] | None = None,
    result_artifacts: list[str] | None = None,
) -> ExperimentRecord:
    """Convert one ``ExperimentRow`` to an immutable ``ExperimentRecord``.

    Args:
        row: A mapped experiment row (detached instances are fine — only
            column attributes are read).
        datasets: Pre-fetched ``experiment_dataset`` links; when ``None`` an
            empty list is used (the store batch-fetches these per call).
        result_artifacts: Pre-fetched result artifact id strings; when
            ``None`` an empty list is used.

    Returns:
        The equivalent ``ExperimentRecord`` (see the module docstring for the
        exact field mapping).
    """
    code = {"artifact_id": str(row.code_artifact_id)} if row.code_artifact_id is not None else {}
    environment = {"artifact_id": str(row.environment_artifact_id)} if row.environment_artifact_id is not None else {}
    started = _iso(row.started_at)
    return ExperimentRecord(
        id=str(row.id),
        project_id=str(row.project_id),
        created_by_run_id=str(row.created_by_run_id),
        hypothesis=row.hypothesis,
        methodology=dict(row.methodology or {}),
        parameters=dict(row.parameters or {}),
        datasets=list(datasets or []),
        code=code,
        environment=environment,
        family_hash=row.experiment_family_hash,
        execution_hash=row.execution_hash,
        status=row.status,
        outcome=row.outcome,
        failure_class=row.failure_class,
        metrics=dict(row.metrics) if row.metrics is not None else None,
        result_artifacts=list(result_artifacts or []),
        parent_experiment_id=str(row.parent_experiment_id) if row.parent_experiment_id is not None else None,
        replicated_experiment_id=str(row.replicated_experiment_id) if row.replicated_experiment_id is not None else None,
        idempotency_key=None,
        commit_idempotency_key=None,
        started_at=started if started is not None else "",
        completed_at=_iso(row.completed_at),
    )


class SQLExperimentSearchStore:
    """``ExperimentSearchStore`` backed by the Phase 1 ORM models.

    Args:
        session_factory: Zero-argument callable returning a new synchronous
            SQLAlchemy ``Session`` (e.g. a ``sessionmaker``). One session is
            opened and closed per call.
    """

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._sessions = session_factory

    def find_by_execution_hash(self, execution_hash: str) -> ExperimentRecord | None:
        """Return the experiment with this execution hash, or None (unique lookup)."""
        with self._sessions() as session:
            row = session.scalars(select(ExperimentRow).where(ExperimentRow.execution_hash == execution_hash)).one_or_none()
            if row is None:
                return None
            links = self._fetch_links(session, [row.id])
            return row_to_record(row, datasets=links[0][0], result_artifacts=links[0][1])

    def find_by_family_hash(self, family_hash: str, *, limit: int, offset: int) -> list[ExperimentRecord]:
        """Return one page of experiments in this family, oldest first."""
        with self._sessions() as session:
            rows = list(session.scalars(self._ordered(select(ExperimentRow).where(ExperimentRow.experiment_family_hash == family_hash)).limit(limit).offset(offset)))
            return self._to_records(session, rows)

    def search_experiments(self, filters: ExperimentFilter, *, limit: int, offset: int) -> list[ExperimentRecord]:
        """Return one page of experiments matching all non-None filter fields, oldest first."""
        statement = select(ExperimentRow)
        if filters.family_hash is not None:
            statement = statement.where(ExperimentRow.experiment_family_hash == filters.family_hash)
        if filters.project_id is not None:
            statement = statement.where(ExperimentRow.project_id == uuid.UUID(filters.project_id))
        if filters.status is not None:
            statement = statement.where(ExperimentRow.status == filters.status)
        if filters.outcome is not None:
            statement = statement.where(ExperimentRow.outcome == filters.outcome)
        if filters.failure_class is not None:
            statement = statement.where(ExperimentRow.failure_class == filters.failure_class)
        if filters.hypothesis_contains is not None:
            statement = statement.where(ExperimentRow.hypothesis.ilike(f"%{_like_escape(filters.hypothesis_contains)}%", escape="\\"))
        with self._sessions() as session:
            rows = list(session.scalars(self._ordered(statement).limit(limit).offset(offset)))
            return self._to_records(session, rows)

    @staticmethod
    def _ordered(statement):  # type: ignore[no-untyped-def]
        """Apply the oldest-first page ordering (NULL started_at sorts last)."""
        return statement.order_by(ExperimentRow.started_at.asc().nulls_last(), ExperimentRow.id.asc())

    def _to_records(self, session: Session, rows: list[ExperimentRow]) -> list[ExperimentRecord]:
        """Convert rows to records with one batched link fetch (no N+1)."""
        links = self._fetch_links(session, [row.id for row in rows])
        return [row_to_record(row, datasets=datasets, result_artifacts=artifacts) for row, (datasets, artifacts) in zip(rows, links, strict=True)]

    @staticmethod
    def _fetch_links(session: Session, experiment_ids: list[uuid.UUID]) -> list[tuple[list[dict[str, str]], list[str]]]:
        """Batch-fetch dataset + result-artifact links for the given experiments.

        Returns one ``(datasets, result_artifacts)`` pair per id, in input
        order. Both lists are sorted for deterministic pages.
        """
        datasets_by_experiment: dict[uuid.UUID, list[dict[str, str]]] = {experiment_id: [] for experiment_id in experiment_ids}
        artifacts_by_experiment: dict[uuid.UUID, list[str]] = {experiment_id: [] for experiment_id in experiment_ids}
        if not experiment_ids:
            return []
        for dataset_row in session.scalars(select(ExperimentDatasetRow).where(ExperimentDatasetRow.experiment_id.in_(experiment_ids))):
            datasets_by_experiment[dataset_row.experiment_id].append({"dataset_version_id": str(dataset_row.dataset_version_id), "role": dataset_row.role})
        for artifact_row in session.scalars(select(ExperimentArtifactRow).where(ExperimentArtifactRow.experiment_id.in_(experiment_ids))):
            if artifact_row.role in RESULT_ARTIFACT_ROLES:
                artifacts_by_experiment[artifact_row.experiment_id].append(str(artifact_row.artifact_id))
        return [
            (
                sorted(datasets_by_experiment[experiment_id], key=lambda link: (link["dataset_version_id"], link["role"])),
                sorted(artifacts_by_experiment[experiment_id]),
            )
            for experiment_id in experiment_ids
        ]


def open_search_store(dsn: str) -> SQLExperimentSearchStore:
    """Build a search store from a SQLAlchemy DSN.

    Reuses the process-wide sync-engine cache (one engine/pool per URL), so
    repeated calls with the same DSN share connections.

    Args:
        dsn: SQLAlchemy URL, e.g. ``postgresql+psycopg://...`` (canonical) or
            ``sqlite:////path/to/kb.db`` (dev/test).

    Returns:
        A ``SQLExperimentSearchStore`` bound to the database at ``dsn``.
    """
    from deerflow.persistence.agents.sql import get_sync_sessionmaker

    return SQLExperimentSearchStore(get_sync_sessionmaker(dsn))

"""Tests for the SQLAlchemy ``ExperimentSearchStore`` binding (integration).

Covers :mod:`deerflow.knowledge.pg_store` against a file-backed SQLite
database (same query code paths as PostgreSQL; SQLite is the dev/test
dialect per the schema conventions):

* ``ExperimentRow`` -> ``ExperimentRecord`` field mapping (datasets, code /
  environment artifact pointers, result roles, ISO timestamps);
* ``find_by_execution_hash`` unique lookup (hit + miss);
* ``find_by_family_hash`` oldest-first ordering + limit/offset pagination;
* ``search_experiments`` structured filters (AND-combined) and
  case-insensitive literal ``hypothesis_contains``;
* end-to-end ``experiment_search`` / ``experiment_get_by_execution_hash``
  over the PG-backed store;
* ``open_search_store`` DSN helper smoke test.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

import deerflow.knowledge.schema as kb_schema
from deerflow.knowledge.pg_store import (
    RESULT_ARTIFACT_ROLES,
    SQLExperimentSearchStore,
    open_search_store,
    row_to_record,
)
from deerflow.knowledge.search import (
    ExperimentFilter,
    ExperimentSearchStore,
    experiment_get_by_execution_hash,
    experiment_search,
)
from deerflow.persistence.base import Base

KB_TABLES: tuple[str, ...] = (
    "research_project",
    "agent_run",
    "artifact",
    "dataset_version",
    "experiment",
    "experiment_dataset",
    "experiment_artifact",
    "assumption",
)

FAMILY_A = "aa" * 32
FAMILY_B = "bb" * 32


def _engine(tmp_path: Path, name: str = "kb_search.db") -> sa.Engine:
    engine = sa.create_engine(f"sqlite:///{(tmp_path / name).as_posix()}")
    for table in KB_TABLES:
        Base.metadata.tables[table].create(engine, checkfirst=True)
    return engine


def _seed() -> dict[str, object]:
    """Build one project + run + artifacts + dataset versions (explicit UUIDs)."""
    project_id = uuid.uuid4()
    run_id = uuid.uuid4()
    code_id = uuid.uuid4()
    env_id = uuid.uuid4()
    result_id = uuid.uuid4()
    log_id = uuid.uuid4()
    chart_id = uuid.uuid4()
    notebook_id = uuid.uuid4()
    dataset_v1 = uuid.uuid4()
    dataset_v2 = uuid.uuid4()
    return {
        "project": kb_schema.ResearchProjectRow(id=project_id, name="momentum", visibility_scope={}),
        "run": kb_schema.AgentRunRow(id=run_id, project_id=project_id, agent_type="research", task="search binding", status="running"),
        "code": kb_schema.ArtifactRow(id=code_id, sha256="c0" * 32, kind="code", storage_uri="s3://evidence/code", byte_size=10, created_by_run_id=run_id),
        "env": kb_schema.ArtifactRow(id=env_id, sha256="e0" * 32, kind="environment", storage_uri="s3://evidence/env", byte_size=20, created_by_run_id=run_id),
        "result": kb_schema.ArtifactRow(id=result_id, sha256="d0" * 32, kind="result", storage_uri="s3://evidence/result", byte_size=30, created_by_run_id=run_id),
        "log": kb_schema.ArtifactRow(id=log_id, sha256="d1" * 32, kind="log", storage_uri="s3://evidence/log", byte_size=31, created_by_run_id=run_id),
        "chart": kb_schema.ArtifactRow(id=chart_id, sha256="d2" * 32, kind="chart", storage_uri="s3://evidence/chart", byte_size=32, created_by_run_id=run_id),
        "notebook": kb_schema.ArtifactRow(id=notebook_id, sha256="d3" * 32, kind="notebook", storage_uri="s3://evidence/nb", byte_size=33, created_by_run_id=run_id),
        "dataset_v1": kb_schema.DatasetVersionRow(id=dataset_v1, dataset_key="sp500-daily", provider="lse", lineage={}, access_metadata={}),
        "dataset_v2": kb_schema.DatasetVersionRow(id=dataset_v2, dataset_key="sp500-daily", provider="lse", lineage={}, access_metadata={}),
    }


def _experiment(
    seed: dict[str, object],
    *,
    execution_hash: str,
    family_hash: str = FAMILY_A,
    hypothesis: str = "Momentum persists net of costs.",
    status: str = "completed",
    outcome: str | None = "success",
    failure_class: str | None = None,
    started_at: datetime | None = datetime(2026, 1, 10, 12, 0, tzinfo=UTC),
    completed_at: datetime | None = datetime(2026, 1, 11, 12, 0, tzinfo=UTC),
    with_links: bool = True,
) -> kb_schema.ExperimentRow:
    project = seed["project"]
    run = seed["run"]
    assert isinstance(project, kb_schema.ResearchProjectRow)
    assert isinstance(run, kb_schema.AgentRunRow)
    code = seed["code"]
    env = seed["env"]
    assert isinstance(code, kb_schema.ArtifactRow)
    assert isinstance(env, kb_schema.ArtifactRow)
    row = kb_schema.ExperimentRow(
        id=uuid.uuid4(),
        project_id=project.id,
        created_by_run_id=run.id,
        experiment_family_hash=family_hash,
        execution_hash=execution_hash,
        hypothesis=hypothesis,
        methodology={"universe": "sp500-pit", "horizon": "6m"},
        parameters={"lookback_days": 126, "seed": 7},
        metrics={"sharpe": 0.82} if outcome is not None else None,
        code_artifact_id=code.id if with_links else None,
        environment_artifact_id=env.id if with_links else None,
        outcome=outcome,
        failure_class=failure_class,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
    )
    return row


def _link_datasets(seed: dict[str, object], experiment_id: uuid.UUID) -> list[kb_schema.ExperimentDatasetRow]:
    v1 = seed["dataset_v1"]
    v2 = seed["dataset_v2"]
    assert isinstance(v1, kb_schema.DatasetVersionRow)
    assert isinstance(v2, kb_schema.DatasetVersionRow)
    return [
        kb_schema.ExperimentDatasetRow(experiment_id=experiment_id, dataset_version_id=v1.id, role="features"),
        kb_schema.ExperimentDatasetRow(experiment_id=experiment_id, dataset_version_id=v2.id, role="labels"),
    ]


def _link_artifacts(seed: dict[str, object], experiment_id: uuid.UUID) -> list[kb_schema.ExperimentArtifactRow]:
    rows = []
    for key, role in (("result", "result"), ("log", "log"), ("chart", "chart"), ("notebook", "notebook"), ("code", "code")):
        artifact = seed[key]
        assert isinstance(artifact, kb_schema.ArtifactRow)
        rows.append(kb_schema.ExperimentArtifactRow(experiment_id=experiment_id, artifact_id=artifact.id, role=role))
    return rows


def _store(engine: sa.Engine) -> SQLExperimentSearchStore:
    factory: sessionmaker[Session] = sessionmaker(bind=engine, expire_on_commit=False)
    return SQLExperimentSearchStore(factory)


def _seed_session(engine: sa.Engine) -> Session:
    """Return a seeding session that keeps attributes usable after commit."""
    return Session(engine, expire_on_commit=False)


def _seed_full(engine: sa.Engine) -> dict[str, object]:
    seed = _seed()
    with _seed_session(engine) as session:
        session.add_all(list(seed.values()))
        session.commit()
    return seed


def test_result_roles_cover_outputs_logs_and_charts() -> None:
    assert RESULT_ARTIFACT_ROLES == frozenset({"result", "log", "chart"})


def test_find_by_execution_hash_round_trip(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    seed = _seed_full(engine)
    with _seed_session(engine) as session:
        row = _experiment(seed, execution_hash="01" * 32)
        session.add(row)
        session.add_all(_link_datasets(seed, row.id))
        session.add_all(_link_artifacts(seed, row.id))
        session.commit()
        experiment_id = row.id

    record = _store(engine).find_by_execution_hash("01" * 32)

    assert record is not None
    assert record.id == str(experiment_id)
    assert record.project_id == str(seed["project"].id)
    assert record.created_by_run_id == str(seed["run"].id)
    assert record.hypothesis == "Momentum persists net of costs."
    assert record.methodology == {"universe": "sp500-pit", "horizon": "6m"}
    assert record.parameters == {"lookback_days": 126, "seed": 7}
    assert record.metrics == {"sharpe": 0.82}
    assert record.family_hash == FAMILY_A
    assert record.execution_hash == "01" * 32
    assert record.status == "completed"
    assert record.outcome == "success"
    assert record.failure_class is None
    assert record.datasets == sorted(
        [
            {"dataset_version_id": str(seed["dataset_v1"].id), "role": "features"},
            {"dataset_version_id": str(seed["dataset_v2"].id), "role": "labels"},
        ],
        key=lambda link: (link["dataset_version_id"], link["role"]),
    )
    assert record.code == {"artifact_id": str(seed["code"].id)}
    assert record.environment == {"artifact_id": str(seed["env"].id)}
    # Only result/log/chart roles surface; notebook + code links are excluded.
    assert record.result_artifacts == sorted([str(seed["result"].id), str(seed["log"].id), str(seed["chart"].id)])
    assert record.parent_experiment_id is None
    assert record.replicated_experiment_id is None
    assert record.idempotency_key is None
    assert record.commit_idempotency_key is None
    assert record.started_at == "2026-01-10T12:00:00+00:00"
    assert record.completed_at == "2026-01-11T12:00:00+00:00"


def test_find_by_execution_hash_missing_returns_none(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _seed_full(engine)
    assert _store(engine).find_by_execution_hash("ff" * 32) is None


def test_row_to_record_without_links_uses_empty_defaults() -> None:
    row = kb_schema.ExperimentRow(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        created_by_run_id=uuid.uuid4(),
        experiment_family_hash=FAMILY_A,
        execution_hash="02" * 32,
        hypothesis="Sparse row.",
        methodology={},
        parameters={},
        status="planned",
    )
    record = row_to_record(row)
    assert record.datasets == []
    assert record.code == {}
    assert record.environment == {}
    assert record.metrics is None
    assert record.result_artifacts == []
    assert record.started_at == ""
    assert record.completed_at is None


def test_find_by_family_hash_orders_oldest_first_with_pagination(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    seed = _seed_full(engine)
    with _seed_session(engine) as session:
        old = _experiment(seed, execution_hash="10" * 32, started_at=datetime(2026, 1, 1, tzinfo=UTC))
        new = _experiment(seed, execution_hash="11" * 32, started_at=datetime(2026, 3, 1, tzinfo=UTC))
        undated = _experiment(seed, execution_hash="12" * 32, started_at=None, status="planned", outcome=None, completed_at=None)
        other_family = _experiment(seed, execution_hash="13" * 32, family_hash=FAMILY_B, started_at=datetime(2025, 1, 1, tzinfo=UTC))
        session.add_all([old, new, undated, other_family])
        session.commit()

    store = _store(engine)
    page = store.find_by_family_hash(FAMILY_A, limit=10, offset=0)
    assert [rec.execution_hash for rec in page] == ["10" * 32, "11" * 32, "12" * 32]
    assert [rec.execution_hash for rec in store.find_by_family_hash(FAMILY_A, limit=1, offset=1)] == ["11" * 32]
    assert store.find_by_family_hash(FAMILY_A, limit=10, offset=99) == []
    assert [rec.execution_hash for rec in store.find_by_family_hash(FAMILY_B, limit=10, offset=0)] == ["13" * 32]


def test_search_experiments_combines_structured_filters(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    seed = _seed_full(engine)
    other_project = kb_schema.ResearchProjectRow(id=uuid.uuid4(), name="other", visibility_scope={})
    with _seed_session(engine) as session:
        session.add(other_project)
        session.flush()
        failed = _experiment(
            seed,
            execution_hash="20" * 32,
            hypothesis="Costs erase momentum.",
            status="failed",
            outcome="failure",
            failure_class="execution",
        )
        ok = _experiment(seed, execution_hash="21" * 32, hypothesis="Momentum survives costs.")
        session.add_all([failed, ok])
        session.commit()
        project_id = str(seed["project"].id)

    store = _store(engine)
    assert [rec.execution_hash for rec in store.search_experiments(ExperimentFilter(status="failed"), limit=10, offset=0)] == ["20" * 32]
    assert [rec.execution_hash for rec in store.search_experiments(ExperimentFilter(outcome="failure", failure_class="execution"), limit=10, offset=0)] == ["20" * 32]
    assert store.search_experiments(ExperimentFilter(outcome="failure", failure_class="data"), limit=10, offset=0) == []
    assert {rec.execution_hash for rec in store.search_experiments(ExperimentFilter(project_id=project_id), limit=10, offset=0)} == {"20" * 32, "21" * 32}
    assert store.search_experiments(ExperimentFilter(project_id=str(other_project.id)), limit=10, offset=0) == []
    both = store.search_experiments(ExperimentFilter(), limit=10, offset=0)
    assert {rec.execution_hash for rec in both} == {"20" * 32, "21" * 32}


def test_search_hypothesis_contains_is_case_insensitive_and_literal(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    seed = _seed_full(engine)
    with _seed_session(engine) as session:
        session.add(_experiment(seed, execution_hash="30" * 32, hypothesis="Momentum 100% net of costs_Monday."))
        session.add(_experiment(seed, execution_hash="31" * 32, hypothesis="Unrelated value study."))
        session.commit()

    store = _store(engine)
    assert [rec.execution_hash for rec in store.search_experiments(ExperimentFilter(hypothesis_contains="momentum 100%"), limit=10, offset=0)] == ["30" * 32]
    # LIKE metacharacters match literally: "100_" must NOT match "100%".
    assert store.search_experiments(ExperimentFilter(hypothesis_contains="100_"), limit=10, offset=0) == []
    assert [rec.execution_hash for rec in store.search_experiments(ExperimentFilter(hypothesis_contains="COSTS_monday"), limit=10, offset=0)] == ["30" * 32]


def test_experiment_search_end_to_end_over_pg_store(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    seed = _seed_full(engine)
    with _seed_session(engine) as session:
        session.add(_experiment(seed, execution_hash="40" * 32, hypothesis="Family member one."))
        session.add(_experiment(seed, execution_hash="41" * 32, hypothesis="Family member two.", status="failed", outcome="failure", failure_class="code"))
        session.commit()

    store = _store(engine)
    page = experiment_search(store, family_hash=FAMILY_A)
    assert page.family_hash_used == FAMILY_A
    assert {rec.execution_hash for rec in page.experiments} == {"40" * 32, "41" * 32}

    failed_only = experiment_search(store, family_hash=FAMILY_A, status="failed")
    assert [rec.execution_hash for rec in failed_only.experiments] == ["41" * 32]

    assert experiment_get_by_execution_hash(store, "40" * 32) is not None
    assert experiment_get_by_execution_hash(store, "ff" * 32) is None


def test_store_satisfies_search_protocol(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    assert isinstance(_store(engine), ExperimentSearchStore)


def test_open_search_store_from_sqlite_dsn(tmp_path: Path) -> None:
    db_path = tmp_path / "dsn.db"
    dsn = f"sqlite:///{db_path.as_posix()}"
    setup_engine = sa.create_engine(dsn)
    for table in KB_TABLES:
        Base.metadata.tables[table].create(setup_engine, checkfirst=True)
    seed = _seed()
    with Session(setup_engine) as session:
        session.add_all(list(seed.values()))
        session.add(_experiment(seed, execution_hash="50" * 32))
        session.commit()
    setup_engine.dispose()

    store = open_search_store(dsn)
    record = store.find_by_execution_hash("50" * 32)
    assert record is not None
    assert record.family_hash == FAMILY_A
    assert record.datasets == []

"""Tests for the Knowledge Plane Phase 1 schema (KB episodic half).

Covers the eight ORM models in ``deerflow.knowledge.schema`` and alembic
revision ``0022_knowledge_phase1``:

* revision chain integrity (direct child of ``0019_thread_incarnations``,
  single alembic head);
* ``Base.metadata.create_all`` + full object-graph persistence across all
  eight tables, including JSONB payload and UUID default round-trips;
* contract enforcement: unique ``sha256`` / ``execution_hash``, enum
  CHECK constraints, and foreign keys;
* migration ``upgrade()`` / ``downgrade()`` against a scratch SQLite
  database (idempotent re-run included);
* PostgreSQL DDL compilation (JSONB / UUID / TIMESTAMPTZ rendering).

Standalone: uses a file-backed SQLite database under ``tmp_path`` and
imports only the new schema modules plus the migration under test. No
PostgreSQL server or driver is required.
"""

from __future__ import annotations

import importlib
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import deerflow.knowledge.schema.evidence as evidence
import deerflow.knowledge.schema.experiments as experiments
import deerflow.knowledge.schema.research as research
from deerflow.knowledge.schema.evidence import ArtifactRow, DatasetVersionRow
from deerflow.knowledge.schema.experiments import (
    AssumptionRow,
    ExperimentArtifactRow,
    ExperimentDatasetRow,
    ExperimentRow,
)
from deerflow.knowledge.schema.research import AgentRunRow, ResearchProjectRow
from deerflow.persistence.base import Base
from deerflow.persistence.migrations import _helpers as migration_helpers

# Digit-prefixed migration modules are not importable with a plain import
# statement; importlib handles the dotted name fine.
migration = importlib.import_module("deerflow.persistence.migrations.versions.0022_knowledge_phase1")
migration_0028 = importlib.import_module("deerflow.persistence.migrations.versions.0028_knowledge_experiment_embeddings")

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


def _engine(tmp_path: Path, name: str = "kb.db") -> sa.Engine:
    return sa.create_engine(f"sqlite:///{(tmp_path / name).as_posix()}")


def _seed_graph() -> dict[str, object]:
    """Build a connected object graph spanning all eight tables.

    Every row gets an explicit UUID up front: Python-side ``default=``
    values only materialize at flush time, so FK references must use
    pre-assigned ids rather than reading ``.id`` off unflushed peers.
    """
    project = ResearchProjectRow(id=uuid.uuid4(), name="momentum", description="XS momentum", visibility_scope={"org": "quantflow"})
    run = AgentRunRow(id=uuid.uuid4(), project_id=project.id, agent_type="research", task="test momentum", status="running")
    code = ArtifactRow(
        id=uuid.uuid4(),
        sha256="c" * 64,
        kind="code",
        storage_uri="s3://kb/code/bundle.zip",
        media_type="application/zip",
        byte_size=128,
        created_by_run_id=run.id,
        artifact_metadata={"git_commit": "abc123"},
    )
    dataset = DatasetVersionRow(
        id=uuid.uuid4(),
        dataset_key="crsp-daily",
        provider="crsp",
        vintage_at=datetime(2026, 1, 31, tzinfo=UTC),
        coverage_start=datetime(2000, 1, 1, tzinfo=UTC),
        coverage_end=datetime(2025, 12, 31, tzinfo=UTC),
        schema_hash="schema-1",
        artifact_id=code.id,
        lineage={"source": "vendor"},
        access_metadata={"license": "internal"},
    )
    experiment = ExperimentRow(
        id=uuid.uuid4(),
        project_id=project.id,
        created_by_run_id=run.id,
        experiment_family_hash="family-1",
        execution_hash="exec-1",
        hypothesis="12-1 momentum survives costs",
        methodology={"universe": "liquid-common-stocks", "frequency": "monthly"},
        parameters={"lookback": 252, "skip": 21},
        metrics={"sharpe": 0.9},
        code_artifact_id=code.id,
        outcome="success",
        failure_class=None,
        status="completed",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        completed_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    link_ds = ExperimentDatasetRow(experiment_id=experiment.id, dataset_version_id=dataset.id, role="features")
    link_artifact = ExperimentArtifactRow(experiment_id=experiment.id, artifact_id=code.id, role="code")
    assumption = AssumptionRow(
        id=uuid.uuid4(),
        experiment_id=experiment.id,
        statement="Zero transaction costs",
        category="execution",
        sensitivity="high",
        tested=False,
        status="active",
        evidence_artifact_id=None,
    )
    return {
        "project": project,
        "run": run,
        "code": code,
        "dataset": dataset,
        "experiment": experiment,
        "link_ds": link_ds,
        "link_artifact": link_artifact,
        "assumption": assumption,
    }


def _run_migration(engine: sa.Engine, direction: str, module=None) -> None:
    """Execute a migration module's upgrade/downgrade against *engine*.

    Rebinds the module's ``op`` proxy to an ``Operations`` bound to this
    connection (public API only; no private patching). Defaults to the
    0022 revision under test; pass ``migration_0028`` for the Phase 3
    experiment-embedding column.
    """
    target = module if module is not None else migration
    with engine.begin() as connection:
        context = MigrationContext.configure(connection, opts={"render_as_batch": True})
        ops = Operations(context)
        original_op = target.op
        original_helpers_op = migration_helpers.op
        target.op = ops
        # Column revisions (0028) act through ``_helpers.safe_*``; their
        # module holds its own ``op`` proxy, so bind it as well.
        migration_helpers.op = ops
        try:
            if direction == "upgrade":
                target.upgrade()
            else:
                target.downgrade()
        finally:
            target.op = original_op
            migration_helpers.op = original_helpers_op


class TestRevisionChain:
    def test_revision_attributes(self) -> None:
        assert migration.revision == "0022_knowledge_phase1"
        assert migration.down_revision == "0019_thread_incarnations"

    def test_single_head_is_new_revision(self) -> None:
        # Phase 2 landed revision 0023 on top of this one, the upstream
        # merge added 0027 on top of that, and Phase 3 added 0028; the
        # head pin follows the chain tip (single head, no branches).
        migrations_dir = Path(migration.__file__).resolve().parent.parent
        config = Config()
        config.set_main_option("script_location", migrations_dir.as_posix())
        script = ScriptDirectory.from_config(config)
        assert script.get_heads() == ["0028_knowledge_experiment_embeddings"]


class TestModels:
    def test_tables_registered_on_base(self) -> None:
        for table in KB_TABLES:
            assert table in Base.metadata.tables, f"{table} missing from Base.metadata"

    def test_full_graph_round_trip(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        try:
            Base.metadata.create_all(engine)
            graph = _seed_graph()
            with Session(engine, expire_on_commit=False) as session:
                session.add_all(
                    [
                        graph["project"],
                        graph["run"],
                        graph["code"],
                        graph["dataset"],
                        graph["experiment"],
                        graph["link_ds"],
                        graph["link_artifact"],
                        graph["assumption"],
                    ]
                )
                session.commit()

            with Session(engine, expire_on_commit=False) as session:
                experiment = session.get(ExperimentRow, graph["experiment"].id)
                assert experiment is not None
                assert experiment.methodology == {"universe": "liquid-common-stocks", "frequency": "monthly"}
                assert experiment.parameters == {"lookback": 252, "skip": 21}
                assert experiment.metrics == {"sharpe": 0.9}
                assert experiment.outcome == "success"
                assert experiment.failure_class is None
                assert experiment.status == "completed"
                assert isinstance(experiment.id, uuid.UUID)

                artifact = session.get(ArtifactRow, graph["code"].id)
                assert artifact is not None
                assert artifact.artifact_metadata == {"git_commit": "abc123"}
                assert ArtifactRow.__table__.c["metadata"] is not None

                dataset = session.get(DatasetVersionRow, graph["dataset"].id)
                assert dataset is not None
                assert dataset.lineage == {"source": "vendor"}
                assert dataset.access_metadata == {"license": "internal"}

                assumption = session.get(AssumptionRow, graph["assumption"].id)
                assert assumption is not None
                assert assumption.tested is False
                assert assumption.sensitivity == "high"

                links = session.query(ExperimentDatasetRow).all()
                assert [(row.experiment_id, row.dataset_version_id, row.role) for row in links] == [(experiment.id, dataset.id, "features")]

                payload = experiment.to_dict()
                assert payload["execution_hash"] == "exec-1"
                assert payload["experiment_family_hash"] == "family-1"
        finally:
            engine.dispose()

    def test_python_side_defaults(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        try:
            Base.metadata.create_all(engine)
            with Session(engine, expire_on_commit=False) as session:
                project = ResearchProjectRow(name="defaults")
                session.add(project)
                session.commit()
                assert isinstance(project.id, uuid.UUID)
                assert project.visibility_scope == {}
                assert project.created_at is not None
                assert project.created_at.tzinfo is not None

                run = AgentRunRow(project_id=project.id, agent_type="t", task="t")
                session.add(run)
                session.commit()
                assert run.status == "running"
                assert run.started_at is not None
        finally:
            engine.dispose()


class TestContractEnforcement:
    def _seeded_engine(self, tmp_path: Path) -> tuple[sa.Engine, dict[str, object]]:
        engine = _engine(tmp_path)
        Base.metadata.create_all(engine)
        graph = _seed_graph()
        with Session(engine, expire_on_commit=False) as session:
            session.add_all(
                [
                    graph["project"],
                    graph["run"],
                    graph["code"],
                    graph["dataset"],
                    graph["experiment"],
                    graph["link_ds"],
                    graph["link_artifact"],
                    graph["assumption"],
                ]
            )
            session.commit()
        return engine, graph

    def test_duplicate_artifact_sha256_rejected(self, tmp_path: Path) -> None:
        engine, graph = self._seeded_engine(tmp_path)
        try:
            with Session(engine, expire_on_commit=False) as session:
                session.add(
                    ArtifactRow(
                        sha256="c" * 64,
                        kind="log",
                        storage_uri="s3://kb/other",
                        created_by_run_id=graph["run"].id,
                    )
                )
                with pytest.raises(IntegrityError):
                    session.commit()
        finally:
            engine.dispose()

    def test_duplicate_execution_hash_rejected(self, tmp_path: Path) -> None:
        engine, graph = self._seeded_engine(tmp_path)
        try:
            with Session(engine, expire_on_commit=False) as session:
                session.add(
                    ExperimentRow(
                        project_id=graph["project"].id,
                        created_by_run_id=graph["run"].id,
                        experiment_family_hash="other-family",
                        execution_hash="exec-1",
                        hypothesis="dup",
                        methodology={},
                        parameters={},
                        status="planned",
                    )
                )
                with pytest.raises(IntegrityError):
                    session.commit()
        finally:
            engine.dispose()

    def test_same_family_hash_allowed(self, tmp_path: Path) -> None:
        """Family hash groups replications; only execution hash is unique."""
        engine, graph = self._seeded_engine(tmp_path)
        try:
            with Session(engine, expire_on_commit=False) as session:
                session.add(
                    ExperimentRow(
                        project_id=graph["project"].id,
                        created_by_run_id=graph["run"].id,
                        experiment_family_hash="family-1",
                        execution_hash="exec-2",
                        hypothesis="replication",
                        methodology={},
                        parameters={},
                        status="planned",
                    )
                )
                session.commit()
            with Session(engine, expire_on_commit=False) as session:
                rows = session.query(ExperimentRow).filter_by(experiment_family_hash="family-1").all()
                assert {row.execution_hash for row in rows} == {"exec-1", "exec-2"}
        finally:
            engine.dispose()

    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(lambda g: ArtifactRow(sha256="d" * 64, kind="bogus-kind", storage_uri="s3://x"), id="artifact-kind"),
            pytest.param(
                lambda g: ExperimentRow(
                    project_id=g["project"].id,
                    created_by_run_id=g["run"].id,
                    experiment_family_hash="f",
                    execution_hash="e-bad-status",
                    hypothesis="h",
                    methodology={},
                    parameters={},
                    status="bogus",
                ),
                id="experiment-status",
            ),
            pytest.param(
                lambda g: ExperimentRow(
                    project_id=g["project"].id,
                    created_by_run_id=g["run"].id,
                    experiment_family_hash="f",
                    execution_hash="e-bad-outcome",
                    hypothesis="h",
                    methodology={},
                    parameters={},
                    status="completed",
                    outcome="bogus",
                ),
                id="experiment-outcome",
            ),
            pytest.param(
                lambda g: ExperimentRow(
                    project_id=g["project"].id,
                    created_by_run_id=g["run"].id,
                    experiment_family_hash="f",
                    execution_hash="e-bad-failure",
                    hypothesis="h",
                    methodology={},
                    parameters={},
                    status="failed",
                    outcome="failure",
                    failure_class="bogus",
                ),
                id="experiment-failure-class",
            ),
            pytest.param(
                lambda g: ExperimentDatasetRow(
                    experiment_id=g["experiment"].id,
                    dataset_version_id=g["dataset"].id,
                    role="bogus-role",
                ),
                id="dataset-role",
            ),
            pytest.param(
                lambda g: AssumptionRow(experiment_id=g["experiment"].id, statement="s", category="bogus", tested=False, status="active"),
                id="assumption-category",
            ),
            pytest.param(
                lambda g: AssumptionRow(
                    experiment_id=g["experiment"].id,
                    statement="s",
                    category="data",
                    sensitivity="bogus",
                    tested=False,
                    status="active",
                ),
                id="assumption-sensitivity",
            ),
            pytest.param(
                lambda g: AssumptionRow(experiment_id=g["experiment"].id, statement="s", category="data", tested=False, status="bogus"),
                id="assumption-status",
            ),
            pytest.param(
                lambda g: AgentRunRow(project_id=g["project"].id, agent_type="t", task="t", status="bogus"),
                id="agent-run-status",
            ),
        ],
    )
    def test_enum_check_constraints_reject_bad_values(self, tmp_path: Path, factory) -> None:
        engine, graph = self._seeded_engine(tmp_path)
        try:
            with Session(engine, expire_on_commit=False) as session:
                session.add(factory(graph))
                with pytest.raises(IntegrityError):
                    session.commit()
        finally:
            engine.dispose()

    def test_experiment_requires_project_fk(self, tmp_path: Path) -> None:
        engine, graph = self._seeded_engine(tmp_path)
        try:
            with engine.begin() as connection:
                connection.execute(sa.text("PRAGMA foreign_keys=ON"))
            with Session(engine, expire_on_commit=False) as session:
                session.add(
                    ExperimentRow(
                        project_id=uuid.uuid4(),
                        created_by_run_id=graph["run"].id,
                        experiment_family_hash="f",
                        execution_hash="e-no-project",
                        hypothesis="h",
                        methodology={},
                        parameters={},
                        status="planned",
                    )
                )
                with pytest.raises(IntegrityError):
                    session.commit()
        finally:
            engine.dispose()

    def test_nullable_outcome_and_failure_class_accepted(self, tmp_path: Path) -> None:
        engine, graph = self._seeded_engine(tmp_path)
        try:
            with Session(engine, expire_on_commit=False) as session:
                row = ExperimentRow(
                    project_id=graph["project"].id,
                    created_by_run_id=graph["run"].id,
                    experiment_family_hash="f-null",
                    execution_hash="e-null",
                    hypothesis="h",
                    methodology={},
                    parameters={},
                    status="running",
                    outcome=None,
                    failure_class=None,
                )
                session.add(row)
                session.commit()
                assert row.id is not None
        finally:
            engine.dispose()


class TestMigration:
    def test_upgrade_creates_all_tables_and_indexes(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path, "mig.db")
        try:
            _run_migration(engine, "upgrade")
            inspector = sa.inspect(engine)
            tables = set(inspector.get_table_names())
            assert set(KB_TABLES) <= tables

            index_names = {index["name"] for index in inspector.get_indexes("experiment")}
            assert {"ix_experiment_family_hash", "ix_experiment_project", "ix_experiment_created_by_run", "ix_experiment_status"} <= index_names

            uniques = {constraint["name"] for constraint in inspector.get_unique_constraints("experiment")}
            assert "uq_experiment_execution_hash" in uniques
            artifact_uniques = {constraint["name"] for constraint in inspector.get_unique_constraints("artifact")}
            assert "uq_artifact_sha256" in artifact_uniques
        finally:
            engine.dispose()

    def test_upgrade_is_idempotent(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path, "mig-idem.db")
        try:
            _run_migration(engine, "upgrade")
            _run_migration(engine, "upgrade")
            inspector = sa.inspect(engine)
            assert set(KB_TABLES) <= set(inspector.get_table_names())
        finally:
            engine.dispose()

    def test_downgrade_drops_all_tables(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path, "mig-down.db")
        try:
            _run_migration(engine, "upgrade")
            _run_migration(engine, "downgrade")
            tables = set(sa.inspect(engine).get_table_names())
            assert set(KB_TABLES).isdisjoint(tables)
            # Downgrade is also re-runnable.
            _run_migration(engine, "downgrade")
        finally:
            engine.dispose()

    def test_migrated_schema_accepts_orm_writes(self, tmp_path: Path) -> None:
        """Migration DDL and ORM models agree: write via models on migrated DDL."""
        engine = _engine(tmp_path, "mig-orm.db")
        try:
            _run_migration(engine, "upgrade")
            # The ORM carries the Phase 3 ``experiment.embedding`` column,
            # so the migrated schema needs revision 0028 on top of 0022.
            _run_migration(engine, "upgrade", migration_0028)
            graph = _seed_graph()
            with Session(engine, expire_on_commit=False) as session:
                session.add_all(
                    [
                        graph["project"],
                        graph["run"],
                        graph["code"],
                        graph["dataset"],
                        graph["experiment"],
                        graph["link_ds"],
                        graph["link_artifact"],
                        graph["assumption"],
                    ]
                )
                session.commit()
        finally:
            engine.dispose()

    def test_postgres_ddl_compilation(self) -> None:
        """ORM tables compile to PG DDL with JSONB / UUID / TIMESTAMPTZ."""
        dialect = postgresql.dialect()
        ddl = "\n".join(str(sa.schema.CreateTable(Base.metadata.tables[table]).compile(dialect=dialect)) for table in KB_TABLES)
        assert "JSONB" in ddl
        assert "UUID" in ddl
        assert "TIMESTAMP WITH TIME ZONE" in ddl
        assert "uq_experiment_execution_hash" in ddl
        assert "uq_artifact_sha256" in ddl
        assert "ck_experiment_failure_class" in ddl

    def test_schema_modules_expose_expected_models(self) -> None:
        assert research.ResearchProjectRow.__tablename__ == "research_project"
        assert research.AgentRunRow.__tablename__ == "agent_run"
        assert evidence.ArtifactRow.__tablename__ == "artifact"
        assert evidence.DatasetVersionRow.__tablename__ == "dataset_version"
        assert experiments.ExperimentRow.__tablename__ == "experiment"
        assert experiments.ExperimentDatasetRow.__tablename__ == "experiment_dataset"
        assert experiments.ExperimentArtifactRow.__tablename__ == "experiment_artifact"
        assert experiments.AssumptionRow.__tablename__ == "assumption"

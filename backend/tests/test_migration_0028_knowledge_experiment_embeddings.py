"""Tests for the Knowledge Plane Phase 3 experiment embedding column.

Covers ``ExperimentRow.embedding`` in
``deerflow.knowledge.schema.experiments`` and alembic revision
``0028_knowledge_experiment_embeddings``:

* revision chain integrity (direct child of
  ``0027_merge_knowledge_upstream``, single alembic head — this file owns
  the chain-head pin);
* ``Base.metadata.create_all`` + experiment persistence, including exact
  embedding round-trip through the SQLite JSON fallback;
* migration ``upgrade()`` / ``downgrade()`` against a scratch SQLite
  database (idempotent re-run included), including the chained
  0022-then-0028 upgrade;
* ORM/migration DDL parity for the ``experiment`` table (columns,
  nullability, defaults, type affinity, checks, uniques, indexes) —
  the per-table equivalent of the persistence-bootstrap hard gate;
* PostgreSQL DDL compilation (VECTOR(768) rendering).

Standalone: uses file-backed SQLite databases under ``tmp_path``. No
PostgreSQL server or driver is required, and the ``pgvector`` Python
package must not be needed.
"""

from __future__ import annotations

import importlib
import importlib.util
import uuid
from pathlib import Path

import sqlalchemy as sa
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session

import deerflow.knowledge.schema.experiments as experiments
from deerflow.knowledge.schema.experiments import ExperimentRow
from deerflow.knowledge.schema.findings import (
    EMBEDDING_DIMENSIONS,
    EmbeddingVector,
    FindingRow,
    embedding_vector,
)
from deerflow.knowledge.schema.research import AgentRunRow, ResearchProjectRow
from deerflow.persistence import bootstrap
from deerflow.persistence.base import Base
from deerflow.persistence.migrations import _helpers
from deerflow.persistence.migrations._helpers import _normalize_default

# Digit-prefixed migration modules are not importable with a plain import
# statement; importlib handles the dotted name fine.
migration = importlib.import_module("deerflow.persistence.migrations.versions.0028_knowledge_experiment_embeddings")
migration_0022 = importlib.import_module("deerflow.persistence.migrations.versions.0022_knowledge_phase1")

REVISION = "0028_knowledge_experiment_embeddings"
PARENT = "0027_merge_knowledge_upstream"

EXPERIMENT_INDEXES: tuple[str, ...] = (
    "ix_experiment_family_hash",
    "ix_experiment_project",
    "ix_experiment_created_by_run",
    "ix_experiment_status",
)


def _engine(tmp_path: Path, name: str = "kb-experiment-embeddings.db") -> sa.Engine:
    return sa.create_engine(f"sqlite:///{(tmp_path / name).as_posix()}")


def _vector(dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    """Deterministic 768-wide embedding fixture (exact JSON round-trip)."""
    return [i / dimensions for i in range(dimensions)]


def _seed_parents() -> tuple[ResearchProjectRow, AgentRunRow]:
    """Build the project/run provenance anchors for experiment rows.

    Ids are pre-assigned: Python-side ``default=`` values only
    materialize at flush time, so FK references must use ids known up
    front rather than reading ``.id`` off unflushed peers.
    """
    project = ResearchProjectRow(id=uuid.uuid4(), name="momentum", visibility_scope={"org": "quantflow"})
    run = AgentRunRow(id=uuid.uuid4(), project_id=project.id, agent_type="research", task="test momentum", status="running")
    return project, run


def _experiment(**overrides) -> ExperimentRow:
    """Build a fully populated experiment (caller supplies ids/FKs)."""
    params: dict = {
        "experiment_family_hash": "family-1",
        "execution_hash": "exec-1",
        "hypothesis": "12-1 momentum survives transaction costs.",
        "methodology": {"universe": "liquid-common-stocks", "frequency": "monthly"},
        "parameters": {"lookback": 252, "skip": 21},
        "metrics": {"sharpe": 0.9},
        "outcome": "success",
        "failure_class": None,
        "status": "completed",
        "embedding": _vector(),
    }
    params.update(overrides)
    return ExperimentRow(**params)


def _run_migration(module, engine: sa.Engine, direction: str) -> None:
    """Execute a migration module's upgrade/downgrade against *engine*.

    Rebinds the module's ``op`` proxy — and the shared ``_helpers.op``
    proxy used by ``safe_add_column`` / ``safe_drop_column`` — to an
    ``Operations`` bound to this connection (public API only; no private
    patching).
    """
    with engine.begin() as connection:
        context = MigrationContext.configure(connection, opts={"render_as_batch": True})
        operations = Operations(context)
        original_op = module.op
        original_helpers_op = _helpers.op
        module.op = operations
        _helpers.op = operations
        try:
            if direction == "upgrade":
                module.upgrade()
            else:
                module.downgrade()
        finally:
            module.op = original_op
            _helpers.op = original_helpers_op


class TestRevisionChain:
    def test_revision_attributes(self) -> None:
        assert migration.revision == REVISION
        assert migration.down_revision == PARENT

    def test_0028_is_the_chain_head(self) -> None:
        assert bootstrap._get_head_revision() == REVISION

    def test_single_head_no_branches(self) -> None:
        migrations_dir = Path(migration.__file__).resolve().parent.parent
        config = Config()
        config.set_main_option("script_location", migrations_dir.as_posix())
        script = ScriptDirectory.from_config(config)
        assert script.get_heads() == [REVISION]

    def test_embedding_dimensions_match_findings(self) -> None:
        """Experiment embeddings share the finding dimensionality contract."""
        assert EMBEDDING_DIMENSIONS == 768
        assert experiments.ExperimentRow.__tablename__ == "experiment"
        pg_dialect = postgresql.dialect()
        sqlite_dialect = sqlite.dialect()
        experiment_type = ExperimentRow.__table__.c.embedding.type
        finding_type = FindingRow.__table__.c.embedding.type
        assert isinstance(experiment_type, EmbeddingVector)
        assert experiment_type.compile(dialect=pg_dialect) == finding_type.compile(dialect=pg_dialect) == "VECTOR(768)"
        assert experiment_type.compile(dialect=sqlite_dialect) == finding_type.compile(dialect=sqlite_dialect) == "JSON"


class TestPortableTypes:
    def test_migration_embedding_vector_compiles_per_dialect(self) -> None:
        assert str(migration._embedding_vector().compile(dialect=postgresql.dialect())) == "VECTOR(768)"
        assert str(migration._embedding_vector().compile(dialect=sqlite.dialect())) == "JSON"

    def test_migration_snapshot_matches_orm(self) -> None:
        """The frozen migration snapshot renders exactly like the ORM type."""
        pg_dialect = postgresql.dialect()
        sqlite_dialect = sqlite.dialect()
        assert migration._embedding_vector().compile(dialect=pg_dialect) == embedding_vector().compile(dialect=pg_dialect)
        assert migration._embedding_vector().compile(dialect=sqlite_dialect) == embedding_vector().compile(dialect=sqlite_dialect)

    def test_embedding_vector_without_pgvector_package(self) -> None:
        """The portable type must not require the ``pgvector`` package."""
        assert importlib.util.find_spec("pgvector") is None
        assert isinstance(EmbeddingVector(768).compile(dialect=postgresql.dialect()), str)


class TestModels:
    def test_table_registered_on_base(self) -> None:
        assert "experiment" in Base.metadata.tables
        assert "embedding" in Base.metadata.tables["experiment"].c

    def test_full_round_trip(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        try:
            Base.metadata.create_all(engine)
            project, run = _seed_parents()
            row = _experiment(id=uuid.uuid4(), project_id=project.id, created_by_run_id=run.id)
            with Session(engine, expire_on_commit=False) as session:
                session.add_all([project, run, row])
                session.commit()

            with Session(engine, expire_on_commit=False) as session:
                stored = session.get(ExperimentRow, row.id)
                assert stored is not None
                assert stored.experiment_family_hash == "family-1"
                assert stored.execution_hash == "exec-1"
                assert stored.status == "completed"
                assert stored.methodology == {"universe": "liquid-common-stocks", "frequency": "monthly"}
                assert stored.parameters == {"lookback": 252, "skip": 21}
                assert stored.metrics == {"sharpe": 0.9}
                assert stored.embedding == _vector()
                assert isinstance(stored.id, uuid.UUID)

                payload = stored.to_dict()
                assert payload["execution_hash"] == "exec-1"
                assert payload["embedding"] == _vector()
        finally:
            engine.dispose()

    def test_embedding_defaults_to_null(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path, "kb-exp-emb-defaults.db")
        try:
            Base.metadata.create_all(engine)
            project, run = _seed_parents()
            with Session(engine, expire_on_commit=False) as session:
                session.add_all([project, run])
                session.commit()
                row = _experiment(
                    project_id=project.id,
                    created_by_run_id=run.id,
                    execution_hash="exec-no-embedding",
                    embedding=None,
                )
                session.add(row)
                session.commit()
                assert isinstance(row.id, uuid.UUID)
                assert row.embedding is None
        finally:
            engine.dispose()


class TestMigration:
    def test_upgrade_adds_nullable_embedding_column(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path, "mig.db")
        try:
            _run_migration(migration_0022, engine, "upgrade")
            _run_migration(migration, engine, "upgrade")
            columns = {col["name"]: col for col in sa.inspect(engine).get_columns("experiment")}
            assert "embedding" in columns
            assert columns["embedding"]["nullable"] is True
            # SQLite portable fallback renders as documented.
            assert str(columns["embedding"]["type"]) == "JSON"
            assert _normalize_default(columns["embedding"].get("default")) is None
        finally:
            engine.dispose()

    def test_upgrade_without_experiment_table_is_noop(self, tmp_path: Path) -> None:
        """The column helper skips databases that lack the parent table."""
        engine = _engine(tmp_path, "mig-no-table.db")
        try:
            _run_migration(migration, engine, "upgrade")
            assert set(sa.inspect(engine).get_table_names()) == set()
        finally:
            engine.dispose()

    def test_upgrade_is_idempotent(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path, "mig-idem.db")
        try:
            _run_migration(migration_0022, engine, "upgrade")
            _run_migration(migration, engine, "upgrade")
            _run_migration(migration, engine, "upgrade")
            columns = {col["name"]: col for col in sa.inspect(engine).get_columns("experiment")}
            assert "embedding" in columns
        finally:
            engine.dispose()

    def test_downgrade_drops_column(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path, "mig-down.db")
        try:
            _run_migration(migration_0022, engine, "upgrade")
            _run_migration(migration, engine, "upgrade")
            _run_migration(migration, engine, "downgrade")
            columns = {col["name"] for col in sa.inspect(engine).get_columns("experiment")}
            assert "embedding" not in columns
            # The rest of the experiment table survives the downgrade.
            assert {"execution_hash", "hypothesis", "status"} <= columns
            # Downgrade is also re-runnable.
            _run_migration(migration, engine, "downgrade")
        finally:
            engine.dispose()

    def test_chained_upgrade_then_orm_writes(self, tmp_path: Path) -> None:
        """Migration DDL and ORM models agree: write via models on migrated DDL.

        Runs the real 0022-then-0028 chain so the experiment FK parents
        exist, then persists the full object graph through the ORM.
        """
        engine = _engine(tmp_path, "mig-orm.db")
        try:
            _run_migration(migration_0022, engine, "upgrade")
            _run_migration(migration, engine, "upgrade")
            project, run = _seed_parents()
            with Session(engine, expire_on_commit=False) as session:
                session.add_all([project, run, _experiment(project_id=project.id, created_by_run_id=run.id)])
                session.commit()
            with Session(engine, expire_on_commit=False) as session:
                row = session.query(ExperimentRow).one()
                assert row.embedding == _vector()
                assert row.parameters == {"lookback": 252, "skip": 21}
        finally:
            engine.dispose()

    def test_vector_extension_step_is_noop_on_sqlite(self, tmp_path: Path) -> None:
        """The pgvector extension step must not touch SQLite databases."""
        engine = _engine(tmp_path, "mig-ext.db")
        try:
            with engine.begin() as connection:
                context = MigrationContext.configure(connection)
                original_op = migration.op
                migration.op = Operations(context)
                try:
                    assert migration._ensure_vector_extension() is None
                finally:
                    migration.op = original_op
            assert set(sa.inspect(engine).get_table_names()) == set()
        finally:
            engine.dispose()

    def test_create_all_matches_migrated_ddl(self, tmp_path: Path) -> None:
        """Per-table gate: ORM ``create_all`` and alembic DDL agree exactly."""
        fresh = _engine(tmp_path, "fresh.db")
        migrated = _engine(tmp_path, "migrated.db")
        try:
            Base.metadata.create_all(fresh)
            _run_migration(migration_0022, migrated, "upgrade")
            _run_migration(migration, migrated, "upgrade")

            fresh_cols = {col["name"]: col for col in sa.inspect(fresh).get_columns("experiment")}
            migrated_cols = {col["name"]: col for col in sa.inspect(migrated).get_columns("experiment")}
            assert set(fresh_cols) == set(migrated_cols)
            for name in sorted(fresh_cols):
                f_col, m_col = fresh_cols[name], migrated_cols[name]
                assert f_col["nullable"] == m_col["nullable"], f"experiment.{name}: nullable drift"
                assert _normalize_default(f_col.get("default")) == _normalize_default(m_col.get("default")), f"experiment.{name}: default drift create_all={f_col.get('default')!r} alembic={m_col.get('default')!r}"
                assert str(f_col["type"]) == str(m_col["type"]), f"experiment.{name}: type drift create_all={f_col['type']!r} alembic={m_col['type']!r}"

            # SQLite portable fallbacks render as documented.
            assert str(fresh_cols["embedding"]["type"]) == "JSON"

            fresh_checks = {(c["name"], c["sqltext"]) for c in sa.inspect(fresh).get_check_constraints("experiment")}
            migrated_checks = {(c["name"], c["sqltext"]) for c in sa.inspect(migrated).get_check_constraints("experiment")}
            assert fresh_checks == migrated_checks
            assert {"ck_experiment_status", "ck_experiment_outcome", "ck_experiment_failure_class"} <= {name for name, _ in fresh_checks}

            fresh_uniques = {c["name"] for c in sa.inspect(fresh).get_unique_constraints("experiment")}
            migrated_uniques = {c["name"] for c in sa.inspect(migrated).get_unique_constraints("experiment")}
            assert fresh_uniques == migrated_uniques == {"uq_experiment_execution_hash"}

            fresh_indexes = {i["name"] for i in sa.inspect(fresh).get_indexes("experiment")}
            migrated_indexes = {i["name"] for i in sa.inspect(migrated).get_indexes("experiment")}
            assert fresh_indexes == migrated_indexes
            assert set(EXPERIMENT_INDEXES) <= fresh_indexes
        finally:
            fresh.dispose()
            migrated.dispose()

    def test_postgres_ddl_compilation(self) -> None:
        """The ORM experiment table compiles to PG DDL with VECTOR(768)."""
        dialect = postgresql.dialect()
        ddl = str(sa.schema.CreateTable(Base.metadata.tables["experiment"]).compile(dialect=dialect))
        assert "VECTOR(768)" in ddl
        assert "uq_experiment_execution_hash" in ddl
        assert "ck_experiment_status" in ddl

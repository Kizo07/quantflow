"""Tests for the Knowledge Plane Phase 2 findings schema.

Covers ``FindingRow`` in ``deerflow.knowledge.schema.findings`` and alembic
revision ``0023_knowledge_findings``:

* revision chain integrity (direct child of ``0022_knowledge_phase1``,
  single alembic head);
* ``Base.metadata.create_all`` + finding persistence, including JSON
  scope/confidence payloads, UUID defaults, the supersession chain, and
  exact embedding round-trip through the SQLite JSON fallback;
* contract enforcement: unique ``canonical_key``, ``finding_type`` /
  candidate-only ``status`` CHECK constraints, and foreign keys;
* migration ``upgrade()`` / ``downgrade()`` against a scratch SQLite
  database (idempotent re-run included), including the chained
  0022-then-0023 upgrade;
* ORM/migration DDL parity for the ``finding`` table (columns,
  nullability, defaults, type affinity, checks, uniques, indexes) —
  the per-table equivalent of the persistence-bootstrap hard gate;
* PostgreSQL DDL compilation (VECTOR(768) / TSVECTOR / JSONB / UUID /
  TIMESTAMPTZ / GIN rendering).

Standalone: uses file-backed SQLite databases under ``tmp_path`` and
imports only the new schema module (plus its Phase 1 parents), the
``Base`` declarative registry, and the migration under test. No
PostgreSQL server or driver is required, and the ``pgvector`` Python
package must not be needed.
"""

from __future__ import annotations

import importlib
import importlib.util
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import deerflow.knowledge.schema.findings as findings
from deerflow.knowledge.schema.findings import (
    EMBEDDING_DIMENSIONS,
    FINDING_STATUSES,
    FINDING_STATUSES_PHASE2,
    FINDING_TYPES,
    EmbeddingVector,
    FindingRow,
    embedding_vector,
    tsvector,
)
from deerflow.knowledge.schema.research import AgentRunRow, ResearchProjectRow
from deerflow.persistence.base import Base
from deerflow.persistence.migrations._helpers import _normalize_default

# Digit-prefixed migration modules are not importable with a plain import
# statement; importlib handles the dotted name fine.
migration = importlib.import_module("deerflow.persistence.migrations.versions.0023_knowledge_findings")
migration_0022 = importlib.import_module("deerflow.persistence.migrations.versions.0022_knowledge_phase1")

FINDING_INDEXES: tuple[str, ...] = (
    "ix_finding_project",
    "ix_finding_type",
    "ix_finding_status",
    "ix_finding_created_by_run",
    "ix_finding_supersedes",
    "ix_finding_search_document",
)


def _engine(tmp_path: Path, name: str = "kb-findings.db") -> sa.Engine:
    return sa.create_engine(f"sqlite:///{(tmp_path / name).as_posix()}")


def _vector(dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    """Deterministic 768-wide embedding fixture (exact JSON round-trip)."""
    return [i / dimensions for i in range(dimensions)]


def _seed_parents() -> tuple[ResearchProjectRow, AgentRunRow]:
    """Build the project/run provenance anchors for finding rows.

    Ids are pre-assigned: Python-side ``default=`` values only
    materialize at flush time, so FK references must use ids known up
    front rather than reading ``.id`` off unflushed peers.
    """
    project = ResearchProjectRow(id=uuid.uuid4(), name="momentum", visibility_scope={"org": "quantflow"})
    run = AgentRunRow(id=uuid.uuid4(), project_id=project.id, agent_type="research", task="test momentum", status="running")
    return project, run


def _finding(**overrides) -> FindingRow:
    """Build a fully populated candidate finding (caller supplies ids/FKs)."""
    params: dict = {
        "canonical_key": "mom-12-1-survives-costs",
        "finding_type": "empirical",
        "statement": "12-1 momentum survives transaction costs on liquid US stocks.",
        "scope": {"asset_class": "equity", "market": "US", "horizon": "12-1m"},
        "status": "candidate",
        "confidence": {"overall_tier": "moderate", "reproducibility": "single-run"},
        "effective_from": datetime(2000, 1, 1, tzinfo=UTC),
        "effective_to": datetime(2025, 12, 31, tzinfo=UTC),
        "retired_at": None,
        "supersedes_id": None,
        "search_document": "momentum 12-1 transaction costs liquid US stocks",
        "embedding": _vector(),
    }
    params.update(overrides)
    return FindingRow(**params)


def _run_migration(module, engine: sa.Engine, direction: str) -> None:
    """Execute a migration module's upgrade/downgrade against *engine*.

    Rebinds the module's ``op`` proxy to an ``Operations`` bound to this
    connection (public API only; no private patching).
    """
    with engine.begin() as connection:
        context = MigrationContext.configure(connection, opts={"render_as_batch": True})
        original_op = module.op
        module.op = Operations(context)
        try:
            if direction == "upgrade":
                module.upgrade()
            else:
                module.downgrade()
        finally:
            module.op = original_op


class TestRevisionChain:
    def test_revision_attributes(self) -> None:
        assert migration.revision == "0023_knowledge_findings"
        assert migration.down_revision == "0022_knowledge_phase1"

    def test_single_head_is_new_revision(self) -> None:
        # The upstream merge (0022-0026) forked the chain; merge revision
        # 0027 rejoins it, and Phase 3 revision 0028 extends it (single
        # head, no branches).
        migrations_dir = Path(migration.__file__).resolve().parent.parent
        config = Config()
        config.set_main_option("script_location", migrations_dir.as_posix())
        script = ScriptDirectory.from_config(config)
        assert script.get_heads() == ["0028_knowledge_experiment_embeddings"]

    def test_vocabulary_matches_kb_contract(self) -> None:
        assert FINDING_TYPES == ("empirical", "methodological", "data_quality", "failure", "prior")
        assert FINDING_STATUSES == ("candidate", "reviewed", "validated", "disputed", "superseded", "rejected")
        assert FINDING_STATUSES_PHASE2 == ("candidate",)
        assert EMBEDDING_DIMENSIONS == 768

    def test_schema_module_exposes_expected_model(self) -> None:
        assert findings.FindingRow.__tablename__ == "finding"


class TestPortableTypes:
    def test_tsvector_compiles_per_dialect(self) -> None:
        assert str(tsvector().compile(dialect=postgresql.dialect())) == "TSVECTOR"
        assert str(tsvector().compile(dialect=sqlite.dialect())) == "TEXT"

    def test_embedding_vector_compiles_per_dialect(self) -> None:
        assert str(embedding_vector().compile(dialect=postgresql.dialect())) == "VECTOR(768)"
        assert str(embedding_vector().compile(dialect=sqlite.dialect())) == "JSON"

    def test_embedding_vector_without_pgvector_package(self) -> None:
        """The portable type must not require the ``pgvector`` package."""
        assert importlib.util.find_spec("pgvector") is None
        assert isinstance(EmbeddingVector(768).compile(dialect=postgresql.dialect()), str)


class TestModels:
    def test_table_registered_on_base(self) -> None:
        assert "finding" in Base.metadata.tables

    def test_full_round_trip(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        try:
            Base.metadata.create_all(engine)
            project, run = _seed_parents()
            first = _finding(id=uuid.uuid4(), project_id=project.id, created_by_run_id=run.id)
            second = _finding(
                id=uuid.uuid4(),
                project_id=project.id,
                created_by_run_id=run.id,
                canonical_key="mom-12-1-survives-costs-v2",
                supersedes_id=first.id,
                statement="12-1 momentum survives costs; turnover update.",
            )
            with Session(engine, expire_on_commit=False) as session:
                session.add_all([project, run, first, second])
                session.commit()

            with Session(engine, expire_on_commit=False) as session:
                row = session.get(FindingRow, first.id)
                assert row is not None
                assert row.canonical_key == "mom-12-1-survives-costs"
                assert row.finding_type == "empirical"
                assert row.status == "candidate"
                assert row.scope == {"asset_class": "equity", "market": "US", "horizon": "12-1m"}
                assert row.confidence == {"overall_tier": "moderate", "reproducibility": "single-run"}
                assert row.search_document == "momentum 12-1 transaction costs liquid US stocks"
                assert row.embedding == _vector()
                assert isinstance(row.id, uuid.UUID)
                assert row.recorded_at is not None
                assert row.retired_at is None
                assert row.supersedes_id is None

                child = session.get(FindingRow, second.id)
                assert child is not None
                assert child.supersedes_id == first.id

                payload = row.to_dict()
                assert payload["canonical_key"] == "mom-12-1-survives-costs"
                assert payload["finding_type"] == "empirical"
        finally:
            engine.dispose()

    def test_python_side_defaults(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        try:
            Base.metadata.create_all(engine)
            project, run = _seed_parents()
            with Session(engine, expire_on_commit=False) as session:
                session.add_all([project, run])
                session.commit()
                row = FindingRow(
                    canonical_key="defaults-key",
                    finding_type="prior",
                    statement="defaults",
                    project_id=project.id,
                    created_by_run_id=run.id,
                )
                session.add(row)
                session.commit()
                assert isinstance(row.id, uuid.UUID)
                assert row.status == "candidate"
                assert row.scope == {}
                assert row.confidence == {}
                assert row.recorded_at is not None
                assert row.recorded_at.tzinfo is not None
                assert row.search_document is None
                assert row.embedding is None
        finally:
            engine.dispose()

    def test_all_finding_types_accepted(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        try:
            Base.metadata.create_all(engine)
            project, run = _seed_parents()
            with Session(engine, expire_on_commit=False) as session:
                session.add_all([project, run])
                session.add_all(
                    [
                        _finding(
                            canonical_key=f"type-key-{finding_type}",
                            finding_type=finding_type,
                            project_id=project.id,
                            created_by_run_id=run.id,
                        )
                        for finding_type in FINDING_TYPES
                    ]
                )
                session.commit()
            with Session(engine, expire_on_commit=False) as session:
                rows = session.query(FindingRow).all()
                assert {row.finding_type for row in rows} == set(FINDING_TYPES)
        finally:
            engine.dispose()


class TestContractEnforcement:
    def _seeded_engine(self, tmp_path: Path, name: str = "kb-findings-contract.db") -> tuple[sa.Engine, tuple[ResearchProjectRow, AgentRunRow]]:
        engine = _engine(tmp_path, name)
        Base.metadata.create_all(engine)
        project, run = _seed_parents()
        with Session(engine, expire_on_commit=False) as session:
            session.add_all([project, run, _finding(project_id=project.id, created_by_run_id=run.id)])
            session.commit()
        return engine, (project, run)

    def test_duplicate_canonical_key_rejected(self, tmp_path: Path) -> None:
        engine, (project, run) = self._seeded_engine(tmp_path)
        try:
            with Session(engine, expire_on_commit=False) as session:
                session.add(_finding(project_id=project.id, created_by_run_id=run.id, statement="exact duplicate"))
                with pytest.raises(IntegrityError):
                    session.commit()
        finally:
            engine.dispose()

    def test_bad_finding_type_rejected(self, tmp_path: Path) -> None:
        engine, (project, run) = self._seeded_engine(tmp_path)
        try:
            with Session(engine, expire_on_commit=False) as session:
                session.add(
                    _finding(
                        canonical_key="bad-type-key",
                        finding_type="bogus",
                        project_id=project.id,
                        created_by_run_id=run.id,
                    )
                )
                with pytest.raises(IntegrityError):
                    session.commit()
        finally:
            engine.dispose()

    @pytest.mark.parametrize("status", ["reviewed", "validated", "disputed", "superseded", "rejected", "bogus"])
    def test_non_candidate_status_rejected_at_this_phase(self, tmp_path: Path, status: str) -> None:
        """Phase 2 enforces candidate-only status; transitions land in Phase 3."""
        engine, (project, run) = self._seeded_engine(tmp_path, f"kb-status-{status}.db")
        try:
            with Session(engine, expire_on_commit=False) as session:
                session.add(
                    _finding(
                        canonical_key=f"status-key-{status}",
                        status=status,
                        project_id=project.id,
                        created_by_run_id=run.id,
                    )
                )
                with pytest.raises(IntegrityError):
                    session.commit()
        finally:
            engine.dispose()

    def test_created_by_run_fk_enforced(self, tmp_path: Path) -> None:
        engine, (project, run) = self._seeded_engine(tmp_path)
        try:
            with engine.begin() as connection:
                connection.execute(sa.text("PRAGMA foreign_keys=ON"))
            with Session(engine, expire_on_commit=False) as session:
                session.add(
                    _finding(
                        canonical_key="no-run-key",
                        project_id=project.id,
                        created_by_run_id=uuid.uuid4(),
                    )
                )
                with pytest.raises(IntegrityError):
                    session.commit()
        finally:
            engine.dispose()

    def test_created_by_run_not_null(self, tmp_path: Path) -> None:
        engine, (project, run) = self._seeded_engine(tmp_path)
        try:
            with Session(engine, expire_on_commit=False) as session:
                session.add(
                    _finding(
                        canonical_key="null-run-key",
                        project_id=project.id,
                        created_by_run_id=None,
                    )
                )
                with pytest.raises(IntegrityError):
                    session.commit()
        finally:
            engine.dispose()


class TestMigration:
    def test_upgrade_creates_table_and_indexes(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path, "mig.db")
        try:
            _run_migration(migration, engine, "upgrade")
            inspector = sa.inspect(engine)
            assert "finding" in set(inspector.get_table_names())

            index_names = {index["name"] for index in inspector.get_indexes("finding")}
            assert set(FINDING_INDEXES) <= index_names

            uniques = {constraint["name"] for constraint in inspector.get_unique_constraints("finding")}
            assert "uq_finding_canonical_key" in uniques
        finally:
            engine.dispose()

    def test_upgrade_is_idempotent(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path, "mig-idem.db")
        try:
            _run_migration(migration, engine, "upgrade")
            _run_migration(migration, engine, "upgrade")
            assert "finding" in set(sa.inspect(engine).get_table_names())
        finally:
            engine.dispose()

    def test_downgrade_drops_table(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path, "mig-down.db")
        try:
            _run_migration(migration, engine, "upgrade")
            _run_migration(migration, engine, "downgrade")
            assert "finding" not in set(sa.inspect(engine).get_table_names())
            # Downgrade is also re-runnable.
            _run_migration(migration, engine, "downgrade")
        finally:
            engine.dispose()

    def test_chained_upgrade_then_orm_writes(self, tmp_path: Path) -> None:
        """Migration DDL and ORM models agree: write via models on migrated DDL.

        Runs the real 0022-then-0023 chain so the finding FK parents exist,
        then persists the full object graph through the ORM.
        """
        engine = _engine(tmp_path, "mig-orm.db")
        try:
            _run_migration(migration_0022, engine, "upgrade")
            _run_migration(migration, engine, "upgrade")
            project, run = _seed_parents()
            with Session(engine, expire_on_commit=False) as session:
                session.add_all([project, run, _finding(project_id=project.id, created_by_run_id=run.id)])
                session.commit()
            with Session(engine, expire_on_commit=False) as session:
                row = session.query(FindingRow).one()
                assert row.embedding == _vector()
                assert row.scope["market"] == "US"
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
            _run_migration(migration, migrated, "upgrade")

            fresh_cols = {col["name"]: col for col in sa.inspect(fresh).get_columns("finding")}
            migrated_cols = {col["name"]: col for col in sa.inspect(migrated).get_columns("finding")}
            assert set(fresh_cols) == set(migrated_cols)
            for name in sorted(fresh_cols):
                f_col, m_col = fresh_cols[name], migrated_cols[name]
                assert f_col["nullable"] == m_col["nullable"], f"finding.{name}: nullable drift"
                assert _normalize_default(f_col.get("default")) == _normalize_default(m_col.get("default")), f"finding.{name}: default drift create_all={f_col.get('default')!r} alembic={m_col.get('default')!r}"
                assert str(f_col["type"]) == str(m_col["type"]), f"finding.{name}: type drift create_all={f_col['type']!r} alembic={m_col['type']!r}"

            # SQLite portable fallbacks render as documented.
            assert str(fresh_cols["embedding"]["type"]) == "JSON"
            assert str(fresh_cols["search_document"]["type"]) == "TEXT"

            fresh_checks = {(c["name"], c["sqltext"]) for c in sa.inspect(fresh).get_check_constraints("finding")}
            migrated_checks = {(c["name"], c["sqltext"]) for c in sa.inspect(migrated).get_check_constraints("finding")}
            assert fresh_checks == migrated_checks
            assert {"ck_finding_type", "ck_finding_status"} <= {name for name, _ in fresh_checks}

            fresh_uniques = {c["name"] for c in sa.inspect(fresh).get_unique_constraints("finding")}
            migrated_uniques = {c["name"] for c in sa.inspect(migrated).get_unique_constraints("finding")}
            assert fresh_uniques == migrated_uniques == {"uq_finding_canonical_key"}

            fresh_indexes = {i["name"] for i in sa.inspect(fresh).get_indexes("finding")}
            migrated_indexes = {i["name"] for i in sa.inspect(migrated).get_indexes("finding")}
            assert fresh_indexes == migrated_indexes
            assert set(FINDING_INDEXES) <= fresh_indexes
        finally:
            fresh.dispose()
            migrated.dispose()

    def test_check_sql_frozen_in_migrated_ddl(self, tmp_path: Path) -> None:
        """The migrated CHECK text repeats the ORM vocabulary literally."""
        engine = _engine(tmp_path, "mig-sql.db")
        try:
            _run_migration(migration, engine, "upgrade")
            with engine.connect() as connection:
                create_sql = connection.execute(sa.text("SELECT sql FROM sqlite_master WHERE name = 'finding'")).scalar_one()
            assert "finding_type IN ('empirical', 'methodological', 'data_quality', 'failure', 'prior')" in create_sql
            assert "status IN ('candidate')" in create_sql
        finally:
            engine.dispose()

    def test_postgres_ddl_compilation(self) -> None:
        """ORM tables compile to PG DDL with VECTOR / TSVECTOR / JSONB."""
        dialect = postgresql.dialect()
        ddl = str(sa.schema.CreateTable(Base.metadata.tables["finding"]).compile(dialect=dialect))
        assert "VECTOR(768)" in ddl
        assert "TSVECTOR" in ddl
        assert "JSONB" in ddl
        assert "UUID" in ddl
        assert "TIMESTAMP WITH TIME ZONE" in ddl
        assert "uq_finding_canonical_key" in ddl
        assert "ck_finding_type" in ddl
        assert "ck_finding_status" in ddl

        (search_index,) = [index for index in FindingRow.__table__.indexes if index.name == "ix_finding_search_document"]
        index_ddl = str(sa.schema.CreateIndex(search_index).compile(dialect=dialect))
        assert "USING gin" in index_ddl

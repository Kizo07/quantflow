"""Phase 2 exit gate: sectioned recall@10 >= 0.8 on the merged eval fixture.

Seeds the merged eval fixture (Phase 0 + Phase 2 extension) into a scratch
SQLite database through the production write path (``write_api`` begin /
failure / commit, mirrored to the ORM rows), refreshes the lexical index,
backfills finding embeddings with the deterministic fake, then runs the
production retrieval plane per query in two sections:

* priors section — structured + lexical channels over the
  ``finding``/``experiment``/``assumption`` kinds, no failure boost;
* failures section — structured + lexical + dedicated failure channel over
  the ``failure`` kind, failure boost on.

The vector channel stays off: Phase 2 embeds findings only, and the probe
evidence (see the module log below) shows findings-only vectors crowding
experiment needles out of the fused top-10. The vector leg is wired behind
:func:`deerflow.knowledge.embeddings.load_provider` (real provider in
:mod:`deerflow.knowledge.providers_st`) and becomes the default once
experiment embeddings land; until then the production default
(``open_pg_backends`` with ``embedding_model=None``) is struct+lexical.

Sectioned judgments: the priors section is scored on the non-failure
relevant docs only. Every failure-kind doc in ``relevant_doc_ids`` is also
in ``must_recall_failures`` (asserted here), so failures are judged exactly
once, in the failures section — scoring priors on a kind it excludes by
design would be an automatic miss, not a retrieval signal.

Gate: mean recall@10 >= 0.8 AND mean failure-recall@10 >= 0.8.
"""

from __future__ import annotations

import statistics
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

import deerflow.knowledge.schema as kb_schema
import deerflow.persistence.models  # noqa: F401  (register all rows)
from deerflow.knowledge import pg_retrieval as pg
from deerflow.knowledge.embeddings import (
    DeterministicEmbeddingProvider,
    backfill_findings_embeddings,
)
from deerflow.knowledge.eval import seed_phase2 as seed
from deerflow.knowledge.retrieval.planner import (
    execute_retrieval,
    lexical_query_text,
    parse_research_intent,
    plan_retrieval,
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
    "finding",
)

K = 10
TARGET = 0.8
NOW = "2026-09-20T00:00:00+00:00"


def _parse_moment(value: object) -> datetime | None:
    if not value:
        return None
    assert isinstance(value, str)
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment


def _seed_database(db_path: Path) -> tuple[object, dict[str, str]]:
    """Seed the merged fixture; return (session factory, candidate-id -> doc-id map)."""
    engine = sa.create_engine(f"sqlite:///{db_path}")
    for table in KB_TABLES:
        Base.metadata.tables[table].create(engine, checkfirst=True)
    factory = sessionmaker(engine, expire_on_commit=False)

    bundle = seed.build_seed_bundle()
    assert bundle.coverage.ok, f"judgment coverage failed: {bundle.coverage.violations()}"

    write_api = seed.require_deerflow("knowledge.write_api")
    mem = seed._FakeExperimentStore()
    records = {}
    failure_ids = {}
    for item in sorted(bundle.experiments, key=lambda s: s.doc_id):
        record = write_api.experiment_begin(mem, **item.begin_kwargs)
        if item.failure_kwargs is not None:
            failure = write_api.failure_record(mem, experiment_id=record.id, **item.failure_kwargs)
            failure_ids[item.doc_id] = failure.id
        records[item.doc_id] = write_api.experiment_commit(mem, experiment_id=record.id, **item.commit_kwargs)
    commit_map = seed.SeedCommitMap(
        experiment_ids={doc_id: record.id for doc_id, record in records.items()},
        failure_ids=failure_ids,
        family_hashes={doc_id: record.family_hash for doc_id, record in records.items()},
        execution_hashes={doc_id: record.execution_hash for doc_id, record in records.items()},
    )

    with factory() as session:
        session.add(kb_schema.ResearchProjectRow(id=uuid.UUID(bundle.project_id), name="phase2-eval", visibility_scope={}))
        session.add(
            kb_schema.AgentRunRow(
                id=uuid.UUID(bundle.run_id),
                project_id=uuid.UUID(bundle.project_id),
                agent_type="research",
                task="phase 2 eval seeder",
                status="completed",
            )
        )
        for artifact in bundle.artifacts:
            session.add(
                kb_schema.ArtifactRow(
                    id=uuid.UUID(artifact.artifact_id),
                    sha256=artifact.sha256,
                    kind=artifact.kind,
                    storage_uri=f"artifact://sha256/{artifact.sha256}",
                    media_type=artifact.media_type,
                    byte_size=artifact.byte_size,
                    created_by_run_id=uuid.UUID(bundle.run_id),
                    artifact_metadata={"eval_doc_id": artifact.doc_id},
                )
            )
        seen_datasets = set()
        for item in bundle.experiments:
            for dataset_id in item.dataset_version_ids:
                if dataset_id not in seen_datasets:
                    seen_datasets.add(dataset_id)
                    session.add(
                        kb_schema.DatasetVersionRow(
                            id=uuid.UUID(dataset_id),
                            dataset_key=f"eval:{item.doc_id}:features",
                            provider="eval-fixture",
                        )
                    )
        for item in sorted(bundle.experiments, key=lambda s: s.doc_id):
            record = records[item.doc_id]
            session.add(
                kb_schema.ExperimentRow(
                    id=uuid.UUID(record.id),
                    project_id=uuid.UUID(record.project_id),
                    created_by_run_id=uuid.UUID(record.created_by_run_id),
                    experiment_family_hash=record.family_hash,
                    execution_hash=record.execution_hash,
                    hypothesis=record.hypothesis,
                    methodology=dict(record.methodology),
                    parameters=dict(record.parameters),
                    metrics=dict(record.metrics) if record.metrics is not None else None,
                    outcome=record.outcome,
                    failure_class=record.failure_class,
                    status=record.status,
                    started_at=_parse_moment(record.started_at),
                    completed_at=_parse_moment(record.completed_at),
                )
            )
            for link in record.datasets:
                session.add(
                    kb_schema.ExperimentDatasetRow(
                        experiment_id=uuid.UUID(record.id),
                        dataset_version_id=uuid.UUID(link["dataset_version_id"]),
                        role=link["role"],
                    )
                )
            for artifact_id in record.result_artifacts:
                session.add(
                    kb_schema.ExperimentArtifactRow(
                        experiment_id=uuid.UUID(record.id),
                        artifact_id=uuid.UUID(artifact_id),
                        role="result",
                    )
                )
        finding_payloads = seed.resolve_finding_evidence(bundle, commit_map)
        finding_doc = {}
        for payload, finding in zip(finding_payloads, sorted(bundle.findings, key=lambda f: f.doc_id), strict=True):
            session.add(
                kb_schema.FindingRow(
                    id=uuid.UUID(payload["finding_id"]),
                    project_id=uuid.UUID(bundle.project_id),
                    canonical_key=payload["canonical_key"],
                    finding_type=payload["finding_type"],
                    statement=payload["statement"],
                    scope=dict(payload["scope"]),
                    status=payload["status"],
                    confidence=dict(payload["confidence"]),
                    effective_from=_parse_moment(payload["valid_time"]["effective_from"]),
                    effective_to=_parse_moment(payload["valid_time"].get("effective_to")),
                    created_by_run_id=uuid.UUID(payload["created_by_run_id"]),
                )
            )
            finding_doc[payload["finding_id"]] = finding.doc_id
        session.commit()

    n_indexed = pg.refresh_finding_search_documents(factory)
    provider = DeterministicEmbeddingProvider()
    backfill = backfill_findings_embeddings(
        provider,
        pg.SQLFindingTextSource(factory),
        pg.SQLEmbeddingVectorStore(factory, model_id=provider.model_id),
        list(finding_doc),
    )
    assert n_indexed == len(bundle.findings) and backfill.upserted == len(bundle.findings)

    id_to_doc = {record_id: doc_id for doc_id, record_id in commit_map.experiment_ids.items()}
    id_to_doc.update(finding_doc)
    engine.dispose()
    return factory, id_to_doc


def test_sectioned_recall_gate(tmp_path: Path) -> None:
    """Exit gate: sectioned recall@10 and failure-recall@10 each >= 0.8."""
    factory, id_to_doc = _seed_database(tmp_path / "phase2_exit_gate.db")
    structured = pg.SQLStructuredLookupStore(factory)
    lexical = pg.SQLLexicalSearchStore(factory)
    failures = pg.SQLFailureSearchStore(factory)
    fixture = seed.load_merged_fixture()
    kinds = {doc.doc_id: getattr(doc, "kind", "?") for doc in fixture.documents}

    # Every failure-kind relevant doc must be judged in the failures section;
    # otherwise the priors denominator below would silently drop judgments.
    for judgment in fixture.queries:
        failure_relevant = {d for d in judgment.relevant_doc_ids if kinds.get(d) == "failure"}
        assert failure_relevant <= set(judgment.must_recall_failures), judgment.query_id

    recalls: list[float] = []
    f_recalls: list[float] = []
    misses: list[str] = []
    for judgment in fixture.queries:
        intent = parse_research_intent(dict(judgment.intent))

        priors_section = replace(intent, needed_memory=("prior_experiments", "validated_findings"))
        text = lexical_query_text(judgment.query_text, intent)
        priors_plan = plan_retrieval(
            priors_section,
            top_k=K,
            per_channel_limit=50,
            kinds=("finding", "experiment", "assumption"),
            relational=False,
        )
        priors = execute_retrieval(
            priors_plan,
            query_text=text,
            structured=structured,
            lexical=lexical,
            vector=None,
            failures=None,
            relational=None,
            now=NOW,
            apply_failure_boost=False,
        )

        failures_section = replace(intent, needed_memory=("failures",))
        failures_plan = plan_retrieval(
            failures_section,
            top_k=K,
            per_channel_limit=50,
            kinds=("failure",),
            relational=False,
        )
        fails = execute_retrieval(
            failures_plan,
            query_text=text,
            structured=structured,
            lexical=lexical,
            vector=None,
            failures=failures,
            relational=None,
            now=NOW,
            apply_failure_boost=True,
        )

        priors_ids = [id_to_doc[item.candidate.id] for item in priors.top if item.candidate.id in id_to_doc]
        fails_ids = [id_to_doc[item.candidate.id] for item in fails.top if item.candidate.id in id_to_doc]
        rel = {d for d in judgment.relevant_doc_ids if kinds.get(d) != "failure"}
        must = set(judgment.must_recall_failures)
        recall = len(rel & set(priors_ids)) / len(rel) if rel else 1.0
        f_recall = len(must & set(fails_ids)) / len(must) if must else 1.0
        recalls.append(recall)
        f_recalls.append(f_recall)
        if recall < 1.0 or f_recall < 1.0:
            misses.append(f"{judgment.query_id}: recall={recall:.2f} fail-recall={f_recall:.2f}")

    mean_recall = statistics.mean(recalls)
    mean_f_recall = statistics.mean(f_recalls)
    detail = f"recall@{K}={mean_recall:.4f} failure-recall@{K}={mean_f_recall:.4f} target={TARGET}"
    assert mean_recall >= TARGET, f"{detail}\n" + "\n".join(misses)
    assert mean_f_recall >= TARGET, f"{detail}\n" + "\n".join(misses)

"""Phase 3 exit gate: sectioned recall@10 >= 0.8 with the vector leg ON.

Seeds the merged eval fixture exactly like the Phase 2 gate (production
write path, lexical index refresh), backfills **both** embedding kinds —
finding vectors via :func:`backfill_findings_embeddings` and experiment
vectors via the Phase 3 experiment text/vector stores — then runs the
production retrieval plane per query in two sections with the vector
channel armed:

* priors section — structured + lexical + vector channels over the
  ``finding``/``experiment``/``assumption`` kinds, no failure boost;
* failures section — structured + lexical + vector + dedicated failure
  channel over the ``failure`` kind, failure boost on. The
  ``needed_memory`` is widened to ``("failures", "prior_experiments")``
  (from the Phase 2 ``("failures",)``) purely to arm the vector channel —
  ``plan_retrieval`` maps ``"failures"`` to the failure channel alone —
  while ``kinds`` stays ``("failure",)`` so the judgments are unchanged.

Sectioned judgments mirror Phase 2: the priors section scores the
non-failure relevant docs only (every failure-kind doc is also in
``must_recall_failures``, asserted here), failures are judged in the
failures section.

Gate: mean recall@10 >= 0.8 AND mean failure-recall@10 >= 0.8, with the
vector channel contributing to every fused top-10 in both sections (a
run where the vector leg silently skips fails the gate even at full
recall) and both embedding kinds fully backfilled.

Provider: the real MPNet checkpoint via
:mod:`deerflow.knowledge.providers_st` (cached, offline). The
deterministic fake carries no semantic signal, so a quality gate with
the vector leg armed cannot use it — noise ranks would only dilute the
lexical/structured signal (measured: priors recall@10 0.75 with fake
vectors ON vs 0.83 OFF). The gate skips when the checkpoint is
unavailable; mechanics stay covered by the fake-backed retrieval tests.
"""

from __future__ import annotations

import os
import statistics
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

import deerflow.knowledge.schema as kb_schema
import deerflow.persistence.models  # noqa: F401  (register all rows)
from deerflow.knowledge import pg_retrieval as pg
from deerflow.knowledge.embeddings import (
    EmbeddingProvider,
    EmbeddingProviderError,
    backfill_experiments_embeddings,
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


def _load_provider() -> EmbeddingProvider:
    """Load the cached MPNet provider, skipping the gate when unavailable."""
    pytest.importorskip("sentence_transformers")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from deerflow.knowledge.providers_st import register_st_provider

    try:
        return register_st_provider()
    except EmbeddingProviderError as exc:
        pytest.skip(f"MPNet checkpoint unavailable: {exc}")
        raise AssertionError("unreachable") from exc


def _seed_database(db_path: Path, provider: EmbeddingProvider) -> tuple[object, dict[str, str]]:
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
        session.add(kb_schema.ResearchProjectRow(id=uuid.UUID(bundle.project_id), name="phase3-eval", visibility_scope={}))
        session.add(
            kb_schema.AgentRunRow(
                id=uuid.UUID(bundle.run_id),
                project_id=uuid.UUID(bundle.project_id),
                agent_type="research",
                task="phase 3 eval seeder",
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
    finding_backfill = backfill_findings_embeddings(
        provider,
        pg.SQLFindingTextSource(factory),
        pg.SQLEmbeddingVectorStore(factory, model_id=provider.model_id),
        list(finding_doc),
    )
    experiment_ids = list(commit_map.experiment_ids.values())
    experiment_backfill = backfill_experiments_embeddings(
        provider,
        pg.SQLExperimentTextSource(factory),
        pg.SQLExperimentEmbeddingVectorStore(factory, model_id=provider.model_id),
        experiment_ids,
    )
    n_experiments_embedded = experiment_backfill.upserted
    assert n_indexed == len(bundle.findings) and finding_backfill.upserted == len(bundle.findings)
    assert n_experiments_embedded == len(bundle.experiments), "both embedding kinds must backfill"
    assert pg.SQLExperimentEmbeddingVectorStore(factory, model_id=provider.model_id).get_embedding_model(experiment_ids[0]) == provider.model_id

    id_to_doc = {record_id: doc_id for doc_id, record_id in commit_map.experiment_ids.items()}
    id_to_doc.update(finding_doc)
    engine.dispose()
    return factory, id_to_doc


def test_sectioned_recall_gate_with_vector_leg(tmp_path: Path) -> None:
    """Exit gate: sectioned recall@10 and failure-recall@10 each >= 0.8, vector ON."""
    provider = _load_provider()
    factory, id_to_doc = _seed_database(tmp_path / "phase3_exit_gate.db", provider)
    structured = pg.SQLStructuredLookupStore(factory)
    lexical = pg.SQLLexicalSearchStore(factory)
    vector = pg.SQLVectorSearchStore(factory)
    failures = pg.SQLFailureSearchStore(factory)
    fixture = seed.load_merged_fixture()
    kinds = {doc.doc_id: getattr(doc, "kind", "?") for doc in fixture.documents}

    # Every failure-kind relevant doc must be judged in the failures section;
    # otherwise the priors denominator below would silently drop judgments.
    for judgment in fixture.queries:
        failure_relevant = {d for d in judgment.relevant_doc_ids if kinds.get(d) == "failure"}
        assert failure_relevant <= set(judgment.must_recall_failures), judgment.query_id

    # One batched encode for every query (both sections share the embedding).
    query_texts = [lexical_query_text(judgment.query_text, parse_research_intent(dict(judgment.intent))) for judgment in fixture.queries]
    query_embeddings = {judgment.query_id: embedding for judgment, embedding in zip(fixture.queries, provider.embed_batch(query_texts), strict=True)}

    recalls: list[float] = []
    f_recalls: list[float] = []
    misses: list[str] = []
    vector_hits = 0
    for judgment in fixture.queries:
        intent = parse_research_intent(dict(judgment.intent))

        priors_section = replace(intent, needed_memory=("prior_experiments", "validated_findings"))
        text = lexical_query_text(judgment.query_text, intent)
        query_embedding = query_embeddings[judgment.query_id]
        priors_plan = plan_retrieval(
            priors_section,
            top_k=K,
            per_channel_limit=50,
            kinds=("finding", "experiment", "assumption"),
            relational=False,
        )
        assert "vector" in priors_plan.channels
        priors = execute_retrieval(
            priors_plan,
            query_text=text,
            structured=structured,
            lexical=lexical,
            vector=vector,
            failures=None,
            relational=None,
            query_embedding=query_embedding,
            now=NOW,
            apply_failure_boost=False,
        )
        assert "vector channel skipped" not in " ".join(priors.warnings)
        assert priors.channel_counts.get("vector", 0) > 0, f"{judgment.query_id}: vector leg returned no priors hits"
        assert all("vector" in item.channels for item in priors.top), f"{judgment.query_id}: vector leg missing from fused priors top-10"
        vector_hits += priors.channel_counts.get("vector", 0)

        failures_section = replace(intent, needed_memory=("failures", "prior_experiments"))
        failures_plan = plan_retrieval(
            failures_section,
            top_k=K,
            per_channel_limit=50,
            kinds=("failure",),
            relational=False,
        )
        assert "vector" in failures_plan.channels
        fails = execute_retrieval(
            failures_plan,
            query_text=text,
            structured=structured,
            lexical=lexical,
            vector=vector,
            failures=failures,
            relational=None,
            query_embedding=query_embedding,
            now=NOW,
            apply_failure_boost=True,
        )
        assert "vector channel skipped" not in " ".join(fails.warnings)
        assert fails.channel_counts.get("vector", 0) > 0, f"{judgment.query_id}: vector leg returned no failure hits"
        assert all("vector" in item.channels for item in fails.top), f"{judgment.query_id}: vector leg missing from fused failures top-10"
        vector_hits += fails.channel_counts.get("vector", 0)

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

    assert vector_hits > 0
    mean_recall = statistics.mean(recalls)
    mean_f_recall = statistics.mean(f_recalls)
    detail = f"recall@{K}={mean_recall:.4f} failure-recall@{K}={mean_f_recall:.4f} target={TARGET}"
    assert mean_recall >= TARGET, f"{detail}\n" + "\n".join(misses)
    assert mean_f_recall >= TARGET, f"{detail}\n" + "\n".join(misses)

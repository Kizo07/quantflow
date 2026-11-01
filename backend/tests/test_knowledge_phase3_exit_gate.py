"""Phase 3 exit gate: sectioned recall@10 >= 0.8 with the vector leg ON.

Seeds the merged eval fixture exactly like the Phase 2 gate (production
write path, lexical index refresh), backfills **both** embedding kinds —
finding vectors via :func:`backfill_findings_embeddings` and experiment
vectors via the Phase 3 experiment text/vector stores — then serves the
scored retrieval through the fork provider boundary (Phase 2c;
``deerflow.knowledge.quantflow_provider.QuantflowKnowledgeProvider`` bound
to the scratch database with the MPNet embedding provider), one sectioned
search per query with the vector channel armed:

* priors section — ``search_scoped`` over the ``findings``/``experiments``/
  ``assumptions`` datasets (planner kinds
  ``finding``/``experiment``/``assumption``), vector armed;
* failures section — ``search_scoped`` over the ``failures`` dataset
  (planner kind ``failure``), vector armed.

Hit → judgment mapping mirrors the Phase 2 gate: provider ``documents()``
title → row rejoin plus a verbatim fixture-text cross-check that must
agree exactly.

The vector-contribution proof stays on the shared serving stack (same
session factory, same MPNet provider, same channel stores, same planner):
per section the gate also runs the direct retrieval plan and asserts the
vector channel contributes to every fused top-10. ``RetrievalHit`` carries
no channel provenance, so per-hit vector attribution cannot cross the
provider boundary; the direct probe plus the provider armed-check below
(the provider must embed a probe query) jointly prove the leg is ON.

Sectioned judgments mirror Phase 2: the priors section scores the
non-failure relevant docs only (every failure-kind doc is also in
``must_recall_failures``, asserted here), failures are judged in the
failures section.

Gate: mean recall@10 >= 0.8 AND mean failure-recall@10 >= 0.8 on the
provider-served top-10 per section, with the vector channel contributing
to every fused top-10 in both sections (a run where the vector leg
silently skips fails the gate even at full recall) and both embedding
kinds fully backfilled.

Provider: the real MPNet checkpoint via
:mod:`deerflow.knowledge.providers_st` (cached, offline). The
deterministic fake carries no semantic signal, so a quality gate with
the vector leg armed cannot use it — noise ranks would only dilute the
lexical/structured signal (measured: priors recall@10 0.75 with fake
vectors ON vs 0.83 OFF). The gate skips when the checkpoint is
unavailable; mechanics stay covered by the fake-backed retrieval tests.
Requires ``deerflow.knowledge.quantflow_provider`` (Phase 2a); skips
until it exists.
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
from deerflow.knowledge import KnowledgeRequestContext, RetrievalCatalogPage
from deerflow.knowledge import pg_retrieval as pg
from deerflow.knowledge.embeddings import (
    EMBEDDING_DIM,
    EmbeddingProvider,
    EmbeddingProviderError,
    backfill_experiments_embeddings,
    backfill_findings_embeddings,
)
from deerflow.knowledge.eval import seed_phase2 as seed
from deerflow.knowledge.providers import search_provider
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

PRIORS_DATASETS = ("findings", "experiments", "assumptions")
FAILURES_DATASETS = ("failures",)
ALL_DATASETS = ("findings", "experiments", "failures", "assumptions")
GATE_USER_ID = "phase3-exit-gate"
GATE_AGENT = "lead_agent"


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


async def _assert_fork_catalog(provider: object, context: KnowledgeRequestContext) -> None:
    """Assert the provider serves exactly the four fixed fork datasets."""
    page = await provider.catalog(context=context, page=1, page_size=100, search="")  # type: ignore[attr-defined]
    assert isinstance(page, RetrievalCatalogPage)
    assert page.page == 1
    assert page.total == len(ALL_DATASETS) == len(page.items)
    assert {item.id for item in page.items} == set(ALL_DATASETS)


async def _title_to_row_map(provider: object, context: KnowledgeRequestContext) -> dict[str, str]:
    """Map provider document titles to row UUIDs across the four datasets.

    ``documents()`` ids are row UUIDs and names are candidate titles, so this
    map rejoins served hits (``source.title``) to seeded rows. Titles must be
    unique and datasets disjoint, else the rejoin would be ambiguous.
    """
    mapping: dict[str, str] = {}
    seen_rows: set[str] = set()
    for dataset_id in ALL_DATASETS:
        page_no = 1
        collected = 0
        while True:
            docs = await provider.documents(dataset_id, context=context, page=page_no, page_size=100, search="")  # type: ignore[attr-defined]
            assert isinstance(docs, RetrievalCatalogPage)
            for item in docs.items:
                assert item.name not in mapping, f"duplicate document title {item.name!r}: rows {mapping[item.name]} and {item.id}"
                assert item.id not in seen_rows, f"row {item.id} listed in two datasets"
                mapping[item.name] = item.id
                seen_rows.add(item.id)
                collected += 1
            if collected >= docs.total or not docs.items:
                break
            page_no += 1
    return mapping


def _resolve_hit_doc_ids(hits: object, title_map: dict[str, str], id_to_doc: dict[str, str]) -> set[str]:
    """Resolve served hits to fixture doc_ids via title → row → doc."""
    doc_ids: set[str] = set()
    for hit in hits:  # type: ignore[union-attr]
        title = hit.source.title if hit.source is not None else None
        row_id = title_map.get(title or "")
        if row_id is not None and row_id in id_to_doc:
            doc_ids.add(id_to_doc[row_id])
    return doc_ids


def _text_match_doc_ids(hits: object, raw_texts: dict[str, str]) -> set[str]:
    """Resolve served hits to doc_ids by verbatim fixture-text matching (cross-check)."""
    blobs = []
    for hit in hits:  # type: ignore[union-attr]
        parts = [hit.content or ""]
        if hit.source is not None:
            parts.append(hit.source.excerpt or "")
        blobs.append(" ".join(" ".join(parts).split()))
    matched: set[str] = set()
    for doc_id, text in raw_texts.items():
        needle = " ".join(text.split())
        if needle and any(needle in blob for blob in blobs):
            matched.add(doc_id)
    return matched


def _assert_hit_attribution(hits: object, allowed_datasets: frozenset[str]) -> None:
    """Assert every served hit carries fork evidence for the section's datasets."""
    for hit in hits:  # type: ignore[union-attr]
        assert hit.source is not None, "gated provider hits must be attributed"
        assert hit.source.provider == "quantflow-pg"
        assert hit.source.excerpt in hit.content
        assert hit.source.dataset_name in allowed_datasets, f"hit {hit.source.title!r} leaks dataset {hit.source.dataset_name!r}"


@pytest.mark.asyncio
async def test_sectioned_recall_gate_with_vector_leg(tmp_path: Path) -> None:
    """Exit gate: sectioned recall@10 and failure-recall@10 each >= 0.8, vector ON, served via the fork provider."""
    qp_module = pytest.importorskip(
        "deerflow.knowledge.quantflow_provider",
        reason="Phase 2c: fork provider not implemented yet (Phase 2a in flight)",
    )
    assert qp_module.PROVIDER_NAME == "quantflow-pg"
    provider_cls = getattr(qp_module, "QuantflowKnowledgeProvider", None) or qp_module.QuantflowPgProvider
    mpnet = _load_provider()
    factory, id_to_doc = _seed_database(tmp_path / "phase3_exit_gate.db", mpnet)
    fork = provider_cls(session_factory=factory, embedding_provider=mpnet)
    # The serving provider itself must embed: a vector-off provider binding
    # fails here, before any per-query probe, even at full recall.
    armed = fork._query_embedding("momentum backtest placebo")
    assert armed is not None and len(armed) == EMBEDDING_DIM, "fork provider must serve the vector leg in the phase 3 gate"
    context = KnowledgeRequestContext(user_id=GATE_USER_ID, agent_name=GATE_AGENT)
    await _assert_fork_catalog(fork, context)
    title_map = await _title_to_row_map(fork, context)
    structured = pg.SQLStructuredLookupStore(factory)
    lexical = pg.SQLLexicalSearchStore(factory)
    vector = pg.SQLVectorSearchStore(factory)
    failures = pg.SQLFailureSearchStore(factory)
    fixture = seed.load_merged_fixture()
    kinds = {doc.doc_id: getattr(doc, "kind", "?") for doc in fixture.documents}
    raw_texts = {doc_id: doc.get("text", "") for doc_id, doc in seed.fixture_raw_documents().items()}

    # Every failure-kind relevant doc must be judged in the failures section;
    # otherwise the priors denominator below would silently drop judgments.
    for judgment in fixture.queries:
        failure_relevant = {d for d in judgment.relevant_doc_ids if kinds.get(d) == "failure"}
        assert failure_relevant <= set(judgment.must_recall_failures), judgment.query_id

    # One batched encode for every query (both sections share the embedding).
    query_texts = [lexical_query_text(judgment.query_text, parse_research_intent(dict(judgment.intent))) for judgment in fixture.queries]
    query_embeddings = {judgment.query_id: embedding for judgment, embedding in zip(fixture.queries, mpnet.embed_batch(query_texts), strict=True)}

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

        priors_result = await search_provider(
            fork,
            text,
            scope={"version": 1, "mode": "selected", "dataset_ids": list(PRIORS_DATASETS)},
            user_id=GATE_USER_ID,
            agent_name=GATE_AGENT,
            now=NOW,
        )
        fails_result = await search_provider(
            fork,
            text,
            scope={"version": 1, "mode": "selected", "dataset_ids": list(FAILURES_DATASETS)},
            user_id=GATE_USER_ID,
            agent_name=GATE_AGENT,
            now=NOW,
        )
        priors_hits = priors_result.hits[:K]
        fails_hits = fails_result.hits[:K]
        _assert_hit_attribution(priors_hits, frozenset(PRIORS_DATASETS))
        _assert_hit_attribution(fails_hits, frozenset(FAILURES_DATASETS))
        priors_ids = _resolve_hit_doc_ids(priors_hits, title_map, id_to_doc)
        fails_ids = _resolve_hit_doc_ids(fails_hits, title_map, id_to_doc)
        assert _text_match_doc_ids(priors_hits, raw_texts) == priors_ids, f"{judgment.query_id}: title/text mapping disagree on priors hits"
        assert _text_match_doc_ids(fails_hits, raw_texts) == fails_ids, f"{judgment.query_id}: title/text mapping disagree on failures hits"
        rel = {d for d in judgment.relevant_doc_ids if kinds.get(d) != "failure"}
        must = set(judgment.must_recall_failures)
        recall = len(rel & priors_ids) / len(rel) if rel else 1.0
        f_recall = len(must & fails_ids) / len(must) if must else 1.0
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

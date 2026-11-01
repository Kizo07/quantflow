"""Phase 2 exit gate: sectioned recall@10 >= 0.8 on the merged eval fixture.

Seeds the merged eval fixture (Phase 0 + Phase 2 extension) into a scratch
SQLite database through the production write path (``write_api`` begin /
failure / commit, mirrored to the ORM rows), refreshes the lexical index,
backfills finding embeddings with the deterministic fake, then serves
retrieval through the fork provider boundary —
``deerflow.knowledge.quantflow_provider.QuantflowKnowledgeProvider`` bound to
the scratch database — one sectioned search per query:

* priors section — ``search_scoped`` over the ``findings``/``experiments``/
  ``assumptions`` datasets (planner kinds
  ``finding``/``experiment``/``assumption``);
* failures section — ``search_scoped`` over the ``failures`` dataset
  (planner kind ``failure``).

The vector channel stays off: the provider is bound without an embedding
provider. Phase 2 embeds findings only, and the probe evidence shows
findings-only vectors crowding experiment needles out of the fused top-10;
the vector leg arms in the Phase 3 gate once experiment embeddings land.

Hit → judgment mapping crosses the boundary on provider identity: the gate
enumerates the provider's own ``documents()`` listing (row-UUID ids,
candidate-title names) into a title → row map, resolves each served hit's
``source.title`` back to a fixture doc_id, and cross-checks the recalled
set against verbatim fixture-text matching (served excerpts are verbatim
candidate text). A title/text disagreement fails loudly — the mapping must
be sound, never approximate.

Sectioned judgments: the priors section is scored on the non-failure
relevant docs only. Every failure-kind doc in ``relevant_doc_ids`` is also
in ``must_recall_failures`` (asserted here), so failures are judged exactly
once, in the failures section — scoring priors on a kind it excludes by
design would be an automatic miss, not a retrieval signal.

Gate: mean recall@10 >= 0.8 AND mean failure-recall@10 >= 0.8, scored on
the provider-served top-10 per section, plus fork-surface checks (provider
name, exact four-dataset catalog, per-hit attribution carrying the
section's dataset names).

Requires ``deerflow.knowledge.quantflow_provider`` (Phase 2a); skips until
it exists.
"""

from __future__ import annotations

import statistics
import uuid
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
    DeterministicEmbeddingProvider,
    backfill_findings_embeddings,
)
from deerflow.knowledge.eval import seed_phase2 as seed
from deerflow.knowledge.providers import search_provider
from deerflow.knowledge.retrieval.planner import (
    lexical_query_text,
    parse_research_intent,
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

PRIORS_DATASETS = ("findings", "experiments", "assumptions")
FAILURES_DATASETS = ("failures",)
ALL_DATASETS = ("findings", "experiments", "failures", "assumptions")
GATE_USER_ID = "phase2-exit-gate"
GATE_AGENT = "lead_agent"
# Pinned fusion reference clock: fixtures are anchored in 2026, so wall-clock
# scoring would decay past the 365d recency window. Same value as phase3.
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
async def test_sectioned_recall_gate(tmp_path: Path) -> None:
    """Exit gate: sectioned recall@10 and failure-recall@10 each >= 0.8, served via the fork provider."""
    qp_module = pytest.importorskip(
        "deerflow.knowledge.quantflow_provider",
        reason="Phase 2c: fork provider not implemented yet (Phase 2a in flight)",
    )
    assert qp_module.PROVIDER_NAME == "quantflow-pg"
    provider_cls = getattr(qp_module, "QuantflowKnowledgeProvider", None) or qp_module.QuantflowPgProvider
    factory, id_to_doc = _seed_database(tmp_path / "phase2_exit_gate.db")
    provider = provider_cls(session_factory=factory)
    context = KnowledgeRequestContext(user_id=GATE_USER_ID, agent_name=GATE_AGENT)
    await _assert_fork_catalog(provider, context)
    title_map = await _title_to_row_map(provider, context)
    fixture = seed.load_merged_fixture()
    kinds = {doc.doc_id: getattr(doc, "kind", "?") for doc in fixture.documents}
    raw_texts = {doc_id: doc.get("text", "") for doc_id, doc in seed.fixture_raw_documents().items()}

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
        text = lexical_query_text(judgment.query_text, intent)
        priors_result = await search_provider(
            provider,
            text,
            scope={"version": 1, "mode": "selected", "dataset_ids": list(PRIORS_DATASETS)},
            user_id=GATE_USER_ID,
            agent_name=GATE_AGENT,
            now=NOW,
        )
        fails_result = await search_provider(
            provider,
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

    mean_recall = statistics.mean(recalls)
    mean_f_recall = statistics.mean(f_recalls)
    detail = f"recall@{K}={mean_recall:.4f} failure-recall@{K}={mean_f_recall:.4f} target={TARGET}"
    assert mean_recall >= TARGET, f"{detail}\n" + "\n".join(misses)
    assert mean_f_recall >= TARGET, f"{detail}\n" + "\n".join(misses)

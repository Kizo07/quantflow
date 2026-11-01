"""PostgreSQL/SQLite bindings for the Research Knowledge Plane retrieval stores.

This module is the Phase 2 integration seam: it binds every read boundary the
retrieval plane defines to the real KB tables (migrations
``0022_knowledge_phase1`` + ``0023_knowledge_findings``):

* :class:`SQLStructuredLookupStore` →
  :class:`deerflow.knowledge.retrieval.planner.StructuredLookupStore`
* :class:`SQLLexicalSearchStore` →
  :class:`deerflow.knowledge.retrieval.planner.LexicalSearchStore`
* :class:`SQLVectorSearchStore` →
  :class:`deerflow.knowledge.retrieval.planner.VectorSearchStore`
* :class:`SQLFailureSearchStore` →
  :class:`deerflow.knowledge.retrieval.planner.FailureSearchStore`
* :class:`SQLRelationalExpansionStore` →
  :class:`deerflow.knowledge.retrieval.planner.RelationalExpansionStore`
* :class:`KnowledgeRetrievalService` →
  :class:`deerflow.knowledge.agent_tools.lookup.KnowledgeRetrievalBackend`
  (adapter over :func:`deerflow.knowledge.retrieval.planner.execute_retrieval`)
* :class:`SQLExperimentLookupStore` →
  :class:`deerflow.knowledge.agent_tools.lookup.ExperimentLookupStore`
* :class:`SQLFindingTextSource` / :class:`SQLEmbeddingVectorStore` →
  :class:`deerflow.knowledge.embeddings.FindingTextSource` /
  :class:`deerflow.knowledge.embeddings.EmbeddingVectorStore`
* :class:`SQLExperimentTextSource` / :class:`SQLExperimentEmbeddingVectorStore` →
  the experiment text/vector boundaries (Phase 3 experiment embeddings)

Canonical store is PostgreSQL; SQLite works for dev/test with identical
filter semantics (ranking differs only where the portable fallback must:
token-overlap scoring instead of ``ts_rank_cd``, brute-force cosine instead
of the ``<=>`` operator).

Row-to-candidate mapping (stable across every channel so fusion dedups
by id onto one payload — first occurrence wins there):

* ``finding`` rows map to kind ``"finding"``, except
  ``finding_type == "failure"`` rows which map to kind ``"failure"``.
  ``title`` is the ``canonical_key`` (findings carry no title column);
  ``text`` is the ``statement``; scope/confidence/validity come straight
  from the row.
* ``experiment`` rows map to kind ``"experiment"``, except
  ``outcome == "failure"`` rows which map to kind ``"failure"`` (the
  outcome also travels in ``metadata`` so fusion's failure boost applies
  either way). Scope derives from the ``methodology`` JSON
  (``asset_class``/``market``/``universe``/``horizon``/``frequency``) and
  the validity window from ``methodology["sample_period"]``; the lexical
  surface is the hypothesis plus the parameters summary (which carries
  signal names, lookbacks, and tool params).
* ``assumption`` rows map to kind ``"assumption"`` with an empty (neutral)
  scope — assumptions carry no scope columns, and unknown scope is never
  incompatibility.

Filter-before-rank: every channel maps rows to candidates, screens them
with :func:`deerflow.knowledge.retrieval.fusion.apply_hard_filters` (zero
logic drift by construction), then ranks the survivors and trims to
``limit``. The ``relational`` channel binds at Phase 3 to the FK graph
(:class:`SQLRelationalExpansionStore`); the service plans two-pass
retrieval with it.

Phase 1a evidence shape: every channel-store search method returns
:class:`EvidenceCandidate` records (and :class:`SQLExperimentLookupStore`
returns :class:`EvidenceExperimentRecord` records) — the planner
``Candidate`` / ``ExperimentRecord`` payload plus upstream retrieval
evidence (``content`` + ``source: KnowledgeSource``). The excerpt always
equals ``content`` verbatim with the content-hash ``source.id`` computed
by ``KnowledgeSource`` itself, so
:func:`deerflow.knowledge.sources.format_result` never strips the
attribution; ``provider`` is ``quantflow-pg``, ``dataset_name`` is the
fork dataset (findings/experiments/failures/assumptions), and
``url``/``page`` stay None (rows carry no locator evidence and upstream
forbids inferred locators). Fusion, the planner, and the service see
unchanged ``Candidate`` payloads — the evidence subclass passes every
``isinstance`` gate and ranking read.

Dialect behavior:

* Lexical: PostgreSQL ranks findings with
  ``ts_rank_cd(search_document, plainto_tsquery('english', :q))`` over the
  indexed ``search_document`` column (populated by
  :func:`refresh_finding_search_documents`; NULL rows stay invisible until
  indexed) and experiments/assumptions with ad-hoc
  ``to_tsvector('english', ...)`` expressions. Everywhere else the portable
  LIKE fallback scores ``title + text`` with
  :func:`deerflow.knowledge.retrieval.planner.like_fallback_idf_scores`
  (IDF-weighted token overlap, BM25-lite) and keeps only positive scores.
* Vector: Phase 3 embeds ``finding`` and ``experiment`` rows alike, so
  the vector channel searches both tables (exact scan first per the KB —
  HNSW only after benchmarking) and merges the two row sets in Python
  before screening. PostgreSQL orders by the ``<=>`` cosine-distance
  operator with the query vector sent as ``CAST(:qv AS VECTOR(768))``
  text, so no ``pgvector`` Python package is needed; SQLite scores the
  JSON fallback with the brute-force :func:`cosine_similarity <deerflow.knowledge.retrieval.planner.cosine_similarity>`
  kernel. Rows without embeddings (NULL, not yet backfilled) never match
  on any dialect.
* Query text reaches PostgreSQL FTS only through ``plainto_tsquery`` with
  bound parameters, so FTS query-syntax injection is impossible by
  construction.

Scaling seams (deliberate; correct-simple now, pushed-down later):
kind-filtered fetches scan the requested tables per channel call (exactly
like the in-test fakes), which is trivial at Phase 2 corpus sizes and keeps
cross-dialect semantics identical. When corpora outgrow full scans, push the
scope predicates into SQL (``scope @>`` / JSON ``->>`` containment plus
validity overlap) and add the HNSW index — the channel result contract
(best-first survivors, trimmed to ``limit``) does not change.

Sessions: every store holds a synchronous session factory and opens one
short session per call, mirroring :mod:`deerflow.knowledge.pg_store`, so a
single instance is safe to share across threads. :func:`open_pg_backends`
builds the full backend bundle from a DSN through the process-wide
sync-engine cache. :func:`bind_knowledge_backends_from_config` is the
startup integration step: it reads the DSN + embedding model +
artifact-store URI from :class:`deerflow.knowledge.config.KnowledgeConfig`
and registers the bundle process-wide (the Gateway lifespan calls it once
at startup).
"""

from __future__ import annotations

import logging
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import bindparam, func, select
from sqlalchemy.orm import Session

from deerflow.knowledge.agent_tools.lookup import (
    ArtifactStoreReader,
    KnowledgeBackends,
    RetrievalDocument,
    RetrievalPage,
    bind_knowledge_backends,
)
from deerflow.knowledge.config import get_knowledge_config
from deerflow.knowledge.contract import KnowledgeRequestContext, KnowledgeSource, RetrievalHit
from deerflow.knowledge.embeddings import (
    EMBEDDING_DIM,
    FAKE_MODEL_ID,
    EmbeddingError,
    EmbeddingProvider,
    embed_texts,
    load_provider,
    register_provider_factory,
    render_embeddable_text,
)
from deerflow.knowledge.pg_store import SQLExperimentSearchStore, row_to_record
from deerflow.knowledge.retrieval.fusion import SCOPE_PLACEHOLDERS, FusionWeights, apply_hard_filters
from deerflow.knowledge.retrieval.planner import (
    DEFAULT_PER_CHANNEL_LIMIT,
    EDGE_TYPES,
    MAX_PER_CHANNEL_LIMIT,
    MAX_TOP_K,
    Candidate,
    ResearchIntent,
    ScopeFilter,
    cosine_similarity,
    execute_retrieval,
    like_fallback_idf_scores,
    plan_retrieval,
)
from deerflow.knowledge.schema.experiments import (
    AssumptionRow,
    ExperimentArtifactRow,
    ExperimentDatasetRow,
    ExperimentRow,
)
from deerflow.knowledge.schema.findings import FindingRow, NativeVector
from deerflow.knowledge.search import ExperimentFilter
from deerflow.knowledge.write_api import ExperimentRecord, KnowledgeError, KnowledgeValidationError
from deerflow.knowledge_scope import execution_scope

logger = logging.getLogger(__name__)

__all__ = [
    "SUMMARY_CHARS",
    "EVIDENCE_PROVIDER",
    "EvidenceCandidate",
    "EvidenceExperimentRecord",
    "candidate_to_evidence_hit",
    "candidate_to_retrieval_hit",
    "record_to_evidence_hit",
    "SQLStructuredLookupStore",
    "SQLLexicalSearchStore",
    "SQLVectorSearchStore",
    "SQLFailureSearchStore",
    "SQLRelationalExpansionStore",
    "SQLExperimentLookupStore",
    "SQLFindingTextSource",
    "SQLEmbeddingVectorStore",
    "SQLExperimentTextSource",
    "SQLExperimentEmbeddingVectorStore",
    "KnowledgeRetrievalService",
    "finding_row_to_candidate",
    "experiment_row_to_candidate",
    "assumption_row_to_candidate",
    "refresh_finding_search_documents",
    "open_pg_backends",
    "bind_knowledge_backends_from_config",
]

#: Summary ceiling for ``RetrievalDocument.summary`` (full text stays one
#: ``ledger_get`` away; truncation is flagged in the payload).
SUMMARY_CHARS = 4000

#: Methodology keys promoted to candidate scope (KB scope fields).
_EXPERIMENT_SCOPE_KEYS = ("asset_class", "market", "universe", "horizon", "frequency")

#: Planner kinds each KB table can satisfy (``skill``/``prior``/``summary``/
#: ``conflict`` arrive in Phases 3-5 and match no rows at Phase 2).
_FINDING_PLANNER_KINDS = frozenset({"finding", "failure"})
_EXPERIMENT_PLANNER_KINDS = frozenset({"experiment", "failure"})
_ASSUMPTION_PLANNER_KINDS = frozenset({"assumption"})

#: Lookup (tool-layer) kinds mapped to planner kinds. ``dossier`` (L0/L1
#: summaries, Phase 4) maps to planner ``summary`` and matches no rows yet;
#: ``artifact`` blobs resolve via ``artifact_manifest``/``artifact_open``,
#: never through fused search, so they contribute no planner kinds.
_LOOKUP_KIND_MAP: dict[str, tuple[str, ...]] = {
    "finding": ("finding",),
    "experiment": ("experiment",),
    "failure": ("failure",),
    "assumption": ("assumption",),
    "conflict": ("conflict",),
    "dossier": ("summary",),
    "artifact": (),
}

#: Planner kinds mapped back to lookup (tool-layer) document kinds.
_PLANNER_KIND_MAP: dict[str, str] = {
    "finding": "finding",
    "experiment": "experiment",
    "failure": "failure",
    "assumption": "assumption",
    "conflict": "conflict",
    "summary": "dossier",
}


def _iso(value: datetime | None) -> str:
    """Render a timestamp as ISO-8601 (``""`` when NULL; naive reads as UTC)."""
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


def _dialect_name(session: Session) -> str:
    """Return the SQL dialect name for ``session`` (``postgresql``/``sqlite``/...)."""
    return session.connection().engine.dialect.name


def _valid_moment(value: Any) -> str | None:
    """Return ``value`` when it parses as ISO-8601, else None (never raises)."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return value.strip()


def _finding_scope(row: FindingRow) -> dict[str, Any]:
    """Return a JSON-safe copy of a finding scope (non-dicts become {})."""
    scope = row.scope
    if not isinstance(scope, dict):
        return {}
    return {str(key): scope[key] for key in scope}


def _experiment_scope_and_window(row: ExperimentRow) -> tuple[dict[str, Any], str | None, str | None]:
    """Derive candidate scope + validity window from an experiment methodology.

    Scope promotes the KB fields when the methodology declares them as
    non-empty strings; the validity window comes from
    ``methodology["sample_period"]`` (``[start, end]``, either end optional).
    Undeclared or malformed values stay absent (neutral), never guessed.
    """
    scope: dict[str, Any] = {}
    methodology = row.methodology if isinstance(row.methodology, dict) else {}
    for key in _EXPERIMENT_SCOPE_KEYS:
        value = methodology.get(key)
        if isinstance(value, str) and value.strip():
            scope[key] = value.strip()
    valid_from: str | None = None
    valid_to: str | None = None
    period = methodology.get("sample_period")
    if isinstance(period, (list, tuple)) and len(period) == 2:
        valid_from = _valid_moment(period[0])
        valid_to = _valid_moment(period[1])
    return scope, valid_from, valid_to


def _experiment_text(row: ExperimentRow | ExperimentRecord) -> str:
    """Build the lexical surface for an experiment row.

    The hypothesis leads; the parameters summary follows (signal names,
    lookbacks, tool params, and any seeder-provided retrieval text); the
    outcome/failure class trails so failure phrasing stays matchable.
    Parts are whitespace-joined and stripped of empties. Records carry the
    same four attributes, so evidence content rebuilds identically from a
    stored row or a fetched record.
    """
    parts = [row.hypothesis or ""]
    parameters = row.parameters if isinstance(row.parameters, dict) else {}
    salient: list[str] = []
    for key in ("eval_retrieval_text", "signal_name", "tool", "type"):
        value = parameters.get(key)
        if isinstance(value, str) and value.strip():
            salient.append(value.strip())
    tool_params = parameters.get("tool_params")
    if isinstance(tool_params, dict):
        for key in sorted(tool_params):
            value = tool_params[key]
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                salient.append(f"{key} {value}")
    if salient:
        parts.append(" ".join(salient))
    if row.outcome:
        parts.append(str(row.outcome))
    if row.failure_class:
        parts.append(f"failure {row.failure_class}")
    return " ".join(part for part in parts if part and part.strip()).strip()


def finding_row_to_candidate(row: FindingRow) -> Candidate:
    """Map one ``FindingRow`` to a retrieval :class:`Candidate`.

    Kind is ``"failure"`` for ``finding_type == "failure"`` rows, else
    ``"finding"``. ``evidence`` is empty at Phase 2 (``finding_evidence``
    edges land in Phase 3); the open path is ``ledger_get``.
    """
    return Candidate(
        id=str(row.id),
        kind="failure" if row.finding_type == "failure" else "finding",
        title=row.canonical_key or "",
        text=row.statement or "",
        scope=_finding_scope(row),
        status=row.status or "",
        finding_type=row.finding_type,
        project_id=str(row.project_id) if row.project_id is not None else None,
        valid_from=_iso(row.effective_from) or None,
        valid_to=_iso(row.effective_to) or None,
        recorded_at=_iso(row.recorded_at),
    )


def experiment_row_to_candidate(row: ExperimentRow) -> Candidate:
    """Map one ``ExperimentRow`` to a retrieval :class:`Candidate`.

    Kind is ``"failure"`` for ``outcome == "failure"`` rows, else
    ``"experiment"``; the outcome/failure class also travel in
    ``metadata`` so fusion recognizes failures by marker as well as kind.
    """
    scope, valid_from, valid_to = _experiment_scope_and_window(row)
    metadata: dict[str, Any] = {}
    if row.outcome is not None:
        metadata["outcome"] = row.outcome
    if row.failure_class is not None:
        metadata["failure_class"] = row.failure_class
    if row.replicated_experiment_id is not None:
        metadata["replicated_experiment_id"] = str(row.replicated_experiment_id)
    family_hash = row.experiment_family_hash if isinstance(row.experiment_family_hash, str) else None
    return Candidate(
        id=str(row.id),
        kind="failure" if row.outcome == "failure" else "experiment",
        title=row.hypothesis or "",
        text=_experiment_text(row),
        scope=scope,
        status=row.status or "",
        project_id=str(row.project_id) if row.project_id is not None else None,
        valid_from=valid_from,
        valid_to=valid_to,
        recorded_at=_iso(row.started_at),
        family_hash=family_hash,
        metadata=metadata,
    )


def assumption_row_to_candidate(row: AssumptionRow) -> Candidate:
    """Map one ``AssumptionRow`` to a retrieval :class:`Candidate`.

    Assumptions carry no scope columns, so scope stays empty (neutral);
    the category/sensitivity/test state travel in ``metadata``.
    """
    statement = row.statement or ""
    metadata: dict[str, Any] = {"category": row.category}
    if row.sensitivity is not None:
        metadata["sensitivity"] = row.sensitivity
    metadata["tested"] = bool(row.tested)
    if row.experiment_id is not None:
        metadata["experiment_id"] = str(row.experiment_id)
    return Candidate(
        id=str(row.id),
        kind="assumption",
        title=statement[:120].strip(),
        text=statement,
        status=row.status or "",
        metadata=metadata,
    )


#: Upstream provider name stamped on every channel-hit source (the fork
#: extension; see ``backend/docs/KNOWLEDGE.md`` "Local-only deployment").
EVIDENCE_PROVIDER = "quantflow-pg"

#: Planner kind → upstream dataset name (KNOWLEDGE.md "Dataset to kind
#: mapping"). ``failure`` draws from both tables but reports one dataset.
_KIND_DATASETS: dict[str, str] = {
    "finding": "findings",
    "experiment": "experiments",
    "failure": "failures",
    "assumption": "assumptions",
}

#: Evidence ceilings mirror the upstream ``KnowledgeSource`` field limits.
#: Content and excerpt share one string, so the pair truncates together and
#: stays verbatim-equal at any length.
_MAX_EVIDENCE_CHARS = 100_000
_MAX_EVIDENCE_TITLE_CHARS = 500

#: Base dataclass field names copied when attaching evidence (computed once;
#: the evidence subclasses add only ``content``/``source``).
_CANDIDATE_FIELD_NAMES = tuple(entry.name for entry in fields(Candidate))
_RECORD_FIELD_NAMES = tuple(entry.name for entry in fields(ExperimentRecord))


def _evidence_text(text: str, title: str, *, kind: str, row_id: str) -> str:
    """Derive non-empty evidence content for one hit (stripped, ceilinged).

    Falls back from body text to title to a stable ``kind:id`` label so the
    excerpt is never empty (upstream requires ``min_length=1``); the
    fallbacks only fire for degenerate rows.
    """
    content = (text or "").strip() or (title or "").strip() or f"{kind}:{row_id}"
    if len(content) > _MAX_EVIDENCE_CHARS:
        content = content[:_MAX_EVIDENCE_CHARS].rstrip()
    return content


def _evidence_title(title: str, *, kind: str, row_id: str) -> str:
    """Derive a valid upstream title (non-empty, within the 500-char limit)."""
    cleaned = (title or "").strip() or f"{kind}:{row_id}"
    if len(cleaned) > _MAX_EVIDENCE_TITLE_CHARS:
        cleaned = cleaned[:_MAX_EVIDENCE_TITLE_CHARS]
    return cleaned


def _checked_evidence(content: Any, source: Any) -> str:
    """Validate one evidence pair, returning the stripped content.

    Raises:
        KnowledgeValidationError: On empty content, a non-``KnowledgeSource``
            source, or an excerpt that differs from content (``format_result``
            strips such pairs, so they are refused at construction).
    """
    cleaned = content.strip() if isinstance(content, str) else ""
    if not cleaned:
        raise KnowledgeValidationError("evidence content must be a non-empty string.")
    if source is not None:
        if not isinstance(source, KnowledgeSource):
            raise KnowledgeValidationError(f"evidence source must be a KnowledgeSource or None, got {type(source).__name__}.")
        if source.excerpt != cleaned:
            raise KnowledgeValidationError("evidence source.excerpt must equal content verbatim (else format_result strips the attribution).")
    return cleaned


def _copied_field_values(obj: Any, names: Sequence[str]) -> dict[str, Any]:
    """Copy dataclass field values, shallow-copying mutable containers.

    The evidence hit never aliases the source object's dicts/lists, so a
    mutated ranking payload cannot corrupt the converted hit or vice versa.
    """
    values: dict[str, Any] = {}
    for name in names:
        value = getattr(obj, name)
        if isinstance(value, dict):
            value = dict(value)
        elif isinstance(value, list):
            value = list(value)
        values[name] = value
    return values


@dataclass(frozen=True)
class EvidenceCandidate(Candidate):
    """A planner candidate carrying its upstream retrieval evidence (Phase 1a).

    Every channel-store search method returns these instead of bare
    :class:`Candidate` records. Fusion, the planner, and the service see an
    unchanged candidate (row-UUID ``id``, ``kind``, ``scope``, validity —
    the ranking identity never moves to the evidence fields); upstream
    readers see ``content`` plus ``source`` (verbatim excerpt, content-hash
    ``source.id`` computed by :class:`KnowledgeSource` itself). ``source``
    mirrors upstream optionality (None = unattributed content), but the
    channel boundaries always attach one.
    """

    content: str = ""
    source: KnowledgeSource | None = None

    def __post_init__(self) -> None:
        """Validate the candidate fields plus the evidence pair."""
        super().__post_init__()
        object.__setattr__(self, "content", _checked_evidence(self.content, self.source))


@dataclass(frozen=True)
class EvidenceExperimentRecord(ExperimentRecord):
    """An experiment record carrying its upstream retrieval evidence (Phase 1a).

    :class:`SQLExperimentLookupStore` returns these from every lookup/search
    method. Tool consumers see an unchanged record (``to_dict()`` keeps the
    base keys); upstream readers see ``content`` plus ``source``.
    """

    content: str = ""
    source: KnowledgeSource | None = None

    def __post_init__(self) -> None:
        """Validate the evidence pair (the base record needs no validation)."""
        object.__setattr__(self, "content", _checked_evidence(self.content, self.source))


def candidate_to_evidence_hit(candidate: Candidate) -> EvidenceCandidate:
    """Attach upstream retrieval evidence to one channel candidate (Phase 1a).

    Idempotent: ``EvidenceCandidate`` input passes through untouched. The
    excerpt is the candidate text verbatim (``format_result`` keeps the
    attribution); ``provider`` is ``quantflow-pg``, ``title`` the candidate
    title, ``dataset_name`` the fork dataset for the kind, and ``url``/
    ``page`` None (rows carry no locator evidence and upstream forbids
    inferred locators). ``source.id`` is the content hash computed by
    ``KnowledgeSource`` — never invented, never the row UUID.

    Raises:
        KnowledgeValidationError: When ``candidate`` is not a Candidate.
    """
    if isinstance(candidate, EvidenceCandidate):
        return candidate
    if not isinstance(candidate, Candidate):
        raise KnowledgeValidationError(f"candidate must be a Candidate, got {type(candidate).__name__}.")
    content = _evidence_text(candidate.text, candidate.title, kind=candidate.kind, row_id=candidate.id)
    source = KnowledgeSource(
        provider=EVIDENCE_PROVIDER,
        title=_evidence_title(candidate.title, kind=candidate.kind, row_id=candidate.id),
        excerpt=content,
        dataset_name=_KIND_DATASETS.get(candidate.kind),
    )
    return EvidenceCandidate(**_copied_field_values(candidate, _CANDIDATE_FIELD_NAMES), content=content, source=source)


def candidate_to_retrieval_hit(candidate: Candidate, *, score: float | None = None) -> RetrievalHit:
    """Render one channel candidate as a genuine upstream ``RetrievalHit`` (Phase 1a).

    Normalizes through :func:`candidate_to_evidence_hit` first, so bare
    candidates convert identically. ``score`` stays None unless the caller
    supplies a fused/rank score (upstream leaves unscored hits None).
    """
    hit = candidate_to_evidence_hit(candidate)
    return RetrievalHit(content=hit.content, source=hit.source, score=score)


def record_to_evidence_hit(record: ExperimentRecord) -> EvidenceExperimentRecord:
    """Attach upstream retrieval evidence to one experiment record (Phase 1a).

    Idempotent: ``EvidenceExperimentRecord`` input passes through untouched.
    Content is the record's lexical surface (hypothesis plus the parameters
    summary, mirroring :func:`_experiment_text`); ``dataset_name`` is
    ``"failures"`` for ``outcome == "failure"`` records, else
    ``"experiments"`` (the same fork rule as the candidate kind mapping).

    Raises:
        KnowledgeValidationError: When ``record`` is not an ExperimentRecord.
    """
    if isinstance(record, EvidenceExperimentRecord):
        return record
    if not isinstance(record, ExperimentRecord):
        raise KnowledgeValidationError(f"record must be an ExperimentRecord, got {type(record).__name__}.")
    kind = "failure" if record.outcome == "failure" else "experiment"
    content = _evidence_text(_experiment_text(record), record.hypothesis, kind=kind, row_id=record.id)
    source = KnowledgeSource(
        provider=EVIDENCE_PROVIDER,
        title=_evidence_title(record.hypothesis, kind=kind, row_id=record.id),
        excerpt=content,
        dataset_name=_KIND_DATASETS[kind],
    )
    return EvidenceExperimentRecord(**_copied_field_values(record, _RECORD_FIELD_NAMES), content=content, source=source)


def _stable_order(candidates: list[Candidate], score_fn: Callable[[Candidate], float] | None = None) -> list[Candidate]:
    """Order candidates deterministically: score, then recency, then id.

    Primary key is ``score_fn`` descending when given; the tie-break chain
    is recorded moment descending (unknown moments sort last), then id
    ascending. Stable sorts compose the keys without tuple-type mixing.
    """
    ordered = sorted(candidates, key=lambda item: item.id)
    with_moment = sorted([item for item in ordered if item.recorded_at], key=lambda item: item.recorded_at, reverse=True)
    ordered = with_moment + [item for item in ordered if not item.recorded_at]
    if score_fn is not None:
        ordered = sorted(ordered, key=score_fn, reverse=True)
    return ordered


def _screen(candidates: list[Candidate], scope: ScopeFilter) -> list[Candidate]:
    """Apply the fusion hard filters, returning the survivors in order."""
    survivors, _ = apply_hard_filters(candidates, scope)
    return survivors


def _ranked_like_hits(survivors: list[Candidate], query_text: str, *, limit: int) -> list[Candidate]:
    """Rank screened candidates with IDF-weighted LIKE scores (positive only).

    Score ties break by recency then id via :func:`_stable_order`, so pages
    stay deterministic on every dialect.
    """
    scores = like_fallback_idf_scores([f"{item.title} {item.text}" for item in survivors], query_text)
    positive = [(score, item) for score, item in zip(scores, survivors) if score > 0.0]
    by_id = {item.id: score for score, item in positive}
    return _stable_order([item for _, item in positive], score_fn=lambda item: by_id[item.id])[:limit]


#: Scope fields counted for structured match-strength ranking.
_MATCH_FIELDS = ("asset_class", "universe", "horizon", "frequency", "dataset_family")


def _filter_declares_scope(scope: ScopeFilter) -> bool:
    """Return True when the filter constrains any scope field (placeholders excluded)."""
    for field_name in _MATCH_FIELDS:
        value = _match_text(getattr(scope, field_name, None))
        if value is not None and value not in SCOPE_PLACEHOLDERS:
            return True
    return any(isinstance(market, str) and market.strip() for market in scope.markets)


def _declares_scope(candidate_scope: Mapping[str, Any]) -> bool:
    """Return True when a candidate scope states any non-placeholder value."""
    for field_name in _MATCH_FIELDS:
        value = _match_text(candidate_scope.get(field_name))
        if value is not None and value not in SCOPE_PLACEHOLDERS:
            return True
    raw = candidate_scope.get("markets", candidate_scope.get("market"))
    if isinstance(raw, str):
        raw = [raw]
    return any(isinstance(item, str) and item.strip() for item in raw) if isinstance(raw, (list, tuple)) else False


def _match_text(value: Any) -> str | None:
    """Normalize one scope value to a comparable lowercase string (None when undeclared)."""
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def scope_match_count(candidate_scope: Mapping[str, Any], scope: ScopeFilter) -> int:
    """Count scope fields where the filter and candidate agree (0..6).

    Each of ``asset_class``/``universe``/``horizon``/``frequency``/
    ``dataset_family`` contributes 1 when both sides declare it and the
    values match case-insensitively; ``markets`` contributes 1 on any
    overlap. Undeclared fields are neutral (0, never negative): an empty
    filter scores every candidate 0, preserving recency order. Placeholder
    values (``SCOPE_PLACEHOLDERS``) on either side are likewise neutral —
    unspecified never matches. Mismatches score 0 here — incompatibility
    is fusion's hard-filter job, and the channel must never be stricter
    than fusion.
    """
    matches = 0
    for field_name in _MATCH_FIELDS:
        wanted = _match_text(getattr(scope, field_name, None))
        if wanted is None or wanted in SCOPE_PLACEHOLDERS:
            continue
        actual = _match_text(candidate_scope.get(field_name))
        if actual is not None and actual not in SCOPE_PLACEHOLDERS and actual == wanted:
            matches += 1
    wanted_markets = {market.strip().lower() for market in scope.markets if isinstance(market, str) and market.strip()}
    if wanted_markets:
        raw = candidate_scope.get("markets", candidate_scope.get("market"))
        if isinstance(raw, str):
            raw = [raw]
        actual = {str(item).strip().lower() for item in raw} if isinstance(raw, (list, tuple)) else set()
        actual.discard("")
        if actual & wanted_markets:
            matches += 1
    return matches


def _finding_fts_statement(query_text: str):
    """Build the PostgreSQL FTS select over indexed ``finding.search_document``."""
    from sqlalchemy import Text

    ts_query = func.plainto_tsquery("english", bindparam("lex_q", value=query_text, type_=Text()))
    rank = func.ts_rank_cd(FindingRow.search_document, ts_query).label("lex_rank")
    return select(FindingRow, rank).where(FindingRow.search_document.op("@@")(ts_query)).order_by(rank.desc(), FindingRow.recorded_at.desc().nulls_last(), FindingRow.id.asc())


def _experiment_fts_statement(query_text: str):
    """Build the PostgreSQL FTS select over hypothesis + parameters text."""
    from sqlalchemy import Text

    ts_query = func.plainto_tsquery("english", bindparam("lex_q", value=query_text, type_=Text()))
    document = func.to_tsvector("english", ExperimentRow.hypothesis.op("||")(" ").op("||")(func.coalesce(ExperimentRow.parameters.cast(Text()), "")))
    rank = func.ts_rank_cd(document, ts_query).label("lex_rank")
    return select(ExperimentRow, rank).where(document.op("@@")(ts_query)).order_by(rank.desc(), ExperimentRow.started_at.desc().nulls_last(), ExperimentRow.id.asc())


def _assumption_fts_statement(query_text: str):
    """Build the PostgreSQL FTS select over assumption statements."""
    from sqlalchemy import Text

    ts_query = func.plainto_tsquery("english", bindparam("lex_q", value=query_text, type_=Text()))
    document = func.to_tsvector("english", AssumptionRow.statement)
    rank = func.ts_rank_cd(document, ts_query).label("lex_rank")
    return select(AssumptionRow, rank).where(document.op("@@")(ts_query)).order_by(rank.desc(), AssumptionRow.id.asc())


def _finding_vector_statement(literal: str):
    """Build the PostgreSQL exact cosine select over findings (``<=>`` over ``VECTOR(768)``)."""
    from sqlalchemy import Text

    probe = FindingRow.embedding.op("<=>")(bindparam("qv", value=literal, type_=Text()).cast(NativeVector(EMBEDDING_DIM)))
    return select(FindingRow, probe.label("vec_distance")).where(FindingRow.embedding.is_not(None)).order_by(probe.asc(), FindingRow.recorded_at.desc().nulls_last(), FindingRow.id.asc())


def _experiment_vector_statement(literal: str):
    """Build the PostgreSQL exact cosine select over experiments (``<=>`` over ``VECTOR(768)``)."""
    from sqlalchemy import Text

    probe = ExperimentRow.embedding.op("<=>")(bindparam("qv", value=literal, type_=Text()).cast(NativeVector(EMBEDDING_DIM)))
    return select(ExperimentRow, probe.label("vec_distance")).where(ExperimentRow.embedding.is_not(None)).order_by(probe.asc(), ExperimentRow.started_at.desc().nulls_last(), ExperimentRow.id.asc())


def _finding_reindex_update(target_ids):
    """Build the PostgreSQL TSVECTOR refresh update for the given finding ids."""
    return FindingRow.__table__.update().where(FindingRow.id.in_(list(target_ids))).values(search_document=func.to_tsvector("english", FindingRow.canonical_key.op("||")(" ").op("||")(FindingRow.statement)))


class _ChannelStoreBase:
    """Shared session handling for the SQL channel stores (mirrors ``pg_store``)."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        """Bind the store to a zero-argument session factory (one session per call)."""
        self._sessions = session_factory

    @staticmethod
    def _wants(kinds: Sequence[str], supported: frozenset[str]) -> bool:
        """Return True when any requested kind is served by this table."""
        return any(kind in supported for kind in kinds)


class SQLStructuredLookupStore(_ChannelStoreBase):
    """``StructuredLookupStore`` over finding/experiment/assumption rows.

    Returns scope-compatible rows ranked by scope-match strength
    (:func:`scope_match_count` descending, then recency, then id),
    trimmed to ``limit``. Scope screening is the fusion hard-filter set
    (via :func:`_screen`), so a great scope score can never rescue an
    incompatible row — and the store itself is never stricter than
    fusion. An empty filter scores every row 0, degrading honestly to
    recency order (no constraint, no signal).
    """

    def structured_lookup(self, scope: ScopeFilter, *, kinds: Sequence[str], limit: int) -> list[Candidate]:
        """Return up to ``limit`` scope-compatible candidates, best scope match first.

        Hits carry upstream retrieval evidence (Phase 1a).
        """
        with self._sessions() as session:
            candidates: list[Candidate] = []
            if self._wants(kinds, _FINDING_PLANNER_KINDS):
                rows = session.scalars(select(FindingRow).order_by(FindingRow.recorded_at.desc().nulls_last(), FindingRow.id.asc()))
                candidates.extend(finding_row_to_candidate(row) for row in rows)
            if self._wants(kinds, _EXPERIMENT_PLANNER_KINDS):
                rows = session.scalars(select(ExperimentRow).order_by(ExperimentRow.started_at.desc().nulls_last(), ExperimentRow.id.asc()))
                candidates.extend(experiment_row_to_candidate(row) for row in rows)
            if self._wants(kinds, _ASSUMPTION_PLANNER_KINDS):
                rows = session.scalars(select(AssumptionRow).order_by(AssumptionRow.id.asc()))
                candidates.extend(assumption_row_to_candidate(row) for row in rows)
        wanted = frozenset(kinds)
        candidates = [item for item in candidates if item.kind in wanted]
        survivors = _screen(candidates, scope)
        counts = {item.id: scope_match_count(item.scope, scope) for item in survivors}
        if _filter_declares_scope(scope):
            # A filter that constrains scope prunes rows matching nothing:
            # zero-match rows contribute pure RRF noise (every row would
            # otherwise appear in every channel). Scope-empty rows stay —
            # declaring nothing contradicts nothing (neutral, like "none").
            survivors = [item for item in survivors if counts[item.id] > 0 or not _declares_scope(item.scope)]
        ranked = _stable_order(survivors, score_fn=lambda item: float(counts[item.id]))[:limit]
        return [candidate_to_evidence_hit(item) for item in ranked]


class SQLLexicalSearchStore(_ChannelStoreBase):
    """``LexicalSearchStore`` with PostgreSQL FTS + portable LIKE fallback.

    PostgreSQL ranks findings with ``ts_rank_cd`` over the indexed
    ``search_document`` column (rows with NULL documents stay invisible
    until :func:`refresh_finding_search_documents` indexes them) and
    experiments/assumptions with ad-hoc ``to_tsvector`` expressions over
    the hypothesis/statement text. Every other dialect scores
    ``title + text`` with :func:`like_fallback_idf_scores` (IDF-weighted,
    BM25-lite) and keeps only positive scores. Cross-table ranks merge in
    Python (score descending,
    then recency, then id) so pages stay deterministic on every dialect.
    """

    def lexical_search(self, query_text: str, scope: ScopeFilter, *, kinds: Sequence[str], limit: int) -> list[Candidate]:
        """Return up to ``limit`` lexical matches, best-first.

        Hits carry upstream retrieval evidence (Phase 1a).
        """
        with self._sessions() as session:
            if _dialect_name(session) == "postgresql":
                found = self._search_postgres(session, query_text, scope, kinds, limit)
            else:
                found = self._search_portable(session, query_text, scope, kinds, limit)
        return [candidate_to_evidence_hit(item) for item in found]

    def _search_portable(self, session: Session, query_text: str, scope: ScopeFilter, kinds: Sequence[str], limit: int) -> list[Candidate]:
        """Score kind-filtered rows with the IDF-weighted LIKE fallback (positive scores only)."""
        candidates = self._candidates_for_kinds(session, kinds)
        return _ranked_like_hits(_screen(candidates, scope), query_text, limit=limit)

    def _search_postgres(self, session: Session, query_text: str, scope: ScopeFilter, kinds: Sequence[str], limit: int) -> list[Candidate]:
        """Rank kind-filtered rows with ``ts_rank_cd`` (FTS matches only)."""
        ranked: list[tuple[float, Candidate]] = []
        if self._wants(kinds, _FINDING_PLANNER_KINDS):
            for row, score in session.execute(_finding_fts_statement(query_text)):
                ranked.append((float(score or 0.0), finding_row_to_candidate(row)))
        if self._wants(kinds, _EXPERIMENT_PLANNER_KINDS):
            for row, score in session.execute(_experiment_fts_statement(query_text)):
                ranked.append((float(score or 0.0), experiment_row_to_candidate(row)))
        if self._wants(kinds, _ASSUMPTION_PLANNER_KINDS):
            for row, score in session.execute(_assumption_fts_statement(query_text)):
                ranked.append((float(score or 0.0), assumption_row_to_candidate(row)))
        wanted = frozenset(kinds)
        ranked = [(score, item) for score, item in ranked if item.kind in wanted]
        survivors = _screen([item for _, item in ranked], scope)
        survivor_ids = {item.id for item in survivors}
        kept = [(score, item) for score, item in ranked if item.id in survivor_ids]
        by_id = {item.id: score for score, item in kept}
        return _stable_order([item for _, item in kept], score_fn=lambda item: by_id[item.id])[:limit]

    def _candidates_for_kinds(self, session: Session, kinds: Sequence[str]) -> list[Candidate]:
        """Fetch kind-filtered rows and map them to candidates (portable path)."""
        candidates: list[Candidate] = []
        if self._wants(kinds, _FINDING_PLANNER_KINDS):
            candidates.extend(finding_row_to_candidate(row) for row in session.scalars(select(FindingRow)))
        if self._wants(kinds, _EXPERIMENT_PLANNER_KINDS):
            candidates.extend(experiment_row_to_candidate(row) for row in session.scalars(select(ExperimentRow)))
        if self._wants(kinds, _ASSUMPTION_PLANNER_KINDS):
            candidates.extend(assumption_row_to_candidate(row) for row in session.scalars(select(AssumptionRow)))
        wanted = frozenset(kinds)
        return [item for item in candidates if item.kind in wanted]


def _vector_literal(vector: Sequence[float]) -> str:
    """Render a query embedding as a pgvector input literal (``[1,2,...]``).

    Raises:
        KnowledgeValidationError: On an empty vector, a dimension other
            than :data:`EMBEDDING_DIM`, or a non-finite component.
    """
    values = list(vector)
    if len(values) != EMBEDDING_DIM:
        raise KnowledgeValidationError(f"query_embedding has dimension {len(values)}, expected {EMBEDDING_DIM} (the KB VECTOR column width).")
    parts: list[str] = []
    for component in values:
        if not isinstance(component, (int, float)) or isinstance(component, bool) or not math.isfinite(component):
            raise KnowledgeValidationError(f"query_embedding must contain only finite numbers, got {component!r}.")
        parts.append(repr(float(component)))
    return "[" + ",".join(parts) + "]"


class SQLVectorSearchStore(_ChannelStoreBase):
    """``VectorSearchStore`` over finding + experiment embeddings (exact scan first).

    Phase 3 embeds both tables, so the ``finding``/``experiment``/``failure``
    kinds can all match (``failure`` draws from both tables: failed
    experiments map to kind ``"failure"``, as do failure findings); other
    requested kinds simply contribute no rows. PostgreSQL orders each table
    by the ``<=>`` cosine-distance operator with the query vector bound as
    ``CAST(:qv AS VECTOR(768))`` text (no ``pgvector`` Python package
    required) and merges the two row sets in Python; every other dialect
    scores the JSON fallback with the brute-force cosine kernel. Rows
    without embeddings (NULL, not yet backfilled) never match on any
    dialect; malformed stored vectors are skipped with a warning so one
    corrupt row cannot break retrieval.
    """

    def vector_search(self, query_embedding: Sequence[float], scope: ScopeFilter, *, kinds: Sequence[str], limit: int) -> list[Candidate]:
        """Return up to ``limit`` embedded findings/experiments nearest the query, best-first.

        Hits carry upstream retrieval evidence (Phase 1a).
        """
        literal = _vector_literal(query_embedding)
        with self._sessions() as session:
            if _dialect_name(session) == "postgresql":
                found = self._search_postgres(session, literal, scope, kinds, limit)
            else:
                found = self._search_portable(session, query_embedding, scope, kinds, limit)
        return [candidate_to_evidence_hit(item) for item in found]

    def _search_portable(self, session: Session, query_embedding: Sequence[float], scope: ScopeFilter, kinds: Sequence[str], limit: int) -> list[Candidate]:
        """Brute-force cosine search over the SQLite JSON fallback (both tables)."""
        wanted = frozenset(kinds)
        want_findings = self._wants(kinds, _FINDING_PLANNER_KINDS)
        want_experiments = self._wants(kinds, _EXPERIMENT_PLANNER_KINDS)
        if not want_findings and not want_experiments:
            return []
        query = tuple(float(component) for component in query_embedding)
        scored: list[tuple[float, Candidate]] = []
        if want_findings:
            for row in session.scalars(select(FindingRow).where(FindingRow.embedding.is_not(None))):
                candidate = finding_row_to_candidate(row)
                if candidate.kind not in wanted:
                    continue
                score = self._portable_score(query, row.embedding, label=f"finding {row.id}")
                if score is not None:
                    scored.append((score, candidate))
        if want_experiments:
            for row in session.scalars(select(ExperimentRow).where(ExperimentRow.embedding.is_not(None))):
                candidate = experiment_row_to_candidate(row)
                if candidate.kind not in wanted:
                    continue
                score = self._portable_score(query, row.embedding, label=f"experiment {row.id}")
                if score is not None:
                    scored.append((score, candidate))
        survivors = _screen([item for _, item in scored], scope)
        survivor_ids = {item.id for item in survivors}
        kept = [(score, item) for score, item in scored if item.id in survivor_ids]
        by_id = {item.id: score for score, item in kept}
        return _stable_order([item for _, item in kept], score_fn=lambda item: by_id[item.id])[:limit]

    @staticmethod
    def _portable_score(query: tuple[float, ...], stored: Any, *, label: str) -> float | None:
        """Score one stored vector, returning None (with a warning) when malformed."""
        if not isinstance(stored, Sequence) or isinstance(stored, (str, bytes)) or not stored:
            logger.warning("SQLVectorSearchStore: skipping %s with malformed stored embedding", label)
            return None
        try:
            return cosine_similarity(query, tuple(float(component) for component in stored))
        except (KnowledgeValidationError, ValueError, TypeError):
            logger.warning("SQLVectorSearchStore: skipping %s with malformed stored embedding", label)
            return None

    def _search_postgres(self, session: Session, literal: str, scope: ScopeFilter, kinds: Sequence[str], limit: int) -> list[Candidate]:
        """Exact cosine search with the ``<=>`` operator (NULLs excluded, both tables)."""
        wanted = frozenset(kinds)
        rows: list[tuple[float, Candidate]] = []
        if self._wants(kinds, _FINDING_PLANNER_KINDS):
            rows.extend((float(distance), finding_row_to_candidate(row)) for row, distance in session.execute(_finding_vector_statement(literal)))
        if self._wants(kinds, _EXPERIMENT_PLANNER_KINDS):
            rows.extend((float(distance), experiment_row_to_candidate(row)) for row, distance in session.execute(_experiment_vector_statement(literal)))
        rows = [(distance, item) for distance, item in rows if item.kind in wanted]
        survivors = _screen([item for _, item in rows], scope)
        survivor_ids = {item.id for item in survivors}
        kept = [(distance, item) for distance, item in rows if item.id in survivor_ids]
        # SQL already ordered by distance ascending; re-sort in Python so the
        # recency/id tie-break matches every other channel exactly.
        by_id = {item.id: -distance for distance, item in kept}
        return _stable_order([item for _, item in kept], score_fn=lambda item: by_id[item.id])[:limit]


class SQLFailureSearchStore(_ChannelStoreBase):
    """``FailureSearchStore``: failed experiments + failure findings.

    The dedicated negative-evidence channel (Reflexion/ExpeL): the row set
    is fixed (``experiment WHERE outcome = 'failure'`` plus
    ``finding WHERE finding_type = 'failure'``), ranked by lexical
    relevance to the query — PostgreSQL FTS expressions, portable LIKE
    fallback elsewhere — then recency, then id. Fusion adds the failure
    boost on top of channel inclusion.
    """

    def search_failures(self, query_text: str, scope: ScopeFilter, *, limit: int) -> list[Candidate]:
        """Return up to ``limit`` failure candidates relevant to the query, best-first.

        Hits carry upstream retrieval evidence (Phase 1a).
        """
        with self._sessions() as session:
            if _dialect_name(session) == "postgresql":
                found = self._search_postgres(session, query_text, scope, limit)
            else:
                found = self._search_portable(session, query_text, scope, limit)
        return [candidate_to_evidence_hit(item) for item in found]

    def _failure_candidates(self, session: Session) -> list[Candidate]:
        """Fetch the fixed failure row set and map it to candidates."""
        candidates = [finding_row_to_candidate(row) for row in session.scalars(select(FindingRow).where(FindingRow.finding_type == "failure"))]
        candidates.extend(experiment_row_to_candidate(row) for row in session.scalars(select(ExperimentRow).where(ExperimentRow.outcome == "failure")))
        return candidates

    def _search_portable(self, session: Session, query_text: str, scope: ScopeFilter, limit: int) -> list[Candidate]:
        """Rank failure rows with the IDF-weighted LIKE fallback (positive scores only)."""
        return _ranked_like_hits(_screen(self._failure_candidates(session), scope), query_text, limit=limit)

    def _search_postgres(self, session: Session, query_text: str, scope: ScopeFilter, limit: int) -> list[Candidate]:
        """Rank failure rows with ``ts_rank_cd`` (FTS matches only)."""
        ranked: list[tuple[float, Candidate]] = []
        statement = _finding_fts_statement(query_text).where(FindingRow.finding_type == "failure")
        for row, score in session.execute(statement):
            ranked.append((float(score or 0.0), finding_row_to_candidate(row)))
        statement = _experiment_fts_statement(query_text).where(ExperimentRow.outcome == "failure")
        for row, score in session.execute(statement):
            ranked.append((float(score or 0.0), experiment_row_to_candidate(row)))
        survivors = _screen([item for _, item in ranked], scope)
        survivor_ids = {item.id for item in survivors}
        kept = [(score, item) for score, item in ranked if item.id in survivor_ids]
        by_id = {item.id: score for score, item in kept}
        return _stable_order([item for _, item in kept], score_fn=lambda item: by_id[item.id])[:limit]


class SQLRelationalExpansionStore(_ChannelStoreBase):
    """``RelationalExpansionStore`` over the KB foreign-key graph (one hop).

    Phase 3 binds the planner's relational seam to the relations already in
    the schema — no ``knowledge_edge`` table, no migration. Each requested
    edge type traverses its backing relation in both directions (undirected:
    a seed reaches its parents and its children alike), and ``related_to``
    / ``uses`` reach laterally-linked experiments:

    * ``supersedes`` — ``finding.supersedes_id`` (the seed's target plus
      rows pointing at the seed);
    * ``derived_from`` — ``experiment.parent_experiment_id`` (parent plus
      children, so follow-ups and their failures surface together);
    * ``replicates`` — ``experiment.replicated_experiment_id`` (target plus
      replicators);
    * ``applies_to`` — ``assumption.experiment_id`` (an experiment seed
      reaches its assumptions, an assumption seed its experiment);
    * ``related_to`` — shared ``experiment.experiment_family_hash``
      (same-design siblings);
    * ``uses`` — shared dataset versions / artifacts (experiments consuming
      the same ``dataset_version`` or ``artifact`` rows through the
      ``experiment_dataset`` / ``experiment_artifact`` link tables);
    * ``supports`` / ``contradicts`` — no backing relation yet (the
      ``finding_evidence`` / conflict structures are still unlanded), so
      they contribute no neighbors, never an error.

    Single-pass semantics: exactly one hop — neighbors of neighbors stay
    out (the planner calls this once per retrieval and the implementation
    never recurses). Seeds themselves are excluded (fusion already ranks
    them; the channel budget is for new nodes). Like every channel,
    neighbors map to candidates, screen through the fusion hard filters,
    and trim to ``limit``; ordering is seed order first (neighbors of the
    top seed win), then the standard recency/id tie-break. Seed ids that
    are not UUIDs, or that match no row, simply contribute no neighbors.
    """

    def expand_neighbors(self, seed_ids: Sequence[str], scope: ScopeFilter, *, edge_types: Sequence[str], limit: int) -> list[Candidate]:
        """Return up to ``limit`` one-hop neighbors of the seeds, best-first.

        Hits carry upstream retrieval evidence (Phase 1a).

        Raises:
            KnowledgeValidationError: When any edge type falls outside the
                :data:`EDGE_TYPES <deerflow.knowledge.retrieval.planner.EDGE_TYPES>`
                vocabulary.
        """
        if isinstance(edge_types, (str, bytes)):
            raise KnowledgeValidationError(f"edge_types must be a sequence of edge-type names, got {edge_types!r}.")
        unknown = [edge for edge in edge_types if edge not in EDGE_TYPES]
        if unknown:
            raise KnowledgeValidationError(f"edge_types entries must be one of {list(EDGE_TYPES)}, got {unknown[0]!r}.")
        wanted = frozenset(edge_types)
        seeds: list[uuid.UUID] = []
        for raw in seed_ids or ():
            try:
                seeds.append(uuid.UUID(raw.strip() if isinstance(raw, str) else ""))
            except (ValueError, AttributeError):
                continue
        if not seeds or not wanted or limit < 1:
            return []
        seed_set = set(seeds)
        seed_rank: dict[uuid.UUID, int] = {}
        for position, key in enumerate(seeds):
            seed_rank.setdefault(key, position)
        discovered: dict[uuid.UUID, int] = {}

        def _note(neighbor_id: uuid.UUID | None, rank: int) -> None:
            """Record a neighbor unless it is a seed (keeping the best seed rank)."""
            if neighbor_id is None or neighbor_id in seed_set:
                return
            if neighbor_id not in discovered or rank < discovered[neighbor_id]:
                discovered[neighbor_id] = rank

        with self._sessions() as session:
            seed_experiments = {row.id: row for row in session.scalars(select(ExperimentRow).where(ExperimentRow.id.in_(seeds)))}
            seed_findings = {row.id: row for row in session.scalars(select(FindingRow).where(FindingRow.id.in_(seeds)))}
            seed_assumptions = {row.id: row for row in session.scalars(select(AssumptionRow).where(AssumptionRow.id.in_(seeds)))}
            if "derived_from" in wanted and seed_experiments:
                for row in seed_experiments.values():
                    _note(row.parent_experiment_id, seed_rank[row.id])
                children = session.execute(select(ExperimentRow.id, ExperimentRow.parent_experiment_id).where(ExperimentRow.parent_experiment_id.in_(list(seed_experiments))))
                for child_id, parent_id in children:
                    _note(child_id, seed_rank[parent_id])
            if "replicates" in wanted and seed_experiments:
                for row in seed_experiments.values():
                    _note(row.replicated_experiment_id, seed_rank[row.id])
                replicators = session.execute(select(ExperimentRow.id, ExperimentRow.replicated_experiment_id).where(ExperimentRow.replicated_experiment_id.in_(list(seed_experiments))))
                for replica_id, target_id in replicators:
                    _note(replica_id, seed_rank[target_id])
            if "supersedes" in wanted and seed_findings:
                for row in seed_findings.values():
                    _note(row.supersedes_id, seed_rank[row.id])
                superseders = session.execute(select(FindingRow.id, FindingRow.supersedes_id).where(FindingRow.supersedes_id.in_(list(seed_findings))))
                for newer_id, older_id in superseders:
                    _note(newer_id, seed_rank[older_id])
            if "applies_to" in wanted:
                if seed_experiments:
                    linked = session.execute(select(AssumptionRow.id, AssumptionRow.experiment_id).where(AssumptionRow.experiment_id.in_(list(seed_experiments))))
                    for assumption_id, experiment_id in linked:
                        _note(assumption_id, seed_rank[experiment_id])
                for row in seed_assumptions.values():
                    _note(row.experiment_id, seed_rank[row.id])
            if "related_to" in wanted and seed_experiments:
                hash_rank: dict[str, int] = {}
                for row in seed_experiments.values():
                    family = row.experiment_family_hash
                    if isinstance(family, str) and family:
                        hash_rank[family] = min(hash_rank.get(family, len(seeds)), seed_rank[row.id])
                if hash_rank:
                    siblings = session.execute(select(ExperimentRow.id, ExperimentRow.experiment_family_hash).where(ExperimentRow.experiment_family_hash.in_(list(hash_rank))))
                    for sibling_id, family in siblings:
                        _note(sibling_id, hash_rank[family])
            if "uses" in wanted and seed_experiments:
                seed_experiment_ids = list(seed_experiments)
                dataset_links = session.execute(select(ExperimentDatasetRow.experiment_id, ExperimentDatasetRow.dataset_version_id).where(ExperimentDatasetRow.experiment_id.in_(seed_experiment_ids)))
                dataset_rank: dict[uuid.UUID, int] = {}
                for experiment_id, dataset_version_id in dataset_links:
                    dataset_rank[dataset_version_id] = min(dataset_rank.get(dataset_version_id, len(seeds)), seed_rank[experiment_id])
                if dataset_rank:
                    co_users = session.execute(select(ExperimentDatasetRow.experiment_id, ExperimentDatasetRow.dataset_version_id).where(ExperimentDatasetRow.dataset_version_id.in_(list(dataset_rank))))
                    for experiment_id, dataset_version_id in co_users:
                        _note(experiment_id, dataset_rank[dataset_version_id])
                artifact_links = session.execute(select(ExperimentArtifactRow.experiment_id, ExperimentArtifactRow.artifact_id).where(ExperimentArtifactRow.experiment_id.in_(seed_experiment_ids)))
                artifact_rank: dict[uuid.UUID, int] = {}
                for experiment_id, artifact_id in artifact_links:
                    artifact_rank[artifact_id] = min(artifact_rank.get(artifact_id, len(seeds)), seed_rank[experiment_id])
                if artifact_rank:
                    co_users = session.execute(select(ExperimentArtifactRow.experiment_id, ExperimentArtifactRow.artifact_id).where(ExperimentArtifactRow.artifact_id.in_(list(artifact_rank))))
                    for experiment_id, artifact_id in co_users:
                        _note(experiment_id, artifact_rank[artifact_id])
            if not discovered:
                return []
            neighbor_ids = list(discovered)
            neighbors: dict[uuid.UUID, Candidate] = {}
            for row in session.scalars(select(FindingRow).where(FindingRow.id.in_(neighbor_ids))):
                neighbors.setdefault(row.id, finding_row_to_candidate(row))
            for row in session.scalars(select(ExperimentRow).where(ExperimentRow.id.in_(neighbor_ids))):
                neighbors.setdefault(row.id, experiment_row_to_candidate(row))
            for row in session.scalars(select(AssumptionRow).where(AssumptionRow.id.in_(neighbor_ids))):
                neighbors.setdefault(row.id, assumption_row_to_candidate(row))
        candidates = [neighbors[neighbor_id] for neighbor_id in neighbor_ids if neighbor_id in neighbors]
        rank_by_id = {str(neighbor_id): rank for neighbor_id, rank in discovered.items()}
        survivors = _screen(candidates, scope)
        ranked = _stable_order(survivors, score_fn=lambda item: -rank_by_id[item.id])[:limit]
        return [candidate_to_evidence_hit(item) for item in ranked]


@dataclass(frozen=True)
class _ScopeSelection:
    """A constraining per-message scope selection (Phase 3c).

    ``dataset_ids`` is the canonical selected datasets; ``kinds`` the
    planner kinds those datasets own (via the provider mapping);
    ``allowed_doc_ids`` the document-id allowlist (None when no document
    filters are present, meaning "every id in the selected datasets").
    """

    dataset_ids: tuple[str, ...]
    kinds: tuple[str, ...]
    allowed_doc_ids: frozenset[str] | None


def _scope_selection(context: KnowledgeRequestContext | None) -> _ScopeSelection | None:
    """Resolve a scoped call's dataset selection, mirroring the provider exactly.

    Mirrors :meth:`QuantflowPgProvider.search_scoped
    <deerflow.knowledge.quantflow_provider.QuantflowPgProvider.search_scoped>`
    (``quantflow_provider.py``): the dataset→kind table is imported from the
    provider module (never duplicated here); ``disabled`` refuses with
    ``ValueError``; an empty ``selected`` list or any unknown dataset id
    fails the whole call with ``PermissionError`` (never a best-effort
    subset); document filters validate per entry and union into an id
    allowlist. Returns None when the call is unconstrained (absent scope or
    ``all`` mode), in which case scoped methods delegate straight to the
    legacy entry points.

    Raises:
        ValueError: When knowledge is disabled for the turn.
        PermissionError: On an empty selection or unknown dataset ids.
    """
    from deerflow.knowledge.quantflow_provider import dataset_kinds

    dataset_kind_map = dataset_kinds()
    raw = context.scope if context is not None else None
    if raw is None:
        return None
    scope = execution_scope(raw)
    if scope.get("mode") == "disabled":
        raise ValueError("Knowledge search is disabled for this turn")
    if scope.get("mode") != "selected":
        return None
    dataset_ids = list(scope.get("dataset_ids") or [])
    if not dataset_ids:
        # Fail closed, mirroring the provider invariant (unreachable through
        # execution_scope, which requires >= 1 dataset id, but the store must
        # not depend on the validation layer for it).
        raise PermissionError("Knowledge scope selects no datasets")
    unknown = [dataset_id for dataset_id in dataset_ids if dataset_id not in dataset_kind_map]
    if unknown:
        raise PermissionError(f"Knowledge scope selects unknown dataset(s): {', '.join(sorted(unknown))}")
    kinds = tuple(dict.fromkeys(kind for dataset_id in dataset_ids for kind in dataset_kind_map[dataset_id]))
    filters = list(scope.get("document_filters") or [])
    for entry in filters:
        if not isinstance(entry, dict) or entry.get("dataset_id") not in dataset_kind_map:
            raise PermissionError("Knowledge scope selects unknown dataset(s)")
    allowed_doc_ids: frozenset[str] | None = None
    if filters:
        allowed_doc_ids = frozenset(document_id for entry in filters for document_id in (entry.get("document_ids") or []) if isinstance(document_id, str))
    return _ScopeSelection(dataset_ids=tuple(dataset_ids), kinds=kinds, allowed_doc_ids=allowed_doc_ids)


def _experiment_dataset(record: ExperimentRecord) -> str:
    """Return the fork dataset owning an experiment record.

    Mirrors the provider ``documents`` split (``quantflow_provider.py``):
    ``outcome == "failure"`` rows belong to ``failures``; every other row
    — including NULL (undecided) outcomes — belongs to ``experiments``.
    """
    return "failures" if record.outcome == "failure" else "experiments"


class SQLExperimentLookupStore(SQLExperimentSearchStore):
    """``ExperimentLookupStore``: search binding plus primary-key lookup.

    Extends :class:`SQLExperimentSearchStore` with ``find_by_id`` (UUID
    primary-key point lookup with the same batched link fetch), completing
    the experiment read boundary the lookup tools need. Every method returns
    :class:`EvidenceExperimentRecord` hits (upstream retrieval evidence,
    Phase 1a); the record payload itself is unchanged.
    """

    def find_by_id(self, experiment_id: str) -> ExperimentRecord | None:
        """Return the experiment with this id, or None (primary-key lookup).

        The hit carries upstream retrieval evidence (Phase 1a).

        Raises:
            KnowledgeValidationError: When ``experiment_id`` is not a UUID.
        """
        try:
            key = uuid.UUID(experiment_id.strip() if isinstance(experiment_id, str) else "")
        except (ValueError, AttributeError):
            raise KnowledgeValidationError(f"experiment_id must be a valid UUID, got {experiment_id!r}.") from None
        with self._sessions() as session:
            row = session.get(ExperimentRow, key)
            if row is None:
                return None
            links = self._fetch_links(session, [row.id])
            return record_to_evidence_hit(row_to_record(row, datasets=links[0][0], result_artifacts=links[0][1]))

    def find_by_execution_hash(self, execution_hash: str) -> ExperimentRecord | None:
        """Return the experiment with this execution hash, or None (unique lookup).

        Same lookup as :class:`SQLExperimentSearchStore`; the hit carries
        upstream retrieval evidence (Phase 1a).
        """
        record = super().find_by_execution_hash(execution_hash)
        return record_to_evidence_hit(record) if record is not None else None

    def find_by_id_scoped(self, experiment_id: str, *, context: KnowledgeRequestContext) -> ExperimentRecord | None:
        """Return the experiment with this id under ``context.scope``, or None.

        Phase 3c :class:`ScopedExperimentLookupStore
        <deerflow.knowledge.agent_tools.lookup.ScopedExperimentLookupStore>`
        binding: reuses :meth:`find_by_id` (no duplicated SQL), then hides
        out-of-scope records as None — indistinguishable from unknown ids —
        instead of erroring. Records whose dataset (``failures`` for
        ``outcome == "failure"``, else ``experiments``, mirroring the
        provider split) is not selected, or whose id falls outside the
        document-id allowlist, read as misses.

        Raises:
            ValueError: When knowledge is disabled for the turn.
            PermissionError: On an empty selection or unknown dataset ids.
        """
        selection = _scope_selection(context)
        record = self.find_by_id(experiment_id)
        if selection is None or record is None:
            return record
        if _experiment_dataset(record) not in selection.dataset_ids:
            return None
        if selection.allowed_doc_ids is not None and record.id not in selection.allowed_doc_ids:
            return None
        return record

    def find_by_execution_hash_scoped(self, execution_hash: str, *, context: KnowledgeRequestContext) -> ExperimentRecord | None:
        """Return the experiment with this execution hash under ``context.scope``, or None.

        Same scope rule as :meth:`find_by_id_scoped` over
        :meth:`find_by_execution_hash`: out-of-scope hashes read as None
        (existence-hiding), never as errors.
        """
        selection = _scope_selection(context)
        record = self.find_by_execution_hash(execution_hash)
        if selection is None or record is None:
            return record
        if _experiment_dataset(record) not in selection.dataset_ids:
            return None
        if selection.allowed_doc_ids is not None and record.id not in selection.allowed_doc_ids:
            return None
        return record

    def find_by_family_hash(self, family_hash: str, *, limit: int, offset: int) -> list[ExperimentRecord]:
        """Return one page of experiments in this family, oldest first.

        Same page as :class:`SQLExperimentSearchStore`; the hits carry
        upstream retrieval evidence (Phase 1a).
        """
        return [record_to_evidence_hit(item) for item in super().find_by_family_hash(family_hash, limit=limit, offset=offset)]

    def search_experiments(self, filters: ExperimentFilter, *, limit: int, offset: int) -> list[ExperimentRecord]:
        """Return one page of experiments matching all non-None filter fields, oldest first.

        Same page as :class:`SQLExperimentSearchStore`; the hits carry
        upstream retrieval evidence (Phase 1a).
        """
        return [record_to_evidence_hit(item) for item in super().search_experiments(filters, limit=limit, offset=offset)]


class SQLFindingTextSource:
    """``FindingTextSource``: canonical embeddable text per finding id.

    Renders ``canonical_key`` (title) + ``statement`` (body) with
    :func:`render_embeddable_text` so backfills embed exactly what the
    retrieval surface ranks.
    """

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        """Bind the source to a zero-argument session factory."""
        self._sessions = session_factory

    def get_finding_text(self, finding_id: str) -> str | None:
        """Return the embeddable text for ``finding_id``, or None when missing.

        Raises:
            KnowledgeValidationError: When ``finding_id`` is not a UUID.
        """
        try:
            key = uuid.UUID(finding_id.strip() if isinstance(finding_id, str) else "")
        except (ValueError, AttributeError):
            raise KnowledgeValidationError(f"finding_id must be a valid UUID, got {finding_id!r}.") from None
        with self._sessions() as session:
            row = session.get(FindingRow, key)
            if row is None:
                return None
            return render_embeddable_text(title=row.canonical_key or "", body=row.statement or "")


class SQLEmbeddingVectorStore:
    """``EmbeddingVectorStore``: persist finding vectors beside their rows.

    Writes go to ``finding.embedding`` (native ``VECTOR(768)`` on
    PostgreSQL, JSON fallback on SQLite). Every upserted vector is
    fail-closed-validated (exactly :data:`EMBEDDING_DIM` finite floats);
    unknown finding ids raise :class:`KnowledgeError` so backfills record
    them as failures instead of silently dropping them.

    Model identity: the Phase 2 schema carries no per-row model stamp, so
    this store operates in single-model mode — :meth:`get_embedding_model`
    reports the configured ``model_id`` whenever a vector is present.
    Switching embedding models must therefore backfill with
    ``skip_up_to_date=False`` (a later revision may add the
    ``embedding_model``/``embedding_updated_at`` columns the embeddings
    protocol sketches).
    """

    def __init__(self, session_factory: Callable[[], Session], *, model_id: str = FAKE_MODEL_ID) -> None:
        """Bind the store to a session factory plus its single model id.

        Raises:
            KnowledgeValidationError: When ``model_id`` is empty.
        """
        if not isinstance(model_id, str) or not model_id.strip():
            raise KnowledgeValidationError("model_id must be a non-empty string.")
        self._sessions = session_factory
        self._model_id = model_id.strip()

    @property
    def model_id(self) -> str:
        """The model id this store writes and reports."""
        return self._model_id

    def get_embedding_model(self, finding_id: str) -> str | None:
        """Return the model id of the stored vector, or None when unembedded.

        Unknown finding ids also return None (the backfill reports them as
        missing once the text source likewise misses).

        Raises:
            KnowledgeValidationError: When ``finding_id`` is not a UUID.
        """
        try:
            key = uuid.UUID(finding_id.strip() if isinstance(finding_id, str) else "")
        except (ValueError, AttributeError):
            raise KnowledgeValidationError(f"finding_id must be a valid UUID, got {finding_id!r}.") from None
        with self._sessions() as session:
            row = session.get(FindingRow, key)
            if row is None or row.embedding is None:
                return None
            return self._model_id

    def upsert_embedding(self, finding_id: str, vector: Sequence[float], *, model_id: str) -> None:
        """Persist ``vector`` for ``finding_id`` (validated, single-model).

        Args:
            finding_id: Finding UUID string.
            vector: Exactly :data:`EMBEDDING_DIM` finite floats.
            model_id: Must equal this store's :attr:`model_id` (vectors
                from any other model are refused, never silently stored).

        Raises:
            KnowledgeValidationError: On a bad id, a malformed vector, or
                a model-id mismatch.
            KnowledgeError: When the finding does not exist.
        """
        try:
            key = uuid.UUID(finding_id.strip() if isinstance(finding_id, str) else "")
        except (ValueError, AttributeError):
            raise KnowledgeValidationError(f"finding_id must be a valid UUID, got {finding_id!r}.") from None
        if not isinstance(model_id, str) or model_id.strip() != self._model_id:
            raise KnowledgeValidationError(f"model_id {model_id!r} does not match this store's model {self._model_id!r}; refusing to mix models in one column.")
        values = list(vector) if isinstance(vector, Sequence) and not isinstance(vector, (str, bytes)) else None
        if values is None or len(values) != EMBEDDING_DIM:
            got = "non-sequence" if values is None else f"dimension {len(values)}"
            raise KnowledgeValidationError(f"vector for finding {finding_id} has {got}, expected {EMBEDDING_DIM} (the finding VECTOR column width).")
        checked: list[float] = []
        for position, entry in enumerate(values):
            if not isinstance(entry, (int, float)) or isinstance(entry, bool) or not math.isfinite(entry):
                raise KnowledgeValidationError(f"vector for finding {finding_id} entry {position} is not a finite float (got {entry!r}).")
            checked.append(float(entry))
        with self._sessions() as session:
            row = session.get(FindingRow, key)
            if row is None:
                raise KnowledgeError(f"Finding not found: {finding_id}")
            row.embedding = checked
            session.commit()


class SQLExperimentTextSource:
    """Experiment text source: canonical embeddable text per experiment id.

    Renders the hypothesis (title) plus the full lexical surface
    (:func:`_experiment_text` — hypothesis, parameters summary, outcome /
    failure class) with :func:`render_embeddable_text` so backfills embed
    exactly what the retrieval surface ranks.
    """

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        """Bind the source to a zero-argument session factory."""
        self._sessions = session_factory

    def get_experiment_text(self, experiment_id: str) -> str | None:
        """Return the embeddable text for ``experiment_id``, or None when missing.

        Raises:
            KnowledgeValidationError: When ``experiment_id`` is not a UUID.
        """
        try:
            key = uuid.UUID(experiment_id.strip() if isinstance(experiment_id, str) else "")
        except (ValueError, AttributeError):
            raise KnowledgeValidationError(f"experiment_id must be a valid UUID, got {experiment_id!r}.") from None
        with self._sessions() as session:
            row = session.get(ExperimentRow, key)
            if row is None:
                return None
            return render_embeddable_text(title=row.hypothesis or "", body=_experiment_text(row) or "")


class SQLExperimentEmbeddingVectorStore:
    """Experiment vector store: persist experiment vectors beside their rows.

    Writes go to ``experiment.embedding`` (native ``VECTOR(768)`` on
    PostgreSQL, JSON fallback on SQLite). Every upserted vector is
    fail-closed-validated (exactly :data:`EMBEDDING_DIM` finite floats);
    unknown experiment ids raise :class:`KnowledgeError` so backfills record
    them as failures instead of silently dropping them.

    Model identity mirrors :class:`SQLEmbeddingVectorStore`: the Phase 3
    schema carries no per-row model stamp, so this store operates in
    single-model mode — :meth:`get_embedding_model` reports the configured
    ``model_id`` whenever a vector is present. Switching embedding models
    must therefore backfill with ``skip_up_to_date=False``.
    """

    def __init__(self, session_factory: Callable[[], Session], *, model_id: str = FAKE_MODEL_ID) -> None:
        """Bind the store to a session factory plus its single model id.

        Raises:
            KnowledgeValidationError: When ``model_id`` is empty.
        """
        if not isinstance(model_id, str) or not model_id.strip():
            raise KnowledgeValidationError("model_id must be a non-empty string.")
        self._sessions = session_factory
        self._model_id = model_id.strip()

    @property
    def model_id(self) -> str:
        """The model id this store writes and reports."""
        return self._model_id

    def get_embedding_model(self, experiment_id: str) -> str | None:
        """Return the model id of the stored vector, or None when unembedded.

        Unknown experiment ids also return None (the backfill reports them as
        missing once the text source likewise misses).

        Raises:
            KnowledgeValidationError: When ``experiment_id`` is not a UUID.
        """
        try:
            key = uuid.UUID(experiment_id.strip() if isinstance(experiment_id, str) else "")
        except (ValueError, AttributeError):
            raise KnowledgeValidationError(f"experiment_id must be a valid UUID, got {experiment_id!r}.") from None
        with self._sessions() as session:
            row = session.get(ExperimentRow, key)
            if row is None or row.embedding is None:
                return None
            return self._model_id

    def upsert_embedding(self, experiment_id: str, vector: Sequence[float], *, model_id: str) -> None:
        """Persist ``vector`` for ``experiment_id`` (validated, single-model).

        Args:
            experiment_id: Experiment UUID string.
            vector: Exactly :data:`EMBEDDING_DIM` finite floats.
            model_id: Must equal this store's :attr:`model_id` (vectors
                from any other model are refused, never silently stored).

        Raises:
            KnowledgeValidationError: On a bad id, a malformed vector, or
                a model-id mismatch.
            KnowledgeError: When the experiment does not exist.
        """
        try:
            key = uuid.UUID(experiment_id.strip() if isinstance(experiment_id, str) else "")
        except (ValueError, AttributeError):
            raise KnowledgeValidationError(f"experiment_id must be a valid UUID, got {experiment_id!r}.") from None
        if not isinstance(model_id, str) or model_id.strip() != self._model_id:
            raise KnowledgeValidationError(f"model_id {model_id!r} does not match this store's model {self._model_id!r}; refusing to mix models in one column.")
        values = list(vector) if isinstance(vector, Sequence) and not isinstance(vector, (str, bytes)) else None
        if values is None or len(values) != EMBEDDING_DIM:
            got = "non-sequence" if values is None else f"dimension {len(values)}"
            raise KnowledgeValidationError(f"vector for experiment {experiment_id} has {got}, expected {EMBEDDING_DIM} (the experiment VECTOR column width).")
        checked: list[float] = []
        for position, entry in enumerate(values):
            if not isinstance(entry, (int, float)) or isinstance(entry, bool) or not math.isfinite(entry):
                raise KnowledgeValidationError(f"vector for experiment {experiment_id} entry {position} is not a finite float (got {entry!r}).")
            checked.append(float(entry))
        with self._sessions() as session:
            row = session.get(ExperimentRow, key)
            if row is None:
                raise KnowledgeError(f"Experiment not found: {experiment_id}")
            row.embedding = checked
            session.commit()


def refresh_finding_search_documents(session_factory: Callable[[], Session], finding_ids: Sequence[str] | None = None) -> int:
    """Populate ``finding.search_document`` from the canonical key + statement.

    This is the synchronous stand-in for the Phase 6 indexing worker:
    PostgreSQL sets the native TSVECTOR server-side with
    ``to_tsvector('english', canonical_key || ' ' || statement)`` while
    SQLite stores the same surface as plain text for the LIKE fallback.
    Indexing never touches canonical columns — only the retrieval
    projection — so a failed refresh cannot corrupt findings.

    Args:
        session_factory: Zero-argument session factory.
        finding_ids: Finding UUID strings to refresh (None refreshes every
            finding row).

    Returns:
        The number of rows refreshed.

    Raises:
        KnowledgeValidationError: When any id is not a UUID.
    """
    keys: list[uuid.UUID] | None = None
    if finding_ids is not None:
        if isinstance(finding_ids, (str, bytes)):
            raise KnowledgeValidationError(f"finding_ids must be a sequence of UUID strings, got {type(finding_ids).__name__}.")
        keys = []
        for raw in finding_ids:
            try:
                keys.append(uuid.UUID(raw.strip() if isinstance(raw, str) else ""))
            except (ValueError, AttributeError):
                raise KnowledgeValidationError(f"finding_id must be a valid UUID, got {raw!r}.") from None
    with session_factory() as session:
        if _dialect_name(session) == "postgresql":
            statement = select(FindingRow.id)
            if keys is not None:
                statement = statement.where(FindingRow.id.in_(keys))
            target_ids = list(session.scalars(statement))
            if not target_ids:
                return 0
            session.execute(_finding_reindex_update(target_ids))
            session.commit()
            return len(target_ids)
        statement = select(FindingRow)
        if keys is not None:
            statement = statement.where(FindingRow.id.in_(keys))
        count = 0
        for row in session.scalars(statement):
            row.search_document = f"{row.canonical_key or ''} {row.statement or ''}".strip()
            count += 1
        session.commit()
        return count


def _summarize(text: str) -> tuple[str, bool]:
    """Trim ``text`` to :data:`SUMMARY_CHARS`, reporting truncation."""
    if len(text) <= SUMMARY_CHARS:
        return text, False
    return text[: SUMMARY_CHARS - 1].rstrip() + "…", True


class KnowledgeRetrievalService:
    """``KnowledgeRetrievalBackend`` over :func:`execute_retrieval` + SQL channels.

    The tool layer's view of hybrid retrieval: :meth:`search` converts the
    tool call into a planner :class:`RetrievalPlan` (lookup kinds map to
    planner kinds per :data:`_LOOKUP_KIND_MAP`, structured filters become
    the :class:`ScopeFilter`, ``project_id`` becomes the ACL enforcement
    set, ``concept`` folds into the query text for the lexical/failure
    channels, and ``status`` applies as a post-fusion exact match), runs
    the four SQL channels plus the one-hop relational expansion
    (:class:`SQLRelationalExpansionStore` over the FK graph), and renders
    the fused ranking as :class:`RetrievalDocument` pages.
    :meth:`get_documents` is a primary-key fetch across the three
    retrievable tables.

    The vector channel runs only when an ``embedding_provider`` is bound;
    otherwise retrieval skips it with a recorded warning (identical to the
    planner's no-embedding path). Provider outages degrade the same way —
    a log line plus the warning — because an embedding outage must narrow
    recall, never fail the whole search.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        embedding_provider: EmbeddingProvider | None = None,
        fusion_weights: FusionWeights | None = None,
    ) -> None:
        """Bind the service to a session factory plus an optional provider.

        Args:
            session_factory: Zero-argument session factory shared by all
                four channel stores.
            embedding_provider: Query-embedding computation boundary (None
                disables the vector channel).
            fusion_weights: Explicit rank-fusion weights (None reads the
                ``knowledge:`` config mapping per search via
                :func:`deerflow.knowledge.config.get_knowledge_config`, so
                the default is ``FusionWeights()``, identical to before).
        """
        self._sessions = session_factory
        self._structured = SQLStructuredLookupStore(session_factory)
        self._lexical = SQLLexicalSearchStore(session_factory)
        self._vector = SQLVectorSearchStore(session_factory)
        self._failures = SQLFailureSearchStore(session_factory)
        self._relational = SQLRelationalExpansionStore(session_factory)
        self._embedding_provider = embedding_provider
        self._fusion_weights = fusion_weights

    @property
    def embedding_provider(self) -> EmbeddingProvider | None:
        """The bound query-embedding provider (None when vector is off)."""
        return self._embedding_provider

    @property
    def fusion_weights(self) -> FusionWeights | None:
        """The explicit rank-fusion weights (None means read from config per search)."""
        return self._fusion_weights

    def search(
        self,
        query: str,
        *,
        kinds: tuple[str, ...],
        filters: Mapping[str, str],
        limit: int,
        offset: int,
    ) -> RetrievalPage:
        """Return one page of fused retrieval hits for ``query``.

        Raises:
            KnowledgeValidationError: On invalid planner inputs derived
                from the call (the tool layer normally validates first).
            KnowledgeError: When the fused ranking cannot be built.
        """
        planner_kinds: list[str] = []
        for kind in kinds:
            for mapped in _LOOKUP_KIND_MAP.get(kind, ()):
                if mapped not in planner_kinds:
                    planner_kinds.append(mapped)
        if not planner_kinds:
            return RetrievalPage(documents=[], limit=limit, offset=offset, total=0)
        concept = filters.get("concept")
        query_text = f"{query} {concept}".strip() if concept else query
        intent = ResearchIntent(
            topic=query_text,
            asset_class=filters.get("asset_class"),
            markets=(filters["market"],) if filters.get("market") else (),
            universe=filters.get("universe"),
            horizon=filters.get("horizon"),
            concepts=(concept,) if concept else (),
            requested_period=(filters.get("period_start"), filters.get("period_end")),
        )
        needed = limit + offset
        plan = plan_retrieval(
            intent,
            per_channel_limit=min(MAX_PER_CHANNEL_LIMIT, max(DEFAULT_PER_CHANNEL_LIMIT, needed)),
            top_k=min(MAX_TOP_K, needed),
            kinds=planner_kinds,
            relational=True,
        )
        query_embedding = self._embed_query(query_text)
        allowed = [filters["project_id"]] if filters.get("project_id") else None
        weights = self._fusion_weights if self._fusion_weights is not None else get_knowledge_config().get_fusion_weights()
        result = execute_retrieval(
            plan,
            query_text=query_text,
            structured=self._structured,
            lexical=self._lexical,
            vector=self._vector,
            failures=self._failures,
            relational=self._relational,
            query_embedding=query_embedding,
            allowed_project_ids=allowed,
            weights=weights,
        )
        # Slice the full fused ranking (not plan.top_k): total stays honest
        # across pages and deep offsets keep working.
        # Kinds apply as a post-fusion exact filter: channels that ignore
        # kinds (notably the failure channel) must never leak rows the
        # caller did not ask for into the page.
        wanted_kinds = frozenset(planner_kinds)
        ranked = [item for item in result.fusion.candidates if item.candidate.kind in wanted_kinds]
        wanted_status = filters.get("status")
        if wanted_status is not None:
            ranked = [item for item in ranked if item.candidate.status.lower() == wanted_status.lower()]
        total = len(ranked)
        page = ranked[offset : offset + limit]
        return RetrievalPage(documents=[self._to_document(item.candidate, score=item.score, channels=item.channels, modifiers=item.modifiers, rank=item.rank) for item in page], limit=limit, offset=offset, total=total)

    def search_scoped(
        self,
        query: str,
        *,
        kinds: tuple[str, ...],
        filters: Mapping[str, str],
        limit: int,
        offset: int,
        context: KnowledgeRequestContext,
    ) -> RetrievalPage:
        """Return one page of fused hits constrained by ``context.scope``.

        Phase 3c :class:`ScopedKnowledgeRetrievalBackend
        <deerflow.knowledge.agent_tools.lookup.ScopedKnowledgeRetrievalBackend>`
        binding: reuses :meth:`search` (no duplicated SQL). A selected scope
        intersects the caller ``kinds`` with the selected datasets' planner
        kinds (via the provider mapping, so ``conflict``/``dossier``/
        ``artifact`` — owned by no dataset — drop out), then post-filters
        the fused ranking by document id when document filters are present
        (mirroring :meth:`QuantflowPgProvider.search_scoped
        <deerflow.knowledge.quantflow_provider.QuantflowPgProvider.search_scoped>`).
        ``all`` scopes behave as the legacy search.

        Raises:
            ValueError: When knowledge is disabled for the turn.
            PermissionError: On an empty selection or unknown dataset ids.
        """
        selection = _scope_selection(context)
        if selection is None:
            return self.search(query, kinds=kinds, filters=filters, limit=limit, offset=offset)
        wanted = frozenset(selection.kinds)
        restricted = tuple(kind for kind in kinds if set(_LOOKUP_KIND_MAP.get(kind, ())) & wanted)
        if selection.allowed_doc_ids is None:
            return self.search(query, kinds=restricted, filters=filters, limit=limit, offset=offset)
        # The id allowlist applies post-fusion (provider parity), so fetch the
        # full ranking through the unscoped entry point and slice after filtering.
        fetch = limit + offset
        page = self.search(query, kinds=restricted, filters=filters, limit=fetch, offset=0)
        if page.total is not None and page.total > fetch:
            page = self.search(query, kinds=restricted, filters=filters, limit=page.total, offset=0)
        filtered = [doc for doc in page.documents if doc.id in selection.allowed_doc_ids]
        return RetrievalPage(documents=filtered[offset : offset + limit], limit=limit, offset=offset, total=len(filtered))

    def get_documents(self, ids: Sequence[str], *, include_evidence: bool) -> list[RetrievalDocument | None]:
        """Return one entry per requested id, ``None`` for unknown ids (order preserved).

        Ids resolve as UUID primary keys across ``finding``, ``experiment``
        and ``assumption`` in that order; non-UUID ids resolve to None.
        ``include_evidence=True`` additionally attaches the full experiment
        record (datasets + result artifacts) to experiment documents; the
        citation-lock rule itself is enforced by the caller/run state.
        """
        documents: list[RetrievalDocument | None] = []
        with self._sessions() as session:
            for raw in ids:
                try:
                    key = uuid.UUID(raw.strip() if isinstance(raw, str) else "")
                except (ValueError, AttributeError):
                    documents.append(None)
                    continue
                documents.append(self._fetch_document(session, key, include_evidence=include_evidence))
        return documents

    def get_documents_scoped(
        self,
        ids: Sequence[str],
        *,
        include_evidence: bool,
        context: KnowledgeRequestContext,
    ) -> list[RetrievalDocument | None]:
        """Return one entry per requested id under ``context.scope`` (order preserved).

        Phase 3c scoped binding: reuses :meth:`get_documents` (no duplicated
        SQL), then hides out-of-scope documents as None — unknown AND
        out-of-scope ids both read as misses so a miss never confirms
        whether an id exists outside the scope. ``all`` scopes behave as the
        legacy fetch.

        Raises:
            ValueError: When knowledge is disabled for the turn.
            PermissionError: On an empty selection or unknown dataset ids.
        """
        selection = _scope_selection(context)
        rows = self.get_documents(ids, include_evidence=include_evidence)
        if selection is None:
            return rows
        allowed_lookup = {_PLANNER_KIND_MAP[kind] for kind in selection.kinds if kind in _PLANNER_KIND_MAP}
        scoped: list[RetrievalDocument | None] = []
        for doc in rows:
            if doc is None or doc.kind not in allowed_lookup:
                scoped.append(None)
            elif selection.allowed_doc_ids is not None and doc.id not in selection.allowed_doc_ids:
                scoped.append(None)
            else:
                scoped.append(doc)
        return scoped

    def _embed_query(self, query_text: str) -> list[float] | None:
        """Embed the query, degrading to None (vector skipped) on any failure."""
        provider = self._embedding_provider
        if provider is None:
            return None
        try:
            vectors = embed_texts(provider, [query_text])
        except Exception:
            logger.warning("KnowledgeRetrievalService: query embedding failed; running retrieval without the vector channel", exc_info=True)
            return None
        if not vectors:
            return None
        if len(vectors[0]) != EMBEDDING_DIM:
            logger.warning(
                "KnowledgeRetrievalService: provider %r emitted %d-d vectors, expected %d; running retrieval without the vector channel",
                provider.model_id,
                len(vectors[0]),
                EMBEDDING_DIM,
            )
            return None
        return vectors[0]

    @staticmethod
    def _to_document(candidate: Candidate, *, score: float, channels: tuple[str, ...], modifiers: dict[str, float], rank: int) -> RetrievalDocument:
        """Render one fused candidate as a tool-layer document."""
        summary, truncated = _summarize(candidate.text)
        payload: dict[str, Any] = {
            "channels": list(channels),
            "modifiers": dict(modifiers),
            "rank": rank,
            "scope": dict(candidate.scope),
        }
        if truncated:
            payload["truncated"] = True
        if candidate.finding_type is not None:
            payload["finding_type"] = candidate.finding_type
        if candidate.metadata:
            payload["candidate_metadata"] = dict(candidate.metadata)
        return RetrievalDocument(
            id=candidate.id,
            kind=_PLANNER_KIND_MAP.get(candidate.kind, candidate.kind),
            title=candidate.title,
            summary=summary,
            score=float(score),
            status=candidate.status or None,
            evidence_refs=list(candidate.evidence),
            payload=payload,
        )

    def _fetch_document(self, session: Session, key: uuid.UUID, *, include_evidence: bool) -> RetrievalDocument | None:
        """Fetch one document by primary key across the retrievable tables."""
        finding = session.get(FindingRow, key)
        if finding is not None:
            candidate = finding_row_to_candidate(finding)
            document = self._to_document(candidate, score=0.0, channels=(), modifiers={}, rank=0)
            payload = dict(document.payload)
            payload["canonical_key"] = finding.canonical_key
            payload["confidence"] = finding.confidence if isinstance(finding.confidence, dict) else {}
            payload["evidence"] = []
            return RetrievalDocument(
                id=document.id,
                kind=document.kind,
                title=document.title,
                summary=document.summary,
                status=document.status,
                evidence_refs=document.evidence_refs,
                payload=payload,
            )
        experiment = session.get(ExperimentRow, key)
        if experiment is not None:
            candidate = experiment_row_to_candidate(experiment)
            document = self._to_document(candidate, score=0.0, channels=(), modifiers={}, rank=0)
            payload = dict(document.payload)
            payload["family_hash"] = candidate.family_hash
            if include_evidence:
                links = SQLExperimentSearchStore._fetch_links(session, [experiment.id])
                payload["record"] = row_to_record(experiment, datasets=links[0][0], result_artifacts=links[0][1]).to_dict()
            return RetrievalDocument(
                id=document.id,
                kind=document.kind,
                title=document.title,
                summary=document.summary,
                status=document.status,
                evidence_refs=document.evidence_refs,
                payload=payload,
            )
        assumption = session.get(AssumptionRow, key)
        if assumption is not None:
            return self._to_document(assumption_row_to_candidate(assumption), score=0.0, channels=(), modifiers={}, rank=0)
        return None


def open_pg_backends(
    dsn: str,
    *,
    embedding_model: str | None = None,
    artifact_store: Any | None = None,
    artifact_store_uri: str | None = None,
    fusion_weights: FusionWeights | None = None,
) -> KnowledgeBackends:
    """Build the tool-layer backend bundle bound to the database at ``dsn``.

    Reuses the process-wide sync-engine cache (one engine/pool per URL), so
    repeated calls with the same DSN share connections. Pass the result to
    :func:`deerflow.knowledge.agent_tools.lookup.bind_knowledge_backends` to arm
    the ``@tool`` wrappers and the bootstrap middleware.

    Args:
        dsn: SQLAlchemy URL, e.g. ``postgresql+psycopg://...`` (canonical)
            or ``sqlite:////path/to/kb.db`` (dev/test).
        embedding_model: Embedding model id resolved via
            :func:`load_provider` for the vector channel. None (default)
            disables the vector channel entirely — no silent test-fake in
            the production path; pass ``"test-fake/v1"`` explicitly for
            offline mechanics checks.
        artifact_store: Optional :class:`ArtifactStore` to expose through
            an :class:`ArtifactStoreReader` (None leaves artifacts unbound
            unless ``artifact_store_uri`` is set; wins over the URI).
        artifact_store_uri: Optional object-store URI (``file://`` /
            ``s3://`` / ``minio://`` / bare path) resolved via
            :func:`create_backend <deerflow.knowledge.artifacts.backends.create_backend>`
            into the reader's store. None leaves artifacts unbound (current
            behavior when no explicit store is given either).
        fusion_weights: Explicit rank-fusion weights for the retrieval
            service (None reads the ``knowledge:`` config mapping per
            search, defaulting to ``FusionWeights()``).

    Returns:
        A :class:`KnowledgeBackends` bundle (retrieval + experiments always
        bound, artifacts bound when a store or URI is given).
    """
    from deerflow.persistence.agents.sql import get_sync_sessionmaker

    session_factory = get_sync_sessionmaker(dsn)
    provider = load_provider(embedding_model) if embedding_model is not None else None
    if artifact_store is None and artifact_store_uri is not None:
        from deerflow.knowledge.artifacts.backends import create_backend
        from deerflow.knowledge.artifacts.store import ArtifactStore

        artifact_store = ArtifactStore(create_backend(artifact_store_uri))
    return KnowledgeBackends(
        retrieval=KnowledgeRetrievalService(session_factory, embedding_provider=provider, fusion_weights=fusion_weights),
        experiments=SQLExperimentLookupStore(session_factory),
        artifacts=ArtifactStoreReader(artifact_store) if artifact_store is not None else None,
    )


def bind_knowledge_backends_from_config(*, artifact_store: Any | None = None) -> KnowledgeBackends | None:
    """Bind the process-wide knowledge backends from the ``knowledge:`` config section.

    The startup integration step (called once from the Gateway lifespan):
    reads the database DSN, the embedding model, and the artifact-store URI
    from :func:`deerflow.knowledge.config.get_knowledge_config`, opens the
    backend bundle with :func:`open_pg_backends`, and registers it via
    :func:`deerflow.knowledge.agent_tools.lookup.bind_knowledge_backends` so the
    ``ledger_*``/``experiment_get`` tools and the run-start bootstrap
    middleware resolve real PG bindings.

    Vector-channel behavior follows the existing ``needed_memory`` design,
    not a feature flag: the planner arms the vector channel for the
    ``validated_findings`` / ``prior_experiments`` needs, and the channel
    runs whenever a query embedding is available. Availability is decided
    here — a configured ``embedding_model`` is resolved through
    :func:`load_provider` (the sentence-transformers factory is registered
    for it first, unless the id is the ``test-fake`` fake or already has a
    factory), while ``None`` leaves the provider unbound so the channel
    skips with its standard warning. A configured model that fails to
    load degrades the same way — a warning plus retrieval without the
    vector channel — because an embedding outage must narrow recall,
    never fail the whole search.

    Args:
        artifact_store: Optional explicit artifact store to expose through
            an :class:`ArtifactStoreReader` (wins over the configured URI;
            None builds the reader from the configured ``artifact_store``
            URI when set, else leaves artifacts unbound).

    Returns:
        The effective process-wide bindings, or None when the plane is
        disabled or no DSN is configured (previous bindings untouched).

    Raises:
        sqlalchemy.exc.ArgumentError: When the configured DSN is not a
            usable SQLAlchemy URL (the lifespan logs and keeps serving).
        ValueError: When the configured ``artifact_store`` URI has an
            unsupported scheme (fail closed: no partial binding).
    """
    config = get_knowledge_config()
    if not config.is_enabled():
        logger.debug("bind_knowledge_backends_from_config: knowledge plane disabled; leaving backends unbound")
        return None
    dsn = config.get_database_dsn()
    if not dsn:
        logger.info("bind_knowledge_backends_from_config: no knowledge database_dsn configured; leaving backends unbound")
        return None
    model = config.get_embedding_model()
    if model is not None and model not in {"test-fake", FAKE_MODEL_ID}:
        from deerflow.knowledge import embeddings as embeddings_mod
        from deerflow.knowledge.providers_st import SentenceTransformerEmbeddingProvider

        if model not in embeddings_mod._provider_factories:
            register_provider_factory(model, lambda: SentenceTransformerEmbeddingProvider(model))
    artifact_store_uri = config.get_artifact_store() if artifact_store is None else None
    try:
        backends = open_pg_backends(dsn, embedding_model=model, artifact_store=artifact_store, artifact_store_uri=artifact_store_uri)
    except EmbeddingError:
        logger.warning(
            "bind_knowledge_backends_from_config: embedding model %r failed to load; binding knowledge backends without the vector channel",
            model,
            exc_info=True,
        )
        backends = open_pg_backends(dsn, artifact_store=artifact_store, artifact_store_uri=artifact_store_uri)
    bound = bind_knowledge_backends(retrieval=backends.retrieval, experiments=backends.experiments, artifacts=backends.artifacts)
    logger.info(
        "bind_knowledge_backends_from_config: knowledge backends bound (vector=%s, artifacts=%s)",
        "on" if model is not None and bound.retrieval is not None and bound.retrieval.embedding_provider is not None else "off",
        "on" if bound.artifacts is not None else "off",
    )
    return bound

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
* :class:`KnowledgeRetrievalService` →
  :class:`deerflow.knowledge.tools.lookup.KnowledgeRetrievalBackend`
  (adapter over :func:`deerflow.knowledge.retrieval.planner.execute_retrieval`)
* :class:`SQLExperimentLookupStore` →
  :class:`deerflow.knowledge.tools.lookup.ExperimentLookupStore`
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
``limit``. The ``relational`` channel stays unbound at Phase 2 (no
``knowledge_edge`` table yet); the service plans single-pass retrieval.

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
sync-engine cache.
"""

from __future__ import annotations

import logging
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import bindparam, func, select
from sqlalchemy.orm import Session

from deerflow.knowledge.embeddings import (
    EMBEDDING_DIM,
    FAKE_MODEL_ID,
    EmbeddingProvider,
    embed_texts,
    load_provider,
    render_embeddable_text,
)
from deerflow.knowledge.pg_store import SQLExperimentSearchStore, row_to_record
from deerflow.knowledge.retrieval.fusion import SCOPE_PLACEHOLDERS, apply_hard_filters
from deerflow.knowledge.retrieval.planner import (
    DEFAULT_PER_CHANNEL_LIMIT,
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
from deerflow.knowledge.schema.experiments import AssumptionRow, ExperimentRow
from deerflow.knowledge.schema.findings import FindingRow, NativeVector
from deerflow.knowledge.tools.lookup import (
    ArtifactStoreReader,
    KnowledgeBackends,
    RetrievalDocument,
    RetrievalPage,
)
from deerflow.knowledge.write_api import ExperimentRecord, KnowledgeError, KnowledgeValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "SUMMARY_CHARS",
    "SQLStructuredLookupStore",
    "SQLLexicalSearchStore",
    "SQLVectorSearchStore",
    "SQLFailureSearchStore",
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


def _experiment_text(row: ExperimentRow) -> str:
    """Build the lexical surface for an experiment row.

    The hypothesis leads; the parameters summary follows (signal names,
    lookbacks, tool params, and any seeder-provided retrieval text); the
    outcome/failure class trails so failure phrasing stays matchable.
    Parts are whitespace-joined and stripped of empties.
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
        """Return up to ``limit`` scope-compatible candidates, best scope match first."""
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
        return _stable_order(survivors, score_fn=lambda item: float(counts[item.id]))[:limit]


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
        """Return up to ``limit`` lexical matches, best-first."""
        with self._sessions() as session:
            if _dialect_name(session) == "postgresql":
                return self._search_postgres(session, query_text, scope, kinds, limit)
            return self._search_portable(session, query_text, scope, kinds, limit)

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
        """Return up to ``limit`` embedded findings/experiments nearest the query, best-first."""
        literal = _vector_literal(query_embedding)
        with self._sessions() as session:
            if _dialect_name(session) == "postgresql":
                return self._search_postgres(session, literal, scope, kinds, limit)
            return self._search_portable(session, query_embedding, scope, kinds, limit)

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
        """Return up to ``limit`` failure candidates relevant to the query, best-first."""
        with self._sessions() as session:
            if _dialect_name(session) == "postgresql":
                return self._search_postgres(session, query_text, scope, limit)
            return self._search_portable(session, query_text, scope, limit)

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


class SQLExperimentLookupStore(SQLExperimentSearchStore):
    """``ExperimentLookupStore``: search binding plus primary-key lookup.

    Extends :class:`SQLExperimentSearchStore` with ``find_by_id`` (UUID
    primary-key point lookup with the same batched link fetch), completing
    the experiment read boundary the lookup tools need.
    """

    def find_by_id(self, experiment_id: str) -> ExperimentRecord | None:
        """Return the experiment with this id, or None (primary-key lookup).

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
            return row_to_record(row, datasets=links[0][0], result_artifacts=links[0][1])


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
    the four SQL channels single-pass (``relational=False`` — no edge
    table at Phase 2), and renders the fused ranking as
    :class:`RetrievalDocument` pages. :meth:`get_documents` is a
    primary-key fetch across the three retrievable tables.

    The vector channel runs only when an ``embedding_provider`` is bound;
    otherwise retrieval skips it with a recorded warning (identical to the
    planner's no-embedding path). Provider outages degrade the same way —
    a log line plus the warning — because an embedding outage must narrow
    recall, never fail the whole search.
    """

    def __init__(self, session_factory: Callable[[], Session], *, embedding_provider: EmbeddingProvider | None = None) -> None:
        """Bind the service to a session factory plus an optional provider.

        Args:
            session_factory: Zero-argument session factory shared by all
                four channel stores.
            embedding_provider: Query-embedding computation boundary (None
                disables the vector channel).
        """
        self._sessions = session_factory
        self._structured = SQLStructuredLookupStore(session_factory)
        self._lexical = SQLLexicalSearchStore(session_factory)
        self._vector = SQLVectorSearchStore(session_factory)
        self._failures = SQLFailureSearchStore(session_factory)
        self._embedding_provider = embedding_provider

    @property
    def embedding_provider(self) -> EmbeddingProvider | None:
        """The bound query-embedding provider (None when vector is off)."""
        return self._embedding_provider

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
            relational=False,
        )
        query_embedding = self._embed_query(query_text)
        allowed = [filters["project_id"]] if filters.get("project_id") else None
        result = execute_retrieval(
            plan,
            query_text=query_text,
            structured=self._structured,
            lexical=self._lexical,
            vector=self._vector,
            failures=self._failures,
            relational=None,
            query_embedding=query_embedding,
            allowed_project_ids=allowed,
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


def open_pg_backends(dsn: str, *, embedding_model: str | None = None, artifact_store: Any | None = None) -> KnowledgeBackends:
    """Build the tool-layer backend bundle bound to the database at ``dsn``.

    Reuses the process-wide sync-engine cache (one engine/pool per URL), so
    repeated calls with the same DSN share connections. Pass the result to
    :func:`deerflow.knowledge.tools.lookup.bind_knowledge_backends` to arm
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
            an :class:`ArtifactStoreReader` (None leaves artifacts unbound).

    Returns:
        A :class:`KnowledgeBackends` bundle (retrieval + experiments always
        bound, artifacts bound when a store is given).
    """
    from deerflow.persistence.agents.sql import get_sync_sessionmaker

    session_factory = get_sync_sessionmaker(dsn)
    provider = load_provider(embedding_model) if embedding_model is not None else None
    return KnowledgeBackends(
        retrieval=KnowledgeRetrievalService(session_factory, embedding_provider=provider),
        experiments=SQLExperimentLookupStore(session_factory),
        artifacts=ArtifactStoreReader(artifact_store) if artifact_store is not None else None,
    )

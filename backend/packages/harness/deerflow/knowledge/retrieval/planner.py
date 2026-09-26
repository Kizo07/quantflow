"""Retrieval planner for the Research Knowledge Plane (Phase 2: shared recall).

Implements ``knowledge_base.md`` § "Retrieval should be hybrid, scoped, and
evidence-aware": every run starts from a structured **research intent**
(:func:`parse_research_intent`), the planner converts it into a
:class:`RetrievalPlan` (scope filter + channels + limits), and
:func:`execute_retrieval` runs the channels concurrently-from-the-caller's
view (sequentially in this synchronous API) and fuses the rankings with
:mod:`deerflow.knowledge.retrieval.fusion`.

Channels (see ``CHANNELS``):

* ``structured`` — exact strategy/factor identifiers, dataset families,
  experiment families, date ranges, and scope fields (``WHERE`` clauses).
* ``lexical`` — PostgreSQL full-text search over ``search_document``
  (``plainto_tsquery`` + ``ts_rank_cd``); portable ``LIKE`` fallback behind
  the same method (see :func:`like_fallback_score`).
* ``vector`` — exact cosine search over ``embedding`` (pgvector ``<=>``
  distance operator, exact scan first; HNSW later per the KB).
* ``failure`` — dedicated failure channel: failed experiments plus failure
  findings, boosted in fusion so negative evidence is never buried.
* ``relational`` — clean Phase 3 seam: when a
  :class:`RelationalExpansionStore` is provided, top fused hits expand over
  ``knowledge_edge`` types; when None, retrieval is single-pass.

Storage boundary: every channel reads through a small ``Protocol`` (see
below for the PostgreSQL binding sketch of each). This module performs no
I/O of its own, holds no connections, and is safe to call from any thread
or event loop as long as the store implementations are.

Research semantics filter before ranking: the planner threads the
:class:`ScopeFilter` to the stores for index-side filtering, and fusion
applies the documented hard-incompatibility filters before RRF, so a great
embedding score can never rescue a rejected finding or an impossible
asset-class mismatch.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from deerflow.knowledge.hashing import normalize
from deerflow.knowledge.retrieval.fusion import FusedCandidate, FusionResult, FusionWeights, fuse
from deerflow.knowledge.write_api import KnowledgeValidationError

__all__ = [
    "CHANNEL_STRUCTURED",
    "CHANNEL_LEXICAL",
    "CHANNEL_VECTOR",
    "CHANNEL_FAILURE",
    "CHANNEL_RELATIONAL",
    "CHANNELS",
    "CANDIDATE_KINDS",
    "FINDING_TYPES",
    "FINDING_STATUSES",
    "EDGE_TYPES",
    "NEED_MEMORY_DEFAULT",
    "NEED_MEMORY_VOCAB",
    "DEFAULT_KINDS",
    "DEFAULT_PER_CHANNEL_LIMIT",
    "MAX_PER_CHANNEL_LIMIT",
    "DEFAULT_TOP_K",
    "MAX_TOP_K",
    "DEFAULT_RELATIONAL_SEED_K",
    "ScopeFilter",
    "ResearchIntent",
    "Candidate",
    "ChannelHit",
    "RetrievalPlan",
    "RetrievalResult",
    "StructuredLookupStore",
    "LexicalSearchStore",
    "VectorSearchStore",
    "FailureSearchStore",
    "RelationalExpansionStore",
    "parse_research_intent",
    "build_scope_filter",
    "plan_retrieval",
    "execute_retrieval",
    "cosine_similarity",
    "like_fallback_score",
]

#: Exact-match channel over identifiers, dataset families, and scope fields.
CHANNEL_STRUCTURED = "structured"
#: Full-text channel (PG FTS, portable LIKE fallback).
CHANNEL_LEXICAL = "lexical"
#: Cosine-similarity channel over finding/experiment embeddings.
CHANNEL_VECTOR = "vector"
#: Dedicated channel for failed experiments + failure findings.
CHANNEL_FAILURE = "failure"
#: Phase 3 knowledge-edge expansion channel (optional seam).
CHANNEL_RELATIONAL = "relational"

#: All channels in canonical (deterministic) order.
CHANNELS = (
    CHANNEL_STRUCTURED,
    CHANNEL_LEXICAL,
    CHANNEL_VECTOR,
    CHANNEL_FAILURE,
    CHANNEL_RELATIONAL,
)

#: Closed candidate-kind vocabulary (KB canonical schema + packet kinds).
CANDIDATE_KINDS = frozenset({"finding", "experiment", "failure", "assumption", "skill", "prior", "summary", "conflict"})

#: Closed finding-type vocabulary (KB: findings are claims, not notes).
FINDING_TYPES = frozenset({"empirical", "methodological", "data_quality", "failure", "prior"})

#: Finding lifecycle statuses (KB canonical schema; Phase 2 writes candidates only).
FINDING_STATUSES = frozenset({"candidate", "reviewed", "validated", "disputed", "superseded", "rejected"})

#: Knowledge-edge types for relational expansion (KB § lightweight graph; bound in Phase 3).
EDGE_TYPES = ("supports", "contradicts", "derived_from", "replicates", "supersedes", "related_to", "uses", "applies_to")

#: Default ``needed_memory`` when the intent omits it (the five KB bootstrap needs).
NEED_MEMORY_DEFAULT = ("validated_findings", "prior_experiments", "failures", "conflicts", "relevant_skills")

#: Closed ``needed_memory`` vocabulary (KB five + assumptions, which the packet also serves).
NEED_MEMORY_VOCAB = frozenset({*NEED_MEMORY_DEFAULT, "assumptions"})

#: Default retrieval kinds (conflict rows and L0/L1 summaries arrive in Phases 3/4).
DEFAULT_KINDS = ("finding", "experiment", "failure", "assumption", "skill", "prior")

#: Default per-channel row limit (channels over-fetch; fusion + packet trim to budget).
DEFAULT_PER_CHANNEL_LIMIT = 50
#: Hard ceiling for per-channel limits (rejected, never silently clamped).
MAX_PER_CHANNEL_LIMIT = 200
#: Default fused shortlist depth.
DEFAULT_TOP_K = 10
#: Hard ceiling for ``top_k``.
MAX_TOP_K = 100
#: Default seed count for relational expansion (top-N pass-1 survivors).
DEFAULT_RELATIONAL_SEED_K = 10

_HASH_RE = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _require_non_empty_str(name: str, value: Any, *, max_len: int = 100_000) -> str:
    """Validate a required string and return it stripped of outer whitespace."""
    if not isinstance(value, str):
        raise KnowledgeValidationError(f"{name} must be a string, got {type(value).__name__}.")
    stripped = value.strip()
    if not stripped:
        raise KnowledgeValidationError(f"{name} must be a non-empty string.")
    if len(value) > max_len:
        raise KnowledgeValidationError(f"{name} exceeds {max_len} characters ({len(value)}).")
    return stripped


def _optional_str(name: str, value: Any, *, max_len: int = 100_000) -> str | None:
    """Validate an optional string (None passes through; blanks become None)."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise KnowledgeValidationError(f"{name} must be a string or None, got {type(value).__name__}.")
    if len(value) > max_len:
        raise KnowledgeValidationError(f"{name} exceeds {max_len} characters ({len(value)}).")
    stripped = value.strip()
    return stripped or None


def _require_str_tuple(name: str, value: Any, *, allow_empty: bool = True) -> tuple[str, ...]:
    """Validate a sequence of non-empty strings and return it as a tuple."""
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise KnowledgeValidationError(f"{name} must be a sequence of strings, got {type(value).__name__}.")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise KnowledgeValidationError(f"{name}[{index}] must be a non-empty string, got {item!r}.")
        items.append(item.strip())
    if not allow_empty and not items:
        raise KnowledgeValidationError(f"{name} must not be empty.")
    return tuple(items)


def _optional_moment(name: str, value: Any) -> str | None:
    """Validate an optional ISO-8601 date/datetime string (None passes through)."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeValidationError(f"{name} must be an ISO-8601 date/datetime string or None, got {value!r}.")
    try:
        datetime.fromisoformat(value.strip())
    except ValueError:
        raise KnowledgeValidationError(f"{name} must be an ISO-8601 date/datetime string or None, got {value!r}.") from None
    return value.strip()


def _require_jsonable(name: str, value: Any) -> Any:
    """Validate that a value canonicalizes, returning its normalized copy."""
    try:
        return normalize(value)
    except (TypeError, ValueError) as exc:
        raise KnowledgeValidationError(f"{name} must be JSON-canonicalizable: {exc}") from None


def _require_mapping(name: str, value: Any) -> dict[str, Any]:
    """Validate a required mapping and return its normalized plain-dict copy."""
    if not isinstance(value, Mapping):
        raise KnowledgeValidationError(f"{name} must be a mapping, got {type(value).__name__}.")
    normalized = _require_jsonable(name, dict(value))
    assert isinstance(normalized, dict)
    return normalized


def _optional_uuid(name: str, value: Any) -> str | None:
    """Validate an optional UUID string, returning its canonical lowercase form."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise KnowledgeValidationError(f"{name} must be a UUID string or None, got {type(value).__name__}.")
    import uuid as _uuid

    try:
        return str(_uuid.UUID(value.strip()))
    except ValueError:
        raise KnowledgeValidationError(f"{name} must be a valid UUID or None, got {value!r}.") from None


def _optional_hash(name: str, value: Any) -> str | None:
    """Validate an optional 64-char lowercase hex digest (None passes through)."""
    if value is None:
        return None
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise KnowledgeValidationError(f"{name} must be a 64-character lowercase hex digest or None, got {value!r}.")
    return value


@dataclass(frozen=True)
class ScopeFilter:
    """Structured research-semantics filter threaded to every channel.

    Fields left as None/empty impose no constraint. ``valid_from`` /
    ``valid_to`` declare the *required* validity window (from the intent's
    ``requested_period``): fusion drops candidates whose declared effective
    window cannot overlap it. ``allow_restricted_datasets`` opts in to
    restricted-dataset candidates (default False: they filter out).
    """

    asset_class: str | None = None
    markets: tuple[str, ...] = ()
    universe: str | None = None
    horizon: str | None = None
    frequency: str | None = None
    dataset_family: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    allow_restricted_datasets: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the scope filter."""
        return {
            "asset_class": self.asset_class,
            "markets": list(self.markets),
            "universe": self.universe,
            "horizon": self.horizon,
            "frequency": self.frequency,
            "dataset_family": self.dataset_family,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "allow_restricted_datasets": self.allow_restricted_datasets,
        }


@dataclass(frozen=True)
class ResearchIntent:
    """Structured research intent: the task converted to retrieval semantics.

    Mirrors the KB bootstrap shape (``topic`` plus scope fields plus
    ``needed_memory``). Use :func:`parse_research_intent` to build one from
    untrusted input; the constructor assumes validated values.
    """

    topic: str = ""
    asset_class: str | None = None
    markets: tuple[str, ...] = ()
    universe: str | None = None
    horizon: str | None = None
    frequency: str | None = None
    concepts: tuple[str, ...] = ()
    requested_period: tuple[str | None, str | None] = (None, None)
    needed_memory: tuple[str, ...] = NEED_MEMORY_DEFAULT

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the research intent."""
        return {
            "topic": self.topic,
            "asset_class": self.asset_class,
            "markets": list(self.markets),
            "universe": self.universe,
            "horizon": self.horizon,
            "frequency": self.frequency,
            "concepts": list(self.concepts),
            "requested_period": [self.requested_period[0], self.requested_period[1]],
            "needed_memory": list(self.needed_memory),
        }


@dataclass(frozen=True)
class Candidate:
    """One retrievable research object, normalized across kinds.

    ``scope`` carries the KB scope fields (``asset_class``, ``market`` /
    ``markets``, ``universe``, ``horizon``, ``frequency``,
    ``dataset_family``, ``dataset_restricted``) plus any future additions;
    ``metadata`` carries ranking signals (``replication_count``,
    ``replicates``, ``replicated_experiment_id``, ``near_dup_group``,
    ``outcome``). ``valid_from`` / ``valid_to`` are the effective
    (valid-time) window; ``recorded_at`` is the transaction-time moment
    used for recency (ISO-8601, "" when unknown). ``family_hash`` links
    experiments to their design family for the same-family boost.
    ``evidence`` lists L2 pointer ids (experiment/artifact/finding rows)
    the packet renders without fetching.
    """

    id: str = ""
    kind: str = ""
    title: str = ""
    text: str = ""
    scope: dict[str, Any] = field(default_factory=dict)
    status: str = ""
    finding_type: str | None = None
    project_id: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    recorded_at: str = ""
    family_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and normalize every field (raises KnowledgeValidationError)."""
        candidate_id = _require_non_empty_str("id", self.id, max_len=255)
        object.__setattr__(self, "id", candidate_id)
        if not isinstance(self.kind, str) or self.kind not in CANDIDATE_KINDS:
            raise KnowledgeValidationError(f"kind must be one of {sorted(CANDIDATE_KINDS)}, got {self.kind!r}.")
        if not isinstance(self.title, str):
            raise KnowledgeValidationError(f"title must be a string, got {type(self.title).__name__}.")
        if not isinstance(self.text, str):
            raise KnowledgeValidationError(f"text must be a string, got {type(self.text).__name__}.")
        object.__setattr__(self, "scope", _require_mapping("scope", self.scope))
        object.__setattr__(self, "status", _require_non_empty_str("status", self.status, max_len=64))
        if self.finding_type is not None and (not isinstance(self.finding_type, str) or self.finding_type not in FINDING_TYPES):
            raise KnowledgeValidationError(f"finding_type must be one of {sorted(FINDING_TYPES)} or None, got {self.finding_type!r}.")
        object.__setattr__(self, "project_id", _optional_uuid("project_id", self.project_id))
        object.__setattr__(self, "valid_from", _optional_moment("valid_from", self.valid_from))
        object.__setattr__(self, "valid_to", _optional_moment("valid_to", self.valid_to))
        if not isinstance(self.recorded_at, str):
            raise KnowledgeValidationError(f"recorded_at must be an ISO-8601 string or '', got {type(self.recorded_at).__name__}.")
        if self.recorded_at:
            _optional_moment("recorded_at", self.recorded_at)
        object.__setattr__(self, "family_hash", _optional_hash("family_hash", self.family_hash))
        object.__setattr__(self, "metadata", _require_mapping("metadata", self.metadata))
        object.__setattr__(self, "evidence", _require_str_tuple("evidence", self.evidence))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the candidate."""
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "text": self.text,
            "scope": normalize(self.scope),
            "status": self.status,
            "finding_type": self.finding_type,
            "project_id": self.project_id,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "recorded_at": self.recorded_at,
            "family_hash": self.family_hash,
            "metadata": normalize(self.metadata),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ChannelHit:
    """One candidate placed at a 1-based rank by one channel (RRF consumes ranks, not scores)."""

    candidate: Candidate
    channel: str = ""
    rank: int = 0

    def __post_init__(self) -> None:
        """Validate the channel name and rank."""
        if not isinstance(self.candidate, Candidate):
            raise KnowledgeValidationError(f"candidate must be a Candidate, got {type(self.candidate).__name__}.")
        if not isinstance(self.channel, str) or self.channel not in CHANNELS:
            raise KnowledgeValidationError(f"channel must be one of {list(CHANNELS)}, got {self.channel!r}.")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 1:
            raise KnowledgeValidationError(f"rank must be an int >= 1, got {self.rank!r}.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the channel hit."""
        return {"candidate": self.candidate.to_dict(), "channel": self.channel, "rank": self.rank}


@runtime_checkable
class StructuredLookupStore(Protocol):
    """Read boundary for exact structured lookup; the integration step binds this to PostgreSQL.

    PG binding sketch: parameterized ``WHERE`` over the ``finding`` /
    ``experiment`` / ``assumption`` / ``skill_version`` / ``research_prior``
    tables — ``kind = ANY($1)``, scope JSONB containment
    (``scope @> $2``), family-hash equality, validity-overlap predicates —
    ordered by ``recorded_at DESC`` with ``LIMIT``.
    """

    def structured_lookup(self, scope: ScopeFilter, *, kinds: Sequence[str], limit: int) -> list[Candidate]:
        """Return up to ``limit`` candidates matching the scope, best-first."""
        ...


@runtime_checkable
class LexicalSearchStore(Protocol):
    """Read boundary for lexical retrieval; the integration step binds this to PostgreSQL FTS.

    PG binding sketch (canonical)::
        SELECT ..., ts_rank_cd(search_document, plainto_tsquery('english', $1)) AS rank
          FROM finding
         WHERE search_document @@ plainto_tsquery('english', $1)
           AND <scope predicates>
         ORDER BY rank DESC LIMIT $2;
    with a GIN index on ``search_document`` (experiments/failures union the
    same shape). Portable fallback: implementations without FTS satisfy this
    method with parameterized ``LIKE`` substring predicates
    (``%``/``_``/``\\\\`` escaped) ordered by :func:`like_fallback_score`.
    """

    def lexical_search(self, query_text: str, scope: ScopeFilter, *, kinds: Sequence[str], limit: int) -> list[Candidate]:
        """Return up to ``limit`` candidates matching the query text, best-first."""
        ...


@runtime_checkable
class VectorSearchStore(Protocol):
    """Read boundary for exact vector cosine search; the integration step binds this to pgvector.

    PG binding sketch (exact scan first, per the KB — HNSW only after
    benchmarking shows exact search missing latency objectives)::
        SELECT ..., embedding <=> $1 AS distance
          FROM finding
         WHERE <scope predicates>
         ORDER BY distance ASC LIMIT $2;
    The ``<=>`` cosine-distance operator orders conceptually similar rows
    first; fusion consumes the rank order (RRF), never the raw distance.
    """

    def vector_search(self, query_embedding: Sequence[float], scope: ScopeFilter, *, kinds: Sequence[str], limit: int) -> list[Candidate]:
        """Return up to ``limit`` candidates nearest the query embedding, best-first."""
        ...


@runtime_checkable
class FailureSearchStore(Protocol):
    """Read boundary for the dedicated failure channel (Reflexion/ExpeL negative evidence).

    PG binding sketch: union of failed experiments
    (``experiment WHERE outcome = 'failure'``) and failure findings
    (``finding WHERE finding_type = 'failure'``) matching the query text
    lexically, ordered by ``ts_rank_cd`` then ``recorded_at DESC`` with
    ``LIMIT``. Fusion adds the failure boost on top of channel inclusion so
    relevant failures surface even when topical overlap is thin (the eval
    fixture's keyword baseline proves one similarity ranking cannot do this).
    """

    def search_failures(self, query_text: str, scope: ScopeFilter, *, limit: int) -> list[Candidate]:
        """Return up to ``limit`` failure candidates relevant to the query, best-first."""
        ...


@runtime_checkable
class RelationalExpansionStore(Protocol):
    """Read boundary for Phase 3 knowledge-edge expansion (bound to the FK graph).

    Phase 3 binds this to the KB relations (finding supersession,
    experiment lineage / replication / family, assumption links, shared
    dataset/artifact co-usage) with shallow single-hop SQL from the seed
    ids, e.g.: from a strategy to its experiments, from an experiment to
    its failures, from a finding to the revision that supersedes it.
    ``supports`` / ``contradicts`` have no backing relation yet (the
    ``finding_evidence`` / conflict structures are still unlanded) and
    contribute no neighbors. The planner calls it only when provided;
    ``None`` means single-pass retrieval with no expansion round.
    """

    def expand_neighbors(self, seed_ids: Sequence[str], scope: ScopeFilter, *, edge_types: Sequence[str], limit: int) -> list[Candidate]:
        """Return up to ``limit`` candidates adjacent to the seeds, best-first."""
        ...


@dataclass(frozen=True)
class RetrievalPlan:
    """Executable retrieval plan: scope + channels + limits + expansion policy.

    ``channels`` never contains ``relational`` (expansion is a second pass
    armed by ``relational=True`` and executed only when the caller supplies
    a :class:`RelationalExpansionStore`).
    """

    intent: ResearchIntent = field(default_factory=ResearchIntent)
    scope: ScopeFilter = field(default_factory=ScopeFilter)
    channels: tuple[str, ...] = (CHANNEL_STRUCTURED, CHANNEL_LEXICAL, CHANNEL_VECTOR, CHANNEL_FAILURE)
    kinds: tuple[str, ...] = DEFAULT_KINDS
    per_channel_limit: int = DEFAULT_PER_CHANNEL_LIMIT
    top_k: int = DEFAULT_TOP_K
    edge_types: tuple[str, ...] = EDGE_TYPES
    relational: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the retrieval plan."""
        return {
            "intent": self.intent.to_dict(),
            "scope": self.scope.to_dict(),
            "channels": list(self.channels),
            "kinds": list(self.kinds),
            "per_channel_limit": self.per_channel_limit,
            "top_k": self.top_k,
            "edge_types": list(self.edge_types),
            "relational": self.relational,
        }


@dataclass(frozen=True)
class RetrievalResult:
    """Fused retrieval output with per-channel diagnostics."""

    plan: RetrievalPlan = field(default_factory=RetrievalPlan)
    fusion: FusionResult = field(default_factory=FusionResult)
    channel_counts: dict[str, int] = field(default_factory=dict)
    rounds: int = 1
    warnings: tuple[str, ...] = ()

    @property
    def top(self) -> list[FusedCandidate]:
        """Return the fused shortlist trimmed to ``plan.top_k`` (rank order)."""
        return list(self.fusion.candidates[: self.plan.top_k])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the retrieval result."""
        return {
            "plan": self.plan.to_dict(),
            "fusion": self.fusion.to_dict(),
            "channel_counts": dict(self.channel_counts),
            "rounds": self.rounds,
            "warnings": list(self.warnings),
        }


def parse_research_intent(data: Mapping[str, Any]) -> ResearchIntent:
    """Parse and validate untrusted input into a :class:`ResearchIntent`.

    Accepts the KB bootstrap shape (also used by the eval fixture intents):
    ``topic`` (required), ``asset_class``, ``markets``, ``universe``,
    ``horizon``, ``frequency``, ``concepts``, ``requested_period``
    (``[start, end]`` ISO dates, either end optional), ``needed_memory``.
    Unknown keys are ignored so future intent fields never break readers.

    Raises:
        KnowledgeValidationError: On any invalid field.
    """
    if not isinstance(data, Mapping):
        raise KnowledgeValidationError(f"intent must be a mapping, got {type(data).__name__}.")
    topic = _require_non_empty_str("topic", data.get("topic"))
    asset_class = _optional_str("asset_class", data.get("asset_class"), max_len=128)
    markets = _require_str_tuple("markets", data.get("markets", ()))
    universe = _optional_str("universe", data.get("universe"), max_len=256)
    horizon = _optional_str("horizon", data.get("horizon"), max_len=128)
    frequency = _optional_str("frequency", data.get("frequency"), max_len=128)
    concepts = _require_str_tuple("concepts", data.get("concepts", ()))
    raw_period = data.get("requested_period")
    period: tuple[str | None, str | None] = (None, None)
    if raw_period is not None:
        if not isinstance(raw_period, (list, tuple)) or len(raw_period) != 2:
            raise KnowledgeValidationError(f"requested_period must be a [start, end] pair, got {raw_period!r}.")
        start = _optional_moment("requested_period[0]", raw_period[0])
        end = _optional_moment("requested_period[1]", raw_period[1])
        if start is not None and end is not None and datetime.fromisoformat(start) > datetime.fromisoformat(end):
            raise KnowledgeValidationError(f"requested_period start {start!r} is after end {end!r}.")
        period = (start, end)
    raw_needed = data.get("needed_memory")
    if raw_needed is None:
        needed = NEED_MEMORY_DEFAULT
    else:
        items = _require_str_tuple("needed_memory", raw_needed, allow_empty=False)
        for item in items:
            if item not in NEED_MEMORY_VOCAB:
                raise KnowledgeValidationError(f"needed_memory entries must be one of {sorted(NEED_MEMORY_VOCAB)}, got {item!r}.")
        needed = tuple(dict.fromkeys(items))
    return ResearchIntent(
        topic=topic,
        asset_class=asset_class,
        markets=markets,
        universe=universe,
        horizon=horizon,
        frequency=frequency,
        concepts=concepts,
        requested_period=period,
        needed_memory=needed,
    )


def build_scope_filter(intent: ResearchIntent) -> ScopeFilter:
    """Derive the structured :class:`ScopeFilter` from a research intent.

    The intent's ``requested_period`` becomes the required validity window;
    scope fields carry over verbatim. ``allow_restricted_datasets``
    defaults to False (callers opt in explicitly at the agent-tool layer).
    """
    if not isinstance(intent, ResearchIntent):
        raise KnowledgeValidationError(f"intent must be a ResearchIntent, got {type(intent).__name__}.")
    return ScopeFilter(
        asset_class=intent.asset_class,
        markets=intent.markets,
        universe=intent.universe,
        horizon=intent.horizon,
        frequency=intent.frequency,
        valid_from=intent.requested_period[0],
        valid_to=intent.requested_period[1],
    )


def _channels_for_needed_memory(needed_memory: Sequence[str]) -> tuple[str, ...]:
    """Map bootstrap needs to channels (canonical order, deduped)."""
    selected: list[str] = []
    for need in needed_memory:
        if need in ("validated_findings", "prior_experiments"):
            selected.extend((CHANNEL_STRUCTURED, CHANNEL_LEXICAL, CHANNEL_VECTOR))
        elif need == "failures":
            selected.append(CHANNEL_FAILURE)
        elif need in ("conflicts", "relevant_skills", "assumptions"):
            selected.append(CHANNEL_STRUCTURED)
    ordered = [channel for channel in CHANNELS if channel in selected and channel != CHANNEL_RELATIONAL]
    return tuple(ordered)


def plan_retrieval(
    intent: ResearchIntent | Mapping[str, Any],
    *,
    per_channel_limit: int = DEFAULT_PER_CHANNEL_LIMIT,
    top_k: int = DEFAULT_TOP_K,
    kinds: Sequence[str] | None = None,
    edge_types: Sequence[str] | None = None,
    relational: bool = True,
) -> RetrievalPlan:
    """Build an executable :class:`RetrievalPlan` from an intent (mapping or parsed).

    Channels derive from ``needed_memory``: findings/experiments enable the
    structured + lexical + vector channels, failures enable the dedicated
    failure channel, conflicts/skills/assumptions enable structured lookup
    (conflict sets and edge expansion arrive in Phase 3).

    Raises:
        KnowledgeValidationError: On invalid limits, kinds, edge types, or intent.
    """
    parsed = intent if isinstance(intent, ResearchIntent) else parse_research_intent(intent)
    if not isinstance(per_channel_limit, int) or isinstance(per_channel_limit, bool):
        raise KnowledgeValidationError(f"per_channel_limit must be an int, got {type(per_channel_limit).__name__}.")
    if per_channel_limit < 1 or per_channel_limit > MAX_PER_CHANNEL_LIMIT:
        raise KnowledgeValidationError(f"per_channel_limit must be in 1..{MAX_PER_CHANNEL_LIMIT}, got {per_channel_limit}.")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise KnowledgeValidationError(f"top_k must be an int, got {type(top_k).__name__}.")
    if top_k < 1 or top_k > MAX_TOP_K:
        raise KnowledgeValidationError(f"top_k must be in 1..{MAX_TOP_K}, got {top_k}.")
    effective_kinds = DEFAULT_KINDS if kinds is None else tuple(kinds)
    if not effective_kinds:
        raise KnowledgeValidationError("kinds must not be empty.")
    for kind in effective_kinds:
        if kind not in CANDIDATE_KINDS:
            raise KnowledgeValidationError(f"kinds entries must be one of {sorted(CANDIDATE_KINDS)}, got {kind!r}.")
    effective_edges = EDGE_TYPES if edge_types is None else tuple(edge_types)
    for edge in effective_edges:
        if edge not in EDGE_TYPES:
            raise KnowledgeValidationError(f"edge_types entries must be one of {list(EDGE_TYPES)}, got {edge!r}.")
    if not isinstance(relational, bool):
        raise KnowledgeValidationError(f"relational must be a bool, got {type(relational).__name__}.")
    return RetrievalPlan(
        intent=parsed,
        scope=build_scope_filter(parsed),
        channels=_channels_for_needed_memory(parsed.needed_memory),
        kinds=tuple(dict.fromkeys(effective_kinds)),
        per_channel_limit=per_channel_limit,
        top_k=top_k,
        edge_types=tuple(dict.fromkeys(effective_edges)),
        relational=relational,
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two embeddings (pure brute-force kernel).

    Used by the in-test vector fake (and any portable implementation);
    production binds :class:`VectorSearchStore` to pgvector's ``<=>``
    operator instead. Zero vectors score 0.0 (no NaN); range is [-1, 1].

    Raises:
        KnowledgeValidationError: On empty vectors, length mismatch, or
            non-finite / non-numeric components.
    """
    for name, vector in (("left", left), ("right", right)):
        if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence) or len(vector) == 0:
            raise KnowledgeValidationError(f"{name} must be a non-empty sequence of floats.")
        for component in vector:
            if not isinstance(component, (int, float)) or isinstance(component, bool) or not math.isfinite(component):
                raise KnowledgeValidationError(f"{name} must contain only finite numbers, got {component!r}.")
    if len(left) != len(right):
        raise KnowledgeValidationError(f"embedding lengths differ: {len(left)} != {len(right)}.")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def like_fallback_score(document_text: str, query_text: str) -> float:
    """Score a document for the portable LIKE fallback (deterministic, in [0, 1]).

    Formula: fraction of distinct lowercase alphanumeric query tokens
    appearing as substrings in the document, plus a 0.2 phrase bonus when
    the full normalized query appears verbatim (capped at 1.0). Empty
    queries score 0.0. Portable store implementations order ``LIKE``
    matches by this score; it is a ranking signal only, never an identity
    signal (exact identity stays hash-based per :mod:`deerflow.knowledge.hashing`).
    """
    if not isinstance(document_text, str) or not isinstance(query_text, str):
        raise KnowledgeValidationError(f"document_text and query_text must be strings, got {type(document_text).__name__} and {type(query_text).__name__}.")
    query_tokens = set(_TOKEN_RE.findall(query_text.lower()))
    if not query_tokens:
        return 0.0
    lowered = document_text.lower()
    matched = sum(1 for token in query_tokens if token in lowered)
    score = matched / len(query_tokens)
    normalized_query = " ".join(query_text.lower().split())
    if normalized_query and normalized_query in " ".join(lowered.split()):
        score = min(1.0, score + 0.2)
    return score


#: English stopwords dropped from LIKE-fallback query tokens, mirroring
#: PostgreSQL's ``english`` text-search configuration used by the FTS path
#: (``plainto_tsquery`` drops the same class of words). Without this, the
#: portable path counts noise tokens (``how``/``do``/``in``/``up``/...) that
#: the production path ignores, and substring matches like ``"in"`` inside
#: ``"Brinson"`` flatten every score toward the mean.
_LIKE_STOPWORDS = frozenset(
    """
    a an the and or but if then else when what why how do does did done is are
    was were be been being am an as at by for from in into of off on onto out
    over to up with about into over after before between during under again
    further once here there where which who whom whose this that these those
    it its we you he she they them him her their our your my me us him i s t
    d m ll ve re not no nor only own same so than too very can will just
    should now get show vs via per
    """.split()
)


def lexical_query_text(query_text: str, intent: ResearchIntent | Mapping[str, Any] | None) -> str:
    """Fold an intent's concepts into lexical query text (deduped).

    Planner-direct callers pass this to the lexical and failure channels
    instead of the raw query: concepts (``reversal``, ``Brinson``) carry
    vocabulary the natural-language question often lacks, and the
    service path (``ledger_search``) already folds them the same way
    via its ``concept`` filter. Concepts already present as query
    substrings are skipped; a missing/empty intent returns the text
    unchanged.
    """
    if not isinstance(query_text, str):
        raise KnowledgeValidationError(f"query_text must be a string, got {type(query_text).__name__}.")
    concepts: Sequence[str] = ()
    if isinstance(intent, ResearchIntent):
        concepts = intent.concepts
    elif isinstance(intent, Mapping):
        raw = intent.get("concepts", ())
        concepts = tuple(raw) if isinstance(raw, (list, tuple)) else ()
    lowered = query_text.lower()
    extra = [str(concept).strip() for concept in concepts if str(concept).strip() and str(concept).strip().lower() not in lowered]
    if not extra:
        return query_text
    return f"{query_text} {' '.join(extra)}"


def like_fallback_idf_scores(document_texts: Sequence[str], query_text: str) -> list[float]:
    """IDF-weighted token-overlap scores in [0, 1], one per document, in order.

    BM25-lite for the portable LIKE path: stopwords are dropped from the
    query (mirroring the PG ``english`` config), then each matched token
    weighs ``log((N + 1) / (df + 1)) + 1`` (N = candidate count,
    df = candidates containing it), so rare terms (``brinson``,
    ``survivorship``) outvote ubiquitous ones (``risk``, ``momentum``).
    The score is matched-IDF mass over total-query-IDF mass, plus the
    same 0.2 verbatim-phrase bonus as :func:`like_fallback_score`
    (capped at 1.0). Query tokens absent from every candidate still
    count in the denominator; a query with no content tokens scores all
    zeros. Deterministic: same inputs, same outputs.
    """
    if not isinstance(document_texts, Sequence) or isinstance(document_texts, (str, bytes)):
        raise KnowledgeValidationError(f"document_texts must be a sequence of strings, got {type(document_texts).__name__}.")
    for index, text in enumerate(document_texts):
        if not isinstance(text, str):
            raise KnowledgeValidationError(f"document_texts[{index}] must be a string, got {type(text).__name__}.")
    if not isinstance(query_text, str):
        raise KnowledgeValidationError(f"query_text must be a string, got {type(query_text).__name__}.")
    query_tokens = sorted(set(_TOKEN_RE.findall(query_text.lower())) - _LIKE_STOPWORDS)
    if not query_tokens:
        return [0.0 for _ in document_texts]
    lowered_docs = [text.lower() for text in document_texts]
    total = len(lowered_docs)
    doc_freq = {token: sum(1 for lowered in lowered_docs if token in lowered) for token in query_tokens}
    idf = {token: math.log((total + 1) / (doc_freq[token] + 1)) + 1.0 for token in query_tokens}
    denominator = sum(idf[token] for token in query_tokens)
    normalized_query = " ".join(query_text.lower().split())
    scores = []
    for lowered in lowered_docs:
        matched = sum(idf[token] for token in query_tokens if token in lowered)
        score = matched / denominator
        if normalized_query and normalized_query in " ".join(lowered.split()):
            score = min(1.0, score + 0.2)
        scores.append(score)
    return scores


def _validate_embedding(name: str, value: Any) -> tuple[float, ...]:
    """Validate a query embedding (non-empty finite floats)."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) == 0:
        raise KnowledgeValidationError(f"{name} must be a non-empty sequence of floats.")
    components: list[float] = []
    for component in value:
        if not isinstance(component, (int, float)) or isinstance(component, bool) or not math.isfinite(component):
            raise KnowledgeValidationError(f"{name} must contain only finite numbers, got {component!r}.")
        components.append(float(component))
    return tuple(components)


def _as_hits(candidates: Sequence[Candidate], channel: str) -> list[ChannelHit]:
    """Assign 1-based ranks to a store's best-first candidates (validates kind)."""
    hits: list[ChannelHit] = []
    for position, candidate in enumerate(candidates):
        if not isinstance(candidate, Candidate):
            raise KnowledgeValidationError(f"{channel} store must return Candidate records, got {type(candidate).__name__}.")
        hits.append(ChannelHit(candidate=candidate, channel=channel, rank=position + 1))
    return hits


def execute_retrieval(
    plan: RetrievalPlan,
    *,
    query_text: str,
    structured: StructuredLookupStore,
    lexical: LexicalSearchStore,
    vector: VectorSearchStore,
    failures: FailureSearchStore,
    relational: RelationalExpansionStore | None = None,
    query_embedding: Sequence[float] | None = None,
    weights: FusionWeights | None = None,
    allowed_project_ids: Sequence[str] | None = None,
    query_family_hash: str | None = None,
    now: datetime | str | None = None,
    relational_seed_k: int = DEFAULT_RELATIONAL_SEED_K,
    apply_failure_boost: bool = True,
) -> RetrievalResult:
    """Execute a retrieval plan across the channel stores and fuse the rankings.

    Pass 1 runs the planned channels (structured / lexical / vector /
    failure); the vector channel is skipped with a warning when no
    ``query_embedding`` is supplied. Pass 2 runs only when the plan arms
    relational expansion (``plan.relational``) *and* a relational store is
    provided: the top ``relational_seed_k`` pass-1 survivors expand over
    ``plan.edge_types`` and everything re-fuses. Store errors propagate to
    the caller (no silent channel drops — partial rankings must never
    masquerade as complete recall).

    Args:
        plan: Executable plan from :func:`plan_retrieval`.
        query_text: Raw query prose for the lexical + failure channels.
        structured: Structured-lookup store implementation.
        lexical: Lexical-search store implementation.
        vector: Vector-search store implementation.
        failures: Failure-channel store implementation.
        relational: Optional Phase 3 edge-expansion store (None = single pass).
        query_embedding: Query embedding for the vector channel (None = skip).
        weights: Fusion weights (defaults to :class:`FusionWeights`).
        allowed_project_ids: Optional project-visibility enforcement set.
        query_family_hash: Optional query experiment-family digest (same-family boost).
        now: Recency reference moment (ISO-8601 or datetime; default current UTC).
        relational_seed_k: Seed count for the expansion pass (must be >= 1).
        apply_failure_boost: Forwarded to :func:`fuse` (priors/consensus
            rankings pass False so boosted failures cannot flood the
            general top-k; the failures section fuses separately).

    Returns:
        A :class:`RetrievalResult` with fused ranking + diagnostics.

    Raises:
        KnowledgeValidationError: On invalid query text/embedding/seed count.
    """
    if not isinstance(plan, RetrievalPlan):
        raise KnowledgeValidationError(f"plan must be a RetrievalPlan, got {type(plan).__name__}.")
    effective_query = _require_non_empty_str("query_text", query_text)
    if not isinstance(relational_seed_k, int) or isinstance(relational_seed_k, bool) or relational_seed_k < 1:
        raise KnowledgeValidationError(f"relational_seed_k must be an int >= 1, got {relational_seed_k!r}.")
    embedding = _validate_embedding("query_embedding", query_embedding) if query_embedding is not None else None

    warnings: list[str] = []
    hits: list[ChannelHit] = []
    channel_counts: dict[str, int] = {}
    if CHANNEL_STRUCTURED in plan.channels:
        found = list(structured.structured_lookup(plan.scope, kinds=list(plan.kinds), limit=plan.per_channel_limit))
        channel_counts[CHANNEL_STRUCTURED] = len(found)
        hits.extend(_as_hits(found, CHANNEL_STRUCTURED))
    if CHANNEL_LEXICAL in plan.channels:
        found = list(lexical.lexical_search(effective_query, plan.scope, kinds=list(plan.kinds), limit=plan.per_channel_limit))
        channel_counts[CHANNEL_LEXICAL] = len(found)
        hits.extend(_as_hits(found, CHANNEL_LEXICAL))
    if CHANNEL_VECTOR in plan.channels:
        if embedding is None:
            warnings.append("vector channel skipped: no query_embedding supplied")
            channel_counts[CHANNEL_VECTOR] = 0
        else:
            found = list(vector.vector_search(embedding, plan.scope, kinds=list(plan.kinds), limit=plan.per_channel_limit))
            channel_counts[CHANNEL_VECTOR] = len(found)
            hits.extend(_as_hits(found, CHANNEL_VECTOR))
    if CHANNEL_FAILURE in plan.channels:
        found = list(failures.search_failures(effective_query, plan.scope, limit=plan.per_channel_limit))
        channel_counts[CHANNEL_FAILURE] = len(found)
        hits.extend(_as_hits(found, CHANNEL_FAILURE))

    fused = fuse(hits, plan.scope, weights=weights, allowed_project_ids=allowed_project_ids, query_family_hash=query_family_hash, now=now, apply_failure_boost=apply_failure_boost)
    rounds = 1
    if plan.relational and relational is not None and fused.candidates:
        seeds = [item.candidate.id for item in fused.candidates[:relational_seed_k]]
        neighbors = list(relational.expand_neighbors(seeds, plan.scope, edge_types=list(plan.edge_types), limit=plan.per_channel_limit))
        channel_counts[CHANNEL_RELATIONAL] = len(neighbors)
        if neighbors:
            hits.extend(_as_hits(neighbors, CHANNEL_RELATIONAL))
            fused = fuse(hits, plan.scope, weights=weights, allowed_project_ids=allowed_project_ids, query_family_hash=query_family_hash, now=now, apply_failure_boost=apply_failure_boost)
        rounds = 2
    return RetrievalResult(plan=plan, fusion=fused, channel_counts=channel_counts, rounds=rounds, warnings=tuple(warnings))

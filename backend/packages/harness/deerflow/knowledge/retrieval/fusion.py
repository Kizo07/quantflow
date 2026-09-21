"""Reciprocal Rank Fusion + research-quality modifiers for the Research Knowledge Plane.

Each retrieval channel (structured lookup, lexical FTS, vector search, failure
search, relational expansion) produces its own best-first ranking. This module
combines those rankings with **Reciprocal Rank Fusion** (RRF) and then applies
deterministic research-quality modifiers, per ``knowledge_base.md``
§ "Use rank fusion rather than one magical similarity score":

.. code-block:: text

    candidate_score =
        fused_retrieval_rank
      × scope_compatibility
      × authority_modifier
      × validity_modifier
      × evidence_quality_modifier
      × diversity_modifier

The most important rule — **hard incompatibilities filter before ranking** —
is implemented by :func:`apply_hard_filters`. The documented hard-filter list:

1. ``rejected`` — findings with ``status == "rejected"`` are never retrievable.
2. ``acl_project_not_visible`` — when the caller enforces project visibility
   (``allowed_project_ids``), candidates scoped to any other project are
   dropped. Unscoped candidates (``project_id is None``) pass: enforcement
   applies to scoped rows only.
3. ``asset_class_mismatch`` — the scope filter and the candidate both declare
   ``asset_class`` (case-insensitive) and they differ.
4. ``restricted_dataset`` — the candidate scope declares
   ``dataset_restricted: true`` and the scope filter did not opt in via
   ``allow_restricted_datasets``.
5. ``validity_window_disjoint`` — the scope filter requires a validity window
   (``valid_from``/``valid_to``) and the candidate declares an effective
   window that does not overlap it. Candidates with no declared window pass:
   unknown validity is not incompatibility.

Within the compatible set, :func:`fuse` applies these multiplicative
modifiers (defaults in :class:`FusionWeights`; every applied factor is
recorded on :attr:`FusedCandidate.modifiers` so rankings stay auditable):

* exact-scope boost (×1.5) / partial-scope boost (×1.15)
* validated (×1.4) > reviewed (×1.15) > candidate (×1.0)
* replication boost (×1.25; ×1.4 with 3+ independent replications)
* same-family boost (×1.2 when the candidate family matches the query family)
* recency boost (×1.1 inside the recency window)
* dispute-visibility (×1.05; disputed findings stay visible and flagged)
* near-duplicate diversity penalty (×0.5 for later members of a group)
* superseded demote (×0.2; historical context stays findable, never prominent)
* failure boost (×1.3 for failures relevant to the proposed methodology)

Storage boundary: none — this module is pure ranking math over the
:class:`~deerflow.knowledge.retrieval.planner.ChannelHit` records the
planner built. It performs no I/O and holds no connections.

Determinism contract: RRF sums over channels in sorted channel order;
final ordering breaks ties by ``candidate.id`` ascending, so identical
inputs always produce identical rankings regardless of channel call order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from deerflow.knowledge.write_api import KnowledgeValidationError

if TYPE_CHECKING:
    from deerflow.knowledge.retrieval.planner import Candidate, ChannelHit, ScopeFilter

__all__ = [
    "REASON_REJECTED",
    "REASON_ACL",
    "REASON_ASSET_CLASS",
    "REASON_DATASET",
    "REASON_VALIDITY",
    "HARD_FILTER_REASONS",
    "MODIFIER_SCOPE_EXACT",
    "MODIFIER_SCOPE_PARTIAL",
    "MODIFIER_VALIDATED",
    "MODIFIER_REVIEWED",
    "MODIFIER_REPLICATION",
    "MODIFIER_STRONG_REPLICATION",
    "MODIFIER_SAME_FAMILY",
    "MODIFIER_RECENCY",
    "MODIFIER_DISPUTED",
    "MODIFIER_NEAR_DUP",
    "MODIFIER_SUPERSEDED",
    "MODIFIER_FAILURE",
    "DEFAULT_RRF_K",
    "FusionWeights",
    "FusedCandidate",
    "FilteredCandidate",
    "FusionResult",
    "apply_hard_filters",
    "reciprocal_rank_fusion",
    "scope_match_degree",
    "fuse",
]

#: Hard-filter reason: candidate status is ``rejected``.
REASON_REJECTED = "rejected"
#: Hard-filter reason: candidate project is outside the enforced visible set.
REASON_ACL = "acl_project_not_visible"
#: Hard-filter reason: declared asset classes disagree.
REASON_ASSET_CLASS = "asset_class_mismatch"
#: Hard-filter reason: restricted dataset without filter opt-in.
REASON_DATASET = "restricted_dataset"
#: Hard-filter reason: validity windows do not overlap.
REASON_VALIDITY = "validity_window_disjoint"

#: The documented hard-filter list, in evaluation order.
HARD_FILTER_REASONS = (
    REASON_REJECTED,
    REASON_ACL,
    REASON_ASSET_CLASS,
    REASON_DATASET,
    REASON_VALIDITY,
)

#: Modifier key recorded when the candidate scope exactly matches the filter scope.
MODIFIER_SCOPE_EXACT = "exact_scope"
#: Modifier key recorded on a partial scope match.
MODIFIER_SCOPE_PARTIAL = "partial_scope"
#: Modifier key recorded for validated findings.
MODIFIER_VALIDATED = "validated"
#: Modifier key recorded for reviewed findings.
MODIFIER_REVIEWED = "reviewed"
#: Modifier key recorded when independent replication evidence exists.
MODIFIER_REPLICATION = "replication"
#: Modifier key recorded when 3+ independent replications exist.
MODIFIER_STRONG_REPLICATION = "strong_replication"
#: Modifier key recorded when the candidate family matches the query family.
MODIFIER_SAME_FAMILY = "same_family"
#: Modifier key recorded for recently confirmed candidates.
MODIFIER_RECENCY = "recency"
#: Modifier key recorded for disputed findings (kept visible, flagged).
MODIFIER_DISPUTED = "dispute_visibility"
#: Modifier key recorded for later members of a near-duplicate group.
MODIFIER_NEAR_DUP = "near_dup_diversity"
#: Modifier key recorded for superseded candidates.
MODIFIER_SUPERSEDED = "superseded"
#: Modifier key recorded for failure candidates.
MODIFIER_FAILURE = "failure"

#: Default RRF smoothing constant (standard ``1 / (k + rank)``).
DEFAULT_RRF_K = 60

#: Scope fields compared for the exact/partial scope modifiers.
_SCOPE_FIELDS = ("asset_class", "universe", "horizon", "frequency", "dataset_family")


def _parse_moment(value: Any) -> datetime | None:
    """Parse an ISO-8601 date/datetime to an aware UTC datetime, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str) and value.strip():
        try:
            moment = datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    else:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _normalize_scope_value(value: Any) -> str | None:
    """Normalize one scope field to a comparable lowercase string."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


#: Scope values meaning "unspecified" rather than a real constraint.
#: A placeholder on either side makes the field neutral for match
#: degree, match-count ranking, and hard screening alike: filter
#: ``horizon="mixed"`` must not exactly-match (or mismatch) document
#: horizons, and a document tagged ``universe="n/a"`` constrains
#: nothing. Without this, vague values dominate ranking — every
#: ``mixed``-horizon row would outrank topically perfect rows on a
#: ``mixed``-horizon query via the exact-scope boost.
SCOPE_PLACEHOLDERS = frozenset({"mixed", "n/a", "na", "unknown", "unspecified"})


def _is_placeholder(normalized: str | None) -> bool:
    """Return True when a normalized scope value is an unspecified marker."""
    return normalized is not None and normalized in SCOPE_PLACEHOLDERS


@dataclass(frozen=True)
class FusionWeights:
    """Tunable RRF + modifier weights (KB: learned from retrieval evals, not hard-coded forever).

    The defaults encode the KB's qualitative ordering (exact-scope and
    validated findings win strongly; superseded rows sink; failures and
    disputes stay visible). Every multiplier must be positive; ``rrf_k``
    and ``recency_window_days`` must be >= 1.
    """

    rrf_k: int = DEFAULT_RRF_K
    exact_scope_boost: float = 1.5
    partial_scope_boost: float = 1.15
    validated_boost: float = 1.4
    reviewed_boost: float = 1.15
    disputed_visibility: float = 1.05
    replication_boost: float = 1.25
    strong_replication_boost: float = 1.4
    strong_replication_threshold: int = 3
    same_family_boost: float = 1.2
    recency_boost: float = 1.1
    recency_window_days: int = 365
    near_dup_penalty: float = 0.5
    superseded_demote: float = 0.2
    failure_boost: float = 1.3

    def __post_init__(self) -> None:
        """Validate that every weight is in range."""
        if not isinstance(self.rrf_k, int) or isinstance(self.rrf_k, bool) or self.rrf_k < 1:
            raise KnowledgeValidationError(f"rrf_k must be an int >= 1, got {self.rrf_k!r}.")
        if not isinstance(self.recency_window_days, int) or isinstance(self.recency_window_days, bool) or self.recency_window_days < 1:
            raise KnowledgeValidationError(f"recency_window_days must be an int >= 1, got {self.recency_window_days!r}.")
        if not isinstance(self.strong_replication_threshold, int) or isinstance(self.strong_replication_threshold, bool) or self.strong_replication_threshold < 1:
            raise KnowledgeValidationError(f"strong_replication_threshold must be an int >= 1, got {self.strong_replication_threshold!r}.")
        for name in (
            "exact_scope_boost",
            "partial_scope_boost",
            "validated_boost",
            "reviewed_boost",
            "disputed_visibility",
            "replication_boost",
            "strong_replication_boost",
            "same_family_boost",
            "recency_boost",
            "near_dup_penalty",
            "superseded_demote",
            "failure_boost",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not (value > 0):
                raise KnowledgeValidationError(f"{name} must be a positive number, got {value!r}.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the weights."""
        return {
            "rrf_k": self.rrf_k,
            "exact_scope_boost": self.exact_scope_boost,
            "partial_scope_boost": self.partial_scope_boost,
            "validated_boost": self.validated_boost,
            "reviewed_boost": self.reviewed_boost,
            "disputed_visibility": self.disputed_visibility,
            "replication_boost": self.replication_boost,
            "strong_replication_boost": self.strong_replication_boost,
            "strong_replication_threshold": self.strong_replication_threshold,
            "same_family_boost": self.same_family_boost,
            "recency_boost": self.recency_boost,
            "recency_window_days": self.recency_window_days,
            "near_dup_penalty": self.near_dup_penalty,
            "superseded_demote": self.superseded_demote,
            "failure_boost": self.failure_boost,
        }


@dataclass(frozen=True)
class FusedCandidate:
    """One ranked candidate with its fused score and applied modifiers."""

    candidate: Candidate
    score: float
    rrf_score: float
    modifiers: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    channels: tuple[str, ...] = ()
    disputed: bool = False
    failure: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the fused candidate."""
        return {
            "candidate": self.candidate.to_dict(),
            "score": self.score,
            "rrf_score": self.rrf_score,
            "modifiers": dict(self.modifiers),
            "rank": self.rank,
            "channels": list(self.channels),
            "disputed": self.disputed,
            "failure": self.failure,
        }


@dataclass(frozen=True)
class FilteredCandidate:
    """One candidate dropped by a hard filter, with the documented reason."""

    candidate: Candidate
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the filtered candidate."""
        return {"candidate": self.candidate.to_dict(), "reason": self.reason, "detail": self.detail}


@dataclass(frozen=True)
class FusionResult:
    """Ranked fusion output plus the audit trail of hard-filter drops."""

    candidates: list[FusedCandidate] = field(default_factory=list)
    filtered: list[FilteredCandidate] = field(default_factory=list)
    weights: FusionWeights = field(default_factory=FusionWeights)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the fusion result."""
        return {
            "candidates": [item.to_dict() for item in self.candidates],
            "filtered": [item.to_dict() for item in self.filtered],
            "weights": self.weights.to_dict(),
        }


def reciprocal_rank_fusion(rankings: Mapping[str, Sequence[str]], *, k: int = DEFAULT_RRF_K) -> dict[str, float]:
    """Combine per-channel rankings with RRF: ``score(id) = Σ 1 / (k + rank)``.

    Args:
        rankings: Mapping of channel name to candidate ids in best-first
            order (rank = position + 1). Channels iterate in sorted order so
            floating-point summation is deterministic.
        k: Smoothing constant; must be >= 1.

    Returns:
        Mapping of candidate id to fused RRF score (higher is better).

    Raises:
        KnowledgeValidationError: When ``k`` is not an int >= 1, a ranking
            is not a sequence of non-empty strings, or a channel lists the
            same id twice.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise KnowledgeValidationError(f"k must be an int >= 1, got {k!r}.")
    if not isinstance(rankings, Mapping):
        raise KnowledgeValidationError(f"rankings must be a mapping of channel name to id lists, got {type(rankings).__name__}.")
    scores: dict[str, float] = {}
    for channel in sorted(rankings):
        ordered = rankings[channel]
        if isinstance(ordered, (str, bytes)) or not isinstance(ordered, Sequence):
            raise KnowledgeValidationError(f"ranking for channel {channel!r} must be a sequence of candidate ids.")
        seen: set[str] = set()
        for position, candidate_id in enumerate(ordered):
            if not isinstance(candidate_id, str) or not candidate_id:
                raise KnowledgeValidationError(f"ranking for channel {channel!r} must contain non-empty string ids, got {candidate_id!r}.")
            if candidate_id in seen:
                raise KnowledgeValidationError(f"ranking for channel {channel!r} lists {candidate_id!r} twice.")
            seen.add(candidate_id)
            scores[candidate_id] = scores.get(candidate_id, 0.0) + 1.0 / (k + position + 1)
    return scores


def scope_match_degree(candidate_scope: Mapping[str, Any], scope: ScopeFilter) -> str:
    """Classify scope compatibility as ``"exact"``, ``"partial"``, or ``"none"``.

    Only filter-declared fields participate (``asset_class``, ``markets``,
    ``universe``, ``horizon``, ``frequency``, ``dataset_family``). A declared
    field whose candidate value is missing is neutral; a present value that
    matches (case-insensitive; markets match on any overlap) counts as a
    match, otherwise as a mismatch. Placeholder values on either side
    (``SCOPE_PLACEHOLDERS``, e.g. ``"mixed"``/``"n/a"``) make the field
    neutral — unspecified never matches or mismatches. ``"exact"`` means
    >= 1 match and no mismatch; ``"partial"`` means >= 1 match with >= 1
    mismatch; anything else is ``"none"`` (including a filter that
    declares no scope fields).
    """
    matches = 0
    mismatches = 0
    compared = 0
    for field_name in _SCOPE_FIELDS:
        wanted = _normalize_scope_value(getattr(scope, field_name))
        if wanted is None or _is_placeholder(wanted):
            continue
        compared += 1
        actual = _normalize_scope_value(candidate_scope.get(field_name))
        if actual is None or _is_placeholder(actual):
            continue
        if actual == wanted:
            matches += 1
        else:
            mismatches += 1
    wanted_markets = {market.strip().lower() for market in scope.markets if isinstance(market, str) and market.strip()}
    if wanted_markets:
        compared += 1
        raw_markets = candidate_scope.get("markets", candidate_scope.get("market"))
        if isinstance(raw_markets, str):
            raw_markets = [raw_markets]
        actual_markets = {str(item).strip().lower() for item in raw_markets} if isinstance(raw_markets, (list, tuple)) else set()
        actual_markets.discard("")
        if actual_markets:
            if actual_markets & wanted_markets:
                matches += 1
            else:
                mismatches += 1
    if compared == 0 or matches == 0:
        return "none"
    if mismatches == 0:
        return "exact"
    return "partial"


def _is_failure(candidate: Candidate) -> bool:
    """Return True when the candidate is failure evidence."""
    if candidate.kind == "failure":
        return True
    if candidate.finding_type == "failure":
        return True
    outcome = candidate.metadata.get("outcome")
    return isinstance(outcome, str) and outcome.strip().lower() == "failure"


def _replication_count(candidate: Candidate) -> int:
    """Return the independent-replication count carried by the candidate."""
    raw = candidate.metadata.get("replication_count")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    if candidate.metadata.get("replicates") is True:
        return 1
    if candidate.metadata.get("replicated_experiment_id"):
        return 1
    return 0


def _hard_filter_reason(candidate: Candidate, scope: ScopeFilter, allowed_project_ids: frozenset[str] | None) -> FilteredCandidate | None:
    """Return the FilteredCandidate when a hard filter drops the candidate, else None."""
    if candidate.status == "rejected":
        return FilteredCandidate(candidate=candidate, reason=REASON_REJECTED, detail="status is rejected")
    if allowed_project_ids is not None and candidate.project_id is not None and candidate.project_id not in allowed_project_ids:
        return FilteredCandidate(candidate=candidate, reason=REASON_ACL, detail=f"project {candidate.project_id} is outside the visible set")
    wanted_asset = _normalize_scope_value(scope.asset_class)
    actual_asset = _normalize_scope_value(candidate.scope.get("asset_class"))
    if wanted_asset is not None and actual_asset is not None and wanted_asset != actual_asset:
        return FilteredCandidate(candidate=candidate, reason=REASON_ASSET_CLASS, detail=f"filter requires asset_class {scope.asset_class!r}, candidate declares {candidate.scope.get('asset_class')!r}")
    if candidate.scope.get("dataset_restricted") is True and not scope.allow_restricted_datasets:
        return FilteredCandidate(candidate=candidate, reason=REASON_DATASET, detail="candidate dataset is restricted and the filter did not opt in")
    required_from = _parse_moment(scope.valid_from)
    required_to = _parse_moment(scope.valid_to)
    if required_from is not None or required_to is not None:
        candidate_from = _parse_moment(candidate.valid_from)
        candidate_to = _parse_moment(candidate.valid_to)
        if candidate_from is not None or candidate_to is not None:
            disjoint = (candidate_to is not None and required_from is not None and candidate_to < required_from) or (candidate_from is not None and required_to is not None and candidate_from > required_to)
            if disjoint:
                return FilteredCandidate(
                    candidate=candidate,
                    reason=REASON_VALIDITY,
                    detail=f"candidate window [{candidate.valid_from}, {candidate.valid_to}] does not overlap required window [{scope.valid_from}, {scope.valid_to}]",
                )
    return None


def apply_hard_filters(
    candidates: Sequence[Candidate],
    scope: ScopeFilter,
    *,
    allowed_project_ids: Sequence[str] | None = None,
) -> tuple[list[Candidate], list[FilteredCandidate]]:
    """Split candidates into survivors and hard-filter drops (documented list, see module docstring).

    Args:
        candidates: Unique candidates to screen (no dedup here; :func:`fuse`
            dedups channel hits before calling this).
        scope: Effective scope filter (validity window, asset class, dataset opt-in).
        allowed_project_ids: When given, candidates scoped to any other
            project are dropped (reason ``acl_project_not_visible``); when
            None, no project enforcement applies.

    Returns:
        ``(survivors, filtered)`` preserving input order in both lists.

    Raises:
        KnowledgeValidationError: When ``allowed_project_ids`` is not a
            sequence of non-empty strings.
    """
    allowed: frozenset[str] | None = None
    if allowed_project_ids is not None:
        if isinstance(allowed_project_ids, (str, bytes)) or not isinstance(allowed_project_ids, Sequence):
            raise KnowledgeValidationError(f"allowed_project_ids must be a sequence of project-id strings, got {type(allowed_project_ids).__name__}.")
        for project_id in allowed_project_ids:
            if not isinstance(project_id, str) or not project_id.strip():
                raise KnowledgeValidationError(f"allowed_project_ids must contain non-empty strings, got {project_id!r}.")
        allowed = frozenset(allowed_project_ids)
    survivors: list[Candidate] = []
    filtered: list[FilteredCandidate] = []
    for candidate in candidates:
        drop = _hard_filter_reason(candidate, scope, allowed)
        if drop is None:
            survivors.append(candidate)
        else:
            filtered.append(drop)
    return survivors, filtered


def _resolve_now(now: datetime | str | None) -> datetime:
    """Resolve the recency reference moment (default: current UTC)."""
    if now is None:
        return datetime.now(UTC)
    parsed = _parse_moment(now)
    if parsed is None:
        raise KnowledgeValidationError(f"now must be an ISO-8601 datetime or datetime instance, got {now!r}.")
    return parsed


def fuse(
    hits: Sequence[ChannelHit],
    scope: ScopeFilter,
    *,
    weights: FusionWeights | None = None,
    allowed_project_ids: Sequence[str] | None = None,
    query_family_hash: str | None = None,
    now: datetime | str | None = None,
    apply_failure_boost: bool = True,
) -> FusionResult:
    """Fuse channel rankings with RRF, then apply research-quality modifiers.

    Pipeline: (1) dedup hits by candidate id, merging channel names (first
    occurrence wins on payload; input order is the documented tie-break);
    (2) hard-filter incompatibilities (see module docstring); (3) RRF over
    original channel ranks; (4) multiplicative modifiers in this fixed
    order — scope, status (validated/reviewed/disputed), replication,
    same-family, recency, near-dup diversity (greedy in pre-penalty score
    order), superseded, failure; (5) final sort by score descending with
    ``candidate.id`` ascending as the deterministic tie-break; ranks are
    1-based over the survivors.

    The failure boost (``MODIFIER_FAILURE``) exists to keep negative
    evidence visible inside the failures packet section. Callers fusing a
    priors/consensus ranking pass ``apply_failure_boost=False`` so boosted
    failures cannot flood the general top-k; the failures section fuses
    separately with the boost on. The ``failure`` marker on
    :class:`FusedCandidate` is unaffected (it always reflects the
    candidate, never the flag).

    Args:
        hits: Channel hits from the planner (rank >= 1 within each channel).
        scope: Effective scope filter for compatibility screening + modifiers.
        weights: Fusion weights (defaults to :class:`FusionWeights`).
        allowed_project_ids: Optional project-visibility enforcement set.
        query_family_hash: Optional 64-char hex experiment-family digest of
            the query; matching candidates earn the same-family boost. None
            disables the boost.
        now: Recency reference moment (ISO-8601 string or datetime; default
            current UTC). Pass an explicit value in tests and evals.
        apply_failure_boost: Apply the failure-visibility boost (defaults
            True). Priors/consensus fusions pass False.

    Returns:
        A :class:`FusionResult` with ranked survivors and the filter audit trail.

    Raises:
        KnowledgeValidationError: On an invalid family hash, ``now`` value,
            or non-bool ``apply_failure_boost``.
    """
    effective_weights = weights or FusionWeights()
    if not isinstance(apply_failure_boost, bool):
        raise KnowledgeValidationError(f"apply_failure_boost must be a bool, got {type(apply_failure_boost).__name__}.")
    if query_family_hash is not None:
        if not isinstance(query_family_hash, str) or len(query_family_hash) != 64:
            raise KnowledgeValidationError(f"query_family_hash must be a 64-character hex digest, got {query_family_hash!r}.")
        try:
            int(query_family_hash, 16)
        except ValueError:
            raise KnowledgeValidationError(f"query_family_hash must be a 64-character hex digest, got {query_family_hash!r}.") from None
        if query_family_hash != query_family_hash.lower():
            raise KnowledgeValidationError(f"query_family_hash must be lowercase hex, got {query_family_hash!r}.")
    reference = _resolve_now(now)

    by_id: dict[str, Candidate] = {}
    channels_by_id: dict[str, list[str]] = {}
    ranks_by_channel: dict[str, dict[str, int]] = {}
    for hit in hits:
        if hit.candidate.id not in by_id:
            by_id[hit.candidate.id] = hit.candidate
            channels_by_id[hit.candidate.id] = []
        if hit.channel not in channels_by_id[hit.candidate.id]:
            channels_by_id[hit.candidate.id].append(hit.channel)
        channel_ranks = ranks_by_channel.setdefault(hit.channel, {})
        if hit.candidate.id not in channel_ranks or hit.rank < channel_ranks[hit.candidate.id]:
            channel_ranks[hit.candidate.id] = hit.rank

    ordered_unique = [by_id[candidate_id] for candidate_id in by_id]
    survivors, filtered = apply_hard_filters(ordered_unique, scope, allowed_project_ids=allowed_project_ids)
    survivor_ids = {candidate.id for candidate in survivors}
    # RRF over ORIGINAL channel ranks: hard-filtered candidates simply stop
    # contributing; survivors keep the rank their channel assigned them.
    rrf_scores: dict[str, float] = {}
    for channel in sorted(ranks_by_channel):
        for candidate_id, rank in ranks_by_channel[channel].items():
            if candidate_id in survivor_ids:
                rrf_scores[candidate_id] = rrf_scores.get(candidate_id, 0.0) + 1.0 / (effective_weights.rrf_k + rank)

    scored: list[FusedCandidate] = []
    for candidate in survivors:
        rrf = rrf_scores.get(candidate.id, 0.0)
        modifiers: dict[str, float] = {}
        score = rrf
        degree = scope_match_degree(candidate.scope, scope)
        if degree == "exact":
            modifiers[MODIFIER_SCOPE_EXACT] = effective_weights.exact_scope_boost
            score *= effective_weights.exact_scope_boost
        elif degree == "partial":
            modifiers[MODIFIER_SCOPE_PARTIAL] = effective_weights.partial_scope_boost
            score *= effective_weights.partial_scope_boost
        if candidate.status == "validated":
            modifiers[MODIFIER_VALIDATED] = effective_weights.validated_boost
            score *= effective_weights.validated_boost
        elif candidate.status == "reviewed":
            modifiers[MODIFIER_REVIEWED] = effective_weights.reviewed_boost
            score *= effective_weights.reviewed_boost
        disputed = candidate.status == "disputed"
        if disputed:
            modifiers[MODIFIER_DISPUTED] = effective_weights.disputed_visibility
            score *= effective_weights.disputed_visibility
        replications = _replication_count(candidate)
        if replications >= effective_weights.strong_replication_threshold:
            modifiers[MODIFIER_STRONG_REPLICATION] = effective_weights.strong_replication_boost
            score *= effective_weights.strong_replication_boost
        elif replications >= 1:
            modifiers[MODIFIER_REPLICATION] = effective_weights.replication_boost
            score *= effective_weights.replication_boost
        if query_family_hash is not None and candidate.family_hash == query_family_hash:
            modifiers[MODIFIER_SAME_FAMILY] = effective_weights.same_family_boost
            score *= effective_weights.same_family_boost
        recorded = _parse_moment(candidate.recorded_at) if candidate.recorded_at else None
        if recorded is not None and recorded <= reference and (reference - recorded).days <= effective_weights.recency_window_days:
            modifiers[MODIFIER_RECENCY] = effective_weights.recency_boost
            score *= effective_weights.recency_boost
        if candidate.status == "superseded":
            modifiers[MODIFIER_SUPERSEDED] = effective_weights.superseded_demote
            score *= effective_weights.superseded_demote
        failure = _is_failure(candidate)
        if failure and apply_failure_boost:
            modifiers[MODIFIER_FAILURE] = effective_weights.failure_boost
            score *= effective_weights.failure_boost
        scored.append(
            FusedCandidate(
                candidate=candidate,
                score=score,
                rrf_score=rrf,
                modifiers=modifiers,
                channels=tuple(sorted(channels_by_id[candidate.id])),
                disputed=disputed,
                failure=failure,
            )
        )

    scored.sort(key=lambda item: (-item.score, item.candidate.id))
    seen_groups: set[str] = set()
    penalized: list[FusedCandidate] = []
    for item in scored:
        group = item.candidate.metadata.get("near_dup_group")
        if isinstance(group, str) and group:
            if group in seen_groups:
                adjusted = dict(item.modifiers)
                adjusted[MODIFIER_NEAR_DUP] = effective_weights.near_dup_penalty
                item = FusedCandidate(
                    candidate=item.candidate,
                    score=item.score * effective_weights.near_dup_penalty,
                    rrf_score=item.rrf_score,
                    modifiers=adjusted,
                    channels=item.channels,
                    disputed=item.disputed,
                    failure=item.failure,
                )
            else:
                seen_groups.add(group)
        penalized.append(item)
    penalized.sort(key=lambda item: (-item.score, item.candidate.id))
    ranked = [
        FusedCandidate(
            candidate=item.candidate,
            score=item.score,
            rrf_score=item.rrf_score,
            modifiers=dict(item.modifiers),
            rank=position + 1,
            channels=item.channels,
            disputed=item.disputed,
            failure=item.failure,
        )
        for position, item in enumerate(penalized)
    ]
    return FusionResult(candidates=ranked, filtered=filtered, weights=effective_weights)

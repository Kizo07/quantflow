"""Run-start knowledge bootstrap for the Research Knowledge Plane (Phase 2).

Implements the compulsory KB ``knowledge_bootstrap`` step: the agent's
task is first converted into a structured :class:`ResearchIntent`
(topic, asset class, markets, universe, horizon, concepts, period,
needed memory), the intent is turned into an explicit retrieval plan
(:func:`plan_retrieval`), and the plan executes against the retrieval,
experiment, artifact and skill interfaces to assemble a fixed-budget
:class:`ContextPacket` — current consensus, closest prior experiments,
relevant failures, known contradictions, important assumptions,
reusable priors/skills and open questions.

Like :mod:`deerflow.knowledge.tools.lookup`, this module performs no
I/O of its own: :func:`knowledge_bootstrap` takes explicit
:class:`BootstrapStores` (the integration step binds real PG there),
while the ``@tool`` wrapper resolves the process-wide bindings from
:mod:`deerflow.knowledge.tools.lookup`. Per-channel failures never fail
the whole bootstrap; they are captured in the packet's
``channel_errors`` so the run can proceed with partial recall and the
agent can retry individual channels mid-run.
"""

import json
import logging
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from langchain.tools import tool

from deerflow.knowledge.hashing import normalize
from deerflow.knowledge.search import ExperimentSearchStore, experiment_search
from deerflow.knowledge.write_api import KnowledgeError, KnowledgeValidationError
from deerflow.runtime.user_context import resolve_runtime_user_id
from deerflow.tools.types import Runtime

from .lookup import (
    MAX_LIMIT,
    ArtifactReadStore,
    KnowledgeRetrievalBackend,
    RetrievalDocument,
    bind_knowledge_backends,
    get_knowledge_backends,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PACKET_ITEMS",
    "DEFAULT_PACKET_CHARS",
    "DEFAULT_CHANNEL_LIMIT",
    "MAX_TOPIC_CHARS",
    "MAX_CONCEPTS",
    "NEEDED_MEMORY_VOCABULARY",
    "DEFAULT_NEEDED_MEMORY",
    "ASSET_CLASSES",
    "ResearchIntent",
    "RetrievalPlanStep",
    "PacketSection",
    "ContextPacket",
    "PacketBudget",
    "SkillCatalog",
    "NullSkillCatalog",
    "BootstrapStores",
    "parse_research_intent",
    "intent_from_text",
    "intent_from_mapping",
    "plan_retrieval",
    "knowledge_bootstrap",
    "format_context_packet",
    "knowledge_bootstrap_tool",
    "get_knowledge_bootstrap_tool",
    "bind_knowledge_backends",
    "bind_knowledge_skills",
    "get_knowledge_skills",
]

#: Default cap on total packet entries across all sections.
DEFAULT_PACKET_ITEMS = 40
#: Default cap on rendered packet characters (summaries are truncated to fit).
DEFAULT_PACKET_CHARS = 12_000
#: Default per-channel retrieval depth behind the packet.
DEFAULT_CHANNEL_LIMIT = 10
#: Maximum topic characters kept from a raw task string.
MAX_TOPIC_CHARS = 500
#: Maximum concepts kept on an intent (frequency order wins).
MAX_CONCEPTS = 20

#: Closed vocabulary of retrievable memory channels (KB research-intent
#: ``needed_memory`` plus ``assumptions``, which the packet always renders).
NEEDED_MEMORY_VOCABULARY = frozenset(
    {
        "validated_findings",
        "prior_experiments",
        "failures",
        "conflicts",
        "assumptions",
        "relevant_skills",
    }
)

#: Channels fetched when the caller does not narrow ``needed_memory``.
DEFAULT_NEEDED_MEMORY = (
    "validated_findings",
    "prior_experiments",
    "failures",
    "conflicts",
    "assumptions",
    "relevant_skills",
)

#: Closed asset-class vocabulary for intent parsing.
ASSET_CLASSES = frozenset({"equity", "fixed_income", "futures", "fx", "crypto", "multi_asset", "unknown"})

_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_HORIZON_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years|d|w|m|y)\b"
    r"(?:\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years|d|w|m|y))?",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")

_ASSET_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("equity", ("equity", "equities", "stock", "stocks", "share", "shares", "etf")),
    ("fixed_income", ("bond", "bonds", "fixed income", "fixed-income", "treasury", "treasuries", "credit", "yield")),
    (
        "futures",
        (
            "future",
            "futures",
            "commodity",
            "commodities",
            "cme",
        ),
    ),
    ("fx", ("fx", "forex", "foreign exchange", "currency", "currencies", "exchange rate")),
    ("crypto", ("crypto", "bitcoin", "ethereum", "digital asset", "token")),
)

_MARKET_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("US", ("us equities", "us equity", "us stocks", "us stock", "us market", "u.s.", "united states", "s&p", "sp500", "nyse", "nasdaq", "russell")),
    ("EU", ("europe", "european", "eurozone", "stoxx", "euronext")),
    ("UK", ("uk", "u.k.", "united kingdom", "london", "ftse")),
    ("JP", ("japan", "japanese", "nikkei", "topix")),
    ("CN", ("china", "chinese", "csi 300", "a-share", "a share")),
    ("EM", ("emerging markets", "emerging market")),
    ("GL", ("global", "worldwide", "all-country", "all country", "developed markets")),
)

_FREQUENCY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tick", ("tick",)),
    ("intraday", ("intraday", "intraday", "minute", "hourly", "5-min", "high-frequency", "high frequency", "hft")),
    ("daily", ("daily",)),
    ("weekly", ("weekly",)),
    ("monthly", ("monthly",)),
    ("quarterly", ("quarterly",)),
    ("annual", ("annual", "yearly")),
)

_CONCEPT_VOCABULARY: tuple[str, ...] = (
    "momentum",
    "reversal",
    "value",
    "quality",
    "carry",
    "volatility",
    "turnover",
    "transaction costs",
    "slippage",
    "market impact",
    "neutralization",
    "sector-neutral",
    "backtest",
    "walk-forward",
    "cross-validation",
    "overfitting",
    "multiple testing",
    "factor",
    "alpha",
    "beta",
    "sharpe",
    "drawdown",
    "liquidity",
    "short interest",
    "earnings",
    "sentiment",
    "machine learning",
    "regression",
    "portfolio construction",
    "risk model",
    "rebalance",
    "point-in-time",
    "survivorship bias",
)


@dataclass(frozen=True)
class ResearchIntent:
    """Structured research intent converted from an agent task (KB §retrieval).

    Attributes:
        topic: Normalized research topic (required, non-empty).
        asset_class: One of :data:`ASSET_CLASSES`.
        markets: Market codes (``US``/``EU``/``UK``/``JP``/``CN``/``EM``/
            ``GL``); empty means "unspecified".
        universe: Tradable-universe description when known.
        horizon: Holding/forecast horizon description when known.
        frequency: Data frequency when known.
        concepts: Key research concepts (frequency order, max
            :data:`MAX_CONCEPTS`).
        requested_period: ``[start, end]`` ISO dates when the task names a
            sample period.
        needed_memory: Channels to retrieve; defaults to
            :data:`DEFAULT_NEEDED_MEMORY`.
    """

    topic: str
    asset_class: str = "unknown"
    markets: list[str] = field(default_factory=list)
    universe: str | None = None
    horizon: str | None = None
    frequency: str | None = None
    concepts: list[str] = field(default_factory=list)
    requested_period: list[str] | None = None
    needed_memory: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the intent."""
        return {
            "topic": self.topic,
            "asset_class": self.asset_class,
            "markets": list(self.markets),
            "universe": self.universe,
            "horizon": self.horizon,
            "frequency": self.frequency,
            "concepts": list(self.concepts),
            "requested_period": list(self.requested_period) if self.requested_period is not None else None,
            "needed_memory": list(self.needed_memory),
        }


@dataclass(frozen=True)
class PacketBudget:
    """Fixed context budget for packet assembly."""

    max_items: int = DEFAULT_PACKET_ITEMS
    max_chars: int = DEFAULT_PACKET_CHARS
    channel_limit: int = DEFAULT_CHANNEL_LIMIT

    def __post_init__(self) -> None:
        if self.max_items < 1 or self.max_chars < 100 or self.channel_limit < 1:
            raise KnowledgeValidationError(f"PacketBudget needs max_items >= 1, max_chars >= 100 and channel_limit >= 1, got {self!r}.")
        if self.channel_limit > MAX_LIMIT:
            raise KnowledgeValidationError(f"PacketBudget channel_limit {self.channel_limit} exceeds MAX_LIMIT={MAX_LIMIT}.")


@dataclass(frozen=True)
class RetrievalPlanStep:
    """One planned retrieval channel behind the context packet."""

    channel: str
    query: str
    kinds: tuple[str, ...] = ()
    filters: dict[str, str] = field(default_factory=dict)
    limit: int = DEFAULT_CHANNEL_LIMIT

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the plan step."""
        return {
            "channel": self.channel,
            "query": self.query,
            "kinds": list(self.kinds),
            "filters": dict(self.filters),
            "limit": self.limit,
        }


@dataclass(frozen=True)
class PacketSection:
    """One rendered packet section (entries already budget-trimmed)."""

    name: str
    title: str
    entries: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the section."""
        return {
            "name": self.name,
            "title": self.title,
            "entries": [dict(entry) for entry in self.entries],
            "count": len(self.entries),
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class ContextPacket:
    """Fixed-budget research context packet for agent consumption."""

    intent: ResearchIntent
    sections: list[PacketSection] = field(default_factory=list)
    channel_errors: list[dict[str, str]] = field(default_factory=list)
    generated_at: str = ""
    budget: PacketBudget = field(default_factory=PacketBudget)
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the packet."""
        return {
            "intent": self.intent.to_dict(),
            "sections": [section.to_dict() for section in self.sections],
            "channel_errors": [dict(entry) for entry in self.channel_errors],
            "generated_at": self.generated_at,
            "budget": {"max_items": self.budget.max_items, "max_chars": self.budget.max_chars, "channel_limit": self.budget.channel_limit},
            "truncated": self.truncated,
            "text": format_context_packet(self),
        }


@runtime_checkable
class SkillCatalog(Protocol):
    """Read boundary for reusable skills/priors; integration binds the skills index.

    PG/skills binding sketch: match ``concepts`` against skill metadata
    (name, description, tags) with the same lexical channel used for KB
    documents; return installed skill versions newest first.
    """

    def find_skills(self, concepts: Sequence[str], *, limit: int) -> list[dict[str, Any]]:
        """Return up to ``limit`` JSON-safe skill dicts for ``concepts`` (may be empty)."""
        ...


class NullSkillCatalog:
    """Fallback skill catalog reporting no skills (used until integration binds)."""

    def find_skills(self, concepts: Sequence[str], *, limit: int) -> list[dict[str, Any]]:
        """Return an empty skill list (no catalog bound)."""
        return []


@dataclass(frozen=True)
class BootstrapStores:
    """Interface bundle behind :func:`knowledge_bootstrap`.

    At least one of ``retrieval``/``experiments`` must be present; channels
    whose backend is missing are recorded in ``channel_errors`` instead of
    failing the packet. ``skills`` defaults to :class:`NullSkillCatalog`;
    ``artifacts`` is currently unused by the packet (locators resolve
    lazily via ``artifact_manifest``/``artifact_open``) and is carried so
    the middleware can pass one bundle through.
    """

    retrieval: KnowledgeRetrievalBackend | None = None
    experiments: ExperimentSearchStore | None = None
    artifacts: ArtifactReadStore | None = None
    skills: SkillCatalog | None = None

    def effective_skills(self) -> SkillCatalog:
        """Return the bound skill catalog, or a null catalog when unbound."""
        return self.skills if self.skills is not None else NullSkillCatalog()


def _normalize_text(text: str) -> str:
    """Collapse whitespace and NFC-normalize free text."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def _require_topic(topic: Any) -> str:
    """Validate a required intent topic."""
    if not isinstance(topic, str):
        raise KnowledgeValidationError(f"topic must be a string, got {type(topic).__name__}.")
    normalized = _normalize_text(topic)
    if not normalized:
        raise KnowledgeValidationError("topic must be a non-empty string.")
    if len(normalized) > MAX_TOPIC_CHARS:
        normalized = normalized[:MAX_TOPIC_CHARS].rstrip()
    return normalized


def _optional_text(name: str, value: Any, *, max_len: int = 500) -> str | None:
    """Validate an optional short text field (None passes through)."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise KnowledgeValidationError(f"{name} must be a string or null, got {type(value).__name__}.")
    normalized = _normalize_text(value)
    if not normalized:
        return None
    if len(normalized) > max_len:
        raise KnowledgeValidationError(f"{name} exceeds {max_len} characters ({len(normalized)}).")
    return normalized


def _keyword_pattern(keyword: str) -> str:
    """Build a case-insensitive word-boundary regex for one keyword.

    ``(?<!\\w)`` / ``(?!\\w)`` lookarounds (instead of ``\\b``) keep
    dotted tokens like ``u.s.`` matchable while still rejecting
    mid-word hits (``us`` in ``versus``, ``share`` in ``shared``).
    """
    return r"(?<!\w)" + re.escape(keyword.strip().lower()) + r"(?!\w)"


def _keyword_hits(text: str, keywords: tuple[str, ...]) -> bool:
    """Return True when any keyword appears in ``text`` (case-insensitive, word-boundary)."""
    lowered = text.lower()
    return any(re.search(_keyword_pattern(keyword), lowered) for keyword in keywords)


def _keyword_count(text: str, keyword: str) -> int:
    """Count word-boundary occurrences of ``keyword`` in ``text`` (case-insensitive)."""
    return len(re.findall(_keyword_pattern(keyword), text.lower()))


def _detect_asset_class(text: str) -> str:
    """Detect the asset class from task keywords (first match wins, else unknown)."""
    hits = [name for name, keywords in _ASSET_KEYWORDS if _keyword_hits(text, keywords)]
    if len(hits) > 1:
        return "multi_asset"
    return hits[0] if hits else "unknown"


def _detect_markets(text: str) -> list[str]:
    """Detect market codes from task keywords (stable order)."""
    return [code for code, keywords in _MARKET_KEYWORDS if _keyword_hits(text, keywords)]


def _detect_frequency(text: str) -> str | None:
    """Detect the data frequency from task keywords (first match wins)."""
    for name, keywords in _FREQUENCY_KEYWORDS:
        if _keyword_hits(text, keywords):
            return name
    return None


def _detect_concepts(text: str) -> list[str]:
    """Scan the task for known research concepts (frequency order, capped)."""
    scored: list[tuple[int, str]] = []
    for concept in _CONCEPT_VOCABULARY:
        count = _keyword_count(text, concept)
        if count:
            scored.append((-count, concept))
    scored.sort()
    return [concept for _, concept in scored[:MAX_CONCEPTS]]


def _hypothesis_needles(intent: ResearchIntent) -> list[str]:
    """Pick PG-substring needles for experiment recall from an intent.

    The whole topic is usually too specific for an ``ILIKE '%...%'``
    lookup (word order differs between runs), and a single needle misses
    hypotheses phrased around any other concept — so callers union one
    lookup per needle: detected concepts longest-first (most distinctive
    phrases), plus the topic's first five words as a phrase fallback.
    """
    needles: list[str] = []
    for concept in sorted(intent.concepts, key=len, reverse=True):
        term = concept.strip()[:200]
        if len(term) >= 3 and term not in needles:
            needles.append(term)
    words = intent.topic.split()
    phrase = " ".join(words[:5]).strip()
    if len(phrase) >= 3 and phrase not in needles:
        needles.append(phrase[:200])
    if not needles:
        needles.append(intent.topic[:200])
    return needles


def _union_needle_search(store: Any, needles: Sequence[str], *, limit: int, **filters: Any) -> list[Any]:
    """Union one ``experiment_search`` per needle, deduped and ranked.

    Records matching more needles rank first (descending hit count),
    ties keep first-seen order, and the union trims to ``limit``. Every
    needle is a non-empty substring by construction.
    """
    hits: dict[str, list[Any]] = {}
    order = 0
    for needle in needles:
        page = experiment_search(store, hypothesis_contains=needle, limit=limit, offset=0, **filters)
        for record in page.experiments:
            key = str(getattr(record, "id", order))
            if key in hits:
                hits[key][1] += 1
            else:
                hits[key] = [record, 1, order]
                order += 1
    ranked = sorted(hits.values(), key=lambda entry: (-entry[1], entry[2]))
    return [record for record, _, _ in ranked[:limit]]


def _detect_period(text: str) -> list[str] | None:
    """Detect a requested sample period from ISO dates or year ranges."""
    dates = sorted(set(_DATE_RE.findall(text)))
    if len(dates) >= 2:
        return [dates[0], dates[-1]]
    if len(dates) == 1:
        return [dates[0], dates[0]]
    years = sorted(set(_YEAR_RE.findall(text)))
    if len(years) >= 2:
        return [f"{years[0]}-01-01", f"{years[-1]}-12-31"]
    return None


def _detect_horizon(text: str) -> str | None:
    """Detect a holding/forecast horizon span from the task text."""
    match = _HORIZON_RE.search(text)
    if match is None:
        return None
    first, first_unit, second, second_unit = match.group(1), match.group(2), match.group(3), match.group(4)
    if second is None:
        return f"{first} {first_unit}"
    return f"{first} {first_unit} to {second} {second_unit}"


def intent_from_text(task: str, *, needed_memory: Sequence[str] | None = None) -> ResearchIntent:
    """Convert a raw task string into a structured research intent.

    The conversion is deterministic keyword extraction (documented
    heuristic, no model calls): topic from the task text, asset class /
    markets / frequency / concepts from closed vocabularies, horizon from
    duration spans, and the requested period from ISO dates or year
    ranges. Callers with structured context should prefer
    :func:`intent_from_mapping`.

    Args:
        task: Raw agent task text (non-empty).
        needed_memory: Optional channel subset; defaults to
            :data:`DEFAULT_NEEDED_MEMORY`.

    Raises:
        KnowledgeValidationError: On empty tasks or unknown channels.
    """
    if not isinstance(task, str):
        raise KnowledgeValidationError(f"task must be a string, got {type(task).__name__}.")
    topic = _require_topic(task)
    universe = "liquid common stocks" if _detect_asset_class(topic) == "equity" else None
    return ResearchIntent(
        topic=topic,
        asset_class=_detect_asset_class(topic),
        markets=_detect_markets(topic),
        universe=universe,
        horizon=_detect_horizon(topic),
        frequency=_detect_frequency(topic),
        concepts=_detect_concepts(topic),
        requested_period=_detect_period(topic),
        needed_memory=_require_needed_memory(needed_memory),
    )


def _require_needed_memory(needed_memory: Any) -> list[str]:
    """Validate a needed_memory channel list (None becomes the default)."""
    if needed_memory is None:
        return list(DEFAULT_NEEDED_MEMORY)
    if isinstance(needed_memory, str):
        needed_memory = [part.strip() for part in needed_memory.split(",")]
    if not isinstance(needed_memory, Sequence) or isinstance(needed_memory, (bytes, bytearray)):
        raise KnowledgeValidationError(f"needed_memory must be a sequence of channel names, got {type(needed_memory).__name__}.")
    normalized = [str(channel).strip() for channel in needed_memory if str(channel).strip()]
    if not normalized:
        return list(DEFAULT_NEEDED_MEMORY)
    unknown = [channel for channel in normalized if channel not in NEEDED_MEMORY_VOCABULARY]
    if unknown:
        raise KnowledgeValidationError(f"unknown needed_memory channels {unknown}; expected a subset of {sorted(NEEDED_MEMORY_VOCABULARY)}.")
    return list(dict.fromkeys(normalized))


def _require_markets(markets: Any) -> list[str]:
    """Validate an optional market-code list (None becomes [])."""
    if markets is None:
        return []
    if isinstance(markets, str):
        markets = [part.strip() for part in markets.split(",")]
    if not isinstance(markets, Sequence) or isinstance(markets, (bytes, bytearray)):
        raise KnowledgeValidationError(f"markets must be a sequence of market codes, got {type(markets).__name__}.")
    allowed = {code for code, _ in _MARKET_KEYWORDS}
    normalized = [str(code).strip().upper() for code in markets if str(code).strip()]
    unknown = [code for code in normalized if code not in allowed]
    if unknown:
        raise KnowledgeValidationError(f"unknown market codes {unknown}; expected a subset of {sorted(allowed)}.")
    return list(dict.fromkeys(normalized))


def _require_period(period: Any) -> list[str] | None:
    """Validate an optional [start, end] ISO-date period (None passes through)."""
    if period is None:
        return None
    if not isinstance(period, Sequence) or isinstance(period, (str, bytes, bytearray)) or len(period) != 2:
        raise KnowledgeValidationError(f"requested_period must be a [start, end] pair of YYYY-MM-DD dates, got {period!r}.")
    start, end = str(period[0]).strip(), str(period[1]).strip()
    if not _DATE_RE.fullmatch(start) or not _DATE_RE.fullmatch(end):
        raise KnowledgeValidationError(f"requested_period must be a [start, end] pair of YYYY-MM-DD dates, got {period!r}.")
    if start > end:
        raise KnowledgeValidationError(f"requested_period start {start!r} is after end {end!r}.")
    return [start, end]


def intent_from_mapping(data: Mapping[str, Any]) -> ResearchIntent:
    """Build a validated research intent from a structured mapping.

    Accepts the KB research-intent shape (``topic`` plus optional
    ``asset_class``/``markets``/``universe``/``horizon``/``frequency``/
    ``concepts``/``requested_period``/``needed_memory``). Unknown keys are
    rejected so misspelled fields fail loudly instead of being dropped.

    Raises:
        KnowledgeValidationError: On missing topics, unknown keys, or any
            field outside its vocabulary.
    """
    if not isinstance(data, Mapping):
        raise KnowledgeValidationError(f"research intent mapping must be a mapping, got {type(data).__name__}.")
    allowed = {"topic", "asset_class", "markets", "universe", "horizon", "frequency", "concepts", "requested_period", "needed_memory"}
    unknown = [key for key in data if key not in allowed]
    if unknown:
        raise KnowledgeValidationError(f"unknown research-intent keys {sorted(str(k) for k in unknown)}; expected a subset of {sorted(allowed)}.")
    if "topic" not in data:
        raise KnowledgeValidationError("research intent mapping requires a 'topic' field.")
    topic = _require_topic(data["topic"])
    asset_class = data.get("asset_class", "unknown")
    if not isinstance(asset_class, str) or asset_class.strip().lower() not in ASSET_CLASSES:
        raise KnowledgeValidationError(f"asset_class must be one of {sorted(ASSET_CLASSES)}, got {asset_class!r}.")
    concepts = data.get("concepts")
    if concepts is None:
        concept_list: list[str] = []
    elif isinstance(concepts, str):
        concept_list = [_normalize_text(concepts)] if _normalize_text(concepts) else []
    elif isinstance(concepts, Sequence) and not isinstance(concepts, (bytes, bytearray)):
        concept_list = []
        for concept in concepts:
            if not isinstance(concept, str) or not _normalize_text(concept):
                raise KnowledgeValidationError(f"concepts must be non-empty strings, got {concept!r}.")
            concept_list.append(_normalize_text(concept)[:200])
    else:
        raise KnowledgeValidationError(f"concepts must be a sequence of strings, got {type(concepts).__name__}.")
    return ResearchIntent(
        topic=topic,
        asset_class=asset_class.strip().lower(),
        markets=_require_markets(data.get("markets")),
        universe=_optional_text("universe", data.get("universe")),
        horizon=_optional_text("horizon", data.get("horizon")),
        frequency=_optional_text("frequency", data.get("frequency"), max_len=100),
        concepts=list(dict.fromkeys(concept_list))[:MAX_CONCEPTS],
        requested_period=_require_period(data.get("requested_period")),
        needed_memory=_require_needed_memory(data.get("needed_memory")),
    )


def parse_research_intent(research_intent: str | Mapping[str, Any] | ResearchIntent) -> ResearchIntent:
    """Convert task input into a validated :class:`ResearchIntent`.

    Accepts a ready-made intent (returned as-is), a structured mapping
    (:func:`intent_from_mapping`), or a raw task string
    (:func:`intent_from_text`).
    """
    if isinstance(research_intent, ResearchIntent):
        return research_intent
    if isinstance(research_intent, Mapping):
        return intent_from_mapping(research_intent)
    if isinstance(research_intent, str):
        return intent_from_text(research_intent)
    raise KnowledgeValidationError(f"research_intent must be a task string, mapping, or ResearchIntent, got {type(research_intent).__name__}.")


def plan_retrieval(intent: ResearchIntent, *, budget: PacketBudget | None = None) -> list[RetrievalPlanStep]:
    """Turn an intent into an explicit per-channel retrieval plan.

    Each needed-memory channel becomes one step with its query text, kind
    scope, structured filters and depth. ``prior_experiments`` and
    ``failures`` carry experiment-store hints (``kinds == ("experiment",)``
    with the experiment channel flag in ``filters["_store"]``) so the
    executor routes them to :func:`experiment_search
    <deerflow.knowledge.search.experiment_search>` instead of the document
    backend; ``relevant_skills`` carries the skill-catalog flag.

    Args:
        intent: Validated research intent.
        budget: Packet budget (drives per-step ``limit``).

    Returns:
        Plan steps in ``needed_memory`` order.
    """
    limits = budget or PacketBudget()
    query_bits = [intent.topic, *intent.concepts[:5]]
    if intent.universe:
        query_bits.append(intent.universe)
    query = _normalize_text(" ".join(query_bits))
    scope_filters: dict[str, str] = {}
    if intent.asset_class != "unknown":
        scope_filters["asset_class"] = intent.asset_class
    if intent.markets:
        scope_filters["market"] = intent.markets[0]
    if intent.universe:
        scope_filters["universe"] = intent.universe
    if intent.horizon:
        scope_filters["horizon"] = intent.horizon
    if intent.requested_period:
        scope_filters["period_start"] = intent.requested_period[0]
        scope_filters["period_end"] = intent.requested_period[1]
    steps: list[RetrievalPlanStep] = []
    for channel in intent.needed_memory:
        if channel == "validated_findings":
            # No hard status filter: fusion already ranks validated above
            # reviewed above candidate, and a validated-only filter would
            # empty the consensus section until Phase 3 validation exists
            # (Phase 2 writes candidates only). Statuses stay visible on
            # every entry so agents can weigh them.
            steps.append(RetrievalPlanStep(channel=channel, query=query, kinds=("finding",), filters=dict(scope_filters), limit=limits.channel_limit))
        elif channel == "prior_experiments":
            steps.append(RetrievalPlanStep(channel=channel, query=query, kinds=("experiment",), filters={"_store": "experiments"}, limit=limits.channel_limit))
        elif channel == "failures":
            steps.append(RetrievalPlanStep(channel=channel, query=query, kinds=("failure", "experiment"), filters={"_store": "experiments+retrieval"}, limit=limits.channel_limit))
        elif channel == "conflicts":
            steps.append(RetrievalPlanStep(channel=channel, query=query, kinds=("conflict",), filters=dict(scope_filters), limit=limits.channel_limit))
        elif channel == "assumptions":
            steps.append(RetrievalPlanStep(channel=channel, query=query, kinds=("assumption",), filters=dict(scope_filters), limit=limits.channel_limit))
        elif channel == "relevant_skills":
            steps.append(RetrievalPlanStep(channel=channel, query=", ".join(intent.concepts[:8]) or intent.topic, kinds=("skill",), filters={"_store": "skills"}, limit=min(limits.channel_limit, 5)))
    return steps


def _doc_entry(doc: RetrievalDocument) -> dict[str, Any]:
    """Render a retrieval hit as a compact packet entry."""
    return {
        "id": doc.id,
        "kind": doc.kind,
        "title": doc.title,
        "summary": doc.summary,
        "score": doc.score,
        "status": doc.status,
        "evidence_refs": list(doc.evidence_refs),
    }


def _experiment_entry(record: Any) -> dict[str, Any]:
    """Render an experiment record as a compact packet entry."""
    to_dict = getattr(record, "to_dict", None)
    payload = to_dict() if callable(to_dict) else dict(record)
    methodology = payload.get("methodology") or {}
    return {
        "id": payload.get("id"),
        "kind": "experiment",
        "title": str(payload.get("hypothesis", ""))[:300],
        "summary": f"status={payload.get('status')} outcome={payload.get('outcome')} family={str(payload.get('family_hash', ''))[:12]}",
        "score": None,
        "status": payload.get("status"),
        "evidence_refs": [*(payload.get("result_artifacts") or []), *(str(link.get("dataset_version_id")) for link in (payload.get("datasets") or []) if isinstance(link, dict))],
        "methodology": methodology if isinstance(methodology, dict) else {},
    }


def _run_validated_findings(step: RetrievalPlanStep, stores: BootstrapStores) -> list[dict[str, Any]]:
    """Execute the validated-findings channel over the retrieval backend."""
    assert stores.retrieval is not None
    page = stores.retrieval.search(step.query, kinds=("finding",), filters={k: v for k, v in step.filters.items() if not k.startswith("_")}, limit=step.limit, offset=0)
    return [_doc_entry(doc) for doc in page.documents]


def _experiment_design(intent: ResearchIntent) -> dict[str, Any]:
    """Build a same-family experiment design mapping from the intent."""
    methodology: dict[str, Any] = {}
    if intent.universe:
        methodology["universe"] = intent.universe
    if intent.frequency:
        methodology["frequency"] = intent.frequency
    if intent.horizon:
        methodology["horizon"] = intent.horizon
    if intent.requested_period:
        methodology["sample_period"] = list(intent.requested_period)
    if intent.concepts:
        methodology["concepts"] = list(intent.concepts[:8])
    return {"hypothesis": intent.topic, "methodology": methodology}


def _run_prior_experiments(intent: ResearchIntent, step: RetrievalPlanStep, stores: BootstrapStores) -> list[dict[str, Any]]:
    """Execute the prior-experiments channel (family lookup, substring fallback)."""
    assert stores.experiments is not None
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        page = experiment_search(stores.experiments, spec=_experiment_design(intent), limit=step.limit, offset=0)
        for record in page.experiments:
            entry = _experiment_entry(record)
            if entry["id"] not in seen:
                seen.add(entry["id"])  # type: ignore[arg-type]
                entries.append(entry)
    except KnowledgeValidationError:
        pass  # Fall through to the substring fallback below.
    if not entries:
        for record in _union_needle_search(stores.experiments, _hypothesis_needles(intent), limit=step.limit):
            entry = _experiment_entry(record)
            if entry["id"] not in seen:
                seen.add(entry["id"])  # type: ignore[arg-type]
                entries.append(entry)
    return entries


def _run_failures(intent: ResearchIntent, step: RetrievalPlanStep, stores: BootstrapStores) -> list[dict[str, Any]]:
    """Execute the dedicated failure channel (experiments + failure docs).

    The two legs are independent: when one leg raises but the other
    already produced entries, the partial set is returned (and the
    outage is logged) so a retrieval blip cannot hide known experiment
    failures; when nothing was retrieved the error propagates so the
    packet records a channel gap.
    """
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    if stores.experiments is not None:
        for record in _union_needle_search(stores.experiments, _hypothesis_needles(intent), limit=step.limit, outcome="failure"):
            entry = _experiment_entry(record)
            entry["kind"] = "failure"
            if entry["id"] not in seen:
                seen.add(entry["id"])  # type: ignore[arg-type]
                entries.append(entry)
    if stores.retrieval is not None:
        try:
            page = stores.retrieval.search(step.query, kinds=("failure",), filters={}, limit=step.limit, offset=0)
        except Exception:
            if entries:
                logger.warning("knowledge_bootstrap failures channel: retrieval leg failed; returning %d experiment failure(s)", len(entries), exc_info=True)
                return entries[: step.limit]
            raise
        for doc in page.documents:
            if doc.id not in seen:
                seen.add(doc.id)
                entries.append(_doc_entry(doc))
    return entries[: step.limit]


def _run_retrieval_channel(step: RetrievalPlanStep, stores: BootstrapStores, kinds: tuple[str, ...]) -> list[dict[str, Any]]:
    """Execute a plain document channel (conflicts / assumptions)."""
    assert stores.retrieval is not None
    page = stores.retrieval.search(step.query, kinds=kinds, filters={k: v for k, v in step.filters.items() if not k.startswith("_")}, limit=step.limit, offset=0)
    return [_doc_entry(doc) for doc in page.documents]


def _run_skills(intent: ResearchIntent, step: RetrievalPlanStep, stores: BootstrapStores) -> list[dict[str, Any]]:
    """Execute the reusable-skills channel over the skill catalog."""
    raw = stores.effective_skills().find_skills(intent.concepts or [intent.topic], limit=step.limit)
    entries: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        entries.append(
            {
                "id": str(item.get("id", item.get("name", "skill"))),
                "kind": "skill",
                "title": str(item.get("name", item.get("id", "skill"))),
                "summary": str(item.get("description", item.get("summary", "")))[:500],
                "score": item.get("score"),
                "status": str(item.get("version", item.get("status", "")) or ""),
                "evidence_refs": [],
            }
        )
    return entries


_SECTION_TITLES = {
    "validated_findings": "Current consensus",
    "prior_experiments": "Closest prior experiments",
    "failures": "Relevant failures / negative evidence",
    "conflicts": "Known contradictions",
    "assumptions": "Important assumptions",
    "relevant_skills": "Reusable priors / skills",
}


def _execute_step(intent: ResearchIntent, step: RetrievalPlanStep, stores: BootstrapStores) -> list[dict[str, Any]]:
    """Route one plan step to its channel implementation."""
    if step.channel == "validated_findings":
        if stores.retrieval is None:
            raise KnowledgeError("retrieval backend is not bound")
        return _run_validated_findings(step, stores)
    if step.channel == "prior_experiments":
        if stores.experiments is None:
            raise KnowledgeError("experiment store is not bound")
        return _run_prior_experiments(intent, step, stores)
    if step.channel == "failures":
        if stores.retrieval is None and stores.experiments is None:
            raise KnowledgeError("neither retrieval backend nor experiment store is bound")
        return _run_failures(intent, step, stores)
    if step.channel == "conflicts":
        if stores.retrieval is None:
            raise KnowledgeError("retrieval backend is not bound")
        return _run_retrieval_channel(step, stores, ("conflict",))
    if step.channel == "assumptions":
        if stores.retrieval is None:
            raise KnowledgeError("retrieval backend is not bound")
        return _run_retrieval_channel(step, stores, ("assumption",))
    if step.channel == "relevant_skills":
        return _run_skills(intent, step, stores)
    raise KnowledgeValidationError(f"unknown retrieval channel {step.channel!r}.")


def _trim_text(text: str, budget: int) -> tuple[str, bool]:
    """Trim ``text`` to ``budget`` chars with an ellipsis marker."""
    if len(text) <= budget:
        return text, False
    if budget <= 1:
        return "…", True
    return text[: budget - 1].rstrip() + "…", True


def knowledge_bootstrap(
    research_intent: str | Mapping[str, Any] | ResearchIntent,
    *,
    stores: BootstrapStores,
    budget: PacketBudget | None = None,
) -> dict[str, Any]:
    """Build the fixed-budget research context packet for a task.

    Converts ``research_intent`` to a :class:`ResearchIntent`, plans
    retrieval (:func:`plan_retrieval`), executes every channel, and
    assembles the KB packet sections (consensus, prior experiments,
    failures, contradictions, assumptions, skills) plus derived open
    questions. Totals are capped at ``budget.max_items`` entries and
    ``budget.max_chars`` rendered characters; overflow sets section and
    packet ``truncated`` flags instead of silently dropping context.

    Args:
        research_intent: Raw task string, KB intent mapping, or ready-made
            :class:`ResearchIntent`.
        stores: Interface bundle (integration binds real PG + skills).
        budget: Packet budget; defaults to :class:`PacketBudget`.

    Returns:
        JSON-safe context packet dict (see :class:`ContextPacket`).

    Raises:
        KnowledgeValidationError: On invalid intents or budgets, or when
            ``stores`` binds neither retrieval nor experiments.
    """
    intent = parse_research_intent(research_intent)
    limits = budget or PacketBudget()
    if not isinstance(limits, PacketBudget):
        raise KnowledgeValidationError(f"budget must be a PacketBudget, got {type(limits).__name__}.")
    if stores.retrieval is None and stores.experiments is None:
        raise KnowledgeValidationError("knowledge_bootstrap needs at least one of retrieval/experiments bound in stores.")
    plan = plan_retrieval(intent, budget=limits)
    sections: list[PacketSection] = []
    channel_errors: list[dict[str, str]] = []
    remaining_items = limits.max_items
    remaining_chars = limits.max_chars
    packet_truncated = False
    for step in plan:
        try:
            entries = _execute_step(intent, step, stores)
        except NotImplementedError as exc:
            channel_errors.append({"channel": step.channel, "error": f"backend does not support this operation: {exc}"})
            entries = []
        except KnowledgeError as exc:
            channel_errors.append({"channel": step.channel, "error": str(exc)})
            entries = []
        except Exception as exc:  # noqa: BLE001 — one bad channel must not fail the packet
            logger.exception("knowledge_bootstrap channel %s failed", step.channel)
            channel_errors.append({"channel": step.channel, "error": str(exc)})
            entries = []
        normalized = [normalize(entry) for entry in entries]
        assert all(isinstance(entry, dict) for entry in normalized)
        kept: list[dict[str, Any]] = []
        section_truncated = False
        for entry in normalized:
            if remaining_items <= 0:
                section_truncated = True
                packet_truncated = True
                break
            summary = str(entry.get("summary", ""))
            title = str(entry.get("title", ""))
            cost = len(title) + len(summary) + len(str(entry.get("id", ""))) + 16
            if cost > remaining_chars:
                if remaining_chars > 64:
                    entry["summary"], _ = _trim_text(summary, max(0, remaining_chars - len(title) - 32))
                    entry["title"], _ = _trim_text(title, 200)
                    kept.append(entry)
                    remaining_items -= 1
                    remaining_chars = 0
                section_truncated = True
                packet_truncated = True
                break
            kept.append(entry)
            remaining_items -= 1
            remaining_chars -= cost
        if len(normalized) > len(kept) and remaining_items <= 0:
            section_truncated = True
            packet_truncated = True
        sections.append(PacketSection(name=step.channel, title=_SECTION_TITLES[step.channel], entries=kept, truncated=section_truncated))
    open_questions = _derive_open_questions(sections, channel_errors)
    if open_questions and remaining_items > 0:
        sections.append(PacketSection(name="open_questions", title="Open questions", entries=open_questions[:remaining_items], truncated=len(open_questions) > remaining_items))
    packet = ContextPacket(
        intent=intent,
        sections=sections,
        channel_errors=channel_errors,
        generated_at=datetime.now(UTC).isoformat(),
        budget=limits,
        truncated=packet_truncated,
    )
    return packet.to_dict()


def _derive_open_questions(sections: Sequence[PacketSection], channel_errors: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Derive open-question pointers from conflicts and channel gaps."""
    questions: list[dict[str, Any]] = []
    for section in sections:
        if section.name == "conflicts":
            for entry in section.entries:
                questions.append(
                    {
                        "id": f"oq:{entry.get('id', 'conflict')}",
                        "kind": "open_question",
                        "title": f"Resolve contradiction {entry.get('id', '')}",
                        "summary": str(entry.get("summary", entry.get("title", "")))[:300],
                        "score": None,
                        "status": "open",
                        "evidence_refs": [str(entry.get("id", ""))],
                    }
                )
    for failure in channel_errors:
        questions.append(
            {
                "id": f"oq:channel-{failure.get('channel', 'unknown')}",
                "kind": "open_question",
                "title": f"Channel unavailable: {failure.get('channel', 'unknown')}",
                "summary": f"Recall for this channel failed ({failure.get('error', 'unknown error')}); verify manually before relying on packet completeness.",
                "score": None,
                "status": "open",
                "evidence_refs": [],
            }
        )
    return questions


def format_context_packet(packet: ContextPacket | Mapping[str, Any]) -> str:
    """Render a packet (or its ``to_dict`` form) as agent-facing text.

    Follows the KB "research context packet" layout: one titled block per
    section with ``- <id>: <title>`` lines, plus truncation and channel
    error notes so the model never mistakes a partial packet for complete
    recall.
    """
    if isinstance(packet, Mapping):
        sections = packet.get("sections", [])
        intent = packet.get("intent", {})
        channel_errors = packet.get("channel_errors", [])
        truncated = packet.get("truncated", False)
        topic = intent.get("topic", "") if isinstance(intent, Mapping) else ""
    else:
        sections = [section.to_dict() for section in packet.sections]
        topic = packet.intent.topic
        channel_errors = [dict(entry) for entry in packet.channel_errors]
        truncated = packet.truncated
    lines = ["RESEARCH CONTEXT", f"topic: {topic}", ""]
    for section in sections:
        title = section.get("title", section.get("name", ""))
        lines.append(str(title))
        entries = section.get("entries", [])
        if not entries:
            lines.append("  - (none retrieved)")
        for entry in entries:
            entry_id = entry.get("id", "?")
            title_text = str(entry.get("title", "") or entry.get("summary", "")).strip().replace("\n", " ")
            status = entry.get("status")
            suffix = f" [{status}]" if status else ""
            lines.append(f"  - {entry_id}: {title_text[:300]}{suffix}")
        if section.get("truncated"):
            lines.append("  (section truncated to budget; narrow the query or page per-channel tools)")
        lines.append("")
    if channel_errors:
        lines.append("Channel gaps (verify manually):")
        for failure in channel_errors:
            lines.append(f"  - {failure.get('channel')}: {failure.get('error', '')[:200]}")
        lines.append("")
    if truncated:
        lines.append("(packet truncated to budget; use ledger_search/experiment_get for depth)")
    return "\n".join(lines).rstrip() + "\n"


_skill_catalog: SkillCatalog = NullSkillCatalog()


def bind_knowledge_skills(catalog: SkillCatalog | None) -> SkillCatalog:
    """Register the process-wide skill catalog for the bootstrap ``@tool`` wrapper.

    Args:
        catalog: Skill catalog implementation (integration binds the
            skills index; tests bind fakes). ``None`` restores the null
            catalog.

    Returns:
        The registered catalog.
    """
    global _skill_catalog
    _skill_catalog = catalog if catalog is not None else NullSkillCatalog()
    return _skill_catalog


def get_knowledge_skills() -> SkillCatalog:
    """Return the current process-wide skill catalog (null catalog by default)."""
    return _skill_catalog


def _resolve_scope(runtime: Runtime | None = None) -> tuple[str | None, str]:
    """Resolve agent_name and user_id for tool handler scope (cf. memory/tools.py)."""
    context = getattr(runtime, "context", None)
    agent_name = None
    if isinstance(context, dict) and context.get("agent_name"):
        agent_name = str(context["agent_name"])
    return agent_name, resolve_runtime_user_id(runtime)


@tool("knowledge_bootstrap", parse_docstring=True)
def knowledge_bootstrap_tool(
    runtime: Runtime,
    research_intent: str,
    max_items: int = DEFAULT_PACKET_ITEMS,
    max_chars: int = DEFAULT_PACKET_CHARS,
) -> str:
    """Build the run-start research context packet (consensus, priors, failures, conflicts).

    Call this before substantive research: it converts the task into a
    structured research intent, retrieves validated findings, prior
    experiments, failures, contradictions, assumptions and reusable
    skills, and returns one fixed-budget packet.

    Args:
        research_intent: Raw task text, or a JSON object with the
            research-intent shape (topic plus optional asset_class,
            markets, universe, horizon, frequency, concepts,
            requested_period, needed_memory).
        max_items: Packet entry cap (default 40).
        max_chars: Rendered packet character cap (default 12000).

    Returns:
        JSON string context packet ("intent", "sections", "channel_errors",
        "text") — or "error".
    """
    _agent_name, _user_id = _resolve_scope(runtime)
    try:
        bindings = get_knowledge_backends()
        if bindings.retrieval is None and bindings.experiments is None:
            return json.dumps({"error": "knowledge backends are not bound (need retrieval and/or experiments)"})
        intent_input: str | Mapping[str, Any] = research_intent
        stripped = research_intent.strip() if isinstance(research_intent, str) else ""
        if stripped.startswith("{"):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError as exc:
                return json.dumps({"error": f"research_intent JSON is invalid: {exc}"})
            if not isinstance(decoded, dict):
                return json.dumps({"error": "research_intent JSON must decode to an object."})
            intent_input = decoded
        packet = knowledge_bootstrap(
            intent_input,
            stores=BootstrapStores(
                retrieval=bindings.retrieval,
                experiments=bindings.experiments,
                artifacts=bindings.artifacts,
                skills=get_knowledge_skills(),
            ),
            budget=PacketBudget(max_items=max_items, max_chars=max_chars),
        )
        return json.dumps(packet, ensure_ascii=False)
    except KnowledgeError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        logger.exception("knowledge_bootstrap_tool failed")
        return json.dumps({"error": str(exc)})


def get_knowledge_bootstrap_tool() -> list:
    """Return the run-start bootstrap ``@tool`` wrapper for agent registration.

    Called by the integration step alongside
    ``lookup.get_knowledge_tools()``; this module registers nothing itself.
    """
    return [knowledge_bootstrap_tool]

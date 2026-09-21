"""Standalone tests for the Research Knowledge Plane retrieval stack (Phase 2).

Covers ``retrieval/planner.py`` (research intent, channels, orchestration),
``retrieval/fusion.py`` (hard filters, RRF, research-quality modifiers), and
``retrieval/packet.py`` (fixed-budget context packets). Runs with no
database, no config files, and no integration wiring: in-memory fakes
implement the five channel-store protocols, and the vector fake scores with
the brute-force :func:`planner.cosine_similarity` kernel.

Run from anywhere (the bootstrap below locates the harness package)::

    python -m pytest backend/packages/harness/deerflow/knowledge/test_retrieval_plane.py -q
"""

import json
import sys
from pathlib import Path

_HARNESS_DIR = Path(__file__).resolve().parents[2]
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

import pytest  # noqa: E402

from deerflow.knowledge.retrieval import fusion as fusion_mod  # noqa: E402
from deerflow.knowledge.retrieval import packet as packet_mod  # noqa: E402
from deerflow.knowledge.retrieval import planner as planner_mod  # noqa: E402
from deerflow.knowledge.retrieval.fusion import (  # noqa: E402
    HARD_FILTER_REASONS,
    FusionWeights,
)
from deerflow.knowledge.retrieval.packet import (  # noqa: E402
    SECTION_ORDER,
    ConflictView,
    ContextPacket,
    PacketBudget,
    SummaryPlaceholder,
)
from deerflow.knowledge.retrieval.planner import (  # noqa: E402
    DEFAULT_KINDS,
    Candidate,
    ChannelHit,
    ResearchIntent,
    ScopeFilter,
)
from deerflow.knowledge.write_api import KnowledgeValidationError  # noqa: E402

PROJECT_ID = "11111111-1111-1111-1111-111111111111"
PROJECT_ID_2 = "22222222-2222-2222-2222-222222222222"
FAMILY_HASH = "ab" * 32
NOW = "2026-09-20T00:00:00+00:00"

EVAL_INTENT = {
    "topic": "cross-sectional momentum",
    "asset_class": "equity",
    "markets": ["US"],
    "universe": "sp500-pit",
    "horizon": "6m",
    "frequency": "daily",
    "concepts": ["momentum", "backtest", "sharpe", "turnover", "transaction costs"],
    "requested_period": ["2015-01-01", "2026-09-19"],
    "needed_memory": ["validated_findings", "prior_experiments", "failures"],
}


def make_candidate(candidate_id, **overrides):
    """Build a Candidate with valid defaults (eval-fixture-flavored scope)."""
    kwargs = {
        "id": candidate_id,
        "kind": "finding",
        "title": f"Finding {candidate_id}",
        "text": f"momentum backtest sharpe summary for {candidate_id}",
        "scope": {"asset_class": "equity", "market": "US", "universe": "sp500-pit", "horizon": "6m", "frequency": "daily"},
        "status": "candidate",
        "recorded_at": "2026-06-01T00:00:00+00:00",
    }
    kwargs.update(overrides)
    scope = dict(kwargs.get("scope") or {})
    metadata = dict(kwargs.get("metadata") or {})
    kwargs["scope"] = scope
    kwargs["metadata"] = metadata
    return Candidate(**kwargs)


def make_scope(**overrides):
    """Build a ScopeFilter matching the eval intent by default."""
    kwargs = {
        "asset_class": "equity",
        "markets": ("US",),
        "universe": "sp500-pit",
        "horizon": "6m",
        "frequency": "daily",
    }
    kwargs.update(overrides)
    return ScopeFilter(**kwargs)


def make_hit(candidate, channel="lexical", rank=1):
    """Wrap a candidate in a ChannelHit."""
    return ChannelHit(candidate=candidate, channel=channel, rank=rank)


class FakeStructuredStore:
    """In-memory structured lookup: kinds filter, corpus order, honors limit."""

    def __init__(self, corpus):
        self.corpus = list(corpus)
        self.calls = []

    def structured_lookup(self, scope, *, kinds, limit):
        self.calls.append({"kinds": list(kinds), "limit": limit})
        return [item for item in self.corpus if item.kind in kinds][:limit]


class FakeLexicalStore:
    """In-memory lexical search ranked by the portable LIKE fallback score."""

    def __init__(self, corpus):
        self.corpus = list(corpus)
        self.calls = []

    def lexical_search(self, query_text, scope, *, kinds, limit):
        self.calls.append(query_text)
        scored = [(planner_mod.like_fallback_score(f"{item.title} {item.text}", query_text), item) for item in self.corpus if item.kind in kinds]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]


class FakeVectorStore:
    """In-memory brute-force cosine search over fixed test embeddings."""

    def __init__(self, corpus, embeddings):
        self.corpus = list(corpus)
        self.embeddings = dict(embeddings)
        self.calls = []

    def vector_search(self, query_embedding, scope, *, kinds, limit):
        self.calls.append(tuple(query_embedding))
        scored = []
        for item in self.corpus:
            if item.kind not in kinds or item.id not in self.embeddings:
                continue
            scored.append((planner_mod.cosine_similarity(query_embedding, self.embeddings[item.id]), item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]


class FakeFailureStore:
    """In-memory failure channel over failed experiments + failure findings."""

    def __init__(self, corpus):
        self.corpus = list(corpus)
        self.calls = []

    def search_failures(self, query_text, scope, *, limit):
        self.calls.append(query_text)
        scored = [(planner_mod.like_fallback_score(f"{item.title} {item.text}", query_text), item) for item in self.corpus]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]


class FakeRelationalStore:
    """In-memory knowledge-edge expansion over a fixed adjacency map."""

    def __init__(self, adjacency, corpus_by_id):
        self.adjacency = dict(adjacency)
        self.corpus_by_id = dict(corpus_by_id)
        self.calls = []

    def expand_neighbors(self, seed_ids, scope, *, edge_types, limit):
        self.calls.append({"seeds": list(seed_ids), "edge_types": list(edge_types)})
        neighbors = []
        for seed in seed_ids:
            for neighbor_id in self.adjacency.get(seed, ()):
                if neighbor_id in self.corpus_by_id and neighbor_id not in neighbors:
                    neighbors.append(neighbor_id)
        return [self.corpus_by_id[neighbor_id] for neighbor_id in neighbors[:limit]]


@pytest.fixture
def corpus():
    return [
        make_candidate("exp_mom_126", kind="experiment", status="completed", title="Momentum 126d backtest", text="cross-sectional momentum backtest sharpe 0.82"),
        make_candidate("fail_costs", kind="failure", status="completed", title="Costs erase momentum", text="transaction costs erase momentum profits turnover"),
        make_candidate("f_mom_valid", status="validated", title="Momentum persists", text="cross-sectional momentum persists net of costs"),
        make_candidate("f_mom_cand", status="candidate", title="Momentum candidate", text="momentum may persist in small caps"),
    ]


@pytest.fixture
def stores(corpus):
    return {
        "structured": FakeStructuredStore(corpus),
        "lexical": FakeLexicalStore(corpus),
        "vector": FakeVectorStore(
            corpus,
            {
                "exp_mom_126": (1.0, 0.0, 0.0),
                "fail_costs": (0.0, 1.0, 0.0),
                "f_mom_valid": (0.9, 0.1, 0.0),
                "f_mom_cand": (0.0, 0.0, 1.0),
            },
        ),
        "failures": FakeFailureStore([corpus[1]]),
    }


# ---------------------------------------------------------------------------
# planner.py: research intent
# ---------------------------------------------------------------------------


class TestParseResearchIntent:
    def test_eval_shaped_intent(self):
        intent = planner_mod.parse_research_intent(EVAL_INTENT)
        assert isinstance(intent, ResearchIntent)
        assert intent.topic == "cross-sectional momentum"
        assert intent.asset_class == "equity"
        assert intent.markets == ("US",)
        assert intent.universe == "sp500-pit"
        assert intent.horizon == "6m"
        assert intent.frequency == "daily"
        assert intent.concepts == ("momentum", "backtest", "sharpe", "turnover", "transaction costs")
        assert intent.requested_period == ("2015-01-01", "2026-09-19")
        assert intent.needed_memory == ("validated_findings", "prior_experiments", "failures")
        as_dict = intent.to_dict()
        assert as_dict["requested_period"] == ["2015-01-01", "2026-09-19"]

    def test_defaults(self):
        intent = planner_mod.parse_research_intent({"topic": "factors"})
        assert intent.asset_class is None
        assert intent.markets == ()
        assert intent.concepts == ()
        assert intent.requested_period == (None, None)
        assert intent.needed_memory == planner_mod.NEED_MEMORY_DEFAULT

    def test_needed_memory_dedupes_preserving_order(self):
        intent = planner_mod.parse_research_intent({"topic": "t", "needed_memory": ["failures", "failures", "assumptions"]})
        assert intent.needed_memory == ("failures", "assumptions")

    def test_unknown_keys_ignored(self):
        intent = planner_mod.parse_research_intent({"topic": "t", "future_field": {"nested": 1}})
        assert intent.topic == "t"

    def test_open_ended_period(self):
        intent = planner_mod.parse_research_intent({"topic": "t", "requested_period": ["2020-01-01", None]})
        assert intent.requested_period == ("2020-01-01", None)

    @pytest.mark.parametrize(
        "data",
        [
            "not-a-mapping",
            {},
            {"topic": "   "},
            {"topic": 42},
            {"topic": "t", "needed_memory": []},
            {"topic": "t", "needed_memory": ["telepathy"]},
            {"topic": "t", "needed_memory": "failures"},
            {"topic": "t", "requested_period": ["2020-01-01"]},
            {"topic": "t", "requested_period": ["not-a-date", None]},
            {"topic": "t", "requested_period": ["2026-01-01", "2020-01-01"]},
            {"topic": "t", "markets": ["US", "  "]},
            {"topic": "t", "markets": "US"},
            {"topic": "t", "asset_class": 7},
        ],
    )
    def test_validation_errors(self, data):
        with pytest.raises(KnowledgeValidationError):
            planner_mod.parse_research_intent(data)


class TestEvalFixtureIntents:
    def test_all_fixture_intents_parse(self):
        fixture_path = Path(__file__).resolve().parent / "eval" / "eval_fixture.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        assert len(fixture["queries"]) == 30
        for query in fixture["queries"]:
            intent = planner_mod.parse_research_intent(query["intent"])
            assert intent.topic
            assert set(intent.needed_memory) <= set(planner_mod.NEED_MEMORY_VOCAB)


class TestScopeFilter:
    def test_build_from_intent(self):
        scope = planner_mod.build_scope_filter(planner_mod.parse_research_intent(EVAL_INTENT))
        assert scope.asset_class == "equity"
        assert scope.markets == ("US",)
        assert scope.universe == "sp500-pit"
        assert scope.valid_from == "2015-01-01"
        assert scope.valid_to == "2026-09-19"
        assert scope.allow_restricted_datasets is False
        assert scope.to_dict()["markets"] == ["US"]

    def test_rejects_non_intent(self):
        with pytest.raises(KnowledgeValidationError):
            planner_mod.build_scope_filter({"topic": "t"})


# ---------------------------------------------------------------------------
# planner.py: records
# ---------------------------------------------------------------------------


class TestCandidate:
    def test_to_dict_roundtrip(self):
        candidate = make_candidate("x1", finding_type="empirical", project_id=PROJECT_ID, family_hash=FAMILY_HASH, evidence=("e1",), metadata={"replication_count": 2})
        as_dict = candidate.to_dict()
        assert as_dict["id"] == "x1"
        assert as_dict["scope"]["universe"] == "sp500-pit"
        assert as_dict["metadata"] == {"replication_count": 2}
        assert Candidate(**{**as_dict, "evidence": tuple(as_dict["evidence"])}) == candidate

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"id": "  "},
            {"kind": "telegram"},
            {"title": 42},
            {"text": None},
            {"scope": ["not-a-mapping"]},
            {"scope": {"x": float("nan")}},
            {"status": ""},
            {"finding_type": "rumor"},
            {"project_id": "not-a-uuid"},
            {"valid_from": "not-a-date"},
            {"recorded_at": "not-a-date"},
            {"recorded_at": 42},
            {"family_hash": "short"},
            {"metadata": ["not-a-mapping"]},
            {"evidence": ["ok", "  "]},
        ],
    )
    def test_validation_errors(self, kwargs):
        base = {"id": "x1", "kind": "finding", "status": "candidate"}
        base.update(kwargs)
        with pytest.raises(KnowledgeValidationError):
            Candidate(**base)


class TestChannelHit:
    def test_happy_path(self):
        candidate = make_candidate("x1")
        hit = make_hit(candidate, channel="vector", rank=3)
        assert hit.to_dict() == {"candidate": candidate.to_dict(), "channel": "vector", "rank": 3}

    @pytest.mark.parametrize("kwargs", [{"channel": "pagerank"}, {"rank": 0}, {"rank": -2}, {"rank": True}, {"rank": "1"}])
    def test_validation_errors(self, kwargs):
        with pytest.raises(KnowledgeValidationError):
            ChannelHit(candidate=make_candidate("x1"), channel=kwargs.get("channel", "lexical"), rank=kwargs.get("rank", 1))

    def test_rejects_non_candidate(self):
        with pytest.raises(KnowledgeValidationError):
            ChannelHit(candidate={"id": "x1"}, channel="lexical", rank=1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# planner.py: pure kernels
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_is_one(self):
        assert planner_mod.cosine_similarity((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)) == pytest.approx(1.0)

    def test_orthogonal_is_zero(self):
        assert planner_mod.cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)

    def test_opposite_is_minus_one(self):
        assert planner_mod.cosine_similarity((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)

    def test_known_angle(self):
        assert planner_mod.cosine_similarity((1.0, 1.0), (1.0, 0.0)) == pytest.approx(2**0.5 / 2)

    def test_zero_vector_scores_zero_without_nan(self):
        assert planner_mod.cosine_similarity((0.0, 0.0), (1.0, 0.0)) == 0.0

    @pytest.mark.parametrize("left,right", [[(), (1.0,)], [(1.0,), (1.0, 2.0)], [(1.0, float("nan")), (1.0, 0.0)], [(1.0, float("inf")), (1.0, 0.0)], [("1",), ("1",)], ["ab", "ab"]])
    def test_validation_errors(self, left, right):
        with pytest.raises(KnowledgeValidationError):
            planner_mod.cosine_similarity(left, right)


class TestLikeFallbackScore:
    def test_exact_phrase_scores_one(self):
        assert planner_mod.like_fallback_score("Momentum persists net of costs", "momentum persists") == 1.0

    def test_partial_coverage_is_fractional(self):
        assert planner_mod.like_fallback_score("momentum everywhere", "momentum costs") == pytest.approx(0.5)

    def test_case_insensitive(self):
        assert planner_mod.like_fallback_score("MOMENTUM", "momentum") == 1.0

    def test_empty_query_scores_zero(self):
        assert planner_mod.like_fallback_score("momentum", "   ") == 0.0

    def test_no_match_scores_zero(self):
        assert planner_mod.like_fallback_score("value investing", "momentum costs") == 0.0

    def test_substring_matching(self):
        assert planner_mod.like_fallback_score("antimomentum drift", "momentum") == pytest.approx(1.0)

    def test_rejects_non_strings(self):
        with pytest.raises(KnowledgeValidationError):
            planner_mod.like_fallback_score("doc", 42)  # type: ignore[arg-type]


class TestLikeFallbackIdfScores:
    def test_rare_term_outweighs_common_term(self):
        docs = ["risk factor", "risk brinson", "value investing"]
        scores = planner_mod.like_fallback_idf_scores(docs, "risk brinson")
        assert scores[1] > scores[0] > scores[2] == 0.0

    def test_all_tokens_matched_scores_one_without_bonus(self):
        assert planner_mod.like_fallback_idf_scores(["costs momentum", "costs momentum"], "momentum costs") == [pytest.approx(1.0), pytest.approx(1.0)]

    def test_empty_query_scores_zero(self):
        assert planner_mod.like_fallback_idf_scores(["momentum", "costs"], "   ") == [0.0, 0.0]

    def test_empty_corpus_scores_empty(self):
        assert planner_mod.like_fallback_idf_scores([], "momentum") == []

    def test_corpus_absent_tokens_still_count(self):
        # "zzz" matches nothing but dilutes the denominator, like the flat scorer.
        scores = planner_mod.like_fallback_idf_scores(["momentum"], "momentum zzz")
        assert 0.0 < scores[0] < 1.0

    def test_phrase_bonus_capped_at_one(self):
        assert planner_mod.like_fallback_idf_scores(["momentum persists"], "momentum persists") == [pytest.approx(1.0)]

    def test_stopwords_dropped_from_query(self):
        # Noise words must neither dilute the denominator nor match "Brinson"
        # via substring ("in"); parity with the PG english text-search config.
        docs = ["brinson attribution", "risk factor"]
        noisy = planner_mod.like_fallback_idf_scores(docs, "how do brinson and attribution show up in risk?")
        content_only = planner_mod.like_fallback_idf_scores(docs, "brinson attribution risk")
        assert noisy == content_only
        assert noisy[0] > noisy[1] > 0.0
        assert planner_mod.like_fallback_idf_scores(docs, "how do you do") == [0.0, 0.0]

    def test_rejects_bad_inputs(self):
        with pytest.raises(KnowledgeValidationError):
            planner_mod.like_fallback_idf_scores("not-a-sequence", "momentum")  # type: ignore[arg-type]
        with pytest.raises(KnowledgeValidationError):
            planner_mod.like_fallback_idf_scores(["doc", 42], "momentum")  # type: ignore[list-item]
        with pytest.raises(KnowledgeValidationError):
            planner_mod.like_fallback_idf_scores(["doc"], 42)  # type: ignore[arg-type]


class TestLexicalQueryText:
    def test_folds_missing_concepts(self):
        intent = planner_mod.parse_research_intent({"topic": "reversal study", "concepts": ["reversal", "quantile backtest"]})
        assert planner_mod.lexical_query_text("what IC at 21 days?", intent) == "what IC at 21 days? reversal quantile backtest"

    def test_skips_concepts_already_present(self):
        intent = planner_mod.parse_research_intent({"topic": "momentum", "concepts": ["momentum", "turnover"]})
        assert planner_mod.lexical_query_text("momentum premia", intent) == "momentum premia turnover"

    def test_none_and_empty_intent_return_text(self):
        assert planner_mod.lexical_query_text("momentum", None) == "momentum"
        assert planner_mod.lexical_query_text("momentum", {}) == "momentum"
        assert planner_mod.lexical_query_text("momentum", {"concepts": []}) == "momentum"

    def test_rejects_non_string_query(self):
        with pytest.raises(KnowledgeValidationError):
            planner_mod.lexical_query_text(42, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# planner.py: planning + execution
# ---------------------------------------------------------------------------


class TestPlanRetrieval:
    def test_default_channels(self):
        plan = planner_mod.plan_retrieval(EVAL_INTENT)
        assert plan.channels == ("structured", "lexical", "vector", "failure")
        assert plan.kinds == DEFAULT_KINDS
        assert plan.per_channel_limit == planner_mod.DEFAULT_PER_CHANNEL_LIMIT
        assert plan.top_k == planner_mod.DEFAULT_TOP_K
        assert plan.relational is True
        assert plan.to_dict()["channels"] == ["structured", "lexical", "vector", "failure"]

    def test_failures_only_intent_arms_only_failure_channel(self):
        plan = planner_mod.plan_retrieval({"topic": "t", "needed_memory": ["failures"]})
        assert plan.channels == ("failure",)

    def test_conflicts_and_skills_use_structured_lookup(self):
        plan = planner_mod.plan_retrieval({"topic": "t", "needed_memory": ["conflicts", "relevant_skills", "assumptions"]})
        assert plan.channels == ("structured",)

    def test_accepts_parsed_intent(self):
        plan = planner_mod.plan_retrieval(planner_mod.parse_research_intent(EVAL_INTENT), top_k=5, per_channel_limit=7)
        assert plan.top_k == 5
        assert plan.per_channel_limit == 7

    def test_kinds_deduped(self):
        plan = planner_mod.plan_retrieval({"topic": "t"}, kinds=["finding", "finding", "failure"])
        assert plan.kinds == ("finding", "failure")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"per_channel_limit": 0},
            {"per_channel_limit": 201},
            {"per_channel_limit": True},
            {"top_k": 0},
            {"top_k": 101},
            {"top_k": "10"},
            {"kinds": []},
            {"kinds": ["finding", "bogus"]},
            {"edge_types": ["supports", "bogus"]},
            {"relational": "yes"},
        ],
    )
    def test_validation_errors(self, kwargs):
        with pytest.raises(KnowledgeValidationError):
            planner_mod.plan_retrieval({"topic": "t"}, **kwargs)


class TestExecuteRetrieval:
    def test_full_pass_single_round(self, corpus, stores):
        plan = planner_mod.plan_retrieval(EVAL_INTENT)
        result = planner_mod.execute_retrieval(
            plan,
            query_text="cross-sectional momentum backtest",
            structured=stores["structured"],
            lexical=stores["lexical"],
            vector=stores["vector"],
            failures=stores["failures"],
            query_embedding=(1.0, 0.0, 0.0),
            now=NOW,
        )
        assert result.rounds == 1
        assert result.channel_counts == {"structured": 4, "lexical": 4, "vector": 4, "failure": 1}
        assert result.warnings == ()
        assert [item.rank for item in result.fusion.candidates] == [1, 2, 3, 4]
        assert result.top == result.fusion.candidates[: plan.top_k]
        assert result.to_dict()["rounds"] == 1

    def test_vector_skipped_without_embedding(self, corpus, stores):
        plan = planner_mod.plan_retrieval(EVAL_INTENT)
        result = planner_mod.execute_retrieval(
            plan,
            query_text="momentum",
            structured=stores["structured"],
            lexical=stores["lexical"],
            vector=stores["vector"],
            failures=stores["failures"],
            now=NOW,
        )
        assert result.channel_counts["vector"] == 0
        assert stores["vector"].calls == []
        assert result.warnings == ("vector channel skipped: no query_embedding supplied",)

    def test_unplanned_channels_not_called(self, corpus, stores):
        plan = planner_mod.plan_retrieval({"topic": "t", "needed_memory": ["failures"]})
        result = planner_mod.execute_retrieval(
            plan,
            query_text="costs",
            structured=stores["structured"],
            lexical=stores["lexical"],
            vector=stores["vector"],
            failures=stores["failures"],
            now=NOW,
        )
        assert result.channel_counts == {"failure": 1}
        assert stores["structured"].calls == []
        assert stores["lexical"].calls == []

    def test_per_channel_limit_propagates(self, corpus, stores):
        plan = planner_mod.plan_retrieval(EVAL_INTENT, per_channel_limit=2)
        result = planner_mod.execute_retrieval(
            plan,
            query_text="momentum",
            structured=stores["structured"],
            lexical=stores["lexical"],
            vector=stores["vector"],
            failures=stores["failures"],
            query_embedding=(1.0, 0.0, 0.0),
            now=NOW,
        )
        assert result.channel_counts == {"structured": 2, "lexical": 2, "vector": 2, "failure": 1}

    def test_relational_second_pass(self, corpus, stores):
        neighbor = make_candidate("f_replicating", status="reviewed", title="Independent replication", text="independent replication of momentum")
        relational = FakeRelationalStore({"f_mom_valid": ("f_replicating",)}, {"f_replicating": neighbor})
        plan = planner_mod.plan_retrieval(EVAL_INTENT)
        result = planner_mod.execute_retrieval(
            plan,
            query_text="momentum",
            structured=stores["structured"],
            lexical=stores["lexical"],
            vector=stores["vector"],
            failures=stores["failures"],
            relational=relational,
            query_embedding=(1.0, 0.0, 0.0),
            relational_seed_k=2,
            now=NOW,
        )
        pass1 = planner_mod.execute_retrieval(
            plan,
            query_text="momentum",
            structured=stores["structured"],
            lexical=stores["lexical"],
            vector=stores["vector"],
            failures=stores["failures"],
            query_embedding=(1.0, 0.0, 0.0),
            now=NOW,
        )
        assert result.rounds == 2
        assert result.channel_counts["relational"] == 1
        assert relational.calls[0]["seeds"] == [item.candidate.id for item in pass1.fusion.candidates[:2]]
        assert "f_replicating" in {item.candidate.id for item in result.fusion.candidates}
        assert relational.calls[0]["edge_types"] == list(planner_mod.EDGE_TYPES)

    def test_relational_disarmed_by_plan(self, corpus, stores):
        relational = FakeRelationalStore({}, {})
        plan = planner_mod.plan_retrieval(EVAL_INTENT, relational=False)
        result = planner_mod.execute_retrieval(
            plan,
            query_text="momentum",
            structured=stores["structured"],
            lexical=stores["lexical"],
            vector=stores["vector"],
            failures=stores["failures"],
            relational=relational,
            now=NOW,
        )
        assert result.rounds == 1
        assert "relational" not in result.channel_counts
        assert relational.calls == []

    def test_store_returning_non_candidate_rejected(self, corpus, stores):
        class BadStore:
            def structured_lookup(self, scope, *, kinds, limit):
                return [{"id": "bogus"}]

        plan = planner_mod.plan_retrieval({"topic": "t", "needed_memory": ["conflicts"]})
        with pytest.raises(KnowledgeValidationError, match="must return Candidate"):
            planner_mod.execute_retrieval(
                plan,
                query_text="momentum",
                structured=BadStore(),
                lexical=stores["lexical"],
                vector=stores["vector"],
                failures=stores["failures"],
                now=NOW,
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"query_text": "   "},
            {"query_embedding": ()},
            {"query_embedding": (1.0, float("nan"))},
            {"query_embedding": "abc"},
            {"relational_seed_k": 0},
        ],
    )
    def test_validation_errors(self, corpus, stores, kwargs):
        plan = planner_mod.plan_retrieval({"topic": "t"})
        call_kwargs = {
            "query_text": "momentum",
            "structured": stores["structured"],
            "lexical": stores["lexical"],
            "vector": stores["vector"],
            "failures": stores["failures"],
        }
        call_kwargs.update(kwargs)
        with pytest.raises(KnowledgeValidationError):
            planner_mod.execute_retrieval(plan, **call_kwargs)

    def test_rejects_non_plan(self, stores):
        with pytest.raises(KnowledgeValidationError):
            planner_mod.execute_retrieval(
                {"not": "a plan"},
                query_text="momentum",
                structured=stores["structured"],
                lexical=stores["lexical"],
                vector=stores["vector"],
                failures=stores["failures"],
            )


# ---------------------------------------------------------------------------
# fusion.py: hard filters
# ---------------------------------------------------------------------------


class TestHardFilters:
    def test_documented_filter_list(self):
        assert HARD_FILTER_REASONS == (
            "rejected",
            "acl_project_not_visible",
            "asset_class_mismatch",
            "restricted_dataset",
            "validity_window_disjoint",
        )

    def test_rejected_never_retrievable(self):
        candidate = make_candidate("r1", status="rejected")
        survivors, filtered = fusion_mod.apply_hard_filters([candidate, make_candidate("k1")], make_scope())
        assert [item.id for item in survivors] == ["k1"]
        assert [(item.candidate.id, item.reason) for item in filtered] == [("r1", "rejected")]

    def test_acl_enforcement(self):
        scoped_in = make_candidate("in1", project_id=PROJECT_ID)
        scoped_out = make_candidate("out1", project_id=PROJECT_ID_2)
        unscoped = make_candidate("u1")
        survivors, filtered = fusion_mod.apply_hard_filters([scoped_in, scoped_out, unscoped], make_scope(), allowed_project_ids=[PROJECT_ID])
        assert [item.id for item in survivors] == ["in1", "u1"]
        assert [(item.candidate.id, item.reason) for item in filtered] == [("out1", "acl_project_not_visible")]

    def test_acl_disabled_by_default(self):
        survivors, filtered = fusion_mod.apply_hard_filters([make_candidate("o1", project_id=PROJECT_ID_2)], make_scope())
        assert [item.id for item in survivors] == ["o1"]
        assert filtered == []

    def test_acl_rejects_bad_allowlist(self):
        with pytest.raises(KnowledgeValidationError):
            fusion_mod.apply_hard_filters([make_candidate("x1")], make_scope(), allowed_project_ids="not-a-sequence")
        with pytest.raises(KnowledgeValidationError):
            fusion_mod.apply_hard_filters([make_candidate("x1")], make_scope(), allowed_project_ids=[PROJECT_ID, "  "])

    def test_asset_class_mismatch(self):
        scope = make_scope(asset_class="equity")
        mismatch = make_candidate("m1", scope={"asset_class": "credit"})
        match = make_candidate("m2", scope={"asset_class": "Equity"})
        undeclared = make_candidate("m3", scope={"universe": "sp500-pit"})
        survivors, filtered = fusion_mod.apply_hard_filters([mismatch, match, undeclared], scope)
        assert [item.id for item in survivors] == ["m2", "m3"]
        assert [item.reason for item in filtered] == ["asset_class_mismatch"]

    def test_restricted_dataset(self):
        restricted = make_candidate("d1", scope={"dataset_restricted": True})
        survivors, _ = fusion_mod.apply_hard_filters([restricted], make_scope())
        assert survivors == []
        survivors, filtered = fusion_mod.apply_hard_filters([restricted], make_scope(allow_restricted_datasets=True))
        assert [item.id for item in survivors] == ["d1"]
        assert filtered == []

    def test_validity_window(self):
        scope = make_scope(valid_from="2020-01-01", valid_to="2024-12-31")
        disjoint = make_candidate("v1", valid_from="2025-01-01", valid_to="2025-12-31")
        overlap = make_candidate("v2", valid_from="2024-06-01", valid_to="2025-06-01")
        timeless = make_candidate("v3")
        survivors, filtered = fusion_mod.apply_hard_filters([disjoint, overlap, timeless], scope)
        assert [item.id for item in survivors] == ["v2", "v3"]
        assert [(item.candidate.id, item.reason) for item in filtered] == [("v1", "validity_window_disjoint")]

    def test_no_required_window_passes_all(self):
        candidate = make_candidate("v1", valid_from="1999-01-01", valid_to="1999-12-31")
        survivors, filtered = fusion_mod.apply_hard_filters([candidate], make_scope())
        assert [item.id for item in survivors] == ["v1"]
        assert filtered == []

    def test_order_preserved(self):
        survivors, _ = fusion_mod.apply_hard_filters([make_candidate("b"), make_candidate("a")], make_scope())
        assert [item.id for item in survivors] == ["b", "a"]


# ---------------------------------------------------------------------------
# fusion.py: RRF
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_multi_channel_rank_one_wins(self):
        scores = fusion_mod.reciprocal_rank_fusion({"lexical": ["a", "b"], "vector": ["a", "c"]})
        assert scores["a"] == pytest.approx(2 / 61)
        assert scores["b"] == pytest.approx(1 / 62)
        assert scores["a"] > scores["b"] > 0

    def test_custom_k(self):
        scores = fusion_mod.reciprocal_rank_fusion({"lexical": ["a"]}, k=1)
        assert scores == {"a": pytest.approx(1 / 2)}

    def test_empty_rankings(self):
        assert fusion_mod.reciprocal_rank_fusion({}) == {}

    @pytest.mark.parametrize(
        "rankings,kwargs",
        [
            ({"lexical": ["a", "a"]}, {}),
            ({"lexical": ["a", ""]}, {}),
            ({"lexical": "ab"}, {}),
            ("not-a-mapping", {}),
            ({"lexical": ["a"]}, {"k": 0}),
            ({"lexical": ["a"]}, {"k": True}),
        ],
    )
    def test_validation_errors(self, rankings, kwargs):
        with pytest.raises(KnowledgeValidationError):
            fusion_mod.reciprocal_rank_fusion(rankings, **kwargs)


# ---------------------------------------------------------------------------
# fusion.py: modifiers
# ---------------------------------------------------------------------------


def fuse_ids(hits, scope=None, **kwargs):
    """Fuse hits and return (ordered ids, result)."""
    result = fusion_mod.fuse(hits, scope or make_scope(), now=NOW, **kwargs)
    return [item.candidate.id for item in result.candidates], result


class TestFuse:
    def test_dedups_and_merges_channels(self):
        candidate = make_candidate("d1")
        ids, result = fuse_ids([make_hit(candidate, "lexical", 2), make_hit(candidate, "vector", 1)])
        assert ids == ["d1"]
        assert result.candidates[0].channels == ("lexical", "vector")
        assert result.candidates[0].rrf_score == pytest.approx(1 / 61 + 1 / 62)
        assert result.candidates[0].rank == 1

    def test_tie_breaks_by_id(self):
        first = make_candidate("a1", scope={}, recorded_at="")
        second = make_candidate("b1", scope={}, recorded_at="")
        ids, result = fuse_ids([make_hit(second, "lexical", 1), make_hit(first, "vector", 1)], scope=ScopeFilter())
        assert result.candidates[0].score == result.candidates[1].score
        assert ids == ["a1", "b1"]

    def test_exact_scope_beats_partial_beats_none(self):
        scope = make_scope()
        exact = make_candidate("exact", scope={"asset_class": "equity", "market": "US", "universe": "sp500-pit", "horizon": "6m", "frequency": "daily"}, recorded_at="")
        partial = make_candidate("partial", scope={"asset_class": "equity", "market": "EU"}, recorded_at="")
        assert fusion_mod.scope_match_degree(exact.scope, scope) == "exact"
        assert fusion_mod.scope_match_degree(partial.scope, scope) == "partial"
        # "none" fixture would hard-filter on asset class, so scope it neutrally here.
        neutral_none = make_candidate("none", scope={}, recorded_at="")
        assert fusion_mod.scope_match_degree(neutral_none.scope, scope) == "none"
        ids, result = fuse_ids(
            [make_hit(exact, "lexical", 1), make_hit(partial, "lexical", 2), make_hit(neutral_none, "lexical", 3)],
            scope=scope,
        )
        assert ids == ["exact", "partial", "none"]
        by_id = {item.candidate.id: item for item in result.candidates}
        assert by_id["exact"].modifiers["exact_scope"] == pytest.approx(1.5)
        assert by_id["partial"].modifiers["partial_scope"] == pytest.approx(1.15)
        assert "exact_scope" not in by_id["none"].modifiers

    def test_undeclared_filter_scope_is_neutral(self):
        candidate = make_candidate("n1", recorded_at="")
        assert fusion_mod.scope_match_degree(candidate.scope, ScopeFilter()) == "none"
        _, result = fuse_ids([make_hit(candidate, "lexical", 1)], scope=ScopeFilter())
        assert "exact_scope" not in result.candidates[0].modifiers

    def test_placeholder_scope_values_are_neutral_both_sides(self):
        # "mixed"/"n/a" mean unspecified: placeholder-only agreement must
        # not read as "exact", and placeholders must neither match (boost)
        # nor mismatch (demote) when real fields are compared.
        assert fusion_mod.scope_match_degree({"horizon": "mixed"}, ScopeFilter(horizon="mixed")) == "none"
        assert fusion_mod.scope_match_degree({"universe": "n/a"}, ScopeFilter(universe="sp500-pit")) == "none"
        mixed_filter = make_scope(horizon="mixed")
        full = {"asset_class": "equity", "universe": "sp500-pit", "frequency": "daily", "market": "US"}
        assert fusion_mod.scope_match_degree({**full, "horizon": "mixed"}, mixed_filter) == "exact"
        assert fusion_mod.scope_match_degree({**full, "horizon": "63d"}, mixed_filter) == "exact"
        assert fusion_mod.scope_match_degree({**full, "universe": "n/a", "horizon": "6m"}, make_scope()) == "exact"

    def test_validated_beats_candidate_at_equal_rrf(self):
        validated = make_candidate("v1", status="validated", scope={}, recorded_at="")
        candidate = make_candidate("c1", status="candidate", scope={}, recorded_at="")
        _, result = fuse_ids([make_hit(candidate, "lexical", 1), make_hit(validated, "vector", 1)], scope=ScopeFilter())
        by_id = {item.candidate.id: item for item in result.candidates}
        assert by_id["v1"].modifiers["validated"] == pytest.approx(1.4)
        assert by_id["v1"].score == pytest.approx(by_id["c1"].score * 1.4)
        assert [item.candidate.id for item in result.candidates] == ["v1", "c1"]

    def test_reviewed_boost(self):
        reviewed = make_candidate("r1", status="reviewed", scope={}, recorded_at="")
        _, result = fuse_ids([make_hit(reviewed, "lexical", 1)], scope=ScopeFilter())
        assert result.candidates[0].modifiers["reviewed"] == pytest.approx(1.15)

    def test_replication_boost_and_strong_threshold(self):
        one = make_candidate("rep1", scope={}, recorded_at="", metadata={"replication_count": 1})
        strong = make_candidate("rep3", scope={}, recorded_at="", metadata={"replication_count": 3})
        linked = make_candidate("repL", scope={}, recorded_at="", metadata={"replicated_experiment_id": PROJECT_ID})
        _, result = fuse_ids([make_hit(one, "lexical", 1), make_hit(strong, "lexical", 2), make_hit(linked, "lexical", 3)], scope=ScopeFilter())
        by_id = {item.candidate.id: item for item in result.candidates}
        assert by_id["rep1"].modifiers["replication"] == pytest.approx(1.25)
        assert by_id["rep3"].modifiers["strong_replication"] == pytest.approx(1.4)
        assert "replication" not in by_id["rep3"].modifiers
        assert by_id["repL"].modifiers["replication"] == pytest.approx(1.25)

    def test_same_family_boost(self):
        member = make_candidate("fam1", scope={}, recorded_at="", family_hash=FAMILY_HASH)
        outsider = make_candidate("fam2", scope={}, recorded_at="", family_hash="00" * 32)
        _, result = fuse_ids([make_hit(member, "lexical", 1), make_hit(outsider, "lexical", 2)], scope=ScopeFilter(), query_family_hash=FAMILY_HASH)
        by_id = {item.candidate.id: item for item in result.candidates}
        assert by_id["fam1"].modifiers["same_family"] == pytest.approx(1.2)
        assert "same_family" not in by_id["fam2"].modifiers

    def test_same_family_rejects_bad_hash(self):
        with pytest.raises(KnowledgeValidationError):
            fuse_ids([make_hit(make_candidate("x1"), "lexical", 1)], query_family_hash="short")
        with pytest.raises(KnowledgeValidationError):
            fuse_ids([make_hit(make_candidate("x1"), "lexical", 1)], query_family_hash="AB" * 32)

    def test_recency_window(self):
        recent = make_candidate("new1", scope={}, recorded_at="2026-09-01T00:00:00+00:00")
        old = make_candidate("old1", scope={}, recorded_at="2020-01-01T00:00:00+00:00")
        future = make_candidate("fut1", scope={}, recorded_at="2027-01-01T00:00:00+00:00")
        missing = make_candidate("mis1", scope={}, recorded_at="")
        _, result = fuse_ids(
            [make_hit(recent, "lexical", 1), make_hit(old, "lexical", 2), make_hit(future, "lexical", 3), make_hit(missing, "lexical", 4)],
            scope=ScopeFilter(),
        )
        by_id = {item.candidate.id: item for item in result.candidates}
        assert by_id["new1"].modifiers["recency"] == pytest.approx(1.1)
        assert "recency" not in by_id["old1"].modifiers
        assert "recency" not in by_id["fut1"].modifiers
        assert "recency" not in by_id["mis1"].modifiers

    def test_bad_now_rejected(self):
        with pytest.raises(KnowledgeValidationError):
            fusion_mod.fuse([make_hit(make_candidate("x1"), "lexical", 1)], make_scope(), now="not-a-date")

    def test_disputed_stays_visible_and_flagged(self):
        disputed = make_candidate("d1", status="disputed", scope={}, recorded_at="")
        ids, result = fuse_ids([make_hit(disputed, "lexical", 1)], scope=ScopeFilter())
        assert ids == ["d1"]
        assert result.candidates[0].disputed is True
        assert result.candidates[0].modifiers["dispute_visibility"] == pytest.approx(1.05)
        assert result.filtered == []

    def test_near_dup_diversity_penalty(self):
        first = make_candidate("n1", scope={}, recorded_at="", metadata={"near_dup_group": "g1"})
        second = make_candidate("n2", scope={}, recorded_at="", metadata={"near_dup_group": "g1"})
        other = make_candidate("n3", scope={}, recorded_at="", metadata={"near_dup_group": "g2"})
        _, result = fuse_ids([make_hit(first, "lexical", 1), make_hit(second, "lexical", 2), make_hit(other, "lexical", 3)], scope=ScopeFilter())
        by_id = {item.candidate.id: item for item in result.candidates}
        assert "near_dup_diversity" not in by_id["n1"].modifiers
        assert by_id["n2"].modifiers["near_dup_diversity"] == pytest.approx(0.5)
        assert "near_dup_diversity" not in by_id["n3"].modifiers
        assert by_id["n2"].score < by_id["n3"].score

    def test_superseded_demoted_but_present(self):
        current = make_candidate("cur1", status="validated", scope={}, recorded_at="")
        old = make_candidate("old1", status="superseded", scope={}, recorded_at="")
        ids, result = fuse_ids([make_hit(old, "lexical", 1), make_hit(current, "lexical", 2)], scope=ScopeFilter())
        by_id = {item.candidate.id: item for item in result.candidates}
        assert by_id["old1"].modifiers["superseded"] == pytest.approx(0.2)
        assert ids == ["cur1", "old1"]

    def test_failure_boost_all_forms(self):
        kind_failure = make_candidate("k1", kind="failure", status="completed", scope={}, recorded_at="")
        type_failure = make_candidate("t1", finding_type="failure", scope={}, recorded_at="")
        outcome_failure = make_candidate("o1", kind="experiment", status="failed", scope={}, recorded_at="", metadata={"outcome": "failure"})
        plain = make_candidate("p1", scope={}, recorded_at="")
        _, result = fuse_ids(
            [make_hit(kind_failure, "lexical", 1), make_hit(type_failure, "lexical", 2), make_hit(outcome_failure, "lexical", 3), make_hit(plain, "lexical", 4)],
            scope=ScopeFilter(),
        )
        by_id = {item.candidate.id: item for item in result.candidates}
        for doomed in ("k1", "t1", "o1"):
            assert by_id[doomed].failure is True
            assert by_id[doomed].modifiers["failure"] == pytest.approx(1.3)
        assert by_id["p1"].failure is False
        assert "failure" not in by_id["p1"].modifiers

    def test_failure_boost_flag_scopes_boost_to_failures_section(self):
        # Priors ranking: no failure-channel hits, boost off — the
        # lexical-#1 prior outranks the lexical-#2 failure on RRF merit
        # (1/61 > 1/62). Failures ranking: failure-channel hits included,
        # boost on — the failure wins (RRF double presence x 1.3). The
        # failure MARKER is unaffected by the flag in both cases.
        prior = make_candidate("prior1", scope={}, recorded_at="")
        flop = make_candidate("flop1", kind="failure", status="completed", scope={}, recorded_at="")
        priors_hits = [make_hit(prior, "lexical", 1), make_hit(flop, "lexical", 2)]
        ids_off, result_off = fuse_ids(priors_hits, scope=ScopeFilter(), apply_failure_boost=False)
        assert ids_off == ["prior1", "flop1"]
        by_off = {item.candidate.id: item for item in result_off.candidates}
        assert by_off["flop1"].failure is True
        assert "failure" not in by_off["flop1"].modifiers
        all_hits = [*priors_hits, make_hit(flop, "failure", 1)]
        ids_on, result_on = fuse_ids(all_hits, scope=ScopeFilter(), apply_failure_boost=True)
        assert ids_on == ["flop1", "prior1"]
        by_on = {item.candidate.id: item for item in result_on.candidates}
        assert by_on["flop1"].modifiers["failure"] == pytest.approx(1.3)

    def test_failure_boost_flag_rejects_non_bool(self):
        prior = make_candidate("prior1", scope={}, recorded_at="")
        with pytest.raises(KnowledgeValidationError):
            fuse_ids([make_hit(prior, "lexical", 1)], scope=ScopeFilter(), apply_failure_boost="yes")

    def test_hard_filter_beats_great_rank(self):
        rejected = make_candidate("rej1", status="rejected", scope={}, recorded_at="")
        plain = make_candidate("ok1", scope={}, recorded_at="")
        _, result = fuse_ids([make_hit(rejected, "lexical", 1), make_hit(rejected, "vector", 1), make_hit(plain, "lexical", 2)], scope=ScopeFilter())
        assert [item.candidate.id for item in result.candidates] == ["ok1"]
        assert [(item.candidate.id, item.reason) for item in result.filtered] == [("rej1", "rejected")]

    def test_custom_weights(self):
        weights = FusionWeights(validated_boost=2.0, rrf_k=10)
        validated = make_candidate("v1", status="validated", scope={}, recorded_at="")
        _, result = fuse_ids([make_hit(validated, "lexical", 1)], scope=ScopeFilter(), weights=weights)
        assert result.candidates[0].rrf_score == pytest.approx(1 / 11)
        assert result.candidates[0].modifiers["validated"] == pytest.approx(2.0)
        assert result.weights.validated_boost == 2.0
        assert result.to_dict()["weights"]["rrf_k"] == 10

    def test_fused_to_dict(self):
        _, result = fuse_ids([make_hit(make_candidate("x1", status="validated"), "lexical", 1)])
        as_dict = result.to_dict()
        assert as_dict["candidates"][0]["rank"] == 1
        assert as_dict["candidates"][0]["channels"] == ["lexical"]
        assert as_dict["filtered"] == []

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rrf_k": 0},
            {"rrf_k": True},
            {"recency_window_days": 0},
            {"strong_replication_threshold": 0},
            {"validated_boost": 0},
            {"validated_boost": -1.0},
            {"near_dup_penalty": "half"},
        ],
    )
    def test_weights_validation(self, kwargs):
        with pytest.raises(KnowledgeValidationError):
            FusionWeights(**kwargs)


# ---------------------------------------------------------------------------
# packet.py: budget primitives
# ---------------------------------------------------------------------------


class TestBudgetPrimitives:
    def test_estimate_tokens(self):
        assert packet_mod.estimate_tokens("") == 1
        assert packet_mod.estimate_tokens("abcd") == 1
        assert packet_mod.estimate_tokens("abcde") == 2
        with pytest.raises(KnowledgeValidationError):
            packet_mod.estimate_tokens(42)  # type: ignore[arg-type]

    def test_utf8_bytes(self):
        assert packet_mod.utf8_bytes("abc") == 3
        assert packet_mod.utf8_bytes("é") == 2
        with pytest.raises(KnowledgeValidationError):
            packet_mod.utf8_bytes(None)  # type: ignore[arg-type]

    def test_budget_validation(self):
        assert PacketBudget().to_dict() == {"max_tokens": 4000, "max_bytes": 16384, "max_snippet_chars": 600}
        for kwargs in ({"max_tokens": 0}, {"max_bytes": -1}, {"max_snippet_chars": True}):
            with pytest.raises(KnowledgeValidationError):
                PacketBudget(**kwargs)


def fused_fixture():
    """Build a ranked FusionResult spanning every packet section."""
    scope = make_scope()
    hits = [
        make_hit(make_candidate("F17", status="validated", title="Momentum persists", text="cross-sectional momentum persists net of costs", evidence=("E542", "A9")), "lexical", 1),
        make_hit(make_candidate("E542", kind="experiment", status="completed", title="Momentum 126d backtest", text="top-decile rebalance ME sharpe 0.82"), "structured", 1),
        make_hit(make_candidate("E304", kind="failure", status="failed", title="Costs erase momentum", text="failed because transaction costs erase paper profits", evidence=("A11",)), "failure", 1),
        make_hit(make_candidate("F198", status="disputed", title="Momentum is dead", text="contradicting evidence after costs"), "lexical", 2),
        make_hit(make_candidate("A54", kind="assumption", status="active", title="Zero slippage", text="backtest assumes zero slippage"), "structured", 2),
        make_hit(make_candidate("SK1", kind="skill", status="active", title="robust-factor-backtest", text="reusable backtest harness"), "structured", 3),
        make_hit(make_candidate("P4", kind="prior", status="active", title="turnover-sensitive-signals", text="penalize high-turnover signals"), "structured", 4),
    ]
    return fusion_mod.fuse(hits, scope, now=NOW)


class TestBuildContextPacket:
    def test_all_sections_present_in_order(self):
        packet = packet_mod.build_context_packet(fused_fixture(), open_questions=["Does momentum survive 2026 costs?"], topic_key="momentum")
        assert isinstance(packet, ContextPacket)
        assert [section.name for section in packet.sections] == list(SECTION_ORDER)
        assert packet.section("consensus").items[0].id == "F17"
        assert packet.section("prior-experiments").items[0].id == "E542"
        assert packet.section("failures").items[0].id == "E304"
        assert packet.section("conflicts").items[0].id == "F198"
        assert packet.section("assumptions").items[0].id == "A54"
        assert [item.id for item in packet.section("skills").items] == ["SK1", "P4"]
        assert packet.section("open-questions").items[0].title == "Does momentum survive 2026 costs?"
        assert packet.within_budget is True
        assert packet.warnings == ()

    def test_accepts_plain_sequence(self):
        result = fused_fixture()
        packet = packet_mod.build_context_packet(result.candidates)
        assert packet.section("consensus").items[0].id == "F17"

    def test_l0_l1_placeholders_marked_phase4(self):
        packet = packet_mod.build_context_packet(fused_fixture(), topic_key="momentum", l1_topics=["costs", "universes"])
        assert packet.l0.level == "L0"
        assert packet.l0.topic_key == "momentum"
        assert packet.l0.status == "pending_phase4"
        assert [item.topic_key for item in packet.l1] == ["costs", "universes"]
        assert all(item.status == "pending_phase4" for item in packet.l1)
        assert isinstance(packet.l0, SummaryPlaceholder)

    def test_l2_evidence_pointers_preserved(self):
        packet = packet_mod.build_context_packet(fused_fixture())
        consensus_item = packet.section("consensus").items[0]
        assert consensus_item.evidence == ("E542", "A9")
        assert "(evidence: E542, A9)" in consensus_item.to_text()
        assert packet.section("failures").items[0].evidence == ("A11",)

    def test_conflict_views_render_first(self):
        view = ConflictView(id="C31", status="open", member_ids=("F17", "F198"), summary="F17 says momentum persists; F198 says it is dead.")
        packet = packet_mod.build_context_packet(fused_fixture(), conflicts=[view])
        items = packet.section("conflicts").items
        assert [item.id for item in items] == ["C31", "F198"]
        assert items[0].evidence == ("F17", "F198")
        assert items[0].kind == "conflict"

    def test_flags_rendered(self):
        packet = packet_mod.build_context_packet(fused_fixture())
        text = packet.to_text()
        assert "[validated]" in text
        assert "[failure]" in text
        assert "[disputed]" in text

    def test_to_text_shape(self):
        packet = packet_mod.build_context_packet(fused_fixture(), open_questions=["q?"], topic_key="momentum")
        text = packet.to_text()
        assert text.startswith("RESEARCH CONTEXT\n")
        for title in (
            "Current consensus",
            "Closest prior experiments",
            "Relevant failures / negative evidence",
            "Known contradictions",
            "Important assumptions",
            "Reusable priors / skills",
            "Open questions",
            "Derived summaries (L0/L1)",
        ):
            assert title in text
        assert "pending_phase4" in text
        assert text.rstrip().endswith("]")

    def test_accounting_matches_rendered_text(self):
        packet = packet_mod.build_context_packet(fused_fixture(), open_questions=["a?", "b?"], l1_topics=["t1"])
        rendered = packet.to_text()
        assert packet.tokens_used == packet_mod.estimate_tokens(rendered)
        assert packet.bytes_used == packet_mod.utf8_bytes(rendered)

    def test_tiny_budget_truncates_with_warnings(self):
        packet = packet_mod.build_context_packet(fused_fixture(), open_questions=["q?"], budget=PacketBudget(max_tokens=40, max_bytes=400))
        assert any(section.truncated for section in packet.sections)
        assert packet.warnings
        assert packet.to_dict()["sections"][0]["name"] == "consensus"

    def test_snippet_clipping_marks_ellipsis(self):
        long_text = "word " * 500
        candidate = make_candidate("L1", title="Long", text=long_text)
        _, result = fuse_ids([make_hit(candidate, "lexical", 1)])
        packet = packet_mod.build_context_packet(result, budget=PacketBudget(max_snippet_chars=50))
        snippet = packet.section("consensus").items[0].snippet
        assert len(snippet) <= 50
        assert snippet.endswith("…")

    def test_empty_fusion_yields_empty_sections(self):
        packet = packet_mod.build_context_packet(fusion_mod.FusionResult())
        assert all(section.items == () for section in packet.sections)
        assert all(section.truncated is False for section in packet.sections)
        assert packet.l0.status == "pending_phase4"
        assert packet.within_budget is True
        assert "(none retrieved)" in packet.to_text()

    def test_section_accessor_rejects_unknown(self):
        packet = packet_mod.build_context_packet(fusion_mod.FusionResult())
        with pytest.raises(KnowledgeValidationError):
            packet.section("bogus")

    def test_custom_shares(self):
        shares = {name: 1.0 / 7 for name in SECTION_ORDER}
        packet = packet_mod.build_context_packet(fused_fixture(), section_shares=shares)
        assert packet.within_budget is True

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"conflicts": ["not-a-view"]},
            {"conflicts": "C31"},
            {"open_questions": ["ok", "  "]},
            {"open_questions": "just-ask"},
            {"section_shares": {"consensus": 1.0}},
            {"section_shares": {name: 0.5 for name in SECTION_ORDER}},
            {"section_shares": {**{name: 1.0 / 7 for name in SECTION_ORDER}, "consensus": -1.0}},
            {"l1_topics": ["ok", 42]},
            {"l1_topics": "topic"},
            {"topic_key": 42},
            {"budget": "generous"},
        ],
    )
    def test_validation_errors(self, kwargs):
        with pytest.raises(KnowledgeValidationError):
            packet_mod.build_context_packet(fused_fixture(), **kwargs)

    def test_rejects_non_fused_entries(self):
        with pytest.raises(KnowledgeValidationError):
            packet_mod.build_context_packet([{"id": "x1"}])  # type: ignore[list-item]
        with pytest.raises(KnowledgeValidationError):
            packet_mod.build_context_packet("fused")  # type: ignore[arg-type]


class TestPacketRecords:
    def test_conflict_view_text(self):
        view = ConflictView(id="C1", member_ids=("F1", "F2"), summary="disagreement")
        assert view.to_text() == "- C1 [conflict-set, open]: disagreement (members: F1, F2)"
        assert view.to_dict()["member_ids"] == ["F1", "F2"]
        with pytest.raises(KnowledgeValidationError):
            ConflictView(id="  ")
        with pytest.raises(KnowledgeValidationError):
            ConflictView(id="C1", member_ids=["F1"])  # type: ignore[arg-type]

    def test_placeholder_validation(self):
        with pytest.raises(KnowledgeValidationError):
            SummaryPlaceholder(level="L2")
        with pytest.raises(KnowledgeValidationError):
            SummaryPlaceholder(status="ready")

    def test_item_validation(self):
        with pytest.raises(KnowledgeValidationError):
            packet_mod.PacketItem(id="", kind="finding")
        with pytest.raises(KnowledgeValidationError):
            packet_mod.PacketItem(id="x", kind="finding", flags=["a"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# end-to-end: intent -> plan -> execute -> packet
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_bootstrap_flow(self, corpus, stores):
        plan = planner_mod.plan_retrieval(EVAL_INTENT, top_k=3)
        result = planner_mod.execute_retrieval(
            plan,
            query_text="cross-sectional momentum transaction costs",
            structured=stores["structured"],
            lexical=stores["lexical"],
            vector=stores["vector"],
            failures=stores["failures"],
            query_embedding=(1.0, 0.0, 0.0),
            now=NOW,
        )
        assert len(result.top) == 3
        packet = packet_mod.build_context_packet(
            fusion_mod.FusionResult(candidates=result.top, filtered=result.fusion.filtered, weights=result.fusion.weights),
            open_questions=["Do costs erase momentum out of sample?"],
            topic_key="cross-sectional momentum",
        )
        assert packet.within_budget is True
        assert packet.section("failures").items, "dedicated failure channel must surface negative evidence"
        text = packet.to_text()
        assert "fail_costs" in text
        assert packet.to_dict()["l0"]["status"] == "pending_phase4"

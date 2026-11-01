"""Unit + opt-in live tests for the LLM-judge eval supplement (Phase 5c).

Unmarked tests always run (offline, ``FakeLLMTextClient``): grade-output
parsing, prompt wiring, metric math, and invalid-output handling. Live
tests (``pytest.mark.live``) need ``DEER_FLOW_RUN_LIVE_TESTS=1`` plus a
working ``muse`` CLI and run the real Muse judge over two fixture
queries, asserting valid grade-output shape only — reported metrics are
informational and never gate on thresholds.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from deerflow.knowledge.eval.judge_llm import (
    JudgeReport,
    build_grade_prompt,
    evaluate_judge,
    grade_document,
    parse_grade,
)
from deerflow.knowledge.eval.runner import (
    EvalDocument,
    EvalFixture,
    EvalQuery,
    FixtureError,
    RetrievalFn,
    default_fixture_path,
    keyword_retriever_factory,
    load_fixture,
)
from deerflow.knowledge.llm_text_client import FakeLLMTextClient

LIVE_QUERY_IDS = ("q01_momentum_6m_backtest", "q02_weekly_momentum_turnover")
LIVE_K = 3


def _doc(doc_id: str, kind: str = "experiment", title: str = "T", text: str = "x") -> EvalDocument:
    return EvalDocument(doc_id=doc_id, kind=kind, title=title, text=text)


def _mini_fixture() -> EvalFixture:
    """Two queries over three docs with known gold grades (2/1/0)."""
    return EvalFixture(
        version="0.1.0",
        documents=(
            _doc("exp_a", title="Momentum backtest A", text="momentum backtest sharpe"),
            _doc("exp_b", title="Factor IC B", text="factor information coefficient"),
            _doc("fail_c", kind="failure", title="Costs erase momentum", text="costs turnover erase"),
        ),
        queries=(
            EvalQuery(
                query_id="q_mom",
                query_text="momentum backtest with turnover and costs",
                intent={"topic": "momentum"},
                relevant_doc_ids=("exp_a", "exp_b"),
                grades={"exp_a": 2, "exp_b": 1},
            ),
            EvalQuery(
                query_id="q_cost",
                query_text="why did momentum fail net of costs",
                intent={"topic": "costs"},
                relevant_doc_ids=("fail_c",),
                grades={"fail_c": 2},
            ),
        ),
    )


def _fixed_ranking(order: dict[str, list[str]]) -> RetrievalFn:
    """Build a retrieval fn returning a fixed ranking per query id."""

    def retrieve(query: EvalQuery, k: int) -> list[str]:
        return list(order.get(query.query_id, []))[: max(k, 0)]

    return retrieve


# ---------------------------------------------------------------------------
# Grade-output parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("0", 0),
        ("1", 1),
        ("2", 2),
        ("  2\n", 2),
        ("```\n1\n```", 1),
        ("```json\n2\n```", 2),
        ('{"grade": 2}', 2),
        ('{"grade": 0, "rationale": "off-topic"}', 0),
        ("grade: 1", 1),
        ("Grade = 0.", 0),
        ("GRADE: 2", 2),
    ],
)
def test_parse_grade_accepts_valid_forms(raw: str, want: int) -> None:
    assert parse_grade(raw) == want


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "maybe",
        "3",
        "-1",
        "1.5",
        "12",
        "grade 2 out of 2",
        "the grade is 2",
        '{"grade": 5}',
        '{"grade": "2"}',
        '{"grade": true}',
        "[2]",
        "true",
        "```\n\n```",
        None,
        2,
        ["2"],
    ],
)
def test_parse_grade_rejects_invalid_forms(raw: object) -> None:
    assert parse_grade(raw) is None


# ---------------------------------------------------------------------------
# Prompt wiring + single-document grading
# ---------------------------------------------------------------------------


def test_build_grade_prompt_carries_query_and_doc() -> None:
    fixture = _mini_fixture()
    prompt = build_grade_prompt(fixture.queries[0].query_text, fixture.documents[0])
    assert fixture.queries[0].query_text in prompt
    assert fixture.documents[0].title in prompt
    assert fixture.documents[0].text in prompt
    assert "exactly one digit" in prompt
    assert "Grade (0, 1, or 2)" in prompt


def test_grade_document_returns_parsed_grade_and_records_prompt() -> None:
    fixture = _mini_fixture()
    fake = FakeLLMTextClient(['{"grade": 2}'])
    assert grade_document(fixture.queries[0], fixture.documents[0], fake) == 2
    assert len(fake.calls) == 1
    assert fixture.queries[0].query_text in fake.calls[0]["prompt"]
    assert fixture.documents[0].title in fake.calls[0]["prompt"]


def test_grade_document_invalid_output_is_none() -> None:
    fixture = _mini_fixture()
    fake = FakeLLMTextClient(["definitely not a grade {{{"])
    assert grade_document(fixture.queries[0], fixture.documents[0], fake) is None


def test_grade_document_client_error_is_none_not_raise() -> None:
    fixture = _mini_fixture()
    fake = FakeLLMTextClient([RuntimeError("boom")])
    assert grade_document(fixture.queries[0], fixture.documents[0], fake) is None


def test_grade_document_forwards_timeout_and_max_chars() -> None:
    fixture = _mini_fixture()
    fake = FakeLLMTextClient(["1"])
    assert grade_document(fixture.queries[0], fixture.documents[0], fake, timeout=7.5, max_chars=9) == 1
    assert fake.calls[0]["timeout"] == 7.5
    assert fake.calls[0]["max_chars"] == 9


# ---------------------------------------------------------------------------
# Metric math (FakeLLM judge path)
# ---------------------------------------------------------------------------


def test_perfect_judge_scores_one() -> None:
    fixture = _mini_fixture()
    retrieve = _fixed_ranking({"q_mom": ["exp_a", "exp_b", "fail_c"], "q_cost": ["fail_c", "exp_a"]})
    # Scripted gold grades in grading order: q_mom -> 2/1/0, q_cost -> 2/0.
    fake = FakeLLMTextClient(["2", "1", "0", "2", "0"])
    report = evaluate_judge(fixture, retrieve, fake, 3, retriever_name="fixed")
    assert isinstance(report, JudgeReport)
    assert report.retriever == "fixed"
    assert report.k == 3
    assert report.n_queries == 2
    assert report.n_queries_errored == 0
    assert report.n_graded == 5
    assert report.n_invalid == 0
    assert report.mean_grade_accuracy == 1.0
    assert report.mean_ndcg == 1.0
    assert [q.grade_accuracy for q in report.queries] == [1.0, 1.0]
    assert report.queries[0].gold == {"exp_a": 2, "exp_b": 1, "fail_c": 0}
    assert len(fake.calls) == 5


def test_wrong_judge_scores_zero_accuracy() -> None:
    fixture = _mini_fixture()
    retrieve = _fixed_ranking({"q_mom": ["exp_a", "exp_b"]})
    fake = FakeLLMTextClient(["0", "0"])  # gold is 2/1
    report = evaluate_judge(fixture, retrieve, fake, 2)
    assert report.queries[0].grade_accuracy == 0.0
    assert report.queries[0].n_graded == 2
    # q_cost retrieves nothing: no predictions, metrics None, still counted.
    assert report.queries[1].retrieved == ()
    assert report.queries[1].grade_accuracy is None
    assert report.queries[1].ndcg is None
    # Means average only queries with >= 1 valid prediction.
    assert report.mean_grade_accuracy == 0.0


def test_swapped_grades_ndcg_math() -> None:
    fixture = _mini_fixture()
    retrieve = _fixed_ranking({"q_mom": ["exp_a", "exp_b"]})  # gold 2, 1
    fake = FakeLLMTextClient(["1", "2"])  # judge swaps them: predicted order [exp_b, exp_a]
    report = evaluate_judge(fixture, retrieve, fake, 2)
    score = report.queries[0]
    assert score.grade_accuracy == 0.0
    dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    ideal = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    assert score.ndcg == pytest.approx(dcg / ideal)
    assert 0.0 < score.ndcg < 1.0


def test_ties_keep_retrieval_order_for_ndcg() -> None:
    fixture = _mini_fixture()
    retrieve = _fixed_ranking({"q_mom": ["exp_b", "exp_a"]})  # gold 1, 2 in this order
    fake = FakeLLMTextClient(["1", "1"])  # tie: ranking stays [exp_b, exp_a]
    score = evaluate_judge(fixture, retrieve, fake, 2).queries[0]
    assert score.grade_accuracy == 0.5
    dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    ideal = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    assert score.ndcg == pytest.approx(dcg / ideal)


def test_all_irrelevant_candidates_ndcg_is_one_by_convention() -> None:
    fixture = _mini_fixture()
    retrieve = _fixed_ranking({"q_cost": ["exp_a", "exp_b"]})  # gold 0, 0 for q_cost
    fake = FakeLLMTextClient(["0", "0"])
    score = evaluate_judge(fixture, retrieve, fake, 2).queries[1]
    assert score.grade_accuracy == 1.0
    assert score.ndcg == 1.0  # ideal DCG is 0: trivially perfectly ordered


def test_invalid_outputs_counted_and_excluded_from_accuracy() -> None:
    fixture = _mini_fixture()
    retrieve = _fixed_ranking({"q_mom": ["exp_a", "exp_b", "fail_c"]})
    fake = FakeLLMTextClient(["2", "garbage {{{", RuntimeError("timeout")])
    report = evaluate_judge(fixture, retrieve, fake, 3)
    score = report.queries[0]
    assert score.predicted == {"exp_a": 2, "exp_b": None, "fail_c": None}
    assert score.n_graded == 1
    assert score.n_invalid == 2
    assert score.grade_accuracy == 1.0  # valid-only: the single 2 matches gold
    assert score.ndcg == 1.0  # single valid candidate is trivially ideal
    assert report.n_graded == 1
    assert report.n_invalid == 2


def test_all_invalid_yields_none_metrics() -> None:
    fixture = _mini_fixture()
    retrieve = _fixed_ranking({"q_mom": ["exp_a"], "q_cost": ["fail_c"]})
    fake = FakeLLMTextClient(["nope", "still nope"])
    report = evaluate_judge(fixture, retrieve, fake, 1)
    assert report.n_graded == 0
    assert report.n_invalid == 2
    assert report.mean_grade_accuracy is None
    assert report.mean_ndcg is None
    assert all(q.grade_accuracy is None and q.ndcg is None for q in report.queries)


def test_retriever_error_records_and_skips_grading() -> None:
    fixture = _mini_fixture()

    def retrieve(query: EvalQuery, k: int) -> list[str]:
        _ = k
        if query.query_id == "q_mom":
            raise RuntimeError("boom")
        return ["fail_c"]

    fake = FakeLLMTextClient(["2"])
    report = evaluate_judge(fixture, retrieve, fake, 3)
    assert report.n_queries_errored == 1
    assert "RuntimeError" in report.queries[0].error
    assert report.queries[0].retrieved == ()
    assert report.queries[1].grade_accuracy == 1.0
    assert len(fake.calls) == 1  # errored query triggered no completions


def test_unknown_and_duplicate_ids_never_graded() -> None:
    fixture = _mini_fixture()
    retrieve = _fixed_ranking({"q_mom": ["ghost", "exp_a", "exp_a", "exp_b"]})
    fake = FakeLLMTextClient(["2", "1"])
    score = evaluate_judge(fixture, retrieve, fake, 10).queries[0]
    assert score.retrieved == ("exp_a", "exp_b")  # runner cleaning reused
    assert score.predicted == {"exp_a": 2, "exp_b": 1}
    assert len(fake.calls) == 2


def test_ranking_truncated_to_k_before_grading() -> None:
    fixture = _mini_fixture()
    retrieve = _fixed_ranking({"q_mom": ["exp_a", "exp_b", "fail_c"]})
    fake = FakeLLMTextClient(["2"])
    score = evaluate_judge(fixture, retrieve, fake, 1).queries[0]
    assert score.retrieved == ("exp_a",)
    assert score.n_graded == 1
    assert len(fake.calls) == 1


@pytest.mark.parametrize("bad_k", [0, -3, True, "10"])
def test_invalid_k_rejected(bad_k: object) -> None:
    fixture = _mini_fixture()
    with pytest.raises(FixtureError, match="positive integer"):
        evaluate_judge(fixture, _fixed_ranking({}), FakeLLMTextClient(), bad_k)  # type: ignore[arg-type]


def test_report_serializes_to_json() -> None:
    fixture = _mini_fixture()
    retrieve = _fixed_ranking({"q_mom": ["exp_a", "exp_b"]})
    fake = FakeLLMTextClient(["2", "garbage"])
    report = evaluate_judge(fixture, retrieve, fake, 2)
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["retriever"] == "custom"
    assert payload["mean_grade_accuracy"] == 1.0
    assert payload["queries"][0]["predicted"] == {"exp_a": 2, "exp_b": None}
    assert payload["queries"][0]["gold"] == {"exp_a": 2, "exp_b": 1}


def test_evaluate_judge_is_pure_same_inputs_same_report() -> None:
    fixture = _mini_fixture()
    retrieve = _fixed_ranking({"q_mom": ["exp_a", "exp_b"]})
    first = evaluate_judge(fixture, retrieve, FakeLLMTextClient(["2", "1"]), 2)
    second = evaluate_judge(fixture, retrieve, FakeLLMTextClient(["2", "1"]), 2)
    assert first.to_dict() == second.to_dict()


# ---------------------------------------------------------------------------
# Live tests (opt-in): real `muse exec` judge
# ---------------------------------------------------------------------------


def _require_live() -> None:
    if os.getenv("DEER_FLOW_RUN_LIVE_TESTS") != "1":
        pytest.skip("live knowledge checks are opt-in")


def _live_two_query_fixture() -> EvalFixture:
    full = load_fixture(default_fixture_path())
    wanted = [q for q in full.queries if q.query_id in LIVE_QUERY_IDS]
    assert [q.query_id for q in wanted] == list(LIVE_QUERY_IDS)
    return EvalFixture(version=full.version, documents=full.documents, queries=tuple(wanted))


def _assert_valid_judge_shape(report: JudgeReport) -> None:
    assert isinstance(report, JudgeReport)
    assert report.n_queries == len(LIVE_QUERY_IDS)
    assert report.n_queries_errored == 0
    for score in report.queries:
        assert 0 < len(score.retrieved) <= LIVE_K
        assert set(score.predicted) == set(score.retrieved)
        for grade in score.predicted.values():
            assert grade is None or grade in (0, 1, 2)
        assert score.n_graded + score.n_invalid == len(score.retrieved)
        if score.grade_accuracy is not None:
            assert 0.0 <= score.grade_accuracy <= 1.0
        if score.ndcg is not None:
            assert 0.0 <= score.ndcg <= 1.0
    if report.mean_grade_accuracy is not None:
        assert 0.0 <= report.mean_grade_accuracy <= 1.0
    if report.mean_ndcg is not None:
        assert 0.0 <= report.mean_ndcg <= 1.0


@pytest.mark.live
def test_live_muse_judge_single_grade_shape() -> None:
    _require_live()
    from app.gateway.knowledge_llm_client import MuseIntentClient

    full = load_fixture(default_fixture_path())
    query = next(q for q in full.queries if q.query_id == LIVE_QUERY_IDS[0])
    doc = next(d for d in full.documents if d.doc_id == query.relevant_doc_ids[0])
    grade = grade_document(query, doc, MuseIntentClient(default_timeout=300.0))
    assert grade is None or grade in (0, 1, 2)
    print(f"live judge smoke: query={query.query_id} doc={doc.doc_id} gold={query.grades.get(doc.doc_id)} predicted={grade}")


@pytest.mark.live
def test_live_muse_judge_two_fixture_queries_shape_only() -> None:
    _require_live()
    from app.gateway.knowledge_llm_client import MuseIntentClient

    fixture = _live_two_query_fixture()
    retrieve = keyword_retriever_factory(fixture)
    report = evaluate_judge(
        fixture,
        retrieve,
        MuseIntentClient(default_timeout=300.0),
        LIVE_K,
        retriever_name="keyword",
    )
    _assert_valid_judge_shape(report)
    # Informational only: printed for the operator, never gated on thresholds.
    for score in report.queries:
        print(f"live judge: query={score.query_id} retrieved={list(score.retrieved)} gold={score.gold} predicted={score.predicted} accuracy={score.grade_accuracy} ndcg={score.ndcg}")
    print(f"live judge totals: graded={report.n_graded} invalid={report.n_invalid} mean_accuracy={report.mean_grade_accuracy} mean_ndcg={report.mean_ndcg}")

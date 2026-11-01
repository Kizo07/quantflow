"""LLM-as-judge eval supplement for the knowledge retrieval eval (Phase 5c).

Scores an LLM judge's ability to reproduce the fixture's per-document
relevance grades (``1``/``2`` recorded in ``eval_fixture.json``; ``0``
means "not relevant", i.e. any retrieved document without a judgment)
for whatever a retrieval callable returns.

This module sits *beside* :mod:`runner` — it imports the runner's
fixture/scoring types but the runner itself stays stdlib-only and
deterministic (untouched by this phase). The judge side is a pure
function of ``(retrieval fn, fixture, client)``: no I/O, no globals,
no prompt caching. The client is the harness
:class:`~deerflow.knowledge.llm_text_client.LLMTextClient` protocol, so
unit tests bind :class:`FakeLLMTextClient
<deerflow.knowledge.llm_text_client.FakeLLMTextClient>` (the FakeLLM
judge path) while live runs bind the app-layer ``muse exec`` client.

Grading protocol: one completion per (query, retrieved document), each
asking for exactly one digit (``0``/``1``/``2``). A completion that is
unparseable — or a client that raises — grades as ``None`` (invalid):
it is counted, excluded from accuracy, and never aborts the run.

Metrics (see :class:`JudgeReport`):

- ``grade_accuracy``: fraction of valid predictions equal to the gold
  grade (``None`` when a query — or the whole run — has no valid
  prediction).
- ``ndcg``: per query, retrieved documents with valid predictions are
  re-ranked by predicted grade (stable: ties keep retrieval order) and
  scored with gold gains ``2**grade - 1`` and ``log2(rank + 1)``
  discounts, normalized by the ideal (gold-sorted) DCG over the same
  candidate set. ``None`` when there is no valid prediction;
  ``1.0`` when the ideal DCG is 0 (an all-irrelevant candidate set is
  trivially perfectly ordered — retrieval misses are the runner's
  recall, not the judge's).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

try:  # Normal path: imported as deerflow.knowledge.eval.judge_llm.
    from deerflow.knowledge.eval.runner import (
        EvalDocument,
        EvalFixture,
        EvalQuery,
        FixtureError,
        RetrievalFn,
        score_query,
    )
    from deerflow.knowledge.llm_text_client import (
        DEFAULT_LLM_MAX_CHARS,
        DEFAULT_LLM_TIMEOUT,
        LLMTextClient,
    )
except ImportError:  # Standalone/script path: load sibling files by location.
    import importlib.util
    from pathlib import Path

    def _load_sibling(name: str, path: Path):  # type: ignore[no-untyped-def]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError(f"cannot load sibling module {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    _HERE = Path(__file__).resolve().parent
    _runner = _load_sibling("kp_eval_runner", _HERE / "runner.py")
    _llm = _load_sibling("kp_llm_text_client", _HERE.parent / "llm_text_client.py")
    EvalDocument = _runner.EvalDocument
    EvalFixture = _runner.EvalFixture
    EvalQuery = _runner.EvalQuery
    FixtureError = _runner.FixtureError
    RetrievalFn = _runner.RetrievalFn
    score_query = _runner.score_query
    DEFAULT_LLM_MAX_CHARS = _llm.DEFAULT_LLM_MAX_CHARS
    DEFAULT_LLM_TIMEOUT = _llm.DEFAULT_LLM_TIMEOUT
    LLMTextClient = _llm.LLMTextClient

__all__ = [
    "GRADE_LABELS",
    "JudgeQueryScore",
    "JudgeReport",
    "build_grade_prompt",
    "evaluate_judge",
    "grade_document",
    "parse_grade",
]

#: Human meaning of each judge grade (mirrors the fixture rubric).
GRADE_LABELS = {
    0: "not relevant to the research question",
    1: "partially relevant (useful background)",
    2: "highly relevant (directly answers or warns)",
}

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
_GRADE_PREFIX_RE = re.compile(r"^\s*grade\s*[:=]\s*([012])\s*\.?\s*$", re.IGNORECASE)


def build_grade_prompt(query_text: str, doc: EvalDocument) -> str:
    """Build the one-shot grading prompt for (query, document).

    Asks for exactly one digit so both scripted fakes and live models
    have an unambiguous output contract.
    """
    return (
        "You grade how relevant a retrieved knowledge document is to a research question.\n"
        "Reply with exactly one digit and nothing else.\n"
        "Grades: 0 = not relevant to the research question; "
        "1 = partially relevant (useful background); "
        "2 = highly relevant (directly answers it or warns against a pitfall).\n"
        f"Research question: {query_text.strip()}\n"
        f"Document title: {doc.title.strip()}\n"
        f"Document text: {doc.text.strip()}\n"
        "Grade (0, 1, or 2):"
    )


def parse_grade(text: object) -> int | None:
    """Parse a judge completion into a grade, or ``None`` when invalid.

    Accepted (after optional surrounding whitespace): a bare ``0``/``1``/``2``,
    a single fenced block holding one of those (or equivalent JSON), a JSON
    ``{"grade": N}`` object, a top-level JSON ``N``, or a ``grade: N`` line.
    Anything else — empty, out of range, floats, prose — is ``None``.
    """
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    if not candidate:
        return None
    fence = _FENCE_RE.match(candidate)
    if fence:
        candidate = fence.group(1).strip()
        if not candidate:
            return None
    if candidate in ("0", "1", "2"):
        return int(candidate)
    try:
        payload: Any = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        payload = None
    if isinstance(payload, bool):
        return None
    if isinstance(payload, int) and payload in (0, 1, 2):
        return int(payload)
    if isinstance(payload, dict):
        grade = payload.get("grade")
        if isinstance(grade, bool):
            return None
        if isinstance(grade, int) and grade in (0, 1, 2):
            return int(grade)
    prefixed = _GRADE_PREFIX_RE.match(candidate)
    if prefixed:
        return int(prefixed.group(1))
    return None


def grade_document(
    query: EvalQuery,
    doc: EvalDocument,
    client: LLMTextClient,
    *,
    timeout: float = DEFAULT_LLM_TIMEOUT,
    max_chars: int = DEFAULT_LLM_MAX_CHARS,
) -> int | None:
    """Grade one retrieved document with the LLM judge.

    Returns the parsed grade, or ``None`` when the client raises or its
    completion is unparseable. Never raises for judge-side failures.
    """
    try:
        completion = client.complete(
            build_grade_prompt(query.query_text, doc),
            timeout=timeout,
            max_chars=max_chars,
        )
    except Exception:  # noqa: BLE001 - judge failures degrade the score, not the run
        return None
    return parse_grade(completion)


def _dcg(gold_in_rank_order: list[int]) -> float:
    """Discounted cumulative gain with ``2**grade - 1`` gains."""
    return sum((2**g - 1) / math.log2(rank + 2) for rank, g in enumerate(gold_in_rank_order))


@dataclass(frozen=True)
class JudgeQueryScore:
    """Per-query judge outcome over the retriever's top-``k``."""

    query_id: str
    k: int
    retrieved: tuple[str, ...]  # cleaned ranking actually graded (<= k known IDs)
    gold: dict[str, int] = field(default_factory=dict)  # doc_id -> fixture grade (0 when unjudged)
    predicted: dict[str, int | None] = field(default_factory=dict)  # doc_id -> judge grade (None = invalid)
    n_graded: int = 0  # retrieved docs with a valid prediction
    n_invalid: int = 0  # retrieved docs graded None (unparseable output or client error)
    grade_accuracy: float | None = None  # valid-only exact-match rate (None when n_graded == 0)
    ndcg: float | None = None  # predicted-grade ranking vs ideal (None when n_graded == 0)
    error: str = ""  # non-empty when the retriever raised on this query


@dataclass(frozen=True)
class JudgeReport:
    """Whole-fixture judge outcome for one retriever at one cutoff."""

    retriever: str
    k: int
    n_queries: int
    mean_grade_accuracy: float | None  # mean over queries with >= 1 valid prediction
    mean_ndcg: float | None  # mean over queries with >= 1 valid prediction
    n_graded: int
    n_invalid: int
    queries: tuple[JudgeQueryScore, ...]
    n_queries_errored: int

    def to_dict(self) -> dict:
        """Return a JSON-serializable mapping of this report."""
        return {
            "retriever": self.retriever,
            "k": self.k,
            "n_queries": self.n_queries,
            "mean_grade_accuracy": round(self.mean_grade_accuracy, 6) if self.mean_grade_accuracy is not None else None,
            "mean_ndcg": round(self.mean_ndcg, 6) if self.mean_ndcg is not None else None,
            "n_graded": self.n_graded,
            "n_invalid": self.n_invalid,
            "n_queries_errored": self.n_queries_errored,
            "queries": [
                {
                    "query_id": q.query_id,
                    "k": q.k,
                    "retrieved": list(q.retrieved),
                    "gold": dict(q.gold),
                    "predicted": dict(q.predicted),
                    "n_graded": q.n_graded,
                    "n_invalid": q.n_invalid,
                    "grade_accuracy": round(q.grade_accuracy, 6) if q.grade_accuracy is not None else None,
                    "ndcg": round(q.ndcg, 6) if q.ndcg is not None else None,
                    "error": q.error,
                }
                for q in self.queries
            ],
        }


def _score_one_query(
    query: EvalQuery,
    retrieve: RetrievalFn,
    index: dict[str, EvalDocument],
    known: frozenset[str],
    k: int,
    client: LLMTextClient,
    *,
    timeout: float,
    max_chars: int,
) -> JudgeQueryScore:
    """Grade the retriever's cleaned top-``k`` for one query."""
    base = score_query(query, retrieve, known, k)
    if base.error:
        return JudgeQueryScore(query_id=query.query_id, k=k, retrieved=(), error=base.error)
    gold = {doc_id: query.grades.get(doc_id, 0) for doc_id in base.retrieved}
    predicted: dict[str, int | None] = {}
    for doc_id in base.retrieved:
        predicted[doc_id] = grade_document(query, index[doc_id], client, timeout=timeout, max_chars=max_chars)
    valid = [(doc_id, grade) for doc_id, grade in predicted.items() if grade is not None]
    n_invalid = len(predicted) - len(valid)
    if not valid:
        return JudgeQueryScore(
            query_id=query.query_id,
            k=k,
            retrieved=base.retrieved,
            gold=gold,
            predicted=predicted,
            n_graded=0,
            n_invalid=n_invalid,
        )
    accuracy = sum(1 for doc_id, grade in valid if grade == gold[doc_id]) / len(valid)
    ranked = sorted(valid, key=lambda item: -item[1])  # stable: ties keep retrieval order
    ideal = sorted((gold[doc_id] for doc_id, _ in valid), reverse=True)
    ideal_dcg = _dcg(ideal)
    ndcg = _dcg([gold[doc_id] for doc_id, _ in ranked]) / ideal_dcg if ideal_dcg > 0 else 1.0
    return JudgeQueryScore(
        query_id=query.query_id,
        k=k,
        retrieved=base.retrieved,
        gold=gold,
        predicted=predicted,
        n_graded=len(valid),
        n_invalid=n_invalid,
        grade_accuracy=accuracy,
        ndcg=ndcg,
    )


def evaluate_judge(
    fixture: EvalFixture,
    retrieve: RetrievalFn,
    client: LLMTextClient,
    k: int,
    *,
    retriever_name: str = "custom",
    timeout: float = DEFAULT_LLM_TIMEOUT,
    max_chars: int = DEFAULT_LLM_MAX_CHARS,
) -> JudgeReport:
    """Grade ``retrieve``'s top-``k`` per query with the LLM judge.

    Args:
        fixture: Validated eval fixture (gold grades live here).
        retrieve: Ranking callable under test (runner ``RetrievalFn``).
        client: Judge completion client (fake-bindable protocol).
        k: Cutoff; must be a positive integer.
        retriever_name: Label recorded on the report.
        timeout: Per-completion timeout forwarded to the client.
        max_chars: Per-completion cap forwarded to the client.

    Raises:
        FixtureError: If ``k`` is not a positive integer.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise FixtureError(f"k must be a positive integer, got {k!r}")
    index = fixture.doc_index()
    known = fixture.doc_ids
    scores = tuple(_score_one_query(q, retrieve, index, known, k, client, timeout=timeout, max_chars=max_chars) for q in fixture.queries)
    with_valid = [s for s in scores if s.n_graded > 0]
    accuracies = [s.grade_accuracy for s in with_valid if s.grade_accuracy is not None]
    ndcgs = [s.ndcg for s in with_valid if s.ndcg is not None]
    return JudgeReport(
        retriever=retriever_name,
        k=k,
        n_queries=len(scores),
        mean_grade_accuracy=(sum(accuracies) / len(accuracies)) if accuracies else None,
        mean_ndcg=(sum(ndcgs) / len(ndcgs)) if ndcgs else None,
        n_graded=sum(s.n_graded for s in scores),
        n_invalid=sum(s.n_invalid for s in scores),
        queries=scores,
        n_queries_errored=sum(1 for s in scores if s.error),
    )

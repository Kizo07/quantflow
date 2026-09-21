"""Phase 0 retrieval-eval runner for the Quantflow Research Knowledge Plane.

Scores a retrieval callable against the curated fixture in
``eval_fixture.json`` (known prior alpha_engine experiments, failures, and
findings paired with research-intent queries and relevance judgments).

Metrics:

- ``recall@k``: per query, the fraction of judged-relevant documents that
  appear in the retriever's top-``k`` ranking, averaged over queries.
- ``failure-recall@k``: per query, the fraction of must-recall failure
  documents retrieved in the top ``k``, averaged over queries that name at
  least one must-recall failure.

Phase 0 exit behaviour (see ``implementation_plan.md`` Phase 0): with no
retrieval wired, the default ``null`` retriever returns nothing, both means
are ``0.0``, and this runner exits **nonzero** (exit code 2). Phase 2 plugs
a real retriever in (see ``README.md``) and the same command turns green.

Usage::

    python runner.py                          # null baseline -> exit 2
    python runner.py --retriever oracle       # perfect ranking -> exit 0
    python runner.py --retriever keyword      # token-overlap smoke baseline
    python runner.py --self-test              # embedded unit-test suite
    python runner.py --json                   # machine-readable report

The runner is stdlib-only and standalone: it makes no imports from the
``deerflow`` package so it runs before any integration wiring exists.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

FIXTURE_FILENAME = "eval_fixture.json"
FIXTURE_VERSION = "0.1.0"
DEFAULT_K = 10
DEFAULT_TARGET = 0.8

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_BELOW_TARGET = 2

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


class FixtureError(ValueError):
    """Raised when the eval fixture is missing, unreadable, or invalid."""


@dataclass(frozen=True)
class EvalDocument:
    """One retrievable corpus item: a prior experiment, failure, or finding."""

    doc_id: str
    kind: str  # "experiment" | "failure" | "finding"
    title: str
    text: str  # retrieval surface: title + summary + key terms
    tool: str = ""  # originating alpha_engine MCP tool, when applicable
    failure_class: str = ""  # set for kind == "failure"


@dataclass(frozen=True)
class EvalQuery:
    """One eval case: a research-intent query with relevance judgments."""

    query_id: str
    query_text: str
    intent: dict
    relevant_doc_ids: tuple[str, ...]
    must_recall_failures: tuple[str, ...] = ()
    grades: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalFixture:
    """Validated in-memory fixture: corpus documents plus eval queries."""

    version: str
    documents: tuple[EvalDocument, ...]
    queries: tuple[EvalQuery, ...]

    @property
    def doc_ids(self) -> frozenset[str]:
        """IDs of every document in the corpus."""
        return frozenset(d.doc_id for d in self.documents)

    def doc_index(self) -> dict[str, EvalDocument]:
        """Map each corpus ID to its document."""
        return {d.doc_id: d for d in self.documents}


class RetrievalFn(Protocol):
    """Ranking callable under test.

    Given a query and a cutoff ``k``, return ranked document IDs, best
    first. Unknown IDs are ignored (and counted); duplicates are dropped
    keeping the first occurrence; rankings longer than ``k`` are
    truncated. Raising on a query scores that query as zero and records
    the error on the report instead of aborting the run.
    """

    def __call__(self, query: EvalQuery, k: int) -> list[str]: ...


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise FixtureError(msg)


def _require_str(value: object, what: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{what} must be a non-empty string")
    return str(value).strip()


def load_fixture(path: str | Path) -> EvalFixture:
    """Load and validate the eval fixture at ``path``.

    Args:
        path: Path to the fixture JSON file.

    Returns:
        The validated :class:`EvalFixture`.

    Raises:
        FixtureError: If the file is missing, is not valid JSON, or fails
            schema validation (bad version, duplicate/empty IDs, dangling
            relevance judgments, queries without relevant documents, or a
            research intent missing required keys).
    """
    raw_path = Path(path)
    try:
        text = raw_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FixtureError(f"fixture not found: {raw_path}") from exc
    except OSError as exc:
        raise FixtureError(f"cannot read fixture {raw_path}: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FixtureError(f"fixture {raw_path} is not valid JSON: {exc}") from exc
    _require(isinstance(payload, dict), "fixture root must be a JSON object")

    version = _require_str(payload.get("version"), "fixture 'version'")
    _require(version == FIXTURE_VERSION, f"unsupported fixture version {version!r}; want {FIXTURE_VERSION!r}")

    raw_docs = payload.get("documents")
    _require(isinstance(raw_docs, list) and raw_docs, "fixture 'documents' must be a non-empty list")
    documents: list[EvalDocument] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_docs):
        _require(isinstance(raw, dict), f"documents[{i}] must be an object")
        doc_id = _require_str(raw.get("doc_id"), f"documents[{i}].'doc_id'")
        _require(doc_id not in seen, f"duplicate doc_id {doc_id!r}")
        seen.add(doc_id)
        kind = _require_str(raw.get("kind"), f"documents[{i}].'kind'")
        _require(kind in ("experiment", "failure", "finding"), f"documents[{i}].'kind' must be experiment|failure|finding, got {kind!r}")
        documents.append(
            EvalDocument(
                doc_id=doc_id,
                kind=kind,
                title=_require_str(raw.get("title"), f"documents[{i}].'title'"),
                text=_require_str(raw.get("text"), f"documents[{i}].'text'"),
                tool=str(raw.get("tool", "") or ""),
                failure_class=str(raw.get("failure_class", "") or ""),
            )
        )
    if any(d.kind == "failure" for d in documents):
        for d in documents:
            if d.kind == "failure":
                _require(bool(d.failure_class.strip()), f"failure {d.doc_id!r} must set 'failure_class'")

    raw_queries = payload.get("queries")
    _require(isinstance(raw_queries, list) and raw_queries, "fixture 'queries' must be a non-empty list")
    required_intent = ("topic", "asset_class", "markets", "universe", "horizon", "frequency", "concepts", "requested_period", "needed_memory")
    queries: list[EvalQuery] = []
    seen_q: set[str] = set()
    for i, raw in enumerate(raw_queries):
        _require(isinstance(raw, dict), f"queries[{i}] must be an object")
        query_id = _require_str(raw.get("query_id"), f"queries[{i}].'query_id'")
        _require(query_id not in seen_q, f"duplicate query_id {query_id!r}")
        seen_q.add(query_id)
        intent = raw.get("intent")
        _require(isinstance(intent, dict), f"queries[{i}].'intent' must be an object")
        missing = [k for k in required_intent if k not in intent]
        _require(not missing, f"queries[{i}].'intent' missing keys: {', '.join(missing)}")
        judgments = raw.get("relevant")
        _require(isinstance(judgments, list) and judgments, f"queries[{i}].'relevant' must be a non-empty list")
        relevant: list[str] = []
        grades: dict[str, int] = {}
        for j, judge in enumerate(judgments):
            _require(isinstance(judge, dict), f"queries[{i}].'relevant'[{j}] must be an object")
            doc_id = _require_str(judge.get("doc_id"), f"queries[{i}].'relevant'[{j}].'doc_id'")
            _require(doc_id in seen, f"queries[{i}].'relevant'[{j}] dangles: unknown doc_id {doc_id!r}")
            _require(doc_id not in relevant, f"queries[{i}].'relevant'[{j}] duplicates doc_id {doc_id!r}")
            grade = judge.get("grade", 1)
            _require(grade in (1, 2), f"queries[{i}].'relevant'[{j}].'grade' must be 1 or 2, got {grade!r}")
            relevant.append(doc_id)
            grades[doc_id] = int(grade)
        failures = raw.get("must_recall_failures", [])
        _require(isinstance(failures, list), f"queries[{i}].'must_recall_failures' must be a list")
        must: list[str] = []
        for f in failures:
            _require(isinstance(f, str) and f.strip(), f"queries[{i}].'must_recall_failures' entries must be non-empty strings")
            fid = f.strip()
            _require(fid in seen, f"queries[{i}].'must_recall_failures' dangles: unknown doc_id {fid!r}")
            _require(fid not in must, f"queries[{i}].'must_recall_failures' duplicates doc_id {fid!r}")
            must.append(fid)
        queries.append(
            EvalQuery(
                query_id=query_id,
                query_text=_require_str(raw.get("query_text"), f"queries[{i}].'query_text'"),
                intent=dict(intent),
                relevant_doc_ids=tuple(relevant),
                must_recall_failures=tuple(must),
                grades=grades,
            )
        )
    return EvalFixture(version=version, documents=tuple(documents), queries=tuple(queries))


def default_fixture_path() -> Path:
    """Return the ``eval_fixture.json`` sibling of this runner."""
    return Path(__file__).resolve().parent / FIXTURE_FILENAME


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------


def null_retriever(query: EvalQuery, k: int) -> list[str]:
    """Phase 0 baseline: nothing is wired, so retrieve nothing.

    Scoring this retriever must yield ``recall@k == 0`` and
    ``failure-recall@k == 0`` so the runner exits nonzero until Phase 2
    plugs real retrieval in.
    """
    _ = (query, k)
    return []


def oracle_retriever_factory(fixture: EvalFixture) -> RetrievalFn:
    """Build a perfect retriever that returns each query's gold ranking.

    Relevant documents come first (grade 2 before grade 1), then any
    must-recall failures not already listed. Used to validate the
    scoring math and the ``--target`` exit path; not a retrieval method.
    """

    def retrieve(query: EvalQuery, k: int) -> list[str]:
        ranked = sorted(query.relevant_doc_ids, key=lambda d: (-query.grades.get(d, 1), d))
        for fid in query.must_recall_failures:
            if fid not in query.relevant_doc_ids:
                ranked.append(fid)
        return ranked[: max(k, 0)]

    _ = fixture
    return retrieve


def _tokenize(text: str) -> frozenset[str]:
    """Lowercase alphanumeric token set for the keyword smoke baseline."""
    return frozenset(_TOKEN_RE.findall(text.lower()))


def keyword_retriever_factory(fixture: EvalFixture) -> RetrievalFn:
    """Build a token-overlap smoke baseline over the fixture corpus.

    Ranks documents by shared tokens between the query surface
    (``query_text`` plus intent topic/concepts) and each document's
    ``text``, breaking ties by ``doc_id`` for determinism. This is a
    wiring smoke test only — it has no scope filtering, rank fusion, or
    failure channel, so it is not expected to meet the Phase 2 target.
    """
    doc_tokens = {d.doc_id: _tokenize(f"{d.title} {d.text}") for d in fixture.documents}
    doc_ids = sorted(doc_tokens)

    def retrieve(query: EvalQuery, k: int) -> list[str]:
        surface = " ".join(
            [
                query.query_text,
                str(query.intent.get("topic", "")),
                " ".join(str(c) for c in query.intent.get("concepts", [])),
            ]
        )
        q_tokens = _tokenize(surface)
        scored = sorted(doc_ids, key=lambda d: (-len(q_tokens & doc_tokens[d]), d))
        return scored[: max(k, 0)]

    return retrieve


RETRIEVERS: dict[str, str] = {
    "null": "Phase 0 baseline: retrieve nothing (default; exits nonzero).",
    "oracle": "Perfect gold ranking (validates scoring math and exit path).",
    "keyword": "Token-overlap smoke baseline (deterministic, below-target).",
}


def build_retriever(name: str, fixture: EvalFixture) -> RetrievalFn:
    """Build a named built-in retriever for ``fixture``.

    Raises:
        FixtureError: If ``name`` is not a known retriever.
    """
    if name == "null":
        return null_retriever
    if name == "oracle":
        return oracle_retriever_factory(fixture)
    if name == "keyword":
        return keyword_retriever_factory(fixture)
    raise FixtureError(f"unknown retriever {name!r}; want one of {', '.join(sorted(RETRIEVERS))}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryScore:
    """Per-query outcome at cutoff ``k``."""

    query_id: str
    k: int
    retrieved: tuple[str, ...]  # cleaned ranking actually scored (<= k known IDs)
    n_relevant: int
    n_relevant_hit: int
    recall: float
    n_failures: int
    n_failures_hit: int
    failure_recall: float | None  # None when the query names no must-recall failures
    missed_failures: tuple[str, ...]
    unknown_ids_ignored: int
    duplicates_dropped: int
    error: str = ""  # non-empty when the retriever raised on this query


@dataclass(frozen=True)
class EvalReport:
    """Whole-fixture outcome for one retriever at one cutoff."""

    retriever: str
    k: int
    target: float
    n_queries: int
    n_failure_queries: int
    mean_recall: float
    mean_failure_recall: float | None  # None when no query names failures
    queries: tuple[QueryScore, ...]
    n_queries_errored: int
    n_unknown_ids_ignored: int

    @property
    def passed(self) -> bool:
        """True when both means meet ``target`` (and nothing errored)."""
        if self.n_queries_errored:
            return False
        if self.mean_recall < self.target:
            return False
        if self.mean_failure_recall is not None and self.mean_failure_recall < self.target:
            return False
        return True

    def to_dict(self) -> dict:
        """Return a JSON-serializable mapping of this report."""
        return {
            "retriever": self.retriever,
            "k": self.k,
            "target": self.target,
            "n_queries": self.n_queries,
            "n_failure_queries": self.n_failure_queries,
            "mean_recall": round(self.mean_recall, 6),
            "mean_failure_recall": round(self.mean_failure_recall, 6) if self.mean_failure_recall is not None else None,
            "passed": self.passed,
            "n_queries_errored": self.n_queries_errored,
            "n_unknown_ids_ignored": self.n_unknown_ids_ignored,
            "queries": [
                {
                    "query_id": q.query_id,
                    "k": q.k,
                    "retrieved": list(q.retrieved),
                    "n_relevant": q.n_relevant,
                    "n_relevant_hit": q.n_relevant_hit,
                    "recall": round(q.recall, 6),
                    "n_failures": q.n_failures,
                    "n_failures_hit": q.n_failures_hit,
                    "failure_recall": round(q.failure_recall, 6) if q.failure_recall is not None else None,
                    "missed_failures": list(q.missed_failures),
                    "unknown_ids_ignored": q.unknown_ids_ignored,
                    "duplicates_dropped": q.duplicates_dropped,
                    "error": q.error,
                }
                for q in self.queries
            ],
        }


def _clean_ranking(raw: object, known: frozenset[str], k: int) -> tuple[tuple[str, ...], int, int]:
    """Deduplicate, filter to known IDs, and truncate a raw ranking.

    Returns ``(cleaned, unknown_ignored, duplicates_dropped)``. Non-list
    rankings and non-string entries count as unknown rather than raising,
    so a misbehaving retriever degrades the score instead of the run.
    """
    items = list(raw) if isinstance(raw, (list, tuple)) else []
    non_list_penalty = 0 if isinstance(raw, (list, tuple)) else 1
    cleaned: list[str] = []
    seen: set[str] = set()
    unknown = non_list_penalty
    dups = 0
    for entry in items:
        if not isinstance(entry, str) or entry not in known:
            unknown += 1
            continue
        if entry in seen:
            dups += 1
            continue
        seen.add(entry)
        cleaned.append(entry)
        if len(cleaned) >= k:
            break
    return tuple(cleaned), unknown, dups


def score_query(query: EvalQuery, retrieve: RetrievalFn, known: frozenset[str], k: int) -> QueryScore:
    """Score one query, converting retriever failures into zero scores."""
    try:
        raw = retrieve(query, k)
    except Exception as exc:  # noqa: BLE001 - a raising retriever must not abort the run
        return QueryScore(
            query_id=query.query_id,
            k=k,
            retrieved=(),
            n_relevant=len(query.relevant_doc_ids),
            n_relevant_hit=0,
            recall=0.0,
            n_failures=len(query.must_recall_failures),
            n_failures_hit=0,
            failure_recall=0.0 if query.must_recall_failures else None,
            missed_failures=query.must_recall_failures,
            unknown_ids_ignored=0,
            duplicates_dropped=0,
            error=f"{type(exc).__name__}: {exc}",
        )
    cleaned, unknown, dups = _clean_ranking(raw, known, k)
    hit = set(cleaned) & set(query.relevant_doc_ids)
    missed = tuple(f for f in query.must_recall_failures if f not in set(cleaned))
    n_fail = len(query.must_recall_failures)
    return QueryScore(
        query_id=query.query_id,
        k=k,
        retrieved=cleaned,
        n_relevant=len(query.relevant_doc_ids),
        n_relevant_hit=len(hit),
        recall=len(hit) / len(query.relevant_doc_ids),
        n_failures=n_fail,
        n_failures_hit=n_fail - len(missed),
        failure_recall=(n_fail - len(missed)) / n_fail if n_fail else None,
        missed_failures=missed,
        unknown_ids_ignored=unknown,
        duplicates_dropped=dups,
    )


def evaluate(fixture: EvalFixture, retrieve: RetrievalFn, k: int, retriever_name: str = "custom", target: float = DEFAULT_TARGET) -> EvalReport:
    """Score ``retrieve`` over every fixture query at cutoff ``k``.

    Args:
        fixture: Validated eval fixture.
        retrieve: Ranking callable under test.
        k: Cutoff; must be a positive integer.
        retriever_name: Label recorded on the report.
        target: Recall target recorded (and checked) on the report.

    Raises:
        FixtureError: If ``k`` is not a positive integer.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise FixtureError(f"k must be a positive integer, got {k!r}")
    known = fixture.doc_ids
    scores = tuple(score_query(q, retrieve, known, k) for q in fixture.queries)
    failure_scores = [s for s in scores if s.failure_recall is not None]
    return EvalReport(
        retriever=retriever_name,
        k=k,
        target=float(target),
        n_queries=len(scores),
        n_failure_queries=len(failure_scores),
        mean_recall=sum(s.recall for s in scores) / len(scores),
        mean_failure_recall=(sum(s.failure_recall for s in failure_scores if s.failure_recall is not None) / len(failure_scores)) if failure_scores else None,
        queries=scores,
        n_queries_errored=sum(1 for s in scores if s.error),
        n_unknown_ids_ignored=sum(s.unknown_ids_ignored for s in scores),
    )


def format_text_report(report: EvalReport) -> str:
    """Render ``report`` as a human-readable summary with per-query rows."""
    lines = [
        f"retriever={report.retriever} k={report.k} target={report.target:.2f} queries={report.n_queries}",
        f"mean recall@{report.k}: {report.mean_recall:.4f}",
    ]
    if report.mean_failure_recall is None:
        lines.append(f"mean failure-recall@{report.k}: n/a (no must-recall failures)")
    else:
        lines.append(f"mean failure-recall@{report.k}: {report.mean_failure_recall:.4f} over {report.n_failure_queries} queries")
    lines.append("")
    lines.append(f"{'query':34} {'recall':>7} {'fail-r':>7} {'hits':>8} {'missed-failures'}")
    for q in report.queries:
        fail = "n/a" if q.failure_recall is None else f"{q.failure_recall:.3f}"
        missed = ",".join(q.missed_failures) if q.missed_failures else "-"
        err = f" ERROR {q.error}" if q.error else ""
        lines.append(f"{q.query_id:34} {q.recall:7.3f} {fail:>7} {q.n_relevant_hit:>3}/{q.n_relevant:<3} {missed}{err}")
    lines.append("")
    if report.n_queries_errored:
        lines.append(f"VERDICT: FAIL ({report.n_queries_errored} queries errored)")
    elif report.passed:
        lines.append("VERDICT: PASS")
    else:
        lines.append("VERDICT: FAIL (below target — retrieval not wired or not good enough yet)")
    if report.n_unknown_ids_ignored:
        lines.append(f"note: ignored {report.n_unknown_ids_ignored} unknown ranking ids")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="runner.py",
        description="Phase 0 retrieval-eval runner: score a retriever against eval_fixture.json.",
    )
    parser.add_argument("--fixture", default=str(default_fixture_path()), help="Path to eval_fixture.json (default: sibling of this runner).")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"Retrieval cutoff (default: {DEFAULT_K}).")
    parser.add_argument("--retriever", choices=sorted(RETRIEVERS), default="null", help="Built-in retriever to score. " + " ".join(f"{n}: {d}" for n, d in sorted(RETRIEVERS.items())))
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET, help=f"Pass target for both means (default: {DEFAULT_TARGET}).")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON instead of text.")
    parser.add_argument("--self-test", action="store_true", help="Run the embedded unit-test suite and exit.")
    parser.add_argument("--list-queries", action="store_true", help="List fixture query IDs and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0/1/2)."""
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()
    try:
        fixture = load_fixture(args.fixture)
    except FixtureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if args.list_queries:
        for q in fixture.queries:
            print(f"{q.query_id}\t{q.query_text}")
        return EXIT_OK
    try:
        retrieve = build_retriever(args.retriever, fixture)
        report = evaluate(fixture, retrieve, args.k, retriever_name=args.retriever, target=args.target)
    except FixtureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=False))
    else:
        print(format_text_report(report), end="")
    return EXIT_OK if report.passed else EXIT_BELOW_TARGET


# ---------------------------------------------------------------------------
# Self-test (stdlib unittest; `python runner.py --self-test`)
# ---------------------------------------------------------------------------


def _mini_intent(topic: str, concepts: list[str], needed: list[str]) -> dict:
    """Build a minimal valid research intent for self-test queries."""
    return {"topic": topic, "asset_class": "equity", "markets": ["US"], "universe": "sp500", "horizon": "6m", "frequency": "daily", "concepts": concepts, "requested_period": ["2020-01-01", "2025-12-31"], "needed_memory": needed}


def _mini_fixture_dict(**overrides: object) -> dict:
    """Build a minimal valid fixture payload, with per-test overrides."""
    payload: dict = {
        "version": FIXTURE_VERSION,
        "documents": [
            {"doc_id": "exp_a", "kind": "experiment", "title": "Momentum backtest A", "text": "cross-sectional momentum backtest lookback 126 top 20 monthly sharpe", "tool": "run_momentum_backtest"},
            {"doc_id": "exp_b", "kind": "experiment", "title": "Factor IC B", "text": "factor information coefficient quantile long short value", "tool": "run_factor_analysis"},
            {"doc_id": "fail_c", "kind": "failure", "title": "Costs erase momentum", "text": "transaction costs turnover erase momentum net sharpe negative", "tool": "run_backtest", "failure_class": "execution"},
        ],
        "queries": [
            {
                "query_id": "q_mom",
                "query_text": "momentum backtest with turnover and costs",
                "intent": _mini_intent("momentum", ["momentum", "turnover"], ["prior_experiments", "failures"]),
                "relevant": [{"doc_id": "exp_a", "grade": 2}, {"doc_id": "exp_b", "grade": 1}],
                "must_recall_failures": ["fail_c"],
            },
            {
                "query_id": "q_cost",
                "query_text": "why did momentum fail net of costs",
                "intent": _mini_intent("costs", ["costs"], ["failures"]),
                "relevant": [{"doc_id": "fail_c", "grade": 2}],
                "must_recall_failures": ["fail_c"],
            },
        ],
    }
    payload.update(overrides)
    return payload


class _FixtureLoadingTests(unittest.TestCase):
    def _write(self, tmp_dir: str, payload: object) -> str:
        path = Path(tmp_dir) / "fixture.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_loads_shipped_fixture(self) -> None:
        fixture = load_fixture(default_fixture_path())
        self.assertEqual(fixture.version, FIXTURE_VERSION)
        self.assertGreaterEqual(len(fixture.documents), 30)
        self.assertGreaterEqual(len(fixture.queries), 30)
        kinds = {d.kind for d in fixture.documents}
        self.assertTrue({"experiment", "failure", "finding"} <= kinds)
        self.assertTrue(all(q.must_recall_failures for q in fixture.queries))

    def test_missing_file(self) -> None:
        with self.assertRaises(FixtureError):
            load_fixture("/nonexistent/eval_fixture.json")

    def test_invalid_json(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(FixtureError):
                load_fixture(path)

    def test_rejects_dangling_relevant_id(self) -> None:
        import tempfile

        payload = _mini_fixture_dict()
        payload["queries"][0]["relevant"].append({"doc_id": "ghost", "grade": 1})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FixtureError):
                load_fixture(self._write(tmp, payload))

    def test_rejects_duplicate_doc_id(self) -> None:
        import tempfile

        payload = _mini_fixture_dict()
        payload["documents"].append(dict(payload["documents"][0]))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FixtureError):
                load_fixture(self._write(tmp, payload))

    def test_rejects_failure_without_class(self) -> None:
        import tempfile

        payload = _mini_fixture_dict()
        del payload["documents"][2]["failure_class"]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FixtureError):
                load_fixture(self._write(tmp, payload))

    def test_rejects_query_without_relevant(self) -> None:
        import tempfile

        payload = _mini_fixture_dict()
        payload["queries"][0]["relevant"] = []
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FixtureError):
                load_fixture(self._write(tmp, payload))

    def test_rejects_bad_intent(self) -> None:
        import tempfile

        payload = _mini_fixture_dict()
        del payload["queries"][0]["intent"]["topic"]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FixtureError):
                load_fixture(self._write(tmp, payload))


class _ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / "fixture.json"
        path.write_text(json.dumps(_mini_fixture_dict()), encoding="utf-8")
        self.fixture = load_fixture(path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_null_retriever_scores_zero(self) -> None:
        report = evaluate(self.fixture, null_retriever, 10, retriever_name="null")
        self.assertEqual(report.mean_recall, 0.0)
        self.assertEqual(report.mean_failure_recall, 0.0)
        self.assertFalse(report.passed)

    def test_oracle_scores_perfect(self) -> None:
        report = evaluate(self.fixture, oracle_retriever_factory(self.fixture), 10, retriever_name="oracle")
        self.assertEqual(report.mean_recall, 1.0)
        self.assertEqual(report.mean_failure_recall, 1.0)
        self.assertTrue(report.passed)

    def test_partial_recall_math(self) -> None:
        def retrieve(query: EvalQuery, k: int) -> list[str]:
            _ = k
            return [query.relevant_doc_ids[0]] if query.relevant_doc_ids else []

        report = evaluate(self.fixture, retrieve, 10)
        # q_mom hits 1/2 relevant, q_cost hits 1/1 -> mean 0.75.
        self.assertAlmostEqual(report.mean_recall, 0.75)
        # q_mom misses fail_c (0.0), q_cost hits it (1.0) -> mean 0.5.
        assert report.mean_failure_recall is not None
        self.assertAlmostEqual(report.mean_failure_recall, 0.5)

    def test_unknown_ids_ignored_and_counted(self) -> None:
        def retrieve(query: EvalQuery, k: int) -> list[str]:
            _ = (query, k)
            return ["ghost", "exp_a", "exp_a", 42]  # type: ignore[list-item]

        score = score_query(self.fixture.queries[0], retrieve, self.fixture.doc_ids, 10)
        self.assertEqual(score.retrieved, ("exp_a",))
        self.assertEqual(score.unknown_ids_ignored, 2)
        self.assertEqual(score.duplicates_dropped, 1)

    def test_ranking_truncated_to_k(self) -> None:
        def retrieve(query: EvalQuery, k: int) -> list[str]:
            _ = (query, k)
            return ["exp_b", "exp_a", "fail_c"]

        score = score_query(self.fixture.queries[0], retrieve, self.fixture.doc_ids, 1)
        # Only exp_b (grade 1) survives the k=1 cutoff: 1 of 2 relevant.
        self.assertEqual(score.recall, 0.5)
        self.assertEqual(score.failure_recall, 0.0)

    def test_retriever_exception_scores_zero_and_records(self) -> None:
        def retrieve(query: EvalQuery, k: int) -> list[str]:
            _ = (query, k)
            raise RuntimeError("boom")

        report = evaluate(self.fixture, retrieve, 10)
        self.assertEqual(report.n_queries_errored, 2)
        self.assertEqual(report.mean_recall, 0.0)
        self.assertFalse(report.passed)
        self.assertIn("RuntimeError", report.queries[0].error)

    def test_invalid_k_rejected(self) -> None:
        for bad in (0, -3, True, "10"):
            with self.assertRaises(FixtureError):
                evaluate(self.fixture, null_retriever, bad)  # type: ignore[arg-type]

    def test_keyword_baseline_is_deterministic_and_bounded(self) -> None:
        first = evaluate(self.fixture, keyword_retriever_factory(self.fixture), 2, retriever_name="keyword")
        second = evaluate(self.fixture, keyword_retriever_factory(self.fixture), 2, retriever_name="keyword")
        self.assertEqual(first.to_dict(), second.to_dict())
        for q in first.queries:
            self.assertLessEqual(len(q.retrieved), 2)
            self.assertTrue(set(q.retrieved) <= self.fixture.doc_ids)

    def test_report_serializes_to_json(self) -> None:
        report = evaluate(self.fixture, oracle_retriever_factory(self.fixture), 10)
        payload = json.loads(json.dumps(report.to_dict()))
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["n_queries"], 2)


class _CliTests(unittest.TestCase):
    def test_unknown_retriever_rejected(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.json"
            path.write_text(json.dumps(_mini_fixture_dict()), encoding="utf-8")
            fixture = load_fixture(path)
        with self.assertRaises(FixtureError):
            build_retriever("phase2", fixture)

    def test_main_exit_codes(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.json"
            path.write_text(json.dumps(_mini_fixture_dict()), encoding="utf-8")
            self.assertEqual(main(["--fixture", str(path), "--retriever", "null"]), EXIT_BELOW_TARGET)
            self.assertEqual(main(["--fixture", str(path), "--retriever", "oracle"]), EXIT_OK)
            self.assertEqual(main(["--fixture", str(path), "--retriever", "null", "--json"]), EXIT_BELOW_TARGET)
            self.assertEqual(main(["--fixture", "/nonexistent/fixture.json"]), EXIT_ERROR)
            self.assertEqual(main(["--fixture", str(path), "--k", "0"]), EXIT_ERROR)


def run_self_test() -> int:
    """Run the embedded unittest suite. Returns 0 on success, 1 on failure."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        [
            loader.loadTestsFromTestCase(_FixtureLoadingTests),
            loader.loadTestsFromTestCase(_ScoringTests),
            loader.loadTestsFromTestCase(_CliTests),
        ]
    )
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(suite)
    return EXIT_OK if result.wasSuccessful() else EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())

"""Deterministic Phase 2 eval seeder for the Quantflow Research Knowledge Plane.

Turns the curated retrieval-eval fixtures (``eval_fixture.json`` from Phase 0
plus ``eval_fixture_phase2.json``, the Phase 2 extension) into committable
payloads:

* experiment payloads shaped exactly as :mod:`deerflow.knowledge.write_api`
  kwargs (``experiment_begin`` / ``experiment_commit`` / ``failure_record``),
* finding payloads mirroring the ``knowledge_base.md`` ``finding`` contract
  (the ``FindingRow`` fields the Phase 2 schema owner will bind; no finding
  table exists yet, so these validate against a local forward-compatible
  validator instead of an ORM model),
* content-addressed result/log artifact payloads shaped for
  :class:`deerflow.knowledge.artifacts.store.ArtifactStore`,
* the merged known-relevance judgments, including the failure-recall cases.

Determinism contract: every derived value (UUIDs, idempotency keys, hashes,
methodology mappings, artifact bytes) is a pure function of the fixture
bytes plus the constants in this module. No timestamps, no ``uuid4``, no
dict-order dependence: building twice yields byte-identical JSON. Store-
assigned values (``ExperimentRecord.id``, ``started_at``, artifact
``created_at``) are the only non-deterministic surface, and idempotent
replay (same keys, same payload) returns the original records.

This module never edits ``runner.py``: it loads the frozen Phase 0 runner
by file path for fixture validation and scoring, and *extends* the runner
contract with packet-budget assertions plus a Phase 2 exit verdict
(``recall@k`` and ``failure-recall@k`` each ``>= 0.8`` at ``k=10`` **and**
every context packet within budget).

Usage::

    python seed_phase2.py --self-test            # embedded unit-test suite
    python seed_phase2.py --check                # judgment coverage + counts
    python seed_phase2.py --check --json         # machine-readable check
    python seed_phase2.py --emit bundle.json     # write the seed bundle
    python seed_phase2.py --validate-writes      # write_api round-trip check

The payload builders are stdlib-only. Only the write/artifact round-trip
validation imports ``deerflow`` (via the same harness ``sys.path``
bootstrap ``test_knowledge_api.py`` uses), and only inside the functions
that need it, so fixture loading, seeding math, and packet checks run
anywhere ``runner.py`` runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import unittest
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PHASE0_FIXTURE_FILENAME = "eval_fixture.json"
PHASE2_FIXTURE_FILENAME = "eval_fixture_phase2.json"
SEED_VERSION = "2.0.0"
DEFAULT_K = 10
DEFAULT_TARGET = 0.8

#: Default fixture sample period (the common ``requested_period`` across the
#: shipped fixtures). Fixture scopes carry universe/horizon/frequency but no
#: sample window, so every seeded methodology pins this window explicitly.
FIXTURE_SAMPLE_PERIOD: tuple[str, str] = ("2015-01-01", "2026-09-19")

#: UUIDv5 namespace anchoring every derived eval UUID.
SEED_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "quantflow-knowledge-eval/phase2/v1")

#: Fixed eval project / seeder-run anchors (derived, not random).
EVAL_PROJECT_ID = str(uuid.uuid5(SEED_NAMESPACE, "project:phase2-eval"))
EVAL_RUN_ID = str(uuid.uuid5(SEED_NAMESPACE, "run:phase2-eval-seeder"))

#: Idempotency-key prefix; keys are ``{prefix}:{doc_id}:{op}`` (no whitespace).
IDEMPOTENCY_PREFIX = "phase2-eval-v1"

#: KB ``finding_type`` vocabulary (knowledge_base.md: finding object).
FINDING_TYPES = frozenset({"empirical", "methodological", "data_quality", "failure", "prior"})

#: KB ``finding.status`` vocabulary; Phase 2 seeds ``candidate`` only
#: (implementation_plan.md Phase 2: "finding table (candidate status only)").
FINDING_STATUSES = frozenset({"candidate", "reviewed", "validated", "disputed", "superseded", "rejected"})

#: KB ``finding.confidence.overall_tier`` vocabulary.
CONFIDENCE_TIERS = frozenset({"low", "moderate", "high"})

#: KB ``finding.scope`` keys (knowledge_base.md: finding object).
FINDING_SCOPE_KEYS = frozenset({"asset_class", "market", "universe", "instrument", "factor_or_strategy", "horizon", "frequency", "regime", "cost_model", "dataset_family"})

#: KB ``finding_evidence`` vocabularies (knowledge_base.md: provenance edge).
EVIDENCE_TYPES = frozenset({"experiment", "source", "dataset", "prior_finding"})
EVIDENCE_RELATIONS = frozenset({"supports", "contradicts", "derives", "qualifies"})

#: Context-packet sections in KB display order (knowledge_base.md: packet).
PACKET_SECTIONS = ("consensus", "prior_experiments", "failures", "contradictions", "assumptions", "priors_skills", "open_questions")

_CANONICAL_KEY_RE = re.compile(r"[a-z0-9][a-z0-9:_-]*")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class SeedError(ValueError):
    """Base class for all seeder errors (fixture, payload, or budget)."""


class SeedFixtureError(SeedError):
    """Raised when fixtures are missing, version-skewed, or collide on IDs."""


class SeedValidationError(SeedError):
    """Raised when a built payload violates the write/finding/artifact contract."""


class PacketBudgetError(SeedError):
    """Raised when a context packet exceeds its fixed budget."""

    def __init__(self, query_id: str, violations: list[str]):
        super().__init__(f"packet for {query_id!r} exceeds budget: {'; '.join(violations)}")
        self.query_id = query_id
        self.violations = list(violations)


class SeedDependencyError(SeedError):
    """Raised when ``deerflow`` is needed but not importable."""


def _this_dir() -> Path:
    """Return the directory containing this seeder (the ``eval/`` dir)."""
    return Path(__file__).resolve().parent


def default_phase0_path() -> Path:
    """Return the Phase 0 fixture sibling (``eval_fixture.json``)."""
    return _this_dir() / PHASE0_FIXTURE_FILENAME


def default_phase2_path() -> Path:
    """Return the Phase 2 fixture sibling (``eval_fixture_phase2.json``)."""
    return _this_dir() / PHASE2_FIXTURE_FILENAME


_runner_module: Any = None


def load_runner() -> Any:
    """Load the frozen Phase 0 ``runner.py`` sibling by file path.

    The runner is loaded under a private module name so this seeder never
    depends on ``sys.path`` ordering or collides with any other module
    named ``runner``. ``runner.py`` itself is never modified.
    """
    global _runner_module
    if _runner_module is not None:
        return _runner_module
    path = _this_dir() / "runner.py"
    if not path.is_file():
        raise SeedFixtureError(f"Phase 0 runner not found: {path}")
    spec = importlib.util.spec_from_file_location("seed_phase2_phase0_runner", path)
    if spec is None or spec.loader is None:
        raise SeedFixtureError(f"cannot load Phase 0 runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _runner_module = module
    return module


_deerflow_modules: dict[str, Any] = {}


def _harness_dir() -> Path:
    """Return the ``deerflow-harness`` package dir (parents[3] of this file)."""
    return Path(__file__).resolve().parents[3]


def require_deerflow(name: str) -> Any:
    """Import ``deerflow.<name>`` with the harness ``sys.path`` bootstrap.

    Mirrors the bootstrap in ``test_knowledge_api.py`` so the self-test
    runs standalone from anywhere. Results are cached per process.

    Raises:
        SeedDependencyError: If the harness package is not importable.
    """
    if name in _deerflow_modules:
        return _deerflow_modules[name]
    harness = str(_harness_dir())
    if harness not in sys.path:
        sys.path.insert(0, harness)
    try:
        module = importlib.import_module(f"deerflow.{name}")
    except ImportError as exc:
        raise SeedDependencyError(f"deerflow.{name} is not importable (harness dir {harness}): {exc}") from exc
    _deerflow_modules[name] = module
    return module


def stable_uuid(name: str) -> str:
    """Return the deterministic UUIDv5 for ``name`` under :data:`SEED_NAMESPACE`.

    Args:
        name: Namespaced seed string, e.g. ``"experiment:exp_mom_126_top20_me"``.

    Returns:
        Lowercase canonical UUID string; identical across processes/machines.
    """
    return str(uuid.uuid5(SEED_NAMESPACE, name))


def idempotency_key(doc_id: str, operation: str) -> str:
    """Return the deterministic idempotency key for one seeded write.

    Args:
        doc_id: Fixture document ID (never contains whitespace by fixture validation).
        operation: One of ``begin`` / ``commit`` / ``failure`` / ``finding`` / ``artifact``.

    Returns:
        ``"phase2-eval-v1:{doc_id}:{operation}"`` (opaque, whitespace-free).
    """
    if operation not in ("begin", "commit", "failure", "finding", "artifact", "log"):
        raise SeedValidationError(f"unknown idempotency operation {operation!r}")
    return f"{IDEMPOTENCY_PREFIX}:{doc_id}:{operation}"


def canonical_json_text(value: Any) -> str:
    """Serialize ``value`` to canonical JSON (sorted keys, compact, ASCII).

    This is the seeder's stdlib-only canonical form for artifact bytes and
    digest inputs; it intentionally matches the ``sort_keys``/``separators``/
    ``ensure_ascii`` shape of ``deerflow.knowledge.hashing.canonical_json``
    (verified by round-trip in the self-test) without importing deerflow.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_hex(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def slugify(doc_id: str) -> str:
    """Return the canonical-key slug for a fixture document ID."""
    return _SLUG_RE.sub("-", doc_id.strip().lower()).strip("-")


# ---------------------------------------------------------------------------
# Fixture loading + merge
# ---------------------------------------------------------------------------


def load_merged_fixture(phase0_path: str | Path | None = None, phase2_path: str | Path | None = None) -> Any:
    """Load and merge the Phase 0 + Phase 2 eval fixtures.

    Each file is validated by the frozen Phase 0 runner loader (schema,
    version, dangling judgments), then merged with collision checks so the
    union keeps stable IDs for seeding and scoring.

    Args:
        phase0_path: Path to ``eval_fixture.json`` (default: sibling file).
        phase2_path: Path to ``eval_fixture_phase2.json`` (default: sibling).

    Returns:
        A merged ``EvalFixture`` (the runner's frozen dataclass).

    Raises:
        SeedFixtureError: On version skew or duplicate doc/query IDs.
    """
    runner = load_runner()
    try:
        base = runner.load_fixture(str(phase0_path or default_phase0_path()))
        extra = runner.load_fixture(str(phase2_path or default_phase2_path()))
    except runner.FixtureError as exc:
        raise SeedFixtureError(str(exc)) from exc
    if base.version != extra.version:
        raise SeedFixtureError(f"fixture version skew: phase0={base.version!r} vs phase2={extra.version!r}")
    doc_ids = [d.doc_id for d in base.documents]
    dup_docs = sorted({d.doc_id for d in extra.documents} & set(doc_ids))
    if dup_docs:
        raise SeedFixtureError(f"duplicate doc_id(s) across fixtures: {', '.join(dup_docs)}")
    query_ids = [q.query_id for q in base.queries]
    dup_queries = sorted({q.query_id for q in extra.queries} & set(query_ids))
    if dup_queries:
        raise SeedFixtureError(f"duplicate query_id(s) across fixtures: {', '.join(dup_queries)}")
    return runner.EvalFixture(version=base.version, documents=tuple(base.documents) + tuple(extra.documents), queries=tuple(base.queries) + tuple(extra.queries))


def fixture_raw_documents(phase0_path: str | Path | None = None, phase2_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Return the raw fixture document dicts keyed by ``doc_id`` (merged).

    The runner's ``EvalDocument`` keeps only the retrieval surface; seeding
    needs the full params/metrics/scope shapes, so this reads the raw JSON
    dicts (validated for ID parity with the merged fixture).
    """
    merged = load_merged_fixture(phase0_path, phase2_path)
    raw: dict[str, dict[str, Any]] = {}
    for path in (phase0_path or default_phase0_path(), phase2_path or default_phase2_path()):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for doc in payload.get("documents", []):
            raw[doc["doc_id"]] = dict(doc)
    want = {d.doc_id for d in merged.documents}
    if set(raw) != want:
        raise SeedFixtureError(f"raw/validated doc_id parity failure: missing={sorted(want - set(raw))} extra={sorted(set(raw) - want)}")
    return raw


# ---------------------------------------------------------------------------
# Judgments + coverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgmentCoverage:
    """Coverage of the merged known-relevance judgments.

    Attributes:
        n_documents: Corpus size (seedable docs).
        n_queries: Number of eval queries.
        n_failure_queries: Queries naming at least one must-recall failure.
        queries_without_relevant: Query IDs with empty ``relevant`` (must be empty).
        queries_without_failures: Query IDs with empty ``must_recall_failures`` (must be empty).
        dangling_ids: Judged IDs missing from the corpus (must be empty).
        must_recall_non_failures: Must-recall IDs whose doc kind is not ``failure`` (must be empty).
        failures_never_must_recalled: Failure doc IDs no query must-recalls (must be empty).
        documents_never_judged: Doc IDs in neither ``relevant`` nor ``must_recall_failures`` (must be empty).
    """

    n_documents: int
    n_queries: int
    n_failure_queries: int
    queries_without_relevant: tuple[str, ...] = ()
    queries_without_failures: tuple[str, ...] = ()
    dangling_ids: tuple[str, ...] = ()
    must_recall_non_failures: tuple[str, ...] = ()
    failures_never_must_recalled: tuple[str, ...] = ()
    documents_never_judged: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when every coverage invariant holds."""
        return not (self.queries_without_relevant or self.queries_without_failures or self.dangling_ids or self.must_recall_non_failures or self.failures_never_must_recalled or self.documents_never_judged)

    def violations(self) -> list[str]:
        """Return one human-readable line per violated invariant."""
        problems: list[str] = []
        if self.queries_without_relevant:
            problems.append(f"queries without relevant judgments: {', '.join(self.queries_without_relevant)}")
        if self.queries_without_failures:
            problems.append(f"queries without must-recall failures: {', '.join(self.queries_without_failures)}")
        if self.dangling_ids:
            problems.append(f"dangling judged IDs: {', '.join(self.dangling_ids)}")
        if self.must_recall_non_failures:
            problems.append(f"must-recall IDs that are not failures: {', '.join(self.must_recall_non_failures)}")
        if self.failures_never_must_recalled:
            problems.append(f"failures never must-recalled: {', '.join(self.failures_never_must_recalled)}")
        if self.documents_never_judged:
            problems.append(f"documents never judged: {', '.join(self.documents_never_judged)}")
        return problems

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of this coverage."""
        return {
            "n_documents": self.n_documents,
            "n_queries": self.n_queries,
            "n_failure_queries": self.n_failure_queries,
            "ok": self.ok,
            "queries_without_relevant": list(self.queries_without_relevant),
            "queries_without_failures": list(self.queries_without_failures),
            "dangling_ids": list(self.dangling_ids),
            "must_recall_non_failures": list(self.must_recall_non_failures),
            "failures_never_must_recalled": list(self.failures_never_must_recalled),
            "documents_never_judged": list(self.documents_never_judged),
        }


def build_judgments(fixture: Any) -> dict[str, dict[str, Any]]:
    """Return the known-relevance judgments keyed by ``query_id``.

    Each entry carries ``relevant`` (ranked doc IDs, grade 2 first),
    ``grades`` (doc ID to 1/2), and ``must_recall_failures`` (the
    failure-recall cases Phase 2 must surface through its dedicated
    failure channel).
    """
    judgments: dict[str, dict[str, Any]] = {}
    for query in fixture.queries:
        ranked = sorted(query.relevant_doc_ids, key=lambda d: (-query.grades.get(d, 1), d))
        judgments[query.query_id] = {"query_text": query.query_text, "relevant": list(ranked), "grades": {d: query.grades.get(d, 1) for d in ranked}, "must_recall_failures": list(query.must_recall_failures)}
    return judgments


def check_judgment_coverage(fixture: Any) -> JudgmentCoverage:
    """Compute judgment coverage over a (possibly merged) fixture."""
    kinds = {d.doc_id: d.kind for d in fixture.documents}
    relevant_union: set[str] = set()
    must_union: set[str] = set()
    dangling: set[str] = set()
    non_failure_must: set[str] = set()
    no_relevant: list[str] = []
    no_failures: list[str] = []
    for query in fixture.queries:
        if not query.relevant_doc_ids:
            no_relevant.append(query.query_id)
        if not query.must_recall_failures:
            no_failures.append(query.query_id)
        for doc_id in query.relevant_doc_ids:
            (relevant_union if doc_id in kinds else dangling).add(doc_id)
        for doc_id in query.must_recall_failures:
            if doc_id not in kinds:
                dangling.add(doc_id)
            else:
                must_union.add(doc_id)
                if kinds[doc_id] != "failure":
                    non_failure_must.add(doc_id)
    failures = {d for d, k in kinds.items() if k == "failure"}
    judged = relevant_union | must_union
    return JudgmentCoverage(
        n_documents=len(kinds),
        n_queries=len(fixture.queries),
        n_failure_queries=sum(1 for q in fixture.queries if q.must_recall_failures),
        queries_without_relevant=tuple(sorted(no_relevant)),
        queries_without_failures=tuple(sorted(no_failures)),
        dangling_ids=tuple(sorted(dangling)),
        must_recall_non_failures=tuple(sorted(non_failure_must)),
        failures_never_must_recalled=tuple(sorted(failures - must_union)),
        documents_never_judged=tuple(sorted(set(kinds) - judged)),
    )


def assert_judgment_coverage(fixture: Any) -> JudgmentCoverage:
    """Return coverage, raising :class:`SeedValidationError` when it fails."""
    coverage = check_judgment_coverage(fixture)
    if not coverage.ok:
        raise SeedValidationError("judgment coverage failed: " + "; ".join(coverage.violations()))
    return coverage


# ---------------------------------------------------------------------------
# Experiment payloads (write_api shapes)
# ---------------------------------------------------------------------------


def build_methodology(doc: dict[str, Any]) -> dict[str, Any]:
    """Build the KB-canonical methodology mapping for one fixture document.

    Covers the identity-bearing design keys from
    ``deerflow.knowledge.hashing.experiment_family_hash`` (universe,
    frequency, horizon, sample period, protocol, portfolio construction,
    cost/slippage models, neutralization, rebalance rule) plus the
    originating tool name. Purely conceptual: no run-specific values
    (seeds, UUIDs, timestamps) so replications group by family hash.

    Args:
        doc: Raw fixture document dict (params/metrics/scope optional).

    Returns:
        A JSON-canonicalizable methodology mapping.
    """
    params = doc.get("params") or {}
    scope = doc.get("scope") or {}
    if not isinstance(params, dict) or not isinstance(scope, dict):
        raise SeedValidationError(f"document {doc.get('doc_id')!r} has non-object params/scope")
    costs = params.get("costs")
    if isinstance(costs, dict):
        cost_model: Any = {"bps": costs.get("bps", 0), "slippage": costs.get("slippage", 0), "borrow": costs.get("borrow", 0)}
        slippage_model: Any = {"bps": costs.get("slippage", 0)}
    elif "fees" in params:
        cost_model = {"fees": params["fees"]}
        slippage_model = "none"
    else:
        cost_model = "unspecified"
        slippage_model = "unspecified"
    construction: Any = {"tool": doc.get("tool") or "none"}
    construction_keys = (
        "lookback_days",
        "lookback",
        "top_n",
        "holding",
        "signal_name",
        "expression",
        "method",
        "type",
        "features",
        "factors",
        "factor_model",
        "symbols_weights",
        "holdings_uri",
        "symbols",
        "targets",
        "views",
        "join",
        "splits",
        "horizon_days",
        "q",
    )
    for key in construction_keys:
        if key in params:
            construction[key] = params[key]
    neutralization: Any = "sector" if params.get("sector_neutralize") else ("none" if "sector_neutralize" in params else "unspecified")
    methodology = {
        "universe": scope.get("universe", "unspecified"),
        "frequency": scope.get("frequency", "unspecified"),
        "horizon": scope.get("horizon", "unspecified"),
        "sample_period": list(FIXTURE_SAMPLE_PERIOD),
        "train_test_protocol": "point-in-time" if "pit" in str(scope.get("universe", "")).lower() else "as-stated",
        "portfolio_construction": construction,
        "transaction_cost_model": cost_model,
        "slippage_model": slippage_model,
        "neutralization": neutralization,
        "rebalance_rule": params.get("rebalance", "unspecified"),
    }
    if scope.get("asset_class"):
        methodology["asset_class"] = scope["asset_class"]
    if scope.get("market"):
        methodology["market"] = scope["market"]
    return methodology


def build_code_identity(doc: dict[str, Any]) -> dict[str, Any]:
    """Build the exact code-identity mapping for one fixture document."""
    params = doc.get("params") or {}
    digest = sha256_hex(canonical_json_text({"tool": doc.get("tool") or "none", "params": params}).encode("utf-8"))
    return {"mcp_tool": doc.get("tool") or "none", "tool_params_digest": digest, "spec_version": "phase2-eval-v1"}


def build_environment_identity(doc: dict[str, Any]) -> dict[str, Any]:
    """Build the exact environment-identity mapping for one fixture document."""
    digest = sha256_hex(f"alpha_engine-eval|3.12|{doc.get('tool') or 'none'}".encode())
    return {"engine": "alpha_engine-eval", "python": "3.12", "lock_hash": digest}


def build_parameters(doc: dict[str, Any]) -> dict[str, Any]:
    """Build the full parameter mapping (execution-level, includes the seed)."""
    params = doc.get("params") or {}
    seed = int.from_bytes(hashlib.sha256(f"seed:{doc['doc_id']}".encode()).digest()[:4], "big")
    return {"eval_doc_id": doc["doc_id"], "tool": doc.get("tool") or "none", "tool_params": params, "random_seed": seed}


@dataclass(frozen=True)
class SeedExperiment:
    """Committable experiment payloads for one fixture document.

    ``begin_kwargs`` / ``commit_kwargs`` / ``failure_kwargs`` splat directly
    into :func:`deerflow.knowledge.write_api.experiment_begin`,
    :func:`~deerflow.knowledge.write_api.experiment_commit` (plus
    ``experiment_id``), and
    :func:`~deerflow.knowledge.write_api.failure_record` (plus
    ``experiment_id``). Finding-kind documents produce no experiment;
    failure-kind documents produce all three (the failure detail record
    plus a ``outcome="failure"`` commit).
    """

    doc_id: str
    kind: str
    begin_kwargs: dict[str, Any]
    commit_kwargs: dict[str, Any]
    failure_kwargs: dict[str, Any] | None
    dataset_version_ids: tuple[str, ...]
    result_artifact_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of this seed."""
        return {
            "doc_id": self.doc_id,
            "kind": self.kind,
            "begin_kwargs": self.begin_kwargs,
            "commit_kwargs": self.commit_kwargs,
            "failure_kwargs": self.failure_kwargs,
            "dataset_version_ids": list(self.dataset_version_ids),
            "result_artifact_ids": list(self.result_artifact_ids),
        }


def build_experiment_seed(doc: dict[str, Any]) -> SeedExperiment | None:
    """Build the experiment seed for one raw fixture document.

    Args:
        doc: Raw fixture document dict with ``doc_id``/``kind``/``title``/``text``.

    Returns:
        The :class:`SeedExperiment`, or None for ``finding`` documents
        (which seed findings instead of experiments).

    Raises:
        SeedValidationError: On missing identity fields or a failure
            document without a valid ``failure_class``.
    """
    doc_id = doc.get("doc_id", "")
    kind = doc.get("kind", "")
    title = doc.get("title", "")
    text = doc.get("text", "")
    if not doc_id or not title or not text:
        raise SeedValidationError(f"document {doc_id!r} is missing doc_id/title/text")
    if kind == "finding":
        return None
    if kind not in ("experiment", "failure"):
        raise SeedValidationError(f"document {doc_id!r} has unknown kind {kind!r}")
    metrics = doc.get("metrics") or {}
    if not isinstance(metrics, dict):
        raise SeedValidationError(f"document {doc_id!r} has non-object metrics")
    dataset_id = stable_uuid(f"dataset:{doc_id}:features")
    result_artifact_id = stable_uuid(f"artifact:result:{doc_id}")
    begin_kwargs: dict[str, Any] = {
        "project_id": EVAL_PROJECT_ID,
        "hypothesis": title,
        "methodology": build_methodology(doc),
        "parameters": build_parameters(doc),
        "datasets": [{"dataset_version_id": dataset_id, "role": "features"}],
        "code": build_code_identity(doc),
        "environment": build_environment_identity(doc),
        "created_by_run_id": EVAL_RUN_ID,
        "status": "running",
        "idempotency_key": idempotency_key(doc_id, "begin"),
    }
    failure_kwargs: dict[str, Any] | None = None
    if kind == "failure":
        failure_class = doc.get("failure_class", "")
        if failure_class not in ("data", "code", "statistical", "execution", "hypothesis"):
            raise SeedValidationError(f"failure {doc_id!r} has invalid failure_class {failure_class!r}")
        failure_kwargs = {"failure_class": failure_class, "notes": text, "idempotency_key": idempotency_key(doc_id, "failure")}
        commit_kwargs = {"metrics": metrics, "outcome": "failure", "failure_class": failure_class, "result_artifacts": [result_artifact_id], "idempotency_key": idempotency_key(doc_id, "commit")}
    else:
        commit_kwargs = {"metrics": metrics, "outcome": "success", "result_artifacts": [result_artifact_id], "idempotency_key": idempotency_key(doc_id, "commit")}
    # The retrieval surface travels in parameters so committed rows stay
    # self-describing for the Phase 2 indexing worker (execution-level by design).
    begin_kwargs["parameters"]["eval_retrieval_text"] = text
    return SeedExperiment(doc_id=doc_id, kind=kind, begin_kwargs=begin_kwargs, commit_kwargs=commit_kwargs, failure_kwargs=failure_kwargs, dataset_version_ids=(dataset_id,), result_artifact_ids=(result_artifact_id,))


# ---------------------------------------------------------------------------
# Finding payloads (FindingRow fields, forward-compatible)
# ---------------------------------------------------------------------------

#: KB ``finding_type`` per shipped finding doc. New finding docs must be
#: added here explicitly: the type is a curatorial claim, never inferred.
FINDING_TYPE_BY_DOC = {
    "find_mom_6_12_premia": "empirical",
    "find_turnover_kills_short_mom": "empirical",
    "find_month_period_join": "methodological",
    "find_ewma_halflife66": "methodological",
    "find_p2_borrow_costs_short_leg": "methodological",
    "find_p2_pit_fundamentals": "data_quality",
}


def supporting_experiments(finding_doc_id: str, fixture: Any) -> tuple[str, ...]:
    """Return experiment doc IDs sharing a query with a finding.

    A finding's evidence edges point at the experiments it generalizes:
    the experiment-kind documents co-judged ``relevant`` on any query
    where the finding itself is judged relevant. Sorted for determinism.
    """
    kinds = {d.doc_id: d.kind for d in fixture.documents}
    supporting: set[str] = set()
    for query in fixture.queries:
        if finding_doc_id not in query.relevant_doc_ids:
            continue
        for doc_id in query.relevant_doc_ids:
            if kinds.get(doc_id) == "experiment":
                supporting.add(doc_id)
    return tuple(sorted(supporting))


def build_finding_payload(doc: dict[str, Any], fixture: Any) -> dict[str, Any]:
    """Build the KB finding payload for one raw finding document.

    Field-for-field this mirrors the ``finding`` object in
    ``knowledge_base.md`` (the ``FindingRow`` columns the Phase 2 schema
    owner will bind: id, canonical key, statement, type, scope, status,
    confidence, valid/transaction time, run, supersession, evidence).
    ``embedding`` / ``search_document`` are index projections owned by the
    Phase 2 retrieval worker, so the payload carries the retrieval text
    they derive from instead. ``experiment_id`` inside evidence edges is
    None until :func:`resolve_finding_evidence` fills it post-commit.

    Args:
        doc: Raw fixture document dict of kind ``finding``.
        fixture: Merged fixture (derives the evidence edges).

    Returns:
        A JSON-serializable finding payload (validated before return).
    """
    doc_id = doc.get("doc_id", "")
    if doc.get("kind") != "finding":
        raise SeedValidationError(f"document {doc_id!r} is not a finding (kind={doc.get('kind')!r})")
    statement = doc.get("text", "")
    if not statement:
        raise SeedValidationError(f"finding {doc_id!r} has empty statement text")
    finding_type = FINDING_TYPE_BY_DOC.get(doc_id)
    if finding_type is None:
        raise SeedValidationError(f"finding {doc_id!r} has no curated finding_type in FINDING_TYPE_BY_DOC")
    scope = doc.get("scope") or {}
    params = doc.get("params") or {}
    finding_scope: dict[str, Any] = {}
    if scope.get("asset_class"):
        finding_scope["asset_class"] = scope["asset_class"]
    if scope.get("market"):
        finding_scope["market"] = scope["market"]
    if scope.get("universe"):
        finding_scope["universe"] = scope["universe"]
    if scope.get("horizon"):
        finding_scope["horizon"] = scope["horizon"]
    if scope.get("frequency"):
        finding_scope["frequency"] = scope["frequency"]
    if params.get("signal_name"):
        finding_scope["factor_or_strategy"] = params["signal_name"]
    evidence = [
        {
            "evidence_type": "experiment",
            "experiment_doc_id": exp_id,
            "experiment_id": None,
            "relation": "supports",
            "locator": {"table": "experiment", "metric": "metrics"},
            "evidence_weight": 1.0,
        }
        for exp_id in supporting_experiments(doc_id, fixture)
    ]
    payload = {
        "finding_id": stable_uuid(f"finding:{doc_id}"),
        "canonical_key": f"{finding_type}:{slugify(doc_id)}",
        "statement": statement,
        "finding_type": finding_type,
        "scope": finding_scope,
        "status": "candidate",
        "confidence": {"overall_tier": "moderate", "rationale": f"eval seed: corroborated by {len(evidence)} supporting experiment(s)"},
        "valid_time": {"effective_from": FIXTURE_SAMPLE_PERIOD[0], "effective_to": None},
        "created_by_run_id": EVAL_RUN_ID,
        "supersedes_finding_id": None,
        "evidence": evidence,
        "idempotency_key": idempotency_key(doc_id, "finding"),
        "retrieval_text": f"{doc.get('title', '')} {statement}",
    }
    validate_finding_payload(payload)
    return payload


def validate_finding_payload(payload: dict[str, Any]) -> None:
    """Validate a finding payload against the KB finding contract.

    Mirrors the enum vocabularies and shapes of ``knowledge_base.md``
    (finding object + finding_evidence edge) so payloads bind cleanly to
    the Phase 2 ``FindingRow`` schema when its migration lands.

    Raises:
        SeedValidationError: On any contract violation.
    """
    if not isinstance(payload, dict):
        raise SeedValidationError(f"finding payload must be an object, got {type(payload).__name__}")
    try:
        uuid.UUID(str(payload.get("finding_id", "")))
    except ValueError:
        raise SeedValidationError(f"finding {payload.get('canonical_key')!r} has invalid finding_id {payload.get('finding_id')!r}") from None
    key = payload.get("canonical_key", "")
    if not isinstance(key, str) or not _CANONICAL_KEY_RE.fullmatch(key):
        raise SeedValidationError(f"finding has invalid canonical_key {key!r}")
    if not isinstance(payload.get("statement"), str) or not payload["statement"].strip():
        raise SeedValidationError(f"finding {key!r} has empty statement")
    if payload.get("finding_type") not in FINDING_TYPES:
        raise SeedValidationError(f"finding {key!r} has invalid finding_type {payload.get('finding_type')!r}")
    if payload.get("status") != "candidate":
        raise SeedValidationError(f"finding {key!r} must seed with status 'candidate', got {payload.get('status')!r}")
    scope = payload.get("scope")
    if not isinstance(scope, dict):
        raise SeedValidationError(f"finding {key!r} has non-object scope")
    unknown_scope = sorted(set(scope) - FINDING_SCOPE_KEYS)
    if unknown_scope:
        raise SeedValidationError(f"finding {key!r} has unknown scope keys: {', '.join(unknown_scope)}")
    for scope_key, scope_value in scope.items():
        if not isinstance(scope_value, str) or not scope_value.strip():
            raise SeedValidationError(f"finding {key!r} scope[{scope_key!r}] must be a non-empty string")
    confidence = payload.get("confidence")
    if not isinstance(confidence, dict) or confidence.get("overall_tier") not in CONFIDENCE_TIERS:
        raise SeedValidationError(f"finding {key!r} has invalid confidence {confidence!r}")
    valid_time = payload.get("valid_time")
    if not isinstance(valid_time, dict) or not valid_time.get("effective_from"):
        raise SeedValidationError(f"finding {key!r} has invalid valid_time {valid_time!r}")
    try:
        uuid.UUID(str(payload.get("created_by_run_id", "")))
    except ValueError:
        raise SeedValidationError(f"finding {key!r} has invalid created_by_run_id") from None
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        raise SeedValidationError(f"finding {key!r} has non-list evidence")
    for index, edge in enumerate(evidence):
        if not isinstance(edge, dict):
            raise SeedValidationError(f"finding {key!r} evidence[{index}] must be an object")
        if edge.get("evidence_type") not in EVIDENCE_TYPES:
            raise SeedValidationError(f"finding {key!r} evidence[{index}] has invalid evidence_type {edge.get('evidence_type')!r}")
        if edge.get("relation") not in EVIDENCE_RELATIONS:
            raise SeedValidationError(f"finding {key!r} evidence[{index}] has invalid relation {edge.get('relation')!r}")
        if not isinstance(edge.get("locator"), dict):
            raise SeedValidationError(f"finding {key!r} evidence[{index}] has non-object locator")
        weight = edge.get("evidence_weight", 1.0)
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
            raise SeedValidationError(f"finding {key!r} evidence[{index}] has invalid evidence_weight {weight!r}")
        experiment_id = edge.get("experiment_id")
        if experiment_id is not None:
            try:
                uuid.UUID(str(experiment_id))
            except ValueError:
                raise SeedValidationError(f"finding {key!r} evidence[{index}] has invalid experiment_id") from None


@dataclass(frozen=True)
class SeedFinding:
    """A validated finding payload plus its fixture provenance."""

    doc_id: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of this seed."""
        return {"doc_id": self.doc_id, "payload": self.payload}


# ---------------------------------------------------------------------------
# Artifact payloads (ArtifactStore shapes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedArtifact:
    """A content-addressed artifact payload for one fixture document.

    ``sha256`` is the content address (``artifact://sha256/<digest>``);
    ``bytes_text`` is the exact UTF-8 JSON the digest covers, so the
    integration step materializes it with
    ``ArtifactStore.put_bytes(bytes_text.encode("utf-8"), kind=..., ...)``
    and the store recomputes the same digest. Every doc seeds one
    ``result`` artifact; failure docs additionally seed a ``log`` artifact
    carrying the failure notes.
    """

    doc_id: str
    kind: str
    media_type: str
    byte_size: int
    sha256: str
    bytes_text: str
    artifact_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of this seed."""
        return {"doc_id": self.doc_id, "kind": self.kind, "media_type": self.media_type, "byte_size": self.byte_size, "sha256": self.sha256, "bytes_text": self.bytes_text, "artifact_id": self.artifact_id}


def build_artifact_seeds(doc: dict[str, Any]) -> tuple[SeedArtifact, ...]:
    """Build the artifact seeds for one raw fixture document.

    Args:
        doc: Raw fixture document dict.

    Returns:
        One ``result`` seed (every doc) plus one ``log`` seed (failures).
    """
    doc_id = doc.get("doc_id", "")
    if not doc_id:
        raise SeedValidationError("artifact seed requires doc_id")
    result_envelope = {
        "seed_version": SEED_VERSION,
        "doc_id": doc_id,
        "kind": doc.get("kind"),
        "title": doc.get("title"),
        "tool": doc.get("tool") or "none",
        "params": doc.get("params") or {},
        "metrics": doc.get("metrics") or {},
        "scope": doc.get("scope") or {},
        "text": doc.get("text"),
    }
    result_text = canonical_json_text(result_envelope)
    result_bytes = result_text.encode("utf-8")
    seeds = [
        SeedArtifact(doc_id=doc_id, kind="result", media_type="application/json", byte_size=len(result_bytes), sha256=sha256_hex(result_bytes), bytes_text=result_text, artifact_id=stable_uuid(f"artifact:result:{doc_id}")),
    ]
    if doc.get("kind") == "failure":
        log_text = canonical_json_text({"seed_version": SEED_VERSION, "doc_id": doc_id, "failure_class": doc.get("failure_class"), "notes": doc.get("text")})
        log_bytes = log_text.encode("utf-8")
        seeds.append(SeedArtifact(doc_id=doc_id, kind="log", media_type="application/json", byte_size=len(log_bytes), sha256=sha256_hex(log_bytes), bytes_text=log_text, artifact_id=stable_uuid(f"artifact:log:{doc_id}")))
    return tuple(seeds)


def validate_artifact_seed(seed: SeedArtifact) -> None:
    """Validate one artifact seed (digest, size, kind, UUID).

    Raises:
        SeedValidationError: When the digest does not cover the bytes, the
            size is wrong, the kind is outside the KB vocabulary, or the
            artifact ID is not a UUID.
    """
    if seed.kind not in ("dataset_snapshot", "source", "code", "notebook", "result", "log", "chart", "environment"):
        raise SeedValidationError(f"artifact {seed.doc_id!r} has invalid kind {seed.kind!r}")
    raw = seed.bytes_text.encode("utf-8")
    if sha256_hex(raw) != seed.sha256:
        raise SeedValidationError(f"artifact {seed.doc_id!r} sha256 does not cover bytes_text")
    if len(raw) != seed.byte_size:
        raise SeedValidationError(f"artifact {seed.doc_id!r} byte_size {seed.byte_size} != actual {len(raw)}")
    try:
        uuid.UUID(seed.artifact_id)
    except ValueError:
        raise SeedValidationError(f"artifact {seed.doc_id!r} has invalid artifact_id") from None


# ---------------------------------------------------------------------------
# Seed bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedBundle:
    """The full deterministic seed: experiments, findings, artifacts, judgments."""

    seed_version: str
    fixture_version: str
    project_id: str
    run_id: str
    experiments: tuple[SeedExperiment, ...]
    findings: tuple[SeedFinding, ...]
    artifacts: tuple[SeedArtifact, ...]
    judgments: dict[str, dict[str, Any]]
    coverage: JudgmentCoverage

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of this bundle (sorted by doc ID)."""
        return {
            "seed_version": self.seed_version,
            "fixture_version": self.fixture_version,
            "project_id": self.project_id,
            "run_id": self.run_id,
            "experiments": [e.to_dict() for e in sorted(self.experiments, key=lambda s: s.doc_id)],
            "findings": [f.to_dict() for f in sorted(self.findings, key=lambda s: s.doc_id)],
            "artifacts": [a.to_dict() for a in sorted(self.artifacts, key=lambda s: (s.doc_id, s.kind))],
            "judgments": {qid: self.judgments[qid] for qid in sorted(self.judgments)},
            "coverage": self.coverage.to_dict(),
        }


def build_seed_bundle(phase0_path: str | Path | None = None, phase2_path: str | Path | None = None) -> SeedBundle:
    """Build the full deterministic seed bundle from both fixtures.

    Stdlib-only: no ``deerflow`` import, no I/O beyond reading the two
    fixture files. Raises on coverage failure so a bundle always ships
    with complete judgments.
    """
    fixture = load_merged_fixture(phase0_path, phase2_path)
    coverage = assert_judgment_coverage(fixture)
    raw_docs = fixture_raw_documents(phase0_path, phase2_path)
    experiments: list[SeedExperiment] = []
    findings: list[SeedFinding] = []
    artifacts: list[SeedArtifact] = []
    for doc_id in sorted(raw_docs):
        doc = raw_docs[doc_id]
        seed = build_experiment_seed(doc)
        if seed is not None:
            experiments.append(seed)
        if doc.get("kind") == "finding":
            findings.append(SeedFinding(doc_id=doc_id, payload=build_finding_payload(doc, fixture)))
        for artifact in build_artifact_seeds(doc):
            validate_artifact_seed(artifact)
            artifacts.append(artifact)
    return SeedBundle(
        seed_version=SEED_VERSION,
        fixture_version=fixture.version,
        project_id=EVAL_PROJECT_ID,
        run_id=EVAL_RUN_ID,
        experiments=tuple(experiments),
        findings=tuple(findings),
        artifacts=tuple(artifacts),
        judgments=build_judgments(fixture),
        coverage=coverage,
    )


def emit_bundle(bundle: SeedBundle, path: str | Path) -> Path:
    """Write ``bundle`` as pretty canonical JSON to ``path``; return the path."""
    out = Path(path)
    out.write_text(json.dumps(bundle.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Context packets + fixed-budget assertions (runner-contract extension)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PacketBudget:
    """Fixed context-packet budget (KB: packet fits a fixed context budget).

    Attributes:
        max_total_chars: Ceiling on the canonical-JSON size of packet items.
        max_items: Ceiling on items per packet (aligns with the retrieval
            cutoff ``k``: a packet never carries more than was retrieved).
        max_item_chars: Ceiling on ``len(title) + len(text)`` per item, so
            one long document cannot starve the failure channel.
        require_evidence_pointers: When True, every consensus/failure item
            must carry at least one evidence pointer (the cite-open rule
            needs something to open).
    """

    max_total_chars: int = 12000
    max_items: int = 10
    max_item_chars: int = 2000
    require_evidence_pointers: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of this budget."""
        return {"max_total_chars": self.max_total_chars, "max_items": self.max_items, "max_item_chars": self.max_item_chars, "require_evidence_pointers": self.require_evidence_pointers}


def build_evidence_map(bundle: SeedBundle, experiment_ids: dict[str, str] | None = None) -> dict[str, list[str]]:
    """Return evidence pointers per doc ID for packet assembly.

    Args:
        bundle: The seed bundle (provides finding edges + artifact digests).
        experiment_ids: Optional ``doc_id`` to committed experiment-UUID map
            (from :func:`seed_store`); when absent, pointers use the stable
            ``doc:`` pseudo-scheme so packets still validate structurally.

    Returns:
        ``doc_id`` to pointer list. Experiments point at their result
        artifact; failures at their experiment plus log artifact; findings
        at their supporting experiments.
    """
    artifact_by_doc: dict[str, list[str]] = {}
    for artifact in bundle.artifacts:
        artifact_by_doc.setdefault(artifact.doc_id, []).append(f"artifact://sha256/{artifact.sha256}")
    evidence: dict[str, list[str]] = {}
    for seed in bundle.experiments:
        pointers = [f"experiment:{experiment_ids[seed.doc_id]}" if experiment_ids and seed.doc_id in experiment_ids else f"doc:{seed.doc_id}"]
        pointers.extend(sorted(artifact_by_doc.get(seed.doc_id, [])))
        evidence[seed.doc_id] = pointers
    for finding in bundle.findings:
        pointers = []
        for edge in finding.payload.get("evidence", []):
            exp_doc = edge.get("experiment_doc_id", "")
            if experiment_ids and exp_doc in experiment_ids:
                pointers.append(f"experiment:{experiment_ids[exp_doc]}")
            elif exp_doc:
                pointers.append(f"doc:{exp_doc}")
        pointers.extend(sorted(artifact_by_doc.get(finding.doc_id, [])))
        evidence[finding.doc_id] = pointers
    return evidence


def assemble_packet(query_id: str, ranked_ids: list[str], doc_index: dict[str, Any], evidence_map: dict[str, list[str]], budget: PacketBudget | None = None) -> dict[str, Any]:
    """Assemble a KB-shaped context packet from one ranking.

    Sections follow ``knowledge_base.md`` (consensus / closest prior
    experiments / relevant failures / ...): findings land in
    ``consensus``, experiments in ``prior_experiments``, failures in
    ``failures``. The remaining sections (contradictions, assumptions,
    priors/skills, open questions) are empty lists until the Phase 3+
    owners populate them — the packet shape stays stable from Phase 2 on.

    Args:
        query_id: Eval query ID (recorded on the packet).
        ranked_ids: Ranked doc IDs, best first (already cut at ``k``).
        doc_index: ``doc_id`` to runner ``EvalDocument``.
        evidence_map: ``doc_id`` to evidence pointers (see :func:`build_evidence_map`).
        budget: Budget recorded on the packet (default: :class:`PacketBudget`).

    Returns:
        A JSON-serializable packet dict with sections, items, and sizes.
    """
    active = budget or PacketBudget()
    items: list[dict[str, Any]] = []
    sections: dict[str, list[dict[str, Any]]] = {name: [] for name in PACKET_SECTIONS}
    for doc_id in ranked_ids:
        doc = doc_index.get(doc_id)
        if doc is None:
            continue
        item = {"id": doc_id, "kind": doc.kind, "title": doc.title, "text": doc.text, "evidence": list(evidence_map.get(doc_id, []))}
        items.append(item)
        if doc.kind == "finding":
            sections["consensus"].append(item)
        elif doc.kind == "failure":
            sections["failures"].append(item)
        else:
            sections["prior_experiments"].append(item)
    total_chars = len(canonical_json_text(items))
    return {"query_id": query_id, "k": len(ranked_ids), "budget": active.to_dict(), "sections": sections, "items": items, "item_count": len(items), "total_chars": total_chars}


def check_packet_budget(packet: dict[str, Any], budget: PacketBudget | None = None) -> list[str]:
    """Return budget violations for one packet (empty when within budget)."""
    active = budget or PacketBudget()
    violations: list[str] = []
    items = packet.get("items", [])
    if len(items) > active.max_items:
        violations.append(f"item_count {len(items)} exceeds max_items {active.max_items}")
    if packet.get("total_chars", 0) > active.max_total_chars:
        violations.append(f"total_chars {packet.get('total_chars')} exceeds max_total_chars {active.max_total_chars}")
    for item in items:
        size = len(item.get("title", "")) + len(item.get("text", ""))
        if size > active.max_item_chars:
            violations.append(f"item {item.get('id')!r} size {size} exceeds max_item_chars {active.max_item_chars}")
        if active.require_evidence_pointers and item.get("kind") in ("finding", "failure") and not item.get("evidence"):
            violations.append(f"item {item.get('id')!r} ({item.get('kind')}) has no evidence pointer")
    return violations


def assert_packet_budget(packet: dict[str, Any], budget: PacketBudget | None = None) -> None:
    """Raise :class:`PacketBudgetError` when a packet exceeds its budget."""
    violations = check_packet_budget(packet, budget)
    if violations:
        raise PacketBudgetError(str(packet.get("query_id", "?")), violations)


@dataclass(frozen=True)
class PacketReport:
    """Whole-fixture packet-budget outcome for one retriever at one cutoff."""

    retriever: str
    k: int
    n_packets: int
    n_over_budget: int
    max_total_chars: int
    max_items: int
    violations_by_query: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when every packet fits the budget."""
        return self.n_over_budget == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of this report."""
        return {
            "retriever": self.retriever,
            "k": self.k,
            "n_packets": self.n_packets,
            "n_over_budget": self.n_over_budget,
            "max_total_chars": self.max_total_chars,
            "max_items": self.max_items,
            "passed": self.passed,
            "violations_by_query": {qid: list(v) for qid, v in sorted(self.violations_by_query.items())},
        }


def evaluate_with_packet_checks(fixture: Any, retrieve: Any, evidence_map: dict[str, list[str]], k: int, budget: PacketBudget | None = None, retriever_name: str = "custom", target: float = DEFAULT_TARGET) -> tuple[Any, PacketReport]:
    """Score a retriever and assert packet budgets (extended runner contract).

    Runs the frozen runner's :func:`evaluate` untouched, then assembles one
    context packet per query from the cleaned ranking the runner scored and
    checks each packet against ``budget``. ``runner.py`` is not modified:
    this wrapper is the extension point the Phase 2 exit criterion adds.

    Args:
        fixture: Validated (possibly merged) eval fixture.
        retrieve: Ranking callable ``(query, k) -> list[str]``.
        evidence_map: ``doc_id`` to evidence pointers.
        k: Retrieval cutoff (positive int).
        budget: Packet budget (default: :class:`PacketBudget`).
        retriever_name: Label recorded on both reports.
        target: Recall target recorded on the eval report.

    Returns:
        ``(eval_report, packet_report)``; the eval report is the runner's
        ``EvalReport`` with its ``passed`` semantics unchanged.
    """
    runner = load_runner()
    report = runner.evaluate(fixture, retrieve, k, retriever_name=retriever_name, target=target)
    active = budget or PacketBudget()
    doc_index = fixture.doc_index()
    violations: dict[str, tuple[str, ...]] = {}
    max_chars = 0
    max_items = 0
    over = 0
    for scored in report.queries:
        packet = assemble_packet(scored.query_id, list(scored.retrieved), doc_index, evidence_map, active)
        max_chars = max(max_chars, packet["total_chars"])
        max_items = max(max_items, packet["item_count"])
        problems = check_packet_budget(packet, active)
        if problems:
            over += 1
            violations[scored.query_id] = tuple(problems)
    packet_report = PacketReport(retriever=retriever_name, k=k, n_packets=len(report.queries), n_over_budget=over, max_total_chars=max_chars, max_items=max_items, violations_by_query=violations)
    return report, packet_report


def phase2_exit_verdict(report: Any, packet_report: PacketReport, target: float = DEFAULT_TARGET) -> tuple[bool, list[str]]:
    """Judge the Phase 2 exit criterion: recall + failure-recall + packets.

    Exit requires **all** of: ``mean_recall >= target``,
    ``mean_failure_recall >= target`` (when the fixture names failures),
    zero errored queries, and every context packet within budget.

    Args:
        report: The runner's ``EvalReport``.
        packet_report: The :class:`PacketReport` from :func:`evaluate_with_packet_checks`.
        target: Pass target for both recall means (default 0.8).

    Returns:
        ``(passed, reasons)`` with one line per unmet condition (empty when green).
    """
    reasons: list[str] = []
    if report.n_queries_errored:
        reasons.append(f"{report.n_queries_errored} queries errored")
    if report.mean_recall < target:
        reasons.append(f"mean recall@{report.k} {report.mean_recall:.4f} < target {target}")
    if report.mean_failure_recall is not None and report.mean_failure_recall < target:
        reasons.append(f"mean failure-recall@{report.k} {report.mean_failure_recall:.4f} < target {target}")
    if not packet_report.passed:
        reasons.append(f"{packet_report.n_over_budget}/{packet_report.n_packets} packets over budget")
    return (not reasons, reasons)


# ---------------------------------------------------------------------------
# Commit path (real write_api round-trip)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedCommitMap:
    """Result of committing a bundle: fixture IDs to stored record IDs."""

    experiment_ids: dict[str, str]
    failure_ids: dict[str, str]
    family_hashes: dict[str, str]
    execution_hashes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of this commit map."""

        def sorted_map(mapping: dict[str, str]) -> dict[str, str]:
            return dict(sorted(mapping.items()))

        return {"experiment_ids": sorted_map(self.experiment_ids), "failure_ids": sorted_map(self.failure_ids), "family_hashes": sorted_map(self.family_hashes), "execution_hashes": sorted_map(self.execution_hashes)}


def seed_store(store: Any, bundle: SeedBundle) -> SeedCommitMap:
    """Commit every experiment seed through the real ``write_api`` validators.

    For each seed: ``experiment_begin`` (idempotent on the begin key),
    then ``failure_record`` for failures, then ``experiment_commit``. All
    validation is the production path — no reimplementation — so a clean
    return proves the payloads are committable.

    Args:
        store: An ``ExperimentStore`` implementation (fake, SQLite, or PG).
        bundle: The seed bundle to commit.

    Returns:
        The :class:`SeedCommitMap` linking fixture IDs to stored rows.
    """
    write_api = require_deerflow("knowledge.write_api")
    experiment_ids: dict[str, str] = {}
    failure_ids: dict[str, str] = {}
    family_hashes: dict[str, str] = {}
    execution_hashes: dict[str, str] = {}
    for seed in sorted(bundle.experiments, key=lambda s: s.doc_id):
        record = write_api.experiment_begin(store, **seed.begin_kwargs)
        if seed.failure_kwargs is not None:
            failure = write_api.failure_record(store, experiment_id=record.id, **seed.failure_kwargs)
            failure_ids[seed.doc_id] = failure.id
        committed = write_api.experiment_commit(store, experiment_id=record.id, **seed.commit_kwargs)
        experiment_ids[seed.doc_id] = committed.id
        family_hashes[seed.doc_id] = committed.family_hash
        execution_hashes[seed.doc_id] = committed.execution_hash
    return SeedCommitMap(experiment_ids=experiment_ids, failure_ids=failure_ids, family_hashes=family_hashes, execution_hashes=execution_hashes)


def resolve_finding_evidence(bundle: SeedBundle, commit_map: SeedCommitMap) -> tuple[dict[str, Any], ...]:
    """Return finding payloads with evidence ``experiment_id`` resolved.

    Args:
        bundle: The seed bundle (evidence edges carry ``experiment_doc_id``).
        commit_map: The :class:`SeedCommitMap` from :func:`seed_store`.

    Returns:
        One validated payload per finding, in doc-ID order.

    Raises:
        SeedValidationError: When an edge references an uncommitted experiment.
    """
    resolved: list[dict[str, Any]] = []
    for finding in sorted(bundle.findings, key=lambda f: f.doc_id):
        payload = json.loads(json.dumps(finding.payload))
        for edge in payload.get("evidence", []):
            exp_doc = edge.get("experiment_doc_id", "")
            if exp_doc and exp_doc not in commit_map.experiment_ids:
                raise SeedValidationError(f"finding {finding.doc_id!r} references uncommitted experiment {exp_doc!r}")
            edge["experiment_id"] = commit_map.experiment_ids.get(exp_doc)
        validate_finding_payload(payload)
        resolved.append(payload)
    return tuple(resolved)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(prog="seed_phase2.py", description="Phase 2 eval seeder: fixture documents to committable knowledge payloads.")
    parser.add_argument("--phase0", default=str(default_phase0_path()), help="Path to eval_fixture.json (default: sibling file).")
    parser.add_argument("--phase2", default=str(default_phase2_path()), help="Path to eval_fixture_phase2.json (default: sibling file).")
    parser.add_argument("--check", action="store_true", help="Check judgment coverage + payload counts and exit.")
    parser.add_argument("--json", action="store_true", help="With --check: emit machine-readable JSON.")
    parser.add_argument("--emit", metavar="PATH", default=None, help="Write the deterministic seed bundle JSON to PATH.")
    parser.add_argument("--validate-writes", action="store_true", help="Commit the bundle through write_api on an in-memory store (needs deerflow).")
    parser.add_argument("--self-test", action="store_true", help="Run the embedded unit-test suite and exit.")
    return parser


def format_check_text(bundle: SeedBundle) -> str:
    """Render a human-readable summary of bundle coverage and counts."""
    coverage = bundle.coverage
    lines = [
        f"seed_version={bundle.seed_version} fixture_version={bundle.fixture_version}",
        f"project_id={bundle.project_id}",
        f"run_id={bundle.run_id}",
        f"documents={coverage.n_documents} queries={coverage.n_queries} failure_queries={coverage.n_failure_queries}",
        f"experiments={len(bundle.experiments)} findings={len(bundle.findings)} artifacts={len(bundle.artifacts)}",
        f"judgments={len(bundle.judgments)} coverage={'OK' if coverage.ok else 'FAIL'}",
    ]
    for problem in coverage.violations():
        lines.append(f"  violation: {problem}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0/1/2)."""
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()
    try:
        bundle = build_seed_bundle(args.phase0, args.phase2)
    except SeedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.emit:
        try:
            out = emit_bundle(bundle, args.emit)
        except OSError as exc:
            print(f"error: cannot write {args.emit}: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {out} ({len(bundle.experiments)} experiments, {len(bundle.findings)} findings, {len(bundle.artifacts)} artifacts)")
    if args.validate_writes:
        try:
            commit_map = seed_store(_FakeExperimentStore(), bundle)
        except SeedError as exc:
            print(f"error: write validation failed: {exc}", file=sys.stderr)
            return 1
        print(f"write_api round-trip OK: {len(commit_map.experiment_ids)} experiments committed, {len(commit_map.failure_ids)} failures recorded")
    if args.check or (args.emit is None and not args.validate_writes):
        if args.json:
            summary = {"seed_version": bundle.seed_version, "fixture_version": bundle.fixture_version, "n_experiments": len(bundle.experiments), "n_findings": len(bundle.findings), "n_artifacts": len(bundle.artifacts)}
            print(json.dumps({"bundle": summary, "coverage": bundle.coverage.to_dict()}, indent=2, sort_keys=True))
        else:
            print(format_check_text(bundle), end="")
    return 0 if bundle.coverage.ok else 2


# ---------------------------------------------------------------------------
# Self-test (stdlib unittest; `python seed_phase2.py --self-test`)
# ---------------------------------------------------------------------------


class _FakeExperimentStore:
    """In-memory ``ExperimentStore`` for write-path validation (no DB)."""

    def __init__(self) -> None:
        self.experiments: dict[str, Any] = {}
        self.by_begin_key: dict[str, Any] = {}
        self.by_commit_key: dict[str, Any] = {}
        self.by_execution: dict[str, Any] = {}
        self.failures: dict[str, Any] = {}
        self.by_failure_key: dict[str, Any] = {}
        self.assumptions: dict[str, Any] = {}
        self.by_assumption_key: dict[str, Any] = {}

    def find_experiment_by_idempotency_key(self, key: str) -> Any | None:
        """Return the experiment begun under ``key``, or None."""
        return self.by_begin_key.get(key)

    def find_experiment_by_commit_key(self, key: str) -> Any | None:
        """Return the experiment committed under ``key``, or None."""
        return self.by_commit_key.get(key)

    def find_experiment_by_execution_hash(self, value: str) -> Any | None:
        """Return the experiment with this execution hash, or None."""
        return self.by_execution.get(value)

    def get_experiment(self, experiment_id: str) -> Any | None:
        """Return the experiment by ID, or None."""
        return self.experiments.get(experiment_id)

    def insert_experiment(self, record: Any) -> None:
        """Persist a new experiment row."""
        self.experiments[record.id] = record
        self.by_execution[record.execution_hash] = record
        if record.idempotency_key is not None:
            self.by_begin_key[record.idempotency_key] = record
        if record.commit_idempotency_key is not None:
            self.by_commit_key[record.commit_idempotency_key] = record

    def update_experiment(self, record: Any) -> None:
        """Persist a full-row update of an existing experiment."""
        self.insert_experiment(record)

    def find_failure_by_idempotency_key(self, key: str) -> Any | None:
        """Return the failure recorded under ``key``, or None."""
        return self.by_failure_key.get(key)

    def insert_failure(self, record: Any) -> None:
        """Persist a new failure detail row."""
        self.failures[record.id] = record
        if record.idempotency_key is not None:
            self.by_failure_key[record.idempotency_key] = record

    def find_assumption_by_idempotency_key(self, key: str) -> Any | None:
        """Return the assumption recorded under ``key``, or None."""
        return self.by_assumption_key.get(key)

    def insert_assumption(self, record: Any) -> None:
        """Persist a new assumption row."""
        self.assumptions[record.id] = record
        if record.idempotency_key is not None:
            self.by_assumption_key[record.idempotency_key] = record


class _MemoryObjectBackend:
    """In-memory object backend for artifact round-trip tests (no FS/S3)."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, key: str, stream: Any, *, content_type: str = "application/octet-stream", content_length: int | None = None) -> str:
        """Store stream bytes under ``key``; return a memory URI."""
        del content_type, content_length
        self.objects[key] = stream.read()
        return f"memory://{key}"

    def get_object(self, key: str) -> bytes:
        """Return stored bytes; raise KeyError when missing."""
        return self.objects[key]

    def open_stream(self, key: str) -> Any:
        """Return a readable stream; raise KeyError when missing."""
        import io as _io

        return _io.BytesIO(self.objects[key])

    def exists(self, key: str) -> bool:
        """Return True when ``key`` exists."""
        return key in self.objects

    def stat(self, key: str) -> int:
        """Return the byte size under ``key``; raise KeyError when missing."""
        return len(self.objects[key])

    def delete(self, key: str) -> None:
        """Delete ``key`` if present."""
        self.objects.pop(key, None)

    def storage_uri(self, key: str) -> str:
        """Return the memory URI locating ``key``."""
        return f"memory://{key}"


class _DeterminismTests(unittest.TestCase):
    def test_bundle_is_byte_identical_across_builds(self) -> None:
        """Two builds from the same fixtures serialize identically."""
        first = json.dumps(build_seed_bundle().to_dict(), sort_keys=True)
        second = json.dumps(build_seed_bundle().to_dict(), sort_keys=True)
        self.assertEqual(first, second)

    def test_stable_uuids_are_valid_and_unique(self) -> None:
        """Derived UUIDs parse and never collide within a bundle."""
        bundle = build_seed_bundle()
        ids = [bundle.project_id, bundle.run_id]
        for seed in bundle.experiments:
            ids.extend([seed.begin_kwargs["project_id"], seed.begin_kwargs["created_by_run_id"], *seed.dataset_version_ids, *seed.result_artifact_ids])
        for finding in bundle.findings:
            ids.append(finding.payload["finding_id"])
        for artifact in bundle.artifacts:
            ids.append(artifact.artifact_id)
        for value in ids:
            uuid.UUID(value)
        result_ids = [a.artifact_id for a in bundle.artifacts if a.kind == "result"]
        self.assertEqual(len(set(result_ids)), len(result_ids))
        self.assertEqual(len({stable_uuid(f"experiment:{s.doc_id}") for s in bundle.experiments}), len(bundle.experiments))

    def test_idempotency_keys_unique_and_whitespace_free(self) -> None:
        """Every seeded write carries a unique opaque key."""
        bundle = build_seed_bundle()
        keys: list[str] = []
        for seed in bundle.experiments:
            keys.append(seed.begin_kwargs["idempotency_key"])
            keys.append(seed.commit_kwargs["idempotency_key"])
            if seed.failure_kwargs is not None:
                keys.append(seed.failure_kwargs["idempotency_key"])
        for finding in bundle.findings:
            keys.append(finding.payload["idempotency_key"])
        self.assertEqual(len(set(keys)), len(keys))
        for key in keys:
            self.assertTrue(key)
            self.assertFalse(any(ch.isspace() for ch in key))
            self.assertLessEqual(len(key), 255)

    def test_canonical_json_matches_hashing_spec(self) -> None:
        """The stdlib canonical form agrees with deerflow hashing on seeds."""
        hashing = require_deerflow("knowledge.hashing")
        bundle = build_seed_bundle()
        seed = bundle.experiments[0]
        self.assertEqual(canonical_json_text(seed.begin_kwargs["methodology"]), hashing.canonical_json(seed.begin_kwargs["methodology"]))
        self.assertEqual(sha256_hex(b"abc"), hashing.sha256_hex(b"abc"))


class _WriteApiTests(unittest.TestCase):
    def test_full_bundle_commits_cleanly(self) -> None:
        """Every experiment seed begins + commits through real validators."""
        bundle = build_seed_bundle()
        commit_map = seed_store(_FakeExperimentStore(), bundle)
        self.assertEqual(len(commit_map.experiment_ids), len(bundle.experiments))
        n_failures = sum(1 for s in bundle.experiments if s.kind == "failure")
        self.assertEqual(len(commit_map.failure_ids), n_failures)
        self.assertEqual(len(set(commit_map.execution_hashes.values())), len(bundle.experiments))

    def test_terminal_statuses_and_outcomes(self) -> None:
        """Commits land terminal with the kind-correct outcome."""
        write_api = require_deerflow("knowledge.write_api")
        bundle = build_seed_bundle()
        store = _FakeExperimentStore()
        seed_store(store, bundle)
        for seed in bundle.experiments:
            record = store.get_experiment(next(r.id for r in store.experiments.values() if r.idempotency_key == seed.begin_kwargs["idempotency_key"]))
            assert record is not None
            if seed.kind == "failure":
                self.assertEqual(record.status, "failed")
                self.assertEqual(record.outcome, "failure")
                self.assertIn(record.failure_class, write_api.FAILURE_CLASSES)
            else:
                self.assertEqual(record.status, "completed")
                self.assertEqual(record.outcome, "success")
                self.assertIsNone(record.failure_class)

    def test_idempotent_replay_returns_originals(self) -> None:
        """Seeding the same store twice replays instead of duplicating."""
        bundle = build_seed_bundle()
        store = _FakeExperimentStore()
        first = seed_store(store, bundle)
        second = seed_store(store, bundle)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(len(store.experiments), len(bundle.experiments))

    def test_tampered_payload_is_rejected(self) -> None:
        """write_api validators still guard seeded shapes (negative test)."""
        write_api = require_deerflow("knowledge.write_api")
        bundle = build_seed_bundle()
        seed = next(s for s in bundle.experiments if s.kind == "failure")
        bad = dict(seed.commit_kwargs)
        bad["failure_class"] = "not-a-class"
        store = _FakeExperimentStore()
        record = write_api.experiment_begin(store, **seed.begin_kwargs)
        with self.assertRaises(write_api.KnowledgeValidationError):
            write_api.experiment_commit(store, experiment_id=record.id, **bad)

    def test_family_hash_matches_hashing_recomputation(self) -> None:
        """Stored family hashes equal hashing-spec recomputation."""
        hashing = require_deerflow("knowledge.hashing")
        bundle = build_seed_bundle()
        store = _FakeExperimentStore()
        commit_map = seed_store(store, bundle)
        by_key = {r.idempotency_key: r for r in store.experiments.values()}
        for seed in bundle.experiments[:5]:
            record = by_key[seed.begin_kwargs["idempotency_key"]]
            design = {"hypothesis": hashing.normalize_text(seed.begin_kwargs["hypothesis"]), "methodology": seed.begin_kwargs["methodology"]}
            self.assertEqual(record.family_hash, hashing.experiment_family_hash(design))
            self.assertEqual(commit_map.family_hashes[seed.doc_id], record.family_hash)


class _FindingPayloadTests(unittest.TestCase):
    def test_all_findings_validate_and_stay_candidate(self) -> None:
        """Every finding payload validates with candidate status + evidence."""
        bundle = build_seed_bundle()
        self.assertGreaterEqual(len(bundle.findings), 6)
        for finding in bundle.findings:
            validate_finding_payload(finding.payload)
            self.assertEqual(finding.payload["status"], "candidate")
            self.assertIn(finding.payload["finding_type"], FINDING_TYPES)
            self.assertGreaterEqual(len(finding.payload["evidence"]), 1)

    def test_canonical_keys_unique(self) -> None:
        """Canonical keys are unique across the bundle."""
        bundle = build_seed_bundle()
        keys = [f.payload["canonical_key"] for f in bundle.findings]
        self.assertEqual(len(set(keys)), len(keys))

    def test_evidence_resolves_post_commit(self) -> None:
        """Evidence edges resolve to committed experiment UUIDs."""
        bundle = build_seed_bundle()
        commit_map = seed_store(_FakeExperimentStore(), bundle)
        resolved = resolve_finding_evidence(bundle, commit_map)
        self.assertEqual(len(resolved), len(bundle.findings))
        for payload in resolved:
            for edge in payload["evidence"]:
                self.assertIn(edge["experiment_doc_id"], commit_map.experiment_ids)
                self.assertEqual(edge["experiment_id"], commit_map.experiment_ids[edge["experiment_doc_id"]])

    def test_validator_rejects_bad_payloads(self) -> None:
        """The finding validator rejects each corruption class."""
        bundle = build_seed_bundle()
        good = bundle.findings[0].payload
        corruptions = [
            ("finding_type", "vibes"),
            ("status", "validated"),
            ("statement", "  "),
            ("canonical_key", "Has Spaces!"),
            ("scope", {"nonsense_key": "x"}),
            ("confidence", {"overall_tier": "extreme"}),
        ]
        for key, bad_value in corruptions:
            with self.subTest(field=key):
                payload = json.loads(json.dumps(good))
                payload[key] = bad_value
                with self.assertRaises(SeedValidationError):
                    validate_finding_payload(payload)
        payload = json.loads(json.dumps(good))
        payload["evidence"].append({"evidence_type": "experiment", "relation": "supports", "locator": {}, "evidence_weight": 0})
        with self.assertRaises(SeedValidationError):
            validate_finding_payload(payload)


class _ArtifactPayloadTests(unittest.TestCase):
    def test_digests_cover_bytes(self) -> None:
        """Every artifact digest recomputes from its bytes."""
        bundle = build_seed_bundle()
        self.assertGreaterEqual(len(bundle.artifacts), bundle.coverage.n_documents)
        for artifact in bundle.artifacts:
            validate_artifact_seed(artifact)

    def test_result_artifact_ids_match_commit_links(self) -> None:
        """Commit result_artifacts reference the seeded result artifact IDs."""
        bundle = build_seed_bundle()
        result_ids = {a.doc_id: a.artifact_id for a in bundle.artifacts if a.kind == "result"}
        for seed in bundle.experiments:
            self.assertEqual(list(seed.commit_kwargs["result_artifacts"]), [result_ids[seed.doc_id]])
            self.assertEqual(list(seed.result_artifact_ids), [result_ids[seed.doc_id]])

    def test_store_round_trip_recomputes_digest(self) -> None:
        """ArtifactStore.put_bytes over seeds yields identical digests."""
        artifacts = require_deerflow("knowledge.artifacts.store")
        bundle = build_seed_bundle()
        store = artifacts.ArtifactStore(_MemoryObjectBackend())
        for artifact in bundle.artifacts[:8]:
            record = store.put_bytes(artifact.bytes_text.encode("utf-8"), kind=artifact.kind, media_type=artifact.media_type)
            self.assertEqual(record.sha256, artifact.sha256)
            self.assertEqual(record.byte_size, artifact.byte_size)
            self.assertEqual(store.get_bytes(record.sha256), artifact.bytes_text.encode("utf-8"))


class _JudgmentCoverageTests(unittest.TestCase):
    def test_merged_coverage_is_complete(self) -> None:
        """Merged fixtures satisfy every coverage invariant."""
        coverage = assert_judgment_coverage(load_merged_fixture())
        self.assertEqual(coverage.n_documents, 46)
        self.assertEqual(coverage.n_queries, 36)
        self.assertEqual(coverage.n_failure_queries, 36)
        self.assertTrue(coverage.ok)

    def test_phase2_extension_standalone_complete(self) -> None:
        """The Phase 2 extension alone satisfies every invariant."""
        runner = load_runner()
        coverage = assert_judgment_coverage(runner.load_fixture(str(default_phase2_path())))
        self.assertEqual(coverage.n_documents, 10)
        self.assertEqual(coverage.n_queries, 6)
        self.assertTrue(coverage.ok)

    def test_judgments_carry_failure_recall_cases(self) -> None:
        """Every query judgment includes failure-recall cases."""
        judgments = build_judgments(load_merged_fixture())
        self.assertEqual(len(judgments), 36)
        for query_id, judgment in judgments.items():
            self.assertTrue(judgment["relevant"], query_id)
            self.assertTrue(judgment["must_recall_failures"], query_id)
            for doc_id, grade in judgment["grades"].items():
                self.assertIn(grade, (1, 2), f"{query_id}:{doc_id}")

    def test_coverage_reports_violations(self) -> None:
        """The coverage checker names violations instead of hiding them."""
        runner = load_runner()
        fixture = runner.load_fixture(str(default_phase2_path()))
        broken = runner.EvalFixture(version=fixture.version, documents=fixture.documents, queries=fixture.queries[:1])
        coverage = check_judgment_coverage(broken)
        self.assertFalse(coverage.ok)
        self.assertTrue(coverage.violations())
        with self.assertRaises(SeedValidationError):
            assert_judgment_coverage(broken)


class _PacketBudgetTests(unittest.TestCase):
    def test_oracle_packets_fit_default_budget(self) -> None:
        """Perfect rankings still fit the fixed packet budget at k=10."""
        runner = load_runner()
        fixture = load_merged_fixture()
        bundle = build_seed_bundle()
        evidence = build_evidence_map(bundle)
        report, packet_report = evaluate_with_packet_checks(fixture, runner.oracle_retriever_factory(fixture), evidence, 10, retriever_name="oracle")
        self.assertTrue(report.passed)
        self.assertTrue(packet_report.passed, packet_report.to_dict())
        verdict, reasons = phase2_exit_verdict(report, packet_report)
        self.assertTrue(verdict, reasons)

    def test_null_retriever_fails_exit_on_recall(self) -> None:
        """Empty rankings fit the budget but fail the recall gates."""
        runner = load_runner()
        fixture = load_merged_fixture()
        evidence = build_evidence_map(build_seed_bundle())
        report, packet_report = evaluate_with_packet_checks(fixture, runner.null_retriever, evidence, 10, retriever_name="null")
        self.assertTrue(packet_report.passed)
        verdict, reasons = phase2_exit_verdict(report, packet_report)
        self.assertFalse(verdict)
        self.assertTrue(any("recall" in reason for reason in reasons))

    def test_budget_violations_detected(self) -> None:
        """Oversized / pointer-less packets raise with named violations."""
        fixture = load_merged_fixture()
        bundle = build_seed_bundle()
        doc_index = fixture.doc_index()
        query = fixture.queries[0]
        ranked = list(query.relevant_doc_ids) + list(query.must_recall_failures)
        packet = assemble_packet(query.query_id, ranked, doc_index, build_evidence_map(bundle))
        assert_packet_budget(packet)
        tight = PacketBudget(max_total_chars=10, max_items=100, max_item_chars=100000, require_evidence_pointers=False)
        with self.assertRaises(PacketBudgetError) as ctx:
            assert_packet_budget(packet, tight)
        self.assertIn("total_chars", str(ctx.exception))
        bare = assemble_packet(query.query_id, ranked, doc_index, {})
        with self.assertRaises(PacketBudgetError) as ctx2:
            assert_packet_budget(bare)
        self.assertIn("evidence pointer", str(ctx2.exception))

    def test_packet_sections_follow_kb_layout(self) -> None:
        """Packets carry all KB sections with kinds routed correctly."""
        runner = load_runner()
        fixture = load_merged_fixture()
        doc_index = fixture.doc_index()
        evidence = build_evidence_map(build_seed_bundle())
        report = runner.evaluate(fixture, runner.oracle_retriever_factory(fixture), 10, retriever_name="oracle")
        packet = assemble_packet(report.queries[0].query_id, list(report.queries[0].retrieved), doc_index, evidence)
        self.assertEqual(tuple(packet["sections"]), PACKET_SECTIONS)
        for item in packet["sections"]["consensus"]:
            self.assertEqual(item["kind"], "finding")
        for item in packet["sections"]["failures"]:
            self.assertEqual(item["kind"], "failure")
        for item in packet["sections"]["prior_experiments"]:
            self.assertEqual(item["kind"], "experiment")


class _RunnerCompatTests(unittest.TestCase):
    def test_phase2_fixture_loads_in_frozen_runner(self) -> None:
        """The extension fixture scores under unmodified runner.py."""
        runner = load_runner()
        fixture = runner.load_fixture(str(default_phase2_path()))
        oracle = runner.evaluate(fixture, runner.oracle_retriever_factory(fixture), 10, retriever_name="oracle")
        self.assertTrue(oracle.passed)
        null = runner.evaluate(fixture, runner.null_retriever, 10, retriever_name="null")
        self.assertFalse(null.passed)
        self.assertEqual(null.mean_recall, 0.0)

    def test_runner_module_was_not_edited_for_phase2(self) -> None:
        """runner.py carries no Phase 2 symbols (extension lives here)."""
        runner = load_runner()
        for symbol in ("seed", "packet", "PacketBudget", "phase2", "SeedBundle"):
            self.assertFalse(any(symbol in name.lower() for name in dir(runner)), symbol)


class _CliTests(unittest.TestCase):
    def test_check_commands(self) -> None:
        """--check exits 0 in text and JSON modes."""
        self.assertEqual(main(["--check"]), 0)
        self.assertEqual(main(["--check", "--json"]), 0)

    def test_emit_and_validate_writes(self) -> None:
        """--emit writes a reloadable bundle; --validate-writes commits it."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "bundle.json")
            self.assertEqual(main(["--emit", out]), 0)
            payload = json.loads(Path(out).read_text(encoding="utf-8"))
            self.assertEqual(payload["seed_version"], SEED_VERSION)
            self.assertEqual(len(payload["experiments"]) + len(payload["findings"]), 46)
            self.assertEqual(main(["--validate-writes"]), 0)

    def test_missing_fixture_exits_1(self) -> None:
        """Unknown fixture paths exit 1 (usage/data error, not crash)."""
        self.assertEqual(main(["--phase2", "/nonexistent/eval_fixture_phase2.json", "--check"]), 1)


def run_self_test() -> int:
    """Run the embedded unittest suite. Returns 0 on success, 1 on failure."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        [
            loader.loadTestsFromTestCase(_DeterminismTests),
            loader.loadTestsFromTestCase(_WriteApiTests),
            loader.loadTestsFromTestCase(_FindingPayloadTests),
            loader.loadTestsFromTestCase(_ArtifactPayloadTests),
            loader.loadTestsFromTestCase(_JudgmentCoverageTests),
            loader.loadTestsFromTestCase(_PacketBudgetTests),
            loader.loadTestsFromTestCase(_RunnerCompatTests),
            loader.loadTestsFromTestCase(_CliTests),
        ]
    )
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

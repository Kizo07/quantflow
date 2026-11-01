"""Standalone tests for the Research Knowledge Plane agent tools (Phase 2).

Covers ``deerflow.knowledge.agent_tools`` (bootstrap intent parsing, retrieval
planning, context-packet assembly, lookup pure functions, artifact
locator reads, the ``@tool`` wrappers and the run-start middleware)
with in-memory fakes behind the storage-boundary protocols — no
database, no config files, no integration wiring.

Run from anywhere (the bootstrap below locates the harness package)::

    python -m pytest backend/packages/harness/deerflow/knowledge/agent_tools/test_knowledge_tools.py -q
"""

import asyncio
import json
import sys
import types
import uuid
from datetime import UTC, datetime
from pathlib import Path

_HARNESS_DIR = Path(__file__).resolve().parents[3]
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

import pytest  # noqa: E402
import sqlalchemy as sa  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

import deerflow.knowledge.schema as kb_schema  # noqa: E402
from deerflow.knowledge import pg_retrieval as pg_mod  # noqa: E402
from deerflow.knowledge.agent_tools import bootstrap as kb_bootstrap  # noqa: E402
from deerflow.knowledge.agent_tools import lookup as kb_lookup  # noqa: E402
from deerflow.knowledge.agent_tools import middleware as kb_middleware  # noqa: E402
from deerflow.knowledge.artifacts.backends import LocalFilesystemBackend  # noqa: E402
from deerflow.knowledge.artifacts.store import ArtifactStore  # noqa: E402
from deerflow.knowledge.contract import KnowledgeRequestContext  # noqa: E402
from deerflow.knowledge.schema.experiments import AssumptionRow, ExperimentRow  # noqa: E402
from deerflow.knowledge.schema.findings import FindingRow  # noqa: E402
from deerflow.knowledge.write_api import ExperimentRecord, KnowledgeError, KnowledgeValidationError  # noqa: E402
from deerflow.knowledge_scope import KNOWLEDGE_SCOPE_RUNTIME_KEY  # noqa: E402
from deerflow.persistence.base import Base  # noqa: E402
from deerflow.runtime.user_context import resolve_runtime_user_id  # noqa: E402

PROJECT_ID = "11111111-1111-1111-1111-111111111111"
RUN_ID = "33333333-3333-3333-3333-333333333333"
EXP_ID = "55555555-5555-5555-5555-555555555555"
EXEC_HASH = "a" * 64
FAMILY_HASH = "b" * 64


def make_experiment(**overrides):
    payload = {
        "id": EXP_ID,
        "project_id": PROJECT_ID,
        "created_by_run_id": RUN_ID,
        "hypothesis": "Cross-sectional momentum persists net of costs.",
        "methodology": {"universe": "sp500-pit", "frequency": "daily"},
        "parameters": {},
        "datasets": [],
        "code": {},
        "environment": {},
        "family_hash": FAMILY_HASH,
        "execution_hash": EXEC_HASH,
        "status": "completed",
        "outcome": "success",
        "started_at": "2026-01-01T00:00:00+00:00",
    }
    payload.update(overrides)
    return ExperimentRecord(**payload)


class FakeRetrieval:
    """In-memory KnowledgeRetrievalBackend fake."""

    def __init__(self, docs=None, fail_search=False, fail_get=False):
        self.docs = {doc.id: doc for doc in (docs or [])}
        self.search_calls: list[dict] = []
        self.fail_search = fail_search
        self.fail_get = fail_get

    def search(self, query, *, kinds, filters, limit, offset):
        self.search_calls.append({"query": query, "kinds": kinds, "filters": filters, "limit": limit, "offset": offset})
        if self.fail_search:
            raise RuntimeError("retrieval outage")
        hits = [doc for doc in self.docs.values() if doc.kind in kinds]
        page = hits[offset : offset + limit]
        return kb_lookup.RetrievalPage(documents=list(page), limit=limit, offset=offset, total=len(hits))

    def get_documents(self, ids, *, include_evidence):
        if self.fail_get:
            raise RuntimeError("retrieval outage")
        return [self.docs.get(doc_id) for doc_id in ids]


class FakeExperimentLookup:
    """In-memory ExperimentLookupStore fake."""

    def __init__(self, records=None):
        self.records = list(records or [])
        self.used_family = 0
        self.used_search = 0

    def find_by_id(self, experiment_id):
        return next((record for record in self.records if record.id == experiment_id), None)

    def find_by_execution_hash(self, execution_hash):
        return next((record for record in self.records if record.execution_hash == execution_hash), None)

    def find_by_family_hash(self, family_hash, *, limit, offset):
        self.used_family += 1
        hits = [record for record in self.records if record.family_hash == family_hash]
        return hits[offset : offset + limit]

    def search_experiments(self, filters, *, limit, offset):
        self.used_search += 1
        hits = list(self.records)
        if filters.status is not None:
            hits = [record for record in hits if record.status == filters.status]
        if filters.outcome is not None:
            hits = [record for record in hits if record.outcome == filters.outcome]
        if filters.hypothesis_contains is not None:
            needle = filters.hypothesis_contains.lower()
            hits = [record for record in hits if needle in record.hypothesis.lower()]
        return hits[offset : offset + limit]


class FakeSkills:
    """In-memory SkillCatalog fake."""

    def __init__(self, skills=None):
        self.skills = (
            skills
            if skills is not None
            else [
                {"id": "cross-sectional-factor-backtest@2.3", "name": "cross-sectional-factor-backtest", "description": "Robust factor backtest harness.", "version": "2.3", "score": 0.9},
            ]
        )
        self.calls: list[dict] = []

    def find_skills(self, concepts, *, limit):
        self.calls.append({"concepts": list(concepts), "limit": limit})
        return list(self.skills[:limit])


def make_runtime(**context):
    return types.SimpleNamespace(context=dict(context), server_info=None)


def make_doc(doc_id="F17", kind="finding", **overrides):
    payload = {
        "id": doc_id,
        "kind": kind,
        "title": f"Title {doc_id}",
        "summary": f"Summary {doc_id}",
        "score": 0.8,
        "status": "validated",
        "evidence_refs": [EXP_ID],
        "payload": {},
    }
    payload.update(overrides)
    return kb_lookup.RetrievalDocument(**payload)


@pytest.fixture(autouse=True)
def _clean_bindings():
    yield
    kb_lookup.reset_knowledge_backends()
    kb_bootstrap.bind_knowledge_skills(None)


# -- research intent parsing ------------------------------------------------


def test_intent_from_text_extracts_structured_fields():
    intent = kb_bootstrap.intent_from_text("Study cross-sectional momentum in US equities with daily data, 3 months to 12 months horizon, turnover and transaction costs, 2000-01-01 to 2025-12-31.")
    assert intent.asset_class == "equity"
    assert "US" in intent.markets
    assert intent.universe == "liquid common stocks"
    assert intent.frequency == "daily"
    assert "momentum" in intent.concepts
    assert "turnover" in intent.concepts
    assert intent.horizon is not None and "month" in intent.horizon
    assert intent.requested_period == ["2000-01-01", "2025-12-31"]
    assert set(intent.needed_memory) == set(kb_bootstrap.DEFAULT_NEEDED_MEMORY)


def test_intent_from_text_multi_asset_and_years():
    intent = kb_bootstrap.intent_from_text("Compare equity momentum with bond carry from 2010 to 2020 on global futures.")
    assert intent.asset_class == "multi_asset"
    assert "GL" in intent.markets
    assert intent.requested_period == ["2010-01-01", "2020-12-31"]


def test_intent_from_text_rejects_empty():
    with pytest.raises(KnowledgeValidationError):
        kb_bootstrap.intent_from_text("   ")


def test_intent_from_mapping_round_trip():
    intent = kb_bootstrap.intent_from_mapping(
        {
            "topic": "cross-sectional momentum",
            "asset_class": "equity",
            "markets": ["US"],
            "universe": "liquid common stocks",
            "horizon": "3-12 months",
            "frequency": "daily/monthly",
            "concepts": ["momentum", "turnover"],
            "requested_period": ["2000-01-01", "2026-09-19"],
            "needed_memory": ["validated_findings", "failures"],
        }
    )
    assert intent.topic == "cross-sectional momentum"
    assert intent.needed_memory == ["validated_findings", "failures"]
    assert kb_bootstrap.intent_from_mapping(intent.to_dict()).to_dict() == intent.to_dict()


def test_intent_from_mapping_rejects_unknown_keys_and_bad_values():
    with pytest.raises(KnowledgeValidationError):
        kb_bootstrap.intent_from_mapping({"topic": "x", "bogus": 1})
    with pytest.raises(KnowledgeValidationError):
        kb_bootstrap.intent_from_mapping({"topic": "x", "asset_class": "wine"})
    with pytest.raises(KnowledgeValidationError):
        kb_bootstrap.intent_from_mapping({"topic": "x", "requested_period": ["2026-01-01", "2025-01-01"]})
    with pytest.raises(KnowledgeValidationError):
        kb_bootstrap.intent_from_mapping({"topic": "x", "needed_memory": ["dreams"]})
    with pytest.raises(KnowledgeValidationError):
        kb_bootstrap.intent_from_mapping({"asset_class": "equity"})


def test_parse_research_intent_dispatch():
    assert kb_bootstrap.parse_research_intent("momentum in US stocks").topic.startswith("momentum")
    assert kb_bootstrap.parse_research_intent({"topic": "value"}).asset_class == "unknown"
    ready = kb_bootstrap.ResearchIntent(topic="carry")
    assert kb_bootstrap.parse_research_intent(ready) is ready
    with pytest.raises(KnowledgeValidationError):
        kb_bootstrap.parse_research_intent(42)


# -- retrieval planning -----------------------------------------------------


def test_plan_retrieval_covers_needed_memory_with_scope_filters():
    intent = kb_bootstrap.intent_from_mapping(
        {
            "topic": "momentum",
            "asset_class": "equity",
            "markets": ["US"],
            "requested_period": ["2000-01-01", "2020-01-01"],
            "needed_memory": ["validated_findings", "prior_experiments", "relevant_skills"],
        }
    )
    plan = kb_bootstrap.plan_retrieval(intent)
    assert [step.channel for step in plan] == ["validated_findings", "prior_experiments", "relevant_skills"]
    findings = plan[0].to_dict()
    assert findings["kinds"] == ["finding"]
    # No hard status filter: validated-first ordering comes from fusion
    # modifiers, and candidates ARE the Phase 2 consensus (a validated-only
    # filter would empty the section until Phase 3 validation exists).
    assert "status" not in findings["filters"]
    assert findings["filters"]["asset_class"] == "equity"
    assert findings["filters"]["period_start"] == "2000-01-01"
    assert plan[1].filters["_store"] == "experiments"
    assert plan[2].kinds == ("skill",)


# -- bootstrap packet -------------------------------------------------------


def make_stores(**overrides):
    stores = {
        "retrieval": FakeRetrieval(docs=[make_doc(), make_doc("C31", kind="conflict", status="open"), make_doc("A54", kind="assumption", status="active")]),
        "experiments": FakeExperimentLookup(records=[make_experiment()]),
        "skills": FakeSkills(),
    }
    stores.update(overrides)
    return kb_bootstrap.BootstrapStores(**stores)


def test_knowledge_bootstrap_builds_all_sections():
    packet = kb_bootstrap.knowledge_bootstrap("momentum in US equities", stores=make_stores())
    assert packet["intent"]["asset_class"] == "equity"
    names = [section["name"] for section in packet["sections"]]
    assert names[:6] == list(kb_bootstrap.DEFAULT_NEEDED_MEMORY)
    assert packet["channel_errors"] == []
    assert "RESEARCH CONTEXT" in packet["text"]
    assert "Current consensus" in packet["text"]
    assert "Reusable priors / skills" in packet["text"]
    skills = next(section for section in packet["sections"] if section["name"] == "relevant_skills")
    assert skills["entries"] and skills["entries"][0]["id"] == "cross-sectional-factor-backtest@2.3"


def test_knowledge_bootstrap_failure_channel_boosts_failures():
    failed = make_experiment(id="66666666-6666-6666-6666-666666666666", outcome="failure", status="failed", hypothesis="Momentum fails after costs.")
    stores = make_stores(experiments=FakeExperimentLookup(records=[make_experiment(), failed]))
    packet = kb_bootstrap.knowledge_bootstrap("momentum after costs", stores=stores)
    failures = next(section for section in packet["sections"] if section["name"] == "failures")
    assert any(entry["id"] == failed.id for entry in failures["entries"])


def test_hypothesis_needles_cover_all_concepts_longest_first():
    intent = kb_bootstrap.intent_from_mapping({"topic": "momentum turnover study", "concepts": ["turnover", "cross-sectional momentum"]})
    needles = kb_bootstrap._hypothesis_needles(intent)
    assert needles[0] == "cross-sectional momentum"
    assert "turnover" in needles
    assert "momentum turnover study" in needles  # topic phrase fallback


def test_hypothesis_needles_short_topic_falls_back_to_topic():
    intent = kb_bootstrap.intent_from_mapping({"topic": "ab"})
    assert kb_bootstrap._hypothesis_needles(intent) == ["ab"]


def test_union_needle_search_ranks_by_hit_count_then_first_seen():
    both = make_experiment(id="11111111-1111-1111-1111-111111111111", hypothesis="Momentum premia turnover study.")
    first_only = make_experiment(id="22222222-2222-2222-2222-222222222222", hypothesis="Momentum premia study.")
    second_only = make_experiment(id="33333333-3333-3333-3333-333333333333", hypothesis="Turnover drag study.")
    store = FakeExperimentLookup(records=[first_only, second_only, both])
    # A single longest-concept needle ("momentum premia") would miss
    # second_only entirely; the union finds all three.
    found = kb_bootstrap._union_needle_search(store, ["turnover", "momentum premia"], limit=10)
    assert [record.id for record in found] == [both.id, second_only.id, first_only.id]
    trimmed = kb_bootstrap._union_needle_search(store, ["study"], limit=2)
    assert [record.id for record in trimmed] == [first_only.id, second_only.id]


def test_knowledge_bootstrap_truncates_to_budget():
    docs = [make_doc(f"F{i:03d}", summary="s" * 200) for i in range(10)]
    stores = make_stores(retrieval=FakeRetrieval(docs=docs))
    packet = kb_bootstrap.knowledge_bootstrap(
        "momentum",
        stores=stores,
        budget=kb_bootstrap.PacketBudget(max_items=3, max_chars=500, channel_limit=5),
    )
    assert packet["truncated"] is True
    assert packet["channel_errors"] == []
    findings = next(section for section in packet["sections"] if section["name"] == "validated_findings")
    assert findings["truncated"] is True
    assert "(packet truncated to budget" in packet["text"]


def test_knowledge_bootstrap_captures_channel_errors():
    stores = make_stores(retrieval=FakeRetrieval(docs=[make_doc()], fail_search=True))
    packet = kb_bootstrap.knowledge_bootstrap("momentum", stores=stores)
    assert any(entry["channel"] == "validated_findings" for entry in packet["channel_errors"])
    assert any(section["name"] == "open_questions" for section in packet["sections"])
    assert "Channel gaps" in packet["text"]
    # The experiments-only channels still resolved despite the outage.
    priors = next(section for section in packet["sections"] if section["name"] == "prior_experiments")
    assert priors["entries"] and priors["entries"][0]["id"] == EXP_ID


def test_knowledge_bootstrap_requires_a_store():
    with pytest.raises(KnowledgeValidationError):
        kb_bootstrap.knowledge_bootstrap("momentum", stores=kb_bootstrap.BootstrapStores())
    with pytest.raises(KnowledgeValidationError):
        kb_bootstrap.PacketBudget(max_items=0, max_chars=50, channel_limit=0)


def test_knowledge_bootstrap_works_with_experiments_only():
    packet = kb_bootstrap.knowledge_bootstrap(
        {"topic": "momentum", "needed_memory": ["prior_experiments"]},
        stores=kb_bootstrap.BootstrapStores(experiments=FakeExperimentLookup(records=[make_experiment()])),
    )
    priors = next(section for section in packet["sections"] if section["name"] == "prior_experiments")
    assert priors["entries"] and priors["entries"][0]["id"] == EXP_ID


# -- lookup pure functions --------------------------------------------------


def test_ledger_search_happy_path_and_validation():
    backend = FakeRetrieval(docs=[make_doc(), make_doc("E1", kind="experiment")])
    page = kb_lookup.ledger_search("momentum", kinds="finding", filters={"status": "validated"}, backend=backend)
    assert page["count"] == 1
    assert page["documents"][0]["id"] == "F17"
    assert page["query"] == "momentum"
    with pytest.raises(KnowledgeValidationError):
        kb_lookup.ledger_search("  ", backend=backend)
    with pytest.raises(KnowledgeValidationError):
        kb_lookup.ledger_search("x", kinds=["skill"], backend=backend)
    with pytest.raises(KnowledgeValidationError):
        kb_lookup.ledger_search("x", filters={"bogus": "y"}, backend=backend)
    with pytest.raises(KnowledgeValidationError):
        kb_lookup.ledger_search("x", limit=500, backend=backend)
    with pytest.raises(KnowledgeValidationError):
        kb_lookup.ledger_search("x", filters={"project_id": "nope"}, backend=backend)


def test_ledger_get_reports_per_id_misses():
    backend = FakeRetrieval(docs=[make_doc()])
    payload = kb_lookup.ledger_get("F17, F404", include_evidence=True, backend=backend)
    assert [doc["id"] for doc in payload["documents"]] == ["F17"]
    assert payload["errors"] == [{"id": "F404", "error": "not_found"}]
    assert payload["include_evidence"] is True
    with pytest.raises(KnowledgeValidationError):
        kb_lookup.ledger_get(" , ", backend=backend)
    with pytest.raises(KnowledgeValidationError):
        kb_lookup.ledger_get([f"F{i}" for i in range(60)], backend=backend)


def test_experiment_get_by_id_and_execution_hash():
    store = FakeExperimentLookup(records=[make_experiment()])
    by_id = kb_lookup.experiment_get(EXP_ID, store=store)
    assert by_id["lookup"] == "id"
    assert by_id["experiment"]["execution_hash"] == EXEC_HASH
    by_hash = kb_lookup.experiment_get(EXEC_HASH, store=store)
    assert by_hash["lookup"] == "execution_hash"
    assert by_hash["experiment"]["id"] == EXP_ID
    with pytest.raises(KnowledgeValidationError):
        kb_lookup.experiment_get("not-a-ref", store=store)
    with pytest.raises(Exception, match="Experiment not found"):
        kb_lookup.experiment_get("9" * 64, store=store)


def test_parse_locator_forms_and_budgets():
    assert kb_lookup.parse_locator("bytes:0-1023") == {"kind": "bytes", "offset": 0, "length": 1024}
    assert kb_lookup.parse_locator("lines:10-59") == {"kind": "lines", "offset": 10, "length": 50}
    assert kb_lookup.parse_locator("head:20") == {"kind": "head", "offset": 0, "length": 20}
    assert kb_lookup.parse_locator("tail:20") == {"kind": "tail", "offset": 0, "length": 20}
    assert kb_lookup.parse_locator({"kind": "bytes", "offset": 5, "length": 5})["offset"] == 5
    with pytest.raises(KnowledgeValidationError):
        kb_lookup.parse_locator("bytes:10-5")
    with pytest.raises(KnowledgeValidationError):
        kb_lookup.parse_locator("pages:1-2")
    with pytest.raises(KnowledgeValidationError):
        kb_lookup.parse_locator({"kind": "bytes", "offset": 0, "length": kb_lookup.MAX_OPEN_BYTES + 1})
    with pytest.raises(KnowledgeValidationError):
        kb_lookup.parse_locator({"kind": "lines", "offset": 0, "length": kb_lookup.MAX_OPEN_LINES + 1})


def make_tmp_artifacts(tmp_path):
    store = ArtifactStore(LocalFilesystemBackend(tmp_path / "artifacts"))
    text_record = store.put_bytes(b"\n".join(f"line {i:04d}".encode() for i in range(100)), kind="log", media_type="text/plain")
    binary_record = store.put_bytes(bytes([0, 159, 146, 250, 0, 1]), kind="result", media_type="application/octet-stream")
    return kb_lookup.ArtifactStoreReader(store), text_record, binary_record


def test_artifact_manifest_and_bytes_window(tmp_path):
    reader, text_record, _ = make_tmp_artifacts(tmp_path)
    manifest = kb_lookup.artifact_manifest(text_record.uri, artifacts=reader)
    assert manifest["sha256"] == text_record.sha256
    assert manifest["open_limits"]["max_open_bytes"] == kb_lookup.MAX_OPEN_BYTES
    assert manifest["locator_schemes"]
    window = kb_lookup.artifact_open(text_record.uri, "bytes:0-9", artifacts=reader)
    assert window["encoding"] == "utf-8"
    assert window["content"] == "line 0000\n"
    assert window["bytes_returned"] == 10
    assert window["has_more"] is True


def test_artifact_open_line_windows_and_tail(tmp_path):
    reader, text_record, _ = make_tmp_artifacts(tmp_path)
    window = kb_lookup.artifact_open(text_record.uri, "lines:2-4", artifacts=reader)
    assert window["content"] == "line 0002\nline 0003\nline 0004"
    assert window["lines_returned"] == 3
    assert window["has_more"] is True
    head = kb_lookup.artifact_open(text_record.uri, "head:2", artifacts=reader)
    assert head["content"] == "line 0000\nline 0001"
    tail = kb_lookup.artifact_open(text_record.uri, "tail:2", artifacts=reader)
    assert tail["content"] == "line 0098\nline 0099"
    assert tail["has_more"] is False


def test_artifact_open_binary_falls_back_to_hex_and_missing_errors(tmp_path):
    reader, _, binary_record = make_tmp_artifacts(tmp_path)
    window = kb_lookup.artifact_open(binary_record.uri, "bytes:0-5", artifacts=reader)
    assert window["encoding"] == "hex"
    assert window["content"] == bytes([0, 159, 146, 250, 0, 1]).hex()
    with pytest.raises(KnowledgeValidationError, match="not valid UTF-8"):
        kb_lookup.artifact_open(binary_record.uri, "head:2", artifacts=reader)
    with pytest.raises(Exception, match="Artifact not found"):
        kb_lookup.artifact_manifest("artifact://sha256/" + "0" * 64, artifacts=reader)
    with pytest.raises(KnowledgeValidationError, match="invalid artifact ref"):
        kb_lookup.artifact_manifest("not-a-ref", artifacts=reader)


def test_artifact_open_scan_budget_forces_bytes_locators(tmp_path):
    store = ArtifactStore(LocalFilesystemBackend(tmp_path / "artifacts"))
    record = store.put_bytes(b"x" * (kb_lookup.MAX_SCAN_BYTES + 1024), kind="log", media_type="text/plain")
    reader = kb_lookup.ArtifactStoreReader(store)
    with pytest.raises(KnowledgeValidationError, match="MAX_SCAN_BYTES"):
        kb_lookup.artifact_open(record.uri, "tail:2", artifacts=reader)
    window = kb_lookup.artifact_open(record.uri, "bytes:0-9", artifacts=reader)
    assert window["bytes_returned"] == 10


# -- @tool wrappers ---------------------------------------------------------


def bind_all(tmp_path):
    reader, text_record, _ = make_tmp_artifacts(tmp_path)
    kb_lookup.bind_knowledge_backends(
        retrieval=FakeRetrieval(docs=[make_doc()]),
        experiments=FakeExperimentLookup(records=[make_experiment()]),
        artifacts=reader,
    )
    kb_bootstrap.bind_knowledge_skills(FakeSkills())
    return text_record


def test_tool_wrappers_report_unbound_backends():
    # Mirrors tests/test_memory_tools.py: exercise the wrapper via .func with
    # an explicit runtime (bare .invoke demands the injected runtime schema).
    runtime = make_runtime()
    assert "not bound" in json.loads(kb_lookup.ledger_search_tool.func(runtime, "x"))["error"]
    assert "not bound" in json.loads(kb_lookup.ledger_get_tool.func(runtime, "F17"))["error"]
    assert "not bound" in json.loads(kb_lookup.experiment_get_tool.func(runtime, EXP_ID))["error"]
    assert "not bound" in json.loads(kb_lookup.artifact_manifest_tool.func(runtime, "x"))["error"]
    assert "not bound" in json.loads(kb_lookup.artifact_open_tool.func(runtime, "x", "head:1"))["error"]
    assert "not bound" in json.loads(kb_bootstrap.knowledge_bootstrap_tool.func(runtime, "x"))["error"]


def test_tool_wrappers_happy_path(tmp_path):
    text_record = bind_all(tmp_path)
    runtime = make_runtime(thread_id="t1", agent_name="lead", project_id=PROJECT_ID)
    search = json.loads(kb_lookup.ledger_search_tool.func(runtime, "momentum", kinds="finding"))
    assert search["count"] == 1
    # The run's project scope merges under explicit filters.
    assert search["filters"] == {"project_id": PROJECT_ID}
    scoped = json.loads(kb_lookup.ledger_search_tool.func(runtime, "momentum", filters_json=json.dumps({"status": "validated"})))
    assert scoped["filters"] == {"status": "validated", "project_id": PROJECT_ID}
    bad_filters = json.loads(kb_lookup.ledger_search_tool.func(runtime, "x", filters_json="{oops"))
    assert "error" in bad_filters
    got = json.loads(kb_lookup.ledger_get_tool.func(runtime, "F17,F404", include_evidence=True))
    assert [doc["id"] for doc in got["documents"]] == ["F17"]
    assert got["errors"] == [{"id": "F404", "error": "not_found"}]
    exp = json.loads(kb_lookup.experiment_get_tool.func(runtime, EXEC_HASH))
    assert exp["experiment"]["id"] == EXP_ID
    missing_exp = json.loads(kb_lookup.experiment_get_tool.func(runtime, "9" * 64))
    assert "not found" in missing_exp["error"]
    manifest = json.loads(kb_lookup.artifact_manifest_tool.func(runtime, text_record.uri))
    assert manifest["sha256"] == text_record.sha256
    opened = json.loads(kb_lookup.artifact_open_tool.func(runtime, text_record.uri, "head:1"))
    assert opened["content"] == "line 0000"
    packet = json.loads(kb_bootstrap.knowledge_bootstrap_tool.func(runtime, "momentum in US equities"))
    assert "RESEARCH CONTEXT" in packet["text"]
    packet_json = json.loads(kb_bootstrap.knowledge_bootstrap_tool.func(runtime, json.dumps({"topic": "value", "needed_memory": ["failures"]})))
    assert packet_json["intent"]["topic"] == "value"


def test_get_knowledge_tools_registry():
    assert [tool.name for tool in kb_lookup.get_knowledge_tools()] == [
        "ledger_search",
        "ledger_get",
        "experiment_get",
        "artifact_manifest",
        "artifact_open",
    ]
    assert [tool.name for tool in kb_bootstrap.get_knowledge_bootstrap_tool()] == ["knowledge_bootstrap"]


# -- registration gate ------------------------------------------------------


@pytest.fixture()
def _knowledge_config_enabled(monkeypatch):
    """Enable the plane for the test, restoring the previous singleton after."""
    from deerflow.knowledge.config import KnowledgeConfig, get_knowledge_config, set_knowledge_config

    monkeypatch.delenv("DEER_FLOW_KNOWLEDGE_ENABLED", raising=False)
    previous = get_knowledge_config()
    set_knowledge_config(KnowledgeConfig(enabled=True))
    try:
        yield
    finally:
        set_knowledge_config(previous)


@pytest.fixture()
def _knowledge_config_disabled(monkeypatch):
    from deerflow.knowledge.config import KnowledgeConfig, get_knowledge_config, set_knowledge_config

    monkeypatch.delenv("DEER_FLOW_KNOWLEDGE_ENABLED", raising=False)
    previous = get_knowledge_config()
    set_knowledge_config(KnowledgeConfig(enabled=False))
    try:
        yield
    finally:
        set_knowledge_config(previous)


def test_knowledge_tools_available_needs_enabled_and_bound_backends(_knowledge_config_enabled):
    assert kb_lookup.knowledge_tools_available() is False
    kb_lookup.bind_knowledge_backends(retrieval=FakeRetrieval(docs=[make_doc()]))
    assert kb_lookup.knowledge_tools_available() is True


def test_knowledge_tools_available_experiments_only_counts(_knowledge_config_enabled):
    kb_lookup.bind_knowledge_backends(experiments=FakeExperimentLookup(records=[make_experiment()]))
    assert kb_lookup.knowledge_tools_available() is True


def test_knowledge_tools_available_artifacts_only_does_not_count(tmp_path, _knowledge_config_enabled):
    reader, _, _ = make_tmp_artifacts(tmp_path)
    kb_lookup.bind_knowledge_backends(artifacts=reader)
    assert kb_lookup.knowledge_tools_available() is False


def test_knowledge_tools_available_disabled_plane_is_unavailable(_knowledge_config_disabled):
    kb_lookup.bind_knowledge_backends(retrieval=FakeRetrieval(docs=[make_doc()]))
    assert kb_lookup.knowledge_tools_available() is False


def test_factory_registers_knowledge_tools_when_available(tmp_path, _knowledge_config_enabled):
    from deerflow.agents.factory import _assemble_from_features
    from deerflow.agents.features import RuntimeFeatures
    from deerflow.knowledge.agent_tools.middleware import KnowledgeBootstrapMiddleware

    bind_all(tmp_path)
    chain, extra_tools = _assemble_from_features(RuntimeFeatures(knowledge=True), name="test-agent")
    assert any(isinstance(item, KnowledgeBootstrapMiddleware) for item in chain)
    assert [tool.name for tool in extra_tools if tool.name in {"ledger_search", "ledger_get", "experiment_get", "artifact_manifest", "artifact_open"}] == [
        "ledger_search",
        "ledger_get",
        "experiment_get",
        "artifact_manifest",
        "artifact_open",
    ]


def test_factory_skips_knowledge_tools_when_unbound(_knowledge_config_enabled):
    from deerflow.agents.factory import _assemble_from_features
    from deerflow.agents.features import RuntimeFeatures
    from deerflow.knowledge.agent_tools.middleware import KnowledgeBootstrapMiddleware

    chain, extra_tools = _assemble_from_features(RuntimeFeatures(knowledge=True), name="test-agent")
    assert any(isinstance(item, KnowledgeBootstrapMiddleware) for item in chain)
    assert not {"ledger_search", "ledger_get", "experiment_get", "artifact_manifest", "artifact_open"} & {tool.name for tool in extra_tools}


def test_factory_skips_knowledge_tools_when_feature_off(tmp_path, _knowledge_config_enabled):
    from deerflow.agents.factory import _assemble_from_features
    from deerflow.agents.features import RuntimeFeatures
    from deerflow.knowledge.agent_tools.middleware import KnowledgeBootstrapMiddleware

    bind_all(tmp_path)
    chain, extra_tools = _assemble_from_features(RuntimeFeatures(knowledge=False), name="test-agent")
    assert not any(isinstance(item, KnowledgeBootstrapMiddleware) for item in chain)
    assert not {"ledger_search", "ledger_get", "experiment_get", "artifact_manifest", "artifact_open"} & {tool.name for tool in extra_tools}


# -- middleware -------------------------------------------------------------


def test_middleware_injects_packet_once_per_run():
    middleware = kb_middleware.KnowledgeBootstrapMiddleware(stores=make_stores(), enabled=True)
    runtime = make_runtime(thread_id="t1", agent_name="lead")
    state = {"messages": [HumanMessage(content="Study momentum in US equities.")]}
    update = middleware.before_agent(state, runtime)
    assert update is not None
    injected = update["messages"][0]
    assert isinstance(injected, SystemMessage)
    assert injected.id == kb_middleware.KNOWLEDGE_BOOTSTRAP_MESSAGE_ID
    assert injected.additional_kwargs[kb_middleware.KNOWLEDGE_BOOTSTRAP_MARKER] is True
    assert "RESEARCH CONTEXT" in injected.content
    rerun = middleware.before_agent({"messages": [*state["messages"], injected]}, runtime)
    assert rerun is None


def test_middleware_degrades_cleanly():
    disabled = kb_middleware.KnowledgeBootstrapMiddleware(stores=make_stores(), enabled=False)
    state = {"messages": [HumanMessage(content="momentum")]}
    assert disabled.before_agent(state, make_runtime(thread_id="t1")) is None
    unbound = kb_middleware.KnowledgeBootstrapMiddleware(enabled=True)
    assert unbound.before_agent(state, make_runtime(thread_id="t1")) is None
    assert kb_middleware.KnowledgeBootstrapMiddleware(stores=make_stores(), enabled=True).before_agent({"messages": []}, make_runtime()) is None
    system_only = {"messages": [SystemMessage(content="hello")]}
    assert kb_middleware.KnowledgeBootstrapMiddleware(stores=make_stores(), enabled=True).before_agent(system_only, make_runtime()) is None
    failing = kb_middleware.KnowledgeBootstrapMiddleware(stores=make_stores(retrieval=FakeRetrieval(fail_search=True), experiments=None), enabled=True)
    assert failing.before_agent(state, make_runtime(thread_id="t1")) is not None  # partial packet, channel error captured
    exploding = kb_middleware.KnowledgeBootstrapMiddleware(stores=make_stores(), enabled=True, intent_builder=lambda *a, **k: 1 / 0)
    assert exploding.before_agent(state, make_runtime(thread_id="t1")) is not None  # falls back to raw task text


def test_middleware_async_path_and_policy():
    middleware = kb_middleware.KnowledgeBootstrapMiddleware(stores=make_stores(), enabled=True)
    state = {"messages": [HumanMessage(content="momentum in US equities")]}
    update = asyncio.run(middleware.abefore_agent(state, make_runtime(thread_id="t1")))
    assert update is not None and "RESEARCH CONTEXT" in update["messages"][0].content
    policy = middleware.release_policy_parameters()
    assert policy["max_items"] == kb_bootstrap.DEFAULT_PACKET_ITEMS
    assert policy["has_explicit_stores"] is True
    assert set(policy) == {"enabled", "agent_name", "max_items", "max_chars", "channel_limit", "has_explicit_stores", "has_intent_builder"}


# -- Phase 3b: per-message knowledge scope -------------------------------------
#
# The fork tools read the admitted execution scope from the tool runtime
# context (KNOWLEDGE_SCOPE_RUNTIME_KEY), canonicalize it, and pass a
# KnowledgeRequestContext to the retrieval backends — the upstream
# knowledge/tools.py pattern. Enforcement lives in the backends: the
# test doubles below mirror the provider dataset→kind mapping so the
# tests prove the threading, while production fork code never maps.


#: Test-double mirror of the provider dataset→kind mapping (the fork
#: production path never maps; backends own enforcement). ``conflict``/
#: ``dossier``/``artifact`` kinds belong to no dataset and drop out of
#: every selected scope, exactly like the provider's kind filter.
_FAKE_DATASET_KINDS = {
    "findings": {"finding"},
    "experiments": {"experiment"},
    "failures": {"failure"},
    "assumptions": {"assumption"},
}

DISABLED_SCOPE = {"version": 1, "mode": "disabled"}
ALL_SCOPE = {"version": 1, "mode": "all"}
FINDINGS_SCOPE = {"version": 1, "mode": "selected", "dataset_ids": ["findings", "assumptions"]}
EXPERIMENTS_SCOPE = {"version": 1, "mode": "selected", "dataset_ids": ["experiments"]}
UNKNOWN_SCOPE = {"version": 1, "mode": "selected", "dataset_ids": ["nope"]}

# None means "absent scope" and is covered by the legacy-path tests instead.
INVALID_SCOPES = [
    {"mode": "selected"},
    {"version": 1, "mode": "selected", "dataset_ids": []},
    {"version": 1, "mode": "selected", "dataset_ids": ["findings"], "bogus": 1},
    {"version": 2, "mode": "all"},
    {"version": 1, "mode": "all", "dataset_ids": ["findings"]},
    "just-a-string",
    42,
]


class ScopedFakeRetrieval(FakeRetrieval):
    """FakeRetrieval with the Phase 3b scoped entry points."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scoped_search_calls: list[dict] = []
        self.scoped_get_calls: list[dict] = []
        self.get_calls: list[dict] = []

    def get_documents(self, ids, *, include_evidence):
        self.get_calls.append({"ids": list(ids), "include_evidence": include_evidence})
        return super().get_documents(ids, include_evidence=include_evidence)

    def _scope_filter(self, context):
        assert isinstance(context, KnowledgeRequestContext), f"scoped entry points take a KnowledgeRequestContext, got {type(context).__name__}"
        scope = context.scope or {"mode": "all"}
        if scope.get("mode") != "selected":
            return None, None
        dataset_ids = list(scope.get("dataset_ids") or [])
        unknown = [dataset_id for dataset_id in dataset_ids if dataset_id not in _FAKE_DATASET_KINDS]
        if unknown:
            raise PermissionError(f"unknown dataset(s): {', '.join(sorted(unknown))}")
        kinds = set().union(*[_FAKE_DATASET_KINDS[dataset_id] for dataset_id in dataset_ids])
        doc_filters = scope.get("document_filters") or []
        allowed_ids = None
        if doc_filters:
            allowed_ids = {doc_id for entry in doc_filters for doc_id in entry.get("document_ids", [])}
        return kinds, allowed_ids

    def search_scoped(self, query, *, kinds, filters, limit, offset, context):
        self.scoped_search_calls.append({"query": query, "kinds": kinds, "filters": filters, "limit": limit, "offset": offset, "context": context})
        kinds_allowed, ids_allowed = self._scope_filter(context)
        hits = [doc for doc in self.docs.values() if doc.kind in kinds]
        if kinds_allowed is not None:
            hits = [doc for doc in hits if doc.kind in kinds_allowed]
        if ids_allowed is not None:
            hits = [doc for doc in hits if doc.id in ids_allowed]
        page = hits[offset : offset + limit]
        return kb_lookup.RetrievalPage(documents=list(page), limit=limit, offset=offset, total=len(hits))

    def get_documents_scoped(self, ids, *, include_evidence, context):
        self.scoped_get_calls.append({"ids": list(ids), "include_evidence": include_evidence, "context": context})
        kinds_allowed, ids_allowed = self._scope_filter(context)
        rows = []
        for doc_id in ids:
            doc = self.docs.get(doc_id)
            if doc is None:
                rows.append(None)
            elif kinds_allowed is not None and doc.kind not in kinds_allowed:
                rows.append(None)  # out-of-scope reads as not_found (existence-hiding)
            elif ids_allowed is not None and doc.id not in ids_allowed:
                rows.append(None)
            else:
                rows.append(doc)
        return rows


class ScopedFakeExperiments(FakeExperimentLookup):
    """FakeExperimentLookup with the Phase 3b scoped finds."""

    def _in_scope(self, record, context):
        assert isinstance(context, KnowledgeRequestContext), f"scoped entry points take a KnowledgeRequestContext, got {type(context).__name__}"
        if record is None:
            return False
        scope = context.scope or {"mode": "all"}
        if scope.get("mode") != "selected":
            return True
        dataset_ids = list(scope.get("dataset_ids") or [])
        unknown = [dataset_id for dataset_id in dataset_ids if dataset_id not in _FAKE_DATASET_KINDS]
        if unknown:
            raise PermissionError(f"unknown dataset(s): {', '.join(sorted(unknown))}")
        want = "failures" if record.outcome == "failure" else "experiments"
        if want not in dataset_ids:
            return False
        doc_filters = scope.get("document_filters") or []
        if doc_filters:
            allowed = {doc_id for entry in doc_filters for doc_id in entry.get("document_ids", [])}
            return record.id in allowed
        return True

    def find_by_id_scoped(self, experiment_id, *, context):
        record = self.find_by_id(experiment_id)
        return record if self._in_scope(record, context) else None

    def find_by_execution_hash_scoped(self, execution_hash, *, context):
        record = self.find_by_execution_hash(execution_hash)
        return record if self._in_scope(record, context) else None


def make_scoped_runtime(scope, **context):
    return make_runtime(**{KNOWLEDGE_SCOPE_RUNTIME_KEY: scope}, **context)


def make_scoped_docs():
    return [
        make_doc("F17", kind="finding"),
        make_doc("E1", kind="experiment"),
        make_doc("FL1", kind="failure", status="failed"),
        make_doc("A54", kind="assumption", status="active"),
        make_doc("C31", kind="conflict", status="open"),
    ]


def _all_tool_calls():
    """One no-bindings call per fork @tool wrapper (Phase 3b refusal order)."""
    return [
        lambda rt: kb_lookup.ledger_search_tool.func(rt, "momentum"),
        lambda rt: kb_lookup.ledger_get_tool.func(rt, "F17"),
        lambda rt: kb_lookup.experiment_get_tool.func(rt, EXP_ID),
        lambda rt: kb_lookup.artifact_manifest_tool.func(rt, "x"),
        lambda rt: kb_lookup.artifact_open_tool.func(rt, "x", "head:1"),
        lambda rt: kb_bootstrap.knowledge_bootstrap_tool.func(rt, "momentum"),
    ]


def test_disabled_scope_refuses_every_fork_tool_before_backends():
    # Disabled refuses even with nothing bound: the scope gate precedes the
    # binding checks, mirroring upstream tools.py (scope before provider).
    runtime = make_scoped_runtime(DISABLED_SCOPE, agent_name="lead")
    for call in _all_tool_calls():
        assert json.loads(call(runtime)) == {"error": kb_lookup.SCOPE_DISABLED_MESSAGE}


def test_disabled_scope_touches_no_backend(tmp_path):
    retrieval = ScopedFakeRetrieval(docs=make_scoped_docs())
    experiments = ScopedFakeExperiments(records=[make_experiment()])
    reader, text_record, _ = make_tmp_artifacts(tmp_path)
    kb_lookup.bind_knowledge_backends(retrieval=retrieval, experiments=experiments, artifacts=reader)
    kb_bootstrap.bind_knowledge_skills(FakeSkills())
    runtime = make_scoped_runtime(DISABLED_SCOPE, agent_name="lead")
    assert json.loads(kb_lookup.ledger_search_tool.func(runtime, "x")) == {"error": kb_lookup.SCOPE_DISABLED_MESSAGE}
    assert json.loads(kb_lookup.ledger_get_tool.func(runtime, "F17")) == {"error": kb_lookup.SCOPE_DISABLED_MESSAGE}
    assert json.loads(kb_lookup.experiment_get_tool.func(runtime, EXP_ID)) == {"error": kb_lookup.SCOPE_DISABLED_MESSAGE}
    assert json.loads(kb_lookup.artifact_manifest_tool.func(runtime, text_record.uri)) == {"error": kb_lookup.SCOPE_DISABLED_MESSAGE}
    assert json.loads(kb_lookup.artifact_open_tool.func(runtime, text_record.uri, "head:1")) == {"error": kb_lookup.SCOPE_DISABLED_MESSAGE}
    assert json.loads(kb_bootstrap.knowledge_bootstrap_tool.func(runtime, "x")) == {"error": kb_lookup.SCOPE_DISABLED_MESSAGE}
    assert retrieval.search_calls == [] and retrieval.scoped_search_calls == []
    assert retrieval.get_calls == [] and retrieval.scoped_get_calls == []
    assert experiments.used_family == 0 and experiments.used_search == 0
    # Direct service callers fail the same way (defense in depth).
    disabled_ctx = KnowledgeRequestContext(user_id="u", agent_name="lead", scope=dict(DISABLED_SCOPE))
    with pytest.raises(KnowledgeError, match="disabled for this turn"):
        kb_lookup.ledger_search("x", backend=retrieval, request_context=disabled_ctx)
    with pytest.raises(KnowledgeError, match="disabled for this turn"):
        kb_bootstrap.knowledge_bootstrap("x", stores=kb_bootstrap.BootstrapStores(retrieval=retrieval), request_context=disabled_ctx)


def test_selected_scope_constrains_search_to_datasets():
    retrieval = ScopedFakeRetrieval(docs=make_scoped_docs())
    kb_lookup.bind_knowledge_backends(retrieval=retrieval)
    runtime = make_scoped_runtime(FINDINGS_SCOPE, agent_name="lead", user_id="scope-user")
    payload = json.loads(kb_lookup.ledger_search_tool.func(runtime, "momentum"))
    assert {doc["id"] for doc in payload["documents"]} == {"F17", "A54"}
    assert payload["count"] == 2 and payload["total"] == 2
    # Frozen agent API: same keys as the legacy path.
    assert set(payload) == {"documents", "count", "limit", "offset", "total", "query", "kinds", "filters"}
    # The scope narrowed server-side: caller asked for all kinds, the backend
    # got the full kind list plus a request context carrying identity + scope.
    assert retrieval.search_calls == []
    (scoped_call,) = retrieval.scoped_search_calls
    assert set(scoped_call["kinds"]) == set(kb_lookup.RETRIEVAL_KINDS)
    context = scoped_call["context"]
    assert isinstance(context, KnowledgeRequestContext)
    assert context.user_id == resolve_runtime_user_id(runtime) == "scope-user"
    assert context.agent_name == "lead"
    assert context.scope == {"version": 1, "mode": "selected", "dataset_ids": ["findings", "assumptions"]}
    # Direct service call threads the same way.
    direct = kb_lookup.ledger_search("momentum", backend=retrieval, request_context=KnowledgeRequestContext(user_id="u", scope=dict(FINDINGS_SCOPE)))
    assert {doc["id"] for doc in direct["documents"]} == {"F17", "A54"}


def test_selected_scope_intersects_caller_kinds():
    retrieval = ScopedFakeRetrieval(docs=make_scoped_docs())
    kb_lookup.bind_knowledge_backends(retrieval=retrieval)
    runtime = make_scoped_runtime({"version": 1, "mode": "selected", "dataset_ids": ["findings"]})
    payload = json.loads(kb_lookup.ledger_search_tool.func(runtime, "momentum", kinds="finding,experiment"))
    assert [doc["id"] for doc in payload["documents"]] == ["F17"]
    assert payload["kinds"] == ["finding", "experiment"]  # echo unchanged; narrowing is backend-side


def test_selected_scope_unknown_dataset_fails_closed():
    retrieval = ScopedFakeRetrieval(docs=make_scoped_docs())
    kb_lookup.bind_knowledge_backends(retrieval=retrieval)
    runtime = make_scoped_runtime(UNKNOWN_SCOPE)
    assert json.loads(kb_lookup.ledger_search_tool.func(runtime, "x")) == {"error": kb_lookup.SCOPE_INACCESSIBLE_MESSAGE}
    assert json.loads(kb_lookup.ledger_get_tool.func(runtime, "F17")) == {"error": kb_lookup.SCOPE_INACCESSIBLE_MESSAGE}
    with pytest.raises(PermissionError):
        kb_lookup.ledger_search("x", backend=retrieval, request_context=KnowledgeRequestContext(user_id="u", scope=dict(UNKNOWN_SCOPE)))


def test_selected_scope_legacy_backend_cannot_apply():
    # A selected scope on a backend without the scoped entry points refuses —
    # never an unconstrained retry (upstream search_provider parity).
    retrieval = FakeRetrieval(docs=make_scoped_docs())
    kb_lookup.bind_knowledge_backends(retrieval=retrieval, experiments=FakeExperimentLookup(records=[make_experiment()]))
    runtime = make_scoped_runtime(FINDINGS_SCOPE)
    assert json.loads(kb_lookup.ledger_search_tool.func(runtime, "x")) == {"error": kb_lookup.SCOPE_NOT_APPLICABLE_MESSAGE}
    assert json.loads(kb_lookup.ledger_get_tool.func(runtime, "F17")) == {"error": kb_lookup.SCOPE_NOT_APPLICABLE_MESSAGE}
    assert json.loads(kb_lookup.experiment_get_tool.func(runtime, EXP_ID)) == {"error": kb_lookup.SCOPE_NOT_APPLICABLE_MESSAGE}
    assert retrieval.search_calls == []
    with pytest.raises(KnowledgeError, match="cannot apply the selected scope"):
        kb_lookup.ledger_search("x", backend=retrieval, request_context=KnowledgeRequestContext(user_id="u", scope=dict(FINDINGS_SCOPE)))


def test_selected_scope_document_filters_allowlist_get():
    retrieval = ScopedFakeRetrieval(docs=make_scoped_docs())
    kb_lookup.bind_knowledge_backends(retrieval=retrieval)
    scope = {
        "version": 1,
        "mode": "selected",
        "dataset_ids": ["findings", "experiments"],
        "document_filters": [{"dataset_id": "findings", "document_ids": ["F17"]}],
    }
    payload = json.loads(kb_lookup.ledger_get_tool.func(make_scoped_runtime(scope), "F17,E1", include_evidence=True))
    assert [doc["id"] for doc in payload["documents"]] == ["F17"]
    assert payload["errors"] == [{"id": "E1", "error": "not_found"}]
    assert payload["include_evidence"] is True
    assert retrieval.get_calls == []  # scoped path only
    (scoped_call,) = retrieval.scoped_get_calls
    assert scoped_call["ids"] == ["F17", "E1"]


def test_selected_scope_strips_display_block():
    retrieval = ScopedFakeRetrieval(docs=make_scoped_docs())
    kb_lookup.bind_knowledge_backends(retrieval=retrieval)
    scope = {
        "version": 1,
        "mode": "selected",
        "dataset_ids": ["findings"],
        "display": {"datasets": [{"id": "findings", "name": "Findings"}]},
    }
    payload = json.loads(kb_lookup.ledger_search_tool.func(make_scoped_runtime(scope), "momentum"))
    assert {doc["id"] for doc in payload["documents"]} == {"F17"}
    (scoped_call,) = retrieval.scoped_search_calls
    assert scoped_call["context"].scope == {"version": 1, "mode": "selected", "dataset_ids": ["findings"]}


def test_experiment_get_selected_scope():
    experiments = ScopedFakeExperiments(records=[make_experiment()])
    kb_lookup.bind_knowledge_backends(experiments=experiments)
    ok = json.loads(kb_lookup.experiment_get_tool.func(make_scoped_runtime(EXPERIMENTS_SCOPE), EXP_ID))
    assert ok["experiment"]["id"] == EXP_ID and ok["lookup"] == "id"
    by_hash = json.loads(kb_lookup.experiment_get_tool.func(make_scoped_runtime(EXPERIMENTS_SCOPE), EXEC_HASH))
    assert by_hash["lookup"] == "execution_hash"
    # Out-of-scope refs read as not-found (existence-hiding), not as scope errors.
    missed = json.loads(kb_lookup.experiment_get_tool.func(make_scoped_runtime(FINDINGS_SCOPE), EXP_ID))
    assert missed == {"error": f"Experiment not found: {EXP_ID}"}
    with pytest.raises(KnowledgeError, match="Experiment not found"):
        kb_lookup.experiment_get(EXP_ID, store=experiments, request_context=KnowledgeRequestContext(user_id="u", scope=dict(FINDINGS_SCOPE)))


def test_absent_scope_uses_legacy_path_unchanged():
    # Absent scope never touches the scoped entry points, even on backends
    # that implement them: byte-identical legacy behavior for direct and
    # non-agent callers.
    retrieval = ScopedFakeRetrieval(docs=make_scoped_docs())
    experiments = ScopedFakeExperiments(records=[make_experiment()])
    kb_lookup.bind_knowledge_backends(retrieval=retrieval, experiments=experiments)
    runtime = make_runtime(agent_name="lead")
    payload = json.loads(kb_lookup.ledger_search_tool.func(runtime, "momentum"))
    assert payload["count"] == 5
    got = json.loads(kb_lookup.ledger_get_tool.func(runtime, "F17"))
    assert [doc["id"] for doc in got["documents"]] == ["F17"]
    exp = json.loads(kb_lookup.experiment_get_tool.func(runtime, EXP_ID))
    assert exp["experiment"]["id"] == EXP_ID
    assert len(retrieval.search_calls) == 1 and retrieval.scoped_search_calls == []
    assert len(retrieval.get_calls) == 1 and retrieval.scoped_get_calls == []
    # Direct service calls without a context stay legacy too.
    kb_lookup.ledger_search("x", backend=retrieval)
    kb_lookup.ledger_get("F17", backend=retrieval)
    kb_lookup.experiment_get(EXP_ID, store=experiments)
    assert len(retrieval.search_calls) == 2 and retrieval.scoped_search_calls == []
    assert len(retrieval.get_calls) == 2 and retrieval.scoped_get_calls == []


def test_all_scope_supplies_context_when_supported():
    scoped = ScopedFakeRetrieval(docs=make_scoped_docs())
    kb_lookup.bind_knowledge_backends(retrieval=scoped)
    payload = json.loads(kb_lookup.ledger_search_tool.func(make_scoped_runtime(ALL_SCOPE), "momentum"))
    assert payload["count"] == 5
    assert scoped.search_calls == []
    (call,) = scoped.scoped_search_calls
    assert call["context"].scope == {"version": 1, "mode": "all"}
    # ... and stays on the legacy path for backends without scoped methods.
    legacy = FakeRetrieval(docs=make_scoped_docs())
    kb_lookup.bind_knowledge_backends(retrieval=legacy)
    payload = json.loads(kb_lookup.ledger_search_tool.func(make_scoped_runtime(ALL_SCOPE), "momentum"))
    assert payload["count"] == 5
    assert len(legacy.search_calls) == 1


def test_invalid_scope_fails_closed_without_touching_backends():
    retrieval = ScopedFakeRetrieval(docs=make_scoped_docs())
    kb_lookup.bind_knowledge_backends(retrieval=retrieval, experiments=ScopedFakeExperiments(records=[make_experiment()]))
    for scope in INVALID_SCOPES:
        runtime = make_scoped_runtime(scope)
        for call in _all_tool_calls():
            assert json.loads(call(runtime)) == {"error": kb_lookup.SCOPE_INACCESSIBLE_MESSAGE}, scope
    assert retrieval.search_calls == [] and retrieval.scoped_search_calls == []
    assert retrieval.get_calls == [] and retrieval.scoped_get_calls == []
    # Direct service callers fail closed as well.
    with pytest.raises(KnowledgeValidationError, match="request_context must be"):
        kb_lookup.ledger_search("x", backend=retrieval, request_context="nope")
    with pytest.raises(KnowledgeValidationError, match="invalid knowledge scope"):
        kb_lookup.ledger_search("x", backend=retrieval, request_context=KnowledgeRequestContext(user_id="u", scope={"bogus": 1}))


def test_artifact_tools_gate_disabled_but_pass_selected_through(tmp_path):
    reader, text_record, _ = make_tmp_artifacts(tmp_path)
    kb_lookup.bind_knowledge_backends(artifacts=reader)
    disabled = make_scoped_runtime(DISABLED_SCOPE)
    assert json.loads(kb_lookup.artifact_manifest_tool.func(disabled, text_record.uri)) == {"error": kb_lookup.SCOPE_DISABLED_MESSAGE}
    assert json.loads(kb_lookup.artifact_open_tool.func(disabled, text_record.uri, "head:1")) == {"error": kb_lookup.SCOPE_DISABLED_MESSAGE}
    # Selected scopes pass through: artifact refs are unguessable capability
    # locators minted by scoped reads; no dataset mapping exists to enforce.
    selected = make_scoped_runtime(FINDINGS_SCOPE)
    assert json.loads(kb_lookup.artifact_manifest_tool.func(selected, text_record.uri))["sha256"] == text_record.sha256
    assert json.loads(kb_lookup.artifact_open_tool.func(selected, text_record.uri, "head:1"))["content"] == "line 0000"


def test_bootstrap_disabled_scope_refuses_packet():
    kb_lookup.bind_knowledge_backends(retrieval=ScopedFakeRetrieval(docs=make_scoped_docs()), experiments=ScopedFakeExperiments(records=[make_experiment()]))
    kb_bootstrap.bind_knowledge_skills(FakeSkills())
    assert json.loads(kb_bootstrap.knowledge_bootstrap_tool.func(make_scoped_runtime(DISABLED_SCOPE), "momentum")) == {"error": kb_lookup.SCOPE_DISABLED_MESSAGE}
    assert json.loads(kb_bootstrap.knowledge_bootstrap_tool.func(make_scoped_runtime({"mode": "bogus"}), "momentum")) == {"error": kb_lookup.SCOPE_INACCESSIBLE_MESSAGE}


def test_bootstrap_selected_scope_constrains_documents_and_gaps_experiments():
    retrieval = ScopedFakeRetrieval(docs=make_scoped_docs())
    experiments = ScopedFakeExperiments(records=[make_experiment()])
    stores = kb_bootstrap.BootstrapStores(retrieval=retrieval, experiments=experiments, skills=FakeSkills())
    packet = kb_bootstrap.knowledge_bootstrap(
        {"topic": "momentum", "needed_memory": ["validated_findings", "prior_experiments", "failures", "conflicts", "assumptions", "relevant_skills"]},
        stores=stores,
        request_context=KnowledgeRequestContext(user_id="u", scope=dict(FINDINGS_SCOPE)),
    )
    sections = {section["name"]: section for section in packet["sections"]}
    # Document channels went through search_scoped: only in-scope kinds survived.
    assert {entry["id"] for entry in sections["validated_findings"]["entries"]} == {"F17"}
    assert sections["conflicts"]["entries"] == []  # conflict kind belongs to no dataset
    assert {entry["id"] for entry in sections["assumptions"]["entries"]} == {"A54"}
    assert retrieval.scoped_search_calls and retrieval.search_calls == []
    # The experiment store has no scoped entry point yet, so those channels
    # gap loudly instead of running an unrestricted retry.
    assert {entry["channel"] for entry in packet["channel_errors"]} == {"prior_experiments", "failures"}
    assert all(entry["error"] == kb_lookup.SCOPE_NOT_APPLICABLE_MESSAGE for entry in packet["channel_errors"])
    assert experiments.used_family == 0 and experiments.used_search == 0
    assert any(section["name"] == "open_questions" for section in packet["sections"])
    # Skills are scope-independent tool metadata, not KB evidence.
    assert sections["relevant_skills"]["entries"]


def test_bootstrap_selected_scope_legacy_retrieval_gaps_document_channels():
    stores = make_stores()
    packet = kb_bootstrap.knowledge_bootstrap(
        "momentum",
        stores=stores,
        request_context=KnowledgeRequestContext(user_id="u", scope=dict(FINDINGS_SCOPE)),
    )
    assert {entry["channel"] for entry in packet["channel_errors"]} >= {"validated_findings", "prior_experiments"}
    assert all(entry["error"] == kb_lookup.SCOPE_NOT_APPLICABLE_MESSAGE for entry in packet["channel_errors"])


def test_bootstrap_scoped_tool_wrapper_and_all_scope():
    retrieval = ScopedFakeRetrieval(docs=make_scoped_docs())
    experiments = ScopedFakeExperiments(records=[make_experiment()])
    kb_lookup.bind_knowledge_backends(retrieval=retrieval, experiments=experiments)
    kb_bootstrap.bind_knowledge_skills(FakeSkills())
    selected = json.loads(kb_bootstrap.knowledge_bootstrap_tool.func(make_scoped_runtime(FINDINGS_SCOPE, agent_name="lead"), "momentum"))
    assert {entry["channel"] for entry in selected["channel_errors"]} == {"prior_experiments", "failures"}
    assert retrieval.scoped_search_calls and retrieval.search_calls == []
    # ... while all-scopes run every channel (scoped retrieval where supported).
    retrieval2 = ScopedFakeRetrieval(docs=make_scoped_docs())
    experiments2 = ScopedFakeExperiments(records=[make_experiment()])
    kb_lookup.bind_knowledge_backends(retrieval=retrieval2, experiments=experiments2)
    packet = json.loads(kb_bootstrap.knowledge_bootstrap_tool.func(make_scoped_runtime(ALL_SCOPE), "momentum"))
    assert packet["channel_errors"] == []
    assert retrieval2.scoped_search_calls and retrieval2.search_calls == []
    assert experiments2.used_family + experiments2.used_search > 0


def test_bootstrap_absent_scope_legacy_path_unchanged():
    retrieval = ScopedFakeRetrieval(docs=make_scoped_docs())
    stores = kb_bootstrap.BootstrapStores(retrieval=retrieval, experiments=ScopedFakeExperiments(records=[make_experiment()]), skills=FakeSkills())
    packet = kb_bootstrap.knowledge_bootstrap("momentum in US equities", stores=stores)
    assert packet["channel_errors"] == []
    assert retrieval.search_calls and retrieval.scoped_search_calls == []


# -- Phase 3c: selected scope on real PG backends -------------------------------
#
# The Phase 3b tests above prove scope threading against fakes; these prove
# the real PostgreSQL/SQLite bindings (``pg_retrieval.KnowledgeRetrievalService``
# + ``pg_retrieval.SQLExperimentLookupStore``) enforce the same dataset
# selection end to end, and that the bootstrap middleware threads the
# runtime scope into ``knowledge_bootstrap``. The SQLite fixture mirrors
# ``backend/tests/test_knowledge_pg_retrieval.py`` (same tables, same row
# shapes, same query code paths as PostgreSQL).

FAILURES_SCOPE = {"version": 1, "mode": "selected", "dataset_ids": ["failures"]}
ASSUMPTIONS_SCOPE = {"version": 1, "mode": "selected", "dataset_ids": ["assumptions"]}

_PG3C_TABLES = (
    "research_project",
    "agent_run",
    "finding",
    "experiment",
    "experiment_dataset",
    "experiment_artifact",
    "assumption",
)

PG3C_FAMILY_OK = "c1" * 32
PG3C_FAMILY_FAIL = "d2" * 32
PG3C_EXEC_OK = "c3" * 32
PG3C_EXEC_FAIL = "d4" * 32


@pytest.fixture()
def pg3c_store(tmp_path):
    """Seeded SQLite KB plus the real PG service/store (no fakes)."""
    engine = sa.create_engine(f"sqlite:///{(tmp_path / 'kb_phase3c.db').as_posix()}")
    for table in _PG3C_TABLES:
        Base.metadata.tables[table].create(engine, checkfirst=True)
    factory = sessionmaker(engine, expire_on_commit=False)
    project_id = uuid.uuid4()
    run_id = uuid.uuid4()
    finding_id = uuid.uuid4()
    failure_finding_id = uuid.uuid4()
    experiment_id = uuid.uuid4()
    failure_experiment_id = uuid.uuid4()
    assumption_id = uuid.uuid4()
    with factory() as session:
        session.add(kb_schema.ResearchProjectRow(id=project_id, name="momentum", visibility_scope={}))
        session.add(kb_schema.AgentRunRow(id=run_id, project_id=project_id, agent_type="research", task="phase3c", status="running"))
        session.add(
            FindingRow(
                id=finding_id,
                project_id=project_id,
                canonical_key="empirical:momentum-persists-net-costs",
                finding_type="empirical",
                statement="Cross-sectional momentum persists net of transaction costs on liquid United States stocks.",
                scope={"asset_class": "equity", "market": "US", "universe": "sp500-pit", "horizon": "6m", "frequency": "daily"},
                status="candidate",
                confidence={"overall_tier": "moderate"},
                effective_from=datetime(2015, 1, 1, tzinfo=UTC),
                effective_to=None,
                recorded_at=datetime(2026, 3, 10, tzinfo=UTC),
                created_by_run_id=run_id,
            )
        )
        session.add(
            FindingRow(
                id=failure_finding_id,
                project_id=project_id,
                canonical_key="failure:borrow-costs-short-leg",
                finding_type="failure",
                statement="Short-leg borrow costs erase reversal strategy profits in hard-to-borrow names.",
                scope={"asset_class": "equity", "market": "US", "universe": "sp500-pit", "horizon": "5d", "frequency": "daily"},
                status="candidate",
                confidence={"overall_tier": "moderate"},
                effective_from=datetime(2015, 1, 1, tzinfo=UTC),
                effective_to=None,
                recorded_at=datetime(2026, 3, 11, tzinfo=UTC),
                created_by_run_id=run_id,
            )
        )
        session.add(
            ExperimentRow(
                id=experiment_id,
                project_id=project_id,
                created_by_run_id=run_id,
                experiment_family_hash=PG3C_FAMILY_OK,
                execution_hash=PG3C_EXEC_OK,
                hypothesis="Cross-sectional momentum 126-day backtest",
                methodology={
                    "asset_class": "equity",
                    "market": "US",
                    "universe": "sp500-pit",
                    "horizon": "6m",
                    "frequency": "daily",
                    "sample_period": ["2015-01-01", "2026-09-19"],
                },
                parameters={"signal_name": "mom_126d", "top_n": 20, "eval_retrieval_text": "quintile long-short momentum backtest sharpe turnover"},
                metrics={"sharpe": 0.82},
                outcome="success",
                failure_class=None,
                status="completed",
                started_at=datetime(2026, 1, 10, tzinfo=UTC),
                completed_at=datetime(2026, 1, 11, tzinfo=UTC),
            )
        )
        session.add(
            ExperimentRow(
                id=failure_experiment_id,
                project_id=project_id,
                created_by_run_id=run_id,
                experiment_family_hash=PG3C_FAMILY_FAIL,
                execution_hash=PG3C_EXEC_FAIL,
                hypothesis="Short-horizon reversal backtest with borrow costs",
                methodology={
                    "asset_class": "equity",
                    "market": "US",
                    "universe": "sp500-pit",
                    "horizon": "5d",
                    "frequency": "daily",
                    "sample_period": ["2015-01-01", "2026-09-19"],
                },
                parameters={"signal_name": "reversal_5d", "eval_retrieval_text": "borrow costs short leg reversal drawdown"},
                metrics={"sharpe": -0.4},
                outcome="failure",
                failure_class="execution",
                status="completed",
                started_at=datetime(2026, 2, 10, tzinfo=UTC),
                completed_at=datetime(2026, 2, 11, tzinfo=UTC),
            )
        )
        session.add(
            AssumptionRow(
                id=assumption_id,
                experiment_id=None,
                statement="Closing auction volume absorbs the rebalanced notional without material slippage.",
                category="execution",
                sensitivity="high",
                tested=False,
                status="active",
            )
        )
        session.commit()
    try:
        yield {
            "factory": factory,
            "service": pg_mod.KnowledgeRetrievalService(factory),
            "experiments": pg_mod.SQLExperimentLookupStore(factory),
            "finding_id": str(finding_id),
            "failure_finding_id": str(failure_finding_id),
            "experiment_id": str(experiment_id),
            "failure_experiment_id": str(failure_experiment_id),
            "assumption_id": str(assumption_id),
        }
    finally:
        engine.dispose()


def _pg3c_ctx(scope):
    return KnowledgeRequestContext(user_id="pg3c-user", agent_name="lead", scope=dict(scope))


def test_pg3c_real_backends_satisfy_scoped_protocols(pg3c_store):
    assert isinstance(pg3c_store["service"], kb_lookup.ScopedKnowledgeRetrievalBackend)
    assert isinstance(pg3c_store["experiments"], kb_lookup.ScopedExperimentLookupStore)


def test_pg3c_search_selected_scope_constrains_kinds(pg3c_store):
    service = pg3c_store["service"]
    # Sanity: the unconstrained legacy path recalls every content kind.
    legacy = kb_lookup.ledger_search("momentum", backend=service)
    assert {doc["kind"] for doc in legacy["documents"]} >= {"finding", "experiment", "failure", "assumption"}
    assert {doc["id"] for doc in legacy["documents"]} >= {
        pg3c_store["finding_id"],
        pg3c_store["experiment_id"],
        pg3c_store["failure_experiment_id"],
        pg3c_store["assumption_id"],
    }
    # Findings (+assumptions) scope: only those kinds survive.
    findings = kb_lookup.ledger_search("momentum", backend=service, request_context=_pg3c_ctx(FINDINGS_SCOPE))
    assert {doc["kind"] for doc in findings["documents"]} <= {"finding", "assumption"}
    assert pg3c_store["finding_id"] in {doc["id"] for doc in findings["documents"]}
    assert findings["total"] == len(findings["documents"])
    # Experiments scope: the experiment only (failures are their own dataset).
    experiments = kb_lookup.ledger_search("momentum", backend=service, request_context=_pg3c_ctx(EXPERIMENTS_SCOPE))
    assert {doc["id"] for doc in experiments["documents"]} == {pg3c_store["experiment_id"]}
    # Failures scope: both tables' failure rows, nothing else.
    failures = kb_lookup.ledger_search("momentum", backend=service, request_context=_pg3c_ctx(FAILURES_SCOPE))
    assert {doc["id"] for doc in failures["documents"]} == {pg3c_store["failure_finding_id"], pg3c_store["failure_experiment_id"]}
    assert {doc["kind"] for doc in failures["documents"]} == {"failure"}
    # Assumptions scope: the assumption only.
    assumptions = kb_lookup.ledger_search("momentum", backend=service, request_context=_pg3c_ctx(ASSUMPTIONS_SCOPE))
    assert {doc["id"] for doc in assumptions["documents"]} == {pg3c_store["assumption_id"]}


def test_pg3c_search_scoped_errors_mirror_provider(pg3c_store):
    service = pg3c_store["service"]
    kwargs = {"kinds": ("finding", "experiment", "failure", "assumption"), "filters": {}, "limit": 10, "offset": 0}
    with pytest.raises(ValueError, match="disabled for this turn"):
        service.search_scoped("momentum", **kwargs, context=_pg3c_ctx(DISABLED_SCOPE))
    with pytest.raises(PermissionError, match="unknown dataset"):
        service.search_scoped("momentum", **kwargs, context=_pg3c_ctx(UNKNOWN_SCOPE))
    # all-scopes (and absent scopes) behave as the legacy search.
    scoped_all = service.search_scoped("momentum", **kwargs, context=_pg3c_ctx(ALL_SCOPE))
    legacy = service.search("momentum", **kwargs)
    assert [doc.id for doc in scoped_all.documents] == [doc.id for doc in legacy.documents]
    assert scoped_all.total == legacy.total
    scoped_none = service.search_scoped("momentum", **kwargs, context=KnowledgeRequestContext(user_id="u", scope=None))
    assert [doc.id for doc in scoped_none.documents] == [doc.id for doc in legacy.documents]


def test_pg3c_get_documents_scoped_hides_out_of_scope(pg3c_store):
    service = pg3c_store["service"]
    ctx = _pg3c_ctx(FINDINGS_SCOPE)
    rows = service.get_documents_scoped(
        [pg3c_store["finding_id"], pg3c_store["experiment_id"], str(uuid.uuid4())],
        include_evidence=True,
        context=ctx,
    )
    assert rows[0] is not None and rows[0].id == pg3c_store["finding_id"]
    assert rows[0].payload["canonical_key"] == "empirical:momentum-persists-net-costs"  # real PG row, not a fake
    assert rows[1] is None  # out-of-scope reads as a miss (existence-hiding)
    assert rows[2] is None  # unknown ids read identically
    payload = kb_lookup.ledger_get(
        [pg3c_store["finding_id"], pg3c_store["experiment_id"]],
        include_evidence=True,
        backend=service,
        request_context=ctx,
    )
    assert [doc["id"] for doc in payload["documents"]] == [pg3c_store["finding_id"]]
    assert payload["errors"] == [{"id": pg3c_store["experiment_id"], "error": "not_found"}]


def test_pg3c_experiment_scoped_lookup(pg3c_store):
    store = pg3c_store["experiments"]
    ok_ctx = _pg3c_ctx(EXPERIMENTS_SCOPE)
    findings_ctx = _pg3c_ctx(FINDINGS_SCOPE)
    failures_ctx = _pg3c_ctx(FAILURES_SCOPE)
    # In-scope id + hash lookups resolve through the real store.
    record = store.find_by_id_scoped(pg3c_store["experiment_id"], context=ok_ctx)
    assert record is not None and record.execution_hash == PG3C_EXEC_OK
    by_hash = store.find_by_execution_hash_scoped(PG3C_EXEC_OK, context=ok_ctx)
    assert by_hash is not None and by_hash.id == pg3c_store["experiment_id"]
    # Out-of-scope refs read as not-found (existence-hiding), never as errors.
    assert store.find_by_id_scoped(pg3c_store["experiment_id"], context=findings_ctx) is None
    assert store.find_by_execution_hash_scoped(PG3C_EXEC_OK, context=findings_ctx) is None
    assert store.find_by_id_scoped(str(uuid.uuid4()), context=ok_ctx) is None
    # Failed experiments belong to the failures dataset, not experiments.
    assert store.find_by_id_scoped(pg3c_store["failure_experiment_id"], context=ok_ctx) is None
    failed = store.find_by_id_scoped(pg3c_store["failure_experiment_id"], context=failures_ctx)
    assert failed is not None and failed.execution_hash == PG3C_EXEC_FAIL
    # The service layer surfaces the miss as not-found, like unknown refs.
    with pytest.raises(KnowledgeError, match="Experiment not found"):
        kb_lookup.experiment_get(pg3c_store["experiment_id"], store=store, request_context=findings_ctx)
    # Scope denials mirror the provider error contract.
    with pytest.raises(PermissionError, match="unknown dataset"):
        store.find_by_id_scoped(pg3c_store["experiment_id"], context=_pg3c_ctx(UNKNOWN_SCOPE))
    with pytest.raises(ValueError, match="disabled for this turn"):
        store.find_by_execution_hash_scoped(PG3C_EXEC_OK, context=_pg3c_ctx(DISABLED_SCOPE))


def test_pg3c_document_filters_allowlist_real_backend(pg3c_store):
    service = pg3c_store["service"]
    scope = {
        "version": 1,
        "mode": "selected",
        "dataset_ids": ["findings", "experiments"],
        "document_filters": [{"dataset_id": "findings", "document_ids": [pg3c_store["finding_id"]]}],
    }
    ctx = _pg3c_ctx(scope)
    page = kb_lookup.ledger_search("momentum", backend=service, request_context=ctx)
    assert {doc["id"] for doc in page["documents"]} == {pg3c_store["finding_id"]}
    rows = service.get_documents_scoped(
        [pg3c_store["finding_id"], pg3c_store["experiment_id"]],
        include_evidence=False,
        context=ctx,
    )
    assert [doc.id if doc is not None else None for doc in rows] == [pg3c_store["finding_id"], None]


def test_pg3c_bootstrap_selected_scope_real_backend(pg3c_store):
    stores = kb_bootstrap.BootstrapStores(retrieval=pg3c_store["service"], experiments=pg3c_store["experiments"], skills=FakeSkills())
    packet = kb_bootstrap.knowledge_bootstrap(
        {"topic": "momentum in US equities", "needed_memory": ["validated_findings", "prior_experiments", "failures", "conflicts", "assumptions", "relevant_skills"]},
        stores=stores,
        request_context=_pg3c_ctx(FINDINGS_SCOPE),
    )
    sections = {section["name"]: section for section in packet["sections"]}
    assert {entry["kind"] for entry in sections["validated_findings"]["entries"]} == {"finding"}
    assert pg3c_store["finding_id"] in {entry["id"] for entry in sections["validated_findings"]["entries"]}
    assert sections["conflicts"]["entries"] == []  # conflict kind belongs to no dataset
    assert {entry["kind"] for entry in sections["assumptions"]["entries"]} <= {"assumption"}
    # The experiment-store channels still gap loudly (unscoped store, fail closed).
    assert {entry["channel"] for entry in packet["channel_errors"]} == {"prior_experiments", "failures"}
    assert all(entry["error"] == kb_lookup.SCOPE_NOT_APPLICABLE_MESSAGE for entry in packet["channel_errors"])
    assert sections["relevant_skills"]["entries"]  # skills stay scope-independent


def _pg3c_middleware_stores(pg3c_store):
    return kb_bootstrap.BootstrapStores(retrieval=pg3c_store["service"], experiments=pg3c_store["experiments"], skills=FakeSkills())


def test_pg3c_middleware_scope_gates_injection(pg3c_store):
    middleware = kb_middleware.KnowledgeBootstrapMiddleware(stores=_pg3c_middleware_stores(pg3c_store), enabled=True)
    state = {"messages": [HumanMessage(content="Study momentum in US equities.")]}
    assert middleware.before_agent(state, make_scoped_runtime(DISABLED_SCOPE, thread_id="t1")) is None
    assert middleware.before_agent(state, make_scoped_runtime({"mode": "bogus"}, thread_id="t1")) is None
    for scope in INVALID_SCOPES:
        assert middleware.before_agent(state, make_scoped_runtime(scope, thread_id="t1")) is None, scope
    # Absent scope still injects the full unconstrained packet.
    update = middleware.before_agent(state, make_runtime(thread_id="t1"))
    assert update is not None and "RESEARCH CONTEXT" in update["messages"][0].content


def test_pg3c_middleware_selected_scope_constrains_packet(pg3c_store):
    middleware = kb_middleware.KnowledgeBootstrapMiddleware(stores=_pg3c_middleware_stores(pg3c_store), enabled=True)
    state = {"messages": [HumanMessage(content="Study momentum in US equities.")]}
    update = middleware.before_agent(state, make_scoped_runtime(FINDINGS_SCOPE, thread_id="t1", agent_name="lead"))
    assert update is not None
    content = update["messages"][0].content
    assert "RESEARCH CONTEXT" in content
    # Entries render as "id: title [status]": the in-scope finding's
    # canonical key is present, the out-of-scope titles are absent.
    assert "empirical:momentum-persists-net-costs" in content  # the in-scope finding
    assert "126-day backtest" not in content  # the out-of-scope experiment
    assert "Short-horizon reversal" not in content  # the out-of-scope failure


def test_pg3c_middleware_absent_scope_uses_legacy_path():
    retrieval = ScopedFakeRetrieval(docs=make_scoped_docs())
    stores = kb_bootstrap.BootstrapStores(retrieval=retrieval, experiments=ScopedFakeExperiments(records=[make_experiment()]), skills=FakeSkills())
    middleware = kb_middleware.KnowledgeBootstrapMiddleware(stores=stores, enabled=True)
    update = middleware.before_agent({"messages": [HumanMessage(content="momentum in US equities")]}, make_runtime(thread_id="t1"))
    assert update is not None and "RESEARCH CONTEXT" in update["messages"][0].content
    assert retrieval.search_calls and retrieval.scoped_search_calls == []
    # ... while a selected scope routes the same middleware through search_scoped.
    retrieval2 = ScopedFakeRetrieval(docs=make_scoped_docs())
    stores2 = kb_bootstrap.BootstrapStores(retrieval=retrieval2, experiments=ScopedFakeExperiments(records=[make_experiment()]), skills=FakeSkills())
    middleware2 = kb_middleware.KnowledgeBootstrapMiddleware(stores=stores2, enabled=True)
    update2 = middleware2.before_agent({"messages": [HumanMessage(content="momentum in US equities")]}, make_scoped_runtime(FINDINGS_SCOPE, thread_id="t1"))
    assert update2 is not None
    assert retrieval2.scoped_search_calls and retrieval2.search_calls == []


# -- Phase 4a: converge onto the upstream tools path ---------------------------
#
# The generic upstream search tool (``knowledge_search`` via
# ``execute_knowledge_search`` with provider resolution rebound to
# ``quantflow-pg``) is registered ALONGSIDE the fork ledger tools — not
# instead of them. These tests prove the convergence contract:
#
# * the upstream entry path reaches a ``quantflow-pg``-bound provider,
# * the upstream tool set holds exactly the ``quantflow-pg``-compatible
#   generic tools (``knowledge_search`` only),
# * every ledger operation keeps the bound-backend path (the upstream
#   entry cannot serve its frozen JSON semantics — see
#   ``LEDGER_SEARCH_ROUTES``), with Phase 3 scope threading and frozen
#   payload shapes intact,
# * both assembly sites register the converged set without conflicts or
#   duplicates, and the lead-agent site registers the built-in provider.


@pytest.fixture()
def _phase4a_clean_extensions():
    """Reset the process-wide extension registry, restoring it after."""
    from deerflow.extensions import get_loaded_extensions, reset_loaded_extensions, set_loaded_extensions

    previous = get_loaded_extensions()
    reset_loaded_extensions()
    try:
        yield
    finally:
        set_loaded_extensions(previous)


def _phase4a_provider_names():
    from deerflow.extensions import get_loaded_extensions

    return [name for _, (name, _) in get_loaded_extensions().knowledge_providers]


def test_phase4a_routing_table_declares_backend_path():
    assert kb_lookup.LEDGER_SEARCH_ROUTES == {
        "ledger_search": kb_lookup.SEARCH_ROUTE_BACKEND,
        "ledger_get": kb_lookup.SEARCH_ROUTE_BACKEND,
        "experiment_get": kb_lookup.SEARCH_ROUTE_BACKEND,
    }
    assert kb_lookup.resolve_search_route("ledger_search") == "backend"
    assert kb_lookup.resolve_search_route("ledger_get") == "backend"
    assert kb_lookup.resolve_search_route("experiment_get") == "backend"
    with pytest.raises(KnowledgeValidationError, match="unknown ledger operation"):
        kb_lookup.resolve_search_route("knowledge_search")


def test_phase4a_declared_route_matches_observed_backend_path():
    # The routing table is not aspirational: every ledger operation calls
    # the bound backend fakes (legacy or scoped entry points per the
    # Phase 3 thread), never the upstream entry, with frozen JSON shapes.
    retrieval = ScopedFakeRetrieval(docs=make_scoped_docs())
    experiments = ScopedFakeExperiments(records=[make_experiment()])
    kb_lookup.bind_knowledge_backends(retrieval=retrieval, experiments=experiments)
    assert kb_lookup.resolve_search_route("ledger_search") == kb_lookup.SEARCH_ROUTE_BACKEND
    page = kb_lookup.ledger_search("momentum", backend=retrieval)
    assert set(page) == {"documents", "count", "limit", "offset", "total", "query", "kinds", "filters"}
    assert len(retrieval.search_calls) == 1 and retrieval.scoped_search_calls == []
    scoped_page = kb_lookup.ledger_search("momentum", backend=retrieval, request_context=KnowledgeRequestContext(user_id="u", scope=dict(FINDINGS_SCOPE)))
    assert {doc["id"] for doc in scoped_page["documents"]} == {"F17", "A54"}
    assert len(retrieval.scoped_search_calls) == 1
    assert kb_lookup.resolve_search_route("ledger_get") == kb_lookup.SEARCH_ROUTE_BACKEND
    got = kb_lookup.ledger_get("F17,F404", backend=retrieval)
    assert [doc["id"] for doc in got["documents"]] == ["F17"]
    assert got["errors"] == [{"id": "F404", "error": "not_found"}]
    assert len(retrieval.get_calls) == 1 and retrieval.scoped_get_calls == []
    assert kb_lookup.resolve_search_route("experiment_get") == kb_lookup.SEARCH_ROUTE_BACKEND
    by_hash = kb_lookup.experiment_get(EXEC_HASH, store=experiments)
    assert by_hash["lookup"] == "execution_hash" and by_hash["experiment"]["id"] == EXP_ID


def test_phase4a_upstream_entry_reaches_quantflow_pg(_phase4a_clean_extensions, monkeypatch):
    from dataclasses import replace

    from deerflow.extensions import get_loaded_extensions, set_loaded_extensions
    from deerflow.knowledge import tools as kb_tools
    from deerflow.knowledge.contract import KnowledgeCapabilities, KnowledgeSource, RetrievalHit, RetrievalResult

    class Phase4aFakeProvider:
        capabilities = KnowledgeCapabilities(operations=frozenset({"search"}))

        def __init__(self):
            self.calls: list[dict] = []

        async def search(self, query):
            self.calls.append({"query": query})
            return RetrievalResult(
                hits=[
                    RetrievalHit(
                        content="momentum persists net of costs",
                        source=KnowledgeSource(provider="quantflow-pg", title="Finding F17", excerpt="momentum persists net of costs"),
                    )
                ]
            )

        async def search_scoped(self, query, *, context):
            return await self.search(query)

    fake = Phase4aFakeProvider()
    set_loaded_extensions(replace(get_loaded_extensions(), knowledge_providers=(("test:phase4a", ("quantflow-pg", fake)),)))

    def _tool_config(name):
        if name != "knowledge_search":
            return None
        return types.SimpleNamespace(use="deerflow.knowledge.tools:knowledge_search_tool", model_extra={"provider": "quantflow-pg"})

    monkeypatch.setattr(kb_tools, "get_app_config", lambda: types.SimpleNamespace(get_tool_config=_tool_config))
    text, artifact = asyncio.run(kb_tools.execute_knowledge_search("momentum", make_runtime(agent_name="lead")))
    assert len(fake.calls) == 1 and fake.calls[0]["query"] == "momentum"
    assert "momentum persists net of costs" in text
    assert isinstance(artifact, dict)


def test_phase4a_upstream_tool_set_is_quantflow_pg_compatible():
    from deerflow.knowledge.providers import KNOWLEDGE_TOOL_OPERATIONS, operation_capabilities
    from deerflow.knowledge.quantflow_provider import CAPABILITIES

    upstream = kb_lookup.get_upstream_knowledge_tools()
    assert [tool.name for tool in upstream] == ["knowledge_search"]
    for tool in upstream:
        operation = KNOWLEDGE_TOOL_OPERATIONS[tool.name]
        assert operation_capabilities(CAPABILITIES, operation).supports("search")
    # The excluded generic tools' operations are not advertised, so
    # registering them would register dead tools (search_provider refuses
    # unadvertised operations with "operation is unavailable").
    assert not CAPABILITIES.supports("entries_search")
    assert not CAPABILITIES.supports("relations_search")
    # Fork and upstream names are disjoint; the ledger_* aliases resolve.
    fork_names = [tool.name for tool in kb_lookup.get_knowledge_tools()]
    assert {"ledger_search", "ledger_get", "experiment_get"} <= set(fork_names)
    assert not (set(fork_names) & {tool.name for tool in upstream})


def test_phase4a_factory_registers_converged_set(tmp_path, _knowledge_config_enabled, _phase4a_clean_extensions):
    from deerflow.agents.factory import _assemble_from_features
    from deerflow.agents.features import RuntimeFeatures

    bind_all(tmp_path)
    _, extra_tools = _assemble_from_features(RuntimeFeatures(knowledge=True), name="test-agent")
    names = [tool.name for tool in extra_tools]
    assert len(names) == len(set(names))  # no duplicates
    assert names[:6] == ["knowledge_search", "ledger_search", "ledger_get", "experiment_get", "artifact_manifest", "artifact_open"]
    assert _phase4a_provider_names().count("quantflow-pg") == 1
    # Repeated builds neither duplicate tools nor re-register the provider.
    _, extra_tools2 = _assemble_from_features(RuntimeFeatures(knowledge=True), name="test-agent")
    assert [tool.name for tool in extra_tools2][:6] == names[:6]
    assert _phase4a_provider_names().count("quantflow-pg") == 1


def test_phase4a_lead_agent_appends_converged_set_without_conflicts(tmp_path, _knowledge_config_enabled, _phase4a_clean_extensions):
    from deerflow.agents.lead_agent.agent import _append_knowledge_tools_without_name_conflicts

    bind_all(tmp_path)
    tools: list = []
    _append_knowledge_tools_without_name_conflicts(tools)
    assert [tool.name for tool in tools] == ["knowledge_search", "ledger_search", "ledger_get", "experiment_get", "artifact_manifest", "artifact_open"]
    assert _phase4a_provider_names().count("quantflow-pg") == 1
    # Skip-on-conflict: pre-existing same-name tools (operator- or
    # user-provided) win; nothing is dropped or duplicated.
    sentinel_search = types.SimpleNamespace(name="knowledge_search")
    sentinel_ledger = types.SimpleNamespace(name="ledger_search")
    tools2: list = [sentinel_search, sentinel_ledger]
    _append_knowledge_tools_without_name_conflicts(tools2)
    assert tools2[0] is sentinel_search and tools2[1] is sentinel_ledger
    assert [tool.name for tool in tools2] == ["knowledge_search", "ledger_search", "ledger_get", "experiment_get", "artifact_manifest", "artifact_open"]


def test_phase4a_lead_agent_noop_when_unbound(_knowledge_config_enabled, _phase4a_clean_extensions):
    from deerflow.agents.lead_agent.agent import _append_knowledge_tools_without_name_conflicts

    tools: list = []
    _append_knowledge_tools_without_name_conflicts(tools)
    assert tools == []
    assert "quantflow-pg" not in _phase4a_provider_names()

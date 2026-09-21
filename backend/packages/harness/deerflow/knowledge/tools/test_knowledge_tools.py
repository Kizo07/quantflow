"""Standalone tests for the Research Knowledge Plane agent tools (Phase 2).

Covers ``deerflow.knowledge.tools`` (bootstrap intent parsing, retrieval
planning, context-packet assembly, lookup pure functions, artifact
locator reads, the ``@tool`` wrappers and the run-start middleware)
with in-memory fakes behind the storage-boundary protocols — no
database, no config files, no integration wiring.

Run from anywhere (the bootstrap below locates the harness package)::

    python -m pytest backend/packages/harness/deerflow/knowledge/tools/test_knowledge_tools.py -q
"""

import asyncio
import json
import sys
import types
from pathlib import Path

_HARNESS_DIR = Path(__file__).resolve().parents[3]
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

import pytest  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from deerflow.knowledge.artifacts.backends import LocalFilesystemBackend  # noqa: E402
from deerflow.knowledge.artifacts.store import ArtifactStore  # noqa: E402
from deerflow.knowledge.tools import bootstrap as kb_bootstrap  # noqa: E402
from deerflow.knowledge.tools import lookup as kb_lookup  # noqa: E402
from deerflow.knowledge.tools import middleware as kb_middleware  # noqa: E402
from deerflow.knowledge.write_api import ExperimentRecord, KnowledgeValidationError  # noqa: E402

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

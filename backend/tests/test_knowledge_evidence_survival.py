"""Phase 1b: packet/ledger rendering through upstream citation helpers.

(a) Byte-parity: packet + ledger rendering on fixed fixtures equals the
    committed pre-Phase-1b outputs below (captured BEFORE the production
    edits by rendering the same fixtures; any render change fails here).
(b) Evidence survival: fork-sourced evidence (packet rows, ledger docs,
    bootstrap entries) passes through upstream ``format_result`` uncut --
    excerpts verbatim, content-hashed IDs, sources retained with
    attribution -- and truncation keeps citations paired with visible
    excerpts via ``prune_sources``.

Run from backend/::

    PYTHONPATH=. .venv/bin/python -m pytest tests/test_knowledge_evidence_survival.py -q
"""

import json

import pytest

from deerflow.knowledge.agent_tools import bootstrap as bootstrap_mod
from deerflow.knowledge.agent_tools import lookup as lookup_mod
from deerflow.knowledge.contract import KnowledgeSource, RetrievalResult
from deerflow.knowledge.retrieval import fusion as fusion_mod
from deerflow.knowledge.retrieval import packet as packet_mod
from deerflow.knowledge.retrieval.packet import ConflictView, PacketBudget
from deerflow.knowledge.retrieval.planner import Candidate, ChannelHit, ScopeFilter
from deerflow.knowledge.sources import ARTIFACT_VERSION, TRUNCATION_MARKER, format_result, prune_sources, truncate
from deerflow.knowledge.write_api import ExperimentRecord

NOW = "2026-09-20T00:00:00+00:00"
PROJECT_ID = "11111111-1111-1111-1111-111111111111"
RUN_ID = "33333333-3333-3333-3333-333333333333"
EXP_ID = "55555555-5555-5555-5555-555555555555"

# ---------------------------------------------------------------------------
# Committed pre-Phase-1b outputs (captured before the production edits).
# ---------------------------------------------------------------------------

EXPECTED_PACKET_FULL_TEXT = "RESEARCH CONTEXT\n\nCurrent consensus\n- F17 [validated]: Momentum persists — cross-sectional momentum persists net of costs (evidence: E542, A9)\n\nClosest prior experiments\n- E542: Momentum 126d backtest — top-decile rebalance ME sharpe 0.82\n\nRelevant failures / negative evidence\n- E304 [failure]: Costs erase momentum — failed because transaction costs erase paper profits (evidence: A11)\n\nKnown contradictions\n- F198 [disputed]: Momentum is dead — contradicting evidence after costs\n\nImportant assumptions\n- A54: Zero slippage — backtest assumes zero slippage\n\nReusable priors / skills\n- SK1: robust-factor-backtest — reusable backtest harness\n- P4: turnover-sensitive-signals — penalize high-turnover signals\n\nOpen questions\n- q1: Does momentum survive 2026 costs?\n\nDerived summaries (L0/L1)\n  - L0 [momentum] (pending_phase4): Phase 4 consolidation has not run yet; open L2 evidence directly.\n\n[budget: 235/4000 tokens, 952/16384 bytes]"  # noqa: E501

EXPECTED_PACKET_TINY_TEXT = "RESEARCH CONTEXT\n\nCurrent consensus\n- (none retrieved)\n- ... truncated to budget (narrow the intent or raise the budget)\n\nClosest prior experiments\n- (none retrieved)\n- ... truncated to budget (narrow the intent or raise the budget)\n\nRelevant failures / negative evidence\n- (none retrieved)\n- ... truncated to budget (narrow the intent or raise the budget)\n\nKnown contradictions\n- (none retrieved)\n- ... truncated to budget (narrow the intent or raise the budget)\n\nImportant assumptions\n- (none retrieved)\n- ... truncated to budget (narrow the intent or raise the budget)\n\nReusable priors / skills\n- (none retrieved)\n- ... truncated to budget (narrow the intent or raise the budget)\n\nOpen questions\n- (none retrieved)\n- ... truncated to budget (narrow the intent or raise the budget)\n\nDerived summaries (L0/L1)\n  - L0 (pending_phase4): Phase 4 consolidation has not run yet; open L2 evidence directly.\n\n[budget: 236/40 tokens, 941/400 bytes]"  # noqa: E501

EXPECTED_PACKET_EMPTY_TEXT = "RESEARCH CONTEXT\n\nCurrent consensus\n- (none retrieved)\n\nClosest prior experiments\n- (none retrieved)\n\nRelevant failures / negative evidence\n- (none retrieved)\n\nKnown contradictions\n- (none retrieved)\n\nImportant assumptions\n- (none retrieved)\n\nReusable priors / skills\n- (none retrieved)\n\nOpen questions\n- (none retrieved)\n\nDerived summaries (L0/L1)\n  - L0 (pending_phase4): Phase 4 consolidation has not run yet; open L2 evidence directly.\n\n[budget: 121/4000 tokens, 483/16384 bytes]"  # noqa: E501

EXPECTED_PACKET_CONFLICT_TEXT = "RESEARCH CONTEXT\n\nCurrent consensus\n- F17 [validated]: Momentum persists — cross-sectional momentum persists net of costs (evidence: E542, A9)\n\nClosest prior experiments\n- E542: Momentum 126d backtest — top-decile rebalance ME sharpe 0.82\n\nRelevant failures / negative evidence\n- E304 [failure]: Costs erase momentum — failed because transaction costs erase paper profits (evidence: A11)\n\nKnown contradictions\n- C31: Conflict C31 [open] — F17 says momentum persists; F198 says it is dead. (evidence: F17, F198)\n- F198 [disputed]: Momentum is dead — contradicting evidence after costs\n\nImportant assumptions\n- A54: Zero slippage — backtest assumes zero slippage\n\nReusable priors / skills\n- SK1: robust-factor-backtest — reusable backtest harness\n- P4: turnover-sensitive-signals — penalize high-turnover signals\n\nOpen questions\n- (none retrieved)\n\nDerived summaries (L0/L1)\n  - L0 (pending_phase4): Phase 4 consolidation has not run yet; open L2 evidence directly.\n\n[budget: 252/4000 tokens, 1024/16384 bytes]"  # noqa: E501

EXPECTED_PACKET_CLIP_TEXT = "RESEARCH CONTEXT\n\nCurrent consensus\n- L1 [candidate]: Long — word word word word word word word word word word…\n\nClosest prior experiments\n- (none retrieved)\n\nRelevant failures / negative evidence\n- (none retrieved)\n\nKnown contradictions\n- (none retrieved)\n\nImportant assumptions\n- (none retrieved)\n\nReusable priors / skills\n- (none retrieved)\n\nOpen questions\n- (none retrieved)\n\nDerived summaries (L0/L1)\n  - L0 (pending_phase4): Phase 4 consolidation has not run yet; open L2 evidence directly.\n\n[budget: 135/4000 tokens, 544/16384 bytes]"  # noqa: E501

EXPECTED_PACKET_FULL_DICT = {
    "budget": {"max_bytes": 16384, "max_snippet_chars": 600, "max_tokens": 4000},
    "bytes_used": 952,
    "l0": {"level": "L0", "note": "Phase 4 consolidation has not run yet; open L2 evidence directly.", "status": "pending_phase4", "topic_key": "momentum"},
    "l1": [],
    "sections": [
        {
            "bytes_used": 125,
            "items": [{"evidence": ["E542", "A9"], "flags": ["validated"], "id": "F17", "kind": "finding", "score": 0.03786885245901639, "snippet": "cross-sectional momentum persists net of costs", "title": "Momentum persists"}],
            "name": "consensus",
            "title": "Current consensus",
            "tokens_used": 32,
            "truncated": False,
        },
        {
            "bytes_used": 95,
            "items": [{"evidence": [], "flags": [], "id": "E542", "kind": "experiment", "score": 0.027049180327868853, "snippet": "top-decile rebalance ME sharpe 0.82", "title": "Momentum 126d backtest"}],
            "name": "prior-experiments",
            "title": "Closest prior experiments",
            "tokens_used": 24,
            "truncated": False,
        },
        {
            "bytes_used": 148,
            "items": [{"evidence": ["A11"], "flags": ["failure"], "id": "E304", "kind": "failure", "score": 0.03516393442622951, "snippet": "failed because transaction costs erase paper profits", "title": "Costs erase momentum"}],
            "name": "failures",
            "title": "Relevant failures / negative evidence",
            "tokens_used": 38,
            "truncated": False,
        },
        {
            "bytes_used": 94,
            "items": [{"evidence": [], "flags": ["disputed"], "id": "F198", "kind": "finding", "score": 0.027943548387096777, "snippet": "contradicting evidence after costs", "title": "Momentum is dead"}],
            "name": "conflicts",
            "title": "Known contradictions",
            "tokens_used": 23,
            "truncated": False,
        },
        {
            "bytes_used": 76,
            "items": [{"evidence": [], "flags": [], "id": "A54", "kind": "assumption", "score": 0.026612903225806454, "snippet": "backtest assumes zero slippage", "title": "Zero slippage"}],
            "name": "assumptions",
            "title": "Important assumptions",
            "tokens_used": 20,
            "truncated": False,
        },
        {
            "bytes_used": 150,
            "items": [
                {"evidence": [], "flags": [], "id": "SK1", "kind": "skill", "score": 0.02619047619047619, "snippet": "reusable backtest harness", "title": "robust-factor-backtest"},
                {"evidence": [], "flags": [], "id": "P4", "kind": "prior", "score": 0.025781250000000002, "snippet": "penalize high-turnover signals", "title": "turnover-sensitive-signals"},
            ],
            "name": "skills",
            "title": "Reusable priors / skills",
            "tokens_used": 38,
            "truncated": False,
        },
        {
            "bytes_used": 53,
            "items": [{"evidence": [], "flags": [], "id": "q1", "kind": "open_question", "score": 0.0, "snippet": "", "title": "Does momentum survive 2026 costs?"}],
            "name": "open-questions",
            "title": "Open questions",
            "tokens_used": 14,
            "truncated": False,
        },
    ],
    "tokens_used": 235,
    "warnings": [],
    "within_budget": True,
}  # noqa: E501

EXPECTED_BOOTSTRAP_FULL = {
    "budget": {"channel_limit": 10, "max_chars": 12000, "max_items": 40},
    "channel_errors": [],
    "intent": {
        "asset_class": "equity",
        "concepts": ["momentum"],
        "frequency": None,
        "horizon": None,
        "markets": ["US"],
        "needed_memory": ["validated_findings", "prior_experiments", "failures", "conflicts", "assumptions", "relevant_skills"],
        "requested_period": None,
        "topic": "momentum in US equities",
        "universe": "liquid common stocks",
    },
    "sections": [
        {
            "count": 1,
            "entries": [{"evidence_refs": ["55555555-5555-5555-5555-555555555555"], "id": "F17", "kind": "finding", "score": 0.8, "status": "validated", "summary": "Summary F17", "title": "Title F17"}],
            "name": "validated_findings",
            "title": "Current consensus",
            "truncated": False,
        },
        {
            "count": 1,
            "entries": [
                {
                    "evidence_refs": [],
                    "id": "55555555-5555-5555-5555-555555555555",
                    "kind": "experiment",
                    "methodology": {"frequency": "daily", "universe": "sp500-pit"},
                    "score": None,
                    "status": "completed",
                    "summary": "status=completed outcome=success family=bbbbbbbbbbbb",
                    "title": "Cross-sectional momentum persists net of costs.",
                }
            ],
            "name": "prior_experiments",
            "title": "Closest prior experiments",
            "truncated": False,
        },
        {"count": 0, "entries": [], "name": "failures", "title": "Relevant failures / negative evidence", "truncated": False},
        {
            "count": 1,
            "entries": [{"evidence_refs": ["55555555-5555-5555-5555-555555555555"], "id": "C31", "kind": "conflict", "score": 0.8, "status": "open", "summary": "Summary C31", "title": "Title C31"}],
            "name": "conflicts",
            "title": "Known contradictions",
            "truncated": False,
        },
        {
            "count": 1,
            "entries": [{"evidence_refs": ["55555555-5555-5555-5555-555555555555"], "id": "A54", "kind": "assumption", "score": 0.8, "status": "active", "summary": "Summary A54", "title": "Title A54"}],
            "name": "assumptions",
            "title": "Important assumptions",
            "truncated": False,
        },
        {
            "count": 1,
            "entries": [{"evidence_refs": [], "id": "cross-sectional-factor-backtest@2.3", "kind": "skill", "score": 0.9, "status": "2.3", "summary": "Robust factor backtest harness.", "title": "cross-sectional-factor-backtest"}],
            "name": "relevant_skills",
            "title": "Reusable priors / skills",
            "truncated": False,
        },
        {
            "count": 1,
            "entries": [{"evidence_refs": ["C31"], "id": "oq:C31", "kind": "open_question", "score": None, "status": "open", "summary": "Summary C31", "title": "Resolve contradiction C31"}],
            "name": "open_questions",
            "title": "Open questions",
            "truncated": False,
        },
    ],
    "text": "RESEARCH CONTEXT\ntopic: momentum in US equities\n\nCurrent consensus\n  - F17: Title F17 [validated]\n\nClosest prior experiments\n  - 55555555-5555-5555-5555-555555555555: Cross-sectional momentum persists net of costs. [completed]\n\nRelevant failures / negative evidence\n  - (none retrieved)\n\nKnown contradictions\n  - C31: Title C31 [open]\n\nImportant assumptions\n  - A54: Title A54 [active]\n\nReusable priors / skills\n  - cross-sectional-factor-backtest@2.3: cross-sectional-factor-backtest [2.3]\n\nOpen questions\n  - oq:C31: Resolve contradiction C31 [open]\n",  # noqa: E501
    "truncated": False,
}  # noqa: E501

EXPECTED_BOOTSTRAP_TRUNCATED = {
    "budget": {"channel_limit": 5, "max_chars": 500, "max_items": 3},
    "channel_errors": [],
    "intent": {
        "asset_class": "unknown",
        "concepts": ["momentum"],
        "frequency": None,
        "horizon": None,
        "markets": [],
        "needed_memory": ["validated_findings", "prior_experiments", "failures", "conflicts", "assumptions", "relevant_skills"],
        "requested_period": None,
        "topic": "momentum",
        "universe": None,
    },
    "sections": [
        {
            "count": 2,
            "entries": [
                {
                    "evidence_refs": ["55555555-5555-5555-5555-555555555555"],
                    "id": "F000",
                    "kind": "finding",
                    "score": 0.8,
                    "status": "validated",
                    "summary": "ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss",
                    "title": "Title F000",
                },
                {
                    "evidence_refs": ["55555555-5555-5555-5555-555555555555"],
                    "id": "F001",
                    "kind": "finding",
                    "score": 0.8,
                    "status": "validated",
                    "summary": "ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss",
                    "title": "Title F001",
                },
            ],
            "name": "validated_findings",
            "title": "Current consensus",
            "truncated": True,
        },
        {"count": 0, "entries": [], "name": "prior_experiments", "title": "Closest prior experiments", "truncated": True},
        {"count": 0, "entries": [], "name": "failures", "title": "Relevant failures / negative evidence", "truncated": False},
        {"count": 0, "entries": [], "name": "conflicts", "title": "Known contradictions", "truncated": False},
        {"count": 0, "entries": [], "name": "assumptions", "title": "Important assumptions", "truncated": False},
        {"count": 0, "entries": [], "name": "relevant_skills", "title": "Reusable priors / skills", "truncated": True},
    ],
    "text": "RESEARCH CONTEXT\ntopic: momentum\n\nCurrent consensus\n  - F000: Title F000 [validated]\n  - F001: Title F001 [validated]\n  (section truncated to budget; narrow the query or page per-channel tools)\n\nClosest prior experiments\n  - (none retrieved)\n  (section truncated to budget; narrow the query or page per-channel tools)\n\nRelevant failures / negative evidence\n  - (none retrieved)\n\nKnown contradictions\n  - (none retrieved)\n\nImportant assumptions\n  - (none retrieved)\n\nReusable priors / skills\n  - (none retrieved)\n  (section truncated to budget; narrow the query or page per-channel tools)\n\n(packet truncated to budget; use ledger_search/experiment_get for depth)\n",  # noqa: E501
    "truncated": True,
}  # noqa: E501

EXPECTED_BOOTSTRAP_FORMAT_MAPPING = "RESEARCH CONTEXT\ntopic: momentum in US equities\n\nCurrent consensus\n  - F17: Title F17 [validated]\n\nClosest prior experiments\n  - 55555555-5555-5555-5555-555555555555: Cross-sectional momentum persists net of costs. [completed]\n\nRelevant failures / negative evidence\n  - (none retrieved)\n\nKnown contradictions\n  - C31: Title C31 [open]\n\nImportant assumptions\n  - A54: Title A54 [active]\n\nReusable priors / skills\n  - cross-sectional-factor-backtest@2.3: cross-sectional-factor-backtest [2.3]\n\nOpen questions\n  - oq:C31: Resolve contradiction C31 [open]\n"  # noqa: E501

EXPECTED_LEDGER_SEARCH = {
    "count": 1,
    "documents": [{"evidence_refs": ["55555555-5555-5555-5555-555555555555"], "id": "F17", "kind": "finding", "payload": {}, "score": 0.8, "status": "validated", "summary": "Summary F17", "title": "Title F17"}],
    "filters": {"status": "validated"},
    "kinds": ["finding"],
    "limit": 10,
    "offset": 0,
    "query": "momentum",
    "total": 1,
}  # noqa: E501

EXPECTED_LEDGER_GET = {
    "documents": [{"evidence_refs": ["55555555-5555-5555-5555-555555555555"], "id": "F17", "kind": "finding", "payload": {}, "score": 0.8, "status": "validated", "summary": "Summary F17", "title": "Title F17"}],
    "errors": [{"error": "not_found", "id": "F404"}],
    "include_evidence": True,
}  # noqa: E501


# ---------------------------------------------------------------------------
# Fixed fixtures (same builders the baseline was captured with).
# ---------------------------------------------------------------------------


def make_candidate(candidate_id, **overrides):
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
    kwargs["scope"] = dict(kwargs.get("scope") or {})
    kwargs["metadata"] = dict(kwargs.get("metadata") or {})
    return Candidate(**kwargs)


def make_scope(**overrides):
    kwargs = {
        "asset_class": "equity",
        "markets": ("US",),
        "universe": "sp500-pit",
        "horizon": "6m",
        "frequency": "daily",
    }
    kwargs.update(overrides)
    return ScopeFilter(**kwargs)


def fused_fixture():
    """Ranked FusionResult spanning every packet section (parity fixture)."""
    scope = make_scope()
    hits = [
        ChannelHit(candidate=make_candidate("F17", status="validated", title="Momentum persists", text="cross-sectional momentum persists net of costs", evidence=("E542", "A9")), channel="lexical", rank=1),
        ChannelHit(candidate=make_candidate("E542", kind="experiment", status="completed", title="Momentum 126d backtest", text="top-decile rebalance ME sharpe 0.82"), channel="structured", rank=1),
        ChannelHit(candidate=make_candidate("E304", kind="failure", status="failed", title="Costs erase momentum", text="failed because transaction costs erase paper profits", evidence=("A11",)), channel="failure", rank=1),
        ChannelHit(candidate=make_candidate("F198", status="disputed", title="Momentum is dead", text="contradicting evidence after costs"), channel="lexical", rank=2),
        ChannelHit(candidate=make_candidate("A54", kind="assumption", status="active", title="Zero slippage", text="backtest assumes zero slippage"), channel="structured", rank=2),
        ChannelHit(candidate=make_candidate("SK1", kind="skill", status="active", title="robust-factor-backtest", text="reusable backtest harness"), channel="structured", rank=3),
        ChannelHit(candidate=make_candidate("P4", kind="prior", status="active", title="turnover-sensitive-signals", text="penalize high-turnover signals"), channel="structured", rank=4),
    ]
    return fusion_mod.fuse(hits, scope, now=NOW)


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
        "family_hash": "b" * 64,
        "execution_hash": "a" * 64,
        "status": "completed",
        "outcome": "success",
        "started_at": "2026-01-01T00:00:00+00:00",
    }
    payload.update(overrides)
    return ExperimentRecord(**payload)


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
    return lookup_mod.RetrievalDocument(**payload)


class FakeRetrieval:
    """In-memory KnowledgeRetrievalBackend fake (parity fixture)."""

    def __init__(self, docs=None):
        self.docs = {doc.id: doc for doc in (docs or [])}

    def search(self, query, *, kinds, filters, limit, offset):
        hits = [doc for doc in self.docs.values() if doc.kind in kinds]
        page = hits[offset : offset + limit]
        return lookup_mod.RetrievalPage(documents=list(page), limit=limit, offset=offset, total=len(hits))

    def get_documents(self, ids, *, include_evidence):
        return [self.docs.get(doc_id) for doc_id in ids]


class FakeExperimentLookup:
    """In-memory ExperimentLookupStore fake (parity fixture)."""

    def __init__(self, records=None):
        self.records = list(records or [])

    def find_by_id(self, experiment_id):
        return next((record for record in self.records if record.id == experiment_id), None)

    def find_by_execution_hash(self, execution_hash):
        return next((record for record in self.records if record.execution_hash == execution_hash), None)

    def find_by_family_hash(self, family_hash, *, limit, offset):
        return [record for record in self.records if record.family_hash == family_hash][offset : offset + limit]

    def search_experiments(self, filters, *, limit, offset):
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
    """In-memory SkillCatalog fake (parity fixture)."""

    def __init__(self, skills=None):
        self.skills = (
            skills
            if skills is not None
            else [
                {"id": "cross-sectional-factor-backtest@2.3", "name": "cross-sectional-factor-backtest", "description": "Robust factor backtest harness.", "version": "2.3", "score": 0.9},
            ]
        )

    def find_skills(self, concepts, *, limit):
        return list(self.skills[:limit])


def make_stores(**overrides):
    stores = {
        "retrieval": FakeRetrieval(docs=[make_doc(), make_doc("C31", kind="conflict", status="open"), make_doc("A54", kind="assumption", status="active")]),
        "experiments": FakeExperimentLookup(records=[make_experiment()]),
        "skills": FakeSkills(),
    }
    stores.update(overrides)
    return bootstrap_mod.BootstrapStores(**stores)


def stable_packet_dict(packet):
    """Drop the run-varying timestamp so packets compare byte-stable."""
    packet = dict(packet)
    packet.pop("generated_at", None)
    return packet


# ---------------------------------------------------------------------------
# (a) Byte-parity with pre-Phase-1b rendering.
# ---------------------------------------------------------------------------


class TestPacketParity:
    def test_full_packet_text(self):
        packet = packet_mod.build_context_packet(fused_fixture(), open_questions=["Does momentum survive 2026 costs?"], topic_key="momentum")
        assert packet.to_text() == EXPECTED_PACKET_FULL_TEXT

    def test_tiny_budget_truncation_text(self):
        packet = packet_mod.build_context_packet(fused_fixture(), open_questions=["q?"], budget=PacketBudget(max_tokens=40, max_bytes=400))
        assert packet.to_text() == EXPECTED_PACKET_TINY_TEXT

    def test_empty_packet_text(self):
        assert packet_mod.build_context_packet(fusion_mod.FusionResult()).to_text() == EXPECTED_PACKET_EMPTY_TEXT

    def test_conflict_view_text(self):
        view = ConflictView(id="C31", status="open", member_ids=("F17", "F198"), summary="F17 says momentum persists; F198 says it is dead.")
        packet = packet_mod.build_context_packet(fused_fixture(), conflicts=[view])
        assert packet.to_text() == EXPECTED_PACKET_CONFLICT_TEXT

    def test_snippet_clip_text(self):
        candidate = make_candidate("L1", title="Long", text="word " * 500)
        result = fusion_mod.fuse([ChannelHit(candidate=candidate, channel="lexical", rank=1)], make_scope(), now=NOW)
        packet = packet_mod.build_context_packet(result, budget=PacketBudget(max_snippet_chars=50))
        assert packet.to_text() == EXPECTED_PACKET_CLIP_TEXT

    def test_full_packet_dict(self):
        packet = packet_mod.build_context_packet(fused_fixture(), open_questions=["Does momentum survive 2026 costs?"], topic_key="momentum")
        assert packet.to_dict() == EXPECTED_PACKET_FULL_DICT


class TestBootstrapParity:
    def test_full_packet(self):
        packet = stable_packet_dict(bootstrap_mod.knowledge_bootstrap("momentum in US equities", stores=make_stores()))
        assert packet == EXPECTED_BOOTSTRAP_FULL
        assert packet["text"] == EXPECTED_BOOTSTRAP_FULL["text"]

    def test_truncated_packet(self):
        docs = [make_doc(f"F{i:03d}", summary="s" * 200) for i in range(10)]
        stores = make_stores(retrieval=FakeRetrieval(docs=docs))
        packet = stable_packet_dict(bootstrap_mod.knowledge_bootstrap("momentum", stores=stores, budget=bootstrap_mod.PacketBudget(max_items=3, max_chars=500, channel_limit=5)))
        assert packet == EXPECTED_BOOTSTRAP_TRUNCATED

    def test_format_mapping(self):
        packet = stable_packet_dict(bootstrap_mod.knowledge_bootstrap("momentum in US equities", stores=make_stores()))
        assert bootstrap_mod.format_context_packet(packet) == EXPECTED_BOOTSTRAP_FORMAT_MAPPING


class TestLedgerParity:
    def test_search_payload(self):
        backend = FakeRetrieval(docs=[make_doc(), make_doc("E1", kind="experiment")])
        assert lookup_mod.ledger_search("momentum", kinds="finding", filters={"status": "validated"}, backend=backend) == EXPECTED_LEDGER_SEARCH

    def test_get_payload(self):
        backend = FakeRetrieval(docs=[make_doc(), make_doc("E1", kind="experiment")])
        assert lookup_mod.ledger_get("F17, F404", include_evidence=True, backend=backend) == EXPECTED_LEDGER_GET


class TestTrimEquivalence:
    """The fork-rolled clips now delegate to upstream truncate (Phase 1b)."""

    @pytest.mark.parametrize(
        ("text", "budget"),
        [
            ("", 5),
            ("abc", 5),
            ("abc", 3),
            ("abcdef", 5),
            ("abcdef", 1),
            ("ab cdef", 4),  # rstrip branch: "ab " -> "ab…"
            ("  padded   text  ", 10),
            ("word " * 50, 50),
            ("éclair split mé", 8),
        ],
    )
    def test_clip_snippet_matches_upstream_truncate(self, text, budget):
        assert packet_mod._clip_snippet(text, budget) == truncate(" ".join(text.split()), budget)

    def test_trim_text_matches_upstream_truncate(self):
        assert bootstrap_mod._trim_text("abcdef", 3) == (truncate("abcdef", 3), True)
        assert bootstrap_mod._trim_text("abc", 5) == ("abc", False)
        assert bootstrap_mod._trim_text("abc", 3) == ("abc", False)

    def test_trim_text_preserves_zero_budget_quirk(self):
        # Legacy quirk, pinned: a zero budget still emits the marker
        # (upstream truncate would emit ""); parity over purity.
        assert bootstrap_mod._trim_text("hello world", 0) == ("…", True)
        assert bootstrap_mod._trim_text("hello world", 1) == ("…", True)


class TestUpstreamRouting:
    """The renderers genuinely call the upstream helpers (spies, not just parity)."""

    def test_packet_to_text_routes_through_helpers(self, monkeypatch):
        seen = {}
        orig_format, orig_prune = packet_mod.format_result, packet_mod.prune_sources

        def spy_format(result, **kwargs):
            seen["format_result"] = result
            return orig_format(result, **kwargs)

        def spy_prune(content, artifact):
            seen["prune_sources"] = (content, artifact)
            return orig_prune(content, artifact)

        monkeypatch.setattr(packet_mod, "format_result", spy_format)
        monkeypatch.setattr(packet_mod, "prune_sources", spy_prune)
        packet = packet_mod.build_context_packet(fused_fixture(), open_questions=["Does momentum survive 2026 costs?"], topic_key="momentum")
        assert packet.to_text() == EXPECTED_PACKET_FULL_TEXT
        assert isinstance(seen["format_result"], RetrievalResult)
        assert len(seen["format_result"].hits) == 8  # 7 rows + 1 open question
        pruned_content, artifact = seen["prune_sources"]
        assert artifact["knowledge_sources_version"] == ARTIFACT_VERSION
        assert pruned_content == EXPECTED_PACKET_FULL_TEXT

    def test_bootstrap_format_routes_through_helpers(self, monkeypatch):
        seen = {}
        orig_format, orig_prune = bootstrap_mod.format_result, bootstrap_mod.prune_sources

        def spy_format(result, **kwargs):
            seen["format_result"] = result
            return orig_format(result, **kwargs)

        def spy_prune(content, artifact):
            seen["prune_sources"] = (content, artifact)
            return orig_prune(content, artifact)

        monkeypatch.setattr(bootstrap_mod, "format_result", spy_format)
        monkeypatch.setattr(bootstrap_mod, "prune_sources", spy_prune)
        packet = stable_packet_dict(bootstrap_mod.knowledge_bootstrap("momentum in US equities", stores=make_stores()))
        assert packet["text"] == EXPECTED_BOOTSTRAP_FULL["text"]
        assert isinstance(seen["format_result"], RetrievalResult)
        assert seen["format_result"].hits
        assert seen["prune_sources"][1]["knowledge_sources_version"] == ARTIFACT_VERSION


# ---------------------------------------------------------------------------
# (b) Evidence survival through upstream format_result.
# ---------------------------------------------------------------------------


def assert_sources_survive(text, artifact, hits, provider):
    """Fork-sourced evidence passes through uncut: verbatim excerpts, hashed IDs, attribution."""
    assert artifact["knowledge_sources_version"] == ARTIFACT_VERSION
    sources = artifact["knowledge_sources"]
    assert len(sources) == len(hits)
    for hit, raw in zip(hits, sources):
        source = KnowledgeSource.model_validate(raw)
        assert raw["id"] == source.id  # committed id matches the content hash
        assert raw["provider"] == provider
        assert raw["excerpt"] == hit.content  # verbatim, uncut
        assert raw["excerpt"] in text
        assert f"#knowledge-source-{source.id}" in text  # retained with attribution


class TestPacketEvidenceSurvival:
    def test_packet_rows_survive_uncut(self):
        packet = packet_mod.build_context_packet(fused_fixture(), open_questions=["Does momentum survive 2026 costs?"], topic_key="momentum")
        result = packet_mod.packet_to_retrieval_result(packet)
        assert len(result.hits) == 8
        text, artifact = format_result(result, max_chars_per_chunk=100_000, max_total_chars=1_000_000)
        assert_sources_survive(text, artifact, result.hits, packet_mod.PACKET_EVIDENCE_PROVIDER)

    def test_render_packet_evidence_marks_citations(self):
        packet = packet_mod.build_context_packet(fused_fixture(), topic_key="momentum")
        text, artifact = packet_mod.render_packet_evidence(packet)
        assert "[citation:" in text
        sources = artifact["knowledge_sources"]
        assert sources
        for raw in sources:  # every retained source stays paired with text
            assert raw["excerpt"] in text
            assert f"#knowledge-source-{raw['id']}" in text


class TestLedgerEvidenceSurvival:
    def test_ledger_hits_survive_uncut(self):
        docs = [make_doc(), make_doc("E1", kind="experiment"), make_doc("C31", kind="conflict", status="open")]
        text, artifact = lookup_mod.format_ledger_hits(docs, max_chars_per_chunk=100_000, max_total_chars=1_000_000)
        assert_sources_survive(text, artifact, lookup_mod.ledger_page_to_retrieval_result(docs).hits, lookup_mod.LEDGER_EVIDENCE_PROVIDER)

    def test_source_ids_are_content_hashes(self):
        base = lookup_mod.ledger_doc_to_hit(make_doc())
        twin = lookup_mod.ledger_doc_to_hit(make_doc())
        assert base is not None and twin is not None
        assert base.source is not None and twin.source is not None
        assert base.source.id == twin.source.id  # deterministic
        assert len(base.source.id) == 32
        altered = lookup_mod.ledger_doc_to_hit(make_doc(summary="Summary F17, revised"))
        assert altered is not None and altered.source is not None
        assert altered.source.id != base.source.id  # content change -> id change
        retitled = lookup_mod.ledger_doc_to_hit(make_doc(title="Other title"))
        assert retitled is not None and retitled.source is not None
        assert retitled.source.id != base.source.id

    def test_truncation_drops_whole_hits_paired(self):
        docs = [make_doc(f"F{i:03d}", summary="evidence " * 40) for i in range(6)]
        text, artifact = lookup_mod.format_ledger_hits(docs, max_total_chars=600)
        assert TRUNCATION_MARKER in text
        retained = artifact["knowledge_sources"]
        assert 0 < len(retained) < len(docs)
        for raw in retained:
            assert raw["excerpt"] in text
            assert f"#knowledge-source-{raw['id']}" in text
        dropped = lookup_mod.ledger_page_to_retrieval_result(docs).hits[len(retained) :]
        for hit in dropped:
            assert hit.source is not None
            assert f"#knowledge-source-{hit.source.id}" not in text

    def test_prune_keeps_citations_paired_after_cut(self):
        docs = [make_doc(), make_doc("E1", kind="experiment")]
        text, artifact = lookup_mod.format_ledger_hits(docs, max_chars_per_chunk=100_000, max_total_chars=1_000_000)
        cut = text[: len(text) // 2]  # simulate a downstream truncation
        pruned, pruned_artifact = prune_sources(cut, artifact)
        for raw in pruned_artifact["knowledge_sources"]:
            assert raw["excerpt"] in pruned
            assert f"#knowledge-source-{raw['id']}" in pruned
        assert len(pruned_artifact["knowledge_sources"]) <= len(artifact["knowledge_sources"])

    def test_prune_is_identity_on_marker_free_text(self):
        docs = [make_doc()]
        text, _ = lookup_mod.prune_ledger_text("plain ledger line, no markers", docs)
        assert text == "plain ledger line, no markers"


class TestBootstrapEvidenceSurvival:
    def test_packet_entries_survive_uncut(self):
        packet = stable_packet_dict(bootstrap_mod.knowledge_bootstrap("momentum in US equities", stores=make_stores()))
        result = bootstrap_mod.packet_sections_to_retrieval_result(packet["sections"])
        assert result.hits
        text, artifact = format_result(result, max_chars_per_chunk=100_000, max_total_chars=1_000_000)
        assert_sources_survive(text, artifact, result.hits, bootstrap_mod.BOOTSTRAP_EVIDENCE_PROVIDER)

    def test_truncated_entries_leave_no_orphan_evidence(self):
        docs = [make_doc(f"F{i:03d}", summary="s" * 200) for i in range(10)]
        stores = make_stores(retrieval=FakeRetrieval(docs=docs))
        packet = stable_packet_dict(bootstrap_mod.knowledge_bootstrap("momentum", stores=stores, budget=bootstrap_mod.PacketBudget(max_items=3, max_chars=500, channel_limit=5)))
        assert packet["truncated"] is True
        result = bootstrap_mod.packet_sections_to_retrieval_result(packet["sections"])
        text, artifact = format_result(result, max_chars_per_chunk=100_000, max_total_chars=1_000_000)
        assert_sources_survive(text, artifact, result.hits, bootstrap_mod.BOOTSTRAP_EVIDENCE_PROVIDER)

    def test_empty_entry_yields_no_hit(self):
        assert bootstrap_mod.packet_entry_to_hit({}) is None
        assert bootstrap_mod.packet_entry_to_hit({"title": "", "summary": ""}) is None
        bare = bootstrap_mod.packet_entry_to_hit({"id": "x", "title": "", "summary": ""})
        assert bare is not None and bare.content == "x"  # id fallback, never a silent drop
        hit = bootstrap_mod.packet_entry_to_hit({"id": "x", "title": "t", "summary": ""})
        assert hit is not None and hit.content == "t"


class TestEvidenceJsonRoundTrip:
    def test_artifact_survives_json_serialization(self):
        docs = [make_doc(), make_doc("E1", kind="experiment")]
        text, artifact = lookup_mod.format_ledger_hits(docs, max_chars_per_chunk=100_000, max_total_chars=1_000_000)
        restored = json.loads(json.dumps(artifact))
        assert restored == artifact
        for raw in restored["knowledge_sources"]:
            assert KnowledgeSource.model_validate(raw).id == raw["id"]
            assert raw["excerpt"] in text

"""Unit + opt-in live tests for the Muse intent parser (Phase 4b).

Unmarked tests always run (offline, ``FakeLLMTextClient``): shape
validation, fallback on client errors/garbage/shape violations, and
flag-off keyword behavior. Live tests (``pytest.mark.live``) need
``DEER_FLOW_RUN_LIVE_TESTS=1`` plus a working ``muse`` CLI and assert
shape validity + smoke on three fixed prompts; keyword parity is
reported, not asserted.
"""

from __future__ import annotations

import json
import logging
import os

import pytest

from app.gateway.knowledge_llm_client import (
    CliValidationError,
    MuseExecError,
    MuseIntentClient,
    extract_exec_text,
)
from deerflow.knowledge.agent_tools.bootstrap import (
    ASSET_CLASSES,
    DEFAULT_NEEDED_MEMORY,
    NEEDED_MEMORY_VOCABULARY,
    ResearchIntent,
    _intent_from_text_keyword,
    bind_intent_llm_client,
    get_intent_llm_client,
    intent_from_text,
)
from deerflow.knowledge.config import KnowledgeConfig, set_knowledge_config
from deerflow.knowledge.llm_text_client import FakeLLMTextClient
from deerflow.knowledge.write_api import KnowledgeValidationError

TASK_EQUITY = "momentum backtest on US equities with daily bars"
VALID_INTENT_JSON = json.dumps(
    {
        "topic": "momentum backtest on US equities with daily bars",
        "asset_class": "equity",
        "markets": ["US"],
        "universe": "liquid common stocks",
        "horizon": None,
        "frequency": "daily",
        "concepts": ["momentum", "backtest"],
        "requested_period": None,
        "needed_memory": ["prior_experiments", "failures"],
    }
)

LIVE_PROMPTS = (
    "momentum backtest on US equities with daily bars from 2020-01-01 to 2024-12-31",
    "carry strategy in FX majors on weekly data",
    "intraday reversal in crypto perpetuals",
)


@pytest.fixture(autouse=True)
def _clean_intent_state(monkeypatch):
    """Isolate intent-parser config/env/binding per test (no cross-test leaks)."""
    monkeypatch.delenv("DEER_FLOW_KNOWLEDGE_INTENT_PARSER", raising=False)
    set_knowledge_config(KnowledgeConfig())
    bind_intent_llm_client(None)
    yield
    bind_intent_llm_client(None)
    set_knowledge_config(KnowledgeConfig())


def _use_muse() -> None:
    set_knowledge_config(KnowledgeConfig(intent_parser="muse"))


def _assert_valid_shape(intent: ResearchIntent) -> None:
    assert isinstance(intent, ResearchIntent)
    assert isinstance(intent.topic, str) and intent.topic.strip()
    assert intent.asset_class in ASSET_CLASSES
    assert isinstance(intent.markets, list)
    assert set(intent.markets) <= {"US", "EU", "UK", "JP", "CN", "EM", "GL"}
    assert intent.universe is None or isinstance(intent.universe, str)
    assert intent.horizon is None or isinstance(intent.horizon, str)
    assert intent.frequency is None or isinstance(intent.frequency, str)
    assert isinstance(intent.concepts, list) and all(isinstance(c, str) and c for c in intent.concepts)
    if intent.requested_period is not None:
        assert len(intent.requested_period) == 2
        assert all(isinstance(d, str) for d in intent.requested_period)
        assert intent.requested_period[0] <= intent.requested_period[1]
    assert isinstance(intent.needed_memory, list) and intent.needed_memory
    assert set(intent.needed_memory) <= set(NEEDED_MEMORY_VOCABULARY)


# ---------------------------------------------------------------------------
# Flag / binding selection
# ---------------------------------------------------------------------------


def test_flag_off_uses_keyword_without_calling_llm() -> None:
    fake = FakeLLMTextClient([VALID_INTENT_JSON])
    bind_intent_llm_client(fake)
    assert intent_from_text(TASK_EQUITY).to_dict() == _intent_from_text_keyword(TASK_EQUITY).to_dict()
    assert fake.calls == []


def test_unbound_client_uses_keyword_even_when_flag_on() -> None:
    _use_muse()
    assert get_intent_llm_client() is None
    assert intent_from_text(TASK_EQUITY).to_dict() == _intent_from_text_keyword(TASK_EQUITY).to_dict()


def test_bind_unbind_roundtrip() -> None:
    fake = FakeLLMTextClient()
    assert bind_intent_llm_client(fake) is fake
    assert get_intent_llm_client() is fake
    assert bind_intent_llm_client(None) is None
    assert get_intent_llm_client() is None


def test_env_override_selects_muse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_INTENT_PARSER", "muse")
    fake = FakeLLMTextClient([VALID_INTENT_JSON])
    bind_intent_llm_client(fake)
    intent = intent_from_text(TASK_EQUITY)
    assert len(fake.calls) == 1
    assert intent.needed_memory == ["prior_experiments", "failures"]


def test_invalid_env_value_fails_closed_to_keyword(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_INTENT_PARSER", "bogus")
    fake = FakeLLMTextClient([VALID_INTENT_JSON])
    bind_intent_llm_client(fake)
    with caplog.at_level(logging.WARNING, logger="deerflow.knowledge.agent_tools.bootstrap"):
        intent = intent_from_text(TASK_EQUITY)
    assert intent.to_dict() == _intent_from_text_keyword(TASK_EQUITY).to_dict()
    assert fake.calls == []


def test_get_intent_parser_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    assert KnowledgeConfig().get_intent_parser() == "keyword"
    assert KnowledgeConfig(intent_parser="muse").get_intent_parser() == "muse"
    assert KnowledgeConfig(intent_parser=" MUSE ").get_intent_parser() == "muse"
    monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_INTENT_PARSER", "muse")
    assert KnowledgeConfig().get_intent_parser() == "muse"
    monkeypatch.setenv("DEER_FLOW_KNOWLEDGE_INTENT_PARSER", "bogus")
    with pytest.raises(ValueError, match="intent_parser"):
        KnowledgeConfig().get_intent_parser()
    with pytest.raises(ValueError, match="intent_parser"):
        KnowledgeConfig(intent_parser="bogus").get_intent_parser()


# ---------------------------------------------------------------------------
# LLM happy path + prompt wiring
# ---------------------------------------------------------------------------


def test_muse_valid_json_returns_llm_values() -> None:
    _use_muse()
    bind_intent_llm_client(FakeLLMTextClient([VALID_INTENT_JSON]))
    intent = intent_from_text("something entirely different about bonds")
    _assert_valid_shape(intent)
    assert intent.asset_class == "equity"
    assert intent.markets == ["US"]
    assert intent.frequency == "daily"
    assert intent.concepts == ["momentum", "backtest"]
    assert intent.needed_memory == ["prior_experiments", "failures"]


def test_muse_fenced_json_accepted() -> None:
    _use_muse()
    bind_intent_llm_client(FakeLLMTextClient(["```json\n" + VALID_INTENT_JSON + "\n```"]))
    intent = intent_from_text(TASK_EQUITY)
    assert intent.asset_class == "equity"
    assert intent.concepts == ["momentum", "backtest"]


def test_muse_prompt_carries_task_text() -> None:
    _use_muse()
    fake = FakeLLMTextClient([VALID_INTENT_JSON])
    bind_intent_llm_client(fake)
    intent_from_text(TASK_EQUITY)
    assert len(fake.calls) == 1
    assert TASK_EQUITY in fake.calls[0]["prompt"]
    assert "asset_class" in fake.calls[0]["prompt"]


def test_caller_needed_memory_overrides_model() -> None:
    _use_muse()
    bind_intent_llm_client(FakeLLMTextClient([VALID_INTENT_JSON]))
    intent = intent_from_text(TASK_EQUITY, needed_memory=["assumptions"])
    assert intent.needed_memory == ["assumptions"]


def test_muse_defaults_needed_memory_when_model_omits() -> None:
    _use_muse()
    bind_intent_llm_client(FakeLLMTextClient([json.dumps({"topic": TASK_EQUITY})]))
    intent = intent_from_text(TASK_EQUITY)
    assert intent.needed_memory == list(DEFAULT_NEEDED_MEMORY)


# ---------------------------------------------------------------------------
# Fallback on ANY failure
# ---------------------------------------------------------------------------


def test_fallback_on_client_error(caplog: pytest.LogCaptureFixture) -> None:
    _use_muse()
    bind_intent_llm_client(FakeLLMTextClient([RuntimeError("boom")]))
    with caplog.at_level(logging.WARNING, logger="deerflow.knowledge.agent_tools.bootstrap"):
        intent = intent_from_text(TASK_EQUITY)
    assert intent.to_dict() == _intent_from_text_keyword(TASK_EQUITY).to_dict()
    assert "falling back" in caplog.text


def test_fallback_on_garbage_output() -> None:
    _use_muse()
    bind_intent_llm_client(FakeLLMTextClient(["definitely not json {{{"]))
    assert intent_from_text(TASK_EQUITY).to_dict() == _intent_from_text_keyword(TASK_EQUITY).to_dict()


@pytest.mark.parametrize(
    "bad_json",
    [
        json.dumps({"topic": TASK_EQUITY, "asset_class": "plutonium"}),
        json.dumps({"topic": TASK_EQUITY, "markets": ["XX"]}),
        json.dumps({"topic": TASK_EQUITY, "bogus_key": 1}),
        json.dumps({"topic": TASK_EQUITY, "requested_period": ["2024-01-01"]}),
        json.dumps(["not", "an", "object"]),
        json.dumps({"no_topic": True}),
    ],
)
def test_fallback_on_shape_violation(bad_json: str) -> None:
    _use_muse()
    bind_intent_llm_client(FakeLLMTextClient([bad_json]))
    assert intent_from_text(TASK_EQUITY).to_dict() == _intent_from_text_keyword(TASK_EQUITY).to_dict()


def test_invalid_input_raises_without_llm_call() -> None:
    _use_muse()
    fake = FakeLLMTextClient([VALID_INTENT_JSON])
    bind_intent_llm_client(fake)
    with pytest.raises(KnowledgeValidationError, match="task must be a string"):
        intent_from_text(123)  # type: ignore[arg-type]
    assert fake.calls == []


def test_empty_task_raises_on_both_paths() -> None:
    with pytest.raises(KnowledgeValidationError, match="non-empty"):
        intent_from_text("   ")
    _use_muse()
    bind_intent_llm_client(FakeLLMTextClient([VALID_INTENT_JSON]))
    with pytest.raises(KnowledgeValidationError, match="non-empty"):
        intent_from_text("   ")


# ---------------------------------------------------------------------------
# Fake client + exec-output parser (offline, deterministic)
# ---------------------------------------------------------------------------


def test_fake_client_replays_and_records() -> None:
    fake = FakeLLMTextClient(["first", RuntimeError("second fails")], default_response="dflt")
    assert fake.complete("p1") == "first"
    with pytest.raises(RuntimeError, match="second fails"):
        fake.complete("p2", timeout=5.0, max_chars=10)
    assert fake.complete("p3") == "dflt"
    assert [call["prompt"] for call in fake.calls] == ["p1", "p2", "p3"]
    assert fake.calls[1]["timeout"] == 5.0
    assert isinstance(fake, FakeLLMTextClient)


def test_fake_client_truncates_and_validates() -> None:
    assert FakeLLMTextClient(["abcdef"]).complete("p", max_chars=3) == "abc"
    with pytest.raises(ValueError, match="non-empty"):
        FakeLLMTextClient().complete("   ")


@pytest.mark.asyncio
async def test_fake_client_async_mirror() -> None:
    fake = FakeLLMTextClient(["hello"])
    assert await fake.acomplete("p") == "hello"
    assert len(fake.calls) == 1


def _delta(text: str) -> str:
    return json.dumps({"payload_type": "run.output.delta", "payload": {"kind": "run_output_delta", "text": text}})


def _terminal(text: str) -> str:
    return json.dumps({"payload_type": "run.terminal.completed", "payload": {"kind": "run_terminal", "terminal": "completed", "text": text}})


def test_extract_exec_text_joins_deltas() -> None:
    assert extract_exec_text(_delta("echo: ") + "\n" + _delta("hi") + "\n") == "echo: hi"


def test_extract_exec_text_prefers_terminal() -> None:
    stdout = "\n".join([_delta("partial"), _terminal("full text"), '{"payload_type":"run.usage","payload":{}}'])
    assert extract_exec_text(stdout) == "full text"


def test_extract_exec_text_skips_noise_and_rejects_empty() -> None:
    assert extract_exec_text("\nnot json\n" + _delta("ok") + "\n[1,2]\n") == "ok"
    with pytest.raises(MuseExecError, match="no output text"):
        extract_exec_text("")
    with pytest.raises(MuseExecError, match="no output text"):
        extract_exec_text('{"payload_type":"run.usage","payload":{}}\n')


def test_extract_exec_text_rejects_failed_terminal() -> None:
    failed = json.dumps({"payload_type": "run.terminal.failed", "payload": {"kind": "run_terminal", "terminal": "failed", "reason": "killed"}})
    with pytest.raises(MuseExecError, match="did not complete"):
        extract_exec_text(_delta("partial") + "\n" + failed + "\n")


def test_muse_client_rejects_bad_input_without_spawning() -> None:
    client = MuseIntentClient()
    with pytest.raises(CliValidationError, match="prompt must not be empty"):
        client.complete("   ")
    with pytest.raises(ValueError, match="timeout"):
        client.complete("hi", timeout=0)
    with pytest.raises(ValueError, match="max_chars"):
        client.complete("hi", max_chars=0)


def test_muse_client_pins_provider_and_model() -> None:
    client = MuseIntentClient()
    assert client.provider == "meta"
    assert client.model == "muse-spark-1.3"
    params = client._launch_params("hello")
    assert params.provider == "meta"
    assert params.model == "muse-spark-1.3"


# ---------------------------------------------------------------------------
# Live tests (opt-in): real `muse exec` parser
# ---------------------------------------------------------------------------


def _require_live() -> None:
    if os.getenv("DEER_FLOW_RUN_LIVE_TESTS") != "1":
        pytest.skip("live knowledge checks are opt-in")


def _parity_report(task: str, keyword: ResearchIntent, llm: ResearchIntent) -> str:
    keyword_concepts, llm_concepts = set(keyword.concepts), set(llm.concepts)
    overlap = sorted(keyword_concepts & llm_concepts)
    return (
        f"task={task!r}\n"
        f"  asset_class: keyword={keyword.asset_class} llm={llm.asset_class} "
        f"match={keyword.asset_class == llm.asset_class}\n"
        f"  markets: keyword={keyword.markets} llm={llm.markets} "
        f"match={keyword.markets == llm.markets}\n"
        f"  concepts overlap: {overlap} "
        f"(keyword={sorted(keyword_concepts)} llm={sorted(llm_concepts)})"
    )


@pytest.mark.live
def test_live_muse_client_smoke() -> None:
    _require_live()
    client = MuseIntentClient(default_timeout=300.0)
    text = client.complete("Reply with exactly this token and nothing else: smoke-ok", max_chars=64)
    assert isinstance(text, str) and text.strip(), "live muse exec must return non-empty text"
    assert len(text) <= 64


@pytest.mark.live
def test_live_muse_parser_shape_on_fixed_prompts() -> None:
    _require_live()
    keyword_intents = [_intent_from_text_keyword(task) for task in LIVE_PROMPTS]
    _use_muse()
    bind_intent_llm_client(MuseIntentClient(default_timeout=300.0))
    try:
        for task, keyword in zip(LIVE_PROMPTS, keyword_intents, strict=True):
            intent = intent_from_text(task)
            _assert_valid_shape(intent)
            assert intent.topic.strip(), "LLM intent topic must be non-empty"
            print(_parity_report(task, keyword, intent))
    finally:
        bind_intent_llm_client(None)

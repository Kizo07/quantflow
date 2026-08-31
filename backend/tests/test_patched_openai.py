"""Tests for deerflow.models.patched_openai.PatchedChatOpenAI.

These tests verify that _restore_tool_call_signatures correctly re-injects
``thought_signature`` onto tool-call objects stored in
``additional_kwargs["tool_calls"]``, covering id-based matching, positional
fallback, camelCase keys, and several edge-cases.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

from deerflow.models.patched_openai import StreamingPatchedChatOpenAI, _restore_tool_call_signatures

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RAW_TC_SIGNED = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "web_fetch", "arguments": '{"url":"http://example.com"}'},
    "thought_signature": "SIG_A==",
}

RAW_TC_UNSIGNED = {
    "id": "call_2",
    "type": "function",
    "function": {"name": "bash", "arguments": '{"cmd":"ls"}'},
}

PAYLOAD_TC_1 = {
    "type": "function",
    "id": "call_1",
    "function": {"name": "web_fetch", "arguments": '{"url":"http://example.com"}'},
}

PAYLOAD_TC_2 = {
    "type": "function",
    "id": "call_2",
    "function": {"name": "bash", "arguments": '{"cmd":"ls"}'},
}


def _ai_msg_with_raw_tool_calls(raw_tool_calls: list[dict]) -> AIMessage:
    return AIMessage(content="", additional_kwargs={"tool_calls": raw_tool_calls})


# ---------------------------------------------------------------------------
# Core: signed tool-call restoration
# ---------------------------------------------------------------------------


def test_tool_call_signature_restored_by_id():
    """thought_signature is copied to the payload tool-call matched by id."""
    payload_msg = {"role": "assistant", "content": None, "tool_calls": [PAYLOAD_TC_1.copy()]}
    orig = _ai_msg_with_raw_tool_calls([RAW_TC_SIGNED])

    _restore_tool_call_signatures(payload_msg, orig)

    assert payload_msg["tool_calls"][0]["thought_signature"] == "SIG_A=="


def test_tool_call_signature_for_parallel_calls():
    """For parallel function calls, only the first has a signature (per Gemini spec)."""
    payload_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [PAYLOAD_TC_1.copy(), PAYLOAD_TC_2.copy()],
    }
    orig = _ai_msg_with_raw_tool_calls([RAW_TC_SIGNED, RAW_TC_UNSIGNED])

    _restore_tool_call_signatures(payload_msg, orig)

    assert payload_msg["tool_calls"][0]["thought_signature"] == "SIG_A=="
    assert "thought_signature" not in payload_msg["tool_calls"][1]


def test_tool_call_signature_camel_case():
    """thoughtSignature (camelCase) from some gateways is also handled."""
    raw_camel = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "web_fetch", "arguments": "{}"},
        "thoughtSignature": "SIG_CAMEL==",
    }
    payload_msg = {"role": "assistant", "content": None, "tool_calls": [PAYLOAD_TC_1.copy()]}
    orig = _ai_msg_with_raw_tool_calls([raw_camel])

    _restore_tool_call_signatures(payload_msg, orig)

    assert payload_msg["tool_calls"][0]["thought_signature"] == "SIG_CAMEL=="


def test_tool_call_signature_positional_fallback():
    """When ids don't match, falls back to positional matching."""
    raw_no_id = {
        "type": "function",
        "function": {"name": "web_fetch", "arguments": "{}"},
        "thought_signature": "SIG_POS==",
    }
    payload_tc = {
        "type": "function",
        "id": "call_99",
        "function": {"name": "web_fetch", "arguments": "{}"},
    }
    payload_msg = {"role": "assistant", "content": None, "tool_calls": [payload_tc]}
    orig = _ai_msg_with_raw_tool_calls([raw_no_id])

    _restore_tool_call_signatures(payload_msg, orig)

    assert payload_tc["thought_signature"] == "SIG_POS=="


# ---------------------------------------------------------------------------
# Edge cases: no-op scenarios for tool-call signatures
# ---------------------------------------------------------------------------


def test_tool_call_no_raw_tool_calls_is_noop():
    """No change when additional_kwargs has no tool_calls."""
    payload_msg = {"role": "assistant", "content": None, "tool_calls": [PAYLOAD_TC_1.copy()]}
    orig = AIMessage(content="", additional_kwargs={})

    _restore_tool_call_signatures(payload_msg, orig)

    assert "thought_signature" not in payload_msg["tool_calls"][0]


def test_tool_call_no_payload_tool_calls_is_noop():
    """No change when payload has no tool_calls."""
    payload_msg = {"role": "assistant", "content": "just text"}
    orig = _ai_msg_with_raw_tool_calls([RAW_TC_SIGNED])

    _restore_tool_call_signatures(payload_msg, orig)

    assert "tool_calls" not in payload_msg


def test_tool_call_unsigned_raw_entries_is_noop():
    """No signature added when raw tool-calls have no thought_signature."""
    payload_msg = {"role": "assistant", "content": None, "tool_calls": [PAYLOAD_TC_2.copy()]}
    orig = _ai_msg_with_raw_tool_calls([RAW_TC_UNSIGNED])

    _restore_tool_call_signatures(payload_msg, orig)

    assert "thought_signature" not in payload_msg["tool_calls"][0]


def test_tool_call_multiple_sequential_signatures():
    """Sequential tool calls each carry their own signature."""
    raw_tc_a = {
        "id": "call_a",
        "type": "function",
        "function": {"name": "check_flight", "arguments": "{}"},
        "thought_signature": "SIG_STEP1==",
    }
    raw_tc_b = {
        "id": "call_b",
        "type": "function",
        "function": {"name": "book_taxi", "arguments": "{}"},
        "thought_signature": "SIG_STEP2==",
    }
    payload_tc_a = {"type": "function", "id": "call_a", "function": {"name": "check_flight", "arguments": "{}"}}
    payload_tc_b = {"type": "function", "id": "call_b", "function": {"name": "book_taxi", "arguments": "{}"}}
    payload_msg = {"role": "assistant", "content": None, "tool_calls": [payload_tc_a, payload_tc_b]}
    orig = _ai_msg_with_raw_tool_calls([raw_tc_a, raw_tc_b])

    _restore_tool_call_signatures(payload_msg, orig)

    assert payload_tc_a["thought_signature"] == "SIG_STEP1=="
    assert payload_tc_b["thought_signature"] == "SIG_STEP2=="


# Integration behavior for PatchedChatOpenAI is validated indirectly via
# _restore_tool_call_signatures unit coverage above.


# ---------------------------------------------------------------------------
# StreamingPatchedChatOpenAI — always-stream behavior (Token Plan fix)
# ---------------------------------------------------------------------------


def _make_streaming_model() -> StreamingPatchedChatOpenAI:
    return StreamingPatchedChatOpenAI(
        model="test-model",
        api_key="***",
        base_url="http://127.0.0.1:9/v1",
    )


def _patch_stream(monkeypatch, counter: dict):
    """Replace _astream with a fake matching the real contract: it yields
    ChatGenerationChunk objects (langchain-core >= 1.x)."""

    async def fake_astream(self, messages, stop=None, run_manager=None, **kwargs):
        counter["stream"] += 1
        yield ChatGenerationChunk(message=AIMessageChunk(content="ok"))

    monkeypatch.setattr(StreamingPatchedChatOpenAI, "_astream", fake_astream)


def _guard_parent_generate(monkeypatch):
    """Make any regression to the non-streaming parent path fail loudly."""
    from langchain_openai import ChatOpenAI

    from deerflow.models import patched_openai as po

    def boom(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("regressed to non-streaming ChatOpenAI._generate — the exact request shape the Token Plan endpoint drops after ~60s")

    monkeypatch.setattr(ChatOpenAI, "_generate", boom)
    return po


def test_streaming_patched_agenerate_uses_stream(monkeypatch):
    """_agenerate must assemble the result from _astream, never a plain POST."""
    counter = {"stream": 0}
    _patch_stream(monkeypatch, counter)
    _guard_parent_generate(monkeypatch)

    model = _make_streaming_model()
    result = asyncio.run(model._agenerate([HumanMessage(content="hi")]))

    assert counter["stream"] == 1
    assert isinstance(result.generations[0].message, AIMessage)
    assert result.generations[0].message.content == "ok"


def test_streaming_patched_generate_outside_loop_uses_stream(monkeypatch):
    """Sync call with no running loop must still stream internally."""
    counter = {"stream": 0}
    _patch_stream(monkeypatch, counter)
    _guard_parent_generate(monkeypatch)

    model = _make_streaming_model()
    result = model._generate([HumanMessage(content="hi")])

    assert counter["stream"] == 1
    assert result.generations[0].message.content == "ok"


def test_streaming_patched_generate_inside_loop_still_streams(monkeypatch):
    """Sync call from inside a running loop must NOT fall back to the
    non-streaming parent POST; it must keep streaming via a worker loop."""
    counter = {"stream": 0}
    _patch_stream(monkeypatch, counter)
    _guard_parent_generate(monkeypatch)

    model = _make_streaming_model()

    async def driver():
        # Called synchronously while an event loop IS running (happens when
        # sync LangChain helpers are invoked inside async graph code).
        return model._generate([HumanMessage(content="hi")])

    result = asyncio.run(driver())

    assert counter["stream"] == 1
    assert result.generations[0].message.content == "ok"


def test_streaming_patched_installs_keepalive_recycling_clients():
    """Long-lived gateway processes must recycle pooled connections before
    NAT/LB idle cutoffs turn them into dead sockets (the midnight
    RemoteProtocolError storm): keepalive_expiry <= 30s, retries enabled."""
    model = _make_streaming_model()

    assert model.http_client is not None, "sync http_client must be installed"
    assert model.http_async_client is not None, "async http_async_client must be installed"

    sync_expiry = model.http_client._transport._pool._keepalive_expiry
    async_expiry = model.http_async_client._transport._pool._keepalive_expiry
    assert sync_expiry is not None and sync_expiry <= 30.0
    assert async_expiry is not None and async_expiry <= 30.0

    assert model.max_retries >= 2


def test_streaming_patched_does_not_override_user_http_clients():
    """Caller-supplied clients win — the recycling default is a setdefault."""
    import httpx

    custom = httpx.Client()
    custom_async = httpx.AsyncClient()
    model = StreamingPatchedChatOpenAI(
        model="test-model",
        api_key="***",
        base_url="http://127.0.0.1:9/v1",
        http_client=custom,
        http_async_client=custom_async,
    )
    assert model.http_client is custom
    assert model.http_async_client is custom_async
    custom.close()
    asyncio.run(custom_async.aclose())

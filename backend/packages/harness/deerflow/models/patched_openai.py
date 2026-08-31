"""Patched ChatOpenAI that preserves thought_signature for Gemini thinking models.

When using Gemini with thinking enabled via an OpenAI-compatible gateway (e.g.
Vertex AI, Google AI Studio, or any proxy), the API requires that the
``thought_signature`` field on tool-call objects is echoed back verbatim in
every subsequent request.

The OpenAI-compatible gateway stores the raw tool-call dicts (including
``thought_signature``) in ``additional_kwargs["tool_calls"]``, but standard
``langchain_openai.ChatOpenAI`` only serialises the standard fields (``id``,
``type``, ``function``) into the outgoing payload, silently dropping the
signature.  That causes an HTTP 400 ``INVALID_ARGUMENT`` error:

    Unable to submit request because function call `<tool>` in the N. content
    block is missing a `thought_signature`.

This module fixes the problem by overriding ``_get_request_payload`` to
re-inject tool-call signatures back into the outgoing payload for any assistant
message that originally carried them.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import agenerate_from_stream
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from deerflow.models.assistant_payload_replay import restore_assistant_payloads


class PatchedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with ``thought_signature`` preservation for Gemini thinking via OpenAI gateway.

    When using Gemini with thinking enabled via an OpenAI-compatible gateway,
    the API expects ``thought_signature`` to be present on tool-call objects in
    multi-turn conversations.  This patched version restores those signatures
    from ``AIMessage.additional_kwargs["tool_calls"]`` into the serialised
    request payload before it is sent to the API.

    Usage in ``config.yaml``::

        - name: gemini-2.5-pro-thinking
          display_name: Gemini 2.5 Pro (Thinking)
          use: deerflow.models.patched_openai:PatchedChatOpenAI
          model: google/gemini-2.5-pro-preview
          api_key: $GEMINI_API_KEY
          base_url: https://<your-openai-compat-gateway>/v1
          max_tokens: 16384
          supports_thinking: true
          supports_vision: true
          when_thinking_enabled:
            extra_body:
              thinking:
                type: enabled
    """

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get request payload with ``thought_signature`` preserved on tool-call objects.

        Overrides the parent method to re-inject ``thought_signature`` fields
        on tool-call objects that were stored in
        ``additional_kwargs["tool_calls"]`` by LangChain but dropped during
        serialisation.
        """
        # Capture the original LangChain messages *before* conversion so we can
        # access fields that the serialiser might drop.
        original_messages = self._convert_input(input_).to_messages()

        # Obtain the base payload from the parent implementation.
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        restore_assistant_payloads(payload.get("messages", []), original_messages, _restore_tool_call_signatures)

        return payload


class StreamingPatchedChatOpenAI(PatchedChatOpenAI):
    """PatchedChatOpenAI that always talks to the provider over SSE internally.

    Some OpenAI-compatible endpoints (observed: Alibaba Cloud Token Plan,
    ``token-plan.*.maas.aliyuncs.com/compatible-mode/v1``) silently drop
    non-streaming requests whose first response byte takes longer than a
    server-side budget (~60s): the connection is closed before any status
    line, surfacing as ``httpx.RemoteProtocolError: Server disconnected
    without sending a response`` / ``openai.APIConnectionError``.  The
    identical payload with ``stream=true`` gets a first chunk in seconds and
    completes fine, so the failure is the endpoint's non-stream idle budget,
    not the request content.

    LangChain agents call the model through ``ainvoke``/``generate`` (no
    streaming).  This subclass converts those into an internal SSE stream and
    re-assembles the final message, so every call gets a first byte quickly
    and the endpoint never sees a non-streaming request.  ``stream_usage``
    defaults on so token accounting keeps working (langchain-openai only
    emits ``usage`` on the final chunk when this is set).
    """

    def __init__(self, **kwargs: Any) -> None:
        import httpx

        kwargs.setdefault("stream_usage", True)
        # The endpoint retries nothing for us on its own: enable SDK-level
        # retries so a recycled-but-racy connection gets one more attempt.
        kwargs.setdefault("max_retries", 3)
        # Recycle pooled keep-alive connections every 30 seconds.
        #
        # Long-lived gateway processes otherwise hold pooled sockets past
        # NAT/LB/server idle cutoffs; the next POST on such a socket dies
        # with ``httpx.RemoteProtocolError: Server disconnected without
        # sending a response`` and — POSTs being non-idempotent — httpx will
        # NOT transparently retry it. That failure mode took down every LLM
        # call in the gateway after ~5h of uptime until a process restart.
        # With a 30s keep-alive expiry a pooled connection is never old
        # enough to be dead server-side. Caller-supplied clients win
        # (setdefault), so custom transports/proxies stay in control.
        limits = httpx.Limits(max_keepalive_connections=5, keepalive_expiry=30.0)
        timeout = httpx.Timeout(600.0, connect=10.0)
        kwargs.setdefault("http_client", httpx.Client(limits=limits, timeout=timeout))
        kwargs.setdefault("http_async_client", httpx.AsyncClient(limits=limits, timeout=timeout))
        super().__init__(**kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await agenerate_from_stream(self._astream(messages, stop=stop, run_manager=run_manager, **kwargs))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Sync path: bridge through an event loop so in-process/sync callers
        # (tests, scripts) get the same protection instead of silently
        # regressing to the endpoint-hostile non-streaming request.  The
        # gateway's hot path is async (_agenerate above).
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs))
        # Already inside a loop but called synchronously. We must NOT fall
        # back to the parent's non-streaming request here — that is the exact
        # request shape the endpoint drops after ~60s of first-byte silence.
        # Run the streaming coroutine on a dedicated worker thread with its
        # own event loop instead (cheap: this path is rare; the gateway hot
        # path is async via _agenerate above).
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="streaming-llm") as pool:
            return pool.submit(
                asyncio.run,
                self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs),
            ).result()


def _restore_tool_call_signatures(payload_msg: dict, orig_msg: AIMessage) -> None:
    """Re-inject ``thought_signature`` onto tool-call objects in *payload_msg*.

    When the Gemini OpenAI-compatible gateway returns a response with function
    calls, each tool-call object may carry a ``thought_signature``.  LangChain
    stores the raw tool-call dicts in ``additional_kwargs["tool_calls"]`` but
    only serialises the standard fields (``id``, ``type``, ``function``) into
    the outgoing payload, silently dropping the signature.

    This function matches raw tool-call entries (by ``id``, falling back to
    positional order) and copies the signature back onto the serialised
    payload entries.
    """
    raw_tool_calls: list[dict] = orig_msg.additional_kwargs.get("tool_calls") or []
    payload_tool_calls: list[dict] = payload_msg.get("tool_calls") or []

    if not raw_tool_calls or not payload_tool_calls:
        return

    # Build an id → raw_tc lookup for efficient matching.
    raw_by_id: dict[str, dict] = {}
    for raw_tc in raw_tool_calls:
        tc_id = raw_tc.get("id")
        if tc_id:
            raw_by_id[tc_id] = raw_tc

    for idx, payload_tc in enumerate(payload_tool_calls):
        # Try matching by id first, then fall back to positional.
        raw_tc = raw_by_id.get(payload_tc.get("id", ""))
        if raw_tc is None and idx < len(raw_tool_calls):
            raw_tc = raw_tool_calls[idx]

        if raw_tc is None:
            continue

        # The gateway may use either snake_case or camelCase.
        sig = raw_tc.get("thought_signature") or raw_tc.get("thoughtSignature")
        if sig:
            payload_tc["thought_signature"] = sig

import { describe, expect, it } from "@rstest/core";

import {
  appendCliChatFrame,
  buildCliChatSendPayload,
  CLI_CHAT_INHERIT,
  CLI_CHAT_NONE,
  mergeCliChatMessages,
  parseCliChatMessageFrame,
  parseCliChatToolFrame,
  resolveCliChatYolo,
} from "@/core/cli-sessions/chat";
import type { CliChatMessage } from "@/core/cli-sessions/types";

describe("cli chat frame parsers", () => {
  it("parses assistant message deltas", () => {
    expect(
      parseCliChatMessageFrame(
        JSON.stringify({ role: "assistant", delta: "hello", run_id: "r1" }),
        "r1",
      ),
    ).toEqual({
      kind: "message",
      role: "assistant",
      delta: "hello",
      run_id: "r1",
    });
  });

  it("falls back to the stream run id", () => {
    expect(
      parseCliChatMessageFrame(JSON.stringify({ delta: "hi" }), "r9"),
    ).toEqual({
      kind: "message",
      role: "assistant",
      delta: "hi",
      run_id: "r9",
    });
  });

  it("rejects non-JSON, non-object, and empty-delta lines", () => {
    expect(parseCliChatMessageFrame("not json", "r1")).toBeNull();
    expect(parseCliChatMessageFrame("[1,2]", "r1")).toBeNull();
    expect(
      parseCliChatMessageFrame(JSON.stringify({ delta: "" }), "r1"),
    ).toBeNull();
    expect(
      parseCliChatMessageFrame(JSON.stringify({ type: "nope" }), "r1"),
    ).toBeNull();
  });

  it("parses tool frames with missing fields defaulted", () => {
    expect(
      parseCliChatToolFrame(
        JSON.stringify({
          run_id: "r1",
          name: "shell",
          status: "ok",
          detail: "done",
        }),
        "r1",
      ),
    ).toEqual({
      kind: "tool",
      run_id: "r1",
      name: "shell",
      status: "ok",
      detail: "done",
    });
    expect(parseCliChatToolFrame("junk", "r2")).toBeNull();
  });
});

describe("appendCliChatFrame", () => {
  it("folds deltas into the trailing assistant bubble of the same run", () => {
    const first = appendCliChatFrame([], {
      kind: "message",
      role: "assistant",
      delta: "hel",
      run_id: "r1",
    });
    const second = appendCliChatFrame(first, {
      kind: "message",
      role: "assistant",
      delta: "lo",
      run_id: "r1",
    });
    expect(second).toEqual([
      { run_id: "r1", role: "assistant", content: "hello" },
    ]);
  });

  it("starts a new bubble after a tool frame or a new run", () => {
    const withTool = appendCliChatFrame(
      [{ run_id: "r1", role: "assistant", content: "a" }],
      { kind: "tool", run_id: "r1", name: "t", status: "ok", detail: "d" },
    );
    expect(withTool).toHaveLength(2);
    const afterTool = appendCliChatFrame(withTool, {
      kind: "message",
      role: "assistant",
      delta: "b",
      run_id: "r1",
    });
    expect(afterTool).toHaveLength(3);
    const newRun = appendCliChatFrame(afterTool, {
      kind: "message",
      role: "assistant",
      delta: "c",
      run_id: "r2",
    });
    expect(newRun).toHaveLength(4);
  });
});

describe("mergeCliChatMessages", () => {
  const backfill: CliChatMessage[] = [
    { run_id: "r0", role: "user", content: "first" },
    { run_id: "r0", role: "assistant", content: "old-replay" },
    { run_id: "r1", role: "user", content: "follow-up" },
  ];

  it("prefers the live stream over the stale replay for the live run", () => {
    const live: CliChatMessage[] = [
      { run_id: "r0", role: "assistant", content: "live" },
    ];
    expect(mergeCliChatMessages(backfill, live, "r0")).toEqual([
      { run_id: "r0", role: "user", content: "first" },
      { run_id: "r1", role: "user", content: "follow-up" },
      { run_id: "r0", role: "assistant", content: "live" },
    ]);
  });

  it("concatenates when no run is live", () => {
    const live: CliChatMessage[] = [
      { run_id: "r2", role: "assistant", content: "x" },
    ];
    expect(mergeCliChatMessages(backfill, live, null)).toEqual([
      ...backfill,
      ...live,
    ]);
  });
});

describe("cli chat send payload", () => {
  const base = {
    prompt: "  again  ",
    model: CLI_CHAT_INHERIT,
    provider: CLI_CHAT_INHERIT,
    effort: CLI_CHAT_INHERIT,
    yolo: CLI_CHAT_INHERIT,
  };

  it("inherits every flag by default", () => {
    const built = buildCliChatSendPayload(base, { yolo: false });
    expect(built).toEqual({
      ok: true,
      payload: { prompt: "again" },
      yolo: false,
    });
  });

  it("sends explicit overrides", () => {
    const built = buildCliChatSendPayload(
      {
        ...base,
        model: "m1",
        provider: CLI_CHAT_NONE,
        effort: "low",
        yolo: "on",
      },
      { yolo: false },
    );
    expect(built).toEqual({
      ok: true,
      payload: {
        prompt: "again",
        model: "m1",
        provider: null,
        effort: "low",
        yolo: true,
      },
      yolo: true,
    });
  });

  it("rejects empty prompts", () => {
    expect(
      buildCliChatSendPayload({ ...base, prompt: "  " }, { yolo: false }),
    ).toEqual({
      ok: false,
      error: "empty_prompt",
    });
  });

  it("accepts yolo without confirmation, explicit or inherited", () => {
    expect(
      buildCliChatSendPayload({ ...base, yolo: "on" }, { yolo: false }),
    ).toEqual({
      ok: true,
      payload: { prompt: "again", yolo: true },
      yolo: true,
    });
    expect(buildCliChatSendPayload(base, { yolo: true })).toEqual({
      ok: true,
      payload: { prompt: "again" },
      yolo: true,
    });
  });

  it("resolves yolo with override-wins semantics", () => {
    expect(resolveCliChatYolo({ yolo: "on" }, { yolo: false })).toBe(true);
    expect(resolveCliChatYolo({ yolo: "off" }, { yolo: true })).toBe(false);
    expect(resolveCliChatYolo({ yolo: CLI_CHAT_INHERIT }, { yolo: true })).toBe(
      true,
    );
  });
});

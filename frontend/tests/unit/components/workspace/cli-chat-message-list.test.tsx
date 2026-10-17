import { expect, test } from "@rstest/core";

import { cliChatToThreadMessages } from "@/components/workspace/cli-sessions/cli-chat-view";
import type { CliChatMessage } from "@/core/cli-sessions/types";

const MESSAGES: CliChatMessage[] = [
  { run_id: "r1", role: "user", content: "do the thing" },
  { run_id: "r1", role: "assistant", content: "on it" },
  { run_id: "r1", role: "tool", name: "shell", status: "ok", detail: "ran" },
];

test("maps user and assistant turns to human/ai messages", () => {
  const [human, ai] = cliChatToThreadMessages(MESSAGES);
  expect(human?.type).toBe("human");
  expect(human).toMatchObject({ content: "do the thing" });
  expect(ai?.type).toBe("ai");
  expect(ai).toMatchObject({ content: "on it" });
  expect(human?.id).toBe("r1:0");
  expect(ai?.id).toBe("r1:1");
});

test("maps tool turns to paired ai-tool_calls + tool messages", () => {
  const [, , call, result] = cliChatToThreadMessages(MESSAGES);
  expect(call?.type).toBe("ai");
  expect(call).toMatchObject({
    content: "",
    tool_calls: [{ id: "r1:2:call", name: "shell", args: {} }],
  });
  expect(result?.type).toBe("tool");
  expect(result).toMatchObject({
    tool_call_id: "r1:2:call",
    name: "shell",
    content: "ran",
    status: "success",
  });
});

test("marks failed tool turns as error results", () => {
  for (const status of ["failed", "error", "rejected", "stream_failed"]) {
    const [, result] = cliChatToThreadMessages([
      { run_id: "r1", role: "tool", name: "t", status, detail: "boom" },
    ]);
    expect(result).toMatchObject({ type: "tool", status: "error" });
  }
  const [, ok] = cliChatToThreadMessages([
    { run_id: "r1", role: "tool", name: "t", status: "result", detail: "ok" },
  ]);
  expect(ok).toMatchObject({ type: "tool", status: "success" });
});

test("maps empty tool turns to a placeholder result", () => {
  const [call, result] = cliChatToThreadMessages([
    { run_id: "r1", role: "tool", name: "", status: "", detail: "" },
  ]);
  expect(call).toMatchObject({
    type: "ai",
    tool_calls: [{ id: "r1:0:call", name: "tool", args: {} }],
  });
  expect(result).toMatchObject({
    type: "tool",
    tool_call_id: "r1:0:call",
    content: "—",
    status: "success",
  });
});

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

test("maps tool turns to assistant markdown preserving name/status/detail", () => {
  const [, , tool] = cliChatToThreadMessages(MESSAGES);
  expect(tool?.type).toBe("ai");
  const content = (tool as unknown as { content: string }).content;
  expect(content).toContain("shell");
  expect(content).toContain("ok");
  expect(content).toContain("ran");
});

test("maps empty tool turns to a placeholder bubble", () => {
  const [tool] = cliChatToThreadMessages([
    { run_id: "r1", role: "tool", name: "", status: "", detail: "" },
  ]);
  expect(tool?.type).toBe("ai");
  expect((tool as unknown as { content: string }).content).toContain("—");
});

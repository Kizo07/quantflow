import { expect, rs, test } from "@rstest/core";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { CliChatMessageList } from "@/components/workspace/cli-sessions/cli-chat-view";
import type { CliChatMessage } from "@/core/cli-sessions/types";

rs.mock("@/core/i18n/hooks", () => ({
  useI18n: () => ({
    t: {
      cliSessions: {
        chat: {
          empty: "No messages yet.",
          truncated: "Trimmed.",
          userLabel: "You",
          assistantLabel: "Assistant",
          toolLabel: "Tool",
        },
      },
    },
  }),
}));

const MESSAGES: CliChatMessage[] = [
  { run_id: "r1", role: "user", content: "do the thing" },
  { run_id: "r1", role: "assistant", content: "on it" },
  { run_id: "r1", role: "tool", name: "shell", status: "ok", detail: "ran" },
];

test("renders user, assistant, and tool bubbles", () => {
  const html = renderToStaticMarkup(
    createElement(CliChatMessageList, { messages: MESSAGES, truncated: false }),
  );
  expect(html).toContain('data-testid="cli-chat-message-user"');
  expect(html).toContain('data-testid="cli-chat-message-assistant"');
  expect(html).toContain('data-testid="cli-chat-message-tool"');
  expect(html).toContain("do the thing");
  expect(html).toContain("Tool: shell");
  expect(html).not.toContain("cli-chat-truncated");
});

test("shows the trimmed notice and the empty state", () => {
  const trimmed = renderToStaticMarkup(
    createElement(CliChatMessageList, { messages: MESSAGES, truncated: true }),
  );
  expect(trimmed).toContain('data-testid="cli-chat-truncated"');

  const empty = renderToStaticMarkup(
    createElement(CliChatMessageList, { messages: [], truncated: false }),
  );
  expect(empty).toContain('data-testid="cli-chat-empty"');
});

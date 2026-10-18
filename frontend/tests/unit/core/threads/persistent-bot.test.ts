import { describe, expect, it } from "@rstest/core";

import {
  findPersistentBotThread,
  isPersistentBotThread,
  persistentBotHistory,
  persistentBotUnreadCount,
} from "@/core/threads/persistent-bot";
import type { AgentThread } from "@/core/threads/types";

function thread(
  id: string,
  overrides: Record<string, unknown> = {},
): AgentThread {
  return {
    thread_id: id,
    updated_at: "2026-01-01T00:00:00.000Z",
    created_at: "2026-01-01T00:00:00.000Z",
    metadata: {},
    status: "idle",
    values: {},
    ...overrides,
  } as AgentThread;
}

function botThread(overrides: Record<string, unknown> = {}): AgentThread {
  return thread("bot-thread", {
    metadata: { persistent_bot: true, deerflow_pinned: true },
    context: { agent_name: "persistent-bot" },
    ...overrides,
  });
}

describe("persistent bot thread helpers", () => {
  it("recognizes only the Gateway-stamped marker", () => {
    expect(isPersistentBotThread(botThread())).toBe(true);
    expect(isPersistentBotThread(thread("plain", { metadata: {} }))).toBe(
      false,
    );
    expect(
      isPersistentBotThread(
        thread("pinned-only", { metadata: { deerflow_pinned: true } }),
      ),
    ).toBe(false);
    expect(isPersistentBotThread(null)).toBe(false);
    expect(isPersistentBotThread(undefined)).toBe(false);
  });

  it("finds no bot thread by default (feature off)", () => {
    const threads = [
      thread("a"),
      thread("b", { metadata: { deerflow_pinned: true } }),
    ];
    expect(findPersistentBotThread(threads)).toBeNull();
  });

  it("finds the singleton and skips an archived stale row", () => {
    const archivedBot = botThread({
      metadata: { persistent_bot: true, deerflow_archived: true },
    });
    const liveBot = botThread();
    expect(findPersistentBotThread([archivedBot, liveBot])).toBe(liveBot);
    expect(findPersistentBotThread([archivedBot])).toBeNull();
  });

  it("collects history from the bot agent only, excluding the singleton and archived", () => {
    const bot = botThread();
    const sameAgent = thread("same-agent", {
      context: { agent_name: "persistent-bot" },
    });
    const otherAgent = thread("other-agent", {
      context: { agent_name: "researcher" },
    });
    const archivedSameAgent = thread("archived-same", {
      metadata: { deerflow_archived: true },
      context: { agent_name: "persistent-bot" },
    });
    const defaultChat = thread("default-chat");

    expect(
      persistentBotHistory(
        [bot, sameAgent, otherAgent, archivedSameAgent, defaultChat],
        bot,
      ),
    ).toEqual([sameAgent]);
  });

  it("returns no history when the bot thread has no resolvable agent name yet", () => {
    const bot = thread("bot-thread", {
      metadata: { persistent_bot: true },
    });
    const someChat = thread("some-chat", {
      context: { agent_name: "persistent-bot" },
    });
    expect(persistentBotHistory([bot, someChat], bot)).toEqual([]);
  });

  it("aggregates unread across the bot thread and its history only", () => {
    const bot = botThread({ unread: true });
    const historyUnread = thread("history-unread", {
      unread: true,
      context: { agent_name: "persistent-bot" },
    });
    const historyRead = thread("history-read", {
      unread: false,
      context: { agent_name: "persistent-bot" },
    });
    const foreignUnread = thread("foreign-unread", { unread: true });

    expect(
      persistentBotUnreadCount(
        [bot, historyUnread, historyRead, foreignUnread],
        bot,
      ),
    ).toBe(2);
    expect(
      persistentBotUnreadCount(
        [botThread({ unread: false }), historyRead],
        botThread({ unread: false }),
      ),
    ).toBe(0);
  });
});

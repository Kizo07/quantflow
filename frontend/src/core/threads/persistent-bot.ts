import type { AgentThread } from "./types";
import { agentNameOfThread, isThreadArchived } from "./utils";

/**
 * Thread metadata key the Gateway stamps on the persistent bot's singleton
 * thread at seed/repair time (backend `BOT_THREAD_METADATA_KEY` in
 * `deerflow/persistent_bot/service.py`). The thread only exists when the
 * deployment enabled `persistent_bot` (off by default), so every UI surface
 * keyed on this marker stays hidden until the bot is seeded — the frontend
 * needs no separate feature flag.
 */
export const PERSISTENT_BOT_METADATA_KEY = "persistent_bot";

export function isPersistentBotThread(
  thread: Pick<AgentThread, "metadata"> | null | undefined,
): boolean {
  return thread?.metadata?.[PERSISTENT_BOT_METADATA_KEY] === true;
}

/**
 * The singleton pinned bot thread among loaded threads, or `null` when the
 * deployment has no persistent bot (the default). Archived rows never count:
 * startup repair un-archives the live singleton.
 */
export function findPersistentBotThread(
  threads: readonly AgentThread[],
): AgentThread | null {
  return (
    threads.find(
      (thread) => isPersistentBotThread(thread) && !isThreadArchived(thread),
    ) ?? null
  );
}

/**
 * Other chats of the bot's agent (its history next to the singleton thread),
 * in list order. Empty when the bot thread carries no resolvable agent name
 * yet (seeded but never run): without the agent scope there is nothing safe
 * to attribute to the bot.
 */
export function persistentBotHistory(
  threads: readonly AgentThread[],
  botThread: AgentThread,
): AgentThread[] {
  const agentName = agentNameOfThread(botThread);
  if (!agentName) {
    return [];
  }
  return threads.filter(
    (thread) =>
      thread.thread_id !== botThread.thread_id &&
      !isThreadArchived(thread) &&
      agentNameOfThread(thread) === agentName,
  );
}

/**
 * Aggregate unread count across the bot thread and its agent history — the
 * number the pinned-assistant sidebar badge shows. Opening a thread clears
 * its share through the existing mark-read endpoint, which patches the same
 * cached `unread` flags this counts.
 */
export function persistentBotUnreadCount(
  threads: readonly AgentThread[],
  botThread: AgentThread,
): number {
  const scoped = [botThread, ...persistentBotHistory(threads, botThread)];
  return scoped.filter((thread) => thread.unread === true).length;
}

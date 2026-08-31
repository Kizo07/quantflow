import type { AgentThread } from "./types";
import { isThreadArchived, isThreadPinned, sortPinnedThreads } from "./utils";

const MAX_VISIBLE_THREADS = 200;
const modelCache = new WeakMap<object, ThreadListModel>();

export type ThreadListModel = {
  byId: ReadonlyMap<string, AgentThread>;
  threads: readonly AgentThread[];
  displayedThreads: readonly AgentThread[];
  archivedThreads: readonly AgentThread[];
  canLoadMore: boolean;
};

export function buildThreadListModel(
  pages: readonly (readonly AgentThread[])[],
): ThreadListModel {
  const cacheKey = pages as object;
  const cached = modelCache.get(cacheKey);
  if (cached) return cached;

  const byId = new Map<string, AgentThread>();
  for (const page of pages) {
    for (const thread of page) {
      if (!byId.has(thread.thread_id)) {
        byId.set(thread.thread_id, thread);
      }
    }
  }
  const threads = [...byId.values()];
  const sortedThreads = sortPinnedThreads(threads);
  // Archived chats are organizational noise: keep them out of the Recent
  // list but expose them separately so an "Archived" section and deep links
  // into archived threads still work.
  const archivedThreads = sortedThreads.filter(isThreadArchived);
  const visibleThreads = sortedThreads.filter(
    (thread) => !isThreadArchived(thread),
  );
  const pinnedThreads = visibleThreads.filter(isThreadPinned);
  const recentThreads = visibleThreads
    .filter((thread) => !isThreadPinned(thread))
    .slice(0, MAX_VISIBLE_THREADS);
  const model: ThreadListModel = {
    byId,
    threads: sortedThreads,
    displayedThreads: [...pinnedThreads, ...recentThreads],
    archivedThreads,
    canLoadMore: recentThreads.length < MAX_VISIBLE_THREADS,
  };
  modelCache.set(cacheKey, model);
  return model;
}

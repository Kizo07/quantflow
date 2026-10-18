import { afterEach, describe, expect, rs, test } from "@rstest/core";
import {
  QueryClient,
  QueryClientProvider,
  type InfiniteData,
} from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";

rs.mock("next/navigation", () => ({
  useRouter: () => ({ push: rs.fn(), replace: rs.fn(), refresh: rs.fn() }),
  usePathname: () => "/workspace/chats",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

import { SidebarProvider } from "@/components/ui/sidebar";
import { PinnedAssistantSection } from "@/components/workspace/pinned-assistant-section";
import type { Locale } from "@/core/i18n";
import { I18nProvider } from "@/core/i18n/context";
import { INFINITE_THREADS_QUERY_KEY_PREFIX } from "@/core/threads/hooks";
import {
  PERSISTENT_BOT_STATUS_QUERY_KEY,
  type PersistentBotStatus,
} from "@/core/threads/persistent-bot-status";
import type { AgentThread } from "@/core/threads/types";

const BOT_THREAD = "b07b0b7d-0000-4000-8000-000000000001";

function thread(
  id: string,
  metadata: Record<string, unknown>,
): AgentThread {
  return {
    thread_id: id,
    created_at: "2026-10-05T12:00:00Z",
    updated_at: "2026-10-05T12:22:04Z",
    metadata,
    status: "idle",
    values: { title: "Assistant" },
  } as unknown as AgentThread;
}

const clients: QueryClient[] = [];

function renderSection(
  threads: AgentThread[],
  status: PersistentBotStatus | null = null,
  locale: Locale = "en-US",
) {
  document.cookie = `locale=${locale}; path=/`;
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  });
  clients.push(queryClient);
  queryClient.setQueryData<InfiniteData<AgentThread[]>>(
    [...INFINITE_THREADS_QUERY_KEY_PREFIX, { archived: false }],
    { pages: [threads], pageParams: [0] },
  );
  if (status) {
    queryClient.setQueryData(PERSISTENT_BOT_STATUS_QUERY_KEY, status);
  }
  return render(
    <I18nProvider initialLocale={locale}>
      <QueryClientProvider client={queryClient}>
        <SidebarProvider>
          <PinnedAssistantSection />
        </SidebarProvider>
      </QueryClientProvider>
    </I18nProvider>,
  );
}

function botStatus(paused: boolean): PersistentBotStatus {
  return {
    enabled: true,
    seeded: true,
    user_paused: paused,
    agent_name: "assistant",
    thread_id: BOT_THREAD,
    heartbeat_task_id: null,
    heartbeat_status: null,
  };
}

afterEach(() => {
  cleanup();
  document.cookie = "locale=; max-age=0; path=/";
  clients.splice(0).forEach((client) => client.clear());
});

describe("PinnedAssistantSection empty and pause states", () => {
  test("renders a disabled row with the reason when no bot thread exists", async () => {
    renderSection([thread("plain-chat", {})]);
    const row = await screen.findByTestId("pinned-assistant-disabled");
    expect(row.getAttribute("aria-disabled")).toBe("true");
    const reasonId = row.getAttribute("aria-describedby");
    expect(reasonId).toBe("pinned-assistant-disabled-reason");
    expect(document.getElementById(reasonId!)?.textContent).toBe(
      "Assistant is not enabled — no session yet",
    );
    // No navigation target while disabled.
    expect(row.tagName).not.toBe("A");
  });

  test("pause toggle exposes its pressed state and keeps the Paused label", async () => {
    renderSection(
      [thread(BOT_THREAD, { persistent_bot: true })],
      botStatus(true),
    );
    const toggle = await screen.findByTestId("pinned-assistant-pause-toggle");
    expect(toggle.getAttribute("aria-pressed")).toBe("true");
    expect(toggle.textContent).toContain("Paused");
  });
});

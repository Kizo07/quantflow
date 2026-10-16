import { afterEach, beforeEach, describe, expect, it, rs } from "@rstest/core";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, within } from "@testing-library/react";

const mocks = rs.hoisted(() => ({
  search: "",
  fetchSessions: rs.fn(),
  fetchChat: rs.fn(),
}));

rs.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(mocks.search),
}));

rs.mock("@/core/cli-sessions/api", () => ({
  fetchCliSessions: mocks.fetchSessions,
  fetchCliChatMessages: mocks.fetchChat,
  cliSessionEventsUrl: (id: string) => `/api/cli-sessions/${id}/events`,
}));

import { CliSidebarSessions } from "@/components/workspace/cli-sessions/cli-sidebar-sessions";
import type { CliChatBackfill, CliSession } from "@/core/cli-sessions/types";
import { I18nProvider } from "@/core/i18n/context";

function session(overrides: Partial<CliSession> = {}): CliSession {
  return {
    id: "session-aaa",
    session_ref: "ref-aaa",
    resumed_from: null,
    status: "completed",
    mode: "coding",
    model: "muse-spark-1.3",
    provider: "meta",
    preset: null,
    effort: "high",
    contributor_mode: false,
    approval_mode: null,
    yolo: false,
    workspace: null,
    pid: null,
    tmux_session: null,
    error: null,
    started_at: "2026-10-14T10:00:00Z",
    ended_at: "2026-10-14T10:05:00Z",
    created_at: "2026-10-14T10:00:00Z",
    ...overrides,
  };
}

function backfill(id: string, firstPrompt: string): CliChatBackfill {
  return {
    id,
    runs: [id],
    truncated: false,
    messages: [{ run_id: id, role: "user", content: firstPrompt }],
  };
}

const clients: QueryClient[] = [];

function renderSidebar() {
  document.cookie = "locale=en-US; path=/";
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  clients.push(client);
  render(
    <QueryClientProvider client={client}>
      <I18nProvider initialLocale="en-US">
        <ul>
          <CliSidebarSessions />
        </ul>
      </I18nProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  mocks.search = "";
  mocks.fetchSessions.mockResolvedValue([session()]);
  mocks.fetchChat.mockImplementation(async (id: string) =>
    backfill(id, `first prompt of ${id}`),
  );
});

afterEach(() => {
  cleanup();
  clients.splice(0).forEach((client) => client.clear());
  mocks.fetchSessions.mockReset();
  mocks.fetchChat.mockReset();
  document.cookie = "locale=; max-age=0; path=/";
});

describe("CliSidebarSessions", () => {
  it("links New session and orders sessions newest first", async () => {
    const older = session({
      id: "session-old",
      started_at: "2026-10-14T09:00:00Z",
      created_at: "2026-10-14T09:00:00Z",
    });
    const newer = session({
      id: "session-new",
      started_at: "2026-10-14T11:00:00Z",
      created_at: "2026-10-14T11:00:00Z",
    });
    mocks.fetchSessions.mockResolvedValue([older, newer]);
    mocks.fetchChat.mockImplementation(async (id: string) =>
      backfill(id, id === "session-new" ? "newer prompt" : "older prompt"),
    );
    renderSidebar();

    const sub = await screen.findByTestId("cli-sidebar-sessions");
    const newLink = within(sub).getByRole("link", { name: "New session" });
    expect(newLink.getAttribute("href")).toBe("/workspace/cli-sessions?new=1");

    await within(sub).findByText("newer prompt");
    await within(sub).findByText("older prompt");
    const links = within(sub).getAllByRole("link");
    expect(links.map((link) => link.getAttribute("href"))).toEqual([
      "/workspace/cli-sessions?new=1",
      "/workspace/cli-sessions?session_id=session-new",
      "/workspace/cli-sessions?session_id=session-old",
    ]);
  });

  it("marks the active session and the new composer", async () => {
    mocks.search = "session_id=session-aaa";
    renderSidebar();
    const sub = await screen.findByTestId("cli-sidebar-sessions");
    await within(sub).findByText("first prompt of session-aaa");
    const active = within(sub).getByRole("link", {
      name: "first prompt of session-aaa",
    });
    expect(active.getAttribute("data-active")).toBe("true");
    const newLink = within(sub).getByRole("link", { name: "New session" });
    expect(newLink.getAttribute("data-active")).toBe("false");
  });

  it("highlights the New row under ?new=1", async () => {
    mocks.search = "new=1";
    renderSidebar();
    const sub = await screen.findByTestId("cli-sidebar-sessions");
    const newLink = within(sub).getByRole("link", { name: "New session" });
    expect(newLink.getAttribute("data-active")).toBe("true");
  });

  it("falls back to the model id before backfills resolve", async () => {
    mocks.fetchChat.mockImplementation(
      () => new Promise<never>(() => undefined),
    );
    renderSidebar();
    const sub = await screen.findByTestId("cli-sidebar-sessions");
    await within(sub).findByText("muse-spark-1.3");
  });
});

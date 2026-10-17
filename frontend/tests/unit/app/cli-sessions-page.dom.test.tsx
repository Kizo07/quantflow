import { afterEach, beforeEach, describe, expect, it, rs } from "@rstest/core";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import type { ReactNode } from "react";

const mocks = rs.hoisted(() => ({
  search: "",
  replace: rs.fn(),
  mode: "coding" as "research" | "coding",
  setMode: rs.fn(),
  fetchSessions: rs.fn(),
  fetchSession: rs.fn(),
  launch: rs.fn(),
  resume: rs.fn(),
  exportRun: rs.fn(),
  inspect: rs.fn(),
  stop: rs.fn(),
  fetchPreset: rs.fn(),
  putPreset: rs.fn(),
  fetchCapabilities: rs.fn(),
  fetchChat: rs.fn(),
  sendChat: rs.fn(),
}));

rs.mock("next/navigation", () => ({
  usePathname: () => "/workspace/cli-sessions",
  useRouter: () => ({ push: rs.fn(), replace: mocks.replace }),
  useSearchParams: () => new URLSearchParams(mocks.search),
}));
rs.mock("next/link", () => ({
  default: ({
    href,
    children,
    ...rest
  }: {
    href: string;
    children: ReactNode;
  } & Record<string, unknown>) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));
rs.mock("@/core/cli-sessions/harness-mode", () => ({
  HARNESS_MODE_DEFAULT: "research",
  useHarnessMode: () => ({ mode: mocks.mode, setMode: mocks.setMode }),
}));
rs.mock("@/core/cli-sessions/api", () => ({
  fetchCliSessions: mocks.fetchSessions,
  fetchCliSession: mocks.fetchSession,
  launchCliSession: mocks.launch,
  resumeCliSession: mocks.resume,
  exportCliSession: mocks.exportRun,
  inspectCliSession: mocks.inspect,
  stopCliSession: mocks.stop,
  fetchCliPresets: async () => [],
  fetchCliPreset: mocks.fetchPreset,
  putCliPreset: mocks.putPreset,
  fetchCliCapabilities: mocks.fetchCapabilities,
  fetchCliChatMessages: mocks.fetchChat,
  sendCliChatMessage: mocks.sendChat,
  cliSessionEventsUrl: (id: string) => `/api/cli-sessions/${id}/events`,
}));

import CliSessionsPage from "@/app/workspace/cli-sessions/page";
import { SidebarProvider } from "@/components/ui/sidebar";
import type {
  CliCapabilities,
  CliChatBackfill,
  CliPreset,
  CliSession,
} from "@/core/cli-sessions/types";
import { I18nProvider } from "@/core/i18n/context";

function session(overrides: Partial<CliSession> = {}): CliSession {
  return {
    id: "session-aaa",
    session_ref: "ref-aaa",
    resumed_from: null,
    status: "running",
    mode: "coding",
    model: "muse-spark-1.3",
    provider: "meta",
    preset: null,
    effort: "high",
    contributor_mode: false,
    approval_mode: null,
    yolo: false,
    workspace: null,
    pid: 123,
    tmux_session: null,
    error: null,
    started_at: "2026-10-14T10:00:00Z",
    ended_at: null,
    created_at: "2026-10-14T10:00:00Z",
    ...overrides,
  };
}

function backfill(
  id: string,
  firstPrompt: string,
  assistant = "ack",
): CliChatBackfill {
  return {
    id,
    runs: [id],
    truncated: false,
    messages: [
      { run_id: id, role: "user", content: firstPrompt },
      { run_id: id, role: "assistant", content: assistant },
    ],
  };
}

const PRESET_13: CliPreset = {
  name: "muse-1.3",
  model: "muse-spark-1.3",
  provider: null,
  preset: null,
  effort: "ultra",
  contributor_mode: true,
  yolo: true,
  approval_mode: null,
  updated_at: null,
};

const CAPABILITIES: CliCapabilities = {
  models: [
    {
      id: "muse-spark-1.3",
      label: "Muse Spark 1.3",
      effortTiers: ["low", "high", "ultra"],
      defaultEffort: "high",
    },
    {
      id: "muse-wonder-1.0",
      label: "muse-wonder-1.0",
      effortTiers: ["minimal", "low"],
      defaultEffort: "low",
    },
  ],
  defaultModel: "muse-spark-1.3",
  providers: [{ id: "meta", label: "Meta" }],
  defaultProvider: "meta",
};

const clients: QueryClient[] = [];

function renderPage() {
  document.cookie = "locale=en-US; path=/";
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  clients.push(client);
  render(
    <QueryClientProvider client={client}>
      <I18nProvider initialLocale="en-US">
        <SidebarProvider>
          <CliSessionsPage />
        </SidebarProvider>
      </I18nProvider>
    </QueryClientProvider>,
  );
  return client;
}

beforeEach(() => {
  mocks.search = "";
  mocks.mode = "coding";
  mocks.fetchSessions.mockResolvedValue([session()]);
  mocks.fetchSession.mockImplementation(async (id: string) => session({ id }));
  mocks.fetchPreset.mockResolvedValue(PRESET_13);
  mocks.fetchCapabilities.mockResolvedValue(CAPABILITIES);
  mocks.fetchChat.mockImplementation(async (id: string) =>
    backfill(id, `first prompt of ${id}`),
  );
});

afterEach(() => {
  cleanup();
  clients.splice(0).forEach((client) => client.clear());
  for (const fn of [
    mocks.replace,
    mocks.setMode,
    mocks.fetchSessions,
    mocks.fetchSession,
    mocks.launch,
    mocks.resume,
    mocks.exportRun,
    mocks.inspect,
    mocks.stop,
    mocks.fetchPreset,
    mocks.putPreset,
    mocks.fetchCapabilities,
    mocks.fetchChat,
    mocks.sendChat,
  ]) {
    fn.mockReset();
  }
  document.cookie = "locale=; max-age=0; path=/";
});

async function openWelcomeComposer() {
  mocks.fetchSessions.mockResolvedValue([]);
  renderPage();
  await screen.findByTestId("cli-launch-composer");
}

describe("CliSessionsPage", () => {
  it("renders chat-shaped: no H1/tabs/aside, research message list", async () => {
    const older = session({
      id: "session-old",
      status: "completed",
      started_at: "2026-10-14T09:00:00Z",
      ended_at: "2026-10-14T09:05:00Z",
      created_at: "2026-10-14T09:00:00Z",
    });
    const newer = session({
      id: "session-new",
      status: "completed",
      started_at: "2026-10-14T11:00:00Z",
      ended_at: "2026-10-14T11:05:00Z",
      created_at: "2026-10-14T11:00:00Z",
    });
    // Unsorted input: the page must select newest first.
    mocks.fetchSessions.mockResolvedValue([older, newer]);
    mocks.fetchChat.mockImplementation(async (id: string) =>
      backfill(id, id === "session-new" ? "newer prompt" : "older prompt"),
    );
    renderPage();

    const list = await screen.findByTestId("cli-chat-messages");
    expect(screen.queryByRole("heading", { level: 1 })).toBeNull();
    expect(screen.queryByRole("radio")).toBeNull();
    expect(screen.queryByTestId("cli-session-list")).toBeNull();

    // Newest session auto-selected: user pill + assistant text via MessageList.
    expect(
      within(list).getByText("newer prompt").closest(".is-user"),
    ).not.toBeNull();
    expect(
      within(list).getByText("ack").closest(".is-assistant"),
    ).not.toBeNull();
    expect(screen.getByTestId("cli-send-composer")).toBeDefined();
  });

  it("condenses tool turns to one-liners with a details expander", async () => {
    mocks.fetchSessions.mockResolvedValue([
      session({ status: "completed", ended_at: "2026-10-14T10:05:00Z" }),
    ]);
    mocks.fetchChat.mockImplementation(async (id: string) => ({
      id,
      runs: [id],
      truncated: false,
      messages: [
        { run_id: id, role: "user", content: "do the thing" },
        {
          run_id: id,
          role: "tool",
          name: "shell",
          status: "ok",
          detail: "SECRET-OUTPUT-123",
        },
        { run_id: id, role: "assistant", content: "done" },
      ],
    }));
    renderPage();
    const list = await screen.findByTestId("cli-chat-messages");
    await within(list).findByText('Use "shell" tool');
    // Condensed: the raw output stays hidden until expanded.
    expect(within(list).queryByText("SECRET-OUTPUT-123")).toBeNull();
    fireEvent.click(within(list).getByRole("button", { name: /Tool details/ }));
    await within(list).findByText("SECRET-OUTPUT-123");
    expect(within(list).getByText("Result")).toBeDefined();
  });

  it("pins the newest session into the URL on a bare landing", async () => {
    renderPage();
    await screen.findByTestId("cli-chat-view");
    await waitFor(() =>
      expect(mocks.replace).toHaveBeenCalledWith(
        "/workspace/cli-sessions?session_id=session-aaa",
        { scroll: false },
      ),
    );
  });

  it("shows the welcome composer for ?new=1", async () => {
    mocks.search = "new=1";
    renderPage();
    await screen.findByTestId("cli-launch-composer");
    expect(screen.queryByTestId("cli-chat-view")).toBeNull();
  });

  it("shows the run stats line where research shows tokens", async () => {
    renderPage();
    const stats = await screen.findByTestId("cli-run-stats");
    expect(stats.textContent).toContain("Took");
    expect(stats.textContent).toContain("muse-spark-1.3");
    expect(stats.textContent).toContain("Running");
  });

  it("keeps the run inspector reachable via the drawer", async () => {
    renderPage();
    await screen.findByTestId("cli-chat-view");
    expect(screen.queryByTestId("cli-session-detail")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Run inspector" }));
    await screen.findByTestId("cli-session-detail");
  });

  it("opens the welcome composer from New session", async () => {
    renderPage();
    await screen.findByTestId("cli-chat-view");
    fireEvent.click(screen.getByRole("button", { name: "New session" }));
    await screen.findByTestId("cli-launch-composer");
    expect(screen.queryByTestId("cli-chat-view")).toBeNull();
    // Starter chips fill the composer.
    fireEvent.click(
      screen.getByRole("button", { name: /Fix the failing tests/ }),
    );
    expect(
      screen.getByPlaceholderText("What should the run do?"),
    ).toHaveProperty("value", expect.stringContaining("Fix the failing tests"));
  });

  it("sends follow-ups with inherit-only payloads by default", async () => {
    mocks.fetchSessions.mockResolvedValue([
      session({ status: "completed", ended_at: "2026-10-14T10:05:00Z" }),
    ]);
    mocks.sendChat.mockResolvedValue(session({ id: "session-bbb" }));
    renderPage();
    await screen.findByTestId("cli-send-composer");
    fireEvent.change(screen.getByPlaceholderText("Follow-up prompt…"), {
      target: { value: "next step" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(mocks.sendChat).toHaveBeenCalledTimes(1));
    expect(mocks.sendChat).toHaveBeenCalledWith("session-aaa", {
      prompt: "next step",
    });
  });

  it("launches yolo runs without a confirm checkbox", async () => {
    mocks.launch.mockResolvedValue(session({ id: "session-new" }));
    await openWelcomeComposer();

    fireEvent.change(screen.getByPlaceholderText("What should the run do?"), {
      target: { value: "do the thing" },
    });
    const startButton = screen.getByRole("button", { name: "Start run" });
    // The 1.3 preset defaults yolo on; the warning shows but never gates.
    await screen.findByTestId("cli-yolo-warning");
    expect(
      screen.queryByRole("checkbox", {
        name: "I understand this run skips approvals",
      }),
    ).toBeNull();
    expect((startButton as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(startButton);
    await waitFor(() => expect(mocks.launch).toHaveBeenCalledTimes(1));
    expect(mocks.launch).toHaveBeenCalledWith(
      expect.objectContaining({
        yolo: true,
      }),
    );
    expect(mocks.launch).not.toHaveBeenCalledWith(
      expect.objectContaining({
        yolo_confirm: expect.anything(),
      }),
    );
  });

  it("applies the 1.3 preset defaults and shows the preset badge", async () => {
    await openWelcomeComposer();
    await screen.findByTestId("cli-preset-badge");
    // Preset yolo default lands on the toggle (warning shows, never gates).
    const yoloSwitch = screen.getByRole("switch", { name: "Yolo mode" });
    await waitFor(() =>
      expect(yoloSwitch.getAttribute("aria-checked")).toBe("true"),
    );
  });

  it("offers preset persistence only for the 1.3 model", async () => {
    mocks.putPreset.mockResolvedValue(PRESET_13);
    await openWelcomeComposer();
    await screen.findByTestId("cli-preset-badge");
    const saveButton = screen.getByTestId("cli-save-preset");
    fireEvent.click(saveButton);
    await waitFor(() => expect(mocks.putPreset).toHaveBeenCalledTimes(1));
    expect(mocks.putPreset).toHaveBeenCalledWith(
      "muse-1.3",
      expect.objectContaining({ model: "muse-spark-1.3" }),
    );

    // Switching to the CLI-default model drops the badge and the save action.
    fireEvent.click(screen.getByRole("combobox", { name: "Model" }));
    fireEvent.click(await screen.findByRole("option", { name: "CLI default" }));
    await waitFor(() =>
      expect(screen.queryByTestId("cli-preset-badge")).toBeNull(),
    );
    expect(screen.queryByTestId("cli-save-preset")).toBeNull();
  });

  it("lists capability models with the CLI default and custom hatch", async () => {
    await openWelcomeComposer();
    await screen.findByTestId("cli-preset-badge");
    fireEvent.click(screen.getByRole("combobox", { name: "Model" }));
    await screen.findByRole("option", { name: "CLI default" });
    expect(
      screen.getByRole("option", {
        name: "Muse Spark 1.3 (muse-spark-1.3)",
      }),
    ).toBeDefined();
    expect(
      screen.getByRole("option", { name: "muse-wonder-1.0" }),
    ).toBeDefined();
    expect(screen.getByRole("option", { name: "Custom…" })).toBeDefined();
  });

  it("offers only capability providers", async () => {
    await openWelcomeComposer();
    await screen.findByTestId("cli-preset-badge");
    fireEvent.click(screen.getByRole("combobox", { name: "Provider" }));
    await screen.findByRole("option", { name: "Meta (meta)" });
    expect(screen.queryByRole("option", { name: "echo" })).toBeNull();
    expect(screen.queryByRole("option", { name: "local" })).toBeNull();
  });

  it("scopes effort tiers to the selected model", async () => {
    await openWelcomeComposer();
    await screen.findByTestId("cli-preset-badge");
    // The 1.3 preset lands effort on ultra, one of its advertised tiers.
    const effortBox = () =>
      screen.getByRole("combobox", { name: "Reasoning effort" });
    await waitFor(() => expect(effortBox().textContent).toContain("ultra"));
    fireEvent.click(effortBox());
    await screen.findByRole("option", { name: "ultra" });
    expect(screen.queryByRole("option", { name: "minimal" })).toBeNull();
    // Re-select the current tier to close the menu.
    fireEvent.click(screen.getByRole("option", { name: "ultra" }));

    fireEvent.click(screen.getByRole("combobox", { name: "Model" }));
    fireEvent.click(
      await screen.findByRole("option", { name: "muse-wonder-1.0" }),
    );
    // Ultra is invalid for wonder, so effort snaps to its default (low).
    await waitFor(() => expect(effortBox().textContent).toContain("low"));
    fireEvent.click(effortBox());
    await screen.findByRole("option", { name: "minimal" });
    expect(screen.queryByRole("option", { name: "ultra" })).toBeNull();
  });

  it("has no run-preset selector and omits the preset field on launch", async () => {
    mocks.launch.mockResolvedValue(session({ id: "session-new" }));
    await openWelcomeComposer();
    await screen.findByTestId("cli-preset-badge");
    expect(screen.queryByRole("combobox", { name: "Run preset" })).toBeNull();
    fireEvent.change(screen.getByPlaceholderText("What should the run do?"), {
      target: { value: "do the thing" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start run" }));
    await waitFor(() => expect(mocks.launch).toHaveBeenCalledTimes(1));
    const payload = mocks.launch.mock.calls[0]?.[0] as Record<string, unknown>;
    expect("preset" in payload).toBe(false);
  });

  it("shows the launch composer and starts a run when nothing is selected", async () => {
    mocks.launch.mockResolvedValue(session({ id: "session-new" }));
    await openWelcomeComposer();
    expect(screen.getByTestId("cli-preset-badge")).toBeDefined();
    fireEvent.change(screen.getByPlaceholderText("What should the run do?"), {
      target: { value: "do the thing" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start run" }));
    await waitFor(() => expect(mocks.launch).toHaveBeenCalledTimes(1));
    expect(mocks.launch).toHaveBeenCalledWith(
      expect.objectContaining({
        prompt: "do the thing",
        model: "muse-spark-1.3",
        mode: "coding",
      }),
    );
  });

  it("surfaces a capabilities failure without breaking the form", async () => {
    mocks.fetchCapabilities.mockRejectedValue(new Error("down"));
    await openWelcomeComposer();
    await screen.findByTestId("cli-capabilities-error");
    // CLI default + custom stay available; providers collapse to None only.
    fireEvent.click(screen.getByRole("combobox", { name: "Model" }));
    await screen.findByRole("option", { name: "CLI default" });
  });

  it("routes new sessions to chats in research mode", async () => {
    mocks.mode = "research";
    mocks.search = "new=1";
    renderPage();
    await screen.findByTestId("cli-research-hint");
    const link = screen.getByRole("link", { name: "New session" });
    expect(link.getAttribute("href")).toBe("/workspace/chats");
  });

  it("links the operator guide from the header", async () => {
    renderPage();
    await screen.findByTestId("cli-chat-view");
    const link = screen.getByRole("link", { name: "Operator guide" });
    expect(link.getAttribute("href")).toBe("/workspace/cli-sessions/guide");
  });
});

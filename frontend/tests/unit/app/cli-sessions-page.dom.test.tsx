import { afterEach, beforeEach, describe, expect, it, rs } from "@rstest/core";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import type { PropsWithChildren, ReactNode } from "react";

const mocks = rs.hoisted(() => ({
  search: "",
  replace: rs.fn(),
  mode: "coding" as "research" | "coding",
  setMode: rs.fn(),
  threads: [] as Array<{
    thread_id: string;
    values?: { title?: string };
    created_at?: string;
    updated_at?: string;
  }>,
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
rs.mock("@/components/workspace/workspace-container", () => ({
  WorkspaceContainer: ({ children }: PropsWithChildren) => (
    <div>{children}</div>
  ),
  WorkspaceHeader: () => <div />,
  WorkspaceBody: ({ children }: PropsWithChildren) => <div>{children}</div>,
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
  cliSessionEventsUrl: (id: string) => `/api/cli-sessions/${id}/events`,
}));
rs.mock("@/core/threads/hooks", () => ({
  useThreads: () => ({ data: mocks.threads, isPending: false }),
}));

import CliSessionsPage from "@/app/workspace/cli-sessions/page";
import type {
  CliCapabilities,
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
        <CliSessionsPage />
      </I18nProvider>
    </QueryClientProvider>,
  );
  return client;
}

beforeEach(() => {
  mocks.search = "";
  mocks.mode = "coding";
  mocks.threads = [];
  mocks.fetchSessions.mockResolvedValue([session()]);
  mocks.fetchSession.mockImplementation(async (id: string) => session({ id }));
  mocks.fetchPreset.mockResolvedValue(PRESET_13);
  mocks.fetchCapabilities.mockResolvedValue(CAPABILITIES);
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
  ]) {
    fn.mockReset();
  }
  document.cookie = "locale=; max-age=0; path=/";
});

async function openLaunchDialog() {
  fireEvent.click(screen.getByRole("button", { name: "New session" }));
  await screen.findByRole("dialog");
}

describe("CliSessionsPage", () => {
  it("lists coding sessions, shows chat first, and toggles to run detail", async () => {
    renderPage();
    await screen.findByTestId("cli-session-list");
    expect(screen.getByTestId("cli-chat-view")).toBeDefined();
    fireEvent.click(screen.getByRole("radio", { name: "Run" }));
    expect(screen.getByTestId("cli-session-detail")).toBeDefined();
  });

  it("launches yolo runs without a confirm checkbox", async () => {
    mocks.launch.mockResolvedValue(session({ id: "session-new" }));
    renderPage();
    await openLaunchDialog();

    fireEvent.change(screen.getByPlaceholderText("What should the run do?"), {
      target: { value: "do the thing" },
    });
    const launchButton = screen.getByRole("button", { name: "Launch" });
    // The 1.3 preset defaults yolo on; the warning shows but never gates.
    await screen.findByTestId("cli-yolo-warning");
    expect(
      screen.queryByRole("checkbox", {
        name: "I understand this run skips approvals",
      }),
    ).toBeNull();
    expect((launchButton as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(launchButton);
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
    renderPage();
    await openLaunchDialog();
    await screen.findByTestId("cli-preset-badge");
    // Preset yolo default lands on the toggle (warning shows, never gates).
    const yoloSwitch = screen.getByRole("switch", { name: "Yolo mode" });
    await waitFor(() =>
      expect(yoloSwitch.getAttribute("aria-checked")).toBe("true"),
    );
  });

  it("offers preset persistence only for the 1.3 model", async () => {
    mocks.putPreset.mockResolvedValue(PRESET_13);
    renderPage();
    await openLaunchDialog();
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
    renderPage();
    await openLaunchDialog();
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
    renderPage();
    await openLaunchDialog();
    await screen.findByTestId("cli-preset-badge");
    fireEvent.click(screen.getByRole("combobox", { name: "Provider" }));
    await screen.findByRole("option", { name: "Meta (meta)" });
    expect(screen.queryByRole("option", { name: "echo" })).toBeNull();
    expect(screen.queryByRole("option", { name: "local" })).toBeNull();
  });

  it("scopes effort tiers to the selected model", async () => {
    renderPage();
    await openLaunchDialog();
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
    renderPage();
    await openLaunchDialog();
    await screen.findByTestId("cli-preset-badge");
    expect(screen.queryByRole("combobox", { name: "Run preset" })).toBeNull();
    fireEvent.change(screen.getByPlaceholderText("What should the run do?"), {
      target: { value: "do the thing" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Launch" }));
    await waitFor(() => expect(mocks.launch).toHaveBeenCalledTimes(1));
    const payload = mocks.launch.mock.calls[0]?.[0] as Record<string, unknown>;
    expect("preset" in payload).toBe(false);
  });

  it("shows the start panel and starts a run when nothing is selected", async () => {
    mocks.fetchSessions.mockResolvedValue([]);
    mocks.launch.mockResolvedValue(session({ id: "session-new" }));
    renderPage();
    await screen.findByTestId("cli-start-panel");
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
    renderPage();
    await openLaunchDialog();
    await screen.findByTestId("cli-capabilities-error");
    // CLI default + custom stay available; providers collapse to None only.
    fireEvent.click(screen.getByRole("combobox", { name: "Model" }));
    await screen.findByRole("option", { name: "CLI default" });
  });

  it("routes new sessions to chats in research mode", async () => {
    mocks.mode = "research";
    renderPage();
    await screen.findByTestId("cli-session-list");
    expect(screen.getByTestId("cli-research-hint")).toBeDefined();
    const link = screen.getByRole("link", { name: "New session" });
    expect(link.getAttribute("href")).toBe("/workspace/chats");
  });

  it("links the operator guide from the dashboard header", async () => {
    renderPage();
    await screen.findByTestId("cli-session-list");
    const link = screen.getByRole("link", { name: "Operator guide" });
    expect(link.getAttribute("href")).toBe("/workspace/cli-sessions/guide");
  });

  it("shows the unified list with mode badges", async () => {
    mocks.threads = [
      {
        thread_id: "thread-1",
        values: { title: "Research notes" },
        updated_at: "2026-10-14T11:00:00Z",
      },
    ];
    renderPage();
    await screen.findByTestId("cli-session-list");
    fireEvent.click(screen.getByRole("radio", { name: "All sessions" }));
    const badges = await screen.findAllByTestId("cli-mode-badge");
    const texts = badges.map((badge) => badge.textContent).sort();
    expect(texts).toEqual(["Coding", "Research"]);
    expect(screen.getByRole("link", { name: /Research notes/ })).toBeDefined();
  });
});

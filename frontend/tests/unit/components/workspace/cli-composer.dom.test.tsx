import { afterEach, beforeEach, describe, expect, it, rs } from "@rstest/core";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";

const mocks = rs.hoisted(() => ({
  launch: rs.fn(),
  sendChat: rs.fn(),
  fetchPreset: rs.fn(),
  putPreset: rs.fn(),
  fetchCapabilities: rs.fn(),
}));

rs.mock("@/core/cli-sessions/api", () => ({
  fetchCliSessions: async () => [],
  fetchCliSession: async () => null,
  launchCliSession: mocks.launch,
  resumeCliSession: async () => null,
  exportCliSession: async () => null,
  inspectCliSession: async () => null,
  stopCliSession: async () => null,
  fetchCliPresets: async () => [],
  fetchCliPreset: mocks.fetchPreset,
  putCliPreset: mocks.putPreset,
  fetchCliCapabilities: mocks.fetchCapabilities,
  fetchCliChatMessages: async () => null,
  sendCliChatMessage: mocks.sendChat,
  cliSessionEventsUrl: (id: string) => `/api/cli-sessions/${id}/events`,
}));

import {
  CliLaunchComposer,
  CliSendComposer,
} from "@/components/workspace/cli-sessions/cli-composer";
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

function renderComposer(node: React.ReactNode) {
  document.cookie = "locale=en-US; path=/";
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  clients.push(client);
  render(
    <QueryClientProvider client={client}>
      <I18nProvider initialLocale="en-US">{node}</I18nProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  mocks.fetchPreset.mockResolvedValue(PRESET_13);
  mocks.fetchCapabilities.mockResolvedValue(CAPABILITIES);
});

afterEach(() => {
  cleanup();
  clients.splice(0).forEach((client) => client.clear());
  for (const fn of [
    mocks.launch,
    mocks.sendChat,
    mocks.fetchPreset,
    mocks.putPreset,
    mocks.fetchCapabilities,
  ]) {
    fn.mockReset();
  }
  document.cookie = "locale=; max-age=0; path=/";
});

describe("CliSendComposer", () => {
  it("sends only explicit overrides, inheriting the rest", async () => {
    mocks.sendChat.mockResolvedValue(session({ id: "session-bbb" }));
    const onSent = rs.fn();
    renderComposer(
      <CliSendComposer
        session={session()}
        streaming={false}
        onSent={onSent}
        onStop={rs.fn()}
      />,
    );
    await screen.findByTestId("cli-send-composer");

    fireEvent.click(screen.getByRole("combobox", { name: "Model" }));
    fireEvent.click(
      await screen.findByRole("option", { name: "muse-wonder-1.0" }),
    );
    fireEvent.click(screen.getByRole("combobox", { name: "Yolo mode" }));
    fireEvent.click(await screen.findByRole("option", { name: "On" }));

    fireEvent.change(screen.getByPlaceholderText("Follow-up prompt…"), {
      target: { value: "next step" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(mocks.sendChat).toHaveBeenCalledTimes(1));
    expect(mocks.sendChat).toHaveBeenCalledWith("session-aaa", {
      prompt: "next step",
      model: "muse-wonder-1.0",
      yolo: true,
    });
    expect(onSent).toHaveBeenCalledWith("session-bbb", "next step");
  });

  it("warns when yolo is effective and morphs submit into stop while streaming", async () => {
    const onStop = rs.fn();
    renderComposer(
      <CliSendComposer
        session={session({ yolo: true })}
        streaming={true}
        onSent={rs.fn()}
        onStop={onStop}
      />,
    );
    await screen.findByTestId("cli-yolo-warning");
    fireEvent.change(screen.getByPlaceholderText("Follow-up prompt…"), {
      target: { value: "next step" },
    });
    // C-1: the submit affordance is the stop button while streaming — it
    // stays enabled and never queues a second send.
    const stopButton = screen.getByRole("button", { name: "Stop" });
    expect(stopButton).toHaveProperty("disabled", false);
    expect(screen.queryByRole("button", { name: "Send" })).toBeNull();
    fireEvent.click(stopButton);
    await waitFor(() => expect(onStop).toHaveBeenCalledTimes(1));
    expect(mocks.sendChat).not.toHaveBeenCalled();
    expect(screen.getByText(/still active/)).toBeDefined();
  });

  it("disables the stop state while a stop is in flight", async () => {
    renderComposer(
      <CliSendComposer
        session={session()}
        streaming={true}
        onSent={rs.fn()}
        onStop={rs.fn()}
        stopping={true}
      />,
    );
    await screen.findByTestId("cli-send-composer");
    expect(screen.getByRole("button", { name: "Stopping…" })).toHaveProperty(
      "disabled",
      true,
    );
  });
});

describe("CliLaunchComposer", () => {
  it("launches with toggles, approval, sandbox, and custom model", async () => {
    mocks.launch.mockResolvedValue(session({ id: "session-new" }));
    const onStarted = rs.fn();
    renderComposer(<CliLaunchComposer onStarted={onStarted} />);
    await screen.findByTestId("cli-preset-badge");
    // Selects, toggles, and the sandbox field live behind the
    // default-closed Run options disclosure (V-3).
    fireEvent.click(screen.getByTestId("cli-run-options"));

    fireEvent.click(screen.getByRole("combobox", { name: "Model" }));
    fireEvent.click(await screen.findByRole("option", { name: "Custom…" }));
    fireEvent.change(screen.getByPlaceholderText("model id"), {
      target: { value: "custom-model-9" },
    });
    fireEvent.click(screen.getByRole("combobox", { name: "Approval mode" }));
    fireEvent.click(await screen.findByRole("option", { name: "never" }));
    fireEvent.change(screen.getByPlaceholderText("Optional profile name"), {
      target: { value: "locked-down" },
    });
    fireEvent.click(screen.getByRole("switch", { name: "Trust workspace" }));
    // Preset left contributor on; flip it back off explicitly.
    fireEvent.click(screen.getByRole("switch", { name: "Contributor mode" }));

    fireEvent.change(screen.getByPlaceholderText("What should the run do?"), {
      target: { value: "do the thing" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start run" }));
    await waitFor(() => expect(mocks.launch).toHaveBeenCalledTimes(1));
    expect(mocks.launch).toHaveBeenCalledWith(
      expect.objectContaining({
        prompt: "do the thing",
        model: "custom-model-9",
        approval_mode: "never",
        sandbox_network: "locked-down",
        trust_workspace: true,
        contributor_mode: false,
        mode: "coding",
      }),
    );
    expect(onStarted).toHaveBeenCalled();
  });

  it("shows starter chips and the disclaimer in welcome mode", async () => {
    renderComposer(<CliLaunchComposer isWelcomeMode onStarted={rs.fn()} />);
    await screen.findByTestId("cli-starter-suggestions");
    fireEvent.click(
      screen.getByRole("button", { name: /Add regression tests/ }),
    );
    expect(
      screen.getByPlaceholderText("What should the run do?"),
    ).toHaveProperty("value", expect.stringContaining("Add regression tests"));
  });
});

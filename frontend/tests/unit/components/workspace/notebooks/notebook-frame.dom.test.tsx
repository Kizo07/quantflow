import { afterEach, beforeEach, describe, expect, it, rs } from "@rstest/core";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";

const mocks = rs.hoisted(() => ({
  fetch: rs.fn(),
}));

rs.mock("@/core/api/fetcher", () => ({
  fetch: mocks.fetch,
}));

import {
  NOTEBOOK_HEALTH_POLL_INTERVAL_MS,
  NotebookFrame,
} from "@/components/workspace/notebooks/notebook-frame";
import { NOTEBOOK_BRIDGE_SOURCE } from "@/core/notebooks/bridge";

const HEALTHY = {
  status: "healthy",
  bridge_contract_version: 1,
  sidecar: {
    reachable: true,
    component_version: "0.3.0",
    contract_version: 1,
  },
};

const TOKEN = {
  token: "sidecar-jwt-1",
  token_type: "Bearer",
  // Far-future so the refresh timer never fires mid-test (delay math is
  // covered in bridge.test.ts); unmount clears the pending timer.
  expires_at: "2030-01-01T00:00:00.000Z",
};

function jsonResponse(status: number, body: unknown): Response {
  return {
    status,
    ok: status >= 200 && status < 300,
    json: async () => body,
  } as Response;
}

function mockConnector({
  health = HEALTHY,
  healthStatus = 200,
  token = TOKEN,
}: {
  health?: unknown;
  healthStatus?: number;
  token?: unknown;
} = {}) {
  mocks.fetch.mockImplementation(async (url: string) => {
    if (url.endsWith("/api/notebooks/health")) {
      return jsonResponse(healthStatus, health);
    }
    if (url.endsWith("/api/notebooks/token")) {
      return jsonResponse(200, token);
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
}

// NOTE: `await act(async () => …)` hangs under rs.useFakeTimers() in
// this stack (verified: even bare act with no render times out), and
// microtask spins alone never flush React's scheduler queue (it needs
// macrotask turns). The fake-timer tests below flush with
// microtask spins + real MessageChannel macrotask yields (MessageChannel
// is not faked) and drive the fake clock with the sync
// advanceTimersByTime inside sync act() instead.
async function flushAsyncWork(rounds = 10): Promise<void> {
  const yieldMacrotask = () =>
    new Promise<void>((resolve) => {
      const ch = new MessageChannel();
      ch.port1.onmessage = () => {
        ch.port1.close();
        resolve();
      };
      ch.port2.postMessage(null);
    });
  for (let i = 0; i < rounds; i++) {
    for (let j = 0; j < 20; j++) {
      await Promise.resolve();
    }
    await yieldMacrotask();
  }
}

async function findNotebookFrame(): Promise<HTMLIFrameElement> {
  const el = await screen.findByTitle("Notebook");
  if (!(el instanceof HTMLIFrameElement)) {
    throw new Error("expected the Notebook frame to be an iframe");
  }
  return el;
}

function getNotebookFrame(): HTMLIFrameElement {
  const el = screen.getByTitle("Notebook");
  if (!(el instanceof HTMLIFrameElement)) {
    throw new Error("expected the Notebook frame to be an iframe");
  }
  return el;
}

function frameContentWindow(iframe: HTMLIFrameElement): Window {
  const win = iframe.contentWindow;
  if (win === null) {
    throw new Error("expected the Notebook frame to have a content window");
  }
  return win;
}

function frameMessage(
  iframe: HTMLIFrameElement,
  data: unknown,
  origin = "null",
): void {
  const source = iframe.contentWindow;
  expect(source).not.toBeNull();
  window.dispatchEvent(new MessageEvent("message", { data, origin, source }));
}

describe("NotebookFrame", () => {
  beforeEach(() => {
    mockConnector();
  });

  afterEach(() => {
    cleanup();
    rs.useRealTimers();
    rs.restoreAllMocks();
    mocks.fetch.mockReset();
  });

  it("auto-mounts a healthy connector with a token-bearing frame URL", async () => {
    render(<NotebookFrame notebookId="nb-1" />);

    const iframe = await findNotebookFrame();
    const src = iframe.getAttribute("src") ?? "";
    expect(src).toContain("/api/notebooks/?");
    expect(src).toContain(`sidecar_token=${TOKEN.token}`);
    expect(src).toContain("notebook=nb-1");
    expect(
      screen.queryByRole("button", { name: "Start notebooks" }),
    ).toBeNull();
  });

  it("pins the exact ?notebook= shape, encoding slashes as %2F", async () => {
    render(<NotebookFrame notebookId="proj/nb-1" />);

    const iframe = await findNotebookFrame();
    const src = iframe.getAttribute("src") ?? "";
    const url = new URL(src, "http://shell.test");
    expect(url.pathname).toBe("/api/notebooks/");
    expect(url.searchParams.get("sidecar_token")).toBe(TOKEN.token);
    expect(url.searchParams.get("notebook")).toBe("proj/nb-1");
    expect(src).toContain("notebook=proj%2Fnb-1");
  });

  it("sandboxes the frame without allow-same-origin (opaque origin)", async () => {
    render(<NotebookFrame />);

    const iframe = await findNotebookFrame();
    const sandbox = iframe.getAttribute("sandbox") ?? "";
    expect(sandbox).toContain("allow-scripts");
    expect(sandbox).not.toContain("allow-same-origin");
  });

  it("shows Start notebooks only as failure recovery, and retries on click", async () => {
    mocks.fetch.mockImplementation(async (url: string) => {
      if (url.endsWith("/api/notebooks/health")) {
        return jsonResponse(503, { status: "unreachable" });
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    render(<NotebookFrame />);

    const start = await screen.findByRole("button", {
      name: "Start notebooks",
    });
    expect(screen.queryByTitle("Notebook")).toBeNull();

    // Idle-cull recovery: the sidecar is back on retry.
    mockConnector();
    fireEvent.click(start);
    await screen.findByTitle("Notebook");
  });

  it("hard-fails a health-reported contract mismatch with a banner", async () => {
    mockConnector({
      health: {
        ...HEALTHY,
        sidecar: { ...HEALTHY.sidecar, contract_version: 2 },
      },
    });
    render(<NotebookFrame />);

    await screen.findByRole("button", { name: "Start notebooks" });
    expect(screen.getByText(/contract mismatch/).textContent).toMatch(
      /contract mismatch/,
    );
    expect(screen.queryByTitle("Notebook")).toBeNull();
    // No token is minted for a mismatched sidecar.
    const tokenMinted = mocks.fetch.mock.calls.some(
      (call) =>
        typeof call[0] === "string" && call[0].endsWith("/api/notebooks/token"),
    );
    expect(tokenMinted).toBe(false);
  });

  it("delivers the mount token after a matching ready handshake", async () => {
    render(<NotebookFrame onCoordinationEvent={rs.fn()} />);

    const iframe = await findNotebookFrame();
    const postMessage = rs
      .spyOn(frameContentWindow(iframe), "postMessage")
      .mockImplementation(() => undefined);
    frameMessage(iframe, {
      source: NOTEBOOK_BRIDGE_SOURCE,
      version: 1,
      type: "ready",
      bridgeVersion: 1,
      componentVersion: "0.3.0",
    });

    await waitFor(() => {
      expect(postMessage).toHaveBeenCalled();
    });
    const [message, targetOrigin] = postMessage.mock.calls[0] as [
      { type: string; token: string; nonce: string; theme: object },
      string,
    ];
    expect(message.type).toBe("mount");
    expect(message.token).toBe(TOKEN.token);
    expect(message.nonce).not.toBe("");
    // Theme is sent at mount (and only at mount — there is no live push).
    expect(message.theme).toEqual({
      background: "#ffffff",
      foreground: "#000000",
      muted: "#f4f4f5",
      accent: "#18181b",
      border: "#e4e4e7",
      fontFamily: "system-ui, sans-serif",
    });
    expect(targetOrigin).toBe("*");
  });

  it("hard-fails a ready version mismatch and never sends the token", async () => {
    render(<NotebookFrame />);

    const iframe = await findNotebookFrame();
    const postMessage = rs
      .spyOn(frameContentWindow(iframe), "postMessage")
      .mockImplementation(() => undefined);
    frameMessage(iframe, {
      source: NOTEBOOK_BRIDGE_SOURCE,
      version: 1,
      type: "ready",
      bridgeVersion: 2,
      componentVersion: "9.9.9",
    });

    await screen.findByRole("button", { name: "Start notebooks" });
    expect(screen.getByText(/version mismatch/).textContent).toMatch(
      /version mismatch/,
    );
    expect(postMessage).not.toHaveBeenCalled();
  });

  it("ignores messages from real origins and foreign sources", async () => {
    const onEvent = rs.fn();
    render(<NotebookFrame onCoordinationEvent={onEvent} />);

    const iframe = await findNotebookFrame();
    const ready = {
      source: NOTEBOOK_BRIDGE_SOURCE,
      version: 1,
      type: "ready",
      bridgeVersion: 1,
      componentVersion: "0.3.0",
    };
    // Real origin, correct source: dropped by the origin allowlist.
    window.dispatchEvent(
      new MessageEvent("message", {
        data: ready,
        origin: "https://evil.example.com",
        source: iframe.contentWindow,
      }),
    );
    // Opaque origin, foreign source: dropped by the source check.
    window.dispatchEvent(
      new MessageEvent("message", {
        data: ready,
        origin: "null",
        source: window,
      }),
    );
    // Malformed payload: dropped by schema validation.
    frameMessage(iframe, { source: NOTEBOOK_BRIDGE_SOURCE, type: "ready" });

    // Still mounting: no handshake completed, no events, no banner.
    expect(screen.getByText("Mounting notebook UI…").textContent).toBe(
      "Mounting notebook UI…",
    );
    expect(onEvent).not.toHaveBeenCalled();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("delivers current-generation coordination events and drops stale ones", async () => {
    const onEvent = rs.fn();
    render(<NotebookFrame onCoordinationEvent={onEvent} />);

    const iframe = await findNotebookFrame();
    const postMessage = rs
      .spyOn(frameContentWindow(iframe), "postMessage")
      .mockImplementation(() => undefined);
    frameMessage(iframe, {
      source: NOTEBOOK_BRIDGE_SOURCE,
      version: 1,
      type: "ready",
      bridgeVersion: 1,
      componentVersion: "0.3.0",
    });
    await waitFor(() => {
      expect(postMessage).toHaveBeenCalled();
    });
    const [mount] = postMessage.mock.calls[0] as [{ nonce: string }];

    frameMessage(iframe, {
      source: NOTEBOOK_BRIDGE_SOURCE,
      version: 1,
      type: "event",
      nonce: "stale-generation",
      name: "execution-state",
      payload: { cell: 1 },
    });
    expect(onEvent).not.toHaveBeenCalled();

    const live = {
      source: NOTEBOOK_BRIDGE_SOURCE,
      version: 1,
      type: "event",
      nonce: mount.nonce,
      name: "dirty-state",
      payload: { dirty: true },
    };
    frameMessage(iframe, live);
    await waitFor(() => {
      expect(onEvent).toHaveBeenCalledWith(live);
    });
  });

  it("surfaces session-expiry failures as a reconnect banner", async () => {
    mocks.fetch.mockImplementation(async (url: string) => {
      if (url.endsWith("/api/notebooks/health")) {
        return jsonResponse(200, HEALTHY);
      }
      if (url.endsWith("/api/notebooks/token")) {
        return jsonResponse(401, { detail: "expired" });
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    render(<NotebookFrame />);

    await screen.findByRole("button", { name: "Start notebooks" });
    expect(screen.queryByTitle("Notebook")).toBeNull();
  });

  it("prompts re-login on 401 but plain retry on 503", async () => {
    // 401 on the token mint: the session cookie died.
    mocks.fetch.mockImplementation(async (url: string) => {
      if (url.endsWith("/api/notebooks/health")) {
        return jsonResponse(200, HEALTHY);
      }
      return jsonResponse(401, { detail: "expired" });
    });
    const first = render(<NotebookFrame />);
    await screen.findByRole("button", { name: "Start notebooks" });
    expect(screen.getByText(/log in again/i).textContent).toMatch(
      /log in again/i,
    );
    first.unmount();

    // 401 on health itself: same re-login prompt.
    mocks.fetch.mockImplementation(async (url: string) => {
      if (url.endsWith("/api/notebooks/health")) {
        return jsonResponse(401, { detail: "expired" });
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    const second = render(<NotebookFrame />);
    await screen.findByRole("button", { name: "Start notebooks" });
    expect(screen.getByText(/log in again/i).textContent).toMatch(
      /log in again/i,
    );
    second.unmount();

    // 503/refused: the sidecar was culled — retry, no login prompt.
    mocks.fetch.mockImplementation(async (url: string) => {
      if (url.endsWith("/api/notebooks/health")) {
        return jsonResponse(503, { status: "unreachable" });
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    render(<NotebookFrame />);
    await screen.findByRole("button", { name: "Start notebooks" });
    expect(screen.queryByText(/log in again/i)).toBeNull();
  });

  it("polls health while mounted: 200 → 503 drops to Start, then re-mounts", async () => {
    expect(NOTEBOOK_HEALTH_POLL_INTERVAL_MS).toBe(30_000);
    // Fake timers from the start: the liveness interval is created at mount.
    rs.useFakeTimers();
    let healthy = true;
    mocks.fetch.mockImplementation(async (url: string) => {
      if (url.endsWith("/api/notebooks/health")) {
        return healthy
          ? jsonResponse(200, HEALTHY)
          : jsonResponse(503, { status: "unreachable" });
      }
      if (url.endsWith("/api/notebooks/token")) {
        return jsonResponse(200, TOKEN);
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    render(<NotebookFrame onCoordinationEvent={rs.fn()} />);
    await flushAsyncWork();

    const iframe = getNotebookFrame();
    act(() => {
      frameMessage(iframe, {
        source: NOTEBOOK_BRIDGE_SOURCE,
        version: 1,
        type: "ready",
        bridgeVersion: 1,
        componentVersion: "0.3.0",
      });
    });
    await flushAsyncWork();
    expect(screen.queryByText("Mounting notebook UI…")).toBeNull();

    // The sidecar is culled mid-session: the next poll drops to Start.
    healthy = false;
    act(() => {
      rs.advanceTimersByTime(NOTEBOOK_HEALTH_POLL_INTERVAL_MS);
    });
    await flushAsyncWork();
    expect(
      screen.getByRole("button", { name: "Start notebooks" }),
    ).not.toBeNull();
    expect(screen.queryByTitle("Notebook")).toBeNull();

    // Recovery: the sidecar is back, Start re-mounts the frame.
    healthy = true;
    fireEvent.click(screen.getByRole("button", { name: "Start notebooks" }));
    await flushAsyncWork();
    expect(screen.getByTitle("Notebook")).not.toBeNull();
  });

  it("drops a stale token-refresh that resolves after a re-mount", async () => {
    rs.useFakeTimers();
    const soon = () => ({
      ...TOKEN,
      expires_at: new Date(Date.now() + 62_000).toISOString(),
    });
    let tokenResponder: () => Promise<Response> = async () =>
      jsonResponse(200, soon());
    mocks.fetch.mockImplementation(async (url: string) => {
      if (url.endsWith("/api/notebooks/health")) {
        return jsonResponse(200, HEALTHY);
      }
      if (url.endsWith("/api/notebooks/token")) {
        return tokenResponder();
      }
      throw new Error(`unexpected fetch: ${url}`);
    });

    // Generation 1 mounts with a ~2s refresh delay.
    const view = render(<NotebookFrame notebookId="nb-1" />);
    await flushAsyncWork();
    expect(screen.getByTitle("Notebook")).not.toBeNull();

    // Fire generation-1's refresh, but hold its fetch mid-flight…
    let resolveStale: (value: Response) => void = () => {
      throw new Error("stale refresh fetch not started");
    };
    tokenResponder = () =>
      new Promise<Response>((resolve) => {
        resolveStale = resolve;
      });
    act(() => {
      rs.advanceTimersByTime(2_000);
    });
    await flushAsyncWork();

    // …and re-mount (generation 2) before it resolves.
    tokenResponder = async () => jsonResponse(200, soon());
    act(() => {
      view.rerender(<NotebookFrame notebookId="nb-2" />);
    });
    await flushAsyncWork();
    const iframe2 = getNotebookFrame();
    const postMessage = rs
      .spyOn(frameContentWindow(iframe2), "postMessage")
      .mockImplementation(() => undefined);

    resolveStale(jsonResponse(200, TOKEN));
    await flushAsyncWork();
    // Stale generation: no post, and generation 2's timer survives.
    expect(postMessage).not.toHaveBeenCalled();

    // Generation 2's own refresh still fires with the current nonce.
    tokenResponder = async () => jsonResponse(200, TOKEN);
    act(() => {
      rs.advanceTimersByTime(2_000);
    });
    await flushAsyncWork();
    expect(postMessage).toHaveBeenCalledTimes(1);
    const [message] = postMessage.mock.calls[0] as [
      { type: string; nonce: string },
    ];
    expect(message.type).toBe("token-refresh");
    expect(message.nonce).not.toBe("");
  });
});

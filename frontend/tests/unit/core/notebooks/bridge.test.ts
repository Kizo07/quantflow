import { describe, expect, it } from "@rstest/core";

import {
  NOTEBOOK_BRIDGE_SOURCE,
  NOTEBOOK_BRIDGE_VERSION,
  NOTEBOOK_COORDINATION_EVENTS,
  NOTEBOOK_RELAY_PATH_PREFIX,
  buildNotebookFrameUrl,
  buildNotebookMountMessage,
  buildNotebookTokenRefreshMessage,
  isAllowedNotebookOrigin,
  isCurrentMountNonce,
  negotiateNotebookBridgeVersion,
  notebookRefreshDelayMs,
  parseNotebookFrameMessage,
  type NotebookThemeTokens,
} from "@/core/notebooks/bridge";

const theme: NotebookThemeTokens = {
  background: "#fff",
  foreground: "#000",
  muted: "#f4f4f5",
  accent: "#18181b",
  border: "#e4e4e7",
  fontFamily: "system-ui",
};

describe("NOTEBOOK_BRIDGE_VERSION", () => {
  it("is pinned at 1 on both sides of the contract", () => {
    expect(NOTEBOOK_BRIDGE_VERSION).toBe(1);
  });
});

describe("isAllowedNotebookOrigin", () => {
  it("accepts only the opaque sandbox origin", () => {
    expect(isAllowedNotebookOrigin("null")).toBe(true);
  });

  it("rejects real origins", () => {
    expect(isAllowedNotebookOrigin("https://app.example.com")).toBe(false);
    expect(isAllowedNotebookOrigin("http://localhost:2026")).toBe(false);
    expect(isAllowedNotebookOrigin("")).toBe(false);
  });
});

describe("parseNotebookFrameMessage", () => {
  it("accepts a well-formed ready message", () => {
    expect(
      parseNotebookFrameMessage({
        source: NOTEBOOK_BRIDGE_SOURCE,
        version: 1,
        type: "ready",
        bridgeVersion: 1,
        componentVersion: "0.3.0",
      }),
    ).toEqual({
      source: NOTEBOOK_BRIDGE_SOURCE,
      version: 1,
      type: "ready",
      bridgeVersion: 1,
      componentVersion: "0.3.0",
    });
  });

  it("accepts a well-formed coordination event", () => {
    expect(
      parseNotebookFrameMessage({
        source: NOTEBOOK_BRIDGE_SOURCE,
        version: 1,
        type: "event",
        nonce: "abc",
        name: "execution-state",
        payload: { cell: 3, state: "running" },
      }),
    ).toEqual({
      source: NOTEBOOK_BRIDGE_SOURCE,
      version: 1,
      type: "event",
      nonce: "abc",
      name: "execution-state",
      payload: { cell: 3, state: "running" },
    });
  });

  it("rejects every coordination event name outside the allowlist", () => {
    for (const name of ["exec", "write-cell", "run-cell", "eval", ""]) {
      expect(
        parseNotebookFrameMessage({
          source: NOTEBOOK_BRIDGE_SOURCE,
          version: 1,
          type: "event",
          nonce: "abc",
          name,
          payload: null,
        }),
      ).toBeNull();
    }
    expect(NOTEBOOK_COORDINATION_EVENTS).toEqual([
      "execution-state",
      "dirty-state",
      "cell-focus",
    ]);
  });

  it("rejects wrong-source, stale-version, and malformed messages", () => {
    const good = {
      source: NOTEBOOK_BRIDGE_SOURCE,
      version: 1,
      type: "ready",
      bridgeVersion: 1,
      componentVersion: "0.3.0",
    };
    expect(parseNotebookFrameMessage(null)).toBeNull();
    expect(parseNotebookFrameMessage("ready")).toBeNull();
    expect(
      parseNotebookFrameMessage({ ...good, source: "something-else" }),
    ).toBeNull();
    expect(parseNotebookFrameMessage({ ...good, version: 0 })).toBeNull();
    expect(parseNotebookFrameMessage({ ...good, version: 2 })).toBeNull();
    expect(parseNotebookFrameMessage({ ...good, type: "mystery" })).toBeNull();
    expect(
      parseNotebookFrameMessage({ ...good, bridgeVersion: "1" }),
    ).toBeNull();
    expect(
      parseNotebookFrameMessage({ ...good, componentVersion: undefined }),
    ).toBeNull();
    expect(
      parseNotebookFrameMessage({
        source: NOTEBOOK_BRIDGE_SOURCE,
        version: 1,
        type: "event",
        nonce: "",
        name: "dirty-state",
        payload: null,
      }),
    ).toBeNull();
  });
});

describe("negotiateNotebookBridgeVersion", () => {
  it("matches the pinned version and mismatches anything else", () => {
    expect(negotiateNotebookBridgeVersion(1)).toBe("match");
    expect(negotiateNotebookBridgeVersion(0)).toBe("mismatch");
    expect(negotiateNotebookBridgeVersion(2)).toBe("mismatch");
  });
});

describe("isCurrentMountNonce", () => {
  it("accepts the current generation and drops stale or empty nonces", () => {
    expect(isCurrentMountNonce({ nonce: "n1" }, "n1")).toBe(true);
    expect(isCurrentMountNonce({ nonce: "n0" }, "n1")).toBe(false);
    expect(isCurrentMountNonce({ nonce: "" }, "")).toBe(false);
  });
});

describe("shell-to-frame builders", () => {
  it("mount carries source, version, token, nonce, and theme", () => {
    expect(buildNotebookMountMessage("tok", "n1", theme)).toEqual({
      source: NOTEBOOK_BRIDGE_SOURCE,
      version: NOTEBOOK_BRIDGE_VERSION,
      type: "mount",
      token: "tok",
      nonce: "n1",
      theme,
    });
  });

  it("token-refresh carries source, version, token, and nonce", () => {
    expect(buildNotebookTokenRefreshMessage("tok2", "n1")).toEqual({
      source: NOTEBOOK_BRIDGE_SOURCE,
      version: NOTEBOOK_BRIDGE_VERSION,
      type: "token-refresh",
      token: "tok2",
      nonce: "n1",
    });
  });
});

describe("notebookRefreshDelayMs", () => {
  it("schedules refresh one margin before expiry", () => {
    const now = Date.parse("2026-10-31T00:00:00.000Z");
    const expiresAt = "2026-10-31T00:05:00.000Z";
    expect(notebookRefreshDelayMs(expiresAt, now)).toBe(240_000);
  });

  it("fires immediately for expired or unparseable timestamps", () => {
    const now = Date.parse("2026-10-31T00:10:00.000Z");
    expect(notebookRefreshDelayMs("2026-10-31T00:05:00.000Z", now)).toBe(0);
    expect(notebookRefreshDelayMs("not-a-date", now)).toBe(0);
  });
});

describe("NOTEBOOK_RELAY_PATH_PREFIX", () => {
  it("is pinned at the gateway relay prefix", () => {
    expect(NOTEBOOK_RELAY_PATH_PREFIX).toBe("/api/notebooks/");
  });
});

describe("buildNotebookFrameUrl", () => {
  it("pins the exact mount shape: relay prefix + document token + notebook", () => {
    expect(buildNotebookFrameUrl("", "tok-1", "nb-1")).toBe(
      "/api/notebooks/?sidecar_token=tok-1&notebook=nb-1",
    );
  });

  it("omits the notebook selector when no notebook is selected", () => {
    expect(buildNotebookFrameUrl("", "tok-1")).toBe(
      "/api/notebooks/?sidecar_token=tok-1",
    );
  });

  it("encodes slashes in notebook ids so they survive as one value", () => {
    const url = buildNotebookFrameUrl("", "tok-1", "proj/nb-1");
    expect(url).toContain("notebook=proj%2Fnb-1");
    expect(new URL(url, "http://shell.test").searchParams.get("notebook")).toBe(
      "proj/nb-1",
    );
  });

  it("stays contained under the relay prefix for hostile notebook ids (D2)", () => {
    for (const id of ["../../admin", "/etc/passwd", "..%2Fadmin"]) {
      const pathname = new URL(
        buildNotebookFrameUrl("", "t", id),
        "http://shell.test",
      ).pathname;
      expect(pathname).toBe("/api/notebooks/");
    }
  });
});

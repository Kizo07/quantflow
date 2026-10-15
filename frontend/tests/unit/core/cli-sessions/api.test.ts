import { beforeEach, describe, expect, it, rs } from "@rstest/core";

const mocks = rs.hoisted(() => ({ fetch: rs.fn() }));
rs.mock("@/core/api/fetcher", () => ({ fetch: mocks.fetch }));
rs.mock("@/core/config", () => ({ getBackendBaseURL: () => "" }));

import { GatewayApiError } from "@/core/api/errors";
import {
  exportCliSession,
  fetchCliCapabilities,
  fetchCliChatMessages,
  fetchCliPreset,
  fetchCliSession,
  fetchCliSessions,
  inspectCliSession,
  launchCliSession,
  putCliPreset,
  resumeCliSession,
  sendCliChatMessage,
  stopCliSession,
} from "@/core/cli-sessions/api";

function ok(body: unknown): Response {
  return Response.json(body);
}

function fail(status: number, detail = "nope"): Response {
  return Response.json({ detail }, { status });
}

beforeEach(() => {
  mocks.fetch.mockReset();
});

describe("cli-sessions api", () => {
  it("lists sessions", async () => {
    mocks.fetch.mockResolvedValue(ok([{ id: "a" }]));
    await expect(fetchCliSessions()).resolves.toEqual([{ id: "a" }]);
    expect(mocks.fetch).toHaveBeenCalledWith("/api/cli-sessions");
  });

  it("fetches one session", async () => {
    mocks.fetch.mockResolvedValue(ok({ id: "a" }));
    await expect(fetchCliSession("a")).resolves.toEqual({ id: "a" });
    expect(mocks.fetch).toHaveBeenCalledWith("/api/cli-sessions/a");
  });

  it("launches with a JSON body", async () => {
    mocks.fetch.mockResolvedValue(ok({ id: "b" }));
    await launchCliSession({ prompt: "hi", effort: "low" });
    expect(mocks.fetch).toHaveBeenCalledWith("/api/cli-sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "hi", effort: "low" }),
    });
  });

  it("resumes, exports (redacted default), inspects, and stops", async () => {
    mocks.fetch.mockResolvedValue(ok({ id: "b" }));
    await resumeCliSession("b", { prompt: "again" });
    expect(mocks.fetch).toHaveBeenCalledWith("/api/cli-sessions/b/resume", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "again" }),
    });

    mocks.fetch.mockResolvedValue(ok({ id: "b", redacted: true }));
    await exportCliSession("b");
    expect(mocks.fetch).toHaveBeenCalledWith("/api/cli-sessions/b/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ redacted: true }),
    });

    mocks.fetch.mockResolvedValue(ok({ id: "b", output: "trace" }));
    await expect(inspectCliSession("b")).resolves.toEqual({
      id: "b",
      output: "trace",
    });

    mocks.fetch.mockResolvedValue(ok({ id: "b", status: "stopped" }));
    await stopCliSession("b");
    expect(mocks.fetch).toHaveBeenCalledWith("/api/cli-sessions/b", {
      method: "DELETE",
    });
  });

  it("gets and puts presets", async () => {
    mocks.fetch.mockResolvedValue(ok({ name: "muse-1.3" }));
    await expect(fetchCliPreset("muse-1.3")).resolves.toEqual({
      name: "muse-1.3",
    });

    mocks.fetch.mockResolvedValue(ok({ name: "muse-1.3", yolo: true }));
    await putCliPreset("muse-1.3", { yolo: true });
    expect(mocks.fetch).toHaveBeenCalledWith("/api/cli-presets/muse-1.3", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yolo: true }),
    });
  });

  it("backfills chat messages and sends follow-ups", async () => {
    mocks.fetch.mockResolvedValue(
      ok({ id: "b", runs: ["b"], messages: [], truncated: false }),
    );
    await expect(fetchCliChatMessages("b")).resolves.toEqual({
      id: "b",
      runs: ["b"],
      messages: [],
      truncated: false,
    });
    expect(mocks.fetch).toHaveBeenCalledWith("/api/cli-sessions/b/messages");

    mocks.fetch.mockResolvedValue(ok({ id: "c" }));
    await sendCliChatMessage("b", { prompt: "again", effort: "low" });
    expect(mocks.fetch).toHaveBeenCalledWith("/api/cli-sessions/b/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "again", effort: "low" }),
    });
  });

  it("throws GatewayApiError on failure", async () => {
    mocks.fetch.mockResolvedValue(fail(422, "bad effort"));
    await expect(fetchCliSessions()).rejects.toBeInstanceOf(GatewayApiError);
  });

  it("fetches host CLI capabilities", async () => {
    const capabilities = {
      models: [
        {
          id: "muse-spark-1.3",
          label: "Muse Spark 1.3",
          effortTiers: ["low", "high"],
          defaultEffort: "high",
        },
      ],
      defaultModel: "muse-spark-1.3",
      providers: [{ id: "meta", label: "Meta" }],
      defaultProvider: "meta",
    };
    mocks.fetch.mockResolvedValue(ok(capabilities));
    await expect(fetchCliCapabilities()).resolves.toEqual(capabilities);
    expect(mocks.fetch).toHaveBeenCalledWith("/api/cli-capabilities");
  });

  it("throws GatewayApiError when capabilities fail", async () => {
    mocks.fetch.mockResolvedValue(fail(403, "forbidden"));
    await expect(fetchCliCapabilities()).rejects.toBeInstanceOf(
      GatewayApiError,
    );
  });
});

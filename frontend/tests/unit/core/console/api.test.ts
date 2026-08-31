import { beforeEach, describe, expect, rs, test } from "@rstest/core";

rs.mock("@/core/api/fetcher", () => ({
  fetch: rs.fn(),
}));

rs.mock("@/core/config", () => ({
  getBackendBaseURL: () => "/backend",
}));

import { fetch as fetcher } from "@/core/api/fetcher";
import {
  ConsoleUnavailableError,
  fetchConsoleRuns,
  fetchConsoleStats,
  fetchConsoleUsage,
} from "@/core/console/api";

const mockedFetch = rs.mocked(fetcher);

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    statusText: status >= 400 ? "Bad Request" : "OK",
    headers: { "Content-Type": "application/json" },
  });
}

const USAGE_BODY = {
  days: [
    {
      date: "2026-08-30",
      total_tokens: 1200,
      input_tokens: 800,
      output_tokens: 400,
      runs: 2,
      cost: 0.01,
    },
    {
      date: "2026-08-31",
      total_tokens: 0,
      input_tokens: 0,
      output_tokens: 0,
      runs: 0,
      cost: 0,
    },
  ],
  by_model: {
    "qwen3.8-max": {
      tokens: 1200,
      runs: 2,
      cost: 0.01,
      input_tokens: 800,
      cache_read_tokens: 250,
    },
  },
  total_tokens: 1200,
  total_runs: 2,
  total_cost: 0.01,
  currency: "USD",
};

beforeEach(() => {
  mockedFetch.mockReset();
});

describe("console api", () => {
  test("fetchConsoleStats parses the headline counters", async () => {
    mockedFetch.mockResolvedValueOnce(
      jsonResponse(200, {
        total_runs: 12,
        active_runs: 1,
        failed_runs: 2,
        total_threads: 30,
        total_agents: 3,
        total_tokens: 500000,
        total_cost: 1.23,
        currency: "USD",
      }),
    );

    await expect(fetchConsoleStats()).resolves.toMatchObject({
      total_runs: 12,
      active_runs: 1,
      failed_runs: 2,
      total_tokens: 500000,
      total_cost: 1.23,
    });
    const [url] = mockedFetch.mock.calls[0]!;
    expect(url).toBe("/backend/api/console/stats");
  });

  test("fetchConsoleUsage builds days and tz query params", async () => {
    mockedFetch.mockResolvedValueOnce(jsonResponse(200, USAGE_BODY));

    const usage = await fetchConsoleUsage({
      days: 7,
      tzOffsetMinutes: -300,
    });

    expect(usage.total_tokens).toBe(1200);
    expect(usage.days).toHaveLength(2);
    expect(usage.by_model["qwen3.8-max"]?.tokens).toBe(1200);
    const [url] = mockedFetch.mock.calls[0]!;
    expect(url).toContain("/backend/api/console/usage");
    expect(url).toContain("days=7");
    expect(url).toContain("tz_offset_minutes=-300");
  });

  test("fetchConsoleUsage defaults days to 14", async () => {
    mockedFetch.mockResolvedValueOnce(jsonResponse(200, USAGE_BODY));

    await fetchConsoleUsage();

    const [url] = mockedFetch.mock.calls[0]!;
    expect(url).toContain("days=14");
  });

  test("fetchConsoleRuns passes pagination and status", async () => {
    mockedFetch.mockResolvedValueOnce(
      jsonResponse(200, { runs: [], has_more: false }),
    );

    await expect(
      fetchConsoleRuns({ limit: 50, offset: 100, status: "error" }),
    ).resolves.toMatchObject({ runs: [], has_more: false });

    const [url] = mockedFetch.mock.calls[0]!;
    expect(url).toContain("/backend/api/console/runs");
    expect(url).toContain("limit=50");
    expect(url).toContain("offset=100");
    expect(url).toContain("status=error");
  });

  test("503 surfaces as ConsoleUnavailableError", async () => {
    mockedFetch.mockResolvedValueOnce(
      jsonResponse(503, { detail: "SQL backend required" }),
    );

    await expect(fetchConsoleUsage()).rejects.toBeInstanceOf(
      ConsoleUnavailableError,
    );
  });

  test("non-503 errors surface as ConsoleUsageRequestError with detail", async () => {
    mockedFetch.mockResolvedValueOnce(
      jsonResponse(500, { detail: "boom" }),
    );

    await expect(fetchConsoleStats()).rejects.toMatchObject({
      name: "ConsoleUsageRequestError",
      message: "boom",
      status: 500,
    });
  });

  test("non-JSON error bodies degrade to a status message", async () => {
    mockedFetch.mockResolvedValueOnce(
      new Response("not json", { status: 502, statusText: "Bad Gateway" }),
    );

    await expect(fetchConsoleRuns()).rejects.toMatchObject({
      status: 502,
      message: "HTTP 502: Bad Gateway",
    });
  });
});

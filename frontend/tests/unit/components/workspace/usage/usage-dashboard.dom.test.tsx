import { afterEach, beforeEach, describe, expect, it, rs } from "@rstest/core";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import type { PropsWithChildren } from "react";

rs.mock("@/core/console/api", () => ({
  fetchConsoleStats: rs.fn(),
  fetchConsoleUsage: rs.fn(),
  ConsoleUnavailableError: class ConsoleUnavailableError extends Error {
    constructor(message: string) {
      super(message);
      this.name = "ConsoleUnavailableError";
    }
  },
}));

rs.mock("@/core/i18n/hooks", () => ({
  useI18n: () => ({
    locale: "en-US",
    t: {
      usage: {
        title: "Usage",
        unavailable: "Usage statistics are unavailable on this deployment.",
        totalRuns: "Runs",
        totalThreads: "Chats",
        totalTokens: "Tokens",
        totalCost: "Cost",
        costUnavailable: "—",
        lastDays: "Last {days} days",
        byModel: "By model",
        model: "Model",
        tokens: "Tokens",
        runs: "Runs",
        recentRuns: "Recent runs",
        status: "Status",
        thread: "Chat",
        duration: "Duration",
        noRuns: "No runs yet.",
      },
    },
    changeLocale: rs.fn(),
  }),
}));

import { UsageDashboard } from "@/components/workspace/usage/usage-dashboard";
import {
  ConsoleUnavailableError,
  fetchConsoleStats,
  fetchConsoleUsage,
} from "@/core/console/api";

const mockedStats = rs.mocked(fetchConsoleStats);
const mockedUsage = rs.mocked(fetchConsoleUsage);

const USAGE = {
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
      total_tokens: 300,
      input_tokens: 100,
      output_tokens: 200,
      runs: 1,
      cost: 0.005,
    },
  ],
  by_model: {
    "qwen3.8-max": {
      tokens: 1500,
      runs: 3,
      cost: 0.015,
      input_tokens: 900,
      cache_read_tokens: 250,
    },
  },
  total_tokens: 1500,
  total_runs: 3,
  total_cost: 0.015,
  currency: "USD",
};

const STATS = {
  total_runs: 12,
  active_runs: 0,
  failed_runs: 2,
  total_threads: 30,
  total_agents: 3,
  total_tokens: 500000,
  total_cost: null,
  currency: null,
};

function wrapper({ children }: PropsWithChildren) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

beforeEach(() => {
  mockedStats.mockReset();
  mockedUsage.mockReset();
});

afterEach(() => {
  cleanup();
});

describe("UsageDashboard", () => {
  it("renders headline stats, bar chart, and per-model table", async () => {
    mockedStats.mockResolvedValue(STATS);
    mockedUsage.mockResolvedValue(USAGE);

    render(<UsageDashboard />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText("500,000")).toBeTruthy();
    });
    expect(screen.getByText("12")).toBeTruthy();
    expect(screen.getByText("30")).toBeTruthy();
    // Cost null → dash, not 0
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
    // Per-model breakdown shows the model name and its tokens
    expect(screen.getByText("qwen3.8-max")).toBeTruthy();
    expect(screen.getByText("1,500")).toBeTruthy();
    // Bar chart renders one bar per day
    expect(document.querySelectorAll("[data-usage-bar]").length).toBe(2);
  });

  it("shows the unavailable state on ConsoleUnavailableError", async () => {
    mockedStats.mockRejectedValue(
      new ConsoleUnavailableError("memory backend"),
    );
    mockedUsage.mockRejectedValue(
      new ConsoleUnavailableError("memory backend"),
    );

    render(<UsageDashboard />, { wrapper });

    await waitFor(() => {
      expect(
        screen.getByText(
          "Usage statistics are unavailable on this deployment.",
        ),
      ).toBeTruthy();
    });
  });
});

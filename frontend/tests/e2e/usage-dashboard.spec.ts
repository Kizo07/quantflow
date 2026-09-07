import { expect, test } from "@playwright/test";

import { mockLangGraphAPI } from "./utils/mock-api";

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

const USAGE = {
  days: [
    {
      date: "2026-08-30",
      total_tokens: 1200,
      input_tokens: 800,
      output_tokens: 400,
      runs: 2,
      cost: 0,
    },
    {
      date: "2026-08-31",
      total_tokens: 300,
      input_tokens: 100,
      output_tokens: 200,
      runs: 1,
      cost: 0,
    },
  ],
  by_model: {
    "qwen3.8-max": {
      tokens: 1500,
      runs: 3,
      cost: null,
      input_tokens: 900,
      cache_read_tokens: 250,
    },
  },
  total_tokens: 1500,
  total_runs: 3,
  total_cost: null,
  currency: null,
};

function mockConsoleAPI(page: import("@playwright/test").Page) {
  void page.route("**/api/console/stats", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(STATS),
    }),
  );
  void page.route("**/api/console/usage**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(USAGE),
    }),
  );
  void page.route("**/api/console/runs**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ runs: [], has_more: false }),
    }),
  );
}

test.describe("Usage dashboard", () => {
  test("renders headline stats, bars, and per-model table", async ({
    page,
  }) => {
    mockLangGraphAPI(page);
    mockConsoleAPI(page);

    await page.goto("/workspace/usage");

    await expect(page.getByText("500,000")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText("12")).toBeVisible();
    await expect(page.getByText("30")).toBeVisible();
    await expect(page.getByText("qwen3.8-max")).toBeVisible();
    // One bar per day in the mocked series.
    await expect(page.locator("[data-usage-bar]")).toHaveCount(2);
  });

  test("shows unavailable state when the backend is memory-backed", async ({
    page,
  }) => {
    mockLangGraphAPI(page);
    void page.route("**/api/console/stats", (route) =>
      route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "SQL backend required" }),
      }),
    );
    void page.route("**/api/console/usage**", (route) =>
      route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "SQL backend required" }),
      }),
    );

    await page.goto("/workspace/usage");

    await expect(
      page.getByText("Usage statistics are unavailable on this deployment."),
    ).toBeVisible({ timeout: 15_000 });
  });
});

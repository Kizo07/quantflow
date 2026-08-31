import { describe, expect, test } from "@rstest/core";

import type { ConsoleUsageDay } from "@/core/console/types";
import { computeBars } from "@/core/console/usage-chart";

function day(date: string, total_tokens: number): ConsoleUsageDay {
  return { date, total_tokens, input_tokens: 0, output_tokens: 0, runs: 0, cost: 0 };
}

describe("computeBars", () => {
  test("normalizes bar heights against the series maximum", () => {
    const bars = computeBars(
      [day("2026-08-29", 100), day("2026-08-30", 400), day("2026-08-31", 200)],
      120,
    );

    expect(bars).toHaveLength(3);
    expect(bars[0]).toMatchObject({ date: "2026-08-29", heightPct: 25, heightPx: 30 });
    expect(bars[1]).toMatchObject({ date: "2026-08-30", heightPct: 100, heightPx: 120 });
    expect(bars[2]).toMatchObject({ date: "2026-08-31", heightPct: 50, heightPx: 60 });
  });

  test("all-zero series produces zero heights without NaN", () => {
    const bars = computeBars([day("2026-08-30", 0), day("2026-08-31", 0)], 120);

    for (const bar of bars) {
      expect(bar.heightPct).toBe(0);
      expect(bar.heightPx).toBe(0);
      expect(Number.isNaN(bar.heightPx)).toBe(false);
    }
  });

  test("empty series returns no bars", () => {
    expect(computeBars([], 120)).toEqual([]);
  });

  test("single day renders at full height", () => {
    const bars = computeBars([day("2026-08-31", 55)], 100);

    expect(bars[0]).toMatchObject({ heightPct: 100, heightPx: 100 });
  });

  test("rounds pixel heights to whole numbers", () => {
    const bars = computeBars([day("2026-08-31", 3)], 100);

    expect(bars[0]!.heightPx).toBe(100);
    const third = computeBars(
      [day("2026-08-31", 1), day("2026-08-30", 3)],
      100,
    );
    expect(third[0]!.heightPx).toBe(33);
  });

  test("preserves token counts for labels", () => {
    const bars = computeBars([day("2026-08-31", 12345)], 120);

    expect(bars[0]).toMatchObject({ tokens: 12345 });
  });
});

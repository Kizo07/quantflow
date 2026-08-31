import type { ConsoleUsageDay } from "./types";

export interface UsageBar {
  /** Local date (YYYY-MM-DD). */
  date: string;
  /** Bar height as a percentage of the series maximum (0-100). */
  heightPct: number;
  /** Bar height in pixels for the requested chart height. */
  heightPx: number;
  /** Token total for this day (label source). */
  tokens: number;
}

/**
 * Convert a daily token series into dependency-free bar geometry.
 *
 * Heights normalize against the series maximum; an all-zero or empty series
 * yields zero heights (never NaN). Pixel heights are rounded to whole
 * numbers so CSS rendering stays crisp.
 */
export function computeBars(
  series: readonly ConsoleUsageDay[],
  heightPx: number,
): UsageBar[] {
  if (series.length === 0) return [];

  const max = series.reduce((acc, d) => Math.max(acc, d.total_tokens), 0);
  return series.map((d) => {
    const ratio = max > 0 ? d.total_tokens / max : 0;
    return {
      date: d.date,
      heightPct: Math.round(ratio * 100),
      heightPx: Math.round(ratio * heightPx),
      tokens: d.total_tokens,
    };
  });
}

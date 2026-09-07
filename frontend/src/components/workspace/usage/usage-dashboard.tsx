"use client";

import { useQuery } from "@tanstack/react-query";

import {
  ConsoleUnavailableError,
  fetchConsoleStats,
  fetchConsoleUsage,
} from "@/core/console/api";
import { computeBars } from "@/core/console/usage-chart";
import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

const USAGE_DAYS = 14;
const CHART_HEIGHT_PX = 120;

function formatNumber(value: number, locale: string): string {
  return new Intl.NumberFormat(locale).format(value);
}

function formatCost(
  cost: number | null,
  currency: string | null,
  locale: string,
  fallback: string,
): string {
  if (cost === null || currency === null) return fallback;
  return new Intl.NumberFormat(locale, {
    style: "currency",
    currency,
    maximumFractionDigits: 4,
  }).format(cost);
}

export function UsageDashboard() {
  const { t, locale } = useI18n();

  const stats = useQuery({
    queryKey: ["consoleStats"],
    queryFn: ({ signal }) => fetchConsoleStats(signal),
    retry: false,
  });

  const usage = useQuery({
    queryKey: ["consoleUsage", USAGE_DAYS],
    queryFn: ({ signal }) =>
      fetchConsoleUsage(
        { days: USAGE_DAYS, tzOffsetMinutes: new Date().getTimezoneOffset() },
        signal,
      ),
    retry: false,
  });

  const unavailable =
    stats.error instanceof ConsoleUnavailableError ||
    usage.error instanceof ConsoleUnavailableError;

  if (unavailable) {
    return (
      <p className="text-muted-foreground text-sm">{t.usage.unavailable}</p>
    );
  }

  const usageData = usage.data;
  const bars = usageData ? computeBars(usageData.days, CHART_HEIGHT_PX) : [];

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatCard
          label={t.usage.totalRuns}
          value={stats.data ? formatNumber(stats.data.total_runs, locale) : "—"}
        />
        <StatCard
          label={t.usage.totalThreads}
          value={
            stats.data ? formatNumber(stats.data.total_threads, locale) : "—"
          }
        />
        <StatCard
          label={t.usage.totalTokens}
          value={
            stats.data ? formatNumber(stats.data.total_tokens, locale) : "—"
          }
        />
        <StatCard
          label={t.usage.totalCost}
          value={formatCost(
            stats.data?.total_cost ?? null,
            stats.data?.currency ?? null,
            locale,
            t.usage.costUnavailable,
          )}
        />
      </div>

      <section>
        <h3 className="text-muted-foreground mb-2 text-sm font-medium">
          {t.usage.lastDays.replace("{days}", String(USAGE_DAYS))}
        </h3>
        {usageData ? (
          <div
            className="flex items-end gap-1"
            role="img"
            aria-label={t.usage.lastDays.replace("{days}", String(USAGE_DAYS))}
          >
            {bars.map((bar) => (
              <div key={bar.date} className="flex flex-1 flex-col items-center">
                <div
                  className="flex w-full items-end justify-center"
                  style={{ height: CHART_HEIGHT_PX }}
                >
                  <div
                    data-usage-bar
                    title={`${bar.date}: ${formatNumber(bar.tokens, locale)}`}
                    className={cn(
                      "bg-primary/70 w-full max-w-8 rounded-t",
                      bar.heightPx === 0 && "bg-muted",
                    )}
                    style={{ height: Math.max(bar.heightPx, 2) }}
                  />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-muted-foreground text-sm">—</p>
        )}
      </section>

      <section>
        <h3 className="text-muted-foreground mb-2 text-sm font-medium">
          {t.usage.byModel}
        </h3>
        {usageData && Object.keys(usageData.by_model).length > 0 ? (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-muted-foreground border-b text-left">
                <th className="py-1.5 pr-2 font-normal">{t.usage.model}</th>
                <th className="py-1.5 pr-2 text-right font-normal">
                  {t.usage.tokens}
                </th>
                <th className="py-1.5 pr-2 text-right font-normal">
                  {t.usage.runs}
                </th>
                <th className="py-1.5 text-right font-normal">
                  {t.usage.totalCost}
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(usageData.by_model).map(([model, breakdown]) => (
                <tr key={model} className="border-b last:border-0">
                  <td className="py-1.5 pr-2">{model}</td>
                  <td className="py-1.5 pr-2 text-right">
                    {formatNumber(breakdown.tokens, locale)}
                  </td>
                  <td className="py-1.5 pr-2 text-right">
                    {formatNumber(breakdown.runs, locale)}
                  </td>
                  <td className="py-1.5 text-right">
                    {formatCost(
                      breakdown.cost,
                      usageData.currency,
                      locale,
                      t.usage.costUnavailable,
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-muted-foreground text-sm">—</p>
        )}
      </section>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-card rounded-lg border p-3">
      <div className="text-muted-foreground text-xs">{label}</div>
      <div className="mt-1 text-xl font-semibold tabular-nums">{value}</div>
    </div>
  );
}

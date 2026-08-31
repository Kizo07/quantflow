import { fetch } from "@/core/api/fetcher";
import { getBackendBaseURL } from "@/core/config";

import type {
  ConsoleRunsResponse,
  ConsoleStatsResponse,
  ConsoleUsageResponse,
} from "./types";

/** Raised when the gateway cannot serve console data (503, e.g. memory backend). */
export class ConsoleUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConsoleUnavailableError";
  }
}

/** Any non-503 console request failure. */
export class ConsoleUsageRequestError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ConsoleUsageRequestError";
    this.status = status;
  }
}

async function readErrorDetail(response: Response): Promise<string> {
  const data = (await response.json().catch(() => ({}))) as {
    detail?: string;
  };
  return data.detail ?? `HTTP ${response.status}: ${response.statusText}`;
}

async function consoleFetch<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    const detail = await readErrorDetail(response);
    if (response.status === 503) {
      throw new ConsoleUnavailableError(detail);
    }
    throw new ConsoleUsageRequestError(response.status, detail);
  }
  return response.json();
}

export async function fetchConsoleStats(
  signal?: AbortSignal,
): Promise<ConsoleStatsResponse> {
  return consoleFetch<ConsoleStatsResponse>(
    `${getBackendBaseURL()}/api/console/stats`,
    signal,
  );
}

export interface ConsoleUsageOptions {
  /** Window size in days (1-90, backend default 14). */
  days?: number;
  /** Local-time offset from UTC in minutes, for day bucketing. */
  tzOffsetMinutes?: number;
}

export async function fetchConsoleUsage(
  options: ConsoleUsageOptions = {},
  signal?: AbortSignal,
): Promise<ConsoleUsageResponse> {
  const params = new URLSearchParams({
    days: String(options.days ?? 14),
    tz_offset_minutes: String(
      options.tzOffsetMinutes ?? new Date().getTimezoneOffset(),
    ),
  });
  return consoleFetch<ConsoleUsageResponse>(
    `${getBackendBaseURL()}/api/console/usage?${params.toString()}`,
    signal,
  );
}

export interface ConsoleRunsOptions {
  limit?: number;
  offset?: number;
  status?: string;
}

export async function fetchConsoleRuns(
  options: ConsoleRunsOptions = {},
  signal?: AbortSignal,
): Promise<ConsoleRunsResponse> {
  const params = new URLSearchParams();
  if (options.limit !== undefined) params.set("limit", String(options.limit));
  if (options.offset !== undefined) {
    params.set("offset", String(options.offset));
  }
  if (options.status !== undefined) params.set("status", options.status);
  const query = params.toString();
  return consoleFetch<ConsoleRunsResponse>(
    `${getBackendBaseURL()}/api/console/runs${query ? `?${query}` : ""}`,
    signal,
  );
}

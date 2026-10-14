import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { fetch as apiFetch } from "@/core/api/fetcher";
import { useAuth } from "@/core/auth/AuthProvider";
import { getBackendBaseURL } from "@/core/config";

import type { HarnessMode } from "./types";

/**
 * Research/Coding harness mode (F2.7), persisted per user.
 *
 * Store (Phase-0 U0.7 decision): the `user_preferences` table via
 * `PATCH /api/v1/auth/preferences` (`harness_mode` key); a per-user
 * localStorage entry is cache/fallback only. Fail-closed default is
 * `research` (the built-in harness) per the U0.8 verdict.
 */
export const HARNESS_MODE_DEFAULT: HarnessMode = "research";

const CACHE_PREFIX = "deerflow.harness-mode.";

function cacheKey(userId: string): string {
  return `${CACHE_PREFIX}${userId}`;
}

export function readCachedHarnessMode(userId: string): HarnessMode | null {
  try {
    const value = window.localStorage.getItem(cacheKey(userId));
    return value === "research" || value === "coding" ? value : null;
  } catch {
    return null;
  }
}

function writeCachedHarnessMode(userId: string, mode: HarnessMode): void {
  try {
    window.localStorage.setItem(cacheKey(userId), mode);
  } catch {
    // Cache failure must never break the toggle; the server stays canonical.
  }
}

function parseHarnessMode(value: unknown): HarnessMode | null {
  if (
    typeof value === "object" &&
    value !== null &&
    (Reflect.get(value, "harness_mode") === "research" ||
      Reflect.get(value, "harness_mode") === "coding")
  ) {
    return Reflect.get(value, "harness_mode") as HarnessMode;
  }
  return null;
}

async function fetchHarnessMode(userId: string): Promise<HarnessMode> {
  try {
    const response = await apiFetch(
      `${getBackendBaseURL()}/api/v1/auth/preferences`,
      {
        method: "GET",
        headers: { "X-Expected-User-Id": userId },
        cache: "no-store",
      },
    );
    if (response.ok) {
      const remote = parseHarnessMode(await response.json());
      if (remote !== null) {
        writeCachedHarnessMode(userId, remote);
        return remote;
      }
    }
  } catch {
    // Offline or sessionless: fall through to the local cache.
  }
  return readCachedHarnessMode(userId) ?? HARNESS_MODE_DEFAULT;
}

async function saveHarnessMode(
  userId: string,
  mode: HarnessMode,
): Promise<void> {
  const response = await apiFetch(
    `${getBackendBaseURL()}/api/v1/auth/preferences`,
    {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        "X-Expected-User-Id": userId,
      },
      body: JSON.stringify({ harness_mode: mode }),
    },
  );
  if (!response.ok) {
    throw new Error(`Saving harness mode failed: ${response.status}`);
  }
}

export function useHarnessMode() {
  const { user } = useAuth();
  const userId = user?.id ?? null;
  const queryClient = useQueryClient();

  const query = useQuery({
    queryKey: ["harness-mode", userId],
    queryFn: () => fetchHarnessMode(userId!),
    enabled: Boolean(userId),
    retry: false,
    staleTime: 300000,
  });

  const mutation = useMutation({
    mutationFn: (mode: HarnessMode) => saveHarnessMode(userId!, mode),
    onSuccess: (_, mode) =>
      queryClient.setQueryData(["harness-mode", userId], mode),
    // On PATCH failure the local cache (written eagerly below) keeps the
    // toggle usable; the next successful read reconciles the server value.
    onError: () => undefined,
  });

  const mode: HarnessMode =
    query.data ??
    (userId ? readCachedHarnessMode(userId) : null) ??
    HARNESS_MODE_DEFAULT;

  const setMode = (next: HarnessMode) => {
    if (!userId || next === mode) return;
    writeCachedHarnessMode(userId, next);
    queryClient.setQueryData(["harness-mode", userId], next);
    mutation.mutate(next);
  };

  return { mode, setMode, isLoading: query.isLoading };
}

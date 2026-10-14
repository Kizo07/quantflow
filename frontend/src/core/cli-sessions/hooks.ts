import {
  useMutation,
  useQuery,
  useQueryClient,
  type QueryClient,
} from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { GatewayApiError } from "@/core/api/errors";

import {
  cliSessionEventsUrl,
  exportCliSession,
  fetchCliPreset,
  fetchCliSession,
  fetchCliSessions,
  inspectCliSession,
  launchCliSession,
  putCliPreset,
  resumeCliSession,
  stopCliSession,
} from "./api";
import {
  CLI_PRESET_MUSE_13,
  isCliSessionActive,
  type CliExportResult,
  type CliInspectResult,
  type CliLaunchPayload,
  type CliPreset,
  type CliPresetPayload,
  type CliResumePayload,
  type CliSession,
} from "./types";

const CLI_SESSIONS_KEY = ["cli-sessions"] as const;
const CLI_PRESETS_KEY = ["cli-presets"] as const;

const ACTIVE_POLL_MS = 5000;
const IDLE_POLL_MS = 30000;

function isNotFound(error: unknown): boolean {
  return error instanceof GatewayApiError && error.status === 404;
}

function invalidateCliSessions(queryClient: QueryClient) {
  void queryClient.invalidateQueries({ queryKey: CLI_SESSIONS_KEY });
}

/** List polling: fast while any session is still active. */
export function useCliSessions() {
  return useQuery({
    queryKey: CLI_SESSIONS_KEY,
    queryFn: fetchCliSessions,
    refetchInterval: (query) =>
      (query.state.data ?? []).some(isCliSessionActive)
        ? ACTIVE_POLL_MS
        : IDLE_POLL_MS,
    refetchIntervalInBackground: false,
  });
}

export function useCliSession(
  sessionId: string | null | undefined,
  { initialData }: { initialData?: CliSession } = {},
) {
  return useQuery({
    queryKey: [...CLI_SESSIONS_KEY, "session", sessionId],
    queryFn: () => fetchCliSession(sessionId ?? ""),
    enabled: Boolean(sessionId),
    initialData,
    retry: (failureCount, error) => !isNotFound(error) && failureCount < 2,
    refetchInterval: (query) => {
      if (isNotFound(query.state.error)) return false;
      const data = query.state.data;
      return data && isCliSessionActive(data) ? ACTIVE_POLL_MS : IDLE_POLL_MS;
    },
    refetchIntervalInBackground: false,
  });
}

export function useLaunchCliSession() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: CliLaunchPayload) => launchCliSession(payload),
    onSuccess: () => invalidateCliSessions(queryClient),
  });
}

export function useResumeCliSession() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      sessionId,
      payload,
    }: {
      sessionId: string;
      payload?: CliResumePayload;
    }) => resumeCliSession(sessionId, payload),
    onSuccess: () => invalidateCliSessions(queryClient),
  });
}

export function useExportCliSession() {
  return useMutation({
    mutationFn: ({
      sessionId,
      redacted,
    }: {
      sessionId: string;
      redacted: boolean;
    }): Promise<CliExportResult> => exportCliSession(sessionId, redacted),
  });
}

export function useInspectCliSession() {
  return useMutation({
    mutationFn: (sessionId: string): Promise<CliInspectResult> =>
      inspectCliSession(sessionId),
  });
}

export function useStopCliSession() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) => stopCliSession(sessionId),
    onSuccess: () => invalidateCliSessions(queryClient),
  });
}

/** The persisted Muse 1.3 launch defaults (built-in fallback server-side). */
export function useCliLaunchPreset(
  model: string | null | undefined,
  { enabled = true }: { enabled?: boolean } = {},
) {
  return useQuery<CliPreset>({
    queryKey: [...CLI_PRESETS_KEY, CLI_PRESET_MUSE_13],
    queryFn: () => fetchCliPreset(CLI_PRESET_MUSE_13),
    enabled: enabled && Boolean(model),
    retry: false,
    staleTime: 60000,
  });
}

export function usePutCliPreset() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      name,
      payload,
    }: {
      name: string;
      payload: CliPresetPayload;
    }) => putCliPreset(name, payload),
    onSuccess: (_, { name }) =>
      void queryClient.invalidateQueries({
        queryKey: [...CLI_PRESETS_KEY, name],
      }),
  });
}

export interface CliSessionEvent {
  seq: number;
  raw: string;
  payload: unknown;
}

function parseEventPayload(raw: string): unknown {
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

const MAX_BUFFERED_EVENTS = 500;

/**
 * Live JSONL events over the session's SSE endpoint (cookie auth via
 * EventSource credentials). Connects only while `active`; the gateway
 * closes the stream with an `end` event carrying the exit code.
 */
export function useCliSessionEvents(
  sessionId: string | null | undefined,
  active: boolean,
) {
  const [events, setEvents] = useState<CliSessionEvent[]>([]);
  const [exitCode, setExitCode] = useState<number | null>(null);
  const [connected, setConnected] = useState(false);
  const [unavailable, setUnavailable] = useState(false);
  const seqRef = useRef(0);

  useEffect(() => {
    setEvents([]);
    setExitCode(null);
    setConnected(false);
    setUnavailable(false);
    seqRef.current = 0;
    if (!active || !sessionId) return;
    if (typeof EventSource === "undefined") {
      setUnavailable(true);
      return;
    }
    const source = new EventSource(cliSessionEventsUrl(sessionId), {
      withCredentials: true,
    });
    const append = (raw: string) => {
      seqRef.current += 1;
      const event: CliSessionEvent = {
        seq: seqRef.current,
        raw,
        payload: parseEventPayload(raw),
      };
      setEvents((previous) =>
        previous.length >= MAX_BUFFERED_EVENTS
          ? [
              ...previous.slice(previous.length - MAX_BUFFERED_EVENTS + 1),
              event,
            ]
          : [...previous, event],
      );
    };
    source.onopen = () => setConnected(true);
    source.onmessage = (message: MessageEvent) => append(String(message.data));
    source.addEventListener("end", (message: Event) => {
      const data = (message as MessageEvent).data;
      try {
        const parsed: unknown = JSON.parse(String(data));
        if (
          typeof parsed === "object" &&
          parsed !== null &&
          typeof Reflect.get(parsed, "exit_code") === "number"
        ) {
          setExitCode(Reflect.get(parsed, "exit_code") as number);
        } else {
          setExitCode(0);
        }
      } catch {
        setExitCode(0);
      }
      source.close();
      setConnected(false);
    });
    source.onerror = () => {
      // A finished run closes the stream; the error event fires on the
      // close. The list poll reconciles the final status independently.
      setConnected(false);
    };
    return () => source.close();
  }, [active, sessionId]);

  return { events, exitCode, connected, unavailable };
}

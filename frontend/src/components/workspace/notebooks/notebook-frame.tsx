"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { fetch as fetchWithAuth } from "@/core/api/fetcher";
import { getBackendBaseURL } from "@/core/config";
import {
  NOTEBOOK_BRIDGE_VERSION,
  buildNotebookFrameUrl,
  buildNotebookMountMessage,
  buildNotebookTokenRefreshMessage,
  isAllowedNotebookOrigin,
  isCurrentMountNonce,
  negotiateNotebookBridgeVersion,
  notebookRefreshDelayMs,
  parseNotebookFrameMessage,
  type NotebookFrameEventMessage,
  type NotebookShellToFrameMessage,
  type NotebookThemeTokens,
} from "@/core/notebooks/bridge";

/** Liveness poll cadence while the frame is mounted (sidecar idle-cull watch). */
export const NOTEBOOK_HEALTH_POLL_INTERVAL_MS = 30_000;

export type NotebookFrameStatus =
  | "probing"
  | "mounting"
  | "mounted"
  | "mismatch"
  | "unreachable";

export type NotebookTokenResponse = {
  token: string;
  token_type: string;
  expires_at: string;
};

export type NotebookHealthResponse = {
  status: "healthy" | "unreachable";
  bridge_contract_version: number;
  sidecar: {
    reachable: boolean;
    component_version: string | null;
    contract_version: number | null;
  } | null;
  /** Operator hint on unreachable (portfile/DEER_FLOW_HOME, cull, …). */
  hint?: string | null;
};

/** The session cookie died: the user must log in again, not just retry. */
export class NotebookUnauthorizedError extends Error {
  constructor(message = "Notebook connector needs re-authentication") {
    super(message);
    this.name = "NotebookUnauthorizedError";
  }
}

function readThemeTokens(): NotebookThemeTokens {
  const style = getComputedStyle(document.documentElement);
  const read = (name: string, fallback: string) =>
    style.getPropertyValue(name).trim() || fallback;
  return {
    background: read("--background", "#ffffff"),
    foreground: read("--foreground", "#000000"),
    muted: read("--muted", "#f4f4f5"),
    accent: read("--accent", "#18181b"),
    border: read("--border", "#e4e4e7"),
    fontFamily: read("--font-sans", "system-ui, sans-serif"),
  };
}

function newNonce(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}

async function fetchConnectorHealth(): Promise<NotebookHealthResponse | null> {
  const res = await fetchWithAuth(
    `${getBackendBaseURL()}/api/notebooks/health`,
  );
  if (res.status === 401) throw new NotebookUnauthorizedError();
  if (res.status === 503) return null;
  if (!res.ok)
    throw new Error(`Notebook connector health failed: ${res.status}`);
  return (await res.json()) as NotebookHealthResponse;
}

async function fetchSidecarToken(): Promise<NotebookTokenResponse> {
  const res = await fetchWithAuth(`${getBackendBaseURL()}/api/notebooks/token`);
  if (res.status === 401) throw new NotebookUnauthorizedError();
  if (!res.ok) throw new Error(`Notebook token mint failed: ${res.status}`);
  return (await res.json()) as NotebookTokenResponse;
}

function isUnauthorized(error: unknown): boolean {
  return error instanceof NotebookUnauthorizedError;
}

/**
 * Embedded ReportForge notebook UI (Phase 3 connector).
 *
 * Mount is health-gated: the iframe mounts automatically when the
 * connector reports healthy, and "Start notebooks" appears only as
 * failure/idle-cull recovery. Auth reaches the sandboxed frame two ways —
 * a `?sidecar_token=` query param on the frame URL (navigations cannot
 * set headers; the D1 subresource bundle loads on the document token
 * alone) plus a postMessage mount delivery the UI uses for API calls,
 * refreshed before expiry. While mounted, connector health is polled so
 * an idle-culled sidecar drops to unreachable + Start; a 401 anywhere
 * means the session cookie died and prompts re-login instead of retry.
 */
export function NotebookFrame({
  notebookId,
  onCoordinationEvent,
  className,
}: {
  notebookId?: string;
  onCoordinationEvent?: (message: NotebookFrameEventMessage) => void;
  className?: string;
}) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const nonceRef = useRef<string>("");
  const refreshTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [status, setStatus] = useState<NotebookFrameStatus>("probing");
  const [frameSrc, setFrameSrc] = useState<string | undefined>(undefined);
  const [banner, setBanner] = useState<string | null>(null);

  const clearRefreshTimer = useCallback(() => {
    if (refreshTimerRef.current !== null) {
      clearTimeout(refreshTimerRef.current);
      refreshTimerRef.current = null;
    }
  }, []);

  const clearPollTimer = useCallback(() => {
    if (pollTimerRef.current !== null) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  const dropToUnreachable = useCallback(
    (message: string | null) => {
      clearRefreshTimer();
      clearPollTimer();
      setBanner(message);
      setStatus("unreachable");
    },
    [clearRefreshTimer, clearPollTimer],
  );

  const postToFrame = useCallback((message: NotebookShellToFrameMessage) => {
    // Target "*" is required: the sandbox-without-allow-same-origin frame
    // has an opaque origin (the artifact-file-preview.tsx precedent).
    iframeRef.current?.contentWindow?.postMessage(message, "*");
  }, []);

  const scheduleTokenRefresh = useCallback(
    (expiresAt: string) => {
      clearRefreshTimer();
      // Generation captured at schedule time: the fetch below awaits, and a
      // re-mount may start a new generation mid-flight. The stale callback
      // must neither post nor reschedule (rescheduling would clobber the new
      // generation's timer).
      const generation = nonceRef.current;
      const delay = notebookRefreshDelayMs(expiresAt);
      const refreshOnce = async () => {
        try {
          const refreshed = await fetchSidecarToken();
          if (nonceRef.current !== generation) return;
          postToFrame(
            buildNotebookTokenRefreshMessage(refreshed.token, generation),
          );
          scheduleTokenRefresh(refreshed.expires_at);
        } catch (error) {
          if (nonceRef.current !== generation) return;
          if (isUnauthorized(error)) {
            dropToUnreachable(
              "Your session expired. Log in again, then use Start notebooks to reconnect.",
            );
            return;
          }
          dropToUnreachable(
            "Notebook session refresh failed. Your edits are safe — use Start notebooks to reconnect.",
          );
        }
      };
      refreshTimerRef.current = setTimeout(() => {
        void refreshOnce();
      }, delay);
    },
    [clearRefreshTimer, dropToUnreachable, postToFrame],
  );

  const mount = useCallback(async () => {
    clearRefreshTimer();
    clearPollTimer();
    setBanner(null);
    setStatus("probing");
    setFrameSrc(undefined);
    try {
      const health = await fetchConnectorHealth();
      if (health?.status !== "healthy" || health.sidecar === null) {
        // 503/refused: the sidecar may have been culled while idle — Start
        // retries the mount. The connector hint (when present) says why.
        dropToUnreachable(health?.hint ?? null);
        return;
      }
      // Contract skew hard-fails with a banner — never a silent mount.
      if (
        health.bridge_contract_version !== NOTEBOOK_BRIDGE_VERSION ||
        health.sidecar.contract_version !== NOTEBOOK_BRIDGE_VERSION
      ) {
        setBanner(
          `Notebook UI contract mismatch (shell ${NOTEBOOK_BRIDGE_VERSION}, sidecar ${String(health.sidecar.contract_version)}). Update QuantFlow or ReportForge so both sides agree.`,
        );
        setStatus("mismatch");
        return;
      }
      const { token, expires_at } = await fetchSidecarToken();
      nonceRef.current = newNonce();
      setFrameSrc(
        buildNotebookFrameUrl(getBackendBaseURL(), token, notebookId),
      );
      setStatus("mounting");
      scheduleTokenRefresh(expires_at);
    } catch (error) {
      // 401: the session cookie died — retrying alone cannot help.
      if (isUnauthorized(error)) {
        dropToUnreachable(
          "Your session expired. Log in again, then use Start notebooks to reconnect.",
        );
        return;
      }
      dropToUnreachable(
        "Could not reach the notebook connector. Use Start notebooks to retry.",
      );
    }
  }, [
    clearPollTimer,
    clearRefreshTimer,
    dropToUnreachable,
    notebookId,
    scheduleTokenRefresh,
  ]);

  // Auto-mount when healthy; "Start notebooks" is failure/idle-cull
  // recovery only, rendered by the unreachable/mismatch states below.
  useEffect(() => {
    void mount();
    return () => {
      clearRefreshTimer();
      clearPollTimer();
    };
  }, [mount, clearRefreshTimer, clearPollTimer]);

  // Liveness: while mounted, poll connector health so an idle-culled or
  // crashed sidecar drops to unreachable + Start instead of stranding a
  // dead frame on screen. A 401 mid-session means the cookie died — the
  // user must log in again, not just retry.
  useEffect(() => {
    if (status !== "mounted") return;
    const poll = async () => {
      try {
        const health = await fetchConnectorHealth();
        if (
          health?.status !== "healthy" ||
          health.sidecar === null ||
          health.bridge_contract_version !== NOTEBOOK_BRIDGE_VERSION ||
          health.sidecar.contract_version !== NOTEBOOK_BRIDGE_VERSION
        ) {
          dropToUnreachable(
            health?.hint ??
              "Notebooks stopped responding. The sidecar may have been culled while idle — use Start notebooks to reconnect.",
          );
        }
      } catch (error) {
        if (isUnauthorized(error)) {
          dropToUnreachable(
            "Your session expired. Log in again, then use Start notebooks to reconnect.",
          );
          return;
        }
        dropToUnreachable(
          "Notebooks stopped responding. The sidecar may have been culled while idle — use Start notebooks to reconnect.",
        );
      }
    };
    pollTimerRef.current = setInterval(
      () => void poll(),
      NOTEBOOK_HEALTH_POLL_INTERVAL_MS,
    );
    return () => clearPollTimer();
  }, [status, clearPollTimer, dropToUnreachable]);

  // Versioned bridge: origin allowlist + schema validation on every
  // inbound message; stale-generation events are dropped.
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (event.source !== iframeRef.current?.contentWindow) return;
      if (!isAllowedNotebookOrigin(event.origin)) return;
      const message = parseNotebookFrameMessage(event.data);
      if (message === null) return;
      if (message.type === "ready") {
        if (
          negotiateNotebookBridgeVersion(message.bridgeVersion) === "mismatch"
        ) {
          clearRefreshTimer();
          setBanner(
            `Notebook UI version mismatch (shell ${NOTEBOOK_BRIDGE_VERSION}, UI ${message.bridgeVersion}). Update QuantFlow or ReportForge so both sides agree.`,
          );
          setStatus("mismatch");
          return;
        }
        // Negotiation succeeded: deliver the token + theme at mount.
        void fetchSidecarToken()
          .then((refreshed) => {
            postToFrame(
              buildNotebookMountMessage(
                refreshed.token,
                nonceRef.current,
                readThemeTokens(),
              ),
            );
            scheduleTokenRefresh(refreshed.expires_at);
            setStatus("mounted");
          })
          .catch((error: unknown) => {
            if (isUnauthorized(error)) {
              dropToUnreachable(
                "Your session expired. Log in again, then use Start notebooks to reconnect.",
              );
              return;
            }
            dropToUnreachable(
              "Notebook session expired. Use Start notebooks to reconnect.",
            );
          });
        return;
      }
      if (!isCurrentMountNonce(message, nonceRef.current)) return;
      onCoordinationEvent?.(message);
    };
    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, [
    clearRefreshTimer,
    dropToUnreachable,
    onCoordinationEvent,
    postToFrame,
    scheduleTokenRefresh,
  ]);

  if (status === "mismatch" || status === "unreachable") {
    return (
      <div className={className}>
        <div className="flex size-full flex-col items-center justify-center gap-4 p-6 text-center">
          <p className="text-muted-foreground max-w-sm text-sm">
            {banner ??
              "Notebooks are unavailable. The sidecar may have been culled while idle."}
          </p>
          <Button onClick={() => void mount()}>Start notebooks</Button>
        </div>
      </div>
    );
  }

  // The frame renders as soon as its URL is known (mounting) so the UI can
  // load and answer the ready handshake; "mounted" only records that the
  // handshake completed. Gating the element on "mounted" would deadlock.
  if (frameSrc === undefined) {
    return (
      <div className={className}>
        <p className="text-muted-foreground text-sm">Starting notebooks…</p>
      </div>
    );
  }

  return (
    <div className={className}>
      {status === "mounting" && (
        <p className="text-muted-foreground px-4 py-2 text-sm">
          Mounting notebook UI…
        </p>
      )}
      {banner !== null && (
        <p role="alert" className="text-destructive px-4 py-2 text-sm">
          {banner}
        </p>
      )}
      <iframe
        ref={iframeRef}
        className="size-full"
        title="Notebook"
        // allow-scripts + allow-forms only: allow-same-origin is
        // deliberately omitted (the artifact-file-preview.tsx precedent,
        // NOT web-preview.tsx) so the frame runs at an opaque origin with
        // no access to parent.document or cookies.
        sandbox="allow-scripts allow-forms"
        src={frameSrc}
      />
    </div>
  );
}

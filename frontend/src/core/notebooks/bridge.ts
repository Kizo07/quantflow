/**
 * Versioned postMessage bridge between the QuantFlow shell and the
 * ReportForge notebook UI iframe (Phase 3 connector).
 *
 * Schema ownership: ReportForge owns the bridge schema and the sidecar
 * API; QuantFlow owns the theme tokens it pushes at mount. This module is
 * QuantFlow's PINNED copy of the contract — the version, message shapes,
 * and origin rules are asserted in frontend tests so shell↔iframe skew
 * hard-fails with a banner instead of drifting silently.
 *
 * The iframe is sandboxed WITHOUT allow-same-origin (opaque origin, the
 * artifact-file-preview.tsx precedent), so inbound events arrive with
 * origin "null" and outbound postMessage calls must target "*".
 */

export const NOTEBOOK_BRIDGE_VERSION = 1;

export const NOTEBOOK_BRIDGE_SOURCE = "reportforge-notebooks";

/**
 * Gateway relay prefix the notebook iframe lives under. The frame URL and
 * every sidecar API base derived from it MUST stay contained here (D2):
 * the shell mounts `${base}${NOTEBOOK_RELAY_PATH_PREFIX}?...` and the
 * sidecar UI derives its API base from its own location, never from an
 * absolute `/api/*` URL — so a relayed page can never address gateway
 * routes outside the notebook connector.
 */
export const NOTEBOOK_RELAY_PATH_PREFIX = "/api/notebooks/";

/**
 * Build the iframe mount URL: always contained under the relay prefix,
 * carrying the single document `sidecar_token` plus the optional
 * `?notebook=<id>` selector (URL-encoded, so ids containing `/` survive
 * as `%2F`). The D1 subresource bundle (app.js, bridge.js, sanitize.js,
 * styles.css) loads relative to this URL on the document token alone.
 */
export function buildNotebookFrameUrl(
  baseUrl: string,
  token: string,
  notebookId?: string,
): string {
  const params = new URLSearchParams({ sidecar_token: token });
  if (notebookId) params.set("notebook", notebookId);
  return `${baseUrl}${NOTEBOOK_RELAY_PATH_PREFIX}?${params.toString()}`;
}

/** Read-only coordination events the shell accepts from the iframe. */
export const NOTEBOOK_COORDINATION_EVENTS = [
  "execution-state",
  "dirty-state",
  "cell-focus",
] as const;

export type NotebookCoordinationEventName =
  (typeof NOTEBOOK_COORDINATION_EVENTS)[number];

export type NotebookThemeTokens = {
  background: string;
  foreground: string;
  muted: string;
  accent: string;
  border: string;
  fontFamily: string;
};

// Shell-to-frame messages the shell actually sends: the mount delivery
// (token + theme at handshake) and token refreshes. Theme travels ONLY
// in the mount message — there is no live "theme" push (dropped: the
// frame re-reads theme on every mount, and an unsent variant is dead
// contract surface the sidecar must not implement against).
export type NotebookShellToFrameMessage =
  | {
      source: typeof NOTEBOOK_BRIDGE_SOURCE;
      version: typeof NOTEBOOK_BRIDGE_VERSION;
      type: "mount";
      token: string;
      nonce: string;
      theme: NotebookThemeTokens;
    }
  | {
      source: typeof NOTEBOOK_BRIDGE_SOURCE;
      version: typeof NOTEBOOK_BRIDGE_VERSION;
      type: "token-refresh";
      token: string;
      nonce: string;
    };

export type NotebookFrameReadyMessage = {
  source: typeof NOTEBOOK_BRIDGE_SOURCE;
  version: typeof NOTEBOOK_BRIDGE_VERSION;
  type: "ready";
  bridgeVersion: number;
  componentVersion: string;
};

export type NotebookFrameEventMessage = {
  source: typeof NOTEBOOK_BRIDGE_SOURCE;
  version: typeof NOTEBOOK_BRIDGE_VERSION;
  type: "event";
  nonce: string;
  name: NotebookCoordinationEventName;
  payload: unknown;
};

export type NotebookFrameToShellMessage =
  | NotebookFrameReadyMessage
  | NotebookFrameEventMessage;

/**
 * True only for the opaque origin a sandbox-without-allow-same-origin
 * iframe posts from. Real origins are rejected — the relayed UI must
 * never run with ambient authority.
 */
export function isAllowedNotebookOrigin(origin: string): boolean {
  return origin === "null";
}

function isRecord(data: unknown): data is Record<string, unknown> {
  return typeof data === "object" && data !== null;
}

function hasBridgeEnvelope(data: unknown): data is Record<string, unknown> & {
  source: string;
  version: number;
  type: string;
} {
  return (
    isRecord(data) &&
    data.source === NOTEBOOK_BRIDGE_SOURCE &&
    typeof data.version === "number" &&
    typeof data.type === "string"
  );
}

/** Validate an inbound iframe message (shape only; version checked separately). */
export function parseNotebookFrameMessage(
  data: unknown,
): NotebookFrameToShellMessage | null {
  if (!hasBridgeEnvelope(data)) return null;
  if (data.version !== NOTEBOOK_BRIDGE_VERSION) return null;
  if (data.type === "ready") {
    if (
      typeof data.bridgeVersion !== "number" ||
      typeof data.componentVersion !== "string"
    ) {
      return null;
    }
    return {
      source: NOTEBOOK_BRIDGE_SOURCE,
      version: NOTEBOOK_BRIDGE_VERSION,
      type: "ready",
      bridgeVersion: data.bridgeVersion,
      componentVersion: data.componentVersion,
    };
  }
  if (data.type === "event") {
    if (
      typeof data.nonce !== "string" ||
      data.nonce === "" ||
      typeof data.name !== "string" ||
      !(NOTEBOOK_COORDINATION_EVENTS as readonly string[]).includes(data.name)
    ) {
      return null;
    }
    return {
      source: NOTEBOOK_BRIDGE_SOURCE,
      version: NOTEBOOK_BRIDGE_VERSION,
      type: "event",
      nonce: data.nonce,
      name: data.name as NotebookCoordinationEventName,
      payload: data.payload,
    };
  }
  return null;
}

/**
 * Negotiation verdict for a `ready` message. A version mismatch is a
 * HARD failure (banner, no token) — never a silent fallback.
 */
export function negotiateNotebookBridgeVersion(
  bridgeVersion: number,
): "match" | "mismatch" {
  return bridgeVersion === NOTEBOOK_BRIDGE_VERSION ? "match" : "mismatch";
}

/** True when an event message belongs to the current mount generation. */
export function isCurrentMountNonce(
  message: Pick<NotebookFrameEventMessage, "nonce">,
  nonce: string,
): boolean {
  return nonce !== "" && message.nonce === nonce;
}

export function buildNotebookMountMessage(
  token: string,
  nonce: string,
  theme: NotebookThemeTokens,
): NotebookShellToFrameMessage {
  return {
    source: NOTEBOOK_BRIDGE_SOURCE,
    version: NOTEBOOK_BRIDGE_VERSION,
    type: "mount",
    token,
    nonce,
    theme,
  };
}

export function buildNotebookTokenRefreshMessage(
  token: string,
  nonce: string,
): NotebookShellToFrameMessage {
  return {
    source: NOTEBOOK_BRIDGE_SOURCE,
    version: NOTEBOOK_BRIDGE_VERSION,
    type: "token-refresh",
    token,
    nonce,
  };
}

/** Refresh margin before `expires_at`: the 5-minute token renews early. */
export const NOTEBOOK_TOKEN_REFRESH_MARGIN_MS = 60_000;

export function notebookRefreshDelayMs(
  expiresAtIso: string,
  nowMs: number = Date.now(),
): number {
  const expiresMs = Date.parse(expiresAtIso);
  if (!Number.isFinite(expiresMs)) return 0;
  return Math.max(0, expiresMs - nowMs - NOTEBOOK_TOKEN_REFRESH_MARGIN_MS);
}

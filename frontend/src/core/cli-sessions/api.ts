import { throwGatewayApiError } from "@/core/api/errors";
import { fetch } from "@/core/api/fetcher";
import { getBackendBaseURL } from "@/core/config";

import type {
  CliExportResult,
  CliInspectResult,
  CliLaunchPayload,
  CliPreset,
  CliPresetPayload,
  CliResumePayload,
  CliSession,
} from "./types";

function cliSessionsUrl(path: string): string {
  return `${getBackendBaseURL()}/api/cli-sessions${path}`;
}

function cliSessionUrl(sessionId: string, suffix = ""): string {
  return cliSessionsUrl(`/${encodeURIComponent(sessionId)}${suffix}`);
}

function cliPresetsUrl(path: string): string {
  return `${getBackendBaseURL()}/api/cli-presets${path}`;
}

const JSON_HEADERS = { "Content-Type": "application/json" };

// Every failure goes through `throwGatewayApiError`, matching the
// scheduled-tasks API module's contract.

export async function fetchCliSessions(): Promise<CliSession[]> {
  const response = await fetch(cliSessionsUrl(""));
  if (!response.ok) {
    await throwGatewayApiError(
      response,
      `Failed to load CLI sessions: ${response.statusText}`,
    );
  }
  return response.json();
}

export async function fetchCliSession(sessionId: string): Promise<CliSession> {
  const response = await fetch(cliSessionUrl(sessionId));
  if (!response.ok) {
    await throwGatewayApiError(
      response,
      `Failed to load CLI session: ${response.statusText}`,
    );
  }
  return response.json();
}

export async function launchCliSession(
  payload: CliLaunchPayload,
): Promise<CliSession> {
  const response = await fetch(cliSessionsUrl(""), {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    await throwGatewayApiError(
      response,
      `Failed to launch CLI session: ${response.statusText}`,
    );
  }
  return response.json();
}

export async function resumeCliSession(
  sessionId: string,
  payload: CliResumePayload = {},
): Promise<CliSession> {
  const response = await fetch(cliSessionUrl(sessionId, "/resume"), {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    await throwGatewayApiError(
      response,
      `Failed to resume CLI session: ${response.statusText}`,
    );
  }
  return response.json();
}

export async function exportCliSession(
  sessionId: string,
  redacted = true,
): Promise<CliExportResult> {
  const response = await fetch(cliSessionUrl(sessionId, "/export"), {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ redacted }),
  });
  if (!response.ok) {
    await throwGatewayApiError(
      response,
      `Failed to export CLI session: ${response.statusText}`,
    );
  }
  return response.json();
}

/** SSE endpoint URL for the live-event hook (EventSource, cookie auth). */
export function cliSessionEventsUrl(sessionId: string): string {
  return cliSessionUrl(sessionId, "/events");
}

export async function inspectCliSession(
  sessionId: string,
): Promise<CliInspectResult> {
  const response = await fetch(cliSessionUrl(sessionId, "/inspect"));
  if (!response.ok) {
    await throwGatewayApiError(
      response,
      `Failed to inspect CLI session: ${response.statusText}`,
    );
  }
  return response.json();
}

export async function stopCliSession(sessionId: string): Promise<CliSession> {
  const response = await fetch(cliSessionUrl(sessionId), {
    method: "DELETE",
  });
  if (!response.ok) {
    await throwGatewayApiError(
      response,
      `Failed to stop CLI session: ${response.statusText}`,
    );
  }
  return response.json();
}

export async function fetchCliPresets(): Promise<CliPreset[]> {
  const response = await fetch(cliPresetsUrl(""));
  if (!response.ok) {
    await throwGatewayApiError(
      response,
      `Failed to load CLI presets: ${response.statusText}`,
    );
  }
  return response.json();
}

export async function fetchCliPreset(name: string): Promise<CliPreset> {
  const response = await fetch(cliPresetsUrl(`/${encodeURIComponent(name)}`));
  if (!response.ok) {
    await throwGatewayApiError(
      response,
      `Failed to load CLI preset: ${response.statusText}`,
    );
  }
  return response.json();
}

export async function putCliPreset(
  name: string,
  payload: CliPresetPayload,
): Promise<CliPreset> {
  const response = await fetch(cliPresetsUrl(`/${encodeURIComponent(name)}`), {
    method: "PUT",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    await throwGatewayApiError(
      response,
      `Failed to save CLI preset: ${response.statusText}`,
    );
  }
  return response.json();
}

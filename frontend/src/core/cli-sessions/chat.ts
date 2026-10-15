/**
 * Pure chat helpers for the CLI streaming chat UI (F2).
 *
 * The gateway's SSE contract (`routers/cli_sessions.py::_event_generator`)
 * emits raw-JSONL `data:` frames plus typed `message` / `tool` frames and a
 * terminal `end` frame. These parsers consume only the typed frames; raw
 * passthrough frames are ignored here because the backfill replay already
 * covers history, and unknown JSONL lines must never break the stream.
 */

import type { CliChatMessage, CliChatSendPayload, CliSession } from "./types";

export interface CliChatMessageFrame {
  kind: "message";
  role: "assistant";
  delta: string;
  run_id: string;
}

export interface CliChatToolFrame {
  kind: "tool";
  run_id: string;
  name: string;
  status: string;
  detail: string;
}

export type CliChatStreamFrame = CliChatMessageFrame | CliChatToolFrame;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function textField(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  return typeof value === "string" ? value : "";
}

/** Parse an `event: message` SSE data payload; null when unrecognized. */
export function parseCliChatMessageFrame(
  data: string,
  fallbackRunId: string,
): CliChatMessageFrame | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(data);
  } catch {
    return null;
  }
  if (!isRecord(parsed)) return null;
  const delta = textField(parsed, "delta");
  if (!delta) return null;
  const runId = textField(parsed, "run_id") || fallbackRunId;
  return { kind: "message", role: "assistant", delta, run_id: runId };
}

/** Parse an `event: tool` SSE data payload; null when unrecognized. */
export function parseCliChatToolFrame(
  data: string,
  fallbackRunId: string,
): CliChatToolFrame | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(data);
  } catch {
    return null;
  }
  if (!isRecord(parsed)) return null;
  const runId = textField(parsed, "run_id") || fallbackRunId;
  return {
    kind: "tool",
    run_id: runId,
    name: textField(parsed, "name"),
    status: textField(parsed, "status"),
    detail: textField(parsed, "detail"),
  };
}

/**
 * Fold one typed frame into the live message list. Assistant deltas extend
 * the trailing assistant bubble of the same run; anything else appends.
 */
export function appendCliChatFrame(
  messages: CliChatMessage[],
  frame: CliChatStreamFrame,
): CliChatMessage[] {
  if (frame.kind === "tool") {
    return [
      ...messages,
      {
        run_id: frame.run_id,
        role: "tool",
        name: frame.name,
        status: frame.status,
        detail: frame.detail,
      },
    ];
  }
  const last = messages[messages.length - 1];
  if (last?.role === "assistant" && last.run_id === frame.run_id) {
    return [
      ...messages.slice(0, -1),
      { ...last, content: last.content + frame.delta },
    ];
  }
  return [
    ...messages,
    { run_id: frame.run_id, role: "assistant", content: frame.delta },
  ];
}

/**
 * Merge backfill history with live stream messages. Backfill text/tool rows
 * for the live run are dropped in favor of the stream (the replay only saw
 * the log as of request time); the run's user prompt is kept.
 */
export function mergeCliChatMessages(
  backfill: CliChatMessage[],
  live: CliChatMessage[],
  liveRunId: string | null,
): CliChatMessage[] {
  if (liveRunId === null) return [...backfill, ...live];
  const kept = backfill.filter(
    (message) => message.run_id !== liveRunId || message.role === "user",
  );
  return [...kept, ...live];
}

export const CLI_CHAT_INHERIT = "__inherit__" as const;
/** Explicit "none / CLI default" option. Never "" (Radix SelectItem rejects it). */
export const CLI_CHAT_NONE = "__none__" as const;

export interface CliChatControlState {
  model: string;
  provider: string;
  effort: string;
  yolo: typeof CLI_CHAT_INHERIT | "on" | "off";
}

export const CLI_CHAT_DEFAULT_CONTROLS: CliChatControlState = {
  model: CLI_CHAT_INHERIT,
  provider: CLI_CHAT_INHERIT,
  effort: CLI_CHAT_INHERIT,
  yolo: CLI_CHAT_INHERIT,
};

export interface CliChatSendInput extends CliChatControlState {
  prompt: string;
}

/** Resolve the effective yolo flag: explicit override wins, else the prior run. */
export function resolveCliChatYolo(
  input: Pick<CliChatControlState, "yolo">,
  prior: Pick<CliSession, "yolo">,
): boolean {
  if (input.yolo === "on") return true;
  if (input.yolo === "off") return false;
  return prior.yolo;
}

export type CliChatSendBuild =
  | { ok: true; payload: CliChatSendPayload; yolo: boolean }
  | { ok: false; error: "empty_prompt" };

/**
 * Build the `POST .../messages` body from composer input. Fail-closed:
 * empty prompts never produce a payload.
 */
export function buildCliChatSendPayload(
  input: CliChatSendInput,
  prior: Pick<CliSession, "yolo">,
): CliChatSendBuild {
  const prompt = input.prompt.trim();
  if (!prompt) return { ok: false, error: "empty_prompt" };
  const yolo = resolveCliChatYolo(input, prior);
  const noneToNull = (value: string): string | null =>
    value === CLI_CHAT_NONE ? null : value || null;
  const payload: CliChatSendPayload = { prompt };
  if (input.model !== CLI_CHAT_INHERIT) payload.model = noneToNull(input.model);
  if (input.provider !== CLI_CHAT_INHERIT) {
    payload.provider = noneToNull(input.provider);
  }
  if (input.effort !== CLI_CHAT_INHERIT)
    payload.effort = noneToNull(input.effort);
  if (input.yolo !== CLI_CHAT_INHERIT) payload.yolo = input.yolo === "on";
  return { ok: true, payload, yolo };
}

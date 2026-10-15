/**
 * Types for the headless CLI session dashboard (Phase 2).
 *
 * Mirrors the gateway's `/api/cli-sessions` + `/api/cli-presets` shapes
 * (`backend/app/gateway/routers/cli_sessions.py`). All identifiers stay
 * model-agnostic (`cli_*`); the single verified model id below comes from
 * Phase-0 evidence (U0.5), not from any external project.
 */

export type CliSessionStatus =
  | "running"
  | "completed"
  | "failed"
  | "stopped"
  | "interrupted";

export type CliSessionMode = "research" | "coding";

/** Research runs the built-in harness; coding runs the CLI backend. */
export type HarnessMode = "research" | "coding";

export interface CliSession {
  id: string;
  session_ref: string | null;
  resumed_from: string | null;
  status: CliSessionStatus;
  mode: CliSessionMode;
  model: string | null;
  provider: string | null;
  preset: string | null;
  effort: string | null;
  contributor_mode: boolean | null;
  approval_mode: string | null;
  yolo: boolean;
  workspace: string | null;
  pid: number | null;
  tmux_session: string | null;
  error: string | null;
  started_at: string | null;
  ended_at: string | null;
  created_at: string | null;
}

export interface CliPreset {
  name: string;
  model: string | null;
  provider: string | null;
  preset: string | null;
  effort: string | null;
  contributor_mode: boolean | null;
  yolo: boolean;
  approval_mode: string | null;
  updated_at: string | null;
}

export interface CliLaunchPayload {
  prompt: string;
  model?: string | null;
  provider?: string | null;
  preset?: string | null;
  effort?: string | null;
  contributor_mode?: boolean | null;
  approval_mode?: string | null;
  sandbox_network?: string | null;
  trust_workspace?: boolean;
  yolo?: boolean;
  workspace?: string | null;
  mode?: CliSessionMode;
}

export interface CliResumePayload {
  prompt?: string | null;
}

/** One chat bubble. Mirrors the backfill message shape in
 * `routers/cli_sessions.py::_replay_run_messages`: text turns carry
 * `content`, tool turns carry `name`/`status`/`detail`. */
export type CliChatRole = "user" | "assistant" | "tool";

export interface CliChatTextMessage {
  run_id: string;
  role: "user" | "assistant";
  content: string;
}

export interface CliChatToolMessage {
  run_id: string;
  role: "tool";
  name: string;
  status: string;
  detail: string;
}

export type CliChatMessage = CliChatTextMessage | CliChatToolMessage;

export interface CliChatBackfill {
  id: string;
  runs: string[];
  messages: CliChatMessage[];
  truncated: boolean;
}

/** Per-send overrides for `POST .../messages`. `undefined` inherits the
 * prior run's flag; only set fields are sent. */
export interface CliChatSendPayload {
  prompt: string;
  model?: string | null;
  provider?: string | null;
  effort?: string | null;
  yolo?: boolean;
}

export interface CliExportResult {
  id: string;
  session_ref: string;
  redacted: boolean;
  document: unknown;
}

export interface CliInspectResult {
  id: string;
  session_ref: string;
  session_log: string;
  output: string;
}

export interface CliPresetPayload {
  model?: string | null;
  provider?: string | null;
  preset?: string | null;
  effort?: string | null;
  contributor_mode?: boolean | null;
  yolo?: boolean;
  approval_mode?: string | null;
}

/**
 * Host CLI capabilities (`GET /api/cli-capabilities`).
 *
 * `models` carries every model id verified on this machine's CLI with the
 * effort tiers verified valid for that model; `providers` carries only the
 * real-model providers on this host (echo/local are never advertised,
 * though the backend still accepts echo for smoke/tests).
 */
export interface CliCapabilityModel {
  id: string;
  label: string;
  effortTiers: string[];
  defaultEffort: string | null;
}

export interface CliCapabilityProvider {
  id: string;
  label: string;
}

export interface CliCapabilities {
  models: CliCapabilityModel[];
  defaultModel: string | null;
  providers: CliCapabilityProvider[];
  defaultProvider: string | null;
}

/** Phase-0 pinned values (U0.5): the verified model id, its preset row,
 *  the top reasoning-effort tier, and the CLI default effort. */
export const CLI_MODEL_MUSE_13 = "muse-spark-1.3";
export const CLI_PRESET_MUSE_13 = "muse-1.3";
export const CLI_TOP_EFFORT = "ultra";
export const CLI_DEFAULT_EFFORT = "high";

export const CLI_EFFORT_TIERS = [
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
  "ultra",
] as const;

export const CLI_PROVIDERS = ["echo", "meta", "local"] as const;

export const CLI_APPROVAL_MODES = ["untrusted", "on-request", "never"] as const;

export const CLI_TERMINAL_STATUSES: ReadonlySet<CliSessionStatus> = new Set([
  "completed",
  "failed",
  "stopped",
  "interrupted",
]);

export function isCliSessionActive(session: CliSession): boolean {
  return !CLI_TERMINAL_STATUSES.has(session.status);
}

/**
 * Effort tiers for a model choice. A null model resolves to the
 * capabilities default model; unknown/custom models and a missing
 * capabilities payload fall back to the global tier list.
 */
export function selectCliEffortTiers(
  capabilities: CliCapabilities | null | undefined,
  model: string | null,
): { tiers: readonly string[]; defaultEffort: string } {
  const fallback = {
    tiers: CLI_EFFORT_TIERS,
    defaultEffort: CLI_DEFAULT_EFFORT,
  };
  if (!capabilities) return fallback;
  const resolved = capabilities.models.find(
    (entry) => entry.id === (model ?? capabilities.defaultModel),
  );
  if (!resolved) return fallback;
  return {
    tiers: resolved.effortTiers,
    defaultEffort: resolved.defaultEffort ?? CLI_DEFAULT_EFFORT,
  };
}

/**
 * Snap a concrete effort value to the model's tiers: keep it when valid,
 * else the model's default tier, else the first advertised tier.
 */
export function coerceCliEffort(
  capabilities: CliCapabilities | null | undefined,
  model: string | null,
  effort: string,
): string {
  const { tiers, defaultEffort } = selectCliEffortTiers(capabilities, model);
  if (tiers.includes(effort)) return effort;
  if (tiers.includes(defaultEffort)) return defaultEffort;
  return tiers[0] ?? CLI_DEFAULT_EFFORT;
}

/**
 * Effort options for a launch selector: the model's advertised tiers plus
 * the currently selected value when it is a globally valid tier the model
 * row does not advertise (e.g. the persisted Muse 1.3 `ultra` seed). The
 * selector must always be able to display the value it would send.
 */
export function selectCliEffortOptions(
  capabilities: CliCapabilities | null | undefined,
  model: string | null,
  current: string,
): readonly string[] {
  const { tiers } = selectCliEffortTiers(capabilities, model);
  if (tiers.includes(current)) return tiers;
  if (!(CLI_EFFORT_TIERS as readonly string[]).includes(current)) return tiers;
  return [...tiers, current];
}

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
  yolo_confirm?: string | null;
  workspace?: string | null;
  mode?: CliSessionMode;
}

export interface CliResumePayload {
  prompt?: string | null;
  yolo_confirm?: string | null;
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

/** Phase-0 pinned values (U0.5): the verified model id, its preset row,
 *  the top reasoning-effort tier, and the CLI default effort. */
export const CLI_MODEL_MUSE_13 = "muse-spark-1.3";
export const CLI_PRESET_MUSE_13 = "muse-1.3";
export const CLI_TOP_EFFORT = "ultra";
export const CLI_DEFAULT_EFFORT = "high";
export const CLI_YOLO_CONFIRM_TOKEN = "I-CONFIRM-YOLO";

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

/** CLI run presets (`--preset`), distinct from launch presets (`CliPreset`). */
export const CLI_RUN_PRESETS = [
  "native-basic",
  "miniswe",
  "openai-apply-patch",
] as const;

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

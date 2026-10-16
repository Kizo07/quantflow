"use client";

import { useEffect, useState } from "react";

import {
  PromptInput,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
  type PromptInputMessage,
} from "@/components/ai-elements/prompt-input";
import { Suggestion, Suggestions } from "@/components/ai-elements/suggestion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tooltip } from "@/components/workspace/tooltip";
import { GatewayApiError } from "@/core/api/errors";
import {
  CLI_CHAT_DEFAULT_CONTROLS,
  CLI_CHAT_INHERIT,
  CLI_CHAT_NONE,
  buildCliChatSendPayload,
  resolveCliChatYolo,
  type CliChatControlState,
} from "@/core/cli-sessions/chat";
import {
  useCliCapabilities,
  useCliLaunchPreset,
  useLaunchCliSession,
  usePutCliPreset,
  useSendCliChatMessage,
} from "@/core/cli-sessions/hooks";
import {
  CLI_APPROVAL_MODES,
  CLI_DEFAULT_EFFORT,
  CLI_MODEL_MUSE_13,
  CLI_PRESET_MUSE_13,
  coerceCliEffort,
  selectCliEffortOptions,
  selectCliEffortTiers,
  type CliCapabilities,
  type CliSession,
} from "@/core/cli-sessions/types";
import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

const CUSTOM_MODEL_VALUE = "__custom__";
const NONE_VALUE = "__none__";

function capabilityLabel(id: string, label: string): string {
  return label === id ? id : `${label} (${id})`;
}

function modelIdForChoice(
  modelChoice: string,
  customModel: string,
): string | null {
  if (modelChoice === CUSTOM_MODEL_VALUE) {
    return customModel.trim() || null;
  }
  return modelChoice === NONE_VALUE ? null : modelChoice;
}

/** Compact footer select mirroring the research composer's picker pills. */
function CliFooterSelect({
  label,
  value,
  onChange,
  options,
  testId,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: Array<{ value: string; label: string }>;
  testId?: string;
}) {
  return (
    <span className="inline-flex min-w-0 items-center gap-1">
      <span aria-hidden="true" className="text-muted-foreground text-xs">
        {label}
      </span>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger
          aria-label={label}
          data-testid={testId}
          className="h-7 w-auto min-w-0 gap-1 px-2 text-xs"
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </span>
  );
}

/** Compact footer toggle mirroring the research composer's tool buttons. */
function CliFooterToggle({
  label,
  checked,
  onChange,
  tooltip,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  tooltip?: string;
}) {
  const control = (
    <span className="inline-flex items-center gap-1.5">
      <Switch aria-label={label} checked={checked} onCheckedChange={onChange} />
      <span aria-hidden="true" className="text-muted-foreground text-xs">
        {label}
      </span>
    </span>
  );
  return tooltip ? <Tooltip content={tooltip}>{control}</Tooltip> : control;
}

function CliCapabilitiesError() {
  const { t } = useI18n();
  return (
    <p
      role="alert"
      data-testid="cli-capabilities-error"
      className="text-destructive text-sm"
    >
      {t.cliSessions.errors.capabilitiesFailed}
    </p>
  );
}

function CliStarterSuggestions({
  onSelect,
}: {
  onSelect: (text: string) => void;
}) {
  const { t } = useI18n();
  return (
    <div
      data-testid="cli-starter-suggestions"
      className="flex items-center justify-center pt-2"
    >
      <Suggestions className="min-h-16 w-full max-w-full justify-center px-4 sm:w-fit sm:px-0">
        {t.cliSessions.chatStarters.map((starter) => (
          <Suggestion
            key={starter}
            suggestion={starter}
            onClick={() => onSelect(starter)}
          />
        ))}
      </Suggestions>
    </div>
  );
}

/**
 * Launch composer: the research welcome-composer shape (centered rounded-2xl
 * bar, footer tools, chips + disclaimer) wired to the coding launch. Owns the
 * 1.3 preset defaults, capability-driven options, and the yolo warning.
 */
export function CliLaunchComposer({
  onStarted,
  isWelcomeMode = false,
  className,
}: {
  onStarted: (session: CliSession) => void;
  isWelcomeMode?: boolean;
  className?: string;
}) {
  const { t } = useI18n();
  const st = t.cliSessions.launch;
  const [prompt, setPrompt] = useState("");
  const [modelChoice, setModelChoice] = useState<string>(CLI_MODEL_MUSE_13);
  const [customModel, setCustomModel] = useState("");
  const [provider, setProvider] = useState<string>(NONE_VALUE);
  const [effort, setEffort] = useState<string>(CLI_DEFAULT_EFFORT);
  const [contributorMode, setContributorMode] = useState<boolean | null>(null);
  const [approvalMode, setApprovalMode] = useState<string>(NONE_VALUE);
  const [sandboxNetwork, setSandboxNetwork] = useState("");
  const [trustWorkspace, setTrustWorkspace] = useState(false);
  const [yolo, setYolo] = useState(false);
  const [presetApplied, setPresetApplied] = useState(false);
  const [presetSaved, setPresetSaved] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  const capabilitiesQuery = useCliCapabilities();
  const capabilities: CliCapabilities | null = capabilitiesQuery.data ?? null;
  const model = modelIdForChoice(modelChoice, customModel);
  const isMuse13 = model === CLI_MODEL_MUSE_13;
  const effortTiers = selectCliEffortOptions(capabilities, model, effort);
  const effectiveEffort = effort === NONE_VALUE ? null : effort;

  const presetQuery = useCliLaunchPreset(model, { enabled: isMuse13 });
  const launch = useLaunchCliSession();
  const savePreset = usePutCliPreset();

  // When the Muse 1.3 model is chosen, contributor/effort/yolo default
  // from the persisted preset (editable per launch); every other model
  // is per-run only and never reads or writes the preset.
  useEffect(() => {
    if (!isMuse13 || !presetQuery.data || presetApplied) return;
    const preset = presetQuery.data;
    if (preset.effort !== null) setEffort(preset.effort);
    if (preset.contributor_mode !== null) {
      setContributorMode(preset.contributor_mode);
    }
    if (preset.yolo) setYolo(true);
    if (preset.provider !== null) setProvider(preset.provider);
    if (preset.approval_mode !== null) setApprovalMode(preset.approval_mode);
    setPresetApplied(true);
  }, [isMuse13, presetQuery.data, presetApplied]);

  useEffect(() => {
    if (!isMuse13) setPresetApplied(false);
    setPresetSaved(false);
  }, [isMuse13]);

  const handleModelChoice = (value: string) => {
    setModelChoice(value);
    const nextModel = modelIdForChoice(value, customModel);
    setEffort((current) =>
      current === NONE_VALUE
        ? current
        : coerceCliEffort(capabilities, nextModel, current),
    );
  };

  const canSubmit = prompt.trim().length > 0 && !launch.isPending;

  // The `preset` launch field stays server-side only: the gateway keeps it
  // as a passthrough for future run presets, but the dashboard no longer
  // sends it.
  const handleStart = (message: PromptInputMessage) => {
    setFormError(null);
    if (message.text.trim().length === 0) {
      setFormError(st.promptRequired);
      return;
    }
    launch.mutate(
      {
        prompt: message.text.trim(),
        model,
        provider: provider === NONE_VALUE ? null : provider,
        effort: effectiveEffort,
        contributor_mode: contributorMode,
        approval_mode: approvalMode === NONE_VALUE ? null : approvalMode,
        sandbox_network: sandboxNetwork.trim() || null,
        trust_workspace: trustWorkspace,
        yolo,
        mode: "coding",
      },
      {
        onSuccess: onStarted,
        onError: (error) =>
          setFormError(
            error instanceof Error
              ? error.message
              : t.cliSessions.errors.launchFailed,
          ),
      },
    );
  };

  const handleSavePreset = () => {
    setFormError(null);
    setPresetSaved(false);
    savePreset.mutate(
      {
        name: CLI_PRESET_MUSE_13,
        payload: {
          model: CLI_MODEL_MUSE_13,
          provider: provider === NONE_VALUE ? null : provider,
          effort: effectiveEffort,
          contributor_mode: contributorMode,
          yolo,
          approval_mode: approvalMode === NONE_VALUE ? null : approvalMode,
        },
      },
      {
        onSuccess: () => setPresetSaved(true),
        onError: (error) =>
          setFormError(
            error instanceof Error
              ? error.message
              : t.cliSessions.errors.presetFailed,
          ),
      },
    );
  };

  return (
    <div
      data-testid="cli-launch-composer"
      className={cn("flex w-full flex-col", className)}
    >
      {isMuse13 && (
        <div
          data-testid="cli-preset-badge"
          className="flex flex-wrap items-center justify-center gap-2 pb-2"
        >
          <Badge variant="secondary">{st.presetBadge}</Badge>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={handleSavePreset}
            disabled={savePreset.isPending}
            data-testid="cli-save-preset"
          >
            {st.savePreset}
          </Button>
          {presetSaved && (
            <span role="status" className="text-muted-foreground text-xs">
              {st.presetSaved}
            </span>
          )}
        </div>
      )}
      <PromptInput
        className="bg-background/85 relative z-10 rounded-2xl backdrop-blur-sm transition-all duration-300 ease-out *:data-[slot='input-group']:rounded-2xl"
        onSubmit={handleStart}
      >
        <PromptInputTextarea
          aria-label={st.prompt}
          placeholder={st.promptPlaceholder}
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
        />
        <PromptInputFooter className="flex flex-wrap gap-2">
          <PromptInputTools className="min-w-0 flex-1 flex-wrap gap-x-3 gap-y-2">
            <CliFooterSelect
              label={st.model}
              value={modelChoice}
              onChange={handleModelChoice}
              options={[
                { value: NONE_VALUE, label: st.modelCliDefault },
                ...(capabilities?.models ?? []).map((entry) => ({
                  value: entry.id,
                  label: capabilityLabel(entry.id, entry.label),
                })),
                { value: CUSTOM_MODEL_VALUE, label: st.modelCustom },
              ]}
            />
            <CliFooterSelect
              label={st.effort}
              value={effort}
              onChange={setEffort}
              options={[
                { value: NONE_VALUE, label: st.effortNone },
                ...effortTiers.map((value) => ({ value, label: value })),
              ]}
            />
            <CliFooterSelect
              label={st.provider}
              value={provider}
              onChange={setProvider}
              options={[
                { value: NONE_VALUE, label: st.providerNone },
                ...(capabilities?.providers ?? []).map((entry) => ({
                  value: entry.id,
                  label: capabilityLabel(entry.id, entry.label),
                })),
              ]}
            />
            <CliFooterSelect
              label={st.approvalMode}
              value={approvalMode}
              onChange={setApprovalMode}
              options={[
                { value: NONE_VALUE, label: st.approvalNone },
                ...CLI_APPROVAL_MODES.map((value) => ({
                  value,
                  label: value,
                })),
              ]}
            />
            <span className="inline-flex min-w-0 items-center gap-1">
              {contributorMode !== null && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-xs"
                  onClick={() => setContributorMode(null)}
                >
                  {st.contributorUnset}
                </Button>
              )}
              <CliFooterToggle
                label={
                  contributorMode === null
                    ? `${st.contributorMode} (${st.contributorUnset})`
                    : st.contributorMode
                }
                checked={contributorMode === true}
                onChange={(checked) => setContributorMode(checked)}
              />
            </span>
            <CliFooterToggle
              label={st.trustWorkspace}
              checked={trustWorkspace}
              onChange={setTrustWorkspace}
            />
            <CliFooterToggle
              label={st.yolo}
              checked={yolo}
              onChange={setYolo}
            />
            <PromptInputSubmit
              aria-label={st.startRun}
              className="rounded-full"
              disabled={!canSubmit}
              status={launch.isPending ? "submitted" : undefined}
            >
              {launch.isPending ? st.submitting : st.startRun}
            </PromptInputSubmit>
          </PromptInputTools>
        </PromptInputFooter>
      </PromptInput>

      {modelChoice === CUSTOM_MODEL_VALUE && (
        <label className="flex flex-col gap-1.5 pt-2 text-sm">
          <span className="font-medium">{st.modelCustom}</span>
          <Input
            aria-label={st.modelCustom}
            placeholder={st.customModelPlaceholder}
            value={customModel}
            onChange={(event) => setCustomModel(event.target.value)}
          />
        </label>
      )}

      <div className="flex flex-col gap-1.5 pt-2">
        <label className="flex flex-col gap-1 text-xs">
          <span className="text-muted-foreground font-medium">
            {st.sandboxNetwork}
          </span>
          <Input
            aria-label={st.sandboxNetwork}
            placeholder={st.sandboxNetworkPlaceholder}
            value={sandboxNetwork}
            onChange={(event) => setSandboxNetwork(event.target.value)}
            className="h-8 text-xs"
          />
        </label>
      </div>

      {capabilitiesQuery.isError && (
        <div className="pt-2">
          <CliCapabilitiesError />
        </div>
      )}

      {yolo && (
        <div
          className="border-destructive/40 bg-destructive/5 mt-2 flex flex-col gap-2 rounded-lg border p-3 text-sm"
          data-testid="cli-yolo-warning"
        >
          <p className="text-muted-foreground text-xs">{st.yoloWarning}</p>
        </div>
      )}

      {formError && (
        <p role="alert" className="text-destructive pt-2 text-sm">
          {formError}
        </p>
      )}

      {isWelcomeMode && (
        <>
          <CliStarterSuggestions onSelect={setPrompt} />
          <p className="text-muted-foreground/67 z-10 px-4 pt-1 text-center text-xs leading-4">
            {t.inputBox.disclaimer}
          </p>
        </>
      )}
    </div>
  );
}

/** Model id the effort tiers resolve against for the current controls. */
function resolveControlModel(
  controls: CliChatControlState,
  session: CliSession,
  customModel: string,
): string | null {
  if (controls.model === CLI_CHAT_INHERIT) return session.model;
  if (controls.model === CLI_CHAT_NONE) return null;
  if (controls.model === CUSTOM_MODEL_VALUE) {
    return customModel.trim() || null;
  }
  return controls.model;
}

/**
 * Follow-up composer: the research thread-composer shape (bottom rounded-2xl
 * bar, footer tools, disclaimer) with per-send inherit overrides. Only set
 * fields are sent; everything else inherits the prior run.
 */
export function CliSendComposer({
  session,
  streaming,
  onSent,
  className,
}: {
  session: CliSession;
  streaming: boolean;
  onSent: (runId: string, prompt: string) => void;
  className?: string;
}) {
  const { t } = useI18n();
  const ct = t.cliSessions.chat;
  const lt = t.cliSessions.launch;
  const send = useSendCliChatMessage();
  const capabilitiesQuery = useCliCapabilities();
  const capabilities: CliCapabilities | null = capabilitiesQuery.data ?? null;

  const [prompt, setPrompt] = useState("");
  const [controls, setControls] = useState<CliChatControlState>(
    CLI_CHAT_DEFAULT_CONTROLS,
  );
  const [customModel, setCustomModel] = useState("");
  const [sendError, setSendError] = useState<string | null>(null);

  const effectiveModel =
    controls.model === CUSTOM_MODEL_VALUE
      ? customModel.trim() || CLI_CHAT_INHERIT
      : controls.model;
  const effectiveYolo = resolveCliChatYolo(controls, session);
  const { tiers: effortTiers, defaultEffort } = selectCliEffortTiers(
    capabilities,
    resolveControlModel(controls, session, customModel),
  );

  const handleSend = (message: PromptInputMessage) => {
    setSendError(null);
    const built = buildCliChatSendPayload(
      { ...controls, model: effectiveModel, prompt: message.text },
      session,
    );
    if (!built.ok) {
      setSendError(ct.promptRequired);
      return;
    }
    setPrompt("");
    send.mutate(
      { sessionId: session.id, payload: built.payload },
      {
        onSuccess: (row) => onSent(row.id, built.payload.prompt),
        onError: (error) => {
          if (error instanceof GatewayApiError && error.status === 409) {
            setSendError(t.cliSessions.errors.chatConflict);
          } else {
            setSendError(
              error instanceof Error && error.message
                ? error.message
                : t.cliSessions.errors.chatSendFailed,
            );
          }
        },
      },
    );
  };

  const setControl = <K extends keyof CliChatControlState>(
    key: K,
    value: CliChatControlState[K],
  ) => setControls((previous) => ({ ...previous, [key]: value }));

  const canSend = !send.isPending && !streaming && prompt.trim().length > 0;

  return (
    <div
      data-testid="cli-send-composer"
      className={cn("flex w-full flex-col", className)}
    >
      <PromptInput
        className="bg-background/85 relative z-10 rounded-2xl backdrop-blur-sm transition-all duration-300 ease-out *:data-[slot='input-group']:rounded-2xl"
        onSubmit={handleSend}
      >
        <PromptInputTextarea
          aria-label={ct.composerPlaceholder}
          placeholder={ct.composerPlaceholder}
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          className="min-h-10"
        />
        <PromptInputFooter className="flex flex-wrap gap-2">
          <PromptInputTools className="min-w-0 flex-1 flex-wrap gap-x-3 gap-y-2">
            <CliFooterSelect
              label={lt.model}
              value={controls.model}
              onChange={(value) => setControl("model", value)}
              options={[
                { value: CLI_CHAT_INHERIT, label: ct.inherit },
                { value: CLI_CHAT_NONE, label: lt.modelCliDefault },
                ...(capabilities?.models ?? []).map((entry) => ({
                  value: entry.id,
                  label: capabilityLabel(entry.id, entry.label),
                })),
                { value: CUSTOM_MODEL_VALUE, label: lt.modelCustom },
              ]}
            />
            <CliFooterSelect
              label={lt.provider}
              value={controls.provider}
              onChange={(value) => setControl("provider", value)}
              options={[
                { value: CLI_CHAT_INHERIT, label: ct.inherit },
                { value: CLI_CHAT_NONE, label: lt.providerNone },
                ...(capabilities?.providers ?? []).map((entry) => ({
                  value: entry.id,
                  label: capabilityLabel(entry.id, entry.label),
                })),
              ]}
            />
            <CliFooterSelect
              label={lt.effort}
              value={controls.effort}
              onChange={(value) => setControl("effort", value)}
              options={[
                { value: CLI_CHAT_INHERIT, label: ct.inherit },
                { value: CLI_CHAT_NONE, label: lt.effortNone },
                ...effortTiers.map((tier) => ({
                  value: tier,
                  label: tier === defaultEffort ? `${tier} (default)` : tier,
                })),
              ]}
            />
            <CliFooterSelect
              label={lt.yolo}
              value={controls.yolo}
              onChange={(value) =>
                setControl(
                  "yolo",
                  value === "on" || value === "off" ? value : CLI_CHAT_INHERIT,
                )
              }
              options={[
                { value: CLI_CHAT_INHERIT, label: ct.inherit },
                { value: "on", label: ct.yoloOn },
                { value: "off", label: ct.yoloOff },
              ]}
            />
            <PromptInputSubmit
              aria-label={ct.send}
              className="rounded-full"
              disabled={!canSend}
              status={send.isPending ? "submitted" : undefined}
            >
              {send.isPending ? ct.sending : ct.send}
            </PromptInputSubmit>
          </PromptInputTools>
        </PromptInputFooter>
      </PromptInput>

      {controls.model === CUSTOM_MODEL_VALUE && (
        <input
          aria-label={lt.customModelPlaceholder}
          placeholder={lt.customModelPlaceholder}
          value={customModel}
          onChange={(event) => setCustomModel(event.target.value)}
          className="border-input bg-background mt-2 h-8 rounded-md border px-2 text-xs"
        />
      )}

      {streaming && (
        <p role="status" className="text-muted-foreground pt-2 text-xs">
          {ct.activeHint}
        </p>
      )}

      {effectiveYolo && (
        <p
          role="note"
          data-testid="cli-yolo-warning"
          className="border-destructive/40 bg-destructive/5 mt-2 rounded-md border px-2 py-1 text-xs"
        >
          {lt.yoloWarning}
        </p>
      )}

      {capabilitiesQuery.isError && (
        <div className="pt-2">
          <CliCapabilitiesError />
        </div>
      )}

      {sendError && (
        <p role="alert" className="text-destructive pt-2 text-sm">
          {sendError}
        </p>
      )}

      <p className="text-muted-foreground/67 z-10 px-4 pt-1 text-center text-xs leading-4">
        {t.inputBox.disclaimer}
      </p>
    </div>
  );
}

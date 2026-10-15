"use client";

import { useEffect, useState } from "react";

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
import { Textarea } from "@/components/ui/textarea";
import {
  useCliCapabilities,
  useCliLaunchPreset,
  useLaunchCliSession,
  usePutCliPreset,
} from "@/core/cli-sessions/hooks";
import {
  CLI_APPROVAL_MODES,
  CLI_DEFAULT_EFFORT,
  CLI_MODEL_MUSE_13,
  CLI_PRESET_MUSE_13,
  coerceCliEffort,
  selectCliEffortOptions,
  type CliCapabilities,
  type CliLaunchPayload,
  type CliSession,
} from "@/core/cli-sessions/types";
import { useI18n } from "@/core/i18n/hooks";

const CUSTOM_MODEL_VALUE = "__custom__";
const NONE_VALUE = "__none__";

function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: React.ReactNode;
  hint?: string;
}) {
  return (
    <label className="flex flex-col gap-1.5 text-sm">
      <span className="font-medium">{label}</span>
      {children}
      {hint ? (
        <span className="text-muted-foreground text-xs">{hint}</span>
      ) : null}
    </label>
  );
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

function capabilityLabel(id: string, label: string): string {
  return label === id ? id : `${label} (${id})`;
}

export function CliLaunchForm({
  onLaunched,
  onCancel,
}: {
  onLaunched: (session: CliSession) => void;
  onCancel: () => void;
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
  const buildPayload = (): CliLaunchPayload => ({
    prompt: prompt.trim(),
    model,
    provider: provider === NONE_VALUE ? null : provider,
    effort: effectiveEffort,
    contributor_mode: contributorMode,
    approval_mode: approvalMode === NONE_VALUE ? null : approvalMode,
    sandbox_network: sandboxNetwork.trim() || null,
    trust_workspace: trustWorkspace,
    yolo,
    mode: "coding",
  });

  const handleLaunch = () => {
    setFormError(null);
    if (prompt.trim().length === 0) {
      setFormError(st.promptRequired);
      return;
    }
    launch.mutate(buildPayload(), {
      onSuccess: onLaunched,
      onError: (error) =>
        setFormError(
          error instanceof Error
            ? error.message
            : t.cliSessions.errors.launchFailed,
        ),
    });
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
    <div className="flex flex-col gap-4">
      <Field label={st.prompt}>
        <Textarea
          aria-label={st.prompt}
          placeholder={st.promptPlaceholder}
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          rows={4}
        />
      </Field>

      <div className="grid gap-4 sm:grid-cols-2">
        <Field label={st.model}>
          <Select value={modelChoice} onValueChange={handleModelChoice}>
            <SelectTrigger aria-label={st.model}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={NONE_VALUE}>{st.modelCliDefault}</SelectItem>
              {(capabilities?.models ?? []).map((entry) => (
                <SelectItem key={entry.id} value={entry.id}>
                  {capabilityLabel(entry.id, entry.label)}
                </SelectItem>
              ))}
              <SelectItem value={CUSTOM_MODEL_VALUE}>
                {st.modelCustom}
              </SelectItem>
            </SelectContent>
          </Select>
        </Field>
        <Field label={st.provider}>
          <Select value={provider} onValueChange={setProvider}>
            <SelectTrigger aria-label={st.provider}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={NONE_VALUE}>{st.providerNone}</SelectItem>
              {(capabilities?.providers ?? []).map((entry) => (
                <SelectItem key={entry.id} value={entry.id}>
                  {capabilityLabel(entry.id, entry.label)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
      </div>

      {capabilitiesQuery.isError && (
        <p
          role="alert"
          data-testid="cli-capabilities-error"
          className="text-destructive text-sm"
        >
          {t.cliSessions.errors.capabilitiesFailed}
        </p>
      )}

      {modelChoice === CUSTOM_MODEL_VALUE && (
        <Field label={st.modelCustom}>
          <Input
            aria-label={st.modelCustom}
            placeholder={st.customModelPlaceholder}
            value={customModel}
            onChange={(event) => setCustomModel(event.target.value)}
          />
        </Field>
      )}

      {isMuse13 && (
        <div data-testid="cli-preset-badge">
          <Badge variant="secondary">{st.presetBadge}</Badge>
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2">
        <Field label={st.effort}>
          <Select value={effort} onValueChange={setEffort}>
            <SelectTrigger aria-label={st.effort}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={NONE_VALUE}>{st.effortNone}</SelectItem>
              {effortTiers.map((value) => (
                <SelectItem key={value} value={value}>
                  {value}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <Field label={st.approvalMode}>
          <Select value={approvalMode} onValueChange={setApprovalMode}>
            <SelectTrigger aria-label={st.approvalMode}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={NONE_VALUE}>{st.approvalNone}</SelectItem>
              {CLI_APPROVAL_MODES.map((value) => (
                <SelectItem key={value} value={value}>
                  {value}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Field label={st.sandboxNetwork}>
          <Input
            aria-label={st.sandboxNetwork}
            placeholder={st.sandboxNetworkPlaceholder}
            value={sandboxNetwork}
            onChange={(event) => setSandboxNetwork(event.target.value)}
          />
        </Field>
      </div>

      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between gap-3 text-sm">
          <span className="font-medium">
            {st.contributorMode}
            {contributorMode === null && (
              <span className="text-muted-foreground font-normal">
                {" "}
                ({st.contributorUnset})
              </span>
            )}
          </span>
          <div className="flex items-center gap-2">
            {contributorMode !== null && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => setContributorMode(null)}
              >
                {st.contributorUnset}
              </Button>
            )}
            <Switch
              aria-label={st.contributorMode}
              checked={contributorMode === true}
              onCheckedChange={(checked) => setContributorMode(checked)}
            />
          </div>
        </div>
        <div className="flex items-center justify-between gap-3 text-sm">
          <span className="font-medium">{st.trustWorkspace}</span>
          <Switch
            aria-label={st.trustWorkspace}
            checked={trustWorkspace}
            onCheckedChange={setTrustWorkspace}
          />
        </div>
        <div className="flex items-center justify-between gap-3 text-sm">
          <span className="font-medium">{st.yolo}</span>
          <Switch
            aria-label={st.yolo}
            checked={yolo}
            onCheckedChange={setYolo}
          />
        </div>
      </div>

      {yolo && (
        <div
          className="border-destructive/40 bg-destructive/5 flex flex-col gap-2 rounded-lg border p-3 text-sm"
          data-testid="cli-yolo-warning"
        >
          <p className="text-muted-foreground text-xs">{st.yoloWarning}</p>
        </div>
      )}

      {formError && (
        <p role="alert" className="text-destructive text-sm">
          {formError}
        </p>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <Button type="button" onClick={handleLaunch} disabled={!canSubmit}>
          {launch.isPending ? st.submitting : st.submit}
        </Button>
        <Button type="button" variant="outline" onClick={onCancel}>
          {st.cancel}
        </Button>
        {isMuse13 && (
          <Button
            type="button"
            variant="secondary"
            onClick={handleSavePreset}
            disabled={savePreset.isPending}
            data-testid="cli-save-preset"
          >
            {st.savePreset}
          </Button>
        )}
        {presetSaved && (
          <span role="status" className="text-muted-foreground text-sm">
            {st.presetSaved}
          </span>
        )}
      </div>
    </div>
  );
}

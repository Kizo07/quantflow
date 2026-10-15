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
import {
  useCliCapabilities,
  useCliLaunchPreset,
  useLaunchCliSession,
} from "@/core/cli-sessions/hooks";
import {
  CLI_DEFAULT_EFFORT,
  CLI_MODEL_MUSE_13,
  coerceCliEffort,
  selectCliEffortOptions,
  type CliCapabilities,
  type CliSession,
} from "@/core/cli-sessions/types";
import { useI18n } from "@/core/i18n/hooks";

const CUSTOM_MODEL_VALUE = "__custom__";
const NONE_VALUE = "__none__";

/**
 * Welcome-mode start panel for the CLI chat: the research tab's centered
 * composer shape (prompt + model/effort + toggles + start action), wired to
 * the coding launch. Shown when no CLI session is selected.
 */
export function CliStartPanel({
  onStarted,
}: {
  onStarted: (session: CliSession) => void;
}) {
  const { t } = useI18n();
  const st = t.cliSessions.launch;
  const [prompt, setPrompt] = useState("");
  const [modelChoice, setModelChoice] = useState<string>(CLI_MODEL_MUSE_13);
  const [customModel, setCustomModel] = useState("");
  const [effort, setEffort] = useState<string>(CLI_DEFAULT_EFFORT);
  const [contributorMode, setContributorMode] = useState<boolean | null>(null);
  const [trustWorkspace, setTrustWorkspace] = useState(false);
  const [yolo, setYolo] = useState(false);
  const [presetApplied, setPresetApplied] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  const capabilitiesQuery = useCliCapabilities();
  const capabilities: CliCapabilities | null = capabilitiesQuery.data ?? null;
  const model =
    modelChoice === CUSTOM_MODEL_VALUE
      ? customModel.trim() || null
      : modelChoice === NONE_VALUE
        ? null
        : modelChoice;
  const isMuse13 = model === CLI_MODEL_MUSE_13;
  const effortTiers = selectCliEffortOptions(capabilities, model, effort);

  const presetQuery = useCliLaunchPreset(model, { enabled: isMuse13 });
  const launch = useLaunchCliSession();

  useEffect(() => {
    if (!isMuse13 || !presetQuery.data || presetApplied) return;
    const preset = presetQuery.data;
    if (preset.effort !== null) setEffort(preset.effort);
    if (preset.contributor_mode !== null) {
      setContributorMode(preset.contributor_mode);
    }
    if (preset.yolo) setYolo(true);
    setPresetApplied(true);
  }, [isMuse13, presetQuery.data, presetApplied]);

  useEffect(() => {
    if (!isMuse13) setPresetApplied(false);
  }, [isMuse13]);

  const handleModelChoice = (value: string) => {
    setModelChoice(value);
    const nextModel =
      value === CUSTOM_MODEL_VALUE
        ? customModel.trim() || null
        : value === NONE_VALUE
          ? null
          : value;
    setEffort((current) =>
      current === NONE_VALUE
        ? current
        : coerceCliEffort(capabilities, nextModel, current),
    );
  };

  const canStart = prompt.trim().length > 0 && !launch.isPending;

  const handleStart = (message: PromptInputMessage) => {
    setStartError(null);
    if (message.text.trim().length === 0) {
      setStartError(st.promptRequired);
      return;
    }
    launch.mutate(
      {
        prompt: message.text.trim(),
        model,
        effort: effort === NONE_VALUE ? null : effort,
        contributor_mode: contributorMode,
        trust_workspace: trustWorkspace,
        yolo,
        mode: "coding",
      },
      {
        onSuccess: onStarted,
        onError: (error) =>
          setStartError(
            error instanceof Error
              ? error.message
              : t.cliSessions.errors.launchFailed,
          ),
      },
    );
  };

  return (
    <div
      data-testid="cli-start-panel"
      className="mx-auto flex w-full max-w-(--container-width-sm) min-w-0 flex-col gap-4 py-6"
    >
      <div className="flex min-w-0 flex-col gap-1 text-center">
        <h2 className="text-xl font-semibold">{st.title}</h2>
        <p className="text-muted-foreground text-sm">{st.description}</p>
      </div>

      {isMuse13 && (
        <div data-testid="cli-preset-badge" className="self-center">
          <Badge variant="secondary">{st.presetBadge}</Badge>
        </div>
      )}

      <PromptInput onSubmit={handleStart}>
        <PromptInputTextarea
          aria-label={st.prompt}
          placeholder={st.promptPlaceholder}
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
        />
        <PromptInputFooter>
          <PromptInputTools />
          <PromptInputSubmit
            aria-label={st.startRun}
            size="sm"
            disabled={!canStart}
            status={launch.isPending ? "submitted" : undefined}
          >
            {launch.isPending ? st.submitting : st.startRun}
          </PromptInputSubmit>
        </PromptInputFooter>
      </PromptInput>

      <div className="grid gap-4 sm:grid-cols-2">
        <label className="flex flex-col gap-1.5 text-sm">
          <span className="font-medium">{st.model}</span>
          <Select value={modelChoice} onValueChange={handleModelChoice}>
            <SelectTrigger aria-label={st.model}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={NONE_VALUE}>{st.modelCliDefault}</SelectItem>
              {(capabilities?.models ?? []).map((entry) => (
                <SelectItem key={entry.id} value={entry.id}>
                  {entry.label === entry.id
                    ? entry.id
                    : `${entry.label} (${entry.id})`}
                </SelectItem>
              ))}
              <SelectItem value={CUSTOM_MODEL_VALUE}>
                {st.modelCustom}
              </SelectItem>
            </SelectContent>
          </Select>
        </label>
        <label className="flex flex-col gap-1.5 text-sm">
          <span className="font-medium">{st.effort}</span>
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
        </label>
      </div>

      {modelChoice === CUSTOM_MODEL_VALUE && (
        <label className="flex flex-col gap-1.5 text-sm">
          <span className="font-medium">{st.modelCustom}</span>
          <Input
            aria-label={st.modelCustom}
            placeholder={st.customModelPlaceholder}
            value={customModel}
            onChange={(event) => setCustomModel(event.target.value)}
          />
        </label>
      )}

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

      {startError && (
        <p role="alert" className="text-destructive text-center text-sm">
          {startError}
        </p>
      )}
    </div>
  );
}

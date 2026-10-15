"use client";

import { useEffect, useState } from "react";

import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Message, MessageContent } from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
  type PromptInputMessage,
} from "@/components/ai-elements/prompt-input";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { GatewayApiError } from "@/core/api/errors";
import {
  CLI_CHAT_DEFAULT_CONTROLS,
  CLI_CHAT_INHERIT,
  CLI_CHAT_NONE,
  buildCliChatSendPayload,
  mergeCliChatMessages,
  resolveCliChatYolo,
  type CliChatControlState,
} from "@/core/cli-sessions/chat";
import {
  useCliCapabilities,
  useCliChatMessages,
  useCliChatStream,
  useSendCliChatMessage,
  useStopCliSession,
} from "@/core/cli-sessions/hooks";
import {
  isCliSessionActive,
  selectCliEffortTiers,
  type CliCapabilities,
  type CliChatMessage,
  type CliSession,
} from "@/core/cli-sessions/types";
import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

const CUSTOM_MODEL_VALUE = "__custom__";

function messageLabel(
  message: CliChatMessage,
  labels: { user: string; assistant: string; tool: string },
): string {
  if (message.role === "user") return labels.user;
  if (message.role === "tool") {
    return message.name ? `${labels.tool}: ${message.name}` : labels.tool;
  }
  return labels.assistant;
}

export function CliChatMessageList({
  messages,
  truncated,
}: {
  messages: CliChatMessage[];
  truncated: boolean;
}) {
  const { t } = useI18n();
  const ct = t.cliSessions.chat;
  if (messages.length === 0) {
    return (
      <ConversationEmptyState
        role="status"
        data-testid="cli-chat-empty"
        title={ct.empty}
        description=""
      />
    );
  }
  return (
    <div className="flex flex-col gap-2">
      {truncated && (
        <p
          role="note"
          data-testid="cli-chat-truncated"
          className="text-muted-foreground text-xs"
        >
          {ct.truncated}
        </p>
      )}
      <ol data-testid="cli-chat-messages" className="flex flex-col gap-4">
        {messages.map((message, index) => (
          <li
            // Live deltas mutate the trailing bubble in place, so a positional
            // key keeps React from remounting the streaming bubble.
            key={`${message.run_id}:${index}`}
            data-testid={`cli-chat-message-${message.role}`}
          >
            <Message
              from={message.role === "user" ? "user" : "assistant"}
              className="gap-1"
            >
              <MessageContent
                className={cn(
                  "text-sm break-words whitespace-pre-wrap",
                  message.role === "tool" && "font-mono text-xs",
                )}
              >
                <span className="mb-0.5 block text-[11px] font-semibold tracking-wide uppercase opacity-70">
                  {messageLabel(message, {
                    user: ct.userLabel,
                    assistant: ct.assistantLabel,
                    tool: ct.toolLabel,
                  })}
                  {message.role === "tool" && message.status
                    ? ` · ${message.status}`
                    : ""}
                </span>
                {message.role === "tool"
                  ? message.detail || "—"
                  : message.content}
              </MessageContent>
            </Message>
          </li>
        ))}
      </ol>
    </div>
  );
}

function ControlSelect({
  label,
  value,
  onChange,
  options,
  ariaLabel,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: Array<{ value: string; label: string }>;
  ariaLabel: string;
}) {
  return (
    <label className="flex min-w-32 flex-1 flex-col gap-1 text-xs">
      <span className="text-muted-foreground font-medium">{label}</span>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger aria-label={ariaLabel} className="h-8 text-xs">
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
    </label>
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

export function CliChatView({
  session,
  onSelectRun,
}: {
  session: CliSession;
  onSelectRun: (runId: string) => void;
}) {
  const { t } = useI18n();
  const ct = t.cliSessions.chat;
  const lt = t.cliSessions.launch;
  const st = t.cliSessions.detail;
  const backfill = useCliChatMessages(session.id);
  const send = useSendCliChatMessage();
  const stop = useStopCliSession();
  const capabilitiesQuery = useCliCapabilities();
  const capabilities: CliCapabilities | null = capabilitiesQuery.data ?? null;

  const [liveRunId, setLiveRunId] = useState<string | null>(null);
  const [pendingPrompt, setPendingPrompt] = useState<{
    runId: string;
    content: string;
  } | null>(null);
  const [prompt, setPrompt] = useState("");
  const [controls, setControls] = useState<CliChatControlState>(
    CLI_CHAT_DEFAULT_CONTROLS,
  );
  const [customModel, setCustomModel] = useState("");
  const [sendError, setSendError] = useState<string | null>(null);

  const sessionActive = isCliSessionActive(session);
  const streamTarget = liveRunId ?? (sessionActive ? session.id : null);
  const stream = useCliChatStream(streamTarget);
  const streaming = streamTarget !== null && stream.exitCode === null;

  // When a follow-up run finishes, jump to it: the new row is the chat's
  // leaf, so its backfill shows the full turn sequence.
  useEffect(() => {
    if (stream.exitCode !== null && liveRunId !== null) {
      const finished = liveRunId;
      setLiveRunId(null);
      setPendingPrompt(null);
      onSelectRun(finished);
    }
  }, [liveRunId, onSelectRun, stream.exitCode]);

  const backfillMessages = backfill.data?.messages ?? [];
  // Optimistic user bubble: the stream yields assistant/tool frames only,
  // so the just-sent prompt would be invisible until the end+jump.
  const liveMessages: CliChatMessage[] =
    pendingPrompt !== null && pendingPrompt.runId === streamTarget
      ? [
          {
            run_id: pendingPrompt.runId,
            role: "user",
            content: pendingPrompt.content,
          },
          ...stream.messages,
        ]
      : stream.messages;
  const visibleMessages = mergeCliChatMessages(
    backfillMessages,
    liveMessages,
    streamTarget,
  );

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
        onSuccess: (row) => {
          setLiveRunId(row.id);
          setPendingPrompt({ runId: row.id, content: built.payload.prompt });
        },
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

  const handleStop = () => {
    if (!streamTarget) return;
    setSendError(null);
    stop.mutate(streamTarget, {
      onError: (error) =>
        setSendError(
          error instanceof Error && error.message
            ? error.message
            : t.cliSessions.errors.stopFailed,
        ),
    });
  };

  const setControl = <K extends keyof CliChatControlState>(
    key: K,
    value: CliChatControlState[K],
  ) => setControls((previous) => ({ ...previous, [key]: value }));

  const canSend = !send.isPending && !streaming && prompt.trim().length > 0;

  return (
    <div className="flex min-w-0 flex-col gap-3" data-testid="cli-chat-view">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">
          {ct.messages}
          <span className="text-muted-foreground ml-2 font-normal">
            {stream.unavailable
              ? ct.unavailable
              : streaming
                ? stream.connected
                  ? ct.streaming
                  : ct.messages
                : stream.exitCode !== null
                  ? `${ct.finished} (${stream.exitCode})`
                  : ""}
          </span>
        </h2>
        <div className="flex items-center gap-2">
          {streamTarget && !stream.connected && stream.exitCode === null && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={stream.reconnect}
            >
              {ct.reconnect}
            </Button>
          )}
          {streaming && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleStop}
              disabled={stop.isPending}
            >
              {stop.isPending ? st.stopping : st.stop}
            </Button>
          )}
        </div>
      </div>

      {backfill.isPending ? (
        <p role="status" className="text-muted-foreground text-sm">
          {ct.loading}
        </p>
      ) : backfill.isError ? (
        <p role="alert" className="text-destructive text-sm">
          {t.cliSessions.errors.chatLoadFailed}
        </p>
      ) : (
        <Conversation className="max-h-[46vh] min-h-40 rounded-lg border">
          <ConversationContent className="gap-4 p-3">
            <CliChatMessageList
              messages={visibleMessages}
              truncated={backfill.data?.truncated ?? false}
            />
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>
      )}

      {streaming && (
        <p role="status" className="text-muted-foreground text-xs">
          {ct.activeHint}
        </p>
      )}

      <div className="flex flex-col gap-2 rounded-lg border p-3">
        <span className="text-xs font-semibold">{ct.controls}</span>
        <div className="flex flex-wrap gap-2">
          <ControlSelect
            label={lt.model}
            ariaLabel={lt.model}
            value={controls.model}
            onChange={(value) => setControl("model", value)}
            options={[
              { value: CLI_CHAT_INHERIT, label: ct.inherit },
              { value: CLI_CHAT_NONE, label: lt.modelCliDefault },
              ...(capabilities?.models ?? []).map((entry) => ({
                value: entry.id,
                label: entry.label === entry.id ? entry.id : entry.label,
              })),
              { value: CUSTOM_MODEL_VALUE, label: lt.modelCustom },
            ]}
          />
          <ControlSelect
            label={lt.provider}
            ariaLabel={lt.provider}
            value={controls.provider}
            onChange={(value) => setControl("provider", value)}
            options={[
              { value: CLI_CHAT_INHERIT, label: ct.inherit },
              { value: CLI_CHAT_NONE, label: lt.providerNone },
              ...(capabilities?.providers ?? []).map((entry) => ({
                value: entry.id,
                label: entry.label === entry.id ? entry.id : entry.label,
              })),
            ]}
          />
          <ControlSelect
            label={lt.effort}
            ariaLabel={lt.effort}
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
          <ControlSelect
            label={lt.yolo}
            ariaLabel={lt.yolo}
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
        </div>
        {controls.model === CUSTOM_MODEL_VALUE && (
          <input
            aria-label={lt.customModelPlaceholder}
            placeholder={lt.customModelPlaceholder}
            value={customModel}
            onChange={(event) => setCustomModel(event.target.value)}
            className="border-input bg-background h-8 rounded-md border px-2 text-xs"
          />
        )}
        {effectiveYolo && (
          <p
            role="note"
            className="border-destructive/40 bg-destructive/5 rounded-md border px-2 py-1 text-xs"
          >
            {lt.yoloWarning}
          </p>
        )}
        <PromptInput onSubmit={handleSend}>
          <PromptInputTextarea
            aria-label={ct.composerPlaceholder}
            placeholder={ct.composerPlaceholder}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            className="min-h-10"
          />
          <PromptInputFooter>
            <PromptInputTools />
            <PromptInputSubmit
              aria-label={ct.send}
              size="sm"
              disabled={!canSend}
            >
              {ct.send}
            </PromptInputSubmit>
          </PromptInputFooter>
        </PromptInput>
      </div>

      {sendError && (
        <p role="alert" className="text-destructive text-sm">
          {sendError}
        </p>
      )}
    </div>
  );
}

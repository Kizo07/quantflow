"use client";

import type { Message } from "@langchain/langgraph-sdk";
import { useEffect, useMemo, useState, type ComponentProps } from "react";

import { ConversationEmptyState } from "@/components/ai-elements/conversation";
import { Button } from "@/components/ui/button";
import { ArtifactsProvider } from "@/components/workspace/artifacts";
import {
  MESSAGE_LIST_DEFAULT_PADDING_BOTTOM,
  MessageList,
} from "@/components/workspace/messages";
import { MessageListSkeleton } from "@/components/workspace/messages/skeleton";
import { mergeCliChatMessages } from "@/core/cli-sessions/chat";
import {
  useCliChatMessages,
  useCliChatStream,
  useStopCliSession,
} from "@/core/cli-sessions/hooks";
import {
  isCliSessionActive,
  type CliChatMessage,
  type CliSession,
} from "@/core/cli-sessions/types";
import { useI18n } from "@/core/i18n/hooks";

import { CliSendComposer } from "./cli-composer";

/** Tool statuses that mark the paired result an error (research `Error` label). */
const CLI_TOOL_ERROR_STATUS = /fail|error|reject/i;

/**
 * Adapt CLI chat turns to LangGraph messages so the conversation renders
 * through the research tab's actual `MessageList` (user right-pill +
 * assistant markdown + condensed tool steps). Tool turns become native
 * `ai`-with-`tool_calls` + `tool` pairs, so research renders them exactly
 * like its own tool calls: a one-line `ChainOfThoughtStep` with the
 * `ToolCallDetails` expander (truncated preview + copy) instead of a wall
 * of raw output. An orphan `tool` message without the pairing `ai` turn
 * would render as an empty ChainOfThought shell, hence the pair.
 */
export function cliChatToThreadMessages(messages: CliChatMessage[]): Message[] {
  return messages.flatMap((message, index) => {
    const id = `${message.run_id}:${index}`;
    if (message.role === "tool") {
      const callId = `${id}:call`;
      const name = message.name || "tool";
      const failed = CLI_TOOL_ERROR_STATUS.test(message.status);
      return [
        {
          id: `${id}-call`,
          type: "ai",
          content: "",
          tool_calls: [{ id: callId, name, args: {} }],
          run_id: message.run_id,
        },
        {
          id,
          type: "tool",
          tool_call_id: callId,
          name,
          content: message.detail || "—",
          status: failed ? "error" : "success",
          run_id: message.run_id,
        },
      ] as unknown as Message[];
    }
    if (message.role === "user") {
      return [{ id, type: "human", content: message.content } as Message];
    }
    return [
      {
        id,
        type: "ai",
        content: message.content,
        run_id: message.run_id,
      } as unknown as Message,
    ];
  });
}

export interface CliRunStats {
  exitCode: number | null;
  streaming: boolean;
}

export function CliChatView({
  session,
  onSelectRun,
  onRunStats,
}: {
  session: CliSession;
  onSelectRun: (runId: string) => void;
  onRunStats?: (stats: CliRunStats) => void;
}) {
  const { t } = useI18n();
  const ct = t.cliSessions.chat;
  const backfill = useCliChatMessages(session.id);
  const stop = useStopCliSession();

  const [liveRunId, setLiveRunId] = useState<string | null>(null);
  const [pendingPrompt, setPendingPrompt] = useState<{
    runId: string;
    content: string;
  } | null>(null);
  const [stopError, setStopError] = useState<string | null>(null);

  const sessionActive = isCliSessionActive(session);
  const streamTarget = liveRunId ?? (sessionActive ? session.id : null);
  const stream = useCliChatStream(streamTarget);
  const streaming = streamTarget !== null && stream.exitCode === null;

  useEffect(() => {
    onRunStats?.({ exitCode: stream.exitCode, streaming });
  }, [onRunStats, stream.exitCode, streaming]);

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

  const visibleMessages = useMemo(() => {
    const history = backfill.data?.messages ?? [];
    // Optimistic user bubble: the stream yields assistant/tool frames only,
    // so the just-sent prompt would be invisible until the end+jump.
    const live: CliChatMessage[] =
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
    return mergeCliChatMessages(history, live, streamTarget);
  }, [backfill.data?.messages, pendingPrompt, stream.messages, streamTarget]);
  const threadMessages = useMemo(
    () => cliChatToThreadMessages(visibleMessages),
    [visibleMessages],
  );
  const thread = useMemo(
    () =>
      ({
        messages: threadMessages,
        isLoading: streaming,
        isThreadLoading: false,
        values: {},
        error: undefined,
        getMessagesMetadata: () => undefined,
      }) as unknown as ComponentProps<typeof MessageList>["thread"],
    [streaming, threadMessages],
  );

  const handleSent = (runId: string, prompt: string) => {
    setLiveRunId(runId);
    setPendingPrompt({ runId, content: prompt });
  };

  const handleStop = () => {
    if (!streamTarget) return;
    setStopError(null);
    stop.mutate(streamTarget, {
      onError: (error) =>
        setStopError(
          error instanceof Error && error.message
            ? error.message
            : t.cliSessions.errors.stopFailed,
        ),
    });
  };

  return (
    <div
      className="flex size-full min-h-0 min-w-0 flex-col"
      data-testid="cli-chat-view"
    >
      <div className="flex min-h-0 flex-1 justify-center">
        {backfill.isPending ? (
          <div
            role="status"
            aria-label={ct.loading}
            className="flex min-h-0 flex-1 justify-center"
          >
            <MessageListSkeleton />
          </div>
        ) : backfill.isError ? (
          <div className="flex min-h-0 flex-1 items-start justify-center">
            <p
              role="alert"
              className="text-destructive mx-auto w-full max-w-(--container-width-md) p-4 text-center text-sm"
            >
              {t.cliSessions.errors.chatLoadFailed}
            </p>
          </div>
        ) : visibleMessages.length === 0 && !streaming ? (
          <ConversationEmptyState
            role="status"
            data-testid="cli-chat-empty"
            title={ct.empty}
            description=""
          />
        ) : (
          <div className="flex size-full min-w-0 flex-col">
            {(backfill.data?.truncated ?? false) && (
              <p
                role="note"
                data-testid="cli-chat-truncated"
                className="text-muted-foreground mx-auto w-full max-w-(--container-width-md) px-4 pt-2 text-center text-xs"
              >
                {ct.truncated}
              </p>
            )}
            <ArtifactsProvider>
              <MessageList
                className="size-full"
                testId="cli-chat-messages"
                threadId={session.id}
                thread={thread}
                tokenUsageInlineMode="off"
                // Unlock research's Details expander on the condensed tool
                // steps (no token-debug summaries: CLI carries no token
                // metadata).
                showToolCallDetails
                enableSidecarActions={false}
                paddingBottom={MESSAGE_LIST_DEFAULT_PADDING_BOTTOM}
              />
            </ArtifactsProvider>
          </div>
        )}
      </div>

      <div className="flex shrink-0 flex-wrap items-center gap-2 px-3 pb-1 sm:px-4">
        <span className="text-muted-foreground text-xs" role="status">
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
        <span className="flex-1" />
        {streamTarget && !stream.connected && stream.exitCode === null && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-7 text-xs"
            onClick={stream.reconnect}
          >
            {ct.reconnect}
          </Button>
        )}
      </div>
      {stopError && (
        <p role="alert" className="text-destructive px-3 pb-1 text-sm sm:px-4">
          {stopError}
        </p>
      )}

      <div className="relative right-0 bottom-0 left-0 z-30 flex shrink-0 justify-center px-3 pb-4 sm:px-4">
        <CliSendComposer
          key={session.id}
          session={session}
          streaming={streaming}
          onSent={handleSent}
          onStop={handleStop}
          stopping={stop.isPending}
          className="max-w-(--container-width-md)"
        />
      </div>
    </div>
  );
}

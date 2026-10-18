"use client";

import type { ChatStatus } from "ai";

import {
  PromptInput,
  PromptInputFooter,
  PromptInputProvider,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
  usePromptInputController,
  type PromptInputMessage,
} from "@/components/ai-elements/prompt-input";
import { useI18n } from "@/core/i18n/hooks";

import { SuggestionList } from "./input-box";

export type SlimComposerProps = {
  /** Same `PromptInputMessage` contract as the full `InputBox`. */
  onSubmit: (message: PromptInputMessage) => void | Promise<void>;
  /** Stop the streaming run; the submit button becomes stop while streaming. */
  onStop?: () => void | Promise<void>;
  status?: ChatStatus;
  disabled?: boolean;
  canStopStreaming?: boolean;
  /** Mirrors `InputBox`: a role without runs:create keeps a disabled send. */
  canCreateRuns?: boolean;
  placeholder?: string;
  autoFocus?: boolean;
  className?: string;
  /** Welcome mode: show starter chips above the shared disclaimer (S-1). */
  isWelcomeMode?: boolean;
};

function SlimComposerForm({
  onSubmit,
  onStop,
  status = "ready",
  disabled = false,
  canStopStreaming = true,
  canCreateRuns = true,
  placeholder,
  autoFocus = false,
  className,
  isWelcomeMode = false,
}: SlimComposerProps) {
  const { t } = useI18n();
  const controller = usePromptInputController();
  const isStreaming = status === "streaming" || status === "submitted";
  // Same gate as InputBox's sendDenied: while streaming the button is the
  // stop affordance, so the send denial only applies to the send state.
  const sendDenied = !isStreaming && !canCreateRuns;
  const canSend = controller.textInput.value.trim().length > 0;

  return (
    <>
      <PromptInput
        className={className}
        data-testid="slim-composer"
        onSubmit={async (message) => {
          // While a run streams, the submit affordance is the stop button
          // (same contract as `InputBox`): never queue a second send.
          if (isStreaming) {
            if (canStopStreaming) {
              await onStop?.();
            }
            return;
          }
          if (sendDenied) {
            return;
          }
          if (!message.text.trim() && message.files.length === 0) {
            return;
          }
          await onSubmit(message);
          controller.textInput.clear();
          controller.attachments.clear();
        }}
      >
        <PromptInputTextarea
          autoFocus={autoFocus}
          disabled={disabled}
          placeholder={placeholder}
        />
        <PromptInputFooter>
          <PromptInputTools />
          <PromptInputSubmit
            status={status}
            disabled={disabled || sendDenied || (!isStreaming && !canSend)}
          />
        </PromptInputFooter>
      </PromptInput>
      {isWelcomeMode && (
        <div className="flex items-center justify-center pt-2">
          {/* Shared starters without the research-only surprise-me and
              create chrome; still inside the PromptInputProvider. */}
          <SuggestionList showSurpriseMe={false} showCreateMenu={false} />
        </div>
      )}
      {/* Static disclaimer variant (same copy/classes as InputBox welcome
          mode), not the absolute top-full one. */}
      <p className="text-muted-foreground/67 z-10 px-4 text-center text-xs leading-4">
        {t.inputBox.disclaimer}
      </p>
    </>
  );
}

/**
 * Minimal composer for the persistent bot's pinned thread: the same
 * `PromptInput` primitives the full `InputBox` is built from, without the
 * model/mode/skill/attachment tooling an always-on assistant chat does not
 * need. `InputBox` has no slim/compact configuration, so the primitive
 * composition is the reuse-first option; messages keep the exact
 * `PromptInputMessage` shape the chat page already sends.
 */
export function SlimComposer(props: SlimComposerProps) {
  return (
    <PromptInputProvider>
      <SlimComposerForm {...props} />
    </PromptInputProvider>
  );
}

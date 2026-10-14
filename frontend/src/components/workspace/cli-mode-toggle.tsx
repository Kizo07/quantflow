"use client";

import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  HARNESS_MODE_DEFAULT,
  useHarnessMode,
} from "@/core/cli-sessions/harness-mode";
import type { HarnessMode } from "@/core/cli-sessions/types";
import { useI18n } from "@/core/i18n/hooks";

/**
 * Research/Coding harness toggle (F2.7) for the app-shell header.
 * Research runs new sessions on the built-in harness; coding runs them
 * on the CLI backend. Persisted per user; fail-closed to research.
 */
export function CliModeToggle() {
  const { t } = useI18n();
  const { mode, setMode } = useHarnessMode();

  const select = (value: string) => {
    if (value === "research" || value === "coding") {
      setMode(value as HarnessMode);
    }
  };

  return (
    <ToggleGroup
      type="single"
      value={mode ?? HARNESS_MODE_DEFAULT}
      onValueChange={select}
      aria-label={t.cliSessions.modeToggleLabel}
      size="sm"
      spacing={1}
      data-testid="cli-mode-toggle"
    >
      <ToggleGroupItem value="research" aria-label={t.cliSessions.modeResearch}>
        {t.cliSessions.modeResearch}
      </ToggleGroupItem>
      <ToggleGroupItem value="coding" aria-label={t.cliSessions.modeCoding}>
        {t.cliSessions.modeCoding}
      </ToggleGroupItem>
    </ToggleGroup>
  );
}

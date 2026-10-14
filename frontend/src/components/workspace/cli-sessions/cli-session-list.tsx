"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { CliSession } from "@/core/cli-sessions/types";
import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

function formatTime(value: string | null, locale: string): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(locale, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export function CliStatusBadge({ status }: { status: CliSession["status"] }) {
  const { t } = useI18n();
  return (
    <Badge
      variant={status === "running" ? "default" : "secondary"}
      className={cn(
        status === "failed" && "bg-destructive/15 text-destructive",
      )}
    >
      {t.cliSessions.status[status]}
    </Badge>
  );
}

export function CliSessionList({
  sessions,
  selectedId,
  onSelect,
  isLoading,
}: {
  sessions: CliSession[];
  selectedId: string | null;
  onSelect: (sessionId: string) => void;
  isLoading: boolean;
}) {
  const { t, locale } = useI18n();

  if (isLoading) {
    return (
      <p
        role="status"
        data-testid="cli-session-list-loading"
        className="text-muted-foreground text-sm"
      >
        {t.cliSessions.listLoading}
      </p>
    );
  }

  if (sessions.length === 0) {
    return (
      <div
        role="status"
        data-testid="cli-session-list-empty"
        className="text-muted-foreground flex flex-col gap-1 rounded-lg border border-dashed p-6 text-center text-sm"
      >
        <p>{t.cliSessions.listEmpty}</p>
        <p className="text-xs">{t.cliSessions.listEmptyHint}</p>
      </div>
    );
  }

  return (
    <ul className="flex flex-col gap-2" data-testid="cli-session-list">
      {sessions.map((session) => (
        <li key={session.id}>
          <Button
            type="button"
            variant="outline"
            onClick={() => onSelect(session.id)}
            aria-current={session.id === selectedId}
            className={cn(
              "h-auto w-full flex-col items-start gap-1.5 p-3 text-left",
              session.id === selectedId && "border-primary",
            )}
          >
            <span className="flex w-full items-center justify-between gap-2">
              <CliStatusBadge status={session.status} />
              <span className="text-muted-foreground font-mono text-xs">
                {session.id.slice(0, 8)}
              </span>
            </span>
            <span className="text-sm font-medium">
              {session.model ?? t.cliSessions.launch.modelCliDefault}
              {session.effort ? ` · ${session.effort}` : ""}
            </span>
            <span className="text-muted-foreground text-xs">
              {formatTime(session.started_at ?? session.created_at, locale)}
            </span>
          </Button>
        </li>
      ))}
    </ul>
  );
}

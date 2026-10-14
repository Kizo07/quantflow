"use client";

import Link from "next/link";
import { useMemo } from "react";

import { Badge } from "@/components/ui/badge";
import type { CliSession } from "@/core/cli-sessions/types";
import { useI18n } from "@/core/i18n/hooks";
import { useThreads } from "@/core/threads/hooks";
import type { AgentThread } from "@/core/threads/types";
import { pathOfThread, titleOfThread } from "@/core/threads/utils";
import { cn } from "@/lib/utils";

import { CliStatusBadge } from "./cli-session-list";

type UnifiedRow =
  | { kind: "research"; thread: AgentThread; at: number }
  | { kind: "coding"; session: CliSession; at: number };

function timeOf(value: string | null | undefined): number {
  if (!value) return 0;
  const time = new Date(value).getTime();
  return Number.isNaN(time) ? 0 : time;
}

/**
 * Unified session list (F2.8): research threads from the existing store
 * plus coding sessions from `GET /api/cli-sessions`, each with a mode
 * badge. Coding rows link to the session detail on this page.
 */
export function CliUnifiedList({
  cliSessions,
  selectedId,
  onSelectCli,
}: {
  cliSessions: CliSession[];
  selectedId: string | null;
  onSelectCli: (sessionId: string) => void;
}) {
  const { t } = useI18n();
  const threadsQuery = useThreads();

  const rows: UnifiedRow[] = useMemo(() => {
    const merged: UnifiedRow[] = [
      ...(threadsQuery.data ?? []).map(
        (thread): UnifiedRow => ({
          kind: "research",
          thread,
          at: timeOf(thread.updated_at ?? thread.created_at),
        }),
      ),
      ...cliSessions.map(
        (session): UnifiedRow => ({
          kind: "coding",
          session,
          at: timeOf(session.started_at ?? session.created_at),
        }),
      ),
    ];
    merged.sort((a, b) => b.at - a.at);
    return merged;
  }, [cliSessions, threadsQuery.data]);

  if (threadsQuery.isPending) {
    return (
      <p role="status" className="text-muted-foreground text-sm">
        {t.cliSessions.listLoading}
      </p>
    );
  }

  if (rows.length === 0) {
    return (
      <div
        role="status"
        data-testid="cli-unified-list-empty"
        className="text-muted-foreground flex flex-col gap-1 rounded-lg border border-dashed p-6 text-center text-sm"
      >
        <p>{t.cliSessions.listEmpty}</p>
        <p className="text-xs">{t.cliSessions.listEmptyHint}</p>
      </div>
    );
  }

  return (
    <ul className="flex flex-col gap-2" data-testid="cli-unified-list">
      {rows.map((row) =>
        row.kind === "research" ? (
          <li key={`research-${row.thread.thread_id}`}>
            <Link
              href={pathOfThread(row.thread)}
              className="hover:bg-muted/40 flex w-full items-center justify-between gap-2 rounded-lg border p-3 text-left text-sm"
            >
              <span className="min-w-0 truncate font-medium">
                {titleOfThread(row.thread, t.pages.untitled)}
              </span>
              <Badge variant="outline" data-testid="cli-mode-badge">
                {t.cliSessions.modeResearch}
              </Badge>
            </Link>
          </li>
        ) : (
          <li key={`coding-${row.session.id}`}>
            <button
              type="button"
              onClick={() => onSelectCli(row.session.id)}
              aria-current={row.session.id === selectedId}
              className={cn(
                "hover:bg-muted/40 flex w-full items-center justify-between gap-2 rounded-lg border p-3 text-left text-sm",
                row.session.id === selectedId && "border-primary",
              )}
            >
              <span className="flex min-w-0 items-center gap-2">
                <CliStatusBadge status={row.session.status} />
                <span className="truncate font-medium">
                  {row.session.model ?? t.cliSessions.launch.modelCliDefault}
                </span>
              </span>
              <Badge variant="outline" data-testid="cli-mode-badge">
                {t.cliSessions.modeCoding}
              </Badge>
            </button>
          </li>
        ),
      )}
    </ul>
  );
}

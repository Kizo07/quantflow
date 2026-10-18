"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { CliModeToggle } from "@/components/workspace/cli-mode-toggle";
import {
  CliChatView,
  type CliRunStats,
} from "@/components/workspace/cli-sessions/cli-chat-view";
import { CliLaunchComposer } from "@/components/workspace/cli-sessions/cli-composer";
import { CliSessionDetail } from "@/components/workspace/cli-sessions/cli-session-detail";
import { RunDurationLabel } from "@/components/workspace/messages/run-duration";
import { Welcome } from "@/components/workspace/welcome";
import { selectFirstCliUserPrompt } from "@/core/cli-sessions/chat";
import { useHarnessMode } from "@/core/cli-sessions/harness-mode";
import { useCliChatMessages, useCliSessions } from "@/core/cli-sessions/hooks";
import type { CliSession } from "@/core/cli-sessions/types";
import { useI18n } from "@/core/i18n/hooks";

function sessionTime(session: CliSession): number {
  const value = session.started_at ?? session.created_at;
  if (!value) return 0;
  const time = new Date(value).getTime();
  return Number.isNaN(time) ? 0 : time;
}

function elapsedSeconds(session: CliSession): number | null {
  if (!session.started_at) return null;
  const start = new Date(session.started_at).getTime();
  const end = session.ended_at
    ? new Date(session.ended_at).getTime()
    : Date.now();
  if (Number.isNaN(start) || Number.isNaN(end)) return null;
  return Math.max(0, Math.floor((end - start) / 1000));
}

/**
 * Run stats line in the header slot where the research tab shows token
 * usage: elapsed (the research `RunDurationLabel`) + model + exit code.
 */
function CliHeaderStats({
  session,
  live,
}: {
  session: CliSession;
  live: CliRunStats | null;
}) {
  const { t } = useI18n();
  const seconds = elapsedSeconds(session);
  const model = session.model ?? t.cliSessions.launch.modelCliDefault;
  const tail =
    live?.exitCode !== null && live?.exitCode !== undefined
      ? `${t.cliSessions.statsExit} ${live.exitCode}`
      : t.cliSessions.status[session.status];
  return (
    <span
      data-testid="cli-run-stats"
      className="text-muted-foreground hidden items-center gap-1.5 text-xs sm:inline-flex"
    >
      {seconds !== null && <RunDurationLabel durationSeconds={seconds} />}
      <span aria-hidden="true">·</span>
      <span className="max-w-40 truncate font-medium">{model}</span>
      <span aria-hidden="true">·</span>
      <span>{tail}</span>
    </span>
  );
}

function CliHeaderTitle({ session }: { session: CliSession | null }) {
  const { t } = useI18n();
  const backfill = useCliChatMessages(session?.id);
  if (!session) {
    // V-7a: the page title reads "CLI sessions" while the action button keeps
    // "New session" — previously both rendered the same string.
    return (
      <span className="min-w-0 flex-1 truncate text-sm font-medium">
        {t.cliSessions.title}
      </span>
    );
  }
  const title =
    selectFirstCliUserPrompt(backfill.data?.messages) ??
    `${session.model ?? t.cliSessions.launch.modelCliDefault}${session.effort ? ` · ${session.effort}` : ""}`;
  return (
    <span className="min-w-0 flex-1 truncate text-sm font-medium" title={title}>
      {title}
    </span>
  );
}

export default function CliSessionsPage() {
  const { t } = useI18n();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const sessionIdParam = searchParams.get("session_id");
  const newParam = searchParams.get("new") === "1";

  const { mode } = useHarnessMode();
  const query = useCliSessions();
  const sessions = useMemo(
    () =>
      [...(query.data ?? [])].sort((a, b) => sessionTime(b) - sessionTime(a)),
    [query.data],
  );

  const [shownId, setShownId] = useState<string | null>(null);
  const [isNew, setIsNew] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [liveStats, setLiveStats] = useState<{
    id: string;
    stats: CliRunStats;
  } | null>(null);

  const pinnedId = sessionIdParam ?? shownId;
  const selectedSession: CliSession | null = isNew
    ? null
    : ((pinnedId ? sessions.find((s) => s.id === pinnedId) : undefined) ??
      sessions[0] ??
      null);

  useEffect(() => {
    document.title = `${t.sidebar.cliSessions} - ${t.pages.appName}`;
  }, [t.pages.appName, t.sidebar.cliSessions]);

  // URL params drive the welcome/selection state so the app-sidebar links
  // (`?session_id=` / `?new=1`) work without remounting the page.
  useEffect(() => {
    if (sessionIdParam) setIsNew(false);
  }, [sessionIdParam]);

  useEffect(() => {
    if (newParam) {
      setIsNew(true);
      setShownId(null);
    }
  }, [newParam]);

  const replaceQuery = useCallback(
    (sessionId: string | "new" | null) => {
      const params = new URLSearchParams();
      if (sessionId === "new") params.set("new", "1");
      else if (sessionId) params.set("session_id", sessionId);
      const qs = params.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    },
    [pathname, router],
  );

  const selectSession = useCallback(
    (sessionId: string) => {
      setIsNew(false);
      setShownId(sessionId);
      replaceQuery(sessionId);
    },
    [replaceQuery],
  );

  const newSession = useCallback(() => {
    setIsNew(true);
    setShownId(null);
    replaceQuery("new");
  }, [replaceQuery]);

  // Bare landing with sessions: pin the newest into the URL so the sidebar
  // highlights what the page shows.
  useEffect(() => {
    const newest = sessions[0];
    if (!sessionIdParam && !newParam && !isNew && newest) {
      replaceQuery(newest.id);
    }
  }, [isNew, newParam, replaceQuery, sessionIdParam, sessions]);

  const handleRunStats = useCallback(
    (stats: CliRunStats) => {
      if (!selectedSession) return;
      const id = selectedSession.id;
      setLiveStats((previous) =>
        previous?.id === id &&
        previous.stats.exitCode === stats.exitCode &&
        previous.stats.streaming === stats.streaming
          ? previous
          : { id, stats },
      );
    },
    [selectedSession],
  );

  const live =
    liveStats && liveStats.id === selectedSession?.id ? liveStats.stats : null;

  return (
    <div
      data-testid="cli-sessions-chat"
      className="relative flex size-full min-h-0 flex-col"
    >
      {query.error && (
        <p
          role="alert"
          data-testid="cli-session-load-error"
          className="text-destructive px-4 pt-2 text-sm"
        >
          {t.cliSessions.errors.loadFailed}
        </p>
      )}
      <div className="relative flex min-h-0 min-w-0 flex-1">
        <header className="bg-background/80 absolute top-0 right-0 left-0 z-30 flex h-12 shrink-0 items-center gap-2 px-2 shadow-xs backdrop-blur sm:px-4">
          <div className="flex min-w-0 flex-1 items-center gap-2">
            <CliHeaderTitle session={selectedSession} />
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {/* Mode toggle: without it, default research users land on an
                input-less welcome with no in-page remedy (V-1). */}
            <CliModeToggle />
            {mode === "coding" ? (
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={newSession}
              >
                {t.cliSessions.newSession}
              </Button>
            ) : (
              <Button type="button" variant="outline" size="sm" asChild>
                <Link href="/workspace/chats">{t.cliSessions.newSession}</Link>
              </Button>
            )}
            {selectedSession && (
              <CliHeaderStats session={selectedSession} live={live} />
            )}
            {/* D-2: the CLI Inspector is intentionally a modal Sheet, not the
                research sidecar panel — a session inspector is transient
                run metadata, not a parallel conversation surface. */}
            {selectedSession && (
              <Sheet open={inspectorOpen} onOpenChange={setInspectorOpen}>
                <SheetTrigger asChild>
                  <Button type="button" variant="outline" size="sm">
                    {t.cliSessions.inspector}
                  </Button>
                </SheetTrigger>
                <SheetContent
                  side="right"
                  className="w-full overflow-y-auto sm:max-w-xl"
                >
                  <SheetHeader>
                    <SheetTitle>{t.cliSessions.inspector}</SheetTitle>
                    <SheetDescription>
                      {t.cliSessions.inspectorDescription}
                    </SheetDescription>
                  </SheetHeader>
                  <CliSessionDetail
                    key={selectedSession.id}
                    session={selectedSession}
                  />
                </SheetContent>
              </Sheet>
            )}
            <Button type="button" variant="ghost" size="sm" asChild>
              <Link href="/workspace/cli-sessions/guide">
                {t.cliSessions.guideLink}
              </Link>
            </Button>
          </div>
        </header>

        <main className="flex min-h-0 max-w-full grow flex-col">
          {selectedSession ? (
            <div className="flex min-h-0 flex-1 justify-center pt-12">
              <CliChatView
                key={selectedSession.id}
                session={selectedSession}
                onSelectRun={selectSession}
                onRunStats={handleRunStats}
              />
            </div>
          ) : (
            // Normal flow with pt-12 like the selected-session branch: the
            // old absolute+translate trick pushed the tall coding welcome
            // under the h-12 header (V-2).
            <div className="flex min-h-0 flex-1 justify-center overflow-y-auto px-3 pt-12 sm:px-4">
              <div className="w-full max-w-(--container-width-sm)">
                <Welcome />
                {mode === "coding" ? (
                  <CliLaunchComposer
                    isWelcomeMode
                    onStarted={(session) => selectSession(session.id)}
                  />
                ) : (
                  <>
                    <p
                      className="text-muted-foreground pb-2 text-center text-xs"
                      data-testid="cli-research-hint"
                    >
                      {t.cliSessions.researchNewSessionHint}
                    </p>
                    <p className="text-muted-foreground rounded-lg border border-dashed p-6 text-center text-sm">
                      {t.cliSessions.selectHint}
                    </p>
                  </>
                )}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

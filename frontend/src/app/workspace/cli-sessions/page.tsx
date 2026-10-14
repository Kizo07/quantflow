"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { CliLaunchForm } from "@/components/workspace/cli-sessions/cli-launch-form";
import { CliSessionDetail } from "@/components/workspace/cli-sessions/cli-session-detail";
import { CliSessionList } from "@/components/workspace/cli-sessions/cli-session-list";
import { CliUnifiedList } from "@/components/workspace/cli-sessions/cli-unified-list";
import {
  WorkspaceBody,
  WorkspaceContainer,
  WorkspaceHeader,
} from "@/components/workspace/workspace-container";
import { useHarnessMode } from "@/core/cli-sessions/harness-mode";
import { useCliSessions } from "@/core/cli-sessions/hooks";
import { useI18n } from "@/core/i18n/hooks";

type ListTab = "coding" | "all";

export default function CliSessionsPage() {
  const { t } = useI18n();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const sessionIdParam = searchParams.get("session_id");

  const { mode } = useHarnessMode();
  const query = useCliSessions();
  const sessions = query.data ?? [];

  const [tab, setTab] = useState<ListTab>("coding");
  const [launchOpen, setLaunchOpen] = useState(false);
  const [shownId, setShownId] = useState<string | null>(null);

  const pinnedId = sessionIdParam ?? shownId;
  const selectedSession =
    (pinnedId ? sessions.find((s) => s.id === pinnedId) : undefined) ??
    sessions[0] ??
    null;
  if (!sessionIdParam && selectedSession && selectedSession.id !== shownId) {
    setShownId(selectedSession.id);
  }

  useEffect(() => {
    document.title = `${t.sidebar.cliSessions} - ${t.pages.appName}`;
  }, [t.pages.appName, t.sidebar.cliSessions]);

  const replaceQuery = useCallback(
    (sessionId: string | null) => {
      const params = new URLSearchParams();
      if (sessionId) params.set("session_id", sessionId);
      const qs = params.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    },
    [pathname, router],
  );

  const selectSession = (sessionId: string) => {
    setShownId(sessionId);
    replaceQuery(sessionId);
  };

  return (
    <WorkspaceContainer>
      <WorkspaceHeader />
      <WorkspaceBody>
        <div className="mx-auto flex w-full max-w-6xl min-w-0 flex-col gap-4 px-4 py-5 sm:px-6">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="flex min-w-0 flex-col gap-1">
              <h1 className="text-2xl font-semibold">{t.cliSessions.title}</h1>
              <p className="text-muted-foreground text-sm">
                {t.cliSessions.description}
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button type="button" variant="outline" asChild>
                <Link href="/workspace/cli-sessions/guide">
                  {t.cliSessions.guideLink}
                </Link>
              </Button>
              {mode === "coding" ? (
                <Button type="button" onClick={() => setLaunchOpen(true)}>
                  {t.cliSessions.newSession}
                </Button>
              ) : (
                <Button type="button" asChild>
                  <Link href="/workspace/chats">
                    {t.cliSessions.newSession}
                  </Link>
                </Button>
              )}
            </div>
          </div>

          {mode === "research" && (
            <p
              className="text-muted-foreground text-sm"
              data-testid="cli-research-hint"
            >
              {t.cliSessions.researchNewSessionHint}
            </p>
          )}

          {query.error && (
            <p
              role="alert"
              data-testid="cli-session-load-error"
              className="text-destructive text-sm"
            >
              {t.cliSessions.errors.loadFailed}
            </p>
          )}

          <ToggleGroup
            type="single"
            value={tab}
            onValueChange={(value) => {
              if (value === "coding" || value === "all") setTab(value);
            }}
            aria-label={t.cliSessions.title}
            size="sm"
            spacing={1}
            className="flex-wrap"
          >
            <ToggleGroupItem value="coding">
              {t.cliSessions.tabsCoding}
            </ToggleGroupItem>
            <ToggleGroupItem value="all">
              {t.cliSessions.tabsAll}
            </ToggleGroupItem>
          </ToggleGroup>

          <div className="grid min-w-0 gap-4 lg:grid-cols-[330px_minmax(0,1fr)] lg:items-start">
            <div className="flex min-w-0 flex-col gap-3">
              {tab === "coding" ? (
                <CliSessionList
                  sessions={sessions}
                  selectedId={selectedSession?.id ?? null}
                  onSelect={selectSession}
                  isLoading={query.isPending}
                />
              ) : (
                <CliUnifiedList
                  cliSessions={sessions}
                  selectedId={selectedSession?.id ?? null}
                  onSelectCli={selectSession}
                />
              )}
            </div>
            <div className="min-w-0">
              {selectedSession ? (
                <CliSessionDetail
                  key={selectedSession.id}
                  session={selectedSession}
                />
              ) : query.isSuccess ? (
                <p className="text-muted-foreground rounded-lg border border-dashed p-6 text-center text-sm">
                  {t.cliSessions.selectHint}
                </p>
              ) : null}
            </div>
          </div>
        </div>
      </WorkspaceBody>

      <Dialog open={launchOpen} onOpenChange={setLaunchOpen}>
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>{t.cliSessions.launch.title}</DialogTitle>
            <DialogDescription>
              {t.cliSessions.launch.description}
            </DialogDescription>
          </DialogHeader>
          <CliLaunchForm
            onLaunched={(session) => {
              setLaunchOpen(false);
              selectSession(session.id);
            }}
            onCancel={() => setLaunchOpen(false)}
          />
        </DialogContent>
      </Dialog>
    </WorkspaceContainer>
  );
}

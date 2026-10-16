"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useMemo } from "react";

import {
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
} from "@/components/ui/sidebar";
import { selectFirstCliUserPrompt } from "@/core/cli-sessions/chat";
import { useCliChatMessages, useCliSessions } from "@/core/cli-sessions/hooks";
import type { CliSession } from "@/core/cli-sessions/types";
import { useI18n } from "@/core/i18n/hooks";

function sessionTime(session: CliSession): number {
  const value = session.started_at ?? session.created_at;
  if (!value) return 0;
  const time = new Date(value).getTime();
  return Number.isNaN(time) ? 0 : time;
}

function CliSidebarSessionRow({
  session,
  active,
}: {
  session: CliSession;
  active: boolean;
}) {
  const backfill = useCliChatMessages(session.id);
  const title =
    selectFirstCliUserPrompt(backfill.data?.messages) ??
    session.model ??
    session.id.slice(0, 8);
  return (
    <SidebarMenuSubItem>
      <SidebarMenuSubButton asChild isActive={active} size="sm">
        <Link
          href={`/workspace/cli-sessions?session_id=${session.id}`}
          title={title}
        >
          <span>{title}</span>
        </Link>
      </SidebarMenuSubButton>
    </SidebarMenuSubItem>
  );
}

/**
 * CLI sessions expanded below the "CLI Sessions" nav item (newest first).
 * Mounted only on the cli-sessions route, so the list query (and its poll)
 * never runs on other workspace pages.
 */
export function CliSidebarSessions() {
  const { t } = useI18n();
  const searchParams = useSearchParams();
  const activeId = searchParams.get("session_id");
  const isNew = searchParams.get("new") === "1";
  const query = useCliSessions();
  const sessions = useMemo(
    () =>
      [...(query.data ?? [])].sort((a, b) => sessionTime(b) - sessionTime(a)),
    [query.data],
  );
  return (
    <SidebarMenuSub data-testid="cli-sidebar-sessions">
      <SidebarMenuSubItem>
        <SidebarMenuSubButton asChild isActive={isNew} size="sm">
          <Link href="/workspace/cli-sessions?new=1">
            <span>{t.cliSessions.newSession}</span>
          </Link>
        </SidebarMenuSubButton>
      </SidebarMenuSubItem>
      {sessions.map((session) => (
        <CliSidebarSessionRow
          key={session.id}
          session={session}
          active={session.id === activeId}
        />
      ))}
    </SidebarMenuSub>
  );
}

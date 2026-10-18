"use client";

import { BotIcon, PauseIcon, PlayIcon } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useRef } from "react";
import { toast } from "sonner";

import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useI18n } from "@/core/i18n/hooks";
import { useInfiniteThreads } from "@/core/threads/hooks";
import {
  findPersistentBotThread,
  persistentBotHistory,
  persistentBotUnreadCount,
} from "@/core/threads/persistent-bot";
import {
  usePersistentBotStatus,
  useSetPersistentBotPaused,
} from "@/core/threads/persistent-bot-status";
import { buildThreadListModel } from "@/core/threads/thread-list-model";
import { pathOfThread, titleOfThread } from "@/core/threads/utils";
import { env } from "@/env";

import { ThreadSidebarItem } from "./recent-chat-list";

/**
 * Cross-scope pinned section for the persistent bot: one entry linking to
 * the singleton pinned thread, the bot agent's other chats below it as
 * history (the shared `ThreadSidebarItem` rows), and an aggregate unread
 * badge over both. It renders nothing until a thread carries the
 * Gateway-stamped `persistent_bot` marker, which only exists when the
 * deployment enabled the bot (off by default), so the sidebar is unchanged
 * everywhere else. Opening a listed thread clears its unread share through
 * the chat page's existing mark-read call; the badge counts the same cached
 * `unread` flags that call patches.
 */
export function PinnedAssistantSection() {
  const { t } = useI18n();
  const pathname = usePathname();
  // Same params as RecentChatList/ProjectsSection: one shared query, no
  // extra fetch for this section.
  const { data: infiniteThreads } = useInfiniteThreads({
    archived:
      env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY === "true" ? undefined : false,
  });
  const threads = useMemo(
    () => buildThreadListModel(infiniteThreads?.pages ?? []).threads,
    [infiniteThreads?.pages],
  );
  const botThread = useMemo(() => findPersistentBotThread(threads), [threads]);
  const history = useMemo(
    () => (botThread ? persistentBotHistory(threads, botThread) : []),
    [botThread, threads],
  );
  const unreadCount = botThread
    ? persistentBotUnreadCount(threads, botThread)
    : 0;
  const botStatus = usePersistentBotStatus(botThread !== null);
  const setBotPaused = useSetPersistentBotPaused();
  const isPaused =
    botStatus.data?.user_paused === true ||
    botStatus.data?.heartbeat_status === "paused";

  // Background activity from the bot (a heartbeat landing while the user is
  // elsewhere) is worth a toast only when this tab is hidden; a visible tab
  // already shows the badge. The first observed count just seeds the
  // baseline so initial load never toasts.
  const previousUnreadRef = useRef<number | null>(null);
  useEffect(() => {
    if (!botThread) {
      previousUnreadRef.current = null;
      return;
    }
    const previous = previousUnreadRef.current;
    previousUnreadRef.current = unreadCount;
    if (previous !== null && unreadCount > previous && document.hidden) {
      toast(t.sidebar.pinnedAssistantNewActivity);
    }
  }, [botThread, t.sidebar.pinnedAssistantNewActivity, unreadCount]);

  if (!botThread) {
    // Disabled-with-tooltip row (P-1) instead of null: first-run users see
    // that the Assistant exists but is not enabled yet. Same pattern as the
    // Agents entry in workspace-nav-chat-list.
    return (
      <SidebarGroup>
        <SidebarGroupLabel>{t.sidebar.pinnedAssistant}</SidebarGroupLabel>
        <SidebarGroupContent>
          <SidebarMenu>
            <SidebarMenuItem>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="block w-full cursor-not-allowed">
                    <SidebarMenuButton
                      className="text-muted-foreground/50"
                      aria-disabled
                      aria-describedby="pinned-assistant-disabled-reason"
                      data-testid="pinned-assistant-disabled"
                    >
                      <BotIcon />
                      <span>{t.sidebar.pinnedAssistant}</span>
                    </SidebarMenuButton>
                    <span
                      id="pinned-assistant-disabled-reason"
                      className="sr-only"
                    >
                      {t.sidebar.pinnedAssistantDisabledTooltip}
                    </span>
                  </span>
                </TooltipTrigger>
                <TooltipContent side="right">
                  {t.sidebar.pinnedAssistantDisabledTooltip}
                </TooltipContent>
              </Tooltip>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarGroupContent>
      </SidebarGroup>
    );
  }

  const botPath = pathOfThread(botThread);
  const botTitle = titleOfThread(botThread, t.pages.untitled);

  return (
    <SidebarGroup>
      <SidebarGroupLabel>{t.sidebar.pinnedAssistant}</SidebarGroupLabel>
      <SidebarGroupContent>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton isActive={pathname === botPath} asChild>
              <Link
                className="text-muted-foreground"
                href={botPath}
                data-testid="pinned-assistant-entry"
              >
                <BotIcon />
                <span className="truncate">{botTitle}</span>
              </Link>
            </SidebarMenuButton>
            {unreadCount > 0 && (
              <SidebarMenuBadge data-testid="pinned-assistant-unread-badge">
                {unreadCount}
              </SidebarMenuBadge>
            )}
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton
              type="button"
              disabled={botStatus.isPending || setBotPaused.isPending}
              onClick={() => setBotPaused.mutate(!isPaused)}
              aria-pressed={isPaused}
              data-testid="pinned-assistant-pause-toggle"
            >
              {isPaused ? <PlayIcon /> : <PauseIcon />}
              <span className="truncate">
                {isPaused
                  ? t.sidebar.pinnedAssistantResume
                  : t.sidebar.pinnedAssistantPause}
              </span>
              {isPaused && (
                <span className="text-muted-foreground ml-auto text-xs">
                  {t.sidebar.pinnedAssistantPaused}
                </span>
              )}
            </SidebarMenuButton>
          </SidebarMenuItem>
          {history.map((thread) => (
            <ThreadSidebarItem
              key={thread.thread_id}
              thread={thread}
              isActive={pathname === pathOfThread(thread)}
            />
          ))}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  );
}

import { afterEach, describe, expect, it, rs } from "@rstest/core";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";

import { SidebarProvider } from "@/components/ui/sidebar";
import { RecentChatList } from "@/components/workspace/recent-chat-list";
import { DEFAULT_LOCALE } from "@/core/i18n";
import { I18nProvider } from "@/core/i18n/context";

const mocks = rs.hoisted(() => ({
  replace: rs.fn(),
  mutateDelete: rs.fn(),
  resetThreadChat: rs.fn(),
  archivedThread: {
    thread_id: "archived-1",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    metadata: {},
    status: "idle",
    values: { title: "Old project notes", messages: [] },
    interrupts: {},
  },
}));

rs.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mocks.replace }),
  usePathname: () => "/workspace/chats/archived-1",
  useParams: () => ({ thread_id: "archived-1" }),
}));
rs.mock("sonner", () => ({ toast: { error: rs.fn(), success: rs.fn() } }));
rs.mock("@/components/workspace/chats/use-thread-chat", () => ({
  resetThreadChatAfterDelete: mocks.resetThreadChat,
}));
rs.mock("@/core/threads/hooks", () => ({
  useInfiniteThreads: (params?: { archived?: boolean }) => ({
    data: params?.archived
      ? { pages: [[mocks.archivedThread]] }
      : { pages: [[]] },
    fetchNextPage: rs.fn(),
    hasNextPage: false,
    isFetchingNextPage: false,
  }),
  useMoveThreadToProject: () => ({ mutate: rs.fn() }),
  usePinThread: () => ({ mutate: rs.fn() }),
  useRenameThread: () => ({ mutate: rs.fn() }),
  useDeleteThread: () => ({ mutate: mocks.mutateDelete }),
}));

function renderList(): ReturnType<typeof render> {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <I18nProvider initialLocale={DEFAULT_LOCALE}>
      <QueryClientProvider client={queryClient}>
        <SidebarProvider>
          <RecentChatList />
        </SidebarProvider>
      </QueryClientProvider>
    </I18nProvider>,
  );
}

/** Radix triggers open on pointerdown, not click (verified in happy-dom). */
async function openArchivedRowMenu(): Promise<void> {
  const trigger = await screen.findByRole("button", { name: /more/i });
  fireEvent.pointerDown(trigger, { button: 0, pointerType: "mouse" });
  fireEvent.click(trigger);
}

afterEach(() => {
  rs.restoreAllMocks();
  cleanup();
});

describe("RecentChatList archived section", () => {
  it("renders archived chats without crashing", async () => {
    renderList();
    expect(await screen.findByTestId("archived-chat-list")).not.toBeNull();
    expect(await screen.findByText("Old project notes")).not.toBeNull();
  });

  it("deleting the open archived chat resets chat state and redirects home", async () => {
    renderList();
    await openArchivedRowMenu();
    fireEvent.click(await screen.findByText("Delete"));
    await waitFor(() => {
      expect(mocks.mutateDelete).toHaveBeenCalledTimes(1);
    });
    const variables = mocks.mutateDelete.mock.calls[0]?.[0] as {
      threadId: string;
      onDeleted?: () => void;
    };
    expect(variables.threadId).toBe("archived-1");
    // The delete mutation only honors `onDeleted`; any other key silently
    // drops the post-delete redirect.
    expect(typeof variables.onDeleted).toBe("function");
    act(() => {
      variables.onDeleted?.();
    });
    expect(mocks.resetThreadChat).toHaveBeenCalledWith({
      deletedThreadId: "archived-1",
      nextPath: "/workspace/chats/new",
      force: true,
    });
    expect(mocks.replace).toHaveBeenCalledWith("/workspace/chats/new");
  });
});

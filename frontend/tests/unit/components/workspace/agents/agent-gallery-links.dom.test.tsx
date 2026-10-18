import { afterEach, describe, expect, rs, test } from "@rstest/core";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";

rs.mock("next/navigation", () => ({
  useRouter: () => ({ push: rs.fn(), replace: rs.fn(), refresh: rs.fn() }),
  usePathname: () => "/workspace/agents",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

const agentsState = rs.hoisted(() => ({ agents: [] as unknown[] }));

rs.mock("@/core/agents", () => ({
  useAgents: () => ({ agents: agentsState.agents, isLoading: false }),
  useImportAgentPackage: () => ({ mutateAsync: rs.fn() }),
}));

import { AgentGallery } from "@/components/workspace/agents/agent-gallery";
import { I18nProvider } from "@/core/i18n/context";

function renderGallery() {
  document.cookie = "locale=en-US; path=/";
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <I18nProvider initialLocale="en-US">
      <QueryClientProvider client={queryClient}>
        <AgentGallery />
      </QueryClientProvider>
    </I18nProvider>,
  );
}

afterEach(() => {
  cleanup();
  agentsState.agents = [];
  document.cookie = "locale=; max-age=0; path=/";
});

describe("AgentGallery new-agent navigation", () => {
  test("both New Agent actions are pre-hydration-safe links", () => {
    renderGallery();
    // Empty gallery: header action + empty-state action.
    const links = screen.getAllByRole("link", { name: "New Agent" });
    expect(links).toHaveLength(2);
    for (const link of links) {
      expect(link.getAttribute("href")).toBe("/workspace/agents/new");
    }
  });
});

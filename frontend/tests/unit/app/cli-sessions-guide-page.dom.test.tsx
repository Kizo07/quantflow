import { afterEach, describe, expect, it, rs } from "@rstest/core";
import { cleanup, render, screen } from "@testing-library/react";
import type { PropsWithChildren, ReactNode } from "react";

rs.mock("next/link", () => ({
  default: ({
    href,
    children,
    ...rest
  }: {
    href: string;
    children: ReactNode;
  } & Record<string, unknown>) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));
rs.mock("@/components/workspace/workspace-container", () => ({
  WorkspaceContainer: ({ children }: PropsWithChildren) => (
    <div>{children}</div>
  ),
  WorkspaceHeader: () => <div />,
  WorkspaceBody: ({ children }: PropsWithChildren) => <div>{children}</div>,
}));

import CliSessionsGuidePage from "@/app/workspace/cli-sessions/guide/page";
import { I18nProvider } from "@/core/i18n/context";

function renderGuide() {
  document.cookie = "locale=en-US; path=/";
  render(
    <I18nProvider initialLocale="en-US">
      <CliSessionsGuidePage />
    </I18nProvider>,
  );
}

afterEach(() => {
  cleanup();
  document.cookie = "locale=; max-age=0; path=/";
});

describe("CliSessionsGuidePage", () => {
  it("renders the flags matrix, yolo warning, log locations, mode toggle, and 1.3 preset", () => {
    renderGuide();
    expect(
      screen.getByRole("heading", { name: "CLI sessions operator guide" }),
    ).toBeDefined();
    expect(screen.getByRole("heading", { name: "Launch flags" })).toBeDefined();
    expect(screen.getByText("--reasoning-effort")).toBeDefined();
    expect(screen.getByTestId("cli-guide-yolo-warning")).toBeDefined();
    expect(
      screen.getByRole("heading", { name: "Session log locations" }),
    ).toBeDefined();
    expect(
      screen.getByRole("heading", { name: "Research / Coding mode" }),
    ).toBeDefined();
    expect(
      screen.getByRole("heading", { name: "The 1.3 preset" }),
    ).toBeDefined();
  });

  it("links back to the sessions dashboard", () => {
    renderGuide();
    const link = screen.getByRole("link", { name: "Back to CLI sessions" });
    expect(link.getAttribute("href")).toBe("/workspace/cli-sessions");
  });
});

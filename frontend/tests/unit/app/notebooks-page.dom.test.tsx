import { afterEach, describe, expect, it, rs } from "@rstest/core";
import { cleanup, render, screen } from "@testing-library/react";
import type { PropsWithChildren } from "react";

const mocks = rs.hoisted(() => ({
  search: "",
  frameProps: {} as Record<string, unknown>,
}));

rs.mock("next/navigation", () => ({
  usePathname: () => "/workspace/notebooks",
  useRouter: () => ({ push: rs.fn(), replace: rs.fn() }),
  useSearchParams: () => new URLSearchParams(mocks.search),
}));
rs.mock("@/components/workspace/workspace-container", () => ({
  WorkspaceContainer: ({ children }: PropsWithChildren) => (
    <div>{children}</div>
  ),
  WorkspaceHeader: () => <div />,
  WorkspaceBody: ({ children }: PropsWithChildren) => <div>{children}</div>,
}));
rs.mock("@/components/workspace/notebooks/notebook-frame", () => ({
  NotebookFrame: (props: Record<string, unknown>) => {
    mocks.frameProps = props;
    return <div data-testid="notebook-frame-stub" />;
  },
}));

import NotebooksPage from "@/app/workspace/notebooks/page";

describe("NotebooksPage", () => {
  afterEach(() => {
    cleanup();
    mocks.search = "";
    mocks.frameProps = {};
  });

  it("passes ?notebook= through as the frame notebookId", () => {
    mocks.search = "?notebook=e2e-demo%2Fsmoke.ipynb";
    render(<NotebooksPage />);
    expect(screen.getByTestId("notebook-frame-stub")).toBeDefined();
    expect(mocks.frameProps.notebookId).toBe("e2e-demo/smoke.ipynb");
  });

  it("renders the frame without a notebookId when no param is given", () => {
    render(<NotebooksPage />);
    expect(screen.getByTestId("notebook-frame-stub")).toBeDefined();
    expect(mocks.frameProps.notebookId).toBeUndefined();
  });
});

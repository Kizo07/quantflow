import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it, rs } from "@rstest/core";
import { cleanup, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";

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

rs.mock("next/navigation", () => ({
  useRouter: () => ({ back: rs.fn() }),
}));

rs.mock("@/components/landing/header", () => ({
  Header: () => <div data-testid="site-header" />,
}));

rs.mock("@/components/landing/footer", () => ({
  Footer: () => <div data-testid="site-footer" />,
}));

describe("404 recovery", () => {
  it("renders a way back: Go home link and Go back button with site chrome", async () => {
    cleanup();
    const { default: NotFound } = await import("@/app/not-found");
    render(<NotFound />);
    expect(await screen.findByTestId("site-header")).toBeDefined();
    expect(await screen.findByTestId("site-footer")).toBeDefined();
    const home = await screen.findByRole("link", { name: /go home/i });
    expect(home.getAttribute("href")).toBe("/");
    expect(
      await screen.findByRole("button", { name: /go back/i }),
    ).toBeDefined();
  });

  it("serves spark-mcp-architecture.html from the public dir", () => {
    const publicCopy = join(
      __dirname,
      "..",
      "..",
      "..",
      "public",
      "spark-mcp-architecture.html",
    );
    expect(existsSync(publicCopy)).toBe(true);
    const html = readFileSync(publicCopy, "utf-8");
    expect(html.length).toBeGreaterThan(0);
  });

  it("rewrites legacy absolute report links to the public copy", () => {
    const configPath = join(__dirname, "..", "..", "..", "next.config.js");
    const config = readFileSync(configPath, "utf-8");
    expect(
      config.includes(
        "/home/fire/Documents/quantflow/backend/Audit/spark-mcp-architecture.html",
      ),
    ).toBe(true);
    expect(
      config.includes(
        "/home/fire/Documents/deer-flow/backend/Audit/spark-mcp-architecture.html",
      ),
    ).toBe(true);
    expect(
      config.includes("/home/fire/Documents/Audit/spark-mcp-architecture.html"),
    ).toBe(true);
    expect(config.includes("/spark-mcp-architecture.html")).toBe(true);
  });
});

import { describe, expect, it } from "@rstest/core";
import { cleanup, render, screen } from "@testing-library/react";
import type { JSX } from "react";

import { Section } from "@/components/landing/section";
import { AlphaEngineSection } from "@/components/landing/sections/alpha-engine-section";
import { KizoNLPSection } from "@/components/landing/sections/kizonlp-section";
import { KnowledgeSection } from "@/components/landing/sections/knowledge-section";
import { QuantDeskSection } from "@/components/landing/sections/quant-desk-section";
import { ReportForgeSection } from "@/components/landing/sections/report-forge-section";
import { SparkSection } from "@/components/landing/sections/spark-section";

const SECTIONS: Array<{
  name: string;
  render: () => JSX.Element;
  title: string;
  kicker: string;
}> = [
  {
    name: "QuantDeskSection",
    render: () => <QuantDeskSection />,
    title: "An investment committee, on call",
    kicker: "Harness · Quant Desk",
  },
  {
    name: "KnowledgeSection",
    render: () => <KnowledgeSection />,
    title: "Research memory that outlives the chat",
    kicker: "Harness · Knowledge Plane",
  },
  {
    name: "AlphaEngineSection",
    render: () => <AlphaEngineSection />,
    title: "Market data and research engine",
    kicker: "MCP · alpha_engine",
  },
  {
    name: "KizoNLPSection",
    render: () => <KizoNLPSection />,
    title: "Financial language, measured",
    kicker: "MCP · kizonlp",
  },
  {
    name: "ReportForgeSection",
    render: () => <ReportForgeSection />,
    title: "Flagship reports, mechanically gated",
    kicker: "MCP · reportforge",
  },
  {
    name: "SparkSection",
    render: () => <SparkSection />,
    title: "An institutional memory for ideas",
    kicker: "MCP · spark",
  },
];

describe("landing redesign sections", () => {
  for (const section of SECTIONS) {
    it(`${section.name} renders its kicker and title`, () => {
      try {
        render(section.render());
        expect(screen.getByText(section.title)).not.toBeNull();
        expect(screen.getByText(section.kicker)).not.toBeNull();
      } finally {
        cleanup();
      }
    });

    it(`${section.name} carries no deer-flow branding`, () => {
      try {
        const { container } = render(section.render());
        expect(container.textContent).not.toMatch(/deer-?flow/i);
      } finally {
        cleanup();
      }
    });

    it(`${section.name} uses no gradient headline text`, () => {
      try {
        const { container } = render(section.render());
        expect(
          container.querySelector(
            '[class*="bg-clip-text"], [class*="bg-linear-to-"], [class*="bg-gradient-to-"], [class*="text-transparent"]',
          ),
        ).toBeNull();
      } finally {
        cleanup();
      }
    });
  }

  it("Section renders a solid headline with an optional kicker", () => {
    try {
      const { container } = render(
        <Section title="Solid Title" kicker="Eyebrow" subtitle="Sub">
          body
        </Section>,
      );
      expect(screen.getByText("Solid Title")).not.toBeNull();
      expect(screen.getByText("Eyebrow")).not.toBeNull();
      expect(container.querySelector('[class*="bg-clip-text"]')).toBeNull();
    } finally {
      cleanup();
    }
  });
});

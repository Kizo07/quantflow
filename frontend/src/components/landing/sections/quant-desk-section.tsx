import { cn } from "@/lib/utils";

import { Section } from "../section";

const PHASES = [
  {
    phase: "Phase 0",
    name: "Frame the brief",
    detail:
      "The question becomes a research brief: universe, horizon, objective, constraints — and a freshness check before anything is trusted.",
  },
  {
    phase: "Phase 1",
    name: "Gather evidence in parallel",
    detail:
      "Specialists screen, read, and measure at the same time: quant screens, fresh catalysts, tape setups, earnings, sectors, macro.",
  },
  {
    phase: "Phase 2",
    name: "Debate bull versus bear",
    detail:
      "Independent bull and bear cases collide on the shortlist. Every demotion is recorded with its reason — overlap is signal.",
  },
  {
    phase: "Phase 3",
    name: "Pass the risk veto",
    detail:
      "The risk manager enforces single-name caps, factor concentration limits, and the volatility budget. Vetoes stick.",
  },
  {
    phase: "Phase 4",
    name: "Compile one sourced synthesis",
    detail:
      "A single ranked report with evidence, demotions, a risk dashboard, and caveats. Research only — never financial advice.",
  },
];

const SPECIALISTS = [
  { name: "quant-analyst", role: "Owns every number" },
  { name: "technical-analyst", role: "Setups, regimes, levels" },
  { name: "news-analyst", role: "Catalysts and risk events" },
  { name: "earnings-analyst", role: "Results vs consensus" },
  { name: "sector-researcher", role: "Cycle and peer comps" },
  { name: "macro-analyst", role: "Rates, inflation, scenarios" },
  { name: "thematic-analyst", role: "Theme attribution" },
  { name: "demand-analyst", role: "Volumes, shares, backlog" },
  { name: "document-analyst", role: "Attached filings and notes" },
  { name: "bull-researcher", role: "The bull case" },
  { name: "bear-researcher", role: "The kill case" },
  { name: "risk-manager", role: "Caps, concentration, veto" },
];

export function QuantDeskSection({ className }: { className?: string }) {
  return (
    <Section
      id="quant-desk"
      className={className}
      kicker="Harness · Quant Desk"
      title="An investment committee, on call"
      subtitle="A CIO-style evidence-first workflow: frame, gather, debate, veto, then synthesize."
    >
      <div className="container-md mx-auto mt-8 grid w-full gap-12 px-0 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)] lg:gap-16">
        <ol className="border-km-line min-w-0 border-l-2 pl-0">
          {PHASES.map((step) => (
            <li key={step.phase} className="relative pb-8 pl-8 last:pb-0">
              <span
                aria-hidden="true"
                className="bg-km-cyan absolute top-1.5 -left-[7px] size-3 rounded-full"
              />
              <p className="text-km-gold font-mono text-xs font-medium tracking-[0.18em] uppercase">
                {step.phase}
              </p>
              <h3 className="text-foreground mt-1 text-xl font-semibold">
                {step.name}
              </h3>
              <p className="text-km-muted mt-1 max-w-prose text-base leading-7">
                {step.detail}
              </p>
            </li>
          ))}
        </ol>
        <div className="min-w-0">
          <h3 className="text-km-faint font-mono text-xs font-medium tracking-[0.18em] uppercase">
            Twelve specialists
          </h3>
          <dl className="border-km-line mt-4 grid grid-cols-1 gap-x-8 border-b sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
            {SPECIALISTS.map((specialist) => (
              <div
                key={specialist.name}
                className="border-km-line border-t py-3"
              >
                <dt className="text-km-cyan-bright font-mono text-sm font-medium">
                  {specialist.name}
                </dt>
                <dd className="text-km-muted text-sm">{specialist.role}</dd>
              </div>
            ))}
          </dl>
          <p
            className={cn(
              "border-km-line bg-km-surface text-km-muted mt-6 rounded-lg border p-4 text-sm leading-6",
            )}
          >
            Runs behind an OpenAI-compatible model gateway, so the Muse Spark
            family, Qwen, GLM, DeepSeek, and Kimi all plug in through{" "}
            <code className="text-km-text font-mono text-[0.8125rem]">
              config.example.yaml
            </code>
            .
          </p>
        </div>
      </div>
    </Section>
  );
}

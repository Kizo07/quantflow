import { Section } from "../section";

const GROUPS = [
  {
    name: "Data & universe",
    tools: [
      "engine_status",
      "list_instruments",
      "get_price_history",
      "get_latest_quotes",
      "cross_section_returns",
      "universe_members_as_of",
      "refresh_data",
    ],
  },
  {
    name: "Vendor series",
    tools: ["get_lse_history", "get_lse_yields", "get_lse_fundamentals"],
  },
  {
    name: "Factors & signals",
    tools: ["get_factor_history", "run_factor_analysis", "ml_signal_scores"],
  },
  {
    name: "Backtests",
    tools: [
      "run_momentum_backtest",
      "run_param_sweep",
      "run_backtest",
      "causal_test",
    ],
  },
  {
    name: "Portfolio",
    tools: [
      "optimize_portfolio",
      "attribute_portfolio",
      "portfolio_risk_decomposition",
      "risk_decompose",
      "brinson_attribution",
    ],
  },
  {
    name: "Charts & tearsheets",
    tools: [
      "chart_price",
      "chart_bars",
      "chart_fundamentals",
      "chart_fan",
      "render_report",
    ],
  },
];

export function AlphaEngineSection({ className }: { className?: string }) {
  return (
    <Section
      id="alpha-engine"
      className={className}
      kicker="MCP · alpha_engine"
      title="Market data and research engine"
      subtitle="27 tools, one freshness-first toolkit: S&P 500 prices, factors, backtests, portfolio math, and tearsheets."
    >
      <div className="container-md mx-auto mt-8">
        <dl className="border-km-line border-t">
          {GROUPS.map((group) => (
            <div
              key={group.name}
              className="border-km-line grid grid-cols-1 gap-3 border-b py-5 md:grid-cols-[minmax(0,11rem)_minmax(0,1fr)] md:gap-6"
            >
              <dt className="text-km-gold font-mono text-sm font-medium">
                {group.name}
              </dt>
              <dd className="flex min-w-0 flex-wrap gap-2">
                {group.tools.map((tool) => (
                  <code
                    key={tool}
                    className="border-km-line bg-km-surface text-km-cyan-bright rounded-md border px-2.5 py-1 font-mono text-[0.8125rem]"
                  >
                    {tool}
                  </code>
                ))}
              </dd>
            </div>
          ))}
        </dl>
        <p className="text-km-muted mt-4 text-sm leading-6">
          Freshness gates everything: the desk checks{" "}
          <code className="text-km-text font-mono text-[0.8125rem]">
            engine_status
          </code>{" "}
          before trusting a screen, and historical selections always resolve
          through point-in-time membership — never the current snapshot.
        </p>
      </div>
    </Section>
  );
}

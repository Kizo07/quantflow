import { Section } from "../section";

const TRIALS = [
  {
    hypothesis: "126-day momentum, top 20",
    trials: "M = 20",
    verdict: "DSR 0.92 · promote",
    outcome: "Shipped to desk screen",
  },
  {
    hypothesis: "5-day reversal overlay",
    trials: "M = 20",
    verdict: "DSR 0.31 · hold",
    outcome: "Parked, not tombstoned",
  },
  {
    hypothesis: "Earnings-week timing rule",
    trials: "M = 20",
    verdict: "DSR −0.14 · reject",
    outcome: "Tombstoned with rationale",
  },
];

export function SparkSection({ className }: { className?: string }) {
  return (
    <Section
      id="spark"
      className={className}
      kicker="MCP · spark"
      title="An institutional memory for ideas"
      subtitle="A hypothesis registry with trial accounting, so verdicts are deflated instead of lucky."
    >
      <div className="container-md mx-auto mt-8">
        <div className="border-km-line overflow-x-auto rounded-lg border">
          <table className="bg-km-surface w-full min-w-[36rem] border-collapse text-left text-sm">
            <caption className="sr-only">
              Illustrative trial ledger: hypotheses with trial counts,
              significance verdicts, and outcomes
            </caption>
            <thead>
              <tr className="border-km-line text-km-faint border-b font-mono text-xs tracking-[0.14em] uppercase">
                <th scope="col" className="px-5 py-3 font-medium">
                  Hypothesis
                </th>
                <th scope="col" className="px-5 py-3 font-medium">
                  Trials
                </th>
                <th scope="col" className="px-5 py-3 font-medium">
                  Gate verdict
                </th>
                <th scope="col" className="px-5 py-3 font-medium">
                  Outcome
                </th>
              </tr>
            </thead>
            <tbody>
              {TRIALS.map((trial) => (
                <tr
                  key={trial.hypothesis}
                  className="border-km-line border-b leading-6 last:border-b-0"
                >
                  <th
                    scope="row"
                    className="text-km-text px-5 py-3.5 font-medium"
                  >
                    {trial.hypothesis}
                  </th>
                  <td className="text-km-muted px-5 py-3.5 font-mono">
                    {trial.trials}
                  </td>
                  <td className="text-km-cyan-bright px-5 py-3.5 font-mono">
                    {trial.verdict}
                  </td>
                  <td className="text-km-muted px-5 py-3.5">{trial.outcome}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-km-muted mt-4 max-w-prose text-sm leading-6">
          Every trial is counted, failures surface as tombstones that block
          re-spend, and novelty checks point at unexplored research space.
          Significance gates run on deflated ratios over the family&apos;s trial
          count — a lone high Sharpe without one is luck. Rows above are
          illustrative.
        </p>
      </div>
    </Section>
  );
}

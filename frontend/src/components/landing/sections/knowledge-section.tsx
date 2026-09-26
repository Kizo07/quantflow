import { Section } from "../section";

const STATS = [
  {
    value: "recall@10 ≥ 0.8",
    label: "Sectioned exit gate on the committed eval fixture",
  },
  {
    value: "SHA-256",
    label: "Content-addressed artifacts with dataset vintages",
  },
  {
    value: "RRF fusion",
    label: "Structured + lexical + vector + failure channels",
  },
  {
    value: "First-class",
    label: "Experiments, assumptions, and failures as shared state",
  },
];

export function KnowledgeSection({ className }: { className?: string }) {
  return (
    <Section
      id="knowledge"
      className={className}
      kicker="Harness · Knowledge Plane"
      title="Research memory that outlives the chat"
      subtitle="Experiments, failures, and evidence as shared, queryable state — not buried in conversation history."
    >
      <div className="container-md mx-auto mt-8">
        <dl className="border-km-line bg-km-line grid grid-cols-1 gap-px overflow-hidden rounded-lg border sm:grid-cols-2 lg:grid-cols-4">
          {STATS.map((stat) => (
            <div
              key={stat.value}
              className="bg-km-surface flex flex-col gap-1 p-6"
            >
              <dt className="text-km-muted order-2 text-sm leading-6">
                {stat.label}
              </dt>
              <dd className="text-km-gold order-1 font-mono text-2xl font-semibold">
                {stat.value}
              </dd>
            </div>
          ))}
        </dl>
        <div className="mt-6 flex flex-col items-start gap-3 sm:flex-row sm:items-center">
          <div className="flex flex-wrap gap-2">
            <code className="border-km-line bg-km-surface-2 text-km-cyan-bright rounded-md border px-3 py-1.5 font-mono text-sm">
              ledger_search
            </code>
            <code className="border-km-line bg-km-surface-2 text-km-cyan-bright rounded-md border px-3 py-1.5 font-mono text-sm">
              ledger_get
            </code>
          </div>
          <p className="text-km-muted text-sm">
            Two tools query the whole plane. Knowledge content stays local and
            gitignored — the harness ships the mechanism, never your data.
          </p>
        </div>
      </div>
    </Section>
  );
}

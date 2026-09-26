import { Section } from "../section";

const STEPS = [
  {
    step: "01",
    name: "Scaffold",
    detail: "Branded project, verdict band, and exhibit plan up front.",
  },
  {
    step: "02",
    name: "Write",
    detail: "One source in Quarto — prose, tables, and live charts.",
  },
  {
    step: "03",
    name: "Gate",
    detail: "Mechanical checks refuse to ship bad exhibits.",
  },
  {
    step: "04",
    name: "Publish",
    detail: "Rendered to HTML, PDF, and DOCX from the same source.",
  },
];

const FORMATS = ["HTML", "PDF", "DOCX"];

export function ReportForgeSection({ className }: { className?: string }) {
  return (
    <Section
      id="report-forge"
      className={className}
      kicker="MCP · reportforge"
      title="Flagship reports, mechanically gated"
      subtitle="One-source publishing: scaffold, write, gate, publish — with infographic covers and numbered exhibits."
    >
      <div className="container-md mx-auto mt-8">
        <ol className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {STEPS.map((step, index) => (
            <li key={step.step} className="relative min-w-0">
              <p className="text-km-gold font-mono text-4xl font-semibold">
                {step.step}
              </p>
              <h3 className="text-foreground mt-2 flex items-center gap-2 text-lg font-semibold">
                {step.name}
                {index < STEPS.length - 1 && (
                  <span
                    aria-hidden="true"
                    className="text-km-faint hidden font-mono lg:inline"
                  >
                    →
                  </span>
                )}
              </h3>
              <p className="text-km-muted mt-1 text-sm leading-6">
                {step.detail}
              </p>
            </li>
          ))}
        </ol>
        <div className="border-km-line mt-8 flex flex-col gap-3 border-t pt-6 sm:flex-row sm:items-center">
          <div className="flex gap-2" aria-label="Output formats">
            {FORMATS.map((format) => (
              <span
                key={format}
                className="bg-km-surface-2 text-km-text rounded-md px-3 py-1.5 font-mono text-sm font-medium"
              >
                {format}
              </span>
            ))}
          </div>
          <p className="text-km-muted text-sm leading-6">
            16 templates — 8 base plus 8 domain briefs — with a{" "}
            <code className="text-km-text font-mono text-[0.8125rem]">
              bespoke
            </code>{" "}
            escape hatch, provenance-burned charts, and spreadsheet tables.
          </p>
        </div>
      </div>
    </Section>
  );
}

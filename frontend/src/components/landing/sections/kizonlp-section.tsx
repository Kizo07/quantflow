import { Section } from "../section";

const LABELS = [
  { name: "guidance", score: "0.91" },
  { name: "forward-looking", score: "0.84" },
  { name: "risk", score: "0.22" },
];

const KEYPHRASES = [
  "raised full-year outlook",
  "record backlog",
  "margin pressure",
];

export function KizoNLPSection({ className }: { className?: string }) {
  return (
    <Section
      id="kizonlp"
      className={className}
      kicker="MCP · kizonlp"
      title="Financial language, measured"
      subtitle="Transformer tone scoring, zero-shot labels, summarization, and keyphrases feeding the news and document analysts."
    >
      <div className="container-md mx-auto mt-8 grid w-full gap-6 lg:grid-cols-2">
        <figure className="border-km-line bg-km-surface flex min-w-0 flex-col rounded-lg border p-6">
          <figcaption className="text-km-faint font-mono text-xs font-medium tracking-[0.18em] uppercase">
            Sample input · earnings call excerpt
          </figcaption>
          <blockquote className="text-km-text mt-4 text-base leading-7">
            “We are raising our full-year outlook on the back of a record
            backlog, though component costs keep near-term margins under
            pressure.”
          </blockquote>
          <p className="text-km-muted mt-4 text-sm">
            PDF text from releases, filings, and transcripts flows through{" "}
            <code className="text-km-text font-mono text-[0.8125rem]">
              pdf_text
            </code>{" "}
            first.
          </p>
        </figure>
        <div className="border-km-line bg-km-surface min-w-0 rounded-lg border p-6">
          <p className="text-km-faint font-mono text-xs font-medium tracking-[0.18em] uppercase">
            Sample output · illustrative scores
          </p>
          <div className="mt-4 flex items-baseline gap-3">
            <p className="text-km-gold font-mono text-2xl font-semibold">
              cautiously positive
            </p>
            <p className="text-km-muted font-mono text-sm">fin_sentiment</p>
          </div>
          <div
            className="bg-km-surface-2 mt-3 h-2 w-full overflow-hidden rounded-full"
            role="img"
            aria-label="Illustrative sentiment meter at 68 percent positive"
          >
            <div className="bg-km-cyan h-full w-[68%] rounded-full" />
          </div>
          <ul className="mt-5 space-y-2">
            {LABELS.map((label) => (
              <li
                key={label.name}
                className="flex items-baseline justify-between gap-4 font-mono text-sm"
              >
                <span className="text-km-cyan-bright">{label.name}</span>
                <span className="text-km-muted">{label.score}</span>
              </li>
            ))}
          </ul>
          <div className="mt-5 flex flex-wrap gap-2">
            {KEYPHRASES.map((phrase) => (
              <span
                key={phrase}
                className="border-km-line text-km-muted rounded-full border px-3 py-1 text-sm"
              >
                {phrase}
              </span>
            ))}
          </div>
        </div>
      </div>
    </Section>
  );
}

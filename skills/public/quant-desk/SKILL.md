---
name: quant-desk
description: Use for any analysis of listed stocks, ETFs, sectors, markets, factors, trading signals, strategies, backtests, portfolios, risk, financial news sentiment or company filings, short questions included (e.g. "is momentum still working?", "NVDA vs AMD", "build a defensive portfolio", "how is the market reading TSLA news?", "test this signal idea", "summarize this 10-K"). Runs on the local alpha_engine, kizonlp, spark and reportforge tools and scales from a 1-3 call lookup to direct analysis or a full desk run with subagents and a PDF report.
---

# QuantFlow Quant Desk

Research only, never personal financial advice. Every number you report carries its as-of date.
In anything a reader sees, the product name is QuantFlow.

## 1. Pick the tier first

| Tier | When | How |
|---|---|---|
| **Q** lookup | One fact: a quote, a return, index membership on a date, a factor's recent return | 1–3 tool calls (`alpha_engine_engine_status`, then the one tool). Answer with the as-of. No report. |
| **A** direct (default) | One question: 1–5 names, one factor or signal, one portfolio, one news or filing read | Run the §3 recipes yourself. **The only tier when `task` is not in your tools** (Pro mode). |
| **D** desk | Ideas across a universe, a multi-name recommendation, a portfolio proposal or an IC-style report, **and** `task` is in your tools | §6. |

Desk-sized ask but no `task` tool: run tier A once per track (quant, news, technical, documents if
attached, risk), add a "Strongest bear points" section, label the result "non-debated", and say in
one line that Ultra mode runs the full desk.

## 2. Frame

1. Restate universe, horizon, objective and constraints.
2. Call `alpha_engine_engine_status` once, first. Put the prices `latest` date and any freshness
   `breaches` in the header. Disclose stale data; never refresh it yourself (refresh is a scheduled
   job) and never invent missing data.
3. Before designing a study, check prior work: `spark_research_history` (trials, tombstones) and,
   if `ledger_search` is in your tools, prior desk findings.

## 3. Evidence: tool order and recipes

Order for any number: **alpha_engine op → `alpha_engine_sql`** (read-only SQL; `alpha_engine_describe`
lists the relations) **→ `reportforge_run_code` only on values a tool returned → web** only for facts
not stored locally. Never read alpha_engine data files from any code runner. No package installs.

| Question | Chain |
|---|---|
| Single-name outlook | `alpha_engine_get_price_history` → `alpha_engine_get_lse_fundamentals` (descriptive, not PIT) → news chain below |
| Compare names | Same per name, plus `alpha_engine_portfolio_risk_decomposition` on an equal-weight book for style and sector exposure |
| Does factor or signal X work | `alpha_engine_run_factor_analysis` (DSL `expression` or `signal_name`) with `alpha_engine_get_factor_history` as benchmark; momentum: `alpha_engine_run_momentum_backtest`; robustness: `alpha_engine_run_param_sweep` (every cell is a trial). New idea → §5 |
| Stock ideas, screens | `alpha_engine_ml_signal_scores` → `alpha_engine_sql` for custom filters. New rule → §5 |
| Market-wide (breadth, dispersion, sector moves) | One `alpha_engine_cross_section_returns`, never a per-ticker loop |
| Index membership on a date | `alpha_engine_universe_members_as_of` |
| Portfolio and risk | `alpha_engine_optimize_portfolio` → `alpha_engine_portfolio_risk_decomposition` → `alpha_engine_attribute_portfolio` or `alpha_engine_brinson_attribution` |
| Rates, macro, earnings dates | `alpha_engine_get_lse_yields`; `alpha_engine_sql` on `ust_yields_daily`, `fred_series`, `earnings_dates`; web for prints not stored |
| News, sentiment | `web_search_advanced_exa` (date-bounded) or `web_search` → `web_fetch_exa` (pages over ~4 KB) or `web_fetch` → `kizonlp_fin_sentiment` (one company) or `kizonlp_entity_sentiment` (several) → `kizonlp_zero_shot_classify` (catalyst type) → `alpha_engine_get_price_history` (price reaction) |
| Filings, documents | `kizonlp_parse_document` → `kizonlp_rerank` / `kizonlp_extract` / `kizonlp_forward_looking` → `kizonlp_check_faithfulness` on any summary before you quote it |

Every tool with its gotchas: `references/tool-catalog.md`.

**Finance text goes through kizonlp.** Tone, catalyst type, guidance language and entities come from a
kizonlp score, not from your own reading; report label and score. `texts` is a JSON array; score
sentences or paragraphs, not whole articles. Embedding-backed tools (`kizonlp_extract_keyphrases`,
`kizonlp_fit_topics`) use the Gemini embedder: on a Gemini quota or 429 error,
retry once with `model="harrier-0.6b"` (`embedding_model="harrier-0.6b"` for `kizonlp_fit_topics`), then
report the failure. If `kizonlp_parse_document` refuses an upload path, `read_file` the upload (or
its converted `.md`) and say so.

## 4. Stamps and citations

- Cite local evidence as `[tool_name, as-of YYYY-MM-DD]`, plus the `snapshot_id` that `alpha_engine_sql` returns.
- Backtests and screens keep the default `universe_basis="pit"`. Quote `universe_basis`, `pit_coverage`
  and `n_priced` against `n_universe_pit`. Reject a `snapshot` result unless the user approved it.
- Caller-list tools (optimize, attribute, Brinson, causal, `alpha_engine_risk_decompose`,
  cross-section) are disclosed with their stamp and never called PIT.
- Read `references/data-governance.md` before any historical-universe question, backtest or stale-data call.

## 5. Spark governance (mandatory for self-built signals)

Applies to any signal, screen, factor expression, ML combination or timing rule that you or the user
constructed and that you backtest and present as evidence. Descriptive questions about well-known
factors need only `spark_research_history`.

**Both paths start:** `spark_check_novelty` → `spark_hypothesis_propose` → `spark_register_hypothesis`
(returns `episode_id`), before any backtest.

**Path A, DSR-gated** (momentum or reversal signals): `spark_backtest_run` → `spark_significance_gate`
→ `spark_log_outcome`. Quote a Sharpe ratio only with its DSR.

**Path B, direct alpha_engine strategy tool** (`alpha_engine_run_factor_analysis`,
`alpha_engine_run_momentum_backtest`, `alpha_engine_ml_signal_scores`, `alpha_engine_run_param_sweep`):
run the tool → `spark_log_outcome` with the metrics it returned. Label the result "registered, not
DSR-gated". Do not call the gate and do not hand-compute skew, kurtosis or T.

```
spark_hypothesis_propose(
  idea="12-1 momentum (skip the latest month) beats plain 12-month momentum",
  family="quant", universe="S&P 500 PIT", horizon="20d",
  scope="US large caps, monthly rebalance, 2010-2026",
  mechanism="short-term reversal contaminates the last month of the formation window",
  expected_direction="positive",
  falsifier="out-of-sample Sharpe spread vs plain momentum <= 0 net of 5 bps costs",
  predicted_ic=0.03, predicted_sigma=0.02, hypothesis_type="predictive")
```

Close every registered episode with `spark_log_outcome`, failures included. Spark runs without an
LLM: novelty is a lexical proxy, idea tools return templated hints, and there is no
LLM debate step. Specs, outcomes, tombstones and sweeps: `references/spark-protocol.md`.

## 6. Desk mode (tier D)

1. **Evidence** (parallel, at most 3 at once):
   - quant-analyst: screens, ranks, backtests, sector comps; owns the spark calls.
   - news-analyst: dated catalysts with kizonlp scores, earnings and macro prints, themes.
   - technical-analyst: focal tickers only.
   - document-analyst: only with attached files.

   Merge into a long list (about 15–25 names) and show which track each name came from.
2. **Debate:** bull-researcher and bear-researcher together on the ~10-name shortlist. Keep names
   that survive both. Demote where bear kill-cases are severe and bull rebuttals weak, and record
   each demotion with its reason.
3. **Risk:** risk-manager on the survivors with the risk budget. Enforce its vetoes.
4. **Synthesis:** you (§7).

Only these seven roles exist; never use `general-purpose` for a desk track. Demand and TAM questions
go to your own web leg. Budget 5–7 delegations, at most 12. A failed track is noted in Caveats,
never filled in.

## 7. Synthesis

```
# Research Brief: <question>
As-of: <data date> · Universe: <universe> · Horizon: <horizon> · Data notes: <stale datasets, bases>
## Conclusions / picks (ranked)
| # | Name | Quant evidence | News signal | Debate outcome | Risk check |
## Evidence highlights (2–3 cited bullets per pick)
## What changed our mind (demotions and why)
## Risk dashboard (vol, concentration, top factor exposures)
## Caveats and failure modes
## Appendix: ledger trail (spark ids) and subagent briefs
```

- Every number traces to a tool result or a subagent brief; add no unsourced numbers.
- A result without the bull and bear debate is labelled "non-debated".
- Never invent failure reasons: a call you claim failed must appear in your tool results.
- On a validation error, fix the arguments or switch tools; never repeat an identical call more than twice.

## 8. Reports

For a PDF, DOCX or HTML deliverable, read `references/reporting.md` first. Every report run ends
with `reportforge_render_report` → `reportforge_publish_report` → `present_files` with the returned
`present_paths`. A Markdown answer is written under `/mnt/user-data/outputs` and presented with
`present_files`.

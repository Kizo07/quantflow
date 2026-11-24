# Tool catalog (research set)

Exact model-visible names. Order for any number: alpha_engine op → `alpha_engine_sql` →
`reportforge_run_code` on tool-returned values → web for facts not stored locally.
Tools not listed here are operator, admin or human-gate tools: don't call them.

## alpha_engine: local US-equity data, analytics, portfolios

| Tool | Use when | Gotchas |
|---|---|---|
| `alpha_engine_engine_status` | First call in any market-data task: coverage and freshness | Quote prices `latest` and `breaches`; call once per run |
| `alpha_engine_describe` | Which op fits; relations and columns before writing SQL | `detail=true` is large |
| `alpha_engine_sql` | Any slice, screen, ranking, rolling/window stat, earnings dates, yields, FRED | One read-only SELECT; no PIT masking (join `sp500_membership_intervals`); cite `snapshot_id` |
| `alpha_engine_list_instruments` | Current constituents, ticker and sector lookup | Current snapshot only; never for past dates |
| `alpha_engine_get_price_history` | Daily OHLCV for one symbol | Returns the last `limit` rows (default 250); raise `limit` for long windows; don't loop tickers |
| `alpha_engine_get_latest_quotes` | Latest close and volume for a few symbols | End-of-day, not live |
| `alpha_engine_get_lse_history` | Vendor candles back to 2003 | Local store only |
| `alpha_engine_get_lse_yields` | US2/5/10/30Y, DE10Y, UK10Y, JP10Y | Full UST curve: `alpha_engine_sql` on `ust_yields_daily` |
| `alpha_engine_get_lse_fundamentals` | Descriptive statement rows | Not PIT (no filing date): never in backtests |
| `alpha_engine_get_factor_history` | FF, AQR, JKP, global-q factor returns | AQR/JKP month-end; FF lags by months |
| `alpha_engine_universe_members_as_of` | PIT S&P 500 roster on a date (1996 on) | Disclose `pit_tail_provisional`; `snapshot-fallback` is never PIT |
| `alpha_engine_cross_section_returns` | Breadth, dispersion, "how many went up", sector moves | One call, not a loop; basis `prices-availability` |
| `alpha_engine_run_factor_analysis` | "Does factor/signal X work?": IC, quantile spreads, long-short | Price/volume-only DSL; PIT default; not DSR-gateable |
| `alpha_engine_run_momentum_backtest` | Top-N cross-sectional momentum with fees | PIT default; window clipped to PIT coverage |
| `alpha_engine_run_param_sweep` | Lookback × top_n robustness grid | Every cell is a trial; never quote only the best cell |
| `alpha_engine_ml_signal_scores` | Walk-forward ridge/OLS/RF ranking of candidates | PIT required; heavy |
| `alpha_engine_run_backtest` | Momentum/reversal backtest returning `sr,T,skew,kurt` | Reach it through `spark_backtest_run` once a hypothesis exists |
| `alpha_engine_optimize_portfolio` | Weights: max_sharpe, min_volatility, max_return, HRP, Black-Litterman | Caller list; `max_weight` is the only constraint |
| `alpha_engine_portfolio_risk_decomposition` | Style + sector risk: total, common, specific vol, exposures | PIT required; don't mix with `alpha_engine_risk_decompose` numbers |
| `alpha_engine_attribute_portfolio` | Factor regression of fixed weights | Caller list; factors must be valid ids |
| `alpha_engine_brinson_attribution` | Brinson-Fachler allocation/selection/interaction | Default benchmark is survivorship-biased; disclose `benchmark_basis` |
| `alpha_engine_risk_decompose` | FF3/FF5/Carhart regression of a book | Inside a spark episode use `spark_risk_decompose` |
| `alpha_engine_causal_test` | Pairwise Granger F-tests | Inside a spark episode use `spark_causal_check`; PCMCI refused |
| `alpha_engine_chart_price`, `alpha_engine_chart_bars`, `alpha_engine_chart_fan`, `alpha_engine_chart_fundamentals` | Themed PNG exhibits | Host path: copy into the report project before use |
| `alpha_engine_render_report` | Interactive HTML tear sheet from engine outputs | For PDF/DOCX use reportforge |

## kizonlp: financial NLP on text you already hold

| Tool | Use when | Gotchas |
|---|---|---|
| `kizonlp_kizo_status` | Caps, allowed roots, defaults before file or bulk work | — |
| `kizonlp_fin_sentiment` | Sentiment of headlines, paragraphs, filing or call excerpts | Labels positive/negative/neutral; score sentences, not whole articles |
| `kizonlp_entity_sentiment` | Per-company sentiment in multi-company text | `aggregate_by=ticker` |
| `kizonlp_zero_shot_classify` | Catalyst type, risk category, relevance labels | ≤ 50 labels |
| `kizonlp_forward_looking` | Guidance language in filings or calls | Segment into sentences first |
| `kizonlp_segment` | Sentences, paragraphs, speaker turns, chunks | Prep step for scoring |
| `kizonlp_rerank` | Pick the passages or articles worth reading | ≤ 500 docs |
| `kizonlp_entities` | Map text to tickers | — |
| `kizonlp_parse_document` | PDF, SEC htm/iXBRL, DOCX, PPTX, XLSX → Markdown | Path must be inside the allowed roots; OCR modes are slow |
| `kizonlp_extract` | Schema JSON with verbatim quotes (KPIs, guidance, risks) | Local LLM: slow cold start |
| `kizonlp_summarize` | Very long or batch documents; `mode=headline` | Local LLM: slow cold start |
| `kizonlp_check_faithfulness` | Verify a summary against its source before quoting | — |
| `kizonlp_extract_keyphrases` | Theme tagging | Gemini embedder; on 429 retry once with `model="harrier-0.6b"` |
| `kizonlp_fit_topics` | Cluster many headlines into themes (5–5000 docs) | Gemini embedder; on 429 retry once with `embedding_model="harrier-0.6b"` |

## spark: research ledger and trial accounting

| Tool | Use when | Gotchas |
|---|---|---|
| `spark_research_history` | Before designing any signal or screen: past trials, tombstones | FTS5-only retrieval |
| `spark_memory_recall` | "What do we believe about X" | — |
| `spark_check_novelty` | Before proposing a hypothesis | `family` from the fixed list |
| `spark_hypothesis_propose` | Register a falsifiable hypothesis | All C6 fields required |
| `spark_register_hypothesis` | Create the episode before any backtest | Returns `episode_id` |
| `spark_backtest_run` | Governed momentum/reversal backtest (counted trial) | Tombstone pre-check |
| `spark_significance_gate` | DSR before quoting a Sharpe ratio | Needs `sr,T,skew,kurt`: path A only |
| `spark_log_outcome` | Close every registered episode, failures included | Path B: never `validated` |
| `spark_causal_check`, `spark_risk_decompose` | Episode-scoped causal and risk analytics | — |
| `spark_suggest_next`, `spark_explore_methods`, `spark_find_analogies`, `spark_combine_methods` | When stuck | No LLM: templated hints |
| `spark_trace_hypothesis`, `spark_report_write`, `spark_archive_status`, `spark_sota_status` | Audit trail and appendix | Read-only |

## reportforge: reports and host Python

| Tool | Use when |
|---|---|
| `reportforge_capabilities`, `reportforge_list_templates`, `reportforge_list_recipes`, `reportforge_describe_vocabulary` | Start of a report task |
| `reportforge_scaffold_from_brief`, `reportforge_compose_report`, `reportforge_scaffold_report` | Create the project (prefer `reportforge_scaffold_from_brief`; `formats` is a list) |
| `reportforge_write_report_body`, `reportforge_read_report`, `reportforge_edit_report`, `reportforge_append_section` | Write and revise |
| `reportforge_save_table`, `reportforge_save_chart`, `reportforge_save_asset` | Tables, PNG charts, assets |
| `reportforge_register_source`, `reportforge_register_facts`, `reportforge_register_exhibit` | Evidence registry |
| `reportforge_check_readiness`, `reportforge_render_report`, `reportforge_render_preview`, `reportforge_output_qa` | Check, render, look |
| `reportforge_publish_report` | Mandatory before `present_files` |
| `reportforge_project_status`, `reportforge_read_project_file` | Failure loop |
| `reportforge_run_code` | Host Python on tool-returned values only; never reads alpha_engine data |
| `reportforge_notebook_create`, `reportforge_notebook_read`, `reportforge_notebook_create_cell`, `reportforge_notebook_write_cell`, `reportforge_notebook_run_cell`, `reportforge_notebook_run_all`, `reportforge_run_cell_persistent` | Notebooks; edits and runs are human-gated (see reporting.md) |

Never: `reportforge_run_file`, release/lifecycle/approval tools.

## Web and built-ins

| Tool | Use when |
|---|---|
| `web_search_advanced_exa` | Date-bounded news and filings search |
| `web_fetch_exa` | Full text of pages over ~4 KB |
| `web_search`, `web_fetch` | Quick search and http(s) fetch; never for prices, returns, factors or fundamentals |
| `read_file`, `list_uploaded_files` | Uploads and sandbox files (not alpha_engine or report-forge host dirs) |
| `task` | Desk subagents (tier D only) |
| `present_files` | Deliver files from the thread outputs directory; last action |
| `ask_clarification` | Blocking question; notebook risk confirmation |
| `ledger_search`, `ledger_get`, `experiment_get` | Prior desk findings (lead only, when bound) |

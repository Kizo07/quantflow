# Data governance: freshness, PIT universe, stamps

Read this before any historical-universe question, any backtest or screen, and whenever
`alpha_engine_engine_status` reports stale data.

## Data stack (LSE-first)

- Prices are LSE vendor candles (split-adjusted; `adj_close` recomputed from LSE cash dividends).
  Seven tickers (BNY, ECHO, FDXF, FLEX, HONA, MRSH, VMRK) miss LSE and fall back to yahoo/Stooq
  automatically per file. Never present yahoo as the primary source.
- Vendor extras: `alpha_engine_get_lse_history` (candles back to 2003), `alpha_engine_get_lse_yields`
  (US2/5/10/30Y, DE10Y, UK10Y, JP10Y), `alpha_engine_get_lse_fundamentals` (statements and dividends).
- Prices are end-of-day as of `prices.latest`, never live quotes.
- **Fundamentals are not PIT.** LSE fundamentals carry period end only, no filing date. Use them
  descriptively; never in a backtest or an as-of factor, and disclose the look-ahead risk.
- The factor DSL (`alpha_engine_run_factor_analysis` `expression`) takes `CLOSE`, `OPEN`, `HIGH`,
  `LOW` and `VOLUME` only.
- Not available locally: PIT fundamentals, options, intraday, FX, crypto; macro is a few FRED series.
  Say so instead of approximating.

## Freshness gate (mandatory)

1. Call `alpha_engine_engine_status` once at the start and read its `freshness` block.
2. `breaches` non-empty: disclose in the header ("Data as-of: DATE; N datasets stale: LIST").
   Never silently proceed. Do not try to refresh: data refresh is a scheduled operator job, and
   no agent tool refreshes factor or alternative datasets.
3. A dataset with no `latest` date, or a verdict of `EMPTY`, does not exist for this run. Do not
   invent its contents. Empty datasets can also show up as `STALE`; check `latest` before using one.
4. Vendor-lagged sources (`ff_factors_daily`, `jkp_factors_monthly`, `globalq_factors`, AQR) lag by a
   month or a quarter by design. Disclose the end date; it is not a defect.
5. As-of contract: every table carries an explicit as-of that matches
   `freshness.datasets.<primary dataset>.latest`.

## Universe rule (survivorship-bias-free, mandatory)

alpha_engine is the single source of truth for S&P 500 membership. Never hand-assemble ticker lists.

- **As-of questions** ("what was in the index on DATE", as-of exposures): call
  `alpha_engine_universe_members_as_of(as_of=DATE)` and use exactly that set. Check the stamps
  (`universe_basis`, `pit_covered`, `pit_tail_provisional`, `pit_observed_end`, `pit_extended_to`).
  A `pit_tail_provisional: true` roster is the censored extension past `pit_observed_end` and must be
  disclosed as provisional. A snapshot-fallback result (`universe_basis: snapshot-fallback`,
  `pit_covered: false`, warning present) must be disclosed and never quoted as PIT. Never apply the
  current snapshot (`alpha_engine_list_instruments`) to past dates: delisted names (ENRNQ, LEHMQ, …)
  would silently vanish.
- **Backtests** (`alpha_engine_run_momentum_backtest`, `alpha_engine_run_factor_analysis`,
  `alpha_engine_run_param_sweep`, `alpha_engine_ml_signal_scores`,
  `alpha_engine_portfolio_risk_decomposition`): do not pre-fetch a list. Call the tool with its
  default `universe_basis="pit"` and let the engine resolve membership per rebalance date. Verify the
  returned stamps (`universe_basis`, `pit_coverage`, `member_counts`). A result with
  `universe_basis: "snapshot"` (or `survivorship_biased: true`) may be reported only if the user
  explicitly approved snapshot, and the report header must disclose it. The reject rule applies only
  to tools that return `universe_basis`; the caller-supplied surfaces below are disclosed, not rejected.
- **Survivor subsets before about 2016–2020.** PIT membership covers about 1,200 tickers but only
  about 510 have prices. Results stamped `sp500-pit` for early windows are survivor subsets: always
  quote `n_priced` against `n_universe_pit`. Windows are clipped to PIT coverage
  (`pit_window_start_clipped_to`); report the clipped start.
- **`alpha_engine_run_backtest` (the spark-contract backtest, reached through `spark_backtest_run`)**
  takes a caller list: pass `"SYM1,SYM2,…@YYYY-MM-DD"` in `universe_pit` so the engine validates every
  name against PIT membership on that single date (strict, fail-loud, never silently shrinks the
  list). Check `universe_basis`, `universe_excluded`, `n_validated` and
  `universe_as_of_outside_window`. Without `@date` the basis is `caller-list-unvalidated`.
  Single-date validation is not time-varying survivorship-freedom; disclose it as such.
- **Caller-supplied surfaces:** `alpha_engine_optimize_portfolio`, `alpha_engine_attribute_portfolio`,
  `alpha_engine_causal_test` and `alpha_engine_risk_decompose` are stamped `caller-list-unvalidated`;
  `alpha_engine_brinson_attribution` is stamped `benchmark_basis`/`portfolio_basis` (its default
  benchmark is snapshot equal-weight and survivorship-biased historically);
  `alpha_engine_cross_section_returns` is stamped `prices-availability`, `pit_covered: false` (every
  name with local prices that day, not PIT index membership). Disclose the stamped basis in the
  brief, never reject them for lacking `sp500-pit`, and never quote them as PIT.
- Delisted tickers carry Q-suffixes and have no vendor prices. Treat them as untradable (drop with
  disclosure), not as errors.

## Two risk models: never mix their numbers

- `alpha_engine_portfolio_risk_decomposition`: style (momentum, vol, size) plus sector
  cross-sectional model; total, common and specific vol. PIT required.
- `alpha_engine_risk_decompose` (or `spark_risk_decompose` inside a spark episode): FF3/FF5/Carhart
  regression of a book. Caller list.

## Market-wide questions

Breadth, dispersion, Gini, "how many went up", percentile ranks, sector-day returns: one
`alpha_engine_cross_section_returns(days=N, date=optional)` call. It returns whole-universe returns
with `gini` and summary stats. Never loop `alpha_engine_get_price_history` or
`alpha_engine_get_latest_quotes` per ticker. Disclose the `prices-availability` basis for historical
dates.

## SQL escape hatch

- `alpha_engine_sql` is read-only DuckDB over the pinned snapshot: one SELECT, `max_rows` ≤ 5000,
  `timeout_s` ≤ 120. Use it for any slice, ranking, rolling or window statistic, earnings dates,
  yield curve or FRED series that no fixed op returns. `alpha_engine_describe` lists the relations.
- It applies **no PIT masking**: join `sp500_membership_intervals` yourself for any historical universe.
- Cite the `snapshot_id` it returns with the as-of date.
- Aggregate in SQL and keep row counts small; tool output over about 12,000 characters is
  externalised to a file.

## Never read the data store directly

alpha_engine data files are not an input. Never open them from `reportforge_run_code`, `read_file`
or any other runner; a direct read bypasses snapshot pinning, telemetry and the PIT stamps. Use the
alpha_engine tools or `alpha_engine_sql`, then compute on the values they return.

## Error kinds

alpha_engine errors carry `_meta.alpha_engine.error_kind`: `pit_uncovered`, `stale_data`,
`invalid_param`, `insufficient_history`, `not_found`, `data_unavailable`, `contract_violation`,
`sandbox_refused`, `timeout`, `busy`, `internal`. Quote the kind and message; fix arguments for
`invalid_param`, disclose for `pit_uncovered`/`data_unavailable`, and never fill the gap by hand.

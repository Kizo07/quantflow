# Spark protocol: specs, outcomes, tombstones, sweeps

Spark is the research ledger. It remembers every tested idea (failures included), counts trials per
family so significance is deflated (PSR/DSR), and points at unexplored research space. It never
computes prices: `spark_backtest_run` dispatches to `alpha_engine_run_backtest` under the trial
governor. Spark runs without an LLM (deterministic scoring, FTS5-only retrieval).

## When it is mandatory

Any signal, screen, factor expression, ML combination or timing rule that you or the user
constructed and that you backtest and present as evidence. Single-name notes, prose research and
descriptive questions about well-known factors need only `spark_research_history`.

## Shared start (both paths)

1. `spark_research_history(query=...)` and, for beliefs, `spark_memory_recall`: has this been tried,
   and is it tombstoned?
2. `spark_check_novelty(idea_or_code=..., family=...)`.
3. `spark_hypothesis_propose(...)` → `hypothesis_id`. All of these are required:
   `idea`, `family`, `universe`, `horizon`, `scope`, `mechanism`, `expected_direction`, `falsifier`,
   `predicted_ic`, `predicted_sigma`, `hypothesis_type`.
   - `family`: one of `quant`, `econometrics`, `causal`, `signal_processing`, `network_optimization`,
     `nlp`, `macro`, `portfolio`. There is no catch-all.
   - `horizon`: one of `1d`, `5d`, `20d`, `60d`, `120d`, `long` (exact match).
   - `hypothesis_type`: one of `predictive`, `causal`, `descriptive`, `portfolio`, `risk`, `execution`.
     `causal` also requires `dag` and `adjustment_set`.
   - Pass a stable `client_id` to make retries idempotent.
4. `spark_register_hypothesis(hypothesis_id, spec, seed)` → `episode_id`. `spec` is the exact test you
   are about to run (the BacktestSpec for path A; the alpha_engine tool name and arguments for path B).
   This fixes the protocol hash before any result exists.

## Path A: DSR-gated (momentum or reversal signals)

5. `spark_backtest_run(episode_id, spec)`. `spec` is a BacktestSpec:

   ```
   {"hypothesis_id": "<id>", "signal_ref": "mom",
    "params": {"type": "momentum", "lookback": 20, "holding": 1, "turnover": 1.0},
    "universe_pit": "AAPL,MSFT,XOM,JPM,JNJ@2018-01-02",
    "dates": {"is_start": "2018-01-02", "is_end": "2020-12-31",
              "oos_start": "2021-01-04", "oos_end": "2022-12-30"},
    "costs": {"bps": 5, "slippage": 0, "borrow": 0},
    "splits": {"purge": 5, "embargo": 5}, "seed": 7}
   ```

   - `params.type` is `momentum` or `reversal`; anything else is refused.
   - Dates must satisfy `is_start <= is_end < oos_start <= oos_end`.
   - Costs must total more than zero.
   - Use `@YYYY-MM-DD` in `universe_pit`; see data-governance.md for the stamps to check.
   - The result carries `sr`, `T`, `skew`, `kurt` (net of costs), `oos_sr` and the universe stamps.
6. `spark_significance_gate(episode_id)` → DSR against the family trial count M. A high Sharpe ratio
   without the gate is luck, not evidence. Always report SR paired with DSR.
7. `spark_log_outcome(episode_id, outcome, metrics, costs)`: `validated` or `success` only when the gate
   passed; otherwise `inconclusive`, or `failure` when the falsifier fired.

## Path B: direct alpha_engine strategy tool (registered, not DSR-gated)

Tools: `alpha_engine_run_factor_analysis`, `alpha_engine_run_momentum_backtest`,
`alpha_engine_ml_signal_scores`, `alpha_engine_run_param_sweep`.

5. Run the tool with the arguments you registered.
6. `spark_log_outcome(episode_id, outcome, metrics=<metrics the tool returned>, costs)`. This
   records a reconciled trial. Use `inconclusive`, or `failure` when the falsifier fired; never
   `validated` or `success`, because the result was not gated.
7. Do **not** call `spark_significance_gate`: it refuses episodes without `sr`, `T`, `skew` and
   `kurt`, which these tools do not return. Do not compute those moments yourself in
   `reportforge_run_code`.
8. Label the result "registered, not DSR-gated" wherever it is quoted.

## Sweeps

Every cell of `alpha_engine_run_param_sweep` is a trial. Register the sweep as one episode, log it
with the number of cells in `metrics` (e.g. `"n_cells": 12`), and report the whole surface, never
only the best cell.

## Tombstones

A failure outcome writes a tombstone. `spark_backtest_run` runs a tombstone pre-check and blocks a
re-test of a tombstoned idea unless the call carries
`acknowledge_tombstone={"tombstone_id": "...", "rationale": "what is materially different"}`.
Only acknowledge with a real difference; otherwise report the earlier failure.

## Episode-scoped analytics

Inside a registered episode use `spark_causal_check` (Granger; PCMCI is refused) instead of
`alpha_engine_causal_test`, and `spark_risk_decompose` instead of `alpha_engine_risk_decompose`, so
the result lands on the episode. No causal claim is promoted without a DAG.

## When stuck, and for the appendix

- Next region to probe: `spark_suggest_next`, `spark_explore_methods`, `spark_find_analogies`,
  `spark_combine_methods`. Without an LLM they return deterministic or templated output: treat it
  as a hint, not evidence.
- Audit trail: `spark_trace_hypothesis(hypothesis_id)`, `spark_report_write` for a ledger appendix,
  `spark_archive_status` for family counts.
- Never call `spark_approve` or any other human-gate, skill-evolution or maintenance tool: approval
  belongs to a human, and those tools are outside the research set in `tool-catalog.md`.

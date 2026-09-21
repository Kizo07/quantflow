# Knowledge retrieval eval — Phase 0 fixture

Curated retrieval-eval fixture for the Quantflow Research Knowledge Plane
(Phase 0, scaffolding + eval). It pins the Phase 2 exit criterion —
*agents retrieve known relevant priors and known failures before research
begins* — as runnable numbers: **recall@k** and **failure-recall@k** at
`k=10`, target `>= 0.8`.

## Contents

| File | Purpose |
|---|---|
| `eval_fixture.json` | Corpus (36 docs) + 30 research-intent queries with relevance judgments |
| `runner.py` | Stdlib-only scorer + CLI + embedded self-test; no `deerflow` imports |
| `README.md` | This file |

The package is intentionally standalone (namespace packages import without
`__init__.py`): the runner loads its sibling fixture by path and runs
before any integration wiring exists.

## Fixture schema (`eval_fixture.json`)

```jsonc
{
  "version": "0.1.0",
  "description": "...",
  "documents": [
    {
      "doc_id": "exp_mom_126_top20_me",   // stable corpus ID; rankings return these
      "kind": "experiment|failure|finding",
      "title": "...",
      "tool": "run_momentum_backtest",    // originating alpha_engine MCP tool ("" for findings)
      "params": {"lookback_days": 126, "top_n": 20, "rebalance": "ME", "fees": 0.0005},
      "metrics": {"sharpe": 0.82, "turnover": 3.4},
      "scope": {"asset_class": "equity", "market": "US", "universe": "sp500-pit",
                "horizon": "6m", "frequency": "daily"},
      "failure_class": "execution",       // required when kind == "failure":
                                          // data | code | statistical | execution | hypothesis
      "text": "..."                       // retrieval surface: summary + key terms
    }
  ],
  "queries": [
    {
      "query_id": "q01_momentum_6m_backtest",
      "query_text": "What prior momentum backtests exist ...?",
      "intent": {                         // knowledge_base.md research-intent shape
        "topic": "cross-sectional momentum",
        "asset_class": "equity",
        "markets": ["US"],
        "universe": "sp500-pit",
        "horizon": "6m",
        "frequency": "daily",
        "concepts": ["momentum", "backtest", "sharpe"],
        "requested_period": ["2015-01-01", "2026-09-19"],
        "needed_memory": ["validated_findings", "prior_experiments", "failures"]
      },
      "relevant": [                       // graded relevance judgments (non-empty)
        {"doc_id": "exp_mom_126_top20_me", "grade": 2, "rationale": "..."}
      ],
      "must_recall_failures": ["fail_costs_erase_mom"]  // failure-recall cases
    }
  ]
}
```

Grades: `2` = highly relevant (answer material), `1` = relevant
(background/comparison). Recall is computed over all judged-relevant IDs;
grades are retained for future nDCG-style metrics.

Corpus coverage (36 docs: 20 experiments, 12 failures, 4 findings), all
grounded in real alpha_engine capabilities and their tested names:

- Momentum: `run_momentum_backtest` (`lookback_days`/`top_n`/`ME|W`/`fees`,
  sharpe/turnover/max-drawdown), `run_param_sweep` grids, `run_backtest`
  §15.2 (`type`/`lookback`/`holding`, `bps`/`slippage`/`borrow`,
  `purge`/`embargo`, sr/T/skew/kurt/gross_sr/cost_drag/ic_mean/oos_sr).
- Factors: `run_factor_analysis` (`mom_126d`/`rsi_14`/`vol_63d`/...,
  expression DSL, `horizon_days`/`q`/sector-neutralize, IC mean/ICIR/
  t-stat/hit-rate, quantile long-short, coverage), `ml_signal_scores`.
- Attribution: `attribute_portfolio` (betas/t-stats/alpha/R²),
  `brinson_attribution` (allocation/selection/interaction/active).
- Risk: `portfolio_risk_decomposition` (EWMA `halflife` 66 / sample /
  Newey-West, total/common/specific vol), `risk_decompose`
  (`ff5+momentum`, specific risk).
- Optimizer/causal: `optimize_portfolio` (max_sharpe/HRP/Black-Litterman),
  `causal_test` (Granger; pcmci refused).
- Failure modes: cost erasure, survivorship bias (snapshot vs PIT),
  naive AQR day-join, look-ahead labels (`t+1` rule), Stooq unadjusted
  closes, calendar-vs-trading windows, DuckDB single writer, sweep
  overfitting, Brinson date alignment, thin cross-sections, GPU float32.

## Runner usage

```bash
cd backend/packages/harness/deerflow/knowledge/eval

python runner.py                        # null baseline -> exit 2 (nothing wired)
python runner.py --retriever oracle     # perfect ranking -> exit 0
python runner.py --retriever keyword    # token-overlap smoke baseline
python runner.py --self-test            # embedded unit tests -> exit 0
python runner.py --json                 # machine-readable report
python runner.py --list-queries         # query IDs + text
python runner.py --k 5 --target 0.8     # custom cutoff / target
```

Exit codes: `0` = both means meet `--target` (default 0.8); `2` = below
target or retriever errors (the Phase 0 pre-recall state); `1` = usage or
fixture error (missing/invalid JSON, bad `--k`, unknown retriever).

Observed Phase 0 baselines (`k=10`, 30 queries): `null` → recall 0.0000,
failure-recall 0.0000, exit 2; `keyword` → recall 0.8889 but
failure-recall only 0.4667, exit 2; `oracle` → 1.0000 / 1.0000, exit 0.
The keyword result is the fixture working as designed: naive topical
overlap cannot surface the must-recall failures, which is why Phase 2
needs a dedicated failure channel rather than one similarity ranking.

Scoring rules:

- `recall@k = |top-k ∩ relevant| / |relevant|`, averaged over queries.
- `failure-recall@k = |top-k ∩ must-recall| / |must-recall|`, averaged over
  queries that name failures.
- Rankings are cleaned before scoring: unknown IDs ignored (counted),
  duplicates dropped, truncated to `k`. A retriever raising on a query
  scores that query as zero and records the error; the run continues.

## How Phase 2 plugs retrieval in

Phase 2 replaces the `null` baseline with the real hybrid stack
(structured + FTS + pgvector + failure channel, scope filters, RRF fusion,
fixed-budget context packet). The seam is the `RetrievalFn` protocol in
`runner.py`:

```python
from runner import EvalQuery, evaluate, load_fixture

def knowledge_retriever(query: EvalQuery, k: int) -> list[str]:
    """Phase 2 adapter: research intent -> ledger_search -> doc IDs."""
    packet = knowledge_bootstrap(query.intent)          # Phase 2 agent tool
    hits = ledger_search(                            # Phase 2 agent tool
        query_text=query.query_text,
        filters={"universe": query.intent["universe"],
                 "horizon": query.intent["horizon"]},
        kinds=["experiment", "failure", "finding"],
        top_k=k,
    )
    # Map store IDs back to fixture doc_ids for scoring (fixture IDs are
    # stable; seed the Phase 2 store with the same corpus or keep a map).
    return [to_fixture_id(h) for h in hits["ids"]]

fixture = load_fixture("eval_fixture.json")
report = evaluate(fixture, knowledge_retriever, k=10,
                  retriever_name="ledger_search", target=0.8)
print(f"recall={report.mean_recall:.3f} "
      f"failure-recall={report.mean_failure_recall:.3f} "
      f"passed={report.passed}")
```

Checklist for the Phase 2 owner:

1. Seed (or map) the 36 fixture documents into the Phase 2 store so every
   `doc_id` is retrievable; keep `failure_class` and scope fields as
   structured filters.
2. Implement the adapter above in the integration step (do not modify this
   directory's contract: `doc_id`s, query IDs, and metric definitions are
   frozen for comparability).
3. Run `python runner.py` equivalents via `evaluate(...)` with
   `retriever_name="ledger_search"`; the Phase 2 exit criterion is
   `passed == True` at `k=10`, `target=0.8` for **both** means.
4. Add a `make`-style target that runs this eval (Phase 0 exit: the target
   exists and reports the failing baseline; Phase 2 exit: it turns green).
5. Keep `--self-test` passing; extend the fixture by appending new queries
   (never rewrite judgments in place — history must stay comparable).

## Conventions followed

- Backend Python style: `ruff`-clean, 240-col limit, type hints, double
  quotes, `encoding="utf-8"` on file I/O; stdlib-only so the eval runs
  anywhere.
- Harness/app boundary: nothing here imports `app.*` or any `deerflow`
  module (verified: the runner has zero package imports).
- No existing files were modified; per the Phase 0 concurrency rule all
  three files are new under `knowledge/eval/`.

# QuantFlow

A quantitative-research super-agent harness: deep-exploration agents wired directly
into a local quant stack (market data, factors, backtests, portfolio construction)
and a multi-format report engine — ask for stock ideas, evidence-backed research,
and investment-committee style reports.

> **Fork note:** QuantFlow is forked from
> [**DeerFlow**](https://github.com/bytedance/deer-flow) by ByteDance (the 2.0
> super-agent harness: sub-agents, skills, memory, sandboxes). All harness credit
> goes upstream; this repo layers a quant-research desk on top. For the generic
> framework docs, start with
> [upstream README](https://github.com/bytedance/deer-flow/blob/main/README.md).

## What's different in this fork

- **alpha_engine MCP integration** (`extensions_config.json` → `alpha_engine` server,
  tools catalogued in `config/quant_tools.yaml`, validated by
  `scripts/validate_quant_tools.py`): S&P 500 price history, FF/AQR/JKP factor
  analysis, walk-forward ML signal scores, momentum backtests + GPU param sweeps,
  portfolio optimization (max-Sharpe / HRP / Black-Litterman), Brinson attribution,
  and factor risk decomposition — dealt out to desk agents per tool
  (`quant-analyst`, `technical-analyst`, `risk-manager`).
- **report-forge MCP integration** (host-side stdio server): one-source reports
  rendered via Quarto to HTML / PDF (Typst) / DOCX, with code execution
  (`reportforge_run_code`), static chart export, asset ingestion, and project
  inspection — see `docs/plans/2026-09-01-reportforge-native-flexibility.md`.
- **kizonlp MCP integration**: financial NLP (sentiment scoring, zero-shot
  classification, summarization, keyphrases, PDF extraction) feeding the
  news/document analysts.
- **quant-desk skill** (`skills/custom/quant-desk/SKILL.md`): a CIO-style
  evidence-first workflow — frame the brief, gather evidence in parallel,
  bull-vs-bear debate, risk-manager veto, then a sourced synthesis. Research
  only, never financial advice.
- **More model choices**: the `models:` list in `config.example.yaml` already
  covers OpenAI-compatible gateways, so Meta's Muse Spark family works via the
  Meta Model API (`base_url: https://api.meta.ai/v1`) alongside the existing
  Qwen / GLM / DeepSeek / Kimi entries.

## Quick start

```bash
git clone https://github.com/Kizo07/quantflow.git
cd quantflow
make setup        # interactive wizard (recommended first run)
make doctor       # verify configuration and system requirements
make install      # frontend + backend + hooks
make dev          # all services with hot-reloading
```

Production instead: `make up` (Docker, unified endpoint `http://localhost:2026`).
Background runs: `make dev-daemon` / `make start-daemon`; `make stop` to end them.

Configuration lives in `config.yaml` (generated from `config.example.yaml` via
`make config`) plus `.env` for keys — both are gitignored and never committed.
After editing the example, merge with `make config-upgrade`. `make check`
verifies prerequisites.

## The quant desk

Ask things like *"find 10 momentum names with fresh catalysts"* or
*"build a defensive portfolio from these picks"*. Behind the prompt:

1. **Frame** — the question becomes a brief (universe, horizon, constraints).
2. **Evidence** — `quant-analyst` screens/ranks (factors, ML scores, backtests),
   `news-analyst` adds dated catalysts, technicians cover setups.
3. **Debate** — `bull-researcher` vs `bear-researcher` argue the shortlist;
   weak names are demoted with reasons on record.
4. **Risk gate** — `risk-manager` enforces caps, concentration and vol budgets.
5. **Synthesis** — one ranked report with evidence, demotions, risk dashboard,
   and caveats, rendered through report-forge when a formatted deliverable
   is needed.

Data freshness first: the desk checks `engine_status` and refreshes the local
store (`refresh_data`, quant-analyst only) before trusting a screen.

## Repo map

- `config.example.yaml` / `config.yaml` — models, token budgets, tool policy.
- `config/quant_tools.yaml` — alpha_engine tool catalog per desk agent.
- `extensions_config.json` — MCP servers (alpha_engine, kizonlp, report-forge).
- `skills/custom/quant-desk/` — the desk playbook; `skills/public/` — upstream skills.
- `scripts/validate_quant_tools.py` — keeps the tool catalog in sync with the engine.
- `docs/plans/` — design notes (report-forge flexibility, remediation logs).
- `backend/` + `frontend/` — the harness itself (upstream DeerFlow 2.0).

## Upstream & license

Harness core: [bytedance/deer-flow](https://github.com/bytedance/deer-flow)
(MIT — see [LICENSE](./LICENSE)). Quant layers (desk skill, tool catalog,
MCP wiring, plans) are this fork's additions. Upstream docs in other languages
(`README_zh/ja/fr/ru.md`) describe the unmodified framework.

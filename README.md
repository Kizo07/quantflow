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

- **alpha_engine MCP integration** (local stdio server, wired via
  `extensions_config.json` — see "Quant MCP servers" below; tools catalogued in
  `config/quant_tools.yaml`, validated by `scripts/validate_quant_tools.py`):
  S&P 500 price history, FF/AQR/JKP factor analysis, walk-forward ML signal
  scores, momentum backtests + GPU param sweeps, portfolio optimization
  (max-Sharpe / HRP / Black-Litterman), Brinson attribution, and factor risk
  decomposition — dealt out to desk agents per tool (`quant-analyst`,
  `technical-analyst`, `risk-manager`).
- **report-forge MCP integration** (host-side stdio server): one-source reports
  rendered via Quarto to HTML / PDF (Typst) / pdf-web / DOCX, with code execution
  (`reportforge_run_code`), static chart export, asset ingestion, and project
  inspection — see `docs/plans/2026-09-01-reportforge-native-flexibility.md`.
- **kizonlp MCP integration**: financial NLP (sentiment scoring, zero-shot
  classification, summarization, keyphrases, PDF extraction) feeding the
  news/document analysts.
- **quant-desk skill** (`skills/custom/quant-desk/SKILL.md`): a CIO-style
  evidence-first workflow — frame the brief, gather evidence in parallel,
  bull-vs-bear debate, risk-manager veto, then a sourced synthesis. Research
  only, never financial advice.
- **More model choices**: `config.example.yaml` shows the OpenAI-compatible
  gateway pattern, so you can add Meta's Muse Spark family via the Meta Model
  API alongside the Qwen / GLM / DeepSeek / Kimi examples — see "Muse models"
  below.

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

## Quant MCP servers

`extensions_config.json` is local-only (gitignored). Copy the template, then
enable the quant servers with your own paths:

```bash
cp extensions_config.example.json extensions_config.json
```

The template already contains disabled `alpha_engine`, `kizonlp`, and
`reportforge` entries: replace the `/path/to/...` placeholders with your
checkouts (`alpha_engine` sources, `report-forge/.venv`), set
`"enabled": true` for the ones you run, and restart the gateway. Docker
(`make up`, `make docker-start`) auto-creates `extensions_config.json` from
the template when it is missing.

## Muse models

Nothing Muse-specific ships enabled: add one entry under `models:` in your
`config.yaml`, following the OpenAI-compatible examples in
`config.example.yaml`, and put the key in `.env` (never in the YAML):

```yaml
models:
  - name: muse-spark-1.3-contributor
    display_name: Muse Spark 1.3
    use: deerflow.models.patched_openai:PatchedChatOpenAI
    model: muse-spark-1.3-contributor
    api_key: $MODEL_API_KEY
    base_url: https://api.meta.ai/v1
    context_window: 800000   # total prompt + completion capacity
    supports_reasoning_effort: true
```

(`max_tokens` is the per-call *output* cap — do not set it to the context
size.) Model metadata usually hot-reloads; restart the gateway if the new
model doesn't show up.

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
- `extensions_config.example.json` / `extensions_config.json` (gitignored) —
  MCP servers incl. disabled `alpha_engine`, `kizonlp`, `reportforge` entries.
- `skills/custom/quant-desk/` — the desk playbook; `skills/public/` — upstream skills.
- `scripts/validate_quant_tools.py` — keeps the tool catalog in sync with the engine.
- `docs/plans/` — design notes (report-forge flexibility, remediation logs).
- `backend/` + `frontend/` — the harness itself (upstream DeerFlow 2.0).

## Security notice

This fork adds capabilities upstream's hardening notes don't cover:

- **Host code execution is on by default locally.** `reportforge_run_code` /
  `reportforge_run_file` run with your user permissions by design (see the
  flexibility plan). Treat prompts and report inputs as untrusted, and set
  `REPORTFORGE_EXEC=off` for any shared, CI, or public deployment.
- **Do not publicly host this stack as-is.** It is built for a single local
  operator: MCP servers, sandbox mounts, and the gateway assume localhost
  trust. Review upstream's security recommendations before exposing anything.
- **Keys stay in `.env`** (gitignored, never committed). Session imports
  (`POST /api/threads/import`) sanitize uploads to plain human/AI transcript —
  tool calls, ids, and reasoning payloads are dropped — but never import
  files from sources you don't trust.

## Upstream & license

Harness core: [bytedance/deer-flow](https://github.com/bytedance/deer-flow)
(MIT — see [LICENSE](./LICENSE)). Quant layers (desk skill, tool catalog,
MCP wiring, plans) are this fork's additions. Upstream docs in other languages
(`README_zh/ja/fr/ru.md`) describe the unmodified framework.

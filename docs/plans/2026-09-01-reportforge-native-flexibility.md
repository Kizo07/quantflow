# Report-Forge Flexibility Plan — native-generation capabilities inside report-forge

Date: 2026-09-01
Status: IMPLEMENTED 2026-09-01 (report-forge commit e5e70e3; live E2E verified)
Scope: `~/Documents/report-forge` (engine + MCP + tests) plus deer-flow skill/prompt updates.

## Context

As of 2026-09-01 report-forge (WS-A..E remediation complete, live-verified in smoke run
`e6b472fe`) gives us deterministic, multi-format, professionally typeset output — at the cost
of a template ceiling and an agent that cannot execute code, inspect results, or ingest
arbitrary assets. Native quantflow generation has none of those limits (agent writes files,
runs code, iterates visually) but pays in tokens, PDF fragility, and zero DOCX.

**Goal:** absorb native-generation flexibility into report-forge while keeping its
one-source → html/pdf/docx determinism. Operating assumption per user decision: report-forge
runs on the **host** as a stdio MCP; the sandbox boundary is irrelevant here and the server
may execute code with the host user's permissions. Low risk, explicitly desired.

### What already exists (do not rebuild)

- `write_report_body` accepts a **complete .qmd including YAML frontmatter**, scoped to
  `REPORTS_DIR` — raw HTML blocks, custom divs, arbitrary code chunks are already possible.
  Flexibility gaps are ergonomic (full-body rewrites), not fundamental.
- `save_chart` exports plotly JSON → static PNG + interactive HTML with sandbox-path
  translation into the project (`_translate_sandbox_path`).
- Jupyter kernel `reportforge` is auto-created from a configurable Python:
  `_venv_python()` → `REPORTFORGE_PYTHON` env > running venv > `<repo>/.venv` (engine.py:233).
  This is the hook for pointing execution at the alpha_engine conda env.
- `publish_report` bridges rendered artifacts into thread outputs (WS-C, verified).
- Quarto config already sets `freeze: auto`, `output-dir: output`, 300 dpi figures.

### Gaps this plan closes

| # | Gap | Native equivalent |
|---|-----|-------------------|
| G1 | Agent cannot execute code outside embedded chunks (data prep, models, asset generation) | sandbox bash/python |
| G2 | Only plotly charts can be ingested (`save_chart` is plotly-JSON-only) | write any file in sandbox |
| G3 | Agent cannot read project files or render logs between renders — blind iteration | read_file on own outputs |
| G4 | Every body edit rewrites the whole .qmd; no incremental composition | incremental write_file edits |
| G5 | PDF requires Typst-compatible content; no HTML-first → PDF path (plotly/JS-heavy layouts) | Chromium print-to-PDF recovery |
| G6 | Execution env is a bare venv — no pandas/pyarrow/statsmodels for real quant work | alpha_engine env |

---

## WS-1 — Code execution surface (`reportforge_run_code`)

The single biggest flexibility unlock: the agent runs Python on the host, sees real
stdout/stderr, and can generate data/files that chunks and figures then consume.

**New tool:** `reportforge_run_code(code: str, project: str | None, timeout: int = 300)`
- Executes `code` with the reportforge kernel's Python (`_venv_python()` resolution — same
  interpreter Quarto chunks use, so run_code and chunks share one environment).
- **Working directory = the project root** (cwd scoping; no cwd for non-project calls —
  require `project` unless `REPORTFORGE_PROJECT_OPTIONAL=1`).
- Captures stdout + stderr (tail ~8KB), exit status, wall time, and lists files created or
  modified under the project during execution (diff of mtimes before/after) so the agent
  sees its own artifacts without a separate inspection call.
- Returns `{ok, stdout_tail, stderr_tail, created: [...], modified: [...], duration_s}`.

**New tool:** `reportforge_run_file(path: str, project: str, args: list[str] | None, timeout)`
- Runs a script that already lives inside the project (`.py`/`.sh`/`.R` via extension map),
  same capture semantics. Lets the agent write a long script with `write_report_body`-style
  file tools and execute it separately from its context window.

**Security model (explicit, per user decision):**
- These run with host user permissions — documented in the tool docstrings.
- Scoping is ergonomic, not a security boundary: cwd pinned to project, but code *can* reach
  the full host filesystem. That is accepted. An env switch
  `REPORTFORGE_EXEC=off` disables both tools for shared/CI deployments.
- No shell-string execution: `run_code` is Python only; `run_file` runs files that exist
  inside the project. deer-flow already gives the agent bash elsewhere — no need to
  duplicate it here.

**Tests:** run_code captures stdout, sees created files, times out, respects REPORTFORGE_EXEC=off;
run_file rejects paths escaping the project.

## WS-2 — Asset ingestion (`reportforge_save_asset`)

Generalize `save_chart` beyond plotly:

**New tool:** `reportforge_save_asset(project, dest_relpath, content_b64 | content_text, mode="text")`
- Writes arbitrary text or base64 binary into the project (figures/, data/, assets/).
- Path translation reuses `_translate_sandbox_path`; destination confined to project root.
- Returns the relative path ready for `.qmd` embedding (`![](figures/x.png)`) plus bytes written.

Keep `save_chart` as the plotly fast path (it also emits the interactive HTML twin).
Update `save_chart` docstring: for matplotlib/altair/static PNGs use `save_asset`.

**Tests:** text + binary round-trip, path-confined, sandbox-path translation preserved.

## WS-3 — Inspection & iteration

Blind iteration is why native runs hand-polish and report-forge runs guess. Add:

**New tool:** `reportforge_project_status(project)`
- File tree (relpaths + sizes), `_quarto.yml` formats, presence of output/, last render
  timestamp (persisted by render_report into `<project>/.reportforge-state.json` — new).

**New tool:** `reportforge_read_project_file(project, relpath, max_bytes=32768)`
- Read any text file in the project (qmd, render log, generated CSV head, typst errors).
  Binary → returns size + mime guess only.

**Enhancement:** `render_report` persists the full quarto log to
`<project>/output/.render-log-<format>.txt` and returns its path; `log_tail` grows from
~2KB to 8KB. Typst/PDF failures are the #1 iteration blocker — the agent must see them.

**Tests:** state file written on render; read confined to project; log persisted.

## WS-4 — Incremental composition

**New tool:** `reportforge_append_section(project, markdown, before: str | None)`
- Appends a markdown section to `index.qmd` body (after frontmatter), or inserts before a
  heading matched by `before`. Match = first heading whose text contains `before`
  (case-insensitive substring); no match is a clean error, never a silent append.
  No full-body rewrite for additive edits.
- Implementation: split frontmatter (`---` fence) from body; append/insert; rewrite.

**New template: `bespoke`** (escape hatch)
- `scaffold_report(template="bespoke", frontmatter_yaml: str, body: str)` — scaffold writes
  the caller's frontmatter verbatim (validated as parseable YAML with a `format:` key) and
  the caller's body. Keeps project machinery (`output/`, `freeze`, kernel, publish bridge)
  while removing all template opinions. This is "native freedom, report-forge plumbing".

**Tests:** append preserves frontmatter; insert-before anchors correctly; bespoke rejects
invalid YAML; bespoke renders at least html.

## WS-5 — HTML-first PDF path (`pdf-web` format)

The validated AAPL recovery (self-contained HTML + static charts → Chromium headless) becomes
first-class, so JS-heavy or plotly-rich layouts can ship as PDF:

- `PUBLIC_FORMATS += ("pdf-web",)`.
- `render_report` for `pdf-web`: render `html` (self-contained via quarto
  `format.html.embed-resources: true`), then
  `chromium --headless --no-sandbox --print-to-pdf=output/<slug>.pdf file://.../index.html`
  using the same headless Chromium deer-flow uses (resolve via `REPORTFORGE_CHROMIUM` env >
  `chromium` on PATH).
- Interactive plotly stays alive in the HTML artifact; PDF is the print snapshot. Documented
  policy: `pdf` = Typst (best typography, static figures only); `pdf-web` = print of the
  HTML (keeps JS-rendered visuals); `html` = full interactivity.
- `publish_report` already picks up everything under `output/` — no change needed there.

**Tests:** unit-mocked chromium invocation; live smoke renders the smoke project to pdf-web
and asserts `pdfinfo` page count ≥ 1.

## WS-6 — Quant-capable execution environment

> **SUPERSEDED 2026-09-01 (see "Implementation status" below):** do NOT point
> `REPORTFORGE_PYTHON` at the alpha_engine conda env — it has no ipykernel and
> is unsuitable as a Quarto kernel host. The original proposal is kept for the
> record; the shipped behavior is the report-forge `.venv` default (pandas,
> pyarrow, statsmodels pre-installed there), with `REPORTFORGE_PYTHON` retained
> only as an override for another kernel-ready env.

Original proposal (not implemented as written):

- Document + support `REPORTFORGE_PYTHON=/opt/anaconda/envs/alpha_engine/bin/python`.
  The alpha_engine env already has pandas/pyarrow/statsmodels — chunks and `run_code` then
  do genuine factor/backtest work, not toy math.
- deer-flow side: add `REPORTFORGE_PYTHON` to the reportforge stdio MCP entry in
  `extensions_config.json` (gitignored — config only, no code change; documented in plan).
- If the alpha_engine env lacks ipykernel/jupyter deps, one-time install there (pre-flight
  check script `scripts/check_kernel_env.sh` in report-forge, run during implementation).
- Fallback behavior unchanged (reportforge venv → python3) so nothing breaks without the env var.

## WS-7 — Skills, prompts, docs

- **quant-desk SKILL.md**: capability table (when to use run_code/save_asset/pdf-web vs
  plain scaffold), bespoke-template guidance, publish-before-present rule already there.
- **autonomous-report-runs skill**: new template `bespoke-report-run-prompt.txt` for
  design-heavy runs; pitfall entry: "run_code is host execution — treat stdout as ground
  truth, never fabricate results you didn't capture" (extends the WS-A integrity rules).
- **report-forge README**: new tools, permission model (host execution, accepted risk),
  env vars (`REPORTFORGE_PYTHON`, `REPORTFORGE_EXEC`, `REPORTFORGE_CHROMIUM`).
- forensic_lint.sh: extend census to include the new tool names.

## WS-8 — Verification

1. Unit tests per WS above (target: suite 44 → ~65 tests).
2. Live MCP boundary smoke: JSON-string formats already covered; add run_code + save_asset
   through the fastmcp boundary.
3. **Capstone smoke run** (deer-flow thread, minimal effort): bespoke template → run_code
   computes a table from a CSV it writes via save_asset → save_chart plotly figure →
   append_section twice → render html+pdf-web → publish → present_files. Assert: run
   success, both PDF and HTML in thread outputs, forensic lint clean.
4. Restart not needed for report-forge changes (MCP spawns per run); deer-flow
   `extensions_config.json` env change needs `systemctl --user restart deer-flow-dev`.

---

## Sequencing & effort

| Order | WS | Depends on | Est. effort |
|-------|----|------------|-------------|
| 1 | WS-6 (env) | — | S (config + pre-flight) |
| 2 | WS-1 (run_code/run_file) | WS-6 for real value | M |
| 3 | WS-2 (save_asset) | — | S |
| 4 | WS-3 (inspection) | — | S |
| 5 | WS-4 (append + bespoke) | — | M |
| 6 | WS-5 (pdf-web) | — | M (chromium wiring) |
| 7 | WS-7 (skills/docs) | 1–6 | S |
| 8 | WS-8 (verification) | all | M |

WS-1..WS-5 are independent of each other and could be parallelized; WS-6 first because it
determines which interpreter WS-1 executes.

## Open decisions (defaults assumed if no answer)

1. **Default execution env**: ~~assume alpha_engine conda env via `REPORTFORGE_PYTHON`
   (WS-6)~~ — DECIDED 2026-09-01: report-forge `.venv` default (alpha_engine env
   has no ipykernel; see WS-6 supersede note + Implementation status).
   `REPORTFORGE_PYTHON` stays as a config-side override only.
2. **Shell tool**: none — Python-only execution inside report-forge; agent already has bash
   in deer-flow. → *Default: no shell tool.*
3. **pdf-web default**: opt-in per render (formats list), never replaces `pdf`. → *Default: opt-in.*
4. **Kill-switch**: `REPORTFORGE_EXEC=off` disables run_code/run_file for non-local deploys.
   → *Default: on locally.*

## Risks & mitigations

- **Host execution blast radius**: accepted per user decision; mitigated by docstrings,
  kill-switch env, and cwd pinning. No secrets are read by report-forge beyond what
  deer-flow already injects.
- **Chromium drift**: pdf-web depends on headless chromium; pinned via REPORTFORGE_CHROMIUM;
  render returns the chromium stderr tail on failure (same log discipline as WS-3).
- **Kernel env mismatch**: if alpha_engine env lacks ipykernel, Quarto falls back to
  python3 and chunks break — pre-flight script in WS-6 catches this before first render.
- **Context bloat**: run_code stdout tail capped (8KB) + file-diff summary keeps tool
  results small; full logs stay on disk, readable via WS-3 tools.

## Success criteria

- A report run can: compute data on the host, ingest arbitrary assets, iterate against real
  render logs, compose sections incrementally, and ship html + Typst-PDF + pdf-web + docx —
  all through report-forge tools, ending with publish_report + present_files.
- Zero regression: existing 44 tests + smoke flow (scaffold→render→publish) unchanged.

## Implementation status (2026-09-01)

- report-forge commit `e5e70e3` implements WS-1..WS-6: `reportforge_run_code`,
  `reportforge_run_file`, `reportforge_save_asset`, `reportforge_project_status`,
  `reportforge_read_project_file`, `reportforge_append_section`, `bespoke` template,
  `pdf-web` format, render-log persistence (output/.render-log-<fmt>.txt +
  .reportforge-state.json), and `scripts/preflight_env.sh`.
- WS-6 design decision recorded: the execution interpreter defaults to the report-forge
  `.venv` (pandas 3.0.5, pyarrow 25.0.1, numpy 2.5.2, statsmodels 0.15.0, plotly,
  matplotlib, scipy installed there; the alpha_engine conda env has no ipykernel and is
  unsuitable as a Quarto kernel host). `REPORTFORGE_PYTHON` remains the override for
  pointing chunks/run_code at another env.
- Tests: 76 passed (was 44). Live E2E on the host: bespoke scaffold → run_code
  (matplotlib figure) → append_section → render html + typst pdf + pdf-web all produced
  real artifacts (`index.html`, `index.pdf`, `index-web.pdf`; chromium at /usr/bin/chromium).
- One real bug found by the E2E and fixed: BESPOKE_YML's pdf block needed explicit
  linkcolor/urlcolor/citecolor (Typst rejected Quarto's default non-hex linkcolor).
- WS-7: quant-desk skill gained the execution/flexibility capability table; the
  autonomous-report-runs skill documents the new tools and the compute-first prompt rule.
- Open follow-ups: none blocking. A capstone run exercising the full surface through the
  live MCP boundary is tracked separately (WS-8).

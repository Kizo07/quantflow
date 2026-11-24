# Reporting: report-forge pipeline, house style, notebooks, delivery

Read this before producing any PDF, DOCX or HTML deliverable.

## Pipeline

1. `reportforge_capabilities` (templates, formats, exec availability).
2. Scaffold: `reportforge_scaffold_from_brief` (one `report_brief` v1 dict), or
   `reportforge_compose_report` with `dry_run=true` first for the flagship path.
3. Write: `reportforge_write_report_body` for the first body, then `reportforge_read_report` +
   `reportforge_edit_report` for revisions; `reportforge_append_section` for additive sections.
4. Evidence: `reportforge_save_table`, `reportforge_register_source`, `reportforge_register_facts`,
   `reportforge_register_exhibit` (exhibit ids start with `fig-`).
5. `reportforge_check_readiness` until `ready_for_review`.
6. `reportforge_render_report` with the report **slug** (a .qmd path or project dir also works).
7. `reportforge_render_preview(inline=true)` and look at the pages.
8. Delivery (below): `reportforge_publish_report` → `present_files`.

Failure loop: `reportforge_project_status` → `reportforge_read_project_file` (render log) → fix →
re-render. Approval, release and lifecycle tools are human or operator workflow; don't call them.

## Branding (mandatory)

The public name is QuantFlow, never DeerFlow (the upstream project name), in any reader-facing
field: titles, headers, footers, body prose. `DEERFLOW_*` environment identifiers are plumbing;
leave them alone.

## Scaffold defaults (mandatory for flagship PDFs)

Every flagship scaffold passes these explicitly. Never rely on report-forge's unbranded `standard`
default, and never use `modern` or `thematic-deepdive` for flagships (both shipped unbranded or
off-theme PDFs).

- `template="portfolio-light"` (flagship default; `ledger-light` is the only approved alternate).
- `author="QuantFlow Desk"` (append a role such as "— CIO Office" when useful),
  `organization="QuantFlow Research"` and `firm="QuantFlow Research"` on every scaffold.
- `engine_charts_only=true`: render then refuses fallback or hand-rolled charts instead of shipping them.
- Chart theme matches the page theme: `quantflow-light` charts on light templates (`ledger-light`
  on ledger pages), `quantflow-dark` only on dark templates. Dark charts on light pages block shipping.

## Length gate (long-form briefs)

When the brief demands length ("25+ pages", "N words"), do not call `reportforge_render_report`
until the assembled body meets the minimum. Count mechanically with `reportforge_run_code`
(`len(Path("index.qmd").read_text().split())`). If short, expand thin sections with
`reportforge_append_section` and recount. Never declare a target met without a count.

## Compute surface

- `reportforge_run_code(code, project)`: host Python, cwd = project root. Use it only for
  (a) maths on values a tool returned that no alpha_engine op covers, (b) assembling figures and
  assets, (c) copying alpha_engine chart PNGs into the project. Never read alpha_engine data files
  in it, and never recompute returns, vol, Sharpe, drawdown, IC, factor regressions or optimisation
  that an alpha_engine op returns. Compute before you write: print the numbers, then embed only
  numbers you saw in `stdout_tail`.
- The reportforge_run_code environment has pandas, numpy, scipy, statsmodels, plotly and matplotlib; check
  `reportforge_capabilities` before relying on anything else. Never install packages.
- `reportforge_save_asset(project, dest_relpath, content_text|content_b64)`: CSVs, CSS, HTML
  partials, non-plotly images.
- Do not use `reportforge_run_file`.
- Templates: `bespoke` gives you the whole layout (custom or html-first reports). Formats: `pdf`
  (Typst) for typography-grade documents, `pdf-web` (headless-Chromium print of self-contained
  HTML; needs `html`, lands as `<stem>-web.pdf`) for JS-rendered visuals, `html` for interactivity.
- Host execution runs with user permissions by operator decision. Kill-switch: `REPORTFORGE_EXEC=off`.

## Charts: static PNG only

Inline `{python}` Plotly chunks render in HTML but break PDF ("Unable to display output") and leave
DOCX `word/media/` empty. Never ship them.

- Static PNG only: `reportforge_save_chart` (plotly figure JSON → PNG in the QuantFlow theme; write
  to `charts/<name>.png` and reference it relatively; `preview=true` returns the PNG in the result)
  or matplotlib in `reportforge_run_code` with `fig.savefig(...)`. If `reportforge_save_chart`
  fails, quote the real error and switch to matplotlib; disclose the switch.
- alpha_engine charts (`alpha_engine_chart_price`, `alpha_engine_chart_bars`,
  `alpha_engine_chart_fan`, `alpha_engine_chart_fundamentals`) return a host path you cannot view or
  present. Copy the PNG into the project's `charts/` with `reportforge_run_code` before referencing it.
- Prefer the engine builders over hand-rolled plotly. In `reportforge_run_code` (`alpha_engine` is
  importable), feed them data that a tool returned:
  `from alpha_engine.viz import price_technicals, waterfall_bridge, scenario_fan, attribution_bars,
  comps_bars, event_timeline, returns_dist, price_levels, brinson_effects, risk_decomposition,
  weights_donut, sector_allocation, rolling_risk, financials_trend, dataframe_table, scatter_xy,
  corr_heatmap, seasonality_grid, distribution_box, treemap_weights, mix_area, radar_profile,
  coef_intervals, metrics_markdown, save_figure` — e.g.
  `fig = price_technicals(px, volume=vol, theme="quantflow-light")`,
  `save_figure(fig, "charts/<name>.png", caption_text=caption(asof="<date>", source="<tool>"))`.
  `theme=` matches the page (`quantflow-light`/`quantflow-dark`, `ledger-light`/`ledger-dark`).
- Match encoding to data; bars are not the default. Comparison → `comps_bars`/`attribution_bars`;
  correlation → `corr_heatmap`; distribution by group → `distribution_box` (never a bar of means);
  static mix → `weights_donut` (≤ 10 slices) or `treemap_weights` (`groups={leaf: parent}`); mix over
  time → `mix_area`; multi-factor profile → `radar_profile` (≤ 4 overlays, pre-normalised);
  regression or event study → `coef_intervals` (zero line included); bivariate → `scatter_xy`
  (`size=` bubbles); seasonal grid → `seasonality_grid`; OHLC → `price_technicals(..., ohlc=df)`
  (single panel by design: candle mode drops volume and RSI). The figure lint enforces ≤ 50 %
  bar-only and ≥ 2 non-default encodings. Dense charts (heatmaps, treemaps, multi-trace) ride
  column-page heroes; at column width their type falls under 4 pt.
- Hand-rolled plotly goes through `apply_quantflow(fig)` / `quantflow_template()`. On export error:
  quote it verbatim, stringify scalar Timestamps, retry once, escalate. No silent substitution.
- Engine metrics dicts → `metrics_markdown(metrics)` tables wrapped in
  `::: {.rf-showtable .rf-nums-right} ... :::`, never hand-authored.

## Figures, captions, tables

- Every exhibit gets 2–5 lines of read-through (what it shows, what to conclude). A lone chart is
  never a section's only content.
- One caption system: every figure gets `{#fig-exN}` plus caption text, rendered as native
  `Exhibit N:` captions. Never hand-write `*Exhibit N — ...*` paragraphs. Refer to figures as `@fig-exN`.
- PDF body is two columns from page 2 on flagship templates (the cover stays single-column); widths
  are % of the column. Heroes (technicals, fan, timeline, major exhibits): `width=100%` with
  `.column-page` on the figure itself (`![cap](charts/x.png){#fig-exN width=100% .column-page}`),
  never a `::: {column-page}` fenced div. Standard charts: `width=85%`. Export sizes: hero 1500×750,
  standard 1000×650, compact 800×520 (scale 2). Never stack more than two 100 % figures without prose.
- Markdown tables wider than about 58 characters split, or move to the appendix as
  `dataframe_table` PNGs on `.column-page` heroes.
- Before render, run the report-forge figure lint (`scripts/figure_lint.py <project-dir>` in the
  report-forge checkout) through `reportforge_run_code` and fix every violation.

## Prose voice (mandatory)

Write as a human desk analyst. Short declarative sentences, one claim each (about 20 words; one
longer sentence per paragraph allowed). Plain exhibit names from the data, never invented labels.
Banned: shouty headers, "delve/tapestry/landscape", "announces itself/adjudicates", "forensic
attention", "honestly labeled", triple-parallel flourishes, non-English slips. Number ladders go in
tables; prose states the read. Every number needs a unit and an as-of. Paragraphs close with the
implication (vary: "So what" / "The read" / "Bottom line"). Alt text uses `&`.

## Notebooks and the diagram loop (two-pass rule)

Pipeline steps run in order; utilities are called as needed.

- **Pass 1, execute:** `reportforge_notebook_create` → cell edits (`reportforge_notebook_create_cell`,
  `reportforge_notebook_write_cell`) → `reportforge_notebook_run_cell` / `reportforge_notebook_run_all`
  (fresh kernel per run) or `reportforge_run_cell_persistent` (one cell in the project's long-lived
  kernel). Re-run after any code edit above a fed cell; stale flags say when. Kill-switch
  `REPORTFORGE_EXEC=off` stops all kernel paths.
- **Pass 2, render:** `reportforge_render_report` on the slug, then publish and present. Render never
  executes: a hollow render means pass 1 was skipped. Re-run the pass; never patch the outputs.
  Unexecuted notebooks render rc 0 with empty outputs (pinned policy, not a bug).
- **Diagram figures (utility):** livefigure sources (`figures/*.dot`, `*.vl.json`, `*.nomnoml`, …)
  go through validate → render → look → fix with the report-forge `scripts/figure_loop.py
  <project-dir> --render` (exit 1 lists violations). Look at the preview before shipping: layout bugs
  render exit 0. Use offline formats only for sealed releases (kroki-backed formats call out).
- **Hygiene is warn-only:** `panache_warnings` on render results are advisory. Fix what is cheap;
  never treat them as render failures.

### Notebook approval protocol (mandatory)

Notebook cell edits and cell runs issued by an agent are human-gated; reads are never gated.

- **Interactive run:** the harness returns a `RISK_CONFIRMATION` directive instead of running the
  tool. Call `ask_clarification(question=..., clarification_type="risk_confirmation", ...)` and wait.
  Only after the human approves, retry the same call; the harness forwards the confirmed retry. Make
  no other tool calls in the `ask_clarification` turn (siblings are dropped).
- **Autonomous run** (scheduled, webhook, no human): the harness returns `BLOCKED`. Never retry a
  `BLOCKED` notebook call. Report the `missing_decision` and continue with read-only work, or stop.
- The human's own clicks in the embedded UI are always allowed.

Never pass `confirmed`, `interaction_mode` or `mode` to a notebook tool. The harness strips any
model-supplied values and sets them per run policy (a confirmed retry carries `confirmed=True` plus
the resolved `interaction_mode`), so a model can never self-approve. Edits you make while the human
watches the embedded UI still count as agent-initiated: observation is not authorship.

## Delivery (mandatory)

report-forge renders on the host; your sandbox cannot read those paths, so presenting a host path
never works and a manifest is not a deliverable. After `reportforge_render_report` succeeds:

1. `reportforge_publish_report(project=<slug>)` copies the artifacts to your thread outputs and
   returns sandbox `present_paths`. It fails closed on `SEAL_*` codes: re-render all formats.
2. `present_files` with exactly those `present_paths`, as the last action.

If publishing fails, report the exact error. Before declaring done, verify the artifacts: DOCX
`word/media/` has the images, the PDF shows no "Unable to display" lines, the HTML `<img>` count
matches the chart count. A successful run is not proof; only the artifacts are.

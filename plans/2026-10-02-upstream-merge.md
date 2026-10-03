# QuantFlow upstream-merge plan (deer-flow → QuantFlow)

- Date: 2026-10-02. Evidence: subagent research (divergence, upstream-features, local-conflicts, critic) against `/home/fire/Documents/deer-flow` and a fresh `/tmp/deer-flow-upstream` clone; headline numbers re-verified by the parent session.
- Yes, QuantFlow is behind: 315 upstream commits since the 2026-09-21 fork point.

## Goal

Merge `upstream/main` (bytedance/deer-flow) into QuantFlow (`main`) to reach parity, preserving the QuantFlow quant-research layer (desk skill, tool map, knowledge package, branding, local config).

## Success Criteria

- `main` contains all 315 upstream commits via one merge commit; no rebase of shared history.
- All 71 overlapping files resolved per the K/T/M checklist below; no conflict markers remain.
- Backend gates green: lint, format check, test, test-knowledge, test-blocking-io, detect-blocking-io, root check.
- Quant gates green: `validate_quant_tools.py`, quant_tools.yaml (27 tools), quant-desk SKILL.md shows only intended changes.
- Frontend builds; touched e2e specs pass or are triaged as selector churn.
- `config_version` reads 50 with staged migrations applied in order.
- CHANGELOG `[Unreleased]` records the merge; README stays QuantFlow-branded with fork note intact.

## Approach

Merge on a branch (`merge/upstream-20261002`), never rebase: the 73-commit local arc stays intact and reviewable. Resolve the 71 overlap files in the risk order listed under Predicted conflicts with a per-file strategy — **K** keep local, **T** take upstream, **M** manual hunk merge — defaulting to local behavior where intent differs and porting upstream additions. Run the 46→50 config migration before resolving code, quarantine the dirty tree with stash+tag first, and land upstream-only new files as-is.

## Behind-status (verified)

- Remotes: `origin=https://github.com/Kizo07/quantflow.git`, `upstream=https://github.com/bytedance/deer-flow.git`.
- Local HEAD: `977b0b55` (2026-09-26, QuantFlow landing redesign).
- Fork point (merge-base): `656db122` (2026-09-21, #5622).
- Upstream HEAD: `8cd7aef5` (2026-10-02, #6175); identical in fetched `upstream/main` and the fresh clone.
- Commits behind: 315. Local-only: 73. Files changed: 310 local vs 1098 upstream. Both-sides overlap: exactly 71 files (via `comm`). Tags: 8 (v2.0-m0 … v2.1.0-rc0).
- CHANGELOG: local `[Unreleased]` → 2.1.0 milestone, 765 PRs (identical to merge-base); upstream released 2.1.0 on 2026-09-24 and targets 2.2.0 (301 PRs).
- README: local 218-line QuantFlow rewrite (`+195/-2158`, fork note L11-15) vs upstream `+308/-24` — full-file conflict certain.
- `config_version`: local 46 (= merge-base) vs upstream 50.
- Working tree at plan time: 5 modified (factory.py, lead_agent/agent.py, knowledge lookup.py + test, quant-desk SKILL.md — the memory-tools wiring) + pre-existing untracked `.extensions_config.json.lock`.

## Features behind (upstream-only)

Pre-fork exclusions, proven in QuantFlow via merge-base ancestry — do NOT re-merge: #5355 #5579 #5251 #5545 #5551 #5238 #5527-base #5528 #5489 #5596 #5582 #5497 #5308 #5564. PR→feature mappings verified by file/content presence, not individually traced to commits.

- **Memory (High):** #3182 cognitiveStyle (`MEMORY_COGNITIVE_STYLE.md` absent); #5565 agent-scoped mgmt API (needs file-scoped recheck during merge — raw counts disputed by critic); #5906 Jev prescreen + `signal`/`deerflow.typesafe` (`prescreen/` absent, tied to config 49→50).
- **Memory (Medium):** #5577 queued-payload redaction and #4066 `--all-users` bulk load — both unprobed, verify during merge.
- **Runtime (High):** #4929 artifact handles `art_*` (capture middleware + config absent); #6105 middleware-declared tool authz (`tool_declarations.py` absent); #5927 `history_search` role — CORRECTED, local HAS the tool, port only the `role` param (small surgical change).
- **Runtime (Medium):** #5947 json-valid criteria, #5570 upload cursor pagination, #5794 prompt overlays (`prompt_overlay.py` absent, tied to 47→48) — contents unprobed.
- **Authz/guardrails (High):** #5712 TypeSafe risk gate (`guardrails/typesafe.py`, 433 lines + eval script +168 `GUARDRAILS.md` lines; tied to 46→47); #4541 skill authz (`authz/skill_filter.py` absent); #5842 plugin targets (`authz/plugin_authz.py` absent, tied to 48→49).
- **Authz/guardrails (Medium):** #4989 route perms + per-user USER.md, #5651 system/developer 400 handling, #5621 #5816 #5799 #5736 — unprobed or not enumerated.
- **Extensions (High):** #5647 plugin router (`backend/app/gateway/routers/plugins.py`, 146 lines, absent).
- **Extensions (Medium/Low):** #5685 `browser_assets.py`, #5796 `model_invocation.py` (tied to 48→49), #5727 run-evidence delta, #6188 `catalog.ts`, examples #5731 #5735 #5833.
- **Storage (High/Medium):** #5361 blob contract (`storage/contract.py` 198 lines + `blob_storage_config.py`, tied to 46→47); #6159 blob-backed tool-output restore (unprobed).
- **Scheduler (Medium):** #6135 IM notification outbox (`notification_delivery.py` 334 lines + `notification_deliveries` table, WeCom `send_notification`).
- **Models/community (High/Medium):** #5780 reasoning contract (local legacy booleans only — port types alongside flags); #5981 Unbrowse (`tools.py` 174 lines); async migrations #5786 #5784 #5782 #5788; #6143 #6113 #5723 (unprobed).
- **Frontend (Medium/Low):** #6063 inline @ refs, #6154 reserved slash menu, #5758 skill snapshots (`middlewares/skill_usage.py`), #6126 cross-worker MCP cache resets — mostly unprobed.
- **Config migration (High, do in order):** 46→47 (#5361+#5712) → 47→48 (#5794) → 48→49 (#5842/#5796/#4929/#5577) → 49→50 (#5906). Exact 46→47 attribution unresolved (both read 47) — read both commits' migration bodies at merge time. Plus #5991 `config-upgrade.sh` live-path fix (local script keeps old order; fix content unprobed).
- **Notable fixes (High — ride with the merge):** #6066 DST double-fire, #6069 SQLite `file:` URI reject, #6058 `DEERFLOW_USER_ID` to bash, #5634 AIO bash timeouts, #5711 AIO fence, #5898 ledger parent scoping, #6083 direct-return results, #5566 compaction-model anchor, #6155 Codex SSE, #5966/#5956 MCP isolation.
- **Docs (Low):** #5845 #5769 #5775 #5761 #5929 #5688, not enumerated.

## Local-only inventory (preserve — do not lose)

- `skills/custom/quant-desk/SKILL.md` (340 lines; flagship skill; upstream has no counterpart; currently dirty).
- `config/quant_tools.yaml` (v1, 27 tools) + `scripts/validate_quant_tools.py` (133 lines).
- `backend/packages/harness/deerflow/knowledge/` (31 py files; lookup.py + test currently dirty).
- `AGENTS.md`, `README.md` (QuantFlow rewrite + fork note), `config.yaml` + `config.example.yaml` (merge upstream keys around local quant keys).
- `docs/plans/` + `plans/` conventions; the 73-commit local arc (preserve via merge + pre-merge tag).

## Predicted conflicts (71 files, resolve in this order)

- **README.md — K.** Guaranteed full-file conflict. Keep local branding + fork note; hand-port upstream README facts only if worthwhile.
- **agents/factory.py, agents/lead_agent/agent.py — M.** Both sides changed AND dirty. Stash first, merge hunk-by-hunk, keep QuantFlow wiring; re-run knowledge/agent tests after.
- **config/app_config.py — M.** Merge keys, then staged 46→50 migration in order.
- **models/factory.py — M.** Keep local wiring; port #5780 reasoning-contract types alongside legacy booleans; migrate callers later.
- **backend/uv.lock — T-then-regenerate.** Take upstream as base, re-add local deps, rebuild. Never hand-edit hunks.
- **backend/pyproject.toml — M.** Union of deps, keep QuantFlow extras; verify with `detect_uv_extras.py`.
- **config.example.yaml — M.** Union of keys with upstream defaults.
- **subagents/{config,executor,registry}.py — M each.** Keep local executor semantics; port upstream features hunk-by-hunk.
- **i18n locale files + e2e specs (chat, sidecar-chat, thread-history) — M.** Confirm exact paths from the 71-file list; union keys/intent, keep local strings.
- **Backend app + harness core — M each:** `app/gateway/app.py` (keep `X-Trace-Id` behavior), `app/channels/manager.py`, `gateway/routers/channel_connections.py`, 10 harness-core files (`client.py`, Jina tools, `subagents_config.py`, `mcp/tools.py`, `patched_openai.py`, `migrations/AGENTS.md`, `persistence/models/__init__.py`, `sandbox/tools.py`, read-before-write middleware), 9 `backend/tests/*` files (keep local intent; note deliberate divergences in the merge commit).
- **Frontend workspace (6 verified files) — M each:** `workspace-content.tsx`, `input-box.tsx`, `subtask-card.tsx`, `recent-chat-list.tsx`, `sidecar-panel.tsx`, `core/threads/hooks.ts`. Keep local branding; port #6063/#6154 hunk-by-hunk.
- **Docs/scripts/misc (low risk) — M:** 22 mdx docs (keep local edits, port upstream updates), `detect_uv_extras.py`, `serve.sh`, `docker/nginx/nginx.local.conf`.
- **Upstream-only new files — T as additions:** `routers/plugins.py`, `scheduler/notification_delivery.py` + migration, `guardrails/typesafe.py` + eval, `storage/contract.py` + `blob_storage_config.py`, `authz/{skill_filter,plugin_authz}.py`, `extensions/{browser_assets,model_invocation}.py`, `config/prompt_overlay.py`, `middlewares/{tool_declarations,skill_usage}.py`, artifact-capture middleware + config, Unbrowse tools, `catalog.ts`, `prescreen/` tree.

## Steps

1. **Quarantine + tag.** `git stash push -u -m pre-merge-20261002`; `git tag quantflow-pre-merge-20261002 HEAD`; `git fetch upstream`. Gate: status clean; tag listed.
2. **Reproduce the 71-file list.** Regenerate local/upstream file lists from `656db122`, `comm` them. Gate: overlap = 71; keep as the merge checklist.
3. **Restore work on a branch.** `git checkout -b merge/upstream-20261002`; `git stash pop`. Gate: 5 modified files + lock file return; diff stat matches.
4. **Config migration first (46→50).** Apply staged migrations in order; resolve 46→47 attribution from both commits' migration bodies. Gate: `config-upgrade` clean; version reads 50; configs validate.
5. **Merge in risk order.** `git merge upstream/main --no-commit --no-ff`; resolve conflicts section by section; land upstream-only files as-is. Gate per file: compiles/lints; `grep -r '<<<<<<<' backend frontend` empty.
6. **Surgical corrections.** (a) #5927 `role` param only; (b) #5565 file-scoped `agent_name` recheck; (c) #5991 `config-upgrade.sh` order. Gate: targeted knowledge + gateway tests.
7. **Lockfile + deps.** Regenerate `uv.lock` from merged pyproject; run `detect_uv_extras.py`. Gate: `uv lock --check` passes; imports resolve.
8. **Backend gates, all must pass:** `make -C backend lint`, format check, `test`, `test-knowledge`, `test-blocking-io`, `detect-blocking-io`, root `make check`. Any failure bisects to its conflict file first.
9. **Quant gates.** `validate_quant_tools.py` passes; quant_tools.yaml = 27 tools; SKILL.md diff shows only intended changes.
10. **Frontend gates.** Typecheck/lint/build; re-run touched e2e specs; triage selector churn with fixes.
11. **Docs + changelog.** CHANGELOG `[Unreleased]` merge entry (upstream 2.1.0, 315 commits, notable ports); README stays branded.
12. **Final review + merge commit.** Full `diff HEAD --stat` against the 71-file checklist; commit with per-file K/T/M summary. Gate: 71/71 accounted; decide `.extensions_config.json.lock` disposition (keep or gitignore).

## Validation Plan

- Highest-risk validation: step 8 backend gate suite on the merged tree — it is the first end-to-end proof the 315-commit bulk did not break QuantFlow behavior; run it before any frontend or docs work.
- Focused commands per step are listed above (stash/tag listing, `comm` count, `config-upgrade`, marker grep, targeted tests, `uv lock --check`, make targets, validator script, frontend build + e2e).
- Manual checks that cannot be automated: 71/71 checklist review, SKILL.md diff review, merge-commit message accuracy, lock-file disposition decision.

## Rollback

- Before merge commit: `git merge --abort`; delete the branch; state recoverable from stash + tag.
- After merge commit, before push: `git reset --hard quantflow-pre-merge-20261002`; delete branch.
- After push: `git revert -m 1 <merge-sha>` on `main` (never force-push shared `main`); re-verify the Local-only inventory after revert.
- Rollback test: status matches pre-merge dirty state or clean tag state, and `make -C backend test-knowledge` passes.

## Risks

- Uncommitted `factory.py`/`lead_agent` work lost or mis-merged (medium/high) — stash + tag before anything; restore on branch.
- 46→50 migration mis-ordered (medium/high) — read both 46→47 migration bodies; gate on `config-upgrade`.
- `uv.lock`/pyproject breakage (medium/high) — regenerate, never hand-edit; extras-detector gate.
- #5712 TypeSafe / #4541 authz behavior deltas vs QuantFlow flows (medium/medium) — merge then run knowledge + quant gates before accepting deltas.
- #5780 reasoning-contract port breaks local model callers (medium/medium) — types alongside flags; migrate callers after.
- 315-commit bulk hides a breaking gateway/authz change (low/high) — 71/71 checklist + full gate suite.
- Rebrand spillover into landing/workspace files (medium/medium) — README first with K; keep-branding default.
- Unprobed items hide conflicts (medium/medium) — verify-each-during-merge; no file marked done until its item is read.
- e2e selector churn blocks frontend gate (high/low) — triage as churn; does not block the backend merge commit.

## Open Questions

- Full 233-entry upstream Fixed list not individually enumerated; only security/notable fixes cited.
- Exact 46→47 attribution between #5361 and #5712 (both read 47 at their commits).
- PR-number→feature mappings verified by file/content presence only, not per-commit traced.
- Unprobed items: #5577, #4066, #5947, #5570, #4989, #5651, #5727, #6159, #6143/#6113/#5723, #6063/#6154/#6126, #5991 script, `GUARDRAILS.md` +168.
- Pre-fork exclusion list verified via merge-base-in-HEAD only, not per-PR is-ancestor.

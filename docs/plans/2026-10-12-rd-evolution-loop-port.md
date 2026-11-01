# RD-Agent Closed Evolution Loop — Native Quantflow Port

## Status

Plan (2026-10-12). Not yet implemented. This document is the build contract
for porting RD-Agent's closed evolution loop
(Hypothesis → Experiment → Feedback → reselect, with EvoStep trace DAG,
checkpoint/SOTA selectors, and scheduler-driven iteration) natively into
Quantflow, integrated with spark-mcp trial governance, the Knowledge Plane,
and Quantflow agent memory.

Reference files (read-only, never imported):

- `rdagent/core/proposal.py` — `Hypothesis`, `ExperimentFeedback`,
  `HypothesisFeedback`, `Trace` (hist + dag_parent DAG), `CheckpointSelector`,
  `SOTAexpSelector`, `ExpGen`/`HypothesisGen`/`Hypothesis2Experiment`/
  `Experiment2Feedback` ABCs
- `rdagent/core/evolving_framework.py` — `EvoStep`, `EvolvingStrategy`,
  `IterEvaluator`, `RAGStrategy`
- `rdagent/components/workflow/rd_loop.py` — `RDLoop` stage pipeline
  (`direct_exp_gen` → `coding` → `running` → `feedback` → `record`) plus the
  human-interaction queues this port **deletes by design**
- spark-mcp `src/spark_mcp/server.py` — trial-governance tool surface
  (`hypothesis_propose`, `register_hypothesis`, `log_outcome`,
  `significance_gate`, `check_novelty`, `memory_recall`, `memory_promote`,
  `suggest_next`, `backtest_run`)

## Placement rationale

Root `AGENTS.md` says to stay in core for "changes to the agent loop", and
`backend/AGENTS.md` enforces the one-way harness/app boundary
(`tests/test_harness_boundary.py`: App imports `deerflow`, never the
reverse). The loop therefore lives in a **new harness package**
`deerflow.evolution`, callable from the existing scheduler occurrence path
(non-interactive runs) and from a thin Gateway router in a later phase.

## Component breakdown

All new files under
`backend/packages/harness/deerflow/evolution/` (import prefix
`deerflow.evolution.*`):

| # | Target file | Contents |
| --- | --- | --- |
| 1 | `evolution/__init__.py` | Public exports only; lazy-load stage modules so importing the package never pulls langchain or MCP transport (mirrors the `deerflow.knowledge` import-cycle contract). |
| 2 | `evolution/models.py` | Frozen dataclasses, clean-room defined in Quantflow style: `Hypothesis` (hypothesis, reason, concise_* fields), `ExperimentSpec` (design + code + data + params + env envelope), `ExperimentFeedback`/`HypothesisFeedback` (decision bool, reason, observations, evaluation, new_hypothesis), `EvoStep` (step_id, iteration, subject, queried_knowledge refs, feedback, parent_step_ids). No rdagent base classes. |
| 3 | `evolution/trace.py` | `Trace` DAG: append-only `hist: list[EvoStep]`, `dag_parent: list[tuple[int, ...]]` with `NEW_ROOT = ()` / `SEL_LATEST_SOTA = (-1,)` semantics, `current_selection`, `get_parents`/`get_children`/`get_sota_experiment` ancestor walks. `TraceStore` Protocol (same no-I/O style as `ExperimentStore`) + `InMemoryTraceStore`. PG binding deferred to phase 2. |
| 4 | `evolution/selectors.py` | `CheckpointSelector` and `SOTASelector` Protocols plus deterministic built-ins: `LatestSOTASelector`, `AncestorSOTASelector`, `RootRestartSelector` (returns `None` = new tree). Pure functions over `Trace`; no LLM calls. |
| 5 | `evolution/loop.py` | `EvolutionLoop` async runner: `propose → design → execute → feedback → record → reselect` until budget (`max_iterations`, wall-clock, or stop-rule) exhausts. Owns step sequencing, idempotency-key derivation, exception→feedback conversion, and tracing spans. No queues, no prompts, no approval waits anywhere in this path. |
| 6 | `evolution/stages/propose.py` | `HypothesisGenerator`: builds proposal context (KB packet + spark recall + agent memory), calls the propose-profile LLM, validates the C6 field set (mechanism, expected_direction, horizon, falsifier, scope, predicted_ic/sigma; dag + adjustment_set iff causal), then `hypothesis_propose`. |
| 7 | `evolution/stages/design.py` | `ExperimentDesigner` (`Hypothesis2Experiment` role): converts hypothesis → `ExperimentSpec`, runs the pre-experiment recheck (`experiment_search` family/execution-hash lookup), then `register_hypothesis` + `experiment_begin`. Maps `DuplicateExecutionError` to link-to-prior-run instead of rerun. |
| 8 | `evolution/stages/execute.py` | Code + run: renders the spec into sandbox code (coder role) and executes it via the existing `SandboxProvider` boundary; quantitative evaluation dispatches through spark `backtest_run` (trial governor; tombstone pre-check honored, retries reuse the winner trial so M never double-counts). |
| 9 | `evolution/stages/feedback.py` | `FeedbackSynthesizer` (`Experiment2Feedback` role): runs `significance_gate` (DSR/MinBTL), builds the decision + reason + observations via the summarize-profile LLM, then `log_outcome` + `experiment_commit` (+ `failure_record` / `assumption_record` as applicable). Exceptions become `decision=False` feedback, mirroring `RDLoop.feedback`. |
| 10 | `evolution/spark_client.py` | Thin typed async client over the spark-mcp trial-governance tools listed above, over MCP transport. Never imports `spark_mcp` (separate repo, reference-only). Owns `client_id` idempotency tokens and error mapping ( refusals / ToolError → typed `EvolutionError`). |
| 11 | `evolution/knowledge_adapter.py` | Binds the loop to `deerflow.knowledge.write_api` (`experiment_begin`/`experiment_commit`/`failure_record`/`assumption_record`), `search.experiment_search`, and the Phase-2 retrieval path (`parse_research_intent` → `execute_retrieval` → `build_context_packet`; bootstrap `intent_from_text` for run start). Takes caller-supplied store Protocols; performs no I/O itself. |
| 12 | `evolution/memory_adapter.py` | Binds the loop to agent memory: pre-proposal `MemoryManager.get_context` / `search` (query-aware recall with the hypothesis topic as hint), post-feedback `add` of the compact turn summary, and `memory_promote` (spark belief) only when the gate + confidence policy passes. Never touches the `MemoryManager` contract itself. |
| 13 | `evolution/config.py` | `EvolutionConfig` (budgets, family/universe defaults, gate thresholds, model-role mapping) + `knowledge:`-style `AppConfig` section registration. Hot-reloadable scalars only; model roles resolve at loop construction. |
| 14 | `evolution/prompts.py` | Versioned static prompt templates for propose/design/summarize roles. Static by policy: spark owns prompt/strategy evolution (see non-goals). |
| 15 | `evolution/errors.py` | Typed errors: `EvolutionError` base, `GovernanceRefusalError`, `GateFailedError`, `BudgetExhaustedError`, `DuplicateRunError`. KB/spark errors propagate unwrapped except where mapped. |

Phase-2 (optional, not in initial build): `app/gateway/routers/evolution.py`
(run start/stop/status over the existing run lifecycle) and a PG-backed
`TraceStore`. The initial build is drivable from scheduled-task occurrences
and tests via `deerflow.client`-style programmatic entry.

## Stage-to-tool mapping

Each loop stage calls exactly these external tools (KB = Knowledge Plane
write_api/search/retrieval; spark = spark-mcp MCP tools; mem = agent memory):

| Loop stage | KB | spark-mcp | Agent memory | LLM role |
| --- | --- | --- | --- | --- |
| Propose (context) | `parse_research_intent` → `execute_retrieval` → `build_context_packet` (consensus, prior experiments, failures, conflicts, assumptions, skills) | `memory_recall`, `check_novelty`, `suggest_next` (underexplored-region bias), `research_history` (optional PIT history) | `get_context` / `search` with hypothesis topic as query hint | propose |
| Propose (register) | — | `hypothesis_propose` (C6 field validation before the call) | — | — |
| Design (recheck) | `experiment_search` (family-hash equivalents + execution-hash exact reruns) | — | — | — |
| Design (register) | `experiment_begin` (status `planned`→`running`) | `register_hypothesis` (episode + protocol_hash) | — | design |
| Execute | result artifacts staged via the artifact store | `backtest_run` (governed dispatch; tombstone ack only from recorded rationale, never prompted) | — | coder (sandbox) |
| Feedback (gate) | — | `significance_gate` (DSR/MinBTL verdict) | — | — |
| Feedback (close-out) | `experiment_commit` (+ `failure_record` when outcome is failure; `assumption_record` for new/tested assumptions) | `log_outcome` (metrics, costs, reward) | `add` (compact summary); `memory_promote` only if gate passed and confidence ≥ threshold, contradictions resolved | summarize |
| Reselect | — | `suggest_next` / `archive_status` as advisory input to selectors | — | none (pure) |
| Record | parent linkage via `parent_experiment_id` (KB) | parent linkage via `parent_hypothesis_ids` (spark) | — | — |

Ordering rule: spark `register_hypothesis` precedes KB `experiment_begin`
(design), and spark `log_outcome` precedes KB `experiment_commit`
(feedback), so the trial governor is always the first writer and the KB
never records an ungoverned run. Every write carries its idempotency key
(see below); on retry the loop replays keys rather than re-registering.

## Data-model mappings (EvoStep id → idempotency keys)

One `EvoStep` (with stable `step_id: UUID`, `run_id: UUID`,
`iteration: int`) fans out to six idempotent writes. Keys derive
deterministically from `deerflow.knowledge.hashing.make_idempotency_key`
(`SHA256(run_id || sequence || canonical_payload)` under the
`quantflow.idempotency/v1` domain):

| Write | Key material (`run_id`, `sequence`, `payload`) | Notes |
| --- | --- | --- |
| spark `hypothesis_propose` (`client_id`) | (`run_id`, `1000 + iteration`, canonical hypothesis fields) | Replays the original `hypothesis_id` on retry. |
| spark `register_hypothesis` (`client_id`) | (`run_id`, `2000 + iteration`, `{hypothesis_id, spec, seed}`) | Replays the original `episode_id` + protocol_hash. |
| spark `backtest_run` (`client_id`) | (`run_id`, `2500 + iteration`, `{episode_id, spec, seed}`) | Own base (not shared with register); replays the governed trial result. |
| KB `experiment_begin` (`idempotency_key`) | (`run_id`, `3000 + iteration`, `{project_id, created_by_run_id, execution_hash}`-aligned payload) | Fingerprint-compatible with `write_api` replay semantics; `DuplicateExecutionError` → link to `existing_experiment_id`, skip execution. |
| spark `log_outcome` (`client_id`) | (`run_id`, `4000 + iteration`, `{episode_id, outcome, metrics}`) | Single close-out per episode. |
| KB `experiment_commit` (`idempotency_key`) | (`run_id`, `5000 + iteration`, `{experiment_id, outcome, metrics, artifacts}`) | Single-shot terminal transition; replay returns the committed record. |
| KB `failure_record` / spark `memory_promote` | (`run_id`, `6000/7000 + iteration`, detail payload) | Only on the failure / promotion branches. |

Identity linkage across systems:

- `EvoStep.step_id` ↔ KB `ExperimentRecord.id` (stored in the step at begin
  time) ↔ spark `episode_id` (stored at register time) ↔
  `hypothesis_id` (stored at propose time). The step record carries all
  three; no reverse lookup by mutable text.
- DAG edges: `Trace.dag_parent` index tuples ↔ KB
  `parent_experiment_id` ↔ spark `parent_hypothesis_ids`. `NEW_ROOT`
  (fresh tree) maps to null parents on both sides.
- Content identity reuses KB hashing, never invented: design equivalence =
  `experiment_family_hash`, exact-rerun detection = `execution_hash`
  (globally unique; the recheck in Design consults it before any compute).
- `created_by_run_id` on every KB record is the loop `run_id`, so one
  evolution run's experiments are queryable as a set.

## LLM profile usage

Hard rule: **every LLM call goes through
`deerflow.models.create_chat_model`** (LangChain model profiles/factory).
No LiteLLM direct calls, no ad-hoc provider clients, no DeerMem-style side
construction (`agents/memory/backends/deermem/.../llm.py` is the
counter-example, not the pattern). No `import litellm` (or
`from litellm ...`) anywhere in `deerflow.evolution` including its
tests; `tests/test_evolution_import_guards.py` fails on any occurrence.

- Role→profile mapping lives in `EvolutionConfig.model_roles`
  (`{propose, design, summarize, coder}` → `config.yaml` model `name`),
  resolved once at loop construction via `AppConfig.get_model_config`.
  Missing profile = fail fast with the model name in the error.
- Standalone loop turns (scheduler-driven, outside a LangGraph run) use
  `attach_tracing=True` so model-level callbacks still emit traces; any
  future in-graph wiring must pass `attach_tracing=False` to avoid
  duplicate spans (factory docstring contract).
- Per-role sampling uses `model_overrides` (`temperature`, `max_tokens`);
  `None` values never clobber profile values. Per-run `reasoning_effort`
  flows through the factory's reasoning contract (issue #5073 semantics);
  the loop never hand-builds thinking payloads.
- Prompts are static versioned strings in `evolution/prompts.py`. The loop
  never rewrites its own prompts (spark owns prompt/strategy evolution).

## Test strategy (TDD mandatory)

Backend `AGENTS.md` makes TDD mandatory: tests in `backend/tests/` named
`test_<feature>.py`, offline (`httpx.MockTransport`, dummy keys), `ruff
format` clean before push. The loop follows the KB modules' own precedent:
pure logic over caller-supplied store Protocols, so unit tests need no DB.

Unit tests (one file per component, all offline):

- `tests/test_evolution_models.py` — dataclass validation, feedback
  truthiness (`__bool__` = decision), serialization round-trips.
- `tests/test_evolution_trace.py` — DAG append/parent/children walks,
  `NEW_ROOT`/`SEL_LATEST_SOTA` semantics, SOTA ancestor search, empty-trace
  edges, multi-tree (`None` selection) behavior.
- `tests/test_evolution_selectors.py` — each selector over scripted traces,
  including empty trace and all-rejected trace.
- `tests/test_evolution_idempotency.py` — key derivation determinism
  (same inputs → same key; any field change → different key), replay
  semantics against the in-memory `ExperimentStore` fake
  (equal-payload replay returns original; different-payload raises
  `IdempotencyConflictError`; execution-hash rerun raises
  `DuplicateExecutionError`).
- `tests/test_evolution_stages.py` — each stage with a stub spark client +
  in-memory KB stores: C6 validation rejects incomplete hypotheses;
  `DuplicateExecutionError` links instead of rerunning; exception in
  execute becomes `decision=False` feedback with `failure_record`.
- `tests/test_evolution_no_rdagent.py` — import-graph guard: fails if
  the string `rdagent` (case-insensitive) appears in any import,
  requirement file (`requirements*.txt`, `pyproject.toml`, lockfiles),
  or test fixture under the evolution package and its tests
  (companion to `test_harness_boundary.py`, which must keep passing:
  `deerflow.evolution` imports `deerflow.*` only, never `app.*`).
- `tests/test_evolution_import_guards.py` — fails if `litellm` is
  imported anywhere in `deerflow.evolution` including its tests (all
  LLM access via `create_chat_model`), and fails if `spark_mcp` is
  imported anywhere in Quantflow (spark is reached over MCP transport
  only, never as a Python dependency).

E2E tests (offline, deterministic):

- `tests/test_evolution_loop_e2e.py` — full loop over scripted fixtures:
  stub spark client (canned gate/promote verdicts), in-memory KB stores,
  deterministic fake chat model (fixed hypothesis/spec/feedback payloads),
  local-sandbox execute stub. Asserts: N iterations → N EvoSteps,
  DAG parents consistent, every write idempotent under loop restart
  (re-run with same `run_id` replays, creates nothing new), budget
  exhaustion stops cleanly, SOTA selector returns the gated winner.
- Scheduler-tick test: drive one `EvolutionLoop.step()` from a
  non-interactive scheduled occurrence context to prove no
  `ask_clarification` / prompt / queue dependency in the loop path.

## Explicit non-goals

1. **No RD-Agent dependency.** No `rdagent` imports anywhere including
   tests, no `rdagent` in requirements files or lockfiles, no vendored
   RD-Agent code. This is a clean-room native reimplementation of the
   loop *concepts* in Quantflow style (frozen dataclasses, store
   Protocols, no-I/O modules), enforced by
   `tests/test_evolution_no_rdagent.py`.
2. **No human-in-the-loop.** The RD-Agent `user_request_q` /
   `user_response_q` interaction points (`_interact_init_params`,
   `_interact_hypo`, `_interact_feedback`) are not ported. No queues,
   prompts, or approval waits exist in the loop path; `deerflow.evolution`
   imports no queue/prompt/approval helpers, and tombstone
   acknowledgements and belief promotions resolve from recorded policy,
   never from a live prompt. The scheduler-tick e2e test proves the loop
   path has no `ask_clarification` / prompt / queue dependency.
3. **No LiteLLM direct.** All LLM access via `create_chat_model` model
   profiles (see above).
4. **Do not modify spark-mcp.** `src/spark_mcp/evolve/gepa.py`,
   skill-promotion tools (including `memory_promote` server-side
   logic), and every other spark-mcp file are untouched; spark keeps
   prompt and strategy evolution. Quantflow integrates as an MCP
   *client* of the trial-governance tools only and never imports
   `spark_mcp` (guarded by
   `tests/test_evolution_import_guards.py`).
5. **No prompt/strategy evolution in Quantflow.** Loop prompts are static
   versioned strings; novelty/surprise scoring, QD archive, GEPA, and
   skill replay stay spark-side.
6. **No Knowledge Plane schema changes in phase 1.** The loop uses the
   existing Phase-1 experiment/failure/assumption schema and Phase-2
   retrieval via their Protocols; a PG-backed `TraceStore` migration is
   explicitly phase 2.
7. **No `MemoryManager` contract changes.** The loop is a client of
   `get_context`/`search`/`add` only.
8. **No Gateway router or frontend UI in phase 1.** Programmatic +
   scheduled-task entry first; HTTP/SSE exposure follows once the loop is
   green e2e.
9. **No new scheduler semantics.** The loop reuses scheduled-task
   occurrences as its tick source; cron/timezone/stop-rule behavior is
   unchanged.

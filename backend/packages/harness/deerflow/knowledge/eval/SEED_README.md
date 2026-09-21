# Phase 2 eval seeder — `seed_phase2.py` + `eval_fixture_phase2.json`

Deterministic seeder turning the curated retrieval-eval fixtures into
committable Knowledge Plane payloads. Phase 0 proved the eval harness
works (`runner.py` + `eval_fixture.json`, frozen); this directory extends
it for Phase 2 (`implementation_plan.md` § Phase 2: *"agents retrieve
known relevant priors + known failures before research begins"*).

## Contents

| File | Purpose |
|---|---|
| `eval_fixture_phase2.json` | Phase 2 corpus extension: 10 docs (5 experiments, 3 failures, 2 findings) + 6 queries with graded relevance + must-recall failure cases |
| `seed_phase2.py` | Deterministic seeder: fixture docs → experiment / finding / artifact payloads + judgments + packet-budget checks + embedded self-test |
| `SEED_README.md` | This file |

No existing file was modified: `runner.py`, `eval_fixture.json`, and
`README.md` are untouched. The extension fixture pins schema version
`0.1.0`, so the frozen runner loads it unchanged:

```bash
cd backend/packages/harness/deerflow/knowledge/eval
python runner.py --fixture eval_fixture_phase2.json --retriever oracle  # exit 0
python runner.py --fixture eval_fixture_phase2.json --retriever null    # exit 2
```

## Seeder usage

```bash
cd backend/packages/harness/deerflow/knowledge/eval

python seed_phase2.py --self-test        # embedded unit tests (29 tests) -> exit 0
python seed_phase2.py --check            # coverage + counts -> exit 0
python seed_phase2.py --check --json     # machine-readable check
python seed_phase2.py --emit bundle.json # write the deterministic seed bundle
python seed_phase2.py --validate-writes  # write_api round-trip on a memory store
```

Observed: 46 merged docs (36 + 10), 36 merged queries (30 + 6), 40
experiment seeds, 6 finding payloads, 61 artifact seeds (46 result + 15
failure logs). `--validate-writes` commits 40 experiments / 15 failures
through the real `write_api` validators.

## Payload shapes

### Experiments — `write_api` kwargs, verbatim

`build_experiment_seed(doc)` returns kwargs that splat directly into the
production write path (`seed_store()` does exactly this):

- `experiment_begin(store, **begin_kwargs)` — `project_id` (fixed eval
  project UUID), `hypothesis` (fixture title), `methodology` (KB-canonical
  design keys: universe, frequency, horizon, sample period, protocol,
  portfolio construction, cost/slippage models, neutralization, rebalance
  rule), `parameters` (tool params + deterministic seed + retrieval text),
  `datasets` (one `features` link, UUIDv5), `code`/`environment` identity
  (tool + digests), `created_by_run_id` (fixed eval-run UUID),
  deterministic `idempotency_key`.
- `failure_record(store, experiment_id, **failure_kwargs)` — failures only
  (`failure_class` + notes + key).
- `experiment_commit(store, experiment_id, **commit_kwargs)` — experiments
  commit `outcome="success"`; failures commit `outcome="failure"` with the
  fixture `failure_class`; metrics are the fixture metrics verbatim;
  `result_artifacts` links the seeded result artifact ID.

Family/execution hashes are computed by `write_api` itself (never
reimplemented); the self-test recomputes family hashes via
`deerflow.knowledge.hashing` and asserts equality with stored values.

### Findings — `FindingRow` fields, forward-compatible

No finding table exists yet (latest migration is `0022_knowledge_phase1`,
episodic half only), so `build_finding_payload()` mirrors the
`finding` object in `knowledge_base.md` field-for-field — the columns the
Phase 2 schema owner will bind: `finding_id`, `canonical_key`
(`{type}:{slug}`), `statement`, `finding_type` (curated per doc in
`FINDING_TYPE_BY_DOC`, never inferred), `scope` (KB scope keys only),
`status` (`candidate` — the only Phase 2 status), `confidence`
(`overall_tier`), `valid_time`, `created_by_run_id`, `supersedes_*`,
plus `evidence` edges (`supports` relations to the co-judged
experiments) and an `idempotency_key`. `validate_finding_payload()`
enforces the KB enum vocabularies; `resolve_finding_evidence()` fills
evidence `experiment_id`s post-commit. `embedding` / `search_document`
are index projections owned by the retrieval worker — the payload carries
the `retrieval_text` they derive from.

### Artifacts — `ArtifactStore` shapes

`build_artifact_seeds(doc)` returns one `result` seed per doc plus a `log`
seed per failure. Each carries `kind`, `media_type`, `byte_size`,
`sha256`, and `bytes_text` (the exact canonical JSON the digest covers).
Materialize with
`ArtifactStore.put_bytes(bytes_text.encode("utf-8"), kind=..., ...)` —
the store recomputes the identical digest (asserted by round-trip test).

## Determinism contract

Every derived value is a pure function of fixture bytes + module
constants: UUIDs are UUIDv5 under `SEED_NAMESPACE`, idempotency keys are
`phase2-eval-v1:{doc_id}:{op}`, seeds/digests are SHA-256 derivations.
No timestamps, no `uuid4`, no dict-order dependence — two builds are
byte-identical (asserted). The only non-deterministic surface is
store-assigned (`ExperimentRecord.id`, `started_at`, artifact
`created_at`), and idempotent replay (same keys, same payload) returns
the original records (asserted by double-seed test).

## Judgments + failure-recall cases

`build_judgments()` exports the merged known-relevance judgments
(`relevant` ranked grade-2-first, `grades`, `must_recall_failures`).
`check_judgment_coverage()` enforces six invariants: every query has
relevant judgments **and** failure-recall cases; no dangling IDs; every
must-recall ID is a `failure` doc; every failure is must-recalled by ≥1
query; every doc is judged somewhere. Both the merged fixture (46/36)
and the Phase 2 extension alone (10/6) satisfy all six.

## Extended runner contract (documented here, `runner.py` not edited)

Phase 2 exit = **recall@k ≥ 0.8 AND failure-recall@k ≥ 0.8 (k=10) AND
every context packet within budget**. The recall halves are the frozen
runner's `evaluate()` (called untouched); the packet half is new in
`seed_phase2.py`:

- `assemble_packet()` builds a KB-shaped packet per scored ranking:
  findings → `consensus`, experiments → `prior_experiments`, failures →
  `failures`; contradictions/assumptions/priors-skills/open-questions are
  empty lists until the Phase 3+ owners populate them (shape-stable).
- `PacketBudget` (defaults: 12000 total chars, 10 items, 2000 chars/item,
  evidence pointers required on consensus/failure items) encodes *"the
  packet fits a fixed context budget; L2 opens on demand"*.
- `check_packet_budget()` / `assert_packet_budget()` (raising
  `PacketBudgetError` with named violations), `evaluate_with_packet_checks()`
  (runner report + `PacketReport`), and `phase2_exit_verdict()` (single
  pass/fail + reasons) compose the exit gate:

```python
from seed_phase2 import (build_evidence_map, build_seed_bundle,
                         evaluate_with_packet_checks, load_merged_fixture,
                         load_runner, phase2_exit_verdict)

runner = load_runner()
fixture = load_merged_fixture()
bundle = build_seed_bundle()
evidence = build_evidence_map(bundle, experiment_ids=None)  # or post-commit UUIDs

def knowledge_retriever(query, k): ...  # Phase 2 adapter -> doc IDs

report, packets = evaluate_with_packet_checks(
    fixture, knowledge_retriever, evidence, k=10, retriever_name="knowledge_search")
passed, reasons = phase2_exit_verdict(report, packets, target=0.8)
```

Oracle rankings pass all three gates; null rankings fit the budget but
fail recall (asserted). `test_runner_module_was_not_edited_for_phase2`
pins that `runner.py` carries no Phase 2 symbols.

## Integration checklist (wiring owner)

1. Commit the bundle: `seed_store(real_store, build_seed_bundle())`;
   resolve findings via `resolve_finding_evidence(bundle, commit_map)`
   and insert them when the finding migration lands.
2. Materialize artifacts with `ArtifactStore.put_bytes(...)` per seed;
   committed `result_artifacts` IDs already match seed artifact IDs.
3. Index `retrieval_text` (experiments/failures) and finding statements
   into the FTS + pgvector projections; keep `failure_class` and scope as
   structured filters and the failure channel separate.
4. Implement the `knowledge_retriever` adapter above (intent → scoped
   hybrid retrieval → fixture doc IDs) and gate Phase 2 exit on
   `phase2_exit_verdict(...)` at `k=10`, `target=0.8`.
5. Extend by appending fixtures with fresh `p2`-style IDs and curated
   `FINDING_TYPE_BY_DOC` entries — never rewrite judgments in place.

## Conventions followed

- Backend Python style: `ruff`-clean (line-length 240, double quotes),
  full docstrings, type hints, `encoding="utf-8"` on file I/O.
- Harness/app boundary: no `app.*` imports; `deerflow` imports are lazy
  (payload math stays stdlib-only) with the `test_knowledge_api.py`
  `sys.path` bootstrap for standalone runs.
- `runner.py` is loaded by file path under a private module name — never
  edited, never shadowed; the extension fixture reuses its frozen schema.
- Concurrency rule: three new files under `knowledge/eval/`, zero edits
  to existing files; tests pass standalone (`--self-test`, no DB/config).

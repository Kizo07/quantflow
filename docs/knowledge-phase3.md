# Knowledge Plane Phase 3: Experiment Embeddings (landed)

Phase 3 generalizes the knowledge plane's vector leg from findings to
experiments. Landed in commit `7c4a9a9a`; the exit gate passes with the
real MPNet checkpoint offline (verified 2026-09-26).

## What it contains

- **Schema**: nullable `experiment.embedding` (`VECTOR(768)` on PostgreSQL /
  pgvector, JSON fallback on SQLite) plus Alembic revision `0028`.
- **Backfill**: `ExperimentTextSource` +
  `backfill_experiments_embeddings()` built on the shared Phase 2
  backfill core next to the findings path.
- **Two-table vector search**: the retrieval plane's vector channel now
  searches findings *and* experiments (`<=>` on PG, portable cosine
  elsewhere), with both embedding kinds fully backfilled.
- **Exit gate**: `backend/tests/test_knowledge_phase3_exit_gate.py` —
  sectioned recall@10 >= 0.8 (priors and failures sections) with the
  vector channel armed and contributing to every fused top-10.

## No longer findings-only

The vector leg previously embedded findings alone; experiment recall
relied on structured + lexical channels. Phase 3 arms the vector
channel over both tables, so experiment priors and failure records get
semantic ranking too.

## Running the gate

Requires the optional `knowledge-st` extra in `backend/.venv` (venv-only;
the core install stays torch-free) and a cached
`sentence-transformers/all-mpnet-base-v2` checkpoint — never downloaded
implicitly:

```sh
cd backend
uv pip install --python .venv/bin/python "sentence-transformers==3.4.1"
HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python -m pytest tests/test_knowledge_phase3_exit_gate.py
```

The gate skips when `sentence_transformers` or the checkpoint is
unavailable; mechanics stay covered by the fake-backed retrieval tests.

# Knowledge Plane: SQLite → PostgreSQL migration (QuantFlow KB)

Date: 2026-09-26. Operator: pg-migrator session (user-approved).

## What / where

| Item | Value |
|---|---|
| Source (kept as rollback) | `/home/fire/Documents/Audit/kb/quantflow_kb.db` (17 MB, untouched) |
| Fresh backup | `/home/fire/Documents/Audit/kb/quantflow_kb.db.bak-20260926-002209` (md5-identical) |
| Target | Postgres 18.6, database `quantflow_kb`, owner `fire`, socket `/run/postgresql` |
| pgvector | server extension `vector` 0.8.6 (installed via `sudo pacman -S pgvector`) |
| Alembic head | `0028_knowledge_experiment_embeddings` (full 0001→0028 chain) |
| Live DSN | `knowledge.database_dsn = postgresql+psycopg://fire@/quantflow_kb?host=/run/postgresql` in gitignored `config.yaml` (~line 543) |
| Loader (durable) | `/home/fire/Documents/Audit/kb/tools/sqlite_to_pg.py` |
| Data | 1 project, 2 runs, 1362 artifacts, 682 findings (all embedded); 5 tables empty as in source |

## How (reproducible steps)

1. **pgvector check.** `CREATE EXTENSION vector` requires superuser; role `fire`
   cannot run it. Verified via the `postgres` socket role in a scratch DB
   (created, verified `vector` 0.8.6 + `<->` operator, dropped).
2. **Backup.** Timestamped `cp` of the SQLite file; size + md5 verified.
   The SQLite file is never deleted.
3. **Database.** As `postgres` superuser:
   `CREATE DATABASE quantflow_kb OWNER fire;` then
   `CREATE EXTENSION IF NOT EXISTS vector;` inside it (pre-installed so the
   0023/0028 `CREATE EXTENSION IF NOT EXISTS` steps are no-ops for `fire`).
4. **Python drivers.** The backend venv had no PG drivers, so installed the
   repo's own `postgres` extra pins surgically (no other package touched):
   `uv pip install --python backend/.venv/bin/python "asyncpg>=0.29" "psycopg[binary]>=3.3.3"`.
   The running gateway (`uv run --no-sync`) is unaffected until restart.
5. **Alembic 0001→0028** via the repo-blessed in-process config
   (`deerflow.persistence.bootstrap._get_alembic_config` + `upgrade head`)
   over `postgresql+asyncpg://fire@/quantflow_kb?host=/run/postgresql`.
   - ⚠️ **Repo bug worked around (no backend/ code touched):** revision id
     `0028_knowledge_experiment_embeddings` is 36 chars but alembic's default
     `alembic_version.version_num` is `VARCHAR(32)`; on Postgres the final
     stamp fails (`StringDataRightTruncationError`) and the whole upgrade
     rolls back. (SQLite ignores VARCHAR lengths, which is why this never
     surfaced.) Workaround: pre-created
     `alembic_version(version_num TEXT NOT NULL)` before upgrading; alembic
     then uses the existing table and stamps 0028 cleanly. Owning team
     should ship a proper fix (e.g. a widening revision) — this doc does
     not change the chain.
   - Benign: one `safe_add_column` drift warning (`runs.change_seq`
     BIGINT vs BIGINTEGER, type-name spelling only, app-tables branch).
   - Note: the full chain also creates the (empty) app tables in
     `quantflow_kb`; the KB is "at migration 0028" per `config.example.yaml`.
6. **Load.** `sqlite_to_pg.py` copies all 9 tables in FK order with
   adaptations: 32-hex → UUID, JSON text → JSONB, naive DATETIME → UTC
   timestamptz, embedding JSON → `VECTOR(768)` literal + `::vector` cast
   (dims asserted == 768, `repr` floats). `finding.search_document` is NOT
   copied (SQLite holds plain text, PG holds TSVECTOR); it is rebuilt with
   the repo's own refresh expression
   `to_tsvector('english', canonical_key || ' ' || statement)`
   (mirrors `pg_retrieval._finding_reindex_update`). Self-referencing FKs
   load in two passes. All per-table counts matched (1/2/1362/0/0/682/0/0/0).
7. **Verify.** Independent script: counts match; 3 finding embeddings
   round-trip with PG cosine-to-self = 1.000000000000 and max abs diff vs
   SQLite ≈ 4e-9 (dims 768); lexical query (`bayesian` → 7 hits) and vector
   ordering query work; 682/682 non-null `search_document` + `embedding`.
8. **Cutover.** Set `knowledge.database_dsn` in local gitignored
   `config.yaml` (field wins over `DEER_FLOW_KNOWLEDGE_DSN`, which is unset).
   Probed the real load path (`load_knowledge_config_from_dict` →
   `get_database_dsn`) plus `open_search_store` (0 experiments = correct,
   source has none), `SQLLexicalSearchStore` (5 hits) and
   `SQLVectorSearchStore` self-probe (top hit = self) — all pass.

## Restart needed to take effect

The gateway (PID 4800, `uv run --no-sync`, port 8001) was deliberately NOT
restarted. It keeps its old (unbound) knowledge binding until restart; then
it picks up the PG DSN from `config.yaml`. After restart, re-run the step-8
probe to confirm the live binding. Durable-env note: a future plain
`uv sync` (without the `postgres` extra) would remove asyncpg/psycopg from
the venv — add the extra to the deploy sync so the PG read path survives
reprovisioning.

## Rollback

1. In `config.yaml`, remove the `knowledge.database_dsn` line (or point it
   back at `sqlite:////home/fire/Documents/Audit/kb/quantflow_kb.db`).
2. Restart the gateway.
3. The SQLite file + timestamped backups are intact; `quantflow_kb` can be
   left alone or dropped (`DROP DATABASE quantflow_kb;` as `postgres`).

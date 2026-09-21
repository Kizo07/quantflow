DEER_FLOW_HOME ?= $(CURDIR)/.deer-flow
DEER_FLOW_HOME := $(abspath $(DEER_FLOW_HOME))
BACKEND_SANDBOX_HOME := $(abspath $(CURDIR)/sandbox)

install:
	uv sync --locked

dev:
	mkdir -p "$(DEER_FLOW_HOME)" "$(BACKEND_SANDBOX_HOME)"
	PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 DEER_FLOW_HOME="$(DEER_FLOW_HOME)" uv run --locked uvicorn app.gateway.app:app --host 0.0.0.0 --port 8001 \
		--reload \
		--reload-include='*.yaml' \
		--reload-include='.env' \
		--reload-exclude='*.pyc' \
		--reload-exclude='__pycache__' \
		--reload-exclude="$(BACKEND_SANDBOX_HOME)" \
		--reload-exclude="$(DEER_FLOW_HOME)"

gateway:
	PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 uv run --locked uvicorn app.gateway.app:app --host 0.0.0.0 --port 8001

test:
	PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 uv run pytest -m "not live" --ignore=tests/blocking_io tests/ -v

test-live:
	DEER_FLOW_RUN_LIVE_TESTS=1 PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 uv run pytest -m live tests/ -v -s

test-blocking-io:
	PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 uv run pytest tests/blocking_io -q --tb=short

# Research Knowledge Plane tests: write/search API (in-memory fake), artifact
# store, Phase 1 schema + migration, PG search-store binding, AppConfig wiring.
test-knowledge:
	PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 uv run pytest -q --tb=short \
		tests/test_knowledge_artifacts.py \
		tests/test_knowledge_schema_phase1.py \
		tests/test_knowledge_pg_store.py \
		tests/test_knowledge_config_wiring.py \
		packages/harness/deerflow/knowledge/test_knowledge_api.py

# Research Knowledge Plane retrieval eval (Phase 0 fixture). Runs the
# runner's embedded self-test (must pass) plus the keyword baseline. Exit 2
# from the baseline run is the expected pre-Phase-2 state (recall below
# target); exit 1 (usage/fixture error) fails the target.
eval-knowledge:
	PYTHONIOENCODING=utf-8 PYTHONUTF8=1 uv run python packages/harness/deerflow/knowledge/eval/runner.py --self-test
	@PYTHONIOENCODING=utf-8 PYTHONUTF8=1 uv run python packages/harness/deerflow/knowledge/eval/runner.py --retriever keyword; code=$$?; \
	if [ $$code -eq 2 ]; then \
		echo "eval-knowledge: baseline below target (exit 2) -- expected pre-Phase-2 state"; \
	elif [ $$code -ne 0 ]; then \
		exit $$code; \
	fi

# Run the backend unit-test suite as one shard of a parallel CI split.
#   SPLITS = total number of shards; GROUP = 1-based shard index.
# Duration-aware: `least_duration` balances the shards by real wall-clock cost
# read from .test_durations. Shards only READ that file (no --store-durations),
# so concurrent CI jobs cannot race writes on it. Refresh it with
# `make test-shard-durations` after a meaningful change to the test set.
# `make test-shard SPLITS=4 GROUP=2` runs shard 2 of 4; a bare `make test-shard`
# runs the whole (non-live) suite as a single split, matching `make test`.
SPLITS ?= 1
GROUP ?= 1
DURATIONS_FILE ?= .test_durations
test-shard:
	@test -f "$(DURATIONS_FILE)" || \
		(echo "error: $(DURATIONS_FILE) is missing; run 'make test-shard-durations' to regenerate the duration baseline" >&2; exit 1)
	PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 uv run pytest -m "not live" --ignore=tests/blocking_io --splits $(SPLITS) --group $(GROUP) --splitting-algorithm least_duration --durations-path=$(DURATIONS_FILE) tests/ -q

# Regenerate .test_durations from the full offline (non-live, non-blocking-I/O)
# suite so duration-aware sharding reflects real wall-clock cost. Run locally or
# in a periodic job, then commit the refreshed file.
test-shard-durations:
	PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 uv run pytest -m "not live" --ignore=tests/blocking_io --store-durations --clean-durations --durations-path=$(DURATIONS_FILE) tests/ -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check . --fix && uv run ruff format .

detect-blocking-io:
	@PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 uv run python ../scripts/detect_blocking_io_static.py --output ../.deer-flow/blocking-io-findings.json

# Generate a new alembic revision by autogenerating the diff against the live
# ORM models. Usage: make migrate-rev MSG="add foo column to runs"
# The Gateway runs `alembic upgrade head` automatically at startup via
# `bootstrap_schema`, so there is intentionally no `migrate` / `migrate-stamp`
# target -- the single execution path keeps ops mistakes off the table.
# The script builds a fresh temp SQLite at head, then diffs the live models
# against it, so a clean checkout does NOT need a pre-existing ./data/deerflow.db.
migrate-rev:
	@if [ -z "$(MSG)" ]; then echo 'usage: make migrate-rev MSG="describe the change"'; exit 1; fi
	PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 uv run python scripts/_autogen_revision.py "$(MSG)"

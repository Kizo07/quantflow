#!/usr/bin/env python3
"""QuantFlow tool-awareness eval: runner and scorer (plan Phase 0b, package F).

Drives the Gateway API the way the web UI does, one run at a time, and scores the
persisted run events offline. No LLM is called by this script; every token is spent
by the Gateway runs it starts.

Modes
  --init-credentials   write a 0600 credentials file for the eval user (random password)
  --setup-user         register the eval user on the Gateway (or verify it exists)
  --pilot              3 pilot runs, per-run token cost, projected pass cost, and the
                       fixed run list both passes must use
  --pass baseline|post execute the fixed run list (Pro x3 repeats + Ultra runs)
  --score DIR          offline scoring of a pass directory (markdown + JSON);
                       --baseline DIR adds paired per-prompt deltas
  --dry-run            validate prompts, fixtures, configs and print the plan; no
                       Gateway contact

Run with the backend venv (httpx + PyYAML):
  backend/.venv/bin/python scripts/qf_eval.py --help

The integrator guide is Audit/quantflow-tool-awareness-impl-2026-11-18-F-eval-harness.md.
Secrets: the credentials file is read but never printed; cookies and CSRF tokens are
kept in memory only.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import json
import os
import re
import secrets
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = REPO / "config" / "eval" / "prompts.yaml"
DEFAULT_SPARK_TEMPLATE = REPO / "config" / "eval" / "spark.eval.yaml"
FIXTURES_DIR = REPO / "scripts" / "fixtures" / "eval"
# jsonl run-event store: JsonlRunEventStore() defaults to the relative ".deer-flow",
# and serve.sh starts uvicorn from backend/, so events land under backend/.deer-flow.
DEFAULT_EVENTS_ROOT = REPO / "backend" / ".deer-flow" / "threads"
DEFAULT_USERS_ROOT = REPO / "backend" / ".deer-flow" / "users"
CONFIG_YAML = REPO / "config.yaml"
EXTENSIONS_JSON = REPO / "extensions_config.json"
SPARK_EVAL_DIR = Path("~/.spark/eval").expanduser()
RENDERED_SPARK_CONFIG = SPARK_EVAL_DIR / "spark.eval.yaml"
LIVE_SPARK_DB = Path("~/.spark/spark.db").expanduser()
AE_ROOT = Path("/home/fire/Documents/alpha_engine")
AE_DATA = AE_ROOT / "data"
AE_CLI = AE_ROOT / ".venv" / "bin" / "alpha-engine"
DEFAULT_REPORTS_DIR = Path(os.environ.get("REPORTFORGE_REPORTS_DIR", "~/Documents/report-forge/reports")).expanduser()
DEFAULT_RESULTS_ROOT = Path("~/quantflow-tool-awareness-eval").expanduser()
DEFAULT_CREDENTIALS = Path(os.environ.get("QF_EVAL_CREDENTIALS_FILE", "~/.config/quantflow-eval/credentials.json")).expanduser()
DEFAULT_GATEWAY = os.environ.get("QF_EVAL_GATEWAY", "http://127.0.0.1:8001")
EVAL_EMAIL = "qf-eval@example.com"

CATEGORIES = ("stock_compare", "factor_idea", "backtest_idea", "news_sentiment", "filing_read", "portfolio_construction", "risk_attribution", "report_request", "pit_historical", "control")
NLP_CATEGORIES = ("news_sentiment", "filing_read")
IDEA_CATEGORIES = ("factor_idea", "backtest_idea")
FAMILIES = ("alpha_engine", "kizonlp", "spark", "reportforge", "exa", "web", "built-in", "task")
DOMAIN_FAMILIES = ("alpha_engine", "kizonlp", "spark")
# Names that must never appear in a prompt (servers, skills, subagents, tool prefixes).
FORBIDDEN_PROMPT_WORDS = ("alpha_engine", "alpha engine", "alpha-engine", "kizonlp", "kizo", "spark", "reportforge", "report-forge", "quant-desk", "quant desk", "exa", "mcp", "subagent", "quant-analyst", "news-analyst", "run_code")
TERMINAL_STATUSES = {"success", "error", "timeout", "interrupted"}
SNAPSHOT_RE = re.compile(r"\b\d{8}T\d{6}Z-[0-9a-f]{8}\b")
INVALID_TOOL_RE = re.compile(r"Error: (\S+) is not a valid tool")
GEMINI_429_RE = re.compile(r"(?i)(\b429\b|resource[_ ]?exhausted|quota|rate.?limit)")
MCP_FAILURE_RE = re.compile(r"(?i)mcp[^\n]{0,60}(session|server|client)[^\n]{0,60}(fail|timed? ?out|unavailable|closed|not (initiali[sz]ed|available))")
DEFAULT_WARN_TOKENS = 60_000_000
DEFAULT_BUDGET_TOKENS = 80_000_000
FLAG_KEYS = ("mode", "is_plan_mode", "subagent_enabled", "reasoning_effort")


class EvalError(RuntimeError):
    """Abort with a clear message (no traceback)."""


# --------------------------------------------------------------------------- utils


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def to_text(value: Any) -> str:
    """Flatten message content (str, list of blocks, dict) to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(to_text(v) for v in value)
    if isinstance(value, dict):
        for key in ("text", "content"):
            if key in value:
                return to_text(value[key])
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def args_text(args: Any) -> str:
    if isinstance(args, str):
        return args
    return json.dumps(args, ensure_ascii=False, default=str)


def name_matches(name: str, patterns: list[str] | tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def family_of(name: str) -> str:
    if name.startswith("alpha_engine_"):
        return "alpha_engine"
    if name.startswith("kizonlp_"):
        return "kizonlp"
    if name.startswith("spark_"):
        return "spark"
    if name.startswith("reportforge_"):
        return "reportforge"
    if name in ("web_search_advanced_exa", "web_fetch_exa") or name.endswith("_exa"):
        return "exa"
    if name in ("web_search", "web_fetch", "image_search"):
        return "web"
    if name == "task":
        return "task"
    return "built-in"


def fmt_tokens(n: float | int | None) -> str:
    if n is None:
        return "n/a"
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{n:.0f}"


def pct(num: float, den: float) -> str:
    return "n/a" if not den else f"{100.0 * num / den:.1f}%"


def spread(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {"mean": statistics.fmean(values), "min": min(values), "max": max(values), "n": len(values)}


# --------------------------------------------------------------------------- prompts


@dataclass
class PromptSet:
    path: Path
    raw: dict[str, Any]
    sha256: str

    @property
    def prompts(self) -> list[dict[str, Any]]:
        return self.raw["prompts"]

    @property
    def defaults(self) -> dict[str, Any]:
        return self.raw.get("defaults", {})

    def by_id(self) -> dict[str, dict[str, Any]]:
        return {p["id"]: p for p in self.prompts}

    def status_tools(self) -> list[str]:
        return list(self.raw.get("status_tools", []))


def registration_required(prompt: dict[str, Any]) -> bool:
    value = prompt.get("registration_required", False)
    if isinstance(value, str):
        return value.strip().lower() in ("yes", "true", "1")
    return bool(value)


def load_prompts(path: Path) -> PromptSet:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PromptSet(path=path, raw=raw, sha256=sha256_file(path) or "")


def validate_prompts(ps: PromptSet) -> list[str]:
    """Return a list of problems (empty = valid)."""
    problems: list[str] = []
    prompts = ps.prompts
    ids = [p.get("id") for p in prompts]
    if len(set(ids)) != len(ids):
        problems.append("duplicate prompt ids")
    if len(prompts) != 20:
        problems.append(f"expected 20 prompts, found {len(prompts)}")
    per_cat = Counter(p.get("category") for p in prompts)
    for cat in CATEGORIES:
        if per_cat.get(cat, 0) != 2:
            problems.append(f"category {cat}: expected 2 prompts, found {per_cat.get(cat, 0)}")
    for cat in per_cat:
        if cat not in CATEGORIES:
            problems.append(f"unknown category {cat!r}")
    status = set(ps.status_tools())
    for p in prompts:
        pid = p.get("id")
        if p.get("mode") not in ("pro", "full"):
            problems.append(f"{pid}: mode must be pro or full (Ultra runs are listed under ultra_runs)")
        if p.get("category") == "report_request" and p.get("mode") != "full":
            problems.append(f"{pid}: report prompts run in full mode (F12)")
        text = str(p.get("text", ""))
        for word in FORBIDDEN_PROMPT_WORDS:
            if re.search(rf"(?i)(?<![a-z0-9_]){re.escape(word)}(?![a-z0-9_])", text):
                problems.append(f"{pid}: prompt text names a tool/server/skill ({word!r})")
        for group in p.get("required_tools", []):
            if not isinstance(group, list) or not group:
                problems.append(f"{pid}: each required_tools entry must be a non-empty list")
                continue
            for name in group:
                if name in status:
                    problems.append(f"{pid}: status tool {name} cannot satisfy a requirement")
                if re.search(rf"(?i)\b{re.escape(name)}\b", text):
                    problems.append(f"{pid}: prompt text names tool {name}")
        if p.get("category") in IDEA_CATEGORIES and "registration_required" not in p:
            problems.append(f"{pid}: idea prompts need registration_required: yes|no")
        if p.get("category") == "control" and not p.get("forbidden_tools"):
            problems.append(f"{pid}: control prompts need forbidden_tools")
        if p.get("upload") and not (FIXTURES_DIR / p["upload"]).is_file():
            problems.append(f"{pid}: upload fixture missing: {FIXTURES_DIR / p['upload']}")
        for ap in p.get("anti_patterns", []) or []:
            if ap.get("args_regex"):
                try:
                    re.compile(ap["args_regex"])
                except re.error as exc:
                    problems.append(f"{pid}: anti_pattern {ap.get('id')} bad regex: {exc}")
    known = set(ids)
    for phase, items in (ps.raw.get("ultra_runs") or {}).items():
        for pid in items:
            if pid not in known:
                problems.append(f"ultra_runs.{phase}: unknown prompt {pid}")
    for item in ps.raw.get("pilot_runs", []):
        if item.get("prompt") not in known:
            problems.append(f"pilot_runs: unknown prompt {item.get('prompt')}")
    return problems


# --------------------------------------------------------------------------- run list


def cap_for(ps: PromptSet, prompt: dict[str, Any], mode: str, phase: str | None = None) -> int | None:
    caps = ps.defaults.get("tool_call_cap", {})
    if mode == "ultra":
        return caps.get("ultra_post" if phase == "post" else "ultra_baseline")
    if "tool_call_cap" in prompt:
        return prompt["tool_call_cap"]
    return caps.get(mode)


def timeout_for(ps: PromptSet, mode: str) -> int:
    return int(ps.defaults.get("timeout_s", {}).get(mode, 1800))


def make_spec(ps: PromptSet, prompt: dict[str, Any], *, run_key: str, mode: str, repeat: int, phase: str | None = None) -> dict[str, Any]:
    return {"run_key": run_key, "prompt_id": prompt["id"], "category": prompt["category"], "mode": mode, "repeat": repeat, "cap": cap_for(ps, prompt, mode, phase), "timeout_s": timeout_for(ps, mode)}


def build_run_list(ps: PromptSet, pilot: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fixed, ordered run list (repeat-major). Both passes must use the same file (F5b)."""
    repeats = int(ps.defaults.get("repeats", 3))
    pro_runs = []
    for rep in range(1, repeats + 1):
        for prompt in ps.prompts:
            pro_runs.append(make_spec(ps, prompt, run_key=f"r{rep}-{prompt['id']}", mode=prompt["mode"], repeat=rep))
    by_id = ps.by_id()
    ultra: dict[str, list[dict[str, Any]]] = {}
    for phase in ("baseline", "post"):
        items = (ps.raw.get("ultra_runs") or {}).get(phase, [])
        ultra[phase] = [make_spec(ps, by_id[pid], run_key=f"u{i}-{pid}-ultra", mode="ultra", repeat=1, phase=phase) for i, pid in enumerate(items, 1)]
    return {"version": 1, "created_at": now_iso(), "prompts_path": str(ps.path), "prompts_sha256": ps.sha256, "pro_runs": pro_runs, "ultra_runs": ultra, "pilot": pilot}


def pilot_specs(ps: PromptSet) -> list[dict[str, Any]]:
    by_id = ps.by_id()
    specs = []
    for i, item in enumerate(ps.raw.get("pilot_runs", []), 1):
        mode = item.get("mode", "pro")
        specs.append(make_spec(ps, by_id[item["prompt"]], run_key=f"pilot{i}-{item['prompt']}-{mode}", mode=mode, repeat=0, phase="baseline"))
    return specs


def project_cost(run_list: dict[str, Any], per_mode_tokens: dict[str, float]) -> dict[str, Any]:
    n_pro = sum(1 for r in run_list["pro_runs"] if r["mode"] == "pro")
    n_full = sum(1 for r in run_list["pro_runs"] if r["mode"] == "full")
    out: dict[str, Any] = {"n_pro_capped": n_pro, "n_full": n_full, "per_run_tokens": per_mode_tokens}
    for phase in ("baseline", "post"):
        n_ultra = len(run_list["ultra_runs"].get(phase, []))
        total = n_pro * per_mode_tokens.get("pro", 0) + n_full * per_mode_tokens.get("full", 0) + n_ultra * per_mode_tokens.get("ultra", 0)
        out[phase] = {"n_ultra": n_ultra, "projected_total_tokens": total}
    out["note"] = "Post-change Ultra runs are uncapped; the pilot Ultra run is capped, so the post projection is a lower bound."
    return out


# --------------------------------------------------------------------------- live checks


def _proc_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def _proc_ppid(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat.rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def _all_pids() -> list[int]:
    return [int(p.name) for p in Path("/proc").iterdir() if p.name.isdigit()]


def gateway_processes() -> list[tuple[int, str]]:
    out = []
    for pid in _all_pids():
        cmd = _proc_cmdline(pid)
        if "uvicorn" in cmd and "app.gateway.app:app" in cmd:
            out.append((pid, cmd))
    return out


def kizonlp_processes() -> dict[str, int]:
    """Count kizonlp-mcp processes: all, and those descending from the Gateway."""
    gw = {pid for pid, _ in gateway_processes()}
    total = under_gw = 0
    for pid in _all_pids():
        cmd = _proc_cmdline(pid)
        if "kizonlp-mcp" not in cmd or "qf_eval" in cmd:
            continue
        total += 1
        cur: int | None = pid
        for _ in range(12):
            cur = _proc_ppid(cur) if cur else None
            if cur is None or cur <= 1:
                break
            if cur in gw:
                under_gw += 1
                break
    return {"all": total, "gateway": under_gw}


def tree_fingerprint() -> dict[str, Any]:
    """Record what the agent is served from, to detect edits during a pass (F1)."""

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    files = {
        "config.yaml": CONFIG_YAML,
        "extensions_config.json": EXTENSIONS_JSON,
        "skills/custom/quant-desk/SKILL.md": REPO / "skills" / "custom" / "quant-desk" / "SKILL.md",
        "skills/public/quant-desk/SKILL.md": REPO / "skills" / "public" / "quant-desk" / "SKILL.md",
        "backend/.deer-flow/SOUL.md": REPO / "backend" / ".deer-flow" / "SOUL.md",
        "config/quant_tools.yaml": REPO / "config" / "quant_tools.yaml",
    }
    status = git("status", "--porcelain")
    return {"git_head": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"), "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(), "files_sha256": {k: sha256_file(v) for k, v in files.items()}}


def check_configs(snapshot_id: str | None, *, allow_live_spark: bool, allow_memory_injection: bool) -> tuple[list[str], list[str], dict[str, Any]]:
    """Read only the isolation-relevant keys. Never prints other values."""
    errors: list[str] = []
    warnings: list[str] = []
    facts: dict[str, Any] = {}
    try:
        cfg = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        cfg = {}
        errors.append(f"cannot read config.yaml: {exc}")
    backend = (cfg.get("run_events") or {}).get("backend")
    facts["run_events.backend"] = backend
    if backend != "jsonl":
        errors.append(f"config.yaml run_events.backend is {backend!r}; the scorer needs 'jsonl' (package B)")
    mem = cfg.get("memory") or {}
    facts["memory.enabled"] = mem.get("enabled")
    facts["memory.injection_enabled"] = mem.get("injection_enabled")
    if mem.get("injection_enabled") is not False:
        (warnings if allow_memory_injection else errors).append("config.yaml memory.injection_enabled is not false (set it false for the pass, F3b)")
    try:
        ext = json.loads(EXTENSIONS_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        ext = {}
        errors.append(f"cannot read extensions_config.json: {exc}")
    servers = ext.get("mcpServers") or {}
    spark_env = (servers.get("spark") or {}).get("env") or {}
    ae_env = (servers.get("alpha_engine") or {}).get("env") or {}
    spark_cfg = spark_env.get("SPARK_CONFIG")
    facts["spark.env.SPARK_CONFIG"] = spark_cfg
    if not spark_cfg or Path(spark_cfg).expanduser() != RENDERED_SPARK_CONFIG:
        msg = f"extensions_config.json spark env SPARK_CONFIG is {spark_cfg!r}; expected {RENDERED_SPARK_CONFIG} during a pass (F3a)"
        (warnings if allow_live_spark else errors).append(msg)
    pinned = ae_env.get("ALPHA_ENGINE_SNAPSHOT_ID")
    facts["alpha_engine.env.ALPHA_ENGINE_SNAPSHOT_ID"] = pinned
    facts["alpha_engine.env.ALPHA_ENGINE_PROFILE"] = ae_env.get("ALPHA_ENGINE_PROFILE")
    if snapshot_id and pinned != snapshot_id:
        errors.append(f"alpha_engine env ALPHA_ENGINE_SNAPSHOT_ID is {pinned!r}; expected {snapshot_id} (F22)")
    if not snapshot_id:
        warnings.append("no --snapshot-id given: alpha_engine data may change across the 18:00 refresh (F22)")
    facts["mcpServers"] = sorted(servers)
    facts["kizonlp.translate_virtual_paths"] = (servers.get("kizonlp") or {}).get("translate_virtual_paths")
    return errors, warnings, facts


def check_snapshot_lease(snapshot_id: str) -> dict[str, Any]:
    snap_dir = AE_DATA / "snapshots" / snapshot_id
    info: dict[str, Any] = {"snapshot_id": snapshot_id, "exists": snap_dir.is_dir(), "valid_leases": []}
    leases_dir = snap_dir / "leases"
    now = time.time()
    if leases_dir.is_dir():
        for p in sorted(leases_dir.glob("*.json")):
            try:
                rec = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            pid = rec.get("pid")
            alive = pid is None or Path(f"/proc/{pid}").exists()
            if float(rec.get("expires_at", 0)) > now and alive:
                info["valid_leases"].append({"file": p.name, "expires_at": rec.get("expires_at"), "note": rec.get("note")})
    return info


def create_snapshot_lease(snapshot_id: str, note: str, ttl: str = "21d") -> None:
    if not AE_CLI.exists():
        raise EvalError(f"alpha-engine CLI not found at {AE_CLI}")
    res = subprocess.run([str(AE_CLI), "snapshot", "pin", snapshot_id, "--ttl", ttl, "--note", note], capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        raise EvalError(f"alpha-engine snapshot pin failed: {res.stdout.strip()[-400:]} {res.stderr.strip()[-400:]}")


def render_spark_config(template: Path, snapshot_id: str | None) -> dict[str, Any]:
    """Render the eval spark config to the fixed path SPARK_CONFIG points at."""
    cfg = yaml.safe_load(template.read_text(encoding="utf-8"))
    db_path = Path(os.path.expanduser(cfg["storage"]["db_path"])).resolve()
    if db_path == LIVE_SPARK_DB.resolve() or SPARK_EVAL_DIR.resolve() not in db_path.parents:
        raise EvalError(f"spark eval db_path {db_path} must live under {SPARK_EVAL_DIR} and not be the live ledger")
    if (cfg.get("llm") or {}).get("degraded_mode") != "never-llm":
        raise EvalError("spark eval config must set llm.degraded_mode: never-llm (decision 8)")
    peer = cfg["peers"]["alpha_engine"]
    if snapshot_id:
        python = peer["command"]
        peer["command"] = "/usr/bin/env"
        peer["args"] = [f"ALPHA_ENGINE_SNAPSHOT_ID={snapshot_id}", python, *peer.get("args", [])]
    SPARK_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    header = f"# RENDERED by scripts/qf_eval.py at {now_iso()} from {template}. Do not edit; re-render instead.\n"
    RENDERED_SPARK_CONFIG.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return {"rendered": str(RENDERED_SPARK_CONFIG), "db_path": str(db_path), "artifacts_dir": os.path.expanduser(cfg["storage"]["artifacts_dir"])}


def reset_spark_scratch(db_path: Path) -> None:
    db_path = db_path.resolve()
    if db_path == LIVE_SPARK_DB.resolve() or SPARK_EVAL_DIR.resolve() not in db_path.parents:
        raise EvalError(f"refusing to reset {db_path}: not a scratch spark DB")
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)


# --------------------------------------------------------------------------- gateway client


def iter_sse(lines):
    """Yield (event, data) from SSE lines."""
    event, data_lines = None, []
    for raw in lines:
        line = raw.rstrip("\r")
        if line == "":
            if event is not None or data_lines:
                payload = "\n".join(data_lines)
                try:
                    data = json.loads(payload) if payload else None
                except json.JSONDecodeError:
                    data = payload
                yield event or "message", data
            event, data_lines = None, []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())


class LiveCounter:
    """Counts distinct tool calls seen on the stream (lead `updates` + subagent `custom`)."""

    def __init__(self, exempt: list[str], skill_tools: list[str], skill_re: re.Pattern[str]):
        self.seen: dict[str, str] = {}
        self.substantive = 0
        self.exempt = exempt
        self.skill_tools = skill_tools
        self.skill_re = skill_re

    def _is_exempt(self, name: str, args: Any) -> bool:
        if name in self.exempt:
            return True
        if name in self.skill_tools and isinstance(args, dict) and self.skill_re.search(str(args.get("path", ""))):
            return True
        return False

    def feed(self, data: Any) -> None:
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                calls = node.get("tool_calls")
                if isinstance(calls, list) and node.get("type") in (None, "ai", "AIMessage", "AIMessageChunk"):
                    for i, tc in enumerate(calls):
                        if not isinstance(tc, dict) or not tc.get("name"):
                            continue
                        key = tc.get("id") or f"{node.get('id')}:{i}:{tc.get('name')}"
                        if key in self.seen:
                            continue
                        self.seen[key] = tc["name"]
                        if not self._is_exempt(tc["name"], tc.get("args")):
                            self.substantive += 1
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)


class Gateway:
    def __init__(self, base_url: str, credentials: Path):
        import httpx

        self.httpx = httpx
        self.base = base_url.rstrip("/")
        self.credentials = credentials
        self.client = httpx.Client(base_url=self.base, timeout=httpx.Timeout(60.0))
        self.cookies: dict[str, str] = {}
        self.user: dict[str, Any] = {}

    # -- auth -------------------------------------------------------------
    def _creds(self) -> dict[str, str]:
        if not self.credentials.is_file():
            raise EvalError(f"credentials file not found: {self.credentials} (run --init-credentials)")
        mode = self.credentials.stat().st_mode & 0o077
        if mode:
            raise EvalError(f"credentials file {self.credentials} must be mode 0600")
        data = json.loads(self.credentials.read_text(encoding="utf-8"))
        if not data.get("email") or not data.get("password"):
            raise EvalError("credentials file needs 'email' and 'password'")
        return data

    def _absorb_cookies(self, resp) -> None:
        for name in ("access_token", "csrf_token", "deerflow_session_persistent"):
            value = resp.cookies.get(name)
            if value:
                self.cookies[name] = value

    def headers(self, write: bool = False) -> dict[str, str]:
        h = {"Cookie": "; ".join(f"{k}={v}" for k, v in self.cookies.items())}
        if write and "csrf_token" in self.cookies:
            h["X-CSRF-Token"] = self.cookies["csrf_token"]
        return h

    def login(self) -> dict[str, Any]:
        creds = self._creds()
        resp = self.client.post("/api/v1/auth/login/local", data={"username": creds["email"], "password": creds["password"]})
        if resp.status_code != 200:
            raise EvalError(f"eval-user login failed: HTTP {resp.status_code}")
        self._absorb_cookies(resp)
        if "access_token" not in self.cookies or "csrf_token" not in self.cookies:
            raise EvalError("login did not return session + CSRF cookies")
        me = self.client.get("/api/v1/auth/me", headers=self.headers())
        if me.status_code != 200:
            raise EvalError(f"/auth/me failed: HTTP {me.status_code}")
        self.user = me.json()
        if self.user.get("email") != creds["email"]:
            raise EvalError("logged-in user does not match the credentials file")
        return self.user

    def register(self) -> str:
        creds = self._creds()
        resp = self.client.post("/api/v1/auth/register", json={"email": creds["email"], "password": creds["password"], "remember_me": False})
        if resp.status_code == 201:
            return "created"
        if resp.status_code == 400:
            return "exists"
        raise EvalError(f"register failed: HTTP {resp.status_code} {resp.text[:200]}")

    # -- threads ----------------------------------------------------------
    def create_thread(self, title: str, metadata: dict[str, Any]) -> str:
        tid = str(uuid.uuid4())
        resp = self.client.post("/api/threads", json={"thread_id": tid, "metadata": metadata}, headers=self.headers(True))
        if resp.status_code not in (200, 201):
            raise EvalError(f"create thread failed: HTTP {resp.status_code} {resp.text[:200]}")
        # Title via state (as the UI rename does). A titled thread also skips the
        # title-generation LLM call, the same in both passes.
        st = self.client.post(f"/api/threads/{tid}/state", json={"values": {"title": title}}, headers=self.headers(True))
        if st.status_code >= 400:
            self.client.patch(f"/api/threads/{tid}", json={"metadata": {"title": title}}, headers=self.headers(True))
        return tid

    def upload(self, tid: str, path: Path) -> list[dict[str, Any]]:
        with path.open("rb") as fh:
            resp = self.client.post(f"/api/threads/{tid}/uploads", files=[("files", (path.name, fh, "application/pdf"))], headers=self.headers(True), timeout=self.httpx.Timeout(300.0))
        if resp.status_code >= 400:
            raise EvalError(f"upload failed: HTTP {resp.status_code} {resp.text[:200]}")
        files = resp.json().get("files", [])
        return [{"filename": f["filename"], "size": f["size"], "path": f["virtual_path"], "status": "uploaded"} for f in files]

    def stream_run(self, tid: str, body: dict[str, Any], counter: LiveCounter, cap: int | None, ceiling: int, deadline: float) -> tuple[str | None, str]:
        """Stream until end, cap or deadline. Returns (run_id, stop_reason)."""
        run_id: str | None = None
        url = f"/api/threads/{tid}/runs/stream"
        method, payload = "POST", body
        limit = cap if cap is not None else ceiling
        while True:
            try:
                with self.client.stream(method, url, json=payload, headers=self.headers(method == "POST"), timeout=self.httpx.Timeout(60.0, read=120.0)) as resp:
                    if resp.status_code >= 400:
                        resp.read()
                        raise EvalError(f"run stream failed: HTTP {resp.status_code} {resp.text[:300]}")
                    loc = resp.headers.get("content-location")
                    if loc and not run_id:
                        run_id = loc.rstrip("/").rsplit("/", 1)[-1]
                    for event, data in iter_sse(resp.iter_lines()):
                        if event == "metadata" and isinstance(data, dict) and not run_id:
                            run_id = data.get("run_id")
                        counter.feed(data)
                        if event == "end":
                            return run_id, "completed"
                        if counter.substantive > limit:
                            return (run_id, "cap" if cap is not None else "safety_ceiling")
                        if time.time() > deadline:
                            return run_id, "timeout"
                    return run_id, "stream_closed"
            except self.httpx.TransportError:
                if time.time() > deadline:
                    return run_id, "timeout"
                if not run_id:
                    raise
                # Re-attach to the same run (on_disconnect=continue keeps it alive).
                method, url, payload = ("GET", f"/api/threads/{tid}/runs/{run_id}/join", None)
                if self.run_status(tid, run_id) in TERMINAL_STATUSES:
                    return run_id, "completed"

    def cancel(self, tid: str, run_id: str) -> int:
        resp = self.client.post(f"/api/threads/{tid}/runs/{run_id}/cancel", params={"wait": "true", "action": "interrupt"}, headers=self.headers(True), timeout=self.httpx.Timeout(180.0))
        return resp.status_code

    def get_run(self, tid: str, run_id: str) -> dict[str, Any]:
        resp = self.client.get(f"/api/threads/{tid}/runs/{run_id}", headers=self.headers())
        return resp.json() if resp.status_code == 200 else {"error": f"HTTP {resp.status_code}"}

    def run_status(self, tid: str, run_id: str) -> str | None:
        return self.get_run(tid, run_id).get("status")

    def latest_run_id(self, tid: str) -> str | None:
        resp = self.client.get(f"/api/threads/{tid}/runs", headers=self.headers())
        if resp.status_code == 200 and resp.json():
            return resp.json()[0].get("run_id")
        return None

    def delete_thread(self, tid: str) -> int:
        for _ in range(10):
            resp = self.client.delete(f"/api/threads/{tid}", headers=self.headers(True), timeout=self.httpx.Timeout(180.0))
            if resp.status_code != 409:
                return resp.status_code
            time.sleep(6)
        return 409


# --------------------------------------------------------------------------- runner


def build_context(ps: PromptSet, mode: str, tid: str) -> dict[str, Any]:
    """Mirror frontend/src/core/threads/hooks.ts buildRunContext for Pro/Ultra."""
    d = ps.defaults
    ui_mode = "ultra" if mode == "ultra" else "pro"
    return {
        "model_name": d.get("model_name"),
        "mode": ui_mode,
        "thinking_enabled": True,
        "is_plan_mode": True,
        "subagent_enabled": ui_mode == "ultra",
        "reasoning_effort": "high" if ui_mode == "ultra" else "medium",
        "thread_id": tid,
        "locale": d.get("locale", "en-US"),
        "client_timezone": d.get("client_timezone", "America/Chicago"),
    }


@dataclass
class PassState:
    pass_name: str
    pass_dir: Path
    ps: PromptSet
    gateway: Gateway
    spark_db: Path | None
    events_root: Path
    users_root: Path
    reports_dir: Path | None
    warn_tokens: int
    budget_tokens: int
    hard_stop: bool
    snapshot_id: str | None
    cumulative_tokens: int = 0
    warned: bool = False
    log: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        line = f"[{now_iso()}] {msg}"
        print(line, flush=True)
        with (self.pass_dir / "runner.log").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def wait_kizonlp(state: PassState, limit: int = 2, wait_s: int = 45) -> dict[str, int]:
    counts = kizonlp_processes()
    t0 = time.time()
    while counts["gateway"] > limit and time.time() - t0 < wait_s:
        time.sleep(3)
        counts = kizonlp_processes()
    if counts["gateway"] > limit:
        raise EvalError(f"{counts['gateway']} kizonlp-mcp processes alive under the Gateway (> {limit}); aborting the pass (F4)")
    if counts["all"] > limit:
        state.note(f"warning: {counts['all']} kizonlp-mcp processes on the host ({counts['gateway']} under the Gateway)")
    return counts


def wait_terminal_event(path: Path, timeout_s: int = 90) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if path.is_file() and '"run.terminal"' in path.read_text(encoding="utf-8", errors="replace"):
            return True
        time.sleep(2)
    return False


def execute_run(state: PassState, spec: dict[str, Any]) -> dict[str, Any]:
    ps, gw = state.ps, state.gateway
    prompt = ps.by_id()[spec["prompt_id"]]
    run_dir = state.pass_dir / "runs" / spec["run_key"]
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "events").mkdir(parents=True)
    meta: dict[str, Any] = {**spec, "pass": state.pass_name, "started_at": now_iso(), "snapshot_id_pinned": state.snapshot_id}
    meta["kizonlp_before"] = wait_kizonlp(state)
    if state.spark_db:
        reset_spark_scratch(state.spark_db)
    reports_before = {p.name for p in state.reports_dir.iterdir()} if state.reports_dir and state.reports_dir.is_dir() else set()
    title = f"eval-{state.pass_name}-{spec['run_key']}"
    tid = gw.create_thread(title, {"qf_eval_pass": state.pass_name, "qf_eval_run": spec["run_key"]})
    meta["thread_id"] = tid
    meta["title"] = title
    files: list[dict[str, Any]] = []
    if prompt.get("upload"):
        files = gw.upload(tid, FIXTURES_DIR / prompt["upload"])
        meta["upload"] = files
    human: dict[str, Any] = {"type": "human", "id": str(uuid.uuid4()), "content": [{"type": "text", "text": prompt["text"].strip()}], "additional_kwargs": {"files": files} if files else {}}
    body = {
        "assistant_id": "lead_agent",
        "input": {"messages": [human]},
        "config": {"recursion_limit": 1000},
        "context": build_context(ps, spec["mode"], tid),
        "stream_mode": ["updates", "custom"],
        "on_disconnect": "continue",
        "multitask_strategy": "reject",
    }
    meta["context"] = body["context"]
    counter = LiveCounter(list(ps.raw.get("cap_exempt_tools", [])), list(ps.raw.get("skill_read_tools", [])), re.compile(ps.raw.get("skill_path_regex", r"$^")))
    ceiling = int(ps.defaults.get("safety_tool_call_ceiling", 250))
    deadline = time.time() + spec["timeout_s"]
    run_id: str | None = None
    try:
        run_id, reason = gw.stream_run(tid, body, counter, spec["cap"], ceiling, deadline)
    except Exception as exc:  # noqa: BLE001 - any transport failure ends this run, not the pass
        reason = f"stream_error: {type(exc).__name__}: {str(exc)[:200]}"
    run_id = run_id or gw.latest_run_id(tid)
    meta["run_id"] = run_id
    meta["stop_reason"] = reason
    meta["live_tool_calls_substantive"] = counter.substantive
    meta["live_tool_calls_total"] = len(counter.seen)
    if run_id and reason in ("cap", "safety_ceiling", "timeout"):
        meta["cancel_http_status"] = gw.cancel(tid, run_id)
    if run_id:
        t0 = time.time()
        while time.time() - t0 < 300 and gw.run_status(tid, run_id) not in TERMINAL_STATUSES:
            time.sleep(3)
        if gw.run_status(tid, run_id) not in TERMINAL_STATUSES:
            meta["cancel_http_status_late"] = gw.cancel(tid, run_id)
            time.sleep(10)
        meta["gateway_run"] = {k: v for k, v in gw.get_run(tid, run_id).items() if k not in ("kwargs",)}
        if meta["gateway_run"].get("status") == "error" and reason == "completed":
            # The stream ends normally even when the run itself failed (e.g. the model
            # endpoint returned 4xx); record that instead of a clean completion.
            meta["stop_reason"] = reason = "run_error"
        meta["terminal_event_seen"] = wait_terminal_event(state.events_root / tid / "runs" / f"{run_id}.jsonl")
    # Copy events, then outputs, then delete the thread (closes its MCP sessions, F4).
    src_events = state.events_root / tid / "runs"
    meta["events_files"] = []
    if src_events.is_dir():
        for f in sorted(src_events.glob("*.jsonl")):
            shutil.copy2(f, run_dir / "events" / f.name)
            meta["events_files"].append(f.name)
    if not meta["events_files"]:
        state.note(f"warning: no run-event jsonl for {spec['run_key']} under {src_events} (is run_events.backend jsonl?)")
    uid = gw.user.get("id")
    outputs = state.users_root / str(uid) / "threads" / tid / "user-data" / "outputs"
    if outputs.is_dir():
        size = sum(p.stat().st_size for p in outputs.rglob("*") if p.is_file())
        if size < 200 * 1024 * 1024:
            shutil.copytree(outputs, run_dir / "outputs")
        meta["outputs"] = sorted(str(p.relative_to(outputs)) for p in outputs.rglob("*") if p.is_file())
    meta["delete_http_status"] = gw.delete_thread(tid)
    if meta["delete_http_status"] not in (200, 204):
        state.note(f"warning: thread delete returned {meta['delete_http_status']} for {tid}")
    if state.reports_dir and state.reports_dir.is_dir():
        new = sorted({p.name for p in state.reports_dir.iterdir()} - reports_before)
        moved = []
        for name in new:
            dest = run_dir / "reportforge_projects"
            dest.mkdir(exist_ok=True)
            shutil.move(str(state.reports_dir / name), str(dest / name))
            moved.append(name)
        meta["reportforge_projects_moved"] = moved
    tokens = int((meta.get("gateway_run") or {}).get("total_tokens") or 0)
    if not tokens:
        tokens = int(parse_run_dir(run_dir).usage.get("total_tokens", 0))
    meta["total_tokens"] = tokens
    meta["ended_at"] = now_iso()
    meta["kizonlp_after"] = kizonlp_processes()
    write_json(run_dir / "meta.json", meta)
    return meta


def run_specs(state: PassState, specs: list[dict[str, Any]], resume: bool) -> None:
    progress = state.pass_dir / "progress.jsonl"
    done: dict[str, int] = {}
    if progress.is_file():
        for line in progress.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["run_key"]] = int(rec.get("total_tokens") or 0)
    if done and not resume:
        raise EvalError(f"{state.pass_dir} already has completed runs; use --resume or a new --pass-dir")
    state.cumulative_tokens = sum(done.values())
    for i, spec in enumerate(specs, 1):
        if spec["run_key"] in done:
            continue
        if state.hard_stop and state.cumulative_tokens >= state.budget_tokens:
            state.note(f"--hard-stop: cumulative {fmt_tokens(state.cumulative_tokens)} >= budget {fmt_tokens(state.budget_tokens)}; stopping (the pass is now INCOMPLETE)")
            break
        state.note(f"run {i}/{len(specs)} {spec['run_key']} ({spec['mode']}, cap={spec['cap']})")
        meta = execute_run(state, spec)
        if meta["stop_reason"] == "run_error" and not meta["total_tokens"]:
            # Zero tokens means the model never answered (auth, quota, endpoint down).
            # Abort before recording progress so --resume re-runs this key.
            raise EvalError(
                f"{spec['run_key']} ended in error with 0 tokens (LLM/infra failure, see logs/gateway.log); "
                "pass aborted, nothing recorded for this run"
            )
        state.cumulative_tokens += meta["total_tokens"]
        with progress.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"run_key": spec["run_key"], "status": (meta.get("gateway_run") or {}).get("status"), "stop_reason": meta["stop_reason"], "total_tokens": meta["total_tokens"], "finished_at": meta["ended_at"]}) + "\n")
        state.note(f"  -> {meta['stop_reason']}, {meta['live_tool_calls_substantive']} calls, {fmt_tokens(meta['total_tokens'])} tokens; cumulative {fmt_tokens(state.cumulative_tokens)}")
        if not state.warned and state.cumulative_tokens >= state.warn_tokens:
            state.warned = True
            state.note(f"WARNING: cumulative tokens {fmt_tokens(state.cumulative_tokens)} passed {fmt_tokens(state.warn_tokens)}; the pass continues (fixed run list, F5)")


@contextlib.contextmanager
def single_runner_lock(root: Path):
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    fh = (root / ".qf_eval.lock").open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise EvalError("another qf_eval.py runner holds the lock; runs are strictly sequential") from exc
    try:
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def live_preflight(args, ps: PromptSet, snapshot_id: str | None) -> dict[str, Any]:
    problems = validate_prompts(ps)
    if problems:
        raise EvalError("prompt set invalid:\n  " + "\n  ".join(problems))
    errors, warnings, facts = check_configs(snapshot_id, allow_live_spark=args.allow_live_spark, allow_memory_injection=args.allow_memory_injection)
    gws = gateway_processes()
    facts["gateway_processes"] = len(gws)
    if any("--reload" in cmd for _, cmd in gws) and not args.allow_reload:
        errors.append("the Gateway runs with --reload; start it without reload (see the harness guide, F21)")
    if not gws:
        warnings.append("no local uvicorn Gateway process found (remote Gateway? kizonlp guard and no-reload check are blind)")
    if snapshot_id:
        lease = check_snapshot_lease(snapshot_id)
        if not lease["exists"]:
            errors.append(f"snapshot {snapshot_id} not found under {AE_DATA / 'snapshots'}")
        elif not lease["valid_leases"]:
            if args.lease:
                create_snapshot_lease(snapshot_id, note=f"qf-eval {args.pass_name or 'pilot'}")
                lease = check_snapshot_lease(snapshot_id)
            else:
                errors.append(f"snapshot {snapshot_id} has no valid lease; pass --lease or run `alpha-engine snapshot pin {snapshot_id} --ttl 21d`")
        facts["snapshot_lease"] = lease
    if errors:
        raise EvalError("preflight failed:\n  " + "\n  ".join(errors))
    for w in warnings:
        print(f"warning: {w}")
    facts["warnings"] = warnings
    return facts


def start_pass(args, pass_name: str, specs: list[dict[str, Any]], extra_manifest: dict[str, Any]) -> PassState:
    ps = load_prompts(args.prompts)
    snapshot_id = args.snapshot_id
    facts = live_preflight(args, ps, snapshot_id)
    pass_dir = args.pass_dir or (args.results_root / pass_name)
    pass_dir.mkdir(parents=True, exist_ok=True)
    spark = None
    if not args.allow_live_spark:
        spark = render_spark_config(args.spark_template, snapshot_id)
        art = Path(spark["artifacts_dir"])
        if art.is_dir() and SPARK_EVAL_DIR.resolve() in art.resolve().parents and not args.resume:
            shutil.rmtree(art)
    gw = Gateway(args.gateway, args.credentials)
    user = gw.login()
    shutil.copy2(args.prompts, pass_dir / "prompts.yaml")
    manifest_path = pass_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    manifest.update(
        {
            "pass": pass_name,
            "started_at": manifest.get("started_at", now_iso()),
            "resumed_at": now_iso() if args.resume else None,
            "prompts_sha256": ps.sha256,
            "snapshot_id": snapshot_id,
            "eval_user": {"id": user.get("id"), "email": user.get("email")},
            "gateway": args.gateway,
            "spark": spark,
            "preflight": facts,
            "tree_at_start": manifest.get("tree_at_start", tree_fingerprint()),
            "n_runs": len(specs),
            **extra_manifest,
        }
    )
    write_json(manifest_path, manifest)
    return PassState(
        pass_name=pass_name,
        pass_dir=pass_dir,
        ps=ps,
        gateway=gw,
        spark_db=Path(spark["db_path"]) if spark else None,
        events_root=args.events_root,
        users_root=args.users_root,
        reports_dir=None if args.no_reportforge_archive else args.reports_dir,
        warn_tokens=args.warn_tokens,
        budget_tokens=args.budget_tokens,
        hard_stop=args.hard_stop,
        snapshot_id=snapshot_id,
    )


def finish_pass(state: PassState) -> None:
    manifest_path = state.pass_dir / "manifest.json"
    manifest = read_json(manifest_path)
    end = tree_fingerprint()
    manifest["tree_at_end"] = end
    manifest["tree_changed_during_pass"] = end != manifest.get("tree_at_start")
    manifest["ended_at"] = now_iso()
    manifest["cumulative_tokens"] = state.cumulative_tokens
    write_json(manifest_path, manifest)
    if manifest["tree_changed_during_pass"]:
        state.note("WARNING: the live tree or a served config changed during the pass; results may not be comparable")


# --------------------------------------------------------------------------- scorer


@dataclass
class Call:
    order: tuple[int, int]
    agent: str
    name: str
    args: Any
    call_id: str
    args_truncated: bool = False


@dataclass
class Result:
    agent: str
    name: str
    text: str
    status: str | None
    call_id: str | None


@dataclass
class Trace:
    calls: list[Call] = field(default_factory=list)
    results: list[Result] = field(default_factory=list)
    llm_errors: list[str] = field(default_factory=list)
    run_errors: list[str] = field(default_factory=list)
    invalid_tool_calls: int = 0
    terminal_status: str | None = None
    usage: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    models: Counter = field(default_factory=Counter)
    subagents: dict[str, str] = field(default_factory=dict)
    n_events: int = 0


def load_events(run_dir: Path) -> list[dict[str, Any]]:
    events = []
    for f in sorted((run_dir / "events").glob("*.jsonl")) if (run_dir / "events").is_dir() else []:
        with f.open("rb") as fh:
            for raw in fh:
                try:
                    events.append(json.loads(raw.decode("utf-8")))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
    events.sort(key=lambda e: e.get("seq", 0))
    return events


def parse_run_dir(run_dir: Path) -> Trace:
    events = load_events(run_dir)
    tr = Trace(n_events=len(events))
    # Pass 1: map task tool-call ids to subagent types.
    for ev in events:
        if ev.get("event_type") == "llm.ai.response" and (ev.get("metadata") or {}).get("caller") == "lead_agent":
            for tc in (ev.get("content") or {}).get("tool_calls") or []:
                if tc.get("name") == "task" and tc.get("id"):
                    tr.subagents[tc["id"]] = str((tc.get("args") or {}).get("subagent_type", "?"))
    seen_ids: set[str] = set()
    seen_usage: set[str] = set()
    subagent_llm_usage = False
    subagent_end_usage: dict[str, int] = defaultdict(int)
    for ev in events:
        et = ev.get("event_type")
        seq = int(ev.get("seq", 0))
        content = ev.get("content")
        meta = ev.get("metadata") or {}
        if et == "llm.ai.response" and isinstance(content, dict):
            caller = meta.get("caller", "lead_agent")
            usage = meta.get("usage") or {}
            ukey = f"{caller}:{content.get('id')}:{meta.get('llm_call_index')}"
            if usage and ukey not in seen_usage:
                seen_usage.add(ukey)
                for k in ("input_tokens", "output_tokens", "total_tokens"):
                    tr.usage[k] += int(usage.get(k) or 0)
                tr.usage["cache_read_tokens"] += int(((usage.get("input_token_details") or {}).get("cache_read")) or 0)
                if caller.startswith("subagent:"):
                    subagent_llm_usage = True
            model = (content.get("response_metadata") or {}).get("model_name")
            if model:
                tr.models[model] += 1
            if caller != "lead_agent":
                continue
            tr.invalid_tool_calls += len(content.get("invalid_tool_calls") or [])
            for i, tc in enumerate(content.get("tool_calls") or []):
                cid = tc.get("id") or f"{seq}:{i}"
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                tr.calls.append(Call((seq, i), "lead", tc.get("name", ""), tc.get("args"), cid))
        elif et == "llm.tool.result" and isinstance(content, dict):
            tr.results.append(Result("lead", content.get("name") or "", to_text(content.get("content")), content.get("status"), content.get("tool_call_id")))
        elif et == "subagent.step" and isinstance(content, dict):
            task_id = content.get("task_id", "")
            agent = f"subagent:{tr.subagents.get(task_id, '?')}"
            if content.get("kind") == "ai":
                for i, tc in enumerate(content.get("tool_calls") or []):
                    cid = f"{task_id}:{content.get('message_index')}:{i}"
                    if cid in seen_ids:
                        continue
                    seen_ids.add(cid)
                    tr.calls.append(Call((seq, i), agent, tc.get("name", ""), tc.get("args"), cid, bool(tc.get("args_truncated"))))
            else:
                tr.results.append(Result(agent, content.get("tool_name") or "", content.get("text") or "", None, None))
        elif et == "subagent.end" and isinstance(content, dict):
            for k, v in (content.get("usage") or {}).items():
                subagent_end_usage[k] += int(v or 0)
        elif et == "llm.error":
            tr.llm_errors.append(to_text(content)[:500])
        elif et == "run.error":
            tr.run_errors.append(to_text(content)[:500])
        elif et == "run.terminal" and isinstance(content, dict):
            tr.terminal_status = content.get("status")
    if not subagent_llm_usage:
        for k in ("input_tokens", "output_tokens", "total_tokens"):
            tr.usage[k] += subagent_end_usage.get(k, 0)
    tr.calls.sort(key=lambda c: c.order)
    return tr


class Scorer:
    def __init__(self, ps: PromptSet, known_tools: set[str] | None = None):
        self.ps = ps
        self.raw = ps.raw
        self.prompts = ps.by_id()
        self.status = set(ps.status_tools())
        self.exempt = list(self.raw.get("cap_exempt_tools", []))
        self.skill_tools = list(self.raw.get("skill_read_tools", []))
        self.skill_re = re.compile(self.raw.get("skill_path_regex", r"$^"))
        self.run_code_tools = list(self.raw.get("run_code_tools", []))
        self.code_tools = list(self.raw.get("code_tools", []))
        self.reg_tools = list(self.raw.get("registration_tools", []))
        self.bt_tools = list(self.raw.get("backtest_tools", []))
        self.global_aps = list(self.raw.get("global_anti_patterns", []))
        self.known = known_tools

    def is_exempt(self, c: Call) -> bool:
        if c.name in self.exempt:
            return True
        if c.name in self.skill_tools:
            path = c.args.get("path", "") if isinstance(c.args, dict) else str(c.args)
            return bool(self.skill_re.search(str(path)))
        return False

    def _ap_hits(self, ap: dict[str, Any], calls: list[Call]) -> list[str]:
        rx = re.compile(ap["args_regex"]) if ap.get("args_regex") else None
        hits = []
        for c in calls:
            if not name_matches(c.name, ap.get("tools", [])):
                continue
            text = args_text(c.args)
            m = rx.search(text) if rx else None
            if rx is None or m:
                start = max(0, (m.start() if m else 0) - 60)
                hits.append(f"{c.agent}/{c.name}: …{text[start : start + 160]}…")
        return hits

    def score_run(self, run_dir: Path) -> dict[str, Any]:
        meta = read_json(run_dir / "meta.json") if (run_dir / "meta.json").is_file() else {}
        tr = parse_run_dir(run_dir)
        prompt = self.prompts.get(meta.get("prompt_id", ""), {})
        calls = list(tr.calls)
        truncated_at_cap = 0
        if meta.get("stop_reason") == "cap" and meta.get("cap"):
            kept, n = [], 0
            for c in calls:
                if not self.is_exempt(c):
                    if n >= meta["cap"]:
                        truncated_at_cap += 1
                        continue
                    n += 1
                kept.append(c)
            calls = kept
        names = [c.name for c in calls]
        substantive = [c for c in calls if not self.is_exempt(c)]
        fam = Counter(family_of(c.name) for c in calls)
        fam_sub = Counter(family_of(c.name) for c in substantive)
        nonstatus = {c.name for c in calls if c.name not in self.status}
        groups = prompt.get("required_tools") or []
        group_ok = [any(t in nonstatus for t in g) for g in groups]
        forbidden_hits = [c.name for c in calls if name_matches(c.name, prompt.get("forbidden_tools") or [])]
        passed = all(group_ok) and not forbidden_hits
        # Errors and infra.
        tool_errors = Counter()
        gemini_429 = 0
        mcp_failures = 0
        nonexistent: list[str] = []
        for r in tr.results:
            txt = r.text or ""
            is_err = r.status == "error" or txt.lstrip().startswith("Error") or '"ok": false' in txt[:400]
            if is_err:
                tool_errors[r.name] += 1
            m = INVALID_TOOL_RE.search(txt[:300])
            if m:
                nonexistent.append(m.group(1))
            if r.name.startswith("kizonlp_") and is_err and GEMINI_429_RE.search(txt[:2000]):
                gemini_429 += 1
            if is_err and MCP_FAILURE_RE.search(txt[:2000]):
                mcp_failures += 1
        if self.known:
            for n in names:
                if family_of(n) in ("alpha_engine", "kizonlp", "spark", "reportforge", "exa") and n not in self.known and n not in nonexistent:
                    nonexistent.append(n)
        llm_infra = len(tr.llm_errors) + len(tr.run_errors)
        status = (meta.get("gateway_run") or {}).get("status") or tr.terminal_status
        infra_total = gemini_429 + mcp_failures + llm_infra + (1 if status in ("error", "timeout") and not calls else 0)
        outcome = "pass" if passed else ("infra" if infra_total else "miss")
        # Anti-patterns.
        aps: dict[str, Any] = {}
        for ap in self.global_aps + list(prompt.get("anti_patterns") or []):
            hits = self._ap_hits(ap, calls)
            aps[ap["id"]] = {"severity": ap.get("severity", "soft"), "count": len(hits), "evidence": hits[:3]}
        # Spark ordering.
        spark_status = None
        if registration_required(prompt):
            reg = next((i for i, c in enumerate(calls) if c.name in self.reg_tools), None)
            bt = next((i for i, c in enumerate(calls) if c.name in self.bt_tools), None)
            if reg is None and bt is None:
                spark_status = "no_register_no_backtest"
            elif reg is None:
                spark_status = "backtest_without_register"
            elif bt is None:
                spark_status = "register_no_backtest"
            else:
                spark_status = "ok" if reg < bt else "backtest_before_register"
        run_code = sum(1 for c in substantive if c.name in self.run_code_tools)
        first_domain = next((i + 1 for i, c in enumerate(substantive) if family_of(c.name) in DOMAIN_FAMILIES and c.name not in self.status), None)
        truncated_code = sum(1 for c in calls if c.args_truncated and name_matches(c.name, self.code_tools))
        snapshots = sorted({s for r in tr.results if r.name.startswith("alpha_engine_") for s in SNAPSHOT_RE.findall(r.text or "")})
        upload_read_file = sum(1 for c in calls if c.name == "read_file" and "/mnt/user-data/uploads" in args_text(c.args))
        gw_run = meta.get("gateway_run") or {}
        tokens = {
            "total_tokens": int(gw_run.get("total_tokens") or tr.usage.get("total_tokens", 0)),
            "input_tokens": int(gw_run.get("total_input_tokens") or tr.usage.get("input_tokens", 0)),
            "output_tokens": int(gw_run.get("total_output_tokens") or tr.usage.get("output_tokens", 0)),
            "subagent_tokens": int(gw_run.get("subagent_tokens") or 0),
            "cache_read_tokens_from_events": int(tr.usage.get("cache_read_tokens", 0)),
            "source": "gateway_run" if gw_run.get("total_tokens") else "events",
        }
        non_muse = sum(v for k, v in tr.models.items() if "muse" not in k.lower())
        return {
            "run_key": meta.get("run_key", run_dir.name),
            "prompt_id": meta.get("prompt_id"),
            "category": meta.get("category") or prompt.get("category"),
            "mode": meta.get("mode"),
            "repeat": meta.get("repeat"),
            "stop_reason": meta.get("stop_reason"),
            "status": status,
            "n_events": tr.n_events,
            "n_calls": len(calls),
            "n_substantive": len(substantive),
            "calls_dropped_after_cap": truncated_at_cap,
            "sequence": [f"{c.agent}:{c.name}" if c.agent != "lead" else c.name for c in calls],
            "families": dict(fam),
            "families_substantive": dict(fam_sub),
            "by_agent": dict(Counter(c.agent for c in calls)),
            "required_groups": [{"any_of": g, "ok": ok} for g, ok in zip(groups, group_ok)],
            "forbidden_hits": forbidden_hits,
            "passed": passed,
            "outcome": outcome,
            "kizonlp_called": any(n.startswith("kizonlp_") and n not in self.status for n in names),
            "alpha_engine_calls": sum(1 for n in names if n.startswith("alpha_engine_")),
            "spark_registration": spark_status,
            "registration_required": registration_required(prompt),
            "run_code_calls": run_code,
            "run_code_share": (run_code / len(substantive)) if substantive else 0.0,
            "first_domain_tool_position": first_domain,
            "anti_patterns": aps,
            "nonexistent_tool_calls": nonexistent,
            "invalid_tool_calls": tr.invalid_tool_calls,
            "tool_errors": dict(tool_errors),
            "infra": {"gemini_429": gemini_429, "mcp_failures": mcp_failures, "llm_or_run_errors": llm_infra, "llm_error_samples": (tr.llm_errors + tr.run_errors)[:2]},
            "subagent_code_args_truncated": truncated_code,
            "snapshot_ids_seen": snapshots,
            "upload_read_via_read_file": upload_read_file,
            "models": dict(tr.models),
            "non_muse_llm_calls": non_muse,
            "tokens": tokens,
        }


def load_known_tools(path: Path | None) -> set[str] | None:
    if not path or not path.is_file():
        return None
    data = read_json(path)
    names: set[str] = set()
    if isinstance(data, dict):
        # quant_tools.lock.json: {"servers": {server: {"tools": {visible_name: {...}}}}}
        servers = data.get("servers") if isinstance(data.get("servers"), dict) else data
        for server in servers.values():
            tools = server.get("tools") if isinstance(server, dict) else None
            if isinstance(tools, dict):
                names.update(tools)
            elif isinstance(tools, list):
                names.update(t if isinstance(t, str) else t.get("name", "") for t in tools)
    elif isinstance(data, list):
        names.update(str(x) for x in data)
    return names or None


def aggregate(runs: list[dict[str, Any]], ps: PromptSet, pass_dir: Path) -> dict[str, Any]:
    pro = [r for r in runs if r["mode"] in ("pro", "full")]
    ultra = [r for r in runs if r["mode"] == "ultra"]
    repeats = int(ps.defaults.get("repeats", 3))
    expected_pro = len(ps.prompts) * repeats
    out: dict[str, Any] = {"n_runs": len(runs), "n_pro": len(pro), "n_ultra": len(ultra), "expected_pro": expected_pro}

    def scored(rs):
        return [r for r in rs if r["outcome"] != "infra"]

    # 1. Required-tool pass rate (headline), infra runs excluded from the denominator.
    den = scored(pro)
    by_rep = defaultdict(list)
    for r in den:
        by_rep[r["repeat"]].append(1.0 if r["passed"] else 0.0)
    rep_rates = [statistics.fmean(v) for _, v in sorted(by_rep.items()) if v]
    out["required_pass"] = {
        "passed": sum(r["passed"] for r in den),
        "denominator": len(den),
        "infra_excluded": len(pro) - len(den),
        "rate": (sum(r["passed"] for r in den) / len(den)) if den else None,
        "per_repeat": {f"r{k}": statistics.fmean(v) for k, v in sorted(by_rep.items())},
        "spread_over_repeats": spread(rep_rates),
        "threshold": 0.85,
    }
    # 2. Raw data reads (all runs, incl. Ultra).
    out["raw_data_reads"] = {"count": sum(r["anti_patterns"].get("raw_data_read", {}).get("count", 0) for r in runs), "threshold": 0, "runs": [r["run_key"] for r in runs if r["anti_patterns"].get("raw_data_read", {}).get("count")]}
    # 3. kizonlp in NLP runs, per category with explicit denominators.
    nlp = {}
    for cat in NLP_CATEGORIES:
        rs = [r for r in pro if r["category"] == cat]
        nlp[cat] = {"called": sum(r["kizonlp_called"] for r in rs), "denominator": len(rs), "gemini_429_runs": sum(1 for r in rs if r["infra"]["gemini_429"])}
    nlp_all = [r for r in pro if r["category"] in NLP_CATEGORIES]
    out["kizonlp_nlp"] = {"per_category": nlp, "overall": {"called": sum(r["kizonlp_called"] for r in nlp_all), "denominator": len(nlp_all)}, "threshold": "≥ 5/6 per category"}
    # 4. Spark register-before-backtest, registration_required: yes only.
    reg = [r for r in pro if r["registration_required"]]
    reg_u = [r for r in ultra if r["registration_required"]]
    out["spark_registration"] = {
        "ok": sum(r["spark_registration"] == "ok" for r in reg),
        "denominator": len(reg),
        "breakdown": dict(Counter(r["spark_registration"] for r in reg)),
        "ultra": {"ok": sum(r["spark_registration"] == "ok" for r in reg_u), "denominator": len(reg_u)},
        "threshold": 0.80,
    }
    # 5. run_code share (todo writes and skill reads excluded).
    sub = sum(r["n_substantive"] for r in pro)
    rc = sum(r["run_code_calls"] for r in pro)
    out["run_code_share"] = {"run_code_calls": rc, "substantive_calls": sub, "pooled": (rc / sub) if sub else None, "per_run": spread([r["run_code_share"] for r in pro]), "threshold": 0.25}
    # 6. Control.
    ctrl = [r for r in runs if r["category"] == "control"]
    out["control_alpha_engine_calls"] = {"count": sum(r["alpha_engine_calls"] for r in ctrl), "denominator_runs": len(ctrl), "threshold": 0}
    # 7. Nonexistent tools.
    out["nonexistent_tool_calls"] = {
        "count": sum(len(r["nonexistent_tool_calls"]) for r in runs),
        "names": sorted({n for r in runs for n in r["nonexistent_tool_calls"]}),
        "invalid_tool_calls": sum(r["invalid_tool_calls"] for r in runs),
        "threshold": 0,
    }
    # Verdicts.
    rp = out["required_pass"]
    nlp_ok = all(v["denominator"] and v["called"] * 6 >= 5 * v["denominator"] for v in nlp.values())
    sr = out["spark_registration"]
    out["criteria"] = {
        "required_tools_≥85%": rp["rate"] is not None and rp["rate"] >= 0.85,
        "raw_data_reads_0": out["raw_data_reads"]["count"] == 0,
        "kizonlp_≥5/6_per_NLP_category": nlp_ok,
        "spark_register_first_≥80%": bool(sr["denominator"]) and sr["ok"] / sr["denominator"] >= 0.80,
        "run_code_share_≤25%": out["run_code_share"]["pooled"] is not None and out["run_code_share"]["pooled"] <= 0.25,
        "control_alpha_engine_0": out["control_alpha_engine_calls"]["count"] == 0,
        "nonexistent_tools_0": out["nonexistent_tool_calls"]["count"] == 0,
    }
    # Family distribution (audit 04 §2 format).
    for label, rs in (("pro", pro), ("ultra", ultra)):
        fam = Counter()
        for r in rs:
            fam.update(r["families"])
        total = sum(fam.values())
        out[f"families_{label}"] = {"total": total, **{f: fam.get(f, 0) for f in FAMILIES}}
    ultra_by_agent = Counter()
    for r in ultra:
        ultra_by_agent.update(r["by_agent"])
    out["ultra_calls_by_agent"] = dict(ultra_by_agent)
    # Per prompt.
    per_prompt = {}
    for p in ps.prompts:
        rs = sorted([r for r in pro if r["prompt_id"] == p["id"]], key=lambda r: r["repeat"] or 0)
        if not rs:
            continue
        marks = "".join("✓" if r["passed"] else ("i" if r["outcome"] == "infra" else "✗") for r in rs)
        sc = scored(rs)
        per_prompt[p["id"]] = {
            "category": p["category"],
            "repeats": marks,
            "pass_rate": (sum(r["passed"] for r in sc) / len(sc)) if sc else None,
            "flaky": len({r["passed"] for r in sc}) > 1,
            "run_code_share": spread([r["run_code_share"] for r in rs]),
            "alpha_engine_share": spread([(r["families_substantive"].get("alpha_engine", 0) / r["n_substantive"]) if r["n_substantive"] else 0.0 for r in rs]),
            "first_domain_tool_position": spread([float(r["first_domain_tool_position"]) for r in rs if r["first_domain_tool_position"]]),
            "tokens": spread([float(r["tokens"]["total_tokens"]) for r in rs]),
        }
    out["per_prompt"] = per_prompt
    per_cat = {}
    for cat in CATEGORIES:
        rs = scored([r for r in pro if r["category"] == cat])
        if rs:
            per_cat[cat] = {"passed": sum(r["passed"] for r in rs), "denominator": len(rs), "flaky": any(per_prompt.get(pid, {}).get("flaky") for pid, pr in per_prompt.items() if pr["category"] == cat)}
    out["per_category"] = per_cat
    # Anti-patterns, infra, tokens, provenance.
    ap_tot: dict[str, dict[str, Any]] = {}
    for r in runs:
        for k, v in r["anti_patterns"].items():
            a = ap_tot.setdefault(k, {"severity": v["severity"], "count": 0, "runs": 0})
            a["count"] += v["count"]
            a["runs"] += 1 if v["count"] else 0
    out["anti_patterns"] = ap_tot
    out["infra"] = {
        "gemini_429": sum(r["infra"]["gemini_429"] for r in runs),
        "mcp_failures": sum(r["infra"]["mcp_failures"] for r in runs),
        "llm_or_run_errors": sum(r["infra"]["llm_or_run_errors"] for r in runs),
        "infra_runs": [r["run_key"] for r in runs if r["outcome"] == "infra"],
    }
    tool_err = Counter()
    for r in runs:
        tool_err.update(r["tool_errors"])
    out["tool_errors"] = dict(tool_err.most_common())
    tok = [r["tokens"]["total_tokens"] for r in runs]
    out["tokens"] = {
        "total": sum(tok),
        "per_run": spread([float(t) for t in tok]),
        "by_mode": {m: sum(r["tokens"]["total_tokens"] for r in runs if r["mode"] == m) for m in ("pro", "full", "ultra")},
        "cache_read_from_events": sum(r["tokens"]["cache_read_tokens_from_events"] for r in runs),
        "note": "total = Gateway run total_tokens (input incl. any cached input + output). Memory-extraction calls run outside the run and are not included.",
    }
    out["snapshot_ids_seen"] = sorted({s for r in runs for s in r["snapshot_ids_seen"]})
    out["subagent_code_args_truncated"] = sum(r["subagent_code_args_truncated"] for r in runs)
    out["non_muse_llm_calls"] = sum(r["non_muse_llm_calls"] for r in runs)
    out["calls_dropped_after_cap"] = sum(r["calls_dropped_after_cap"] for r in runs)
    out["upload_read_via_read_file_runs"] = sum(1 for r in runs if r["upload_read_via_read_file"])
    manifest = read_json(pass_dir / "manifest.json") if (pass_dir / "manifest.json").is_file() else {}
    out["manifest"] = {k: manifest.get(k) for k in ("pass", "started_at", "ended_at", "snapshot_id", "eval_user", "prompts_sha256", "tree_changed_during_pass", "run_list_sha256")}
    out["incomplete"] = len(pro) < expected_pro
    return out


def paired_deltas(base_runs: list[dict[str, Any]], post_runs: list[dict[str, Any]], ps: PromptSet) -> dict[str, Any]:
    base = {r["run_key"]: r for r in base_runs if r["mode"] in ("pro", "full")}
    post = {r["run_key"]: r for r in post_runs if r["mode"] in ("pro", "full")}
    common = sorted(set(base) & set(post))
    per_prompt = {}
    for p in ps.prompts:
        keys = [k for k in common if base[k]["prompt_id"] == p["id"]]
        if not keys:
            continue

        def mean(rs, f):
            return statistics.fmean([f(r) for r in rs]) if rs else 0.0

        b = [base[k] for k in keys]
        a = [post[k] for k in keys]
        d_pass = mean(a, lambda r: 1.0 if r["passed"] else 0.0) - mean(b, lambda r: 1.0 if r["passed"] else 0.0)
        per_prompt[p["id"]] = {
            "n_pairs": len(keys),
            "pass_rate_base": mean(b, lambda r: 1.0 if r["passed"] else 0.0),
            "pass_rate_post": mean(a, lambda r: 1.0 if r["passed"] else 0.0),
            "delta_pass_rate": d_pass,
            "delta_run_code_share": mean(a, lambda r: r["run_code_share"]) - mean(b, lambda r: r["run_code_share"]),
            "delta_domain_share": mean(a, lambda r: sum(r["families_substantive"].get(f, 0) for f in DOMAIN_FAMILIES) / max(1, r["n_substantive"]))
            - mean(b, lambda r: sum(r["families_substantive"].get(f, 0) for f in DOMAIN_FAMILIES) / max(1, r["n_substantive"])),
            "delta_tokens": mean(a, lambda r: r["tokens"]["total_tokens"]) - mean(b, lambda r: r["tokens"]["total_tokens"]),
        }
    deltas = [v["delta_pass_rate"] for v in per_prompt.values()]
    return {
        "n_common_runs": len(common),
        "missing_in_post": sorted(set(base) - set(post)),
        "missing_in_base": sorted(set(post) - set(base)),
        "per_prompt": per_prompt,
        "summary": {"improved": sum(d > 0 for d in deltas), "unchanged": sum(d == 0 for d in deltas), "worse": sum(d < 0 for d in deltas), "mean_delta_pass_rate": statistics.fmean(deltas) if deltas else None},
        "ultra_side_by_side": {
            "base": [{"run_key": r["run_key"], "passed": r["passed"], "spark": r["spark_registration"], "by_agent": r["by_agent"]} for r in base_runs if r["mode"] == "ultra"],
            "post": [{"run_key": r["run_key"], "passed": r["passed"], "spark": r["spark_registration"], "by_agent": r["by_agent"]} for r in post_runs if r["mode"] == "ultra"],
        },
    }


def score_dir(pass_dir: Path, ps: PromptSet, known: set[str] | None) -> list[dict[str, Any]]:
    scorer = Scorer(ps, known)
    runs_dir = pass_dir / "runs"
    if not runs_dir.is_dir():
        raise EvalError(f"no runs/ directory in {pass_dir}")
    return [scorer.score_run(d) for d in sorted(runs_dir.iterdir()) if d.is_dir()]


def render_markdown(agg: dict[str, Any], runs: list[dict[str, Any]], deltas: dict[str, Any] | None, pass_dir: Path) -> str:
    L: list[str] = []
    m = agg["manifest"]
    L.append(f"# QuantFlow tool-awareness eval: {m.get('pass') or pass_dir.name}")
    L.append("")
    L.append(f"- **Pass dir:** `{pass_dir}`")
    L.append(f"- **Runs:** {agg['n_pro']} Pro/full (expected {agg['expected_pro']}), {agg['n_ultra']} Ultra" + (" — **INCOMPLETE**" if agg["incomplete"] else ""))
    L.append(f"- **Snapshot pinned:** `{m.get('snapshot_id')}`; seen in alpha_engine results: {', '.join(f'`{s}`' for s in agg['snapshot_ids_seen']) or 'none'}")
    L.append(f"- **Eval user:** `{(m.get('eval_user') or {}).get('email')}`; prompts sha256 `{(m.get('prompts_sha256') or '')[:12]}`; tree changed during pass: {m.get('tree_changed_during_pass')}")
    L.append(
        f"- **Tokens:** {fmt_tokens(agg['tokens']['total'])} total ("
        + ", ".join(f"{m2} {fmt_tokens(agg['tokens']['by_mode'][m2])}" for m2 in ("pro", "full", "ultra"))
        + f"); cached input reported in events {fmt_tokens(agg['tokens']['cache_read_from_events'])}"
    )
    L.append("")
    L.append("## Pass criteria")
    L.append("")
    L.append("| Criterion | Result | Denominator | Met |")
    L.append("|---|---|---|---|")
    rp = agg["required_pass"]
    sp = rp["spread_over_repeats"]
    rate = "n/a" if rp["rate"] is None else f"{100 * rp['rate']:.1f}%"
    sprd = "" if sp["mean"] is None else f" (repeats mean {100 * sp['mean']:.1f}%, min {100 * sp['min']:.1f}%, max {100 * sp['max']:.1f}%)"
    c = agg["criteria"]
    L.append(f"| Required tools called (status tools excluded) ≥ 85% | {rp['passed']} passed = {rate}{sprd} | {rp['denominator']} Pro/full runs ({rp['infra_excluded']} infra excluded) | {c['required_tools_≥85%']} |")
    L.append(f"| Raw alpha_engine data reads = 0 | {agg['raw_data_reads']['count']} | all {agg['n_runs']} runs | {c['raw_data_reads_0']} |")
    k = agg["kizonlp_nlp"]
    kc = "; ".join(f"{cat} {v['called']}/{v['denominator']}" for cat, v in k["per_category"].items())
    L.append(f"| kizonlp called in NLP runs ≥ 5/6 per category | {kc} (overall {k['overall']['called']}/{k['overall']['denominator']}) | Pro runs of news_sentiment, filing_read | {c['kizonlp_≥5/6_per_NLP_category']} |")
    s = agg["spark_registration"]
    L.append(f"| spark registers before first backtest ≥ 80% | {s['ok']}/{s['denominator']} ({pct(s['ok'], s['denominator'])}); {s['breakdown']} | Pro runs with registration_required: yes | {c['spark_register_first_≥80%']} |")
    rc = agg["run_code_share"]
    L.append(f"| run_code share ≤ 25% (todo + skill reads excluded) | {rc['run_code_calls']}/{rc['substantive_calls']} = {pct(rc['run_code_calls'], rc['substantive_calls'])} | Pro/full substantive calls | {c['run_code_share_≤25%']} |")
    L.append(f"| alpha_engine calls on control = 0 | {agg['control_alpha_engine_calls']['count']} | {agg['control_alpha_engine_calls']['denominator_runs']} control runs | {c['control_alpha_engine_0']} |")
    L.append(f"| Calls to nonexistent tools = 0 | {agg['nonexistent_tool_calls']['count']} {agg['nonexistent_tool_calls']['names'] or ''} | all runs | {c['nonexistent_tools_0']} |")
    L.append("")
    L.append(
        f"Infrastructure (not counted as misses): Gemini 429 {agg['infra']['gemini_429']}, MCP failures {agg['infra']['mcp_failures']}, "
        f"LLM/run errors {agg['infra']['llm_or_run_errors']}; infra runs: {', '.join(agg['infra']['infra_runs']) or 'none'}."
    )
    L.append("")
    L.append("## Tool-call distribution by family")
    L.append("")
    L.append("| Runs | Calls | " + " | ".join(FAMILIES) + " |")
    L.append("|---|---|" + "---|" * len(FAMILIES))
    for label in ("pro", "ultra"):
        f = agg[f"families_{label}"]
        L.append(f"| {label} | {f['total']} | " + " | ".join(f"{f[x]} ({pct(f[x], f['total'])})" for x in FAMILIES) + " |")
    if agg["ultra_calls_by_agent"]:
        L.append("")
        L.append("Ultra calls by agent: " + ", ".join(f"{a} {n}" for a, n in sorted(agg["ultra_calls_by_agent"].items())))
    L.append("")
    L.append("## Per prompt (Pro/full, over repeats)")
    L.append("")
    L.append("| Prompt | Category | Repeats | Pass rate | Flaky | run_code share mean [min–max] | alpha_engine share mean | 1st domain tool pos. | Tokens mean |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for pid, v in agg["per_prompt"].items():
        rcs = v["run_code_share"]
        ae = v["alpha_engine_share"]
        fd = v["first_domain_tool_position"]
        L.append(
            f"| {pid} | {v['category']} | {v['repeats']} | {'n/a' if v['pass_rate'] is None else f'{100 * v["pass_rate"]:.0f}%'} | {'yes' if v['flaky'] else ''} | "
            f"{100 * rcs['mean']:.0f}% [{100 * rcs['min']:.0f}–{100 * rcs['max']:.0f}] | {100 * ae['mean']:.0f}% | {'' if fd['mean'] is None else f'{fd["mean"]:.1f}'} | {fmt_tokens(v['tokens']['mean'])} |"
        )
    L.append("")
    L.append("Per category (descriptive only; n = 6 cannot separate 80% from 60%): " + "; ".join(f"{cat} {v['passed']}/{v['denominator']}{' (flaky)' if v['flaky'] else ''}" for cat, v in agg["per_category"].items()))
    L.append("")
    L.append("## Anti-patterns")
    L.append("")
    L.append("| Id | Severity | Calls | Runs |")
    L.append("|---|---|---|---|")
    for k2, v in sorted(agg["anti_patterns"].items(), key=lambda kv: (kv[1]["severity"] != "hard", kv[0])):
        L.append(f"| {k2} | {v['severity']} | {v['count']} | {v['runs']} |")
    ev = [(r["run_key"], k2, e) for r in runs for k2, v in r["anti_patterns"].items() if v["severity"] == "hard" for e in v["evidence"]]
    if ev:
        L.append("")
        L.append("Hard anti-pattern evidence (first 160 chars of the matching call):")
        for rk, k2, e in ev[:20]:
            L.append(f"- `{rk}` {k2}: `{e.replace('`', "'")}`")
    L.append("")
    L.append(f"Tool errors: {', '.join(f'{k2} {v}' for k2, v in list(agg['tool_errors'].items())[:12]) or 'none'}.")
    if deltas:
        L.append("")
        L.append("## Paired per-prompt deltas (post − baseline)")
        L.append("")
        sm = deltas["summary"]
        L.append(
            f"{deltas['n_common_runs']} paired runs. Prompts improved {sm['improved']}, unchanged {sm['unchanged']}, worse {sm['worse']}; "
            f"mean Δ pass rate {'n/a' if sm['mean_delta_pass_rate'] is None else f'{100 * sm["mean_delta_pass_rate"]:+.1f} pp'}."
        )
        L.append("")
        L.append("| Prompt | Pairs | Pass base → post | Δ run_code share | Δ domain-tool share | Δ tokens |")
        L.append("|---|---|---|---|---|---|")
        for pid, v in deltas["per_prompt"].items():
            L.append(
                f"| {pid} | {v['n_pairs']} | {100 * v['pass_rate_base']:.0f}% → {100 * v['pass_rate_post']:.0f}% | {100 * v['delta_run_code_share']:+.0f} pp | "
                f"{100 * v['delta_domain_share']:+.0f} pp | {('+' if v['delta_tokens'] >= 0 else '-') + fmt_tokens(abs(v['delta_tokens']))} |"
            )
        if deltas["missing_in_post"] or deltas["missing_in_base"]:
            L.append("")
            L.append(f"Unpaired: missing in post {deltas['missing_in_post']}, missing in baseline {deltas['missing_in_base']}.")
        L.append("")
        L.append("Ultra (qualitative; baseline has 1 capped run, post has 5 uncapped):")
        for side in ("base", "post"):
            for u in deltas["ultra_side_by_side"][side]:
                L.append(f"- {side} `{u['run_key']}`: passed {u['passed']}, spark {u['spark']}, calls by agent {u['by_agent']}")
    L.append("")
    L.append("## Per run")
    L.append("")
    L.append("| Run | Mode | Stop | Outcome | Calls (subst.) | Failed groups | Spark | Tokens |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in runs:
        failed = [" or ".join(g["any_of"][:2]) + ("…" if len(g["any_of"]) > 2 else "") for g in r["required_groups"] if not g["ok"]]
        if r["forbidden_hits"]:
            failed.append("forbidden: " + ", ".join(sorted(set(r["forbidden_hits"]))))
        L.append(f"| {r['run_key']} | {r['mode']} | {r['stop_reason']} | {r['outcome']} | {r['n_calls']} ({r['n_substantive']}) | {'; '.join(failed)} | {r['spark_registration'] or ''} | {fmt_tokens(r['tokens']['total_tokens'])} |")
    L.append("")
    L.append("## Caveats")
    L.append("")
    L.append(f"- Subagent tool-call args are capped at 8192 characters in `subagent.step` events; {agg['subagent_code_args_truncated']} code-bearing subagent calls were truncated, so a raw-data read past the cap can be missed.")
    L.append(f"- Capped runs are scored on the first `cap` substantive calls; {agg['calls_dropped_after_cap']} calls issued in the same turn as the cap were dropped.")
    L.append(f"- {agg['upload_read_via_read_file_runs']} runs read an upload with `read_file` (the Gateway also stores a markdown conversion); this is informational, not a miss.")
    L.append(f"- Non-Muse LLM calls (fallback model) seen: {agg['non_muse_llm_calls']}.")
    L.append("- Anti-pattern regexes are heuristics over call arguments; check the evidence lines before acting on a count.")
    L.append("- Per-category figures are descriptive; compare passes with the paired per-prompt deltas.")
    return "\n".join(L) + "\n"


def cmd_score(args) -> int:
    pass_dir = args.score
    prompts_path = args.prompts if args.prompts_explicit else (pass_dir / "prompts.yaml" if (pass_dir / "prompts.yaml").is_file() else args.prompts)
    ps = load_prompts(prompts_path)
    known = load_known_tools(args.known_tools)
    runs = score_dir(pass_dir, ps, known)
    agg = aggregate(runs, ps, pass_dir)
    deltas = None
    if args.baseline:
        base_runs = score_dir(args.baseline, ps, known)
        deltas = paired_deltas(base_runs, runs, ps)
    out = args.out or pass_dir
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "report.json", {"generated_at": now_iso(), "prompts": str(prompts_path), "known_tools": str(args.known_tools) if known else None, "aggregate": agg, "paired_deltas": deltas, "runs": runs})
    (out / "report.md").write_text(render_markdown(agg, runs, deltas, pass_dir), encoding="utf-8")
    print(f"scored {len(runs)} runs -> {out / 'report.md'}, {out / 'report.json'}")
    for k, v in agg["criteria"].items():
        print(f"  {k}: {v}")
    return 0


# --------------------------------------------------------------------------- commands


def cmd_init_credentials(args) -> int:
    path = args.credentials
    if path.exists():
        raise EvalError(f"{path} already exists; not overwriting")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"email": args.email, "password": secrets.token_urlsafe(24)}, fh)
    print(f"wrote {path} (mode 0600) for {args.email}; the password was not printed")
    return 0


def cmd_setup_user(args) -> int:
    gw = Gateway(args.gateway, args.credentials)
    state = gw.register()
    user = gw.login()
    print(f"eval user {state}: id={user.get('id')} email={user.get('email')} role={user.get('system_role')}")
    print(f"memory dir (delete after the eval): {args.users_root / str(user.get('id'))}")
    return 0


def cmd_dry_run(args) -> int:
    ps = load_prompts(args.prompts)
    problems = validate_prompts(ps)
    print(f"prompts: {args.prompts} sha256 {ps.sha256[:12]}: {'OK' if not problems else 'INVALID'}")
    for p in problems:
        print(f"  - {p}")
    for p in ps.prompts:
        if p.get("upload"):
            f = FIXTURES_DIR / p["upload"]
            print(f"fixture {p['id']}: {f.name} {'present' if f.is_file() else 'MISSING'} ({f.stat().st_size if f.is_file() else 0} bytes)")
    errors, warnings, facts = check_configs(args.snapshot_id, allow_live_spark=args.allow_live_spark, allow_memory_injection=args.allow_memory_injection)
    print("isolation config checks (would block a live pass):" if errors else "isolation config checks: OK")
    for e in errors:
        print(f"  ERROR {e}")
    for w in warnings:
        print(f"  warn  {w}")
    print(f"  facts: {json.dumps({k: v for k, v in facts.items() if k != 'mcpServers'})}, servers={facts.get('mcpServers')}")
    tmpl = yaml.safe_load(args.spark_template.read_text(encoding="utf-8"))
    print(f"spark template: db_path={tmpl['storage']['db_path']} degraded_mode={tmpl['llm']['degraded_mode']} peer={tmpl['peers']['alpha_engine']['command']}; rendered target {RENDERED_SPARK_CONFIG}")
    if args.snapshot_id:
        print(f"snapshot: {json.dumps(check_snapshot_lease(args.snapshot_id))}")
    gws = gateway_processes()
    print(f"gateway processes: {len(gws)}{' (one runs with --reload!)' if any('--reload' in c for _, c in gws) else ''}; kizonlp-mcp: {kizonlp_processes()}")
    print(f"credentials file: {args.credentials} {'present' if args.credentials.is_file() else 'absent'}")
    if args.pilot:
        specs = pilot_specs(ps)
        print(f"\npilot plan ({len(specs)} runs):")
    else:
        rl = read_json(args.run_list) if args.run_list and args.run_list.is_file() else build_run_list(ps)
        phase = args.pass_name or "baseline"
        specs = rl["pro_runs"] + rl["ultra_runs"][phase]
        src = args.run_list if args.run_list and args.run_list.is_file() else "(built in memory; a real pass needs the pilot's run list)"
        print(f"\n{phase} pass plan from {src}: {len(specs)} runs")
    for s in specs:
        prompt = ps.by_id()[s["prompt_id"]]
        print(
            f"  {s['run_key']:<34} mode={s['mode']:<5} cap={s['cap']!s:<5} timeout={s['timeout_s']}s "
            f"flags={json.dumps({k: v for k, v in build_context(ps, s['mode'], '<tid>').items() if k in FLAG_KEYS})}"
            f"{' upload=' + prompt['upload'] if prompt.get('upload') else ''}"
        )
    print("\ndry run: no Gateway contact, nothing written.")
    return 1 if problems else 0


def cmd_pilot(args) -> int:
    ps = load_prompts(args.prompts)
    run_list_path = args.run_list or (args.results_root / "runlist.json")
    if run_list_path.exists() and not args.force:
        raise EvalError(f"{run_list_path} exists; both passes must use one run list. Use --force only before the baseline has started.")
    specs = pilot_specs(ps)
    with single_runner_lock(args.results_root):
        state = start_pass(args, "pilot", specs, {})
        run_specs(state, specs, resume=args.resume)
        finish_pass(state)
    per_mode: dict[str, list[int]] = defaultdict(list)
    for spec in specs:
        meta = read_json(state.pass_dir / "runs" / spec["run_key"] / "meta.json")
        per_mode[spec["mode"]].append(int(meta["total_tokens"]))
    per_mode_mean = {m: statistics.fmean(v) for m, v in per_mode.items()}
    rl = build_run_list(ps)
    projection = project_cost(rl, per_mode_mean)
    rl["pilot"] = {"pass_dir": str(state.pass_dir), "per_run_tokens": {m: v for m, v in per_mode.items()}, "projection": projection, "snapshot_id": args.snapshot_id}
    write_json(run_list_path, rl)
    print(f"\npilot per-run tokens: {', '.join(f'{m} {fmt_tokens(v)}' for m, v in per_mode_mean.items())}")
    for phase in ("baseline", "post"):
        print(f"projected {phase} pass: {fmt_tokens(projection[phase]['projected_total_tokens'])} ({projection['n_pro_capped']} capped Pro + {projection['n_full']} full + {projection[phase]['n_ultra']} Ultra)")
    print(projection["note"])
    print(f"run list written: {run_list_path} (sha256 {sha256_file(run_list_path)[:12]})")
    return 0


def cmd_pass(args) -> int:
    run_list_path = args.run_list or (args.results_root / "runlist.json")
    if not run_list_path.is_file():
        raise EvalError(f"run list {run_list_path} not found; run --pilot first (F5b)")
    rl = read_json(run_list_path)
    ps = load_prompts(args.prompts)
    if rl["prompts_sha256"] != ps.sha256:
        raise EvalError("prompts.yaml changed since the run list was written; passes must use the identical prompt set")
    specs = rl["pro_runs"] + rl["ultra_runs"][args.pass_name]
    with single_runner_lock(args.results_root):
        state = start_pass(args, args.pass_name, specs, {"run_list": str(run_list_path), "run_list_sha256": sha256_file(run_list_path)})
        try:
            run_specs(state, specs, resume=args.resume)
        finally:
            finish_pass(state)
    print(f"\npass {args.pass_name} done: {state.pass_dir}; cumulative {fmt_tokens(state.cumulative_tokens)}. Score with --score {state.pass_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qf_eval.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--init-credentials", action="store_true", help="write a 0600 credentials file for the eval user")
    mode.add_argument("--setup-user", action="store_true", help="register (or verify) the eval user on the Gateway")
    mode.add_argument("--pilot", action="store_true", help="run the 3 pilot runs and write the fixed run list")
    mode.add_argument("--pass", dest="pass_name", choices=("baseline", "post"), help="execute the fixed run list")
    mode.add_argument("--score", type=Path, metavar="DIR", help="score a pass directory offline")
    p.add_argument("--dry-run", action="store_true", help="with --pilot/--pass: validate and print the plan; no Gateway contact")
    p.add_argument("--gateway", default=DEFAULT_GATEWAY, help=f"Gateway base URL (default {DEFAULT_GATEWAY}, direct, not via nginx)")
    p.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS, help="eval-user credentials JSON, mode 0600 (env QF_EVAL_CREDENTIALS_FILE)")
    p.add_argument("--email", default=EVAL_EMAIL, help="eval-user email for --init-credentials")
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--spark-template", type=Path, default=DEFAULT_SPARK_TEMPLATE)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--pass-dir", type=Path, help="override <results-root>/<pass>")
    p.add_argument("--run-list", type=Path, help="run list file (default <results-root>/runlist.json)")
    p.add_argument("--snapshot-id", help="alpha_engine snapshot pinned for the pass (must match extensions_config)")
    p.add_argument("--lease", action="store_true", help="create a 21-day CLI lease on --snapshot-id if none is valid")
    p.add_argument("--resume", action="store_true", help="continue a pass, skipping completed runs")
    p.add_argument("--force", action="store_true", help="--pilot: overwrite an existing run list")
    p.add_argument("--warn-tokens", type=int, default=DEFAULT_WARN_TOKENS)
    p.add_argument("--budget-tokens", type=int, default=DEFAULT_BUDGET_TOKENS)
    p.add_argument("--hard-stop", action="store_true", help="stop the pass at --budget-tokens (leaves it incomplete)")
    p.add_argument("--events-root", type=Path, default=DEFAULT_EVENTS_ROOT, help="jsonl run-event root (Gateway cwd/.deer-flow/threads)")
    p.add_argument("--users-root", type=Path, default=DEFAULT_USERS_ROOT)
    p.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR, help="report-forge reports dir; projects created during a run are moved into the run dir")
    p.add_argument("--no-reportforge-archive", action="store_true")
    p.add_argument("--allow-live-spark", action="store_true", help="skip the scratch spark config (not for real passes)")
    p.add_argument("--allow-memory-injection", action="store_true")
    p.add_argument("--allow-reload", action="store_true")
    p.add_argument("--baseline", type=Path, help="--score: baseline pass dir for paired deltas")
    p.add_argument("--known-tools", type=Path, default=REPO / "config" / "quant_tools.lock.json", help="--score: lock file of live tool names")
    p.add_argument("--out", type=Path, help="--score: output dir (default: the pass dir)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.prompts_explicit = "--prompts" in (argv if argv is not None else sys.argv[1:])
    try:
        if args.init_credentials:
            return cmd_init_credentials(args)
        if args.score:
            return cmd_score(args)
        if args.dry_run:
            return cmd_dry_run(args)
        if args.setup_user:
            return cmd_setup_user(args)
        if args.pilot:
            return cmd_pilot(args)
        return cmd_pass(args)
    except EvalError as exc:
        print(f"qf_eval: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

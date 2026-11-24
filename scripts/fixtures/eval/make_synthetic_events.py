"""Usage: backend/.venv/bin/python scripts/fixtures/eval/make_synthetic_events.py scripts/fixtures/eval/synthetic

Generate synthetic run-event fixtures for scripts/qf_eval.py --score (offline test).

Event records follow JsonlRunEventStore.put (thread_id, run_id, event_type, category,
content, metadata, seq, created_at); contents follow journal.py (AIMessage/ToolMessage
model_dump) and subagents/step_events.py (subagent.start/step/end).
"""

import json
import sys
import uuid
from pathlib import Path

OUT = Path(sys.argv[1])
SNAP = "20261118T044522Z-809bbb36"


class Run:
    def __init__(self, key, prompt_id, category, mode, repeat, cap=None, stop="completed", status="success"):
        self.key, self.tid, self.rid = key, str(uuid.uuid4()), "run-" + uuid.uuid4().hex[:12]
        self.meta = {"run_key": key, "prompt_id": prompt_id, "category": category, "mode": mode, "repeat": repeat, "cap": cap, "stop_reason": stop, "thread_id": self.tid, "run_id": self.rid}
        self.status, self.ev, self.seq, self.tokens = status, [], 0, 0
        self.put("run.start", "trace", "", {})
        self.put("llm.human.input", "message", {"type": "human", "content": "synthetic prompt"}, {"caller": "lead_agent"})

    def put(self, et, cat, content, meta):
        self.seq += 1
        self.ev.append({"thread_id": self.tid, "run_id": self.rid, "event_type": et, "category": cat, "content": content, "metadata": meta, "seq": self.seq, "created_at": "2026-11-18T12:00:00+00:00"})

    def lead(self, calls, results=None, inp=50000, out=800):
        mid = "lc_" + uuid.uuid4().hex[:8]
        tcs = [{"name": n, "args": a, "id": "call_" + uuid.uuid4().hex[:10], "type": "tool_call"} for n, a in calls]
        self.put(
            "llm.ai.response",
            "message",
            {
                "type": "ai",
                "id": mid,
                "content": "",
                "tool_calls": tcs,
                "invalid_tool_calls": [],
                "response_metadata": {"model_name": "muse-spark-1.3-contributor"},
                "usage_metadata": {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out},
            },
            {"caller": "lead_agent", "usage": {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out, "input_token_details": {"cache_read": inp // 2}}, "llm_call_index": self.seq},
        )
        self.tokens += inp + out
        results = results or {}
        for tc in tcs:
            text, st = results.get(tc["name"], ('{"ok": true}', "success"))
            self.put("llm.tool.result", "message", {"type": "tool", "name": tc["name"], "tool_call_id": tc["id"], "status": st, "content": text, "id": uuid.uuid4().hex}, {})
        return tcs

    def subagent(self, task_id, steps):
        self.put("subagent.start", "subagent", {"task_id": task_id, "description": "desk work"}, {"task_id": task_id})
        for i, (name, args, truncated, result) in enumerate(steps):
            call = {"name": name, "args": args}
            if truncated:
                call = {"name": name, "args": json.dumps(args)[:8192], "args_truncated": True}
            self.put("subagent.step", "subagent", {"task_id": task_id, "message_index": 2 * i, "kind": "ai", "text": "", "truncated": False, "tool_calls": [call]}, {"task_id": task_id, "message_index": 2 * i})
            self.put("subagent.step", "subagent", {"task_id": task_id, "message_index": 2 * i + 1, "kind": "tool", "text": result, "truncated": False, "tool_name": name}, {"task_id": task_id, "message_index": 2 * i + 1})
        self.put("subagent.end", "subagent", {"task_id": task_id, "status": "completed", "result": "done", "usage": {"input_tokens": 300000, "output_tokens": 5000, "total_tokens": 305000}}, {"task_id": task_id})
        self.tokens += 305000

    def write(self, pass_dir):
        if self.status == "error":
            self.put("llm.error", "trace", "Error code: 429 - rate limit exceeded", {})
            self.put("run.error", "error", "ModelError: 429", {"error_type": "ModelError"})
        else:
            self.put("run.end", "outputs", {}, {"status": "success"})
        self.put("run.terminal", "trace", {"status": self.status}, {"protocol_version": 1, "recovered": False})
        d = pass_dir / "runs" / self.key
        (d / "events").mkdir(parents=True, exist_ok=True)
        (d / "events" / f"{self.rid}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in self.ev))
        self.meta["gateway_run"] = {"status": self.status, "total_tokens": self.tokens, "total_input_tokens": int(self.tokens * 0.98), "total_output_tokens": int(self.tokens * 0.02)}
        self.meta["total_tokens"] = self.tokens
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2))


AE_OK = (json.dumps({"ok": True, "snapshot_id": SNAP, "rows": 252}), "success")


def build(pass_name, post):
    pd = OUT / pass_name
    runs = []
    # stock compare
    r = Run("r1-stock-compare-1", "stock-compare-1", "stock_compare", "pro", 1, cap=20)
    r.lead([("write_todos", {"todos": []}), ("alpha_engine_engine_status", {})])
    if post:
        r.lead([("alpha_engine_get_price_history", {"symbols": ["NVDA", "AMD"]})], {"alpha_engine_get_price_history": AE_OK})
        r.lead([("reportforge_run_code", {"code": "import numpy as np\nprint(np.mean([0.1, 0.2]))"})])
    else:
        r.lead([("reportforge_run_code", {"code": "import pandas as pd\ndf = pd.read_parquet('/home/fire/Documents/alpha_engine/data/prices/NVDA.parquet')\nprint(df.close.pct_change().std())"})])
        r.lead([("web_search", {"query": "NVDA AMD stock price returns 2026"})])
    runs.append(r)
    # factor idea (registration_required: yes)
    r = Run("r1-factor-idea-1", "factor-idea-1", "factor_idea", "pro", 1, cap=20)
    if post:
        for n in ("spark_check_novelty", "spark_hypothesis_propose", "spark_register_hypothesis", "spark_backtest_run", "spark_significance_gate", "spark_log_outcome"):
            r.lead([(n, {"x": 1})])
    else:
        r.lead([("alpha_engine_run_factor_analysis", {"expression": "-vol_60 + mom_126"})], {"alpha_engine_run_factor_analysis": AE_OK})
        r.lead([("spark_register_hypothesis", {"statement": "low vol + momentum"})])
    runs.append(r)
    # news sentiment with Gemini 429 at baseline
    r = Run("r1-news-sentiment-1", "news-sentiment-1", "news_sentiment", "pro", 1, cap=20)
    r.lead([("web_search_advanced_exa" if post else "web_search", {"query": "Micron news this week"})])
    res = {"kizonlp_fin_sentiment": ('{"ok": true, "labels": ["neutral"]}', "success")} if post else {"kizonlp_fin_sentiment": ("Error: gemini embedding failed: 429 RESOURCE_EXHAUSTED quota exceeded", "error")}
    r.lead([("kizonlp_fin_sentiment", {"texts": ["Micron beats"]})], res)
    runs.append(r)
    # filing read
    r = Run("r1-filing-read-1", "filing-read-1", "filing_read", "pro", 1, cap=20)
    if post:
        r.lead([("kizonlp_parse_document", {"path": "/mnt/user-data/uploads/northwind-10q-q3fy2026-excerpt.pdf"})])
    else:
        r.lead([("read_file", {"path": "/mnt/user-data/uploads/northwind-10q-q3fy2026-excerpt.md"})])
        r.lead([("read_file", {"path": "/mnt/skills/custom/quant-desk/SKILL.md"})])
    runs.append(r)
    # control
    r = Run("r1-control-1", "control-1", "control", "pro", 1, cap=20)
    if not post:
        r.lead([("alpha_engine_engine_status", {})])
    r.lead([("reportforge_run_code", {"code": "print('racecar'[::-1])"})])
    runs.append(r)
    # report (full)
    r = Run("r1-report-1", "report-1", "report_request", "full", 1)
    r.lead([("reportforge_scaffold_report", {"project": "costco"})])
    r.lead([("alpha_engine_get_price_history", {"symbols": ["COST"]})], {"alpha_engine_get_price_history": AE_OK})
    if post:
        r.lead([("reportforge_render_report", {"project": "costco"})])
        r.lead([("present_files", {"filepaths": ["/mnt/user-data/outputs/costco.pdf"]})])
    else:
        r.lead([("reportforge_save_chart", {"spec": {}})], {"reportforge_save_chart": ("Error: browser seemed to close immediately", "error")})
        r.lead([("write_file", {"path": "/mnt/user-data/outputs/costco.md", "content": "# Costco"})])
    runs.append(r)
    # PIT: LLM 429 before any tool call -> infra
    r = Run("r1-pit-1", "pit-1", "pit_historical", "pro", 1, cap=20, stop="completed", status="error" if not post else "success")
    if post:
        r.lead([("alpha_engine_universe_members_as_of", {"date": "2009-03-02"})], {"alpha_engine_universe_members_as_of": AE_OK})
        r.lead([("alpha_engine_sql", {"query": "select 1"})], {"alpha_engine_sql": AE_OK})
    runs.append(r)
    # capped portfolio run: 22 substantive calls, cap 20; one nonexistent tool
    r = Run("r1-portfolio-1", "portfolio-1", "portfolio_construction", "pro", 1, cap=20, stop="cap", status="interrupted")
    r.lead([("alpha_engine_sql", {"query": "select symbol from instruments"})], {"alpha_engine_sql": AE_OK})
    r.lead([("alpha_engine_optimise_portfolio", {})], {"alpha_engine_optimise_portfolio": ("Error: alpha_engine_optimise_portfolio is not a valid tool, try one of [alpha_engine_sql].", "error")})
    for i in range(18):
        r.lead([("alpha_engine_get_price_history", {"symbols": [f"S{i}"]})], {"alpha_engine_get_price_history": AE_OK}, inp=20000)
    r.lead([("alpha_engine_optimize_portfolio", {}), ("alpha_engine_portfolio_risk_decomposition", {})])  # 21st/22nd: dropped
    runs.append(r)
    # Ultra desk run with a subagent
    r = Run("u1-factor-idea-1-ultra", "factor-idea-1", "factor_idea", "ultra", 1, cap=60)
    tcs = r.lead([("task", {"subagent_type": "quant-analyst", "description": "test the signal", "prompt": "..."})])
    big = {"code": ("x = 1\n" * 2000) + "pd.read_parquet('/home/fire/Documents/alpha_engine/data/prices/AAPL.parquet')"}
    steps = [
        ("spark_register_hypothesis", {"statement": "lv+mom"}, False, '{"ok": true}'),
        ("alpha_engine_run_backtest", {"signal": "lv+mom"}, False, json.dumps({"ok": True, "snapshot_id": SNAP})),
        ("reportforge_run_code", big, not post, '{"ok": true}'),
    ]
    if post:
        steps = steps[:2]
    r.subagent(tcs[0]["id"], steps)
    runs.append(r)
    for r in runs:
        r.write(pd)
    (pd / "manifest.json").write_text(
        json.dumps(
            {"pass": pass_name, "started_at": "2026-11-18T12:00:00+00:00", "snapshot_id": SNAP, "eval_user": {"id": "synthetic", "email": "qf-eval@example.com"}, "prompts_sha256": "synthetic", "tree_changed_during_pass": False}, indent=2
        )
    )


build("baseline", post=False)
build("post", post=True)
print("ok", OUT)

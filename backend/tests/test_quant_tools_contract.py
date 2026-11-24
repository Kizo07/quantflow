"""Offline contract for config/quant_tools.yaml + config/quant_tools.lock.json.

Runs ``scripts/quant_tools.py`` --emit/--check logic against the committed
manifest and lock with a tiny fixture checkout: emitted allowlists and routing
must check clean, and a prompt with a stale tool name, a bash instruction and
a lead-only allowlist entry must fail.
"""

from __future__ import annotations

from pathlib import Path

import quant_tools  # scripts/ is on sys.path via conftest
import yaml

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "config" / "quant_tools.yaml"
LOCK = REPO / "config" / "quant_tools.lock.json"


def _write_config(root: Path, prompt: str, tools: list[str]) -> None:
    config = {
        "subagents": {
            "custom_agents": {
                "quant-analyst": {
                    "description": "Quantitative analysis from local data.",
                    "system_prompt": prompt,
                    "tools": tools,
                    "skills": [],
                }
            }
        }
    }
    (root / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def test_manifest_lock_and_prompt_surfaces(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEER_FLOW_HOME", str(tmp_path / "home"))
    root = tmp_path / "checkout"
    root.mkdir()

    # Emitted machine data checks clean against the committed manifest + lock.
    quant_tools.run_emit(root, MANIFEST, LOCK)
    catalog = quant_tools.Catalog.build(quant_tools.load_manifest(MANIFEST), quant_tools.load_lock(LOCK))
    good_tools = catalog.manifest_set("quant-analyst")
    _write_config(
        root,
        "Call alpha_engine_engine_status first, then alpha_engine_run_factor_analysis; use alpha_engine_sql before reportforge_run_code. There is no bash.",
        good_tools,
    )
    report = quant_tools.run_check(root, MANIFEST, LOCK)
    assert report.errors == [], [f"{f.where}: {f.message}" for f in report.errors]
    assert any(agent == "quant-analyst" and "allowlist" in source for agent, source, _, _ in report.agent_rows)

    # Known-bad names fail: retired tool, bash instruction, lead-only knowledge tool.
    _write_config(
        root,
        "Parse filings with kizonlp_pdf_text, then run pandas via bash.",
        [*good_tools, "ledger_search"],
    )
    messages = [f.message for f in quant_tools.run_check(root, MANIFEST, LOCK).errors]
    assert any(m.startswith("kizonlp_pdf_text:") and "kizonlp_parse_document" in m for m in messages), messages
    assert any("'bash'" in m for m in messages), messages
    assert any("ledger_search" in m and "lead-only" in m for m in messages), messages

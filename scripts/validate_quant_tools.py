"""Validate config/quant_tools.yaml against the live MCP server tool defs.

Reads tool names by statically parsing ``@mcp.tool()`` decorators in
alpha_engine's mcp_server.py (``ast`` — stdlib only), so the catalog cannot
drift from the actual server without importing its heavy third-party chain
(numpy/pandas/vectorbt/plotly), which has broken validation on unrelated
dependency drift before. Run from the deer-flow repo root or anywhere;
ALPHA_ENGINE_SRC overrides the default sibling-checkout lookup.

Also cross-checks the layer-3 scope contract (warn-only, never fails): the
``alpha_engine`` entry must exist in ``extensions_config.example.json`` so the
enablement template cannot silently drop the server this catalog maps.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def alpha_engine_src() -> Path:
    """Locate the alpha_engine checkout portably.

    ``ALPHA_ENGINE_SRC`` wins when set; otherwise assume a sibling checkout
    of this repo (``../alpha_engine/src``), matching ``source_repo`` in
    config/quant_tools.yaml.
    """
    override = os.environ.get("ALPHA_ENGINE_SRC")
    if override:
        return Path(override)
    return REPO.parent / "alpha_engine" / "src"


def mcp_tool_names(ae_src: Path) -> set[str]:
    """Tool names declared in alpha_engine's mcp_server.py (bare names).

    Static ``ast`` extraction of ``@mcp.tool()``-decorated defs — exact for
    this server (all tools are decorated defs in the one file; verified no
    dynamic add_tool/mount registration). An explicit ``name=`` kwarg wins
    when present.
    """
    import ast  # noqa: PLC0415

    server_py = ae_src / "alpha_engine" / "mcp_server.py"
    if not server_py.is_file():
        print(f"mcp_server.py not found at {server_py}")
        print("Set ALPHA_ENGINE_SRC to the '<checkout>/src' directory and retry.")
        raise SystemExit(2)
    tree = ast.parse(server_py.read_text(), filename=str(server_py))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            func = dec.func if isinstance(dec, ast.Call) else dec
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "tool"
                and isinstance(func.value, ast.Name)
                and func.value.id == "mcp"
            ):
                continue
            name = node.name
            if isinstance(dec, ast.Call):
                for kw in dec.keywords:
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                        name = str(kw.value.value)
            names.add(name)
    return names


def check_extensions_config_layer() -> None:
    """Layer-3 cross-check: warn unless the example template enables alpha_engine.

    Warn-only by design (Unit D): a mismatch must not fail validation until
    the catalog resync lands.

    NOTE: catalog entries carry no ``server`` field today — all current
    entries belong to the single ``alpha_engine`` MCP server, so this check
    can only assert that one server entry. A multi-server catalog will need a
    per-entry schema field (e.g. ``server: <mcpServers key>``) before entries
    can be mapped to their servers.
    """
    example = REPO / "extensions_config.example.json"
    if not example.is_file():
        print(f"WARN: layer 3: {example.name} not found at repo root")
        return
    try:
        config = json.loads(example.read_text())
    except json.JSONDecodeError as exc:
        print(f"WARN: layer 3: {example.name} is not valid JSON: {exc}")
        return
    servers = config.get("mcpServers", {})
    if "alpha_engine" not in servers:
        print(f"WARN: layer 3: no 'alpha_engine' entry in {example.name}")
        return
    print("LAYER 3 OK: alpha_engine entry present in extensions_config.example.json")


def main() -> int:
    ae_src = alpha_engine_src()
    if not ae_src.is_dir():
        print(f"alpha_engine sources not found at {ae_src}")
        print("Set ALPHA_ENGINE_SRC to the '<checkout>/src' directory and retry.")
        return 2
    catalog = yaml.safe_load((REPO / "config" / "quant_tools.yaml").read_text())
    catalog_ids = {t["id"] for t in catalog["tools"]}
    bare_ids = {i.removeprefix("alpha_engine_") for i in catalog_ids}
    actual = mcp_tool_names(ae_src)

    missing_in_catalog = sorted(actual - bare_ids)
    stale_in_catalog = sorted(
        i for i in catalog_ids
        if i.removeprefix("alpha_engine_") not in actual)

    print(f"mcp_server tools: {len(actual)}")
    print(f"catalog tools:    {len(catalog_ids)}")
    if missing_in_catalog:
        print("MISSING from catalog (add these):", missing_in_catalog)
    if stale_in_catalog:
        print("STALE in catalog (remove/rename):", stale_in_catalog)
    ok = not missing_in_catalog and not stale_in_catalog
    print("CATALOG OK" if ok else "CATALOG OUT OF SYNC")
    check_extensions_config_layer()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

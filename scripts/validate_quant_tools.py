"""Validate config/quant_tools.yaml against the live MCP server tool defs.

Reads tool names via fastmcp's registry (import alpha_engine.mcp_server) so
the catalog cannot drift from the actual server. Exit 1 with a diff if the
catalog is stale. Run from the deer-flow repo root or anywhere, with a Python
that has fastmcp (e.g. the alpha_engine env); ALPHA_ENGINE_SRC overrides the
default sibling-checkout lookup.
"""
from __future__ import annotations

import os
import sys
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
    """Tool names declared in alpha_engine.mcp_server (bare names)."""
    sys.path.insert(0, str(ae_src))
    import alpha_engine.mcp_server as mcp_server  # noqa: PLC0415

    import asyncio  # noqa: PLC0415

    async def _collect() -> set[str]:
        listed = await mcp_server.mcp.list_tools()
        return {t.name for t in listed}

    return asyncio.run(_collect())


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
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

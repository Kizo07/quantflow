"""Validate config/quant_tools.yaml against the live MCP server tool defs.

Reads tool names via fastmcp's registry (import alpha_engine.mcp_server) so
the catalog cannot drift from the actual server. Exit 1 with a diff if the
catalog is stale. Run from the deer-flow repo root or anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path("/home/fire/Documents/deer-flow")
AE_SRC = Path("/home/fire/Documents/alpha_engine/src")


def mcp_tool_names() -> set[str]:
    """Tool names declared in alpha_engine.mcp_server (bare names)."""
    sys.path.insert(0, str(AE_SRC))
    import alpha_engine.mcp_server as mcp_server  # noqa: PLC0415

    import asyncio  # noqa: PLC0415

    async def _collect() -> set[str]:
        listed = await mcp_server.mcp.list_tools()
        return {t.name for t in listed}

    return asyncio.run(_collect())


def main() -> int:
    catalog = yaml.safe_load((REPO / "config" / "quant_tools.yaml").read_text())
    catalog_ids = {t["id"] for t in catalog["tools"]}
    bare_ids = {i.removeprefix("alpha_engine_") for i in catalog_ids}
    actual = mcp_tool_names()

    # MCP prefixes bare names with the server name when served
    prefixed = {f"alpha_engine_{n}" for n in actual}
    known = actual | prefixed

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

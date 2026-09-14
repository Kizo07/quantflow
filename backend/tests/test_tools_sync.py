"""Sync tool bridge: parallel callers must share one event loop.

Regression: make_sync_tool_wrapper drove each call with its own
asyncio.run() loop, so parallel MCP calls to the same server arrived on
different loops and the session pool cancelled each other's in-flight
session creations mid-initialize() — surfacing as a bare CancelledError
that kills the whole tools node.
"""

import asyncio
import sys
import threading

sys.path.insert(0, "packages/harness")

from deerflow.tools.sync import make_sync_tool_wrapper  # noqa: E402

_arrived = 0
_arrived_lock = threading.Lock()
_release = asyncio.Event()


async def _probe(results):
    global _arrived
    results.append(asyncio.get_running_loop())
    with _arrived_lock:
        _arrived += 1
        if _arrived == 4:
            _release.set()
    await asyncio.wait_for(_release.wait(), timeout=30)
    return "ok"


def test_parallel_sync_calls_share_one_loop():
    seen: list = []
    wrapped = make_sync_tool_wrapper(_probe, "probe-tool")

    errors: list = []

    def call():
        try:
            assert wrapped(seen) == "ok"
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=call) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive()

    assert not errors, errors
    assert len(seen) == 4
    assert len(set(map(id, seen))) == 1, "each call ran on its own loop"

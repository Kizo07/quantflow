"""Import guards for the native evolution loop (offline).

* No ``litellm`` import anywhere in ``deerflow.evolution`` including its
  tests: every LLM call goes through ``deerflow.models.create_chat_model``.
* No ``spark_mcp`` import anywhere in Quantflow: spark is reached over MCP
  transport only, never as a Python dependency.
* No human-in-the-loop helpers in ``deerflow.evolution``: no queues,
  prompts-for-approval, or approval waits in the loop path.
* Light-import contract: importing ``deerflow.evolution`` never pulls
  langchain, MCP transport, ``deerflow.config``, or the app layer.
"""

import ast
import subprocess
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).parent.parent
HARNESS_ROOT = BACKEND_ROOT / "packages" / "harness" / "deerflow"
EVOLUTION_ROOT = HARNESS_ROOT / "evolution"
EVOLUTION_TESTS = sorted(BACKEND_ROOT.glob("tests/test_evolution_*.py"))

BANNED_LLM_MODULES = ("litellm",)
BANNED_SPARK_MODULES = ("spark_mcp",)
BANNED_HITL_MODULES = ("multiprocessing", "queue")
BANNED_HITL_NAMES = ("ask_clarification", "user_request_q", "user_response_q", "prompt_for_approval", "wait_for_approval")
BANNED_HEAVY_MODULES = ("langchain", "litellm", "spark_mcp", "mcp", "deerflow.config", "app.")


def _collect_imports(path: Path) -> list[tuple[int, str, list[str]]]:
    """Return (lineno, module, imported_names) for every import in *path*."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return []
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append((node.lineno, alias.name, [alias.asname or alias.name.split(".")[0]]))
        elif isinstance(node, ast.ImportFrom) and node.module:
            results.append((node.lineno, node.module, [alias.name for alias in node.names]))
    return results


def _module_matches(module: str, banned: tuple[str, ...]) -> str | None:
    root = module.split(".")[0]
    for entry in banned:
        if entry.endswith("."):
            if module == entry.rstrip(".") or module.startswith(entry):
                return entry
        elif root == entry or module == entry or module.startswith(entry + "."):
            return entry
    return None


def test_evolution_package_has_no_litellm_imports():
    assert EVOLUTION_ROOT.is_dir(), f"evolution package missing: {EVOLUTION_ROOT}"
    violations = []
    for py_file in sorted(EVOLUTION_ROOT.rglob("*.py")):
        for lineno, module, _ in _collect_imports(py_file):
            if _module_matches(module, BANNED_LLM_MODULES):
                violations.append(f"  {py_file.relative_to(BACKEND_ROOT)}:{lineno}  imports {module}")
    assert not violations, "deerflow.evolution must not import litellm (use create_chat_model):\n" + "\n".join(violations)


def test_evolution_tests_have_no_litellm_imports():
    assert EVOLUTION_TESTS, "no test_evolution_*.py files found"
    violations = []
    for test_file in EVOLUTION_TESTS:
        for lineno, module, _ in _collect_imports(test_file):
            if _module_matches(module, BANNED_LLM_MODULES):
                violations.append(f"  {test_file.relative_to(BACKEND_ROOT)}:{lineno}  imports {module}")
    assert not violations, "evolution tests must not import litellm:\n" + "\n".join(violations)


def test_quantflow_has_no_spark_mcp_imports():
    roots = [BACKEND_ROOT / "packages", BACKEND_ROOT / "app", BACKEND_ROOT / "tests", BACKEND_ROOT / "scripts"]
    violations = []
    scanned = 0
    for root in roots:
        if not root.is_dir():
            continue
        for py_file in sorted(root.rglob("*.py")):
            scanned += 1
            for lineno, module, _ in _collect_imports(py_file):
                if _module_matches(module, BANNED_SPARK_MODULES):
                    violations.append(f"  {py_file.relative_to(BACKEND_ROOT)}:{lineno}  imports {module}")
    assert scanned > 0, "no backend sources found to scan"
    assert not violations, "Quantflow must never import spark_mcp (MCP transport only):\n" + "\n".join(violations)


def test_evolution_package_has_no_human_in_the_loop_helpers():
    assert EVOLUTION_ROOT.is_dir(), f"evolution package missing: {EVOLUTION_ROOT}"
    violations = []
    for py_file in sorted(EVOLUTION_ROOT.rglob("*.py")):
        for lineno, module, names in _collect_imports(py_file):
            if _module_matches(module, BANNED_HITL_MODULES):
                violations.append(f"  {py_file.relative_to(BACKEND_ROOT)}:{lineno}  imports queue helper {module}")
            for name in names:
                if name.lower() in BANNED_HITL_NAMES:
                    violations.append(f"  {py_file.relative_to(BACKEND_ROOT)}:{lineno}  imports HITL helper {name}")
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "input":
                violations.append(f"  {py_file.relative_to(BACKEND_ROOT)}:{node.lineno}  calls input()")
    assert not violations, "deerflow.evolution must not use human-in-the-loop helpers:\n" + "\n".join(violations)


def test_evolution_import_stays_light():
    """Importing deerflow.evolution must not pull langchain/MCP/config/app."""
    probe = "import sys; import deerflow.evolution; banned = sys.argv[1:]; loaded = sorted(m for m in sys.modules if any(m == b or m.startswith(b + '.') for b in banned)); print('LOADED:' + ','.join(loaded)); "
    completed = subprocess.run(
        [sys.executable, "-c", probe, *BANNED_HEAVY_MODULES],
        cwd=BACKEND_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, f"probe import failed:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    loaded_line = next((line for line in completed.stdout.splitlines() if line.startswith("LOADED:")), "LOADED:")
    loaded = [entry for entry in loaded_line.removeprefix("LOADED:").split(",") if entry]
    assert not loaded, f"importing deerflow.evolution pulled heavy modules: {loaded}"

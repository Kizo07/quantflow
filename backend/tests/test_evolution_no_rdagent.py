"""Guard: no RD-Agent dependency in the native evolution loop (offline).

Fails if the string ``rdagent`` (case-insensitive) appears in any import,
requirement file, lockfile, or test fixture under the evolution package
and its tests. The loop is a clean-room reimplementation of the loop
*concepts*; it must never import, require, or vendor RD-Agent code.

Note: prose references such as "RD-Agent" (hyphenated) in docstrings and
plans are not the ``rdagent`` module string and do not trip this guard.
"""

import ast
from pathlib import Path

BACKEND_ROOT = Path(__file__).parent.parent
HARNESS_ROOT = BACKEND_ROOT / "packages" / "harness" / "deerflow"
EVOLUTION_ROOT = HARNESS_ROOT / "evolution"
EVOLUTION_TESTS = sorted(BACKEND_ROOT.glob("tests/test_evolution_*.py"))

REQUIREMENT_GLOBS = ("requirements*.txt", "pyproject.toml", "uv.lock", "poetry.lock", "pdm.lock")


def _scan_text(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if "rdagent" in line.lower():
            hits.append(f"  {path.relative_to(BACKEND_ROOT)}:{lineno}: {line.strip()[:160]}")
    return hits


def _scan_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return []
    hits = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            if "rdagent" in name.lower():
                hits.append(f"  {path.relative_to(BACKEND_ROOT)}:{node.lineno}  imports {name}")
    return hits


def test_evolution_package_has_no_rdagent_references():
    assert EVOLUTION_ROOT.is_dir(), f"evolution package missing: {EVOLUTION_ROOT}"
    violations: list[str] = []
    for py_file in sorted(EVOLUTION_ROOT.rglob("*.py")):
        violations.extend(_scan_imports(py_file))
        violations.extend(_scan_text(py_file))
    assert not violations, "evolution package must not reference rdagent:\n" + "\n".join(violations)


def test_evolution_tests_have_no_rdagent_references():
    assert EVOLUTION_TESTS, "no test_evolution_*.py files found"
    violations: list[str] = []
    for test_file in EVOLUTION_TESTS:
        violations.extend(_scan_imports(test_file))
        if test_file.name == Path(__file__).name:
            # This guard necessarily names the string it guards against; its
            # own imports are still scanned above.
            continue
        violations.extend(_scan_text(test_file))
    assert not violations, "evolution tests must not reference rdagent:\n" + "\n".join(violations)


def test_requirements_and_lockfiles_have_no_rdagent():
    candidates: list[Path] = []
    for base in (BACKEND_ROOT, BACKEND_ROOT / "packages"):
        if base.is_dir():
            for glob in REQUIREMENT_GLOBS:
                candidates.extend(sorted(base.rglob(glob)))
    assert candidates, "no requirement/lock files found to scan"
    violations: list[str] = []
    for candidate in candidates:
        if candidate.stat().st_size > 8_000_000:
            continue
        violations.extend(_scan_text(candidate))
    assert not violations, "requirements/lockfiles must not reference rdagent:\n" + "\n".join(violations)

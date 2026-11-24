"""QuantFlow tool manifest: live smoke, offline reference check, fragment emit.

Replaces ``scripts/validate_quant_tools.py``. One manifest
(``config/quant_tools.yaml``, hand-written) and one lock
(``config/quant_tools.lock.json``, written by ``--smoke``).

  --smoke  Live, no LLM. Loads ``extensions_config.json`` through DeerFlow's own
           loader (``$VAR`` resolution included), starts every enabled server
           with the harness's connection params, runs ``initialize`` +
           ``tools/list`` and one harmless read-only call per server, then
           writes the lock (model-visible names, schema hashes, token
           estimates; no host paths, no secrets). Needs the backend venv:
           ``backend/.venv/bin/python scripts/quant_tools.py --smoke``.
  --check  Offline. Every tool name referenced by the prompt surfaces must
           exist in the lock and be bound for the agent that sees it. Prints
           per-agent tool counts and token estimates (reported, not asserted).
  --emit   Writes machine data only (subagent allowlists, routing entries,
           expose/use_when keys) from the manifest to
           ``plans/tool-awareness/fragments/emitted/``. Never edits live
           configs and never generates prose.

``--root`` points ``--check``/``--emit`` (and ``--smoke``'s config lookup) at
another checkout, e.g. the tool-awareness worktree.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import re
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
MANIFEST_REL = Path("config/quant_tools.yaml")
LOCK_REL = Path("config/quant_tools.lock.json")
FRAGMENTS_REL = Path("plans/tool-awareness/fragments")
EMIT_REL = FRAGMENTS_REL / "emitted"

TIERS = ("core", "research", "operator", "hidden")
USE_WHEN_MAX = 200
KEYWORD_MIN = 4

# One harmless, read-only call per server (original tool name, arguments).
# Servers not listed here (exa) are list-only: no paid or remote call.
SMOKE_CALLS: dict[str, tuple[str, dict[str, Any]]] = {
    "alpha_engine": ("engine_status", {}),
    "kizonlp": ("kizo_status", {}),
    "spark": ("spark_doctor", {}),
    "reportforge": ("reportforge_capabilities", {}),
}
REQUIRED_ENV = ("EXA_API_KEY",)
SMOKE_CALL_TIMEOUT = 300.0

_PREFIXED_RE = re.compile(r"(?<![\w./-])((?:alpha_engine|kizonlp|spark|reportforge)_[a-z0-9_]+|web_[a-z0-9_]+_exa)(?![\w/-])")
_UNDERSCORE_RE = re.compile(r"(?<![\w./$-])([a-z][a-z0-9]*_[a-z0-9_]*[a-z0-9])(?![\w/-])")
_BACKTICK_RE = re.compile(r"`([a-z][a-z0-9_]*)(?:\(\))?`")
_BASH_RE = re.compile(r"\bbash\b", re.IGNORECASE)
_BASH_NEGATION_RE = re.compile(r"\b(?:no|not|never|without|don't|do not|cannot|can't|nor)\b[^.\n]{0,40}$", re.IGNORECASE)
_HOST_PATH_RE = re.compile(r"(?<![\w.])/(?:home|opt|root|Users|srv|var|tmp|usr)/[^\s'\"`),;]*")
_TEXT_SUFFIXES = {".md", ".txt", ".yaml", ".yml", ".json", ".py"}


# --------------------------------------------------------------------------- data


def load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("version") != 2:
        raise SystemExit(f"{path}: expected a version 2 manifest")
    return data


def load_lock(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_data_path(root: Path, rel: Path, override: str | None) -> Path:
    """``--manifest``/``--lock`` win; else the root's copy; else this repo's.

    A root whose manifest is still version 1 (e.g. a worktree cut before the
    v2 manifest was committed) falls back to this repo's manifest.
    """
    if override:
        return Path(override)
    candidate = root / rel
    if not candidate.exists():
        return REPO / rel
    if rel == MANIFEST_REL:
        with contextlib.suppress(yaml.YAMLError, OSError, AttributeError):
            if yaml.safe_load(candidate.read_text(encoding="utf-8-sig")).get("version") != 2:
                return REPO / rel
    return candidate


@dataclass
class Catalog:
    """Manifest + lock joined into the lookups --check and --emit need."""

    manifest: dict[str, Any]
    lock: dict[str, Any] | None
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)  # visible name -> manifest entry
    builtins: dict[str, dict[str, Any]] = field(default_factory=dict)
    live: dict[str, dict[str, Any]] = field(default_factory=dict)  # visible name -> lock entry (+server)
    originals: dict[str, list[str]] = field(default_factory=dict)  # original/bare name -> visible names

    @classmethod
    def build(cls, manifest: dict[str, Any], lock: dict[str, Any] | None) -> Catalog:
        cat = cls(manifest=manifest, lock=lock)
        cat.tools = {t["name"]: t for t in manifest.get("tools", [])}
        cat.builtins = {b["name"]: b for b in manifest.get("builtins", [])}
        for server, info in ((lock or {}).get("servers") or {}).items():
            for name, entry in (info.get("tools") or {}).items():
                cat.live[name] = {**entry, "server": server}
        prefixes = cls._prefixes(manifest)
        universe = {n: {"server": t["server"]} for n, t in cat.tools.items()} | cat.live
        for name, entry in universe.items():
            original = entry.get("original") or cat.original_name(name, entry["server"], prefixes)
            for bare in {original, cls._strip_namespace(name)}:
                if bare and bare != name:
                    cat.originals.setdefault(bare, []).append(name)
        return cat

    @staticmethod
    def _prefixes(manifest: dict[str, Any]) -> dict[str, bool]:
        return {s: bool((cfg or {}).get("prefix", True)) for s, cfg in (manifest.get("servers") or {}).items()}

    @staticmethod
    def _strip_namespace(name: str) -> str:
        for ns in ("alpha_engine_", "kizonlp_", "spark_", "reportforge_"):
            if name.startswith(ns):
                return name[len(ns) :]
        return name

    def original_name(self, visible: str, server: str, prefixes: dict[str, bool] | None = None) -> str:
        live = self.live.get(visible)
        if live and live.get("original"):
            return live["original"]
        prefixes = prefixes or self._prefixes(self.manifest)
        prefix = f"{server}_"
        if prefixes.get(server, True) and visible.startswith(prefix):
            return visible[len(prefix) :]
        return visible

    @property
    def known(self) -> set[str]:
        return set(self.live or self.tools) | set(self.builtins)

    def tier(self, name: str) -> str | None:
        entry = self.tools.get(name)
        return entry.get("tier") if entry else None

    def bound_for(self, name: str, agent: str) -> tuple[bool, str]:
        """Whether the runtime can bind ``name`` for ``agent`` at all."""
        if name in self.builtins:
            scope = self.builtins[name].get("scope", "all")
            if scope == "none":
                return False, "is not bound for anyone (host bash disabled)"
            if scope == "lead" and agent != "lead":
                return False, "is lead-only; subagents never receive it"
            return True, ""
        if self.lock is not None and name not in self.live:
            return False, "is not served live (absent from the lock)"
        tier = self.tier(name)
        if tier == "hidden":
            return False, "is tier hidden (never agent-callable)"
        if tier == "operator" and agent != "lead":
            return False, "is tier operator; never in a subagent's toolset"
        return True, ""

    def tokens(self, name: str) -> int:
        if name in self.live:
            return int(self.live[name].get("tokens") or 0)
        return int(self.builtins.get(name, {}).get("tokens") or 0)

    def manifest_set(self, agent: str) -> list[str]:
        names = [b["name"] for b in self.manifest.get("builtins", []) if agent in (b.get("agents") or [])]
        names += [t["name"] for t in self.manifest.get("tools", []) if agent in (t.get("agents") or [])]
        return names


# -------------------------------------------------------------------------- check


@dataclass
class Finding:
    level: str  # "error" | "warn"
    where: str
    message: str


@dataclass
class Surface:
    where: str
    agent: str  # "lead" or a subagent name
    text: str


@dataclass
class CheckReport:
    findings: list[Finding] = field(default_factory=list)
    agent_rows: list[tuple[str, str, int, int]] = field(default_factory=list)
    scanned: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warn"]

    def _add(self, finding: Finding) -> None:
        if finding not in self.findings:
            self.findings.append(finding)

    def error(self, where: str, message: str) -> None:
        self._add(Finding("error", where, message))

    def warn(self, where: str, message: str) -> None:
        self._add(Finding("warn", where, message))


def _check_manifest(cat: Catalog, report: CheckReport) -> None:
    where = "manifest"
    roles = set(cat.manifest.get("roles") or [])
    seen: set[str] = set()
    for entry in cat.manifest.get("tools", []):
        name = entry.get("name", "?")
        if name in seen:
            report.error(where, f"{name}: duplicate entry")
        seen.add(name)
        for key in ("server", "tier", "purpose", "use_when"):
            if not entry.get(key):
                report.error(where, f"{name}: missing {key}")
        if entry.get("tier") not in TIERS:
            report.error(where, f"{name}: tier {entry.get('tier')!r} not in {TIERS}")
        if len(entry.get("use_when") or "") > USE_WHEN_MAX:
            report.error(where, f"{name}: use_when longer than {USE_WHEN_MAX} chars")
        agents = entry.get("agents") or []
        if unknown := sorted(set(agents) - roles):
            report.error(where, f"{name}: unknown roles {unknown}")
        if entry.get("tier") in ("operator", "hidden") and [a for a in agents if a != "lead"]:
            report.error(where, f"{name}: tier {entry['tier']} must not list subagent roles")
        if entry.get("tier") == "hidden" and agents:
            report.error(where, f"{name}: tier hidden must have no agents")
        routing = entry.get("routing")
        if routing:
            if entry.get("tier") in ("operator", "hidden"):
                report.error(where, f"{name}: routing on a tier {entry['tier']} tool")
            for kw in routing.get("keywords") or []:
                if len(kw) < KEYWORD_MIN:
                    report.error(where, f"{name}: routing keyword {kw!r} shorter than {KEYWORD_MIN} chars")
    for b in cat.manifest.get("builtins", []):
        if b.get("scope") == "lead" and [a for a in b.get("agents") or [] if a != "lead"]:
            report.error(where, f"builtin {b['name']}: lead-only but lists subagent roles")
    if cat.lock is None:
        report.error(where, "no lock file; run --smoke first")
        return
    lock_servers = set((cat.lock.get("servers") or {}).keys())
    for name, live in cat.live.items():
        entry = cat.tools.get(name)
        if entry is None:
            report.error(where, f"{name}: live on {live['server']} but missing from the manifest")
        elif entry["server"] != live["server"]:
            report.error(where, f"{name}: manifest server {entry['server']} != live server {live['server']}")
        elif entry["tier"] == "hidden":
            report.warn(where, f"{name}: tier hidden but still served live (hidden only once hook H1 emits expose: false)")
    for name, entry in cat.tools.items():
        if name in cat.live:
            continue
        if entry.get("tier") == "hidden":
            continue
        if entry["server"] not in lock_servers:
            report.error(where, f"{name}: server {entry['server']} is not in the lock")
        else:
            report.error(where, f"{name}: in the manifest but not served live (stale entry or renamed tool)")


def _mask(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for i in range(start, end):
            chars[i] = " "
    return "".join(chars)


def _hidden_target(cat: Catalog, token: str) -> str | None:
    """The hidden tool a token names, directly or by its bare original name."""
    for name in [token, *cat.originals.get(token, [])]:
        if cat.tier(name) == "hidden":
            return name
    return None


def _suggest(cat: Catalog, token: str) -> str | None:
    retired = cat.manifest.get("retired") or {}
    if token in retired:
        return retired[token]
    hits = cat.originals.get(token)
    if hits:
        return " or ".join(sorted(set(hits)))
    return None


def _scan_names(cat: Catalog, surface: Surface, report: CheckReport) -> set[str]:
    """Flag unknown, stale and unprefixed names; return the valid names mentioned."""
    ignore = set(cat.manifest.get("ignore_tokens") or [])
    known = cat.known
    mentioned: set[str] = set()
    spans: list[tuple[int, int]] = []
    for match in _PREFIXED_RE.finditer(surface.text):
        token = match.group(1)
        spans.append(match.span(1))
        if token in ignore:
            continue
        if token in known:
            mentioned.add(token)
            continue
        if hidden := _hidden_target(cat, token):
            report.error(surface.where, f"{token}: names {hidden}, tier hidden (never agent-callable); remove the reference")
            continue
        hint = _suggest(cat, token)
        report.error(surface.where, f"{token}: not a live tool name" + (f"; use {hint}" if hint else ""))
    rest = _mask(surface.text, spans)
    bare_tokens = {m.group(1) for m in _UNDERSCORE_RE.finditer(rest)} | {m.group(1) for m in _BACKTICK_RE.finditer(rest)}
    retired = cat.manifest.get("retired") or {}
    for token in sorted(bare_tokens):
        if token in ignore:
            continue
        if token in known:
            mentioned.add(token)
            continue
        if (hidden := _hidden_target(cat, token)) and token not in retired:
            report.error(surface.where, f"{token}: names {hidden}, tier hidden (never agent-callable); remove the reference")
        elif token in retired or token in cat.originals:
            report.error(surface.where, f"{token}: unprefixed/stale name; use {_suggest(cat, token)}")
    return mentioned


def _all_negated(text: str, token: str) -> bool:
    """True when every occurrence of ``token`` sits in a "never/do not ..." clause."""
    hits = list(re.finditer(rf"(?<![\w]){re.escape(token)}(?![\w])", text))
    if not hits:
        return False
    for match in hits:
        before = text[max(0, match.start() - 60) : match.start()]
        before = before[before.rfind("\n") + 1 :]
        if not _BASH_NEGATION_RE.search(before):
            return False
    return True


def _check_mentions(cat: Catalog, surface: Surface, mentioned: set[str], allowlist: list[str] | None, report: CheckReport) -> None:
    for name in sorted(mentioned):
        tier = cat.tier(name)
        if surface.agent == "lead":
            if tier == "hidden" or (name in cat.builtins and cat.builtins[name].get("scope") == "none"):
                if _all_negated(surface.text, name):
                    report.warn(surface.where, f"{name}: never agent-callable; mentioned only as a prohibition")
                else:
                    report.error(surface.where, f"{name}: mentioned but never agent-callable")
            elif tier == "operator":
                report.warn(surface.where, f"{name}: operator-tier tool mentioned in a lead surface (hidden in Phase 3)")
            elif cat.lock is not None and name not in cat.builtins and name not in cat.live:
                report.error(surface.where, f"{name}: not served live")
            continue
        ok, why = cat.bound_for(name, surface.agent)
        if not ok:
            report.error(surface.where, f"{name}: mentioned in the {surface.agent} prompt but {why}")
        elif allowlist is not None and name not in allowlist:
            report.error(surface.where, f"{name}: mentioned in the {surface.agent} prompt but not in its tools: allowlist")


def _check_bash(surface: Surface, report: CheckReport) -> None:
    for match in _BASH_RE.finditer(surface.text):
        before = surface.text[max(0, match.start() - 60) : match.start()]
        line_start = before.rfind("\n")
        before = before[line_start + 1 :]
        if _BASH_NEGATION_RE.search(before):
            continue
        snippet = surface.text[max(0, match.start() - 30) : match.end() + 30].replace("\n", " ")
        report.error(surface.where, f"'bash' instruction in a subagent prompt (no bash tool is bound): ...{snippet.strip()}...")


def _check_allowlist(cat: Catalog, agent: str, where: str, allowlist: list[str], report: CheckReport) -> None:
    intended = set(cat.manifest_set(agent))
    for name in allowlist:
        if name not in cat.known:
            hint = _suggest(cat, name)
            report.error(where, f"tools: {name} is not a live tool" + (f"; use {hint}" if hint else ""))
            continue
        ok, why = cat.bound_for(name, agent)
        if not ok:
            report.error(where, f"tools: {name} {why}")
        elif intended and name not in intended:
            report.warn(where, f"tools: {name} is not in the manifest set for {agent}")
    if intended and (missing := sorted(intended - set(allowlist))):
        report.warn(where, f"tools: manifest set for {agent} also has {', '.join(missing)}")


def _routing_blocks(data: Any) -> dict[str, Any]:
    """Accept {"mcpServers": {...}} or a bare {server: {...}} mapping."""
    if not isinstance(data, dict):
        return {}
    if isinstance(data.get("mcpServers"), dict):
        return data["mcpServers"]
    return {k: v for k, v in data.items() if isinstance(v, dict) and ("tools" in v or "routing" in v)}


def _check_routing(cat: Catalog, where: str, data: Any, report: CheckReport) -> None:
    lock_servers = (cat.lock or {}).get("servers") or {}
    for server, block in _routing_blocks(data).items():
        server_routing = (block.get("routing") or {}) if isinstance(block, dict) else {}
        if server_routing.get("mode") == "prefer":
            report.warn(where, f"{server}: server-level routing applies to every tool on the server")
        tools = (block.get("tools") or {}) if isinstance(block, dict) else {}
        if not tools:
            continue
        if cat.lock is not None and server not in lock_servers:
            report.error(where, f"{server}: routing for a server not in the lock")
            continue
        originals = {cat.original_name(n, server): n for n, e in cat.live.items() if e["server"] == server}
        if cat.lock is None:
            originals = {cat.original_name(n, server): n for n, t in cat.tools.items() if t["server"] == server}
        for key, override in tools.items():
            visible = originals.get(key)
            if visible is None:
                hint = _suggest(cat, key)
                if key in cat.known and key not in originals:
                    hint = cat.original_name(key, server)
                report.error(where, f"{server}.tools.{key}: not an original tool name on {server}" + (f"; use {hint}" if hint else ""))
                continue
            routing = (override or {}).get("routing") or {}
            if routing.get("mode") == "prefer" and cat.tier(visible) in ("operator", "hidden"):
                report.error(where, f"{server}.tools.{key}: routing prefers a tier {cat.tier(visible)} tool")
            for kw in routing.get("keywords") or []:
                if len(str(kw)) < KEYWORD_MIN:
                    report.error(where, f"{server}.tools.{key}: routing keyword {kw!r} shorter than {KEYWORD_MIN} chars")


def _subagent_blocks(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    if isinstance(data.get("subagents"), dict):
        data = data["subagents"]
    if isinstance(data.get("custom_agents"), dict):
        data = data["custom_agents"]
    return {k: v for k, v in data.items() if isinstance(v, dict) and ("system_prompt" in v or "tools" in v)}


def _text_files(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    if not base.is_dir():
        return []
    out = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or ".bak" in path.name or path.suffix not in _TEXT_SUFFIXES:
            continue
        out.append(path)
    return out


def _soul_path(root: Path) -> Path:
    home = os.environ.get("DEER_FLOW_HOME")
    return Path(home) / "SOUL.md" if home else root / "backend" / ".deer-flow" / "SOUL.md"


def run_check(root: Path, manifest_path: Path | None = None, lock_path: Path | None = None) -> CheckReport:
    """Validate all prompt surfaces under ``root`` against the manifest + lock."""
    manifest_path = manifest_path or _resolve_data_path(root, MANIFEST_REL, None)
    lock_path = lock_path or _resolve_data_path(root, LOCK_REL, None)
    cat = Catalog.build(load_manifest(manifest_path), load_lock(lock_path))
    report = CheckReport()
    _check_manifest(cat, report)

    surfaces: list[tuple[Surface, list[str] | None]] = []
    allowlists: dict[str, tuple[str, list[str] | None]] = {}

    def rel(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    def add_subagents(blocks: dict[str, Any], origin: str) -> None:
        for agent, cfg in blocks.items():
            tools = cfg.get("tools")
            allow = [str(t) for t in tools] if isinstance(tools, list) else None
            if allow is not None:
                _check_allowlist(cat, agent, f"{origin}:{agent}.tools", allow, report)
            allowlists[f"{origin}:{agent}"] = (agent, allow)
            if cfg.get("system_prompt"):
                surfaces.append((Surface(f"{origin}:{agent}.system_prompt", agent, str(cfg["system_prompt"])), allow))
            if cfg.get("description"):
                surfaces.append((Surface(f"{origin}:{agent}.description", "lead", str(cfg["description"])), None))

    config_path = root / "config.yaml"
    if config_path.is_file():
        report.scanned.append(rel(config_path))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
        overlay = config.get("lead_prompt_overlay") or {}
        if isinstance(overlay, dict):
            for key in ("prepend", "append"):
                if overlay.get(key):
                    surfaces.append((Surface(f"config.yaml:lead_prompt_overlay.{key}", "lead", str(overlay[key])), None))
        add_subagents(_subagent_blocks(config), "config.yaml")
        roles = set(cat.manifest.get("roles") or []) - {"lead"}
        configured = set(_subagent_blocks(config))
        if configured and configured != roles:
            report.warn("config.yaml", f"subagents {sorted(configured)} differ from manifest roles {sorted(roles)}")

    soul = _soul_path(root)
    if soul.is_file():
        report.scanned.append(rel(soul))
        surfaces.append((Surface(rel(soul), "lead", soul.read_text(encoding="utf-8-sig")), None))

    for skills_dir in sorted((root / "skills").glob("*/quant-desk")):
        for path in _text_files(skills_dir):
            report.scanned.append(rel(path))
            surfaces.append((Surface(rel(path), "lead", path.read_text(encoding="utf-8-sig")), None))

    ext_path = root / "extensions_config.json"
    if ext_path.is_file():
        report.scanned.append(rel(ext_path))
        try:
            ext = json.loads(ext_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            report.error(rel(ext_path), f"not valid JSON: {exc}")
        else:
            _check_routing(cat, rel(ext_path), ext, report)

    frag_dir = root / FRAGMENTS_REL
    for path in _text_files(frag_dir):
        where = rel(path)
        report.scanned.append(where)
        text = path.read_text(encoding="utf-8-sig")
        data: Any = None
        if path.suffix in (".yaml", ".yml", ".json"):
            try:
                data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
            except (json.JSONDecodeError, yaml.YAMLError) as exc:
                report.error(where, f"unparseable: {exc}")
                continue
        blocks = _subagent_blocks(data)
        if blocks:
            add_subagents(blocks, where)
            continue
        if _routing_blocks(data):
            # routing.json / presentation.json: keys are original names.
            _check_routing(cat, where, data, report)
            continue
        # The emitted allowlist file is {agent: [names]}.
        if isinstance(data, dict) and data and all(isinstance(v, list) for v in data.values()):
            for agent, names in data.items():
                _check_allowlist(cat, str(agent), f"{where}:{agent}", [str(n) for n in names], report)
                allowlists[f"{where}:{agent}"] = (str(agent), [str(n) for n in names])
            continue
        if data is None or isinstance(data, str):
            surfaces.append((Surface(where, "lead", text), None))
        else:
            # Structured fragment (presentation.json etc.): names only.
            surfaces.append((Surface(where, "lead", json.dumps(data)), None))

    for surface, allow in surfaces:
        mentioned = _scan_names(cat, surface, report)
        _check_mentions(cat, surface, mentioned, allow, report)
        if surface.agent != "lead":
            _check_bash(surface, report)

    # Token/tool-count report: what each agent would be bound to.
    # Without an allowlist a subagent inherits every live MCP tool (operator
    # tier included until hook H1) plus the shared built-ins.
    live_mcp = list(cat.live) if cat.lock is not None else [n for n, t in cat.tools.items() if t.get("tier") != "hidden"]
    lead_tools = live_mcp + [n for n, b in cat.builtins.items() if b.get("scope") in ("all", "lead")]
    report.agent_rows.append(("lead", "bound at runtime", len(lead_tools), sum(cat.tokens(n) for n in lead_tools)))
    sub_all = live_mcp + [n for n, b in cat.builtins.items() if b.get("scope") == "all"]
    for origin, (agent, allow) in sorted(allowlists.items()):
        names = [n for n in allow if cat.bound_for(n, agent)[0]] if allow is not None else sub_all
        report.agent_rows.append((agent, origin + (" (allowlist)" if allow is not None else " (inherits all)"), len(names), sum(cat.tokens(n) for n in names)))
    for role in cat.manifest.get("roles") or []:
        names = cat.manifest_set(role)
        report.agent_rows.append((role, "manifest set", len(names), sum(cat.tokens(n) for n in names)))
    return report


def print_check(report: CheckReport) -> None:
    print(f"scanned {len(report.scanned)} surface files")
    for level in ("error", "warn"):
        items = [f for f in report.findings if f.level == level]
        if not items:
            continue
        print(f"\n{level.upper()}S ({len(items)})")
        by_where: dict[str, list[str]] = {}
        for f in items:
            by_where.setdefault(f.where, []).append(f.message)
        for where, msgs in by_where.items():
            print(f"  {where}")
            for msg in msgs:
                print(f"    - {msg}")
    print("\nPer-agent tools (token estimates: o200k over the OpenAI tool schema; reported, not asserted)")
    print(f"  {'agent':22s} {'source':58s} {'tools':>5s} {'~tokens':>8s}")
    for agent, source, count, tokens in report.agent_rows:
        print(f"  {agent:22s} {source[:58]:58s} {count:5d} {tokens:8d}")
    print(f"\n{len(report.errors)} error(s), {len(report.warnings)} warning(s)")


# --------------------------------------------------------------------------- emit


def run_emit(root: Path, manifest_path: Path, lock_path: Path) -> list[Path]:
    cat = Catalog.build(load_manifest(manifest_path), load_lock(lock_path))
    out_dir = root / EMIT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    roles = [r for r in cat.manifest.get("roles") or [] if r != "lead"]
    allowlists = {role: cat.manifest_set(role) for role in roles}
    routing: dict[str, dict[str, Any]] = {}
    presentation: dict[str, dict[str, Any]] = {}
    for entry in cat.manifest.get("tools", []):
        server, name = entry["server"], entry["name"]
        if cat.lock is not None and name not in cat.live:
            continue  # not served (e.g. removed by a server profile): nothing to route or hide
        original = cat.original_name(name, server)
        if entry.get("routing"):
            routing.setdefault(server, {"tools": {}})["tools"][original] = {
                "routing": {"mode": "prefer", "priority": int(entry["routing"]["priority"]), "keywords": list(entry["routing"]["keywords"])},
            }
        pres: dict[str, Any] = {}
        if entry["tier"] in ("operator", "hidden"):
            pres["expose"] = False
        else:
            pres["use_when"] = entry["use_when"]
        presentation.setdefault(server, {"tools": {}})["tools"][original] = pres
    files = {
        out_dir / "subagent_tools.yaml": yaml.safe_dump(allowlists, sort_keys=False, width=200),
        out_dir / "routing.json": json.dumps({"mcpServers": routing}, indent=2) + "\n",
        out_dir / "presentation.json": json.dumps({"mcpServers": presentation}, indent=2) + "\n",
    }
    header = "# Generated by scripts/quant_tools.py --emit from config/quant_tools.yaml. Machine data only.\n"
    for path, body in files.items():
        path.write_text(header + body if path.suffix == ".yaml" else body, encoding="utf-8")
    return list(files)


# -------------------------------------------------------------------------- smoke


def _scrub(text: str) -> str:
    return _HOST_PATH_RE.sub("<host-path>", text)


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def _descendants(pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(int(stat.parent.name))
        except (OSError, IndexError, ValueError):
            continue
    out, stack = [], [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            out.append(child)
            stack.append(child)
    return out


def _kill_descendants() -> list[int]:
    victims = _descendants(os.getpid())
    for sig in (signal.SIGTERM, signal.SIGKILL):
        alive = []
        for pid in victims:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, sig)
                alive.append(pid)
        if not alive:
            break
        time.sleep(1.0)
        victims = [p for p in alive if Path(f"/proc/{p}").exists()]
    return victims


@contextlib.contextmanager
def _stderr_to(path: Path):
    """Send fd 2 (inherited by stdio MCP children) to a log file."""
    sys.stderr.flush()
    saved = os.dup(2)
    with open(path, "ab") as fh:
        os.dup2(fh.fileno(), 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)


def _digest(text: str) -> str:
    """Short, path-free summary of a smoke call result."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return _scrub(text.strip().splitlines()[0][:120]) if text.strip() else "(empty)"
    if not isinstance(data, dict):
        return type(data).__name__
    picks = []
    for key, value in data.items():
        if re.search(r"mode|status|ok|healthy|version|degraded|llm", key, re.I) and isinstance(value, (str, int, float, bool)):
            picks.append(f"{key}={_scrub(str(value))[:60]}")
        elif isinstance(value, dict):
            for sub, sv in value.items():
                if re.search(r"mode|degraded|provider|as_of", sub, re.I) and isinstance(sv, (str, int, float, bool)):
                    picks.append(f"{key}.{sub}={_scrub(str(sv))[:60]}")
    return ", ".join(picks[:6]) or f"keys: {', '.join(list(data)[:8])}"


async def _probe_server(name: str, server_cfg: Any, connection: dict[str, Any], workdir: Path, log: Path) -> dict[str, Any]:
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_mcp_adapters.sessions import create_session
    from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool

    try:
        import tiktoken

        encoder = tiktoken.get_encoding("o200k_base")
        count = lambda s: len(encoder.encode(s))  # noqa: E731
    except Exception:  # offline or not installed: rough estimate
        count = lambda s: len(s) // 4  # noqa: E731

    connection = dict(connection)
    if connection.get("transport", "stdio") == "stdio" and not connection.get("cwd"):
        connection["cwd"] = str(workdir)
    init_timeout = float(server_cfg.session_init_timeout or 120)
    prefix = bool(server_cfg.tool_name_prefix)
    result: dict[str, Any] = {"ok": False}
    started = time.monotonic()
    with _stderr_to(log):
        async with create_session(connection) as session:
            init = await asyncio.wait_for(session.initialize(), init_timeout)
            tools = []
            cursor = None
            while True:
                page = await asyncio.wait_for(session.list_tools(cursor), init_timeout)
                tools.extend(page.tools)
                cursor = page.nextCursor
                if not cursor:
                    break
            entries: dict[str, Any] = {}
            for tool in tools:
                lc_tool = convert_mcp_tool_to_langchain_tool(None, tool, connection=connection, server_name=name, tool_name_prefix=prefix)
                spec = convert_to_openai_tool(lc_tool)
                description = (tool.description or "").strip()
                first = description.splitlines()[0] if description else ""
                entries[lc_tool.name] = {
                    "original": tool.name,
                    "schema_sha256": _sha(tool.inputSchema),
                    "description_sha256": _sha(description),
                    "summary": _scrub(first[:200]),
                    "tokens": count(json.dumps(spec)),
                }
            result.update(
                server_name=init.serverInfo.name,
                version=init.serverInfo.version,
                instructions_sha256=_sha(init.instructions or ""),
                tools=dict(sorted(entries.items())),
            )
            call = SMOKE_CALLS.get(name)
            if call is not None:
                original, args = call
                if original not in {t.name for t in tools}:
                    raise RuntimeError(f"smoke tool {original!r} not served")
                timeout = min(float(server_cfg.tool_call_timeout or SMOKE_CALL_TIMEOUT), SMOKE_CALL_TIMEOUT)
                res = await asyncio.wait_for(session.call_tool(original, args), timeout)
                text = "".join(getattr(c, "text", "") or "" for c in res.content)
                if res.isError:
                    raise RuntimeError(f"{original} returned isError: {_scrub(text[:200])}")
                result["call"] = f"{original}: {_digest(text)}"
            else:
                result["call"] = "list only"
    result["ok"] = True
    result["seconds"] = round(time.monotonic() - started, 1)
    return result


def _interceptor_status(ext: Any) -> list[str]:
    from deerflow.mcp.config_normalization import normalize_mcp_interceptor_paths
    from deerflow.reflection import resolve_variable

    lines = []
    for path in normalize_mcp_interceptor_paths((ext.model_extra or {}).get("mcpInterceptors")):
        try:
            resolve_variable(path)
            lines.append(f"interceptor {path}: importable (not loaded by smoke; calls carry no paths)")
        except Exception as exc:  # report, never fatal
            lines.append(f"WARN interceptor {path}: not importable ({type(exc).__name__}); gateway would skip it")
    return lines


def run_smoke(root: Path, lock_path: Path, manifest_path: Path) -> int:
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        print(f"PREFLIGHT FAIL: {', '.join(missing)} is unset in this environment. Export it from the shell profile (never print it) and retry.")
        return 2
    try:
        from deerflow.config.extensions_config import ExtensionsConfig
        from deerflow.mcp.client import build_server_params
    except ImportError as exc:
        print(f"deerflow is not importable ({exc}); run with backend/.venv/bin/python")
        return 2
    ext_path = root / "extensions_config.json"
    try:
        ext = ExtensionsConfig.from_file(str(ext_path))
    except Exception as exc:  # message may embed resolved values: print the type only
        print(f"cannot load {ext_path.name}: {type(exc).__name__}")
        return 2
    for line in _interceptor_status(ext):
        print(line)
    servers = ext.get_enabled_mcp_servers()
    manifest = load_manifest(manifest_path)
    expected: dict[str, int] = {}  # all manifest entries per server (hidden ones may be absent live)
    for t in manifest.get("tools", []):
        expected[t["server"]] = expected.get(t["server"], 0) + 1
    results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="qf-smoke-") as tmp:
        tmpdir = Path(tmp)

        async def main() -> None:
            for name, cfg in servers.items():
                log = tmpdir / f"{name}.stderr.log"
                try:
                    connection = build_server_params(name, cfg)
                    results[name] = await _probe_server(name, cfg, connection, tmpdir, log)
                except BaseException as exc:  # noqa: BLE001 - report every server
                    if isinstance(exc, KeyboardInterrupt):
                        raise
                    detail = type(exc).__name__
                    if isinstance(exc, (RuntimeError, ValueError)) and str(exc):
                        detail += f": {_scrub(str(exc))[:200]}"
                    tail = log.read_text(errors="replace").strip().splitlines()[-8:] if log.exists() else []
                    results[name] = {"ok": False, "error": detail, "stderr_tail": [_scrub(x)[:200] for x in tail]}

        try:
            asyncio.run(main())
        finally:
            leftovers = _kill_descendants()
    print(f"\n{'server':14s} {'status':6s} {'tools':>5s} {'manif.':>6s} {'~tokens':>8s} {'secs':>6s}  call")
    failed = False
    for name in servers:
        r = results.get(name, {"ok": False, "error": "not run"})
        if r["ok"]:
            n = len(r["tools"])
            tok = sum(t["tokens"] for t in r["tools"].values())
            print(f"{name:14s} {'OK':6s} {n:5d} {expected.get(name, 0):6d} {tok:8d} {r['seconds']:6.1f}  {r['call']}")
        else:
            failed = True
            print(f"{name:14s} {'FAIL':6s} {'-':>5s} {expected.get(name, 0):6d} {'-':>8s} {'-':>6s}  {r.get('error')}")
            for line in r.get("stderr_tail", []):
                print(f"{'':16s}| {line}")
    for server in sorted(set(expected) - set(servers)):
        failed = True
        print(f"{server:14s} {'ABSENT':6s} manifest expects {expected[server]} tools; server not enabled in {ext_path.name}")
    manifest_names = {t["name"]: t for t in manifest.get("tools", [])}
    for name, r in results.items():
        if not r["ok"]:
            continue
        new = sorted(set(r["tools"]) - set(manifest_names))
        gone = sorted(n for n, t in manifest_names.items() if t["server"] == name and t["tier"] != "hidden" and n not in r["tools"])
        live_hidden = sorted(n for n in r["tools"] if manifest_names.get(n, {}).get("tier") == "hidden")
        if new:
            print(f"DRIFT {name}: live but not in the manifest: {', '.join(new)}")
        if gone:
            print(f"DRIFT {name}: in the manifest but not live: {', '.join(gone)}")
        if live_hidden:
            print(f"note  {name}: tier-hidden tools still served (need hook H1 or a server profile): {', '.join(live_hidden)}")
    if leftovers:
        print(f"WARN: {len(leftovers)} child process(es) survived SIGKILL: {leftovers}")
    if failed:
        print("\nsmoke FAILED; lock not written")
        return 1
    lock = {
        "version": 1,
        "generated_by": "scripts/quant_tools.py --smoke",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "servers": {
            name: {
                "server_name": r["server_name"],
                "version": r["version"],
                "tool_name_prefix": bool(servers[name].tool_name_prefix),
                "instructions_sha256": r["instructions_sha256"],
                "tool_count": len(r["tools"]),
                "tools": r["tools"],
            }
            for name, r in sorted(results.items())
        },
    }
    lock_path.write_text(json.dumps(lock, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"\nwrote {lock_path} ({sum(s['tool_count'] for s in lock['servers'].values())} tools)")
    return 0


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="live MCP handshake + one read-only call per server; writes the lock")
    mode.add_argument("--check", action="store_true", help="offline: validate prompt surfaces against manifest + lock")
    mode.add_argument("--emit", action="store_true", help="write allowlists/routing/presentation fragments from the manifest")
    parser.add_argument("--root", default=str(REPO), help="checkout to read configs/skills/fragments from (default: this repo)")
    parser.add_argument("--manifest", help="manifest path (default: <root>/config/quant_tools.yaml, else this repo's)")
    parser.add_argument("--lock", help="lock path (default: <root>/config/quant_tools.lock.json, else this repo's)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    manifest = _resolve_data_path(root, MANIFEST_REL, args.manifest)
    lock = _resolve_data_path(root, LOCK_REL, args.lock)
    if args.smoke:
        lock_out = Path(args.lock) if args.lock else REPO / LOCK_REL
        return run_smoke(root, lock_out, manifest)
    if args.check:
        print(f"root {root}\nmanifest {manifest}\nlock {lock}{'' if lock.exists() else ' (missing)'}")
        report = run_check(root, manifest, lock)
        print_check(report)
        return 1 if report.errors else 0
    for path in run_emit(root, manifest, lock):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

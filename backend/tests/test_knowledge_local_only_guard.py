"""Phase 0b local-only guard: no external knowledge adapters by default.

Proves through the REAL resolution path -- ``providers._configured_tools``
(via ``get_knowledge_capabilities`` / ``resolve_knowledge_provider``) plus
the ``KnowledgeBaseConfig.enabled`` default -- that a default ``AppConfig``
and the shipped example configs expose zero external knowledge adapters
(ragflow / lightrag / weknora) and leave ``knowledge_base`` disabled unless
the operator opts in.

Style precedent: ``tests/test_knowledge_provider.py`` (minimal AppConfig
with stubbed sandbox, ``Path(__file__).parents[2]`` for the repo root).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from deerflow.config.app_config import AppConfig
from deerflow.config.knowledge_base_config import KnowledgeBaseConfig
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.config.tool_config import ToolConfig
from deerflow.extensions import ExtensionRegistry
from deerflow.knowledge.providers import (
    GENERIC_KNOWLEDGE_TOOL,
    GENERIC_KNOWLEDGE_TOOLS,
    KNOWLEDGE_TOOL_OPERATIONS,
    LIGHTRAG_KNOWLEDGE_TOOL,
    RAGFLOW_KNOWLEDGE_TOOL,
    get_knowledge_capabilities,
    resolve_knowledge_provider,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIGS = [REPO_ROOT / "config.yaml", REPO_ROOT / "config.example.yaml"]

# Provider registration names that would resolve to an external (non-fork)
# knowledge backend through the generic tool's `provider:` extra. Weknora has
# no legacy tool path; it arrives via the extension registry under this name.
EXTERNAL_PROVIDER_NAMES = frozenset({"ragflow", "lightrag", "weknora"})
EXTERNAL_TOOL_USES = frozenset({RAGFLOW_KNOWLEDGE_TOOL, LIGHTRAG_KNOWLEDGE_TOOL})


def _default_config() -> AppConfig:
    """Minimal AppConfig with every knowledge default intact.

    ``sandbox`` is a required field unrelated to knowledge; it is stubbed
    exactly as in ``test_knowledge_provider.py`` so the knowledge defaults
    under test (``tools=[]``, ``knowledge_base``) are the real ones.
    """
    return AppConfig(sandbox=SandboxConfig(use="example.remote:Sandbox"))


def _active_knowledge_tool_uses(config: AppConfig) -> list[tuple[str, str | None, str | None]]:
    """Active (name, use, provider-extra) for each configured knowledge tool."""
    found = []
    for name in KNOWLEDGE_TOOL_OPERATIONS:
        tool = config.get_tool_config(name)
        if tool is not None:
            found.append((name, tool.use, (tool.model_extra or {}).get("provider")))
    return found


def test_knowledge_base_enabled_defaults_false():
    assert KnowledgeBaseConfig().enabled is False
    assert KnowledgeBaseConfig().scope_selection_enabled is False
    default = _default_config()
    assert default.knowledge_base.enabled is False
    assert default.knowledge_base.scope_selection_enabled is False


def test_default_config_registers_zero_external_adapters():
    """Resolution outcome on defaults: no knowledge tools, empty caps, LookupError."""
    config = _default_config()
    assert _active_knowledge_tool_uses(config) == []
    assert get_knowledge_capabilities(config).operations == frozenset()
    with pytest.raises(LookupError, match="not configured"):
        resolve_knowledge_provider(config)


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS, ids=[p.name for p in EXAMPLE_CONFIGS])
def test_example_config_registers_zero_external_adapters(config_path):
    """Resolution outcome on shipped example configs: same local-only result.

    Validated with ``AppConfig.model_validate`` on the parsed YAML rather
    than ``AppConfig.from_file``: ``from_file`` writes process-wide singleton
    configs (checkpointer, knowledge, extensions, ...), which a guard test
    must not mutate as a side effect.
    """
    assert config_path.exists(), f"shipped example config missing: {config_path}"
    config = AppConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    assert config.knowledge_base.enabled is False
    active = _active_knowledge_tool_uses(config)
    for name, use, provider_name in active:
        assert use not in EXTERNAL_TOOL_USES, f"{config_path.name}: {name} uses external adapter {use}"
        if use in GENERIC_KNOWLEDGE_TOOLS.values():
            assert provider_name not in EXTERNAL_PROVIDER_NAMES, f"{config_path.name}: {name} resolves to external provider {provider_name!r}"
    assert get_knowledge_capabilities(config).operations == frozenset()
    with pytest.raises(LookupError, match="not configured"):
        resolve_knowledge_provider(config)


def test_exactly_one_provider_once_quantflow_pg_exists():
    """Fork ships exactly one knowledge provider (Phase 2b hardening).

    With default config + fork backends bound, provider resolution yields
    exactly {quantflow-pg}: a fresh registry holding the fork provider
    exposes no other names (in particular no ragflow/lightrag/weknora),
    and the REAL resolution path (``get_knowledge_capabilities`` /
    ``resolve_knowledge_provider``) against a default AppConfig plus the
    fork tool entry resolves to that provider with the six fork operations.

    Pure defaults expose zero tools (see above); the fork tool entry is
    the minimal operator opt-in on top of those defaults. Uses an explicit
    extension snapshot (no global singleton mutation).

    Skipped until ``deerflow.knowledge.quantflow_provider`` exists (Phase 2a).
    Contract source: ``backend/docs/KNOWLEDGE.md`` fork section.
    """
    qp_module = pytest.importorskip(
        "deerflow.knowledge.quantflow_provider",
        reason="Phase 2: Quantflow fork provider not implemented yet",
    )
    provider_name = getattr(qp_module, "PROVIDER_NAME", "quantflow-pg")
    assert provider_name == "quantflow-pg"
    assert provider_name not in EXTERNAL_PROVIDER_NAMES

    # Fork backends bound: offline provider instance, no live credentials.
    provider = qp_module.QuantflowKnowledgeProvider()
    expected_ops = frozenset({"search", "sources", "catalog", "documents", "dataset_filter", "document_filter"})
    assert provider.capabilities.operations == expected_ops

    registry = ExtensionRegistry()
    with registry.attributed_to("test:fork"):
        registry.knowledge_provider(provider_name, provider)
    loaded = registry.build()
    names = {name for _, (name, _) in loaded.knowledge_providers}
    assert names == {"quantflow-pg"}
    assert not (names & EXTERNAL_PROVIDER_NAMES)

    # Default config + fork tool entry through the REAL resolution path.
    config = AppConfig(
        tools=[ToolConfig(name="knowledge_search", group="knowledge", use=GENERIC_KNOWLEDGE_TOOL, provider=provider_name)],
        sandbox=SandboxConfig(use="example.remote:Sandbox"),
        knowledge_base=KnowledgeBaseConfig(enabled=True, scope_selection_enabled=True),
    )
    assert _active_knowledge_tool_uses(config) == [("knowledge_search", GENERIC_KNOWLEDGE_TOOL, provider_name)]
    assert get_knowledge_capabilities(config, extensions=loaded).operations == expected_ops
    assert resolve_knowledge_provider(config, extensions=loaded) is provider

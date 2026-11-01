"""Native Quantflow evolution loop (closed R&D iteration, no human-in-the-loop).

This package ports RD-Agent's closed evolution loop concepts
(Hypothesis → Experiment → Feedback → reselect, with a Trace DAG,
checkpoint/SOTA selectors, and scheduler-driven iteration) as a clean-room
native reimplementation in Quantflow style: frozen dataclasses, store
Protocols, no-I/O modules.

Submodules:

* :mod:`deerflow.evolution.errors` — typed errors (``EvolutionError`` base).
* :mod:`deerflow.evolution.models` — ``Hypothesis`` / ``ExperimentSpec`` /
  ``ExperimentFeedback`` / ``HypothesisFeedback`` / ``EvoStep``.
* :mod:`deerflow.evolution.trace` — ``Trace`` DAG + ``TraceStore`` Protocol
  + ``InMemoryTraceStore``.
* :mod:`deerflow.evolution.selectors` — checkpoint/SOTA selector Protocols
  and deterministic built-ins.
* :mod:`deerflow.evolution.config` — ``EvolutionConfig`` (standalone; also
  registered as the ``evolution:`` section of ``AppConfig``).
* :mod:`deerflow.evolution.prompts` — static versioned prompt templates.
* :mod:`deerflow.evolution.loop` — ``EvolutionLoop`` async runner with
  stage Protocols and idempotency-key derivation.
* :mod:`deerflow.evolution.spark_client` — typed MCP client for trial
  governance (lazy: import via ``deerflow.evolution.spark_client``).
* :mod:`deerflow.evolution.knowledge_adapter` — Knowledge Plane binding
  (lazy: import via ``deerflow.evolution.knowledge_adapter``).
* :mod:`deerflow.evolution.memory_adapter` — agent-memory binding
  (lazy: import via ``deerflow.evolution.memory_adapter``).
* :mod:`deerflow.evolution.stages` — stage implementations over the
  adapters (lazy: import via ``deerflow.evolution.stages``).
* :mod:`deerflow.evolution.sandboxing` — Quantflow sandbox adaptation
  (canonical workdir, provider manifest, artifact harvest; lazy).
* :mod:`deerflow.evolution.wiring` — composition root and entry points
  (lazy: import via ``deerflow.evolution.wiring``).

Import-cycle contract (mirrors ``deerflow.knowledge``): ``deerflow.config``
imports ``deerflow.evolution.config``, so importing this package must never
pull in ``deerflow.config``, langchain, or MCP transport. The eagerly
imported modules below depend only on the stdlib, pydantic, and each other
(plus ``deerflow.knowledge.hashing`` for content identity). Stage bindings
and adapters (later components) must load lazily via :func:`__getattr__`.
"""

from __future__ import annotations

import importlib
from typing import Any

from deerflow.evolution import config, errors, loop, models, prompts, selectors, trace
from deerflow.evolution.config import (
    ENV_DEFAULT_FAMILY,
    ENV_DEFAULT_UNIVERSE,
    ENV_ENABLED,
    ENV_MAX_ITERATIONS,
    ENV_MAX_WALL_CLOCK_SECONDS,
    ENV_SANDBOX_TIMEOUT,
    MODEL_ROLES,
    EvolutionConfig,
    get_evolution_config,
    load_evolution_config_from_dict,
    set_evolution_config,
)
from deerflow.evolution.errors import (
    BudgetExhaustedError,
    DuplicateRunError,
    EvolutionError,
    GateFailedError,
    GovernanceRefusalError,
)
from deerflow.evolution.loop import (
    BACKTEST_SEQUENCE_BASE,
    BEGIN_SEQUENCE_BASE,
    COMMIT_SEQUENCE_BASE,
    FAILURE_SEQUENCE_BASE,
    OUTCOME_SEQUENCE_BASE,
    PROMOTE_SEQUENCE_BASE,
    PROPOSE_SEQUENCE_BASE,
    REGISTER_SEQUENCE_BASE,
    Coder,
    EvolutionLoop,
    ExperimentDesigner,
    FeedbackSynthesizer,
    NullSpanEmitter,
    Proposer,
    Runner,
    RunSummary,
    SpanEmitter,
    Stage,
    StageContext,
    derive_key,
)
from deerflow.evolution.models import (
    CodedExperiment,
    EvoStep,
    ExperimentFeedback,
    ExperimentOutcome,
    ExperimentSpec,
    Hypothesis,
    HypothesisFeedback,
)
from deerflow.evolution.prompts import PROMPT_VERSION, PROMPTS
from deerflow.evolution.selectors import (
    AncestorSOTASelector,
    CheckpointSelector,
    FirstAcceptedSelector,
    LatestSOTASelector,
    RootRestartSelector,
    SOTASelector,
    resolve_selection_for_record,
)
from deerflow.evolution.trace import NEW_ROOT, SEL_LATEST_SOTA, InMemoryTraceStore, Trace, TraceStore

__all__ = [
    "config",
    "errors",
    "loop",
    "models",
    "prompts",
    "selectors",
    "trace",
    "ENV_ENABLED",
    "ENV_MAX_ITERATIONS",
    "ENV_MAX_WALL_CLOCK_SECONDS",
    "ENV_SANDBOX_TIMEOUT",
    "ENV_DEFAULT_FAMILY",
    "ENV_DEFAULT_UNIVERSE",
    "MODEL_ROLES",
    "EvolutionConfig",
    "get_evolution_config",
    "set_evolution_config",
    "load_evolution_config_from_dict",
    "EvolutionError",
    "GovernanceRefusalError",
    "GateFailedError",
    "BudgetExhaustedError",
    "DuplicateRunError",
    "Hypothesis",
    "ExperimentSpec",
    "CodedExperiment",
    "ExperimentOutcome",
    "ExperimentFeedback",
    "HypothesisFeedback",
    "EvoStep",
    "NEW_ROOT",
    "SEL_LATEST_SOTA",
    "Trace",
    "TraceStore",
    "InMemoryTraceStore",
    "CheckpointSelector",
    "SOTASelector",
    "LatestSOTASelector",
    "AncestorSOTASelector",
    "RootRestartSelector",
    "FirstAcceptedSelector",
    "resolve_selection_for_record",
    "PROPOSE_SEQUENCE_BASE",
    "REGISTER_SEQUENCE_BASE",
    "BACKTEST_SEQUENCE_BASE",
    "BEGIN_SEQUENCE_BASE",
    "OUTCOME_SEQUENCE_BASE",
    "COMMIT_SEQUENCE_BASE",
    "FAILURE_SEQUENCE_BASE",
    "PROMOTE_SEQUENCE_BASE",
    "Stage",
    "StageContext",
    "RunSummary",
    "SpanEmitter",
    "NullSpanEmitter",
    "Proposer",
    "ExperimentDesigner",
    "Coder",
    "Runner",
    "FeedbackSynthesizer",
    "EvolutionLoop",
    "derive_key",
    "PROMPT_VERSION",
    "PROMPTS",
]

_LAZY_SUBMODULES = frozenset({"stages", "spark_client", "knowledge_adapter", "memory_adapter", "sandboxing", "wiring"})


def __getattr__(name: str) -> Any:
    """Lazily import later-component submodules (keeps this import light)."""
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

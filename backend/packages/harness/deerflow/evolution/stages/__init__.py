"""Loop stage implementations binding the core to spark/KB/memory/LLM.

This package is loaded lazily (see :mod:`deerflow.evolution`) so importing
the evolution loop never pulls provider SDKs or transport code:

* :mod:`deerflow.evolution.stages.state` — shared per-iteration sidecar.
* :mod:`deerflow.evolution.stages.llm` — model-profile factory boundary.
* :mod:`deerflow.evolution.stages.propose` — hypothesis generation.
* :mod:`deerflow.evolution.stages.design` — experiment design + registration.
* :mod:`deerflow.evolution.stages.execute` — sandbox coding + governed runs.
* :mod:`deerflow.evolution.stages.feedback` — gate, close-out, promotion.
"""

from __future__ import annotations

from deerflow.evolution.stages.design import ExperimentDesignerImpl, derive_seed
from deerflow.evolution.stages.execute import (
    CoderImpl,
    ProviderSandboxExecutor,
    RunnerImpl,
    SandboxExecutor,
    SandboxResult,
)
from deerflow.evolution.stages.feedback import FeedbackSynthesizerImpl, failure_class_for
from deerflow.evolution.stages.llm import ChatModelFactory, LangChainModelFactory, parse_json_object
from deerflow.evolution.stages.propose import C6Payload, HypothesisGenerator, ProposalContext, validate_c6
from deerflow.evolution.stages.state import IterationRecord, IterationState, format_lineage

__all__ = [
    "C6Payload",
    "ProposalContext",
    "HypothesisGenerator",
    "validate_c6",
    "ExperimentDesignerImpl",
    "derive_seed",
    "SandboxResult",
    "SandboxExecutor",
    "ProviderSandboxExecutor",
    "CoderImpl",
    "RunnerImpl",
    "FeedbackSynthesizerImpl",
    "failure_class_for",
    "ChatModelFactory",
    "LangChainModelFactory",
    "parse_json_object",
    "IterationRecord",
    "IterationState",
    "format_lineage",
]

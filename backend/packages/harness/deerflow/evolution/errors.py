"""Typed errors for the native Quantflow evolution loop.

KB/spark errors propagate unwrapped except where explicitly mapped by the
loop and its stages (see :mod:`deerflow.evolution.loop`).
"""

from __future__ import annotations

__all__ = [
    "EvolutionError",
    "GovernanceRefusalError",
    "GateFailedError",
    "BudgetExhaustedError",
    "DuplicateRunError",
]


class EvolutionError(Exception):
    """Base class for all evolution-loop errors."""


class GovernanceRefusalError(EvolutionError):
    """The spark trial governor refused a propose/register/log_outcome call."""

    def __init__(self, message: str, *, tool: str | None = None):
        super().__init__(message)
        self.tool = tool


class GateFailedError(EvolutionError):
    """The significance gate (DSR/MinBTL) rejected the experiment outcome."""

    def __init__(self, message: str, *, verdict: str | None = None):
        super().__init__(message)
        self.verdict = verdict


class BudgetExhaustedError(EvolutionError):
    """The loop budget (iterations, wall-clock, or stop rule) is exhausted."""

    def __init__(self, message: str, *, reason: str = "budget"):
        super().__init__(message)
        self.reason = reason


class DuplicateRunError(EvolutionError):
    """The exact computation already ran; link to the prior run instead."""

    def __init__(self, execution_hash: str, existing_experiment_id: str):
        super().__init__(f"Execution {execution_hash} already ran as experiment {existing_experiment_id}; link to the prior run instead of duplicating it.")
        self.execution_hash = execution_hash
        self.existing_experiment_id = existing_experiment_id

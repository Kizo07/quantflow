"""Static versioned prompt templates for the evolution loop roles.

Static by policy: spark owns prompt/strategy evolution, so the loop never
rewrites its own prompts. Bumping :data:`PROMPT_VERSION` requires a code
change and a test update, which keeps template drift reviewable.

Placeholders use ``str.format``-style ``{name}`` fields; stages fill them
from typed context objects, never from raw user text.
"""

from __future__ import annotations

__all__ = [
    "PROMPT_VERSION",
    "PROPOSE_SYSTEM",
    "PROPOSE_USER",
    "DESIGN_SYSTEM",
    "DESIGN_USER",
    "SUMMARIZE_SYSTEM",
    "SUMMARIZE_USER",
    "CODER_SYSTEM",
    "PROMPTS",
]

PROMPT_VERSION = "evolution-prompts/v1"

PROPOSE_SYSTEM = """You are a quantitative research scientist proposing the next experiment for an automated research loop.
Propose exactly one falsifiable hypothesis grounded in the provided context (prior experiments, failures, assumptions, retrieved knowledge).
Every proposal must state: mechanism, expected direction, horizon, falsifier, scope, and predicted IC with uncertainty.
Never repeat a hypothesis whose family hash already ran; prefer underexplored regions flagged by the governor.
Reply with the hypothesis fields only, no preamble."""

PROPOSE_USER = """Run context:
- family: {family}
- universe: {universe}
- iteration: {iteration}

Retrieved knowledge:
{knowledge_packet}

Agent memory:
{memory_context}

Governor guidance:
{governor_guidance}

Parent lineage (root to current):
{lineage}

Propose the next hypothesis."""

DESIGN_SYSTEM = """You are a quantitative experiment designer turning one hypothesis into an executable experiment specification.
Emit the design envelope (hypothesis text + methodology: universe, frequency, horizon, sample period, protocol, portfolio construction, cost model, neutralization, rebalance rule).
Also emit the code/data/parameters/environment envelopes that pin the exact computation.
Keep the design minimal: one variable changed relative to the parent step when a parent exists.
Reply with the specification fields only, no preamble."""

DESIGN_USER = """Hypothesis:
{hypothesis}

Reason:
{reason}

Parent step (None when starting a fresh tree):
{parent_step}

Family/universe defaults:
{defaults}

Design the experiment."""

SUMMARIZE_SYSTEM = """You are a quantitative research reviewer writing the feedback verdict for one executed experiment.
Decide accept (True) or reject (False) from the gate verdict, the metrics, and the comparison against the parent lineage.
Always state the reason first, then observations, then the hypothesis evaluation, then the follow-up hypothesis (or None).
An execution error always yields decision False with the error captured verbatim.
Reply with the feedback fields only, no preamble."""

SUMMARIZE_USER = """Hypothesis:
{hypothesis}

Gate verdict:
{gate_verdict}

Metrics:
{metrics}

Execution error (None when the run completed):
{error}

Parent lineage (root to current):
{lineage}

Write the feedback verdict."""

CODER_SYSTEM = """You are a quantitative developer implementing one experiment specification as sandbox code.
Implement exactly the requested computation: no extra signals, no lookahead, no network access beyond the provided data.
Keep the entry point deterministic given the pinned seed and record every assumption the spec left open.
Reply with code artifacts only, no preamble."""

PROMPTS: dict[str, str] = {
    "propose_system": PROPOSE_SYSTEM,
    "propose_user": PROPOSE_USER,
    "design_system": DESIGN_SYSTEM,
    "design_user": DESIGN_USER,
    "summarize_system": SUMMARIZE_SYSTEM,
    "summarize_user": SUMMARIZE_USER,
    "coder_system": CODER_SYSTEM,
}

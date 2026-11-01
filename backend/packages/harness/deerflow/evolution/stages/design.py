"""Design stage: hypothesis to registered experiment.

:class:`ExperimentDesignerImpl` implements the loop's ``ExperimentDesigner``
protocol:

1. Calls the design-profile LLM (static versioned templates) to convert the
   hypothesis into an :class:`ExperimentSpec` envelope.
2. Runs the pre-experiment recheck (execution-hash exact rerun +
   family-hash equivalents) before any compute; an exact rerun raises
   :class:`DuplicateRunError` so the loop links to the prior run.
3. Registers with the trial governor first (spark ``register_hypothesis``),
   then opens the KB record (``experiment_begin``) — the governor is always
   the first writer, so the KB never records an ungoverned run.

Design identities (episode, experiment, protocol hash, seed) are recorded
in the shared :class:`IterationState` for the execute/feedback stages. The
trial seed derives deterministically from ``(run_id, iteration)`` so
retries pin the same computation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from deerflow.evolution.config import EvolutionConfig
from deerflow.evolution.errors import EvolutionError
from deerflow.evolution.knowledge_adapter import KnowledgeAdapter
from deerflow.evolution.loop import BEGIN_SEQUENCE_BASE, REGISTER_SEQUENCE_BASE, StageContext, derive_key
from deerflow.evolution.models import ExperimentSpec, Hypothesis
from deerflow.evolution.prompts import DESIGN_SYSTEM, DESIGN_USER
from deerflow.evolution.spark_client import SparkClient
from deerflow.evolution.stages.llm import ChatModelFactory, parse_json_object
from deerflow.evolution.stages.state import IterationState

__all__ = [
    "ExperimentDesignerImpl",
    "derive_seed",
]


def derive_seed(run_id: str, iteration: int) -> int:
    """Derive the deterministic trial seed for one iteration."""
    digest = hashlib.sha256(f"{run_id}:{iteration}".encode()).hexdigest()
    return int(digest[:8], 16)


class ExperimentDesignerImpl:
    """Loop ``ExperimentDesigner`` bound to spark + KB + model profiles.

    Args:
        knowledge: KB binding for the recheck and ``experiment_begin``.
        spark: Trial-governance client for ``register_hypothesis``.
        models: Role-addressed chat model factory.
        config: Evolution config (family/universe defaults).
        state: Shared per-iteration sidecar.
    """

    def __init__(
        self,
        *,
        knowledge: KnowledgeAdapter,
        spark: SparkClient,
        models: ChatModelFactory,
        config: EvolutionConfig,
        state: IterationState | None = None,
    ) -> None:
        self._knowledge = knowledge
        self._spark = spark
        self._models = models
        self._config = config
        self._state = state or IterationState()

    async def design(self, hypothesis: Hypothesis, ctx: StageContext) -> ExperimentSpec:
        """Design, recheck, and register one experiment for *hypothesis*."""
        parents = ctx.trace.get_parent_steps()
        if parents:
            current = parents[-1]
            parent_text = json.dumps(
                {
                    "iteration": current.iteration,
                    "hypothesis": current.hypothesis.hypothesis if current.hypothesis else None,
                    "decision": current.feedback.decision if current.feedback else None,
                },
                sort_keys=True,
            )
        else:
            parent_text = "None"
        defaults = json.dumps(
            {
                "family": self._config.resolve_default_family(),
                "universe": self._config.resolve_default_universe(),
            },
            sort_keys=True,
        )
        user_prompt = DESIGN_USER.format(
            hypothesis=hypothesis.hypothesis,
            reason=hypothesis.reason,
            parent_step=parent_text,
            defaults=defaults,
        )
        raw = await self._models.generate(role="design", system=DESIGN_SYSTEM, user=user_prompt)
        payload = parse_json_object(raw, role="design")
        design_section = payload.get("design", {})
        if not isinstance(design_section, Mapping):
            raise EvolutionError("The design model must return a 'design' object.")
        design_section = dict(design_section)
        design_section.setdefault("hypothesis", hypothesis.hypothesis)
        spec = ExperimentSpec(
            design=design_section,
            code=payload.get("code", {}),
            data=payload.get("data", {}),
            parameters=payload.get("parameters", {}),
            environment=payload.get("environment", {}),
        )
        # Recheck before any write or compute: exact reruns raise here.
        await self._knowledge.recheck(spec)
        hypothesis_id = self._state.require_hypothesis_id(ctx.run_id, ctx.iteration)
        seed = derive_seed(ctx.run_id, ctx.iteration)
        trial_spec: dict[str, Any] = {
            "experiment_spec": spec.to_dict(),
            "family": self._config.resolve_default_family(),
            "universe": self._config.resolve_default_universe(),
        }
        register_key = derive_key(
            ctx.run_id,
            REGISTER_SEQUENCE_BASE,
            ctx.iteration,
            {"hypothesis_id": hypothesis_id, "spec": trial_spec, "seed": seed},
        )
        # Ordering: governor first, KB second — never an ungoverned KB record.
        registration = await self._spark.register(
            hypothesis_id=hypothesis_id,
            spec=trial_spec,
            seed=seed,
            client_id=register_key,
        )
        parent_experiment_id: str | None = None
        if parents:
            parent_experiment_id = self._state.get_experiment_id(ctx.run_id, parents[-1].iteration)
        begin_key = derive_key(
            ctx.run_id,
            BEGIN_SEQUENCE_BASE,
            ctx.iteration,
            {
                "project_id": self._knowledge.project_id,
                "created_by_run_id": ctx.run_id,
                "execution_hash": spec.execution_hash(),
            },
        )
        record = await self._knowledge.begin_experiment(
            spec,
            run_id=ctx.run_id,
            parent_experiment_id=parent_experiment_id,
            status="running",
            idempotency_key=begin_key,
        )
        self._state.record_design(
            ctx.run_id,
            ctx.iteration,
            episode_id=registration.episode_id,
            experiment_id=record.id,
            protocol_hash=registration.protocol_hash,
            seed=seed,
        )
        return spec

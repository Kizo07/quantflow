"""Typed async client over the spark trial-governance MCP tools.

This module is the Quantflow side of the spark integration: it speaks to
spark exclusively over MCP transport through an injected
:class:`SparkTransport`. It never imports the spark server package (a
separate repository, reference-only); the import-guards test enforces that.

Covered tools (names match the spark MCP surface exactly):

* ``hypothesis_propose`` — C6-validated hypothesis registration.
* ``register_hypothesis`` — trial-episode creation + protocol hash.
* ``backtest_run`` — governed backtest dispatch (tombstone pre-check honored
  by the caller; retries reuse the winner trial so M never double-counts).
* ``significance_gate`` — DSR/MinBTL verdict read.
* ``log_outcome`` — episode close-out (metrics, costs, reward).
* ``check_novelty`` — bounded novelty/surprise read.
* ``suggest_next`` — underexplored-region suggestions.
* ``memory_recall`` / ``memory_promote`` — belief recall and promotion.

Error contract: transport failures that name a refusal surface as
:class:`GovernanceRefusalError` (with the tool name attached); other
transport ``ToolError``-shaped failures surface as :class:`EvolutionError`
with the tool name and the server message preserved (tombstone blocks stay
recognizable for the execute stage's recorded-rationale retry); anything
else propagates unwrapped.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from deerflow.evolution.errors import EvolutionError, GovernanceRefusalError

__all__ = [
    "SparkTransport",
    "SparkClient",
    "ProposeResult",
    "RegisterResult",
    "GateVerdict",
    "NoveltyResult",
    "SuggestNextResult",
    "Belief",
    "PromoteResult",
    "BacktestResult",
]


@runtime_checkable
class SparkTransport(Protocol):
    """Minimal MCP transport boundary for spark tool calls.

    The integration step binds this to the configured MCP caller (server
    name ``spark``); unit tests bind it to an in-memory fake. The transport
    returns decoded JSON values and raises its own errors on failure.
    """

    async def call_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> Any:
        """Call one spark MCP tool and return its decoded result."""
        ...


def _is_refusal(exc: BaseException) -> bool:
    """True when *exc* names a governance refusal (C6 or otherwise)."""
    name = type(exc).__name__.lower().replace("_", "")
    if "refus" in name:
        return True
    return "refus" in str(exc).lower()


def _is_tool_error(exc: BaseException) -> bool:
    """True when *exc* is shaped like a server-side tool failure."""
    return type(exc).__name__.lower().replace("_", "").endswith("toolerror")


def _require_mapping(tool: str, result: Any) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise EvolutionError(f"spark tool {tool} returned a non-object result: {type(result).__name__}.")
    return dict(result)


def _check_error_envelope(tool: str, payload: Mapping[str, Any]) -> None:
    """Raise for transports that report failures as data instead of raising."""
    error = payload.get("error")
    if error is not None and error != "":
        message = error if isinstance(error, str) else repr(error)
        if "refus" in message.lower():
            raise GovernanceRefusalError(f"spark tool {tool} refused: {message}", tool=tool)
        raise EvolutionError(f"spark tool {tool} failed: {message}")
    status = payload.get("status")
    if isinstance(status, str) and status.lower() in {"refused", "rejected"}:
        raise GovernanceRefusalError(f"spark tool {tool} refused (status={status!r}).", tool=tool)


def _require_str(tool: str, payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvolutionError(f"spark tool {tool} returned no {key!r}: {payload!r}.")
    return value


@dataclass(frozen=True)
class ProposeResult:
    """Outcome of ``hypothesis_propose``."""

    hypothesis_id: str
    novelty: float | None = None
    surprise: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegisterResult:
    """Outcome of ``register_hypothesis``."""

    episode_id: str
    protocol_hash: str
    spec_hash: str | None = None
    seed: int | None = None
    status: str | None = None
    reused: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateVerdict:
    """Outcome of ``significance_gate`` (DSR/MinBTL, net of costs)."""

    passed: bool
    status: str
    sr: float | None = None
    dsr: float | None = None
    p_value: float | None = None
    m_family: int | None = None
    promotion_blockers: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NoveltyResult:
    """Outcome of ``check_novelty`` (N bounded; S null without realized IC)."""

    novelty: float | None = None
    surprise: float | None = None
    max_cos: float | None = None
    structural_overlap: float | None = None
    cell_emptiness: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SuggestNextResult:
    """Outcome of ``suggest_next`` (underexplored-region suggestions)."""

    suggestions: tuple[dict[str, Any], ...] = ()
    selection_probabilities: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Belief:
    """One recalled spark belief with episode pointers."""

    claim: str
    confidence: float | None = None
    effective_confidence: float | None = None
    supporting: tuple[str, ...] = ()
    contradicting: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromoteResult:
    """Outcome of ``memory_promote``."""

    belief_id: str
    promotion_status: str
    contradictions: tuple[str, ...] = ()
    confidence: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def promoted(self) -> bool:
        """True when spark auto-promoted (vs ``pending_human``)."""
        return self.promotion_status == "auto"


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of ``backtest_run`` (trial + provider metrics)."""

    metrics: dict[str, Any] = field(default_factory=dict)
    trial: dict[str, Any] = field(default_factory=dict)
    simulated: bool | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


class SparkClient:
    """Typed async wrapper over the spark trial-governance MCP tools.

    Args:
        transport: Bound MCP transport for the spark server.
    """

    def __init__(self, transport: SparkTransport) -> None:
        self._transport = transport

    @property
    def transport(self) -> SparkTransport:
        """The bound MCP transport."""
        return self._transport

    async def _call(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        try:
            result = await self._transport.call_tool(tool, dict(arguments))
        except GovernanceRefusalError:
            raise
        except EvolutionError:
            raise
        except Exception as exc:
            if _is_refusal(exc):
                raise GovernanceRefusalError(f"spark tool {tool} refused: {exc}", tool=tool) from exc
            if _is_tool_error(exc):
                raise EvolutionError(f"spark tool {tool} failed: {exc}") from exc
            raise
        payload = _require_mapping(tool, result)
        _check_error_envelope(tool, payload)
        return payload

    async def propose(
        self,
        *,
        idea: str,
        family: str,
        universe: str,
        horizon: str,
        scope: str,
        mechanism: str,
        expected_direction: str,
        falsifier: str,
        predicted_ic: float,
        predicted_sigma: float,
        hypothesis_type: str,
        dag: Mapping[str, Any] | None = None,
        adjustment_set: list[str] | None = None,
        parent_hypothesis_ids: list[str] | None = None,
        client_id: str | None = None,
    ) -> ProposeResult:
        """Propose a C6-validated hypothesis (idempotent on *client_id*)."""
        arguments: dict[str, Any] = {
            "idea": idea,
            "family": family,
            "universe": universe,
            "horizon": horizon,
            "scope": scope,
            "mechanism": mechanism,
            "expected_direction": expected_direction,
            "falsifier": falsifier,
            "predicted_ic": predicted_ic,
            "predicted_sigma": predicted_sigma,
            "hypothesis_type": hypothesis_type,
        }
        if dag is not None:
            arguments["dag"] = dict(dag)
        if adjustment_set is not None:
            arguments["adjustment_set"] = list(adjustment_set)
        if parent_hypothesis_ids is not None:
            arguments["parent_hypothesis_ids"] = list(parent_hypothesis_ids)
        if client_id is not None:
            arguments["client_id"] = client_id
        payload = await self._call("hypothesis_propose", arguments)
        return ProposeResult(
            hypothesis_id=_require_str("hypothesis_propose", payload, "hypothesis_id"),
            novelty=_as_float(payload.get("N")),
            surprise=_as_float(payload.get("S")),
            raw=payload,
        )

    async def register(self, *, hypothesis_id: str, spec: Mapping[str, Any], seed: int, client_id: str | None = None) -> RegisterResult:
        """Create the trial episode for a hypothesis (idempotent on *client_id*)."""
        arguments: dict[str, Any] = {"hypothesis_id": hypothesis_id, "spec": dict(spec), "seed": seed}
        if client_id is not None:
            arguments["client_id"] = client_id
        payload = await self._call("register_hypothesis", arguments)
        seed_value = payload.get("seed")
        return RegisterResult(
            episode_id=_require_str("register_hypothesis", payload, "episode_id"),
            protocol_hash=_require_str("register_hypothesis", payload, "protocol_hash"),
            spec_hash=payload.get("spec_hash") if isinstance(payload.get("spec_hash"), str) else None,
            seed=seed_value if isinstance(seed_value, int) and not isinstance(seed_value, bool) else None,
            status=payload.get("status") if isinstance(payload.get("status"), str) else None,
            reused=bool(payload.get("reused", False)),
            raw=payload,
        )

    async def log_outcome(
        self,
        *,
        episode_id: str,
        outcome: str,
        metrics: Mapping[str, Any] | None = None,
        failure_reason: str | None = None,
        costs: Mapping[str, Any] | None = None,
        reward: float | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        """Close out an episode (idempotent on *client_id*)."""
        arguments: dict[str, Any] = {"episode_id": episode_id, "outcome": outcome}
        if metrics is not None:
            arguments["metrics"] = dict(metrics)
        if failure_reason is not None:
            arguments["failure_reason"] = failure_reason
        if costs is not None:
            arguments["costs"] = dict(costs)
        if reward is not None:
            arguments["reward"] = reward
        if client_id is not None:
            arguments["client_id"] = client_id
        return await self._call("log_outcome", arguments)

    async def significance_gate(self, episode_id: str) -> GateVerdict:
        """Read the DSR/MinBTL gate verdict for an episode (pure read)."""
        payload = await self._call("significance_gate", {"episode_id": episode_id})
        m_value = payload.get("M_family")
        return GateVerdict(
            passed=bool(payload.get("pass", False)),
            status=str(payload.get("status", "")),
            sr=_as_float(payload.get("sr")),
            dsr=_as_float(payload.get("dsr")),
            p_value=_as_float(payload.get("p")),
            m_family=m_value if isinstance(m_value, int) and not isinstance(m_value, bool) else None,
            promotion_blockers=_as_str_tuple(payload.get("promotion_blockers")),
            raw=payload,
        )

    async def check_novelty(
        self,
        idea_or_code: str,
        *,
        family: str | None = None,
        realized_ic: float | None = None,
        predicted_ic: float | None = None,
        predicted_sigma: float | None = None,
    ) -> NoveltyResult:
        """Score bounded novelty/surprise for an idea (pure read)."""
        arguments: dict[str, Any] = {"idea_or_code": idea_or_code}
        if family is not None:
            arguments["family"] = family
        if realized_ic is not None:
            arguments["realized_ic"] = realized_ic
        if predicted_ic is not None:
            arguments["predicted_ic"] = predicted_ic
        if predicted_sigma is not None:
            arguments["predicted_sigma"] = predicted_sigma
        payload = await self._call("check_novelty", arguments)
        return NoveltyResult(
            novelty=_as_float(payload.get("N")),
            surprise=_as_float(payload.get("S")),
            max_cos=_as_float(payload.get("max_cos")),
            structural_overlap=_as_float(payload.get("structural_overlap")),
            cell_emptiness=_as_float(payload.get("cell_emptiness")),
            raw=payload,
        )

    async def suggest_next(self, *, n: int = 3, seed: int | None = None) -> SuggestNextResult:
        """Fetch underexplored-region suggestions (pure read)."""
        arguments: dict[str, Any] = {"n": n}
        if seed is not None:
            arguments["seed"] = seed
        payload = await self._call("suggest_next", arguments)
        raw_suggestions = payload.get("suggestions", ())
        suggestions = tuple(dict(item) for item in raw_suggestions if isinstance(item, Mapping))
        raw_probs = payload.get("family_selection_probabilities", {})
        probabilities = {name: float(prob) for name, prob in raw_probs.items() if isinstance(name, str) and isinstance(prob, (int, float)) and not isinstance(prob, bool)} if isinstance(raw_probs, Mapping) else {}
        return SuggestNextResult(suggestions=suggestions, selection_probabilities=probabilities, raw=payload)

    async def memory_recall(self, query: str, *, k: int = 10) -> tuple[Belief, ...]:
        """Recall beliefs ordered by effective confidence (pure read)."""
        payload = await self._call("memory_recall", {"query": query, "k": k})
        raw_beliefs = payload.get("beliefs", ())
        if not isinstance(raw_beliefs, (list, tuple)):
            raise EvolutionError(f"spark tool memory_recall returned non-list beliefs: {raw_beliefs!r}.")
        beliefs: list[Belief] = []
        for entry in raw_beliefs:
            if not isinstance(entry, Mapping):
                continue
            claim = entry.get("claim")
            if not isinstance(claim, str) or not claim.strip():
                continue
            beliefs.append(
                Belief(
                    claim=claim,
                    confidence=_as_float(entry.get("confidence")),
                    effective_confidence=_as_float(entry.get("effective_confidence")),
                    supporting=_as_str_tuple(entry.get("supporting") or entry.get("supporting_episodes")),
                    contradicting=_as_str_tuple(entry.get("contradicting") or entry.get("contradicting_episodes")),
                    raw=dict(entry),
                )
            )
        return tuple(beliefs)

    async def memory_promote(
        self,
        *,
        episode_ids: list[str],
        claim: str,
        confidence: float,
        scope: str | None = None,
        rationale: str | None = None,
        client_id: str | None = None,
    ) -> PromoteResult:
        """Promote a belief (idempotent on *client_id*).

        Spark auto-promotes only when its confidence threshold passes with
        no contradictions; otherwise the status is ``pending_human``, which
        the caller accepts without escalation (no prompts in the loop path).
        """
        arguments: dict[str, Any] = {"episode_ids": list(episode_ids), "claim": claim, "confidence": confidence}
        if scope is not None:
            arguments["scope"] = scope
        if rationale is not None:
            arguments["rationale"] = rationale
        if client_id is not None:
            arguments["client_id"] = client_id
        payload = await self._call("memory_promote", arguments)
        return PromoteResult(
            belief_id=_require_str("memory_promote", payload, "belief_id"),
            promotion_status=str(payload.get("promotion_status", "")),
            contradictions=_as_str_tuple(payload.get("contradictions")),
            confidence=_as_float(payload.get("confidence")),
            raw=payload,
        )

    async def backtest_run(
        self,
        *,
        episode_id: str,
        spec: Mapping[str, Any],
        acknowledge_tombstone: Mapping[str, Any] | None = None,
        client_id: str | None = None,
    ) -> BacktestResult:
        """Dispatch a governed backtest (idempotent on *client_id*).

        A tombstone block surfaces as an error whose message names the
        blocking tombstone; the caller retries with a recorded
        acknowledgement or not at all (never prompted).
        """
        arguments: dict[str, Any] = {"episode_id": episode_id, "spec": dict(spec)}
        if acknowledge_tombstone is not None:
            arguments["acknowledge_tombstone"] = dict(acknowledge_tombstone)
        if client_id is not None:
            arguments["client_id"] = client_id
        payload = await self._call("backtest_run", arguments)
        metrics = payload.get("metrics")
        trial = payload.get("trial")
        simulated = payload.get("simulated")
        return BacktestResult(
            metrics=dict(metrics) if isinstance(metrics, Mapping) else {},
            trial=dict(trial) if isinstance(trial, Mapping) else {},
            simulated=simulated if isinstance(simulated, bool) else None,
            raw=payload,
        )

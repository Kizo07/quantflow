"""Unit tests for deerflow.evolution.spark_client (offline).

A scripted fake MCP transport stands in for the spark server; no network,
no spark package import (transport-only by contract).
"""

from collections.abc import Mapping
from typing import Any

import pytest

from deerflow.evolution.errors import EvolutionError, GovernanceRefusalError
from deerflow.evolution.spark_client import (
    BacktestResult,
    GateVerdict,
    PromoteResult,
    SparkClient,
)


class FakeTransport:
    """Scripted transport: per-tool results or exceptions, with call log."""

    def __init__(self, results: dict[str, Any] | None = None):
        self.results = dict(results or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> Any:
        self.calls.append((tool_name, dict(arguments)))
        result = self.results.get(tool_name, {})
        if isinstance(result, Exception):
            raise result
        if callable(result):
            return result(tool_name, dict(arguments))
        return result


class FakeToolError(Exception):
    """Server-side tool failure shaped like a ToolError (by name)."""


def _client(results: dict[str, Any] | None = None) -> tuple[SparkClient, FakeTransport]:
    transport = FakeTransport(results)
    return SparkClient(transport), transport


def _c6_arguments(**overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "idea": "momentum persists",
        "family": "quant-alpha",
        "universe": "sp500",
        "horizon": "21d",
        "scope": "us-equities",
        "mechanism": "slow diffusion",
        "expected_direction": "positive",
        "falsifier": "IC <= 0 over 63d",
        "predicted_ic": 0.05,
        "predicted_sigma": 0.02,
        "hypothesis_type": "predictive",
    }
    arguments.update(overrides)
    return arguments


@pytest.mark.asyncio
async def test_propose_returns_hypothesis_id_and_forwards_client_id():
    client, transport = _client({"hypothesis_propose": {"hypothesis_id": "h-1", "N": 0.7, "S": None}})
    result = await client.propose(**_c6_arguments(client_id="key-1000"))
    assert result.hypothesis_id == "h-1"
    assert result.novelty == 0.7
    assert result.surprise is None
    assert transport.calls[0][0] == "hypothesis_propose"
    assert transport.calls[0][1]["client_id"] == "key-1000"
    assert transport.calls[0][1]["hypothesis_type"] == "predictive"


@pytest.mark.asyncio
async def test_propose_omits_unset_optionals():
    client, transport = _client({"hypothesis_propose": {"hypothesis_id": "h-1"}})
    await client.propose(**_c6_arguments())
    sent = transport.calls[0][1]
    assert "dag" not in sent
    assert "adjustment_set" not in sent
    assert "parent_hypothesis_ids" not in sent
    assert "client_id" not in sent


@pytest.mark.asyncio
async def test_propose_missing_hypothesis_id_raises():
    client, _ = _client({"hypothesis_propose": {"N": 0.1}})
    with pytest.raises(EvolutionError, match="hypothesis_id"):
        await client.propose(**_c6_arguments())


@pytest.mark.asyncio
async def test_register_returns_episode_and_protocol_hash():
    client, transport = _client(
        {
            "register_hypothesis": {
                "episode_id": "e-1",
                "hypothesis_id": "h-1",
                "protocol_hash": "ab" * 32,
                "spec_hash": "cd" * 32,
                "seed": 42,
                "status": "registered",
            }
        }
    )
    result = await client.register(hypothesis_id="h-1", spec={"a": 1}, seed=42, client_id="key-2000")
    assert result.episode_id == "e-1"
    assert result.protocol_hash == "ab" * 32
    assert result.reused is False
    assert transport.calls[0][1]["client_id"] == "key-2000"


@pytest.mark.asyncio
async def test_log_outcome_forwards_close_out_fields():
    client, transport = _client({"log_outcome": {"episode_id": "e-1", "status": "validated"}})
    result = await client.log_outcome(
        episode_id="e-1",
        outcome="success",
        metrics={"sr": 1.2},
        reward=1.0,
        client_id="key-4000",
    )
    assert result["status"] == "validated"
    sent = transport.calls[0][1]
    assert sent["outcome"] == "success"
    assert sent["metrics"] == {"sr": 1.2}
    assert sent["reward"] == 1.0
    assert "failure_reason" not in sent
    assert "costs" not in sent


@pytest.mark.asyncio
async def test_significance_gate_parses_verdict():
    payload = {
        "episode_id": "e-1",
        "status": "validated",
        "sr": 1.5,
        "psr": 0.97,
        "dsr": 0.96,
        "p": 0.04,
        "M_family": 12,
        "pass": True,
        "promotion_blockers": [],
    }
    client, _ = _client({"significance_gate": payload})
    verdict: GateVerdict = await client.significance_gate("e-1")
    assert verdict.passed is True
    assert verdict.status == "validated"
    assert verdict.sr == 1.5
    assert verdict.dsr == 0.96
    assert verdict.p_value == 0.04
    assert verdict.m_family == 12


@pytest.mark.asyncio
async def test_check_novelty_parses_bounded_scores():
    client, transport = _client(
        {
            "check_novelty": {
                "N": 0.62,
                "S": None,
                "max_cos": 0.3,
                "structural_overlap": 0.1,
                "cell_emptiness": 0.9,
            }
        }
    )
    result = await client.check_novelty("an idea", family="quant-alpha", predicted_ic=0.05, predicted_sigma=0.02)
    assert result.novelty == 0.62
    assert result.surprise is None
    assert result.cell_emptiness == 0.9
    assert transport.calls[0][1]["family"] == "quant-alpha"


@pytest.mark.asyncio
async def test_suggest_next_parses_suggestions_and_probabilities():
    client, _ = _client(
        {
            "suggest_next": {
                "suggestions": [{"family": "f-a"}, {"family": "f-b"}],
                "family_selection_probabilities": {"f-a": 0.7, "f-b": 0.3},
            }
        }
    )
    result = await client.suggest_next(n=2, seed=7)
    assert len(result.suggestions) == 2
    assert result.selection_probabilities == {"f-a": 0.7, "f-b": 0.3}


@pytest.mark.asyncio
async def test_memory_recall_parses_beliefs_and_skips_malformed():
    client, transport = _client(
        {
            "memory_recall": {
                "beliefs": [
                    {"claim": "momentum works", "confidence": 0.8, "effective_confidence": 0.75, "supporting": ["e-1"]},
                    {"no-claim": True},
                    "not-a-mapping",
                ]
            }
        }
    )
    beliefs = await client.memory_recall("momentum", k=5)
    assert len(beliefs) == 1
    assert beliefs[0].claim == "momentum works"
    assert beliefs[0].supporting == ("e-1",)
    assert transport.calls[0][1] == {"query": "momentum", "k": 5}


@pytest.mark.asyncio
async def test_memory_promote_reports_pending_human_without_raising():
    client, _ = _client(
        {
            "memory_promote": {
                "belief_id": "b-1",
                "promotion_status": "pending_human",
                "contradictions": ["b-0"],
                "confidence": 0.5,
            }
        }
    )
    result: PromoteResult = await client.memory_promote(episode_ids=["e-1"], claim="a claim", confidence=0.5, scope="s", rationale="r", client_id="key-7000")
    assert result.promoted is False
    assert result.promotion_status == "pending_human"
    assert result.contradictions == ("b-0",)


@pytest.mark.asyncio
async def test_backtest_run_parses_metrics_trial_simulated():
    client, transport = _client(
        {
            "backtest_run": {
                "metrics": {"sr": 1.1, "T": 252},
                "trial": {"trial_id": "t-1"},
                "simulated": True,
            }
        }
    )
    result: BacktestResult = await client.backtest_run(episode_id="e-1", spec={"x": 1}, acknowledge_tombstone={"tombstone_id": "t", "rationale": "r"})
    assert result.metrics == {"sr": 1.1, "T": 252}
    assert result.trial == {"trial_id": "t-1"}
    assert result.simulated is True
    assert transport.calls[0][1]["acknowledge_tombstone"] == {"tombstone_id": "t", "rationale": "r"}


@pytest.mark.asyncio
async def test_refusal_maps_to_governance_refusal_with_tool():
    client, _ = _client({"hypothesis_propose": ValueError("hypothesis refused (C6): missing required fields: scope")})
    with pytest.raises(GovernanceRefusalError, match="refused") as exc_info:
        await client.propose(**_c6_arguments())
    assert exc_info.value.tool == "hypothesis_propose"


@pytest.mark.asyncio
async def test_tool_error_maps_to_evolution_error_preserving_message():
    block = FakeToolError('tombstone pre-check blocked dispatch: no. matched tombstone: {"tombstone_id": "tb-1"}')
    client, _ = _client({"backtest_run": block})
    with pytest.raises(EvolutionError, match="tombstone pre-check blocked dispatch") as exc_info:
        await client.backtest_run(episode_id="e-1", spec={})
    assert "tb-1" in str(exc_info.value)


@pytest.mark.asyncio
async def test_unexpected_transport_errors_propagate_unwrapped():
    client, _ = _client({"significance_gate": ConnectionError("mcp down")})
    with pytest.raises(ConnectionError, match="mcp down"):
        await client.significance_gate("e-1")


@pytest.mark.asyncio
async def test_error_envelope_dict_raises():
    client, _ = _client({"log_outcome": {"error": "episode unknown"}})
    with pytest.raises(EvolutionError, match="episode unknown"):
        await client.log_outcome(episode_id="e-x", outcome="success")


@pytest.mark.asyncio
async def test_refused_status_envelope_raises_governance_refusal():
    client, _ = _client({"memory_promote": {"status": "refused", "belief_id": "b-1"}})
    with pytest.raises(GovernanceRefusalError):
        await client.memory_promote(episode_ids=["e-1"], claim="c", confidence=0.1)


@pytest.mark.asyncio
async def test_non_object_result_raises():
    client, _ = _client({"suggest_next": ["not", "an", "object"]})
    with pytest.raises(EvolutionError, match="non-object"):
        await client.suggest_next()

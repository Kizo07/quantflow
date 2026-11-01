"""Unit tests for deerflow.evolution.sandboxing (offline).

A fake provider/sandbox pair proves the Quantflow adaptation: canonical
workdir, code-contract normalization, provider manifest staging, and
outputs/ artifact harvest.
"""

import json
import uuid
from typing import Any

import pytest

from deerflow.evolution.loop import StageContext
from deerflow.evolution.models import CodedExperiment, ExperimentSpec
from deerflow.evolution.sandboxing import (
    CODE_CONTRACT_VERSION,
    MAX_HARVESTED_ARTIFACTS,
    PROVIDER_MANIFEST_FILENAME,
    PROVIDER_NAME,
    QuantflowSandboxExecutor,
    evolution_workdir,
    normalize_code_mapping,
    provider_manifest,
)

RUN_ID = "44444444-4444-4444-4444-444444444444"


def _spec(**overrides: Any) -> ExperimentSpec:
    values: dict[str, Any] = {
        "design": {"hypothesis": "momentum persists", "methodology": {"universe": "sp500"}},
        "code": {"artifact": "code-v1"},
        "data": {"datasets": ["lse/us-equities/daily"]},
        "parameters": {"seed": 3},
        "environment": {"image": "qf:v1"},
    }
    values.update(overrides)
    return ExperimentSpec(**values)


def _ctx(iteration: int = 0) -> StageContext:
    from deerflow.evolution.config import EvolutionConfig
    from deerflow.evolution.trace import Trace

    return StageContext(
        run_id=RUN_ID,
        iteration=iteration,
        step_id=str(uuid.uuid4()),
        trace=Trace(),
        config=EvolutionConfig(max_iterations=5),
    )


class FakeSandbox:
    """In-memory Sandbox double recording writes and commands."""

    def __init__(self, outputs: list[str] | None = None, stdout: str = "ok") -> None:
        self.files: dict[str, str] = {}
        self.commands: list[dict[str, Any]] = []
        self._outputs = list(outputs or [])
        self._stdout = stdout
        self.fail_list = False

    def write_file(self, path: str, content: str, append: bool = False) -> None:
        self.files[path] = content

    def execute_command(self, command: str, env=None, timeout=None) -> str:
        self.commands.append({"command": command, "timeout": timeout})
        return self._stdout

    def list_dir(self, path: str, max_depth: int = 2) -> list[str]:
        if self.fail_list:
            raise OSError("listing unavailable")
        return list(self._outputs)


class FakeProvider:
    """Minimal SandboxProvider double over one FakeSandbox."""

    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox
        self.acquired: list[str | None] = []
        self.released: list[str] = []
        self.vanish = False

    def acquire(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        self.acquired.append(thread_id)
        return f"sb-{len(self.acquired)}"

    def get(self, sandbox_id: str):
        if self.vanish:
            return None
        return self.sandbox

    def release(self, sandbox_id: str) -> None:
        self.released.append(sandbox_id)


def test_evolution_workdir_uses_user_data_contract():
    assert evolution_workdir(RUN_ID, 2) == f"/mnt/user-data/evolution/{RUN_ID}/iter-2"


def test_normalize_code_mapping_accepts_valid_contract():
    files, entrypoint = normalize_code_mapping({"files": {"run.py": "print(1)"}, "entrypoint": "  python run.py  "})
    assert files == {"run.py": "print(1)"}
    assert entrypoint == "python run.py"


def test_normalize_code_mapping_rejects_empty_files_and_entrypoint():
    with pytest.raises(ValueError, match="no files"):
        normalize_code_mapping({"files": {}, "entrypoint": "python run.py"})
    with pytest.raises(ValueError, match="no entrypoint"):
        normalize_code_mapping({"files": {"run.py": "x"}, "entrypoint": "   "})
    with pytest.raises(ValueError, match="must map a path to text"):
        normalize_code_mapping({"files": {"run.py": 42}, "entrypoint": "python run.py"})


def test_provider_manifest_names_provider_and_datasets():
    manifest = provider_manifest(run_id=RUN_ID, iteration=1, spec=_spec())
    assert manifest["contract_version"] == CODE_CONTRACT_VERSION
    assert manifest["provider"] == PROVIDER_NAME
    assert manifest["datasets"] == ["lse/us-equities/daily"]
    assert manifest["run_id"] == RUN_ID
    assert manifest["iteration"] == 1


def test_provider_manifest_defaults_datasets_to_empty():
    manifest = provider_manifest(run_id=RUN_ID, iteration=0, spec=_spec(data={}))
    assert manifest["datasets"] == []


@pytest.mark.asyncio
async def test_executor_writes_files_manifest_and_runs_entrypoint():
    sandbox = FakeSandbox(outputs=["metrics.json"], stdout="sharpe=1.2")
    executor = QuantflowSandboxExecutor(FakeProvider(sandbox), command_timeout=12.5)
    coded = CodedExperiment(spec=_spec(), code={"files": {"run.py": "print('hi')"}, "entrypoint": "python run.py"})

    result = await executor.run(coded, _ctx(iteration=2))

    workdir = evolution_workdir(RUN_ID, 2)
    assert result.error is None
    assert result.stdout == "sharpe=1.2"
    assert sandbox.files[f"{workdir}/run.py"] == "print('hi')"
    staged = json.loads(sandbox.files[f"{workdir}/{PROVIDER_MANIFEST_FILENAME}"])
    assert staged["provider"] == PROVIDER_NAME
    assert staged["iteration"] == 2
    assert sandbox.commands == [{"command": f"cd {workdir} && python run.py", "timeout": 12.5}]
    assert result.artifacts == ("outputs/metrics.json",)


@pytest.mark.asyncio
async def test_executor_releases_sandbox_and_reports_vanished_sandbox():
    sandbox = FakeSandbox()
    provider = FakeProvider(sandbox)
    provider.vanish = True
    executor = QuantflowSandboxExecutor(provider)

    result = await executor.run(
        CodedExperiment(spec=_spec(), code={"files": {"run.py": "x"}, "entrypoint": "python run.py"}),
        _ctx(),
    )

    assert result.error is not None and "vanished" in result.error
    assert provider.acquired == [RUN_ID]
    assert provider.released == ["sb-1"]


@pytest.mark.asyncio
async def test_executor_releases_sandbox_on_command_failure():
    class ExplodingSandbox(FakeSandbox):
        def execute_command(self, command: str, env=None, timeout=None) -> str:
            raise RuntimeError("backend exploded")

    provider = FakeProvider(ExplodingSandbox())
    executor = QuantflowSandboxExecutor(provider)
    with pytest.raises(RuntimeError, match="backend exploded"):
        await executor.run(
            CodedExperiment(spec=_spec(), code={"files": {"run.py": "x"}, "entrypoint": "python run.py"}),
            _ctx(),
        )
    assert provider.released == ["sb-1"]


@pytest.mark.asyncio
async def test_executor_returns_error_result_for_malformed_code():
    sandbox = FakeSandbox()
    provider = FakeProvider(sandbox)
    executor = QuantflowSandboxExecutor(provider)

    result = await executor.run(CodedExperiment(spec=_spec(), code={"files": {}, "entrypoint": ""}), _ctx())

    assert result.error is not None and "no files" in result.error
    assert provider.acquired == []


@pytest.mark.asyncio
async def test_executor_keeps_coder_manifest_and_harvests_nothing_on_list_failure():
    sandbox = FakeSandbox()
    sandbox.fail_list = True
    executor = QuantflowSandboxExecutor(FakeProvider(sandbox))
    coded = CodedExperiment(
        spec=_spec(),
        code={"files": {"run.py": "x", PROVIDER_MANIFEST_FILENAME: '{"custom": true}'}, "entrypoint": "python run.py"},
    )

    result = await executor.run(coded, _ctx())

    workdir = evolution_workdir(RUN_ID, 0)
    assert sandbox.files[f"{workdir}/{PROVIDER_MANIFEST_FILENAME}"] == '{"custom": true}'
    assert result.artifacts == ()


@pytest.mark.asyncio
async def test_executor_caps_harvested_artifacts():
    sandbox = FakeSandbox(outputs=[f"a-{idx}.csv" for idx in range(MAX_HARVESTED_ARTIFACTS + 10)])
    executor = QuantflowSandboxExecutor(FakeProvider(sandbox))

    result = await executor.run(
        CodedExperiment(spec=_spec(), code={"files": {"run.py": "x"}, "entrypoint": "python run.py"}),
        _ctx(),
    )

    assert len(result.artifacts) == MAX_HARVESTED_ARTIFACTS
    assert result.artifacts[0] == "outputs/a-0.csv"

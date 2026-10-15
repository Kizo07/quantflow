import pytest
from pydantic import ValidationError

from deerflow.config.cli_config import CliConfig


def test_tmux_yaml_false_coerces_to_off():
    assert CliConfig(tmux=False).tmux == "off"


def test_tmux_yaml_true_coerces_to_on():
    assert CliConfig(tmux=True).tmux == "on"


def test_tmux_rejects_unknown_value():
    with pytest.raises(ValidationError):
        CliConfig(tmux="sometimes")

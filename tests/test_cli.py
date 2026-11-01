import pytest
from typer.testing import CliRunner

from py_irt.cli import app

runner = CliRunner()


# Regression test: typer < 0.16 crashed rendering --help with click >= 8.2
# (TypeError: Parameter.make_metavar() missing 1 required positional argument: 'ctx').
@pytest.mark.parametrize("command", [[], ["train"], ["evaluate"], ["train-and-evaluate"]])
def test_help(command):
    result = runner.invoke(app, [*command, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output

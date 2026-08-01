from __future__ import annotations

from typer.testing import CliRunner

from oplm.cli import app

runner = CliRunner()


def test_sweep_subcommand_is_registered() -> None:
    result = runner.invoke(app, ["sweep", "--help"])
    assert result.exit_code == 0
    for command in ("smoke", "coarse", "refine", "replicate", "transfer", "bridge", "confirm"):
        assert command in result.stdout


def test_sweep_analyze_is_registered() -> None:
    result = runner.invoke(app, ["sweep", "analyze", "--help"])
    assert result.exit_code == 0


def test_sweep_coord_check_is_registered() -> None:
    result = runner.invoke(app, ["sweep", "coord-check", "--help"])
    assert result.exit_code == 0

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from oplm.cli import app

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_slurm_subcommand_is_registered() -> None:
    result = runner.invoke(app, ["slurm", "--help"])
    assert result.exit_code == 0
    for command in ("generate", "submit", "status"):
        assert command in result.stdout


def test_generate_from_the_committed_scaling_config(tmp_path: Path) -> None:
    """The general layer works from a plain training config, with no sweep artifacts."""
    result = runner.invoke(
        app,
        [
            "slurm",
            "generate",
            "--config",
            str(REPO_ROOT / "configs" / "scaling.yaml"),
            "--preset",
            "400M",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    script = tmp_path / "oplm-400M.sbatch"
    assert script.exists()
    text = script.read_text()
    assert "#SBATCH --nodes=8" in text
    assert "--preset 400M" in text
    assert "sweep" not in text

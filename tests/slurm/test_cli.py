from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from oplm.cli import app
from tests.cli_output import plain
from tests.slurm.test_submit import _install_stub

if TYPE_CHECKING:
    import pytest

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]

# A minimal, valid `slurm:` block -- everything `load_slurm_config` requires, and nothing
# from `model:` / `train:` / `data:`, since generate/submit/status only ever touch the
# `slurm:` key.
_SLURM_BLOCK = textwrap.dedent(
    """\
    slurm:
      partition: hpc-mid
      time_limit: {default: "168:00:00"}
      cpus_per_task: 128
      gpus_per_node: 8
      exclusive: true
      mem: "0"
      log_dir: /mnt/home/briney/logs
      env_file: /mnt/home/briney/.env
      container_image: /mnt/data/containers/dl.sqsh
      container_mounts: [/mnt/data:/mnt/data, /tmp:/tmp]
      install: pip install oplm[train]
      max_concurrent: 4
      nodes: {170M: 1, 400M: 4}
      max_batch_size: {170M: 256, 400M: 256}
    """
)


def _write_config(path: Path) -> Path:
    path.write_text(_SLURM_BLOCK)
    return path


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


# --- generate: relative --config must resolve to an absolute path ----------------------


def test_generate_injects_auto_resume_into_the_rendered_command(tmp_path: Path) -> None:
    """A requeued Slurm job must resume rather than restart -- see Trainer.auto_resume."""
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
    text = (tmp_path / "oplm-400M.sbatch").read_text()
    assert "train.auto_resume=true" in text


def test_generate_resolves_a_relative_config_path_to_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rendered script must embed an absolute --config path.

    The command ultimately runs on a compute node under a job-scoped working directory, not
    the directory `generate` was invoked from, so a relative --config would silently resolve
    to the wrong (likely nonexistent) file there. This test fails if `config.resolve()` is
    removed from `generate` -- confirmed by temporarily deleting that call, observing this
    test fail, and restoring it (see the task report).
    """
    config = _write_config(tmp_path / "scaling.yaml")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        ["slurm", "generate", "--config", "scaling.yaml", "--preset", "400M", "--out", str(out)],
    )
    assert result.exit_code == 0, result.stdout
    text = (out / "oplm-400M.sbatch").read_text()
    assert f"--config {config.resolve()}" in text
    assert "--config scaling.yaml" not in text


# --- generate: failure paths must raise a clean typer.BadParameter, not a traceback -----


def test_generate_rejects_a_config_with_no_slurm_block(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  hidden_size: 64\n")
    out = tmp_path / "out"

    result = runner.invoke(app, ["slurm", "generate", "--config", str(config), "--out", str(out)])
    assert result.exit_code != 0
    # typer.BadParameter's rendering goes to stderr; `.output` mixes stdout+stderr as a user
    # would see them, so a traceback would show up there too if the exception weren't caught.
    assert "Traceback" not in result.output
    assert "has no `slurm:` block" in result.output


def test_generate_rejects_a_preset_missing_from_the_nodes_table(tmp_path: Path) -> None:
    """The `nodes` PhaseTable's KeyError must surface as a clean BadParameter.

    KeyError.__str__ wraps its message in repr() quoting; generate() must unwrap it (using
    exc.args[0]) rather than let the quoted form leak into the CLI error.
    """
    config = _write_config(tmp_path / "config.yaml")
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        ["slurm", "generate", "--config", str(config), "--preset", "9B", "--out", str(out)],
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "nodes has no entry for phase=None preset='9B'" in result.output
    # The raw repr()-quoted KeyError message (e.g. `"nodes has no entry ..."` with an extra
    # layer of quoting) must not leak through unwrapped.
    assert "KeyError" not in result.output


# --- submit ------------------------------------------------------------------------------


def test_submit_writes_job_ids_into_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path / "config.yaml")
    out = tmp_path / "out"
    generated = runner.invoke(
        app,
        ["slurm", "generate", "--config", str(config), "--preset", "400M", "--out", str(out)],
    )
    assert generated.exit_code == 0, generated.stdout

    log = tmp_path / "sbatch.log"
    _install_stub(
        tmp_path,
        monkeypatch,
        "sbatch",
        f'#!/bin/bash\necho "$@" >> "{log}"\necho 812345\n',
    )

    result = runner.invoke(app, ["slurm", "submit", str(out)])
    assert result.exit_code == 0, result.stdout
    assert "JOB_0: 812345" in plain(result.stdout)

    manifest = json.loads((out / "jobs.json").read_text())
    assert manifest["job_ids"] == {"JOB_0": "812345"}
    # The submitted script argument reached sbatch (not just a hardcoded stub reply).
    assert "oplm-400M.sbatch" in log.read_text()


def test_submit_without_a_prior_generate_reports_missing_manifest(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()

    result = runner.invoke(app, ["slurm", "submit", str(out)])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "no jobs.json in" in result.output
    assert "generate` first" in result.output


# --- status --------------------------------------------------------------------------


def test_status_without_a_prior_generate_reports_missing_manifest(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()

    result = runner.invoke(app, ["slurm", "status", str(out)])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "no jobs.json in" in result.output
    assert "generate` first" in result.output


def test_status_before_any_submission_reports_nothing_submitted_yet(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "jobs.json").write_text(json.dumps({"scripts": ["job.sbatch"], "job_ids": {}}))

    result = runner.invoke(app, ["slurm", "status", str(out)])
    assert result.exit_code == 0, result.stdout
    assert "no jobs submitted yet" in result.stdout


def test_status_reports_scheduler_unavailable_when_squeue_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without squeue, status must say the scheduler is unreachable -- not imply every job
    already finished, which is what an empty `running_job_ids()` result would look like if
    this branch didn't exist."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "jobs.json").write_text(
        json.dumps({"scripts": ["job.sbatch"], "job_ids": {"JOB_0": "812345"}})
    )
    monkeypatch.setenv("PATH", "")

    result = runner.invoke(app, ["slurm", "status", str(out)])
    assert result.exit_code == 0, result.stdout
    output = plain(result.stdout)
    # Cause-neutral wording: this branch also fires when squeue exists but its controller
    # wedged past the query timeout, so the message must not claim squeue is missing.
    assert "scheduler unreachable" in output
    assert "squeue not found" not in output
    assert "JOB_0: 812345" in output
    assert "active" not in output
    assert "finished" not in output


def test_status_distinguishes_active_from_finished_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "jobs.json").write_text(
        json.dumps(
            {
                "scripts": ["a.sbatch", "b.sbatch"],
                "job_ids": {"JOB_0": "812345", "JOB_1": "812346"},
            }
        )
    )
    _install_stub(tmp_path, monkeypatch, "squeue", "#!/bin/bash\nprintf '812345\\n'\n")

    result = runner.invoke(app, ["slurm", "status", str(out)])
    assert result.exit_code == 0, result.stdout
    output = plain(result.stdout)
    assert "JOB_0 (812345): active" in output
    assert "JOB_1 (812346): finished or unknown" in output

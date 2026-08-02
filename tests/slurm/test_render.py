from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from oplm.slurm.config import SlurmConfig
from oplm.slurm.render import (
    JobSpec,
    SubmitEntry,
    accelerate_command,
    render_job,
    render_submit_script,
)
from tests.slurm.test_config import RAW

SLURM = SlurmConfig.from_mapping(RAW)


def _array_spec(tmp_path: Path) -> JobSpec:
    return JobSpec(
        name="oplm-coarse-170M",
        nodes=1,
        time_limit="168:00:00",
        command='python -m oplm.sweep.run --config "$RUN_DIR/run.yaml"',
        array_size=7,
        array_index_file=tmp_path / "jobs" / "170M.jobs",
        base_dir=tmp_path,
    )


def test_array_header_includes_throttle(tmp_path: Path) -> None:
    text = render_job(_array_spec(tmp_path), SLURM)
    assert "#SBATCH --array=0-6%4" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks-per-node=1" in text
    assert "#SBATCH --gres=gpu:8" in text
    assert "#SBATCH --time=168:00:00" in text
    assert "#SBATCH --requeue" in text
    assert "#SBATCH --exclusive" in text


def test_array_logs_use_array_placeholders(tmp_path: Path) -> None:
    text = render_job(_array_spec(tmp_path), SLURM)
    assert "%x_%A_%a.out" in text
    assert "%x_%A_%a.err" in text


def test_single_job_logs_use_job_id() -> None:
    spec = JobSpec(
        name="oplm-scale-400M",
        nodes=8,
        time_limit="168:00:00",
        command="python -m oplm.train --config cfg.yaml",
    )
    text = render_job(spec, SLURM)
    assert "%x_%j.out" in text
    assert "--array" not in text
    assert "SLURM_ARRAY_TASK_ID" not in text


def test_pyxis_flags_are_on_srun_not_sbatch(tmp_path: Path) -> None:
    text = render_job(_array_spec(tmp_path), SLURM)
    for line in text.splitlines():
        if line.startswith("#SBATCH"):
            assert "--container-" not in line
    assert "--container-image=/mnt/data/containers/dl.sqsh" in text
    assert (
        "--container-mounts=/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},"
        "/mnt/data:/mnt/data,/tmp:/tmp" in text
    )
    assert "--no-container-mount-home" in text


def test_workdir_is_created_on_every_node(tmp_path: Path) -> None:
    """JOB_WORK_DIR is node-local /tmp and .env only creates it on the batch node."""
    text = render_job(_array_spec(tmp_path), SLURM)
    assert 'srun --nodes=$SLURM_NNODES --ntasks-per-node=1 mkdir -p "$JOB_WORK_DIR"' in text
    # The mkdir fanout must precede the training srun.
    assert text.index("mkdir -p") < text.index("--container-image=")


def test_rendezvous_variables(tmp_path: Path) -> None:
    text = render_job(_array_spec(tmp_path), SLURM)
    # `export MASTER_ADDR=$(...)` reports export's own exit status (always 0), so a failing
    # `hostname` would not trip `set -e`; the assignment and export must be split.
    assert "\nMASTER_ADDR=$(hostname --ip-address)\n" in text
    assert "\nexport MASTER_ADDR\n" in text
    assert "export MASTER_ADDR=$(" not in text
    # SLURM_JOB_ID is unique per array task, so concurrent jobs cannot collide.
    assert "export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))" in text
    assert "export OMP_NUM_THREADS=1" in text


def test_slurm_vars_expand_inside_the_container(tmp_path: Path) -> None:
    """The inner command is single-quoted so $SLURM_PROCID expands in the container shell."""
    spec = JobSpec(
        name="oplm-coarse-170M",
        nodes=1,
        time_limit="168:00:00",
        command=accelerate_command(module="oplm.sweep.run", gpus_per_node=8, args="--config x"),
        array_size=7,
        array_index_file=tmp_path / "jobs" / "170M.jobs",
        base_dir=tmp_path,
    )
    text = render_job(spec, SLURM)
    inner = text.split("bash -c '", 1)[1]
    assert "--machine_rank $SLURM_PROCID" in inner
    assert "--num_machines $SLURM_NNODES" in inner
    # gpus_per_node is a render-time constant, so the arithmetic is already substituted.
    assert "--num_processes $((SLURM_NNODES * 8))" in inner
    assert "pip install oplm[train]" in inner


def test_array_index_maps_through_the_index_file(tmp_path: Path) -> None:
    text = render_job(_array_spec(tmp_path), SLURM)
    assert "170M.jobs" in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "export RUN_DIR" in text


def test_gres_omitted_for_cpu_only_jobs() -> None:
    spec = JobSpec(
        name="oplm-coarse-analyze",
        nodes=1,
        time_limit="01:00:00",
        command="oplm sweep analyze phase.json",
        gres=False,
    )
    text = render_job(spec, SLURM)
    assert "--gres" not in text


def test_submit_script_wires_afterany_across_arrays() -> None:
    text = render_submit_script(
        [
            SubmitEntry(var="A_400M", script=Path("jobs/400M.sbatch")),
            SubmitEntry(var="A_800M", script=Path("jobs/800M.sbatch")),
            SubmitEntry(var="A_1B", script=Path("jobs/1B.sbatch")),
            SubmitEntry(
                var="ANALYZE",
                script=Path("jobs/analyze.sbatch"),
                depends_on=("A_400M", "A_800M", "A_1B"),
            ),
        ]
    )
    assert "A_400M=$(sbatch --parsable jobs/400M.sbatch)" in text
    assert (
        "ANALYZE=$(sbatch --parsable --dependency=afterany:$A_400M:$A_800M:$A_1B "
        "jobs/analyze.sbatch)" in text
    )
    # afterok would wedge the analyze job the first time a cell diverges.
    assert "afterok" not in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_rendered_scripts_are_valid_bash(tmp_path: Path) -> None:
    for index, text in enumerate(
        (
            render_job(_array_spec(tmp_path), SLURM),
            render_submit_script([SubmitEntry(var="A", script=Path("jobs/a.sbatch"))]),
        )
    ):
        script = tmp_path / f"candidate{index}.sh"
        script.write_text(text)
        subprocess.run(["bash", "-n", str(script)], check=True)

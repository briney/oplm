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


def test_header_gains_requeue_signal_and_append_mode(tmp_path: Path) -> None:
    """A requeued job must get an in-job warning signal and append (not truncate) its logs."""
    text = render_job(_array_spec(tmp_path), SLURM)
    assert "#SBATCH --signal=USR1@600" in text
    assert "#SBATCH --open-mode=append" in text


def test_restart_banner_follows_env_source(tmp_path: Path) -> None:
    text = render_job(_array_spec(tmp_path), SLURM)
    banner = 'echo "=== $(date -Is) start; restart_count=${SLURM_RESTART_COUNT:-0} ==="'
    assert banner in text
    assert text.index(f"source {SLURM.env_file}") < text.index(banner)


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


def test_nccl_hardening_env_block(tmp_path: Path) -> None:
    """The full NCCL-hardening env block replaces the old bare `NCCL_DEBUG=INFO` line."""
    text = render_job(_array_spec(tmp_path), SLURM)
    assert f"export NCCL_DEBUG={SLURM.nccl_debug}" in text
    assert "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1" in text
    assert "export TORCH_NCCL_TRACE_BUFFER_SIZE=2000" in text
    assert "export TORCH_NCCL_DUMP_ON_TIMEOUT=1" in text
    assert (
        f"export TORCH_FR_DUMP_TEMP_FILE={SLURM.log_dir}/nccl_trace_${{SLURM_JOB_ID}}_rank"
        in text
    )
    assert "export NCCL_DEBUG=INFO" not in text


def test_nccl_debug_uses_the_configured_value(tmp_path: Path) -> None:
    from dataclasses import replace

    custom = replace(SLURM, nccl_debug="INFO")
    text = render_job(_array_spec(tmp_path), custom)
    assert "export NCCL_DEBUG=INFO" in text


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


def test_wrapper_replaces_the_bare_srun_tail(tmp_path: Path) -> None:
    """The srun must run under `set +e` so a nonzero training exit does not kill the script."""
    text = render_job(_array_spec(tmp_path), SLURM)
    assert "set +e\nsrun" in text
    assert "\nSTATUS=$?\nset -e\n" in text


def test_wrapper_handles_clean_completion(tmp_path: Path) -> None:
    text = render_job(_array_spec(tmp_path), SLURM)
    assert 'if [ "$STATUS" -eq 0 ]; then' in text
    assert 'echo "training complete"' in text
    assert "  exit 0\nfi" in text


def test_wrapper_interpolates_the_configured_requeue_budget(tmp_path: Path) -> None:
    text = render_job(_array_spec(tmp_path), SLURM)
    assert "RESTARTS=${SLURM_RESTART_COUNT:-0}" in text
    assert f'if [ "$RESTARTS" -ge {SLURM.max_requeues} ]; then' in text
    assert f"requeue budget ({SLURM.max_requeues}) exhausted" in text
    assert 'scontrol requeue "$SLURM_JOB_ID"' in text


def test_wrapper_uses_a_custom_requeue_budget(tmp_path: Path) -> None:
    from dataclasses import replace

    custom = replace(SLURM, max_requeues=3)
    text = render_job(_array_spec(tmp_path), custom)
    assert 'if [ "$RESTARTS" -ge 3 ]; then' in text
    assert "requeue budget (3) exhausted" in text


def test_array_job_progress_dir_uses_run_dir_expansion(tmp_path: Path) -> None:
    """Array jobs pass a `$RUN_DIR`-based progress dir; it must reach the shell unexpanded."""
    spec = JobSpec(
        name="oplm-coarse-170M",
        nodes=1,
        time_limit="168:00:00",
        command='python -m oplm.sweep.run --config "$RUN_DIR/run.yaml"',
        array_size=7,
        array_index_file=tmp_path / "jobs" / "170M.jobs",
        base_dir=tmp_path,
        progress_dir="$RUN_DIR/output",
    )
    text = render_job(spec, SLURM)
    assert 'STEP_FILE="$RUN_DIR/output/.last_requeue_step"' in text
    assert 'DRAIN_MARKER="$RUN_DIR/output/.drained"' in text
    assert 'ls -d "$RUN_DIR/output"/checkpoint-*' in text
    # The whole guard -- its step-file *write* included -- is skipped on the drain path,
    # and its comparison is only meaningful once at least one restart has happened.
    assert 'if [ "$DRAINED" -ne 1 ]; then' in text
    assert 'if [ "$RESTARTS" -ge 1 ]; then' in text


def test_no_progress_guard_omitted_when_progress_dir_is_none() -> None:
    """Non-training jobs (progress_dir=None) still requeue, just without checkpoint tracking."""
    spec = JobSpec(
        name="oplm-scale-400M",
        nodes=8,
        time_limit="168:00:00",
        command="python -m oplm.train --config cfg.yaml",
    )
    text = render_job(spec, SLURM)
    assert "STEP_FILE" not in text
    assert "PREV_STEP" not in text
    assert "DRAIN_MARKER" not in text
    assert "no checkpoint progress" not in text
    assert 'scontrol requeue "$SLURM_JOB_ID"' in text


def test_wrapper_recognizes_drain_by_marker_or_exit_code(tmp_path: Path) -> None:
    """Drain detection is marker-file OR exit-85, and the marker is consumed.

    `accelerate launch --multi_gpu` reports any worker failure -- including a clean
    drain's exit 85 -- as its own exit 1 (torchelastic re-raises ChildFailedError), so
    the wrapper cannot key the drain branch on `$STATUS` alone. The trainer's `.drained`
    marker (oplm.training.signals.DRAIN_MARKER_NAME) is the signal that survives the
    flattening; the bare exit-code check remains for launcher-less runs.
    """
    spec = JobSpec(
        name="oplm-scale-400M",
        nodes=8,
        time_limit="168:00:00",
        command="python -m oplm.train --config cfg.yaml",
        progress_dir=str(tmp_path / "output"),
    )
    text = render_job(spec, SLURM)
    assert 'if [ "$STATUS" -eq 85 ]; then DRAINED=1; fi' in text
    assert 'if [ -f "$DRAIN_MARKER" ]; then' in text
    assert 'rm -f "$DRAIN_MARKER"' in text


def test_inner_bash_ignores_usr1_so_the_drain_signal_cannot_kill_the_plumbing(
    tmp_path: Path,
) -> None:
    """The `bash -c` wrapper starts with `trap "" USR1`.

    Slurm's `--signal=USR1@600` is delivered to every process in the step's cgroup; the
    task-leader bash and the `accelerate launch` parent have no USR1 handler and would
    die instantly, tearing the step down before the trainer's drain checkpoint commits.
    SIG_IGN is inherited across exec (shielding `accelerate launch` too), while the
    trainer ranks explicitly re-install their own handler (signal.signal overrides an
    inherited ignore), so the drain itself still works.
    """
    text = render_job(_array_spec(tmp_path), SLURM)
    assert 'bash -c \'trap "" USR1; ' in text
    # SIGTERM stays un-ignored: torchelastic installs its own TERM handler regardless,
    # and scancel's TERM->KILL teardown semantics must not change.
    assert 'trap "" USR1 TERM' not in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_rendered_scripts_are_valid_bash(tmp_path: Path) -> None:
    progress_spec = JobSpec(
        name="oplm-coarse-170M",
        nodes=1,
        time_limit="168:00:00",
        command='python -m oplm.sweep.run --config "$RUN_DIR/run.yaml"',
        array_size=7,
        array_index_file=tmp_path / "jobs" / "170M.jobs",
        base_dir=tmp_path,
        progress_dir="$RUN_DIR/output",
    )
    for index, text in enumerate(
        (
            render_job(_array_spec(tmp_path), SLURM),
            render_job(progress_spec, SLURM),
            render_submit_script([SubmitEntry(var="A", script=Path("jobs/a.sbatch"))]),
        )
    ):
        script = tmp_path / f"candidate{index}.sh"
        script.write_text(text)
        subprocess.run(["bash", "-n", str(script)], check=True)

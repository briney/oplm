"""Render Slurm job scripts from a :class:`~oplm.slurm.config.SlurmConfig`.

One launcher form is used for every job. The multi-node ``srun`` form degrades correctly at
``SLURM_NNODES=1``, so there is no separate single-node path to maintain or test.

Quoting matters here. The inner training command is wrapped in ``bash -c '...'`` with *single*
quotes, so ``$SLURM_NNODES`` / ``$SLURM_PROCID`` / ``$MASTER_ADDR`` expand in the container's
shell (they reach it via ``--export=ALL``), not at render time. Anything that must be substituted
at render time is interpolated into the template directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from oplm.slurm.config import SlurmConfig

# The user's home mount is user-specific, so it is added at render time rather than configured.
_HOME_MOUNT = "/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER}"


@dataclass(frozen=True)
class JobSpec:
    """One Slurm job: a single run, or an array over homogeneous jobs."""

    name: str
    nodes: int
    time_limit: str
    command: str
    array_size: int | None = None
    array_index_file: Path | None = None
    gres: bool = True
    base_dir: Path | None = None
    # Shell-expandable path to this job's training output dir, used by the requeue wrapper's
    # no-progress guard (a plain path for a single job; "$RUN_DIR/..." for an array task). None
    # for non-training jobs, which skips the guard entirely (budget-only requeue).
    progress_dir: str | None = None


@dataclass(frozen=True)
class SubmitEntry:
    """One ``sbatch`` invocation in a generated ``submit.sh``."""

    var: str
    script: Path
    depends_on: tuple[str, ...] = ()


def _header(spec: JobSpec, slurm: SlurmConfig) -> list[str]:
    suffix = "%A_%a" if spec.array_size is not None else "%j"
    lines = [
        "#!/bin/bash",
        "",
        "# --- job ---",
        f"#SBATCH --job-name={spec.name}",
        f"#SBATCH --partition={slurm.partition}",
    ]
    if slurm.account is not None:
        lines.append(f"#SBATCH --account={slurm.account}")
    lines += [
        "",
        "# --- nodes & resources ---",
        f"#SBATCH --nodes={spec.nodes}",
        "#SBATCH --ntasks-per-node=1",
    ]
    if spec.gres:
        lines.append(f"#SBATCH --gres=gpu:{slurm.gpus_per_node}")
    lines.append(f"#SBATCH --cpus-per-task={slurm.cpus_per_task}")
    lines.append(f"#SBATCH --mem={slurm.mem}")
    if slurm.exclusive:
        lines.append("#SBATCH --exclusive")
    lines += [
        f"#SBATCH --time={spec.time_limit}",
        "#SBATCH --requeue",
        # Slurm sends USR1 10 minutes before the time limit so the trainer can drain cleanly
        # (see the Task 1.5 signal handling) instead of being SIGTERM'd mid-checkpoint.
        "#SBATCH --signal=USR1@600",
        # A requeued run reuses the same %j/%A_%a log path; append instead of truncating it so
        # the pre-requeue tail (including the drain/crash-loop diagnosis) survives.
        "#SBATCH --open-mode=append",
        "",
        "# --- logs ---",
        f"#SBATCH --output={slurm.log_dir}/%x_{suffix}.out",
        f"#SBATCH --error={slurm.log_dir}/%x_{suffix}.err",
    ]
    if spec.array_size is not None:
        lines += [
            "",
            "# --- array ---",
            f"#SBATCH --array=0-{spec.array_size - 1}%{slurm.max_concurrent}",
        ]
    return lines


def _array_lookup(spec: JobSpec) -> list[str]:
    if spec.array_index_file is None or spec.base_dir is None:
        return []
    return [
        "",
        "# Map this array index to its run. One run id per line; index = line number - 1.",
        f'BASE_DIR="{spec.base_dir}"',
        f'INDEX_FILE="{spec.array_index_file}"',
        'RUN_ID=$(awk "NR==$((SLURM_ARRAY_TASK_ID + 1))" "$INDEX_FILE")',
        'if [ -z "$RUN_ID" ]; then',
        '  echo "no run for array index $SLURM_ARRAY_TASK_ID" >&2',
        "  exit 1",
        "fi",
        'export RUN_DIR="$BASE_DIR/runs/$RUN_ID"',
    ]


def render_job(spec: JobSpec, slurm: SlurmConfig) -> str:
    """Render one sbatch script.

    Args:
        spec: What to run and at what size.
        slurm: Cluster settings.

    Returns:
        Complete script text, ending in a newline.
    """
    mounts = ",".join((_HOME_MOUNT, *slurm.container_mounts))
    lines = _header(spec, slurm)
    lines += [
        "",
        "set -euo pipefail",
        "",
        "# Creates JOB_WORK_DIR and exports object-storage / W&B credentials.",
        f"source {slurm.env_file}",
        # Marks the top of every attempt (fresh submission or requeue) in the append-mode log,
        # so `restart_count` is visible without cross-referencing `scontrol show job`.
        'echo "=== $(date -Is) start; restart_count=${SLURM_RESTART_COUNT:-0} ==="',
    ]
    lines += _array_lookup(spec)
    lines += [
        "",
        "# JOB_WORK_DIR is node-local /tmp; .env only created it on the batch node.",
        'srun --nodes=$SLURM_NNODES --ntasks-per-node=1 mkdir -p "$JOB_WORK_DIR"',
        "",
        "# Distributed rendezvous. The sbatch body runs on the rank-0 node, and SLURM_JOB_ID is",
        "# unique per array task, so concurrent jobs cannot collide on a port.",
        "# Assignment and export are split rather than combined into one `export VAR=$(cmd)`",
        "# line: bash would then report export's own exit status (always 0) instead of the",
        "# command substitution's, so `set -e` could not catch a failing hostname lookup and the",
        "# job would hang at rendezvous with an empty MASTER_ADDR.",
        "MASTER_ADDR=$(hostname --ip-address)",
        "export MASTER_ADDR",
        "export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))",
        "export NCCL_DEBUG=INFO",
        "export OMP_NUM_THREADS=1",
        "",
        # `set +e` / `set -e` bracket the srun so a nonzero training exit (drain, crash) does
        # not immediately kill the script under the top-level `set -euo pipefail` -- the
        # requeue wrapper below needs to inspect $STATUS first.
        "set +e",
        "srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \\",
        "  --export=ALL \\",
        f"  --container-image={slurm.container_image} \\",
        f"  --container-mounts={mounts} \\",
        '  --container-workdir="$JOB_WORK_DIR" \\',
        "  --no-container-mount-home \\",
        f"  bash -c '{slurm.install} && \\",
        f"    {spec.command}'",
    ]
    lines += _requeue_wrapper(spec, slurm)
    lines.append("")
    return "\n".join(lines)


def _requeue_wrapper(spec: JobSpec, slurm: SlurmConfig) -> list[str]:
    """Budget- and progress-aware requeue logic run after the training `srun` exits.

    Exit 0 is a clean finish. Exit ``DRAIN_EXIT_CODE`` (85, see ``oplm.training.signals``)
    requeues unconditionally as long as the requeue budget (``slurm.max_requeues``) is not
    exhausted -- it bypasses the no-progress guard below, but not the budget cap. Any other
    nonzero exit requeues only if budget remains *and* checkpoint progress was made since the
    previous restart; two consecutive restarts with no step advance is a crash loop, and the
    job exits without requeueing.

    When ``spec.progress_dir`` is ``None`` (a non-training job), the no-progress guard and its
    step-tracking file are omitted entirely -- the wrapper requeues on budget alone.
    """
    lines = [
        "STATUS=$?",
        "set -e",
        'if [ "$STATUS" -eq 0 ]; then',
        '  echo "training complete"',
        "  exit 0",
        "fi",
        "RESTARTS=${SLURM_RESTART_COUNT:-0}",
        f'if [ "$RESTARTS" -ge {slurm.max_requeues} ]; then',
        f'  echo "requeue budget ({slurm.max_requeues}) exhausted; exiting $STATUS" >&2',
        '  exit "$STATUS"',
        "fi",
    ]
    if spec.progress_dir is not None:
        lines += [
            f'STEP_FILE="{spec.progress_dir}/.last_requeue_step"',
            f'CURRENT_STEP=$(ls -d "{spec.progress_dir}"/checkpoint-* 2>/dev/null \\',
            "  | sed 's/.*checkpoint-//' | grep -E '^[0-9]+$' | sort -n | tail -1)",
            "CURRENT_STEP=${CURRENT_STEP:-0}",
            'if [ "$STATUS" -ne 85 ] && [ "$RESTARTS" -ge 1 ]; then',
            '  PREV_STEP=$(cat "$STEP_FILE" 2>/dev/null || echo -1)',
            '  if [ "$CURRENT_STEP" -le "$PREV_STEP" ]; then',
            (
                '    echo "no checkpoint progress since last restart (step $CURRENT_STEP); '
                'crash loop -- not requeueing" >&2'
            ),
            '    exit "$STATUS"',
            "  fi",
            "fi",
            'echo "$CURRENT_STEP" > "$STEP_FILE"',
            'echo "requeueing (exit=$STATUS, restarts=$RESTARTS, step=$CURRENT_STEP)"',
        ]
    else:
        lines.append('echo "requeueing (exit=$STATUS, restarts=$RESTARTS)"')
    lines.append('scontrol requeue "$SLURM_JOB_ID"')
    return lines


def accelerate_command(
    *, module: str, gpus_per_node: int, args: str, mixed_precision: str = "bf16"
) -> str:
    """Build the inner ``accelerate launch`` command for a container shell.

    ``$SLURM_NNODES`` and ``$SLURM_PROCID`` are left unexpanded on purpose: the caller embeds this
    inside single quotes so the container's shell resolves them.
    """
    return (
        "accelerate launch \\\n"
        "    --multi_gpu \\\n"
        f"    --mixed_precision {mixed_precision} \\\n"
        "    --num_machines $SLURM_NNODES \\\n"
        f"    --num_processes $((SLURM_NNODES * {gpus_per_node})) \\\n"
        "    --machine_rank $SLURM_PROCID \\\n"
        '    --main_process_ip "$MASTER_ADDR" \\\n'
        "    --main_process_port $MASTER_PORT \\\n"
        f"    -m {module} \\\n"
        f"    {args}"
    )


def render_submit_script(entries: list[SubmitEntry]) -> str:
    """Render a ``submit.sh`` that submits every job and wires dependencies.

    Dependencies use ``afterany``, not ``afterok``: a downstream job (e.g. one that inspects the
    outputs of several upstream jobs) should still run even if one upstream job fails, so its
    failure is visible rather than silent. Under ``afterok`` the first failed upstream job would
    leave the downstream job in ``DependencyNeverSatisfied`` forever.
    """
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        "# Run from the base directory regardless of the caller's cwd.",
        'cd "$(dirname "$0")/.."',
        "",
    ]
    for entry in entries:
        if entry.depends_on:
            deps = ":".join(f"${var}" for var in entry.depends_on)
            call = f"sbatch --parsable --dependency=afterany:{deps} {entry.script}"
        else:
            call = f"sbatch --parsable {entry.script}"
        lines.append(f"{entry.var}=$({call})")
        lines.append(f'echo "submitted {entry.script}: ${entry.var}"')
    lines.append("")
    return "\n".join(lines)

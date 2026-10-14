#!/bin/bash
# Late-loss investigation (docs/LATE_LOSS_INVESTIGATION.md §4): is the compiled training step
# computing a different loss from eager? Scores checkpoint-600000 on the exact batches rank 0
# trained on, each batch both eagerly and through torch.compile in train mode with a backward
# pass (the trainer's step, weights untouched), behind a single-process DDP wrapper as in the
# runs. In ONE process, so the compile cache carries through, it scores:
#   1. "recent" windows just before step 600k, which this checkpoint had just trained on
#      (if the eager loss there falls to the logged train loss, the gap was within-shard recency);
#   2. "far" windows at 640k-669k, never seen by this checkpoint;
#   3. the epoch's closing short batches (the run's exact sizes) through the compiled step;
#   4. the same "far" windows again.
# compiled_minus_eager ~ -0.06 at 1-2 means the compiled graph differs from eager; a change
# between 2 and 4 means the short batches switched the graph. TORCH_LOGS=recompiles logs every
# recompilation with its reason. A second pass repeats it without DDP.
#
# Submit from a login node:
#   sbatch docs/late-loss/compile-replay.sbatch
# Overrides (environment variables, all optional):
#   OPLM_GIT_REF=feat/stream-replay
#   OPLM_CHECKPOINT=/mnt/home/<you>/projects/oplm/scaling/170M/checkpoint-600000
#   OPLM_WORLD_SIZE=64
#   OPLM_BATCHES=20
#   OPLM_RESULTS=/mnt/home/<you>/late-loss/compile-replay/<jobid>
#
#SBATCH --job-name=oplm-compile-replay
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/home/briney/logs/%x_%j.out
#SBATCH --error=/mnt/home/briney/logs/%x_%j.err

set -euo pipefail

ENV_FILE="${OPLM_ENV_FILE:-/mnt/home/${SLURM_JOB_USER}/.env}"
CONTAINER="${OPLM_CONTAINER:-/mnt/data/containers/deeplearning_v2026-05-26.sqsh}"
GIT_REF="${OPLM_GIT_REF:-feat/stream-replay}"
CHECKPOINT="${OPLM_CHECKPOINT:-/mnt/home/${SLURM_JOB_USER}/projects/oplm/scaling/170M/checkpoint-600000}"
WORLD_SIZE_RUN="${OPLM_WORLD_SIZE:-64}"
BATCHES="${OPLM_BATCHES:-20}"
RESULTS="${OPLM_RESULTS:-/mnt/home/${SLURM_JOB_USER}/late-loss/compile-replay/${SLURM_JOB_ID}}"
export GIT_REF CHECKPOINT WORLD_SIZE_RUN BATCHES RESULTS

source "$ENV_FILE"  # creates JOB_WORK_DIR
mkdir -p "$RESULTS" "$JOB_WORK_DIR"
echo "=== $(date -Is) start; job=$SLURM_JOB_ID node=$(hostname) results=$RESULTS ==="

INNER="$RESULTS/run-inner.sh"
cat > "$INNER" <<'EOF'
set -euo pipefail
python -m pip install --quiet "oplm[train] @ git+https://github.com/briney/oplm@$GIT_REF"
python -m pip freeze | grep -i -E '^(oplm|torch|transformers)' > "$RESULTS/pip-freeze.txt"
export TORCH_LOGS=recompiles

steps() { python -c "import sys; a, b, n = map(int, sys.argv[1:]); print(' '.join(f'--step {a + i * (b - a) // (n - 1)}' for i in range(n)))" "$@"; }
RECENT=$(steps 597000 599700 10)
FAR=$(steps 640000 669000 15)

run() {  # name, extra flags...
  local name=$1; shift
  # shellcheck disable=SC2086
  oplm stream-replay "$CHECKPOINT" --world-size "$WORLD_SIZE_RUN" --batches "$BATCHES" \
    --compiled --partial-after 25 "$@" $RECENT $FAR $FAR --out "$RESULTS/$name.csv" \
    > "$RESULTS/$name.log" 2>&1 || echo "$name failed: $?" >> "$RESULTS/status.txt"
}
run ddp --ddp
run bare
EOF

srun --nodes=1 --ntasks-per-node=1 --export=ALL \
  --container-image="$CONTAINER" \
  --container-mounts="/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},/mnt/data:/mnt/data,/tmp:/tmp" \
  --container-workdir="$JOB_WORK_DIR" \
  --no-container-mount-home \
  bash "$INNER"

echo "=== $(date -Is) done; results in $RESULTS ==="

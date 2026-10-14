#!/bin/bash
# Late-loss investigation (docs/LATE_LOSS_INVESTIGATION.md §4): score the replication's frozen
# checkpoint-600000 on the exact batches rank 0 trained on just before and just after the
# data-epoch boundary (global step 669,312 at world size 64 with 4 workers), plus a control span
# mid-epoch. A jump in the frozen model's loss across the boundary means the second-epoch stream
# is harder data; no jump means the original runs' +0.045 came from the model's trajectory.
# The 600k checkpoint never saw the 640k-669k pre-boundary data, while ~77% of the post-boundary
# deepclust70 rows were seen once in epoch 0, so any bias favours the post-boundary side.
# One GPU, one-shot; output is a CSV per span plus the console tables.
#
# Submit from a login node:
#   sbatch docs/late-loss/stream-replay.sbatch
# Overrides (environment variables, all optional):
#   OPLM_GIT_REF=feat/stream-replay
#   OPLM_CHECKPOINT=/mnt/home/<you>/projects/oplm/scaling/170M/checkpoint-600000
#   OPLM_WORLD_SIZE=64        global batch / train.batch_size of the run
#   OPLM_WINDOWS=30           windows per span
#   OPLM_BATCHES=20           batches (of train.batch_size rows) per window
#   OPLM_RESULTS=/mnt/home/<you>/late-loss/stream-replay/<jobid>
#
#SBATCH --job-name=oplm-stream-replay
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/briney/logs/%x_%j.out
#SBATCH --error=/mnt/home/briney/logs/%x_%j.err

set -euo pipefail

ENV_FILE="${OPLM_ENV_FILE:-/mnt/home/${SLURM_JOB_USER}/.env}"
CONTAINER="${OPLM_CONTAINER:-/mnt/data/containers/deeplearning_v2026-05-26.sqsh}"
GIT_REF="${OPLM_GIT_REF:-feat/stream-replay}"
CHECKPOINT="${OPLM_CHECKPOINT:-/mnt/home/${SLURM_JOB_USER}/projects/oplm/scaling/170M/checkpoint-600000}"
WORLD_SIZE_RUN="${OPLM_WORLD_SIZE:-64}"
WINDOWS="${OPLM_WINDOWS:-30}"
BATCHES="${OPLM_BATCHES:-20}"
RESULTS="${OPLM_RESULTS:-/mnt/home/${SLURM_JOB_USER}/late-loss/stream-replay/${SLURM_JOB_ID}}"
export GIT_REF CHECKPOINT WORLD_SIZE_RUN WINDOWS BATCHES RESULTS

source "$ENV_FILE"  # creates JOB_WORK_DIR
mkdir -p "$RESULTS" "$JOB_WORK_DIR"
echo "=== $(date -Is) start; job=$SLURM_JOB_ID node=$(hostname) results=$RESULTS ==="

INNER="$RESULTS/run-inner.sh"
cat > "$INNER" <<'EOF'
set -euo pipefail
python -m pip install --quiet "oplm[train] @ git+https://github.com/briney/oplm@$GIT_REF"
python -m pip freeze | grep -i -E '^(oplm|torch|transformers)' > "$RESULTS/pip-freeze.txt"

# Evenly spaced window starts over [first, last], as repeated --step options.
steps() { python -c "import sys; a, b, n = map(int, sys.argv[1:]); print(' '.join(f'--step {a + i * (b - a) // (n - 1)}' for i in range(n)))" "$@"; }

run() {  # name first last
  # shellcheck disable=SC2046
  oplm stream-replay "$CHECKPOINT" --world-size "$WORLD_SIZE_RUN" --batches "$BATCHES" \
    $(steps "$2" "$3" "$WINDOWS") --out "$RESULTS/$1.csv" 2>&1 | tee "$RESULTS/$1.log"
}
run control 450000 479000     # mid-epoch reference span, same length
run pre     640000 669000     # last 29k steps of epoch 0 (boundary at 669,312)
run post    669400 698400     # first 29k steps of epoch 1
EOF

srun --nodes=1 --ntasks-per-node=1 --export=ALL \
  --container-image="$CONTAINER" \
  --container-mounts="/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},/mnt/data:/mnt/data,/tmp:/tmp" \
  --container-workdir="$JOB_WORK_DIR" \
  --no-container-mount-home \
  bash "$INNER"

echo "=== $(date -Is) done; results in $RESULTS ==="

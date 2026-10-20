#!/bin/bash
# Time packed attention by row width and kernel (dense-mask SDPA vs FlexAttention) on one GPU
# (docs/perf/attention_width_bench.py), for each preset in OPLM_PRESETS. About twenty minutes.
#
# Submit from a login node:
#   OPLM_CONFIG=/path/to/run.yaml sbatch docs/perf/attention-width-bench.sbatch
# Overrides (environment variables):
#   OPLM_CONFIG=<yaml>      REQUIRED: a training YAML whose data.train supplies sequence lengths
#   OPLM_PRESETS="170M 1B"  presets to time (default below)
#   OPLM_GIT_REF=<commit>   oplm version to install (default below)
#   OPLM_BENCH_ARGS="..."   extra attention_width_bench.py arguments, e.g.
#                           "--checkpointing --shapes 84x512,21x2048" (checkpointing otherwise
#                           follows the run config's model.gradient_checkpointing)
#   OPLM_RESULTS=/mnt/home/<you>/oplm-perf/attention-width/<jobid>
#
#SBATCH --job-name=oplm-attention-width
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/briney/logs/%x_%j.out
#SBATCH --error=/mnt/home/briney/logs/%x_%j.err

set -euo pipefail

: "${OPLM_CONFIG:?set OPLM_CONFIG to a training YAML with data.train}"
ENV_FILE="${OPLM_ENV_FILE:-/mnt/home/${SLURM_JOB_USER}/.env}"
CONTAINER="${OPLM_CONTAINER:-/mnt/data/containers/deeplearning_v2026-05-26.sqsh}"
GIT_REF="${OPLM_GIT_REF:-main}"
PRESETS="${OPLM_PRESETS:-170M 1B}"
RESULTS="${OPLM_RESULTS:-/mnt/home/${SLURM_JOB_USER}/oplm-perf/attention-width/${SLURM_JOB_ID}}"
OPLM_BENCH_ARGS="${OPLM_BENCH_ARGS:-}"
export GIT_REF RESULTS PRESETS OPLM_CONFIG OPLM_BENCH_ARGS

source "$ENV_FILE"  # creates JOB_WORK_DIR
mkdir -p "$RESULTS" "$JOB_WORK_DIR"

INNER="$RESULTS/run-inner.sh"
cat > "$INNER" <<'EOF'
set -euo pipefail
python -m pip install --quiet "oplm[train] @ git+https://github.com/briney/oplm@$GIT_REF"
for script in attention_width_bench.py packing_memory.py; do
  curl -sSfL "https://raw.githubusercontent.com/briney/oplm/$GIT_REF/docs/perf/$script" \
    -o "$RESULTS/$script"
done
python -m pip freeze | grep -i -E '^(oplm|torch|triton)' > "$RESULTS/pip-freeze.txt"
nvidia-smi > "$RESULTS/nvidia-smi.txt" 2>&1 || true
for preset in $PRESETS; do
  # shellcheck disable=SC2086  # OPLM_BENCH_ARGS is intentionally word-split
  python "$RESULTS/attention_width_bench.py" --config "$OPLM_CONFIG" --preset "$preset" \
    $OPLM_BENCH_ARGS 2>&1 | tee "$RESULTS/bench-$preset.txt"
done
EOF

srun --nodes=1 --ntasks-per-node=1 --export=ALL \
  --container-image="$CONTAINER" \
  --container-mounts="/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},/mnt/data:/mnt/data,/tmp:/tmp" \
  --container-workdir="$JOB_WORK_DIR" \
  --no-container-mount-home \
  bash "$INNER"

echo "=== done; results in $RESULTS ==="

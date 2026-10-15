#!/bin/bash
# How many packed rows fit one GPU without activation checkpointing, against how many rows
# real batches need (docs/perf/packing_memory.py). About fifteen minutes.
#
# Submit from a login node:
#   sbatch docs/perf/packing-memory.sbatch
# Overrides (environment variables, all optional):
#   OPLM_GIT_REF=<commit>   oplm version to install and check (default below)
#   OPLM_CHECKPOINT=<dir>   checkpoint with config.yaml and hf/ (default below)
#   OPLM_RESULTS=/mnt/home/<you>/oplm-perf/packing-memory/<jobid>
#
#SBATCH --job-name=oplm-packing-memory
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

ENV_FILE="${OPLM_ENV_FILE:-/mnt/home/${SLURM_JOB_USER}/.env}"
CONTAINER="${OPLM_CONTAINER:-/mnt/data/containers/deeplearning_v2026-05-26.sqsh}"
GIT_REF="${OPLM_GIT_REF:-perf/profile-mode}"
CHECKPOINT="${OPLM_CHECKPOINT:-/mnt/home/${SLURM_JOB_USER}/projects/oplm/scaling/170M/checkpoint-450000}"
RESULTS="${OPLM_RESULTS:-/mnt/home/${SLURM_JOB_USER}/oplm-perf/packing-memory/${SLURM_JOB_ID}}"
export GIT_REF RESULTS CHECKPOINT

source "$ENV_FILE"  # creates JOB_WORK_DIR
mkdir -p "$RESULTS" "$JOB_WORK_DIR"

INNER="$RESULTS/run-inner.sh"
cat > "$INNER" <<'EOF'
set -euo pipefail
python -m pip install --quiet "oplm[train] @ git+https://github.com/briney/oplm@$GIT_REF"
curl -sSfL "https://raw.githubusercontent.com/briney/oplm/$GIT_REF/docs/perf/packing_memory.py" \
  -o "$RESULTS/packing_memory.py"
python -m pip freeze | grep -i -E '^(oplm|torch|triton)' > "$RESULTS/pip-freeze.txt"
nvidia-smi > "$RESULTS/nvidia-smi.txt" 2>&1 || true
python "$RESULTS/packing_memory.py" --checkpoint "$CHECKPOINT" 2>&1 | tee "$RESULTS/packing_memory.txt"
EOF

srun --nodes=1 --ntasks-per-node=1 --export=ALL \
  --container-image="$CONTAINER" \
  --container-mounts="/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},/mnt/data:/mnt/data,/tmp:/tmp" \
  --container-workdir="$JOB_WORK_DIR" \
  --no-container-mount-home \
  bash "$INNER"

echo "=== done; results in $RESULTS ==="

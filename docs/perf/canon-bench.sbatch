#!/bin/bash
# Time candidate Canon depthwise-convolution implementations at 170M training shapes on one
# GPU (docs/perf/canon_bench.py). One-shot, about ten minutes.
#
# Submit from a login node:
#   sbatch docs/perf/canon-bench.sbatch
# Overrides (environment variables, all optional):
#   OPLM_GIT_REF=<commit>   oplm version to install and benchmark (default below)
#   OPLM_RESULTS=/mnt/home/<you>/oplm-perf/canon-bench/<jobid>
#
#SBATCH --job-name=oplm-canon-bench
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
RESULTS="${OPLM_RESULTS:-/mnt/home/${SLURM_JOB_USER}/oplm-perf/canon-bench/${SLURM_JOB_ID}}"
export GIT_REF RESULTS

source "$ENV_FILE"  # creates JOB_WORK_DIR
mkdir -p "$RESULTS" "$JOB_WORK_DIR"

INNER="$RESULTS/run-inner.sh"
cat > "$INNER" <<'EOF'
set -euo pipefail
python -m pip install --quiet "oplm[train] @ git+https://github.com/briney/oplm@$GIT_REF"
curl -sSfL "https://raw.githubusercontent.com/briney/oplm/$GIT_REF/docs/perf/canon_bench.py" \
  -o "$RESULTS/canon_bench.py"
python -m pip freeze | grep -i -E '^(oplm|torch|triton)' > "$RESULTS/pip-freeze.txt"
nvidia-smi > "$RESULTS/nvidia-smi.txt" 2>&1 || true
python "$RESULTS/canon_bench.py" 2>&1 | tee "$RESULTS/canon_bench.txt"
EOF

srun --nodes=1 --ntasks-per-node=1 --export=ALL \
  --container-image="$CONTAINER" \
  --container-mounts="/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},/mnt/data:/mnt/data,/tmp:/tmp" \
  --container-workdir="$JOB_WORK_DIR" \
  --no-container-mount-home \
  bash "$INNER"

echo "=== done; results in $RESULTS ==="

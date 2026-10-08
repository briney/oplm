#!/bin/bash
# Milestone-0 Task 7 acceptance run on one B200 (docs/superpowers/plans/
# 2026-10-07-fold-m0-foundations-and-kernels.md, Task 7): GPU parity tests, the two
# kernel benchmarks, and the MLM/EMA regression files on cuda-bf16. One-shot job,
# no requeue. Follows the SUNK/Pyxis launcher form from docs/SLURM.md §5.
#
# Submit from a login node:
#   sbatch docs/fold/b200-task7.sbatch
# Overrides (environment variables, all optional):
#   OPLM_REPO=/mnt/home/<you>/git/oplm   use a mounted checkout instead of cloning
#   OPLM_GIT_REF=feat/fold-m0            branch/tag/sha to clone when OPLM_REPO is unset
#   OPLM_ENV_FILE=/mnt/home/<you>/.env   the credentials/JOB_WORK_DIR script to source
#   OPLM_CONTAINER=/mnt/data/containers/deeplearning_v2026-05-26.sqsh
#   OPLM_RESULTS=/mnt/home/<you>/fold-m0-b200/<jobid>   where reports and logs land
#   OPLM_CUEQ_OPS=cuequivariance-ops-torch-cu13  swap for -cu12 if the container's torch is CUDA 12
# If the B200 nodes live on another partition: sbatch --partition=<name> docs/fold/b200-task7.sbatch
#
#SBATCH --job-name=fold-m0-b200
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/home/briney/logs/%x_%j.out
#SBATCH --error=/mnt/home/briney/logs/%x_%j.err

set -euo pipefail

ENV_FILE="${OPLM_ENV_FILE:-/mnt/home/${SLURM_JOB_USER}/.env}"
CONTAINER="${OPLM_CONTAINER:-/mnt/data/containers/deeplearning_v2026-05-26.sqsh}"
RESULTS="${OPLM_RESULTS:-/mnt/home/${SLURM_JOB_USER}/fold-m0-b200/${SLURM_JOB_ID}}"
GIT_REF="${OPLM_GIT_REF:-feat/fold-m0}"
REPO="${OPLM_REPO:-}"
CUEQ_OPS="${OPLM_CUEQ_OPS:-cuequivariance-ops-torch-cu13}"
# The in-container script reads these; srun --export=ALL only forwards exported variables.
export RESULTS GIT_REF REPO CUEQ_OPS

# Creates JOB_WORK_DIR (node-local /tmp) and exports object-storage / W&B credentials.
source "$ENV_FILE"
echo "=== $(date -Is) start; job=$SLURM_JOB_ID node=$(hostname) results=$RESULTS ==="
mkdir -p "$RESULTS"
mkdir -p "$JOB_WORK_DIR"

# The in-container script. Written to the (home-mounted) results dir so it is visible
# inside the container at the same path and preserved alongside the outputs it produced.
INNER="$RESULTS/run-inner.sh"
cat > "$INNER" <<'EOF'
set -uo pipefail

log() { echo "=== $(date -Is) $*"; }
run() {  # run <name> <command...>: never abort the job on one failing step; record its status.
  local name=$1; shift
  log "step $name: $*"
  "$@" > "$RESULTS/$name.log" 2>&1
  local status=$?
  echo "$name $status" >> "$RESULTS/status.txt"
  log "step $name exit $status (log: $RESULTS/$name.log)"
  return 0
}

export TRITON_CACHE_DIR="$JOB_WORK_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$JOB_WORK_DIR/inductor"
export CUEQ_TRITON_CACHE_DIR="$JOB_WORK_DIR/cueq"
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
: > "$RESULTS/status.txt"

if [ -z "$REPO" ]; then
  REPO="$JOB_WORK_DIR/oplm"
  log "cloning https://github.com/briney/oplm.git@$GIT_REF -> $REPO"
  git clone --quiet --branch "$GIT_REF" https://github.com/briney/oplm.git "$REPO"
fi
cd "$REPO"
git rev-parse HEAD > "$RESULTS/git-rev.txt"
log "repo $REPO at $(cat "$RESULTS/git-rev.txt")"

# Editable install from the checkout (the fold machinery is not on PyPI). The cuEquivariance
# pair is installed explicitly rather than via the `fold` extra so OPLM_CUEQ_OPS can select
# the ops wheel matching the container's CUDA (the extra pins cu13).
run install pip install --quiet -e ".[dev,train]" "cuequivariance-torch>=0.10.0" "$CUEQ_OPS"
run env-torch python -c "import torch, cuequivariance_torch; from importlib.metadata import version; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print('cuequivariance-torch', version('cuequivariance-torch'))"
nvidia-smi > "$RESULTS/nvidia-smi.txt" 2>&1 || true
pip freeze 2>/dev/null | grep -i -E "^(torch|triton|cuequivariance|transformers|accelerate)" > "$RESULTS/pip-freeze.txt" || true

# Task 7 Step 2: GPU parity (trimul fused paths vs fp32 reference; flex vs dense).
run gpu-tests python -m pytest tests/fold -v -m slow -rs --junitxml="$RESULTS/fold-gpu-tests.xml"

# Task 7 Step 3: the published benchmark (defaults: widths 128,256; lengths 384..2048; both
# directions; all three paths) and the compiled-reference variant.
run bench oplm fold bench-kernels --out "$RESULTS/bench-kernels-b200.json"
run bench-compiled oplm fold bench-kernels --paths reference --compile-reference --out "$RESULTS/bench-kernels-b200-compiled-reference.json"

# Task 7 Step 4: MLM regression on cuda-bf16 (FLOP contract unchanged, EMA under autocast).
run mlm-regression python -m pytest tests/training/test_e2e_logging.py tests/training/test_e2e_precision.py tests/training/test_e2e_ema.py -v --junitxml="$RESULTS/mlm-regression.xml"

log "status summary:"
cat "$RESULTS/status.txt"
EOF

srun --nodes=1 --ntasks-per-node=1 \
  --export=ALL \
  --container-image="$CONTAINER" \
  --container-mounts="/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},/mnt/data:/mnt/data,/tmp:/tmp" \
  --container-workdir="$JOB_WORK_DIR" \
  --no-container-mount-home \
  bash "$INNER"

echo "=== $(date -Is) done; copy $RESULTS/bench-kernels-b200*.json into docs/fold/ and fill docs/FOLD.md §6 ==="

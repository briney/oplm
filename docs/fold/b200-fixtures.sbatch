#!/bin/bash
# Milestone-1 golden-fixture generation + CPU parity run (docs/superpowers/plans/
# 2026-10-09-fold-m1-inference-port.md, Tasks 10 and 12). Builds a pinned `esm==3.4.1.post1`
# venv (Python 3.12, transformers 4.57) on /mnt/data, installs this repo into it, records
# the upstream ESMFold2-Fast oracle on CPU in fp32, then runs tests/fold/test_parity.py
# against it in the same venv. CPU-only: the GPU is requested only because the partition
# schedules by GPU; drop --gres if a CPU partition exists. One-shot job, no requeue.
#
# Submit from a login node:
#   sbatch docs/fold/b200-fixtures.sbatch
# Overrides (environment variables, all optional):
#   OPLM_REPO=/mnt/home/<you>/git/oplm       mounted checkout instead of cloning
#   OPLM_GIT_REF=feat/fold-m1                branch/tag/sha to clone when OPLM_REPO is unset
#   OPLM_ENV_FILE=/mnt/home/<you>/.env       credentials/JOB_WORK_DIR script to source
#   OPLM_CONTAINER=/mnt/data/containers/deeplearning_v2026-05-26.sqsh
#   OPLM_FIXTURES=/mnt/data/<you>/fold-fixtures/<jobid>   where the fixtures land (persistent)
#   OPLM_ESM_VENV=/mnt/data/<you>/fold-fixtures/esm-venv  reused across jobs
#   OPLM_RESULTS=/mnt/home/<you>/fold-m1-fixtures/<jobid> logs, manifest copy, junit XML
#   OPLM_CASES=all                           comma-separated subset of fixture cases
#   OPLM_STEPS="venv install env make-fixtures parity"
#
#SBATCH --job-name=fold-m1-fixtures
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/home/briney/logs/%x_%j.out
#SBATCH --error=/mnt/home/briney/logs/%x_%j.err

set -euo pipefail

ENV_FILE="${OPLM_ENV_FILE:-/mnt/home/${SLURM_JOB_USER}/.env}"
CONTAINER="${OPLM_CONTAINER:-/mnt/data/containers/deeplearning_v2026-05-26.sqsh}"
RESULTS="${OPLM_RESULTS:-/mnt/home/${SLURM_JOB_USER}/fold-m1-fixtures/${SLURM_JOB_ID}}"
FIXTURES="${OPLM_FIXTURES:-/mnt/data/${SLURM_JOB_USER}/fold-fixtures/${SLURM_JOB_ID}}"
ESM_VENV="${OPLM_ESM_VENV:-/mnt/data/${SLURM_JOB_USER}/fold-fixtures/esm-venv}"
GIT_REF="${OPLM_GIT_REF:-feat/fold-m1}"
REPO="${OPLM_REPO:-}"
CASES="${OPLM_CASES:-all}"
STEPS="${OPLM_STEPS:-venv install env make-fixtures parity}"
export RESULTS FIXTURES ESM_VENV GIT_REF REPO CASES STEPS

source "$ENV_FILE"
echo "=== $(date -Is) start; job=$SLURM_JOB_ID node=$(hostname) results=$RESULTS fixtures=$FIXTURES ==="
mkdir -p "$RESULTS" "$FIXTURES" "$(dirname "$ESM_VENV")" "$JOB_WORK_DIR"

INNER="$RESULTS/run-inner.sh"
cat > "$INNER" <<'EOF'
set -uo pipefail

log() { echo "=== $(date -Is) $*"; }
run() {
  local name=$1; shift
  case " $STEPS " in
    *" $name "*) ;;
    *) log "step $name skipped (OPLM_STEPS)"; echo "$name skipped" >> "$RESULTS/status.txt"; return 0 ;;
  esac
  log "step $name: $*"
  "$@" > "$RESULTS/$name.log" 2>&1
  local status=$?
  echo "$name $status" >> "$RESULTS/status.txt"
  log "step $name exit $status (log: $RESULTS/$name.log)"
  return 0
}

export CUDA_VISIBLE_DEVICES=""            # CPU oracle on purpose (fp32, no autocast)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export HF_HOME="/mnt/data/${SLURM_JOB_USER}/hf-cache"   # the 26 GB snapshot is reused across jobs
export UV_PYTHON_INSTALL_DIR="$(dirname "$ESM_VENV")/uv-python"
export WANDB_MODE=offline
: > "$RESULTS/status.txt"

if [ -z "$REPO" ]; then
  REPO="$JOB_WORK_DIR/oplm"
  log "cloning https://github.com/briney/oplm.git@$GIT_REF -> $REPO"
  git clone --quiet --branch "$GIT_REF" https://github.com/briney/oplm.git "$REPO"
fi
cd "$REPO"
git rev-parse HEAD > "$RESULTS/git-rev.txt"

# Pinned upstream environment: esm 3.4.1.post1 needs Python >=3.12 and transformers <5; this
# repo accepts transformers >=4.45, so one venv serves both (and exercises OPLM under 4.x).
run venv bash -c "pip install --quiet uv && ([ -x '$ESM_VENV/bin/python' ] || uv venv --python 3.12 '$ESM_VENV')"
run install bash -c "source '$ESM_VENV/bin/activate' && pip install --quiet 'esm==3.4.1.post1' rdkit && pip install --quiet -e '$REPO[dev]' && pip freeze | grep -i -E '^(esm|torch|transformers|rdkit|safetensors|huggingface)' > '$RESULTS/pip-freeze.txt'"
run env bash -c "source '$ESM_VENV/bin/activate' && python -c 'import esm, torch, transformers; print(esm.__version__, torch.__version__, transformers.__version__)'"
run make-fixtures bash -c "source '$ESM_VENV/bin/activate' && oplm fold make-fixtures --out '$FIXTURES' --cases '$CASES'"
cp "$FIXTURES/manifest.json" "$RESULTS/manifest.json" 2>/dev/null || true
run parity bash -c "source '$ESM_VENV/bin/activate' && OPLM_FOLD_FIXTURES='$FIXTURES' python -m pytest tests/fold/test_parity.py -v -rs --junitxml='$RESULTS/parity.xml'"

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

echo "=== $(date -Is) done; fixtures at $FIXTURES; copy $RESULTS/{manifest.json,parity.log,parity.xml,status.txt,pip-freeze.txt} into docs/fold/m1/ ==="

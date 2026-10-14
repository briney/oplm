#!/bin/bash
# Milestone-2c offline evaluation (docs/FOLD.md §10; docs/fold/m2/README.md): the EMA weights of
# every permanent checkpoint (step % 10000 == 0) of both proof runs, scored with `oplm fold eval`
# on the cameo and casp14 target sets, one checkpoint per GPU, eight at a time. One-shot job; a
# failing evaluation is recorded in status.txt and the others still run.
#
# Submit from a login node after both runs finish:
#   sbatch docs/fold/b200-m2c-eval.sbatch
# Overrides (environment variables, all optional):
#   OPLM_GIT_REF=fold-m2c-runs                        tag or sha to install
#   OPLM_RUNS=/mnt/data/<you>/oplm-runs/fold-m2c      parent of proof-w128/ and proof-w256/
#   OPLM_DATASET=/mnt/data/<you>/oplm-data/hf/wwpdb/2026-10-12
#   OPLM_VENV=/mnt/data/<you>/fold-m2c/venv-eval
#   OPLM_RESULTS=/mnt/home/<you>/fold-m2c/eval/<jobid>
#   OPLM_KEEP_EVERY=10000
#
#SBATCH --job-name=fold-m2c-eval
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/home/briney/logs/%x_%j.out
#SBATCH --error=/mnt/home/briney/logs/%x_%j.err

set -euo pipefail

ENV_FILE="${OPLM_ENV_FILE:-/mnt/home/${SLURM_JOB_USER}/.env}"
CONTAINER="${OPLM_CONTAINER:-/mnt/data/containers/deeplearning_v2026-05-26.sqsh}"
GIT_REF="${OPLM_GIT_REF:-fold-m2c-runs}"
RUNS="${OPLM_RUNS:-/mnt/data/${SLURM_JOB_USER}/oplm-runs/fold-m2c}"
DATASET="${OPLM_DATASET:-/mnt/data/${SLURM_JOB_USER}/oplm-data/hf/wwpdb/2026-10-12}"
VENV="${OPLM_VENV:-/mnt/data/${SLURM_JOB_USER}/fold-m2c/venv-eval}"
RESULTS="${OPLM_RESULTS:-/mnt/home/${SLURM_JOB_USER}/fold-m2c/eval/${SLURM_JOB_ID}}"
KEEP_EVERY="${OPLM_KEEP_EVERY:-10000}"
export GIT_REF RUNS DATASET VENV RESULTS KEEP_EVERY

source "$ENV_FILE"  # creates JOB_WORK_DIR
mkdir -p "$RESULTS" "$(dirname "$VENV")" "$JOB_WORK_DIR"
echo "=== $(date -Is) start; job=$SLURM_JOB_ID node=$(hostname) results=$RESULTS ==="

INNER="$RESULTS/run-inner.sh"
cat > "$INNER" <<'EOF'
set -uo pipefail
export HF_HOME="/mnt/data/${SLURM_JOB_USER}/hf-cache"
export OMP_NUM_THREADS=8
: > "$RESULTS/status.txt"
python -m pip install --quiet uv
python -m uv venv --allow-existing --seed --python 3.12 "$VENV"
python -m uv pip install --quiet --python "$VENV/bin/python" \
  "oplm[train,fold,esmc] @ git+https://github.com/briney/oplm@$GIT_REF"
python -m uv pip check --python "$VENV/bin/python" || { echo "install broken" >> "$RESULTS/status.txt"; exit 1; }
source "$VENV/bin/activate"
python -m uv pip freeze | grep -i -E '^(oplm|esm|torch|transformers|accelerate|cuequivariance)' > "$RESULTS/pip-freeze.txt"
nvidia-smi > "$RESULTS/nvidia-smi.txt" 2>&1 || true

jobs=()
for run in proof-w128 proof-w256; do
  for ckpt in "$RUNS/$run"/checkpoint-*; do
    step="${ckpt##*-}"
    [[ "$step" =~ ^[0-9]+$ && -d "$ckpt/hf_ema" ]] || continue
    (( step % KEEP_EVERY == 0 )) && jobs+=("$run $step $ckpt")
  done
done
echo "evaluating ${#jobs[@]} checkpoints"

gpu=0
for job in "${jobs[@]}"; do
  read -r run step ckpt <<< "$job"
  mkdir -p "$RESULTS/$run"
  (
    CUDA_VISIBLE_DEVICES=$gpu oplm fold eval --model "$ckpt" --weights ema --data "$DATASET" \
      --target-set cameo --target-set casp14 --out "$RESULTS/$run/step-$step" \
      > "$RESULTS/$run/step-$step.log" 2>&1
    echo "$run step-$step $?" >> "$RESULTS/status.txt"
  ) &
  gpu=$(( (gpu + 1) % 8 ))
  if (( gpu == 0 )); then wait; fi
done
wait
echo "=== status ==="; cat "$RESULTS/status.txt"
EOF

srun --nodes=1 --ntasks-per-node=1 --export=ALL \
  --container-image="$CONTAINER" \
  --container-mounts="/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},/mnt/data:/mnt/data,/tmp:/tmp" \
  --container-workdir="$JOB_WORK_DIR" \
  --no-container-mount-home \
  bash "$INNER"

echo "=== $(date -Is) done; copy $RESULTS into docs/fold/m2/eval/ (docs/fold/m2/README.md) ==="

#!/bin/bash
# OPLMFold inference memory against token length on one B200 (docs/FOLD.md, "Inference memory"):
# installs the pushed commit with the [fold] extra in a node-local venv and runs
# docs/fold/inference_length_sweep.py from that same checkout. Submit from the login node:
#   OPLM_REF=<commit> sbatch docs/fold/inference-length-sweep.sbatch    (sbatch exports the env)
# Optional: SWEEP_ARGS="--widths 256 --samples 1" to narrow the sweep.
#SBATCH --job-name=fold-lensweep
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/briney/logs/%x_%j.out
#SBATCH --error=/mnt/home/briney/logs/%x_%j.err

set -euo pipefail

: "${OPLM_REF:?set OPLM_REF to the pushed commit, e.g. OPLM_REF=abc1234 sbatch inference-length-sweep.sbatch}"
export OPLM_REF SWEEP_ARGS="${SWEEP_ARGS:-}"
# Per-job default, so a rerun does not overwrite an earlier run's results.
export OUT="${OUT:-/mnt/home/briney/fold-lensweep/$SLURM_JOB_ID}"

# Creates JOB_WORK_DIR (node-local) and exports credentials, as the other jobs do.
source /mnt/home/briney/.env
mkdir -p "$JOB_WORK_DIR" "$OUT"

srun --export=ALL \
  --container-image=/mnt/data/containers/deeplearning_v2026-05-26.sqsh \
  --container-mounts=/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},/mnt/data:/mnt/data,/tmp:/tmp \
  --container-workdir="$JOB_WORK_DIR" \
  --no-container-mount-home \
  bash -c 'set -euo pipefail
    unset PYTHONPATH
    export TRITON_CACHE_DIR="$JOB_WORK_DIR/triton" TORCHINDUCTOR_CACHE_DIR="$JOB_WORK_DIR/inductor"
    export CUEQ_TRITON_CACHE_DIR="$JOB_WORK_DIR/cueq" OMP_NUM_THREADS=8
    export UV_PYTHON_INSTALL_DIR=/mnt/data/${SLURM_JOB_USER}/fold-lensweep/uv-python
    git clone --quiet https://github.com/briney/oplm oplm-src
    git -C oplm-src checkout --quiet "$OPLM_REF"
    cd oplm-src
    python -m pip install --quiet uv
    python -m uv venv --quiet --python 3.12 "$JOB_WORK_DIR/venv"
    python -m uv pip install --quiet --python "$JOB_WORK_DIR/venv/bin/python" -e ".[fold]"
    { echo "commit $(git rev-parse HEAD)"; nvidia-smi -L
      python -m uv pip freeze --python "$JOB_WORK_DIR/venv/bin/python" \
        | grep -i -E "^(torch|cuequivariance)"
    } | tee "$OUT/versions.txt"
    "$JOB_WORK_DIR/venv/bin/python" docs/fold/inference_length_sweep.py \
      --out "$OUT/sweep.json" $SWEEP_ARGS 2>&1 | tee "$OUT/sweep.log"'

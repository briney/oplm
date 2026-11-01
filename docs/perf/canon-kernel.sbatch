#!/bin/bash
# Canon kernel GPU validation: GPU tests, then the kernel-vs-compiled-reference benchmark and the
# launch-config sweep (fwd and bwd timed separately per config; ~120 configs x 2 dtypes, each
# compiled once, checked against the reference, then timed at D=768 and 2048: ~15-20 min).
# Tests failing stops the job (the benchmark is meaningless on a broken kernel). Submit from the
# login node with the pushed commit of branch perf/fused-canon:
#   OPLM_REF=<commit> sbatch canon-kernel.sbatch      (sbatch exports the environment)
#SBATCH --job-name=canon-kernel
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/briney/logs/%x_%j.out
#SBATCH --error=/mnt/home/briney/logs/%x_%j.err

set -euo pipefail

: "${OPLM_REF:?set OPLM_REF to the pushed commit, e.g. OPLM_REF=abc1234 sbatch canon-kernel.sbatch}"
export OPLM_REF
# Per-job default, so a rerun does not overwrite an earlier run's results.
export OUT="${OUT:-/mnt/home/briney/projects/oplm/mup_redo/canon-kernel/$SLURM_JOB_ID}"

# Creates JOB_WORK_DIR and exports W&B / storage credentials, as the sweep jobs do.
source /mnt/home/briney/.env
mkdir -p "$JOB_WORK_DIR" "$OUT"

srun --export=ALL \
  --container-image=/mnt/data/containers/deeplearning_v2026-05-26.sqsh \
  --container-mounts=/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},/mnt/data:/mnt/data,/tmp:/tmp \
  --container-workdir="$JOB_WORK_DIR" \
  --no-container-mount-home \
  bash -c 'set -euo pipefail
    unset PYTHONPATH
    # The checkout supplies tests/ and is the installed package (editable), so tests and bench
    # exercise exactly $OPLM_REF.
    git clone https://github.com/briney/oplm oplm-src
    git -C oplm-src checkout "$OPLM_REF"
    cd oplm-src
    pip install -e ".[train,dev]"
    # Provenance: the commit, GPU and torch/triton versions behind the results in $OUT.
    { echo "commit $(git rev-parse HEAD)"
      nvidia-smi -L
      python -c "import torch, triton; print(\"torch\", torch.__version__, \"triton\", triton.__version__)"
    } | tee "$OUT/versions.txt"
    python -m pytest tests/model/test_canon_kernel.py tests/model/test_conv.py \
      tests/model/test_canon_semantics.py -q -rs 2>&1 | tee "$OUT/pytest.log"
    python docs/perf/canon_kernel_bench.py --sweep --out "$OUT/bench.json" 2>&1 | tee "$OUT/bench.log"'

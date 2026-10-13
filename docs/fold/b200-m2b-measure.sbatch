#!/bin/bash
# Milestone-2b acceptance run on one B200 (docs/fold/m2/, addendum §5): measure step time and
# peak GPU memory of the real training step at both proof widths so milestone 2c can size
# train.batch_size, gradient_accumulation_steps and max_steps. Builds a Python 3.12 venv with
# this repo's [dev,train,fold,esmc] extras in one resolve, downloads the published
# training dataset, runs the ESM-C weights test (biohub/ESMC-300M for the esmc test step and
# biohub/ESMC-600M for the measure steps; both are public, MIT-licensed and ungated, so no Hub
# token is required, though an exported HF_TOKEN raises Hub rate limits), then the fold CPU suite
# (`-m "not slow"`, on this venv's transformers < 5 / Python 3.12, which has never run it), then
# `oplm fold measure` at pair width 128 and 256 (configs/fold/proof-w{128,256}.yaml, 20 optimizer
# steps each), a per-device batch sweep at both widths (`sweep`: the same 16 samples per optimizer
# step at batch 2, 4 and 8, sizing milestone 2c), and finally the GPU-marked fold tests. One-shot
# job, no requeue; a failing step (an OOM at one sweep size, say) is recorded in status.txt and
# later steps still run.
#
# Submit from a login node:
#   sbatch docs/fold/b200-m2b-measure.sbatch
# Overrides (environment variables, all optional):
#   OPLM_REPO=/mnt/home/<you>/git/oplm       mounted checkout instead of cloning
#   OPLM_GIT_REF=main                        branch/tag/sha to clone when OPLM_REPO is unset
#   OPLM_ENV_FILE=/mnt/home/<you>/.env       JOB_WORK_DIR script to source (optional HF_TOKEN)
#   OPLM_CONTAINER=/mnt/data/containers/deeplearning_v2026-05-26.sqsh
#   OPLM_ESM_VENV=/mnt/data/<you>/fold-m2b/esm-venv   reused across jobs
#   OPLM_DATA_ROOT=/mnt/data/<you>/oplm-data/hf       where the dataset is downloaded
#   OPLM_RESULTS=/mnt/home/<you>/fold-m2b/<jobid>     logs, JSON reports, junit XML
#   OPLM_STEPS="venv install env download esmc cpu-tests measure-w128 measure-w256 sweep gpu-tests"
#                                            a name before its first '-' selects the group
#                                            (sweep = every sweep-w*-b* step)
#   OPLM_SWEEP_BATCHES="2 4 8"               per-device batches for the sweep (must divide 16)
#
#SBATCH --job-name=fold-m2b-measure
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/home/briney/logs/%x_%j.out
#SBATCH --error=/mnt/home/briney/logs/%x_%j.err

set -euo pipefail

ENV_FILE="${OPLM_ENV_FILE:-/mnt/home/${SLURM_JOB_USER}/.env}"
CONTAINER="${OPLM_CONTAINER:-/mnt/data/containers/deeplearning_v2026-05-26.sqsh}"
RESULTS="${OPLM_RESULTS:-/mnt/home/${SLURM_JOB_USER}/fold-m2b/${SLURM_JOB_ID}}"
ESM_VENV="${OPLM_ESM_VENV:-/mnt/data/${SLURM_JOB_USER}/fold-m2b/esm-venv}"
GIT_REF="${OPLM_GIT_REF:-main}"
REPO="${OPLM_REPO:-}"
STEPS="${OPLM_STEPS:-venv install env download esmc cpu-tests measure-w128 measure-w256 sweep gpu-tests}"
SWEEP_BATCHES="${OPLM_SWEEP_BATCHES:-2 4 8}"
# The in-container script reads these; srun --export=ALL only forwards exported variables.
export RESULTS ESM_VENV GIT_REF REPO STEPS SWEEP_BATCHES

# Creates JOB_WORK_DIR (node-local /tmp) and exports object-storage / W&B / HF credentials.
source "$ENV_FILE"
echo "=== $(date -Is) start; job=$SLURM_JOB_ID node=$(hostname) results=$RESULTS ==="
mkdir -p "$RESULTS" "$(dirname "$ESM_VENV")" "$JOB_WORK_DIR"

# The in-container script. Written to the (home-mounted) results dir so it is visible
# inside the container at the same path and preserved alongside the outputs it produced.
INNER="$RESULTS/run-inner.sh"
cat > "$INNER" <<'EOF'
set -uo pipefail

log() { echo "=== $(date -Is) $*"; }
run() {  # run <name> <command...>: never abort the job on one failing step; record its status.
  local name=$1; shift
  case " $STEPS " in
    *" $name "* | *" ${name%%-*} "*) ;;
    *) log "step $name skipped (OPLM_STEPS)"; echo "$name skipped" >> "$RESULTS/status.txt"; return 0 ;;
  esac
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
export UV_PYTHON_INSTALL_DIR="$(dirname "$ESM_VENV")/uv-python"
export WANDB_MODE=offline
export HF_HOME="/mnt/data/${SLURM_JOB_USER}/hf-cache"   # ESM-C weights and the dataset are reused across jobs
DATA="${OPLM_DATA_ROOT:-/mnt/data/${SLURM_JOB_USER}/oplm-data/hf}"
: > "$RESULTS/status.txt"
if [ -z "${HF_TOKEN:-}" ]; then
  log "HF_TOKEN is not set; downloading the public ESM-C weights and dataset anonymously (lower Hub rate limits)"
fi

if [ -z "$REPO" ]; then
  REPO="$JOB_WORK_DIR/oplm"
  log "cloning https://github.com/briney/oplm.git@$GIT_REF -> $REPO"
  git clone --quiet --branch "$GIT_REF" https://github.com/briney/oplm.git "$REPO"
fi
cd "$REPO"
git rev-parse HEAD > "$RESULTS/git-rev.txt"
log "repo $REPO at $(cat "$RESULTS/git-rev.txt")"
nvidia-smi > "$RESULTS/nvidia-smi.txt" 2>&1 || true

# esm 3.4.1.post1 needs Python >=3.12 and transformers <5; this repo accepts transformers
# >=4.45, so one venv serves the ESM-C loader and OPLM. It must be ONE resolve: installing esm
# first and the repo second let the second resolve (which never sees esm) upgrade
# huggingface-hub and with it transformers to 5.3 (first B200 run, docs/fold/m2/README.md), so
# esm comes in through the [esmc] extra and `uv pip check` fails the step on any broken pin. A
# reused venv is downgraded back into range by the same resolve. Every install targets the venv's
# interpreter explicitly (`uv pip --python`): a bare `pip` after `source activate` would fall
# through to the container Python. `python -m uv` avoids depending on where pip put uv.
run venv bash -c "python -m pip install --quiet uv && ([ -x '$ESM_VENV/bin/python' ] || python -m uv venv --seed --python 3.12 '$ESM_VENV')"
run install bash -c "python -m uv pip install --python '$ESM_VENV/bin/python' --quiet -e '$REPO[dev,train,fold,esmc]' && python -m uv pip check --python '$ESM_VENV/bin/python' && python -m uv pip freeze --python '$ESM_VENV/bin/python' | grep -i -E '^(esm|torch|transformers|accelerate|cuequivariance|huggingface)' > '$RESULTS/pip-freeze.txt'"
run env bash -c "source '$ESM_VENV/bin/activate' && python -c 'import esm, torch, transformers; print(esm.__version__, torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))'"
run download bash -c "source '$ESM_VENV/bin/activate' && hf download brineylab/oplmfold-training-data --repo-type dataset --revision wwpdb-2026-10-12 --include 'wwpdb/2026-10-12/*' --local-dir '$DATA'"
run esmc bash -c "source '$ESM_VENV/bin/activate' && python -m pytest tests/fold/test_esmc.py -q -rs"
run cpu-tests bash -c "source '$ESM_VENV/bin/activate' && python -m pytest tests/fold -q -m 'not slow' --junitxml='$RESULTS/fold-cpu-tests.xml'"

# Acceptance measurement: 20 real optimizer steps per width (docs/FOLD.md §9). The run
# configs point data.train at the cluster's default dataset path; override it and the output
# directory so the job is independent of the YAML's paths.
run measure-w128 bash -c "source '$ESM_VENV/bin/activate' && oplm fold measure --config configs/fold/proof-w128.yaml --steps 20 --out '$RESULTS/measure-w128.json' data.train='$DATA/wwpdb/2026-10-12' train.output_dir='$JOB_WORK_DIR/measure-w128'"
run measure-w256 bash -c "source '$ESM_VENV/bin/activate' && oplm fold measure --config configs/fold/proof-w256.yaml --steps 20 --out '$RESULTS/measure-w256.json' data.train='$DATA/wwpdb/2026-10-12' train.output_dir='$JOB_WORK_DIR/measure-w256'"
# Per-device batch sweep for milestone 2c (docs/fold/m2/README.md): tokens always pad to
# crop_size, so per-sample memory is near constant, but whether a larger batch buys throughput on
# a B200 is measured here, not extrapolated. 8 devices x batch x accumulation stays 128.
for w in 128 256; do
  for b in $SWEEP_BATCHES; do
    run sweep-w$w-b$b bash -c "source '$ESM_VENV/bin/activate' && oplm fold measure --config configs/fold/proof-w$w.yaml --steps 20 --out '$RESULTS/sweep-w$w-b$b.json' data.train='$DATA/wwpdb/2026-10-12' train.output_dir='$JOB_WORK_DIR/sweep-w$w-b$b' train.batch_size=$b train.gradient_accumulation_steps=$((16 / b))"
  done
done
run gpu-tests bash -c "source '$ESM_VENV/bin/activate' && python -m pytest tests/fold -m slow -q -rs --junitxml='$RESULTS/fold-gpu-tests.xml'"

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

echo "=== $(date -Is) done; copy $RESULTS/{measure-w128.json,measure-w256.json,sweep-w*.json,status.txt,pip-freeze.txt,git-rev.txt,esmc.log,fold-cpu-tests.xml,fold-gpu-tests.xml} into docs/fold/m2/ ==="

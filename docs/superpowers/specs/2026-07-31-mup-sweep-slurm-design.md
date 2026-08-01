# μP Sweep on SUNK/Slurm Design

Supersedes the launcher and grid portions of
[2026-07-14-mup-sweep-scripts-design.md](2026-07-14-mup-sweep-scripts-design.md). The phase
funnel, selection protocol, and μP parameterization from that spec are unchanged except where
stated here.

## Goal

Run the phased μP learning-rate sweep on the CoreWeave SUNK (Slurm-on-Kubernetes) cluster
instead of one local 8×B200 node, and re-center the learning-rate grid on the region the 170M
coarse sweep actually identified.

Two independent changes, delivered together because both touch the same generator:

1. **Execution.** Replace the sequential local runner with per-phase Slurm job arrays plus a
   dependent analyze job. Move the sweep tooling into the installable package so a job script
   needs nothing but `pip install oplm[train]`.
2. **Search range.** Shift the coarse LR grid down one half-decade and derive the phase gates
   from the grid actually used rather than from hardcoded constants.

## Motivating evidence

The 170M coarse sweep (10,000 steps, seven-point grid `0.0025 … 0.04`) ranked
`0.0025 ≈ 0.004 ≫ 0.0063 > 0.01 > 0.016`. The winner sits at the **lower boundary** of the grid.

This is both a physics problem and a tooling problem. `analyze_phase` only selects a refinement
region when the coarse winner is interior to its finite grid, so that phase would have ranked
and then selected nothing, stalling the funnel with no diagnostic pointing at the cause.

## Wall-clock reality

Scaling the measured 170M datapoint (12 h per 10,000 steps at global batch 2048, i.e. ~1.2 h per
1,000 steps) by parameter count gives single-node estimates:

| Phase | Cell | Steps | Global batch | 1 node |
| --- | --- | ---: | ---: | ---: |
| `transfer` | 400M | 10k | 2048 | ~28 h |
| `transfer` | 800M | 20k | 2048 | ~4.7 d |
| `transfer` | 1B | 10k | 2048 | ~4.5 d |
| `bridge` | 170M | 10k | 8192 | ~2.0 d |
| `confirm` | 800M | 10k | 8192 | ~9.4 d |
| `scale` | 170M | 100k | 8192 | ~20 d |
| `scale` | 1B | 100k | 8192 | ~180 d |

Two consequences drive the design. `scale` is not sweep-tool work at any node count and becomes
generate-only. `confirm` and the large `transfer` cells require multi-node execution; a ~9-day
gate is not an acceptable serialization point in the funnel.

Two caveats on the table. The `1B` preset is ~1.5B parameters (1600 hidden × 50 layers), which is
why it costs more than `800M` despite fewer steps. And the 12 h/10k figure implies roughly 1–2%
MFU on 8×B200 — low enough that it is either not measuring steady-state training (eval overhead,
a larger batch or sequence length than assumed) or indicates a throughput problem. Every estimate
above scales linearly off it. Investigating that is **out of scope** here, but the node counts
chosen below should be revisited if the baseline is wrong.

## Design principles

- **One artifact contract for local and Slurm.** Phase directories keep their existing layout and
  meaning. Slurm adds a `jobs/` subtree; it does not change `phase.json`, ranking, or selection.
- **Generation and submission are separate steps.** Every phase generates reviewable scripts.
  `--submit` is an opt-in convenience, never a requirement.
- **One launcher form.** The multi-node `srun` form degrades correctly to a single node, so there
  is no separate single-node code path to maintain or test.
- **Derived values are computed and validated, never silently adjusted.** Per-device batch and
  gradient accumulation are derived from node count and global batch; indivisibility is an error.
- **Gates track the grid actually used.** No phase logic references a hardcoded learning rate.

## Out of scope

- Multi-node support for the `scale` phase's submission or monitoring. `scale` emits scripts and
  stops.
- Throughput/MFU investigation (see caveat above).
- Any change to the μP parameterization, ranking rules, phase ordering, or selection protocol.
- Retry policy beyond Slurm `--requeue` plus checkpoint resume.
- Cross-cluster portability. The `slurm:` block is validated but its defaults target CoreWeave.

## Change 1 — Package the sweep tooling

`scripts/` is not shipped: `pyproject.toml` has no `[tool.hatch.build]` section, so hatchling
packages only `src/oplm`. Every generated sweep command invokes `-m scripts.mup_run`, which does
not exist after `pip install oplm[train]` inside the container.

Move the tooling into the wheel:

```
src/oplm/sweep/
├── __init__.py
├── common.py       # was scripts/_mup_common.py
├── phases.py       # was scripts/mup_sweep.py  (generators + analyze)
├── coord_check.py  # was scripts/mup_coord_check.py
├── run.py          # was scripts/mup_run.py    (+ auto-resume)
├── slurm.py        # NEW: config model, sbatch rendering, submission, status
└── cli.py          # NEW: `oplm sweep ...` typer sub-app
```

`cli.py` is wired into `src/oplm/cli.py` with `add_typer`, exposing `oplm sweep smoke|coarse|
refine|replicate|transfer|bridge|confirm|scale|analyze|status|coord-check`. Cells are launched as
`-m oplm.sweep.run`.

`scripts/` is deleted. `tests/scripts/test_mup_sweep.py` moves to `tests/sweep/test_phases.py`,
mirroring the new source layout.

The `--local` sequential path is retained: it is what the integration test exercises and what a
single 8×B200 workstation runs.

## Change 2 — Slurm configuration

Cluster settings live in a `slurm:` block in the existing sweep config
(`configs/mup-production.yaml`), so one file describes the whole sweep.

This requires no loader changes. `load_config` calls `OmegaConf.set_struct(base, False)`, so the
unknown top-level key merges and is reachable as `cfg.slurm`; `serialize_config` omits it, so each
cell's generated `run.yaml` stays clean. Both behaviors were verified against the current loader
before adopting this layout.

```yaml
slurm:
  partition: hpc-mid
  time_limit:
    default: "24:00:00"
    confirm: "48:00:00"
  cpus_per_task: 128
  gpus_per_node: 8
  exclusive: true
  mem: "0"
  log_dir: /mnt/home/briney/logs
  env_file: /mnt/home/briney/.env
  container_image: /mnt/data/containers/deeplearning_v2026-05-26.sqsh
  container_mounts:
    - /mnt/home/${USER}:/mnt/home/${USER}
    - /mnt/data:/mnt/data
    - /tmp:/tmp
  install: "pip install oplm[train]"
  max_concurrent: 4
  nodes:
    default: {170M: 1, 400M: 4, 800M: 8, 1B: 8}
    bridge:  {170M: 4}
    confirm: {800M: 8}
    scale:   {170M: 4, 400M: 8, 800M: 16, 1B: 16}
  max_batch_size: {170M: 256, 400M: 256, 800M: 256, 1B: 128}
```

The block is parsed into a validated dataclass in `slurm.py`, not consumed as a raw dict.

### Node counts are explicit; batch is derived

`nodes` is configured per preset with per-phase overrides because the counts are driven by wall
time, not by memory. Per-device batch and gradient accumulation are **derived**: choose the
smallest `accum >= 1` such that

```
per_device = global_examples / (nodes * gpus_per_node * accum)
```

is a positive integer and `per_device <= max_batch_size[preset]`. If no such `accum` exists, the
generator raises rather than adjusting the global batch. The derived table is printed at
generation time and recorded in `phase.json`.

Under the defaults above every cell resolves to `accum = 1`:

| Global batch | Preset | Nodes | Per-device | Accum | Est. wall time |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2048 | 170M | 1 | 256 | 1 | ~12 h |
| 2048 | 400M | 4 | 64 | 1 | ~7 h |
| 2048 | 800M | 8 | 32 | 1 | ~14 h |
| 2048 | 1B | 8 | 32 | 1 | ~13 h |
| 8192 | 170M | 4 | 256 | 1 | ~12 h (`bridge`) |
| 8192 | 800M | 8 | 128 | 1 | ~28 h (`confirm`) |

Note that `confirm` at ~28 h exceeds the 24 h default `time_limit`. `time_limit` therefore takes
the same `default` + per-phase override shape as `nodes`, and `confirm` is raised to 48 h rather
than being left to depend on requeue for its primary path.

This replaces the current behavior, where `_write_run_config` reads `base.train.batch_size` and
computes accumulation from it. The generator now pins `train.batch_size` per cell.

## Change 3 — Per-phase artifacts and execution

```text
sweeps/<phase>/
├── phase.json          # + oplm_version, generated_at, derived batch table, submitted job ids
├── commands.txt        # unchanged: one accelerate command per cell
├── jobs/
│   ├── <preset>.sbatch # array job over that preset's cells
│   ├── <preset>.cells  # one run id per line; SLURM_ARRAY_TASK_ID indexes it
│   ├── analyze.sbatch  # CPU-only ranking job (no --gres)
│   └── submit.sh       # sbatch invocations and dependency wiring
└── runs/<id>/
    ├── run.yaml
    ├── result.json
    └── output/
```

### One array per preset

Slurm job arrays require homogeneous resources, and node count varies by preset. Phases with a
single preset (`smoke`, `coarse`, `refine`, `replicate`, `bridge` at 170M; `confirm` at 800M) emit
one array. `transfer` spans 400M/800M/1B and emits three arrays, with one analyze job depending on
all of them.

Array indices map to cells through the sibling `<preset>.cells` file, so the sbatch body contains
no generated per-cell branching.

### Dependency wiring

```bash
A400=$(sbatch --parsable jobs/400M.sbatch)
A800=$(sbatch --parsable jobs/800M.sbatch)
A1B=$(sbatch --parsable jobs/1B.sbatch)
sbatch --parsable --dependency=afterany:$A400:$A800:$A1B jobs/analyze.sbatch
```

`afterany`, not `afterok`. A divergent learning rate is expected data, not a failure: ranking
already treats missing and non-finite results as ineligible. Under `afterok` the first blown-up
cell would leave the analyze job permanently in `DependencyNeverSatisfied`, and the phase would
appear hung rather than ranked.

The analyze job runs `oplm sweep analyze <phase.json>` inside the container on one node with no
`--gres` and a short time limit. It is a single job, not an array, so `max_concurrent` does not
apply to it.

### Generation, submission, and status

Every phase generates. `--submit` executes the submission sequence directly via
`subprocess.run` with argument lists (never `shell=True`), parsing `sbatch --parsable` output and
recording the job ids into `phase.json`.

Intended usage: `--submit` for the cheap 170M phases (`smoke`, `coarse`, `refine`, `replicate`);
review then run `submit.sh` by hand for `transfer`, `bridge`, `confirm`, and `scale`. This is a
convention, not an enforced restriction — `--submit` is available on every phase except `scale`.

`oplm sweep status <phase.json>` reports per-cell state (pending / running / complete / missing /
non-finite) and prints the `sbatch --array=<indices>` line needed to re-run missing cells.

### Launcher form

Scripts render the multi-node `srun` form from the validated CoreWeave template:

- no container at the sbatch level; Pyxis flags (`--container-image`, `--container-mounts`,
  `--container-workdir`, `--no-container-mount-home`) on `srun`;
- `source ${env_file}` in the sbatch body, then an `srun --ntasks-per-node=1 mkdir -p
  "$JOB_WORK_DIR"` fanout, because `JOB_WORK_DIR` is node-local `/tmp` and `.env` creates it only
  on the batch node;
- `MASTER_ADDR=$(hostname --ip-address)`, `MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))`
  (`SLURM_JOB_ID` is unique per array task, so concurrent cells cannot collide);
- `accelerate launch --num_machines $SLURM_NNODES --num_processes $((SLURM_NNODES * 8))
  --machine_rank $SLURM_PROCID` inside a single-quoted `bash -c`, so the Slurm variables expand in
  the container's shell rather than at render time;
- `#SBATCH --requeue`.

At `SLURM_NNODES=1` this form is correct without modification, so single-node cells use it too.

## Change 4 — Resilience

`--requeue` alone restarts a cell from step 0. Three coordinated changes make cells idempotent:

1. `oplm.sweep.run` discovers the newest `checkpoint-<step>` directory under the cell's
   `train.output_dir` and sets `train.resume_from` when it is unset. Checkpoint naming is
   `checkpoint-{global_step}` (`trainer.py:528`), so discovery is a glob plus an integer max.
2. The generator pins `train.save_every ≈ max_steps // 8`. The current default of `10_000` means a
   10,000-step cell checkpoints only at completion, making resume worthless.
3. The generator pins `train.save_total_limit = 1`. Rotation already exists
   (`checkpoint.py:122`); without pinning it, eight checkpoints per cell across ~24 `transfer`
   cells is substantial disk for artifacts that are never used after the run completes.

Requeue, preemption, wall-clock expiry, and manual re-submission of a failed array index then all
resolve to the same resume path.

### Version recording

Generated scripts keep `pip install oplm[train]` unpinned, per operator preference. To make
version drift across a multi-week sweep detectable rather than invisible, each cell records its
resolved `oplm.__version__` into `result.json`, and the generator records it into `phase.json`.
This imposes no constraint on submission order or release cadence.

## Change 5 — Learning-rate grid and gates

### New grid

```python
SMOKE_LRS  = (0.0004, 0.0016, 0.0063)
COARSE_LRS = (0.0004, 0.00063, 0.001, 0.0016, 0.0025, 0.004, 0.0063)
```

Seven points at the existing 1.6× spacing, shifted down one half-decade. Both observed winners
(0.0025, 0.004) are interior with four points of headroom below, so a lower true optimum is
findable, and 0.0063 remains as an upper guard so a boundary win is still detectable. The dropped
values (0.01, 0.016, 0.025, 0.04) were all measured worse. `SMOKE_LRS` is the new grid's minimum,
midpoint, and maximum, preserving its existing relationship to `COARSE_LRS`.

### Gates derived from the manifest

Both smoke gates currently reference the old grid by value and would be silently wrong after the
shift:

- `_smoke_gated_lrs` (`mup_sweep.py:421`) indexes the `SMOKE_LRS` module constant;
- the smoke branch of `analyze_phase` (`mup_sweep.py:575`) tests `scores.get(0.0025)` and
  `scores.get(0.01)` literally.

Both are rewritten to derive from the phase manifest's own learning-rate list: the two lowest
learning rates in the phase must have finite validation loss, and the highest is dropped from the
downstream coarse grid if it did not. The gates then track whatever `--lrs` was actually passed,
and changing the grid never again requires editing phase logic.

### `smoke` is retained

With the new grid its divergence gate has little to catch — 0.0063 is known to train. It is kept
and reframed as a 1,000-step end-to-end check of the new container, mounts, dataset paths, and the
full Slurm path, run before committing days of GPU time.

### Eval task scope

Sweep cells evaluate one held-out sequence task, which is the selection metric. The full six-task
suite (three sequence evals, CASP14, two ProteinGym tasks) appears only in `scale`.

This resolves a live contradiction: the production job scripts configure six eval tasks, while
`LR_SWEEP.md` instructs the operator to configure exactly one and `_resolve_metric`
(`mup_sweep.py:161`) errors unless there is exactly one or `--metric` is passed. Structure and DMS
eval cost also does not inform ranking, so paying it on ~50 cells is waste.

## Change 6 — `scale` becomes generate-only

`scale` no longer runs or submits anything. It reads the confirmed winner from
`sweeps/confirm/phase.json` and writes one multi-node `.sbatch` per preset, which the operator
reviews and submits on their own schedule.

Production-specific settings come from a `scale:` sub-block in the same sweep config — the full
eval suite, WSD schedule (`stable_steps`, `max_steps`), `wandb_project`, and output-directory
root. Keeping it in the one file preserves the property that the sweep is described by a single
artifact, and the alternative (pointing at a separate production config) adds a file whose drift
from the sweep config would be silent.

The value retained by keeping `scale` in the tool at all is provenance: the winning `lr`,
`output_mult`, `depth_exponent`, and `batch_mult` flow into five job scripts without
hand-transcription, at the point in the workflow where a transcription error is most expensive.

## Testing strategy

No cluster access is required for any test.

**Rendering.** Golden-file tests over generated `.sbatch` for representative phases: node count,
array range and `%K` throttle, `--requeue`, Pyxis flags on `srun` and not on `#SBATCH`, the
`mkdir -p` fanout, and — critically — which variables expand at render time versus inside the
container's single-quoted `bash -c`.

**Syntax.** `bash -n` over every generated script, including `submit.sh`.

**Derivation.** Unit tests over the node/batch/accum solver: each row of the table above, the
`max_batch_size` constraint forcing `accum > 1`, and the indivisible case raising.

**Submission.** `--submit` tested against a fake `sbatch` shim placed on `PATH` in a tmp dir,
asserting argv, `--parsable` usage, dependency string construction across multiple arrays, and
that job ids reach `phase.json`.

**Gates.** The smoke and coarse gate tests are rewritten against the new grid, plus a test that
passing a non-default `--lrs` moves the gates with it — the regression that hardcoded constants
would reintroduce.

**Resume.** Unit test for latest-checkpoint discovery: no checkpoints, one checkpoint, several
(correct integer max, not lexicographic), and an explicit `resume_from` taking precedence.

**Integration.** The existing `--local` end-to-end pilot run is retained: a small model trains for
a few steps and completes one eval through the real phase machinery.

**Migration.** Existing phase-generation, ranking, and selection tests move to `tests/sweep/` and
must pass unchanged apart from import paths and the new grid values.

## Documentation changes

- `docs/LR_SWEEP.md`: replace the local eight-GPU sequence and the short "SUNK/Slurm use" section
  with the Slurm workflow; document the `slurm:` block, the node/batch table, the per-phase
  submit-versus-review convention, `status`, and the new grid. Correct the "exactly one eval task"
  instruction to the lean-proxy/full-scale split.
- `docs/MUP.md`: update the pointer at line 221 to the new command surface.
- `AGENTS.md`: note that sweep tooling lives in `src/oplm/sweep/`, not `scripts/`.

## Acceptance criteria

1. `pip install oplm[train]` provides `oplm sweep` and `python -m oplm.sweep.run`; no generated
   command references `scripts.`.
2. Every phase except `scale` generates `jobs/` with per-preset arrays, an analyze job wired with
   `--dependency=afterany`, and a `submit.sh`; `transfer` emits three arrays and one analyze job.
3. Derived per-device batch and accumulation match the table above and are recorded in
   `phase.json`; an indivisible configuration raises at generation time.
4. A requeued cell resumes from its newest checkpoint rather than from step 0.
5. `COARSE_LRS` and `SMOKE_LRS` are the new values, and no phase logic references a literal
   learning rate.
6. `oplm sweep scale` writes per-preset scripts carrying the confirmed winner and submits nothing.
7. CI gates pass on `src/`: `ruff check`, `ruff format --check`, `ty check`, `pytest`.

# Production μP Learning-Rate Sweep

This is the operator runbook for selecting the OPLM production learning rate. It uses the
implemented phased sweep to tune an affordable proxy, verify width/depth transfer, bridge to
the production batch, and run the winning scaling series.

## Production scaling ray

The presets follow a 32:1 hidden-size-to-depth ratio with a fixed 64-dimensional attention
head:

| Role | Preset | Hidden | Layers | Heads | Head dim |
| --- | --- | ---: | ---: | ---: | ---: |
| Diagnostic | `50M` | 512 | 16 | 8 | 64 |
| μP/production anchor | `170M` | 768 | 24 | 12 | 64 |
| Production | `400M` | 1024 | 32 | 16 | 64 |
| Production | `800M` | 1280 | 40 | 20 | 64 |
| Scaling confirmation | `1B` | 1600 | 50 | 25 | 64 |

The `800M` preset is corrected to 20 heads, so its head dimension is 64. The sweep forces
`mup_base_width=768`, making the `170M` preset its width-μP anchor. This is intentionally
different from the general model-config fallback/default of `mup_base_width=512`; see
[MUP.md](MUP.md).

## One production config

Use one `configs/mup-production.yaml` for every command below. Create that path (including its
`configs/` directory) at the repository root with this base config, replacing only the two
dataset paths and the cluster paths in the `slurm:` block:

```yaml
model:
  norm_strategy: sandwich
  canon_enabled: true
  canon_positions: [A, B, C, D]
  residual_gate: channel

train:
  batch_size: 32
  optimizer: muon
  muon_adjust_lr_fn: original
  weight_decay: 0.01

data:
  train: /path/to/train.parquet
  eval:
    heldout:
      path: /path/to/eval.parquet
      type: sequence
      every: {steps: 500}

slurm:
  partition: hpc-mid
  time_limit:
    default: "168:00:00"
    analyze: "01:00:00"
  cpus_per_task: 128
  gpus_per_node: 8
  exclusive: true
  mem: "0"
  log_dir: /mnt/home/briney/logs
  env_file: /mnt/home/briney/.env
  container_image: /mnt/data/containers/deeplearning_v2026-05-26.sqsh
  container_mounts:
    - /mnt/data:/mnt/data
    - /tmp:/tmp
  install: pip install oplm[train]
  max_concurrent: 4
  nodes:
    default: {170M: 1, 400M: 4, 800M: 8, 1B: 8}
    transfer: {170M: 4}
    bridge: {170M: 4}
    replicate: {170M: 4}
    confirm: {800M: 8}
  max_batch_size: {170M: 256, 400M: 256, 800M: 256, 1B: 128}
```

The file merges over the packaged defaults. Add any production-specific masking, precision,
compile, or checkpoint overrides to this same file before launch. Keep exactly one named
sequence eval task **for every proxy phase** (`smoke` through `confirm`) so the harness can infer
its `eval/<name>/loss` metric — `scale` is the one exception; it trains production checkpoints, so
it uses the full multi-task eval suite from `configs/scaling.yaml` (six tasks: three sequence
losses plus `casp14` structure eval and two ProteinGym tasks), not this single-task metric.

Keep weight decay fixed at `0.01` for the complete workflow. The generator also resolves each
cell with μP enabled, base width 768, reference depth 24, the requested depth exponent, and a
`wsd_linear` schedule whose stable phase runs from the end of warmup to the end of the cell.

The `slurm:` block above follows the general schema documented in [SLURM.md](SLURM.md) — see
that page for every field, its default, and the accepted `nodes`/`time_limit` forms. Every phase
command below reads `nodes`, `time_limit`, and `max_batch_size` from it and derives each cell's
per-device batch and gradient-accumulation steps, so **the `slurm:` block is required even when
running a phase with `--local`** — it is not only consulted for real Slurm submission.

## Parameterization gates

Run both the μP gate and its non-μP control at fixed depth, then check the production preset
ray:

```bash
# Parameterization gates (run μP-on and --no-mup control).
oplm sweep coord-check --config configs/mup-production.yaml \
  --scaling width --widths 384,768,1536 --depth 24 --base-width 768 \
  --out sweeps/coord-width
oplm sweep coord-check --config configs/mup-production.yaml \
  --no-mup --scaling width --widths 384,768,1536 --depth 24 \
  --base-width 768 --out sweeps/coord-width-control
oplm sweep coord-check --config configs/mup-production.yaml \
  --scaling preset_ray --widths 512,768,1024,1280 --base-width 768 \
  --out sweeps/coord-ray
```

With μP enabled, hooked module output RMS should not grow systematically with width. Small
non-systematic variation is acceptable, and readout logits may shrink at initialization by
design. The non-μP control should visibly fan out. The preset-ray check is an empirical test of
combined width/depth behavior, not a replacement for the fixed-depth width gate.

## Where these commands run

The sweep splits cleanly across two machines, and the split is worth setting up deliberately
before you start.

**Generation runs inside the training container.** `oplm sweep <phase>` resolves and validates a
complete training config for every cell — it calls `load_config`, which instantiates the real
Hugging Face model config to reject mistyped `model.*` keys and derive `head_dim` /
`intermediate_size` / rope dims before writing each `run.yaml`. That pulls in `transformers` and
`torch`. Separately, `oplm/__init__.py` imports `transformers` eagerly to register OPLM with the
`AutoModel` classes, so *any* `import oplm` needs it. Neither is avoidable today, and neither is
something you want on a login node.

Since the container already carries the full stack, generate there. The portable way is a
one-CPU `srun` using the same Pyxis flags the training jobs use:

```bash
CONFIG=/mnt/home/$USER/projects/oplm/configs/mup-production.yaml
SWEEP=/mnt/home/$USER/projects/oplm/sweeps

srun --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:30:00 \
  --container-image=/mnt/data/containers/deeplearning_v2026-05-26.sqsh \
  --container-mounts=/mnt/home/$USER:/mnt/home/$USER,/mnt/data:/mnt/data \
  --no-container-mount-home \
  bash -c "pip install 'oplm[train]==0.1.6' && \
    oplm sweep coarse --config $CONFIG \
      --from $SWEEP/smoke/phase.json --out $SWEEP/coarse --submit"
```

Generation is a few seconds of text rendering, so the queue wait usually dominates. If your login
node can run the container image directly (`enroot`, `apptainer`, or `podman` — the exact
invocation is site-specific), doing that instead removes the wait entirely and is worth setting up
if you iterate on configs. The command inside the container is identical either way.

**Use absolute paths for `--config` and `--out`. This is not stylistic.** The generator calls
`Path.resolve()` on both, which resolves against the current working directory — and inside the
container that is `$JOB_WORK_DIR`, which lives on **node-local `/tmp`**. A relative `--out` writes
the entire phase (resolved configs, `jobs/`, the manifest) to scratch that is discarded when the
job exits. The command prints `wrote ...` and exits 0; the artifacts are simply gone. Either pass
absolute paths, or add `--container-workdir=/mnt/home/$USER/...` to move the working directory
onto shared storage.

**Pin the oplm version for the duration of the sweep.** Each generation run reinstalls from PyPI,
so an unpinned `oplm[train]` lets phase 1 and phase 6 be generated by different code. Cells record
their resolved version in `result.json`, which makes drift *detectable*; pinning makes it
impossible. If you are the maintainer, the simplest discipline is to defer releases until the
sweep finishes.

**The login node needs nothing installed.** Submission is `sbatch`, which you already have.
`bash sweeps/<phase>/jobs/submit.sh` is a plain shell script, and every path inside the generated
`.sbatch` files is absolute, so they run correctly from anywhere.

### Weights & Biases

Set the project once in your production config; every generated cell inherits it, so the whole
sweep lands in one place:

```yaml
train:
  wandb_project: oplm-mup-sweep
```

There is no `--wandb-project` flag on the phase commands — the config is the single source, which
is what keeps every phase in the same project without you remembering a flag each time.

The generator sets each cell's run name to `<phase>-<preset>-lr<lr>-om<output_mult>-a<depth_exponent>-s<seed>`,
e.g. `coarse-170M-lr0.0025-om1-a0-s42`, so a run's full identity is legible from the W&B sidebar.
The phase prefix matters: `SMOKE_LRS` is a subset of `COARSE_LRS`, so without it the three smoke
cells and their coarse counterparts would appear under identical names, separable only by step
count. `transfer`'s depth-ray cells additionally insert the layer count and bracket multiplier
(`transfer-170M-d48-lr0.004-om1-a0.5-x1.6-s42`) — the multiplier is not redundant with the
effective LR, since the 1.6×-spaced coarse grid lets one finalist's `x1.6` cell land on another
finalist's `x1` LR. `scale` runs are named `oplm-<preset>-scale`.

### Monitoring without installing oplm

The dependent `analyze` job ranks each phase automatically inside the container (see below), so
the ranking step needs nothing from you. For day-to-day monitoring, W&B covers training health and
`squeue` covers queue state.

`oplm sweep status` adds two things neither of those gives you:

1. **Whether each cell's `result.json` actually landed.** Selection ranks on that file being
   present and finite. A cell that trained fine and died before writing it looks healthy in W&B
   and is invisible to ranking.
2. **The `sbatch --array=...` resubmit line**, with per-preset zero-based indices matching each
   `.jobs` file.

Both are recoverable by hand — `ls sweeps/<phase>/runs/*/result.json` answers the first, and the
`.jobs` file answers the second — so treat `status` as a convenience, not a requirement.

If you do want it locally, note that `status` and `analyze` are inherently light: they read JSON
manifests and shell out to `squeue`, and never call `load_config`. Only the *generation* commands
need the heavy path. What blocks a light install today is the eager `transformers` import in
`oplm/__init__.py`, not those commands themselves. Until that is deferred, a login-node install is:

```bash
pip install --no-deps oplm && pip install omegaconf typer rich
```

Note that `pip install oplm[<extra>]` cannot be lighter than `pip install oplm` — Python extras
only ever *add* dependencies, so no extra can subtract `torch` from the core requirements.

## Running phases on Slurm

The commands below are written as you would type them inside the container, with `$CONFIG` and
`$SWEEP` set to absolute paths:

```bash
CONFIG=/mnt/home/$USER/projects/oplm/configs/mup-production.yaml
SWEEP=/mnt/home/$USER/projects/oplm/sweeps
```

Every phase from `smoke` through `confirm` **generates** artifacts (fully resolved `run.yaml`
configs, a `phase.json` manifest, and one Slurm array job per preset plus a dependent `analyze`
job — see [SLURM.md §6](SLURM.md#6-job-arrays-and-the-homogeneous-resource-constraint)). Do not
generate the next phase until the current one has completed and been ranked. `scale` is different
— see the note at the end of this section.

For the cheap, fast proxy phases (`smoke`, `coarse`, `refine`, the first `replicate`, and
`transfer` — the depth ray runs 170M-class cells on the same 4-node allocation as the first
`replicate`), pass `--submit` so generation and submission happen in one step. For the expensive, multi-day phases (`bridge`, the
second `replicate`, `confirm`), omit `--submit`, inspect the generated `jobs/*.sbatch` scripts
and `jobs/submit.sh`, then run `bash sweeps/<phase>/jobs/submit.sh` yourself once you're ready
to commit the allocation:

The split is by *cost*, not by node count. Both `replicate` invocations share the phase name
`replicate`, so the `nodes` override above applies to both: the first one is cheap (a few hours)
but still lands on four nodes, and `transfer`'s depth-ray cells do too. Check the
[node table](#node-counts-and-derived-per-device-batch)
before auto-submitting if four nodes is a hard allocation to get on your cluster.

```bash
# Cheap phases: generate and submit in one step. Run each `oplm sweep` line in the
# container (see "Where these commands run"); the `bash .../submit.sh` lines are plain
# sbatch calls and run on the login node.
oplm sweep smoke --config $CONFIG --out $SWEEP/smoke --submit
oplm sweep coarse --config $CONFIG \
  --from $SWEEP/smoke/phase.json --out $SWEEP/coarse --submit
oplm sweep refine --config $CONFIG \
  --from $SWEEP/coarse/phase.json --out $SWEEP/refine --submit
oplm sweep replicate --config $CONFIG \
  --from $SWEEP/refine/phase.json --out $SWEEP/refine-replicate --submit
oplm sweep transfer --config $CONFIG \
  --from $SWEEP/refine-replicate/phase.json --out $SWEEP/transfer --submit

# Expensive phases: generate, review, then submit manually.
oplm sweep bridge --config $CONFIG \
  --from $SWEEP/transfer/phase.json --out $SWEEP/bridge
bash $SWEEP/bridge/jobs/submit.sh
oplm sweep replicate --config $CONFIG \
  --from $SWEEP/bridge/phase.json --out $SWEEP/bridge-replicate
bash $SWEEP/bridge-replicate/jobs/submit.sh
oplm sweep confirm --config $CONFIG \
  --from $SWEEP/bridge-replicate/phase.json --out $SWEEP/confirm
bash $SWEEP/confirm/jobs/submit.sh

# scale never submits -- see the phase table below.
oplm sweep scale --config $CONFIG \
  --from $SWEEP/confirm/phase.json --out $SWEEP/scale
```

Each `generate` call's own `--submit` and the `jobs/submit.sh` it always writes are equivalent —
`--submit` just runs that submission immediately instead of leaving it for you to review first.
The generator's dependent `analyze` job runs `oplm sweep analyze <phase>/phase.json` automatically
once every array for that phase has finished (or diverged) via `afterany`, so a completed,
submitted phase ranks itself; you do not need to call `analyze` by hand unless you're re-ranking
after a manual resubmission.

`scale` writes plain per-preset `run.yaml`/`.sbatch` pairs under `sweeps/scale/jobs/`, with no
`phase.json`, `jobs.json`, or `submit.sh` — there is no ranking left to do and nothing for
`oplm slurm status` to track. Submit each one directly once you've reviewed it:

```bash
sbatch $SWEEP/scale/jobs/oplm-170M-scale.sbatch
sbatch $SWEEP/scale/jobs/oplm-400M-scale.sbatch
# ...one per --presets entry
```

`--local` remains available (see `oplm sweep smoke --help`) for a single-machine sanity run: it
executes every cell sequentially on one eight-GPU node with no Slurm involvement at all, then
ranks the phase before the command returns. It is useful for confirming a new production config's
cells generate and train correctly before committing a multi-day Slurm allocation, but it is not
the production path — `--local` and `--submit` are mutually exclusive.

### Node counts and derived per-device batch

Under the `slurm:` block above, every cell resolves to `gradient_accumulation_steps=1`. Wall-time
estimates are derived from a single 170M measurement (~12 h per 10,000 steps at global batch
2048) scaled by parameter count and node count; cells also now pin full gradient checkpointing
(`model.gradient_checkpointing_mode=full`), which was not necessarily in effect when that
measurement was taken. Treat the wall-time column as order-of-magnitude guidance, not a promise.

The `nodes` table's `replicate` entry governs **both** `replicate` invocations: the generator
resolves node count from the phase name `replicate` regardless of which source phase fed it, so
one `{170M: 4}` override reshapes the first `replicate` (source `refine`, still the 2,048 proxy
batch) exactly as much as the second (source `bridge`, the 8,192 production batch). At the
production batch this is exactly what's wanted — it makes the second `replicate` match `bridge`'s
shape. At the proxy batch it also pulls the first `replicate` off the single-node group below it
onto 4 nodes at a quarter the per-device batch, which is why it now gets its own row:

| Phase | Preset | Global batch | Nodes | Per-device batch | Est. wall time |
| --- | --- | ---: | ---: | ---: | ---: |
| `smoke`/`coarse`/`refine` | 170M | 2048 | 1 | 256 | ~12 h (`coarse`, 10k steps) |
| first `replicate` | 170M | 2048 | 4 | 64 | ~6 h (20k steps, inherited from `refine`) |
| `transfer` (depth 12) | 170M | 2048 | 4 | 64 | ~1.5 h (10k steps) |
| `transfer` (depth 24) | 170M | 2048 | 4 | 64 | ~3 h (10k steps) |
| `transfer` (depth 48) | 170M | 2048 | 4 | 64 | ~6 h (10k steps) |
| `bridge` | 170M | 8192 | 4 | 256 | ~12 h (10k steps) |
| second `replicate` | 170M | 8192 | 4 | 256 | ~12 h (10k steps, matches `bridge`) |
| `confirm` | 800M | 8192 | 8 | 128 | ~28 h (10k steps) |

Every row above still resolves to `gradient_accumulation_steps=1`.

Every `transfer` cell resolves node count and batch cap from the `170M` preset regardless of its
layer override; the `transfer: {170M: 4}` entry above distributes each cell across 4 nodes
(32-way data parallel, 64 per device), matching the first `replicate`'s shape. At 64 per device
even the depth-48 leg — roughly twice the 170M activation footprint per example — has ample
memory headroom, and full gradient checkpointing is pinned in every cell besides.

Full gradient checkpointing is what makes the 800M-scale per-device batches above fit in memory;
it costs roughly 30–40% more compute than selective checkpointing.

### Status and resubmission

`oplm sweep status <phase.json>` reports each cell's state (`complete`, `non-finite`, `running`,
`missing`, or `unknown` — see `--help` for the full definitions) and prints a ready-to-run
resubmit line for any preset with incomplete cells:

```text
$ oplm sweep status sweeps/smoke/phase.json
                        smoke (eval/heldout/loss)
┏━━━━━━━━┳━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ preset ┃ idx ┃ cell                     ┃ state    ┃ eval/heldout/loss ┃
┡━━━━━━━━╇━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ 170M   │   0 │ 170M-lr0.0004-om1-a0-s42 │ complete │            2.3100 │
│ 170M   │   1 │ 170M-lr0.0016-om1-a0-s42 │ missing  │                 - │
│ 170M   │   2 │ 170M-lr0.0063-om1-a0-s42 │ missing  │                 - │
└────────┴─────┴──────────────────────────┴──────────┴───────────────────┘
resubmit 170M: sbatch --array=1,2 jobs/170M.sbatch
```

Run the printed `sbatch --array=...` line from inside the phase's own directory (it is relative to
`jobs/`). A preset is withheld from resubmit guidance (with an explicit `skip` line) when every one
of its open cells is `running` or `unknown` — resubmitting a `running` cell risks a duplicate run,
and an `unknown` cell's true state cannot be determined at all because the scheduler could not be
reached.

## The re-centered learning-rate grid

`smoke` and `coarse`'s `--lrs` default to the module constants in `src/oplm/sweep/phases.py`
(confirm with `oplm sweep smoke --help` / `oplm sweep coarse --help`):

- `smoke` (`SMOKE_LRS`): `0.0004,0.0016,0.0063`
- `coarse` (`COARSE_LRS`): `0.0004,0.00063,0.001,0.0016,0.0025,0.004,0.0063`

**This grid moved down one half-decade from an earlier `0.0025,0.01,0.04` (smoke) /
`0.0025,0.004,0.0063,0.01,0.016,0.025,0.04` (coarse) grid.** A real 170M coarse sweep on the old
grid ranked `0.0025 ≈ 0.004 ≫ 0.0063 > 0.01 > 0.016`, putting the winner at the **lower boundary**
of the grid — and `analyze_phase` only selects a refinement region when the coarse winner is
*interior* to its finite grid, so that coarse phase ranked its cells but selected nothing,
stalling the funnel with no candidate to hand `refine`. The new grid keeps the same 1.6× spacing,
shifted one half-decade down, so both of the old grid's observed winners now sit interior with
headroom below them, and `0.0063` remains as an upper guard.

## Phase funnel

The defaults implement this funnel:

| Phase | Purpose |
| --- | --- |
| `smoke` | Check LRs `0.0004,0.0016,0.0063` at 170M for 1,000 steps. |
| `coarse` | Rank the seven-point LR grid at 170M for 10,000 steps. |
| `refine` | Cross the selected local LR region with output multipliers `0.5,1,2`. |
| first `replicate` | Add seeds 43 and 44 to seed 42 for the two finalists. |
| `transfer` | Bracket both finalists' LRs (`×0.625,×1,×1.6`) across depths 12/24/48 at fixed 170M width, exponent pinned at 0.5. |
| `bridge` | Test batch-correction LR multipliers `0.7,1,1.4,2` at the 170M production batch. |
| second `replicate` | Add seeds 43 and 44 to the bridge finalists. |
| `confirm` | Rank the production-batch finalists at 800M. |
| `scale` | **Generate-only** — write production job scripts for the confirmed winner. |

`scale` runs no proxy cells and owns no ranking: it writes 100,000-step production job scripts
carrying the confirmed `lr`, `output_mult`, and `depth_exponent` for each requested preset
(default `170M,400M,800M,1B`) and never submits them — you review and submit the generated
scripts yourself (see [SLURM.md](SLURM.md)).

The proxy phases default to 2,048 global examples (roughly 1M tokens for the intended length
distribution); bridge and later phases use 8,192 (roughly 4M). μP does not make batch size
irrelevant, so the bridge measures this correction rather than assuming a scaling rule.

Each cell warms up over ~10% of its own horizon: 1,000 steps for the 10k phases (smoke uses 100
over 1k), 2,000 for the 20k `refine`, and 5,000 (~5%) for the 100k `scale` cells. Warmup
*fraction*, like batch size, does not transfer across horizon — so the proxy keeps it modest
rather than spending half of a short run ramping, which also avoids the selection bias where a
large warmup fraction tolerates a hotter peak LR than production's ~0.4% warmup can sustain.
`transfer` cells share one horizon (`--steps 10000 --warmup 1000`) across every depth.

## Depth-ray LR-transfer check

`transfer` does **not** sweep the depth-LR exponent. The exponent in the repeated-block
correction `effective_block_lr = width_aware_lr * (24 / L) ** exponent` is a consequence of the
residual parameterization, not a free hyperparameter: OPLM freezes the residual-branch scaling
at `1/√L` (`residual_scaling=sqrt_num_layers`), and under that choice a per-block LR of
`η·(24/L)^e` gives a total one-step feature update across the `L` blocks that scales as
`η·L^(1/2−e)`. Depth invariance therefore pins `e = 0.5`. `e = 0` under-corrects (updates grow
as `√L` — the historical 800M blow-up mode), and `e = 1` **over**-corrects (updates shrink as
`1/√L`, starving deep stacks). Note that `e = 1` is *not* "the CompleteP value" in this
parameterization: CompleteP's depth-flat Adam LR is tied to its `1/L` branch multiplier
(`α = 1`); grafting its exponent onto a `1/√L` branch double-counts the depth correction.

The exponent is therefore pinned (default `--exponent 0.5`) and the phase *verifies* it
empirically instead of selecting it from a grid. The reason verification is still warranted is
architectural: sandwich norm applies a post-norm to each branch output *before* the `1/√L`
multiplier and the learned channel gate, so the branch contribution to the stream is
normalized regardless of weight scale, and Muon's orthogonalized updates change the update-norm
bookkeeping — both weaken the clean derivation above without suggesting a specific alternative.

The check is a **fixed-width depth ray**: every cell keeps the 170M geometry (hidden 768,
12 heads) and varies *only* the layer count over `--depths 12,24,48`, crossing each replicated
finalist with the LR bracket `--lr-mults 0.625,1,1.6` (the coarse grid's own 1.6× spacing)
applied on top of the exponent correction. Depth 24 is the μP reference depth, where the
correction is a no-op — that leg re-checks the finalist itself and anchors the bracket, so
`--depths` must include it. This design isolates depth (no width/heads/horizon confound, unlike
a preset-ray sweep), spans 4× in depth instead of the production ray's 2×, and runs every cell
as a 4-node, 32-GPU data-parallel job on 170M-class hardware.

Reading the result: a finalist **transfers** when the bracket center wins at every depth — the
exponent-corrected LR stayed optimal as depth quadrupled, and the winner proceeds to `bridge`
carrying `depth_exponent=0.5`. If the winning multiplier drifts systematically with depth
(hot at 48 ⇒ the correction is too strong; cold ⇒ too weak), the phase selects nothing and the
per-depth `winning_mults` recorded in `phase.json`'s ranking show the drift direction; the
drift slope across a 4× depth range is itself the measured exponent correction. Production
geometry is exercised where it belongs: `confirm` (800M at the production batch) and the deep
stability probe below.

## Stability diagnostics

Diagnostics are **off by default** and do not affect selection — the sweep ranks by held-out
validation loss regardless. They are an opt-in aid for recording the *mechanism* of a divergence
(the historical 800M run was stable at init but fell apart before 5,000 steps). Enable them per
phase with the `--diagnostics` flag (default `--no-diagnostics`), which pins
`train.stability_diagnostics` in each generated cell and overrides whatever the base config says:

```bash
oplm sweep coarse --config configs/mup-production.yaml \
  --from sweeps/smoke/phase.json --out sweeps/coarse --diagnostics
```

Tune the probe cadence with `train.stability_probe_every` in the base config (default 25 logs;
`0` = grad-norm-only). When enabled, `StabilityDiagnosticsCallback` logs under `diag/*`:

- `diag/grad_norm` — pre-clip global gradient norm, **every training log** (the per-step spike
  tripwire; present only when `max_grad_norm > 0`);
- `diag/residual_rms/{max,mean,argmax_layer,final_layer}` — residual-stream growth and which
  hidden state drives it;
- `diag/logit_rms` — output-logit growth.

Everything except the grad norm comes from **one eager diagnostic forward every
`stability_probe_every` logs** (`output_hidden_states`, the tested SDPA path), run on the
unwrapped model over a small fixed batch. It runs **on the main process only** — model weights are
DDP-identical across ranks, so rank 0's shard is representative, and keeping the extra forward off
the other ranks is what makes it distributed-safe (an earlier version that also used
`output_attentions` and ran on every rank hung multi-GPU runs with an NCCL timeout). There are no
forward hooks, so **`torch.compile` stays on** for the training step — leave `train.compile` at
its production value. Set `stability_probe_every=0` for grad-norm-only (cheapest).

## Deep stability probe

The funnel's only run at the production 800M geometry before `scale` is `confirm`, and its 10k
proxy horizon at one seed cannot by itself certify deep stability (the old 800M run was stable
at init and fell apart before 5,000 steps — a dynamics failure, not an init failure). The
depth-ray `transfer` deliberately trades production geometry for a clean depth measurement, so
this probe is what covers the production stack. After `confirm` selects a winner, run one long
800M probe at the production batch, past the historical failure horizon, with diagnostics on:

```bash
oplm train --preset 800M --config configs/mup-production.yaml \
  train.lr=<confirmed-lr> train.mup_depth_lr_exponent=<confirmed-exponent> \
  train.max_steps=12000 train.stability_diagnostics=true train.stability_probe_every=25 \
  data.train=/path/to/train.parquet
```

If it survives with flat `diag/*` traces, scale with confidence; if it diverges, the `diag/*`
series identify whether the cause is residual-stream growth, attention-logit growth, output-logit
growth, or a gradient spike.

## Head-count control

The `800M` preset was corrected to 20 heads (`head_dim=64`); the earlier unstable run used 16
heads (`head_dim=80`), which violates the width-μP "fixed head dim" assumption independently of
depth. To quantify how much of the original instability was that violation versus depth, run the
old geometry once at the old depth-unaware LR with diagnostics on, as a comparison baseline:

```bash
oplm train --preset 800M --config configs/mup-production.yaml \
  model.num_attention_heads=16 model.head_dim=80 \
  train.lr=<old-50M-derived-lr> train.mup_depth_lr_exponent=0 \
  train.max_steps=12000 train.stability_diagnostics=true train.stability_probe_every=25 \
  data.train=/path/to/train.parquet
```

## Phase artifacts

Every phase directory has the same layout:

```text
sweeps/<phase>/
├── phase.json
├── commands.txt
├── jobs/
│   ├── <preset>.sbatch       # one array job per preset in this phase
│   ├── <preset>.jobs         # array index -> run id, one per line
│   ├── analyze.sbatch        # runs `oplm sweep analyze`, depends on every array (afterany)
│   └── submit.sh             # submits every job above in dependency order
└── runs/
    └── <id>/
        ├── run.yaml
        ├── result.json
        └── output/
```

- `phase.json` is the phase manifest. It records the source phase, planned cells (including each
  cell's resolved `nodes`, `per_device_batch`, and `gradient_accumulation_steps`), ranking,
  selected candidates, the `oplm` version that generated it, and (once submitted) job ids.
- `commands.txt` contains one shell-quoted Accelerate command per generated cell, for reference or
  ad hoc reruns outside Slurm.
- `jobs/` is the Slurm layer's output — see [SLURM.md](SLURM.md) for what it contains and why.
- `runs/<id>/run.yaml` is the fully resolved config for that cell.
- `runs/<id>/result.json` contains the final evaluation values used by ranking.
- `runs/<id>/output/` is the normal training output directory.

Missing, non-numeric, or non-finite validation results are ineligible for selection.

## Selection protocol

Selection is based on finite held-out validation loss from `result.json`:

- ordinary phases rank cells by increasing validation loss;
- each `replicate` phase ranks a candidate by its mean validation loss across seeds 42, 43,
  and 44, using the seed-42 result from its source phase; and
- `transfer` requires the LR bracket's center (`lr_mult=1.0`) to win within every depth — a
  missing or non-finite bracket point counts as worst, so a diverged edge cell loses to any
  finite one, while a missing or non-finite *center* disqualifies the candidate. Transferring
  candidates rank by mean center loss across depths, and selection carries the base `lr` (never
  a bracket edge) with the pinned `depth_exponent`. If no candidate transfers, the phase selects
  nothing; the `winning_mults` on each ranking entry show the per-depth drift direction.

The coarse winner must be interior to its finite LR grid; its neighboring LR values become the
refinement region. Refinement, replication, transfer, bridge, and confirmation then narrow the
candidate set according to the implemented phase rules. Inspect the validation losses and the
`ranking` and `selected` arrays in each `phase.json` before continuing.

**`batch_mult` is ranking metadata only — it is never re-applied to a config.** `bridge` records
each candidate's batch-correction multiplier as `batch_mult` and folds the correction directly
into that candidate's `lr` (`lr = candidate["lr"] * multiplier`); every later phase (`confirm`,
`scale`) carries `batch_mult` forward purely so you can see which batch correction a winner came
from, but reads only `lr`, `output_mult`, and `depth_exponent` back out of a selected candidate.
There is no `train.batch_mult` training-config field, and there must not be one applied by hand
either — the correction is already inside `lr`. Re-applying `batch_mult` on top of that `lr` would
square the correction.

## Production WSD schedule

The sweep and 100k scaling cells select the peak stable learning rate; they do not select the
production cooldown. Production uses:

```text
linear warmup
-> 1–2M stable-LR steps
-> 0.5–1M linear-decay steps
```

Configure `scheduler=wsd_linear`, set `stable_steps` to the plateau length, and set `max_steps`
to warmup plus plateau plus decay. For example:

```yaml
train:
  scheduler: wsd_linear
  lr: 0.01  # illustrative; replace with the confirmed production LR
  min_lr: 0.0
  warmup_steps: 10_000
  stable_steps: 1_500_000
  max_steps: 2_260_000  # 10k warmup + 1.5M stable + 750k decay
```

**Watch out for a zero-decay footgun.** The sweep phases in this document deliberately set
`stable_steps = max_steps - warmup_steps` for every proxy cell — for a *short* proxy run that
convention is fine, since selection only cares about the peak stable LR, not a cooldown. That same
arithmetic is **wrong** for a real production run: it leaves **zero** steps between the end of the
stable plateau and `max_steps`, so a `wsd_linear` schedule trains at peak LR all the way to the
end with no decay at all — silently, since nothing errors. This is not a hypothetical: this
repo's own production job scripts previously shipped with exactly that configuration.
`configs/scaling.yaml` gets this right — `warmup_steps: 5000`, `stable_steps: 85000`,
`max_steps: 100000`, leaving a real 10,000-step (10%) linear-decay tail — and is the reference to
copy the pattern from, not the short-horizon sweep cells.

Preserve optimizer state through the small dataset-mixture change at the stable-to-decay
transition. Do not reset momentum or add a second warmup.

## References

- [SLURM.md](SLURM.md) — the general Slurm job-generation layer this runbook builds on.
- [MUP.md](MUP.md) — OPLM's width-μP parameterization and coordinate-check interpretation.
- [CONFIG.md](CONFIG.md) — model and training configuration fields.
- [TRAIN.md](TRAIN.md) — training, distributed launch, schedules, and resume.

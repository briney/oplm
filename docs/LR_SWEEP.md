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

## Running phases on Slurm

Every phase from `smoke` through `confirm` **generates** artifacts (fully resolved `run.yaml`
configs, a `phase.json` manifest, and one Slurm array job per preset plus a dependent `analyze`
job — see [SLURM.md §6](SLURM.md#6-job-arrays-and-the-homogeneous-resource-constraint)). Do not
generate the next phase until the current one has completed and been ranked. `scale` is different
— see the note at the end of this section.

For the cheap, fast proxy phases (`smoke`, `coarse`, `refine`, and the first `replicate`), pass
`--submit` so generation and submission happen in one step. For the expensive, multi-day,
multi-node phases (`transfer`, `bridge`, the second `replicate`, `confirm`), omit `--submit`,
inspect the generated `jobs/*.sbatch` scripts and `jobs/submit.sh`, then run `bash
sweeps/<phase>/jobs/submit.sh` yourself once you're ready to commit the allocation:

```bash
# Cheap phases: generate and submit in one step.
oplm sweep smoke --config configs/mup-production.yaml --out sweeps/smoke --submit
oplm sweep coarse --config configs/mup-production.yaml \
  --from sweeps/smoke/phase.json --out sweeps/coarse --submit
oplm sweep refine --config configs/mup-production.yaml \
  --from sweeps/coarse/phase.json --out sweeps/refine --submit
oplm sweep replicate --config configs/mup-production.yaml \
  --from sweeps/refine/phase.json --out sweeps/refine-replicate --submit

# Expensive phases: generate, review, then submit manually.
oplm sweep transfer --config configs/mup-production.yaml \
  --from sweeps/refine-replicate/phase.json --out sweeps/transfer
bash sweeps/transfer/jobs/submit.sh
oplm sweep bridge --config configs/mup-production.yaml \
  --from sweeps/transfer/phase.json --out sweeps/bridge
bash sweeps/bridge/jobs/submit.sh
oplm sweep replicate --config configs/mup-production.yaml \
  --from sweeps/bridge/phase.json --out sweeps/bridge-replicate
bash sweeps/bridge-replicate/jobs/submit.sh
oplm sweep confirm --config configs/mup-production.yaml \
  --from sweeps/bridge-replicate/phase.json --out sweeps/confirm
bash sweeps/confirm/jobs/submit.sh

# scale never submits -- see the phase table below.
oplm sweep scale --config configs/mup-production.yaml \
  --from sweeps/confirm/phase.json --out sweeps/scale
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
sbatch sweeps/scale/jobs/oplm-170M-scale.sbatch
sbatch sweeps/scale/jobs/oplm-400M-scale.sbatch
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
| `transfer` | 400M | 2048 | 4 | 64 | ~7 h (10k steps) |
| `transfer` | 800M | 2048 | 8 | 32 | ~14 h (20k steps) |
| `transfer` | 1B | 2048 | 8 | 32 | ~13 h (10k steps) |
| `bridge` | 170M | 8192 | 4 | 256 | ~12 h (10k steps) |
| second `replicate` | 170M | 8192 | 4 | 256 | ~12 h (10k steps, matches `bridge`) |
| `confirm` | 800M | 8192 | 8 | 128 | ~28 h (10k steps) |

Every row above still resolves to `gradient_accumulation_steps=1`.

Full gradient checkpointing is what makes the 400M+ per-device batches above fit in memory; it
costs roughly 30–40% more compute than selective checkpointing.

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
| `transfer` | Test both finalists and depth exponents `0,0.5,0.75,1.0` at 400M, 800M, and 1B. |
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
over 1k), 2,000 for the 20k `refine` and the 20k 800M `transfer` cell, and 5,000 (~5%) for the
100k `scale` cells. Warmup *fraction*, like batch size, does not transfer across horizon — so the
proxy keeps it modest rather than spending half of a short run ramping, which also avoids the
selection bias where a large warmup fraction tolerates a hotter peak LR than production's ~0.4%
warmup can sustain. `transfer` takes a per-preset `--warmup` list aligned with
`--presets`/`--steps`.

## Depth-LR exponent grid

`transfer` sweeps `mup_depth_lr_exponent` over `0,0.5,0.75,1.0` — the empirical repeated-block
LR correction `effective_block_lr = width_aware_lr * (24 / L) ** exponent`. The residual-branch
scaling stays fixed at `1/√L` (`residual_scaling=sqrt_num_layers`); this grid tunes only the
optimizer-side depth correction, which is the dominant lever for depth transfer. The upper end
reaches `1.0` deliberately: it is the CompleteP (Adam) value and the strongest correction worth
testing for a 40–50 layer stack. Because OPLM trains with Muon (an orthogonalized update, not
Adam or SGD), the transferring exponent is not the published CompleteP/Depth-μP number and must
be selected empirically here. Override with `--exponents` to widen or refine the grid.

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

The proxy `transfer`/`confirm` horizons are shorter than the ~5,000-step regime where the old
800M run failed, so they cannot by themselves certify deep stability. After `confirm` selects a
winner, run one long 800M probe at the production batch, past the historical failure horizon,
with diagnostics on:

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
- `transfer` ranks each candidate within each model preset, sums those per-model ranks, and
  prefers the lowest total.

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

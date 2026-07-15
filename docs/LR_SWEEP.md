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
dataset paths:

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
```

The file merges over the packaged defaults. Add any production-specific masking, precision,
compile, or checkpoint overrides to this same file before launch. Keep exactly one named
sequence eval task so the harness can infer its `eval/<name>/loss` metric.

Keep weight decay fixed at `0.01` for the complete workflow. The generator also resolves each
cell with μP enabled, base width 768, reference depth 24, the requested depth exponent, and a
`wsd_linear` schedule whose stable phase runs from the end of warmup to the end of the cell.
The base config's per-device batch size must divide each requested global batch across the
selected process count.

## Parameterization gates

Run both the μP gate and its non-μP control at fixed depth, then check the production preset
ray:

```bash
# Parameterization gates (run μP-on and --no-mup control).
python -m scripts.mup_coord_check --config configs/mup-production.yaml \
  --scaling width --widths 384,768,1536 --depth 24 --base-width 768 \
  --out sweeps/coord-width
python -m scripts.mup_coord_check --config configs/mup-production.yaml \
  --no-mup --scaling width --widths 384,768,1536 --depth 24 \
  --base-width 768 --out sweeps/coord-width-control
python -m scripts.mup_coord_check --config configs/mup-production.yaml \
  --scaling preset_ray --widths 512,768,1024,1280 --base-width 768 \
  --out sweeps/coord-ray
```

With μP enabled, hooked module output RMS should not grow systematically with width. Small
non-systematic variation is acceptable, and readout logits may shrink at initialization by
design. The non-μP control should visibly fan out. The preset-ray check is an empirical test of
combined width/depth behavior, not a replacement for the fixed-depth width gate.

## Local eight-GPU run

`--local` executes all cells in one phase sequentially. Each cell uses one eight-process
Accelerate launch, then the phase is ranked before the command returns. Do not generate the
next phase until the current one has completed and been ranked.

```bash
# Generate jobs; add --local to run each phase sequentially on all eight GPUs.
python -m scripts.mup_sweep smoke --config configs/mup-production.yaml \
  --out sweeps/smoke --num-processes 8 --local
python -m scripts.mup_sweep coarse --config configs/mup-production.yaml \
  --from sweeps/smoke/phase.json --out sweeps/coarse --num-processes 8 --local
python -m scripts.mup_sweep refine --config configs/mup-production.yaml \
  --from sweeps/coarse/phase.json --out sweeps/refine --num-processes 8 --local
python -m scripts.mup_sweep replicate --config configs/mup-production.yaml \
  --from sweeps/refine/phase.json --out sweeps/refine-replicate --num-processes 8 --local
python -m scripts.mup_sweep transfer --config configs/mup-production.yaml \
  --from sweeps/refine-replicate/phase.json --out sweeps/transfer --num-processes 8 --local
python -m scripts.mup_sweep bridge --config configs/mup-production.yaml \
  --from sweeps/transfer/phase.json --out sweeps/bridge --num-processes 8 --local
python -m scripts.mup_sweep replicate --config configs/mup-production.yaml \
  --from sweeps/bridge/phase.json --out sweeps/bridge-replicate --num-processes 8 --local
python -m scripts.mup_sweep confirm --config configs/mup-production.yaml \
  --from sweeps/bridge-replicate/phase.json --out sweeps/confirm --num-processes 8 --local
python -m scripts.mup_sweep scale --config configs/mup-production.yaml \
  --from sweeps/confirm/phase.json --out sweeps/scale --num-processes 8 --local
```

The defaults implement this funnel:

| Phase | Purpose |
| --- | --- |
| `smoke` | Check LRs `0.0025,0.01,0.04` at 170M for 1,000 steps. |
| `coarse` | Rank the seven-point LR grid at 170M for 10,000 steps. |
| `refine` | Cross the selected local LR region with output multipliers `0.5,1,2`. |
| first `replicate` | Add seeds 43 and 44 to seed 42 for the two finalists. |
| `transfer` | Test both finalists and depth exponents `0,0.5,0.75,1.0` at 400M, 800M, and 1B. |
| `bridge` | Test batch-correction LR multipliers `0.7,1,1.4,2` at the 170M production batch. |
| second `replicate` | Add seeds 43 and 44 to the bridge finalists. |
| `confirm` | Rank the production-batch finalists at 800M. |
| `scale` | Run the winner for 100,000 steps across 50M, 170M, 400M, 800M, and 1B. |

The proxy phases default to 2,048 global examples (roughly 1M tokens for the intended length
distribution); bridge and later phases use 8,192 (roughly 4M). μP does not make batch size
irrelevant, so the bridge measures this correction rather than assuming a scaling rule.

Each cell warms up over ~10% of its own horizon: 1,000 steps for the 10k phases (smoke uses 100
over 1k), 2,000 for the 20k `refine` and the 20k 800M `transfer` cell, and 5,000 (~5%) for the
100k `scale` cells. Warmup *fraction*, like batch size, does not transfer across horizon — so the
proxy keeps it modest rather than spending half of a short run ramping, which also avoids the
selection bias where a large warmup fraction tolerates a hotter peak LR than production's ~0.4%
warmup can sustain. `transfer` takes a per-preset `--warmup` list aligned with `--presets`/`--steps`.

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

Deep runs can pass the coordinate check and the short proxy phases yet still diverge later
(the historical 800M run was stable at init but fell apart before 5,000 steps). Enable the
stability diagnostics on the deep phases and probes to record the *mechanism*, not just the loss:

```yaml
train:
  stability_diagnostics: true  # per-step grad norm + periodic residual/logit/entropy probe
  stability_probe_every: 25    # cadence (in logs) of the probe forward; 0 = grad norm only
```

This attaches `StabilityDiagnosticsCallback`, which logs under `diag/*`:

- `diag/grad_norm` — pre-clip global gradient norm, **every training log** (the per-step spike
  tripwire; present only when `max_grad_norm > 0`);
- `diag/residual_rms/{max,mean,argmax_layer,final_layer}` — residual-stream growth and which
  hidden state drives it;
- `diag/logit_rms` — output-logit growth;
- `diag/attn_entropy/{mean,min}`, `diag/attn_max_prob/max` — exact per-layer attention entropy
  (QK-norm and the sigmoid output gate are active by default, so a hard collapse is unlikely, but
  channel-mode QK-norm still permits logit growth via its learned scale).

Everything except the grad norm comes from **one eager diagnostic forward every
`stability_probe_every` logs**, run on the unwrapped model over a small fixed batch. There are no
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

Every phase directory has the same small layout:

```text
sweeps/<phase>/
├── phase.json
├── commands.txt
└── runs/
    └── <id>/
        ├── run.yaml
        ├── result.json
        └── output/
```

- `phase.json` is the phase manifest. It records the source phase, planned cells, ranking, and
  selected candidates.
- `commands.txt` contains one shell-quoted Accelerate command per generated cell.
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

## SUNK/Slurm use

Omit `--local` to generate artifacts without running them. Queue each line from that phase's
`commands.txt` inside its SUNK/Slurm allocation. After all expected results have completed,
rank the phase before generating the next one:

```bash
# Generate without --local, queue each commands.txt line inside its SUNK allocation,
# then rank completed results before generating the next phase.
python -m scripts.mup_sweep analyze sweeps/coarse/phase.json
```

Use the same phase order and `--from` links as the local sequence.

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

Preserve optimizer state through the small dataset-mixture change at the stable-to-decay
transition. Do not reset momentum or add a second warmup.

## References

- [MUP.md](MUP.md) — OPLM's width-μP parameterization and coordinate-check interpretation.
- [CONFIG.md](CONFIG.md) — model and training configuration fields.
- [TRAIN.md](TRAIN.md) — training, distributed launch, schedules, and resume.

# μP and learning rate

How OPLM picks its learning rate: μP with Muon, the LR tuned once at the 170M proxy and reused at
every width, and the sweep funnel that tuned it. The funnel below is the October 2026 μP refresh
(W&B `thebrineylab/oplm-mup-4M`), the run that produced the current defaults.

Related pages:

- [training.md](training.md): launching runs, the schedule, and branching a decay off a checkpoint.
- [slurm.md](slurm.md): the `slurm:` block and job generation that `oplm sweep` builds on.
- [config.md](config.md#μp-maximal-update-parametrization): the `mup_*` fields.
- [architecture.md](architecture.md#μp-hooks): where the model applies μP.

## Recipe

μP + Muon is the default. The tuned set is spread over two files:

- `src/oplm/configs/train/base.yaml`: `train.optimizer` (`muon`), `train.muon_adjust_lr_fn`
  (`original`), `train.lr` (the μP base LR), `train.mup_depth_lr_exponent`,
  `train.mup_depth_reference_layers`, `train.adamw_lr_mult` and `train.weight_decay_to_init`.
- `src/oplm/configs/model/base.yaml`: `model.mup_enable`, `model.mup_base_width` and
  `model.mup_output_mult`.

These values are one tuned set. Don't change one without re-running the funnel.

The method:

1. **Base width 768.** `model.mup_base_width` is the `170M` preset's `hidden_size`, so the width
   multiplier `m = hidden_size / mup_base_width` is 1 there. Every sweep cell pins
   `model.mup_base_width=768` (in `src/oplm/sweep/phases.py`), so a sweep winner drops into the
   defaults unchanged. At the base width μP is a no-op: init, forward multipliers and per-group LRs
   are bit-identical to a non-μP run.
2. **Tune `train.lr` once at the 170M proxy**, at the production batch (4,194,304 tokens per step),
   context 2048, packed rows and flex attention, and rank candidates on loss **after an LR decay**.
3. **Reuse that `train.lr` at every preset.** μP makes the base LR transfer across width. It does
   not transfer across tokens per step, sequence length or training length, so those stay fixed at
   their production values during the sweep (see [LR vs training length](#lr-vs-training-length)).

The run-default vs dataclass-fallback split (bare `OplmConfig()` / `TrainConfig()` and
`from_pretrained` stay μP-off) is in
[architecture.md, *Run defaults and library fallbacks*](architecture.md#run-defaults-and-library-fallbacks).

### Parameterization

Symbols: `d0 = mup_base_width`, `L = num_hidden_layers`, `σ = initializer_range`, and
`head_dim = 64` on every preset (only the head count grows with width).

The one scaling quantity is the per-matrix fan-in multiplier `m_W = fan_in(W) / fan_in_base(W)`
(`OplmConfig.mup_fanin_mult`):

- matrices whose fan-in is `hidden_size` (`q/k/v/o_proj`, `gate/up_proj`, `lm_head.dense`, the
  readout `lm_head.decoder`): `m_W = m = hidden_size / d0`;
- `down_proj` (fan-in `intermediate_size`): `m_W = m_ffn = intermediate_size / i0`, where `i0` is
  the FFN size of a width-`d0` model under the same rounding
  (`OplmConfig.mup_base_intermediate_size`). FFN rounding is absorbed exactly, not approximated by `m`.

Every entry is a no-op when `model.mup_enable=false` or `m_W = 1`.

| Component | Init std (μP) | Forward multiplier | Muon LR | AdamW LR |
| --- | --- | --- | --- | --- |
| `embed_tokens` (input) | `σ` | ×1 | (AdamW) | `lr` ×1 |
| `q/k/v/o_proj`, `gate/up_proj`, `lm_head.dense` | `σ/√m_W` | none | `lr` (×1 via `original`) | `lr/m_W` |
| residual writers `o_proj`, `down_proj` | `σ/(√(2L)·√m_W)` | none | `lr` | `lr/m_W` |
| `lm_head.decoder` (readout, untied) | `σ` (constant) | logits ×`mup_output_mult/m` | (AdamW) | `lr` ×1 |
| norms, biases (1-D) | ones / zeros | none | (AdamW) | `lr` ×1 |
| attention softmax | | `1/√head_dim`, unchanged | | |
| residual add | | `1/√L`, unchanged | | |

Who owns what:

- **Muon mode** (`train.optimizer=muon`): every 2-D weight outside the embedding and `lm_head.*`
  (attention `q/k/v/o_proj` and the output-gate `gate_proj`, FFN `gate/up/down_proj`) goes to Muon
  at a constant `lr`. Everything else stays on AdamW: `lm_head.dense` at `lr/m`, and the embedding,
  the readout, norms, biases and Canon kernels at `lr` ×1.
- **AdamW-only mode** (`train.optimizer=adamw`): hidden matrices take `lr/m_W`. Embedding, readout,
  norms and biases take `lr` ×1.
- Every AdamW group is further multiplied by `train.adamw_lr_mult` (`src/oplm/training/optim.py`).
  Muon groups are not.

Why this is clean for OPLM:

- **Fixed `head_dim`.** The classic μP change of attention scaling to `1/d` isn't needed, because
  `1/√64` is already width-invariant.
- **A standalone readout.** `model.tie_word_embeddings` is off, so `lm_head.decoder` is an
  independent matrix with constant init `σ`, a logit multiplier `mup_output_mult/m` on the matmul
  path only (its bias is untouched), and AdamW LR ×1. That is the standard μP readout (logits
  `Θ(1/√m)`, small at init) without weight tying. Fine-tuning `classifier` heads are tagged as
  readouts too, so their init is constant `σ`; LR transfer for fine-tuning heads is not covered.
- **Muon's `original` factor** scales each matrix's LR by `√(max(1, d_out/d_in))`, a function of
  aspect ratio only. Aspect ratios are nearly constant across presets, so the Muon base LR is
  constant across width. For `d_out < d_in` (`down_proj`) the factor clips at 1. The small error
  left by FFN rounding is checked by the coordinate check.

### Why `muon_adjust_lr_fn=original` is mandatory

`train.muon_adjust_lr_fn=match_rms_adamw` scales each LR by `0.2·√(max(d_out, d_in))`, which grows
like `√width` and breaks transfer. μP + Muon needs `original`, the run default.
`build_optimizers` raises `ValueError` if `model.mup_enable` is on, `train.optimizer=muon` and
`train.muon_adjust_lr_fn=match_rms_adamw`, and the sweep refuses to generate cells with it.

### Depth correction

When depth changes as well as width, parameters inside the repeated blocks
(`oplm.backbone.layers.*`, on both optimizers) get

```text
effective_block_lr(L) = width_aware_lr * (mup_depth_reference_layers / L) ** mup_depth_lr_exponent
```

It doesn't apply to the embedding, the final norm or the MLM head. `train.mup_depth_reference_layers`
is the 170M depth (24), so the correction is exactly 1 at the proxy.

The exponent follows from the residual parameterization; it isn't a free knob. The branch scaling
is frozen at `1/√L` (`model.residual_scaling=sqrt_num_layers`). Summing per-block updates of size
`η·(24/L)^e` over `L` blocks gives a one-step feature update `∝ η·L^(1/2−e)`, so depth invariance
pins `e = 0.5`. `e = 0` under-corrects (updates grow like `√L`) and `e = 1` over-corrects (they
shrink like `1/√L`). CompleteP's depth-flat LR belongs to its `1/L` branch multiplier and doesn't
carry over to a `1/√L` branch. Sandwich norm and Muon both weaken the clean derivation, so it is
checked empirically: the 800M confirmation (40 layers against the proxy's 24) runs with the
correction on.

### Multipliers across the presets

`m = hidden_size / 768`, `m_ffn = intermediate_size / 2048` (2048 is `i0` at `d0 = 768`), and the
depth factor is `(24 / L) ** 0.5` with `L = num_hidden_layers`. `hidden_size` and `L` come from
the preset YAMLs (`src/oplm/configs/model/presets/*.yaml`); `intermediate_size` is as `OplmConfig`
derives it.

| Preset | `hidden_size` | `intermediate_size` | `L` | `m` | `m_ffn` | depth factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `50M` | 512 | 1536 | 16 | 0.67 | 0.75 | 1.225 |
| `170M` | 768 | 2048 | 24 | 1.00 | 1.00 | 1.000 |
| `400M` | 1024 | 2816 | 32 | 1.33 | 1.38 | 0.866 |
| `800M` | 1280 | 3584 | 40 | 1.67 | 1.75 | 0.775 |
| `1B` | 1600 | 4352 | 50 | 2.08 | 2.125 | 0.693 |
| `3B` | 2048 | 5632 | 64 | 2.67 | 2.75 | 0.612 |
| `6B` | 2560 | 6912 | 80 | 3.33 | 3.38 | 0.548 |
| `12B` | 3200 | 8704 | 100 | 4.17 | 4.25 | 0.490 |

## Commands

The funnel: coordinate check, then smoke → coarse → refine at 170M, a decay branch off every
refine cell, replicate the top two and decay them, then confirm at 800M and decay. From `refine`
on, candidates are ranked on the **decayed** eval loss. The harness ranks each phase itself after
it finishes, but on undecayed loss, so its `refine`, `replicate` and `confirm` selections are not
used. Only the `coarse` selection (the LR region that `refine` explores) comes from the harness.

`oplm sweep` also has `transfer` (a fixed-width depth ray), `bridge` (a batch-size correction) and
`scale` (production job scripts) phases. They are not part of this recipe.

### Sweep config

Every command takes one config. Make it from the production recipe:

1. Copy [`configs/scaling.yaml`](../configs/scaling.yaml) to an absolute path on shared storage,
   for example `$HOME/oplm-sweep/sweep.yaml`. It already carries the production context, packing,
   flex attention, static compile, data mix and token mode.
2. Keep only the `uniref70` entry under `data.eval`, evaluated every 500 steps. The harness infers
   its ranking metric (`eval/uniref70/loss`) when there is exactly one eval task, and the structure
   evals only cost time.
3. Set `train.wandb_project` so every cell lands in one W&B project. There is no flag for it.
4. Size the `slurm:` block for the sweep. The refresh ran:

   ```yaml
   slurm:
     nodes: {170M: 4, 800M: 8}
     max_batch_tokens: {170M: 65536, 800M: 32768}
   ```

   At 4,194,304 tokens per step that is 2 accumulation steps per cell on 8-GPU nodes
   (`4194304 / (32 x 65536)` at 170M, `4194304 / (64 x 32768)` at 800M). Token cells keep the
   config's own gradient-checkpointing setting (off), so size `max_batch_tokens` for that. See
   [slurm.md](slurm.md) for every `slurm:` field.

The generator requires `train.weight_decay=0.01` and, for Muon, `train.muon_adjust_lr_fn=original`
(both run defaults). For each cell it pins μP on, `model.mup_base_width=768`, the cell's LR, output
multiplier, seed and depth exponent, a `wsd_linear` schedule with
`stable_steps = max_steps - warmup_steps` (no decay inside the cell), and in token mode
`train.tokens_per_step`, `train.batch_tokens`, the derived accumulation and `data.pack_sequences=true`.

### Where the commands run

- **Generate inside the training container.** Every `oplm sweep <phase>` resolves a full training
  config per cell, which imports `torch` and `transformers`. A one-CPU `srun` with the same
  container flags as the training jobs works. `status` and `analyze` only read JSON manifests.
- **Use absolute paths for `--config` and `--out`.** Inside the container the working directory is
  node-local `/tmp`. A relative `--out` writes the phase there, prints `wrote ...`, exits 0, and the
  files are lost when the job ends.
- **Pin the oplm version for the whole sweep** in `slurm.install` (for example
  `pip install oplm[train]==<version>`), so every phase is generated and trained by the same code.
  Each cell records its version in `result.json`.

### Run the coordinate check

Run the μP gate, its non-μP control, and the preset ray before any sweep, and again after any
change to init, forward multipliers or optimizer grouping:

```bash
CONFIG=$HOME/oplm-sweep/sweep.yaml
SWEEP=$HOME/oplm-sweep/sweeps

oplm sweep coord-check --config $CONFIG \
  --scaling width --widths 384,768,1536 --depth 24 --base-width 768 --out $SWEEP/coord-width
oplm sweep coord-check --config $CONFIG --no-mup \
  --scaling width --widths 384,768,1536 --depth 24 --base-width 768 --out $SWEEP/coord-width-control
oplm sweep coord-check --config $CONFIG \
  --scaling preset_ray --widths 512,768,1024,1280 --base-width 768 --out $SWEEP/coord-ray
```

Each run writes `coord_check_{scaling}_{optimizer}_{mup|nomup}.{csv,png}` into `--out` and prints a
per-module RMS-growth table with a verdict. Reading it is under
[Coordinate check](#coordinate-check).

### Smoke, coarse and refine at 170M

```bash
oplm sweep smoke --config $CONFIG --global-tokens 4194304 --out $SWEEP/smoke --submit
oplm sweep coarse --config $CONFIG --from $SWEEP/smoke/phase.json --global-tokens 4194304 \
  --lrs 0.00063,0.001,0.0016,0.0025,0.004,0.0063,0.01,0.016 --out $SWEEP/coarse --submit
oplm sweep refine --config $CONFIG --from $SWEEP/coarse/phase.json --global-tokens 4194304 \
  --out $SWEEP/refine --submit
```

Generate each phase only after the previous one has finished and been ranked (a dependent
`analyze` job does that automatically).

- `smoke`: the three default LRs (`--lrs` default), 1,000 steps. The two lowest LRs must finish
  with a finite loss.
- `coarse`: an 8-point grid with 1.6× spacing, 10,000 steps. The grid is passed explicitly,
  because the built-in default grid tops out at 0.0063 and the coarse winner must be interior: the
  harness selects the winner and its two grid neighbours for `refine`, and selects nothing if the
  winner is at either end. If smoke's highest LR diverged, it is dropped from the coarse grid.
- `refine`: each selected LR crossed with `model.mup_output_mult` 0.5, 1 and 2 (`--output-mults`
  default), 20,000 steps with a 2,000-step warmup. That is 9 cells.

Refresh wall times on 8×B200 nodes: about 20 minutes per smoke cell, 2.7 h per coarse cell and
5.4 h per refine cell, at 4 nodes per cell.

### Decay every refine cell

Branch each refine cell from its final checkpoint into a 20% linear decay tail and rank on the
decayed loss. The harness doesn't automate this. Each branch is an ordinary training run built
from the cell's own resolved `run.yaml`, with three overrides: `train.branch_from` (the
checkpoint), `train.stable_steps` (the plateau ends at the checkpoint's step) and
`train.max_steps` (the end of the decay). The arithmetic and the rules (fresh `train.output_dir`,
`data.train` left as `run.yaml` has it and the same node count, so the data cursor is restored) are in
[training.md, *Branching a decay off a checkpoint*](training.md#branching-a-decay-off-a-checkpoint).

For a cell of `N` steps with `W` warmup steps and a decay of `D = N / 5` steps:

```bash
# refine: 20k-step cells, 2k warmup, decay 20000 -> 24000 on the cell's own 4 nodes.
PHASE=refine NODES=4 N=20000 W=2000 D=4000

for run in $SWEEP/$PHASE/runs/*; do
  id=$(basename $run)
  out=$SWEEP/$PHASE-decay/$id
  mkdir -p $out
  # run.yaml has no slurm: block; append the sweep config's.
  { cat $run/run.yaml; awk '/^[^ #]/{p=/^slurm:/} p' $CONFIG; } > $out/decay.yaml
  oplm slurm generate --config $out/decay.yaml --nodes $NODES --name $PHASE-decay-$id \
    --out $out/jobs train.branch_from=$run/output/checkpoint-$N train.output_dir=$out/output \
    train.stable_steps=$((N - W)) train.max_steps=$((N + D)) train.wandb_run_name=$PHASE-decay-$id
  oplm slurm submit $out/jobs
done
```

Every cell keeps its final checkpoint (`checkpoint-20000` here), so no extra save setting is
needed. Rank the cells on `eval/uniref70/loss` at step `N + D` (24,000) of each decay run, read
from W&B.

### Replicate the top two and decay them

Pass the two best **decayed** refine cells explicitly as `lr:output_mult`. The refresh replicated
0.0063 and 0.004, both at output multiplier 1:

```bash
oplm sweep replicate --config $CONFIG --from $SWEEP/refine/phase.json \
  --candidates 0.0063:1,0.004:1 --out $SWEEP/replicate --submit
```

This adds seeds 43 and 44 to each candidate, reusing the refine cell as seed 42. Then decay the new
cells with the same loop and `PHASE=replicate` (same `NODES`, `N`, `W`, `D`). Rank each candidate
on the mean decayed loss over the three seeds, with seed 42 taken from `refine-decay`.

### Confirm at 800M and decay

Run the winner and its refine neighbours at the 800M preset (40 layers, depth correction on).
Candidates are `lr:output_mult:depth_exponent:batch_mult`, and `batch_mult` is 1 because the proxy
already ran at the production batch:

```bash
oplm sweep confirm --config $CONFIG --from $SWEEP/replicate/phase.json \
  --candidates 0.004:1:0.5:1,0.0063:1:0.5:1,0.01:1:0.5:1 --global-tokens 4194304 \
  --diagnostics --out $SWEEP/confirm --submit
```

Cells run 10,000 steps with a 1,000-step warmup (the defaults), about 4.3 h each on 8 nodes. Decay
them with the same loop and `PHASE=confirm NODES=8 N=10000 W=1000 D=2000`, and rank on step 12,000.
The winner's LR goes into `train.lr` in both `src/oplm/configs/train/base.yaml` and
[`configs/scaling.yaml`](../configs/scaling.yaml), which pins its own `lr`.

### Status, resubmission and phase artifacts

`oplm sweep status $SWEEP/<phase>/phase.json` lists each cell as `complete`, `non-finite`,
`running`, `missing` or `unknown`, and prints an `sbatch --array=...` resubmit line for incomplete
cells. Run that line from the phase directory. A cell that trained but died before writing
`result.json` looks healthy in W&B and is invisible to ranking, which is what `status` catches.
After a manual resubmission, re-rank with `oplm sweep analyze $SWEEP/<phase>/phase.json`.

```text
<phase>/
├── phase.json          # manifest: cells (nodes, batch, accumulation), ranking, selected, oplm version, job ids
├── commands.txt        # one Accelerate command per cell, for ad hoc reruns
├── jobs/               # <preset>.sbatch + <preset>.jobs (array index -> run id), analyze.sbatch, submit.sh
└── runs/<id>/
    ├── run.yaml        # the fully resolved cell config
    ├── result.json     # final eval values used by ranking
    └── output/         # the training output directory (checkpoints)
```

W&B run names are `<phase>-<preset>-lr<lr>-om<output_mult>-a<depth_exponent>-s<seed>`, for example
`coarse-170M-lr0.0063-om1-a0-s42`. Missing or non-finite results are never selected. `--local`
runs every cell of a phase sequentially on one node and ranks it, for checking a new sweep config
before committing an allocation; it is mutually exclusive with `--submit`.

## What to check

### Coordinate check

The coordinate check is the correctness gate for the μP implementation. The oracle is one-sided:

- With μP on, **no module's RMS grows with width**. Small non-systematic wobble is fine. The
  command prints `μP oracle holds` when the worst module grows at most 2× across the width range,
  and `μP oracle FAIL` otherwise.
- The `--no-mup` control must visibly fan out with width.
- The readout (`lm_head.decoder`) may shrink at init (its logits are `Θ(1/√m)` by design). The
  growth table is taken at the final step, so step 0 is already excluded.
- Attention pre-softmax logits live inside SDPA and aren't hooked.
- `preset_ray` co-scales depth with width. It is an empirical check of combined width and depth
  behaviour, not a substitute for the fixed-depth width gate.

### Refresh reference results

Final `eval/uniref70/loss` from the refresh. A re-run on the same data should land close to these.

- **Smoke (1k steps):** monotone, highest LR best (0.0063 2.354, 0.0016 2.385, 0.0004 2.453).
- **Coarse (10k, undecayed):** 0.0063 won at 2.1639 and was interior (0.004 2.1666, 0.01 2.1670),
  so `refine` explored 0.004, 0.0063 and 0.01.
- **Decaying flipped the refine ranking.** Undecayed at 20k, 0.004 led. After the 20k → 24k decay,
  0.0063 at output multiplier 1 scored **2.0775**, against **2.0792** for the best 0.004 cell
  (output multiplier 2). 0.0063 beat 0.004 at every output multiplier.
- **Output multiplier.** Output multiplier 2 was nominally best at every LR on seed 42 (by
  0.0004–0.0019, unreplicated, within the seed-to-seed spread of about 0.0016); the lowest decayed
  refine cell was 0.0063 at output multiplier 2 (2.0771). The recipe keeps 1.
- **The optimum is bracketed.** 0.004 is clearly worse. At output multiplier 1, 0.01 finished 0.002
  behind 0.0063 on seed 42 (0.0009 behind the 3-seed mean), and within 0.0005 at 0.5 and 2. The
  basin is flat between 0.0063 and 0.01 (vertex ≈0.007), consistent with ~0.008 at 24k in
  [LR vs training length](#lr-vs-training-length).
- **Replicate (3 seeds, decayed, output multiplier 1):** 0.0063 2.0786 against 0.004 2.0812, a gap
  of 0.0026 (about 4.5 standard errors). Undecayed, the order reverses.
- **Confirm (800M):** undecayed at 10k, 0.0063 2.0819 < 0.004 2.0837 < 0.01 2.0845, the same order
  and gaps as 170M coarse at 10k, so the LR transfers. Decayed at 12k, 0.01 2.0301 and 0.0063
  2.0306 tie and 0.004 trails at 2.0350, the same shape as 170M. The recipe kept 0.0063.

### Stability diagnostics

`--diagnostics` on a phase (or `train.stability_diagnostics=true` on any run) logs, under `diag/*`:
the pre-clip global grad norm every training log (`diag/grad_norm`, only when
`train.max_grad_norm > 0`), and every `train.stability_probe_every` logs one eager probe forward
giving residual-stream RMS (`diag/residual_rms/{max,mean,argmax_layer,final_layer}`) and output-logit
RMS (`diag/logit_rms`). The probe runs on the main process only, over a small fixed batch, with no
hooks, so `train.compile` stays on. Diagnostics never affect ranking. In the 800M confirm, look for
flat `diag/logit_rms`, falling grad norms and no spikes. Residual-stream RMS grows with LR; it was
the reason the residual gate was dropped (see [Why and what was ruled out](#why-and-what-was-ruled-out)).

## LR vs training length

The best LR falls as training gets longer. Estimated best LR, on decayed loss:

| Training length (steps at 4,194,304 tokens) | Estimated best LR |
| ---: | ---: |
| 24k | ~0.008 |
| 60k | ~0.0043 |
| 120k | ~0.0033 |

The funnel ranks at 24k, so `train.lr` in `src/oplm/configs/train/base.yaml` (and in
`configs/scaling.yaml`) is a 24k-horizon value, too high for a production run of millions of steps.
The production LR comes from extrapolating this trend to the production length. Width transfer still
holds: the extrapolated value is a base LR, used unchanged at every preset.

> **Pending:** the long-run LR-vs-length results and the extrapolated production LR (waiting on #55).

## Decay recipe

The 20% linear tail in the funnel is a ranking device: it makes each cell's loss comparable to the
end of a real run. It is not the production decay.

> **Pending:** the production decay shape, length fraction and LR floor (waiting on #55).

Whatever the decay recipe turns out to be, keep the optimizer state through the stable-to-decay
transition, including the data-mix change there: no momentum reset and no second warmup. A branch
off the stable-phase checkpoint does this by default.

The mechanics of running a decay off a stable-phase checkpoint are in
[training.md, *Branching a decay off a checkpoint*](training.md#branching-a-decay-off-a-checkpoint).

## How to change it

- **A different preset.** Nothing to retune: pick the preset and keep `train.lr`.
- **A different tokens per step, context length, data mix or training length.** μP doesn't cover
  these. Re-run the funnel with the new values in the sweep config (`--global-tokens` for the
  batch).
- **A different optimizer or architecture recipe.** Re-run the coordinate check and the funnel with
  that recipe as `--config`. For the ESM-C comparison model (AdamW-only μP), merge
  `src/oplm/configs/train/esm-c.yaml` over the sweep config as shown in
  [training.md, *Optimizer and schedule*](training.md#optimizer-and-schedule), pass
  `--optimizer adamw` to `coord-check`, and write the winning `train.lr` and `model.mup_output_mult`
  back into `esm-c.yaml`. Its current values are placeholders until that campaign runs.
- **`model.mup_base_width`, `model.mup_output_mult`, `train.adamw_lr_mult`,
  `train.mup_depth_lr_exponent` or `train.weight_decay_to_init`.** These were tuned together.
  Changing the base width moves the anchor, so results tuned at one base width don't carry over to
  another.
- **μP off.** `model.mup_enable=false` restores standard parameterization; then `train.lr` must be
  tuned per size.

## Why and what was ruled out

- **Rank after decay.** Production runs end in a decay, and at peak LR the comparison favours a
  lower LR: undecayed, 0.004 beat 0.0063 at 170M; decayed, 0.0063 won by 0.0026 over three seeds.
- **Proxy at the production batch; no transfer or bridge.** Tuning at 4,194,304 tokens per step
  leaves no batch correction for `bridge` to measure, and the 800M confirm (wider and deeper than
  the proxy) checks width and depth transfer directly, so `transfer` was skipped too.
- **The residual gate was dropped** (`model.residual_gate` is `none` in
  `src/oplm/configs/model/base.yaml`). A per-channel gate duplicates the sandwich post-norm gain and
  drove residual-stream growth in the top layers. Paired, same-seed, decayed runs with and without
  it (170M at three LRs and 800M at 0.0063) differed by −0.0001 on average, which is noise.
- **`train.weight_decay_to_init`** at 0.1 bounds the post-norm gains at no measurable loss cost:
  paired decayed 170M runs at 0.0063 scored 2.0794 at 0.1 against 2.0790 at 0.01, within noise.
  See [training.md, *Optimizer and schedule*](training.md#optimizer-and-schedule).
- **`train.adamw_lr_mult`** was carried over from earlier 170M runs at a smaller batch (AdamW-side
  LR ÷3 against ÷10, first set up as branches in the late-loss investigation,
  [LATE_LOSS_INVESTIGATION.md](https://github.com/briney/oplm/blob/7db9c1bf0155fc9b4c05c8b358b3cf4cd6648f2c/docs/LATE_LOSS_INVESTIGATION.md));
  it was held fixed and not re-tuned at 4,194,304 tokens per step.
- **AdamW-only μP** is the ESM-C comparison recipe in `src/oplm/configs/train/esm-c.yaml`, not the
  production optimizer.
- The batch-mode funnel this replaces (proxy batch of 2,048 sequences, a depth-ray `transfer`, a
  batch `bridge` and a second replicate) is in
  [LR_SWEEP.md](https://github.com/briney/oplm/blob/7db9c1bf0155fc9b4c05c8b358b3cf4cd6648f2c/docs/LR_SWEEP.md),
  and the earlier μP write-up in
  [MUP.md](https://github.com/briney/oplm/blob/7db9c1bf0155fc9b4c05c8b358b3cf4cd6648f2c/docs/MUP.md).

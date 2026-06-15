# μP: Learning-Rate Transfer Across Width

**Maximal Update Parameterization (μP)** lets a learning rate tuned on a small
pilot model transfer to much larger models without re-sweeping at every scale.
Tune `train.lr` **once** at a small proxy width, then reuse the same number at
every larger preset.

μP is **opt-in** (`model.mup_enable=false` by default) and a **no-op at the base
width** — with it off, or at `hidden_size == mup_base_width`, init, forward
multipliers, and per-group LRs are bit-identical to a non-μP run, so existing
runs, checkpoints, and tests are unchanged. The production optimizer for μP is
**Muon**; AdamW is also supported.

Related references:

- [TRAIN.md](TRAIN.md) — the general training how-to.
- [CONFIG.md](CONFIG.md#μp-maximal-update-parametrization) — the three `mup_*` config knobs.
- [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) — the model itself.

---

## 1. Quick start

The recipe lives in a config overlay you apply on top of a size `--preset`:

```bash
# Tune once at the proxy width (50M, hidden_size = mup_base_width = 512):
python -m scripts.mup_sweep --widths 256,512 --lrs 1e-3,3e-3,1e-2,3e-2 \
  --gpus 4 --steps 400 --data /data/corpus.parquet --out sweeps/run1
# → pick the argmin-loss lr (the sweep prints it and a transfer verdict)

# Reuse that lr unchanged at any larger preset:
oplm train --preset 400M --config src/oplm/configs/train/mup_muon.yaml \
  data.train=/data/uniref50/ train.lr=<base-lr-from-the-sweep>
```

`configs/train/mup_muon.yaml` flips on μP and the Muon settings it requires:

```yaml
train:
  optimizer: muon
  muon_adjust_lr_fn: original   # μP+Muon REQUIRES this (see §4)
model:
  mup_enable: true
  mup_base_width: 512           # the proxy width where the width multiplier m = 1
```

With μP on, **`train.lr` is the base LR** — the value you tuned at the proxy
width. You do not re-tune it per preset.

---

## 2. The workflow

1. **Gate the implementation** with the coordinate check (§5). This is the
   correctness test: with μP, per-module activation RMS must *not grow* with
   width. Run it before trusting any sweep.
2. **Sweep the LR** at two small proxy widths (§6). The loss-vs-LR minimum should
   land at the **same** `lr` for both widths — that shared minimum is your base
   LR, and `MupTransferResult.transferred == True` confirms it.
3. **Reuse** that base LR at the target preset via `--config mup_muon.yaml`.
4. **Confirm** on one larger preset that the loss curve tracks expectation.

The proxy/base width is `mup_base_width` (default `512`, the 50M preset's
`hidden_size`); the width multiplier `m = hidden_size / mup_base_width` is `1`
there, so μP is a no-op at 50M and scales everything else relative to it.

---

## 3. The recipe

Symbols: `d0 = mup_base_width`, `L = num_hidden_layers`, `σ = initializer_range`
(`0.02`), `head_dim = 64` (fixed across presets — only the head *count* grows).

The single scaling quantity is the **per-matrix fan-in multiplier**
`m_W = fan_in(W) / fan_in_base(W)`:

- matrices whose fan-in is `hidden_size` (`q/k/v/o_proj`, `gate/up_proj`,
  `lm_head.dense`, the readout `lm_head.decoder`): `m_W = m = hidden_size / d0`;
- `down_proj` (fan-in is `intermediate_size`): `m_W = m_ffn = intermediate_size /
  i0`, where `i0 = round_up_to(int(8·d0/3), 256)` is the base FFN size — so FFN
  rounding is absorbed exactly, not approximated by `m`.

Every entry below is a no-op when `mup_enable=false` or `m_W = 1`.

| Component | Init std (μP) | Forward mult | Muon LR | AdamW LR |
|---|---|---|---|---|
| `embed_tokens` (input) | `σ` (unchanged) | input ×1 | — (AdamW) | `lr` ×1 |
| `q/k/v/o_proj`, `gate/up_proj`, `lm_head.dense` | `σ/√m_W` | — | `lr` (×1 via `original`) | `lr/m_W` |
| residual writers `o_proj`, `down_proj` | `σ/(√(2L)·√m_W)` | — | `lr` | `lr/m_W` |
| `lm_head.decoder` (readout, **untied**) | `σ` (constant, no `1/√m`) | logits ×`mup_output_mult/m` | — (AdamW) | `lr` ×1 |
| norms, biases (1-D) | ones / zeros | — | — | `lr` ×1 |
| attention softmax | — | `1/√head_dim` **unchanged** (head_dim fixed) | | |
| residual add | — | `1/√L` **unchanged** (already implemented) | | |

**Why this is clean for OPLM:**

- **`head_dim = 64` is fixed across presets** (only the head count grows), so the
  classic μP "`1/d` attention scaling" change is **not needed** — `1/√64` is
  already width-invariant.
- **The readout is a standalone μP readout, not a tied weight.**
  `tie_word_embeddings` defaults to `false`, so `lm_head.decoder` is an
  independent matrix: constant init `σ`, an output-logit multiplier
  `mup_output_mult/m` on the **matmul path only** (so `decoder.bias` is
  untouched), and AdamW LR `×1`. This reproduces the standard μP readout (logits
  `Θ(1/√m)` → small at init, correct update scale) without relying on weight
  tying.
- **Muon's `original` factor** multiplies per-param LR by `√(max(1, d_out/d_in))`
  — a function of *aspect ratio only*. Aspect ratio is approximately constant
  across presets, so the Muon **base LR is constant across width** (the residual
  error from FFN rounding is validated empirically by the coord-check).

### Owner-by-optimizer

- **Muon mode** (`optimizer=muon`): `q/k/v/o, gate/up/down` → Muon at a constant
  `lr`. Only `lm_head.dense` stays on AdamW and takes `lr/m`; `lm_head.decoder`
  (the readout) also stays on AdamW, at `lr ×1`.
- **AdamW-only mode** (`optimizer=adamw`): hidden matrices take `lr/m_W` (`lr/m`
  for hidden-fan-in matrices, `lr/m_ffn` for `down_proj`); embedding, the readout
  `lm_head.decoder`, norms, and biases stay at `lr` (×1).

### Multipliers across the presets

`mup_base_width=512`, so `i0 = 1536` and the 50M preset is the `m = 1` base:

| Preset | `hidden_size` | `intermediate_size` | `m` | `m_ffn` |
|---|---|---|---|---|
| 50M  | 512  | 1536 | 1.00 | 1.00 |
| 170M | 768  | 2048 | 1.50 | 1.33 |
| 400M | 1024 | 2816 | 2.00 | 1.83 |
| 800M | 1280 | 3584 | 2.50 | 2.33 |
| 1B   | 1600 | 4352 | 3.12 | 2.83 |
| 3B   | 2048 | 5632 | 4.00 | 3.67 |

`m_ffn ≠ m` because `intermediate_size` rounds to a multiple of 256; using the
true per-matrix fan-in keeps the scaling exact rather than approximating
`down_proj` by `m`.

---

## 4. Why `muon_adjust_lr_fn=original` is mandatory

The default `muon_adjust_lr_fn="match_rms_adamw"` uses `0.2·√(max(d_out, d_in))`,
which **grows like `√width`** and does **not** transfer. μP+Muon requires
`"original"` (aspect-ratio-only). The guard is enforced — `build_optimizers`
raises `ValueError` if `mup_enable` and `optimizer="muon"` and
`muon_adjust_lr_fn="match_rms_adamw"`. The `mup_muon.yaml` overlay sets it for
you.

---

## 5. Coordinate check — the correctness gate

The coord-check is the authoritative test that the μP implementation is correct.
It builds a model at several widths, runs a few optimizer steps on one fixed
batch, and records every `nn.Linear`/`nn.Embedding` output RMS per step.

```bash
# The gate (vary width only, fixed depth):
python -m scripts.mup_coord_check --widths 128,256,512,1024 --optimizer muon

# The control — should fan out / grow with width:
python -m scripts.mup_coord_check --no-mup --widths 128,256,512,1024

# The preset ray (co-scale depth with width; see Caveats):
python -m scripts.mup_coord_check --scaling preset_ray --widths 256,512,1024 --optimizer muon
```

Each run writes `coord_check_{scaling}_{optimizer}_{mup|nomup}.{csv,png}` into
`--out` (μP and its control land side by side) and prints a per-module
RMS-growth table plus a verdict.

**Pass/fail oracle** — the gate is **one-sided**:

- With μP, **no module's RMS grows with width**. Read it as "does not grow," not
  "stays perfectly flat" — small non-systematic wobble is fine. The `--no-mup`
  control is the contrast: it fans out with width.
- The **readout-logits module** (`lm_head.decoder`) is **allowed to shrink at
  init** (its logits are `Θ(1/√m)` by design). Assess it across widths at steps
  `t ≥ 1` only; the script's growth summary is taken at the final step, so `t=0`
  is already excluded.
- Internal attention pre-softmax logits live inside SDPA and are not hooked as
  named-submodule outputs, so they need no separate exclusion.

---

## 6. LR sweep harness

Once the gate passes, sweep the LR across an `(width, lr)` grid. The orchestrator
fans one pilot subprocess per grid point, each pinned to a single GPU.

```bash
python -m scripts.mup_sweep --widths 256,512 --lrs 1e-3,3e-3,1e-2,3e-2 \
  --gpus 4 --steps 400 --data /data/corpus.parquet --out sweeps/run1
```

- `--widths` are hidden sizes. `--scaling width` (default) fixes depth;
  `--scaling preset_ray` co-scales depth with width to validate the preset ray.
- `--gpus N` is the concurrency (GPU ids `0..N-1`); every run shares `seed`,
  `batch_size`, `warmup_steps`, and `steps` — only `lr` and width vary.

On completion it loads each run's `metrics.json`, prints the per-`(width, lr)`
loss table, the argmin LR per width, and the **transfer verdict**, then writes
`sweep_loss_vs_lr.png` (one line per width, argmin starred). The μP payoff: the
minimum lands at the **same** `lr` for every proxy width
(`MupTransferResult.transferred == True`) — that `lr` is what you put in
`train.lr`.

A single grid point can be run standalone for debugging:

```bash
python -m scripts.mup_pilot_run --width 512 --lr 1e-2 --steps 200 \
  --data /data/corpus.parquet --out runs/w512_lr1e-2
```

These scripts require the `train` extra (`pip install -e '.[train]'`) for pandas
and matplotlib.

---

## 7. Caveats

- **μP guarantees transfer across width at fixed depth.** OPLM's existing `1/√L`
  depth scaling (`residual_scaling="sqrt_num_layers"` + `1/√(2L)` residual-writer
  init) is a *separate, weaker* guarantee. Combined width+depth transfer along the
  preset ray (the constant-aspect-ratio scaling the presets ship) is validated
  **empirically** — coord-check `preset_ray` mode plus a confirmation run — not
  asserted from theory.
- **μP does not transfer across batch size or training horizon.** If batch size
  changes with scale, apply `lr ∝ √(batch ratio)`; for the training horizon,
  prefer WSD schedules / weight-decay adjustment.
- **`original` clips the Muon factor at 1 for `d_out < d_in` matrices** (e.g.
  `down_proj`). It is still ~scale-invariant modulo FFN rounding, so transfer
  holds — the coord-check validates this empirically.
- **Fine-tuning heads are out of scope.**
  `OplmForSequenceClassification`/`OplmForTokenClassification` `classifier` heads
  are tagged `_is_readout=true`, so init treats them as readouts (constant `σ`,
  never `σ/√m`). LR transfer for fine-tuning heads is a follow-up, not covered
  here.

---

## See also

- [TRAIN.md](TRAIN.md) — training how-to (data, launching, checkpointing).
- [CONFIG.md](CONFIG.md#μp-maximal-update-parametrization) — the `mup_*` knobs.
- [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) — depth scaling and the readout.

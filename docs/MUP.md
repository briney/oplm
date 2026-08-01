# μP: Learning-Rate Transfer Across Width

**Maximal Update Parameterization (μP)** lets a learning rate tuned on a small
proxy model transfer to much larger models without re-sweeping at every scale.
Tune `train.lr` **once** at a small proxy width, then reuse the same number at
every larger preset.

μP + Muon is the **default training recipe** (`oplm train` uses it out of the box;
`train.lr` defaults to the μP base LR `0.01`). It is a **no-op at the base width** —
at `hidden_size == mup_base_width`, init, forward multipliers, and per-group LRs
are bit-identical to a non-μP run. The production optimizer for μP is **Muon**;
AdamW is also supported. To turn μP off (and recover the conventional ESM-C
recipe — AdamW plus the pre-2026 architecture), apply
`configs/train/vanilla_esm-c.yaml` (see §1).

Backward-compatibility note: the default lives in the **YAML run-config layer**
(`configs/*/base.yaml`). Bare `OplmConfig()` / `TrainConfig()` construction and
`from_pretrained` stay μP-off (the conservative dataclass fallback), and existing
checkpoints load with whatever `mup_enable` they were saved with.

Related references:

- [TRAIN.md](TRAIN.md) — the general training how-to.
- [CONFIG.md](CONFIG.md#μp-maximal-update-parametrization) — the μP model and training knobs.
- [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) — the model itself.

---

## 1. Quick start

μP + Muon is the default, so training at any preset just works — `train.lr` is the
μP base LR (`0.01`), reused unchanged at every size:

```bash
oplm train --preset 1B data.train=/data/uniref50/   # μP + Muon + lr 0.01, by default
```

The defaults live in `configs/train/base.yaml` (`optimizer: muon`,
`muon_adjust_lr_fn: original`, `lr: 0.01`) and `configs/model/base.yaml`
(`mup_enable: true`, `mup_base_width: 512`). With μP on, **`train.lr` is the base
LR** and transfers across width — you do not re-tune it per preset.

**To re-tune the base LR** for a new optimizer or data regime, follow the phased
production workflow in [LR_SWEEP.md](LR_SWEEP.md). It ranks held-out validation
loss, repeats finalists across three seeds, verifies transfer across the preset
ray, and measures the production-batch correction.

**To disable μP** (vanilla ESM-C recipe — AdamW plus the pre-2026 architecture,
e.g. for an ablation), apply the opt-out overlay — note its `lr` is a plain AdamW
LR that must be tuned per size:

```bash
oplm train --preset 400M --config src/oplm/configs/train/vanilla_esm-c.yaml \
  data.train=/data/uniref50/ train.lr=<adamw-lr-for-this-size>
```

---

## 2. The workflow

1. **Gate the implementation** with the coordinate check (§5). This is the
   correctness test: with μP, per-module activation RMS must *not grow* with
   width. Run it before trusting any sweep.
2. **Run the phased sweep** in [LR_SWEEP.md](LR_SWEEP.md). It selects by held-out
   validation-loss ranking, three-seed means, and summed per-model transfer ranks.
3. **Reuse** the selected base LR across width under the same batch and data
   conditions. The production batch bridge supplies any batch correction.
4. **Confirm** the combined width, depth, and batch result before scaling.

The general model-config fallback/default is `mup_base_width=512`, the `50M`
preset's `hidden_size`; the width multiplier `m = hidden_size / mup_base_width`
is `1` there. The production sweep deliberately overrides that value and forces
`mup_base_width=768`, making the `170M` preset its production anchor. Do not mix
results calibrated against the two different anchors.

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

### Optional repeated-block depth correction

Width-μP supplies `width_aware_lr`. When model depth also changes, the optional
empirical correction for parameters inside repeated Transformer blocks is:

```text
effective_block_lr(L) = width_aware_lr * (mup_depth_reference_layers / L) ** mup_depth_lr_exponent
```

`mup_depth_reference_layers` defaults to `24`, and
`mup_depth_lr_exponent=0.0` makes the correction a no-op. The correction does
not apply to embeddings, the final stack norm, or the MLM head/readout. The
production sweep selects one exponent for the complete scaling ray from a
`0,0.5,0.75,1.0` grid (empirical because Muon is neither Adam nor SGD, so the
published CompleteP/Depth-μP exponents do not carry over); see
[LR_SWEEP.md](LR_SWEEP.md#depth-lr-exponent-grid).

### Multipliers across the presets

For the general `mup_base_width=512` default, `i0 = 1536` and the 50M preset is
the `m = 1` base:

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

`muon_adjust_lr_fn="match_rms_adamw"` uses `0.2·√(max(d_out, d_in))`, which
**grows like `√width`** and does **not** transfer. μP+Muon requires `"original"`
(aspect-ratio-only), which is the **default** (`configs/train/base.yaml`). The
guard is enforced — `build_optimizers` raises `ValueError` if `mup_enable` and
`optimizer="muon"` and `muon_adjust_lr_fn="match_rms_adamw"` — so you only hit it
if you explicitly switch back to `match_rms_adamw` while μP+Muon are on.

---

## 5. Coordinate check — the correctness gate

The coord-check is the authoritative test that the μP implementation is correct.
It builds a model at several widths, runs a few optimizer steps on one fixed
batch, and records every `nn.Linear`/`nn.Embedding` output RMS per step.

```bash
# The gate (vary width only, fixed depth):
oplm sweep coord-check --config configs/my-run.yaml \
  --widths 128,256,512,1024 --depth 24 --base-width 512 --optimizer muon

# The control — should fan out / grow with width:
oplm sweep coord-check --config configs/my-run.yaml --no-mup \
  --widths 128,256,512,1024 --depth 24 --base-width 512

# The preset ray (co-scale depth with width; see Caveats):
oplm sweep coord-check --config configs/my-run.yaml \
  --scaling preset_ray --widths 512,768,1024 --base-width 512 --optimizer muon
```

The production gate uses base width 768 and is listed verbatim in
[LR_SWEEP.md](LR_SWEEP.md#parameterization-gates).

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

## 6. Production sweep harness

The executable `oplm sweep` command sequence, Slurm job generation, phase
artifacts, and selection rules live in [LR_SWEEP.md](LR_SWEEP.md) (job scripts
themselves are rendered by the general layer documented in
[SLURM.md](SLURM.md)). The harness uses one production YAML, keeps weight decay
fixed at `0.01`, and carries selected candidates through refine, replicate,
transfer, batch bridge, confirmation, and winner-only scaling phases.

---

## 7. Caveats

- **μP guarantees transfer across width at fixed depth.** OPLM's existing `1/√L`
  depth scaling (`residual_scaling="sqrt_num_layers"` + `1/√(2L)` residual-writer
  init) is a *separate, weaker* guarantee. Combined width+depth transfer along the
  preset ray (the constant-aspect-ratio scaling the presets ship) is validated
  **empirically** — coord-check `preset_ray` mode plus a confirmation run — not
  asserted from theory.
- **μP does not transfer across batch size, sequence length, or training horizon.**
  The production runbook therefore keeps the sequence-length distribution fixed,
  measures a proxy-to-production batch bridge, and uses a WSD schedule for the
  longer production horizon. The width coordinate-check oracle itself remains
  length-invariant because RoPE and `1/√head_dim` softmax scaling are driven by
  width, not sequence length.
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
- [LR_SWEEP.md](LR_SWEEP.md) — the phased production LR sweep runbook.
- [SLURM.md](SLURM.md) — the general Slurm job-generation layer the sweep builds on.
- [CONFIG.md](CONFIG.md#μp-maximal-update-parametrization) — the `mup_*` knobs.
- [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) — depth scaling and the readout.

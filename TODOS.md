# TODO: Implement μP Learning-Rate Transfer for OPLM (Muon-first)

Add **Maximal Update Parameterization (μP)** so a learning rate tuned on a small
pilot model transfers to much larger models without re-sweeping at every scale.
Production optimizer is **Muon**; AdamW stays supported. μP is **opt-in**
(`mup_enable=False` default) and a no-op at the base width, so all existing
behavior, checkpoints, and tests are preserved when off.

This work adds the **width** half of μP. The **depth** half already exists
(`residual_scaling="sqrt_num_layers"` gives `1/√L` residual branches and
`init_scale_output_projections` gives `1/√(2L)` residual-writer init); μP ties
the two together so transfer holds along the constant-aspect-ratio scaling ray.

---

## Reference: the μP recipe for OPLM

Symbols: `d0 = mup_base_width`, `m = hidden_size / d0` (width multiplier),
`L = num_hidden_layers`, `σ = initializer_range` (0.02), `head_dim = 64` (fixed).
Every entry is a no-op when `mup_enable=False` or `m = 1`.

| Component | Init std (μP) | Forward mult | Muon LR | AdamW LR |
|---|---|---|---|---|
| `embed_tokens` (input, tied to readout) | `σ` (unchanged) | input ×1 | — (AdamW) | `lr` ×1 |
| `q/k/v/o_proj`, `gate/up_proj`, `lm_head.dense` | `σ/√m` | — | `lr` (×1 via `original`) | `lr/m` |
| residual writers `o_proj`, `down_proj` | `σ/(√(2L)·√m)` | — | `lr` | `lr/m` |
| `lm_head.decoder` (readout; weight tied to embedding) | = embedding via tie | logits ×`mup_output_mult/m` | — | `lr` ×1 (as embedding) |
| norms, biases (1-D) | ones / zeros | — | — | `lr` ×1 |
| attention softmax | — | `1/√head_dim` **UNCHANGED** (head_dim fixed) | | |
| residual add | — | `1/√L` **UNCHANGED** (already implemented) | | |

Key reasons this is clean for OPLM:
- **`head_dim=64` is fixed across presets** (only the head *count* grows), so the
  classic μP "`1/d` attention scaling" change is **not needed** — `1/√64` is
  already width-invariant.
- **`lm_head.decoder.weight` is tied to the input embedding**
  (`_tied_weights_keys` in `modeling_oplm.py`), so the readout's width treatment
  collapses to a single **output-logit multiplier** instead of separate init/LR.
- **`torch.optim.Muon` with `adjust_lr_fn="original"`** multiplies per-param LR by
  `√(max(1, d_out/d_in))` — a function of *aspect ratio only*. Under constant
  aspect ratio this is **scale-invariant**, so the Muon base LR is **constant
  across width**. The current default `"match_rms_adamw"` uses
  `0.2·√(max(d_out,d_in))` which grows like `√width` and does **not** transfer;
  μP+Muon must use `"original"`.

**Owner-by-optimizer:**
- *Muon mode:* `q/k/v/o, gate/up/down` → Muon (constant LR). Only `lm_head.dense`
  stays on AdamW and takes `lr/m`. Do **not** move `lm_head.dense` to Muon in this
  pass (it would widen the blast radius into the non-μP path + `test_optim.py`);
  `lm_head.decoder` must stay on AdamW regardless.
- *AdamW-only mode:* all hidden matrices are on AdamW and take `lr/m`; embedding,
  norms, biases stay at `lr`.

**User-facing result:** with μP on, the value in `train.lr` is the *base* LR — tune
once at the proxy width and reuse the same number at every larger preset.

---

## Phase 1: Config Surface And Compatibility

- [ ] Add new `OplmConfig` fields in `src/oplm/model/configuration_oplm.py`, with
  defaults that make μP a no-op:
  - `mup_enable: bool = False`
  - `mup_base_width: int = 512`  (50M preset's `hidden_size`; the `m=1` reference)
  - `mup_output_mult: float = 1.0`  (tunable O(1) readout multiplier)
- [ ] Add a derived helper method on `OplmConfig`:
  - `mup_width_mult(self) -> float` → `hidden_size / mup_base_width` when
    `mup_enable`, else `1.0`. Single source of truth for `m`.
- [ ] Add validation (mirror existing `_resolve_derived_fields` / validation
  style): `mup_base_width >= 1`; `mup_output_mult > 0`.
- [ ] Confirm μP fields serialize into the HF model config (they live on
  `OplmConfig` so they ride in `config.json`/checkpoints automatically) and are
  restored on load — cover in the config round-trip test.
- [ ] Update `src/oplm/configs/model/base.yaml` and `docs/CONFIG.md` so the three
  `mup_*` knobs are documented in the existing config style (not hidden).
- [ ] Update config default / validation / CLI-override / YAML-typo-guard /
  save-load round-trip tests for the new fields (mirror existing `OplmConfig`
  coverage).

## Phase 2: Width-Aware Initialization

- [ ] Edit `OplmPreTrainedModel._init_weights` in
  `src/oplm/model/modeling_oplm.py` (lines ~79–114). `_init_weights(module)`
  receives **no name**, so role must come from per-module flags (reuse the
  existing `_is_residual_writer` pattern; do not parse names).
- [ ] In the `nn.Linear` branch, scale `module_std` by `1/√m` where
  `m = self.config.mup_width_mult()`:
  - base: `module_std = σ / √m`
  - residual writers keep the extra depth factor:
    `module_std = (σ / √(2·num_hidden_layers)) / √m`
  - All `nn.Linear` here are hidden matrices; the tied `lm_head.decoder` is
    overwritten by `tie_weights()` afterward, so scaling it is harmless.
- [ ] Leave the `nn.Embedding` branch unchanged (constant `σ`) — embeddings are μP
  input weights with width-independent init.
- [ ] Verify at `m=1` (base width or μP off) the produced std equals current init
  exactly (regression safety).

## Phase 3: Output-Logit Multiplier (Readout)

- [ ] In `OplmMLMHead.__init__` (`src/oplm/model/modeling_oplm.py` lines
  ~309–322), compute and store
  `self.output_mult = config.mup_output_mult / config.mup_width_mult()`
  (equals `1.0` when μP is off). Tag `self.decoder._is_readout = True` for
  coord-check labeling.
- [ ] In `OplmMLMHead.forward`, multiply the decoder output by `self.output_mult`:
  `return self.decoder(self.norm(self.act(self.dense(x)))) * self.output_mult`.
- [ ] Confirm logits stay `float()` (cast already happens in
  `OplmForMaskedLM.forward`) and the loss path is unaffected when `output_mult=1`.

## Phase 4: Per-Group Learning-Rate Scaling

- [ ] Use the canonical μP role/multiplier helper from `src/oplm/training/mup.py`
  (Phase 5) inside `src/oplm/training/optim.py`:
  - `mup_lr_multiplier(name, param, config) -> float` → `1/m` for hidden weight
    matrices (`q/k/v/o_proj`, `gate/up/down_proj`, `lm_head.dense.weight`), `1.0`
    for embedding / readout / norms / biases; `1.0` whenever `not config.mup_enable`.
- [ ] Extend `partition_optimizer_params` (lines ~34–87): keep the Muon
  eligibility rule (`param.ndim == 2 and not name.startswith("lm_head.")`).
  Additionally tag each AdamW-bound param with its `mup_lr_multiplier`, and group
  AdamW params by `(weight_decay_class, lr_multiplier)`.
- [ ] Update `_build_adamw_optimizer` / `build_optimizers` (lines ~90–160):
  - Emit one AdamW param-group dict per distinct `(wd, mult)` with
    `lr = cfg.lr * mult`, `weight_decay = wd`.
  - Keep Muon at `lr = cfg.lr` (constant); pass `adjust_lr_fn=cfg.muon_adjust_lr_fn`.
  - Mechanics to rely on (verified): `torch.optim.Muon` reads `group["lr"]` and
    `group["adjust_lr_fn"]` per group; `LambdaLR` captures each group's
    `initial_lr` into `base_lrs` and multiplies *each* by the same schedule λ, so
    heterogeneous per-group base LRs compose correctly with the existing scheduler.
- [ ] Add the **guard** in `build_optimizers`: raise `ValueError` when
  `model.config.mup_enable and cfg.optimizer == "muon" and
  cfg.muon_adjust_lr_fn == "match_rms_adamw"` (μP+Muon requires `"original"`).
- [ ] Leave `build_schedulers` unchanged (one `LambdaLR` per optimizer; per-group
  base LRs carry the width scaling, the schedule scales them uniformly).

## Phase 5: μP Core Module (`src/oplm/training/mup.py`, new)

- [ ] `mup_lr_multiplier(name, param, config) -> float` — canonical implementation
  imported by `optim.py` and tests.
- [ ] `coord_check(build_cfg_fn, widths, batch, steps, optimizer, seed) -> pandas.DataFrame`:
  - For each width, build a model with **depth and every other dim fixed**, varying
    only `hidden_size` (and derived `num_attention_heads`, keeping `head_dim=64`).
  - Register forward hooks on named submodules; record per-activation **RMS**
    (`x.pow(2).mean().sqrt()`), at each of `steps` optimizer steps on a fixed `batch`.
  - Return tidy frame with columns `(width, module, step, rms)`.
- [ ] `SweepMetricsCallback(path)` — a `TrainerCallback`
  (`src/oplm/training/callbacks.py` interface: `on_log`, `on_eval_end`,
  `on_train_end`). Capture EMA train loss from `on_log`, eval losses from
  `on_eval_end`, write `metrics.json`
  (`{final_train_loss, eval: {...}, lr, width, steps}`) on `on_train_end`. Needed
  because `trainer_state.json` carries no loss.
- [ ] `summarize_sweep(run_dirs) -> pandas.DataFrame` — load each run's
  `metrics.json` into a frame keyed by `(width, lr)`.
- [ ] `best_lr_per_width(df) -> dict[int, float]` — argmin-loss LR per width, plus a
  boolean **transfer verdict** (do the proxy widths agree on the argmin LR?).

## Phase 6: Coordinate-Check Script (`scripts/mup_coord_check.py`, new, `typer`)

- [ ] Thin CLI over `mup.coord_check`. Options: `--widths` (default
  `128,256,512,1024`), `--depth` (small, default 4), `--steps` (default 3),
  `--optimizer` (`muon|adamw`), `--data` (parquet), `--mup/--no-mup`, `--out`.
- [ ] Output a CSV of `(width, module, step, rms)` and a per-module RMS-vs-width
  plot.
- [ ] Document the pass/fail oracle: **with μP, each module's RMS stays ~flat
  across widths; with `--no-mup` it fans out.** This is the implementation's
  correctness gate — run before trusting any sweep.

## Phase 7: LR Sweep Harness (multi-GPU node)

- [ ] `scripts/mup_pilot_run.py` (new, `typer`): build one `OplmConfig` (μP on)
  from CLI overrides + `TrainConfig`, then run
  `Trainer(cfg, callbacks=[SweepMetricsCallback(out/"metrics.json")]).train()`.
  Single unit a sweep launches (in-process so the callback can capture loss).
  Honors `CUDA_VISIBLE_DEVICES`.
- [ ] `scripts/mup_sweep.py` (new, `typer`, orchestrator): inputs `--lrs` (grid),
  `--widths` (hidden sizes or presets), `--gpus N`, `--steps`, `--data`, `--out`.
  Fan out one `python -m scripts.mup_pilot_run` **subprocess per grid point**, each
  pinned via `CUDA_VISIBLE_DEVICES`, with a GPU-sized concurrency pool (semaphore).
  Use `subprocess.run` with arg lists (no `shell=True`).
- [ ] On completion, call `summarize_sweep` + `best_lr_per_width`; print best LR
  per width, emit a loss-vs-LR plot, print the **transfer verdict**.
- [ ] Each pilot run: `wandb_enabled=false`, fixed `seed`, identical `batch_size`,
  `warmup_steps`, `max_steps` across the grid (only `lr` and `hidden_size` vary).

## Phase 8: Defaults, Preset, Docs

- [ ] Add `src/oplm/configs/train/mup_muon.yaml`: `optimizer=muon`,
  `muon_adjust_lr_fn=original`, and (model side) `mup_enable=true`,
  `mup_base_width=512`.
- [ ] Write `docs/MUP.md`: the recipe table, the "tune-once-reuse-`train.lr`"
  workflow, coord-check + sweep commands, and the caveats below. Link from
  `docs/TRAIN.md`.

## Phase 9: Tests And Validation

- [ ] `tests/training/test_mup.py` (fast):
  - `mup_lr_multiplier` → `1/m` for hidden matrices, `1.0` for
    embedding/readout/norm/bias; `1.0` when `mup_enable=False`.
  - Param-group assembly (Muon mode): `lm_head.dense` group `lr ≈ cfg.lr/m`,
    embedding/norm groups `lr ≈ cfg.lr`, Muon `lr == cfg.lr`.
  - Param-group assembly (AdamW-only mode): all hidden matrices `lr ≈ cfg.lr/m`.
  - Build model at two widths: hidden-weight std ratio ≈ `√(d0/d)`, embedding std
    unchanged, `lm_head.output_mult ≈ mup_output_mult/m`.
  - Guard: `mup_enable + muon + match_rms_adamw` raises `ValueError`.
  - μP off ⇒ init, multipliers, param groups identical to current behavior.
- [ ] `tests/training/test_mup_coordcheck.py` (`@pytest.mark.slow`): run
  `coord_check` at widths `{128,256,512}`, depth 4, ~3 steps on the
  `training_parquet` fixture. Assert max per-module RMS ratio across widths within
  tolerance **with** μP and exceeding it **without** μP (control).
- [ ] `tests/training/test_mup_sweep.py` (`@pytest.mark.slow`): 2-point LR grid at
  one tiny width on the fixture; assert each run writes `metrics.json` and
  `best_lr_per_width` selects the lower-loss LR.
- [ ] Regression: extend `tests/training/test_optim.py` and
  `tests/training/test_e2e_optim.py` with a μP-Muon run asserting the
  two-optimizer path steps and the new groups exist; confirm μP-off is unchanged.
- [ ] Run focused tests (per repo convention, use `python -m`):
  - `python -m pytest tests/training/test_mup.py tests/training/test_optim.py tests/training/test_config.py tests/model/test_config.py tests/model/test_save_load.py`
- [ ] Run final gates:
  - `ruff format src/ tests/ scripts/`
  - `ruff check src/ tests/ scripts/`
  - `ty check src/`
  - `python -m pytest -m "not slow"`

## Phase 10: End-to-End Transfer Validation (run + sign-off)

- [ ] Confirm the real-corpus parquet path and the LR grid + step budget with the
  user before launching real runs.
- [ ] **Coord-check:**
  `python -m scripts.mup_coord_check --widths 128,256,512,1024 --optimizer muon`
  → verify flat per-module RMS curves (and that `--no-mup` fans out).
- [ ] **Pilot sweep:**
  `python -m scripts.mup_sweep --widths 256,512 --lrs <grid> --gpus <N> --data <corpus> --steps <budget>`
  → verify the loss-vs-LR minimum lands at the **same** `train.lr` for both proxy
  widths (transfer verdict = pass).
- [ ] **Confirmation run:** train one larger preset (e.g. 400M or 1B) reusing that
  `train.lr`; check the loss curve tracks the μTransfer expectation.
- [ ] Record results in the lab notebook per the lab-notebook skill.

## Caveats (document in `docs/MUP.md`)

- μP transfers across **width** (and **depth** via the existing `1/√L`), **not**
  across **batch size** or **training horizon**. If batch size changes with scale,
  apply `lr ∝ √(batch ratio)`; for horizon, prefer WSD schedules / weight-decay
  adjustment.
- `"original"` clips the Muon factor at 1 for `d_out < d_in` matrices (e.g.
  `down_proj`); still scale-invariant, so transfer holds — the coord-check
  validates this empirically.
- `OplmForSequenceClassification.classifier` is also a readout; out of scope for
  pretraining LR transfer (follow-up, leave fine-tuning heads alone).

## Assumptions

- [ ] `mup_base_width=512` (50M preset hidden_size) is the proxy/base where `m=1`;
  the pilot sweep tunes there and the result is reused unchanged at larger presets.
- [ ] Aspect ratio stays constant across presets (`hidden_size/num_hidden_layers=32`,
  `head_dim=64` fixed) — this is what makes the Muon `original` factor and the
  per-matrix init/LR scaling scale-invariant.
- [ ] μP and the LR transfer are validated on the **Muon** path; AdamW μP is
  implemented generically and gets a lighter baseline sweep only.

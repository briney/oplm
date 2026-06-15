# TODO: Implement μP Learning-Rate Transfer for OPLM (Muon-first)

Add **Maximal Update Parameterization (μP)** so a learning rate tuned on a small
pilot model transfers to much larger models without re-sweeping at every scale.
Production optimizer is **Muon**; AdamW stays supported. μP is **opt-in**
(`mup_enable=False` default) and a no-op at the base width, so all existing
behavior, checkpoints, and tests are preserved when off.

This work adds the **width** half of μP — the part μP theory guarantees: a base LR
tuned at one width transfers to other widths **at fixed depth**. OPLM already has
the **depth** half (`residual_scaling="sqrt_num_layers"` gives `1/√L` residual
branches and `init_scale_output_projections` gives `1/√(2L)` residual-writer init),
a separate and weaker guarantee than width μP. Combined width+depth transfer along
the constant-aspect-ratio preset ray is therefore validated **empirically**
(coord-check preset-ray mode + confirmation run), not asserted from μP theory.

---

## Reference: the μP recipe for OPLM

Symbols: `d0 = mup_base_width`, `L = num_hidden_layers`,
`σ = initializer_range` (0.02), `head_dim = 64` (fixed). The single scaling
quantity is the **per-matrix fan-in multiplier** `m_W = fan_in(W) / fan_in_base(W)`:
- for every matrix whose fan-in is `hidden_size` (`q/k/v/o_proj`, `gate/up_proj`,
  `lm_head.dense`, the readout `lm_head.decoder`): `m_W = m = hidden_size / d0`;
- for `down_proj` (fan-in is `intermediate_size`): `m_W = m_ffn = intermediate_size
  / i0`, where `i0` is the base intermediate size derived from `d0` via the same
  `round_up_to(int(8·d0/3), 256)` rounding — so FFN rounding is absorbed exactly,
  not approximated by `m`.
Every entry is a no-op when `mup_enable=False` or `m_W = 1`.

| Component | Init std (μP) | Forward mult | Muon LR | AdamW LR |
|---|---|---|---|---|
| `embed_tokens` (input) | `σ` (unchanged) | input ×1 | — (AdamW) | `lr` ×1 |
| `q/k/v/o_proj`, `gate/up_proj`, `lm_head.dense` | `σ/√m_W` | — | `lr` (×1 via `original`) | `lr/m_W` |
| residual writers `o_proj`, `down_proj` | `σ/(√(2L)·√m_W)` | — | `lr` | `lr/m_W` |
| `lm_head.decoder` (readout, **untied**) | `σ` (constant, no `1/√m`) | logits ×`mup_output_mult/m` | — (AdamW) | `lr` ×1 |
| norms, biases (1-D) | ones / zeros | — | — | `lr` ×1 |
| attention softmax | — | `1/√head_dim` **UNCHANGED** (head_dim fixed) | | |
| residual add | — | `1/√L` **UNCHANGED** (already implemented) | | |

Key reasons this is clean for OPLM:
- **`head_dim=64` is fixed across presets** (only the head *count* grows), so the
  classic μP "`1/d` attention scaling" change is **not needed** — `1/√64` is
  already width-invariant.
- **The readout is a standalone μP readout, not a tied weight.**
  `tie_word_embeddings` defaults to `False` (`configuration_oplm.py`, `base.yaml`),
  so `lm_head.decoder` is an independent matrix. Its μP treatment is constant init
  `σ` + an output-logit multiplier `mup_output_mult/m` applied to the **matmul path
  only** + AdamW LR `×1`. This reproduces the standard μP readout (logits `Θ(1/√m)`
  → small at init, correct update scale) without relying on weight tying, and still
  works if a user later sets `tie_word_embeddings=True` (the tied weight then rides
  the embedding's `×1` group; the multiplier still applies).
- **`torch.optim.Muon` with `adjust_lr_fn="original"`** multiplies per-param LR by
  `√(max(1, d_out/d_in))` — a function of *aspect ratio only*. The aspect ratio is
  approximately constant across presets (FFN rounding makes it only ~so), so the
  Muon base LR is **constant across width**; the residual error is validated
  empirically by the coord-check. The current default `"match_rms_adamw"` uses
  `0.2·√(max(d_out,d_in))` which grows like `√width` and does **not** transfer;
  μP+Muon must use `"original"`.

**Owner-by-optimizer:**
- *Muon mode:* `q/k/v/o, gate/up/down` → Muon (constant LR). Only `lm_head.dense`
  stays on AdamW and takes `lr/m`. Do **not** move `lm_head.dense` to Muon in this
  pass (it would widen the blast radius into the non-μP path + `test_optim.py`);
  `lm_head.decoder` (the readout) must stay on AdamW regardless, at `lr ×1`.
- *AdamW-only mode:* hidden matrices are on AdamW and take `lr/m_W` (`lr/m` for
  hidden-fan-in matrices, `lr/m_ffn` for `down_proj`); embedding, **readout
  `lm_head.decoder`**, norms, biases stay at `lr` (×1).

**User-facing result:** with μP on, the value in `train.lr` is the *base* LR — tune
once at the proxy width and reuse the same number at every larger preset.

---

## Phase 1: Config Surface And Compatibility

- [x] Add new `OplmConfig` fields in `src/oplm/model/configuration_oplm.py`, with
  defaults that make μP a no-op:
  - `mup_enable: bool = False`
  - `mup_base_width: int = 512`  (50M preset's `hidden_size`; the `m=1` reference)
  - `mup_output_mult: float = 1.0`  (tunable O(1) readout multiplier)
- [x] Add derived helper(s) on `OplmConfig`:
  - `mup_width_mult(self) -> float` → `hidden_size / mup_base_width` when
    `mup_enable`, else `1.0`. The hidden-fan-in multiplier `m` (used by the readout
    output multiplier and by every hidden-fan-in matrix).
  - expose the base FFN fan-in (e.g. `mup_base_intermediate_size`) by running the
    existing `round_up_to(int(8·mup_base_width/3), 256)` resolution (and the
    `relu2` variant) at the base width, so `down_proj`'s fan-in multiplier
    `m_ffn = intermediate_size / base_intermediate_size` is computed once and not
    re-derived in several places.
- [x] Add validation (mirror existing `_resolve_derived_fields` / validation
  style): `mup_base_width >= 1`; `mup_output_mult > 0`.
- [x] Confirm μP fields serialize into the HF model config (they live on
  `OplmConfig` so they ride in `config.json`/checkpoints automatically) and are
  restored on load — cover in the config round-trip test.
- [x] Update `src/oplm/configs/model/base.yaml` and `docs/CONFIG.md` so the three
  `mup_*` knobs are documented in the existing config style (not hidden).
- [x] Update config default / validation / CLI-override / YAML-typo-guard /
  save-load round-trip tests for the new fields (mirror existing `OplmConfig`
  coverage).

## Phase 2: Width-Aware Initialization

- [x] Edit `OplmPreTrainedModel._init_weights` in
  `src/oplm/model/modeling_oplm.py` (lines ~79–114). `_init_weights(module)`
  receives **no name**, so role must come from per-module flags (reuse the
  existing `_is_residual_writer` pattern; do not parse names). The `nn.Linear`
  branch must dispatch by **role flag** — not treat every Linear as a hidden matrix
  (`lm_head.decoder` and the `classifier` heads are also `nn.Linear`):
  - **readout** (`getattr(module, "_is_readout", False)`): constant `σ` — **no**
    `1/√m_W` (μP readouts have width-independent init). Covers `lm_head.decoder`
    and the fine-tuning `classifier` heads, so the "heads out of scope" caveat
    actually holds (they never get `σ/√m`).
  - **residual writer** (`_is_residual_writer` and `init_scale_output_projections`):
    `module_std = (σ / √(2·num_hidden_layers)) / √m_W`
  - **other hidden** (`q/k/v/o_proj`, `gate/up_proj`, `lm_head.dense`):
    `module_std = σ / √m_W`
- [x] Compute `m_W` per module from its **fan-in** via the canonical helper
  (`fan_in = module.in_features`), not a single global `m`: `m_W = fan_in /
  fan_in_base`, where `fan_in_base` is that role's fan-in at `d0 = mup_base_width`
  (= `d0` for hidden-fan-in matrices; = base `intermediate_size` for `down_proj`).
  Implemented as `OplmConfig.mup_fanin_mult(fan_in)` in the **model** package
  (not `training/mup.py`) so init stays `trust_remote_code`-safe and avoids a
  model→training import; the Phase-5 `mup_fanin_mult(module_or_param, config)`
  will delegate to it (single source of truth).
- [x] Tag readouts at construction: set `_is_readout = True` on `lm_head.decoder`
  (`OplmMLMHead.__init__`) and on the `classifier` linears in
  `OplmForSequenceClassification`/`OplmForTokenClassification`.
- [x] Leave the `nn.Embedding` branch unchanged (constant `σ`) — embeddings are μP
  input weights with width-independent init.
- [x] Verify at `m_W=1` (base width or μP off) the produced std equals current init
  exactly (regression safety), for every role including the readout and classifier.
  Confirmed: μP-off ≡ μP-on@base is bit-identical; width-scaled ratios match
  `1/√m` (hidden/`o_proj`/`lm_head.dense`), `1/√m_ffn` (`down_proj`), and `1.0`
  (embed/decoder/classifier readouts).

## Phase 3: Output-Logit Multiplier (Readout)

- [x] In `OplmMLMHead.__init__` (`src/oplm/model/modeling_oplm.py` lines
  ~309–322), compute and store
  `self.output_mult = config.mup_output_mult / config.mup_width_mult()`
  (equals `1.0` when μP is off; `mup_width_mult()` is the hidden-fan-in `m`, the
  decoder's fan-in). Tag `self.decoder._is_readout = True` — consumed by Phase 2
  init **and** coord-check labeling. (Tag done in Phase 2.)
- [x] In `OplmMLMHead.forward`, multiply the **input to the decoder** by
  `self.output_mult` — NOT the decoder output — so only the matmul path is scaled
  and `decoder.bias` (the layer has `bias=True`) is left untouched, matching
  `mup.MuReadout`:
  `return self.decoder(self.output_mult * self.norm(self.act(self.dense(x))))`.
- [x] Confirm logits stay `float()` (cast already happens in
  `OplmForMaskedLM.forward`) and the loss path is unaffected when `output_mult=1`.
  Verified: `output_mult` is 1.0 (off/base), 0.5 (m=2); matmul-only scaling leaves
  `decoder.bias` untouched; `output_mult=1` ⇒ forward bit-identical to `decoder(z)`.

## Phase 4: Per-Group Learning-Rate Scaling

- [x] Use the canonical μP role/multiplier helper from `src/oplm/training/mup.py`
  (Phase 5) inside `src/oplm/training/optim.py`:
  - `mup_lr_multiplier(name, param, config) -> float` → `1/m_W` (per-matrix
    fan-in multiplier) for hidden weight matrices: `1/m` for `q/k/v/o_proj`,
    `gate/up_proj`, `lm_head.dense.weight` (fan-in = hidden); `1/m_ffn` for
    `down_proj` (fan-in = intermediate). Returns `1.0` for embedding / **readout
    `lm_head.decoder`** / norms / biases, and `1.0` whenever `not config.mup_enable`.
- [x] Extend `partition_optimizer_params` (lines ~34–87): keep the Muon
  eligibility rule (`param.ndim == 2 and not name.startswith("lm_head.")`).
  Additionally tag each AdamW-bound param with its `mup_lr_multiplier`, and group
  AdamW params by `(weight_decay_class, lr_multiplier)`. New `AdamwParamGroup`
  dataclass `(params, weight_decay, lr_mult)`; `OptimizerParamGroups` now carries
  `adamw_groups` with `adamw_decay_params`/`adamw_no_decay_params` kept as derived
  properties so existing callers/tests are unchanged.
- [x] Update `_build_adamw_optimizer` / `build_optimizers` (lines ~90–160):
  - Emit one AdamW param-group dict per distinct `(wd, mult)` with
    `lr = cfg.lr * mult`, `weight_decay = wd`.
  - Keep Muon at `lr = cfg.lr` (constant); pass `adjust_lr_fn=cfg.muon_adjust_lr_fn`.
  - Mechanics to rely on (verified): `torch.optim.Muon` reads `group["lr"]` and
    `group["adjust_lr_fn"]` per group; `LambdaLR` captures each group's
    `initial_lr` into `base_lrs` and multiplies *each* by the same schedule λ, so
    heterogeneous per-group base LRs compose correctly with the existing scheduler.
    Verified: μP-Muon base_lrs `[lr/m (dense), lr (decoder), lr (no-decay)]`.
- [x] Add the **guard** in `build_optimizers`: raise `ValueError` when
  `model.config.mup_enable and cfg.optimizer == "muon" and
  cfg.muon_adjust_lr_fn == "match_rms_adamw"` (μP+Muon requires `"original"`).
- [x] Leave `build_schedulers` unchanged (one `LambdaLR` per optimizer; per-group
  base LRs carry the width scaling, the schedule scales them uniformly).

## Phase 5: μP Core Module (`src/oplm/training/mup.py`, new)

- [x] `mup_fanin_mult(module_or_param, config) -> float` — canonical per-matrix
  fan-in multiplier `m_W = fan_in / fan_in_base` (hidden-fan-in matrices → `m`;
  `down_proj` → `intermediate_size / base_intermediate_size`, base derived from
  `mup_base_width` via the existing `round_up_to` rounding). Single source of truth
  shared by init (Phase 2) and LR (Phase 4). Implemented as a thin training-side
  wrapper over `OplmConfig.mup_fanin_mult(fan_in)` (the model-package core, so init
  stays `trust_remote_code`-safe); returns `1.0` for non-2-D tensors.
- [x] `mup_lr_multiplier(name, param, config) -> float` — `1/mup_fanin_mult(...)`
  for hidden weight matrices, `1.0` for embedding / readout / norms / biases and
  whenever `not config.mup_enable`. Imported by `optim.py` and tests. Readouts
  detected by name suffix (`lm_head.decoder.weight`, `classifier.weight`), mirroring
  the Phase-2 `_is_readout` flags (init has no name; LR has no module flag).
- [x] `coord_check(build_cfg_fn, widths, batch, steps, optimizer, seed, scaling="width") -> pandas.DataFrame`:
  - `scaling="width"` (default, the μP correctness gate): for each width build a
    model with **depth and every other dim fixed**, varying only `hidden_size`
    (and derived `num_attention_heads`, keeping `head_dim=64`).
  - `scaling="preset_ray"`: scale `hidden_size` **and** `num_hidden_layers`
    together at the preset aspect ratio (`hidden/layers ≈ 32`), to validate the
    real scaling the user ships (Finding 4 — combined width+depth is empirical,
    not guaranteed by theory).
  - Register forward hooks on named submodules; record per-activation **RMS**
    (`x.pow(2).mean().sqrt()`) at steps `t = 0..steps` on a fixed `batch` (include
    `t=0`/init so the oracle can exclude it where appropriate).
  - Return tidy frame with columns `(width, module, step, rms)`.
  - Note: the per-width config geometry (incl. depth-vs-width for `preset_ray`) is
    owned by the caller's `build_cfg_fn(width)`; `scaling` is recorded in
    `df.attrs`. Hooks fire on every `nn.Linear`/`nn.Embedding`; the throwaway
    optimizer reuses `build_optimizers` and forces `adjust_lr_fn="original"`.
    Verified on CPU (muon + adamw): μP keeps q_proj RMS flat across widths
    (spread ≈1.05) while `--no-mup` fans out (≈3.4).
- [x] `SweepMetricsCallback(path)` — a `TrainerCallback`
  (`src/oplm/training/callbacks.py` interface: `on_log`, `on_eval_end`,
  `on_train_end`). Capture EMA train loss from `on_log`, eval losses from
  `on_eval_end`, write `metrics.json`
  (`{final_train_loss, eval: {...}, lr, width, steps}`) on `on_train_end`. Needed
  because `trainer_state.json` carries no loss.
- [x] `summarize_sweep(run_dirs) -> pandas.DataFrame` — load each run's
  `metrics.json` into a frame keyed by `(width, lr)`.
- [x] `best_lr_per_width(df) -> MupTransferResult` — return a dataclass
  `MupTransferResult(best_lr: dict[int, float], transferred: bool)` (argmin-loss LR
  per width plus a boolean **transfer verdict**: do the proxy widths agree on the
  argmin LR?). Not a bare `dict` — a `dict` cannot also carry the verdict.

## Phase 6: Coordinate-Check Script (`scripts/mup_coord_check.py`, new, `typer`)

- [x] Thin CLI over `mup.coord_check`. Options: `--widths` (default
  `128,256,512,1024`), `--depth` (small, default 4), `--steps` (default 3),
  `--optimizer` (`muon|adamw`), `--data` (parquet), `--mup/--no-mup`,
  `--scaling width|preset_ray` (default `width`), `--out`. Implemented as
  `typer.run`-style single command in a new `scripts/` package (so
  `python -m scripts.mup_coord_check` works); adds tuning knobs `--base-width`,
  `--output-mult`, `--n-seqs`, `--max-length`, `--lr`, `--seed`, `--device`
  (auto-detects CUDA). `head_dim=64` is held fixed; `width` keeps `--depth`,
  `preset_ray` co-scales depth at the 32:1 preset ratio (`--depth` ignored).
  When `--data` is omitted, falls back to a few built-in real protein sequences.
- [x] Output a CSV of `(width, module, step, rms)` and a per-module RMS-vs-width
  plot (one panel per step, one curve per module type, log-log). Files written
  to `--out/coord_check_{scaling}_{optimizer}_{mup|nomup}.{csv,png}` so μP and
  the control land side by side. Also prints a per-module RMS-growth table +
  verdict (a heuristic eyeball aid; the authoritative gate is the slow test).
- [x] Document the pass/fail oracle precisely (the implementation's correctness
  gate — run before trusting any sweep): in the module docstring, the `--help`,
  and a rich "oracle" panel printed after each run.
  - the oracle is **one-sided**: with μP, **no module's RMS grows with width** (the
    `--no-mup` control fans out / grows). State it as "does not grow", not
    "stays perfectly flat".
  - the **readout-logits module is allowed to shrink at init** (`Θ(1/√m)` by
    design): assess its cross-width behavior at steps `t ≥ 1` only, exclude `t=0`.
    The growth summary is taken at the final step (`t = steps ≥ 1`), so `t=0` is
    excluded by construction.
  - internal attention pre-softmax logits live inside SDPA and are **not** hooked
    as named-submodule outputs, so they need no separate exclusion.
  - Verified on CPU: μP-on holds (worst growth ≈1.15× over widths 128→512), the
    `--no-mup` control fans out (`down_proj` ≈7.2×, `o_proj` ≈5.4×), and the
    `preset_ray`/`adamw`/`--data` paths all run.

## Phase 7: LR Sweep Harness (multi-GPU node)

- [x] `scripts/mup_pilot_run.py` (new, `typer`): builds one μP-on root
  `OplmConfig` (model + `TrainConfig` + `DataConfig`) from CLI options, calls
  `oplm.train._bootstrap_training_environment()` (DeepSpeed opt-out + Triton
  cache, matching `oplm.train.main`), then runs
  `Trainer(cfg, callbacks=[SweepMetricsCallback(out/"metrics.json")]).train()`.
  `wandb_enabled=False`, eval left off (train-loss-only signal), `--no-save`
  default (`save_final=False`, `save_every=steps+1`) so a sweep leaves no
  checkpoints. Honors `CUDA_VISIBLE_DEVICES` (accelerate auto-detects). μP
  geometry comes from `scripts._mup_common` (shared with the coord-check, so the
  gate validates the same model the sweep trains).
- [x] `scripts/mup_sweep.py` (new, `typer`, orchestrator): inputs `--lrs`,
  `--widths` (hidden sizes; `--scaling width` fixes depth, `--scaling preset_ray`
  co-scales depth with width to validate the preset ray, Finding 4), `--gpus N`,
  `--steps`, `--data`, `--out`. Fans one `python -m scripts.mup_pilot_run`
  subprocess per `(width, lr)` grid point via `subprocess.run` with arg lists (no
  `shell=True`), pinned by a per-subprocess `CUDA_VISIBLE_DEVICES`. The GPU-sized
  pool is a `ThreadPoolExecutor(max_workers=gpus)` over a `queue.Queue` of GPU ids
  (the queue is the semaphore: each task borrows an id, runs, returns it). Failed
  runs are collected (stderr tail surfaced), not fatal.
- [x] On completion, calls `summarize_sweep` + `best_lr_per_width`; prints a
  per-`(width, lr)` loss table, the argmin LR per width, the **transfer verdict**
  (`result.transferred`), and writes a loss-vs-LR plot (`sweep_loss_vs_lr.png`,
  one line per width, argmin starred).
- [x] Each pilot run: `wandb_enabled=false`, and the orchestrator passes an
  identical `seed`, `batch_size`, `warmup_steps`, and `steps` to every grid point
  (a shared `common` argv); only `--width` and `--lr` vary (plus depth in
  preset-ray mode, derived from width).
- [x] Refactor: shared `scripts/_mup_common.py` (head_dim=64, preset aspect
  ratio, `Scaling`/`Optimizer` enums, `parse_widths`/`parse_floats`,
  `num_layers_for`) is the single source of width→geometry for both the
  coord-check and the sweep. Verified on CPU/GPU: pilot trains and writes
  `metrics.json`; a 2×2 sweep spawns 4 pinned subprocesses, summarizes, prints
  the verdict, and emits the plot; coord-check regression still holds.

## Phase 8: Defaults, Preset, Docs

- [ ] Add `src/oplm/configs/train/mup_muon.yaml`: `optimizer=muon`,
  `muon_adjust_lr_fn=original`, and (model side) `mup_enable=true`,
  `mup_base_width=512`.
- [ ] Write `docs/MUP.md`: the recipe table, the "tune-once-reuse-`train.lr`"
  workflow, coord-check + sweep commands, and the caveats below. Link from
  `docs/TRAIN.md`.

## Phase 9: Tests And Validation

- [ ] `tests/training/test_mup.py` (fast):
  - `mup_lr_multiplier` → `1/m` for hidden matrices with hidden fan-in, `1/m_ffn`
    for `down_proj` (intermediate fan-in), `1.0` for embedding/**readout**/norm/bias;
    `1.0` when `mup_enable=False`.
  - Param-group assembly (Muon mode): `lm_head.dense` group `lr ≈ cfg.lr/m`,
    `lm_head.decoder` (readout) group `lr ≈ cfg.lr` (×1), embedding/norm groups
    `lr ≈ cfg.lr`, Muon `lr == cfg.lr`.
  - Param-group assembly (AdamW-only mode): hidden matrices `lr ≈ cfg.lr/m`, and
    `down_proj` `lr ≈ cfg.lr/m_ffn` (a distinct group using the intermediate fan-in).
  - Build model at two widths: hidden-weight std ratio ≈ `√(d0/d)`, `down_proj`
    std ratio ≈ `√(i0/i)` (intermediate, not hidden), embedding std unchanged,
    **readout `lm_head.decoder` std unchanged** (constant `σ`, not `σ/√m`),
    **classifier-head std unchanged**, `lm_head.output_mult ≈ mup_output_mult/m`.
  - Readout bias: with `output_mult ≠ 1`, only the matmul path is scaled —
    `lm_head.decoder.bias` is unaffected by `m` (matmul-path scaling check).
  - Guard: `mup_enable + muon + match_rms_adamw` raises `ValueError`.
  - μP off ⇒ init, multipliers, param groups identical to current behavior.
- [ ] `tests/training/test_mup_coordcheck.py` (`@pytest.mark.slow`): run
  `coord_check` at widths `{128,256,512}`, depth 4, ~3 steps on the
  `training_parquet` fixture. Assert the **one-sided** oracle — per-module RMS does
  not grow with width within tolerance **with** μP, and the `--no-mup` control
  exceeds it. Exclude the readout-logits module at `t=0` (allowed to shrink at
  init); assess it at `t ≥ 1`. Add a `scaling="preset_ray"` smoke case.
- [ ] `tests/training/test_mup_sweep.py` (`@pytest.mark.slow`): 2-point LR grid at
  one tiny width on the fixture; assert each run writes `metrics.json` and
  `best_lr_per_width(...).best_lr` selects the lower-loss LR (and `.transferred`
  is populated).
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
- [ ] **Coord-check (width, the gate):**
  `python -m scripts.mup_coord_check --widths 128,256,512,1024 --optimizer muon`
  → verify per-module RMS does **not grow** with width (and that `--no-mup` fans
  out); the readout may shrink at init (assessed at `t ≥ 1`).
- [ ] **Coord-check (preset ray):**
  `python -m scripts.mup_coord_check --scaling preset_ray --widths 256,512,1024 --optimizer muon`
  → empirically validate combined width+depth scaling (Finding 4).
- [ ] **Pilot sweep:**
  `python -m scripts.mup_sweep --widths 256,512 --lrs <grid> --gpus <N> --data <corpus> --steps <budget>`
  → verify the loss-vs-LR minimum lands at the **same** `train.lr` for both proxy
  widths (`MupTransferResult.transferred == True`).
- [ ] **Confirmation run:** train one larger preset (e.g. 400M or 1B) reusing that
  `train.lr`; check the loss curve tracks the μTransfer expectation.
- [ ] Record results in the lab notebook per the lab-notebook skill.

## Caveats (document in `docs/MUP.md`)

- μP guarantees transfer across **width at fixed depth**. The existing `1/√L`
  depth scaling is a separate, weaker guarantee; combined width+depth transfer
  along the preset ray is validated **empirically** (coord-check preset-ray mode +
  confirmation run), not asserted from theory. μP does **not** transfer across
  **batch size** or **training horizon** — if batch size changes with scale, apply
  `lr ∝ √(batch ratio)`; for horizon, prefer WSD schedules / weight-decay
  adjustment.
- `"original"` clips the Muon factor at 1 for `d_out < d_in` matrices (e.g.
  `down_proj`); still ~scale-invariant (modulo FFN rounding), so transfer holds —
  the coord-check validates this empirically.
- `OplmForSequenceClassification`/`OplmForTokenClassification` `classifier` heads
  are readouts and are tagged `_is_readout=True`, so init treats them as readouts
  (constant `σ`, never `σ/√m`); pretraining LR transfer for fine-tuning heads is
  out of scope (follow-up).

## Assumptions

- [ ] `mup_base_width=512` (50M preset hidden_size) is the proxy/base where `m=1`;
  the pilot sweep tunes there and the result is reused unchanged at larger presets.
- [ ] Per-matrix fan-in multipliers (`m` for hidden-fan-in matrices, `m_ffn` for
  `down_proj`) make the init/LR scaling exact under FFN rounding — no reliance on a
  single global `m`. `head_dim=64` is fixed (only head count grows), so attention
  softmax scaling is width-invariant. The Muon `original` factor (`√(d_out/d_in)`)
  uses true shapes; its cross-preset constancy is only **approximate** because the
  FFN aspect ratio varies with `intermediate_size` rounding (e.g. 3.0 / 2.667 /
  2.75 at hidden 512 / 768 / 1024), and is validated empirically by the coord-check.
- [ ] μP and the LR transfer are validated on the **Muon** path; AdamW μP is
  implemented generically and gets a lighter baseline sweep only.

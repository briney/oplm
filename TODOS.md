# TODO: Promote the best-performing recipe to the default config

muP + Muon is already the default optimizer recipe. This change promotes the four
remaining best-performing architecture features to the **run defaults** in
`configs/model/base.yaml`, so production runs get the winning recipe with no extra
flags:

1. `norm_strategy` → **sandwich**
2. `attn_output_gate` → **sigmoid**
3. `value_residual` → **learnable**
4. Canon depthwise convs → **enabled at `[A, B, C, D]`, kernel size 7**

Design invariants:
- **YAML carries production defaults; the dataclass stays a conservative library
  fallback.** Only `configs/model/base.yaml` changes — `OplmConfig.__init__` keeps
  the vanilla values so bare `OplmConfig()` / `from_pretrained` / old checkpoints
  stay backward-compatible (mirrors how `mup_enable` was promoted).
- A single combined escape hatch, `configs/train/vanilla_esm-c.yaml`, turns μP off
  **and** reverts the architecture (replaces the μP-only `baseline_adamw.yaml`).

The four features are already implemented and test-covered; this change only moves
defaults — no model code changes.

---

## Phase 1 — Flip model run-defaults (`src/oplm/configs/model/base.yaml`)

- [x] `norm_strategy`: `pre` → `sandwich`
- [x] `attn_output_gate`: `none` → `sigmoid`
- [x] `value_residual`: `none` → `learnable`
- [x] `canon_enabled`: `false` → `true`
- [x] `canon_positions`: `[]` → `[A, B, C, D]`
- [x] `canon_kernel_sizes`: `4` → `7`
- [x] Point the μP-disable comment at `configs/train/vanilla_esm-c.yaml`
- [x] Leave `canon_residual: true`, `canon_activation: none`,
      `value_residual_lambda_init: 0.5` unchanged

## Phase 2 — Replace the baseline overlay with `vanilla_esm-c.yaml`

- [x] Delete `src/oplm/configs/train/baseline_adamw.yaml`
- [x] Create `src/oplm/configs/train/vanilla_esm-c.yaml` (combined `train:` + `model:`
      overlay: AdamW, μP off, and the four architecture features reset to vanilla)
- [x] Repoint the disable-path comments in `configs/train/base.yaml`

## Phase 3 — Update tests (`tests/training/test_config.py`)

- [x] Rename `test_baseline_adamw_overlay_disables_mup` →
      `test_vanilla_esmc_overlay_restores_conventional_recipe`; load the new overlay
      and assert μP off **and** architecture reverted (`norm_strategy=pre`,
      `attn_output_gate=none`, `value_residual=none`, `canon_enabled=False`)
- [x] `test_dataclass_fallbacks_are_conservative`: also assert the four architecture
      fields stay vanilla on bare `OplmModelConfig()`
- [x] `test_canon_residual_cli_override_applies`: `canon_enabled` now `True` by default
- [x] `test_all_presets_resolve_golden_canon_defaults`: Canon on, `[A,B,C,D]`,
      `[7]*num_hidden_layers`, residual on, activation none
- [x] Replace the old `paper_exact` Canon test with
      `test_canon_default_run_resolves_golden_encoder_config` (default run →
      sandwich + Canon `[A,B,C,D]` + `k=7`)
- [x] No change needed to `tests/model/test_toggles.py` (already exercises the
      combined recipe) or the μP coord-check (builds `OplmConfig` directly)
- [x] Add `tests/test_e2e_default_recipe.py`: a slow e2e test that builds a tiny model
      from `base.yaml` via `load_config` (size overridden only), asserts the production
      defaults are active, and drives the real `Trainer` through a few train steps + an
      eval — guards that the default recipe trains end-to-end. `tiny_train_cfg` builds
      from the dataclass (vanilla), so this is the only test covering the YAML recipe.

## Phase 4 — Update docs and references

- [x] `docs/CONFIG.md`: Default column for the six fields + Canon section header +
      overlay reference + the conservative-fallbacks note
- [x] `docs/MUP.md`, `docs/TRAIN.md`: rename overlay + example commands; reword
      "AdamW baseline" → "vanilla ESM-C recipe"
- [x] `README.md`: rename overlay + recipe description
- [x] `docs/OVERVIEW.md`: reconciled §1 (library-default vs production-default split;
      moved value residual / output gating out of the "Removed" list into toggles;
      added the two features to the toggle list and feature table) and qualified the
      output-gate / value-residual per-field lines
- [x] `docs/MODEL_ARCHITECTURE.md`: clarified the Canon default kernel (library `4`,
      production `7`)

## Phase 5 — Verification

- [x] `python -m pytest tests/training/test_config.py tests/model/test_toggles.py -q` — 585 passed
- [x] `python -m pytest tests/training/test_mup_coordcheck.py -q` — 2 passed (μP unaffected)
- [x] Default-load smoke: `load_config([])` resolves the six new values
- [x] Escape-hatch smoke: `--config vanilla_esm-c.yaml` → μP off + all four features off
- [x] End-to-end train+eval under the new default recipe (sandwich + sigmoid gate +
      learnable value residual + Canon ABCD k=7, μP + Muon) on the real data fixture —
      no shape / optimizer-grouping errors. Captured permanently as
      `tests/test_e2e_default_recipe.py` (passes on both the default device and forced
      CPU); `tests/test_e2e_lifecycle.py` stays on the dataclass/vanilla helper

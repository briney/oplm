# Architecture

How the model works: [../architecture.md](../architecture.md). Sizes are the presets in
`src/oplm/configs/model/presets/`; defaults are in `src/oplm/configs/model/base.yaml`.

| Choice | Where the value lives | Evidence |
|---|---|---|
| OPLM architecture against an ESM-C-style baseline | `src/oplm/configs/model/base.yaml` | [oplm-vs-esmc-recipe](evidence/oplm-vs-esmc-recipe/README.md) |
| Canon layers are on | `model.canon_enabled` in `src/oplm/configs/model/base.yaml` | [canon-ablation](evidence/canon-ablation/README.md) |
| Shared-layer looping layout | `src/oplm/configs/model/base.yaml` | to backfill (looping probe: stack, interleave, mid) |
| Residual gate is off | `model.residual_gate` in `src/oplm/configs/model/base.yaml` | [residual-gate](evidence/residual-gate/README.md) |
| Residual scaling and μP hooks | `src/oplm/configs/model/base.yaml` | to backfill (coordinate check, [../mup-lr.md](../mup-lr.md)) |

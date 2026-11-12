# Architecture

How the model works: [../architecture.md](../architecture.md). Sizes are the presets in
`src/oplm/configs/model/presets/`; defaults are in `src/oplm/configs/model/base.yaml`.

| Choice | Where the value lives | Evidence |
|---|---|---|
| Canon layers are on | `src/oplm/configs/model/base.yaml` | to backfill (no-Canon ablation, W&B and cluster logs) |
| Shared-layer looping layout | `src/oplm/configs/model/base.yaml` | to backfill (looping probe: stack, interleave, mid) |
| Residual scaling and μP hooks | `src/oplm/configs/model/base.yaml` | to backfill (coordinate check, [../mup-lr.md](../mup-lr.md)) |

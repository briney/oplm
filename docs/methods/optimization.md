# Optimization

How the optimizer and schedule work: [../training.md](../training.md), *Optimizer and schedule*.
How the learning rate was tuned: [../mup-lr.md](../mup-lr.md). Defaults are in
`src/oplm/configs/train/base.yaml`; the AdamW comparison recipe is
`src/oplm/configs/train/esm-c.yaml`.

> **Pending:** the production training config (horizon and learning rate) is not committed yet (waiting on #70).

| Choice | Where the value lives | Evidence |
|---|---|---|
| Muon for matrices, AdamW for the rest | `train.optimizer` | [oplm-vs-esmc-recipe](evidence/oplm-vs-esmc-recipe/README.md) (confounded with architecture; isolated ablation is #117) |
| Base learning rate | `train.lr` | to backfill (μP funnel; long-run campaign #55, docs PR #71) |
| Learning rate vs training length | `train.lr` | [lr-vs-training-length](evidence/lr-vs-training-length/README.md) |
| AdamW-side LR multiplier | `train.adamw_lr_mult` | to backfill |
| Depth LR correction | `train.mup_depth_lr_exponent` | to backfill (fixed-width depth ray) |
| Decay toward init for gains, gates and biases | `train.weight_decay_to_init` | to backfill (decay knobs PR #40) |
| Warmup, stable, linear decay schedule | `train.scheduler` | [decay-shape](evidence/decay-shape/README.md) |
| Tokens per optimizer step | `train.tokens_per_step` | to backfill |
| Stable and decay phase lengths | `train.stable_steps`, `train.max_steps` | to backfill (#70) |

# Optimization

How the optimizer and schedule work: [../training.md](../training.md), *Optimizer and schedule*.
How the learning rate was tuned: [../mup-lr.md](../mup-lr.md). Defaults are in
`src/oplm/configs/train/base.yaml`; the AdamW comparison recipe is
`src/oplm/configs/train/esm-c.yaml`.

> **Pending:** the production training config (horizon and learning rate) is not committed yet (waiting on #70).

| Choice | Where the value lives | Evidence |
|---|---|---|
| Muon for matrices, AdamW for the rest | `train.optimizer` | [oplm-vs-esmc-recipe](evidence/oplm-vs-esmc-recipe/README.md) (confounded with architecture; isolated ablation is #117) |
| Base learning rate | `train.lr` | [base-lr-funnel](evidence/base-lr-funnel/README.md) |
| Learning rate vs training length | `train.lr` | [lr-vs-training-length](evidence/lr-vs-training-length/README.md) |
| AdamW-side LR multiplier | `train.adamw_lr_mult` | [adamw-lr-multiplier](evidence/adamw-lr-multiplier/README.md) (runs predate the current recipe; matched rerun is #118) |
| Depth LR correction | `train.mup_depth_lr_exponent` | Derived, not swept: [../mup-lr.md](../mup-lr.md), *Depth correction*. Checked only by the 800M cells in [base-lr-funnel](evidence/base-lr-funnel/README.md); no run has it off |
| Decay toward init for gains, gates and biases | `train.weight_decay_to_init` | [weight-decay-to-init](evidence/weight-decay-to-init/README.md) |
| Warmup, stable, linear decay schedule | `train.scheduler` | [decay-shape](evidence/decay-shape/README.md) |
| Tokens per optimizer step | `train.tokens_per_step` in `configs/scaling.yaml` | Not varied in any run. The learning rate was tuned at this value ([base-lr-funnel](evidence/base-lr-funnel/README.md)) and does not carry to another |
| Stable and decay phase lengths | `train.stable_steps`, `train.max_steps` | [training-length-and-overfitting](evidence/training-length-and-overfitting/README.md) (the lengths themselves are pending #70) |

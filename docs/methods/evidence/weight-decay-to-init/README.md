# Gains, gates and biases decay toward their initial values at 0.1

**Status:** provisional

## Question

`train.weight_decay_to_init` pulls norm gains, residual gates and biases back toward their initial
values, to stop the post-norm gains drifting upward on long runs. Does raising it from 0.01 to 0.1
cost any loss?

## Arms

170M, no residual gate, lr 0.0063, the learning-rate funnel protocol (20k steps constant, 4k decay
to step 24k; see [base-lr-funnel](../base-lr-funnel/README.md)). Only
`train.weight_decay_to_init` differs: 0.01 against 0.1.

## Result

No measurable cost: the two arms are within 0.0005 held-out loss, well inside the seed range
(0.0016).

| weight_decay_to_init | step 20k | step 24k, after decay |
|---|---|---|
| 0.01 | 2.1262 | 2.0790 |
| 0.1 | 2.1267 | 2.0794 |

## Runs

`results.csv` lists the four cells; their configs are in `../base-lr-funnel/configs/`. The
mechanism is described in [training.md](../../../training.md), *Optimizer and schedule*.

## Caveats

- **This record shows only the cost, not the benefit.** The evidence that the gains drift without
  the stronger decay, and that 0.1 bounds them, comes from long-run weight diagnostics that have
  not been assembled here yet.
- One seed, 24k steps, held-out loss only.

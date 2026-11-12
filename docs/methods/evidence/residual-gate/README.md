# The residual gate is off

**Status:** settled

## Question

Does the learnable per-channel residual gate (`model.residual_gate: channel`) improve the model
over no gate (`none`)?

## Arms

The decayed cells of the learning-rate funnel, with and without the gate; everything else is equal
(see [base-lr-funnel](../base-lr-funnel/README.md) for the protocol). 170M at three learning
rates, 800M at one.

## Result

The gate makes no measurable difference: every pair is within 0.0015 held-out loss, inside the
seed range of the gated cells (0.0016), and the sign is not consistent. It is off.

| size | lr | gate (seeds 42 / 43 / 44) | no gate |
|---|---|---|---|
| 170M | 0.004 | 2.0811 / 2.0809 / 2.0816 | 2.0803 |
| 170M | 0.0063 | 2.0775 / 2.0791 / 2.0791 | 2.0790 |
| 170M | 0.01 | 2.0795 | 2.0790 |
| 800M | 0.0063 | 2.0306 | 2.0297 |

## Runs

`results.csv` lists these cells; their configs are in `../base-lr-funnel/configs/`.

## Caveats

- Short runs (24k steps at 170M, 12k at 800M); the no-gate cells have one seed.
- Held-out loss only; no structure evals on these cells.

# The best learning rate falls as training gets longer

**Status:** provisional

## Question

The production run is far longer than any run a learning-rate sweep can afford. How does the best
base learning rate move with total training length, so that it can be extrapolated to the
production horizon?

## Arms

170M model, Muon, 4,194,304 tokens per step, commit 6e8b277, at three learning rates (0.0016,
0.004, 0.0063). Each learning rate has one long run (2k warmup, constant to step 250k, linear decay
to 300k) and two branches off it that decay earlier: from step 50k to 60k and from step 100k to
120k. The decay is one sixth of the total length in every arm.

## Result

At every length 0.004 is the best of the three, but the optimum moves down as training lengthens:
0.0016 closes from 0.010 behind at 60k to 0.002 behind at 300k, while 0.0063 falls from 0.001
behind to 0.003 behind.

Held-out uniref70 loss after the decay:

| total steps | lr 0.0016 | lr 0.004 | lr 0.0063 |
|---|---|---|---|
| 60k | 2.0355 | 2.0256 | 2.0265 |
| 120k | 1.9974 | 1.9910 | 1.9927 |
| 300k | 1.9537 | 1.9516 | 1.9549 |

A parabola in log learning rate through each row puts the optimum near 0.0045, 0.0040 and 0.0030
at 60k, 120k and 300k steps. The fit, its extension with the 24k-step sweep point, and the
extrapolation to the production horizon are in [mup-lr.md](../../../mup-lr.md), *LR vs training
length*.

## Runs

- W&B project `oplm-170M-long`; IDs, Slurm jobs and resolved configs are in `results.csv` and
  `configs/`. Run directories: `~/projects/oplm/mup_redo/runs/170M-lr*-long*` on the cluster.
- Write-up: issue #55, docs PR #71. The production horizon and learning rate: issue #70.

## Caveats

- **Three learning rates per length,** so each parabola is exactly determined and has no error
  estimate. Moving one point by 0.001 loss moves the fitted optimum by about 10%.
- **The basin is flat.** At 300k steps the three arms span 0.003 loss across a fourfold range of
  learning rate.
- **Production is an extrapolation** to at least five times the longest run measured here, at
  one model size and one seed.
- **Held-out loss only.** These runs logged no structure evals.
- **Rank after the decay.** Before the decay the lowest learning rate leads (it gains least from
  the decay). The pre-decay values are in W&B and are not exported here.

# The AdamW-side learning rate is one third of the base learning rate

**Status:** needs matched rerun

## Question

Under Muon, the parameters Muon does not handle (embeddings, head, norm gains, biases, Canon
kernels) are trained by AdamW at `train.lr × train.adamw_lr_mult`. At a multiplier of 1 the
AdamW-side weights grew steadily over a long run. How far should the multiplier be cut: to one
third or to one tenth?

## Arms

170M, Muon, lr 0.0035, seed 42, 4 nodes. Only `train.adamw_lr_mult` differs: 0.33 (**div3**)
against 0.1 (**div10**).

- Each arm ran to step 80k, then was branched from its own step-80k checkpoint to step 800k
  (the **-800k** arms) with the multiplier unchanged.
- The learning rate is constant to step 795k and decays linearly to zero over the last 5k steps.
- The batch is 8,192 sequences at 512 tokens of context (`train.batch_size` 256 per GPU), not the
  token-mode batch the recipe uses now.

## Result

One tenth is ahead early and one third is ahead late; the gaps are 0.001–0.005 held-out loss.
One third was kept. Mean of div10 minus div3 over the evaluations in each window (negative
favors div10):

| steps | uniref70 | deepclust70_gte2 | deepclust70_singletons | omg70 | casp14 P@L |
|---|---|---|---|---|---|
| 85k–300k | −0.0027 | −0.0011 | −0.0022 | −0.0003 | −0.0002 |
| 300k–500k | −0.0011 | +0.0004 | −0.0005 | +0.0004 | −0.0037 |
| 500k–795k | +0.0016 | +0.0036 | +0.0011 | +0.0025 | −0.0023 |

At step 800k:

| arm | uniref70 | deepclust70_gte2 | deepclust70_singletons | omg70 | casp14 P@L |
|---|---|---|---|---|---|
| div3 | 1.9982 | 2.0330 | 2.1176 | 2.1975 | 0.1545 |
| div10 | 1.9999 | 2.0381 | 2.1209 | 2.2008 | 0.1453 |

`curves.csv` has every evaluation (every 5k steps) for all four runs.

## Runs

- W&B `thebrineylab/oplm-scaling`. Cluster: `~/projects/oplm/scaling/170M-v2-adamdiv{3,10}/`
  (first 80k steps; the 800k output directories are no longer on disk).
- `configs/` holds the resolved config of each first leg. The 800k branches used the same config
  with `train.branch_from` set to the step-80k checkpoint, `train.max_steps=800000` and
  `train.stable_steps=795000`.
- How the multiplier is applied: [mup-lr.md](../../../mup-lr.md).

## Caveats

- **These runs predate the current recipe.** Against today's defaults they use lr 0.0035, output
  multiplier 0.5, the residual gate on, `weight_decay_to_init` 0.01, a sequence-count batch at
  512 context, and a 25:75 uniref70:deepclust70 mix read from deepclust70 shards that were still
  in cluster order. The multiplier was carried into the learning-rate funnel unchanged and not
  retuned at 4,194,304 tokens per step.
- **No multiplier of 1 arm on this recipe.** The earlier run at 1 (W&B `426y3e7m`) differs in
  several other settings, so the gain from cutting the multiplier at all is not isolated here.
- **No real decay.** The comparison is at constant learning rate; other decisions in this
  directory changed ranking after a decay.
- **One seed.** The late gaps on uniref70 (0.002) are about the seed range measured at 24k steps
  (0.0016); the gaps on deepclust70_gte2 and omg70 are two to three times that.
- casp14 P@L is noisy between evaluations (±0.01) and does not separate the arms.
- The weight-growth measurements that motivated cutting the multiplier are not assembled here.

# Longer training keeps paying, and repeating the data does not overfit

**Status:** provisional

## Question

The production run passes over uniref70 many times. Does that overfit, and does model quality
still improve at the lengths production will reach? This sets how long the stable and decay
phases should be.

## Arms

The 1/35 proxy: 170M models trained on a random 1/35 of each source, so the data repeats 35 times
sooner than in production and one proxy step stands for about 35 production steps of data
exposure. 4,194,304 tokens per step.

- **Stable runs** to step 252k at constant learning rate, with uniref70 at row fraction 0.05, for
  the OPLM recipe (Muon, lr 0.003) and the ESM-C recipe (AdamW, lr 0.0004). Held-out sets and
  train-probe sets (rows that are in the proxy training data) are scored every 1k steps and the
  structure sets every 7k.
- **Decays** branched from the stable runs at steps 56k, 140k and 252k, with uniref70 at 25% of
  tokens. These are the runs of [oplm-vs-esmc-recipe](../oplm-vs-esmc-recipe/README.md).

## Result

Through 252k proxy steps, held-out loss is still falling and contact precision still rising for
both recipes. The gap between held-out and training-set loss widens early and then levels off. A
decay from a later branch point is better on every metric.

OPLM stable run (`curves.csv` has every eval point for both recipes):

| step | uniref70 gte2 held-out | uniref70 gte2 train probe | gap | deepclust70 gte2 held-out | deepclust70 train probe | gap | P@L pdb40 |
|---|---|---|---|---|---|---|---|
| 28k | 2.1827 | 1.9169 | 0.266 | 2.1866 | 2.0605 | 0.126 | .2971 |
| 56k | 2.1522 | 1.8663 | 0.286 | 2.1604 | 2.0277 | 0.133 | .3284 |
| 84k | 2.1383 | 1.8459 | 0.292 | 2.1474 | 2.0135 | 0.134 | .3391 |
| 140k | 2.1221 | 1.8165 | 0.306 | 2.1374 | 1.9959 | 0.142 | .3517 |
| 196k | 2.1130 | 1.7976 | 0.315 | 2.1296 | 1.9842 | 0.145 | .3622 |
| 252k | 2.1056 | 1.7911 | 0.315 | 2.1233 | 1.9764 | 0.147 | .3671 |

OPLM decays by branch point (end of decay):

| branch step | uniref70 gte2 loss | P@L pdb40 | P@L cameo | P@L casp | P@L pdb40_2048 |
|---|---|---|---|---|---|
| 56k | 2.0839 | .3556 | .3745 | .3037 | .3463 |
| 140k | 2.0364 | .3754 | .4009 | .3267 | .3734 |
| 252k | 2.0076 | .3959 | .4177 | .3370 | .3905 |

Each step up in branch point adds about 0.02 P@L on pdb40, with no sign of saturation. The decay
itself is worth more than the same number of stable steps: the 140k branch ends at step 175k with
P@L .3754, above the stable run at step 196k (.3622) and at 252k (.3671).

## Runs

- W&B project `oplm-170M-overfit`; stable runs `taqdsivi` (OPLM) and `u9bc9lg9` (ESM-C).
  `results.csv` lists the stable runs and the decays; their resolved configs are in
  `../oplm-vs-esmc-recipe/configs/`.
- `curves.csv` was exported from W&B history for the two stable runs: every `eval/*/loss` and
  `eval/*/precision_at_L` point.
- Cluster: `~/projects/oplm/mup_redo/runs/overfit-1of35*/`, driver `overfit.sh`. Discussion:
  issue #58; production horizon: issue #70.

## Caveats

- **The gap's level is not meaningful, only its trend.** The train-probe and held-out sets are
  different sequences, and the held-out sets were screened to have no close relative in training,
  so some gap exists from the first eval.
- **Stable and decay losses are not comparable to the last digit.** The stable runs evaluated on
  8 nodes and the decays on 4, which changes the masked positions and moves held-out loss by about
  0.01. P@L is unaffected.
- **The proxy matches data repetition, not model size.** Production models are far larger than
  170M and may memorize repeated data sooner.
- **The factor of 35 is in data exposure,** not in optimization: a proxy run takes 35 times fewer
  optimizer steps per pass over the data than production would.
- One seed per arm. The mix in the stable phase was not varied.

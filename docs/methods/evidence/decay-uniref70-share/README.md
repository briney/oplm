# uniref70 is 25% of decay-phase tokens

**Status:** provisional

## Question

During the final decay the data mix shifts toward uniref70. How large should its token share be?

## Arms

170M model on the 1/35 proxy data, 4,194,304 tokens per step. Every arm branches from its model's
stable-phase checkpoint at step 140k and decays linearly to step 175k. Only the uniref70 /
deepclust70 mix differs within a model.

- **OPLM** (Muon, lr 0.003): uniref70 at 12%, 25%, 50% and 65% of tokens.
- **ESM-C recipe** (AdamW, lr 0.0004): 25%, 50% and 65%.

## Result

No share improves contact precision. Raising the share trades deepclust70 loss for uniref70 loss,
widens the gap between training-set and held-out uniref70 loss, and at 65% P@L starts to fall.
25% was taken as the production value.

| arm | uniref70 gte2 loss | uniref70 train gte2 loss | deepclust70 gte2 loss | P@L pdb40 | P@L cameo | P@L casp | P@L pdb40_2048 |
|---|---|---|---|---|---|---|---|
| oplm-u12 | 2.0630 | 1.7176 | 2.0761 | .3792 | .3979 | .3214 | .3724 |
| oplm-u25 | 2.0364 | 1.6667 | 2.0822 | .3754 | .4009 | .3267 | .3734 |
| oplm-u50 | 2.0034 | 1.5856 | 2.0967 | .3769 | .3991 | .3272 | .3703 |
| oplm-u65 | 1.9934 | 1.5485 | 2.1133 | .3716 | .3955 | .3235 | .3647 |
| esmc-u25 | 2.0916 | 1.7493 | 2.1218 | .2984 | .3172 | .2619 | .2935 |
| esmc-u50 | 2.0692 | 1.6946 | 2.1323 | .2937 | .3143 | .2545 | .2884 |
| esmc-u65 | 2.0613 | 1.6669 | 2.1443 | .2904 | .3079 | .2513 | .2841 |

For OPLM the held-out minus train uniref70 gap grows from 0.35 at 12% to 0.44 at 65%. The ESM-C
arms show the same direction on every column.

## Runs

- W&B project `oplm-170M-overfit`; IDs, Slurm jobs and resolved configs are in `results.csv` and
  `configs/`. Run directories: `~/projects/oplm/mup_redo/runs/overfit-1of35*/` on the cluster.
- Discussion: issues #58 and #78.

## Caveats

- **The data do not separate 12%, 25% and 50%** on P@L; they differ only in which held-out loss
  they favor. 25% is a judgment call, not a measured optimum.
- One model size per recipe, one seed, one decay length, proxy scale.
- Mix fractions in the configs are row probabilities, not token shares; 25% of tokens is a
  uniref70 row fraction of 0.11.

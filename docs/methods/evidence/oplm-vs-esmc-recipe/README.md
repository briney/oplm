# OPLM (architecture and Muon) against the ESM-C recipe

**Status:** needs matched rerun

## Question

How much better is the OPLM recipe than a like-sized ESM-C-style baseline trained on the same data
with the same protocol? This is the baseline comparison, and for now the only evidence bearing on
the choice of Muon.

## Arms

Both models have hidden size 768 and 24 layers and train on the 1/35 proxy data at 4,194,304
tokens per step, commit 628d7bf.

- **oplm:** the OPLM architecture (sandwich norm, Canon layers) with Muon, lr 0.003.
- **esmc:** the ESM-C recipe (`src/oplm/configs/train/esm-c.yaml`: pre-norm, no Canon) with AdamW,
  lr 0.0004.

Each has a stable run to step 252k and three linear decays with uniref70 at 25% of tokens,
branched at steps 56k, 140k and 252k (ending at 70k, 175k and 315k).

## Result

OPLM is ahead at every point: by 0.05 to 0.06 held-out uniref70 loss and 0.08 to 0.09 P@L on pdb40
after a decay.

| arm | final step | uniref70 gte2 loss | deepclust70 gte2 loss | P@L pdb40 | P@L cameo | P@L casp |
|---|---|---|---|---|---|---|
| oplm stable | 252k | 2.1056 | 2.1233 | .3671 | .3904 | .2980 |
| esmc stable | 252k | 2.1189 | 2.1369 | .2972 | .3181 | .2632 |
| oplm decay56k | 70k | 2.0839 | 2.1125 | .3556 | .3745 | .3037 |
| esmc decay56k | 70k | 2.1339 | 2.1517 | .2768 | .2889 | .2486 |
| oplm decay140k | 175k | 2.0364 | 2.0822 | .3754 | .4009 | .3267 |
| esmc decay140k | 175k | 2.0916 | 2.1218 | .2984 | .3172 | .2619 |
| oplm decay252k | 315k | 2.0076 | 2.0628 | .3959 | .4177 | .3370 |
| esmc decay252k | 315k | 2.0684 | 2.1049 | .3092 | .3248 | .2755 |

## Runs

- W&B project `oplm-170M-overfit`; IDs, Slurm jobs and resolved configs are in `results.csv` and
  `configs/`. Run directories: `~/projects/oplm/mup_redo/runs/overfit-1of35/` and
  `overfit-1of35-esmc-lr0.0004/` on the cluster.
- The ESM-C learning-rate sweep is W&B `oplm-mup-esmc-4M`; see [mup-lr.md](../../../mup-lr.md),
  *ESM-C funnel*.

## Caveats

- **Optimizer and architecture change together,** so this does not say how much of the gap is
  Muon. The isolated ablation is #117.
- **The ESM-C learning rate may not be at its optimum for this length.** Its sweep put the
  24k-step optimum at 0.00063 to 0.001 (from session notes); 0.0004 was used here and was not
  bracketed at 252k steps.
- **Equal steps, not equal compute.** OPLM steps were about twice as slow as ESM-C steps when
  these ran (about 0.9 s against 0.42 s, from session notes, before the fused Canon kernel).
  Parameter counts are also not matched: Canon adds parameters at the same width and depth.
- One seed per arm, proxy scale.

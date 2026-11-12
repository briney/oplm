# Antibody sequences are 1.5% of decay-phase tokens

**Status:** provisional

## Question

How large a share of the decay-phase tokens should be antibody sequences (AntiRef96, unpaired heavy
and light chains)? Too little leaves antibody CDR3 modeling at the protein-only level; too much
repeats the antibody data often enough to memorize it and could cost general protein performance.

## Arms

Four decays of the 170M model on the 1/35 proxy data, branched from the same stable-phase
checkpoint at step 140k and decayed linearly (`wsd_linear`) to step 175k on 4 nodes at 4,194,304
tokens per step.

- **control:** protein only, uniref70 at 25% of tokens, the rest deepclust70.
- **ab1.5, ab3, ab6:** 1.5%, 3% and 6% of tokens from AntiRef96 with `sample_members: true`, taken
  out of deepclust70's share. The uniref70 token share is held at 25%.

The antibody proxy is 6 of 211 shards of `brineylab/oplm-antibody-sequences` v1.0.0 (about 6M
chains, 0.7B tokens), so each arm repeats the antibody data about as often as a production decay
at the same share would. Row fractions are derived from token shares with mean residues per row:
uniref70 363, deepclust70 134.7, AntiRef96 115.6. For ab1.5 the row fractions are uniref70
0.109678, deepclust70 0.869646, antiref96 0.020676.

## Result

1.5% gets nearly all of the antibody gain that generalizes to unseen CDR3s, at no measurable cost
on the protein evals apart from a 0.009 drop in cameo P@L. Larger shares mostly add memorization.

CDR3 loss at step 175k (`antibody_cdr3` eval: one forward pass per CDR3 residue, only that residue
masked), on the held-out AntiRef90 clusters with two or more members (`gte2`). "Seen" means the
exact CDR3 occurs in the 6 training shards.

| arm | heavy | light | heavy unseen | heavy seen | heavy seen-unseen gap | light unseen | light seen |
|---|---|---|---|---|---|---|---|
| control | 2.184 | 1.922 | 2.187 | 2.173 | 0.01 | 2.304 | 1.833 |
| ab1.5 | 1.451 | 0.839 | 1.496 | 1.305 | 0.19 | 1.445 | 0.699 |
| ab3 | 1.366 | 0.812 | 1.458 | 1.069 | 0.39 | 1.418 | 0.672 |
| ab6 | 1.264 | 0.785 | 1.412 | 0.793 | 0.62 | 1.373 | 0.649 |

- Unseen heavy CDR3 loss drops 0.69 at 1.5%. Each further doubling of the share gains about 0.04
  on unseen CDR3s while the seen-unseen gap triples from 0.19 to 0.62.
- The control scores seen and unseen heavy CDR3s almost the same, so the gap is memorization, not
  seen CDR3s being easier.

Protein evals at step 175k:

| arm | deepclust70 gte2 loss | uniref70 gte2 loss | P@L cameo | P@L casp | P@L pdb40 | P@L pdb40_2048 |
|---|---|---|---|---|---|---|
| control | 2.0822 | 2.0364 | .4009 | .3267 | .3754 | .3734 |
| ab1.5 | 2.0814 | 2.0360 | .3918 | .3288 | .3789 | .3740 |
| ab3 | 2.0827 | 2.0354 | .3896 | .3192 | .3770 | .3745 |
| ab6 | 2.0832 | 2.0359 | .3892 | .3228 | .3775 | .3736 |

Every held-out loss is within 0.003 of the control. `results.csv` also has the singleton sets.

## Runs

- W&B IDs and Slurm jobs are in `results.csv`. Each arm's config and sbatch are in its `config`
  directory on the cluster.
- The control's CDR3 numbers come from a standalone 1-GPU eval of its step-175k checkpoint (job
  35558), because the eval did not exist when the control ran. The eval is deterministic.
- Seen/unseen numbers come from jobs 35786 and 35899, using
  `~/projects/oplm/mup_redo/ab_cdr3_eval/{run_eval_split.py,cdr3match.py}` on the cluster and the
  `eval_*_{seen,unseen}.parquet` splits next to them.
- Full write-up and the production decision: issue #78. The eval itself: issue #85, PR #95.

## Caveats

- **Proxy scale.** 170M model on 1/35 of the data, one seed per arm. Not rerun at production scale.
- **Commits differ.** The control ran at 628d7bf, the antibody arms at f28661e, which adds the
  `wsd_sqrt` schedule and the `antibody_cdr3` eval. `wsd_linear` training is unchanged between them.
- **The cameo dip is unexplained.** P@L on cameo (256 chains) is 0.009 to 0.012 lower in all three
  antibody arms. That is outside the 0.001 spread seen across three protein-only decays that
  differ only in schedule shape, and it does not grow with the share. It was accepted without a
  second seed.
- **Seen/unseen is a lower bound on overlap.** CDR3s were pulled from the training shards with a
  regex that agrees with the dataset's CDR mask on 93% of eval chains.
- **Only one antibody train set was tested:** AntiRef96 representatives with AntiRef99 members.
  Other cluster levels are open on #78.
- Seen/unseen metric columns in `results.csv` are named by hand; they are not W&B metrics.

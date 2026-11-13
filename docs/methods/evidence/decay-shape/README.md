# The learning rate decays linearly

**Status:** settled

## Question

Does the shape of the final learning-rate decay matter? Linear (`wsd_linear`) was the default;
square-root (`wsd_sqrt`) and cosine (`wsd_cosine`) were the alternatives.

## Arms

170M model on the 1/35 proxy data, Muon at lr 0.003, 4,194,304 tokens per step. Every arm
branches from the same stable-phase checkpoint at step 140k and decays to zero at step 175k. Only
`train.scheduler` differs within a pair.

- **uniref70 at 25% of tokens:** linear, sqrt, cosine.
- **uniref70 at 12% of tokens:** linear, sqrt.

## Result

The three shapes tie: every held-out loss is within 0.003 of linear and every P@L within 0.004,
except casp under cosine. Linear is kept because it was already the default.

| arm | uniref70 gte2 loss | deepclust70 gte2 loss | P@L pdb40 | P@L cameo | P@L casp | P@L pdb40_2048 |
|---|---|---|---|---|---|---|
| linear-u25 | 2.0364 | 2.0822 | .3754 | .4009 | .3267 | .3734 |
| sqrt-u25 | 2.0374 | 2.0800 | .3780 | .4001 | .3251 | .3731 |
| cosine-u25 | 2.0373 | 2.0824 | .3788 | .4007 | .3177 | .3741 |
| linear-u12 | 2.0630 | 2.0761 | .3792 | .3979 | .3214 | .3724 |
| sqrt-u12 | 2.0612 | 2.0750 | .3778 | .3992 | .3215 | .3731 |

`results.csv` also has the singleton sets and the train-probe losses.

## Runs

- W&B project `oplm-170M-overfit`; IDs, Slurm jobs and commits are in `results.csv`, resolved
  configs in `configs/`. Run directories: `~/projects/oplm/mup_redo/runs/overfit-1of35/` on the
  cluster.
- Discussion and the mid-decay curves: issue #81, PR #83 (the `wsd_sqrt` code), PR #87 (docs).

## Caveats

- **Mid-decay leads are not final leads.** The sqrt arm led linear by about 0.011 loss partway
  through the decay because its learning rate was lower at that step; the lead was gone by step
  175k. Compare shapes only at the end of the decay. (The 0.011 figure is from session notes; the
  curves are in W&B and not exported here.)
- **casp under cosine is 0.009 lower.** casp is 101 chains and the other three sets do not move.
- **Commits differ.** The sqrt arms ran at a140e98, the others at 628d7bf; a140e98 adds only the
  `wsd_sqrt` schedule.
- One seed per arm, proxy scale, one decay length (20% of total steps).

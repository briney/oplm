# The μP base learning rate is 0.0063 at a 24k-step horizon

**Status:** settled

## Question

What base learning rate should `train.lr` default to for μP + Muon, does the output multiplier
need retuning, and does the value found at 170M transfer to a larger model?

## Arms

All cells: Muon, 4,194,304 tokens per step, commit 6e8b277, held-out uniref70 loss. `results.csv`
has every cell with a `phase` column.

- **smoke** (1k steps) and **coarse** (10k steps, constant LR): 170M at eight learning rates from
  0.00063 to 0.016.
- **refine** (20k steps, constant LR): 170M at lr 0.004, 0.0063, 0.01 × output multiplier 0.5, 1, 2.
- **refine-decay:** every refine cell branched at step 20k into a 4k linear decay to step 24k.
- **replicate, replicate-decay:** lr 0.004 and 0.0063 at two more seeds, same protocol.
- **confirm, confirm-decay:** 800M at lr 0.004, 0.0063, 0.01 with the depth LR correction (exponent
  0.5); 10k steps constant, then a 2k decay to step 12k.
- The **-nogate** and **-wdi0.1** phases repeat cells with one setting changed; see
  [residual-gate](../residual-gate/README.md) and
  [weight-decay-to-init](../weight-decay-to-init/README.md).

## Result

After a decay, 0.0063 is the best learning rate at 170M by 0.003 over 0.004, more than the seed
range, and the same value is at the optimum at 800M. Output multiplier 1 is kept.

170M, held-out loss at step 24k after the decay (output multiplier 1):

| lr | seed 42 | seed 43 | seed 44 | mean |
|---|---|---|---|---|
| 0.004 | 2.0811 | 2.0809 | 2.0816 | 2.0812 |
| 0.0063 | 2.0775 | 2.0791 | 2.0791 | 2.0786 |
| 0.01 | 2.0795 | | | |

Output multiplier, 170M seed 42 after the decay:

| lr | mult 0.5 | mult 1 | mult 2 |
|---|---|---|---|
| 0.004 | 2.0813 | 2.0811 | 2.0792 |
| 0.0063 | 2.0793 | 2.0775 | 2.0771 |
| 0.01 | 2.0795 | 2.0795 | 2.0776 |

800M after the decay (step 12k): lr 0.004 2.0350, 0.0063 2.0306, 0.01 2.0301.

## Runs

- Cluster: `~/projects/oplm/mup_redo/sweeps/<phase>/`. W&B IDs and resolved configs for all 53
  cells are in `results.csv` and `configs/`. Slurm job IDs were not recovered (the phases ran as
  job arrays).
- Procedure: [mup-lr.md](../../../mup-lr.md), *Commands*.

## Caveats

- **Rank after a decay.** At constant LR the ranking is different: at step 20k lr 0.004 leads
  0.0063 (2.1239 against 2.1251). The undecayed ranking favors lower learning rates.
- **This is the optimum at 24k steps.** It falls as training lengthens; see
  [lr-vs-training-length](../lr-vs-training-length/README.md).
- **Output multiplier 2 is ahead of 1 by up to 0.002** at all three learning rates. That is about
  the seed range at multiplier 1 (0.0016), and it was not replicated across seeds.
- **The 800M check is short** (12k steps, one seed) and does not separate 0.0063 from 0.01.

# Layer looping: a second stacked pass, added at the decay, is the only layout that helps

**Status:** provisional

## Question

Can re-running shared layers (looping) buy quality without adding parameters, and if so with which
layout? The motivation is the structure head: it is memory-bound on its language-model base, so
depth that costs compute but not parameters is worth having. Looping is off by default
(`model.num_loops: 1`); whether production uses it is not decided.

## Arms

All arms are 170M with 24 layers. Looping is introduced at the start of the decay: each arm loads
weights only (`train.init_from`), warms up for 1k steps, then decays linearly to zero. Global
tokens per step are unchanged (4,194,304); looped arms use a smaller micro-batch.

- **Layout probe.** From the lr 0.004 long run's step-250k weights, a 50k-step decay:
  - `x1`: unlooped control.
  - `stack2`: all 24 layers run, then all 24 run again.
  - `interleave2`: each layer run twice in a row.
  - `mid-interleave2`: interleave on layers 8 to 15 only.
  - `interleave3`: each layer three times in a row.
- **Proxy replication.** On the 1/35 proxy (lr 0.003, uniref70 at 25% of tokens), `stack2` against
  the unlooped decay from the stable checkpoints at steps 56k, 140k and 252k.
- **Loop count at eval.** The proxy checkpoints re-evaluated with the loop count set to 1 to 4,
  without further training (`loop_count_eval.csv`).

## Result

`stack2` is the only layout that improves both held-out loss and contact precision, and it does
so at all three proxy branch points. The gain is tied to the trained loop count.

Layout probe, step 50k:

| arm | uniref70 loss | P@L pdb40 | P@L cameo | P@L casp | P@L pdb40_2048 |
|---|---|---|---|---|---|
| x1 | 1.9517 | .3865 | .4039 | .3283 | .3812 |
| stack2 | 1.9428 | .3928 | .4087 | .3311 | .3902 |
| interleave2 | 1.9475 | .3810 | .4022 | .3243 | .3797 |
| mid-interleave2 | 1.9535 | .3823 | .4032 | .3224 | .3803 |
| interleave3 | not finished | | | | |

Proxy replication, end of decay:

| branch step | arm | uniref70 gte2 loss | deepclust70 gte2 loss | P@L pdb40 | P@L cameo | P@L casp | P@L pdb40_2048 |
|---|---|---|---|---|---|---|---|
| 56k | unlooped | 2.0839 | 2.1125 | .3556 | .3745 | .3037 | .3463 |
| 56k | stack2 | 2.0734 | 2.1050 | .3572 | .3788 | .3007 | .3509 |
| 140k | unlooped | 2.0364 | 2.0822 | .3754 | .4009 | .3267 | .3734 |
| 140k | stack2 | 2.0243 | 2.0708 | .3851 | .4090 | .3303 | .3797 |
| 252k | unlooped | 2.0076 | 2.0628 | .3959 | .4177 | .3370 | .3905 |
| 252k | stack2 | 1.9938 | 2.0523 | .3987 | .4186 | .3459 | .3948 |

`stack2` lowers uniref70 loss by 0.010, 0.012 and 0.014 at the three branch points.

Loop count at eval, checkpoints from the 140k branch (uniref70 gte2 loss / P@L pdb40):

| trained as | 1 loop | 2 loops | 3 loops | 4 loops |
|---|---|---|---|---|
| stack2 | 2.1134 / .3558 | 2.0303 / .3846 | 2.0398 / .3785 | 2.0665 / .3699 |
| unlooped | 2.0427 / .3759 | 2.4191 / .2870 | | |

A model trained with two passes is worse at one, three or four, and an unlooped model collapses
when given a second pass. The `stack2` checkpoints from the 56k and 252k branches show the same
pattern; the unlooped model was tested with two passes at the 140k branch only.

## Runs

- Layout probe: W&B project `oplm-170M-loopdecay`, cluster
  `~/projects/oplm/mup_redo/runs/loopdecay-170M-lr0.004-long/`, driver `loopdecay.sh`.
- Proxy replication: W&B project `oplm-170M-overfit`, cluster `runs/overfit-1of35/`.
- Loop count at eval: jobs 35572 to 35575, script and raw output in
  `~/projects/oplm/mup_redo/kscan/` on the cluster, at commit 628d7bf.
- IDs, Slurm jobs and resolved configs for the training runs are in `results.csv` and `configs/`.

## Caveats

- **Warm start only.** Looping was added to an already-trained network at the decay. `stack`
  leaves the first pass as the pretrained network while `interleave` perturbs every layer, so the
  ranking of layouts may not hold for a model looped from the start. That has not been run.
- **Equal steps, not equal compute.** A two-pass arm executes twice the layers per step; looped
  arms trained at roughly half the tokens per second of the unlooped control. The comparison that
  matters for the structure head is equal memory, where a larger unlooped model is not an option.
- **Unlooped controls differ in how they start.** In the proxy replication the unlooped arms are
  full-state branches (`train.branch_from`) while the `stack2` arms load weights only and re-warm
  for 1k steps. In the layout probe the `x1` control, started the same way as the looped arms,
  finished at 1.9517 against 1.9516 for the long run's own decay, so the restart itself costs
  nothing measurable.
- **`interleave3` did not finish.** Job 33751 was cancelled after its step-45k checkpoint and has
  no final metrics.
- **`loop_count_eval.csv` is not comparable to `results.csv` to the last digit.** It was run on
  one node, so the masked positions differ from the in-training evals. Compare within the file.
- **`step` restarts for weights-only arms.** In `results.csv` the looped arms and `probe-x1` count
  steps from the start of their decay, while the unlooped proxy arms continue the parent's count.
- One seed per arm, one model size.

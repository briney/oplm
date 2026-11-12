# Canon layers are on

**Status:** needs matched rerun

## Question

Do the Canon convolution layers earn their step-time cost? Without them the model trains faster
per step, so they must buy enough loss and contact precision to justify it.

## Arms

170M model, 4,194,304 tokens per step, seed 42, commit 6e8b277. Each arm trains 20k steps at a
constant learning rate, then branches into a 4k-step linear decay to step 24k, with structure evals
at the end.

- **nocanon:** `model.canon_enabled: false`, at four learning rates (0.0025, 0.004, 0.0063, 0.01).
- **canon:** the same recipe with Canon on, at its best learning rate from the matching sweep
  (0.0063).

The only config difference between the arms is `model.canon_enabled`.

## Result

After the decay, Canon is worth 0.009 held-out loss and about 0.01 P@L against the best no-Canon
arm, and 0.011 loss and 0.015 to 0.02 P@L at equal learning rate.

| arm | lr | uniref70 loss | P@L pdb40 | P@L cameo | P@L casp |
|---|---|---|---|---|---|
| canon | 0.0063 | 2.0791 | .3120 | .3377 | .2662 |
| nocanon | 0.0025 | 2.0980 | .2777 | .3060 | .2491 |
| nocanon | 0.004 | 2.0935 | .2856 | .3102 | .2519 |
| nocanon | 0.0063 | 2.0905 | .2968 | .3163 | .2500 |
| nocanon | 0.01 | 2.0884 | .3023 | .3313 | .2550 |

## Runs

- W&B project `oplm-mup-4M-nocanon`; decay-run IDs and Slurm jobs are in `results.csv`.
- The 20k-step constant-LR parents of the no-Canon arms are W&B `ckp33jpc`, `87ljdyr6`, `k6rijgcn`
  and `3cdsir0o` (lr 0.0025 to 0.01), under `sweeps-nocanon/refine/runs/` on the cluster. The Canon
  arm's parent is `sweeps/refine-nogate/runs/170M-lr0.0063-om1-a0-s42`.
- Driver script: `~/projects/oplm/mup_redo/nocanon.sh` on the cluster.

## Caveats

- **The no-Canon learning rate is not bracketed.** Loss and P@L are still improving at the highest
  rate tested (0.01), so the no-Canon arm is not at its optimum and the gap is an upper bound. The
  rerun needs no-Canon cells above 0.01.
- **The comparison is at equal steps, not equal compute.** When these runs were made the Canon
  model took about 0.90 s per step against 0.53 s without (same host, unfused Canon kernels; from
  session notes, not re-read from the logs for this record). The fused Canon kernel (PR #96) has
  since cut that cost, so the step-time ratio needs re-measuring on `main` and the comparison
  restating at equal wall-clock.
- **Short horizon, one seed.** 24k steps at 170M; not checked on a long run or a larger model.

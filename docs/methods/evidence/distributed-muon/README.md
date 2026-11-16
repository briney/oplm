# Under DDP the Muon step is batched and distributed across ranks

**Status:** settled

## Question

Whether to replace `torch.optim.Muon`, which orthogonalizes one matrix at a time and repeats the
identical work on every DDP rank, with a step that stacks same-shape matrices and has each rank
orthogonalize only its share. Had it not been faster, or had it changed training, stock Muon would
have stayed.

## Arms

Old is `main` at `e13d8db` (`torch.optim.Muon`); new is `6a52233` (`oplm.training.optim.Muon`).
Everything else is held fixed within a pair. All runs use 4 nodes (32 B200 GPUs, DDP).

- **170M profile:** 160 steps, 4,194,304 tokens per step; steps 100–119 are timed with
  synchronization, steps 120–139 are traced.
- **6B profile:** the same, at 524,288 tokens per step (16,384 per GPU, no accumulation) and with
  `model.gradient_checkpointing=true`.
- **170M equivalence:** 3,000 steps, seed 42, evaluated every 500 steps.

## Result

The new step is faster at both sizes and trains the same.

Times are medians over ranks, in ms. The optimizer phase covers Muon and the auxiliary AdamW.

| Arm | Step | Optimizer phase | Forward | Backward | `Muon.step`, traced |
|---|---|---|---|---|---|
| 170M old | 626.4 | 51.7 | 150.4 | 409.3 | 77.5 |
| 170M new | 581.5 | 10.9 | 150.4 | 409.6 | 8.1 |
| 6B old | 2165.5 | 471.4 | 322.2 | 1328.5 | 478.4 |
| 6B new | 1860.9 | 188.2 | 319.6 | 1325.1 | 184.0 |

The step is 7.2% shorter at 170M and 14.1% shorter at 6B. Of the 184 ms that the 6B Muon step
still takes, 126 ms is the three bf16 all-reduces that hand the updates to every rank.

Equivalence, `eval/uniref70/loss`:

| Step | Old | New | New − old |
|---|---|---|---|
| 500 | 2.5719 | 2.5720 | 0.0000 |
| 1000 | 2.4685 | 2.4715 | 0.0030 |
| 1500 | 2.4227 | 2.4233 | 0.0006 |
| 2000 | 2.3922 | 2.3935 | 0.0013 |
| 2500 | 2.3649 | 2.3654 | 0.0005 |
| 3000 | 2.3470 | 2.3461 | −0.0008 |

## Runs

W&B project `thebrineylab/oplm-profile`. Cluster outputs under
`/mnt/home/briney/projects/oplm/mup_redo/distmuon-gates/`, submitted by `distmuon-gates.sh` in the
parent directory. Discussion: briney/oplm#53 and PR #129.

## Caveats

- The 6B pair runs with gradient checkpointing because 16,384 tokens per GPU without it runs out
  of memory (jobs 36561 and 36562). Checkpointing lengthens backward, so the optimizer's share of
  the step is lower here than it would be without it; the millisecond figures are unaffected.
- 32 ranks, not the ~256 of a production run. With more ranks each rank orthogonalizes fewer
  matrices, but the all-reduce does not shrink, so it sets the floor.
- One run per arm, one seed. The equivalence gap (at most 0.003) is the size the fused Canon
  kernel gate saw between two builds.
- The traced `Muon.step` column includes profiler overhead; compare it within a pair only.
- HSDP is not covered: it keeps `torch.optim.Muon` (#128).

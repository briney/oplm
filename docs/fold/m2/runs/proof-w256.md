# Proof run w256: launch and forced resume (Task 8)

- **Job:** 35928 (4 nodes x 8 B200, QOS admin_override), started 05:06:15 UTC on 2026-11-12.
- **Code:** tag `fold-m2c-runs` = 7150334 (main after PR #112). Script from `oplm slurm generate`
  in a cluster checkout of the tag (`~/fold-m2c/oplm`), config `configs/fold/proof-w256.yaml`, 50k steps.
- **W&B:** https://wandb.ai/thebrineylab/oplm-fold/runs/x9fr8xv2
- **Output:** `/mnt/data/briney/oplm-runs/fold-m2c/proof-w256`
- **Step time:** median 6.02 s per optimizer step over the first steps (config estimate 4.3 s); 50k steps
  is about 84 h.

## First valid eval

Step 1 (`at_start`) and step 2000: `eval/valid/targets` = 64, `eval/valid/failed` = 0 (TM-score
0.138 at step 1, 0.614 at step 2000). Checkpoint 2000 has
`hf/` and `hf_ema/`.

## Forced resume

The plan's `scancel --signal=USR1 <jobid>` does not drain on this cluster (#113): Slurm never
signals the rank sessions, and Triton's bundled LLVM takes over the ranks' SIGUSR1 handler and does
not chain it. SIGTERM sent straight to the 8 ranks under each node's `accelerate launch` drains
correctly (LLVM restores the saved handlers and re-raises), so that was used instead:

    srun --jobid 35928 --overlap -N4 --ntasks-per-node=1 bash -c \
      'L=$(pgrep -u briney -f "^<venv>/bin/python <venv>/bin/accelerate launch"); kill -TERM $(pgrep -P $L)'

Sent at 09:58:33 UTC. Evidence (`.err` lines have no timestamps; `.out` is in
[fold-proof-w256_35928.out.txt](fold-proof-w256_35928.out.txt)):

1. `.err`: `Drain requested: checkpoint saved at step 2880; exiting 85` (once per rank, 32 lines).
2. `.out`: `requeueing (exit=1, drained=1, restarts=0, step=2880)`.
3. `.out`: `=== 2026-11-12T10:01:40+00:00 start; restart_count=1 ===`, then `Auto-resuming from checkpoint-2880`.
4. W&B continues run `x9fr8xv2`: logged steps go 2870 -> 2890 across the drain (2880 is the drain step itself, which saves before it logs), with no
   duplicated step and no other gap in the history.

Each node's `.err` also has a `ChildFailedError` traceback: `accelerate launch` reports the
ranks' exit 85 as a failure, which is why the wrapper keys on the `.drained` marker.

## Unplanned events

- First attempt (job 35926, 04:51 UTC) died in the install step: hatch-vcs could not parse
  a version from the run tag (`Can't parse version from tag 'fold-m2c-runs'`). Fixed by PR #112
  (versions from `v*` tags only); the tag moved from 200b50f to 7150334 before any run trained.

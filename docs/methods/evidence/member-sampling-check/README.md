# Member sampling serves a uniform cluster member, reproducibly

**Status:** settled

## Question

In the decay phase each uniref70 row serves a randomly drawn member of its cluster instead of the
representative (`sample_members: true`). Does the loader do that correctly at scale: uniform over
members, redrawn each pass, independent of the number of ranks, and identical after a resume?

This record is about correctness. It does not show that member sampling makes the model better;
no run isolates that.

## Arms

- **Unit tests,** `tests/data/sequence/test_member_sampling.py`: members come from their own
  cluster, the draw is uniform, the same member is served across rank and worker layouts, members
  change across passes and refills, and a resume or re-shard serves the same member.
- **One cluster run:** the 170M lr 0.004 long run's step-250k weights, decayed for 10k steps on 4
  nodes on the published dataset, with uniref70 at row fraction 0.05 and `sample_members: true`.
  The job was requeued after its step-5000 checkpoint to test the resume.

## Result

The cluster run trained at the same step time as the equivalent run without member sampling, with
no time spent waiting on the loader, and the steps retrained after the requeue (5010 to 5160)
reproduced the original loss, sequence lengths and gradient norm exactly.

About 80% of uniref70 train clusters have one member ([data.md](../../../data.md), *Clustering*),
so member sampling changes roughly one uniref70 row in eight.

## Runs

- W&B `oplm-170M-phase2check`, run `gxjs1i2j`, Slurm job 34399, commit 628d7bf. Resolved config in
  `configs/`. Cluster: `~/projects/oplm/mup_redo/runs/phase2check-170M-lr0.004-long/`, driver
  `phase2check.sh`.
- The step-time control is the unlooped arm of the looping probe (W&B `xszzn5uj`; see
  [looping](../looping/README.md)).
- Code: PR #69. Discussion: issue #58.

## Caveats

- **The step time (0.94 s), the bit-identical resume and the one-in-eight figure are from session
  notes,** not re-read from the logs for this record. A loader-level check on a development subset
  (24 of 24 checks passed, non-representative rate 12.4%) was run from a script that was not kept.
- **Its effect on the model is not measured.** Every decay arm with member sampling also changes
  the uniref70 share, so the two are confounded.

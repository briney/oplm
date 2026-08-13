# Fault-Tolerant Training for Long Multi-Node Runs — Design

**Date:** 2026-08-12
**Status:** Approved design, pre-implementation
**Scope:** Checkpointing, resume, and automatic failure recovery for training runs on
the order of weeks to months on ~64 nodes / 512 GPUs (B200s) under Slurm
(CoreWeave SUNK), while keeping `oplm train` fully usable for one-off runs on
bare metal and portable for open-source release.

---

## 1. Motivation and failure model

At 512 GPUs for a month or more, node/GPU failures are expected roughly weekly,
possibly every few days. The recovery model is **requeue-centric**: any failure
kills the Slurm job; Slurm requeues it onto healthy nodes; the trainer
auto-resumes from the latest committed checkpoint. Per-failure recovery cost is
minutes (reschedule + container start + checkpoint read), i.e. ~0.1% of run
time at one failure per week. Elastic in-job recovery (torchrun elastic,
hot spares) is deliberately **out of scope** — at this failure rate its
complexity buys almost nothing.

### Current-state audit (2026-08-12)

**Works today:**

- `accelerator.save_state` captures model, both optimizers (incl. Muon momentum
  buffers), both LR schedulers, per-rank RNG; `trainer_state.json` carries
  `global_step` / `epoch` / `samples_seen` / `tokens_seen`; config serialized
  alongside; rotation via `save_total_limit` (`src/oplm/training/checkpoint.py`).
- Resume restores all of the above, with tests for counter restoration, LR
  continuity, loss continuity, and sweep requeue simulation.
- `oplm.sweep.run` auto-detects the latest `checkpoint-<step>`; generated sbatch
  scripts carry `#SBATCH --requeue`; `afterany` dependencies; rank-identical
  `EvalContext` hang-avoidance discipline.
- Shard order and per-shard row order are pure functions of `(seed, epoch)` —
  the foundation the data cursor builds on.

**Gaps this design closes:**

1. Dataloader position is never checkpointed; resume replays the epoch from
   row 0 (data re-seen, `tokens_seen` diverges from an uninterrupted run).
2. `oplm.train` does not auto-resume — only the sweep runner does. A requeued
   production job silently restarts at step 0.
3. No SIGTERM/SIGUSR1 handling; drains and wall-clock expiry lose everything
   since the last periodic save. Cadence is steps-only (`save_every: 10000`).
4. Checkpoint writes are non-atomic and blocking; "latest" is a name scan; a
   torn newest checkpoint crashes auto-resume instead of falling back.
5. `accelerate launch` static rendezvous with `max_restarts=0`; a job-step
   failure ends the batch script (`set -euo pipefail`) with no requeue.
6. W&B starts a new run on every resume (no run-id persistence).
7. NCCL: only `NCCL_DEBUG=INFO`; no async error handling, no flight recorder,
   no explicit process-group timeout.
8. Rank-0 monolithic checkpoint format cannot shard, reshard, or save async —
   and cannot serve the 24–30B model ceiling at all.

**Suspected correctness issue (verify in Phase 0):** accelerate's
`prepare_data_loader` wraps `IterableDataset`s in `IterableDatasetShard`
(per-rank batch striping) *on top of* `ShardedProteinDataset`'s own rank/worker
striping. If both layers are active in multi-GPU runs, each rank sees 1/N of
its already-1/N stripe and most data is never seen. Inferred from accelerate
source, not yet confirmed against a live multi-GPU run.

---

## 2. Decisions and constraints

| Decision | Choice | Rationale |
|---|---|---|
| Parallelism target | FSDP2 (`fully_shard`) with HSDP mesh; DDP remains default | 30B × 12 B/param (Muon) ≈ 360 GB param+grad+optim — unshardable on a 180 GB B200. HSDP keeps sharding collectives NVLink-local. 6B fits DDP comfortably; 12B only barely (~145 GB states). |
| Checkpoint format | `torch.distributed.checkpoint` (DCP) | Parallelism-agnostic (`get_state_dict()`), async save, per-rank parallel writes, reshard-on-load (resume at different world size / debug a big run's checkpoint on one node). Works single-process too. |
| Resume rigor | **Data-exact** | Cursor into the deterministic shuffle: nothing re-seen or skipped, same order. Collator masking RNG untouched — post-resume masks differ bit-for-bit from an uninterrupted run; the data stream does not. |
| Checkpoint storage | Node-local NVMe staging + object store (S3-compatible) as durable source of truth | SUNK pod NVMe evaporates on requeue; every committed checkpoint must be uploaded. Object store optional — unset means pure-local behavior. |
| Recovery unit | Slurm job requeue | See failure model above. |
| Portability | Nothing CoreWeave-specific in `src/` | Trainer is pure PyTorch + POSIX signals; slurm layer is generic Slurm (+pyxis directives already present); CoreWeave specifics live only in user config/`.env`. `train.auto_resume` defaults `false` so interactive runs are unchanged. |

---

## 3. Checkpoint layer (DCP)

Replaces `accelerator.save_state` in `src/oplm/training/checkpoint.py`.

**Contents** — one app-level state dict of `Stateful` components:

- Model + optimizers via `torch.distributed.checkpoint.state_dict.get_state_dict()`
  (uniform across DDP prefix-stripping and FSDP2/DTensor sharding).
- LR schedulers; per-rank RNG bundle (Python, NumPy, torch-CPU, torch-CUDA);
  trainer counters (`global_step`, `epoch`, `samples_seen`, `tokens_seen`);
  dataloader cursor (§4); W&B run id; serialized config.

**Layout:** `checkpoint-<step>/` with DCP shard files + `.metadata`, plus
human-readable `trainer_state.json` and `config.yaml`. The `hf/` export
subdirectory is kept as-is (inference/release artifact, never used for resume).

**Commit protocol:** all writes go to `checkpoint-<step>.tmp/`; after a
`dist.barrier()` confirms every rank finished, rank 0 renames to
`checkpoint-<step>/` and updates a `latest` pointer file (temp + `os.replace`).
Discovery (auto-resume, rotation) considers only committed (renamed)
directories — torn writes are invisible by construction and the resume scan
falls back to the previous committed checkpoint.

**Async save:** `dcp.async_save()` — brief stall to stage tensors to CPU, then
serialization/IO on a background thread. One in-flight save at a time; a new
trigger waits for the pending one. The commit rename runs in the completion
callback. Drain checkpoints (§5) are synchronous.

**Cadence:** `train.save_every` (steps, existing) plus new
`train.save_every_minutes` — whichever fires first, evaluated on the
rank-synchronized step boundary (same discipline as the eval scheduler).
`configs/scaling.yaml` moves to ~30-minute time-based cadence for long runs.

**Retention:**

- `train.save_total_limit` culls **rolling** checkpoints only (newest K kept).
- New `train.keep_every_n_steps` (e.g. 100 000) and
  `train.keep_every_n_hours` mark checkpoints **permanent** — exempt from
  rotation, both rules may be active at once. Steps anchor to
  `global_step % n == 0` (stable across restarts); hours anchor to elapsed
  time since the run's first checkpoint, carried in checkpoint metadata so a
  requeue does not reset the clock.

**Load validation:** on resume, if the newest committed checkpoint fails to
load (corrupt shard, bad metadata), fall back to the next-newest with a loud
warning, bounded to a small number of attempts. Resume verifies the loaded
`config.yaml` is schedule-compatible with the live config (`warmup_steps`,
`max_steps`, LR shape) and fails loudly on mismatch — closing the silent
`LambdaLR` rebuild-from-config hazard. RNG-restore failures are surfaced as
errors, not swallowed.

---

## 4. Data-exact resume (dataloader cursor)

The shuffle is already a pure function of `(seed, epoch)`; the cursor stores a
**position**, not pipeline state.

- **Cursor:** `{epoch, batches_consumed_this_epoch}` — rank-identical by
  construction, stored once in DCP app state. No worker RNG or iterator state
  saved.
- **Skip by arithmetic, not by reading:** on resume, `ShardedProteinDataset`
  rebuilds the epoch permutation, converts the global batch count to a
  per-stream sample offset (rank/worker striping and DataLoader worker
  round-robin are deterministic), then walks the shard-order list subtracting
  whole-shard row counts (parquet metadata only) until the offset lands inside
  a shard, and starts from that row of that shard's permutation. Deep-epoch
  resume is O(#shards) metadata lookups.
- **`InterleavedDataset`:** skip draws the same sequence of source choices
  (advancing the source-choice RNG identically) and advances per-source
  cursors without materializing rows.
- **Validity guard:** cursor records `(world_size, num_workers,
  per_rank_batch, seed)`; resume fails loudly on mismatch.
  `train.resume_data_position=false` is the escape hatch (falls back to
  today's epoch-restart behavior). Data-exact resume assumes requeue at the
  same data-parallel size — the normal case on a fixed allocation.
- **Prerequisite:** resolve the suspected accelerate double-sharding first
  (likely by keeping the dataloader out of `accelerator.prepare`'s sharding
  path, since the dataset self-shards). The cursor math requires exactly one
  striping layer.
- **Acceptance test:** record the sample-identity sequence of an uninterrupted
  N-step run; repeat with a mid-epoch checkpoint + resume in a fresh process;
  assert the sequences are identical and `tokens_seen` matches the
  uninterrupted control exactly (the current e2e test explicitly cannot assert
  this). Runs on real parquet fixtures.

Masking stays as-is: collator RNG untouched (data-exact tier, not bitwise).

---

## 5. Signals, auto-resume, and the requeue loop

**Trainer drain handling:** SIGUSR1/SIGTERM handlers set a flag (only
async-signal-safe action). The flag is folded into the existing per-step
all-reduce (MAX) so all ranks agree on the same stop step. On agreement:
finish the optimizer step, take a **synchronous** checkpoint, exit with
dedicated code **85** ("drained cleanly, resume expected") — distinct from 0
("reached `max_steps`") and other nonzero ("crashed"). Generated sbatch header
gains `#SBATCH --signal=USR1@600` (no `B:` prefix — signal reaches every
rank's process), which also covers wall-clock expiry automatically.

**Auto-resume in `oplm.train`:** `train.auto_resume` (default `false`;
`oplm slurm generate` and sweep generation inject `true` into rendered
commands). Explicit `train.resume_from` always wins. Resolution order:
newest committed checkpoint in `output_dir` → newest committed checkpoint in
the object store → fresh start. Only committed checkpoints are candidates.

**Requeue wrapper in the generated script** (Slurm auto-requeues on node
failure, but a job-step failure just returns nonzero to the batch script):

- exit 0 → training complete; done.
- exit 85 → `scontrol requeue $SLURM_JOB_ID` unconditionally.
- other nonzero → requeue with a budget: `SLURM_RESTART_COUNT` capped by new
  `slurm.max_requeues` (default 20), plus a **no-progress guard** — before
  requeueing, read the latest committed checkpoint step; two consecutive
  restarts without step advance = deterministic crash loop → exit without
  requeue.

Supporting details: `#SBATCH --open-mode=append` + restart banner
(`SLURM_RESTART_COUNT`, timestamp, resumed-from step) for one readable log
across requeues. `MASTER_PORT` from `SLURM_JOB_ID` is stable across requeues;
`MASTER_ADDR` is already recomputed from the fresh node list.

---

## 6. Object-store sync, W&B continuity, NCCL hardening, guardrails

**Object-store sync** (all behind `train.remote_checkpoint_uri`, generic
fsspec/S3, credentials from env; unset = pure-local):

- *Save:* shards land on node-local NVMe; a per-node background uploader
  (local-rank-0) pushes that node's shards; global rank 0 uploads metadata and
  writes a remote **manifest last** (lists every shard + size) as the remote
  commit marker.
- *Load:* no bulk pre-download — DCP's fsspec storage reader reads directly
  from the object store; each rank fetches only the byte ranges its load plan
  needs.
- *Retention:* every committed checkpoint is uploaded (NVMe is ephemeral);
  remote retention mirrors local rules (rolling deleted on rotation,
  `keep_every_n_*` kept forever).

**W&B continuity:** capture the run id at first init; persist in checkpoint
app state + plaintext `wandb_run_id` in `output_dir`; on resume pass
`id=<saved>, resume="allow"` through accelerate `init_kwargs`. Metrics already
log with `step=global_step`, so a resumed run continues one W&B run.

**NCCL / distributed hardening** (generated-script env + `Accelerator` kwargs):

- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` — failed collectives raise everywhere
  instead of hanging; the exception becomes a nonzero exit → requeue → resume.
- Flight recorder: `TORCH_NCCL_TRACE_BUFFER_SIZE` +
  `TORCH_NCCL_DUMP_ON_TIMEOUT=1`, dump path in the job log directory —
  per-rank collective traces identify the stalled rank on a hang.
- Explicit process-group timeout via `InitProcessGroupKwargs`
  (`train.dist_timeout_minutes`, default 15) — tolerates cold-cache compile,
  converts genuine hangs to requeues in minutes.
- `NCCL_DEBUG=WARN` for production (configurable).

**Guardrails:**

- Non-finite loss → log diagnostics, abort nonzero. With the requeue loop this
  is automatic rollback to the last checkpoint; the no-progress guard catches
  deterministic NaNs.
- Loss-spike detection (deviation from the existing EMA-smoothed loss) warns
  only — spikes self-recover often enough that auto-rollback would cause more
  harm than good.
- Pre-flight check at startup, before data loading: per-rank allocate +
  matmul + one small all-reduce, with a logged rank→node map — sick nodes fail
  in seconds and attributably.

---

## 7. FSDP2 / HSDP

- `train.parallelism: ddp | hsdp`, default `ddp` (small runs and open-source
  out-of-box unchanged).
- `hsdp`: 2-D device mesh — shard within the node (8×B200), replicate across
  nodes — via FSDP2 (`fully_shard`/DTensor). Required before any 24–30B run;
  optional headroom for 12B.
- Checkpoints are parallelism-agnostic (§3), so the same checkpoint loads
  under DDP or HSDP at any world size.
- **Muon spike (Phase 0):** verify whether `torch.optim.Muon` accepts
  DTensor-sharded params; if not, adopt a torchtitan-style distributed Muon
  (gather → Newton–Schulz → scatter per matrix). Half-day prototype on 2 GPUs;
  gates only Phase 5.
- Open sub-decision (resolve during Phase 5 planning): keep accelerate's FSDP2
  wrapping vs. go native torch for the sharded path.

---

## 8. Config surface (new/changed knobs)

| Knob | Default | Meaning |
|---|---|---|
| `train.auto_resume` | `false` | Scan for newest committed checkpoint (local, then remote); slurm/sweep generation injects `true`. |
| `train.save_every_minutes` | `None` | Time-based cadence; fires alongside `save_every`, whichever first. |
| `train.keep_every_n_steps` | `None` | Permanent checkpoints at step multiples; exempt from `save_total_limit`. |
| `train.keep_every_n_hours` | `None` | Permanent checkpoints by elapsed time; exempt from `save_total_limit`. |
| `train.resume_data_position` | `true` | Escape hatch to old epoch-restart resume. |
| `train.remote_checkpoint_uri` | `None` | S3-compatible URI for durable checkpoint storage. |
| `train.dist_timeout_minutes` | `15` | Process-group timeout. |
| `train.parallelism` | `"ddp"` | `ddp` or `hsdp`. |
| `slurm.max_requeues` | `20` | Requeue budget for non-drain failures. |

Exit code **85** is reserved for "drained cleanly."

---

## 9. Testing

Per repo convention: `python -m pytest`, real data fixtures, `@pytest.mark.slow`
for heavy cases, e2e pilot-scale runs for every pipeline change.

- **Commit protocol:** kill-mid-save simulation leaves only `.tmp` dirs;
  discovery ignores them; fallback-to-previous on a corrupted newest.
- **Data-exact resume:** sample-identity-sequence equality vs. uninterrupted
  control (§4); cursor validity-guard failure on layout mismatch.
- **Drain:** send SIGUSR1 to a running pilot train; assert synchronous
  checkpoint + exit 85; resume continues at the right step.
- **Requeue wrapper:** shell-level tests of the exit-code branches and the
  no-progress guard (following existing `tests/slurm/` patterns of rendering +
  asserting script text, plus a scripted fake-`scontrol` harness).
- **W&B:** resumed trainer reuses the persisted run id.
- **Retention:** rotation never deletes `keep_every_n_*` checkpoints.
- **DCP round-trip:** save under DDP world-size 2 → load world-size 1 (reshard)
  → loss continuity; async save overlap (training steps proceed during save).
- **Object store:** moto/minio-backed round-trip of upload-manifest-commit and
  direct-read resume (slow-marked).

---

## 10. Phasing

Ordered so each phase is independently shippable and cheap universal wins land
before the format rebuild. **Phase 1 alone is the minimum bar for starting a
long 512-GPU DDP run.**

- **Phase 0 — verify:** accelerate double-sharding check (+ fix if real);
  Muon/FSDP2 spike.
- **Phase 1 — resilience on the current format** (works with
  `accelerator.save_state` today): auto-resume in `oplm.train`; atomic commit
  (tmp + rename); signal handling + drain checkpoint + exit codes; requeue
  wrapper with budget + no-progress guard; time-based cadence;
  `keep_every_n_steps/_hours`; W&B continuity; NCCL hardening; pre-flight
  check; non-finite-loss abort.
- **Phase 2 — DCP checkpoint layer:** format switch, `Stateful` app state,
  async save, load validation with fallback.
- **Phase 3 — data-exact cursor** (needs Phase 0 sharding fix; rides in
  Phase 2 app state).
- **Phase 4 — object-store sync.**
- **Phase 5 — HSDP** (informed by the Phase 0 spike).

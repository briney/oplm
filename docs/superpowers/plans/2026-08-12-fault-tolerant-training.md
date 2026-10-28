# Fault-Tolerant Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make week-to-month training runs on ~64 nodes / 512 B200s survive node/GPU failures
automatically: committed-only checkpoints with atomic writes, data-exact resume, signal-driven
drain checkpoints, a Slurm requeue loop, durable object-store copies, and a DCP checkpoint format
that carries through the DDP→HSDP transition — while keeping `oplm train` unchanged for one-off
bare-metal runs and free of anything CoreWeave-specific.

**Architecture:** The failure-recovery unit is the Slurm job: failures exit nonzero, the generated
script requeues with a budget, and `oplm.train` auto-resumes from the newest *committed*
checkpoint. The checkpoint layer moves from `accelerator.save_state` to
`torch.distributed.checkpoint` (DCP) with a tmp-dir + rename commit protocol, async saves staged
to node-local NVMe, and a background per-node upload to an S3-compatible object store whose
manifest-written-last is the remote commit marker. Resume is data-exact: a tiny
`{epoch, batches_consumed}` cursor is replayed against the already-deterministic `(seed, epoch)`
shuffle by index arithmetic. All rank-coordination (drain, time-based save, non-finite abort)
piggybacks on the existing per-step token all-reduce so every rank agrees on the same step.

**Tech Stack:** Python 3.11+, PyTorch ≥ 2.10 (`torch.distributed.checkpoint`, `torch.optim.Muon`,
FSDP2 `fully_shard`), HuggingFace Accelerate, fsspec (+ s3fs as an optional extra), Slurm
(SUNK/pyxis), pytest.

**Spec:** [docs/superpowers/specs/2026-08-12-fault-tolerant-training-design.md](docs/superpowers/specs/2026-08-12-fault-tolerant-training-design.md)

## Global Constraints

- Python 3.11+; `from __future__ import annotations` in every module. Max line length 100.
- CI gates run before declaring any task done: `ruff check src/`, `ruff format --check src/`,
  `ty check src/`, `python -m pytest -m "not slow"`. Run tests as `python -m pytest` (bare
  `pytest` resolves to a linuxbrew Python without oplm).
- Multi-process tests launch via `sys.executable -m torch.distributed.run` (never bare
  `torchrun`), CPU + gloo backend, and are marked `@pytest.mark.slow`.
- Type hints on every signature; Google-style docstrings on public classes/functions.
- `subprocess.run` with list args; never `shell=True`. `pathlib.Path` throughout.
- **Nothing CoreWeave/SUNK-specific in `src/`** (spec §2): trainer code is pure
  PyTorch + POSIX; Slurm specifics live in `oplm.slurm` render output; site specifics live in the
  user's YAML/`.env`. This tooling ships open source.
- `train.auto_resume` defaults `false`; only generated Slurm/sweep commands inject `true`
  (spec §5). Explicit `train.resume_from` always wins.
- Exit code **85** is reserved: "drained cleanly, resume expected" (spec §5).
- Checkpoint discovery (resume, rotation, status) must only ever consider **committed**
  checkpoints — a `checkpoint-<step>/` directory produced by the rename in the commit protocol.
  `*.tmp` directories are invisible everywhere (spec §3).
- `save_total_limit` culls rolling checkpoints only; checkpoints matching `keep_every_n_steps` /
  `keep_every_n_hours` are permanent and exempt (spec §3).
- Preserve current behavior for configs that set none of the new knobs: same checkpoint cadence,
  same resume semantics, same generated-script content except the additions specified here.
- Plan deviation from spec §5 (deliberate, small): in addition to SIGUSR1/SIGTERM handlers, the
  trainer treats `SLURM_JOB_END_TIME` (already in the environment via `--export=ALL`) minus a
  margin as a drain trigger. Signal delivery through `srun → container bash → accelerate
  launcher → ranks` is fragile (the launcher's default SIGUSR1 action is termination); the env
  clock is deterministic and needs no delivery chain. Both paths set the same drain flag.

## File Structure

**Created:**

| Path | Responsibility |
| --- | --- |
| `src/oplm/training/signals.py` | `DrainSignal` (USR1/TERM flag + `SLURM_JOB_END_TIME` clock), `DRAIN_EXIT_CODE = 85` |
| `src/oplm/training/preflight.py` | Startup GPU/collective sanity check with rank→host map |
| `src/oplm/training/remote.py` | `RemoteStore`: fsspec upload/manifest/rotation/latest-committed (Phase 4) |
| `tests/training/test_signals.py` | Drain flag, env-clock trigger, exit-code constant |
| `tests/training/test_commit_protocol.py` | tmp+rename atomicity, committed-only discovery, retention rules |
| `tests/training/test_e2e_drain.py` | SIGUSR1 mid-train → sync checkpoint + exit 85 → resume |
| `tests/training/test_e2e_dcp.py` | DCP round-trip, world-size 2→1 reshard, async overlap (slow) |
| `tests/training/test_remote.py` | `RemoteStore` over `file://` (+ optional moto, slow) |
| `tests/data/test_resume_cursor.py` | Stream skip arithmetic, interleaved skip, layout guard |
| `tests/training/test_e2e_data_exact.py` | Sample-identity equality vs uninterrupted control |
| `tests/data/test_double_sharding.py` | Phase-0 verification: per-rank coverage under `accelerator.prepare` (slow) |

**Modified:**

| Path | Change |
| --- | --- |
| `src/oplm/training/checkpoint.py` | Commit protocol; shared `latest_checkpoint`; retention exemptions; Phase 2: DCP save/load |
| `src/oplm/training/trainer.py` | Control-bundle reduce; time cadence; auto-resume; drain; preflight; W&B id; cursor wiring |
| `src/oplm/config.py` | New `TrainConfig` fields (spec §8) + validation |
| `src/oplm/data/sequence/dataset.py` | `stream_length()`, skip-aware iteration, `DataCursor`, interleaved skip |
| `src/oplm/data/sequence/loaders.py` | Dataloader kept out of accelerate sharding (if Phase 0 confirms) |
| `src/oplm/slurm/render.py` | `--signal`/`--open-mode` directives; requeue wrapper; NCCL env |
| `src/oplm/slurm/config.py` | `max_requeues`, `nccl_debug` fields |
| `src/oplm/slurm/cli.py` | Inject `train.auto_resume=true`; pass `progress_dir` to render |
| `src/oplm/sweep/phases.py` | Inject `train.auto_resume=true`; pass per-run progress dir |
| `src/oplm/sweep/run.py` | Reuse shared committed-only `latest_checkpoint` |
| `configs/scaling.yaml` | Time cadence, keep rules, requeue budget, remote URI placeholder |
| `docs/TRAIN.md`, `docs/SLURM.md`, `docs/CONFIG.md` | Resilience + requeue semantics docs |

---

## Phase 0 — Verification

### Task 0.1: Double-sharding verification test

**Files:**
- Create: `tests/data/test_double_sharding.py`
- Create: `tests/data/_double_sharding_worker.py` (subprocess entry)

**Interfaces:**
- Produces: an empirical verdict (test + committed note) on whether `accelerator.prepare`'s
  `IterableDatasetShard` stacks on `ShardedProteinDataset`'s own rank striping. Task 0.2 consumes
  the verdict.

- [ ] **Step 1: Write the subprocess worker** — a script run under
  `torch.distributed.run --nproc_per_node=2` (CPU/gloo) that builds the real train dataloader from
  a tiny config over the existing test parquet fixtures, passes it through `accelerator.prepare`
  exactly as `Trainer.__init__` does, iterates one full epoch, and writes the consumed
  `sequence_id`s to `out_dir/rank<i>.json`:

```python
"""Worker for the double-sharding verification test. Run under torch.distributed.run."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration


def main(fixture_path: str, out_dir: str) -> None:
    accelerator = Accelerator(
        cpu=True, dataloader_config=DataLoaderConfiguration(dispatch_batches=False)
    )
    from oplm.data.sequence.collate import MLMCollator
    from oplm.data.sequence.dataset import ShardedProteinDataset
    from oplm.data.tokenizer import get_tokenizer
    from torch.utils.data import DataLoader

    dataset = ShardedProteinDataset(fixture_path, seed=0)
    collator = MLMCollator(get_tokenizer(), max_length=64, keep_sequence_ids=True)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator, num_workers=2)
    dataloader = accelerator.prepare(dataloader)

    seen: list[str] = []
    for batch in dataloader:
        seen.extend(batch["sequence_ids"])
    rank = accelerator.process_index
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(seen))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

  If `MLMCollator` has no way to surface `sequence_id`s, add a private
  `keep_sequence_ids: bool = False` flag to it (returns them as a plain list in the batch dict,
  excluded from tensor collation) — that flag is also needed by Task 3.4.

- [ ] **Step 2: Write the test** — launch the worker, load both rank files, assert the invariant
  we *want*: `set(rank0) ∪ set(rank1)` equals the full fixture id set and
  `set(rank0) ∩ set(rank1)` is empty. Mark `@pytest.mark.slow`:

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_prepared_dataloader_covers_dataset_exactly_once(tmp_path, fixture_parquet_dir):
    worker = Path(__file__).with_name("_double_sharding_worker.py")
    subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node=2", "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0",
            str(worker), str(fixture_parquet_dir), str(tmp_path),
        ],
        check=True,
        timeout=300,
    )
    rank0 = set(json.loads((tmp_path / "rank0.json").read_text()))
    rank1 = set(json.loads((tmp_path / "rank1.json").read_text()))
    all_ids = _fixture_ids(fixture_parquet_dir)
    assert rank0 | rank1 == all_ids, "rows lost: double-sharding is real"
    assert not (rank0 & rank1), "rows duplicated across ranks"
```

  Reuse the repo's existing parquet fixtures (see `tests/data/` conftest); `_fixture_ids` reads
  the fixture with pyarrow directly.

- [ ] **Step 3: Run it** — `python -m pytest tests/data/test_double_sharding.py -v -m slow`.
  Record the verdict (pass = no bug; fail with missing rows = bug confirmed, expected per audit).
- [ ] **Step 4: Commit** the test with the verdict in the commit message (`test:` prefix). If the
  bug is confirmed, the test stays red until Task 0.2 lands — commit it `xfail`-marked with
  `strict=True` and remove the marker in Task 0.2.

### Task 0.2: Single-striping fix — keep the dataloader out of accelerate's sharding

**Only if Task 0.1 confirms the bug.** If 0.1 passes (accelerate skips re-sharding in this
configuration), close this task with a comment in `loaders.py` documenting why, and move on.

**Files:**
- Modify: `src/oplm/training/trainer.py:146` (prepare call), `src/oplm/data/sequence/loaders.py`
- Test: `tests/data/test_double_sharding.py` (goes green), existing `tests/training/test_e2e_*`

**Interfaces:**
- Produces: `oplm.data.sequence.loaders.DeviceDataLoader` — thin wrapper with
  `__iter__` (moves batch tensors to `device`, `non_blocking=True`), `set_epoch(epoch)`
  (forwards to `.dataset`), `.dataset` property, and `__len__` delegation.
- Consumes: nothing new. Later tasks treat `trainer.dataloader` as this wrapper.

- [ ] **Step 1:** Add `DeviceDataLoader` to `loaders.py`:

```python
class DeviceDataLoader:
    """Device-placement wrapper for a self-sharding dataloader.

    ``ShardedProteinDataset`` stripes rows over the joint (rank, worker) index itself, so the
    training dataloader must NOT pass through ``accelerator.prepare`` — accelerate would wrap the
    IterableDataset in ``IterableDatasetShard`` and stripe a second time (each rank would then see
    1/N of its already-1/N stripe). This wrapper supplies the only two things ``prepare`` was
    providing: device placement and ``set_epoch`` forwarding.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device) -> None:
        self._dataloader = dataloader
        self._device = device

    @property
    def dataset(self):  # noqa: ANN201 — mirrors DataLoader.dataset
        return self._dataloader.dataset

    def __len__(self) -> int:
        return len(self._dataloader)

    def set_epoch(self, epoch: int) -> None:
        set_epoch = getattr(self._dataloader.dataset, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

    def __iter__(self):
        for batch in self._dataloader:
            yield {
                k: v.to(self._device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
```

- [ ] **Step 2:** In `Trainer.__init__`, remove `dataloader` from the `accelerator.prepare(...)`
  call and wrap instead: `self.dataloader = DeviceDataLoader(dataloader, self.accelerator.device)`
  (adjust the `prepared` tuple indexing — model, optimizers, schedulers only). Add a comment
  noting the gradient-accumulation caveat: without a prepared dataloader, accelerate cannot
  detect end-of-dataloader for a partial final accumulation window; the training stream is
  effectively infinite (epoch rollover re-iterates), so no partial window ever syncs — same
  semantics as before for `gradient_accumulation_steps == 1`, and for >1 the tail micro-batches
  of an epoch simply roll into the next window.
- [ ] **Step 3:** Un-`xfail` the 0.1 test; run
  `python -m pytest tests/data/test_double_sharding.py -m slow tests/training/ -v`. The e2e
  training tests must still pass (single-process behavior is unchanged: accelerate never sharded
  at world size 1).
- [ ] **Step 4:** Run all CI gates. Commit (`fix: stop double-striping the train dataloader under
  accelerate.prepare`).

### Task 0.3: Muon-under-FSDP2 spike

**Files:**
- Create: `tests/training/test_fsdp2_muon_spike.py` (slow; doubles as the spike artifact)
- Modify: `docs/superpowers/specs/2026-08-12-fault-tolerant-training-design.md` §7 (record verdict)

**Interfaces:**
- Produces: a recorded decision for Phase 5 — `torch.optim.Muon` works on DTensor params as-is,
  or Phase 5 must carry a gather→Newton–Schulz→scatter adapter.

- [ ] **Step 1:** Write a 2-process CPU/gloo subprocess test (same launch pattern as Task 0.1):
  build a 2-layer `nn.Linear` stack, `fully_shard` it over a 1-D mesh, construct
  `torch.optim.Muon` over the (now DTensor) 2-D params, run 3 forward/backward/step iterations,
  and assert (a) no exception, (b) params changed, (c) the loss decreased on a fixed toy
  regression target. Wrap the Muon construction in `try/except NotImplementedError | RuntimeError`
  and make the test **record** rather than fail: `pytest.skip(f"Muon+DTensor unsupported: {e}")`
  so the suite stays green either way — the *verdict* is the skip/pass outcome.
- [ ] **Step 2:** Run it; also try `get_state_dict(model, [muon])` on the sharded model to verify
  the optimizer state round-trips through the Phase-2 checkpoint path.
- [ ] **Step 3:** Record the outcome in spec §7 (one paragraph: works as-is / needs adapter /
  needs torchtitan-style distributed Muon) and in this file next to Task 5.2.
- [ ] **Step 4:** Commit (`test: record Muon-under-FSDP2 spike verdict`).

---

## Phase 1 — Resilience on the current checkpoint format

Everything in this phase works with `accelerator.save_state` unchanged. **After Phase 1, a
512-GPU DDP run survives failures automatically.**

### Task 1.1: Commit protocol + shared committed-only discovery

**Files:**
- Modify: `src/oplm/training/checkpoint.py`, `src/oplm/sweep/run.py:15-40`
- Test: `tests/training/test_commit_protocol.py`, existing `tests/training/test_checkpoint.py`,
  `tests/sweep/test_run.py`

**Interfaces:**
- Produces:
  - `oplm.training.checkpoint.latest_checkpoint(output_dir: Path) -> Path | None` — newest
    committed `checkpoint-<step>` by numeric suffix; ignores `*.tmp` and non-numeric names.
    (Moved from `oplm.sweep.run`; sweep imports it from here.)
  - `save_checkpoint(...)` now writes into `checkpoint-<step>.tmp/`, barriers, then rank 0
    renames to `checkpoint-<step>/` and rewrites `<output_dir>/latest` (one line, the checkpoint
    dir name) via a temp file + `os.replace`.
- Consumes: existing `save_checkpoint` signature (unchanged for callers).

- [ ] **Step 1: Write failing tests:**

```python
def test_save_checkpoint_commits_via_rename(tiny_trainer_cfg, tmp_path):
    # after save: committed dir exists, no *.tmp remains, latest points at it
    ...
def test_torn_tmp_dir_is_invisible(tmp_path):
    (tmp_path / "checkpoint-500.tmp").mkdir()
    (tmp_path / "checkpoint-300").mkdir()
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-300"
def test_rotation_ignores_tmp_dirs(tmp_path):
    # a .tmp dir neither counts toward the limit nor gets deleted
    ...
def test_sweep_run_uses_committed_only(tmp_path):
    # sweep.run.latest_checkpoint is the shared function (identity check)
    ...
```

- [ ] **Step 2:** Run to verify failure (`latest_checkpoint` not in `oplm.training.checkpoint`).
- [ ] **Step 3: Implement.** In `save_checkpoint`: `tmp_dir = Path(output_dir) /
  f"checkpoint-{global_step}.tmp"`; point `accelerator.save_state`, `trainer_state.json`,
  `config.yaml`, and the `hf/` export at `tmp_dir`; then:

```python
    accelerator.wait_for_everyone()  # every rank finished writing into tmp_dir
    if accelerator.is_main_process:
        final_dir = tmp_dir.with_name(f"checkpoint-{global_step}")
        if final_dir.exists():  # re-save at same step after requeue: replace atomically-enough
            shutil.rmtree(final_dir)
        tmp_dir.rename(final_dir)
        _write_latest_pointer(Path(output_dir), final_dir.name)
        _rotate_checkpoints(Path(output_dir), save_total_limit)
    accelerator.wait_for_everyone()
```

  `_rotate_checkpoints` and the moved `latest_checkpoint` both filter with
  `d.name.startswith("checkpoint-") and not d.name.endswith(".tmp") and suffix.isdigit()`.
  Also add stale-tmp cleanup at trainer start (main process): delete `checkpoint-*.tmp` dirs in
  `output_dir` — they are by definition torn.
- [ ] **Step 4:** Update `sweep/run.py` to `from oplm.training.checkpoint import
  latest_checkpoint` (delete the local copy; keep `_CHECKPOINT_PREFIX` references working).
- [ ] **Step 5:** Run `python -m pytest tests/training/test_commit_protocol.py
  tests/training/test_checkpoint.py tests/sweep/ -v` → PASS. Run CI gates. Commit
  (`feat: atomic checkpoint commit + committed-only discovery`).

### Task 1.2: Cadence and retention config

**Files:**
- Modify: `src/oplm/config.py` (TrainConfig, ~line 82-90 block + `__post_init__`),
  `src/oplm/training/checkpoint.py` (`_rotate_checkpoints`)
- Test: `tests/training/test_commit_protocol.py`, `tests/test_config.py` (or wherever TrainConfig
  validation tests live — mirror existing patterns)

**Interfaces:**
- Produces (spec §8): `TrainConfig.save_every_minutes: int | None = None`,
  `keep_every_n_steps: int | None = None`, `keep_every_n_hours: float | None = None`,
  `auto_resume: bool = False`, `resume_data_position: bool = True`,
  `dist_timeout_minutes: int = 15`, `remote_checkpoint_uri: str | None = None`.
  Validation: the three cadence/keep values must be `> 0` when set.
- Produces: `_rotate_checkpoints(output_dir, save_total_limit, *, keep_every_n_steps=None)` —
  a checkpoint is permanent (never rotated) if `step % keep_every_n_steps == 0` or a `KEEP`
  marker file exists in the checkpoint dir.
- Produces: `mark_permanent(checkpoint_dir: Path) -> None` — writes the `KEEP` marker (used by
  the trainer for the hours rule, Task 1.3).

- [ ] **Step 1: Failing tests:** config round-trip + validation errors for each new field;
  rotation with `save_total_limit=1, keep_every_n_steps=100` over dirs
  `checkpoint-100, -150, -200, -250` deletes only `-150` (`-250` is newest, `-100`/`-200`
  permanent); a `KEEP`-marked dir survives regardless of step.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement fields + validation + rotation exemptions. Thread
  `keep_every_n_steps` through `save_checkpoint`'s call to `_rotate_checkpoints`.
- [ ] **Step 4:** Run → PASS. CI gates. Commit (`feat: time/step retention knobs and
  permanent-checkpoint exemptions`).

### Task 1.3: Control-bundle reduce + time-based save trigger

**Files:**
- Modify: `src/oplm/training/trainer.py` (train loop, ~lines 356-407; `_save_checkpoint`)
- Test: extend `tests/training/test_e2e_checkpoint.py`

**Interfaces:**
- Produces: `Trainer._reduce_step_flags(local_tokens: int) -> _StepFlags` where

```python
@dataclass(frozen=True)
class _StepFlags:
    tokens_delta: int
    drain: bool        # any rank saw SIGUSR1/SIGTERM or the SLURM_JOB_END_TIME margin
    nonfinite: bool    # any rank's step loss was non-finite
    save_due: bool     # any rank's save_every_minutes timer fired
```

  Implementation: one `accelerator.reduce` (sum) over a 4-element long tensor
  `[local_tokens, drain_flag, nonfinite_flag, time_due_flag]`, replacing the existing
  tokens-only reduce at `trainer.py:359-362`. Sum > 0 ⇒ flag set. This is the single
  rank-synchronization point the drain (1.5) and guardrail (1.8) tasks plug into; per-rank clock
  skew is harmless because *any* rank's timer firing triggers everyone.
- Produces: trainer state `self._last_save_at: float` (monotonic, reset after every checkpoint)
  and `self._first_checkpoint_at: float | None` persisted into `trainer_state.json` as
  `first_checkpoint_unix` (wall clock) so the `keep_every_n_hours` anchor survives requeue.
- Consumes: `mark_permanent` from Task 1.2.

- [ ] **Step 1: Failing test:** pilot config with `save_every=0, save_every_minutes=1`, patch
  `time.monotonic` (autouse-style monotonic fake advancing 30 s per step) → checkpoint appears at
  the step where 60 s elapses; second test: `keep_every_n_hours=1` with a fake clock crossing the
  1 h boundary → that checkpoint dir contains `KEEP`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `_StepFlags` + `_reduce_step_flags`; checkpoint trigger becomes:

```python
                step_saved = cfg.save_every > 0 and self.global_step % cfg.save_every == 0
                if step_saved or flags.save_due:
                    self._save_checkpoint()
```

  In `_save_checkpoint`, after saving: update `self._last_save_at`; set
  `self._first_checkpoint_at` if unset (loading it back on resume from `trainer_state.json`);
  apply the hours rule — if `keep_every_n_hours` is set and
  `(now_wall - first_checkpoint_unix)` has crossed a new multiple of `keep_every_n_hours * 3600`
  since the last permanent-by-time mark (track `last_time_keep_index` in `trainer_state.json`),
  call `mark_permanent(checkpoint_dir)`.
- [ ] **Step 4:** Run new + existing e2e checkpoint tests → PASS. CI gates. Commit
  (`feat: wall-clock checkpoint cadence on a rank-synchronized control bundle`).

### Task 1.4: Auto-resume in `oplm.train`

**Files:**
- Modify: `src/oplm/training/trainer.py:263-266`, `src/oplm/slurm/cli.py:77-79`,
  `src/oplm/sweep/phases.py` (train-args builders), `src/oplm/sweep/run.py`
- Test: extend `tests/training/test_e2e_checkpoint.py`, `tests/slurm/test_cli.py`,
  `tests/sweep/test_run.py`

**Interfaces:**
- Consumes: `latest_checkpoint` (Task 1.1).
- Produces: `Trainer.__init__` resume block becomes:

```python
        resume_target = cfg.train.resume_from
        if resume_target is None and cfg.train.auto_resume:
            found = latest_checkpoint(Path(cfg.train.output_dir))
            if found is not None:
                resume_target = str(found)
                _status(f"[dim]Auto-resuming from {found.name}[/dim]")
        if resume_target is not None:
            self._resume_from_checkpoint(resume_target)
```

  (Remote resolution is appended here in Task 4.2.)

- [ ] **Step 1: Failing tests:** (a) e2e: train 4 steps with `auto_resume=true`, new
  `Trainer(cfg)` with `resume_from=None` picks up at step 4; (b) fresh `output_dir` +
  `auto_resume=true` starts at step 0 (no scan surprise); (c) `oplm slurm generate` output
  contains `train.auto_resume=true` in the rendered command; (d) sweep-generated commands ditto.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement; in `sweep/run.py`, replace the local
  scan-and-set block with `cfg.train.auto_resume = True` before `Trainer(cfg)` (single code
  path). **Step 4:** PASS + CI gates. Commit (`feat: opt-in auto-resume from newest committed
  checkpoint`).

### Task 1.5: Drain — signals, env clock, sync checkpoint, exit 85

**Files:**
- Create: `src/oplm/training/signals.py`
- Modify: `src/oplm/training/trainer.py` (init + train loop + `finally`)
- Test: `tests/training/test_signals.py`, `tests/training/test_e2e_drain.py`

**Interfaces:**
- Produces:

```python
DRAIN_EXIT_CODE = 85

class DrainSignal:
    """Set-a-flag drain trigger: SIGUSR1/SIGTERM handlers plus the Slurm end-time clock.

    ``install()`` registers handlers (idempotent; chains any previous non-default handler).
    ``requested`` is True once a signal arrived OR ``SLURM_JOB_END_TIME`` (unix seconds, exported
    by Slurm and forwarded by ``--export=ALL``) minus ``margin_seconds`` has passed. Handlers only
    set a bool — the only async-signal-safe action.
    """

    def __init__(self, *, margin_seconds: int = 600, env: Mapping[str, str] | None = None): ...
    def install(self) -> None: ...
    @property
    def requested(self) -> bool: ...
```

- Consumes: `_StepFlags.drain` (Task 1.3) — trainer feeds `drain_signal.requested` into the
  control bundle each step.
- Produces: on `flags.drain`, the trainer saves synchronously and raises
  `SystemExit(DRAIN_EXIT_CODE)`; the existing `finally:` (progress stop, `_emit_train_end`,
  `accelerator.end_training`) runs on the way out, so W&B closes cleanly.

- [ ] **Step 1: Failing unit tests** (`test_signals.py`): flag flips on
  `os.kill(os.getpid(), signal.SIGUSR1)`; env clock with `SLURM_JOB_END_TIME = now + 300`,
  `margin_seconds=600` → `requested` immediately True; no env, no signal → False;
  `DRAIN_EXIT_CODE == 85`.
- [ ] **Step 2: Failing e2e test** (`test_e2e_drain.py`): start a pilot `Trainer.train()` in a
  thread is not signal-safe — instead run it in a **subprocess** (plain `sys.executable -m
  oplm.train --config ...` on a tiny config with `max_steps=200, save_every=0`), wait for
  `output_dir/config.yaml` + a first logged step (poll a `log_every=1` side-effect or just sleep
  2 s), send SIGUSR1, assert: exit code 85, exactly one committed checkpoint exists, and a second
  run with `auto_resume=true` resumes past that step (reuse Task 1.4's assertion helpers). Mark
  slow.
- [ ] **Step 3:** Implement `signals.py`; in trainer: instantiate + `install()` in `__init__`
  (margin from a module constant 600 s for now); in the loop, right after
  `flags = self._reduce_step_flags(...)`:

```python
                if flags.drain:
                    self._save_checkpoint()
                    logger.warning("Drain requested: checkpoint saved at step %d; exiting %d",
                                   self.global_step, DRAIN_EXIT_CODE)
                    raise SystemExit(DRAIN_EXIT_CODE)
```

  (Placed before the eval/periodic-save blocks so a drain never starts a long eval.)
- [ ] **Step 4:** PASS + CI gates. Commit (`feat: drain-to-checkpoint on signal or walltime
  margin, exit 85`).

### Task 1.6: Requeue wrapper + Slurm directives

**Files:**
- Modify: `src/oplm/slurm/render.py` (`_header`, `render_job`, `JobSpec`),
  `src/oplm/slurm/config.py` (`max_requeues: int = 20`, `nccl_debug: str = "WARN"`),
  `src/oplm/slurm/cli.py`, `src/oplm/sweep/phases.py` (pass `progress_dir`)
- Test: `tests/slurm/test_render.py`, `tests/slurm/test_requeue_wrapper.py` (bash-level)

**Interfaces:**
- Produces: `JobSpec.progress_dir: str | None = None` — shell-expandable string naming the
  training output dir (plain path for single jobs; `"$RUN_DIR/..."` for sweep arrays), used by
  the no-progress guard. `oplm slurm generate` fills it from `cfg.train.output_dir`;
  `phases.py` from its per-run dir expression.
- Produces: header gains `#SBATCH --signal=USR1@600` and `#SBATCH --open-mode=append`; body
  gains a restart banner, `set +e` around the `srun`, and this wrapper after it:

```bash
STATUS=$?
set -e
if [ "$STATUS" -eq 0 ]; then
  echo "training complete"
  exit 0
fi
RESTARTS=${SLURM_RESTART_COUNT:-0}
if [ "$RESTARTS" -ge <max_requeues> ]; then
  echo "requeue budget (<max_requeues>) exhausted; exiting $STATUS" >&2
  exit "$STATUS"
fi
STEP_FILE="<progress_dir>/.last_requeue_step"
CURRENT_STEP=$(ls -d "<progress_dir>"/checkpoint-* 2>/dev/null \
  | sed 's/.*checkpoint-//' | grep -E '^[0-9]+$' | sort -n | tail -1)
CURRENT_STEP=${CURRENT_STEP:-0}
if [ "$STATUS" -ne 85 ] && [ "$RESTARTS" -ge 1 ]; then
  PREV_STEP=$(cat "$STEP_FILE" 2>/dev/null || echo -1)
  if [ "$CURRENT_STEP" -le "$PREV_STEP" ]; then
    echo "no checkpoint progress since last restart (step $CURRENT_STEP); crash loop — not requeueing" >&2
    exit "$STATUS"
  fi
fi
echo "$CURRENT_STEP" > "$STEP_FILE"
echo "requeueing (exit=$STATUS, restarts=$RESTARTS, step=$CURRENT_STEP)"
scontrol requeue "$SLURM_JOB_ID"
```

  Restart banner right after `source <env_file>`:
  `echo "=== $(date -Is) start; restart_count=${SLURM_RESTART_COUNT:-0} ==="`.
  When `progress_dir` is `None`, omit the no-progress guard (requeue on budget only) — keeps the
  layer usable for non-training jobs.
- Note: the checkpoint-name `ls` scan here intentionally cannot see `.tmp` dirs as committed —
  the glob matches them, so the `sed`/`grep -E '^[0-9]+$'` pipeline must run on the *basename
  suffix*; `checkpoint-500.tmp` yields `500.tmp` which fails the numeric grep. Keep that property
  (it is the shell-side mirror of committed-only discovery) and assert it in the bash test.

- [ ] **Step 1: Failing render tests** (mirror existing `tests/slurm` text-assertion style):
  header contains both new directives; body contains `scontrol requeue`; `max_requeues` value
  interpolated; array jobs get `"$RUN_DIR"`-based `STEP_FILE`; no guard block when
  `progress_dir=None`.
- [ ] **Step 2: Failing bash test** (`test_requeue_wrapper.py`): extract the wrapper into a
  rendered script against a `tmp_path` progress dir; execute with `bash` and fake `scontrol` /
  `SLURM_*` env (write a stub `scontrol` recording its argv to a file, prepend to `PATH`).
  Cases: exit 0 → no requeue; exit 85 → requeue even at high restart count below budget; exit 1
  twice at same step → second invocation does NOT requeue; exit 1 with advanced step → requeues;
  budget exhausted → no requeue; `checkpoint-500.tmp` present alone → `CURRENT_STEP=0`.
- [ ] **Step 3:** Implement in `render.py` + config field + plumbing. **Step 4:** PASS + CI
  gates. Commit (`feat: requeue wrapper with budget and no-progress guard`).

### Task 1.7: W&B run continuity

**Files:**
- Modify: `src/oplm/training/trainer.py:94-106`, `src/oplm/training/checkpoint.py`
  (`trainer_state.json` gains `wandb_run_id`)
- Test: extend `tests/training/test_e2e_wandb.py`

**Interfaces:**
- Produces: main process persists the run id to `<output_dir>/wandb_run_id` right after
  `init_trackers` (`wandb.run.id`, import guarded by `wandb_enabled`), and `save_checkpoint`
  copies it into `trainer_state.json`. Before `init_trackers`, if resuming (either
  `resume_from` set or `auto_resume` found a checkpoint) read the id — checkpoint's
  `trainer_state.json` first, `wandb_run_id` file second — and extend
  `wandb_kwargs |= {"id": run_id, "resume": "allow"}`.
- Ordering note: wandb currently initializes *before* the resume block. Move the resume-target
  *resolution* (Task 1.4's block, minus the actual `_resume_from_checkpoint` call) ahead of
  `init_trackers`; keep the heavy `_resume_from_checkpoint` where it is.

- [ ] **Step 1: Failing test:** with `WANDB_MODE=offline` on a pilot config, train + checkpoint;
  second trainer with `auto_resume=true` → assert both runs' `wandb_run_id` files are identical
  and the second `wandb.init` received `id=<first id>, resume="allow"` (patch
  `accelerate.tracking.WandBTracker` init kwargs or read the offline run dir metadata —
  follow whatever `test_e2e_wandb.py` already inspects).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** PASS + CI gates. Commit
  (`feat: persist wandb run id; resume continues the same run`).

### Task 1.8: NCCL hardening, process-group timeout, preflight, non-finite guard

**Files:**
- Create: `src/oplm/training/preflight.py`
- Modify: `src/oplm/training/trainer.py` (Accelerator kwargs, preflight call, nonfinite flag),
  `src/oplm/slurm/render.py` (env block)
- Test: `tests/slurm/test_render.py`, `tests/training/test_preflight.py`, e2e nonfinite test

**Interfaces:**
- Produces (render.py env block, replacing the bare `NCCL_DEBUG=INFO` line):

```bash
export NCCL_DEBUG=<slurm.nccl_debug>            # default WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_FR_DUMP_TEMP_FILE=<slurm.log_dir>/nccl_trace_${SLURM_JOB_ID}_rank
```

- Produces: `Accelerator(..., kwargs_handlers=[InitProcessGroupKwargs(
  timeout=timedelta(minutes=cfg.train.dist_timeout_minutes))])` (import from `accelerate.utils`;
  only when launched distributed — harmless single-process).
- Produces: `oplm.training.preflight.run_preflight(accelerator) -> None` — logs
  `rank=<i> host=<socket.gethostname()> device=<accelerator.device>`, allocates a 64 MB tensor,
  runs one 1024² matmul, and (when `num_processes > 1`) one small `accelerator.reduce`; raises
  `RuntimeError` with the host name in the message on any failure. Called in `Trainer.__init__`
  right after the Accelerator is constructed.
- Consumes: `_StepFlags.nonfinite` (Task 1.3) — the trainer feeds
  `not math.isfinite(current_loss)` into the control bundle; on `flags.nonfinite`:
  `raise RuntimeError(f"non-finite training loss at step {self.global_step}")` (all ranks raise
  together; the requeue loop resumes from the last checkpoint = automatic rollback; the
  no-progress guard catches a deterministic NaN).
- Produces (spec §6, warn-only): in `_log_step`, maintain an EMA of `train/loss` (same
  `ema = 0.98 * ema + 0.02 * loss` smoothing `oplm.training.mup` uses for its logging helper);
  when `loss > 3.0 * ema` after the first 50 logged steps, `logger.warning("loss spike: %.4f vs
  EMA %.4f at step %d", ...)`. No abort, no rollback — spikes self-recover often enough that
  acting on them causes more harm than good.

- [ ] **Step 1: Failing tests:** render test for the env block + timeout wiring (assert the
  kwargs handler is passed — construct a Trainer on a pilot config and inspect
  `accelerator.init_handler` or patch `Accelerator` to capture kwargs); preflight unit test
  (runs clean single-process; error message contains hostname when the matmul is patched to
  raise); e2e: patch the model to return a NaN loss at step 3 → `RuntimeError` mentioning step 3,
  and the step-2 checkpoint is intact.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS + CI gates. Commit
  (`feat: NCCL fault env, pg timeout, preflight check, non-finite loss abort`).

### Task 1.9: Production config + docs

**Files:**
- Modify: `configs/scaling.yaml`, `docs/TRAIN.md`, `docs/SLURM.md`, `docs/CONFIG.md`
- Test: `tests/test_docs_links.py` (existing link check), config-load test over `scaling.yaml`

- [ ] **Step 1:** `configs/scaling.yaml` train block gains: `save_every_minutes: 30`,
  `keep_every_n_steps: 100000`, `auto_resume: true` (explicit here — this config only runs under
  Slurm), `save_total_limit: 3`; slurm block: `max_requeues: 20`. Add a commented
  `# remote_checkpoint_uri: s3://...` placeholder (activated in Phase 4).
- [ ] **Step 2:** Docs: `docs/SLURM.md` §"Requeue semantics" rewritten (auto-resume is now real;
  document exit 85, the budget, the no-progress guard, `--signal`); `docs/TRAIN.md` gains a
  "Fault tolerance" section (knob table from spec §8, drain behavior, what a resume restores);
  `docs/CONFIG.md` documents each new field.
- [ ] **Step 3:** `python -m pytest tests/test_docs_links.py -v` + a test asserting
  `load_config` accepts `configs/scaling.yaml` with the new keys. CI gates. Commit
  (`docs: fault-tolerance operations guide + scaling config update`).

---

## Phase 2 — DCP checkpoint layer

### Task 2.1: DCP save path

**Files:**
- Modify: `src/oplm/training/checkpoint.py` (rewrite `save_checkpoint` internals),
  `src/oplm/training/trainer.py` (`_save_checkpoint` passes optimizers/schedulers/extra state)
- Test: `tests/training/test_e2e_dcp.py`, existing checkpoint tests updated

**Interfaces:**
- Produces: `save_checkpoint` signature change (trainer is the only caller):

```python
def save_checkpoint(
    accelerator, model, optimizers, schedulers, cfg, output_dir,
    *, global_step, epoch, samples_seen, tokens_seen,
    cursor: DataCursor | None = None,  # dataclass lands in Task 3.1; TYPE_CHECKING import,
    wandb_run_id: str | None = None,   # callers pass None until Phase 3
    save_total_limit: int = 3, keep_every_n_steps: int | None = None,
) -> Path:  # returns the committed checkpoint dir
```

  Internals: build the state dict via DCP `Stateful` components and write with
  `dcp.save(state_dict, checkpoint_id=str(tmp_dir))`:

```python
class _ModelOptState(Stateful):
    """Model + optimizer state through torch.distributed.checkpoint.state_dict.

    get_state_dict/set_state_dict unwrap DDP ('module.' prefixes) and handle FSDP2/DTensor
    uniformly — this is what makes the checkpoint parallelism- and world-size-agnostic.
    """
    def __init__(self, model, optimizers):
        self._model, self._optimizers = model, [_unwrap_accelerated(o) for o in optimizers]
    def state_dict(self):
        msd, osd = get_state_dict(self._model, self._optimizers)
        return {"model": msd, "optimizers": osd}
    def load_state_dict(self, sd):
        set_state_dict(self._model, self._optimizers,
                       model_state_dict=sd["model"], optim_state_dict=sd["optimizers"])

state_dict = {
    "app": _ModelOptState(model, optimizers),
    "schedulers": [s.state_dict() for s in schedulers],
    "trainer": {"global_step": ..., "epoch": ..., "samples_seen": ..., "tokens_seen": ...,
                "cursor": asdict(cursor) if cursor else None, "wandb_run_id": wandb_run_id},
}
```

  Per-rank RNG is a sidecar: each rank writes `rng_state_<rank>.pt`
  (`{"python": random.getstate(), "numpy": np.random.get_state(),
  "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}`) into
  `tmp_dir` with `torch.save` (DCP dedupes identical keys across ranks, so per-rank blobs don't
  fit its model; tiny files, no collective needed). Keep `trainer_state.json` (now also carrying
  `wandb_run_id`, `first_checkpoint_unix`, `last_time_keep_index`, and the serialized cursor for
  human inspection), `config.yaml`, and the `hf/` export exactly as today. Commit protocol,
  rotation, and pointer from Task 1.1 unchanged.
- Consumes: `_unwrap_accelerated(o)` = `getattr(o, "optimizer", o)` (AcceleratedOptimizer wrap).

- [ ] **Step 1: Failing test:** single-process pilot save → committed dir contains `.metadata` +
  `__0_0.distcp`-style shard files + `rng_state_0.pt` + `trainer_state.json` + `config.yaml` +
  `hf/`; `accelerator.save_state` artifacts (`model.safetensors`, `optimizer.bin`) are ABSENT.
- [ ] **Step 2:** FAIL. **Step 3:** Implement (single-process DCP works without an initialized
  process group). **Step 4:** PASS; existing checkpoint tests updated to the new layout. CI
  gates. Commit (`feat: DCP checkpoint format`).

### Task 2.2: DCP load, validation, fallback

**Files:**
- Modify: `src/oplm/training/checkpoint.py` (`load_checkpoint`),
  `src/oplm/training/trainer.py` (`_resume_from_checkpoint`)
- Test: `tests/training/test_e2e_dcp.py`, `tests/training/test_commit_protocol.py`

**Interfaces:**
- Produces: `load_checkpoint(accelerator, model, optimizers, schedulers, checkpoint_dir, cfg)
  -> dict[str, Any]` — `dcp.load` into the same `Stateful` layout, restore schedulers, restore
  this rank's `rng_state_<rank>.pt` (**missing file = hard error**, closing the
  swallowed-RNG-failure hole; error message explains the world-size-change case and points at
  starting fresh RNG with an explicit `train.resume_from` + documented
  `OPLM_ALLOW_MISSING_RNG=1` escape), and return the `trainer` dict.
- Produces: schedule-compatibility validation before `dcp.load`: parse the checkpoint's
  `config.yaml` and compare `{warmup_steps, stable_steps, max_steps, scheduler, lr, min_lr}`
  against the live config; mismatch → `ValueError` naming each differing field (closes the
  silent `LambdaLR` reshape hazard).
- Produces: fallback in the trainer's auto-resume path (Task 1.4 block): wrap
  `_resume_from_checkpoint` in `try/except (RuntimeError, OSError, ValueError)`; on failure with
  `auto_resume` (never with explicit `resume_from`), log loudly and retry with the next-newest
  committed checkpoint, at most 2 fallback attempts, then re-raise.

- [ ] **Step 1: Failing tests:** save→load round-trip restores step/epoch/tokens + optimizer
  momentum (compare a Muon `momentum_buffer` tensor before/after); schedule mismatch
  (`warmup_steps` changed) raises naming the field; corrupted newest (truncate a `.distcp` file)
  + `auto_resume` → resumes from previous checkpoint with a warning in caplog; corrupted newest
  + explicit `resume_from` → raises.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS + CI gates. Commit
  (`feat: DCP load with schedule validation and committed-fallback`).

### Task 2.3: Async save

**Files:**
- Modify: `src/oplm/training/checkpoint.py`, `src/oplm/training/trainer.py`
- Test: `tests/training/test_e2e_dcp.py`

**Interfaces:**
- Produces: `save_checkpoint(..., blocking: bool = True)`; when `blocking=False` it calls
  `dcp.async_save(...)` (stages to CPU, returns a `Future`) and returns a `PendingSave` handle
  `(future, tmp_dir, final_name, permanent)`.
- Produces: trainer-side finalization on the control-bundle boundary: keep at most one
  `self._pending_save`; each step, fold `pending_done = pending is not None and
  future.done()` into the control bundle (5th element, sum); when the sum equals
  `num_processes`, all ranks' writes are complete → rank 0 performs the rename + pointer +
  rotation (the deferred tail of the commit protocol), `_emit_checkpoint_saved` fires, pending
  clears. A new save trigger or drain or train-end while a save is pending first blocks on
  `future.result()` + finalize. Drain and final saves stay `blocking=True`.

- [ ] **Step 1: Failing test:** pilot run with async on (make it the default for periodic saves)
  and `save_every=2, max_steps=8` → 4 committed checkpoints, `on_checkpoint_saved` fired 4×, and
  a probe asserting at least one training step executed between `async_save` returning and the
  commit rename (instrument with a step-counter captured in a patched rename).
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS + CI gates; also rerun the drain
  e2e (drain during a pending async save must finalize the pending one, then take its own
  blocking save). Commit (`feat: async periodic checkpoints with deferred commit`).

### Task 2.4: Reshard e2e (world-size 2 → 1)

**Files:**
- Test: `tests/training/test_e2e_dcp.py` (+ a subprocess worker like Task 0.1's)

- [ ] **Step 1:** Subprocess test (slow): 2-rank CPU/gloo pilot trains 3 steps, saves; a
  1-process run loads the same checkpoint (`auto_resume`) and trains 2 more steps without error;
  assert restored `global_step` and that a designated weight tensor matches rank 0's saved value.
  This is the resilience property that lets a 64-node run be inspected or resumed at a different
  world size. (Data-exactness across world-size change is *not* asserted — the cursor guard from
  Phase 3 will refuse; pass `train.resume_data_position=false` in this test once Phase 3 lands.)
- [ ] **Step 2:** Run → PASS (no new src code expected; fix what surfaces). CI gates. Commit
  (`test: cross-world-size DCP resume`).

---

## Phase 3 — Data-exact resume

Requires Task 0.2's single-striping verdict/fix. Cursor rides in Task 2.1's `trainer` state.

### Task 3.1: Skip-aware `ShardedProteinDataset`

**Files:**
- Modify: `src/oplm/data/sequence/dataset.py`
- Test: `tests/data/test_resume_cursor.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class DataCursor:
    """Position of the training stream, plus the layout it is only valid under."""
    epoch: int
    batches_in_epoch: int
    world_size: int
    num_workers: int
    per_rank_batch: int
    seed: int

class ShardedProteinDataset:
    def stream_length(self) -> int:
        """Rows this (rank, worker) stream serves in one epoch. Must be called in-context
        (inside a worker, or single-process) — uses _joint_stripe()."""
    def set_resume_skip(self, batches_in_epoch: int, per_rank_batch: int,
                        num_workers: int) -> None:
        """Arm a one-epoch skip; applied by the next __iter__, in each worker."""
    def clear_resume_skip(self) -> None: ...
```

  `__iter__` becomes a thin call to `self._iter_stream(self._resolved_skip())` where
  `_resolved_skip()` converts the armed batch count to this stream's sample offset:
  worker `w` of the local rank contributed `len(range(w, batches_in_epoch, num_workers))`
  batches (DataLoader round-robin), each of `per_rank_batch` consecutive samples from this
  stream → `skip = that_count * per_rank_batch`, then `skip %= max(self.stream_length(), 1)`
  (full wraps re-yield the identical stream, so modulo is exact). `_iter_stream(skip)` walks the
  shard order computing each shard's `selected` count from index arithmetic
  (`len(range(first_selected_offset, n_rows, ...))` — no `read_table`) and only starts reading
  at the shard where the running count crosses `skip`, slicing `selected[skip - consumed:]`.
- The armed skip is plain instance state — it pickles into DataLoader workers with the dataset,
  which is exactly how it reaches them.

- [ ] **Step 1: Failing tests** (single-process, real fixtures): (a) baseline = materialize
  epoch-0 stream as a list; for `skip in {0, 1, shard0_len, shard0_len + 3, stream_len + 2}`:
  armed skip yields `baseline[skip % len(baseline):]` exactly; (b) `pq.read_table` call count
  (patched counter) for a skip landing in shard 2 of 3 is exactly 2 (shards 2 and 3 read,
  shard 1 skipped by arithmetic); (c) `stream_length()` equals `len(baseline)`;
  (d) `num_workers=2` DataLoader: resumed loader's concatenated output equals the uninterrupted
  loader's output from batch k onward (order-exact at the batch level).
- [ ] **Step 2:** FAIL. **Step 3:** Implement (refactor `__iter__`; keep the yielded-dict
  contract byte-identical for skip=0). **Step 4:** PASS + CI gates (whole `tests/data/`).
  Commit (`feat: index-arithmetic resume skip for ShardedProteinDataset`).

### Task 3.2: Skip-aware `InterleavedDataset`

**Files:**
- Modify: `src/oplm/data/sequence/dataset.py`
- Test: `tests/data/test_resume_cursor.py`

**Interfaces:**
- Produces: same `set_resume_skip`/`clear_resume_skip` on `InterleavedDataset`. In `__iter__`,
  after computing `choices`: per-stream skip `k` (same round-robin arithmetic) → count
  per-source draws in `choices[:k]` (`torch.bincount`), arm each source's skip with its count
  **in samples** via a new internal `_arm_source_skip(source, n)` that reduces `n` modulo the
  source's `stream_length()` (sources refill deterministically, so wraps are identity), then
  iterate `choices[k:]`.
- Consumes: `ShardedProteinDataset.stream_length`, `_iter_stream` (Task 3.1).

- [ ] **Step 1: Failing test:** two-source interleaved fixture; baseline epoch-0 stream vs
  armed-skip stream — suffix equality for skips crossing several refills of the smaller source.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS + CI gates. Commit
  (`feat: resume skip for InterleavedDataset`).

### Task 3.3: Trainer cursor wiring + layout guard

**Files:**
- Modify: `src/oplm/training/trainer.py`, `src/oplm/training/checkpoint.py` (cursor already in
  the schema from Task 2.1)
- Test: `tests/training/test_e2e_checkpoint.py`, `tests/data/test_resume_cursor.py`

**Interfaces:**
- Produces: trainer tracks `self._batches_in_epoch` (increment where samples are counted,
  `trainer.py:344`; reset to 0 in the `StopIteration` epoch-rollover branch, which also calls
  `dataset.clear_resume_skip()`). `_save_checkpoint` builds
  `DataCursor(epoch=self.epoch, batches_in_epoch=self._batches_in_epoch,
  world_size=accelerator.num_processes, num_workers=cfg.data.num_workers,
  per_rank_batch=cfg.train.batch_size, seed=cfg.train.seed)`.
- Produces: `_resume_from_checkpoint` — with a cursor present and
  `cfg.train.resume_data_position` true: validate `(world_size, num_workers, per_rank_batch,
  seed)` against the live run; mismatch → `ValueError` listing each differing field and naming
  the `train.resume_data_position=false` escape hatch; match → `self._batches_in_epoch = cursor
  .batches_in_epoch` and arm `dataset.set_resume_skip(...)` (walk `self.dataloader.dataset`;
  arm the top-level dataset — interleaved propagates internally). With the flag false or no
  cursor: today's behavior (epoch restart) plus a WARNING that data will be re-seen.
- Async-save note: the cursor snapshot must be taken at trigger time (it is — the dict is built
  before `async_save` stages), not at commit time.

- [ ] **Step 1: Failing tests:** guard raises on changed `num_workers` naming the field;
  `resume_data_position=false` logs the re-seen warning and starts the epoch at row 0;
  post-resume `_batches_in_epoch` continues from the cursor.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS + CI gates. Commit
  (`feat: data cursor in checkpoints with layout guard`).

### Task 3.4: Data-exactness acceptance test

**Files:**
- Test: `tests/training/test_e2e_data_exact.py`
- Modify (if not done in 0.1): `src/oplm/data/sequence/collate.py` (`keep_sequence_ids` flag)

- [ ] **Step 1:** Control run: pilot config (real fixtures, `num_workers=2`,
  `per_rank_batch=4`, `max_steps=12`), a callback recording every batch's `sequence_ids` in
  order. Interrupted run: same config, `max_steps=12`, checkpoint at step 5 (`save_every=5`),
  stop at step 6 (small `max_steps` first run = 6), fresh `Trainer` with `auto_resume` running
  to 12. Assert: concatenated sequence-id stream of (run-to-6 ++ resumed-7-to-12) equals the
  control's stream **exactly**, and final `tokens_seen` matches the control exactly (retire the
  old test's continuity-band assertion in favor of equality; keep the file's docstring updated —
  it currently documents the epoch-restart limitation, which this phase removes).
- [ ] **Step 2:** Run → PASS (this is the acceptance gate for the whole phase; debug until
  exact). Also assert a mid-epoch-boundary variant: checkpoint lands within epoch 1 (fixture
  small enough that step 5 crosses an epoch), proving `epoch + batches_in_epoch` compose.
- [ ] **Step 3:** CI gates. Commit (`test: bitwise data-stream equality across resume`).

---

## Phase 4 — Object-store sync

### Task 4.1: `RemoteStore`

**Files:**
- Create: `src/oplm/training/remote.py`
- Modify: `pyproject.toml` (add `fsspec` to core deps if not already transitive; `s3fs` under a
  new `[project.optional-dependencies] s3` extra)
- Test: `tests/training/test_remote.py`

**Interfaces:**
- Produces:

```python
class RemoteStore:
    """Checkpoint mirror on an fsspec filesystem (s3://, gs://, file://...).

    Layout mirrors output_dir: <uri>/checkpoint-<step>/<files> + manifest.json written LAST —
    a checkpoint-<step>/ without manifest.json is uncommitted and invisible.
    """
    def __init__(self, uri: str) -> None: ...           # fsspec.core.url_to_fs
    def upload_checkpoint(self, local_dir: Path, *, files: list[Path], permanent: bool,
                          write_manifest: bool) -> None:
        """Upload `files` (relative to local_dir) into <uri>/<local_dir.name>/."""
    def finalize(self, name: str, *, permanent: bool) -> None:
        """Write manifest.json: {"files": {relpath: size}, "permanent": bool}."""
    def latest_committed(self) -> tuple[str, dict] | None:  # (name, manifest)
    def download_checkpoint(self, name: str, dest: Path) -> Path: ...
    def rotate(self, save_total_limit: int, keep_every_n_steps: int | None) -> None:
        """Delete non-permanent committed checkpoints beyond the limit (manifest-aware)."""
```

- [ ] **Step 1: Failing tests** over `file://{tmp_path}`: upload+finalize round-trip; a dir
  without manifest is not `latest_committed`; manifest size mismatch on download →
  `RuntimeError`; rotate keeps permanent + newest K. Optional `@pytest.mark.slow` moto-based
  `s3://` smoke test guarded by `pytest.importorskip("moto")`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS + CI gates. Commit
  (`feat: fsspec RemoteStore with manifest-committed checkpoints`).

### Task 4.2: Trainer integration — background upload + remote resume

**Files:**
- Modify: `src/oplm/training/trainer.py`, `src/oplm/training/checkpoint.py`,
  `configs/scaling.yaml` (uncomment the URI placeholder into a documented example)
- Test: `tests/training/test_remote.py` (e2e over `file://`), drain e2e extended

**Interfaces:**
- Consumes: `RemoteStore` (4.1); `train.remote_checkpoint_uri` (1.2).
- Produces, save side: after a checkpoint commits (sync or async finalization), each node's
  local-rank-0 process (`accelerator.local_process_index == 0`) starts a daemon-thread upload of
  the shard files *this node's ranks wrote* (DCP writer file names carry the writing rank; each
  rank records its own written files via a save-plan hook or simply `rng_state_<rank>.pt` +
  `glob` filtered by rank — implementer picks, test pins behavior: union of all nodes' uploads
  == full dir, no file uploaded twice); global rank 0 additionally uploads `.metadata`,
  `trainer_state.json`, `config.yaml`, `hf/`, then `finalize()`s the manifest **after** a small
  gloo-group barrier confirms all nodes' uploads finished (use a dedicated
  `torch.distributed.new_group(backend="gloo")` created at trainer init when a remote URI is
  set — never the NCCL default group from a background thread). One in-flight upload at a time;
  a new commit while uploading queues (drop-oldest beyond 1 queued: the newer checkpoint
  supersedes). Rank 0 calls `store.rotate(...)` after finalize. Drain path blocks on the upload
  (bounded by a 10-min timeout, then logs and exits 85 anyway — the local checkpoint still
  exists for a same-allocation requeue).
- Produces, resume side (extends Task 1.4's resolution): when `auto_resume` finds no local
  committed checkpoint and a remote URI is set, `latest_committed()` → download to
  `output_dir` (becoming the local committed copy — download to `.tmp` + rename, reusing the
  local commit convention) → resume from it. If both exist, the higher step wins.
- Single-process degradation: no process group → the one process is both uploader and
  finalizer; no barrier needed.

- [ ] **Step 1: Failing e2e test** (single-process, `file://` URI): pilot run with
  `remote_checkpoint_uri` set → remote mirror committed with manifest; wipe `output_dir`
  (simulates NVMe loss on requeue); fresh trainer with `auto_resume` → downloads and resumes at
  the right step; remote rotation honors `keep_every_n_steps`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement (upload manager as a small class inside
  `remote.py`, e.g. `UploadManager(store, accelerator)`, so trainer wiring stays thin).
- [ ] **Step 4:** PASS + CI gates; rerun drain e2e with a remote URI. Commit
  (`feat: background checkpoint upload and remote auto-resume`).

---

## Phase 5 — HSDP

Shaped by Task 0.3's verdict. If the spike showed `torch.optim.Muon` handles DTensor params,
skip Task 5.2.

### Task 5.1: `train.parallelism` knob + FSDP2 wiring

**Files:**
- Modify: `src/oplm/config.py` (`parallelism: str = "ddp"`, validate in
  `{"ddp", "hsdp"}`), `src/oplm/training/trainer.py`, `src/oplm/slurm/render.py`
  (`accelerate_command` drops `--multi_gpu` in favor of plain multi-process launch when hsdp —
  decide per the recorded accelerate-vs-native verdict), `docs/TRAIN.md`
- Test: `tests/training/test_e2e_hsdp.py` (subprocess, slow), config validation test

**Interfaces:**
- Produces: when `parallelism == "hsdp"`, the trainer builds
  `init_device_mesh(device_type, (num_nodes, gpus_per_node), mesh_dim_names=("replicate",
  "shard"))` (dims from `WORLD_SIZE` / `LOCAL_WORLD_SIZE` env; a single-node run degrades to
  `(1, world)` = plain FSDP; world size 1 → refuse with a clear error, use ddp), applies
  `fully_shard(block, mesh=mesh)` per `OplmBlock` + `fully_shard(model, mesh=mesh)` on the
  root, and **does not** pass the model through DDP preparation (record here the exact
  accelerate interplay chosen — spike + implementation decide between
  `accelerator.prepare(model)` with fsdp_version 2 vs preparing only optimizers; the checkpoint
  path is agnostic either way because Phase 2 uses `get_state_dict`).
- Constraint: `torch.compile` ordering (compile after sharding) and gradient-checkpointing
  compatibility must be preserved; the pilot e2e is the gate.

- [ ] **Step 1: Failing test:** 2-process CPU/gloo subprocess pilot with
  `parallelism=hsdp` (mesh (1,2)): trains 3 steps, saves, auto-resumes 2 more, and the saved
  checkpoint loads into a 1-process `ddp` run (Phase 2 reshard test pattern) — proving the
  DDP↔HSDP checkpoint interop claim.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS + CI gates. Commit
  (`feat: hsdp parallelism via FSDP2 device mesh`).

### Task 5.2: Distributed Muon adapter (only if the 0.3 spike requires it)

> **Task 0.3 verdict:** `torch.optim.Muon` works as-is on FSDP2 DTensor params
> (2-process CPU/gloo: no exception, loss decreased, params changed,
> `get_state_dict` round-tripped sharded momentum buffers) — no adapter
> needed; **skip this task**. See spec §7 and
> `.superpowers/sdd/TODOS/task-0.3-report.md`.

**Files:**
- Create: `src/oplm/training/muon_dtensor.py`
- Modify: `src/oplm/training/optim.py` (select adapter when params are DTensor)
- Test: `tests/training/test_fsdp2_muon_spike.py` (un-skip → becomes the regression test)

**Interfaces:**
- Produces: `DTensorMuon(torch.optim.Optimizer)` — per 2-D param: all-gather the sharded
  gradient over the mesh's shard dim (`grad.full_tensor()` on the DTensor), run the same
  momentum + Newton–Schulz update `torch.optim.Muon` applies (reuse
  `torch.optim._muon` helpers where importable; otherwise vendor the ~30-line NS iteration with
  a comment pinning it to the torch implementation), then write back this rank's shard
  (`torch.distributed.tensor.distribute_tensor` slice). Momentum buffers stay sharded
  (allocated like the param) so optimizer-state memory doesn't regress; only the transient
  gathered grad is full-size — per-param, freed each step.
- Consumes: mesh from Task 5.1; `build_optimizers` routing.

- [ ] **Step 1:** Convert the 0.3 spike skip-path into failing assertions (2-proc CPU/gloo:
  loss decreases; single-device `torch.optim.Muon` on the same toy problem and seeds produces
  the same params after 3 steps within `atol=1e-6` — the oracle test).
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS + CI gates. Commit
  (`feat: DTensor-aware Muon for hsdp`).

### Task 5.3: Scale config + docs closeout

**Files:**
- Modify: `configs/scaling.yaml` (document `parallelism` with the 24-30B guidance from spec §2),
  `docs/TRAIN.md`, `docs/OVERVIEW.md` (resilience architecture paragraph), spec (mark §7
  sub-decision resolved)
- Test: `tests/test_docs_links.py`

- [ ] **Step 1:** Write the docs; include the memory table (6B/12B/30B vs DDP/HSDP) and the
  failure-recovery walkthrough (failure → nonzero exit → requeue → auto-resume → same W&B run).
- [ ] **Step 2:** Link check + CI gates. Commit (`docs: fault-tolerance closeout`).

---

## Task Dependencies

```
0.1 → 0.2 → 3.1 → 3.2 → 3.3 → 3.4
0.3 → 5.1 → 5.2 → 5.3
1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6
            1.4 → 1.7        1.3 → 1.8 → 1.9
1.* → 2.1 → 2.2 → 2.3 → 2.4 → (3.3 cursor schema, 4.2 finalization hook)
2.* → 4.1 → 4.2
```

Phase 1 is sequential (each task builds on the previous trainer/render state). Phases 0 and 1
can run in parallel with each other. Phase 3 needs 0.2 and 2.1; Phase 4 needs 2.3; Phase 5
needs 0.3 and 2.2.

# World-Size-Independent Data Cursor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A token-mode checkpoint written at world size N (or worker count W) resumes
data-exactly at a different world size M (or worker count W'): no row skipped or repeated.

**Architecture:** The cursor already records where every old `(rank, worker)` stream stood.
On a layout change, that record becomes a **reshard event**, stored in the cursor and replayed
on every later resume. The dataset gets a cursor for the epoch by removing what each old stream
consumed from the epoch's global order, then striping what is left over the live streams. Events
nest (N→M→K), so a stream is "remaining rows after layers 1..k, striped by the live stride". With
no events this is exactly today's stream. A mix (`InterleavedDataset`) applies the same layers
per source pass and continues with fresh mix draws for the rest of the epoch's draw budget.

**Tech Stack:** Python 3.11+, torch `IterableDataset`, numpy (already a dependency), pyarrow,
accelerate, pytest.

**Spec:** GitHub issue #63 (body + comment). No separate spec doc; the design decisions are
below, and the reviewer should check them against the issue.

## Design decisions (review these first)

1. **Token mode only.** Production runs use token mode (`configs/scaling.yaml`,
   `train.tokens_per_step`). Sequence-mode cursors keep today's same-layout guard: a different
   world size or worker count still raises and names `resume_data_position=false`.
2. **Seed must still match.** The seed defines the epoch order. A seed mismatch still raises.
3. **Mix semantics.** The mix (`InterleavedDataset`) is exact **per source pass**: within each
   source pass, no row is served twice and every row the old streams had not consumed stays
   available. The epoch's total draw count (`num_samples`) is kept. The draws after the resume
   come from new streams, so the mix *order* differs from an uninterrupted run. This can't be
   avoided once the stream count changes.
4. **Ranks in different epochs.** If old stream `j` is already in a later epoch than `e`, all
   of its epoch-`e` rows count as consumed. The uninterrupted run would never have served them
   in epoch `e`. If it is in an earlier epoch, it has consumed nothing of `e`. After a reshard,
   every new rank starts at `min(rank_epochs)`.
5. **Exact per-source position travels with the batch.** `InterleavedDataset` tags each row
   with `(draws_done, ((pass, offset), ...))`. `TokenBatcher` puts the last row's tag in
   `stream_position`. The cursor stores it per worker (`worker_sources`). This replaces
   regenerating old streams' draws, which doesn't work once streams are layered. Token cursors
   from before this change have no `worker_sources`. They still resume a mix at the same layout
   (the existing bincount path), but a reshard of a mix with no `worker_sources` raises.
6. **RNG.** A rank with no sidecar because the checkpoint had fewer ranks is reseeded with
   `cfg.train.seed + rank` and a warning, with no env var. A sidecar missing for
   `rank < sidecar_count` (corruption) still raises unless `OPLM_ALLOW_MISSING_RNG=1`.
7. **`branch_from` compares data by content.** The cursor stores `data_fingerprint`: a sha256
   over each `data.train` entry's fraction and its shards' raw parquet footers, in discovery
   order. `branch_from` keeps the cursor iff the fingerprints match. A cursor without a
   fingerprint falls back to today's raw `data.train` comparison. Plain resume adds no data
   check (out of scope).

## Global Constraints

- `from __future__ import annotations` in every file; type hints on all signatures; Google docstrings on public APIs.
- `pathlib.Path`, never `os.path`; line length 100; no `# type: ignore` without a code.
- `ruff check src/`, `ruff format src/`, `ty check src/` clean.
- Tests run as `.venv/bin/python -m pytest ...`; fast suite `-m "not slow"`; e2e tests are `pytest.mark.slow`.
- With no reshard events, every stream is **byte-identical** to today's (reproducibility contract; the seed-mixing constants don't change).
- Docs: update the home doc in the same task as the behavior; delete superseded text; remove the `> **Pending:** ... (waiting on #63)` marker in docs/training.md; cite docs by heading name.

## Review Focus

- A new stream that gets **zero rows** of a partial epoch (more new streams than remaining rows) yields nothing and rolls over to the next epoch without raising. Pinned in Task 1 (`test_reshard_with_more_streams_than_rows_leaves_some_empty`).
- A checkpoint saved after a reshard but **before a worker produced a batch** (`worker_sources[w] is None`), then resharded again, stays exact. Pinned in Task 2 (`test_nested_reshard_with_an_idle_old_worker`).
- A **same-layout resume of an already-resharded run** (layered stream + skip) is exact. Pinned in Task 2 (`test_same_layout_resume_inside_a_layered_epoch`) and Task 3 (e2e).
- An **old-format token cursor** (no `worker_sources`) resumed into a mix at a new layout fails loudly, not silently wrong. Pinned in Task 3 (`test_reshard_of_a_mix_needs_worker_sources`).
- **Seed change** in token mode still raises after the guard is relaxed. Pinned in Task 3 (`test_token_resume_rejects_a_changed_seed`).

---

### Task 1: Layered, vectorized `ShardedProteinDataset` stream

**Files:**
- Modify: `src/oplm/data/sequence/dataset.py` (`DataCursor` area for `Reshard`; `_iter_stream`, `stream_length`, `_resolved_skip`, `__iter__`, `_iter_pass`; new module helpers)
- Test: `tests/data/sequence/test_reshard.py` (create)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Reshard: stride: int; epochs: list[int]; rows: list[int]; sources: list[list[list[int]] | None] | None = None`. Each per-stream list is in joint-index order (`rank * num_workers + worker`). `sources[j][s] = [pass, offset]`. Exported from `oplm.data`.
  - `Layer = tuple[int, list[int]]`: `(stride, consumed)` for one applicable event, unclamped (the math clamps).
  - `_FULL: int = 2**62`: the "consumed everything" sentinel.
  - `_resolve_layers(reshards: Sequence[Reshard], epoch: int, *, source: int | None = None, pass_index: int = 0) -> list[Layer]`.
  - `_level_sizes(total: int, layers, drop_remainder: bool) -> list[int]`: the size of each level, starting with `total`.
  - `ShardedProteinDataset.set_reshards(reshards: Sequence[Reshard]) -> None`: persists across epochs and is **not** cleared by `clear_resume_skip`.
  - `ShardedProteinDataset.stream_length(layers: Sequence[Layer] | None = None) -> int`
  - `ShardedProteinDataset._iter_pass(pass_index: int, skip: int = 0, layers=None)`, and `_iter_stream(skip, *, pass_index=0, layers=None)`. Here `layers=None` means `_resolve_layers(self._reshards, self._epoch)`.

Math (the only part the signatures leave open):
- Stream lengths of a level with `size` items under `stride`: `ℓ_s = size // stride` if `drop_remainder`, else `len(range(s, size, stride))`. A layer's consumed counts are clamped: `c_s = min(consumed_s, ℓ_s)`.
- An item at level index `x` survives a layer iff `q >= c[s] and q < ℓ[s]`, where `s = x % stride` and `q = x // stride`. Its next-level index is its rank among the survivors. The next level's size is `Σ(ℓ_s − c_s)`.
- Survivors before `x` (O(stride), scalar): `Σ_s max(0, min(n_s(x), ℓ_s) − c_s)`, where `n_s(x) = max(0, ceil((x − s) / stride))`.
- `_resolve_layers`, top level: an event applies iff `min(epochs) <= epoch <= max(epochs)`. Then `consumed_j = _FULL if epochs[j] > epoch, rows[j] if ==, 0 if <`. For a source at `pass_index`: the same epoch test, then for streams in `epoch` use `sources[j][source]`, or `[0, 0]` when `sources[j]` is None: `_FULL if pass > pass_index, offset if ==, 0 if <`. Streams in a later epoch get `_FULL`; streams in an earlier epoch get 0. **Drop an event whose consumed counts are all 0** (an identity layer; keeping it would trim the `drop_remainder` tail twice).
- `_iter_stream`, per shard: the shard's global positions `[base, base+n)` are contiguous at level 0. The survivors of every layer stay contiguous at the next level. So for each layer, build `idx = np.arange(b, b + m)`, apply the survive mask (combining it into a shard-offset mask), set `b = survivors_before(b)` and `m = mask.sum()`. At the live level, `k = np.arange(b, b + m)` and a row is selected iff `k % stride == joint_index and k // stride < limit`, where `limit = stream_length(layers)` computed over the last level size. Map the selected offsets through `torch.randperm(...).numpy()`. Keep the existing skip and no-read-of-skipped-shards logic.

- [ ] **Step 1: Write the failing tests** in `tests/data/sequence/test_reshard.py`. Add a helper `_stream(ds, rank, world, worker, workers, layers=None, skip=0) -> list[str]` that monkeypatches `oplm.data.sequence.dataset._resolve_distributed_context` to `(rank, world, worker, workers)` and returns the ids from `_iter_stream`. Build the shards with `tests.data.conftest.write_sequence_shards` (≈97 rows, 4 shards; an odd count so the strides don't divide it).
  - `test_no_reshards_matches_todays_stream`: for strides 1, 3 and 4, `set_reshards([])` streams equal the ids from `ds._iter_stream(0)` on the pre-change code. Capture those ids once as a literal list for one `(seed, epoch, joint)` and assert equality, with `drop_remainder` both True and False.
  - `test_reshard_is_exact_single_source` (parametrized over `(N, M)` in `[(4, 2), (2, 3), (3, 1), (1, 4)]`): consume ragged prefixes of the N old streams (`rows = [5, 0, 9, 2][:N]`), build `Reshard(stride=N, epochs=[0]*N, rows=rows)`, iterate all M new streams. Assert `Counter(old_consumed + new) == Counter(all_ids)` with `drop_remainder=False`.
  - `test_reshard_straddling_epochs`: old stride 2, `epochs=[0, 1]`, `rows=[3, 4]`. Epoch 0 under stride 3 serves exactly the rows stream 0 hadn't consumed (stream 1's epoch-0 rows count as consumed). Epoch 1 serves every epoch-1 row except stream 1's first 4.
  - `test_nested_reshard_is_exact`: N=4 → M=3 (consume ragged prefixes of the layered streams) → K=2. The union over all three phases is every row exactly once.
  - `test_reshard_with_more_streams_than_rows_leaves_some_empty`: almost everything consumed, M=8. Some new streams are empty, and the union is still exact.
  - `test_reshard_drop_remainder_never_repeats`: with `drop_remainder=True`, there are no duplicates, and the rows missing per level number fewer than that level's stride.
  - `test_reshards_survive_clear_resume_skip`.

- [ ] **Step 2: Run them to verify they fail**
Run: `.venv/bin/python -m pytest tests/data/sequence/test_reshard.py -q`
Expected: FAIL (`ImportError: cannot import name 'Reshard'`).

- [ ] **Step 3: Implement** `Reshard`, the helpers and the layered vectorized `_iter_stream` / `stream_length` described above. Delete the Python per-row selection loop: the vectorized path replaces it, with layers or without.

- [ ] **Step 4: Run the new and existing dataset tests**
Run: `.venv/bin/python -m pytest tests/data -q -m "not slow"`
Expected: PASS, including `test_unarmed_skip_is_byte_identical_to_baseline`, `test_skip_avoids_reading_fully_skipped_shards` and `test_resume_with_two_dataloader_workers_matches_batch_suffix`.

- [ ] **Step 5: Commit**
```bash
git add src/oplm/data tests/data/sequence/test_reshard.py
git commit -m "feat(data): layered reshard events in ShardedProteinDataset streams (#63)"
```

### Task 2: `InterleavedDataset` position tags, layered passes and reshard

**Files:**
- Modify: `src/oplm/data/sequence/dataset.py` (`InterleavedDataset`)
- Modify: `src/oplm/data/sequence/batching.py` (`TokenBatcher`)
- Test: `tests/data/sequence/test_reshard.py`, `tests/data/sequence/test_batching.py`

**Interfaces:**
- Consumes: `Reshard`, `_resolve_layers`, `_level_sizes`, `ShardedProteinDataset._iter_pass(pass_index, skip, layers)` and `stream_length(layers)` (Task 1).
- Produces:
  - `POSITION_KEY = "_stream_position"`: every row `InterleavedDataset` yields carries `row[POSITION_KEY] = (draws_done: int, tuple[tuple[int, int], ...])`, the absolute `(pass, offset)` per source after this draw. The offset counts this stream's rows in the current pass of the *layered* pass stream.
  - `InterleavedDataset.set_reshards(reshards: Sequence[Reshard]) -> None` (persists like Task 1's).
  - `InterleavedDataset.set_resume_sources(sources: Sequence[Sequence[Sequence[int]] | None]) -> None`: per worker, per source `[pass, offset]`. Armed with `set_resume_rows` (which then means *draws*). Cleared by `clear_resume_skip`.
  - `TokenBatcher` batches carry `stream_position = (worker_id, rows_emitted, tag)`, where `tag` is the popped `POSITION_KEY` of the batch's last row, or `None` for a single source. `TokenBatcher.set_resume_sources` and `set_reshards` forward to the wrapped dataset when it has them.

Behavior of `InterleavedDataset.__iter__` for epoch `e`:
- Draw budget: `D = _level_sizes(num_samples, _resolve_layers(self._reshards, e), drop)[-1]`. `n_steps` is the live stream's share of `D` (same formula as today, with `D` in place of `num_samples`). The choices generator and its seed are unchanged.
- Sources:
  - If resume sources are armed for this worker, skip `rows[worker]` draws and start source `s` at `_iter_pass(order_pass(p), offset, layers(e, s, p))`, where `(p, offset) = sources[worker][s]` (`None` → `(start_pass_s, 0)`).
  - Else, if resume rows are armed and no event applies to `e`, use today's bincount path.
  - Else, if resume rows are armed and an event applies, raise `ValueError("... needs per-source positions ...")`.
  - Otherwise start each source at `(start_pass_s, 0)`. `start_pass_s` is the first `p` whose layered size is > 0, searching `p` from 0 up to `1 + max` pass in the applicable events.
- `order_pass(p) = p if reshuffle_on_refill else 0`. A refill moves to `p + 1` with that pass's layers. The tag offset resets to 0 on refill and goes up by one per row served.

- [ ] **Step 1: Write the failing tests**
  - `test_rows_carry_position_tags`: for a 2-source mix, `draws_done` runs 1..n. For each source, `(pass, offset)` matches a hand count over a short stream with `reshuffle_on_refill=True` and a source that refills.
  - `test_mix_reshard_is_exact_per_source_pass` (using `_sources`/`_mix` from `test_refill_and_remainder.py`, moved into `tests/data/conftest.py` or imported): consume ragged draws on old stride 3 and read each worker's last tag. Reshard with `sources` from the tags and iterate stride 2 to the end of the epoch. Assert that (a) the total draws equal `num_samples`, (b) for each source, every pass that new streams finished holds each of the source's rows exactly once, counting old and new, and (c) the last partial pass has no duplicates.
  - `test_same_layout_resume_inside_a_layered_epoch`: after a reshard, consume part of the layered stream, then resume at the same stride via `set_resume_rows` (draws) and `set_resume_sources` (tags). The result is the exact suffix of the uninterrupted layered stream.
  - `test_nested_reshard_with_an_idle_old_worker`: stride 3 → 2 → 4, where one stride-2 worker has a `None` source state at the second event. Each source pass is still exact.
  - `test_layered_resume_without_sources_raises`.
  - In `test_batching.py`, `test_stream_position_carries_the_last_rows_tag`: the batch's tag equals the tag of its last row, and the row deferred to the next batch is not counted.

- [ ] **Step 2: Run them to verify they fail**
Run: `.venv/bin/python -m pytest tests/data/sequence/test_reshard.py tests/data/sequence/test_batching.py -q`
Expected: FAIL (no `POSITION_KEY`, and `stream_position` has 2 fields).

- [ ] **Step 3: Implement** the tags, `set_reshards`, `set_resume_sources`, the layered passes and the `TokenBatcher` changes described above. Keep `_arm_source_skip` only for the bincount path.

- [ ] **Step 4: Run the data suite**
Run: `.venv/bin/python -m pytest tests/data -q -m "not slow"`
Expected: PASS, including `test_resume_rows_continue_each_worker_through_a_reshuffled_mix` and `test_resume_rows_across_reshuffled_refills_is_the_baseline_suffix`.

- [ ] **Step 5: Commit**
```bash
git add src/oplm/data tests/data
git commit -m "feat(data): per-source position tags and layered reshard for mixes (#63)"
```

### Task 3: Trainer cursor: record, reshard on resume, prune

**Files:**
- Modify: `src/oplm/data/sequence/dataset.py` (`DataCursor` fields and docstring)
- Modify: `src/oplm/training/trainer.py` (`__init__` data-cursor state, the loop's `stream_position` read and rollover reset, `_save_checkpoint` gather, `_resume_data_cursor`)
- Modify: `docs/training.md`: *Resuming on different hardware*, the `trainer_state.json` row under *Checkpointing and resume*, the `resume_data_position` knob row under *Fault tolerance*, and the cursor lines under *Token-budget batching*
- Test: `tests/training/test_e2e_token_resume.py`, `tests/training/_token_resume_worker.py`, `tests/data/sequence/test_dataset.py`

**Interfaces:**
- Consumes: `Reshard`, `set_reshards`, `set_resume_rows`, `set_resume_sources`, and the 3-field `stream_position` (Tasks 1–2).
- Produces:
  - New `DataCursor` fields, all defaulted so old dicts still load: `worker_sources: list[list[list[list[int]] | None]] | None = None` (ranks × workers × sources × `[pass, offset]`), `reshards: list[dict] = field(default_factory=list)` (`asdict(Reshard)`), and `data_fingerprint: str | None = None` (filled in Task 5).
  - Trainer state: `self._worker_sources: list[list[list[int]] | None]` (reset to `[None] * workers` on rollover) and `self._reshards: list[Reshard]`.

Behavior:
- Loop: `worker_id, rows, tag = batch["stream_position"]`. Then `_worker_rows[w] = tag[0] if tag else rows` and `_worker_sources[w] = [list(p) for p in tag[1]] if tag else None`.
- Save: gather `(epoch, worker_rows, worker_sources)`. Write `reshards` as `self._reshards`, minus events where `max(epochs) < min(rank_epochs)`.
- `_resume_data_cursor`, token mode:
  - Check the seed. Restore `self._reshards` from `cursor.reshards`.
  - **Same layout** (`world_size` and `num_workers` match): today's behavior, plus `set_reshards`, plus `set_resume_sources(cursor.worker_sources[rank])` when present.
  - **Different layout:** append `Reshard(stride=cursor.world_size * cursor.num_workers, epochs=[rank_epochs[r] for r in ranks for _ in workers], rows=flatten(worker_rows), sources=flatten(worker_sources) or None)`. Set `self.epoch = min(rank_epochs)` on every rank, call `set_reshards`, keep the zeroed `_worker_rows`/`_worker_sources`, and log one INFO line (old → new stream count).
  - Raise `ValueError` naming `resume_data_position=false` if the cursor has no `worker_sources` and `data.train` has more than one entry.
  - A sequence-mode cursor in token mode at a different layout raises as today.
- Delete the `per_rank_batch`/`world_size`/`num_workers` mismatch error for token mode only.

- [ ] **Step 1: Write the failing tests**
  - `test_data_cursor_loads_old_dicts` (extend): a dict without the new fields loads with `worker_sources=None`, `reshards=[]`, `data_fingerprint=None`.
  - In `test_e2e_token_resume.py`, replace `test_token_resume_rejects_a_changed_worker_count` with `test_token_resume_restripes_a_changed_worker_count`: first phase `num_workers=0`, second `num_workers=2`, a 2-source mix, `drop_last=False`. For every epoch that has finished, each sequence id appears exactly once across both phases.
  - `test_token_resume_rejects_a_changed_seed`.
  - `test_reshard_of_a_mix_needs_worker_sources`: strip `worker_sources` from the saved `trainer_state.json`, resume with `num_workers=2`, and expect a `ValueError` that matches `resume_data_position=false`.
  - `test_two_rank_token_reshard` (parametrized over `(first_nproc, second_nproc)` in `[(2, 1), (1, 2)]`): `_launch` takes an `nproc` argument, and the worker accepts `drop_last=false` and a mix. Each finished epoch's ids appear exactly once across both phases. The `(2, 1)` case must have a checkpoint where `rank_epochs` differ, so the straddle is exercised. **This is the issue's "Done when" test.** (`(1, 2)` passes only after Task 4. Mark it `xfail(strict=True)` here and drop the marker in Task 4.)
  - `test_same_layout_resume_after_a_reshard`: phase 2 (resharded) saves, phase 3 resumes at phase 2's layout, and the combined stream matches the exactness check.

- [ ] **Step 2: Run them to verify they fail**
Run: `.venv/bin/python -m pytest tests/training/test_e2e_token_resume.py tests/data/sequence/test_dataset.py -q`
Expected: FAIL (`num_workers` mismatch raises; the new cursor fields are unknown).

- [ ] **Step 3: Implement** the cursor fields and the trainer changes above. Leave the existing sequence-cursor → per-worker-rows mapping as it is.

- [ ] **Step 4: Update docs/training.md.** *Resuming on different hardware* says that a token-mode cursor resumes at any world size and worker count, with the same seed, by re-striping the rows not yet consumed. It gives the mix caveat (exact per source pass, different mix order) and says sequence-mode cursors still need the same layout. Delete the `Pending ... #63` marker. Update the cursor description in the table row and the knob row to match.

- [ ] **Step 5: Run the tests and the docs guard**
Run: `.venv/bin/python -m pytest tests/training/test_e2e_token_resume.py tests/training/test_e2e_checkpoint.py tests/data tests/test_docs.py -q`
Expected: PASS (the `(1, 2)` reshard case XFAIL).

- [ ] **Step 6: Commit**
```bash
git add src/oplm tests docs/training.md
git commit -m "feat(train): token-mode data cursor reshards to any world size (#63)"
```

### Task 4: RNG sidecars at a larger world size

**Files:**
- Modify: `src/oplm/training/checkpoint.py` (`_restore_rng_sidecar`, delete `_validate_rng_sidecar_count` and its call in `validate_checkpoint_for_resume`, docstrings)
- Modify: `docs/training.md` *Resuming on different hardware* (the RNG paragraph)
- Test: `tests/training/test_commit_protocol.py` (the sidecar-count test near line 577), `tests/training/test_e2e_dcp.py`, `tests/training/test_e2e_token_resume.py`

**Interfaces:**
- Produces: `_restore_rng_sidecar(checkpoint_dir: Path, rank: int, seed: int) -> None`. When the sidecar is missing and `rank >= sidecar_count > 0`: log a warning and call `accelerate.utils.set_seed(seed + rank)`. When it is missing otherwise: today's error or env-var behavior.

- [ ] **Step 1: Update the tests.** Rewrite the commit-protocol count test as `test_validate_allows_a_larger_world_size` (no raise at `world_size=3` with 2 sidecars). Add `test_extra_rank_is_reseeded_without_env_var` in `test_e2e_dcp.py`: load at rank 1 of a 1-sidecar checkpoint, expect a warning, and check that `torch.initial_seed() == seed + 1`. Keep `test_load_checkpoint_raises_on_missing_rng_sidecar` (rank 0 missing still raises). Drop the `xfail` from `test_two_rank_token_reshard[1-2]`.

- [ ] **Step 2: Run them to verify they fail**
Run: `.venv/bin/python -m pytest tests/training/test_commit_protocol.py tests/training/test_e2e_dcp.py -q -k "rng or sidecar or world_size"`
Expected: FAIL.

- [ ] **Step 3: Implement**, and update the RNG paragraph in the docs: a larger world size reseeds the extra ranks, and the env var is only for a sidecar missing below the sidecar count.

- [ ] **Step 4: Run the tests**
Run: `.venv/bin/python -m pytest tests/training/test_commit_protocol.py tests/training/test_e2e_dcp.py tests/training/test_e2e_token_resume.py tests/training/test_resume_target_broadcast.py tests/test_docs.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/oplm/training/checkpoint.py tests docs/training.md
git commit -m "feat(checkpoint): reseed extra ranks when resuming at a larger world size (#63)"
```

### Task 5: `branch_from` keeps the cursor when the data matches by content

**Files:**
- Modify: `src/oplm/data/sequence/dataset.py` (new `train_data_fingerprint`)
- Modify: `src/oplm/training/trainer.py` (compute once in `__init__`, write into the cursor, compare in `_branch_from_checkpoint`)
- Modify: `docs/training.md` *Branching a decay off a checkpoint* → *Rules* (the `data.train` comparison)
- Test: `tests/data/sequence/test_dataset.py`, `tests/training/test_e2e_branch.py`

**Interfaces:**
- Produces: `train_data_fingerprint(raw_train: Any) -> str`, hex sha256. For each entry of `parse_train_configs(raw_train)` in order, hash `repr(entry.fraction)`, then the raw footer bytes of each shard from `ShardedProteinDataset._discover_shards(Path(entry.path))`. The footer is the last `8 + n` bytes of the file minus the trailing 8, where `n` is the little-endian uint32 at `-8`.

- [ ] **Step 1: Write the failing tests**
  - `test_fingerprint_ignores_location`: copy the shard dir to another path and the fingerprints are equal.
  - `test_fingerprint_sees_content_and_fraction`: rewriting one shard with the rows in a different order changes it, and so does changing a fraction.
  - In `test_e2e_branch.py`, `test_branch_keeps_cursor_for_data_at_a_new_path`: branch from a checkpoint with `data.train` pointing at a copied directory. The cursor is kept, with no "differs" warning in caplog.

- [ ] **Step 2: Run them to verify they fail**
Run: `.venv/bin/python -m pytest tests/data/sequence/test_dataset.py tests/training/test_e2e_branch.py -q -k "fingerprint or new_path"`
Expected: FAIL.

- [ ] **Step 3: Implement.** In `_branch_from_checkpoint`, when the cursor has a `data_fingerprint`, compare it with the live one. Otherwise keep the current `config.yaml` comparison. Update the *Rules* bullet.

- [ ] **Step 4: Run the full fast suite, the slow resume suites and the linters**
Run: `.venv/bin/python -m pytest -m "not slow" -q && .venv/bin/python -m pytest tests/training/test_e2e_token_resume.py tests/training/test_e2e_data_exact.py tests/training/test_e2e_branch.py tests/training/test_e2e_checkpoint.py -q && ruff check src/ && ruff format --check src/ && ty check src/`
Expected: all PASS and clean.

- [ ] **Step 5: Commit**
```bash
git add src/oplm tests docs/training.md
git commit -m "feat(train): branch_from compares training data by content fingerprint (#63)"
```

The PR body gets `Closes #63`.

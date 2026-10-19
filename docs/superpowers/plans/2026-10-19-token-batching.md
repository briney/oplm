# Token-Budget Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a run fix its global token count per optimizer step (`train.tokens_per_step`, with each GPU's micro-batch capacity `train.batch_tokens`), with fixed `rows × W` micro-batches, data-exact resume from per-worker row counts, and per-GPU epoch rollover.

**Architecture:** A `TokenBatcher` dataset wraps the existing training dataset inside each DataLoader worker. It fills `R` rows of width `W` first-fit in stream order, closes a batch at the first sequence that fits nowhere, masks and packs with the existing collator, and tags each batch with `(worker_id, rows_emitted)`. The trainer derives gradient accumulation from the token budget, records each worker's position from the batches it consumes, and saves a per-GPU, per-worker position table in the checkpoint cursor. On resume each worker skips exactly its recorded rows.

**Tech Stack:** PyTorch 2.11 `DataLoader`/`IterableDataset`, Hugging Face Accelerate 1.15, OmegaConf dataclass configs, pytest (CPU, gloo for two ranks).

**Spec:** `docs/superpowers/specs/2026-10-19-token-batching-flex-attention-design.md` (§5 and §6 are this plan; §3 lists what is out of scope).

## Global Constraints

- Run everything through the repo venv: `.venv/bin/python -m pytest`, `.venv/bin/ruff`, `VIRTUAL_ENV=$PWD/.venv .venv/bin/ty check src/`.
- `ruff format` only the files you touched (`main` has unrelated drift); never commit churn in other files.
- Token mode is on when **both** `train.tokens_per_step` and `train.batch_tokens` are set; setting one is an error. With both unset, every existing code path must behave exactly as before.
- `gradient_accumulation_steps = tokens_per_step / (batch_tokens × world_size)`, computed once the world size is known, written into `cfg.train.gradient_accumulation_steps` and `accelerator.gradient_accumulation_steps`.
- Token length of a sequence: `min(len(sequence), W - 2) + 2`, where `W = model.max_position_embeddings`.
- A batch closes at the **first** sequence that fits in no row; that sequence starts the next batch. Each batch is a contiguous run of its worker's stream.
- Every token-mode micro-batch is exactly `(R, W)` with `R = batch_tokens // W`.
- The final partial batch of a worker's stream is dropped when `data.drop_last` is true (default) and emitted otherwise.
- Resume in token mode is data-exact per worker; the layout check compares only `world_size`, `num_workers`, `seed`, so changing `tokens_per_step`/`batch_tokens` at a resume or branch keeps the position.
- `_SaveDeferral` is disabled in token mode (a save entered by only some ranks deadlocks).
- `train.batch_size` still sizes eval batches. Eval, the fold head and the sweep/Slurm tooling are untouched.
- FlexAttention (the other plan) is independent: nothing here requires `model.packed_attention="flex"`, and CPU tests must use the default `"sdpa"` (FlexAttention has no CPU backward).
- Commit messages use `type(scope): summary` and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**Rulings made while planning (each consistent with the spec's intent):**

1. *Accumulation in saved configs.* The spec says an explicit `gradient_accumulation_steps` other than 1 is an error, and also that the saved `config.yaml` records the derived value. Both hold if the check happens at startup: `resolve_token_accumulation` accepts 1 **or** the derived value, and rejects anything else. A saved token-mode config then reloads at the same world size; at a different world size the error says to set it to 1. Cost if wrong: one confusing error message.
2. *stream-replay guard.* The spec says stream-replay raises on checkpoints whose cursor has `worker_rows`. `StreamReplay` reads only the checkpoint's config, so it checks `cfg.train.batch_tokens` instead (equivalent, and it fails before any data is touched).
3. *Empty epochs.* A rank whose stream yields no full batch in a whole epoch (tiny data, huge `batch_tokens`, `drop_last`) would raise a bare `StopIteration` from the training loop. The rollover now raises a `RuntimeError` naming the cause. It applies in both modes and changes nothing for runs that work today.

## Review Focus

1. **A rank or worker whose epoch yields no full token batch** → a `RuntimeError` naming `train.batch_tokens`/`data.drop_last`, not a bare `StopIteration`. Pinned by `test_token_mode_empty_epoch_raises_clearly` (Task 5).
2. **Resuming a sequence-mode checkpoint into a token-mode run, or the reverse** → `ValueError` naming `train.resume_data_position=false`. Pinned by `test_resume_across_batching_modes_raises` (Task 6).
3. **A saved token-mode `config.yaml` reloaded** (its `gradient_accumulation_steps` is the derived value) → accepted at the same world size, a clear error at a different one. Pinned by `test_resolve_token_accumulation_accepts_saved_value_and_rejects_others` (Task 1).
4. **Sequences longer than `W - 2` residues** → token length `W`, a full row of their own, no crash. Pinned by `test_long_sequence_takes_a_whole_row` (Task 4).
5. **Checkpoints written before this change** (cursor without `rank_epochs`/`worker_rows`) → load as a sequence-mode cursor. Pinned by `test_data_cursor_loads_old_dicts` (Task 6).

---

### Task 1: Config fields, cross-section validation, accumulation

**Files:**
- Modify: `src/oplm/config.py` (`TrainConfig` fields under `# Batch` ~line 44 and `__post_init__` ~line 211; new `validate_token_batching` and `resolve_token_accumulation` after `validate_parallelism_compat` ~line 597; `load_config` call site ~line 853)
- Modify: `src/oplm/configs/train/base.yaml` (batch block ~line 15)
- Modify: `src/oplm/training/trainer.py` (import + call next to `validate_parallelism_compat`, ~lines 254 and 270)
- Modify: `tests/training/conftest.py` (`tiny_train_cfg` gains four kwargs)
- Test: `tests/training/test_config.py`

**Interfaces:**
- Produces:
  - `TrainConfig.tokens_per_step: int | None = None`, `TrainConfig.batch_tokens: int | None = None`;
  - `oplm.config.validate_token_batching(cfg: OplmConfig) -> None`;
  - `oplm.config.resolve_token_accumulation(train: TrainConfig, world_size: int) -> int`;
  - `tiny_train_cfg(..., tokens_per_step=None, batch_tokens=None, pack_sequences=False, drop_last=True)`.

- [ ] **Step 1: Extend `tiny_train_cfg`** (`tests/training/conftest.py`): add keyword parameters `tokens_per_step: int | None = None`, `batch_tokens: int | None = None`, `pack_sequences: bool = False`, `drop_last: bool = True`; pass the first two into `TrainConfig(...)` and the last two into `DataConfig(...)`. (`DataConfig.pack_sequences` and `DataConfig.drop_last` already exist.)

- [ ] **Step 2: Write the failing tests** (append to `tests/training/test_config.py`)

```python
from oplm.config import resolve_token_accumulation, validate_token_batching
from tests.training.conftest import tiny_train_cfg


def test_token_mode_needs_both_fields() -> None:
    with pytest.raises(ValueError, match="set together"):
        TrainConfig(tokens_per_step=4096)
    with pytest.raises(ValueError, match="set together"):
        TrainConfig(batch_tokens=4096)


def test_token_mode_rejects_non_positive_budgets() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        TrainConfig(tokens_per_step=0, batch_tokens=4096)


def test_token_mode_tokens_per_step_must_be_a_multiple_of_batch_tokens() -> None:
    with pytest.raises(ValueError, match="multiple of train.batch_tokens"):
        TrainConfig(tokens_per_step=6000, batch_tokens=4096)


def test_token_mode_rejects_max_epochs() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        TrainConfig(tokens_per_step=8192, batch_tokens=4096, max_epochs=1)


def test_validate_token_batching(tmp_path: Path) -> None:
    data = tmp_path / "unused.parquet"
    ok = tiny_train_cfg(tmp_path, data, tokens_per_step=256, batch_tokens=256, pack_sequences=True)
    validate_token_batching(ok)  # max_position_embeddings is 64 in tiny_train_cfg
    validate_token_batching(tiny_train_cfg(tmp_path, data))  # sequence mode: nothing to check
    bad_width = tiny_train_cfg(
        tmp_path, data, tokens_per_step=400, batch_tokens=200, pack_sequences=True
    )
    with pytest.raises(ValueError, match="multiple of model.max_position_embeddings"):
        validate_token_batching(bad_width)
    unpacked = tiny_train_cfg(tmp_path, data, tokens_per_step=256, batch_tokens=256)
    with pytest.raises(ValueError, match="data.pack_sequences"):
        validate_token_batching(unpacked)


def test_resolve_token_accumulation() -> None:
    train = TrainConfig(tokens_per_step=4 * 8192, batch_tokens=8192)
    assert resolve_token_accumulation(train, 1) == 4
    assert resolve_token_accumulation(train, 2) == 2
    assert resolve_token_accumulation(train, 4) == 1
    for world_size in (3, 8):  # not a divisor; more capacity per step than the budget
        with pytest.raises(ValueError, match="multiple of train.batch_tokens x world size"):
            resolve_token_accumulation(train, world_size)


def test_resolve_token_accumulation_accepts_saved_value_and_rejects_others() -> None:
    # A token-mode run's saved config.yaml records the derived accumulation (2 at world size 2).
    saved = TrainConfig(tokens_per_step=4 * 8192, batch_tokens=8192, gradient_accumulation_steps=2)
    assert resolve_token_accumulation(saved, 2) == 2
    with pytest.raises(ValueError, match="leave it at 1"):
        resolve_token_accumulation(saved, 1)  # derived 4 here, so the saved 2 is stale
```

If `Path` is only imported under `TYPE_CHECKING` in that file, that is fine for annotations (`from __future__ import annotations` is on).

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/training/test_config.py -k "token" -v`
Expected: FAIL (`ImportError: cannot import name 'resolve_token_accumulation'`).

- [ ] **Step 4: Implement**

`TrainConfig`, under `# Batch` after `gradient_accumulation_steps: int = 1`:

```python
    # Token mode (set both, or neither): global token capacity per optimizer step, and each
    # process's micro-batch capacity (rows x model.max_position_embeddings). Micro-batches are
    # then fixed (rows, width) packed batches and gradient_accumulation_steps is derived at
    # startup from the world size. train.batch_size still sizes eval batches.
    tokens_per_step: int | None = None
    batch_tokens: int | None = None
```

`TrainConfig.__post_init__`, after the `gradient_accumulation_steps` check:

```python
        if (self.tokens_per_step is None) != (self.batch_tokens is None):
            raise ValueError(
                "train.tokens_per_step and train.batch_tokens must be set together (token mode) "
                f"or both left unset; got tokens_per_step={self.tokens_per_step}, "
                f"batch_tokens={self.batch_tokens}"
            )
        if self.tokens_per_step is not None and self.batch_tokens is not None:
            if self.tokens_per_step < 1 or self.batch_tokens < 1:
                raise ValueError(
                    "train.tokens_per_step and train.batch_tokens must be >= 1, got "
                    f"{self.tokens_per_step} and {self.batch_tokens}"
                )
            if self.tokens_per_step % self.batch_tokens:
                raise ValueError(
                    f"train.tokens_per_step ({self.tokens_per_step}) must be a multiple of "
                    f"train.batch_tokens ({self.batch_tokens})"
                )
            if self.max_epochs is not None:
                raise ValueError(
                    "token mode (train.batch_tokens) needs train.max_steps: sequences per step "
                    "vary, so train.max_epochs cannot be converted to steps"
                )
```

After `validate_parallelism_compat`:

```python
def validate_token_batching(cfg: OplmConfig) -> None:
    """Reject token-mode configs whose train, data and model sections disagree.

    ``TrainConfig.__post_init__`` checks the train-only rules; these need ``cfg.model`` and
    ``cfg.data`` too. Called from :func:`load_config` and again in ``Trainer.__init__``, like
    :func:`validate_parallelism_compat`. A no-op outside token mode.

    Raises:
        ValueError: ``train.batch_tokens`` is not a multiple of ``model.max_position_embeddings``,
            or ``data.pack_sequences`` is off.
    """
    if cfg.train.batch_tokens is None:
        return
    width = int(cfg.model.max_position_embeddings)
    if cfg.train.batch_tokens % width:
        raise ValueError(
            f"train.batch_tokens ({cfg.train.batch_tokens}) must be a multiple of "
            f"model.max_position_embeddings ({width}): each micro-batch is "
            "batch_tokens / max_position_embeddings rows of that width"
        )
    if not cfg.data.pack_sequences:
        raise ValueError("token mode (train.batch_tokens) requires data.pack_sequences=true")


def resolve_token_accumulation(train: TrainConfig, world_size: int) -> int:
    """Gradient accumulation that makes ``batch_tokens x world_size x accumulation`` equal
    ``tokens_per_step``.

    Accepts a configured ``gradient_accumulation_steps`` of 1 or of the derived value (a token
    run's saved ``config.yaml`` records the value it used), and rejects anything else.

    Raises:
        ValueError: ``tokens_per_step`` is not a multiple of ``batch_tokens x world_size``, or the
            configured accumulation is neither 1 nor the derived value.
    """
    assert train.tokens_per_step is not None and train.batch_tokens is not None
    per_step = train.batch_tokens * world_size
    if train.tokens_per_step % per_step:
        raise ValueError(
            f"train.tokens_per_step ({train.tokens_per_step}) must be a multiple of "
            f"train.batch_tokens x world size ({train.batch_tokens} x {world_size} = {per_step}); "
            "change tokens_per_step, batch_tokens or the number of processes"
        )
    accumulation = train.tokens_per_step // per_step
    if train.gradient_accumulation_steps not in (1, accumulation):
        raise ValueError(
            "token mode derives train.gradient_accumulation_steps "
            f"({accumulation} = tokens_per_step / (batch_tokens x world size)); leave it at 1 "
            f"(got {train.gradient_accumulation_steps})"
        )
    return accumulation
```

In `load_config`, after `validate_parallelism_compat(cfg)`: `validate_token_batching(cfg)`.
In `Trainer.__init__`, import `validate_token_batching` alongside `validate_parallelism_compat` and call it right after `validate_parallelism_compat(cfg)`.

`src/oplm/configs/train/base.yaml`, after `gradient_accumulation_steps: 1`:

```yaml
  # Token mode: set both (global tokens per optimizer step; per-process micro-batch tokens,
  # a multiple of model.max_position_embeddings). Needs data.pack_sequences: true.
  tokens_per_step: null
  batch_tokens: null
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/training/test_config.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
.venv/bin/ruff check src/oplm/config.py src/oplm/training/trainer.py tests/training/test_config.py tests/training/conftest.py
.venv/bin/ruff format src/oplm/config.py tests/training/test_config.py tests/training/conftest.py
git add src/oplm/config.py src/oplm/configs/train/base.yaml src/oplm/training/trainer.py tests/training/test_config.py tests/training/conftest.py
git commit -m "feat(config): train.tokens_per_step and train.batch_tokens select token mode

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: First-fit placement and the collator's fixed-shape packing

**Files:**
- Modify: `src/oplm/data/sequence/collate.py` (new `RowFiller`, `first_fit`, `token_length` above `MLMCollator`; `MLMCollator.__init__`, `__call__`, `_pack`)
- Test: `tests/data/sequence/test_collate.py`

**Interfaces:**
- Produces:
  - `RowFiller(rows: int, width: int)` with `.place(length: int) -> bool` and `.placement: list[int]`;
  - `first_fit(lengths: Sequence[int], rows: int, width: int) -> list[int] | None`;
  - `token_length(sequence: str, width: int) -> int`;
  - `MLMCollator(..., pack_sequences=True, pack_rows: int | None = None)`: with `pack_rows`, output tensors are exactly `(pack_rows, max_length)`, rows laid out by `first_fit` in input order, plus `"num_sequences"`.

- [ ] **Step 1: Write the failing tests** (append to `tests/data/sequence/test_collate.py`; reuse the file's tokenizer fixture/helper, e.g. `get_tokenizer()`)

```python
import oplm.data.sequence.collate as collate_mod
from oplm.data.sequence.collate import MLMCollator, RowFiller, first_fit, token_length


def test_token_length_counts_specials_and_truncates() -> None:
    assert token_length("ACDE", 16) == 6
    assert token_length("A" * 14, 16) == 16
    assert token_length("A" * 40, 16) == 16


def test_first_fit_places_in_order_into_the_first_row_with_room() -> None:
    assert first_fit([12, 12, 4, 3], rows=2, width=16) == [0, 1, 0, 1]
    assert first_fit([16, 16], rows=2, width=16) == [0, 1]
    assert first_fit([12, 12, 7], rows=2, width=16) is None


def test_row_filler_refuses_without_placing() -> None:
    filler = RowFiller(1, 10)
    assert filler.place(6) and not filler.place(5) and filler.place(4)
    assert filler.placement == [0, 0]


def _seqs() -> list[dict[str, str]]:
    lengths = (10, 3, 12, 5, 1)
    return [{"sequence_id": f"s{i}", "sequence": "ACDEFGHIKLMNPQ"[:n]} for i, n in enumerate(lengths)]


def test_fixed_shape_pack_is_exactly_rows_by_width() -> None:
    collator = MLMCollator(get_tokenizer(), max_length=16, pack_sequences=True, pack_rows=4)
    out = collator(_seqs())
    assert out["input_ids"].shape == out["segment_ids"].shape == (4, 16)
    assert out["num_sequences"] == 5
    lengths = [token_length(s["sequence"], 16) for s in _seqs()]  # 12, 5, 14, 7, 3
    placement = first_fit(lengths, 4, 16)
    assert placement == [0, 1, 2, 1, 0]
    for row in range(4):
        expected = sum(n for n, r in zip(lengths, placement, strict=True) if r == row)
        assert int(out["attention_mask"][row].sum()) == expected


def test_fixed_shape_pack_keeps_the_unpacked_masks() -> None:
    kwargs = dict(max_length=16, deterministic=True, seed=3)
    plain = MLMCollator(get_tokenizer(), **kwargs)(_seqs())
    packed = MLMCollator(get_tokenizer(), pack_sequences=True, pack_rows=4, **kwargs)(_seqs())
    lengths = plain["attention_mask"].sum(1).tolist()
    placement = first_fit(lengths, 4, 16)
    offsets = [0] * 4
    for i, (n, row) in enumerate(zip(lengths, placement, strict=True)):
        start = offsets[row]
        for key in ("input_ids", "labels"):
            assert packed[key][row, start : start + n].tolist() == plain[key][i, :n].tolist()
        offsets[row] += n


def test_fixed_shape_pack_raises_when_sequences_do_not_fit() -> None:
    collator = MLMCollator(get_tokenizer(), max_length=16, pack_sequences=True, pack_rows=1)
    with pytest.raises(ValueError, match="do not fit"):
        collator(_seqs())


def test_pack_rows_requires_pack_sequences() -> None:
    with pytest.raises(ValueError, match="pack_rows requires pack_sequences"):
        MLMCollator(get_tokenizer(), max_length=16, pack_rows=2)


def test_fixed_shape_pack_checks_token_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collate_mod, "token_length", lambda seq, width: 1)
    collator = MLMCollator(get_tokenizer(), max_length=16, pack_sequences=True, pack_rows=4)
    with pytest.raises(ValueError, match="token_length"):
        collator(_seqs())
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/sequence/test_collate.py -k "token_length or first_fit or row_filler or fixed_shape or pack_rows" -v`
Expected: FAIL (`ImportError: cannot import name 'RowFiller'`).

- [ ] **Step 3: Implement** in `src/oplm/data/sequence/collate.py`, above `class MLMCollator`:

```python
class RowFiller:
    """First-fit placement of token lengths, in order, into ``rows`` rows of ``width`` tokens.

    The token batcher (:class:`~oplm.data.sequence.batching.TokenBatcher`) and the collator's
    fixed-shape packing both place sequences through this class, so they cannot disagree about
    where a batch closes or which row a sequence lands in.
    """

    def __init__(self, rows: int, width: int) -> None:
        self._free = [width] * rows
        self.placement: list[int] = []

    def place(self, length: int) -> bool:
        """Put ``length`` in the first row with room; return ``False``, placing nothing, if none."""
        for row, room in enumerate(self._free):
            if length <= room:
                self._free[row] -= length
                self.placement.append(row)
                return True
        return False


def first_fit(lengths: Sequence[int], rows: int, width: int) -> list[int] | None:
    """Each length's row under :class:`RowFiller`, or ``None`` if one fits in no row."""
    filler = RowFiller(rows, width)
    return filler.placement if all(filler.place(n) for n in lengths) else None


def token_length(sequence: str, width: int) -> int:
    """Tokens a sequence takes in a row of ``width``: residues (at most ``width - 2``) + 2 specials."""
    return min(len(sequence), width - 2) + 2
```

`MLMCollator.__init__`: add keyword `pack_rows: int | None = None`; document it in the Args (`With pack_sequences: pack into exactly this many rows of max_length (token mode), laid out by first_fit in input order; raises if the sequences do not fit. None keeps first-fit-decreasing into as few rows as needed.`); validate and store:

```python
        if pack_rows is not None and (not pack_sequences or pack_rows < 1):
            raise ValueError(f"pack_rows requires pack_sequences=True and pack_rows >= 1, got {pack_rows}")
        self._pack_rows = pack_rows
```

`MLMCollator.__call__`, replace `if self._pack_sequences: out = self._pack(...)` with:

```python
        if self._pack_sequences:
            if self._pack_rows is not None:
                expected = [token_length(_sequence_of(item), self._max_length) for item in batch]
                if attention_mask.sum(1).tolist() != expected:
                    raise ValueError(
                        "tokenized lengths differ from token_length(); the token batcher's "
                        "placement no longer matches the batch"
                    )
            out = self._pack(input_ids, attention_mask, labels)
```

`MLMCollator._pack`: compute `rows` and `width` per mode, then keep the shared fill loop:

```python
        lengths = attention_mask.sum(1).tolist()
        if self._pack_rows is not None:
            placement = first_fit(lengths, self._pack_rows, self._max_length)
            if placement is None:
                raise ValueError(
                    f"{len(lengths)} sequences ({sum(lengths)} tokens) do not fit "
                    f"{self._pack_rows} rows of {self._max_length}"
                )
            rows = [[i for i, r in enumerate(placement) if r == row] for row in range(self._pack_rows)]
            width = self._max_length
        else:
            rows, free = [], []
            for i in sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True):
                fit = next((r for r, room in enumerate(free) if lengths[i] <= room), None)
                if fit is None:
                    rows.append([i])
                    free.append(self._max_length - lengths[i])
                else:
                    rows[fit].append(i)
                    free[fit] -= lengths[i]
            width = self._max_length - min(free)
            if self._pad_to_multiple_of:
                width = -(-width // self._pad_to_multiple_of) * self._pad_to_multiple_of
        shape = (len(rows), width)
        # ... the existing `packed = {...}` allocation and fill loop, unchanged ...
```

Keep the existing type annotations on `rows`/`free` (`list[list[int]]`, `list[int]`). Update `pack_sequences`' docstring line to mention `pack_rows`.

- [ ] **Step 4: Run the collator tests**

Run: `.venv/bin/python -m pytest tests/data/sequence/test_collate.py tests/model/test_packing.py tests/training/test_packing_train.py -q`
Expected: all PASS (the existing packing path is unchanged).

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check src/oplm/data/sequence/collate.py tests/data/sequence/test_collate.py
.venv/bin/ruff format src/oplm/data/sequence/collate.py
git add src/oplm/data/sequence/collate.py tests/data/sequence/test_collate.py
git commit -m "feat(data): first-fit row placement and fixed-shape packing in MLMCollator

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(`test_collate.py` is one of the files drifted on `main`; check it with `ruff check` but do not `ruff format` it.)

---

### Task 3: Resume datasets by per-worker row counts

**Files:**
- Modify: `src/oplm/data/sequence/dataset.py` (new `_resolve_rows_skip` after `_resolve_sample_skip` ~line 145; `ShardedProteinDataset.__init__`, `clear_resume_skip`, `_resolved_skip`, new `set_resume_rows`; `InterleavedDataset.__init__`, `clear_resume_skip`, `__iter__`, new `set_resume_rows`)
- Test: `tests/data/sequence/test_dataset.py`, `tests/data/sequence/test_refill_and_remainder.py`

**Interfaces:**
- Produces: `set_resume_rows(rows_per_worker: Sequence[int]) -> None` on `ShardedProteinDataset` and `InterleavedDataset`. Each worker skips `rows_per_worker[worker_id]` rows of its stream (clamped to the stream length); `clear_resume_skip()` clears it along with the batch-count skip. A list whose length is not the worker count raises `ValueError` at iteration.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/sequence/test_dataset.py` (it has `_set_context` and `_ids`; import `write_sequence_shards` from `tests.data.conftest`):

```python
from tests.data.conftest import write_sequence_shards


def _rows_dataset(tmp_path: Path) -> ShardedProteinDataset:
    shard_dir = tmp_path / "rows"
    shard_dir.mkdir()
    records = [(f"{i}", "ACDEFGHIKL"[: 3 + i % 7]) for i in range(40)]
    return ShardedProteinDataset(write_sequence_shards(shard_dir, records, n_shards=3), seed=0)


def test_set_resume_rows_skips_each_workers_own_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _rows_dataset(tmp_path)
    baseline = {}
    for worker in range(2):
        _set_context(monkeypatch, 0, 1, worker, 2)
        baseline[worker] = _ids(ds)
    ds.set_resume_rows([3, 7])
    for worker, skip in enumerate((3, 7)):
        _set_context(monkeypatch, 0, 1, worker, 2)
        assert _ids(ds) == baseline[worker][skip:]
    ds.clear_resume_skip()
    _set_context(monkeypatch, 0, 1, 1, 2)
    assert _ids(ds) == baseline[1]


def test_set_resume_rows_clamps_to_the_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _rows_dataset(tmp_path)
    ds.set_resume_rows([1000, 0])
    _set_context(monkeypatch, 0, 1, 0, 2)
    assert _ids(ds) == []


def test_set_resume_rows_needs_one_entry_per_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _rows_dataset(tmp_path)
    ds.set_resume_rows([1, 2, 3])
    _set_context(monkeypatch, 0, 1, 0, 2)
    with pytest.raises(ValueError, match="3 entries for 2 DataLoader workers"):
        _ids(ds)
```

Append to `tests/data/sequence/test_refill_and_remainder.py`:

```python
def test_resume_rows_across_reshuffled_refills_is_the_baseline_suffix(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    baseline = _ids(_mix(sources, reshuffle=True))
    for skip in (3, 10, 27, 61, len(baseline) - 1):  # lands in the first and later passes
        resumed = _mix(sources, reshuffle=True)
        resumed.set_resume_rows([skip])
        assert _ids(resumed) == baseline[skip:], skip
```

If `write_sequence_shards` writes into `directory` and returns `None` rather than the path, pass `shard_dir` to `ShardedProteinDataset` instead (the refill test's `_shards` helper shows the return value is used).

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/sequence/test_dataset.py tests/data/sequence/test_refill_and_remainder.py -k resume_rows -v`
Expected: FAIL (`AttributeError: ... has no attribute 'set_resume_rows'`).

- [ ] **Step 3: Implement** in `src/oplm/data/sequence/dataset.py`

After `_resolve_sample_skip`:

```python
def _resolve_rows_skip(rows_per_worker: Sequence[int], stream_length: int) -> int:
    """This worker's entry of an armed per-worker row-count skip, clamped to its stream.

    Token-mode resume (:meth:`ShardedProteinDataset.set_resume_rows`): the checkpoint records how
    many rows each worker had emitted, so no batch-count arithmetic is involved. Clamped, never
    wrapped, for the reason given in :func:`_resolve_sample_skip`.

    Raises:
        ValueError: ``rows_per_worker`` does not have one entry per DataLoader worker.
    """
    _, _, worker_id, num_workers = _resolve_distributed_context()
    if len(rows_per_worker) != num_workers:
        raise ValueError(
            f"resume rows have {len(rows_per_worker)} entries for {num_workers} DataLoader "
            "workers; the checkpoint's worker count must match data.num_workers"
        )
    return min(int(rows_per_worker[worker_id]), max(stream_length, 0))
```

`ShardedProteinDataset.__init__`, after `self._resume_num_workers = 1`:

```python
        # Token-mode resume skip: rows already emitted, one entry per DataLoader worker.
        self._resume_rows: list[int] | None = None
```

New method after `set_resume_skip`:

```python
    def set_resume_rows(self, rows_per_worker: Sequence[int]) -> None:
        """Arm a one-epoch skip of ``rows_per_worker[worker_id]`` rows in each worker's stream.

        The token-mode counterpart of :meth:`set_resume_skip`: batches hold a varying number of
        rows, so the cursor records rows per worker directly. Pickles into workers the same way;
        cleared by :meth:`clear_resume_skip`.
        """
        self._resume_rows = list(rows_per_worker)
```

`clear_resume_skip`: also set `self._resume_rows = None` (update its docstring: "Disarm both resume skips").

`_resolved_skip`, before the existing return:

```python
        if self._resume_rows is not None:
            return _resolve_rows_skip(self._resume_rows, self.stream_length())
```

`InterleavedDataset`: same `self._resume_rows: list[int] | None = None` in `__init__`, the same `set_resume_rows` method (docstring: per-worker row counts are draws from the mix, resolved through `_arm_source_skip` like the batch-count skip), `clear_resume_skip` clears it too, and in `__iter__` replace the `skip = _resolve_sample_skip(...)` assignment with:

```python
        if self._resume_rows is not None:
            skip = _resolve_rows_skip(self._resume_rows, n_steps)
        else:
            skip = _resolve_sample_skip(
                self._resume_batches_in_epoch,
                self._resume_per_rank_batch,
                self._resume_num_workers,
                n_steps,
            )
```

- [ ] **Step 4: Run the data suite**

Run: `.venv/bin/python -m pytest tests/data -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check src/oplm/data/sequence/dataset.py tests/data/sequence/test_dataset.py tests/data/sequence/test_refill_and_remainder.py
.venv/bin/ruff format src/oplm/data/sequence/dataset.py tests/data/sequence/test_dataset.py tests/data/sequence/test_refill_and_remainder.py
git add src/oplm/data/sequence/dataset.py tests/data/sequence/test_dataset.py tests/data/sequence/test_refill_and_remainder.py
git commit -m "feat(data): resume sequence datasets from per-worker row counts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `TokenBatcher` and the token-mode data loader

**Files:**
- Create: `src/oplm/data/sequence/batching.py`
- Modify: `src/oplm/data/sequence/loaders.py` (`build_train_dataloader`, `build_sequence_eval_dataloader`, `_build_dataloader`)
- Modify: `src/oplm/data/__init__.py` (export `TokenBatcher` if the package re-exports dataset classes; check its `__all__`)
- Test: `tests/data/sequence/test_batching.py` (new)

**Interfaces:**
- Consumes: `RowFiller`, `token_length`, `MLMCollator(pack_rows=...)` (Task 2); `set_resume_rows` (Task 3); `tiny_train_cfg(tokens_per_step=..., batch_tokens=..., pack_sequences=...)` (Task 1).
- Produces: `TokenBatcher(dataset, collator, *, rows: int, width: int, drop_last: bool = True)`, an `IterableDataset` yielding batch dicts with `input_ids`, `attention_mask`, `labels`, `segment_ids` (each `(rows, width)`), `num_sequences: int`, `stream_position: (worker_id, rows_emitted)`; methods `set_epoch`, `set_resume_rows`, `clear_resume_skip`; property `total_length`. In token mode `build_train_dataloader` returns `DataLoader(TokenBatcher(...), batch_size=None)`.

- [ ] **Step 1: Write the failing tests** (`tests/data/sequence/test_batching.py`)

```python
"""TokenBatcher: fixed (rows, width) batches, contiguous runs of the stream, exact resume."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from torch.utils.data import DataLoader

from oplm.data import build_train_dataloader
from oplm.data.sequence.batching import TokenBatcher
from oplm.data.sequence.collate import MLMCollator, token_length
from oplm.data.sequence.dataset import ShardedProteinDataset
from oplm.data.tokenizer import get_tokenizer
from tests.data.conftest import write_sequence_shards
from tests.training.conftest import tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

WIDTH, ROWS = 16, 2


@pytest.fixture(autouse=True)
def _single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


def _dataset(tmp_path: Path, lengths: list[int], *, shuffle: bool = False) -> ShardedProteinDataset:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir(exist_ok=True)
    residues = "ACDEFGHIKLMNPQRSTVWY" * 3
    records = [(f"{i}", residues[:n]) for i, n in enumerate(lengths)]
    path = write_sequence_shards(shard_dir, records, n_shards=1)
    return ShardedProteinDataset(path, shuffle_shards=shuffle, shuffle_rows=shuffle, seed=0)


def _batcher(dataset, *, drop_last: bool = True, rows: int = ROWS) -> TokenBatcher:
    collator = MLMCollator(
        get_tokenizer(), max_length=WIDTH, pack_sequences=True, pack_rows=rows, keep_sequence_ids=True
    )
    return TokenBatcher(dataset, collator, rows=rows, width=WIDTH, drop_last=drop_last)


def test_batches_close_at_the_first_sequence_that_fits_nowhere(tmp_path: Path) -> None:
    # Token lengths 12, 12, 7, 14, 4, 3: [12 | 12] closes at 7; [7 + 4 + 3 | 14] is full; ...
    batches = list(_batcher(_dataset(tmp_path, [10, 10, 5, 12, 2, 1]), drop_last=False))
    assert [b["num_sequences"] for b in batches] == [2, 4]
    assert [b["sequence_ids"] for b in batches] == [["0", "1"], ["2", "3", "4", "5"]]
    assert [tuple(b["stream_position"]) for b in batches] == [(0, 2), (0, 6)]
    for b in batches:
        assert b["input_ids"].shape == b["segment_ids"].shape == (ROWS, WIDTH)


def test_every_batch_is_exactly_rows_by_width_and_a_run_of_the_stream(tmp_path: Path) -> None:
    lengths = [(i * 7) % 15 + 1 for i in range(60)]
    dataset = _dataset(tmp_path, lengths)
    stream = [str(r["sequence_id"]) for r in dataset]
    batches = list(_batcher(dataset))
    served = [sid for b in batches for sid in b["sequence_ids"]]
    assert served == stream[: len(served)]  # contiguous, in order, nothing skipped
    assert len(stream) - len(served) < 2 * ROWS * WIDTH  # only the dropped tail is missing
    for b in batches:
        assert b["input_ids"].shape == (ROWS, WIDTH)
        assert int(b["attention_mask"].sum()) == sum(
            token_length("x" * n, WIDTH) for n in (lengths[int(s)] for s in b["sequence_ids"])
        )


def test_final_partial_batch_follows_drop_last(tmp_path: Path) -> None:
    lengths = [10, 10, 5]  # [12 | 12] then a lone 7
    assert [b["num_sequences"] for b in _batcher(_dataset(tmp_path, lengths))] == [2]
    kept = list(_batcher(_dataset(tmp_path, lengths), drop_last=False))
    assert [b["num_sequences"] for b in kept] == [2, 1]


def test_long_sequence_takes_a_whole_row(tmp_path: Path) -> None:
    batches = list(_batcher(_dataset(tmp_path, [40, 40, 3]), drop_last=False, rows=1))
    assert [b["num_sequences"] for b in batches] == [1, 1, 1]
    assert int(batches[0]["attention_mask"].sum()) == WIDTH


def test_resume_rows_continue_each_worker_exactly(tmp_path: Path) -> None:
    lengths = [(i * 5) % 13 + 1 for i in range(200)]
    dataset = _dataset(tmp_path, lengths, shuffle=True)

    def run(batcher: TokenBatcher) -> list[tuple[int, int, list[str]]]:
        loader = DataLoader(batcher, batch_size=None, num_workers=2)
        return [
            (int(b["stream_position"][0]), int(b["stream_position"][1]), list(b["sequence_ids"]))
            for b in loader
        ]

    baseline = run(_batcher(dataset))
    cut = 7  # batches consumed before the "checkpoint"
    rows = [0, 0]
    for worker, emitted, _ in baseline[:cut]:
        rows[worker] = emitted
    resumed_batcher = _batcher(dataset)
    resumed_batcher.set_resume_rows(rows)
    resumed = run(resumed_batcher)
    for worker in range(2):
        before = [ids for w, _, ids in baseline[:cut] if w == worker]
        after = [ids for w, _, ids in baseline[cut:] if w == worker]
        assert sum(len(ids) for ids in before) == rows[worker]
        assert [ids for w, _, ids in resumed if w == worker] == after


def test_build_train_dataloader_in_token_mode(training_parquet: Path, tmp_path: Path) -> None:
    cfg = tiny_train_cfg(
        tmp_path, training_parquet, tokens_per_step=256, batch_tokens=256, pack_sequences=True
    )
    loader = build_train_dataloader(cfg)
    assert isinstance(loader.dataset, TokenBatcher) and loader.batch_size is None
    assert loader.dataset.total_length > 0
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (4, 64)  # 256 / max_position_embeddings 64
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/sequence/test_batching.py -v`
Expected: FAIL (`ModuleNotFoundError: oplm.data.sequence.batching`).

- [ ] **Step 3: Create `src/oplm/data/sequence/batching.py`**

```python
"""Token-budget batching: fixed ``rows x width`` micro-batches filled from a sequence stream.

:class:`TokenBatcher` wraps the training dataset inside each DataLoader worker. It places
sequences in stream order into ``rows`` rows of ``width`` tokens (first fit) and closes a batch
at the first sequence that fits in no row; that sequence starts the next batch. Every batch is a
contiguous run of the worker's stream, so the number of rows emitted so far is an exact resume
position whatever the batch size (token-batching design spec, §5.3 and §6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import IterableDataset, get_worker_info

from oplm.data.sequence.collate import RowFiller, token_length

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from oplm.data.sequence.collate import MLMCollator

__all__ = ["TokenBatcher"]


class TokenBatcher(IterableDataset[dict[str, object]]):
    """Yield fixed-shape packed, masked batches of whole sequences from ``dataset``.

    Args:
        dataset: The training stream (``InterleavedDataset`` or ``ShardedProteinDataset``); it
            stripes itself over ``(rank, worker)``.
        collator: An :class:`~oplm.data.sequence.collate.MLMCollator` built with
            ``pack_sequences=True, pack_rows=rows, max_length=width``.
        rows: Rows per batch (``train.batch_tokens // width``).
        width: Tokens per row (``model.max_position_embeddings``).
        drop_last: Drop the final partial batch of each worker's stream (``data.drop_last``).

    Each batch carries the collator's tensors (all ``(rows, width)``), ``num_sequences`` and
    ``stream_position = (worker_id, rows_emitted)``: the rows of ``dataset``'s stream this worker
    has emitted this epoch, counting this batch.
    """

    def __init__(
        self,
        dataset: IterableDataset[dict[str, object]],
        collator: MLMCollator,
        *,
        rows: int,
        width: int,
        drop_last: bool = True,
    ) -> None:
        super().__init__()
        if rows < 1 or width < 3:
            raise ValueError(f"TokenBatcher needs rows >= 1 and width >= 3, got {rows}, {width}")
        self._dataset = dataset
        self._collator = collator
        self._rows = rows
        self._width = width
        self._drop_last = drop_last
        self._resume_rows: list[int] | None = None

    @property
    def total_length(self) -> int:
        """Rows in one epoch of the wrapped stream (for the trainer's fractional epoch)."""
        return int(getattr(self._dataset, "total_length", 0))

    def set_epoch(self, epoch: int) -> None:
        """Forward the epoch to the wrapped dataset's shuffling."""
        self._dataset.set_epoch(epoch)  # ty: ignore[unresolved-attribute]  # duck-typed dataset

    def set_resume_rows(self, rows_per_worker: Sequence[int]) -> None:
        """Arm a one-epoch per-worker row skip (the wrapped dataset skips; counting resumes)."""
        self._resume_rows = list(rows_per_worker)
        self._dataset.set_resume_rows(rows_per_worker)  # ty: ignore[unresolved-attribute]

    def clear_resume_skip(self) -> None:
        """Disarm the resume skip here and in the wrapped dataset."""
        self._resume_rows = None
        self._dataset.clear_resume_skip()  # ty: ignore[unresolved-attribute]

    def __iter__(self) -> Iterator[dict[str, object]]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        emitted = self._resume_rows[worker_id] if self._resume_rows is not None else 0
        filler, buffer = RowFiller(self._rows, self._width), []
        for row in self._dataset:
            length = token_length(str(row["sequence"]), self._width)
            if not filler.place(length):
                emitted += len(buffer)
                yield self._batch(buffer, worker_id, emitted)
                filler, buffer = RowFiller(self._rows, self._width), []
                filler.place(length)  # empty rows take any sequence: length <= width
            buffer.append(row)
        if buffer and not self._drop_last:
            emitted += len(buffer)
            yield self._batch(buffer, worker_id, emitted)

    def _batch(
        self, rows: list[Mapping[str, object]], worker_id: int, emitted: int
    ) -> dict[str, object]:
        out: dict[str, object] = dict(self._collator(rows))
        out["stream_position"] = (worker_id, emitted)
        return out
```

If `ty` rejects the `# ty: ignore` codes, run `ty check` and copy the code it reports, as elsewhere in `dataset.py`.

- [ ] **Step 4: Wire the loader** (`src/oplm/data/sequence/loaders.py`)

`_build_dataloader`: add a required keyword `batch_size: int | None` and use it instead of `cfg.train.batch_size`; type `collator` as `MLMCollator | None` (passed as `collate_fn`; `None` with `batch_size=None` gives the DataLoader's per-item conversion, which leaves dicts of tensors, ints, lists and tuples intact). Update the eval builder's call to pass `batch_size=cfg.train.batch_size`.

`build_train_dataloader`: after building `dataset` and the mask-dropout check:

```python
    width = cfg.model.max_position_embeddings
    rows = cfg.train.batch_tokens // width if cfg.train.batch_tokens is not None else None
    collator = MLMCollator(
        get_tokenizer(),
        max_length=width,
        # ... existing kwargs unchanged ...
        pack_sequences=cfg.data.pack_sequences,
        pack_rows=rows,
    )
    if rows is not None:
        # Token mode: each worker builds whole (rows, width) batches; no DataLoader batching.
        batcher = TokenBatcher(dataset, collator, rows=rows, width=width, drop_last=cfg.data.drop_last)
        return _build_dataloader(batcher, None, cfg, batch_size=None)
    return _build_dataloader(
        dataset, collator, cfg, batch_size=cfg.train.batch_size, drop_last=cfg.data.drop_last
    )
```

Update the docstring's Returns: in token mode the loader yields the batcher's fixed-shape batches. Import `TokenBatcher` from `oplm.data.sequence.batching`. If `oplm/data/__init__.py` re-exports `InterleavedDataset`/`ShardedProteinDataset`, add `TokenBatcher` beside them.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/data -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
.venv/bin/ruff check src/oplm/data tests/data/sequence/test_batching.py
.venv/bin/ruff format src/oplm/data/sequence/batching.py src/oplm/data/sequence/loaders.py tests/data/sequence/test_batching.py
git add src/oplm/data/sequence/batching.py src/oplm/data/sequence/loaders.py src/oplm/data/__init__.py tests/data/sequence/test_batching.py
git commit -m "feat(data): TokenBatcher builds fixed rows x width batches in the data workers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Trainer token mode: accumulation, samples, positions, fill, empty epochs, deferral

**Files:**
- Modify: `src/oplm/training/trainer.py`:
  - after the `Accelerator(...)` construction (~line 317): derive accumulation;
  - training-state init (~line 763): `_worker_rows`, `_step_local_samples`, one-time log;
  - `_StepFlags` (~line 136) and `_reduce_step_flags` (~line 1233): add samples;
  - the loop (~lines 898–960 and ~986): empty-epoch guard, worker-row reset, position recording, samples accumulation;
  - both `_SaveDeferral.decide` calls (~lines 1059, 1127) and new `_alignment_workers`;
  - `_log_step` (~line 1405): `train/batch_fill`;
  - `_global_effective_batch_size` (~line 1915);
  - resume and branch paths (~lines 1720–1730, 1786–1790): reset `_step_local_samples`.
- Test: `tests/training/test_e2e_token_batching.py` (new)

**Interfaces:**
- Consumes: `resolve_token_accumulation` (Task 1); the loader's `stream_position` (Task 4).
- Produces (for Task 6): `Trainer._token_mode: bool`, `Trainer._worker_rows: list[int]` (index = DataLoader worker id, value = rows emitted this epoch as of the last consumed batch), `Trainer._alignment_workers() -> int`.

- [ ] **Step 1: Write the failing tests** (`tests/training/test_e2e_token_batching.py`)

```python
"""Token mode end to end on CPU: derived accumulation, batch fill, empty-epoch error."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow


def test_token_mode_derives_accumulation_and_logs_fill(training_parquet: Path, tmp_path: Path) -> None:
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path, training_parquet, max_steps=4, tokens_per_step=512, batch_tokens=256,
        pack_sequences=True,
    )
    callback = FullRecordingCallback()
    trainer = Trainer(cfg, callbacks=[callback])
    trainer.train()
    assert trainer.cfg.train.gradient_accumulation_steps == 2  # 512 / (256 x 1 process)
    assert trainer.accelerator.gradient_accumulation_steps == 2
    logs = [metrics for _, metrics in callback.train_logs]
    assert len(logs) == 4
    for metrics in logs:
        assert 0.5 < metrics["train/batch_fill"] <= 1.0
        assert math.isfinite(metrics["train/loss"])
    assert 0 < trainer.tokens_seen <= 4 * 512
    assert logs[-1]["train/samples"] == trainer._samples_seen > 0


def test_token_mode_turns_off_save_alignment(training_parquet: Path, tmp_path: Path) -> None:
    from oplm.training.trainer import Trainer

    token = tiny_train_cfg(
        tmp_path / "t", training_parquet, num_workers=2, tokens_per_step=256, batch_tokens=256,
        pack_sequences=True,
    )
    assert Trainer(token, callbacks=[])._alignment_workers() == 1
    plain = tiny_train_cfg(tmp_path / "p", training_parquet, num_workers=2)
    assert Trainer(plain, callbacks=[])._alignment_workers() == 2


def test_token_mode_empty_epoch_raises_clearly(tiny_training_parquet: Path, tmp_path: Path) -> None:
    from oplm.training.trainer import Trainer

    # 16 sequences can never fill 64 rows of 64 tokens; drop_last drops the only partial batch.
    cfg = tiny_train_cfg(
        tmp_path, tiny_training_parquet, max_steps=2, tokens_per_step=4096, batch_tokens=4096,
        pack_sequences=True,
    )
    with pytest.raises(RuntimeError, match="produced no batch"):
        Trainer(cfg, callbacks=[]).train()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/training/test_e2e_token_batching.py -v`
Expected: FAIL (`AttributeError: 'Trainer' object has no attribute '_alignment_workers'`, or accumulation still 1).

- [ ] **Step 3: Derive accumulation** — right after the `self.accelerator = Accelerator(...)` statement:

```python
        # Token mode: the global token budget fixes accumulation for this world size. Written
        # into the config and the accelerator before anything reads it, so the loop, the
        # total-step math and the saved config.yaml all see the value in use.
        self._token_mode = cfg.train.batch_tokens is not None
        if self._token_mode:
            from oplm.config import resolve_token_accumulation

            accumulation = resolve_token_accumulation(cfg.train, self.accelerator.num_processes)
            cfg.train.gradient_accumulation_steps = accumulation
            self.accelerator.gradient_accumulation_steps = accumulation
```

(`Accelerator.gradient_accumulation_steps` has a setter in accelerate 1.15.) Confirm nothing between the start of `__init__` and this point reads `gradient_accumulation_steps`; the `Accelerator(...)` call itself receives the configured value and is corrected here.

- [ ] **Step 4: Training state** — after `self._batches_in_epoch = 0` (and its comment):

```python
        # Token mode (spec §6.1): rows each DataLoader worker had emitted this epoch, as of the
        # last batch the loop consumed (its stream_position). The per-worker resume position.
        self._worker_rows = [0] * self._effective_num_workers()
        self._step_local_samples = 0  # local sequences accumulated across the current opt step
        if self._token_mode:
            logger.info(
                "Token mode: %d tokens per step, %d per micro-batch, accumulation %d; "
                "worker-cycle save alignment is off (each GPU rolls over epochs on its own)",
                cfg.train.tokens_per_step,
                cfg.train.batch_tokens,
                cfg.train.gradient_accumulation_steps,
            )
```

- [ ] **Step 5: Samples in the rank reduction**

`_StepFlags`: add `samples_delta: int` after `tokens_delta: int` and mention it in the docstring ("sequence counts add up like tokens; GPUs differ in sequences per batch in token mode").

`_reduce_step_flags`: add keyword `local_samples: int = 0` (documented), pack it second:

```python
        bundle = torch.tensor(
            [
                int(local_tokens),
                int(local_samples),
                int(drain),
                int(nonfinite),
                int(save_due),
                int(pending_done),
            ],
            device=self.accelerator.device,
            dtype=torch.long,
        )
        reduced = self.accelerator.reduce(bundle, reduction="sum").tolist()
        tokens_delta, samples_delta, drain_sum, nonfinite_sum, save_due_sum, pending_done_sum = (
            int(x) for x in reduced
        )
        return _StepFlags(
            tokens_delta=tokens_delta,
            samples_delta=samples_delta,
            # ... the other fields unchanged ...
        )
```

In the loop, replace `self._samples_seen += result.samples * self.accelerator.num_processes` with `self._step_local_samples += result.samples`. At the optimizer-step boundary, next to `local_tokens = self._step_local_tokens; self._step_local_tokens = 0`, add `local_samples = self._step_local_samples; self._step_local_samples = 0`, pass `local_samples=local_samples` to `_reduce_step_flags`, and after `self.tokens_seen += tokens_delta` add `self._samples_seen += flags.samples_delta`. Update the "Track tokens ... samples" comment accordingly. In `_resume_from_checkpoint` and `_branch_from_checkpoint`, next to `self._step_local_tokens = 0`, add `self._step_local_samples = 0`.

- [ ] **Step 6: Epoch rollover and positions**

In the `except StopIteration:` branch, after `self._clear_dataset_resume_skip()`:

```python
                        self._worker_rows = [0] * self._effective_num_workers()
                        data_iter = iter(self.dataloader)
                        try:
                            batch = next(data_iter)
                        except StopIteration:
                            raise RuntimeError(
                                f"the training stream produced no batch in epoch {self.epoch}: too "
                                "little data for one batch per rank and worker (in token mode, "
                                "lower train.batch_tokens or set data.drop_last=false)"
                            ) from None
```

(replacing the two existing lines `data_iter = iter(self.dataloader)` / `batch = next(data_iter)`). In token mode each GPU reaches this branch on its own step; it contains no collective.

After `self._window_data_wait += time.perf_counter() - data_t0`:

```python
                if self._token_mode:
                    worker_id, rows_emitted = batch["stream_position"]
                    self._worker_rows[int(worker_id)] = int(rows_emitted)
```

- [ ] **Step 7: Save alignment off in token mode** — add the method near `_effective_num_workers`:

```python
    def _alignment_workers(self) -> int:
        """Worker count for :class:`_SaveDeferral`; 1 (no deferral) in token mode.

        The deferral assumes ``batches_in_epoch`` is identical on every rank. In token mode each
        GPU rolls over to its next epoch on its own, so ranks could disagree on whether to defer
        and enter a save apart, which deadlocks. Each worker's resume position is exact without
        it; only the interleaving of a GPU's workers can shift within one cycle (spec §6.4).
        """
        return 1 if self._token_mode else self._effective_num_workers()
```

In both `decide(...)` calls, replace `num_workers=self._effective_num_workers()` with `num_workers=self._alignment_workers()`.

- [ ] **Step 8: Batch fill and the global batch figure**

In `_log_step`, inside `if samples_delta > 0:` after `metrics["train/mean_seq_len"] = ...`:

```python
            if self._token_mode and self._window_steps > 0:
                # Real tokens over capacity: what the packing leaves empty (spec §5.5).
                metrics["train/batch_fill"] = tokens_delta / (
                    self._window_steps * self.cfg.train.tokens_per_step
                )
```

(`self._window_steps` is reset only after this block; confirm by reading the end of `_log_step`.)

`_global_effective_batch_size`: return `int(self.cfg.train.tokens_per_step)` when `self._token_mode`, else the existing product; update its docstring ("tokens per step in token mode").

`_compute_total_steps` needs no change: token mode forbids `max_epochs`, so it returns `max_steps`.

- [ ] **Step 9: Run the new tests and the training suite**

Run: `.venv/bin/python -m pytest tests/training/test_e2e_token_batching.py -v`
Expected: all PASS.

Run: `.venv/bin/python -m pytest tests/training -q`
Expected: all PASS (sequence mode unchanged; `train/samples` values are identical with equal per-rank batches).

- [ ] **Step 10: Commit**

```bash
VIRTUAL_ENV=$PWD/.venv .venv/bin/ty check src/oplm/training/trainer.py
.venv/bin/ruff check src/oplm/training/trainer.py tests/training/test_e2e_token_batching.py
.venv/bin/ruff format src/oplm/training/trainer.py tests/training/test_e2e_token_batching.py
git add src/oplm/training/trainer.py tests/training/test_e2e_token_batching.py
git commit -m "feat(training): token mode derives accumulation, tracks worker positions, logs fill

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The token-mode data cursor and data-exact resume

**Files:**
- Modify: `src/oplm/data/sequence/dataset.py` (`DataCursor` ~line 147)
- Modify: `src/oplm/training/trainer.py` (`_save_checkpoint` cursor build ~line 1528; `_resume_data_cursor` ~line 1818)
- Modify: `src/oplm/training/stream_replay.py` (`StreamReplay.__init__` ~line 102)
- Create: `tests/training/test_e2e_token_resume.py`, `tests/training/_token_resume_worker.py`
- Test: also `tests/training/test_stream_replay.py`, `tests/data/sequence/test_dataset.py`

**Interfaces:**
- Consumes: `Trainer._token_mode`, `Trainer._worker_rows`, `Trainer._alignment_workers` (Task 5); `set_resume_rows` on the top-level dataset (Tasks 3–4); test helpers `_SequenceIdRecorder`, `_force_keep_sequence_ids`, `_wrap_recorder` from `tests/training/test_e2e_data_exact.py` (existing).
- Produces: `DataCursor.rank_epochs: list[int] | None`, `DataCursor.worker_rows: list[list[int]] | None` in `trainer_state.json`'s `cursor`.

- [ ] **Step 1: Write the failing unit tests**

Append to `tests/data/sequence/test_dataset.py`:

```python
from dataclasses import asdict

from oplm.data import DataCursor


def test_data_cursor_loads_old_dicts() -> None:
    old = {"epoch": 1, "batches_in_epoch": 5, "world_size": 2, "num_workers": 1,
           "per_rank_batch": 4, "seed": 0}
    cursor = DataCursor(**old)
    assert cursor.rank_epochs is None and cursor.worker_rows is None


def test_data_cursor_round_trips_token_fields() -> None:
    cursor = DataCursor(epoch=0, batches_in_epoch=3, world_size=2, num_workers=2,
                        per_rank_batch=0, seed=0, rank_epochs=[0, 1], worker_rows=[[4, 5], [0, 2]])
    assert DataCursor(**asdict(cursor)) == cursor
```

Append to `tests/training/test_stream_replay.py`:

```python
def test_stream_replay_rejects_token_mode(tmp_path: Path) -> None:
    from oplm.training.stream_replay import StreamReplay
    from tests.training.conftest import tiny_train_cfg

    cfg = tiny_train_cfg(tmp_path, tmp_path / "unused.parquet", tokens_per_step=256,
                         batch_tokens=256, pack_sequences=True)
    with pytest.raises(ValueError, match="token-mode"):
        StreamReplay(cfg, rank=0, world_size=1)
```

Run: `.venv/bin/python -m pytest tests/data/sequence/test_dataset.py -k data_cursor tests/training/test_stream_replay.py -k token -v`
Expected: FAIL (`TypeError: unexpected keyword 'rank_epochs'`; no guard).

- [ ] **Step 2: Cursor fields and the stream-replay guard**

`DataCursor`: add, after `seed: int`:

```python
    rank_epochs: list[int] | None = None
    worker_rows: list[list[int]] | None = None
```

and extend its docstring's Attributes: `rank_epochs`: token mode only, each rank's epoch at the save (ranks roll over on their own); `worker_rows`: token mode only, a ranks × workers table of rows each worker had emitted this epoch, the exact position whatever the batch size; `per_rank_batch` is 0 in token mode.

`StreamReplay.__init__`, first statement:

```python
        if cfg.train.batch_tokens is not None:
            raise ValueError(
                "oplm stream-replay does not support token-mode runs (train.batch_tokens): their "
                "batches hold a varying number of sequences, so a step has no fixed stream offset"
            )
```

Run the Step 1 tests again. Expected: PASS.

- [ ] **Step 3: Write the failing single-process resume tests** (`tests/training/test_e2e_token_resume.py`)

```python
"""Token mode resumes data-exactly: same rows per worker, across workers, epochs and budgets."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.training.conftest import FullRecordingCallback, tiny_train_cfg
from tests.training.test_e2e_data_exact import _force_keep_sequence_ids, _wrap_recorder

pytestmark = pytest.mark.slow

TOKEN = dict(pack_sequences=True, log_every=1)


def _run(cfg, callbacks=None):  # noqa: ANN001, ANN202
    from oplm.training.trainer import Trainer

    trainer = Trainer(cfg, callbacks=callbacks or [])
    recorder = _wrap_recorder(trainer)
    trainer.train()
    return trainer, recorder


def _interrupt_and_resume(data: Path, tmp_path: Path, *, total: int, stop: int, first: dict,
                          second: dict, num_workers: int = 0):  # noqa: ANN202
    common = dict(num_workers=num_workers, **TOKEN)
    control, control_rec = _run(tiny_train_cfg(
        tmp_path / "control", data, max_steps=total, save_final=False, **first, **common))
    run_dir = tmp_path / "interrupt"
    callback = FullRecordingCallback()
    run1, rec1 = _run(tiny_train_cfg(
        run_dir, data, max_steps=stop, save_every=stop, save_final=False, **first, **common),
        [callback])
    assert callback.checkpoint_steps[-1] == run1.global_step == stop  # no deferral in token mode
    state = json.loads((run_dir / f"checkpoint-{stop}" / "trainer_state.json").read_text())
    run2, rec2 = _run(tiny_train_cfg(
        run_dir, data, max_steps=total, save_final=False, auto_resume=True, **second, **common))
    return control, control_rec, run1, rec1, run2, rec2, state


@pytest.mark.parametrize("num_workers", [0, 2])
def test_token_resume_matches_control_mid_epoch(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, num_workers: int
) -> None:
    _force_keep_sequence_ids(monkeypatch)
    budget = dict(tokens_per_step=256, batch_tokens=256)
    control, control_rec, run1, rec1, run2, rec2, state = _interrupt_and_resume(
        training_parquet, tmp_path, total=12, stop=6, first=budget, second=budget,
        num_workers=num_workers,
    )
    cursor = state["cursor"]
    assert cursor["rank_epochs"] == [0] and run1.epoch == 0
    assert len(cursor["worker_rows"]) == 1 and len(cursor["worker_rows"][0]) == max(num_workers, 1)
    # Six batches is a whole number of worker cycles, so even the interleaving is exact.
    assert rec1.flat() + rec2.flat() == control_rec.flat()
    assert run2.tokens_seen == control.tokens_seen


def test_token_resume_keeps_position_when_the_budget_changes(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_keep_sequence_ids(monkeypatch)
    _, control_rec, _, rec1, _, rec2, _ = _interrupt_and_resume(
        training_parquet, tmp_path, total=12, stop=6,
        first=dict(tokens_per_step=256, batch_tokens=256),
        second=dict(tokens_per_step=192, batch_tokens=192),
    )
    combined = rec1.flat() + rec2.flat()
    # One worker, no epoch crossing: any budget serves the stream in order, so the resumed run
    # (smaller batches after step 6) is a prefix of the control's stream.
    assert combined == control_rec.flat()[: len(combined)]


def test_token_resume_matches_control_across_an_epoch(
    tiny_training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_keep_sequence_ids(monkeypatch)
    budget = dict(tokens_per_step=128, batch_tokens=128)
    _, control_rec, run1, rec1, _, rec2, state = _interrupt_and_resume(
        tiny_training_parquet, tmp_path, total=20, stop=10, first=budget, second=budget,
    )
    assert run1.epoch >= 1 and state["cursor"]["rank_epochs"] == [run1.epoch]
    assert rec1.flat() + rec2.flat() == control_rec.flat()


def test_resume_across_batching_modes_raises(training_parquet: Path, tmp_path: Path) -> None:
    from oplm.training.trainer import Trainer

    run_dir = tmp_path / "run"
    Trainer(tiny_train_cfg(run_dir, training_parquet, max_steps=2), callbacks=[]).train()
    resumed = tiny_train_cfg(run_dir, training_parquet, max_steps=4, auto_resume=True,
                             tokens_per_step=256, batch_tokens=256, **TOKEN)
    with pytest.raises(ValueError, match="resume_data_position=false"):
        Trainer(resumed, callbacks=[])


def test_token_resume_rejects_a_changed_worker_count(training_parquet: Path, tmp_path: Path) -> None:
    from oplm.training.trainer import Trainer

    run_dir = tmp_path / "run"
    budget = dict(tokens_per_step=256, batch_tokens=256, **TOKEN)
    Trainer(tiny_train_cfg(run_dir, training_parquet, max_steps=2, **budget), callbacks=[]).train()
    resumed = tiny_train_cfg(run_dir, training_parquet, max_steps=4, auto_resume=True,
                             num_workers=2, **budget)
    with pytest.raises(ValueError, match="num_workers"):
        Trainer(resumed, callbacks=[])
```

Notes for the implementer:
- `FullRecordingCallback` exposes the saved steps; if the property is not called `checkpoint_steps`, use the one `test_e2e_data_exact.py` uses.
- If the resume layout check fires inside `trainer.train()` rather than `Trainer(...)`, move the `pytest.raises` around `.train()`.
- In the epoch test, if `tiny_training_parquet` does not roll over by step 10 at `batch_tokens=128`, lower it to 64 (and `tokens_per_step` with it); the `run1.epoch >= 1` assertion must hold for the test to mean anything.

Run: `.venv/bin/python -m pytest tests/training/test_e2e_token_resume.py -v`
Expected: FAIL (cursor has no `rank_epochs`; resume restarts the stream).

- [ ] **Step 4: Save the token cursor** — in `_save_checkpoint`, replace the `cursor = DataCursor(...)` statement:

```python
        rank_epochs: list[int] | None = None
        worker_rows: list[list[int]] | None = None
        if self._token_mode:
            # Every rank enters this save at the same step (each trigger is rank-reduced, and
            # _SaveDeferral is off in token mode), so the gather cannot strand a rank.
            from accelerate.utils import gather_object

            gathered = gather_object([(self.epoch, list(self._worker_rows))])
            rank_epochs = [int(epoch) for epoch, _ in gathered]
            worker_rows = [[int(n) for n in rows] for _, rows in gathered]
        cursor = DataCursor(
            epoch=self.epoch,
            batches_in_epoch=self._batches_in_epoch,
            world_size=self.accelerator.num_processes,
            num_workers=self._effective_num_workers(),
            per_rank_batch=0 if self._token_mode else self.cfg.train.batch_size,
            seed=self.cfg.train.seed,
            rank_epochs=rank_epochs,
            worker_rows=worker_rows,
        )
```

Confirm `_save_checkpoint` has exactly three call sites (`grep -n "_save_checkpoint(" src/oplm/training/trainer.py`), all in branches every rank takes together: drain (after the reduced `flags.drain`), periodic (after `decide`), and the final save. Note this in the commit message if the count differs.

- [ ] **Step 5: Resume from the token cursor** — in `_resume_data_cursor`, after `cursor = DataCursor(**cursor_dict)`, replace the `live_layout` dict and everything after the mismatch check with:

```python
        if (cursor.worker_rows is not None) != self._token_mode:
            saved, live = ("token", "sequence") if cursor.worker_rows is not None else ("sequence", "token")
            raise ValueError(
                f"Checkpoint {checkpoint_dir}'s data cursor is from a {saved}-mode run, but this "
                f"run is in {live} mode (train.batch_tokens); its position cannot be mapped. To "
                "resume anyway (restarting the epoch's data position from row 0), set "
                "train.resume_data_position=false."
            )
        # Rows per worker depend only on the striping (world size, workers, seed); in token mode
        # the batch size can change at a resume without moving the position.
        live_layout: dict[str, int] = {
            "world_size": self.accelerator.num_processes,
            "num_workers": self._effective_num_workers(),
            "seed": self.cfg.train.seed,
        }
        if not self._token_mode:
            live_layout["per_rank_batch"] = self.cfg.train.batch_size
        mismatches = [...]  # unchanged
        if mismatches:
            raise ValueError(...)  # unchanged

        if self._token_mode:
            assert cursor.rank_epochs is not None and cursor.worker_rows is not None
            rank = self.accelerator.process_index
            self.epoch = cursor.rank_epochs[rank]
            self._set_dataset_epoch(self.epoch)
            self._epoch_at_last_opt_step = self.epoch
            self._worker_rows = list(cursor.worker_rows[rank])
            self._top_level_dataset().set_resume_rows(cursor.worker_rows[rank])  # ty: ignore[unresolved-attribute]
            return

        # ... the existing sequence-mode restore (batches_in_epoch, set_resume_skip, warning) ...
```

Update the method's docstring: add the token-mode case (per-rank epoch, per-worker rows, layout without `per_rank_batch`, cross-mode error).

- [ ] **Step 6: Run the single-process resume tests**

Run: `.venv/bin/python -m pytest tests/training/test_e2e_token_resume.py tests/training/test_e2e_data_exact.py tests/training/test_e2e_branch.py -v`
Expected: all PASS.

- [ ] **Step 7: Write the two-rank test** — worker `tests/training/_token_resume_worker.py`:

```python
"""Worker for the two-rank token-mode resume test. Run under ``torch.distributed.run``.

Trains one phase of a token-mode run on 2 CPU/gloo ranks and writes, per rank, every consumed
batch's sequence ids and the rank's epoch at the time to ``out_dir/<phase>-rank<r>.json``.
Phases: ``control`` (0 -> total), ``first`` (0 -> stop, one checkpoint at stop), ``second``
(auto-resume -> total).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(run_dir: str, train_data: str, out_dir: str, phase: str, total: int, stop: int) -> None:
    import pytest

    from oplm.training.trainer import Trainer
    from tests.training.conftest import tiny_train_cfg
    from tests.training.test_e2e_data_exact import _force_keep_sequence_ids

    monkeypatch = pytest.MonkeyPatch()
    _force_keep_sequence_ids(monkeypatch)
    budget = dict(tokens_per_step=256, batch_tokens=128, pack_sequences=True)  # accumulation 1
    if phase == "first":
        cfg = tiny_train_cfg(Path(run_dir), Path(train_data), max_steps=stop, save_every=stop,
                             save_final=False, **budget)
    else:
        cfg = tiny_train_cfg(Path(run_dir), Path(train_data), max_steps=total, save_final=False,
                             auto_resume=phase == "second", **budget)
    trainer = Trainer(cfg, callbacks=[])
    records: list[dict[str, object]] = []
    inner = trainer.dataloader

    class _Recorder:
        dataset = property(lambda self: inner.dataset)

        def set_epoch(self, epoch: int) -> None:
            inner.set_epoch(epoch)

        def __len__(self) -> int:
            return len(inner)

        def __iter__(self):  # noqa: ANN204
            for batch in inner:
                records.append({"epoch": trainer.epoch, "ids": list(batch["sequence_ids"])})
                yield batch

    trainer.dataloader = _Recorder()  # type: ignore[assignment]
    trainer.train()
    rank = trainer.accelerator.process_index
    (Path(out_dir) / f"{phase}-rank{rank}.json").write_text(json.dumps(records))
    monkeypatch.undo()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))
```

Test, appended to `tests/training/test_e2e_token_resume.py` (the launch environment mirrors `test_dcp_checkpoint_saved_at_world_size_2_resumes_at_world_size_1` in `tests/training/test_e2e_dcp.py`):

```python
def _launch(worker: Path, *args: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join(
        p for p in (str(repo_root), str(repo_root / "src"), os.environ.get("PYTHONPATH", "")) if p
    )
    env = {**os.environ, "ACCELERATE_USE_CPU": "true", "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": pythonpath}
    subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
         "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0", str(worker), *args],
        check=True, env=env,
    )


def _rollover_steps(records: list[dict]) -> list[int]:  # type: ignore[type-arg]
    return [i for i in range(1, len(records)) if records[i]["epoch"] != records[i - 1]["epoch"]]


def test_two_rank_token_resume_with_uneven_rollovers(
    tiny_training_parquet: Path, tmp_path: Path
) -> None:
    worker = Path(__file__).with_name("_token_resume_worker.py")
    out = tmp_path / "out"
    out.mkdir()
    total, stop = 12, 5
    data = str(tiny_training_parquet)
    _launch(worker, str(tmp_path / "control"), data, str(out), "control", str(total), str(stop))
    for phase in ("first", "second"):
        _launch(worker, str(tmp_path / "interrupt"), data, str(out), phase, str(total), str(stop))

    def load(phase: str, rank: int) -> list[dict]:  # type: ignore[type-arg]
        return json.loads((out / f"{phase}-rank{rank}.json").read_text())

    control = [load("control", rank) for rank in range(2)]
    # The point of the test: the two ranks reach a new epoch at different steps.
    assert _rollover_steps(control[0]) != _rollover_steps(control[1])
    for rank in range(2):
        resumed = [r["ids"] for r in load("first", rank) + load("second", rank)]
        assert resumed == [r["ids"] for r in control[rank]], rank
```

If the rollover assertion fails (the two ranks happen to roll over together), change `batch_tokens` in the worker (try 192 with `tokens_per_step=384`) until the ranks' rollover steps differ, and keep the assertion.

Run: `.venv/bin/python -m pytest tests/training/test_e2e_token_resume.py -k two_rank -v`
Expected: PASS (it takes a minute: three two-process launches).

- [ ] **Step 8: Full suite and commit**

Run: `.venv/bin/python -m pytest -m "not slow" -q` then `.venv/bin/python -m pytest tests/training tests/data -q`
Expected: all PASS.

```bash
VIRTUAL_ENV=$PWD/.venv .venv/bin/ty check src/
.venv/bin/ruff check src/oplm tests/training/test_e2e_token_resume.py tests/training/_token_resume_worker.py tests/training/test_stream_replay.py tests/data/sequence/test_dataset.py
.venv/bin/ruff format src/oplm/data/sequence/dataset.py src/oplm/training/trainer.py src/oplm/training/stream_replay.py tests/training/test_e2e_token_resume.py tests/training/_token_resume_worker.py
git add src/oplm/data/sequence/dataset.py src/oplm/training/trainer.py src/oplm/training/stream_replay.py tests/training/test_e2e_token_resume.py tests/training/_token_resume_worker.py tests/training/test_stream_replay.py tests/data/sequence/test_dataset.py
git commit -m "feat(training): token-mode cursor records per-rank epochs and per-worker rows

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Docs

**Files:**
- Modify: `docs/TRAIN.md` (new section after "Sequence packing", ~line 640; metrics paragraph ~line 412)
- Modify: `docs/CONFIG.md` (train table, after `train.batch_size` ~line 263)

- [ ] **Step 1: `docs/CONFIG.md` rows**, after `train.batch_size`:

```markdown
| `train.tokens_per_step` | `int \| null` | `null` | Token mode, with `train.batch_tokens`: global token capacity per optimizer step. Must be a multiple of `batch_tokens` × processes; the quotient becomes `gradient_accumulation_steps` at startup. Needs `max_steps` (not `max_epochs`) and `data.pack_sequences: true`. |
| `train.batch_tokens` | `int \| null` | `null` | Token mode: each process's micro-batch capacity, a multiple of `model.max_position_embeddings` (rows × width). Every micro-batch is exactly that shape. `train.batch_size` then only sizes eval batches. |
```

Also change `train.batch_size`'s description to: `Per-process batch size, before accumulation (eval batch size in token mode).`

- [ ] **Step 2: `docs/TRAIN.md` section** after "Sequence packing":

```markdown
### Token-budget batching (`train.tokens_per_step`, `train.batch_tokens`)

Counting a batch in sequences makes its real size depend on the data: the same 8192 sequences
hold about 1.1M tokens at a deepclust-heavy mix and 1.9M at a uniref-heavy one, and far fewer
than `batch_size × max_position_embeddings` once packed. Token mode fixes the token count instead.

- `train.tokens_per_step` is the global capacity per optimizer step and `train.batch_tokens`
  each process's micro-batch capacity (`rows × model.max_position_embeddings`). Accumulation is
  derived: `tokens_per_step / (batch_tokens × processes)`, which must be a whole number. The same
  `tokens_per_step` therefore works at any node count and model size.
- Each data worker fills `rows` rows first-fit in stream order and closes a batch at the first
  sequence that fits nowhere; that sequence starts the next batch. Batches are exactly
  `(rows, width)`, so shapes never change, and real tokens are 97–99% of capacity
  (`train/batch_fill`). Masking is unchanged.
- Each GPU rolls over to its next epoch on its own, since sequences per batch vary.
  `train/epoch` is rank 0's.
- Resume is data-exact per worker: the checkpoint cursor records each rank's epoch and the rows
  each worker had emitted. Only the world size, worker count and seed must match, so
  `tokens_per_step` and `batch_tokens` can change at a resume or branch without losing the
  position. Resuming between token and sequence mode needs `train.resume_data_position=false`.
- The worker-cycle save alignment is off in token mode; after a resume only the order in which
  a GPU's workers' batches interleave can shift within one cycle.
- Requires `data.pack_sequences: true` and `train.max_steps`. `oplm stream-replay` does not
  support token-mode runs.
```

In the metrics paragraph near line 412 (the one listing `train/mean_seq_len`), add: `` `train/batch_fill` (token mode: real tokens over capacity) ``.

- [ ] **Step 3: Commit**

```bash
git add docs/TRAIN.md docs/CONFIG.md
git commit -m "docs: token-budget batching

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

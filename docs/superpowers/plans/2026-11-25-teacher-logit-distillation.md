# Teacher-Logit Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional per-residue `(L, 20)` teacher-logit KL distillation with runtime `Neff` gating to OPLM's single-sequence, sequence-packed MLM training pipeline for WSD annealing runs.

**Architecture:** An optional top-level `distill: DistillConfig | None = None` block on `OplmConfig` enables reading optional `teacher_logits` (and `teacher_neff` / `teacher_neff_sqrt_l`) Parquet columns in `ShardedProteinDataset`, decoding masked canonical positions into dense `(R, T, 20)` float16 + `(R, T)` bool `teacher_mask` batch tensors in `MLMCollator`, and selecting `DistillMLMTask(MLMTask)` automatically in `Trainer.__init__` to compute the per-masked-token blended CE + `T²·KL` loss over live-tokenizer-mapped canonical channels. When `cfg.distill is None`, every code path is untouched.

**Tech Stack:** Python 3.11+, PyTorch, PyArrow, NumPy, OmegaConf, pytest, Ruff, ty.

**Spec:** `docs/superpowers/specs/2026-11-24-teacher-logit-distillation-design.md`

## Global Constraints

- Python 3.11+, `from __future__ import annotations` in every file, type hints on all function signatures, 100-char line limit.
- Zero changes to `src/oplm/model/` (use `OplmForMaskedLM.forward(..., labels=None)` which returns fp32 logits).
- Zero behavior change or extra batch keys when `cfg.distill is None`.
- Every `teacher_alphabet` channel-to-vocab mapping is derived from `get_tokenizer()`, never hardcoded (`ARNDCEQGHILKMFPSTWYV` != ESM-C's `LAGVSERTIDPKQNFYMHWC`).
- Must pass `.venv/bin/python -m pytest -m "not slow" -q`, `ruff check src/`, `ty check src/`, and `tests/test_docs.py`.
- Keep `pack_sequences=true`, `TokenBatcher`, `compile_dynamic=false`, `Trainer._canon_self_check`, `train_data_fingerprint`, `branch_from`, and `serialize_config` round-tripping intact.

## Review Focus

- **PyArrow `large_binary` shards and read-only `bytes`/`memoryview` buffers:** 100k-row UniRef50 shards with 14 KB/row approach PyArrow's 2 GiB 32-bit binary limit and naturally serialize as `large_binary`, and `np.frombuffer(bytes, ...)` returns a non-writable array that triggers PyTorch warnings unless copied/indexed cleanly (tested in Task 2 and Task 3).
- **Truncated sequences (`len(sequence) > max_length - 2`):** `teacher_logits` byte length must validate against raw `len(sequence) * 40`, then align masked positions within the clipped prefix without raising or misindexing (tested in Task 3).
- **CLI dotlist override onto `distill=None` and bad `teacher_alphabet` inputs:** `load_config(["distill.alpha=0.5"])` must resolve a `DistillConfig` from `None`, while lowercase/non-canonical/duplicate alphabets must fail fast (tested in Task 1).
- **`Trainer._canon_self_check` `[:4]` slicing when rows `0..3` have zero distilled tokens:** `teacher_logits` and `teacher_mask` must slice cleanly to `[:4]` and `DistillMLMTask.step` must handle `n_distill == 0` without NaN or emitting a bogus `0.0` `loss_kl` (tested in Task 4).
- **`branch_from` a plain MLM checkpoint into a `distill:` mixed anneal (distilled + plain + `sample_members: true` plain):** must restore weights/optimizer, drop the stale data cursor, log `train/loss_ce`, `train/loss_kl`, `train/distill_frac`, and round-trip the saved checkpoint `config.yaml` through `load_config` (tested in Task 4).

## File Structure

- `src/oplm/config.py`: `DistillConfig`, `OplmConfig.distill`, `validate_distill_compat`, `load_config` normalization, `serialize_config` emission.
- `src/oplm/data/sequence/dataset.py`: optional `teacher_logits` + `teacher_neff` / `teacher_neff_sqrt_l` reading, per-shard schema guards (`sample_members`, missing `Neff` column), and row-level `Neff` gating.
- `src/oplm/data/sequence/collate.py`: optional `load_teacher_logits` decoding (masked canonical positions only, `NaN` filter, byte-length guard) and packing into `(R, T, 20)` fp16 + `(R, T)` bool `teacher_mask`.
- `src/oplm/data/sequence/loaders.py`: wire `cfg.distill` into `build_train_dataloader` (leaving `build_sequence_eval_dataloader` untouched).
- `src/oplm/training/task.py`: `DistillMLMTask(MLMTask)` with live-tokenizer channel indices and per-masked-token blended CE + `T²·KL` loss + `StepResult.metrics`.
- `src/oplm/training/trainer.py`: auto-select `DistillMLMTask(cfg.distill)` when `task is None and cfg.distill is not None` and call `validate_distill_compat(cfg)`.
- `docs/data.md`, `docs/config.md`, `docs/training.md`, `tests/test_docs.py`: schema, config, training docs, and config-field doc guard.

---

### Task 1: `DistillConfig` and config plumbing

**Files:**
- Modify: `src/oplm/config.py:615-670,905-996`
- Test: `tests/training/test_config.py`

**Interfaces:**
- Consumes: existing `OplmConfig`, `FoldRunConfig`, `load_config`, `serialize_config` in `src/oplm/config.py`.
- Produces:
  - `DistillConfig(alpha: float = 0.5, temperature: float = 1.5, teacher_alphabet: str = "ARNDCEQGHILKMFPSTWYV", min_neff: float | None = None, min_neff_sqrt_l: float | None = None)` exported from `oplm.config`.
  - `OplmConfig.distill: DistillConfig | None = None`.
  - `validate_distill_compat(cfg: OplmConfig) -> None` in `oplm.config` (raises `ValueError` if both `cfg.fold` and `cfg.distill` are set).

- [ ] **Step 1: Write failing tests in `tests/training/test_config.py`**

```python
def test_distill_defaults_to_none() -> None:
    assert load_config([]).distill is None


def test_distill_cli_dotlist_and_round_trip(tmp_path: Path) -> None:
    cfg = load_config(["distill.alpha=0.7", "distill.min_neff_sqrt_l=2.0"])
    assert isinstance(cfg.distill, DistillConfig)
    assert cfg.distill.alpha == 0.7
    assert cfg.distill.temperature == 1.5
    assert cfg.distill.teacher_alphabet == "ARNDCEQGHILKMFPSTWYV"
    assert cfg.distill.min_neff is None
    assert cfg.distill.min_neff_sqrt_l == 2.0

    dumped = tmp_path / "roundtrip.yaml"
    dumped.write_text(serialize_config(cfg))
    reloaded = load_config(["--config", str(dumped)])
    assert reloaded.distill == cfg.distill


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": -0.1},
        {"alpha": 1.1},
        {"alpha": float("nan")},
        {"temperature": 0.0},
        {"temperature": -1.0},
        {"temperature": float("inf")},
        {"teacher_alphabet": "arndceqghilkmfpstwyv"},  # lowercase rejected
        {"teacher_alphabet": "ARNDCEQGHILKMFPSTWYX"},  # non-canonical X
        {"teacher_alphabet": "AARNDCEQGHILKMFPSTWY"},  # duplicate A
        {"teacher_alphabet": "ARND"},                  # wrong length
        {"min_neff": -1.0},
        {"min_neff": float("nan")},
        {"min_neff_sqrt_l": -0.5},
    ],
)
def test_distill_config_rejects_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        DistillConfig(**kwargs)


def test_distill_allows_zero_alpha() -> None:
    assert DistillConfig(alpha=0.0).alpha == 0.0


def test_distill_and_fold_mutually_exclusive(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "fold:\n  lm: esmc:esmc_600m\ndistill:\n  alpha: 0.5\n")
    with pytest.raises(ValueError, match="distill"):
        load_config(["--config", path])
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/training/test_config.py -k distill -v`
Expected: FAIL (`ImportError: cannot import name 'DistillConfig' from 'oplm.config'`).

- [ ] **Step 3: Implement `DistillConfig`, `validate_distill_compat`, and `load_config` / `serialize_config` hooks in `src/oplm/config.py`**

- Define `@dataclass class DistillConfig` right next to `FoldRunConfig` with fields `alpha: float = 0.5`, `temperature: float = 1.5`, `teacher_alphabet: str = "ARNDCEQGHILKMFPSTWYV"`, `min_neff: float | None = None`, `min_neff_sqrt_l: float | None = None`.
- In `DistillConfig.__post_init__`, validate: finite `alpha` in `[0.0, 1.0]`, finite `temperature > 0.0`, `isinstance(teacher_alphabet, str)` with `len == 20` and `set(teacher_alphabet) == set("LAGVSERTIDPKQNFYMHWC")`, finite `min_neff >= 0.0` (if set), finite `min_neff_sqrt_l >= 0.0` (if set). Reject bools masquerading as floats/ints if needed or let finite/range checks run cleanly.
- Add `distill: DistillConfig | None = None` to `OplmConfig`.
- Add `validate_distill_compat(cfg: OplmConfig) -> None` raising `ValueError` when `cfg.fold is not None and cfg.distill is not None`.
- In `load_config`: normalize `if isinstance(cfg.distill, dict): cfg.distill = DistillConfig(**cast("dict[str, Any]", cfg.distill))` (mirroring `cfg.fold`) and call `validate_distill_compat(cfg)`.
- In `serialize_config`: emit `if cfg.distill is not None: config_dict["distill"] = asdict(cfg.distill)`.

- [ ] **Step 4: Run config tests and type/lint checks**

Run: `.venv/bin/python -m pytest tests/training/test_config.py -q && ruff check src/oplm/config.py && ty check src/oplm/config.py`
Expected: PASS, 0 lint/type errors.

- [ ] **Step 5: Commit**

```bash
git add src/oplm/config.py tests/training/test_config.py
git commit -m "feat(config): add DistillConfig block and validation"
```

---

### Task 2: Teacher-logit columns and runtime `Neff` gating in `ShardedProteinDataset`

**Files:**
- Modify: `src/oplm/data/sequence/dataset.py:36-45,399-454,676-726`
- Test: `tests/data/sequence/test_dataset.py`

**Interfaces:**
- Consumes: existing `ShardedProteinDataset`, `InterleavedDataset` in `src/oplm/data/sequence/dataset.py`.
- Produces:
  - Column constants `TEACHER_LOGITS_COLUMN = "teacher_logits"`, `TEACHER_NEFF_COLUMN = "teacher_neff"`, `TEACHER_NEFF_SQRT_L_COLUMN = "teacher_neff_sqrt_l"` in `oplm.data.sequence.dataset`.
  - `ShardedProteinDataset(..., *, load_teacher_logits: bool = False, min_teacher_neff: float | None = None, min_teacher_neff_sqrt_l: float | None = None)`.
  - When `load_teacher_logits=True`, each yielded row carries `"teacher_logits"` as `bytes | None` (raw `<f2` payload or `None` when absent, gated out, or from a plain source).

- [ ] **Step 1: Write failing tests in `tests/data/sequence/test_dataset.py`**

Add a small local helper `_write_distill_shard(path, rows, *, large_binary=False, include_neff=True)` writing a Parquet file via `pyarrow` with `sequence_id`, `sequence`, `teacher_logits` (`pa.large_binary()` when `large_binary=True`, else `pa.binary()`), and optional `teacher_neff` (`pa.float32()`) / `teacher_neff_sqrt_l` (`pa.float32()`), then test:

```python
def test_teacher_logits_ignored_when_disabled(tmp_path: Path) -> None:
    shard = _write_distill_shard(tmp_path / "s.parquet", [("a", "ACDE", b"\x00" * 160, 50.0, 25.0)])
    ds = ShardedProteinDataset(shard, shuffle_shards=False, shuffle_rows=False)
    row = next(iter(ds))
    assert "teacher_logits" not in row


@pytest.mark.parametrize("large_binary", [False, True])
def test_teacher_logits_loaded_and_neff_gated(tmp_path: Path, large_binary: bool) -> None:
    payload = b"\x01" * 160
    shard = _write_distill_shard(
        tmp_path / "s.parquet",
        [
            ("pass_both", "ACDE", payload, 30.0, 5.0),
            ("fail_neff", "ACDE", payload, 10.0, 5.0),
            ("fail_sqrt", "ACDE", payload, 30.0, 1.5),
            ("nan_neff", "ACDE", payload, float("nan"), 5.0),
            ("none_neff", "ACDE", payload, None, 5.0),
            ("null_logits", "ACDE", None, 50.0, 10.0),
        ],
        large_binary=large_binary,
    )
    ds = ShardedProteinDataset(
        shard,
        shuffle_shards=False,
        shuffle_rows=False,
        load_teacher_logits=True,
        min_teacher_neff=20.0,
        min_teacher_neff_sqrt_l=2.0,
    )
    got = {r["sequence_id"]: r["teacher_logits"] for r in ds}
    assert got["pass_both"] == payload
    for key in ("fail_neff", "fail_sqrt", "nan_neff", "none_neff", "null_logits"):
        assert got[key] is None


def test_teacher_logits_missing_neff_column_raises(tmp_path: Path) -> None:
    shard = _write_distill_shard(
        tmp_path / "s.parquet", [("a", "ACDE", b"\x00" * 160, None, None)], include_neff=False
    )
    with pytest.raises(ValueError, match="teacher_neff"):
        ShardedProteinDataset(shard, load_teacher_logits=True, min_teacher_neff=20.0)
    with pytest.raises(ValueError, match="teacher_neff_sqrt_l"):
        ShardedProteinDataset(shard, load_teacher_logits=True, min_teacher_neff_sqrt_l=2.0)


def test_mixed_interleaved_distilled_plain_and_sample_members(tmp_path: Path) -> None:
    distilled = _write_distill_shard(tmp_path / "d.parquet", [("d0", "ACDE", b"\x02" * 160, 40.0, 6.0)])
    plain = write_sequence_shards(tmp_path / "plain", n_shards=1, rows_per_shard=2)
    members_shard = tmp_path / "members.parquet"
    pq.write_table(
        pa.table({
            "sequence_id": ["m0"],
            "sequence": ["ACDE"],
            "member_ids": [["m0_a", "m0_b"]],
            "member_sequences": [["ACDE", "ACDF"]],
        }),
        members_shard,
    )
    mix = InterleavedDataset(
        [
            ShardedProteinDataset(distilled, load_teacher_logits=True, min_teacher_neff_sqrt_l=2.0),
            ShardedProteinDataset(plain, load_teacher_logits=True, min_teacher_neff_sqrt_l=2.0),
            ShardedProteinDataset(
                members_shard, sample_members=True, load_teacher_logits=True, min_teacher_neff_sqrt_l=2.0
            ),
        ],
        [1 / 3, 1 / 3, 1 / 3],
        num_samples=12,
        seed=0,
    )
    rows = list(mix)
    assert any(r.get("teacher_logits") is not None for r in rows)
    assert any(r.get("teacher_logits") is None for r in rows)


def test_sample_members_with_teacher_logits_column_raises(tmp_path: Path) -> None:
    shard = tmp_path / "bad.parquet"
    pq.write_table(
        pa.table({
            "sequence_id": ["m0"],
            "sequence": ["ACDE"],
            "member_ids": [["m0_a"]],
            "member_sequences": [["ACDE"]],
            "teacher_logits": pa.array([b"\x00" * 160], type=pa.binary()),
        }),
        shard,
    )
    with pytest.raises(ValueError, match="sample_members"):
        ShardedProteinDataset(shard, sample_members=True, load_teacher_logits=True)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/data/sequence/test_dataset.py -k "teacher_logits or mixed_interleaved" -v`
Expected: FAIL (`TypeError: unexpected keyword argument 'load_teacher_logits'`).

- [ ] **Step 3: Implement teacher-logit loading and `Neff` gating in `src/oplm/data/sequence/dataset.py`**

- Add `TEACHER_LOGITS_COLUMN = "teacher_logits"`, `TEACHER_NEFF_COLUMN = "teacher_neff"`, `TEACHER_NEFF_SQRT_L_COLUMN = "teacher_neff_sqrt_l"`.
- In `ShardedProteinDataset.__init__`: store `_load_teacher_logits`, `_min_teacher_neff`, `_min_teacher_neff_sqrt_l`, and per-shard `_shard_has_teacher_logits: list[bool]`. When `load_teacher_logits=True` and a shard contains `TEACHER_LOGITS_COLUMN`:
  - If `sample_members=True`, raise `ValueError`.
  - If `min_teacher_neff is not None` and `TEACHER_NEFF_COLUMN` is missing, raise `ValueError`.
  - If `min_teacher_neff_sqrt_l is not None` and `TEACHER_NEFF_SQRT_L_COLUMN` is missing, raise `ValueError`.
- In `_read_rows`: when `_load_teacher_logits` and `_shard_has_teacher_logits[shard_idx]`, append `TEACHER_LOGITS_COLUMN` (and whichever `Neff` column(s) are gated) to `columns`. For each row, convert `neff` / `neff_sqrt_l` via `.as_py()` and gate `logits = None` whenever a configured gate has `val is None or not math.isfinite(val) or val < threshold`. Set `row["teacher_logits"] = logits` (and `None` on shards lacking `teacher_logits` when `_load_teacher_logits=True`). In `_read_members`, when `_load_teacher_logits=True`, include `"teacher_logits": None` (safe because shards with `teacher_logits` were rejected at init).

- [ ] **Step 4: Run dataset tests and lint/type checks**

Run: `.venv/bin/python -m pytest tests/data/sequence/test_dataset.py -q && ruff check src/oplm/data/sequence/dataset.py && ty check src/oplm/data/sequence/dataset.py`
Expected: PASS, 0 lint/type errors.

- [ ] **Step 5: Commit**

```bash
git add src/oplm/data/sequence/dataset.py tests/data/sequence/test_dataset.py
git commit -m "feat(data): read optional teacher_logits and Neff gates in ShardedProteinDataset"
```

---

### Task 3: Teacher-logit collation, packing, and dataloader wiring

**Files:**
- Modify: `src/oplm/data/sequence/collate.py:242-390`
- Modify: `src/oplm/data/sequence/loaders.py:115-172`
- Test: `tests/data/sequence/test_collate.py`, `tests/data/sequence/test_loaders.py`

**Interfaces:**
- Consumes: `cfg.distill: DistillConfig | None` (Task 1); `ShardedProteinDataset(..., load_teacher_logits, min_teacher_neff, min_teacher_neff_sqrt_l)` and row key `"teacher_logits": bytes | None` (Task 2).
- Produces:
  - `MLMCollator(..., load_teacher_logits: bool = False)`.
  - When `load_teacher_logits=True`, emitted batch dicts add `"teacher_logits": Tensor` `(R, T, 20)` `torch.float16` (or `(B, T, 20)` unpacked) and `"teacher_mask": Tensor` `(R, T)` `torch.bool` (or `(B, T)` unpacked), guaranteed `teacher_mask ⊆ (labels != -100)` and `teacher_logits[teacher_mask]` all finite.
  - `build_train_dataloader(cfg)` enables teacher-logit loading whenever `cfg.distill is not None`; `build_sequence_eval_dataloader` never does.

- [ ] **Step 1: Write failing tests in `tests/data/sequence/test_collate.py` and `tests/data/sequence/test_loaders.py`**

In `tests/data/sequence/test_collate.py`, add helper `_encode_logits(arr: np.ndarray) -> bytes` (`np.asarray(arr, dtype="<f2").tobytes()`), and tests:

```python
def test_teacher_logits_preserves_exact_masks_and_alignment(tokenizer) -> None:
    seqs = ["ACDEFGHIKLMNPQRSTVWY", "MKTAYIAKQRQISFVKSHFSRQ"]
    arrays = [np.arange(len(s) * 20, dtype="<f2").reshape(len(s), 20) / 100.0 for s in seqs]
    plain_batch = _batch(seqs)
    distill_batch = [
        {"sequence_id": "0", "sequence": seqs[0], "teacher_logits": _encode_logits(arrays[0])},
        {"sequence_id": "1", "sequence": seqs[1], "teacher_logits": None},
    ]
    c_plain = MLMCollator(tokenizer, max_length=64, mask_prob=0.5, deterministic=True, seed=7)
    c_dist = MLMCollator(
        tokenizer, max_length=64, mask_prob=0.5, deterministic=True, seed=7, load_teacher_logits=True
    )
    out_plain = c_plain(plain_batch)
    out_dist = c_dist(distill_batch)
    assert torch.equal(out_plain["input_ids"], out_dist["input_ids"])
    assert torch.equal(out_plain["labels"], out_dist["labels"])
    assert out_dist["teacher_logits"].dtype == torch.float16
    assert out_dist["teacher_logits"].shape == (*out_dist["labels"].shape, 20)
    assert out_dist["teacher_mask"].dtype == torch.bool
    # Row 0: every masked position has teacher_mask=True and matches arrays[0][pos - 1]
    masked0 = (out_dist["labels"][0] != -100).nonzero(as_tuple=False).squeeze(-1)
    assert torch.equal(out_dist["teacher_mask"][0], out_dist["labels"][0] != -100)
    for pos in masked0.tolist():
        expected = torch.tensor(arrays[0][pos - 1], dtype=torch.float16)
        assert torch.equal(out_dist["teacher_logits"][0, pos], expected)
    # Row 1: teacher_logits=None -> all False
    assert not out_dist["teacher_mask"][1].any()


def test_teacher_logits_truncation_nan_noncanonical_and_memoryview(tokenizer) -> None:
    # Raw length 20 truncated to max_length-2 = 10; position 2 is X (noncanonical);
    # position 4 has NaN in channel 0 (simulating K=2 uncomputed fold).
    seq = "ACXEFGHIKLMNPQRSTVWY"
    arr = np.ones((len(seq), 20), dtype="<f2")
    arr[4, 0] = np.nan
    raw = memoryview(arr.tobytes())
    collator = MLMCollator(
        tokenizer, max_length=12, mask_prob=1.0, deterministic=True, seed=0, load_teacher_logits=True
    )
    out = collator([{"sequence_id": "0", "sequence": seq, "teacher_logits": raw}])
    # Clipped residues occupy token indices 1..10 (raw indices 0..9):
    # raw idx 2 ('X' -> token pos 3) and raw idx 4 (NaN -> token pos 5) must be False.
    assert not out["teacher_mask"][0, 3]
    assert not out["teacher_mask"][0, 5]
    assert out["teacher_mask"][0].sum().item() == 8
    assert torch.isfinite(out["teacher_logits"][out["teacher_mask"]]).all()


def test_teacher_logits_byte_length_mismatch_raises(tokenizer) -> None:
    collator = MLMCollator(tokenizer, max_length=32, mask_prob=0.5, load_teacher_logits=True)
    with pytest.raises(ValueError, match="teacher_logits"):
        collator([{"sequence_id": "0", "sequence": "ACDE", "teacher_logits": b"\x00" * 40}])


def test_teacher_logits_packing_matches_unpacked(tokenizer) -> None:
    seqs = ["ACDEFGHIKL", "MNPQRSTVWY"]
    arrays = [np.arange(200, dtype="<f2").reshape(10, 20) / 10.0, np.ones((10, 20), dtype="<f2") * 3.0]
    items = [
        {"sequence_id": "0", "sequence": seqs[0], "teacher_logits": _encode_logits(arrays[0])},
        {"sequence_id": "1", "sequence": seqs[1], "teacher_logits": _encode_logits(arrays[1])},
    ]
    unpacked = MLMCollator(
        tokenizer, max_length=32, mask_prob=0.5, deterministic=True, seed=3, load_teacher_logits=True
    )(items)
    packed = MLMCollator(
        tokenizer,
        max_length=32,
        mask_prob=0.5,
        deterministic=True,
        seed=3,
        pack_sequences=True,
        pack_rows=1,
        load_teacher_logits=True,
    )(items)
    assert packed["teacher_logits"].shape == (1, 32, 20)
    assert packed["teacher_mask"].shape == (1, 32)
    # Segment 1 occupies [0:12], segment 2 occupies [12:24]
    assert torch.equal(packed["teacher_logits"][0, :12], unpacked["teacher_logits"][0, :12])
    assert torch.equal(packed["teacher_mask"][0, :12], unpacked["teacher_mask"][0, :12])
    assert torch.equal(packed["teacher_logits"][0, 12:24], unpacked["teacher_logits"][1, :12])
    assert torch.equal(packed["teacher_mask"][0, 12:24], unpacked["teacher_mask"][1, :12])
```

And in `tests/data/sequence/test_loaders.py`:
- Test `build_train_dataloader(cfg)` with `cfg.distill = DistillConfig(min_neff_sqrt_l=2.0)` yields batches carrying `"teacher_logits"` and `"teacher_mask"` (both packed token mode and unpacked sequence mode), while `build_sequence_eval_dataloader(path, cfg)` omits both keys even when `cfg.distill` is set.

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/data/sequence/test_collate.py tests/data/sequence/test_loaders.py -k teacher_logits -v`
Expected: FAIL (`TypeError: unexpected keyword argument 'load_teacher_logits'`).

- [ ] **Step 3: Implement collation, packing, and loader wiring**

- In `src/oplm/data/sequence/collate.py`:
  - Add `load_teacher_logits: bool = False` keyword-only arg to `MLMCollator.__init__`.
  - Add helper `_collect_teacher_logits(batch, labels, canonical_ids, raw_lengths) -> tuple[Tensor, Tensor]` allocating `(B, T, 20)` `torch.float16` zeros and `(B, T)` `torch.bool` zeros. For each row with non-`None` `item.get("teacher_logits")`: validate `isinstance(raw, (bytes, bytearray, memoryview))` and `len(raw) == raw_len * 40`; decode via `np.frombuffer(raw, dtype="<f2").reshape(raw_len, 20)`; index only masked canonical positions `pos - 1`, copy via `np.array(picked, copy=True)` (avoids non-writable PyTorch warning), filter by `np.isfinite(picked).all(axis=-1)`, and scatter into `teacher_logits` / `teacher_mask`.
  - Extend `_pack(...)` to accept optional `teacher_logits` and `teacher_mask` and slice them into `(len(rows), width, 20)` and `(len(rows), width)` in the existing `start:end` loop.
- In `src/oplm/data/sequence/loaders.py`:
  - In `build_train_dataloader`, set `distill = cfg.distill`, pass `load_teacher_logits=distill is not None`, `min_teacher_neff=distill.min_neff if distill else None`, `min_teacher_neff_sqrt_l=distill.min_neff_sqrt_l if distill else None` to `ShardedProteinDataset`, and `load_teacher_logits=distill is not None` to `MLMCollator`. Leave `build_sequence_eval_dataloader` unchanged.

- [ ] **Step 4: Run collation, batching, and loader tests + lint/type checks**

Run: `.venv/bin/python -m pytest tests/data/sequence/test_collate.py tests/data/sequence/test_loaders.py tests/data/sequence/test_batching.py -q && ruff check src/oplm/data/sequence/ && ty check src/oplm/data/sequence/`
Expected: PASS, 0 lint/type errors.

- [ ] **Step 5: Commit**

```bash
git add src/oplm/data/sequence/collate.py src/oplm/data/sequence/loaders.py tests/data/sequence/test_collate.py tests/data/sequence/test_loaders.py
git commit -m "feat(data): collate and pack masked teacher logits for distillation"
```

---

### Task 4: `DistillMLMTask` loss, channel mapping, and `Trainer` integration

**Files:**
- Modify: `src/oplm/training/task.py:24,97-138`
- Modify: `src/oplm/training/trainer.py:262-285`
- Test: `tests/training/test_task.py`

**Interfaces:**
- Consumes: `DistillConfig`, `validate_distill_compat` (Task 1); batch keys `"teacher_logits"` `(R, T, 20)` fp16 and `"teacher_mask"` `(R, T)` bool (Task 3); `get_tokenizer()` (`oplm.data.tokenizer`).
- Produces:
  - `DistillMLMTask(MLMTask)` exported in `oplm.training.task.__all__`.
  - `Trainer(cfg)` automatically selects `DistillMLMTask(cfg.distill)` when `task is None and cfg.distill is not None`, and validates `validate_distill_compat(cfg)`.
  - `StepResult.metrics` keys: `"loss_ce"`, `"distill_frac"`, and (when `n_distill > 0`) `"loss_kl"`.

- [ ] **Step 1: Write failing tests in `tests/training/test_task.py`**

```python
def test_distill_task_alpha_zero_matches_mlm_loss_and_grads(tmp_path: Path) -> None:
    cfg = tiny_train_cfg(tmp_path, _write_mixed_distill_parquet(tmp_path / "m.parquet"))
    cfg.distill = DistillConfig(alpha=0.0, temperature=1.5)
    mlm_task = MLMTask()
    distill_task = DistillMLMTask(cfg.distill)
    assert isinstance(distill_task, MLMTask)
    model_a = mlm_task.build_model(cfg, None)
    model_b = distill_task.build_model(cfg, None)
    model_b.load_state_dict(model_a.state_dict())
    model_a.eval()
    model_b.eval()
    batch = next(iter(distill_task.build_dataloader(cfg)))
    r_mlm = mlm_task.step(model_a, batch)
    r_dist = distill_task.step(model_b, batch)
    assert torch.allclose(r_dist.loss, r_mlm.loss, atol=1e-6)
    r_mlm.loss.backward()
    r_dist.loss.backward()
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters(), strict=True):
        if p_a.grad is not None:
            assert torch.allclose(p_b.grad, p_a.grad, atol=1e-5)
    assert r_dist.metrics["loss_ce"] == pytest.approx(r_mlm.loss.item(), abs=1e-6)
    assert "loss_kl" in r_dist.metrics
    assert 0.0 < r_dist.metrics["distill_frac"] < 1.0


def test_distill_task_channel_order_maps_teacher_alphabet_to_tokenizer_ids() -> None:
    # 'A' is channel 0 in ARNDCEQGHILKMFPSTWYV, token id 5 in OplmTokenizerFast ('L' is id 4).
    tok = get_tokenizer()
    assert tok.convert_tokens_to_ids("L") == 4 and tok.convert_tokens_to_ids("A") == 5
    logits = torch.zeros((1, 3, tok.vocab_size), dtype=torch.float32, requires_grad=True)
    teacher = torch.zeros((1, 3, 20), dtype=torch.float16)
    teacher[0, 1, 0] = 10.0  # strong spike on 'A'
    batch = {
        "input_ids": torch.tensor([[0, 32, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": torch.tensor([[-100, 5, -100]]),
        "teacher_logits": teacher,
        "teacher_mask": torch.tensor([[False, True, False]]),
    }
    stub = _FixedLogitsModel(logits)
    result = DistillMLMTask(DistillConfig(alpha=1.0, temperature=1.0)).step(stub, batch)
    result.loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 1, 5].item() < -0.5  # pushes 'A' (id 5) UP
    assert logits.grad[0, 1, 4].item() > 0.0   # pushes 'L' (id 4) DOWN


def test_distill_task_packed_matches_unpacked_loss(tmp_path: Path) -> None:
    # Identical two sequences + teacher logits: packed (1 row) vs unpacked (2 rows) match.
    assert torch.allclose(packed_result.loss, unpacked_result.loss, atol=1e-5)


def test_distill_task_handles_zero_distilled_and_zero_masked_batches() -> None:
    # 1) teacher_mask all False (mirrors _canon_self_check [:4] slicing with no distilled rows):
    assert torch.allclose(r_plain.loss, expected_ce, atol=1e-6)
    assert "loss_kl" not in r_plain.metrics
    assert r_plain.metrics["distill_frac"] == 0.0
    # 2) labels all -100:
    assert r_empty.loss.item() == 0.0 and r_empty.loss.requires_grad
    r_empty.loss.backward()
    assert r_empty.metrics == {}


def test_trainer_distill_mixed_packed_and_branch_from_plain_mlm(tmp_path: Path) -> None:
    # Stage 1: plain MLM for 2 steps -> checkpoint-2.
    # Stage 2: branch_from checkpoint-2 with pack_sequences=True, distill=DistillConfig(
    #   alpha=0.5, temperature=1.5, min_neff_sqrt_l=2.0), mixed {distilled, plain, members(sample_members=True)}.
    assert isinstance(trainer.task, DistillMLMTask) and isinstance(trainer.task, MLMTask)
    assert all(
        math.isfinite(m["train/loss_ce"]) and 0.0 <= m["train/distill_frac"] <= 1.0
        for _, m in cb.train_logs
    )
    assert any("train/loss_kl" in m and math.isfinite(m["train/loss_kl"]) for _, m in cb.train_logs)
    reloaded = load_config(["--config", str(stage2_dir / "checkpoint-4" / "config.yaml")])
    assert reloaded.distill == stage2_cfg.distill
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/training/test_task.py -k distill -v`
Expected: FAIL (`ImportError: cannot import name 'DistillMLMTask' from 'oplm.training.task'`).

- [ ] **Step 3: Implement `DistillMLMTask` in `src/oplm/training/task.py` and wire `Trainer` in `src/oplm/training/trainer.py`**

- In `src/oplm/training/task.py`:
  - Export `"DistillMLMTask"` in `__all__`.
  - Implement `class DistillMLMTask(MLMTask)`:
    - `__init__(self, distill_cfg: DistillConfig) -> None`: save `self._distill = distill_cfg`, resolve `self._teacher_token_ids = torch.tensor([tok.convert_tokens_to_ids(aa) for aa in distill_cfg.teacher_alphabet], dtype=torch.long)` using `get_tokenizer()`, keep a lazy per-device cache `_device_token_ids: Tensor | None = None`.
    - `step(self, model: nn.Module, batch: Any) -> StepResult`:
      - Run `outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=None, segment_ids=batch.get("segment_ids"))`.
      - Compute masks `masked = labels != -100`, `teacher_mask = batch["teacher_mask"] & masked`, counts `n_masked`, `n_distill`.
      - If `n_masked == 0`: return `StepResult(loss=logits.sum() * 0.0, tokens=..., samples=..., metrics={})`.
      - Compute per-token fp32 `ce` with `F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100, reduction="none").view_as(labels)`.
      - `plain_ce_sum = ce[masked & ~teacher_mask].sum()`, `distill_ce_sum = ce[teacher_mask].sum()`.
      - If `n_distill > 0`: slice `s20 = logits[teacher_mask].index_select(-1, token_ids)`, `t20 = batch["teacher_logits"][teacher_mask].float()`, `T = self._distill.temperature`, `kl_sum = (F.kl_div(F.log_softmax(s20 / T, dim=-1), F.softmax(t20 / T, dim=-1), reduction="none").sum(dim=-1) * (T * T)).sum()`. Else `kl_sum = logits.new_zeros(())`.
      - `alpha = self._distill.alpha`; `loss = (plain_ce_sum + (1.0 - alpha) * distill_ce_sum + alpha * kl_sum) / n_masked`.
      - Build `metrics = {"loss_ce": float((plain_ce_sum + distill_ce_sum).detach().item() / n_masked), "distill_frac": float(n_distill / n_masked)}`; if `n_distill > 0`, add `metrics["loss_kl"] = float(kl_sum.detach().item() / n_distill)`.
- In `src/oplm/training/trainer.py`:
  - Import `DistillMLMTask` alongside `MLMTask`, and `validate_distill_compat` alongside `validate_token_batching`.
  - Call `validate_distill_compat(cfg)` right after `validate_token_batching(cfg)`.
  - Set `self.task`: `task` if provided, else `DistillMLMTask(cfg.distill)` if `cfg.distill is not None`, else `MLMTask()`.

- [ ] **Step 4: Run task and trainer tests + lint/type checks**

Run: `.venv/bin/python -m pytest tests/training/test_task.py tests/training/test_packing_train.py -q && ruff check src/oplm/training/task.py src/oplm/training/trainer.py && ty check src/oplm/training/task.py src/oplm/training/trainer.py`
Expected: PASS, 0 lint/type errors.

- [ ] **Step 5: Commit**

```bash
git add src/oplm/training/task.py src/oplm/training/trainer.py tests/training/test_task.py
git commit -m "feat(training): add DistillMLMTask and auto-wire distill config in Trainer"
```

---

### Task 5: Docs and docs-guard update + full verification

**Files:**
- Modify: `docs/data.md:544-570`
- Modify: `docs/config.md:467-510`
- Modify: `docs/training.md:1137-1165`
- Modify: `tests/test_docs.py:12-14,122-139`

**Interfaces:**
- Consumes: `DistillConfig` (Task 1), Parquet schema (`teacher_logits`, `teacher_neff`, `teacher_neff_sqrt_l`) (Tasks 2–3), `DistillMLMTask` metrics (`train/loss_ce`, `train/loss_kl`, `train/distill_frac`) (Task 4).
- Produces: updated home docs (`docs/data.md`, `docs/config.md`, `docs/training.md`) and `distill.*` coverage in `tests/test_docs.py`.

- [ ] **Step 1: Add `"distill": DistillConfig` to `tests/test_docs.py`**

Update `tests/test_docs.py`: import `DistillConfig` from `oplm.config`, add `"distill": DistillConfig` to `classes`, and extend the regex in `test_documented_config_fields_exist` to `(model|train|data|fold|distill)`.

- [ ] **Step 2: Update `docs/data.md`, `docs/config.md`, and `docs/training.md`**

- `docs/data.md` (*Dataset formats*): add `teacher_logits`, `teacher_neff`, `teacher_neff_sqrt_l` rows to the Parquet column table and a short paragraph covering channel-order contract (`distill.teacher_alphabet`), `NaN` partial-validity fallback (`K=2` / non-canonical), `Neff` gating, and `sample_members` compatibility.
- `docs/config.md`: add `## Distillation fields (distill.*)` between `Eval datasets` and `Fold fields` with the five fields, types, defaults, and validation rules (and link to `data.md` and `training.md`).
- `docs/training.md` (under *Training tasks (`TrainTask`)*): document `DistillMLMTask` auto-selection when `distill:` is set, the per-masked-token blended loss (`alpha`, `temperature`), logged `train/loss_ce`, `train/loss_kl`, `train/distill_frac`, and a short YAML snippet showing a `train.branch_from` decay branch with mixed distilled + plain sources.

- [ ] **Step 3: Run full fast test suite, docs guard, linter, and type checker**

Run: `.venv/bin/python -m pytest -m "not slow" -q && ruff check src/ && ty check src/`
Expected: all fast tests (including `tests/test_docs.py`) PASS, `ruff check src/` clean, `ty check src/` clean.

- [ ] **Step 4: Commit**

```bash
git add docs/data.md docs/config.md docs/training.md tests/test_docs.py docs/superpowers/specs/2026-11-24-teacher-logit-distillation-design.md docs/superpowers/plans/2026-11-25-teacher-logit-distillation.md
git commit -m "docs: document teacher-logit distillation schema, config, and training"
```

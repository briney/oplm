# OPLM Data Tooling — Implementation Plan

This plan builds the OPLM **data tooling** from scratch per
[`docs/DATA_TOOLING.md`](docs/DATA_TOOLING.md). It is fully standalone: every
file path, signature, algorithm, constant, and test is specified here.

> **From-scratch.** Assume `src/oplm/data/` does **not exist** at implementation
> time — the previous implementation is removed before you start. Do not edit or
> port old files; write every module new. `src/oplm/eval/data/` is also gone; the
> eval harness rewrite is a **separate** effort (future `docs/EVAL.md`) and is out
> of scope here. This plan delivers the `oplm.data` package that the eval harness
> will later import, plus the trainer integration.

Phases are dependency-ordered; each produces independently testable code. Check a
box only when its code **and** its tests pass.

## Conventions

- Python ≥ 3.11; `from __future__ import annotations` at the top of every file.
- Type hints on all signatures; Google-style docstrings on public APIs.
- `ruff format` / `ruff check`; line length 100; absolute imports only.
- No logic in `__init__.py`. No `os.path` — use `pathlib.Path`. No bare `except`.
- Tensor shape comments on non-trivial ops, e.g. `# (B, T) -> (B, T, V)`.
- Tests: `pytest`, mirror layout under `tests/data/`. Prefer **real** data
  fixtures. Mark heavy/model tests `@pytest.mark.slow`; session-scope large
  fixtures.

## Glossary

- `B` = batch size; `T` = padded sequence length (incl. `<cls>`/`<eos>`).
- `L` = raw residue length of one sequence.
- `n_eligible` = count of maskable positions in a sequence (excludes specials/pad).
- `k` = `round(mask_prob * n_eligible)` = number of positions masked per sequence.
- `V` = vocab size = **33** (canonical tokenizer).
- `w_i` = per-residue masking weight at position `i`.

## Canonical facts (depend on these; do not re-derive)

- **Tokenizer:** `OplmTokenizerFast` from
  [`src/oplm/model/tokenization_oplm.py`](src/oplm/model/tokenization_oplm.py),
  exported by `oplm.model`. 33 tokens, ESM-C order:
  - specials: `<cls>`=0, `<pad>`=1, `<eos>`=2, `<unk>`=3, `<mask>`=32.
  - 20 standard amino acids occupy contiguous IDs **4–23**
    (`L,A,G,V,S,E,R,T,I,D,P,K,Q,N,F,Y,M,H,W,C`).
  - ambiguous/gap/structure tokens: `X`24,`B`25,`U`26,`Z`27,`O`28,`.`29,`-`30,`|`31.
  - sanity: `OplmTokenizerFast()("MEEPQ").input_ids == [0, 20, 9, 9, 14, 16, 2]`.
  - templating wraps every sequence with `<cls> … <eos>`; pad with id 1.
- **Model (for integration tests):** `OplmForMaskedLM` from `oplm.model`.
  `forward(input_ids, attention_mask, labels=None, output_attentions=None,
  output_hidden_states=None)` → `MaskedLMOutput(loss, logits, hidden_states,
  attentions)`. Loss is `cross_entropy(logits.view(-1,V), labels.view(-1),
  ignore_index=-100)`. `output_attentions=True` exposes attention (structure
  eval); `output_hidden_states=True` exposes hidden states (downstream eval).
- **Pooling helpers:** `mean_pool(hidden, attention_mask)` and `cls_pool(hidden)`
  in [`src/oplm/model/embedding.py`](src/oplm/model/embedding.py) (for downstream).
- **Config:** `OplmConfig`/`ModelConfig`/`TrainConfig`/`DataConfig` plus
  `TrainDatasetEntry`/`EvalDatasetEntry` in
  [`src/oplm/config.py`](src/oplm/config.py).
- **Deps:** `pyarrow` and `numpy` are core deps (parquet, math). `biopython` and
  `scikit-learn` are under the `train` extra (structure parsing must lazy-import
  biopython). Use `torch` for all tensor/RNG work.

> **Note:** `ModelConfig.vocab_size` defaults to **33**, matching the 33-token
> tokenizer (the legacy `32` default and the value in `configs/model/base.yaml`
> were corrected). Integration tests can rely on the default; no override needed.

---

## Phase 0 — Scaffolding & dependencies

- [ ] **0.1 Create the package tree** (empty modules, with module docstrings):
  ```
  src/oplm/data/__init__.py
  src/oplm/data/tokenizer.py
  src/oplm/data/config.py
  src/oplm/data/sequence/__init__.py
  src/oplm/data/sequence/dataset.py
  src/oplm/data/sequence/collate.py
  src/oplm/data/sequence/loaders.py
  src/oplm/data/structure/__init__.py
  src/oplm/data/structure/loader.py
  src/oplm/data/variant/__init__.py
  src/oplm/data/variant/loader.py
  src/oplm/data/downstream/__init__.py
  src/oplm/data/downstream/loader.py
  ```
- [ ] **0.2 Test tree:** create `tests/data/__init__.py`,
  `tests/data/sequence/__init__.py`, etc., and `tests/data/conftest.py` for
  shared fixtures. Create `tests/data/fixtures/` for real data.
- [ ] **0.3 Import hygiene:** `import oplm.data` (and any submodule) must **not**
  import `oplm.eval`. `oplm/__init__.py` already imports only `.model`; keep it
  that way. Add a test asserting `import oplm.data` succeeds without the eval
  package present.
- [ ] **0.4 Deps check:** confirm `pyarrow`/`numpy` in core deps and
  `biopython`/`scikit-learn` in the `train` extra of `pyproject.toml` (already
  present). No new deps required.

---

## Phase 1 — Configuration surface

- [ ] **1.1 Extend `DataConfig`** in `src/oplm/config.py` with three fields
  (defaults shown):
  ```python
  mask_token_prob: float = 0.8       # of masked positions -> <mask>
  random_token_prob: float = 0.1     # of masked positions -> random canonical AA
  weighted_masking: bool = False     # honor masking_weights column when True
  ```
  Add validation (in `__post_init__`): `0 <= mask_token_prob <= 1`,
  `0 <= random_token_prob <= 1`, and `mask_token_prob + random_token_prob <= 1`.
  Leave existing fields untouched: `train`, `eval`, `mask_prob=0.15`,
  `num_workers=4`, `pin_memory=True`, `prefetch_factor=4`, `shuffle_shards=True`,
  `shuffle_rows=True`.
- [ ] **1.2 Update `src/oplm/configs/data/base.yaml`** — add:
  ```yaml
  mask_token_prob: 0.8
  random_token_prob: 0.1
  weighted_masking: false   # set true to honor the masking_weights column
  ```
- [ ] **1.3 Implement `data/config.py`** — parsing helpers (entry dataclasses are
  imported from `oplm.config`, not redefined). If equivalent functions still
  exist in `config.py`, remove them there so `data/config.py` is the single home.
  - `parse_train_configs(raw) -> list[TrainDatasetEntry]`:
    - `raw` is a `str` (single path → one entry, `fraction=1.0`, `name="train"`)
      **or** a mapping `{name: {path: str, fraction?: float}}`.
    - Normalize fractions to sum 1.0. Omitted fractions split the remaining mass
      equally among entries lacking one. Validate each `fraction >= 0` and the
      total `> 0`. Raise `ValueError` on missing `path`.
  - `parse_eval_configs(raw, default_eval_every) -> list[EvalDatasetEntry]`:
    - `raw` is a mapping `{name: {path, type, eval_every?, metrics?, **extra}}`.
    - Require `path` and `type`; fold unknown keys into `extra`. `eval_every`
      defaults to `None` (caller falls back to `default_eval_every`).
- [ ] **1.4 Tests** (`tests/data/test_config.py`): single-path expands to one
  full-weight entry; multi-dataset fractions normalize; omitted fractions split
  remainder; missing `path` raises; eval parsing requires `path`+`type` and
  routes extras into `extra`.

---

## Phase 2 — Tokenizer access layer (`data/tokenizer.py`)

Single source of truth = `OplmTokenizerFast`. This module defines **no
vocabulary**; it provides an accessor, derived id constants, and per-residue
vector alignment.

- [ ] **2.1 Accessor:** `get_tokenizer() -> OplmTokenizerFast` returning a fresh
  `OplmTokenizerFast()` (imported from `oplm.model`). Optionally cache a
  module-level singleton; if so, never mutate it.
- [ ] **2.2 Derived id constants** (compute from the tokenizer instance, never
  hardcode literals):
  - `special_ids(tok) -> set[int]` = `set(tok.all_special_ids)` → `{0,1,2,3,32}`.
  - `non_maskable_ids(tok) -> set[int]` = the special ids (cls/pad/eos/unk/mask).
  - `mask_token_id(tok)` = `tok.mask_token_id` (32); `pad_token_id(tok)` =
    `tok.pad_token_id` (1).
  - `canonical_amino_acid_ids(tok) -> Tensor`: the 20 standard-AA token ids. Build
    by converting each char of `"LAGVSERTIDPKQNFYMHWC"` through the tokenizer's
    vocab map (`tok.convert_tokens_to_ids`) → expect `range(4, 24)`. Return a
    `torch.long` tensor. This helper is the shared sampling pool reused by the
    collator and (later) eval metrics.
- [ ] **2.3 Alignment helper:**
  `align_per_residue(values: Sequence[Sequence[float] | None], *, lengths:
  Sequence[int], total_len: int, fill_special: float = 0.0, fill_pad: float =
  0.0) -> Tensor`:
  - Produces a `(B, total_len)` float tensor aligned to tokenized `input_ids`:
    for each row, position 0 (`<cls>`) and the `<eos>` position get
    `fill_special`; residue positions `1..1+len_i` get the (already
    truncation-clipped) values; trailing pad positions get `fill_pad`.
  - Rows whose `values` is `None` → fill residue positions with `1.0` so a
    missing-weight row is treated as uniform; document this choice.
  - Mirror the **exact** truncation rule used for tokens (Phase 4.1): values are
    clipped to `max_length - 2` residues. Raise `ValueError` if a non-`None`
    `values` length disagrees with its sequence length before truncation.
- [ ] **2.4 Tests** (`tests/data/test_tokenizer.py`):
  - **Parity guard:** for a battery of sequences, ids equal
    `OplmTokenizerFast()(seq).input_ids`; `mask_token_id==32`,
    `pad_token_id==1`, `special_ids=={0,1,2,3,32}`,
    `canonical_amino_acid_ids().tolist()==list(range(4,24))`. (This is the test
    that catches any vocab regression.)
  - **Alignment:** weights line up with `input_ids` (specials/pad get fill);
    truncation matches; length mismatch raises; `None` row → uniform.

---

## Phase 3 — Sequence datasets (`data/sequence/dataset.py`)

### On-disk format
Parquet file (`.parquet`/`.parq`/`.pq`) or a directory of such shards. Required
columns `sequence_id` (str), `sequence` (str, raw one-letter AAs). Optional
`masking_weights` (`list[float]`, length == `len(sequence)`).

### Seed mixing (use these exact constants for reproducibility)
```python
_PHI    = 0x9E3779B97F4A7C15   # golden-ratio mix
_PRIME  = 0x0100_0003
_MASK32 = 0xFFFF_FFFF
def _epoch_seed(base, epoch):        return (( _PHI ^ base) + epoch * _PRIME) & _MASK32
def _shard_row_seed(epoch_seed, s):  return (epoch_seed + 1009 + s)        & _MASK32
```

### Distributed/worker context
Provide a helper resolving `(rank, world_size, worker_id, num_workers)`:
- rank/world_size from `torch.distributed` if `is_available() and
  is_initialized()`, else env `RANK`/`WORLD_SIZE`, else `(0, 1)`.
- worker_id/num_workers from `torch.utils.data.get_worker_info()`, else `(0, 1)`.
- joint index = `rank * num_workers + worker_id`; stride =
  `world_size * num_workers`.

- [ ] **3.1 `ShardedProteinDataset(IterableDataset)`**:
  ```python
  ShardedProteinDataset(
      path: str | Path, *,
      shuffle_shards: bool = True,
      shuffle_rows: bool = True,
      seed: int = 0,
      load_masking_weights: bool = False,
  )
  ```
  - `__init__`: resolve `path` to a sorted list of shard files (single file → one
    shard). Read each shard's row count from parquet metadata without loading
    rows (`pyarrow.parquet.ParquetFile(p).metadata.num_rows`). Store shards +
    counts; cache `_total_rows`.
  - `__len__` / `total_length` → `_total_rows`.
  - `set_epoch(epoch)`: store epoch (the iterator uses it to seed shuffles).
    Stable across runs/ranks for a given `(seed, epoch)`.
  - `__iter__`:
    1. compute `epoch_seed = _epoch_seed(seed, epoch)`.
    2. shard order: if `shuffle_shards`, permute shard indices with a generator
       seeded by `epoch_seed`; else natural order.
    3. for each shard `s` (in order): read columns
       `["sequence_id","sequence"] (+ ["masking_weights"] if
       load_masking_weights)` via pyarrow; if `shuffle_rows`, permute row indices
       with a generator seeded by `_shard_row_seed(epoch_seed, s)`.
    4. **striping:** yield only rows whose running global row index satisfies
       `idx % stride == joint_index` (stride/joint from the context helper), so
       each `(rank, worker)` gets a disjoint, gap-free subset.
    5. yield `dict(sequence_id=…, sequence=…)`, plus
       `masking_weights=<list[float] | None>` when `load_masking_weights`.
  - When `load_masking_weights=True` but a shard lacks the column → yield `None`
    for that field (do not error here; collator handles fallback + one-time warn).
- [ ] **3.2 `InterleavedDataset(IterableDataset)`**:
  ```python
  InterleavedDataset(datasets, fractions, *, num_samples=None, seed=0)
  ```
  - Normalize `fractions` to sum 1.0 (validate `>=0`, sum `>0`).
  - `set_epoch` propagates to all sub-datasets.
  - `__iter__`: seed a generator per `(epoch, rank, worker)`. Maintain one live
    iterator per source. For `num_samples` steps (default = Σ source lengths,
    striped per worker): pick a source via `multinomial(fractions)`; pull its
    next item; on `StopIteration`, re-`iter()` that source and continue. Yield
    items unchanged (pass through the dict incl. any `masking_weights`).
- [ ] **3.3 Tests** (`tests/data/sequence/test_dataset.py`), using a tiny real
  parquet fixture (write 2 shards in a fixture; see Phase 10 fixture helper):
  - same `(seed, epoch)` → identical row order; different epochs → different.
  - **striping coverage:** simulate `(world_size, num_workers)` ∈
    {(1,1),(1,2),(2,2)}; union of yielded `sequence_id`s == full set, no dups.
  - interleaving sampling ratio over many draws ≈ `fractions`; exhausted sources
    refill.
  - `load_masking_weights`: column surfaced when present; `None` when absent.

---

## Phase 4 — Collation (`data/sequence/collate.py`)

### 4.1 Pad/tokenize primitive
- [ ] `tokenize_and_pad(batch, tokenizer, max_length) -> dict[str, Tensor]`:
  - Accept `list[dict]` (uses `"sequence"`) or `list[str]`.
  - Truncate each raw sequence to `max_length - 2` chars (room for
    `<cls>`/`<eos>`).
  - Tokenize with the canonical tokenizer; pad to the batch-max length with
    `pad_token_id`. Return `{"input_ids": (B,T) long, "attention_mask": (B,T)
    long}`. **No masking, no labels.** (Used by variant/structure/downstream.)
  - Provide an optional path to also return aligned `masking_weights` (B,T) via
    `align_per_residue` (Phase 2.3) when weights are supplied — used internally by
    the MLM collator; keep the public primitive's default output to the two keys.

### 4.2 MLM-mask layer
- [ ] `MLMCollator`:
  ```python
  MLMCollator(
      tokenizer, max_length=1024,
      mask_prob=0.15, mask_token_prob=0.8, random_token_prob=0.1, *,
      weighted_masking=False, deterministic=False, seed=0,
  )
  __call__(self, batch: list[dict]) -> dict[str, Tensor]
  ```
  Steps inside `__call__` (track a `_batch_idx` counter, increment each call):
  1. Build a `torch.Generator`. If `deterministic`, `manual_seed(seed +
     _batch_idx)` (do **not** disturb global RNG — pass this generator to all
     sampling ops below). Else use a generator seeded from ambient entropy (or
     the default RNG) so masks are fresh each draw (RoBERTa dynamic).
  2. `tokenize_and_pad(...)` → `input_ids`, `attention_mask`. Initialize
     `labels = full_like(input_ids, -100)`.
  3. **Eligibility** mask `(B,T)`: `attention_mask==1` AND id ∉ `non_maskable_ids`.
  4. **Weights** `(B,T)`: if `weighted_masking`, align each row's
     `masking_weights` (from the batch dicts) with `align_per_residue`
     (`fill=0.0` at specials/pad; missing/`None` row → all `1.0`); if the column
     is entirely absent across the batch, **warn once** and fall back to uniform.
     If `weighted_masking=False`, weights are all `1.0` (ignore any column).
     Set weight `0.0` at non-eligible positions.
  5. **Selection — Gumbel-top-k per row** (fixed count, sampling without
     replacement):
     ```
     n_eligible_b = eligibility[b].sum()
     k_b          = round(mask_prob * n_eligible_b)
     key          = log(weight) + gumbel_noise       # gumbel = -log(-log(U)), U~Uniform(0,1) via generator
     key[~eligible or weight==0] = -inf
     masked_idx   = topk(key[b], k_b).indices         # the k_b positions to mask
     ```
     Clamp `k_b` to the number of positive-weight eligible positions. Uniform
     masking is the all-weights-equal special case (same code path).
  6. **Targets:** `labels[b, masked_idx] = input_ids[b, masked_idx]`.
  7. **Replacement (BERT 80/10/10)** over masked positions, drawn with the same
     generator: `mask_token_prob` → `mask_token_id` (32); `random_token_prob` →
     a uniform draw from `canonical_amino_acid_ids`; remainder → keep original.
  8. Return `{"input_ids","attention_mask","labels"}` only — `masking_weights`
     is **not** emitted.
- [ ] **4.3 Tests** (`tests/data/sequence/test_collate.py`):
  - **Batch contract:** keys/shapes/dtypes per above; `T <= max_seq_len`;
    `attention_mask` matches padding.
  - **Fixed-k:** exactly `k=round(mask_prob*n_eligible)` positions per row, no
    dups; specials/pad never masked; `labels==-100` exactly off-masked positions.
  - **80/10/10** proportions within tolerance over many batches; random
    replacements only from `canonical_amino_acid_ids`.
  - **Dynamic (RoBERTa):** with `deterministic=False`, the same sequence in two
    calls gets different masked-position sets (across many trials).
  - **Determinism:** with `deterministic=True`, identical masks for the same
    `_batch_idx`; global RNG unaffected (snapshot `torch.random.get_rng_state()`
    before/after).
  - **Weighted:** count stays `k`; empirical inclusion frequency monotone in
    `w_i` and ≈ ∝ `w_i`; `w_i=0` never masked; scaling all weights leaves the
    distribution unchanged; positive-weight `< k` → all positive ones masked;
    `weighted_masking=False` ignores a present column; entirely-absent column
    warns + uniform; length mismatch raises (from `align_per_residue`).

---

## Phase 5 — Sequence builders (`data/sequence/loaders.py`)

- [ ] **5.1 `build_train_dataloader(cfg: OplmConfig) -> DataLoader`:**
  1. `entries = parse_train_configs(cfg.data.train)`.
  2. one `ShardedProteinDataset(entry.path, shuffle_shards=cfg.data.shuffle_shards,
     shuffle_rows=cfg.data.shuffle_rows, seed=cfg.train.seed,
     load_masking_weights=cfg.data.weighted_masking)` per entry.
  3. if >1 entry → `InterleavedDataset(datasets, [e.fraction…],
     seed=cfg.train.seed)`.
  4. `MLMCollator(get_tokenizer(), max_length=cfg.model.max_seq_len,
     mask_prob=cfg.data.mask_prob, mask_token_prob=cfg.data.mask_token_prob,
     random_token_prob=cfg.data.random_token_prob,
     weighted_masking=cfg.data.weighted_masking, deterministic=False,
     seed=cfg.train.seed)`.
  5. `DataLoader(dataset, batch_size=cfg.train.batch_size, collate_fn=collator,
     num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
     prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers>0 else
     None)`.
- [ ] **5.2 `build_sequence_eval_dataloader(path: str, cfg: OplmConfig) ->
  DataLoader`:** same machinery, **eval policy** — `ShardedProteinDataset(path,
  shuffle_shards=False, shuffle_rows=False, seed=<fixed eval seed, e.g. 42>,
  load_masking_weights=cfg.data.weighted_masking)`; `MLMCollator(...,
  deterministic=True, seed=<fixed eval seed>)`; same batch/worker settings.
- [ ] **5.3 Tests** (`tests/data/sequence/test_loaders.py`): a real fixture
  config yields the documented batch contract; eval builder is deterministic
  across two passes (identical batches); train builder differs across epochs;
  `num_workers=0` path returns `prefetch_factor=None`.

---

## Phase 6 — Structure modality (`data/structure/loader.py`)

Eval-only modality. **Lazy-import biopython** inside functions; raise a clear
`ImportError` pointing at `pip install oplm[train]` when missing.

- [ ] **6.1 `StructureData` dataclass:** `name: str`, `sequence: str`,
  `coords: Tensor  # (L,3,3) backbone N,CA,C; NaN for missing`, `chain_id: str|None`.
- [ ] **6.2 Modified-residue map** (three-letter → one-letter), at least:
  `MSE→M, SEC→C, CSE→C, SEP→S, TPO→T, PTR→Y`. `_residue_to_one_letter` uses the
  map first, then `Bio.Data.IUPACData.protein_letters_3to1`, else `"X"`.
- [ ] **6.3 `_parse_single_structure(path) -> StructureData | None`:** pick parser
  by suffix (`.cif/.mmcif` → `MMCIFParser`, else `PDBParser`, both `QUIET=True`);
  use **first model, first chain**; skip heteroatoms unless in the modified map;
  per residue, extract `N,CA,C` coords (NaN row if an atom is missing); return
  `None` (warn) on parse failure or empty chain.
- [ ] **6.4 `load_structures(directory, max_structures=None) ->
  list[StructureData]`:** glob `*.pdb,*.cif,*.ent,*.mmcif`, sort by filename for
  determinism, parse each, skip `None`, optional cap.
- [ ] **6.5 Tests** (`tests/data/structure/test_loader.py`, `@pytest.mark.slow`):
  parse one real PDB fixture → `coords` shape `(L,3,3)`, `len(sequence)==L`,
  modified-residue mapping works, missing atoms → NaN. Add a non-biopython
  import-error test via monkeypatch if feasible.
  - **Fixture:** place a small real structure at
    `tests/data/fixtures/structures/<id>.pdb` (e.g. a short chain). If absent,
    prompt the user to supply one.

---

## Phase 7 — Variant modality (`data/variant/loader.py`)

Zero-shot variant-effect data. Tokenize+pad at scoring time (no MLM masking).

- [ ] **7.1 `VariantAssay` dataclass:** `name: str`, `wildtype: str`,
  `mutations: list[str]`, `labels: list[float]`.
- [ ] **7.2 CSV parsing** (`load_variant_assays(directory) -> list[VariantAssay]`):
  one assay per CSV. Required columns `mutant` (e.g. `"A42T"`, `:`-joined for
  multi-mutants) and `DMS_score` (float). The wild-type sequence is supplied per
  assay (sidecar/metadata/`EvalDatasetEntry.extra["wildtype"]` or a `wildtype`
  column) — document the accepted source(s).
- [ ] **7.3 Mutation parsing/validation:** `parse_mutation("A42T") -> (wt="A",
  pos=42, mut="T")` (1-based). Validate `wildtype[pos-1] == wt`; raise on
  mismatch. (Scoring itself — masked-marginal log-prob ratios — belongs to the
  eval harness; this module only loads + validates.)
- [ ] **7.4 Tests** (`tests/data/variant/test_loader.py`): tiny real ProteinGym
  CSV fixture at `tests/data/fixtures/variant/<assay>.csv`; parse → correct
  counts; mutation parse + WT-consistency validation; multi-mutant split on `:`;
  mismatch raises.

---

## Phase 8 — Downstream modality (`data/downstream/loader.py`)

Labeled-sequence benchmarks (TAPE/ProteinGLUE). Tokenize+pad (no masking);
embeddings + supervised head are the eval harness's job.

- [ ] **8.1 Loaders** for parquet/CSV with a `sequence` column plus task labels:
  per-residue label lists, or sequence-level scalar/categorical. Define a small
  dataclass (e.g. `DownstreamExample(sequence, label)`) and a
  `load_downstream_dataset(path, task_type)` returning a list (or iterable).
- [ ] **8.2 Label contract** (document + implement collation of labels):
  - per-residue → `(B, T)` long, aligned to non-special positions, pad with `-100`
    (reuse `align_per_residue` with `fill_special=-100`, `fill_pad=-100`).
  - seq-level regression → `(B,)` float; seq-level classification → `(B,)` long.
- [ ] **8.3 Tests** (`tests/data/downstream/test_loader.py`): tiny fixtures for
  one per-residue and one sequence-level task; assert label tensor shapes/dtypes
  and `-100` alignment for per-residue padding.

---

## Phase 9 — Public API & trainer integration

- [ ] **9.1 `data/__init__.py`** re-exports the public surface (no logic):
  `get_tokenizer`, `build_train_dataloader`, `build_sequence_eval_dataloader`,
  `ShardedProteinDataset`, `InterleavedDataset`, `MLMCollator`,
  `tokenize_and_pad`, `load_structures`, `StructureData`,
  `load_variant_assays`, `VariantAssay`, `parse_train_configs`,
  `parse_eval_configs`. Must not import `oplm.eval`.
- [ ] **9.2 Trainer wiring:** confirm
  [`src/oplm/training/trainer.py`](src/oplm/training/trainer.py) builds the loader
  via `from oplm.data.loader import build_train_dataloader` **→ update the import
  to `from oplm.data import build_train_dataloader`** (the old module path is
  gone). Verify the epoch handling calls `set_epoch(epoch)` on the dataset on
  `StopIteration`, and that batches pass `input_ids/attention_mask/labels`
  straight into `model(...)`. Make no behavioral changes beyond the import path.
- [ ] **9.3 Eval-harness boundary (note only):** `oplm.eval` will not import until
  its own rewrite consumes `oplm.data.*` (structure/variant/downstream loaders +
  `build_sequence_eval_dataloader`). That is tracked separately; do not stub it
  here. Ensure nothing in `oplm.data` imports `oplm.eval`.

---

## Phase 10 — End-to-end, fixtures & QA

- [ ] **10.1 Fixture helpers** (`tests/data/conftest.py`): a session-scoped
  fixture that writes a tiny **real** sequence parquet (a handful of real protein
  sequences, with a second shard, optionally a `masking_weights` column) to a
  `tmp_path_factory` dir; return the path. Reuse across dataset/collate/loader
  tests. Document where real structure/variant fixtures must be dropped
  (`tests/data/fixtures/...`); prompt the user if missing.
- [ ] **10.2 Pilot end-to-end** (`tests/data/test_e2e.py`, `@pytest.mark.slow`):
  - Build a tiny `OplmConfig`: keep `model.vocab_size` at its default `33`, small
    `hidden_dim`/`num_layers`/`num_heads`, `max_seq_len≈64`; `train.batch_size`
    small; `data.train=<fixture parquet>`, `data.mask_prob=0.15`.
  - Instantiate `OplmForMaskedLM(model-config)`; run ~3 train steps over
    `build_train_dataloader(cfg)` (forward+backward+step); assert finite loss and
    no shape errors.
  - Run `build_sequence_eval_dataloader(<fixture>, cfg)` once; forward with
    `labels`; assert a finite `loss` and that two eval passes give **identical**
    batches (determinism).
  - Repeat the train-step check with `data.weighted_masking=True` and a fixture
    that has a `masking_weights` column; assert it runs and masks respect weights.
- [ ] **10.3 Lint/type/test gates:** `ruff format src/ tests/`,
  `ruff check src/ tests/`, `mypy src/oplm/data`, and `pytest tests/data`
  (`-m "not slow"` for fast iteration; full run before done).

---

## Done criteria

- [ ] Every phase box checked; `pytest tests/data` green (incl. slow).
- [ ] `import oplm.data` works with no `oplm.eval` dependency.
- [ ] Tokenizer parity test passes (ids match `OplmTokenizerFast`, `<mask>`=32,
  canonical AAs = 4–23).
- [ ] Masking is fixed-`k` Gumbel-top-k; uniform == equal-weights special case;
  dynamic across epochs; deterministic under the eval flag.
- [ ] Trainer imports `build_train_dataloader` from `oplm.data` and trains a pilot
  model end-to-end.
- [ ] `docs/DATA_TOOLING.md` and this plan agree; any intentional deviation is
  noted in the doc.

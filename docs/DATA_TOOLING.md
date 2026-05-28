# OPLM Data Tooling

> Founding reference for the OPLM data pipeline — how raw protein data on disk
> becomes model-ready tensors, for both pretraining and evaluation. This document
> specifies the **target** design: the module layout, the tokenizer contract, the
> per-modality loaders, the configuration schema, and the train/eval boundary, in
> enough detail to implement directly. The model itself is out of scope and lives
> in [`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md); the eval *harness* (tasks,
> metrics, scheduling) is out of scope and will live in a future `docs/EVAL.md`.

---

## 1. Scope and design philosophy

### 1.1 What this document covers

- The PyTorch/data modules under `src/oplm/data/` — datasets, collation, the
  tokenizer access layer, the per-modality loaders, and the dataloader builders.
- The **on-disk data formats** OPLM consumes (parquet sequence shards, PDB/CIF
  structures, variant CSVs, downstream-task label files).
- The **batch contracts** — exact dict keys, tensor shapes, and dtypes — produced
  for the model.
- How training and evaluation **share** data machinery, and the precise points
  where they diverge.
- The `data:` block of `OplmConfig` (the contract between YAML and the loaders).

### 1.2 What this document does not cover

- The model, its forward signature, and the tokenizer's internal construction. →
  [`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md) (the tokenizer is *defined*
  there; this doc only specifies how the data layer *uses* it).
- The eval harness: task registry, metric math (contact P@L, Spearman, APC,
  categorical Jacobian), per-task scheduling. → future `docs/EVAL.md`.
- The trainer loop, optimizer, schedules, checkpointing. → future `docs/TRAINER.md`.
- The CLI. → future `docs/CLI.md`.

### 1.3 Design principles

1. **Organize by modality, not by phase.** The primary axis is the *kind of
   data* (sequence, structure, variant, downstream), not whether it is used for
   training or evaluation. Code lives under `src/oplm/data/<modality>/`.

2. **Share by default; fork only across modalities.** Pretraining and sequence
   evaluation consume the *same* parquet sequences with the *same* tokenizer and
   the *same* collation core. The only differences (shuffle, seed, deterministic
   masking) are **policy parameters**, not separate code paths.

3. **Policy, not subclass.** Eval-specific behavior (no shuffle, fixed seed,
   reproducible masks) is expressed as constructor arguments on shared
   components. There is no `DeterministicMLMCollator` subclass, no parallel
   "eval dataset" class — just different arguments to the same builder.

4. **One tokenizer, one vocabulary.** A single canonical tokenizer
   (`OplmTokenizerFast`) is the source of truth for token IDs. Masking constants
   (special-token IDs, the canonical amino-acid range) are *derived* from it, not
   hardcoded. See §3.

5. **Layered.** Four layers, each consuming only the one below it: tokenizer →
   datasets → collation → builders. See §2.

### 1.4 Status: target design vs. today's code

This is a **forward-looking design**. The current contents of `src/oplm/data/`
and `src/oplm/eval/data/` are an earlier implementation, retained as reference
only and slated for a substantial rewrite to match this document. Where the
current code diverges from the target, §12 records the mapping. Two divergences
are load-bearing and motivate the rewrite:

- The current data path uses a second tokenizer (`ProteinTokenizer`) whose
  vocabulary is **incompatible** with the model's. → resolved in §3.
- Sequence data tooling is split across `data/` and `eval/data/`. → consolidated
  in §2.

---

## 2. Layered architecture overview

The pipeline is four layers. Each layer is consumed by the layer above it and by
the eval harness; nothing reaches around a layer.

```
┌──────────────────────────────────────────────────────────────────────┐
│ Layer 4  BUILDERS  (policy: train vs eval)                             │
│   build_train_dataloader        build_sequence_eval_dataloader         │
│   load_structures               load_variant_dataset   ...             │
├──────────────────────────────────────────────────────────────────────┤
│ Layer 3  COLLATION                                                     │
│   pad/tokenize primitive  ──►  MLM-mask layer (stochastic | det.)      │
├──────────────────────────────────────────────────────────────────────┤
│ Layer 2  DATASETS  (per modality)                                      │
│   ShardedProteinDataset  InterleavedDataset   StructureData   Variant  │
├──────────────────────────────────────────────────────────────────────┤
│ Layer 1  TOKENIZER  (single source of truth)                          │
│   OplmTokenizerFast  +  derived id constants                          │
└──────────────────────────────────────────────────────────────────────┘
        ▲                                                ▲
        │ training loop (trainer.py)                     │ eval harness (eval/)
        └──── build_train_dataloader ────────────────────┴── builders + loaders
```

### 2.1 Target module tree

All data loading lives under `src/oplm/data/`, organized by modality. There is
**no** `src/oplm/eval/data/` directory; the eval harness imports `oplm.data.*`.

```
src/oplm/data/
  __init__.py         # re-exports the public builders + tokenizer accessor
  tokenizer.py        # canonical-tokenizer accessor + derived id constants
  config.py           # parse_train_configs / parse_eval_configs + entry dataclasses
  sequence/
    __init__.py
    dataset.py        # ShardedProteinDataset, InterleavedDataset
    collate.py        # pad/tokenize primitive + MLM-mask layer
    loaders.py        # build_train_dataloader, build_sequence_eval_dataloader
  structure/
    __init__.py
    loader.py         # PDB/CIF → StructureData (lazy biopython)
  variant/
    __init__.py
    loader.py         # ProteinGym / EVEREST CSV → (wildtype, mutations, labels)
  downstream/
    __init__.py
    loader.py         # labeled-sequence tasks (TAPE, ProteinGLUE)

src/oplm/eval/        # NO data/ subdir
  tasks/  metrics/  registry.py  evaluator.py    # all import oplm.data.*
```

> The `DataConfig` schema itself remains in `src/oplm/config.py` alongside
> `ModelConfig`/`TrainConfig` (the root `OplmConfig` is one file). `data/config.py`
> holds only the *parsing* helpers that turn loose YAML into typed entries.

### 2.2 What the eval harness consumes (the shared contract)

| Eval concern                | Imports from `oplm.data`                                  |
| --------------------------- | -------------------------------------------------------- |
| Sequence MLM / pseudo-ppl   | `build_sequence_eval_dataloader` (eval policy, §9.2)     |
| Variant effect (zero-shot)  | `variant.loader` + tokenizer + pad primitive             |
| Structure / contacts        | `structure.loader` (`load_structures`) + tokenizer       |
| Downstream (TAPE/GLUE)      | `downstream.loader` + tokenizer + pad primitive          |

The eval harness owns *task logic, scoring, and metrics*; it does not own data
loading. Every model input it builds flows through the layers above.

---

## 3. Tokenizer (single source of truth)

### 3.1 `OplmTokenizerFast` is canonical

The data layer uses **one** tokenizer: `OplmTokenizerFast`, defined in
[`src/oplm/model/tokenization_oplm.py`](../src/oplm/model/tokenization_oplm.py)
and specified in [`MODEL_ARCHITECTURE.md` §3](MODEL_ARCHITECTURE.md). It is the
same tokenizer returned by `from_pretrained()` alongside the model, so token IDs
produced for data are by construction the IDs the model's embedding table expects.

`data/tokenizer.py` is a thin accessor — it constructs/returns an
`OplmTokenizerFast` and exposes the derived id constants of §3.3. It defines **no
vocabulary of its own**. The legacy `ProteinTokenizer` is **removed** (see §3.5
and §12).

The backend is a Rust-backed `PreTrainedTokenizerFast` (WordLevel, per-character
pre-tokenization, `<cls> … <eos>` templating). Batch encoding is fast enough to
sit in the training hot loop; there is no second, "fast" tokenizer to keep in
sync.

### 3.2 Vocabulary (33 tokens, ESM-C order)

IDs are bit-identical to ESM-C. `index == token id`.

| ID | Token   | Class            |        | ID | Token   | Class            |
| -- | ------- | ---------------- | ------ | -- | ------- | ---------------- |
| 0  | `<cls>` | special (BOS)    |        | 17 | `N`     | standard AA      |
| 1  | `<pad>` | special          |        | 18 | `F`     | standard AA      |
| 2  | `<eos>` | special          |        | 19 | `Y`     | standard AA      |
| 3  | `<unk>` | special          |        | 20 | `M`     | standard AA      |
| 4  | `L`     | standard AA      |        | 21 | `H`     | standard AA      |
| 5  | `A`     | standard AA      |        | 22 | `W`     | standard AA      |
| 6  | `G`     | standard AA      |        | 23 | `C`     | standard AA      |
| 7  | `V`     | standard AA      |        | 24 | `X`     | ambiguous AA     |
| 8  | `S`     | standard AA      |        | 25 | `B`     | ambiguous (D/N)  |
| 9  | `E`     | standard AA      |        | 26 | `U`     | selenocysteine   |
| 10 | `R`     | standard AA      |        | 27 | `Z`     | ambiguous (E/Q)  |
| 11 | `T`     | standard AA      |        | 28 | `O`     | pyrrolysine      |
| 12 | `I`     | standard AA      |        | 29 | `.`     | gap marker       |
| 13 | `D`     | standard AA      |        | 30 | `-`     | alignment gap    |
| 14 | `P`     | standard AA      |        | 31 | `\|`    | chain break      |
| 15 | `K`     | standard AA      |        | 32 | `<mask>`| MLM mask         |
| 16 | `Q`     | standard AA      |        |    |         |                  |

The **20 standard amino acids** occupy the contiguous block **IDs 4–23**
(`L,A,G,V,S,E,R,T,I,D,P,K,Q,N,F,Y,M,H,W,C`). Verification anchor (from the
tokenizer docstring): `tok("MEEPQ").input_ids == [0, 20, 9, 9, 14, 16, 2]`.

### 3.3 Derived id constants (not literals)

The collator and any scoring code derive their constants from the tokenizer
instance so they can never drift from the vocabulary:

| Constant                 | Definition                                          | Value today |
| ------------------------ | --------------------------------------------------- | ----------- |
| `special_ids`            | `tokenizer.all_special_ids` (cls, pad, eos, unk, mask) | {0,1,2,3,32} |
| `non_maskable_ids`       | special IDs that must never become a prediction target | {0,1,2,3,32} |
| `mask_token_id`          | `tokenizer.mask_token_id`                            | 32          |
| `pad_token_id`           | `tokenizer.pad_token_id`                             | 1           |
| `canonical_aa_ids`       | token IDs of the 20 standard AAs (contiguous 4–23)  | range(4,24) |

> `canonical_aa_ids` is the sampling pool for random-token replacement (§4.5).
> The eval harness already needs this set for marginal scoring and the
> categorical Jacobian; expose **one** helper (e.g.
> `canonical_amino_acid_ids(tokenizer)`) and reuse it in both the collator and
> the eval metrics rather than duplicating the literal range.

### 3.4 Hot-loop note

The training collator calls the fast tokenizer's batch encode (Rust) once per
batch. There is no tensor-native second vocabulary and no per-character Python
loop in the hot path. This is the single most important reason a second tokenizer
is unnecessary.

### 3.5 Migration note (why the old vocab is dropped)

The legacy `data/tokenizer.py::ProteinTokenizer` defined a **32-token**
vocabulary in which every amino acid was offset by one (`L`=5 vs the canonical
`L`=4) and `<mask>`=4 (vs canonical 32). Because training used that tokenizer
while the model embeds the canonical 33-token vocabulary, the old data path
produced token IDs that **did not align with the model's embedding rows**. This
is a correctness bug, not a stylistic choice; the rewrite eliminates the second
vocabulary entirely. See §12 for the symbol mapping.

---

## 4. Sequence modality (`data/sequence/`)

The pretraining path, and the substrate for sequence-based evaluation.

### 4.1 On-disk format

- A dataset is a single parquet file (`.parquet` / `.parq` / `.pq`) **or** a
  directory of parquet shards.
- Required columns: **`sequence_id`** (str) and **`sequence`** (str, raw
  one-letter amino acids, no special tokens).
- Only these two columns are read (`pq.read_table(..., columns=[...])`); other
  columns are ignored.

### 4.2 `ShardedProteinDataset`

An `IterableDataset[dict[str, str]]` yielding `{"sequence_id": ..., "sequence": ...}`
one row at a time (raw strings — tokenization happens in collation).

```python
ShardedProteinDataset(
    path: str | Path,
    *,
    shuffle_shards: bool = True,
    shuffle_rows: bool = True,
    seed: int = 0,
)
```

Behavior:

- **Shard discovery** at construction: enumerates shard files and reads per-shard
  row counts from parquet metadata (no row data loaded). `total_length` is the
  sum.
- **Per-epoch deterministic shuffle.** `set_epoch(epoch)` selects the epoch's
  RNG. The epoch seed is mixed from the base `seed` and the epoch index using a
  golden-ratio constant and a large prime so consecutive epochs are well
  decorrelated. Same `(seed, epoch)` ⇒ identical order across runs and ranks.
- **Two shuffle granularities**, independently toggleable: shard *order*
  (`shuffle_shards`) and *row order within a shard* (`shuffle_rows`, seeded
  per-shard so different shards permute differently).
- **Worker / rank striping.** Within a shard's (optionally shuffled) row
  sequence, rows are striped so that each `(rank, worker)` sees a disjoint
  subset with no duplication and no gaps.

> **Design point — distributed sharding.** The legacy implementation strided
> only across DataLoader workers and delegated process-level (DDP rank) sharding
> to the launcher. The target makes rank-awareness **explicit**: striping is over
> the joint `(world_rank, worker_id)` index so correctness does not depend on
> launcher behavior. This must be covered by a test (§11).

### 4.3 `InterleavedDataset`

Mixes multiple sequence datasets by sampling fraction.

```python
InterleavedDataset(
    datasets: Sequence[IterableDataset[dict[str, str]]],
    fractions: list[float],
    *,
    num_samples: int | None = None,
    seed: int = 0,
)
```

Behavior:

- **Fraction normalization** to sum 1.0 (validated `>= 0`, sum `> 0`).
- **Probabilistic step selection.** Each step picks a source dataset by its
  fraction, then pulls the next item from that source's iterator.
- **Exhaustion handling.** When a source iterator is exhausted, it is
  re-initialized and sampling continues — sources of unequal size keep mixing at
  the requested ratio for the whole epoch.
- `set_epoch` propagates to all sub-datasets; sampling RNG is seeded per
  `(epoch, worker/rank)`.
- `num_samples` defaults to the sum of source lengths when available.

### 4.4 Collation: a pad primitive plus a mask layer

Collation is **two composable pieces** so that masking and non-masking consumers
share the same tokenize/pad core.

- **Pad/tokenize primitive** — `tokenize_and_pad(batch)`:
  takes `list[dict[str, str]]` (or raw `list[str]`), truncates each raw sequence
  to `max_length - 2` (leaving room for `<cls>`/`<eos>`), tokenizes with the
  canonical tokenizer, pads to the batch's longest member with `pad_token_id`,
  and returns `{"input_ids", "attention_mask"}`. **No masking, no labels.** This
  is what variant, structure, and downstream consumers use.

- **MLM-mask layer** — `MLMCollator`, which calls the primitive and then applies
  masking, adding `labels`:

```python
MLMCollator(
    tokenizer: OplmTokenizerFast,
    max_length: int = 1024,
    mask_prob: float = 0.15,
    mask_token_prob: float = 0.8,
    random_token_prob: float = 0.1,
    *,
    deterministic: bool = False,   # eval policy; see §4.6
    seed: int = 0,
)
```

### 4.5 MLM masking scheme

BERT-style masking, fully parameterized, with constants derived from the
tokenizer (§3.3):

1. **Eligibility.** A position is maskable iff it is a real token
   (`attention_mask == 1`) **and** its id is not in `non_maskable_ids`
   (cls/pad/eos/unk/mask).
2. **Selection.** Each eligible position is selected with probability
   `mask_prob` (default 0.15).
3. **Targets.** For selected positions, `labels` holds the original id; all
   other positions hold the ignore index **`-100`**.
4. **Replacement** (the 80/10/10 split, now configurable):
   - `mask_token_prob` (0.8) → replace with `mask_token_id` (32).
   - `random_token_prob` (0.1) → replace with a uniformly sampled id from
     `canonical_aa_ids` (the 20 standard AAs, 4–23). Ambiguous AAs and
     gap/structure tokens are **not** sampled.
   - remainder (`1 - 0.8 - 0.1 = 0.1`) → keep the original id.

> **New config knobs.** `mask_token_prob` and `random_token_prob` were hardcoded
> in the legacy collator. The target exposes them via `DataConfig` (§8.2) so the
> split is reproducible from config and ablatable.

### 4.6 Determinism for evaluation (policy, not a fork)

Reproducible eval masks are a **parameter**, not a subclass. When
`deterministic=True`, the collator derives a per-batch RNG state from
`seed + batch_index` before masking and restores the ambient RNG afterward, so
the *same batch always receives the same mask pattern* — making MLM eval metrics
comparable across training steps. Training uses `deterministic=False`; the
sequence-eval builder sets `deterministic=True` and disables shuffling. No
separate collator class exists.

### 4.7 Batch contract

Every sequence batch (training and sequence-eval) has identical structure:

| Key              | Shape   | Dtype        | Notes                                       |
| ---------------- | ------- | ------------ | ------------------------------------------- |
| `input_ids`      | `(B,T)` | `torch.long` | canonical IDs; `<cls> … <eos>`; padded      |
| `attention_mask` | `(B,T)` | `torch.long` | 1 = real token, 0 = padding                 |
| `labels`         | `(B,T)` | `torch.long` | original id at masked positions, else `-100`|

`B` = `train.batch_size`; `T` = batch-max length, capped at `model.max_seq_len`.
The pad primitive's output omits `labels` (it produces only the first two keys).

---

## 5. Structure modality (`data/structure/`)

A genuinely separate modality: 3-D structures, parsed to backbone coordinates.
Consumed by the contact-prediction eval task. Sequences reuse the canonical
tokenizer; **no masking** is applied.

### 5.1 Input

- A directory of `*.pdb`, `*.cif`, `*.ent`, `*.mmcif` files.
- Parsed with BioPython (`PDBParser`, `MMCIFParser`).

### 5.2 `StructureData`

```python
@dataclass
class StructureData:
    name: str               # PDB id / filename stem
    sequence: str           # one-letter amino acids
    coords: Tensor          # (L, 3, 3) backbone N, CA, C; NaN for missing atoms
    chain_id: str | None
```

### 5.3 Parsing policy

- **First model, first chain** (model 0 for X-ray; first conformer for NMR).
- **Modified residues** are mapped to their canonical parent (e.g. `MSE→M`,
  `SEP→S`, `PTR→Y`); other heteroatoms (water, ligands) are skipped.
- Missing backbone atoms become `NaN` rows so `(L, 3, 3)` stays aligned with
  `sequence`.
- Files that fail to parse are skipped with a warning; results are returned
  sorted by filename for determinism.

```python
load_structures(directory: str | Path, max_structures: int | None = None)
    -> list[StructureData]
```

### 5.4 How the eval task consumes it

The contact task tokenizes `StructureData.sequence` via the **pad primitive**
(§4.4) with `attention_mask` all ones (no padding for single sequences, no
masking), runs the model (e.g. requesting attention weights), and compares
attention-derived or Jacobian-derived contacts against the geometric contact map
computed from `coords`. All metric math lives in the eval harness, not here.

### 5.5 Optional dependency

BioPython is an **optional** dependency, imported lazily inside the parser so the
core training install need not carry it. The import error message points at the
extra (e.g. `pip install oplm[train]`). Structure parsing belongs in `data/`
(it produces model inputs), but it is gated so that `import oplm.data` never hard
-requires BioPython.

---

## 6. Variant modality (`data/variant/`)

Zero-shot variant-effect prediction (ProteinGym, EVEREST). Tokenize + pad, **no
MLM masking**; scoring uses position-specific (marginal) masking instead.

### 6.1 Input

- A directory of CSV files (one per assay/DMS experiment).
- Per-row fields: **`mutant`** (e.g. `"A42T"`, or `:`-joined multi-mutants),
  **`DMS_score`** (float). The **wild-type sequence** is supplied per assay
  (file metadata / sidecar / config `extra`).

### 6.2 Parsing

```python
@dataclass
class VariantAssay:
    name: str
    wildtype: str               # one-letter WT sequence
    mutations: list[str]        # raw mutant strings, one per row
    labels: list[float]         # DMS_score per row
```

The loader yields one `VariantAssay` per CSV; mutation strings are parsed
(position, from-AA, to-AA) at scoring time, validated against `wildtype`.

### 6.3 Model-input construction

Scoring is **marginal**, not batch-MLM:

- Encode the wild-type sequence once (pad primitive).
- For each mutated position, mask that single position (`<mask>` at the residue),
  run the model, and read the log-probabilities for the WT and mutant amino
  acids (masked-marginal). A WT-marginal variant reads logits from the
  unmasked WT pass.
- Variant score = `log P(mutant_aa) − log P(wt_aa)` summed over the mutation set.

The masking here is **deterministic and position-specific** — it is not the
stochastic MLM collator. It reuses the tokenizer and the canonical-AA helper
(§3.3) for indexing into the logits, nothing from `MLMCollator`.

### 6.4 Output contract

The loader's job ends at `VariantAssay`. The eval harness produces per-assay
predicted scores and computes Spearman / NDCG / AUROC against `labels`.

---

## 7. Downstream / embedding modality (`data/downstream/`)

Supervised downstream benchmarks (TAPE, ProteinGLUE). Tokenize + pad, **no MLM
masking**; the model is used as a frozen embedder feeding a lightweight head.

### 7.1 Input

- Per-residue label tasks: secondary structure (SS3/SS8), per-residue contacts.
- Sequence-level label tasks: fluorescence, stability (regression); fold,
  enzyme-class, GO (classification).
- Stored as parquet/CSV with a `sequence` column plus task-specific label
  column(s) (a per-residue label list, or a scalar/categorical per sequence).

### 7.2 Model-input construction

- Sequences go through the **pad primitive** (§4.4): `{input_ids, attention_mask}`,
  no masking, no `labels` key from the collator.
- The harness extracts representations — **pooled** (mean / CLS) for
  sequence-level tasks, **per-residue** hidden states for residue-level tasks —
  and trains a small supervised head. Pooling helpers live with the model
  ([`model/embedding.py`](../src/oplm/model/embedding.py)); label handling lives
  in the eval harness.

### 7.3 Label contract

| Task family    | Label tensor                  | Notes                              |
| -------------- | ----------------------------- | ---------------------------------- |
| per-residue    | `(B, T)` long / `(B, T, …)`   | aligned to non-special positions; pad with `-100` |
| seq-level reg. | `(B,)` float                  | one scalar per sequence            |
| seq-level cls. | `(B,)` long                   | class index per sequence           |

---

## 8. Configuration schema

### 8.1 `DataConfig` field table

From [`src/oplm/config.py`](../src/oplm/config.py) (`DataConfig`) — exact names
and defaults:

| Field             | Type                     | Default | Meaning                                                       |
| ----------------- | ------------------------ | ------- | ------------------------------------------------------------- |
| `train`           | `str \| dict \| None`    | `None`  | training dataset(s); parsed by `parse_train_configs` (§8.3)   |
| `eval`            | `dict \| None`           | `None`  | eval dataset(s) by name → `{path, type, …}`; `parse_eval_configs` |
| `mask_prob`       | `float`                  | `0.15`  | MLM selection probability (§4.5)                              |
| `num_workers`     | `int`                    | `4`     | DataLoader workers                                            |
| `pin_memory`      | `bool`                   | `True`  | DataLoader pinned memory                                      |
| `prefetch_factor` | `int`                    | `4`     | DataLoader prefetch (only when `num_workers > 0`)            |
| `shuffle_shards`  | `bool`                   | `True`  | shard-order shuffle (§4.2)                                    |
| `shuffle_rows`    | `bool`                   | `True`  | within-shard row shuffle (§4.2)                              |

Fields consumed from sibling configs:

| Field               | Source        | Default | Used for                            |
| ------------------- | ------------- | ------- | ----------------------------------- |
| `model.max_seq_len` | `ModelConfig` | `512`   | collator `max_length` (§4.4)        |
| `train.batch_size`  | `TrainConfig` | `32`    | DataLoader `batch_size`             |
| `train.seed`        | `TrainConfig` | `42`    | dataset/collator RNG seed           |
| `train.eval_every`  | `TrainConfig` | `10000` | default eval cadence (per-task override via `EvalDatasetEntry.eval_every`) |

### 8.2 New knobs to add

To un-hardcode the masking split (§4.5), add to `DataConfig`:

| Field               | Type    | Default | Meaning                                    |
| ------------------- | ------- | ------- | ------------------------------------------ |
| `mask_token_prob`   | `float` | `0.8`   | fraction of masked positions → `<mask>`    |
| `random_token_prob` | `float` | `0.1`   | fraction of masked positions → random AA   |

Validation: both in `[0, 1]` and `mask_token_prob + random_token_prob <= 1`.

### 8.3 Dataset entries and parsing

`data/config.py` provides the parsing helpers (entry dataclasses defined in
`config.py`):

```python
@dataclass
class TrainDatasetEntry:
    name: str
    path: str
    fraction: float

@dataclass
class EvalDatasetEntry:
    name: str
    path: str
    type: str                       # "sequence" | "structure" | "proteingym" | ...
    eval_every: int | None = None   # per-dataset cadence override
    metrics: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # task-specific config
```

- `parse_train_configs(raw)` accepts a single path string or a `{name: {path,
  fraction}}` map; fractions are normalized to 1.0 and omitted fractions split
  the remaining mass equally.
- `parse_eval_configs(raw, default_eval_every)` parses the `{name: {path, type,
  …}}` map, requiring `path` and `type`, and folds unknown keys into `extra`.

### 8.4 Example YAML

```yaml
data:
  # Single training dataset (100% sampling)
  train: /data/uniref50/

  # ...or interleaved datasets with fractional sampling
  # train:
  #   uniref50: { path: /data/uniref50/, fraction: 0.6 }
  #   bfd:      { path: /data/bfd/,      fraction: 0.4 }

  mask_prob: 0.15
  mask_token_prob: 0.8
  random_token_prob: 0.1
  num_workers: 4
  pin_memory: true
  prefetch_factor: 4
  shuffle_shards: true
  shuffle_rows: true

  eval:
    heldout_seqs:
      path: /data/heldout/
      type: sequence
    casp_contacts:
      path: /data/casp_structures/
      type: structure
      extra: { contact_threshold: 8.0, min_seq_sep: 6 }
    proteingym:
      path: /data/proteingym/
      type: proteingym
      eval_every: 50000
```

See [`src/oplm/configs/data/base.yaml`](../src/oplm/configs/data/base.yaml) for
the canonical defaults.

---

## 9. Builders and the train/eval boundary

### 9.1 `build_train_dataloader`

`data/sequence/loaders.py`:

```python
def build_train_dataloader(cfg: OplmConfig) -> DataLoader[dict[str, Tensor]]:
    ...
```

1. `parse_train_configs(cfg.data.train)` → `list[TrainDatasetEntry]`.
2. One `ShardedProteinDataset` per entry, with
   `shuffle_shards/​shuffle_rows` from `cfg.data` and `seed=cfg.train.seed`.
3. If >1 entry, wrap in `InterleavedDataset` with the entries' fractions.
4. `MLMCollator(tokenizer, max_length=cfg.model.max_seq_len,
   mask_prob=cfg.data.mask_prob, mask_token_prob=…, random_token_prob=…,
   deterministic=False)`.
5. Wrap in `DataLoader(batch_size=cfg.train.batch_size,
   collate_fn=collator, num_workers=…, pin_memory=…, prefetch_factor=…)`.

### 9.2 `build_sequence_eval_dataloader`

Same machinery, **eval policy**:

```python
def build_sequence_eval_dataloader(path: str, cfg: OplmConfig) -> DataLoader[dict[str, Tensor]]:
    ...
```

- `ShardedProteinDataset(path, shuffle_shards=False, shuffle_rows=False, seed=<eval seed>)`.
- `MLMCollator(..., deterministic=True, seed=<eval seed>)`.
- Same batch size / worker settings from `cfg`.

The train/eval difference is entirely the highlighted arguments:

| Parameter        | Training                  | Sequence eval            |
| ---------------- | ------------------------- | ------------------------ |
| dataset class    | `ShardedProteinDataset`   | `ShardedProteinDataset`  |
| `shuffle_shards` | `cfg.data.shuffle_shards` | `False`                  |
| `shuffle_rows`   | `cfg.data.shuffle_rows`   | `False`                  |
| collator         | `MLMCollator`             | `MLMCollator`            |
| `deterministic`  | `False`                   | `True`                   |
| mask probability | `cfg.data.mask_prob`      | fixed eval value         |
| batch contract   | §4.7                      | §4.7 (identical)         |

### 9.3 Modality → builder/task map

| Modality   | Loader / builder (`oplm.data`)        | Eval task type        |
| ---------- | ------------------------------------- | --------------------- |
| sequence   | `build_train_dataloader`, `build_sequence_eval_dataloader` | `sequence` |
| structure  | `structure.loader.load_structures`    | `structure`           |
| variant    | `variant.loader` (`VariantAssay`)     | `proteingym`, `everest` |
| downstream | `downstream.loader`                   | `tape`, `proteinglue` |

### 9.4 Shared vs. not shared

| Component                         | Train | Seq eval | Variant | Structure | Downstream |
| --------------------------------- | :---: | :------: | :-----: | :-------: | :--------: |
| `OplmTokenizerFast` + id consts   | ✅    | ✅       | ✅      | ✅        | ✅         |
| pad/tokenize primitive            | ✅    | ✅       | ✅      | ✅        | ✅         |
| `ShardedProteinDataset`/Interleaved | ✅  | ✅       | –       | –         | – ¹        |
| `MLMCollator` (random masking)    | ✅    | ✅²      | –       | –         | –          |
| position-marginal masking         | –     | –        | ✅      | –         | –          |
| PDB/CIF parser                    | –     | –        | –       | ✅        | –          |
| supervised-head label handling    | –     | –        | –       | –         | ✅          |

¹ downstream datasets are labeled files, not raw parquet shards.
² eval uses the same class with `deterministic=True`.

---

## 10. Integration with training & eval loops

### 10.1 Trainer

[`src/oplm/training/trainer.py`](../src/oplm/training/trainer.py):

- Builds the loader via `build_train_dataloader(cfg)`.
- On `StopIteration`, increments the epoch and calls
  `set_epoch(epoch)` on the dataset before re-iterating (re-shuffles).
- Passes batch keys straight through:
  `model(input_ids=…, attention_mask=…, labels=…)`.

### 10.2 Evaluator

- Constructed from `cfg.data.eval` (when non-`None`).
- Resolves each `EvalDatasetEntry.type` to a task, which imports the appropriate
  `oplm.data` builder/loader (§9.3).
- Owns per-task scheduling (`eval_every`, with per-dataset overrides) and metric
  computation; it does **not** contain data-loading code.

---

## 11. Testing strategy

Tests mirror the module tree under `tests/data/` (currently only `__init__.py`).
Prefer **real data**; provide small fixtures (drop real samples into
`tests/data/fixtures/`). Mark heavy parsing/model tests `@pytest.mark.slow` and
make large fixtures session-scoped.

- **Tokenizer parity (regression guard).** Assert the data layer's token IDs
  equal `OplmTokenizerFast`'s for a battery of sequences, and that
  `mask_token_id == 32`, `canonical_aa_ids == range(4, 24)`,
  `special_ids == {0,1,2,3,32}`. This is the test that would have caught the
  legacy off-by-one vocabulary bug.
- **Sequence determinism.** Same `(seed, epoch)` ⇒ byte-identical batch order
  and (with `deterministic=True`) identical mask patterns; different epochs ⇒
  different order.
- **Masking correctness.** Over many batches: selection rate ≈ `mask_prob`; the
  80/10/10 split within tolerance; special tokens and padding never selected;
  `labels == -100` exactly at unmasked positions and the original id at masked
  positions; random replacements drawn only from `canonical_aa_ids`.
- **Padding/truncation.** Sequences longer than `max_length - 2` are truncated;
  shapes are `(B, T)` with `T <= max_seq_len`; `attention_mask` matches padding.
- **Striping coverage.** Across `(rank, worker)` combinations, the union of
  yielded rows equals the dataset with no duplicates and no omissions.
- **Interleaving.** Empirical sampling ratio over many draws matches `fractions`
  within tolerance; exhausted sources refill.
- **Structure/variant/downstream.** Parse one real PDB and one tiny ProteinGym
  CSV; assert `coords` shape `(L, 3, 3)`, NaN handling, modified-residue
  mapping; assert variant parsing/validation against WT.
- **End-to-end.** A pilot-scale model trains a few steps on a tiny parquet
  fixture and runs one evaluation through each builder without shape errors
  (per the project testing convention for pipelines).

---

## 12. Migration from the current implementation

| Current symbol / location                         | Target                                                         |
| ------------------------------------------------- | -------------------------------------------------------------- |
| `data/tokenizer.py::ProteinTokenizer`             | **Removed.** Use `OplmTokenizerFast` (§3); `data/tokenizer.py` becomes a thin accessor + derived id constants. |
| `data/dataset.py` (Sharded/Interleaved)           | `data/sequence/dataset.py` (add explicit rank-aware striping)  |
| `data/collate.py::MLMCollator`                    | `data/sequence/collate.py` — split into pad primitive + mask layer; add `deterministic`/`seed`; derive id constants from tokenizer; expose `mask_token_prob`/`random_token_prob` from config |
| `data/loader.py::build_train_dataloader`          | `data/sequence/loaders.py::build_train_dataloader`             |
| `eval/data/sequence_loader.py::build_sequence_eval_dataloader` | `data/sequence/loaders.py::build_sequence_eval_dataloader` (eval policy via args) |
| `eval/data/sequence_loader.py::DeterministicMLMCollator` | **Removed.** Replaced by `MLMCollator(deterministic=True)` (§4.6) |
| `eval/data/structure_loader.py`                   | `data/structure/loader.py` (`StructureData`, `load_structures`) |
| (new)                                             | `data/variant/loader.py`, `data/downstream/loader.py`          |
| `eval/data/` (directory)                          | **Deleted** — eval imports `oplm.data.*`                        |

Hardcoded constants to lift out:

- Special-token id set `{0,1,2,3}` and AA range `5–24` in the legacy collator →
  derive from the tokenizer as `{0,1,2,3,32}` (non-maskable) and `range(4, 24)`
  (canonical AAs). Note the legacy values reflect the *wrong* 32-token vocab.
- `mask_token_prob=0.8`, `random_token_prob=0.1` → `DataConfig` (§8.2).
- The fixed eval mask probability / seed in `sequence_loader.py` → builder
  arguments with documented defaults.

---

## See also

- [`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md) — model, forward signature,
  and the canonical tokenizer (§3 there).
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — repository module map and runtime flow.
- [`src/oplm/config.py`](../src/oplm/config.py) — `DataConfig`,
  `TrainDatasetEntry`, `EvalDatasetEntry`, parsing helpers.
- [`src/oplm/configs/data/base.yaml`](../src/oplm/configs/data/base.yaml) —
  default data config and YAML syntax.

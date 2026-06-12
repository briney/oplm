# OPLM Data Tooling

How OPLM turns protein sequences (and structures, variants, labeled datasets)
into model-ready batches. This is the practical reference for dataset formats,
the tokenizer, and the MLM masking scheme. For the full layered narrative — the
modality registry, single-source-of-truth tokenizer design, and every builder —
see [OVERVIEW.md §14–20](OVERVIEW.md). For the `data.*` field reference, see
[CONFIG.md](CONFIG.md).

Code lives under `src/oplm/data/`.

## Dataset formats

The pretraining substrate is **parquet** — a single `.parquet` / `.parq` / `.pq`
file or a directory of shards (`src/oplm/data/sequence/dataset.py`). Required
columns:

| Column | Type | Notes |
| --- | --- | --- |
| `sequence_id` | `str` | Stable identifier. |
| `sequence` | `str` | Raw one-letter amino acids, **no special tokens**. |
| `masking_weights` | `list[float]` | Optional, one weight per residue. Read only when `data.weighted_masking=true`. |

Only the needed columns are read (`pq.read_table(columns=[...])`). There is no
HuggingFace `datasets` integration — parquet is the only on-disk format.

`ShardedProteinDataset` is an `IterableDataset` (streaming, not map-style). It
discovers shards and their row counts from parquet metadata at construction (no
data loaded), shuffles per epoch via `set_epoch(epoch)` with two independent
granularities (shard order and within-shard row order), and stripes rows across
the joint `(world_rank, worker_id)` index so every rank/worker sees a disjoint,
gap-free slice independent of launcher behavior. `InterleavedDataset` mixes
multiple sources by sampling fraction (normalized to 1.0); exhausted sources
re-initialize so unequal sizes keep mixing at the target ratio.

## Tokenizer

`OplmTokenizerFast` (`src/oplm/model/tokenization_oplm.py`) is a 33-token
WordLevel tokenizer with per-character pre-tokenization. Each sequence is wrapped
`<cls> … <eos>`. The special-token IDs are:

| Token | ID | Role |
| --- | --: | --- |
| `<cls>` | 0 | BOS / pooling token |
| `<pad>` | 1 | padding |
| `<eos>` | 2 | end of sequence |
| `<unk>` | 3 | unknown |
| `<mask>` | 32 | MLM mask |

The 20 canonical amino acids occupy IDs 4–23 (`canonical_amino_acid_ids()` in
`src/oplm/data/tokenizer.py`); the remaining IDs cover ambiguous/gap/structure
tokens. Every downstream constant — `mask_token_id`, `pad_token_id`,
`non_maskable_ids`, the canonical-AA set — is derived from the live tokenizer
instance, never hardcoded (see [OVERVIEW.md §15](OVERVIEW.md)).

## Collation

Two composable pieces (`src/oplm/data/sequence/collate.py`):

- **`tokenize_and_pad(batch)`** — the shared primitive. Truncates each raw
  sequence to `max_length - 2` (room for `<cls>`/`<eos>`), tokenizes, pads to the
  batch max with `<pad>`, and returns `{input_ids, attention_mask}`. No masking,
  no labels. Reused by the variant/structure/downstream consumers.
- **`MLMCollator`** — calls the primitive, applies masking, and adds `labels`.

Every sequence batch (train and eval) has exactly three keys:

| Key | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `input_ids` | `(B, T)` | long | canonical IDs, `<cls>…<eos>`, padded |
| `attention_mask` | `(B, T)` | long | `1` = real token, `0` = pad |
| `labels` | `(B, T)` | long | original id at masked positions, `-100` elsewhere |

`T` is the batch-max length, capped at `model.max_position_embeddings`.
`masking_weights`, when used, are consumed inside the collator and never emitted.

## MLM masking scheme

Masking is **dynamic** (RoBERTa-style): masks are regenerated every time an
example is drawn, so a sequence is masked differently across epochs. Only the
80/10/10 replacement split is borrowed from BERT. The full algorithm
(`MLMCollator.__call__` / `_replace`):

1. **Eligibility** — a position is maskable iff `attention_mask == 1` and its id
   is not in `non_maskable_ids` (the special tokens).
2. **Selection** — a **fixed count** `k = round(mask_prob · n_eligible)`
   positions are chosen by weighted sampling without replacement
   (Gumbel-top-k). Uniform masking is the special case where all weights are
   equal. Gumbel noise is resampled per draw, so *which* positions are masked
   varies across epochs; only the count `k` is fixed.
3. **Targets** — `labels` holds the original id at the selected positions and
   `-100` (the cross-entropy ignore index) everywhere else.
4. **Replacement** — of the selected positions, `mask_token_prob` (0.8) become
   `<mask>`, `random_token_prob` (0.1) become a uniform random canonical AA
   (IDs 4–23 only), and the remainder (0.1) keep the original token.

> **Fixed-k rounding is a deliberate policy.** `k = round(mask_prob · n_eligible)`
> uses PyTorch's `torch.round`, which rounds `.5` cases to the nearest even
> integer (banker's rounding). This is intentional: masking a fixed per-row count
> (rather than independent Bernoulli draws per position) keeps the masked-token
> budget stable across a batch. The behavior is pinned by tests in
> `tests/data/sequence/test_collate.py`. If a future change wants
> independent-Bernoulli or half-up rounding instead, treat it as a policy change
> and update those tests.

### Gumbel-top-k weighted selection

For each eligible position `i` with weight `w_i ≥ 0`, the selection key is
`key_i = log(w_i) + g_i` where `g_i = -log(-log(u_i))`, `u_i ~ U(0,1)`; the `k`
positions with the largest keys are masked (Efraimidis–Spirakis weighted
reservoir sampling). First-order inclusion probability is proportional to `w_i`,
sampling is without replacement, and the count is fixed regardless of weight
scale. A position with `w_i = 0` is never selected; weights are relative
(scale-invariant). If fewer than `k` positions have positive weight, all of them
(and only them) are masked.

**Per-position weighted masking** is gated by `data.weighted_masking` (default
`false`), *not* by mere presence of the `masking_weights` column. When off,
weights are ignored and masking is uniform. Per-residue weights flow through
`align_per_residue` (0.0 at `<cls>`/`<eos>`/pad). A row with `None` weights falls
back to uniform; an entirely-absent column warns once and falls back to uniform;
a length-mismatched weight array raises (a data-integrity bug surfaced loudly).

### Deterministic eval masking

The sequence-eval dataloader builds the same `MLMCollator` with
`deterministic=True`: the collator derives a per-batch RNG state from
`seed + batch_index` before masking and restores the ambient RNG afterward, so a
given batch always receives the same mask. This makes MLM eval loss comparable
across training steps. Training uses `deterministic=False`. There is no separate
eval collator class — it is a policy flag, not a fork (`build_train_dataloader`
vs `build_sequence_eval_dataloader` in `src/oplm/data/sequence/loaders.py`).

## Sequence handling: truncation, padding, packing

- **Truncation** — raw sequences are clipped to `max_length - 2` residues before
  tokenization.
- **Padding** — batches pad to the batch-max length with `<pad>`; the emitted
  `attention_mask` marks real vs pad positions.
- **Packing** — sequence packing (concatenating multiple short sequences into one
  fixed-length row) is **not implemented**; it is deferred. Sequences are
  truncated and padded, never packed. (Canon's depthwise convolution would leak
  across packed-sequence boundaries, which is the open blocker — see the project
  notes.)
- **Length grouping / bucketing** — not implemented; the iterable dataset uses
  straightforward streaming with per-epoch shuffling.

## Other modalities

The sequence path above is the pretraining substrate. Three further modalities
reuse the canonical tokenizer (via `tokenize_and_pad`, **no masking**) and feed
the eval harness:

- **Structure** (`src/oplm/data/structure/`) — parses `*.pdb`/`*.cif`/`*.ent`/
  `*.mmcif` with BioPython into backbone coordinates for the contact-prediction
  eval task. See [OVERVIEW.md §17](OVERVIEW.md).
- **Variant** (`src/oplm/data/variant/`) — zero-shot variant-effect assays loaded
  and validated from CSV. DMS-substitution assays (`mutant`, `DMS_score`) load via
  `load_variant_assays`, which reconstructs the wild-type from `mutated_sequence`
  when no explicit WT is given and reads the optional `DMS_score_bin` column into
  `VariantAssay.bin_labels` (used for AUROC). ProteinGym **clinical**-substitution
  assays (`mutant`, `DMS_bin_score` = `Pathogenic`/`Benign`, `protein_sequence`)
  load via `load_clinical_variant_assays`, which takes the wild-type directly from
  the constant `protein_sequence` column (no reconstruction) and maps labels to
  `1.0`/`0.0`. See [OVERVIEW.md §18](OVERVIEW.md).
- **Downstream / embedding** (`src/oplm/data/downstream/`) — labeled sequences
  with per-residue or sequence-level labels for probing/fine-tuning. See
  [OVERVIEW.md §19](OVERVIEW.md).

## Config surface (`data.*`)

Backed by `oplm.config.DataConfig`; defaults live in
`src/oplm/configs/data/base.yaml`.

| Key | Type | Default | Purpose |
| --- | --- | --- | --- |
| `data.train` | `str \| dict` | `None` | Training dataset path(s); a dict maps paths to mixing fractions. |
| `data.eval` | `dict` | `None` | Eval dataset specs — see [EVAL_HARNESS.md](EVAL_HARNESS.md). |
| `data.mask_prob` | `float` | `0.15` | Fraction of eligible positions masked per row. |
| `data.mask_token_prob` | `float` | `0.8` | Of masked positions, fraction replaced with `<mask>`. |
| `data.random_token_prob` | `float` | `0.1` | Of masked positions, fraction replaced with a random AA. |
| `data.weighted_masking` | `bool` | `false` | Honor the `masking_weights` column. |
| `data.num_workers` | `int` | `4` | DataLoader worker processes. |
| `data.pin_memory` | `bool` | `true` | Pin DataLoader memory for GPU transfer. |
| `data.prefetch_factor` | `int` | `4` | DataLoader prefetch buffer per worker. |
| `data.shuffle_shards` | `bool` | `true` | Shuffle shard order per epoch. |
| `data.shuffle_rows` | `bool` | `true` | Shuffle within-shard row order per epoch. |

`__post_init__` validates that `mask_token_prob + random_token_prob ≤ 1.0`.
`data.max_length` was **removed**; use `model.max_position_embeddings` for the
sequence-length cap.

# Data

How OPLM's training and evaluation data are defined, and how sequences (and
structures, variants, labeled datasets) become model-ready batches. Code lives
under `src/oplm/data/`. For the `data.*` field reference, see
[config.md](config.md).

## Training data

The training set is two UniRef-derived sequence sets, mixed by rows:

- **uniref70**: UniRef70 sequences, 129M rows, about 40B tokens.
- **deepclust70**: the deep-clustered 70% identity set, 5.35B rows, about 730B tokens.

The mix is set in `data.train` of [`configs/scaling.yaml`](../configs/scaling.yaml):
uniref70 `fraction: 0.05` and deepclust70 `fraction: 0.95`. Fractions are **rows**,
not tokens, because the loader samples a source per row. Sequences differ in length
between the two sets, so the token share is not the row share.

Rows to tokens: `tokens = rows x mean tokens per row`, with the mean taken per source.

| source | rows | tokens | mean tokens per row |
| --- | --: | --: | --: |
| uniref70 | 129M | about 40B | about 310 |
| deepclust70 | 5.35B | about 730B | about 136 |

With 0.05 / 0.95 as row fractions, `0.05 x 310 / (0.05 x 310 + 0.95 x 136)` is about
11% of tokens from uniref70. One epoch of the natural mix (every row of both sources
once) is about 770B tokens, which is `770B / 4,194,304 tokens per step`, about 183k
steps at 4M tokens per step (`train.tokens_per_step`).

## Download the exact training set

> **Pending:** the published training set and its layout (waiting on #61).

## How the training set was built

### Primary sources

> **Pending:** source databases, release versions and download dates (waiting on #61).

### Filtering

> **Pending:** filters applied to the raw sequences (waiting on #61).

### Clustering

> **Pending:** clustering tool, parameters and cluster counts (waiting on #61).

### Shuffling

deepclust70 was shuffled globally across shards, and the published copy is the
shuffled one. See *Why the data is shuffled* below.

> **Pending:** the shuffle procedure and seed (waiting on #61).

### Final counts

Rows and tokens per source are in *Training data* above.

> **Pending:** train/eval split sizes and shard counts (waiting on #61).

## Evaluation data

### Held-out sequences

The held-out sequence eval is `uniref70/eval.parquet` (`type: sequence`), configured
under `data.eval` in [`configs/scaling.yaml`](../configs/scaling.yaml). It is scored by
masked-LM loss with deterministic masking (see *Deterministic eval masking*).
For `data.eval` fields, see [evaluation.md](evaluation.md), *`data.eval` schema*.

> **Pending:** how the held-out rows were selected and their count (waiting on #61).

### Structure evaluation sets

Provenance for the methods section: every value is copied from the build artifact
named beside it.

Sets: `casp15_16`, `cameo`, `pdb40`, and the seed-42 subsets `cameo_256`, `pdb40_2048`
and `pdb40_256`. They are built by `oplm fold eval-sets` and published as the
Hugging Face dataset
[`brineylab/oplm-structure-evals`](https://huggingface.co/datasets/brineylab/oplm-structure-evals).
They are scored by the `structure` eval type (see *Scoring* below).

Built 2026-10-27 with `oplm fold eval-sets` at commit `c0b5f08` (oplm 0.4.1.dev180+gc0b5f08aa;
MMseqs2 18.8cc5c, installed from bioconda). Output: `$OPLM_DATA/hf/structure-evals/2026-10-12/`,
with a `README.md`, `summary.json` (per-stage counts) and `manifest.parquet` (one row per
structure, with its sha256). Design and rationale:
`docs/superpowers/specs/2026-10-27-structure-eval-sets-design.md`.

#### Sources

| source | origin | obtained | version, size, checksum |
|---|---|---|---|
| wwPDB mmCIF archive | `rsync.rcsb.org::ftp_data/structures/divided/mmCIF/` (port 33444) | 2026-10-12, 13:10–16:49 UTC | snapshot 2026-10-12; 260,626 entries, 92.3 GB |
| Chemical Component Dictionary | `files.wwpdb.org` `components.cif.gz` | 2026-10-12 | 51,389 components; sha256 `405da7e5d06527d45292db9400378d9788c024830cad4e3b43890b6c6a0dd4c3` |
| Preprocessed PDB (fold dataset) | `oplm fold preprocess`, code `e0ea2fd` (PR #34) | 2026-10-12 | 1,080,220 chains; HF `brineylab/oplmfold-training-data`, tag `wwpdb-2026-10-12` |
| CAMEO target list | CAMEO one-year exports: Wayback capture (2023-08-07) of `to_date=2022-01-15`, and the Biodataset4AI mirror of `to_date=2022-12-31` (CAMEO's archive returned HTTP 503) | 2026-10-12 | 755 chains, weekly releases 2021-10-02 .. 2022-09-24 |
| CASP15 domain targets | `predictioncenter.org/download_area/CASP15/targets/casp15.targets.TS-domains.public_12.20.2022.tar.gz` | 2026-10-27, 08:36 UTC | sha256 `ac159c284a1506b460143ef6e22832272c5da1b4078fdb5fec8292b373b6239b` |
| CASP15 target sequences | `predictioncenter.org/download_area/CASP15/sequences/casp15.seq.txt` | 2026-10-27, 08:36 UTC | sha256 `84dd04c1fda2a550af73e77ad25be94da23f16bd7ff6632f2c22c1978b9ee0a5` |
| CASP16 domain targets | `predictioncenter.org/download_area/CASP16/targets/casp16.targets_monomer_trimmed2domains.tgz` | 2026-10-27, 08:36 UTC | sha256 `30e6bf992baed08695ebebe481711adea867b483015d30e18d170a0eee66ceb9` |
| CASP16 target sequences | `predictioncenter.org/download_area/CASP16/sequences/casp16.T1.seq.txt` | 2026-10-27, 08:36 UTC | sha256 `77d0b5c4ffda52ca1be7563a92eabae7ca59a3aab814e79a377ddd962783e286` |
| CASP domain classifications | `predictioncenter.org/casp15/domains_summary.cgi`, `/casp16/domains_summary.cgi` (saved HTML in `sources/`) | 2026-10-27, 15:28 UTC | CASP15 labels domains FM, FM/TBM, TBM-hard or TBM-easy (sha256 `2b8b37076a310e500dd6619b357323f6a56bb01946e44cccc63b15ed9ba3067f`); CASP16 gives a difficulty label (easy, medium or hard), not FM/TBM (sha256 `023a1a0e2dc05dceea93bfaba039a8ec497c60e5a477e5b72fa78bb0466b49c1`). Neither is used for filtering |

#### Processing

1. **Structure parsing** (the preprocessed PDB). gemmi mmCIF parsing of L-peptide polymer
   entities in the asymmetric unit; first model; highest-occupancy alternate location per
   atom; hydrogens removed. Non-standard residues are mapped to their CCD parent when the
   component is a non-D peptide, otherwise to unknown. Entries with resolution > 9 Å and
   chains with fewer than 4 resolved residues are dropped. Each chain keeps its full entity
   sequence, with a per-atom resolved mask.
2. **Input convention.** Each structure is one chain. The model receives the full sequence:
   the deposited entity sequence (CAMEO, PDB40) or the full target sequence (CASP).
   Residues without coordinates stay in the sequence and are not scored: any residue pair
   that involves one is excluded from evaluation.
3. **Quality filter** (CAMEO, PDB40): X-ray diffraction at ≤ 2.5 Å or cryo-EM at ≤ 3.0 Å;
   NMR structures are excluded.
4. **Curation rules** (all sets). Only the 20 standard amino acids. Full-sequence length
   50–1024 (CASP, PDB40) or 50–512 (CAMEO). At least 70% of residues with backbone N, Cα and
   C (CAMEO, PDB40). At least L/5 long-range contacts, where L is the number of modeled
   residues and a contact is virtual Cβ–Cβ < 8 Å with sequence separation ≥ 24.
5. **CASP15 and CASP16.** Official domain-trimmed target coordinates. Combined
   (multi-domain) evaluation units are excluded. CASP16 phase-2 copies (`T2xxx`) and
   monomer-file aliases (`T0xxx`) are mapped to their `T1xxx` target. For versioned targets,
   the lowest version is kept. One structure per (target, domain), with only that domain's
   residues carrying coordinates.
6. **CAMEO.** The listed chains that pass the quality filter and the curation rules.
7. **PDB40.** The 175,496 distinct protein entity sequences of length ≥ 30 in the snapshot were
   clustered with MMseqs2 (`mmseqs easy-cluster --min-seq-id 0.4 -c 0.8 --cov-mode 0 -s 7.5
   --dbtype 1`; `--dbtype 1` forces amino-acid input, because without it MMseqs2 guessed
   nucleotide from the sorted FASTA's leading poly-A sequences) into 49,986 clusters. For each
   cluster, candidate chains passing the quality filter, the length range, the standard-residue
   rule and a screen of at least 70% resolved residues were ranked by resolution, then resolved
   fraction, then entry and chain id. Up to five candidates were tried in that order; the first
   passing the curation rules became the cluster's representative.
8. **Subsets.** Seed-42 random subsets of chains ≤ 512 residues: `cameo_256`, `pdb40_2048`
   and `pdb40_256`, with `pdb40_256` contained in `pdb40_2048`.
9. **Scoring** (the `structure` eval type). Unsupervised contact precision from the model's
   categorical Jacobian (20 single-substitution forward passes per residue of the full input
   sequence; centered, symmetrized, APC-corrected), reported as precision at L, L/2 and L/5 over
   long-range pairs (sequence separation ≥ 24). Pairs involving a residue without backbone
   coordinates are excluded, and L is the number of residues with N, Cα and C coordinates, not
   the full sequence length.

#### Counts

| set | candidates | removed (reason: count) | exported |
|---|---|---|---|
| `casp15_16` | 204 domain files, 122 distinct domains | duplicates: 82 (CASP16 `T2xxx`/`T0xxx` copies and higher versions); length: 12; contacts: 9 | 101 |
| `cameo` | 755 | missing: 3; skipped: 0; quality: 246; length: 54; modeled: 13; contacts: 4 | 435 |
| `pdb40` | 49,986 clusters | no_candidate: 19,298; exhausted: 917; max_rounds: 342 | 29,429 |
| `cameo_256`, `pdb40_2048`, `pdb40_256` | | | 256, 2,048, 256 |

CASP15 has 45 domain files (no duplicates); CASP16 has 159, of which 82 are duplicates, so 45 + 77 =
122 distinct domains. CAMEO `skipped` counts listed chains with fewer than three resolved Cα atoms;
`quality` is 46 NMR structures and 200 X-ray structures above 2.5 Å. The `pdb40` removed counts are
clusters and sum with the 29,429 exported to 49,986. `no_candidate` clusters had no chain passing
the quality filter, length, standard-residue and 70%-resolved screen; `exhausted` clusters had
candidates that all failed the curation rules; `max_rounds` clusters still had untried candidates
after 5 attempts. The build also counted individual candidates rejected by the curation rules while
searching clusters (`contacts`: 3,940; `modeled`: 6). These are candidate-level counts, not
clusters; don't add them to the removed total.

Overlap: 202 of the 435 CAMEO chains are also PDB40 representatives, and all 435 fall in a 40%
sequence cluster that has a PDB40 representative (measured). CAMEO is therefore not held out
from PDB40. Both counts are reported; nothing is removed.

#### Distribution

Published 2026-10-27 as the Hugging Face dataset
[`brineylab/oplm-structure-evals`](https://huggingface.co/datasets/brineylab/oplm-structure-evals),
folder `2026-10-12/`, tag `2026-10-12` (commit `08f188a3`). The Hub limits a directory to
10,000 files, so `pdb40/` is distributed as `2026-10-12/pdb40.tar.gz` (29,429 files; sha256
`2aa92cc7a9b715d7374df1af04bb068be468fabb4afd73b73ff1e2615cc4cc08`), which extracts to `pdb40/`;
the other sets are plain folders, byte-identical to the build output. Licenses: `cameo` and
`pdb40` derive from the wwPDB archive (CC0 1.0); `casp15_16` and the CASP files in `sources/`
fall under the Protein Structure Prediction Center's terms.

## Dataset formats

The pretraining substrate is **parquet**: a single `.parquet` / `.parq` / `.pq`
file or a directory of shards (`src/oplm/data/sequence/dataset.py`). Required
columns:

| Column | Type | Notes |
| --- | --- | --- |
| `sequence_id` | `str` | Stable identifier. |
| `sequence` | `str` | Raw one-letter amino acids, **no special tokens**. |
| `masking_weights` | `list[float]` | Optional, one weight per residue. Read only when `data.weighted_masking=true`. |

Only the needed columns are read (`pq.read_table(columns=[...])`). There is no
HuggingFace `datasets` integration; parquet is the only on-disk format.

`ShardedProteinDataset` is an `IterableDataset` (streaming, not map-style). It
discovers shards and their row counts from parquet metadata at construction (no
data loaded), shuffles per epoch via `set_epoch(epoch)` with two independent
granularities (shard order and within-shard row order), and stripes rows across
the joint `(world_rank, worker_id)` index so every rank/worker sees a disjoint,
gap-free slice independent of launcher behavior. `InterleavedDataset` mixes
multiple sources by sampling fraction (normalized to 1.0); exhausted sources
re-initialize so unequal sizes keep mixing at the target ratio. With
`data.reshuffle_on_refill` (the default) each such refill reads the source in a fresh
order, so an oversampled source is not replayed as the same sequence of rows. With
`data.drop_last` (the default) every `(rank, worker)` stream gets the same number of
rows per epoch and partial batches are dropped, so all ranks run the same number of
full batches per epoch.

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
tokens. The data layer uses this one tokenizer, so data token IDs are exactly the IDs
the model's embedding table expects. Every downstream constant (`mask_token_id`,
`pad_token_id`, `non_maskable_ids`, the canonical-AA set) is derived from the live
tokenizer instance, never hardcoded (`src/oplm/data/tokenizer.py`). The canonical-AA
set is also the random-replacement pool for masking and the masked-marginal scoring
pool for eval. An earlier data-side tokenizer defined a different, off-by-one
vocabulary (its IDs did not match the model's embedding rows) and was removed;
`tests/data/test_tokenizer.py` checks every derived constant against the live tokenizer.

Per-residue features (such as masking weights) follow the same truncation,
`<cls>`/`<eos>` insertion and padding as `input_ids` via `align_per_residue`
(0.0 at `<cls>`/`<eos>`/pad), which lives in the tokenizer layer.

## Collation

Two composable pieces (`src/oplm/data/sequence/collate.py`):

- **`tokenize_and_pad(batch)`**: the shared primitive. Truncates each raw
  sequence to `max_length - 2` (room for `<cls>`/`<eos>`), tokenizes, pads to the
  batch max with `<pad>`, and returns `{input_ids, attention_mask}`. No masking,
  no labels. Reused by the variant/structure/downstream consumers.
- **`MLMCollator`**: calls the primitive, applies masking, and adds `labels`.

Every sequence batch (train and eval) has exactly three keys:

| Key | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `input_ids` | `(B, T)` | long | canonical IDs, `<cls>…<eos>`, padded |
| `attention_mask` | `(B, T)` | long | `1` = real token, `0` = pad |
| `labels` | `(B, T)` | long | original id at masked positions, `-100` elsewhere |

`T` is the batch-max length, capped at `model.max_position_embeddings`.
`masking_weights`, when used, are consumed inside the collator and never emitted.

With `data.pack_sequences: true` (training only), the collator masks as above and
then packs the rows: each batch's sequences go back to back into as few rows of
at most `model.max_position_embeddings` tokens as first-fit-decreasing finds. The
batch then has `R ≤ B` rows and two more keys: `segment_ids` `(R, T)` long (`1..n`
for the sequences in a row, `0` at pads) and `num_sequences` (a plain `int`, `B`).

## Masking

Masking is **dynamic** (RoBERTa-style): masks are regenerated every time an
example is drawn, so a sequence is masked differently across epochs. Only the
80/10/10 replacement split is borrowed from BERT. The full algorithm
(`MLMCollator.__call__` / `_replace`):

1. **Eligibility**: a position is maskable iff `attention_mask == 1` and its id
   is not in `non_maskable_ids` (the special tokens).
2. **Selection**: a **fixed count** `k = round(mask_prob · n_eligible)`
   positions are chosen by weighted sampling without replacement
   (Gumbel-top-k). Uniform masking is the special case where all weights are
   equal. Gumbel noise is resampled per draw, so *which* positions are masked
   varies across epochs; only the count `k` is fixed.
3. **Targets**: `labels` holds the original id at the selected positions and
   `-100` (the cross-entropy ignore index) everywhere else.
4. **Replacement**: of the selected positions, `mask_token_prob` (0.8) become
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
weights are ignored and masking is uniform. A row with `None` weights falls
back to uniform; an entirely-absent column warns once and falls back to uniform;
a length-mismatched weight array raises (a data-integrity bug surfaced loudly).

### Deterministic eval masking

The sequence-eval dataloader builds the same `MLMCollator` with
`deterministic=True`: the collator derives a per-batch RNG state from
`seed + batch_index` before masking and restores the ambient RNG afterward, so a
given batch always receives the same mask. This makes MLM eval loss comparable
across training steps. Training uses `deterministic=False`. There is no separate
eval collator class; it is a policy flag, not a fork (`build_train_dataloader`
vs `build_sequence_eval_dataloader` in `src/oplm/data/sequence/loaders.py`).
The sequence-eval builder also turns shard and row shuffling off.

## Truncation, padding, packing

- **Truncation**: raw sequences are clipped to `max_length - 2` residues before
  tokenization.
- **Padding**: batches pad to the batch-max length with `<pad>`; the emitted
  `attention_mask` marks real vs pad positions.
- **Packing**: opt-in for training batches (`data.pack_sequences`; see
  [config.md](config.md)). The model keeps packed sequences apart: a
  block-diagonal attention mask, and Canon taps that would cross a sequence
  boundary contribute zero. Each sequence's logits and loss match its own padded
  row. Evaluation batches are never packed. `data.pack_sequences` is incompatible
  with `model.mask_dropout`.
- **Length grouping / bucketing**: not implemented; the iterable dataset uses
  straightforward streaming with per-epoch shuffling.

## Other modalities

The sequence path above is the pretraining substrate. Three further modalities
reuse the canonical tokenizer (via `tokenize_and_pad`, **no masking**) and feed
the eval harness. The harness owns task logic, scoring and metrics; the loaders
below only load. See [evaluation.md](evaluation.md), *Tasks and metrics*.

- **Structure** (`src/oplm/data/structure/`): parses `*.pdb`/`*.cif`/`*.ent`/
  `*.mmcif` with BioPython into `StructureData` (`name`, `sequence`, `coords`
  `(L,3,3)` backbone N, CA, C with NaN for missing atoms, `chain_id`) for the
  contact-prediction eval task, via `load_structures(directory, max_structures=None)`.
  Parsing policy: first model, first chain; modified residues mapped to canonical
  parents (`MSE→M`, `SEP→S`, `PTR→Y`); other heteroatoms skipped; unparseable files
  skipped with a warning; results sorted by filename. BioPython is imported lazily,
  so `import oplm.data` never hard-requires it.
- **Variant** (`src/oplm/data/variant/`): zero-shot variant-effect assays loaded
  and validated from CSV (one per assay). The wild-type comes from, in order: an explicit
  override (`EvalDatasetEntry.extra["wildtype"]` or a sidecar), then a constant
  `wildtype` column (`protein_sequence` for clinical assays), then reconstruction from
  `mutated_sequence`. DMS-substitution assays (`mutant`, `DMS_score`) load via
  `load_variant_assays`, which reconstructs the wild-type from `mutated_sequence`
  when no explicit WT is given and reads the optional `DMS_score_bin` column into
  `VariantAssay.bin_labels` (used for AUROC). ProteinGym **clinical**-substitution
  assays (`mutant`, `DMS_bin_score` = `Pathogenic`/`Benign`, `protein_sequence`)
  load via `load_clinical_variant_assays`, which takes the wild-type directly from
  the constant `protein_sequence` column (no reconstruction) and maps labels to
  `1.0`/`0.0`. Scoring is masked-marginal with position-specific masking, not
  `MLMCollator`.
- **Downstream / embedding** (`src/oplm/data/downstream/`): labeled sequences
  (TAPE, ProteinGLUE) with per-residue or sequence-level labels for
  probing/fine-tuning, stored as parquet/CSV with a `sequence` column plus label
  column(s). Per-residue labels are aligned to non-special positions and padded
  with `-100`; sequence-level labels are one scalar (regression) or class index
  (classification) per sequence.

## Why the data is shuffled

Shuffling in the loader is per shard: every rank and worker reads the same shard at
the same time, so each training batch comes from one shard per source for many
consecutive steps. That is only iid if each shard is a random sample of its source.
Data written in a sorted or clustered order (for example, by cluster) must be
shuffled globally across shards before training.

The late-loss investigation found why: the loss jump at epoch boundaries was
train-loss flattery, not harder data. Cluster-ordered deepclust70 shards, read in
lockstep, gave batches of near-duplicate sequences, so training loss in the first
epoch was flattered and then jumped when the order changed. It was resolved by
replacing deepclust70 `train/` with a globally shuffled copy (2026-10-20). The full
record is
[LATE_LOSS_INVESTIGATION.md](https://github.com/briney/oplm/blob/7db9c1bf0155fc9b4c05c8b358b3cf4cd6648f2c/docs/LATE_LOSS_INVESTIGATION.md).

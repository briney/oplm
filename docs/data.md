# Data

How OPLM's training and evaluation data are defined, and how sequences (and
structures, variants, labeled datasets) become model-ready batches. Code lives
under `src/oplm/data/`. For the `data.*` field reference, see
[config.md](config.md).

## Training data

The training set is two UniRef-derived sequence sets, mixed by rows:

- **uniref70**: UniRef90 release 2026_03 clustered at 70% identity, one row per cluster,
  75.7M train rows, about 27.5B tokens. Built by `oplm data uniref70` (see
  *How the training set was built*).
- **deepclust70**: the deep-clustered 70% identity set, 5.35B rows, about 730B tokens.

The mix is set in `data.train` of [`configs/scaling.yaml`](../configs/scaling.yaml):
uniref70 `fraction: 0.05` and deepclust70 `fraction: 0.95`. Fractions are **rows**,
not tokens, because the loader samples a source per row. Sequences differ in length
between the two sets, so the token share is not the row share.

> **Pending:** the production data mix for the stable and decay phases (waiting on #58).

Rows to tokens: `tokens = rows x mean tokens per row`, with the mean taken per source.

| source | rows | tokens | mean tokens per row |
| --- | --: | --: | --: |
| uniref70 | 75.7M | about 27.5B | about 363 |
| deepclust70 | 5.35B | about 730B | about 136 |

uniref70 tokens come from `uniref70/summary.json` (`stages.write.train`), before
truncation. The tokenizer adds `<cls>` and `<eos>` to each sequence, so a row is its
residues plus 2 tokens:

- **Phase 1** (representatives): `27,363,613,505 residues / 75,718,236 rows` = 361.4
  residues, so about 363.4 tokens per row and `27,363,613,505 + 2 x 75,718,236` =
  27,515,049,977 tokens per epoch.
- **Phase 2** (`sample_members: true`): each row serves one member, drawn uniformly
  within its cluster. The expected member length per row is 361.5 residues (measured
  over the train shards; not in `summary.json`), so about 363.5 tokens per row and
  about 27.5B tokens per epoch, the same as phase 1. The member-weighted mean,
  `45,532,832,579 member residues / 121,461,869 members` = 374.9 residues, is higher,
  but it weights each cluster by its size, and the loader draws per cluster.

With 0.05 / 0.95 as row fractions, `0.05 x 363 / (0.05 x 363 + 0.95 x 136)` is about
12% of tokens from uniref70. One epoch of the natural mix (every row of both sources
once) is about 758B tokens (27.5B + 730B), which is `758B / 4,194,304 tokens per step`,
about 181k steps at 4M tokens per step (`train.tokens_per_step`).

## Download the exact training set

The dataset is built but not yet published.

> **Pending:** the Hugging Face location of the published training set (waiting on #61).

The dataset folder (`$OPLM_DATA/hf/oplm-sequence-data/`) holds both sources. Its dataset
card (`README.md`) has the rebuild commands; `manifest.parquet` (one row per file: path,
rows, bytes, sha256; written by `oplm data manifest`) sits beside it.

```
oplm-sequence-data/
  README.md
  manifest.parquet
  uniref70/
    summary.json                 # build record: every uniref70 number below
    train/shard-00000.parquet    # 758 shards, 100,000 rows each (the last has 18,236)
    eval_singletons.parquet      # 5,000 held-out clusters with 1 member
    eval_gte2.parquet            # 5,000 held-out clusters with 2+ members
    test_singletons.parquet      # 5,000
    test_gte2.parquet            # 5,000
  deepclust70/
    train/summary.json           # re-cut record
    train/shard-00000.parquet    # 5,354 shards, 1,000,000 rows each (the last has 586,520)
    eval_singletons.parquet      # 15,000
    eval_gte2.parquet            # 15,000
    test_singletons.parquet      # 150,000
    test_gte2.parquet            # 150,000
```

uniref70 rows have `sequence_id`, `sequence` (the representative), `member_ids` and
`member_sequences`; deepclust70 rows have `sequence_id` and `sequence`. See
*Dataset formats*. Training reads only the two `train/` folders.

## How the training set was built

Provenance for the methods section: every value is copied from the build artifact
named beside it. uniref70 values come from `uniref70/summary.json` unless another
artifact is named; deepclust70 re-cut values come from `deepclust70/train/summary.json`.

uniref70 was built by `oplm data uniref70` (oplm 0.4.1.dev252+ga400dfede; MMseqs2
18.8cc5c), seed 0, 32 threads. Clustering ran 2026-10-30 at commit `ea4b33a`; the eval
screen ran at `e6f331c` and the write at `a400dfe`; output written 2026-11-01. The
`oplm_version` in `summary.json` was corrected by hand: the editable install's metadata
was stale (`summary.json`, `oplm_version_note`). Design and rationale:
`docs/superpowers/specs/2026-10-30-sequence-pretraining-dataset-design.md`.

### Primary sources

| source | origin | obtained | version, size, checksum |
|---|---|---|---|
| UniRef90 | `https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz` | 2026-10-30, 19:31:50–19:54:09 UTC | release 2026_03 (02-Sep-2026), 121,494,192 clusters; 32,107,879,019 bytes; sha256 `f6fa665e27d14fa3dadcff39df6db0052f125ffcf40ba7806f0863c1f262b8dc`; CC BY 4.0 (UniProt Consortium) |
| deepclust70 | the shuffled deepclust70 copy, 10,708 shards (`source_shards`) | 2026-10-20 (shuffled copy) | 5,353,586,520 rows |

UniRef90 values are from `$OPLM_DATA/uniref90/SOURCES.txt`, `SHA256SUMS`,
`download_started_utc.txt`, `download_finished_utc.txt` and `uniref90.release_note`.
`summary.json` records the same input sha256 (`input_sha256`).

> **Pending:** deepclust70's upstream sources, release versions and build procedure
> (waiting on #61).

### Filtering

uniref70: none beyond UniRef's own. Every UniRef90 2026_03 entry is in exactly one
uniref70 cluster: 121,494,192 sequences in (`uniref90_sequences_in`), 121,494,192 member
rows out (train 121,461,869 + eval 16,119 + test 16,204). No length, residue or
taxonomy filter is applied.

> **Pending:** filters applied to deepclust70's raw sequences (waiting on #61).

### Clustering

uniref70: the UniRef90 sequences were clustered with MMseqs2 18.8cc5c `easy-linclust`:

```
mmseqs easy-linclust uniref90.fasta.gz <prefix> <tmp> \
  --min-seq-id 0.7 -c 0.8 --cov-mode 0 --dbtype 1 --threads 32 --split-memory-limit 80G
```

`-c 0.8 --cov-mode 0` follows UniRef's own rule (members overlap the seed by at least
80%). `--dbtype 1` forces amino-acid input. The run took 2,473 s. It ran through
`oplm.data.build.uniref70.cluster_uniref90` at commit `ea4b33a`, not through the full
CLI, so its `summary.json` entry was written by hand from that run's record
(`cluster.manual.json`; `stages.cluster.note`).

A cluster's representative is the sequence MMseqs2 picks as the cluster center; its
members are the UniRef90 entries assigned to it, representative included. Result:
75,738,236 clusters (`clusters_out`).

Members per train cluster (`stages.write.train.members_per_cluster`): min 1, median 1,
mean 1.604, max 2,370.

| members | 1 | 2–3 | 4–7 | 8–15 | 16–31 | 32–63 | 64–127 | 128–255 | 256–511 | 512–1023 | 1024–2047 | 2048–4095 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| clusters | 60,298,749 | 11,193,570 | 3,011,528 | 842,621 | 262,141 | 80,833 | 22,036 | 5,458 | 1,128 | 153 | 17 | 2 |

> **Pending:** deepclust70's clustering tool, parameters and cluster counts (waiting on #61).

### Shuffling

uniref70 was shuffled once, before the eval split, by a seeded out-of-core shuffle
(`shuffle_clusters`; `stages.shuffle.params`: seed 0, 512 buckets). Pass 1 sends each
cluster to a uniformly random bucket (`numpy.random.default_rng(0)`); pass 2 permutes
each bucket `b` with `default_rng([0, b])`. The buckets, in order, are the shuffled
order. It shuffled 75,738,236 clusters (121,494,192 members) in 2,645 s. Eval and test
candidates are drawn in this order, and train is written in it, so shard and row order
carry no cluster order.

deepclust70 was shuffled globally across shards, and the published copy is the
shuffled one. See *Why the data is shuffled* below. The re-cut to 1,000,000-row shards
(`oplm data reshard`) kept every row and the global row order: the sha256 of the
`(sequence_id, sequence)` stream is
`dbf6da39210da1dbbe1f1e63794d91f0b94debbe30a556ccca56e612220a44a4` before and after
(`sha256_in`, `sha256_out`), over 5,353,586,520 rows from 10,708 source shards.

> **Pending:** deepclust70's shuffle procedure and seed (waiting on #61).

### Final counts

| file | rows (clusters) | member rows | representative residues | member residues |
|---|--:|--:|--:|--:|
| `uniref70/train/` | 75,718,236 | 121,461,869 | 27,363,613,505 | 45,532,832,579 |
| `uniref70/eval_singletons.parquet` | 5,000 | 5,000 | 1,749,551 | 1,749,551 |
| `uniref70/eval_gte2.parquet` | 5,000 | 11,119 | 1,973,383 | 4,469,677 |
| `uniref70/test_singletons.parquet` | 5,000 | 5,000 | 1,701,713 | 1,701,713 |
| `uniref70/test_gte2.parquet` | 5,000 | 11,204 | 1,891,172 | 4,295,018 |
| `deepclust70/train/` | 5,353,586,520 | | | |
| `deepclust70/eval_singletons.parquet`, `eval_gte2.parquet` | 15,000 each | | | |
| `deepclust70/test_singletons.parquet`, `test_gte2.parquet` | 150,000 each | | | |

uniref70 train is 758 shards of 100,000 rows (`stages.write`: row groups of 10,000,
zstd). deepclust70 train is 5,354 shards of 1,000,000 rows (zstd). deepclust70 row counts
are read from the parquet metadata; its held-out files were copied as-is. Tokens per
source are in *Training data* above.

## Evaluation data

### Held-out sequences

`oplm data uniref70` writes the held-out clusters as `eval_singletons.parquet` and
`eval_gte2.parquet` (and `test_*` likewise), split by whether the cluster has one member.
The held-out sequence evals are `uniref70_singletons` and `uniref70_gte2` (`type: sequence`),
configured under `data.eval` in [`configs/scaling.yaml`](../configs/scaling.yaml). They are
scored by masked-LM loss with deterministic masking on the representatives (see
*Deterministic eval masking*). For `data.eval` fields, see [evaluation.md](evaluation.md),
*`data.eval` schema*.

**Selection** (`select_eval`; values from `uniref70/summary.json`, `parameters` and
`stages.screen`). Walking the shuffled order, the first 250,000 singleton clusters and
the first 250,000 gte2 clusters are candidates. Every member of every candidate is
searched with MMseqs2 18.8cc5c against deepclust70 `train/` and against every member of
every other uniref70 cluster (a candidate's hits against its own cluster don't count;
hits against other candidates do):

```
--min-seq-id 0.7 -c 0.8 --cov-mode 0 -s 1 --max-seqs 100000 -e 1e6
```

These are the clustering thresholds. `-s 1` is the cheapest prefilter: hits at ≥ 70%
identity share many exact k-mers, so it still finds nearly all of them (`-s 4` was
projected at about 2 days over deepclust70). `--max-seqs` and `-e` are set so identity
and coverage alone decide a hit. A candidate with any hit is dropped. Within each stratum
the first 5,000 survivors, in shuffled order, are eval and the next 5,000 are test.
Eval and test are disjoint and are not in `train/`; candidates not chosen go back into
`train/` at their shuffled position.

| stratum | candidates | hits | survivors | survival |
|---|--:|--:|--:|--:|
| singletons | 250,000 | 185,247 | 64,753 | 25.9% |
| gte2 | 250,000 | 239,168 | 10,832 | 4.3% |
| total | 500,000 | 424,415 | 75,585 | 15.1% |

Of the 500,000 candidates, 412,231 hit deepclust70 and 214,223 hit another uniref70
cluster (a candidate can hit both).

The build stops if a stratum's survival is under 5%, because the survivors are then a
narrow slice. gte2 tripped it. On 2026-11-01 the user chose to use the survivors anyway,
since 10,832 is enough for 5,000 eval and 5,000 test, and reran with
`--min-survival 0.04` (commit `a400dfe`). So the gte2 eval and test sets come from the
4.3% of multi-member clusters with no ≥ 70%-identity relative anywhere in training. They
are a narrower slice than a random gte2 draw: clusters with a close relative elsewhere
in training are excluded. A bigger pool would not help: the pool is a uniform draw from the
stratum, so 4.3% estimates the share of all gte2 clusters with no relative, and more
candidates give more survivors at the same rate and of the same kind.

The old `uniref70/eval.parquet` is dropped: its provenance is unknown and it is not
published. Loss on the new evals is not comparable with earlier runs' curves.

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

Cluster-per-row datasets (one row per cluster, the representative in
`sequence_id` / `sequence`) add two optional parallel `list[str]` columns,
`member_ids` and `member_sequences`. Phase 1 trains on the representatives and ignores
them; phase 2 sets `sample_members: true` on that `data.train` source, and each served
row yields one uniformly chosen member, a pure hash of (pass seed, shard, row), so resume,
loader reshard layers (any-topology resume) and any topology serve the same member and every
epoch redraws it. Re-cutting the files with `oplm data reshard` changes shard and row indices,
and so the members drawn.

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
below only load. The harness gets all its data through `oplm.data`; there is no
`src/oplm/eval/data/`. See [evaluation.md](evaluation.md), *Tasks and metrics*.

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

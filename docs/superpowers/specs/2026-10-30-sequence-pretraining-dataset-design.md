# Sequence pretraining dataset: rebuilt uniref70 with member sampling

Status: design, 2026-10-30. Not built.

## Intent

Production pretraining has two phases (see `docs/data.md`, *Training data*, and #58):

- **Phase 1 (stable).** Sample a uniref70 cluster, train on its representative. The same
  cluster always yields the same sequence.
- **Phase 2 (decay).** Sample a uniref70 cluster exactly the same way, then train on one of its
  UniRef90 members, drawn uniformly. This is the ESM-2 recipe (cluster, then uniform member).

The only difference between the phases is the last step: what a sampled cluster turns into.
Cluster selection, batching, mixing and the resume cursor are the same in both.

The old uniref70 can't do phase 2: it has representatives only, and nobody remembers exactly how
it was built (UniRef90 or UniRef100). So uniref70 is rebuilt from scratch from public UniRef90,
in a form that serves both phases from one copy of the data. At the same time, the full sequence
pretraining set (new uniref70 + existing deepclust70) is assembled into one publishable layout,
which is the "exact training set + rebuild procedure" promise of #61.

## Goals

1. A uniref70 dataset built reproducibly from a pinned public UniRef90 release, with the build
   code in `oplm` and the provenance written down in `docs/data.md`.
2. One dataset that serves phase 1 directly and phase 2 by member sampling, with no second copy.
3. A loader change that turns phase 2 on for one source with one config flag, and keeps
   deterministic resume, any-topology resume (#63) and `branch_from` working.
4. deepclust70 resharded to fit the Hugging Face 10,000-files-per-folder limit, keeping every row
   and the global row order.
5. Both sources laid out as one dataset folder, ready to upload (upload itself is later).

## Non-goals

- **Rebuilding deepclust70.** It took about a month on a large machine. The existing, globally
  shuffled copy is used as-is, only re-cut into bigger shards.
- **Publishing to Hugging Face.** The layout is made ready to upload; uploading is a later step.
- **Filtering beyond what UniRef does.** No length, X-content or fragment filters. Long sequences
  are truncated at load time, as today.
- **Weighted member sampling.** Uniform over UniRef90 members; no weighting by length or by
  UniRef100 count.
- **Changing the production mix.** Fractions are decided in #58.

## Anti-goals

- A dataset sorted or clustered by shard. The late-loss investigation showed cluster-ordered
  shards read in lockstep give near-duplicate batches and flattered loss (`docs/data.md`, *Why
  the data is shuffled*). Rows must be globally shuffled before sharding.
- Eval leaking into training. Eval clusters, and every member of them, are absent from
  `uniref70/train/`, and have no hit at the clustering thresholds in deepclust70 or in any
  other uniref70 cluster.
- A member choice that depends on rank, worker or world size. Resume at a different topology
  must give the same members.
- Losing or reordering deepclust70 rows during the re-cut.

## Constraints

- **Machine.** Build runs on this workstation: 32 cores, 123 GB RAM, 1.3 TB free on `/home`.
  MMseqs2 is in the `mmseqs2` micromamba env.
- **Disk.** UniRef90 + MMseqs2 databases + temp files take a few hundred GB; deepclust70 is just
  under 400 GB and will be copied here by the user. MMseqs2 temp and intermediate databases may be
  deleted once the clusters are extracted; only the final dataset is kept. Stages run one after
  another (uniref70 build, delete its work dir, then the deepclust70 re-cut), with a free-space
  check before each. The dumped deepclust70 source is deleted only after its re-cut is verified,
  and only with the user's OK (my call).
- **Hugging Face.** At most 10,000 files per folder.
- **Loader contract.** Rows reach the collator as `{"sequence_id": str, "sequence": str}` (plus
  optional `masking_weights`). Phase 2 must produce rows of that shape, so nothing downstream of
  the dataset changes.
- **Resume cursor.** `DataCursor`, reshard layers and `_stream_position` count row positions per
  source and pass; they don't look at row contents. This design keeps it that way.

## Decisions

### Source and clustering (user)

- Input: the current public UniRef90 release from <https://www.uniprot.org/uniref/> (FASTA).
  Record the release version, download date, URL and sha256 of what was downloaded.
- Cluster the UniRef90 representative sequences at 70% identity with **MMseqs2 `linclust`**, not
  `mmseqs cluster`. Reasons: speed, and deepclust70 was also built with a linclust-like procedure
  (DIAMOND), so the two sources are clustered in the same spirit.
- Parameters follow UniRef's own rule (members have ≥ 80% overlap with the cluster's seed):
  `--min-seq-id 0.7 -c 0.8 --cov-mode 0`. Pass `--dbtype 1` when creating the database: the
  structure-eval build found MMseqs2 can mis-guess amino-acid input as nucleotide.
- A uniref70 cluster's **representative** is the sequence MMseqs2 picks as the cluster center.
- A uniref70 cluster's **members** are the UniRef90 entries assigned to it. The representative is
  one of its members.

### Dataset layout (agreed in conversation)

One dataset folder, both sources:

```
<dataset>/
  README.md                    # dataset card: sources, versions, rebuild commands, counts
  manifest.parquet             # one row per file: path, rows, bytes, sha256
  uniref70/
    train/shard-00000.parquet  # ~100k clusters per shard (~1,300 shards)
    eval.parquet               # held-out clusters
  deepclust70/
    train/shard-00000.parquet  # 1,000,000 rows per shard (~5,355 shards)
    eval.parquet               # copied as-is, if the user's dump has one
```

uniref70 row schema (one row per uniref70 cluster):

| column | type | meaning |
| --- | --- | --- |
| `sequence_id` | `str` | the representative's UniRef90 id, e.g. `UniRef90_P12345` |
| `sequence` | `str` | the representative's sequence |
| `member_ids` | `list<str>` | UniRef90 ids of every member, representative included |
| `member_sequences` | `list<str>` | their sequences, same order as `member_ids` |

The first two columns are exactly today's schema, so phase 1 reads the new uniref70 with the
current loader and no code change. Parquet reads only the columns asked for, so phase 1 never
reads the member columns. (Parallel lists rather than a `list<struct>`: my call, it keeps reads
to plain string arrays; either works.)

deepclust70 keeps its current schema.

### uniref70 assembly

1. Join clusters to sequences: one row per cluster, as above.
2. **Eval split (user: held out of the whole mix, not just uniref70).** Eval clusters must have no
   close relative in deepclust70 either, so eval loss isn't flattered by cross-source leakage.
   deepclust70 itself is not changed.
   - Draw a candidate pool of clusters uniformly at random with a fixed, recorded seed (pool size
     is the builder's call; start around 100,000 and grow it if too few survive).
   - Search every member of every candidate against deepclust70 with MMseqs2 at the clustering
     thresholds (`--min-seq-id 0.7 -c 0.8 --cov-mode 0`). Drop any candidate with a hit. Hint:
     the search is symmetric at `--cov-mode 0`, so the small candidate set can be the indexed
     side and deepclust70 the streamed side.
   - Of the survivors, keep the first **10,000** in the seeded random order (my call on size; see
     *Calls made*). They go to `uniref70/eval.parquet` (same schema) and appear nowhere in
     `uniref70/train/`. Unused candidates go back into `train/`.
   - Record the pool size, survival rate and search parameters in `summary.json`. If survival is
     very low (say under 5%), stop and report before building: survivors would be a biased slice
     (clusters with no deepclust70 relative at all), and the user decides whether that eval is
     still wanted.
   - **Also screen against uniref70 itself (user, 2026-10-30).** `linclust` under-clusters, so
     an eval cluster can have a member ≥ 70% identical to a member of another uniref70 cluster
     that stays in train. Search every candidate member against every member of every other
     uniref70 cluster at the same thresholds; a candidate's hits against its own cluster don't
     count. Hits against other candidates do count (conservative: a non-chosen candidate goes
     back to train). A candidate survives only with no hit in either screen; record the hit
     counts of each screen separately.
   - This step needs the deepclust70 dump. Clustering and assembly can run before it arrives.

   The old `uniref70/eval.parquet` is dropped (user): its provenance is unknown, and it is not
   published. Loss on the new eval is not comparable with earlier runs' curves.
3. **Global shuffle.** Shuffle all train clusters with a fixed, recorded seed, then write them in
   that order into shards of 100,000 rows. Shard and row order then carry no cluster or source
   order.
4. Record counts in a `summary.json` beside the output: UniRef90 sequences in, clusters out,
   members per cluster (min / median / mean / max, and a histogram), eval and train rows, tokens
   (representative and member), shard count.

### deepclust70 re-cut (agreed in conversation)

The current copy is 10,709 shards × 500k rows (5.35B), just over the 10,000-file limit. Re-cut to
**1,000,000 rows per shard**:

- Read the existing shards in sorted filename order, concatenate, and cut at 1M rows. Every row
  and the global row order stay the same; only shard boundaries move.
- Keep the schema and column types. Keep `masking_weights` if present.
- Verify: total row count equal; a content hash of the concatenated `(sequence_id, sequence)`
  stream equal before and after (streaming, not all in memory); record the source shard list and
  both hashes in `summary.json`.

Why 1M and not 600k: 600k gives about 8,900 files, close to the cap; 1M gives about 5,355.
Shard size doesn't change loader IO per training row. Each worker reads a whole shard and keeps
one row in `stride` (ranks × workers), so bytes read per row trained are `stride × row size`
whatever the shard size. Bigger shards only mean bigger, less frequent reads and about twice the
per-worker table memory (order 100–200 MB).

### Loader: phase-2 member sampling

**Config.** A new optional boolean on a `data.train` entry, `sample_members` (default `false`):

```yaml
data:
  train:
    uniref70:
      path: .../uniref70/train
      fraction: 0.10
      sample_members: true    # phase 2: draw a UniRef90 member per sampled cluster
    deepclust70:
      path: .../deepclust70/train
      fraction: 0.90
```

`TrainDatasetEntry` gains the field; `parse_train_configs` reads it; the train dataloader builder
passes it to that source's `ShardedProteinDataset`. Only the sources with the flag change.

**Behavior.** With `sample_members` on, `ShardedProteinDataset`:

- reads `member_ids` and `member_sequences` (and not `sequence`) in `_read_rows`;
- for each row it serves, picks member `j = h mod n_members`, where `h` is a hash of the pass's
  shuffle seed (the `epoch_seed` that `_iter_stream` already computes for this epoch and pass),
  the shard index and the row index within the shard;
- yields `{"sequence_id": member_ids[j], "sequence": member_sequences[j]}`.

Properties this gives, which are the requirements:

- **Topology-independent.** The choice depends on the physical row and the (epoch, pass) only, not
  on rank, worker, stride or reshard layers. Any-topology resume and step replays get the same
  member.
- **Fresh across epochs and refill passes.** A cluster sampled again in a later epoch or pass
  usually gets a different member.
- **Uniform** over a cluster's members, to within hash bias (negligible for 64-bit hashes).
- **Position counting unchanged.** One cluster row in, one sequence out, so `DataCursor`,
  `_stream_position`, token batching and refill passes are untouched.

With the flag off, behavior is byte-for-byte today's.

**Errors.** With the flag on, a shard missing `member_sequences` or `member_ids` raises at
dataset construction (a misconfigured phase 2 must not silently fall back to representatives).
`masking_weights` with `sample_members` is not supported (weights are per representative); raise
if both are set.

**Fingerprint and branching** (my call). `sample_members` is not part of
`train_data_fingerprint`: the shards and row positions are identical between phases, so a
`branch_from` that only flips the flag keeps the data cursor and continues the same stream. A
decay branch that also changes fractions changes the fingerprint and drops the cursor, as it does
today.

**Eval** (my call). Sequence eval stays on representatives: `uniref70/eval.parquet` is read with
the current eval loader, which reads `sequence_id` and `sequence` only. No phase-2 eval.

### Build code (agreed in conversation)

The build ships in `oplm` as CLI commands, like `oplm fold eval-sets`, so the rebuild procedure is
the code. Suggested shape (names are the builder's call):

- `oplm data uniref70 --uniref90 <fasta.gz> --out <dir>`: database, linclust, cluster extraction,
  assembly, eval split, shuffle, shards, `summary.json`. MMseqs2 is an external binary; take its
  path from an option or `PATH`, and record its version.
- `oplm data reshard --src <dir> --out <dir> --rows-per-shard 1000000`: the deepclust70 re-cut
  with its verification.
- `oplm data manifest <dataset>`: writes `manifest.parquet` and checks the 10,000-files-per-folder
  limit.

The steps must be restartable at stage boundaries (clustering is hours long). Intermediate MMseqs2
databases and temp files go in a work directory that can be deleted once clusters are extracted.

### Docs

- `docs/data.md`: fill the *Pending (#61)* sections for uniref70 (sources, clustering, shuffling,
  final counts) with values copied from `summary.json`, in the structure-eval provenance style.
  Describe `sample_members` and the phase-2 recipe. Replace the old uniref70 row counts.
- `docs/config.md`: the new `data.train.<name>.sample_members` field.
- `configs/scaling.yaml`: no change to the mix (#58); paths stay placeholders until publish.

## Tests (smallest set that fails if the logic breaks)

- Member sampling: same `(seed, epoch, pass, shard, row)` gives the same member across different
  world sizes and worker counts; members differ across epochs; over many draws of a 3-member
  cluster each member appears at roughly 1/3; with the flag off rows are unchanged.
- A missing member column with the flag on raises.
- uniref70 assembly on a tiny synthetic cluster table: every UniRef90 id lands in exactly one row;
  no eval cluster or eval member appears in train; the representative is in its own member list;
  a candidate with a (mocked) deepclust70 hit is never chosen for eval.
- Re-cut on tiny shards: same rows, same order, new boundaries.
- Clustering itself is not unit-tested; a small FASTA end-to-end run under `-m slow` is enough.

## Left to the builder

- Hash function for member choice (any fixed, documented 64-bit mix, in the style of the existing
  `_shard_row_seed` / `_pass_seed` constants).
- `linclust` tuning flags beyond the ones above (`--kmer-per-seq`, `--split-memory-limit`,
  threads), recorded in `summary.json`.
- How the join from cluster TSV to sequences is done (it must fit in 123 GB; streaming or
  on-disk is fine).
- Parquet writer settings (compression, row-group size), as long as they're recorded and
  `masking_weights`-free uniref70 shards stay readable by the current loader.
- CLI names and module layout.

## Calls made (for the user to check)

- **Eval size: 10,000 held-out clusters, not 0.1%.** The earlier default was 0.1% of clusters.
  The sequence eval runs the whole eval file every 500 steps with no sample cap, and 0.1% would
  be on the order of 100k sequences per eval round. 10,000 keeps eval cheap and still holds the
  clusters out.
- **Parallel `member_ids` / `member_sequences` lists**, not a list of structs.
- **`sample_members` is not in the data fingerprint**, so a flag-only branch keeps its cursor.
- **Eval stays on representatives** in phase 2.
- **`masking_weights` + `sample_members` is an error**, not a silent fallback.
- **Eval screening searches all members, not just representatives**, and the build stops for a
  decision if under ~5% of candidates survive.
- **Build stages run sequentially**, and the deepclust70 dump is deleted only with the user's OK.

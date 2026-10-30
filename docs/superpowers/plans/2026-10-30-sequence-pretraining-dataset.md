# Sequence Pretraining Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild uniref70 from public UniRef90 as one-row-per-cluster parquet (representative + UniRef90 members), add phase-2 member sampling to the loader, re-cut deepclust70 to 1M-row shards, and assemble both into one upload-ready dataset folder.

**Architecture:** Phase 2 is a per-source flag on `ShardedProteinDataset`: it reads the member list columns instead of `sequence` and picks a member by a pure hash of (pass seed, shard, row), so row positions, the resume cursor and token batching are untouched. The build is a new `oplm data` CLI group (`src/oplm/data/build/`) with restartable stages: `easy-linclust` → bucketed global shuffle → deepclust70 leakage screen on the head of the shuffled order → shard writing. deepclust70 is re-cut by streaming its shards in sorted order.

**Tech Stack:** Python 3.11+, pyarrow (only dependency for data handling; no duckdb/polars), MMseqs2 (`mmseqs2` micromamba env), typer.

**Spec:** `docs/superpowers/specs/2026-10-30-sequence-pretraining-dataset-design.md`

## Global Constraints

- Clustering: `mmseqs easy-linclust` with `--min-seq-id 0.7 -c 0.8 --cov-mode 0`; databases created with `--dbtype 1`.
- Leakage screen: same thresholds, `--min-seq-id 0.7 -c 0.8 --cov-mode 0`.
- uniref70 row schema: `sequence_id: str`, `sequence: str`, `member_ids: list<str>`, `member_sequences: list<str>`; representative is in its own member list.
- uniref70 train shards: 100,000 rows. deepclust70 shards: 1,000,000 rows. Eval: 10,000 clusters.
- Every folder in the output has at most 10,000 files.
- `sample_members` defaults to `false`; with it off, loader output is byte-for-byte today's.
- `sample_members` is NOT part of `train_data_fingerprint`.
- All seeds, tool versions, input sha256s, parameters and counts go into `summary.json`.
- Run Python as `.venv/bin/python`; fast suite is `.venv/bin/python -m pytest -m "not slow"`.
- Work branch: `feat/sequence-dataset` off `main`. Spec and plan are committed in Task 1.

## Review Focus

- **Singleton clusters** (one member) in phase 2: always yield the representative, no error. → test in Task 1.
- **Corrupt member lists** (empty list, or `member_ids` / `member_sequences` lengths differ) in a served row: raise `ValueError` naming shard and row, never yield a mismatched id/sequence. → test in Task 1.
- **Phase-2 resume at a different topology**: same members for the same rows at world size 1 vs 2, 1 vs 2 workers. → test in Task 1.
- **Re-cut tail**: total rows not a multiple of 1M → last shard short, no row lost or duplicated. → test in Task 2.
- **Wrapped FASTA / headers with descriptions**: UniRef90 headers carry `n=… Tax=… RepID=…` after the id, and sequences may span lines; ids must be the first token and sequences joined. → test in Task 3.

---

### Task 1: Phase-2 member sampling in the loader

**Files:**
- Modify: `src/oplm/config.py` (`TrainDatasetEntry`, ~line 377)
- Modify: `src/oplm/data/config.py` (`_parse_train_mapping`)
- Modify: `src/oplm/data/sequence/dataset.py` (`ShardedProteinDataset.__init__`, `_iter_stream`, `_read_rows`; new `_member_index`)
- Modify: `src/oplm/data/sequence/loaders.py:119-127`
- Modify: `docs/config.md`, `docs/data.md` (*Dataset formats*: optional member columns and phase 2)
- Test: `tests/data/sequence/test_member_sampling.py`, `tests/data/test_config.py`

**Interfaces:**
- Produces: `TrainDatasetEntry.sample_members: bool = False`; `ShardedProteinDataset(..., sample_members: bool = False)`; module constants `MEMBER_IDS_COLUMN = "member_ids"`, `MEMBER_SEQUENCES_COLUMN = "member_sequences"` in `oplm.data.sequence.dataset` (Tasks 4–5 write these names).

- [ ] **Step 0: Branch and commit the spec + plan**

```bash
git checkout -b feat/sequence-dataset
git add docs/superpowers/specs/2026-10-30-sequence-pretraining-dataset-design.md docs/superpowers/plans/2026-10-30-sequence-pretraining-dataset.md
git commit -m "docs: sequence pretraining dataset design and plan"
```

- [ ] **Step 1: Write the failing tests**

A helper writes small cluster shards with pyarrow (2 shards; clusters of 1, 3 and 5 members; `sequence` = rep). Tests:

```python
def test_flag_off_is_unchanged(tmp_path):        # rows == today's {"sequence_id", "sequence"} for same seed/epoch
def test_member_rows_come_from_their_cluster(tmp_path):
    # every yielded (sequence_id, sequence) is a (member_ids[j], member_sequences[j]) pair of the row served
def test_singleton_yields_representative(tmp_path)
def test_same_member_across_topologies(tmp_path):
    # collect {cluster rep -> served member id} with world 1/workers 1 and with world 2 (monkeypatch
    # _resolve_distributed_context like tests/data/sequence/test_reshard.py does); dicts equal
def test_members_change_across_epochs(tmp_path):  # 5-member cluster: set_epoch(0..9) gives >1 distinct member
def test_member_choice_is_uniform():
    # _member_index over 30,000 (shard, row) pairs with n=3: each count within 3% of 10,000
def test_missing_member_columns_raise(tmp_path):  # plain {sequence_id, sequence} shard + sample_members=True -> ValueError at construction
def test_masking_weights_with_members_raise(tmp_path)  # load_masking_weights=True + sample_members=True -> ValueError
def test_mismatched_member_lists_raise(tmp_path)  # row with 2 ids / 3 seqs, and a row with [] -> ValueError on iteration, message has shard path and row index
```

In `tests/data/test_config.py`: `sample_members: true` parses to `True`; omitted → `False`; a non-bool (`"yes"`) → `ValueError`; `train_data_fingerprint` of a config with the flag equals the one without.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/sequence/test_member_sampling.py tests/data/test_config.py -v`
Expected: FAIL (`unexpected keyword argument 'sample_members'`, missing attribute).

- [ ] **Step 3: Implement**

- `TrainDatasetEntry`: add `sample_members: bool = False`. `_parse_train_mapping` reads `value.get("sample_members", False)`, rejects non-`bool`. The bare-string form stays `False`.
- `loaders.py`: pass `sample_members=entry.sample_members` to the train `ShardedProteinDataset`. Eval builder unchanged.
- `ShardedProteinDataset.__init__`: with `sample_members`, raise `ValueError` if `load_masking_weights` is also set, or if any shard's schema lacks either member column (schema is already read per shard).
- `_member_index(epoch_seed: int, shard_idx: int, row_idx: int, n: int) -> int`: splitmix64 finalizer over `epoch_seed`, `shard_idx`, `row_idx` (fold each in with `_PHI`-style mixing), then `% n`. Pure Python ints masked to 64 bits; document the constants as part of the reproducibility contract next to the existing seed constants.
- `_iter_stream` passes its `epoch_seed` (already the pass seed) to `_read_rows`. `_read_rows(shard_idx, row_indices, epoch_seed)`: with `sample_members`, read `["member_ids", "member_sequences"]` instead of `["sequence_id", "sequence"]`, validate the row's two lists (non-empty, equal length), pick `j = _member_index(...)`, and yield `{"sequence_id": ids[j], "sequence": seqs[j]}`.

- [ ] **Step 4: Run tests to verify they pass, then the fast suite**

Run: `.venv/bin/python -m pytest tests/data -m "not slow" -q`
Expected: all pass (resume, reshard and refill tests unchanged).

- [ ] **Step 5: Docs**

`docs/config.md`: `data.train.<name>.sample_members` (bool, default false; phase-2 member sampling; requires member columns; not in fingerprint, so a flag-only `branch_from` keeps the cursor). `docs/data.md` *Dataset formats*: the two optional list columns and the phase-1/phase-2 recipe in two or three sentences. Run `.venv/bin/python -m pytest tests/test_docs.py -q` → PASS.

- [ ] **Step 6: Commit**

```bash
git add -A src/oplm tests docs/config.md docs/data.md
git commit -m "feat(data): phase-2 member sampling via data.train.<name>.sample_members"
```

---

### Task 2: `oplm data` CLI group and deepclust70 re-cut

**Files:**
- Create: `src/oplm/data/build/__init__.py`, `src/oplm/data/build/reshard.py`, `src/oplm/data/build/cli.py`
- Modify: `src/oplm/cli.py` (register `data` group next to `fold`)
- Test: `tests/data/build/test_reshard.py`

**Interfaces:**
- Produces: `reshard(src: Path, out: Path, *, rows_per_shard: int) -> dict` returning `{"source_shards": [names], "rows": int, "shards_out": int, "sha256_in": str, "sha256_out": str}` and writing it to `out/summary.json`; `stream_digest(paths: Sequence[Path]) -> tuple[int, str]` (row count, sha256 over `(sequence_id, sequence)` in order); typer `app` in `oplm.data.build.cli`, registered as `oplm data`; shard naming `shard-{i:05d}.parquet`.

- [ ] **Step 1: Write the failing tests**

```python
def test_reshard_preserves_rows_and_order(tmp_path):
    # 7 source shards of 3 rows (21 rows) -> rows_per_shard=5 -> 5 shards: 5,5,5,5,1
    # concatenated (sequence_id, sequence) identical to source; summary sha256_in == sha256_out
def test_reshard_keeps_extra_columns(tmp_path)   # masking_weights column survives with same values
def test_reshard_refuses_nonempty_out(tmp_path)  # existing shard in out -> FileExistsError
def test_cli_reshard(tmp_path)                    # CliRunner: `data reshard SRC OUT --rows-per-shard 5` exit 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/data/build/test_reshard.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement**

Stream source shards in `_discover_shards` order by record batch (`pq.ParquetFile.iter_batches`), buffer to `rows_per_shard`, write each output with the source schema. Hash while writing; then `stream_digest` re-reads the output for `sha256_out` and raises `RuntimeError` on mismatch. Write to `out/.tmp-shard-*` then rename, so a crash leaves no half shard under a final name. Writer settings: zstd, recorded in `summary.json`.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/oplm/data/build src/oplm/cli.py tests/data/build
git commit -m "feat(data): oplm data reshard re-cuts shards keeping row order"
```

---

### Task 3: Clustering stage and cluster-block parser

**Files:**
- Create: `src/oplm/data/build/mmseqs.py` (move `_run_mmseqs_cmd` → `run_mmseqs_cmd` and `mmseqs_version` here from `src/oplm/fold/evalsets/pdb.py`; `pdb.py` imports them)
- Create: `src/oplm/data/build/uniref70.py`
- Test: `tests/data/build/test_uniref70.py`

**Interfaces:**
- Produces: `run_mmseqs_cmd(mmseqs: str, args: list[str]) -> CompletedProcess[str]`; `mmseqs_version(mmseqs: str = "mmseqs") -> str`; `cluster_uniref90(fasta: Path, workdir: Path, *, mmseqs: str, threads: int, extra_args: Sequence[str] = ()) -> Path` returning the `easy-linclust` `<prefix>_all_seqs.fasta`; `iter_clusters(all_seqs_fasta: Path) -> Iterator[Cluster]` with `Cluster = NamedTuple(rep_id: str, member_ids: list[str], member_sequences: list[str])`, rep's own entry first in the member lists.

- [ ] **Step 1: Write the failing tests**

```python
def test_iter_clusters_parses_all_seqs_format(tmp_path):
    # easy-cluster *_all_seqs.fasta format: a lone rep header line starts a cluster, then
    # header+sequence pairs for each member (rep first). Two clusters, one singleton.
def test_iter_clusters_ids_and_wrapped_sequences(tmp_path):
    # headers ">UniRef90_P1 Cluster: x n=3 Tax=y RepID=Z" -> id "UniRef90_P1"; 60-col wrapped seqs joined
def test_rep_is_first_member(tmp_path)
@pytest.mark.slow
def test_cluster_uniref90_end_to_end(tmp_path):
    # skip unless mmseqs on PATH; 6 sequences: two near-identical pairs + two unrelated -> 4 clusters,
    # every input id appears in exactly one cluster
```

- [ ] **Step 2: Run to verify failure** → FAIL (module missing).

- [ ] **Step 3: Implement**

`cluster_uniref90` runs `easy-linclust <fasta> <workdir>/uniref70 <workdir>/tmp --min-seq-id 0.7 -c 0.8 --cov-mode 0 --dbtype 1 --threads N *extra_args` and returns `<workdir>/uniref70_all_seqs.fasta`. `iter_clusters` streams line by line (never loads the file): a header followed directly by another header opens a new cluster.

- [ ] **Step 4: Run tests** (`-m "not slow"`, then the slow one with `PATH=$HOME/micromamba/envs/mmseqs2/bin:$PATH`) → PASS. Also `tests/fold` fast suite → PASS (moved helpers).

- [ ] **Step 5: Commit** `feat(data): uniref90 linclust stage and cluster parser`

---

### Task 4: Bucketed global shuffle

**Files:**
- Modify: `src/oplm/data/build/uniref70.py`
- Test: `tests/data/build/test_uniref70.py`

**Interfaces:**
- Consumes: `iter_clusters`, `Cluster`, column-name constants from Task 1.
- Produces: `shuffle_clusters(clusters: Iterable[Cluster], out_dir: Path, *, seed: int, n_buckets: int) -> list[Path]` writing `out_dir/part-{b:05d}.parquet` in the uniref70 row schema; reading the parts in returned order is the shuffled order. `iter_rows(parts: Sequence[Path]) -> Iterator[dict]` streams rows back in that order.

- [ ] **Step 1: Write the failing tests**

```python
def test_shuffle_is_a_permutation(tmp_path)        # 1,000 clusters, 8 buckets: same multiset of rep ids, order != input
def test_shuffle_is_deterministic(tmp_path)        # same seed -> identical order; other seed -> different
def test_shuffle_breaks_input_order(tmp_path):
    # input sorted by rep id; Spearman correlation between input and output position |rho| < 0.1
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement**

Pass 1: `numpy.random.default_rng(seed)` draws a bucket per cluster in stream order; append to that bucket's parquet writer (buffered record batches). Pass 2: per bucket, read, permute rows with `default_rng([seed, b])`, rewrite. Buckets sized to fit in memory: default `n_buckets` = 512 (record in summary). Write `.tmp` then rename per bucket so pass 2 can resume per bucket.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit** `feat(data): seeded out-of-core global shuffle of uniref70 clusters`

---

### Task 5: Leakage screen, eval selection and shard writing

**Files:**
- Modify: `src/oplm/data/build/uniref70.py`
- Test: `tests/data/build/test_uniref70.py`

**Interfaces:**
- Consumes: `iter_rows`, `run_mmseqs_cmd`, `mmseqs_version`.
- Produces:
  - `screen_against(candidates: Sequence[dict], reference: Path, workdir: Path, *, mmseqs: str, threads: int, chunk_rows: int) -> set[str]`: rep ids of candidates with any member hitting any sequence in the reference parquet dir (deepclust70) at the global thresholds.
  - `select_eval(rows: Iterable[dict], *, pool: int, n_eval: int, screen: Callable[[Sequence[dict]], set[str]], min_survival: float = 0.05) -> tuple[list[dict], dict]`: takes the first `pool` rows as candidates, returns the first `n_eval` survivors in order plus stats `{pool, hits, survivors, survival_rate}`; raises `LowSurvivalError` (subclass of `RuntimeError`) if `survivors / pool < min_survival`, and `ValueError` if fewer than `n_eval` survive (caller grows `pool`).
  - `write_dataset(parts: Sequence[Path], eval_rows: list[dict], out: Path, *, rows_per_shard: int) -> dict`: `out/eval.parquet` and `out/train/shard-{i:05d}.parquet` in shuffled order with eval rep ids skipped; returns counts (rows, member rows, rep and member residues, members-per-cluster min/median/mean/max and a log2 histogram).

- [ ] **Step 1: Write the failing tests**

```python
def test_select_eval_skips_hits_and_keeps_order()   # pool 10, screen flags 3 -> first n_eval of the 7 survivors, in order
def test_select_eval_low_survival_raises()          # 1 of 100 survive, min_survival .05 -> LowSurvivalError
def test_select_eval_too_few_survivors_raises()     # survival ok but < n_eval -> ValueError
def test_write_dataset_partitions(tmp_path):
    # every rep id in exactly one of eval/train; no eval member id appears in any train row;
    # train shards hold rows_per_shard rows except the last; train order == shuffled order minus eval
@pytest.mark.slow
def test_screen_against_end_to_end(tmp_path):
    # skip unless mmseqs; reference has a 95%-identical copy of candidate A's second member only -> {A}
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement `screen_against`**

Write all candidates' members to one FASTA and `createdb --dbtype 1` + `createindex` it (the small, indexed side). Stream the reference in chunks of `chunk_rows` rows: write chunk FASTA → `createdb` → `search` (query = chunk, target = candidates, `--min-seq-id 0.7 -c 0.8 --cov-mode 0`) → `convertalis` → record hit target ids to `workdir/hits-{k:05d}.txt` → delete the chunk's databases and FASTA. Skip chunks whose hits file exists (restart). Map hit member ids back to rep ids. Default `chunk_rows` 500,000,000; sensitivity (`-s`) is the builder's call, recorded in summary.

- [ ] **Step 4: Implement `select_eval` and `write_dataset`** per the interfaces; write `.tmp` then rename.

- [ ] **Step 5: Run tests** → PASS.

- [ ] **Step 6: Commit** `feat(data): deepclust70 leakage screen, eval split and uniref70 shards`

---

### Task 6: `oplm data uniref70` and `oplm data manifest` commands

**Files:**
- Modify: `src/oplm/data/build/cli.py`, `src/oplm/data/build/uniref70.py`
- Create: `src/oplm/data/build/manifest.py`
- Test: `tests/data/build/test_cli.py`

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces:
  - `oplm data uniref70 --uniref90 FASTA --release TEXT --deepclust70 DIR --workdir DIR --out DIR [--seed 0 --threads 32 --mmseqs mmseqs --pool 100000 --n-eval 10000 --rows-per-shard 100000 --n-buckets 512 --chunk-rows 500000000 --keep-workdir]`
  - Stages, each skipped when `workdir/<stage>.done` (JSON with its outputs and stats) exists: `cluster` → `shuffle` (then deletes `workdir/uniref70*` and `workdir/tmp` unless `--keep-workdir`) → `screen` → `write`. Before each stage, check free space on `workdir` and `out` and refuse to start below an estimate (2.5× the input FASTA size for `cluster`, 1.2× for `shuffle`, 200 GB for `screen`).
  - If `--deepclust70` does not exist yet, run through `shuffle`, then exit 0 with a message that the `screen` stage waits for it (the dump arrives later).
  - `out/summary.json`: UniRef90 release (from `--release`), input sha256, mmseqs version, oplm version, every parameter and seed, per-stage stats.
  - `write_manifest(dataset: Path) -> Path` → `dataset/manifest.parquet` (`path`, `rows`, `bytes`, `sha256`); raises `ValueError` naming any folder with more than 10,000 files. CLI: `oplm data manifest DATASET`.

- [ ] **Step 1: Write the failing tests**

```python
def test_uniref70_stages_resume(tmp_path, monkeypatch):
    # monkeypatch cluster_uniref90 to write a fixture all_seqs.fasta and screen_against to return set();
    # run once; delete write.done and out/; rerun -> cluster and shuffle are NOT called again; outputs identical
def test_uniref70_waits_for_deepclust70(tmp_path, monkeypatch)  # missing --deepclust70 dir: shuffle.done written, no screen.done, exit 0
def test_uniref70_summary_fields(tmp_path, monkeypatch)  # summary.json has release, sha256, seed, counts per stage
def test_manifest_counts_and_limit(tmp_path)             # rows/sha256 right; 10,001 files in one folder -> ValueError
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement** per the interfaces.

- [ ] **Step 4: Run** `.venv/bin/python -m pytest -m "not slow" -q` → all PASS; `ty check src/` and `ruff check` clean.

- [ ] **Step 5: Commit** `feat(data): oplm data uniref70 build and manifest commands`

---

### Task 7: Build the dataset on this machine (operational)

No code unless a run exposes a bug (fix it in the owning task's module with a test, commit, rerun). Dataset root: `$HOME/data/hf/oplm-sequence-data/` (`uniref70/`, `deepclust70/`); workdir `$HOME/data/tmp/uniref70-build/`.

- [ ] **Step 1: Download.** `https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz` and `uniref90.release_note` (same folder) into `$HOME/data/uniref90/`. Record URL, UTC date, release (from the release note) and sha256 in a `SOURCES.txt` beside it.
- [ ] **Step 2: Cluster + shuffle.** Run `oplm data uniref70` with `PATH=$HOME/micromamba/envs/mmseqs2/bin:$PATH`, `--threads 32`, in the background; it stops after `shuffle` if the deepclust70 dump isn't there yet. Check `shuffle.done` stats: clusters ≈ 1e8 scale, every UniRef90 id counted once.
- [ ] **Step 3: Re-cut deepclust70** once the user's dump lands: `oplm data reshard <dump>/train $ROOT/deepclust70/train --rows-per-shard 1000000`; confirm 10,709 source shards, 5.35B rows, ~5,355 output shards, digests equal. Copy `eval.parquet` if present. **Ask the user before deleting the dump.**
- [ ] **Step 4: Screen + write.** Rerun `oplm data uniref70 ... --deepclust70 $ROOT/deepclust70/train`. If `LowSurvivalError`, stop and report the survival rate to the user; do not lower the threshold yourself.
- [ ] **Step 5: Manifest.** `oplm data manifest $ROOT`; confirm no folder over 10,000 files.
- [ ] **Step 6: Docs and configs.** Fill `docs/data.md` *Pending (#61)* uniref70 parts (sources, clustering command and version, shuffle seed and buckets, screen stats, final counts) from `summary.json`, replace the old uniref70 rows/tokens and the rows→tokens table, and note the old eval is retired (curves not comparable). Write `$ROOT/README.md` (dataset card: layout, schema, phase-1/phase-2 usage, rebuild commands, counts, licenses: UniRef CC BY 4.0). Commit `docs(data): uniref70 provenance and dataset card`.
- [ ] **Step 7: Push and open a PR** (no HF upload).

# Milestone 2a acceptance: wwPDB preprocessing run and publication

Plan: `docs/superpowers/plans/2026-10-11-fold-m2a-data.md`, Task 10. Addendum:
`docs/superpowers/specs/2026-10-11-fold-m2-pipeline-proof-design.md` §4.

## Machine and dates

- Development machine: 32 cores, 123 GB RAM, 1.9 TB volume (`/home`), Linux 7.2.5.
- Code: branch `feat/fold-m2a` at `e0ea2fd` (PR #34).
- Run date: 2026-10-12 (all times UTC).
- Data root: `$OPLM_DATA=/home/bryanbriney/data`; staging for upload mirrors the
  repository layout, `$OPLM_DATA/hf/wwpdb/2026-10-12/` (the plan's
  `$OPLM_DATA/fold/pdb-<snapshot>` path was replaced by the repository layout below).

## Inputs

| input | value |
|---|---|
| Archive mirror | `rsync.rcsb.org::ftp_data/structures/divided/mmCIF/` (port 33444), `rsync -rlpt -v -z --delete` |
| rsync wall time | 13:10:08 to 16:48:52 UTC (3 h 39 min, 7.0 MB/s) |
| Snapshot date | 2026-10-12 (`pdb/snapshot-date.txt`) |
| Archive size, files | 92.3 GB (87 GiB on disk), 260,626 `.cif.gz` |
| CCD | `components.cif.gz` from files.wwpdb.org, sha256 `405da7e5d06527d45292db9400378d9788c024830cad4e3b43890b6c6a0dd4c3`, 51,389 components |

## Held-out target lists (`$OPLM_DATA/pdb/targets/`)

Both lists are kept with their sources (`targets/sources/`, URLs and sha256 in
`sources/README.md` and `SHA256SUMS`) and are copied into the published folder's `targets/`.

**CAMEO, 2021-10-01 .. 2022-09-30 (`cameo.txt`).** The CAMEO archive
(`archive.cameo3d.org`, where `www.cameo3d.org` now redirects) answered HTTP 503 for every
URL on 2026-10-12, so the weekly lists were assembled from two copies of CAMEO's own
one-year JSON export (`/modeling/targets/1-year/ajax/?to_date=...`):

1. `to_date=2022-01-15`, Wayback Machine capture of 2023-08-07T16:14:58Z (the only archived
   capture of that endpoint).
2. `to_date=2022-12-31`, mirrored in `github.com/Jianxinnn/Biodataset4AI`
   (`Protein/cameo/2022/raw/`, generated 2026-04-12 from the archive endpoint).

The two sources agree exactly on their overlap weeks (2022-01-08 and 2022-01-15, 31
targets). Two independently published subsets are fully contained in source 2: EigenFold's
183 targets (Aug–Oct 2022) and ManyFold's 143 targets (Mar–May 2022). Result: 755 targets
(747 distinct entries) over 52 weekly releases, 2021-10-02 .. 2022-09-24, one
`PDBID_CHAIN` per line. Chain ids are author chain ids (`auth_asym_id`), which the index
stores in `auth_chain_id`; `chain_id` is the label id.

**CASP14 (`casp14.txt`, `casp14_targets.json`).** The Prediction Center's target list and
domain-definition pages annotate PDB codes for only 29 of the 84 regular targets, so the
official target sequences (`download_area/CASP14/sequences/casp14.seq.txt`) were searched
against the PDB with RCSB's sequence service (MMseqs2, e-value 0.1). A target maps to every
entry with ≥ 95 % identity over ≥ 80 % of the target length (`seq-full`); that gives 61
targets, 62 with either an official code or a hit. Entries matching at 100 % identity over
≥ 50 residues but below 80 % coverage (`seq-partial`, shared domains) are also excluded from
training but are not evaluation targets. `casp14.txt` is the union, 133 PDB ids;
`casp14_targets.json` keeps the per-target mapping by category for M2c. Eighteen targets
have no PDB entry: T1047s1 T1047s2 T1050 T1066s1 T1066s2 T1067 T1069s1 T1069s2 T1073 T1077
T1079 T1083 T1084 T1085 T1086 T1087 T1088 T1098.

Held-out targets missing from the snapshot (`manifest["heldout_missing"]`): CAMEO 3 (7don,
7uw7, 8ahp; 744 of 747 entries are in the `cameo` split), CASP14 2 (9a13, 9a8v). Three CASP14
entries (7mhu, 7rej, 7w6b) are also CAMEO targets and sit in the `cameo` split because the
first held-out list wins, so the `casp14` split holds 128 entries; `targets/casp14.json` still
lists all 131 present entries, and the M2c evaluation reads the target files rather than the
split column alone.

## Preprocessing

Smoke test first: 2,636 entries from the already-synced `a0`–`a9` directories, same
options, 8 s wall time, 2,567 accepted, batches load. Full run:

```bash
oplm fold preprocess --archive $OPLM_DATA/pdb/mmCIF --ccd $OPLM_DATA/pdb/components.cif.gz \
    --out $OPLM_DATA/hf/wwpdb/2026-10-12 \
    --heldout cameo=$OPLM_DATA/pdb/targets/cameo.txt --heldout casp14=$OPLM_DATA/pdb/targets/casp14.txt \
    --workers 28
```

| result | value |
|---|---|
| Wall time | 12 min 47 s (16:49:27 to 17:02:14 UTC), 28 workers, about 340 entries/s |
| inputs / accepted | 260,626 / 254,424 |
| rejected | 6,202: no_protein_chains 5,265, resolution 928 (worse than 9.0 Å), no_resolved_chain 9 |
| dropped entities | ligand 495,831, water 193,831, nucleic_acid 38,942, short_chain 1,186, other_polymer 156 |
| chains | 1,080,220 index rows |
| splits | entries: train 176,486, valid 64, post_cutoff 75,763, excluded_heldout_sequence 1,239, cameo 744, casp14 128 (chain rows: train 602,785, post_cutoff 470,688, excluded_heldout_sequence 4,117, cameo 2,047, casp14 519, valid 64) |
| shards | 82 (3.4 MB to 516 MB each, 38.75 GB of shard rows) |
| Dataset size | 37 GiB on disk (`du -sh`) |

Log: `preprocess.log`; manifest: `manifest.json` (this directory). Rejection details were
checked for local paths before publishing: none of the 6,202 `detail` strings contains a path
(the only non-empty details are resolution values such as `10.0 > 9.0`).

## Local batch load

`.venv/bin/python` on the staged folder, `DataLoader(batch_size=8, num_workers=4)`, crop 384,
seed 0:

```
train entries 175065 valid chains 64
20 batches of 8 in 0.8 s; torch.Size([8, 384]) torch.Size([8, 9216, 3]) torch.Size([8, 9216, 3])
```

`StructureDataset` counts entries with at least one eligible chain (≤ 1024 residues, ≥ 4
resolved), so its length is slightly below the manifest's `train` entry count.

## Publication

Repository `brineylab/oplmfold-training-data` (public, CC0 root card), layout one folder per
source and version (`wwpdb/2026-10-12/`), one tag per publish; see the repository README.

Uploaded 2026-10-12 with `hf upload-large-folder` (107 files, 38.9 GB, 5 min 20 s, no errors)
and tagged; details in `publish.txt`.

| | |
|---|---|
| Repository | https://huggingface.co/datasets/brineylab/oplmfold-training-data |
| Folder | `wwpdb/2026-10-12/` |
| Tag | `wwpdb-2026-10-12` |
| Revision | `0fad20d41cbf7f7e1a8ddea0ab098443bbec3172` |

A first attempt failed with HTTP 403 on every LFS preupload because the fine-grained token had
no scope on the `brineylab` organization; the same command was rerun once the token had org
write access (the upload state in `.cache/huggingface/` made it resumable).

## Clean-environment load

Fresh `uv venv --python 3.11` with `oplm[fold] @ git+https://github.com/briney/oplm@e0ea2fd…`
(no checkout of this repository on its path), then
`hf download brineylab/oplmfold-training-data --repo-type dataset --revision wwpdb-2026-10-12
--include "wwpdb/2026-10-12/*"` (37 GiB, 82 shards) and a batch of 4 (`clean-env.log`):

```
175065 ['2ki4_A', '2ewc_D', '2evi_C', '2jd3_B'] torch.Size([4, 384]) 5067
```

The entry count matches the local run and the batch has finite shapes.

## Deviations from the plan

- Output path `hf/wwpdb/<snapshot>` instead of `fold/pdb-<snapshot>`, because
  `hf upload-large-folder` has no target-path option and the local folder must mirror the
  repository layout.
- CAMEO lists came from archived copies of CAMEO's export (the site is down), cross-checked
  as described above, rather than a live export.
- CASP14 ids came from the official pages plus a sequence search, rather than the pages alone.
- A smoke run on 2,636 entries preceded the full run.
- The clean-environment venv and download live under `$OPLM_DATA/tmp/` instead of `/tmp`,
  which is a 62 GB RAM-backed tmpfs on this machine.

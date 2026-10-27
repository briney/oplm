# Data manifest

Where and when each dataset used by oplm was obtained, and how it was processed. Written to
support the methods section: every value is copied from the build artifact named beside it.

## Structure evaluation sets (`structure-evals/2026-10-12`)

Built 2026-10-27 with `oplm fold eval-sets` at commit `c0b5f08` (oplm 0.4.1.dev180+gc0b5f08aa;
MMseqs2 18.8cc5c, installed from bioconda). Output: `$OPLM_DATA/hf/structure-evals/2026-10-12/`,
with a `README.md`, `summary.json` (per-stage counts) and `manifest.parquet` (one row per
structure, with its sha256). Design and rationale:
`docs/superpowers/specs/2026-10-27-structure-eval-sets-design.md`.

### Sources

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

### Processing

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
   categorical Jacobian (L × 20 single-substitution forward passes, centered, symmetrized,
   APC-corrected), reported as precision at L, L/2 and L/5 over long-range pairs.

### Counts

| set | candidates | removed (reason: count) | exported |
|---|---|---|---|
| `casp15_16` | 204 domain files, 122 distinct domains | duplicates: 82 (CASP16 `T2xxx`/`T0xxx` copies and higher versions); length: 12; contacts: 9 | 101 |
| `cameo` | 755 | missing: 3; skipped: 0; quality: 246; length: 54; modeled: 13; contacts: 4 | 435 |
| `pdb40` | 49,986 clusters | no_candidate: 19,298; exhausted: 917; max_rounds: 342 | 29,429 |
| `cameo_256`, `pdb40_2048`, `pdb40_256` | | | 256, 2,048, 256 |

CASP15 has 45 domain files (no duplicates); CASP16 has 159, of which 82 are duplicates, so
45 + 77 = 122 distinct domains. CAMEO `skipped` counts listed chains with fewer than three
resolved Cα atoms; `quality` is 46 NMR structures and 200 X-ray structures above 2.5 Å. The `pdb40` removed counts are clusters and sum with the 29,429
exported to 49,986. `no_candidate` clusters had no chain passing the quality filter, length,
standard-residue and 70%-resolved screen; `exhausted` clusters had candidates that all failed the
curation rules; `max_rounds` clusters still had untried candidates after 5 attempts. The
build also counted individual candidates rejected by the curation rules while searching clusters
(`contacts`: 3,940; `modeled`: 6). These are candidate-level counts, not clusters; don't add
them to the removed total.

Overlap: 202 of the 435 CAMEO chains are also PDB40 representatives, and all 435 fall in a 40%
sequence cluster that has a PDB40 representative (every curated CAMEO chain is itself a valid
candidate in its cluster). CAMEO is therefore not held out from PDB40. Both counts are reported;
nothing is removed.

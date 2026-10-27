# Structure eval sets: sourcing and curation plan

Status: plan, 2026-10-27; filtering decisions approved. Nothing has been built yet. Counts marked
*measured* come from the downloaded sources and the fold dataset; the rest are estimates.

## Goal

One versioned folder with four sets, all built from pinned sources and scored with unsupervised
contact precision@L (the `structure` eval type: categorical Jacobian, virtual Cβ < 8 Å, sequence
separation ≥ 24):

| set | role | size | cadence |
|---|---|---|---|
| `casp15_16` | hard targets | 101 domains (*measured*) | every 5k steps |
| `cameo_256` | recent PDB, same chains as the fold head's held-out set | 256 of ~420 | every 5k steps |
| `pdb40_256`, `pdb40_2048` | representative PDB, the decision metric | 256 / 2048 | 5k / 50k steps |
| `pdb40` | full set, post-training eval only | ~29k | offline |

## Convention: full sequence in, unresolved residues unscored

This follows the released ESM-C code (`esm/utils/structure/protein_chain.py`,
`esm/utils/structure/metrics.py`). The model sees the full deposited sequence. Residues without
coordinates stay in the sequence, and any pair involving one is unknown: it is left out of the
top-L ranking rather than scored as a non-contact. Nothing is dropped for having a gap, so the
sets keep disordered loops and discontinuous CASP domains, with no tilt toward well-ordered
proteins.

This needs two changes to the current eval, made as part of this work:

1. **Loader** (`oplm.data.structure.loader`). Today the sequence is built from residues that have
   coordinates, so an unmodeled loop is spliced out. New behavior: when a file has SEQRES records,
   the sequence is the SEQRES sequence, and each residue is placed at index `resSeq − 1`. Residues
   absent from the coordinates become NaN rows; a missing N, CA or C atom is NaN within its row,
   as today, so under the default Cβ scoring such a residue has no position and goes unscored.
   Placement is used
   only when every residue fits: inside the sequence, once, with no insertion code, and the same
   residue SEQRES has at that position. Otherwise, as for files without SEQRES, the loader keeps
   today's behavior. That way real PDB files numbered by author convention, and existing sets such
   as your CASP14, still load. The builder's round-trip validation guarantees that its own files
   take the placement path.
2. **Metric** (`oplm.eval.metrics`). Today a pair with a NaN coordinate is labeled a non-contact
   (`compute_contact_map`), so predictions on unresolved residues would count as false positives.
   New behavior: such pairs are unknown and excluded before ranking. L is the number of residues
   with coordinates; ESM uses the full sequence length, which only differs for chains with
   unresolved residues. With this definition, a CASP domain gets L = domain length. Existing
   in-training numbers shift slightly where structures had missing backbone atoms.

The builder numbers residues by sequence position (1..N, no insertion codes), so the loader can
place them without an alignment.

## Shared curation rules (every exported structure)

1. One protein chain, one model, written as PDB with chain id `A`: SEQRES holds the full sequence
   (entity polymer sequence, or CASP target sequence); ATOM records only; no hydrogens;
   highest-occupancy altloc; residues numbered by sequence position.
2. Modified residues renamed to their CCD parent (the fold parser's `protein_parent`). A residue
   with no parent (UNK, D-amino acids) anywhere in the sequence disqualifies the chain.
3. At least 70% of residues have N, CA and C (PDB40, CAMEO), so a mostly disordered chain isn't
   scored on a small ordered core. Residues with an incomplete backbone count as unmodeled.
4. Full-sequence length within the set's bounds. The categorical Jacobian costs scale with the
   input length squared, so the bounds limit cost.
5. At least L/5 long-range contacts (Cβ < 8 Å, |i − j| ≥ 24 by sequence position), where L is the
   number of modeled residues. This drops single helices, coiled coils and extended chains whose
   contacts are all with partner chains.
6. File name is a stable id (`casp16_T1206-D1.pdb`, `7a59_B.pdb`); the loader sorts by name.

The values behind rules 3–5 are stored per structure in the manifest, so thresholds can change
without re-curating.

**Quality filter (CAMEO, PDB40):** X-ray ≤ 2.5 Å or cryo-EM ≤ 3.0 Å; no NMR. Keeping cryo-EM keeps
large complexes and membrane proteins in PDB40. The method is recorded per chain.

## CASP: CASP15 + CASP16 official domains

Sources: the Prediction Center's domain-trimmed target files (the coordinates CASP assessed, cut at
the official domain boundaries) and the official target sequences.

| file (under `predictioncenter.org/download_area/`) | sha256 |
|---|---|
| `CASP15/targets/casp15.targets.TS-domains.public_12.20.2022.tar.gz` | `ac159c284a1506b460143ef6e22832272c5da1b4078fdb5fec8292b373b6239b` |
| `CASP16/targets/casp16.targets_monomer_trimmed2domains.tgz` | `30e6bf992baed08695ebebe481711adea867b483015d30e18d170a0eee66ceb9` |
| `CASP15/sequences/casp15.seq.txt` | `84dd04c1fda2a550af73e77ad25be94da23f16bd7ff6632f2c22c1978b9ee0a5` |
| `CASP16/sequences/casp16.T1.seq.txt` | `77d0b5c4ffda52ca1be7563a92eabae7ca59a3aab814e79a377ddd962783e286` |

Also save each edition's domain summary page (`/casp15/domains_summary.cgi`, and the CASP16
equivalent) for FM/TBM labels, so results can be split by difficulty later.

Each domain becomes one structure: SEQRES is the full target sequence, and only that domain's
residues have coordinates. The model sees the whole target and is scored inside the domain, which
is how CASP assesses domains. Domain files are numbered in target coordinates: the residue name
matched the target sequence at every position in all 122 distinct domains (*measured*).

The CASP16 tarball has three kinds of duplicates:

- Phase-2 copies named `T2xxx`, with coordinates byte-identical to the `T1xxx` domain.
- Monomer files that name target `T1xxx` as `T0xxx`.
- Versioned targets (`T1228v1`, `T1228v2`), which are alternative versions of the same target.

The builder maps `T0` and `T2` to `T1`, keeps the lowest version, and takes one file per
(target, domain). File REMARK lines aren't a reliable target id (some read `TARGET` or `USER`).
Every remaining target has a sequence in `casp16.T1.seq.txt`.

Filter: target length 50–1024 and the contact floor. The 70% modeled rule doesn't apply, because
a domain is deliberately a fraction of its target. *Measured:*

- CASP15: 45 → 36 domains.
- CASP16: 159 files → 77 distinct domains → 65 pass.
- Total 101 domains, of which 29 are gapped or discontinuous, so a gap-free rule would have
  dropped them.

All CASP domains are used every 5k steps.

**CASP14.** The public domain tarball
(`casp14.targets.T-dom.public_11.29.2020.tar.gz`, sha256 `4cff43c2…07ea718`) covers only 4
targets (8 usable domains); most CASP14 targets were never released as domain files. Keep your
existing CASP14 set as `casp14` for continuity. It loads under the old path because it has no
SEQRES. Rebuilding CASP14 from PDB entries plus the official boundaries is possible later, at
about a day's work for perhaps 40–60 domains.

## CAMEO

Source: the archived list already used for the fold head, `$OPLM_DATA/pdb/targets/cameo.txt`.
It has 755 chains from weekly releases 2021-10-02 .. 2022-09-24; CAMEO's site was down in
October 2026, and the provenance is in `docs/fold/m2a/README.md`. Sequences, coordinates and
resolved masks come from the fold dataset (wwPDB snapshot 2026-10-12).

*Measured:* 752 present → 506 pass the quality filter (all X-ray; 46 NMR dropped) → 452 within
50–512 residues → 439 with at least 70% modeled. The contact floor will remove a few more,
leaving roughly 420.

Outputs: `cameo/` (all survivors) and `cameo_256/` (a seed-42 subset of it). These are the same
chains as the fold head's held-out CAMEO set, so LM contact precision and fold-head scores can
later be compared chain by chain.

## PDB40: full set and subsets

Source: the fold dataset (wwPDB snapshot 2026-10-12). `index.parquet` has 1,080,220 chains with
entity sequence, method, resolution, release date and resolved count. The shards hold per-residue
resolved masks and coordinates.

1. **Cluster** the 175,496 distinct protein entity sequences of length ≥ 30 (*measured*), all
   methods, with
   `mmseqs easy-cluster --min-seq-id 0.4 -c 0.8 --cov-mode 0 -s 7.5`.
   Clustering every chain, not only the quality-passing ones, makes clusters reflect whole PDB
   families.
2. **Pick one representative per cluster.** Candidates are chains of member sequences that pass
   the quality filter, with length 50–1024. Sort them by resolution, then modeled fraction, then
   id. The first candidate that passes the shared rules wins. A cluster with no passing candidate
   is dropped and counted.
3. **Outputs:**
   - `pdb40/`: the representatives, i.e. the full post-training set.
   - `pdb40_2048/` and `pdb40_256/`: the first 2048 and 256 chains of a seed-42 permutation of
     the `pdb40/` chains with length ≤ 512. The 256 set is contained in the 2048 set.

Estimate: RCSB's weekly 40% cluster file (2026-10-18) has 29,610 clusters with a member passing
quality, length and the 70% rule, so expect about 29k representatives. Our clustering sets the
real number.

Why not use RCSB's file: it's regenerated weekly and not archived, it clusters about 1.06M
computed models alongside experimental entities (52,698 of its 324,268 clusters contain an
experimental entity), and it doesn't match our snapshot. It can't serve as a cross-check either:
the fold index stores a 0-based polymer-entity index, not the mmCIF entity id that RCSB's file is
keyed on.

PDB40 overlaps the fold head's training data, so it is for LM evaluation only. Fold-head evals
stay on CAMEO and CASP.

## Validation (part of the build; any failure fails the build)

1. Every exported file round-trips through the updated `load_structures`: exactly one structure,
   sequence equal to the manifest, NaN rows exactly at the manifest's unmodeled positions.
2. Long-range contact counts recomputed with the eval's own (updated) contact code match the
   manifest.
3. Overlap report: chains and clusters shared between sets. CAMEO and CASP targets can also be
   PDB40 representatives; this is reported, not removed.
4. Counts after every filter stage are written to the README.
5. First real use is the 0.0016 run and the `decay200k-50` branches. Expect P@L ordered
   PDB40 > CAMEO > CASP.

## Output layout and distribution

```
$OPLM_DATA/hf/structure-evals/2026-10-12/
  README.md              provenance, thresholds, per-stage counts, tool versions
  manifest.parquet       one row per exported structure: set, file, source id, chain, entity,
                         cluster, method, resolution, release date, sequence length, modeled
                         count, long-range contacts, CASP category, sha256
  summary.json           the README's counts and thresholds, machine-readable
  sources/               CASP tarballs, sequence files and domain summary pages, CAMEO list copy,
                         MMseqs2 FASTA and cluster TSV
  casp15_16/  cameo/  cameo_256/  pdb40/  pdb40_2048/  pdb40_256/
```

Publish as an HF dataset (`brineylab/oplm-structure-evals`, tag `2026-10-12`), the same way as the
fold dataset, then pull it to the cluster.

## In-training config

```yaml
data:
  eval:
    pdb40:      {path: .../pdb40_256,  type: structure, every: {steps: 5_000}}
    pdb40_2048: {path: .../pdb40_2048, type: structure, every: {steps: 50_000}}
    cameo:      {path: .../cameo_256,  type: structure, every: {steps: 5_000}}
    casp:       {path: .../casp15_16,  type: structure, every: {steps: 5_000}}
    casp14:     {path: <existing CASP14 set>, type: structure, every: {steps: 5_000}}
```

The subsets are materialized, so `categorical_jacobian_sample_size` isn't needed. About 610–650
chains per 5k-step round is roughly 2–3% of training compute: CASP inputs are full targets up to
1024 residues, which costs more per domain than a PDB40 chain. The 2048 set every 50k steps adds
roughly 0.7%. The full PDB40 post-training eval (about 29k chains up to 1024 residues) is roughly
1 h at 170M and 1.5–2 days at 6B on 32 GPUs.

## Implementation

- Builder: one command next to `oplm fold preprocess` (e.g. `oplm fold eval-sets`), reusing the
  fold data layer: index, shard reader, CCD parent table.
- Loader and metric changes as described above, each with tests: SEQRES placement, NaN rows for
  unmodeled residues, the name-mismatch error, old files without SEQRES unchanged, unknown pairs
  excluded from ranking, L = modeled count.
- Builder tests: parent renaming, the modeled-fraction floor, the contact floor, nested subsets,
  CASP domain placement, loader round-trip.
- External dependency: the MMseqs2 binary (bioconda or a static release), with its version
  recorded. It is not a Python dependency.
- Runtime: clustering 175k sequences takes roughly 10–30 min; export takes minutes. Building and
  testing is about 1.5 days.

## Decisions (approved 2026-10-27)

1. Quality filter: X-ray ≤ 2.5 Å or cryo-EM ≤ 3.0 Å, no NMR.
2. ESM convention: full sequence in, unresolved residues unscored (replaces the earlier gap-free
   rule).
3. At least 70% modeled (PDB40, CAMEO); contact floor of L/5 long-range contacts, with L = modeled
   residues.
4. CASP14: keep the existing set for continuity; CASP15+16 is the primary hard set.
5. The builder lives under `oplm fold`.

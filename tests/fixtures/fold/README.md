# Fold data fixtures

Real wwPDB entries (`mmcif/`: mmCIF, gzipped, downloaded from https://files.rcsb.org/download/ on
2026-10-11; wwPDB data is CC0 1.0) and a trimmed Chemical Component Dictionary. Checksums
are pinned in `docs/superpowers/plans/2026-10-11-fold-m2a-data.md` Task 3.

| File | Why it is here |
| --- | --- |
| `mmcif/1crn.cif.gz` | Crambin, 46-residue X-ray monomer at 1.5 Å, every residue resolved. |
| `mmcif/1ejg.cif.gz` | Crambin at 0.54 Å: altlocs (381 atoms), hydrogens, microheterogeneity (`PRO,SER` at 22, `LEU,ILE` at 25). Sequence identical to 1crn except residue 25 (`L`, the first alternative, vs `I`). |
| `mmcif/4hhb.cif.gz` | Hemoglobin: two entities × two copies (symmetry ids), hemes and phosphate (ligands dropped), waters dropped. |
| `mmcif/2ds7.cif.gz` | Se-Met monomer with residues 1–11 unmodeled (tokens with masked coordinates); `MSE` → `MET` with `SE` → `SD`. |
| `mmcif/1jb6.cif.gz` | Homodimer with `MSE` and `NLE` (norleucine → `LEU`, two atoms unmatched); chain B unmodeled at 1–2 and 31–33. |
| `mmcif/1l2y.cif.gz` | Trp-cage, solution NMR, 38 models (first model only, no resolution). |
| `ccd_subset.cif` | `_chem_comp` category of 45 components from `components.cif.gz` (wwPDB CCD, CC0 1.0, 2026-10-11): the 20 amino acids, `UNK`, modified residues with parents, multi-parent and D-amino-acid cases, nucleotides, water and ligands. |

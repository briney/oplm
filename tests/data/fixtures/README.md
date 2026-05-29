# `oplm.data` test fixtures

Real data fixtures for the data-tooling test suite. Drop files in the
subdirectories below; tests `pytest.skip` when a required fixture is absent.

- `structures/<id>.pdb` — a small real structure (short chain) for the
  structure-loader tests (Phase 6). Provided: `structures/1CRN.pdb` (crambin,
  single chain, 46 residues).
- `variant/<assay>.csv` — a tiny real ProteinGym substitution-assay CSV for the
  variant-loader tests (Phase 7). A real ProteinGym CSV has `mutant`,
  `DMS_score`, and `mutated_sequence` columns; the test reconstructs the
  wild-type from `mutated_sequence` (no separate reference file needed). Keep it
  small (a head slice of an assay is fine). The non-fixture variant tests run
  unconditionally on temporary CSVs built from a real reference sequence, so this
  fixture is optional — its test `pytest.skip`s when the directory is absent.
- `downstream/` — tiny per-residue and sequence-level task fixtures for the
  downstream-loader tests (Phase 8).

The session-scoped sequence parquet fixture (Phase 10) is generated at test time
from the existing `tests/fixtures/training/test_sequences.parquet`.

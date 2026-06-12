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
  small (a head slice of an assay is fine). Provided:
  `variant/A0A2Z5U3Z0_9INFA_Wu_2014.csv` (influenza HA, Wu 2014; head slice).
  The `test_real_proteingym_fixture` test loads this fixture directly.
- `downstream/` — *(no dropped files required)*. The downstream-loader tests
  (Phase 8) build their own per-residue and sequence-level fixtures at test time
  from real protein sequences (the shared `real_records` fixture); only the task
  *labels* are derived deterministically (task targets, not biological data).

The session-scoped sequence parquet fixture (Phase 10) is generated at test time
from the existing `tests/fixtures/training/test_sequences.parquet`.

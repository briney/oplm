# `oplm.data` test fixtures

Real data fixtures for the data-tooling test suite. Drop files in the
subdirectories below; tests `pytest.skip` when a required fixture is absent.

- `structures/<id>.pdb` — a small real structure (short chain) for the
  structure-loader tests (Phase 6). Provided: `structures/1CRN.pdb` (crambin,
  single chain, 46 residues).
- `variant/<assay>.csv` — a tiny real ProteinGym assay CSV (`mutant`,
  `DMS_score` columns) for the variant-loader tests (Phase 7).
- `downstream/` — tiny per-residue and sequence-level task fixtures for the
  downstream-loader tests (Phase 8).

The session-scoped sequence parquet fixture (Phase 10) is generated at test time
from the existing `tests/fixtures/training/test_sequences.parquet`.

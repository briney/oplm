# Notes

Running notes and follow-ups that don't belong in `TODOS.md` (the phased plan)
or `AGENTS.md` (contributor instructions).

## Pending test fixtures

_None currently._

### ProteinGym variant assay (Phase 7) — resolved

A real ProteinGym substitution-assay CSV (a head slice of the influenza-HA
Wu 2014 assay) is committed at
`tests/data/fixtures/variant/A0A2Z5U3Z0_9INFA_Wu_2014.csv`. The CSV carries the
`mutant`, `DMS_score`, and `mutated_sequence` columns;
`tests/data/variant/test_loader.py::test_real_proteingym_fixture` now loads it
directly (reconstructing the wild-type by reverting a single-substitution row's
`mutated_sequence`), and no longer `pytest.skip`s.

See `tests/data/fixtures/README.md` for the full fixture-directory layout.

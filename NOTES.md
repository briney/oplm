# Notes

Running notes and follow-ups that don't belong in `TODOS.md` (the phased plan)
or `AGENTS.md` (contributor instructions).

## Pending test fixtures

### ProteinGym variant assay (Phase 7)

The variant-loader tests need a small **real ProteinGym substitution-assay CSV**
dropped at:

```
tests/data/fixtures/variant/<assay>.csv
```

Until it is present, `tests/data/variant/test_loader.py::test_real_proteingym_fixture`
`pytest.skip`s. (The rest of that file's contract tests run unconditionally on
temporary CSVs built from a real human-ubiquitin reference sequence, so coverage
is not blocked — only the genuine-ProteinGym smoke test is.)

Requirements for the fixture:

- A real ProteinGym substitution CSV with columns `mutant`, `DMS_score`, and
  `mutated_sequence`. A **head slice** of a full assay (a few dozen rows) is fine
  and keeps the repo small.
- No separate wild-type reference file is needed: the test reconstructs the
  wild-type by reverting a single-substitution row's `mutated_sequence`.
- Source: ProteinGym substitution benchmark
  (<https://proteingym.org>, DMS substitutions). Pick a short-protein assay so
  the sequences stay small.

See `tests/data/fixtures/README.md` for the full fixture-directory layout.

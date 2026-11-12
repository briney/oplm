# Structure prediction head

How the fold head works and what has been measured: [../FOLD.md](../FOLD.md). Configs are in
`configs/fold/`.

| Choice | Where the value lives | Evidence |
|---|---|---|
| Language-model base for the fold head | `configs/fold/` | to backfill (M2 decisions) |
| PDB-only, single-chain training data | [../FOLD.md](../FOLD.md) | to backfill (M2a) |
| Pair width (128 vs 256) | `configs/fold/proof-w128.yaml`, `configs/fold/proof-w256.yaml` | to backfill (proof runs, #48) |
| Inference memory and maximum length | [../FOLD.md](../FOLD.md) | to backfill (`../fold/inference-memory/`) |

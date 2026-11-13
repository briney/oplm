# Structure prediction head

How the fold head works and what has been measured: [../FOLD.md](../FOLD.md). Configs are in
`configs/fold/`.

> **Pending:** the pair-width comparison; the two proof runs (jobs 35927 and 35928, 50k steps) are still training (waiting on #48).

| Choice | Where the value lives | Evidence |
|---|---|---|
| Language-model base for the fold head | `fold.lm` in `configs/fold/proof-w128.yaml` | A stand-in, not a comparison: frozen ESM-C 600M, because no OPLM checkpoint exists yet. The rerun on an OPLM base is #49 |
| PDB-only, single-chain training data | [../FOLD.md](../FOLD.md), *Milestone 2a: training data* | The scope of the pipeline proof, not a tested choice. Distillation data and multi-chain examples are deferred to #49 |
| Pair width (128 vs 256) | `configs/fold/proof-w128.yaml`, `configs/fold/proof-w256.yaml` | Cost per step and memory at each width: [../fold/m2/README.md](../fold/m2/README.md). Accuracy: pending #48 |
| Inference memory and maximum length | [../FOLD.md](../FOLD.md), *Inference memory and maximum length* | Same section, *Current* and *Why and what was ruled out*; raw sweeps in `docs/fold/inference-memory/` |

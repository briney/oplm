# Data

How the data was built and how to download it: [../data.md](../data.md). Defaults are in
`src/oplm/configs/data/base.yaml`.

| Choice | Where the value lives | Evidence |
|---|---|---|
| Training set composition and clustering thresholds | `src/oplm/configs/data/base.yaml` | to backfill ([../data.md](../data.md), *How the training set was built*) |
| uniref70 share of decay-phase tokens | production config, not committed yet | [decay-uniref70-share](evidence/decay-uniref70-share/README.md) |
| Stable-phase mix between sources | `src/oplm/configs/data/base.yaml` | to backfill |
| Antibody fraction in the decay phase | production config, not committed yet; proxy row fractions are in the record | [antibody-fraction](evidence/antibody-fraction/README.md) |
| Phase-2 sampling | `src/oplm/configs/data/base.yaml` | to backfill (#58 phase-2 and overfit tests) |
| Global shuffle of the training shards | [../data.md](../data.md) | to backfill (intra-shard similarity test, loader fixes PR #38) |

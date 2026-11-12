# Data

How the data was built and how to download it: [../data.md](../data.md). Defaults are in
`src/oplm/configs/data/base.yaml`.

| Choice | Where the value lives | Evidence |
|---|---|---|
| Training set composition and clustering thresholds | [../data.md](../data.md), *Training data* | provenance, not an experiment: [../data.md](../data.md), *How the training set was built* |
| uniref70 share of decay-phase tokens | production config, not committed yet | [decay-uniref70-share](evidence/decay-uniref70-share/README.md) |
| Stable-phase mix between sources | `data.train` in `configs/scaling.yaml` (pending #58, see [../data.md](../data.md), *Training data*) | not varied in any run; [training-length-and-overfitting](evidence/training-length-and-overfitting/README.md) shows no overfitting at row fraction 0.05 |
| Antibody fraction in the decay phase | production config, not committed yet; proxy row fractions are in the record | [antibody-fraction](evidence/antibody-fraction/README.md) |
| Member sampling for uniref70 in the decay phase | production config, not committed yet (`sample_members` on the `data.train` source) | [member-sampling-check](evidence/member-sampling-check/README.md) (correctness only; effect on quality not isolated) |
| Global shuffle of the training shards | [../data.md](../data.md), *Shuffling* | [../data.md](../data.md), *Why the data is shuffled*, which cites the investigation at a pinned commit; numbers not yet extracted into a record |

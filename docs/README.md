# oplm docs

One line per doc. Start with the group that matches what you want to do.

## Use the model

- [inference.md](inference.md): load a model, get embeddings and logits, `oplm encode`, coming from ESM-C.

## Reproduce or modify training

- [reproduce.md](reproduce.md): the ordered path from data to a trained, evaluated model, including starting from a published checkpoint.
- [data.md](data.md): training and eval data, how they were built, dataset formats, tokenizer and masking.
- [mup-lr.md](mup-lr.md): μP, the learning-rate sweep, the coordinate check and the decay recipe.
- [training.md](training.md): launching, schedule, checkpointing and resume, published checkpoints, branching a decay.
- [slurm.md](slurm.md): turning a training config into Slurm jobs, requeue and arrays.
- [evaluation.md](evaluation.md): eval tasks and metrics, scheduling, the published structure eval sets.

## Reference

- [config.md](config.md): every `model.*`, `train.*`, `data.*` and `fold.*` field.
- [architecture.md](architecture.md): the OPLM model architecture.
- [FOLD.md](FOLD.md): developer architecture doc (fold work in progress).

## Contribute

- [testing.md](testing.md): running the test suite, end-to-end scenarios and cluster recovery drills.
- [superpowers/](superpowers/): dated design specs and plans, kept as design records.

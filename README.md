# OPLM

One open project, two models:

- **OPLM** is a protein language model. Give it protein sequences and it gives back per-residue
  and per-protein embeddings, plus amino-acid logits.
- **OPLMFold** predicts protein structure on top of OPLM.

Everything is open, as it is published: the weights, the exact training data and how it was
built, and the full training recipe.

## Install

```bash
pip install oplm
```

Python 3.11 or newer. That is all you need for embeddings. OPLMFold will ship as an install extra
once it is ready.

## Quickstart

### Embed sequences

```python
import torch
from oplm import LogitsConfig, OplmForMaskedLM

# Placeholder: this name does not work yet. Until checkpoints are published,
# point at a local model directory.
model = OplmForMaskedLM.from_pretrained("brineylab/oplm-170M").eval()

sequences = ["MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRIL", "MEEPQSDPSVEPPLSQ"]
with torch.no_grad():
    out = model.logits(sequences, LogitsConfig(return_embeddings=True))

out.embeddings  # (B, T, hidden_size) per residue, each sequence wrapped as <cls> ... <eos>
per_protein = torch.stack(
    [out.embeddings[i, 1 : 1 + len(s)].mean(0) for i, s in enumerate(sequences)]
)  # (B, hidden_size): the mean over each sequence's residues
```

`out.sequence_logits` holds the per-residue amino-acid logits. Batching, GPUs, pooling options and
the `transformers` `Auto*` classes are covered in [docs/inference.md](docs/inference.md).

**Coming from ESM-C?** The API mirrors it: the same `LogitsConfig` and `LogitsOutput`, and the same
33-token vocabulary. The differences are listed in
[docs/inference.md, *Coming from ESM-C?*](docs/inference.md#coming-from-esm-c).

### Predict a structure

**Coming soon.** OPLMFold is not usable yet. This section will hold the structure-prediction
quickstart when it ships.

## Models

### OPLM

**Coming soon.** Trained checkpoints will be listed here, with the Hugging Face Hub name for each.

> **Pending:** published OPLM checkpoints on the Hugging Face Hub (waiting on #62).

### OPLMFold

**Coming soon.**

## What you can do today, and what's coming

Today:

- Per-residue embeddings, per-protein embeddings and amino-acid logits from Python
  ([docs/inference.md](docs/inference.md)).
- `oplm encode`: per-residue embeddings for a list of sequences, written to a file
  ([docs/inference.md, *Command line*](docs/inference.md#command-line)).

Coming:

- Standalone evaluation of a saved checkpoint, #57
  ([docs/evaluation.md, *Offline evaluation*](docs/evaluation.md#offline-evaluation)).
- Structure prediction with OPLMFold.

## Training and reproducing

The configs and recipe are in this repo, and the exact training data and checkpoints will be
published too (some steps are still pending), so you can retrain OPLM yourself.
[docs/reproduce.md](docs/reproduce.md) walks through it in order, from data to evaluation.
Training at full scale needs a multi-node GPU cluster. Starting from a published checkpoint, for
example to try a different decay phase, is much cheaper and has its own section there. OPLMFold
training will get its own path later.

## Documentation

[docs/README.md](docs/README.md) lists every doc by what you want to do. The model itself is
described in [docs/architecture.md](docs/architecture.md). Contributors start with
[AGENTS.md](AGENTS.md).

## Citation

A paper is coming. Until then, cite the repository URL, https://github.com/briney/oplm, and the
commit you used.

## License

MIT. See [LICENSE](LICENSE).

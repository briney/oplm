# Inference: embeddings and logits

How to get per-residue logits, per-residue embeddings and per-protein embeddings out of an OPLM
model, from Python or from the command line. Nothing here needs the training extras: `pip install
oplm` is enough.

> **Pending:** published OPLM checkpoints on the Hugging Face Hub (waiting on #62).

Until then, the snippets below use `brineylab/oplm-170M` as a **placeholder** name. It does not
resolve today. Swap in a local directory written by `save_pretrained` (or a training checkpoint's
`hf/` export, see [training.md](training.md)) and everything else works unchanged.

For the model itself see [architecture.md](architecture.md). For how the tokenizer is built and
which IDs it uses see [data.md](data.md), *Tokenizer*.

## Load a model

`OplmForMaskedLM` is the main entry point. `from_pretrained` loads the weights and attaches the
tokenizer saved next to them, so the model accepts raw sequences:

```python
import torch
from oplm import OplmForMaskedLM

model = OplmForMaskedLM.from_pretrained("brineylab/oplm-170M").eval()
```

If no tokenizer files sit next to the weights, the `tokenizer` attribute stays `None` and `tokenize` and
`logits` raise an error telling you to attach one (`model.tokenizer = AutoTokenizer.from_pretrained(...)`).

Importing `oplm` also registers OPLM with `transformers`, so the standard `Auto*` classes work
(`AutoModel`, `AutoModelForMaskedLM`, `AutoModelForSequenceClassification`,
`AutoModelForTokenClassification`, `AutoTokenizer`):

```python
import oplm  # registers OPLM with the Auto* classes
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("brineylab/oplm-170M")
auto_model = AutoModelForMaskedLM.from_pretrained("brineylab/oplm-170M").eval()

batch = tokenizer(["MEEPQSDPSVEPPLSQ", "GAGTRWPVQ"], return_tensors="pt", padding=True)
with torch.no_grad():
    out = auto_model(**batch, output_hidden_states=True)
out.logits             # (B, T, 33)
len(out.hidden_states) # layers + 1: the embedding output, then one entry per block
```

Each saved model directory also bundles its modeling code, so a machine without `oplm` installed
can load it with `trust_remote_code=True`.

## Per-residue logits and embeddings

`logits(sequences, LogitsConfig(...))` on the model tokenizes, pads, runs the model and returns a
`LogitsOutput`:

```python
from oplm import LogitsConfig

sequences = [
    "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRIL",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK",
    "MEEPQSDPSVEPPLSQ",
]

with torch.no_grad():
    out = model.logits(sequences, LogitsConfig(sequence=True, return_embeddings=True))

out.sequence_logits   # (B, T, 33) per-residue amino-acid logits
out.embeddings        # (B, T, hidden_size) per-residue embeddings (after the final norm)
```

`B` is the number of sequences. `T` is the padded length: the longest sequence plus two, because
every sequence is wrapped as `<cls> ... <eos>`. Position 0 is `<cls>`, residue `i` is at position
`i + 1`, and shorter sequences are padded on the right. To get one sequence's residues back without
the special tokens or padding:

```python
residues = out.embeddings[2, 1 : 1 + len(sequences[2])]   # (len(sequences[2]), hidden_size)
```

The `LogitsConfig` fields are all booleans, and each unset field comes back as `None`:

| Field (default) | Fills in `LogitsOutput` | Shape |
| --- | --- | --- |
| `sequence` (`True`) | `sequence_logits` | `(B, T, 33)` |
| `return_embeddings` (`False`) | `embeddings`, the last hidden state | `(B, T, hidden_size)` |
| `return_hidden_states` (`False`) | `hidden_states`, a tuple: the embedding output, then one per layer | each `(B, T, hidden_size)` |
| `return_attentions` (`False`) | `attentions`, one per layer; forces the slower manual attention path | each `(B, heads, T, T)` |

The definitions live in `src/oplm/model/outputs.py`.

## Per-protein embeddings

For one fixed-size vector per sequence, average each sequence's residues from `out.embeddings`
(the previous section). This leaves out `<cls>`, `<eos>` and padding:

```python
per_protein = torch.stack(
    [out.embeddings[i, 1 : 1 + len(s)].mean(0) for i, s in enumerate(sequences)]
)  # (B, hidden_size)
```

Alternatively, run the backbone (`OplmModel`) and pool with `mean_pool`. It is mask-aware, so
padding does not dilute the average, but the mean includes `<cls>` and `<eos>` (see below):

```python
from oplm import OplmModel
from oplm.model import mean_pool

backbone = OplmModel.from_pretrained("brineylab/oplm-170M").eval()

batch = backbone.tokenize(sequences)   # BatchEncoding, already on the model's device
with torch.no_grad():
    hidden = backbone(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    ).last_hidden_state                                  # (B, T, hidden_size)

pooled = mean_pool(hidden, batch["attention_mask"])  # (B, hidden_size)
```

Loading the backbone from a masked-LM checkpoint prints a load report listing the `lm_head.*` weights as
`UNEXPECTED`. That is expected: the backbone has no head.

The `mean_pool` average includes the `<cls>` and `<eos>` positions, which the attention mask counts
as real tokens, so it differs slightly from the residue-only mean above.
`oplm.model` also exports `cls_pool`, which returns the `<cls>` vector (position 0) instead.

Pass the attention mask whenever you call the model yourself. Without it, pad tokens are treated
as real input.

## Batching, length and devices

- **Length limit.** The context window is 1024 positions, so a sequence can have at most 1022
  residues. Truncate on the way in with `model.tokenize(seqs, truncation=True, max_length=1024)`,
  or split long proteins yourself.
- **Batch by length.** Padding costs compute. Sort sequences by length and feed batches of similar
  length together, then put the results back in the original order.
- **Inference mode.** Call `.eval()` and wrap forward passes in `torch.no_grad()`, as above.
- **GPU and precision.** Move the model once and let `tokenize` follow it. `bfloat16` roughly halves
  memory on recent GPUs:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

fast = OplmForMaskedLM.from_pretrained("brineylab/oplm-170M", dtype=dtype).to(device).eval()
with torch.no_grad():
    out = fast.logits(sequences, LogitsConfig(return_embeddings=True))
out.embeddings.device   # the model's device; tokenize() moves the batch for you
```

- **Attention weights.** `return_attentions=True` materializes `(B, heads, T, T)` tensors per layer.
  Use small batches.

## Command line

`oplm encode` writes per-residue embeddings to a file, and `oplm info` describes an architecture.
`oplm --help` lists every command.

```bash
oplm encode MKWVTFISLLLLFSSAYS MLPGLALLLLAAWTARA --model ./my-oplm --preset 170M --output embeddings.pt
```

`--model` takes a local directory only: a `save_pretrained` directory, or a training checkpoint
directory (which holds `config.yaml` and an `hf/` export). It does not download from the Hub, so
fetch a published checkpoint first (for example with `huggingface-cli download`). The command also
needs to know the architecture, which it reads from `config.yaml` in a training checkpoint, or from
`--preset` / `--config` for a plain `save_pretrained` directory. When the directory holds a
HuggingFace export, the weights and config come from the export and the preset is only there to
satisfy the lookup.

The output file is a `torch.save` tensor of shape `(num_sequences, T, hidden_size)`, the same
padded per-residue embeddings as `out.embeddings` above, so read it back with `torch.load`:

```python
embeddings = torch.load("embeddings.pt")   # (num_sequences, T, hidden_size)
```

`oplm encode` writes embeddings only. For logits, per-protein vectors or anything else, use the
Python API.

```bash
oplm info --preset 170M
```

prints the parameter count, layer and head counts, hidden size and the enabled architecture
features for a preset or config, without loading any weights. The preset names are the files in
`src/oplm/configs/model/presets/`, and each preset only describes an architecture: it does not
download weights.

## Coming from ESM-C?

OPLM mirrors the ESM-C inference surface, so existing pipelines port with few changes:

- `LogitsConfig(sequence=..., return_embeddings=...)` is the same knob object, and
  `logits(...)` returns a `LogitsOutput`. Read `output.embeddings` as you would with ESM-C.
- The tokenizer has the same 33-token vocabulary and the same `<cls>`, `<pad>`, `<eos>` and `<mask>`
  special tokens.
- `model.logits(...)` and `model.tokenize(...)` take a `list[str]` of sequences directly, not
  pre-encoded protein objects. Per-residue logits are in `output.sequence_logits`.
- `model.encode(seqs)` exists for call-site parity, and, as in ESM-C, returns the padded
  `input_ids` and **not** embeddings, with no attention mask. For embeddings use `logits`
  with `return_embeddings=True`.
- Models are ordinary HuggingFace `PreTrainedModel`s, so `save_pretrained`, `push_to_hub` and
  `Auto*` work too.

## OPLMFold

Structure prediction on top of OPLM is coming and is not usable yet. Developers can read the
design in [FOLD.md](FOLD.md).

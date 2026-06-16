# OPLM

**Open Protein Language Model** — an encoder-only protein language model with a
HuggingFace-native, ESM-C-style API. Load a pretrained checkpoint, hand it a list
of sequences, and get back per-residue logits and embeddings.

> **Status:** Pre-alpha (`v0.0.1`). The model, tokenizer, inference API, and
> HuggingFace integration are stable; pretrained checkpoints and benchmark
> results are still landing.

---

## Highlights

- **ESM-C-style inference API** — `model.logits(sequences, LogitsConfig(...))`
  returns a structured `LogitsOutput` with `sequence_logits`, `embeddings`,
  `hidden_states`, and `attentions`. If you've used ESM-C, you already know it.
- **HuggingFace-native** — every model is a `PreTrainedModel`. Use
  `OplmForMaskedLM.from_pretrained(...)` or the `transformers` `Auto*` classes,
  load from the Hub, and `save_pretrained` / `push_to_hub` like any other model.
- **ESM-C-compatible tokenizer** — a 33-token `OplmTokenizerFast` with the same
  vocabulary and special tokens as ESM-C.
- **Fast attention** — built on PyTorch's `scaled_dot_product_attention` (a fused
  FlashAttention / memory-efficient kernel on CUDA, no separate `flash-attn`
  dependency), with a manual softmax path for returning attention weights.
- **Five sizes** — from a 5M-parameter ablation model up to 13B parameters.

---

## Installation

**Requirements:** Python ≥ 3.11 and PyTorch ≥ 2.10.

```bash
pip install oplm
```

Or from source:

```bash
git clone https://github.com/briney/oplm.git
cd oplm
pip install -e .
```

Inference needs no extras. The optional groups are for contributors:

```bash
pip install "oplm[train]"   # distributed training (Accelerate, W&B, datasets)
pip install "oplm[dev]"     # tests, linting, type checking
```

---

## Quick start

### Per-residue logits and embeddings

The primary entry point is `OplmForMaskedLM`. `from_pretrained` loads the weights
and attaches the matching tokenizer, so `.logits()` takes raw sequences directly:

```python
import torch
from oplm import OplmForMaskedLM, LogitsConfig

model = OplmForMaskedLM.from_pretrained("brineylab/oplm-170M").eval()

sequences = [
    "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRIL",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK",
]

with torch.no_grad():
    out = model.logits(
        sequences,
        LogitsConfig(sequence=True, return_embeddings=True),
    )

out.sequence_logits   # (B, T, 33) per-residue amino-acid logits
out.embeddings        # (B, T, hidden_size) per-residue embeddings
```

`B` is the batch size and `T` is the padded sequence length (each sequence is
wrapped with `<cls> … <eos>`).

### Per-protein embeddings

For a single fixed-size vector per sequence, run the backbone (`OplmModel`) and
mask-aware mean-pool over the residue dimension:

```python
import torch
from oplm import OplmModel
from oplm.model import mean_pool

model = OplmModel.from_pretrained("brineylab/oplm-170M").eval()

batch = model.tokenize(sequences)            # BatchEncoding on the model's device
with torch.no_grad():
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

per_residue = out.last_hidden_state          # (B, T, hidden_size)
per_protein = mean_pool(per_residue, batch["attention_mask"])  # (B, hidden_size)
```

`oplm.model` also exports `cls_pool` if you prefer the `<cls>` representation.

### Using the `transformers` Auto* API

Importing `oplm` registers the config, models, and tokenizer with `transformers`,
so the standard `Auto*` classes work too:

```python
import oplm  # registers OPLM with transformers' Auto* classes
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("brineylab/oplm-170M")
model = AutoModelForMaskedLM.from_pretrained("brineylab/oplm-170M").eval()

batch = tokenizer(sequences, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(**batch).logits   # (B, T, 33)
```

`AutoModel` returns the bare encoder, and `AutoModelForSequenceClassification` /
`AutoModelForTokenClassification` return the corresponding fine-tuning heads.

Each model repo also bundles its modeling code, so consumers who don't have
`oplm` installed can load it with `trust_remote_code=True`:

```python
model = AutoModelForMaskedLM.from_pretrained("brineylab/oplm-170M", trust_remote_code=True)
```

### Coming from ESM-C?

OPLM mirrors the ESM-C inference surface so existing pipelines port with minimal
changes:

- `LogitsConfig(sequence=..., return_embeddings=...)` is the same knob object.
- `model.logits(...)` returns a `LogitsOutput`; read `output.embeddings` exactly
  as you would with ESM-C.
- The tokenizer vocabulary (33 tokens, `<cls>`/`<pad>`/`<eos>`/`<mask>` specials)
  matches ESM-C.

The main difference: OPLM's `.logits()` and `.tokenize()` accept a `list[str]` of
sequences directly, rather than pre-encoded protein tensors. Per-residue logits
live in `output.sequence_logits`.

---

## Model sizes

OPLM ships eight size presets, selected on the command line with
`--preset <name>`. A preset is an **architecture recipe** (it sets the core
dimensions only); it does not download weights. To load pretrained weights, pass
a Hub id or a local checkpoint directory to `--model` (see below).

| Preset | Parameters | Layers | Hidden | Heads |
|--------|-----------:|-------:|-------:|------:|
| `50M`  |     ~50M    |     16 |    512 |     8 |
| `170M` |    ~170M    |     24 |    768 |    12 |
| `400M` |    ~400M    |     32 |   1024 |    16 |
| `800M` |    ~800M    |     40 |   1280 |    16 |
| `1B`   |    ~1.6B    |     50 |   1600 |    25 |
| `3B`   |    ~3.3B    |     64 |   2048 |    32 |
| `6B`   |      ~6B    |     80 |   2560 |    40 |
| `12B`  |    ~12.5B   |    100 |   3200 |    50 |

All sizes share the 33-token tokenizer and a 1024-position context window.

---

## Command line

`oplm` ships a small CLI; `oplm --help` lists every command.

### Encode sequences to a file

```bash
oplm encode MKWVTFISLLLLFSSAYS MLPGLALLLLAAWTARA \
  --model brineylab/oplm-170M \
  --output embeddings.pt
```

`--model` accepts a Hub id, a local HuggingFace export directory, or a training
checkpoint directory. The saved tensor holds the per-residue embeddings,
`(num_sequences, T, hidden_size)`.

### Inspect a model

```bash
oplm info --preset 400M
```

```
──────────────── OPLM Model Info ────────────────
                Architecture
 Parameters        412.3M (412,302,369)
 Hidden size       1024
 Layers            32
 Attention heads   16
 Head dim          64
 Intermediate size 2816
 FFN activation    swiglu
 ...
```

---

## Training

OPLM trains with HuggingFace Accelerate (FSDP, mixed precision, gradient
checkpointing) over parquet sequence datasets, with a built-in eval harness for
MLM metrics and structure-based contact prediction. Training defaults to **μP +
Muon**, so `train.lr` is the μP base LR and transfers across model size — set it
once and reuse it at every preset.

```bash
oplm train --preset 1B data.train=/data/uniref50/
```

See **[docs/TRAIN.md](docs/TRAIN.md)** for the full training guide, and
**[docs/MUP.md](docs/MUP.md)** for μP learning-rate transfer (tune `lr` once on a
small model, reuse it at scale; recover the conventional ESM-C recipe via
`vanilla_esm-c.yaml`).

---

## Configuration

Models and runs are configured through a layered system
(**defaults → preset → YAML → CLI overrides**) built on a HuggingFace
`PretrainedConfig`. Architecture toggles, optimizer/scheduler settings, and the
dataset schema are all set here.

See **[docs/CONFIG.md](docs/CONFIG.md)** for the field-by-field reference.

---

## Architecture

OPLM is a pre-norm, bidirectional encoder transformer with RoPE and QK-norm:
LayerNorm by default (RMSNorm available), SwiGLU feed-forward, untied
input/output embeddings, standard multi-head attention, depth-stable residual
scaling, and a BERT-style MLM head. A curated set of independently togglable
research features (Canon depthwise convolutions, partial-RoPE/NoPE, sandwich /
hybrid / post-SDPA norm) layers on top, each off by default.

For the complete specification, see
**[docs/MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md)**.

---

## Development

```bash
pip install -e ".[dev]"

pytest                  # run tests
pytest -m "not slow"    # skip slow tests
pytest --cov=oplm       # with coverage

ruff check src/         # lint
ruff format src/        # format
ty check src/           # type check (Astral's `ty`, not mypy)
```

Contributor and agent instructions live in [AGENTS.md](AGENTS.md).

---

## License

MIT — see [LICENSE](LICENSE).

# OPLM

**Open Protein Language Model** -- an encoder-only protein language model built for research, combining recent architectural innovations with a fully ablation-friendly design.

OPLM integrates ideas from [Proust](https://arxiv.org/abs/2602.01845) (grouped-query attention with shared K/V projections, cross-layer value residuals, depthwise convolutions) and [Attention Residuals](https://arxiv.org/abs/2603.15031) (learned depth-wise residual connections). Every novel feature is independently togglable via config, enabling clean ablation studies with zero memory overhead when features are disabled.

> **Status:** Pre-alpha (v0.0.1). The core architecture, training loop, and evaluation harness are functional. Benchmark dataset integrations are in progress.

---

## Features

- **Encoder-native architecture** -- fully bidirectional attention for masked language modeling on protein sequences
- **Grouped-query attention** with optional shared K/V projections, Q/K normalization, output gating, and partial RoPE
- **Cross-layer value residuals** and **learned attention residuals** for improved gradient flow
- **Bidirectional depthwise convolutions** at configurable positions with static or scheduled kernel sizes
- **SwiGLU, ReLU^2, or GELU** feed-forward activations
- **Ablation-friendly** -- all architectural features are togglable with no overhead when off
- **Distributed training** via HuggingFace Accelerate with FSDP, mixed precision (bf16/fp16), gradient checkpointing, and optional Muon optimization
- **Optional FlashAttention** support (v2/v3 and v4) for efficient long-sequence training
- **Built-in evaluation** for MLM metrics (loss, accuracy, perplexity) and structure-based contact prediction
- **Five model presets** from 4.8M to 11.0B parameters

---

## Installation

**Requirements:** Python 3.11+ and PyTorch 2.10+

### From PyPI

```bash
pip install oplm
```

### From source

```bash
git clone https://github.com/briney/oplm.git
cd oplm
pip install -e .
```

### Optional extras

```bash
# distributed training with Accelerate + Weights & Biases logging
pip install oplm[train]

# FlashAttention v2/v3 (Ampere/Hopper GPUs)
pip install oplm[flash]

# FlashAttention v4 (Blackwell/Hopper GPUs)
pip install oplm[flash4]

# everything (pick one flash variant)
pip install oplm[train,flash]

# development (pytest, ruff, mypy)
pip install oplm[dev]
```

---

## Quick start

### Tokenize protein sequences

```python
from oplm.data.tokenizer import ProteinTokenizer

tokenizer = ProteinTokenizer()

# single sequence
token_ids = tokenizer.encode("MKWVTFISLLLLFSSAYS")

# batch with padding
batch = tokenizer.batch_encode(
    ["MKWVTFISLLLLFSSAYS", "MLPGLALLLLAAWTARA"],
    max_length=512,
)
print(batch["input_ids"].shape)       # (2, 512)
print(batch["attention_mask"].shape)   # (2, 512)
```

### Build a model

```python
from oplm.config import ModelConfig
from oplm.model import OplmForMLM

cfg = ModelConfig(
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    num_kv_heads=4,
)

model = OplmForMLM(cfg)
```

Or use a built-in preset:

```python
from oplm.config import load_config

cfg = load_config(["--preset", "medium"])
model = OplmForMLM(cfg.model)
```

### Extract embeddings

```python
import torch
from oplm.data.tokenizer import ProteinTokenizer
from oplm.model import OplmForMLM
from oplm.config import ModelConfig

tokenizer = ProteinTokenizer()
cfg = ModelConfig(hidden_dim=768, num_layers=12, num_heads=12)
model = OplmForMLM(cfg)
model.eval()

sequences = ["MKWVTFISLLLLFSSAYS", "MLPGLALLLLAAWTARA"]
batch = tokenizer.batch_encode(sequences, max_length=512)

with torch.no_grad():
    hidden = model.encoder(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )[0]

print(hidden.shape)  # (2, 512, 768)
```

For checkpoint-backed inference, use the shared loader so training-produced checkpoint
directories work without manual state-dict handling:

```python
from oplm.inference import load_model_for_inference, resolve_inference_config

checkpoint = "outputs/medium-run/checkpoint-10000"
cfg = resolve_inference_config(checkpoint)
model = load_model_for_inference(checkpoint, cfg)
```

### Masked language modeling

```python
outputs = model(
    batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=labels,  # token IDs at masked positions, -100 elsewhere
)
loss = outputs["loss"]
logits = outputs["logits"]
```

---

## CLI

OPLM provides a command-line interface for training, embedding extraction, and model inspection.

### Training

```bash
# train with a YAML config
oplm train --config configs/my_run.yaml

# use a size preset
oplm train --preset medium

# preset + overrides
oplm train --preset large \
  --override model.num_layers=16 \
  --override train.lr=3e-4

# distributed training with Accelerate
accelerate launch -m oplm.train --config configs/my_run.yaml
```

Set `train.optimizer=muon` to enable the built-in Muon optimizer for eligible
hidden 2D weights while keeping AdamW on embeddings, norms, biases, and the MLM
head.

Standard one-node multi-GPU runs use plain Accelerate/DDP. If your shell or
launcher environment enables DeepSpeed globally, `oplm.train` disables it by
default to avoid DeepSpeed/Triton startup noise; set `OPLM_ENABLE_DEEPSPEED=1`
to opt back in intentionally.

### Encode sequences

```bash
oplm encode MKWVTFISLLLLFSSAYS MLPGLALLLLAAWTARA \
  --model /path/to/checkpoint-10000 \
  --output embeddings.pt
```

### Inspect a model configuration

```bash
oplm info --preset medium
```

```
──────────── OPLM Model Info ─────────────
       Architecture
 Parameters   150.3M (150,341,409)
 Hidden dim   768
 Layers       12
 Attn heads   12 (KV: 4)
 FFN dim      2048
 Max seq len  2048
 ...
```

---

## Configuration

OPLM uses a layered config system: **defaults -> preset -> YAML file -> CLI overrides**, with later sources taking priority.

The canonical field-by-field reference lives in [configs/README.md](configs/README.md).
Runtime precision is controlled by `train.mixed_precision`; `model.dtype` is currently a
reserved placeholder.
`model.max_seq_len` is the sequence-length setting for training, eval, and inference.
`oplm` CLI commands take repeated `--override key=value` flags, while
`accelerate launch -m oplm.train ...` still passes raw dotlist overrides through to
`load_config()`.

### Model presets

| Preset   | Parameters | Layers | Hidden | Heads | KV Heads |
|----------|-----------|--------|--------|-------|----------|
| `small`  | ~4.8M     | 6      | 256    | 4     | 2        |
| `medium` | ~76.2M    | 12     | 768    | 12    | 4        |
| `base`   | ~271.7M   | 24     | 1024   | 16    | 4        |
| `large`  | ~2.2B     | 32     | 2560   | 32    | 8        |
| `xlarge` | ~11.0B    | 40     | 5120   | 40    | 8        |

### YAML config example

```yaml
model:
  hidden_dim: 768
  num_layers: 12
  num_heads: 12
  num_kv_heads: 4
  max_seq_len: 1024
  ffn_activation: swiglu
  shared_kv: false
  qk_norm: true
  partial_rope: true
  value_residual: true
  conv_positions: "AC"
  conv_kernel_size: 3
  conv_kernel_schedule: block_step
  conv_kernel_increment: 2
  conv_kernel_block_size: 2
  conv_kernel_max_size: 9
  attn_residual: true

train:
  max_steps: 100_000
  batch_size: 128
  lr: 1e-4
  warmup_steps: 5_000
  scheduler: warmup_cosine
  mixed_precision: bf16
  wandb_project: oplm

data:
  train: /path/to/sequences.parquet
  mask_prob: 0.15
```

### Training data format

Training data should be parquet files with two columns:

| Column        | Type   | Description          |
|---------------|--------|----------------------|
| `sequence_id` | string | Unique identifier    |
| `sequence`    | string | Amino acid sequence  |

OPLM supports single parquet files, directories of shards, or interleaved multi-source datasets.

---

## Evaluation

OPLM includes a built-in evaluation harness with support for:

- **Sequence evaluation** -- masked language modeling metrics (loss, accuracy, perplexity)
- **Structure evaluation** -- contact prediction precision@L from attention maps, optional logistic regression fitting, and categorical-Jacobian contact extraction

Configure evaluation datasets in your YAML config:

```yaml
data:
  eval:
    validation:
      path: /path/to/eval_sequences.parquet
      type: sequence
    structures:
      path: /path/to/pdb_directory
      type: structure
      eval_every: 10_000
      contact_threshold: 8.0
      use_logistic_regression: true
      use_categorical_jacobian: true
      categorical_jacobian_sample_size: 12
```

Evaluation runs automatically during training at the configured interval, with results logged to Weights & Biases.

---

## Architecture

The core architecture is an encoder-only transformer with the following optional components:

```
Input tokens
    -> TokenEmbedding (scaled by sqrt(D), optional post-norm)
    -> [ValueEmbedding (first/last N layers)]
    -> TransformerBlock x N:
        -> [BidirectionalDepthwiseConv (position A)]
        -> RMSNorm -> Attention (GQA, optional shared K/V, Q/K norm,
           partial RoPE, output gating, value residuals)
        -> [BlockAttentionResidual | standard residual]
        -> [BidirectionalDepthwiseConv (position C)]
        -> RMSNorm -> FFN (SwiGLU / ReLU^2 / GELU)
            -> [BidirectionalDepthwiseConv (position D)]
        -> [BlockAttentionResidual | standard residual]
    -> MLMHead (Dense -> RMSNorm -> GELU -> projection)
```

For a complete architectural description, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Development

```bash
# install with dev dependencies
pip install -e ".[dev]"

# run tests
pytest

# run tests with coverage
pytest --cov=oplm

# skip slow tests
pytest -m "not slow"

# lint and format
ruff check src/
ruff format src/

# type check
mypy src/
```

---

## License

MIT

# OPLM Config Reference

The canonical, field-by-field reference for configuring OPLM models and training
runs. For the *semantics* of the architecture toggles (what `norm_strategy`,
Canon convolutions, or partial RoPE actually do), see
[MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md). For the eval harness, see
[EVAL_HARNESS.md](EVAL_HARNESS.md). For the training loop, see
[TRAIN.md](TRAIN.md).

## Two config namespaces

A run config has three top-level blocks, each mapping to a different object:

| Block     | Backing object                              | Notes |
|-----------|---------------------------------------------|-------|
| `model.*` | `oplm.model.OplmConfig` (a HuggingFace `PretrainedConfig`) | The model architecture. Serialized into `config.json`. |
| `train.*` | `oplm.config.TrainConfig` (dataclass)        | Optimizer, schedule, logging, checkpointing. |
| `data.*`  | `oplm.config.DataConfig` (dataclass)         | Datasets, masking, DataLoader. |

Only the `model.*` block travels with a saved model. `train.*` and `data.*`
describe a *run* and are not part of a published checkpoint.

> **Unknown `model.*` keys are silently absorbed.** `PretrainedConfig` accepts
> arbitrary `**kwargs`, so a misspelled `model.*` key (e.g. `model.hidden_dimm`)
> is retained without error. Double-check model overrides. `train.*` and
> `data.*`, by contrast, are validated against their dataclasses.

## Merge order

Config sources are merged in this order; later sources win:

1. In-package defaults — `src/oplm/configs/{model,train,data}/base.yaml`
2. Optional size preset — `--preset {small,medium,base,large,xlarge}`
3. Optional YAML file — `--config path.yaml`
4. Dotlist overrides — e.g. `model.num_hidden_layers=24 train.lr=3e-4`

Notes:

- `model.head_dim`, `model.intermediate_size`, and `model.rope_dim` / `nope_dim`
  are **derived** from the core dimensions unless you set them explicitly (see
  [Derived fields](#derived-and-validated-fields)).
- `train.config_path` is populated automatically when `--config` is used.
- `data.max_length` was **removed**. Use `model.max_position_embeddings` as the
  sequence-length setting; passing `data.max_length` raises an error.

## Override syntax

```bash
# bare key=value positional overrides (train, info)
oplm train --preset base \
  train.max_steps=100_000 \
  data.train=/data/train.parquet

oplm info --preset large model.num_hidden_layers=40

# encode keeps --override, since its positional slot holds the sequences
oplm encode SEQ1 SEQ2 --model outputs/base-run/checkpoint-10000 \
  --override model.attention_dropout=0.0

# the distributed entry point forwards trailing dotlist args to load_config()
accelerate launch -m oplm.train --config configs/my_run.yaml \
  train.resume_from=outputs/base-run/checkpoint-10000
```

Flags: `--preset/-p` (size preset), `--config/-c` (YAML file), `--name/-n`
(sets `train.wandb_run_name` unless explicitly set elsewhere).

## Size presets

`--preset <name>` loads `src/oplm/configs/model/presets/<name>.yaml`, which sets
only the core dimensions; everything else inherits the defaults below.

| Preset   | Parameters | Layers | Hidden | Heads | Head dim |
|----------|-----------:|-------:|-------:|------:|---------:|
| `small`  |       5.2M |      6 |    256 |     4 |       64 |
| `medium` |      85.6M |     12 |    768 |    12 |       64 |
| `base`   |     309.5M |     24 |   1024 |    16 |       64 |
| `large`  |       2.5B |     32 |   2560 |    32 |       80 |
| `xlarge` |      12.7B |     40 |   5120 |    40 |      128 |

`large` and `xlarge` also enable `model.gradient_checkpointing`. Run
`oplm info --preset <name>` to print the resolved architecture and exact
parameter count.

## Model fields (`model.*`)

Backed by `oplm.model.OplmConfig`. Defaults shown are the effective defaults
from `configs/model/base.yaml`.

### Core dimensions

| Override | Type | Default | Notes |
| --- | --- | --- | --- |
| `model.vocab_size` | `int` | `33` | Tokenizer size. Values other than `33` warn (custom vocabularies are not yet supported). |
| `model.hidden_size` | `int` | `768` | Residual-stream width. Must be divisible by `num_attention_heads`. |
| `model.num_hidden_layers` | `int` | `12` | Number of transformer blocks. |
| `model.num_attention_heads` | `int` | `12` | Standard multi-head attention (no GQA). |
| `model.head_dim` | `int \| null` | derived | Defaults to `hidden_size // num_attention_heads`. |
| `model.intermediate_size` | `int \| null` | derived | FFN inner width. Defaults to `round_up_to(8/3 · hidden_size, 256)`. |
| `model.max_position_embeddings` | `int` | `1024` | Context length used by train, eval, and inference. |

### Positional encoding

| Override | Type | Default | Notes |
| --- | --- | --- | --- |
| `model.rope_theta` | `float` | `10000.0` | RoPE base frequency. |
| `model.rope_dim` | `int \| null` | derived | Rotated channels per head. Defaults to `head_dim` (full RoPE). Must be even. |
| `model.nope_dim` | `int` | `0` | Un-rotated (NoPE) channels per head. For partial RoPE set `rope_dim < head_dim` and `nope_dim = head_dim − rope_dim`. |

### Normalization

| Override | Type | Default | Valid values / notes |
| --- | --- | --- | --- |
| `model.norm_type` | `str` | `layernorm` | `layernorm` or `rmsnorm`. |
| `model.norm_eps` | `float` | `1e-6` | Norm epsilon. |
| `model.norm_strategy` | `str` | `pre` | `pre`, `sandwich`, `hybrid`, or `post_sdpa`. |
| `model.qk_norm` | `bool` | `true` | Normalize Q and K before attention. |
| `model.post_embed_norm` | `bool` | `false` | Apply a norm after the token-embedding lookup. |
| `model.residual_scaling` | `str` | `sqrt_num_layers` | `sqrt_num_layers` (scale each sublayer's residual write by `1/√L`) or `none`. |
| `model.init_scale_output_projections` | `bool` | `true` | Shrink residual-writing projections by `1/√(2L)` at init. |

### Feed-forward and dropout

| Override | Type | Default | Valid values / notes |
| --- | --- | --- | --- |
| `model.ffn_activation` | `str` | `swiglu` | `swiglu` or `geglu` (both gated, 3 projections). |
| `model.ffn_bias` | `bool` | `false` | Bias terms on FFN projections. |
| `model.attention_dropout` | `float` | `0.0` | Dropout on attention probabilities. |
| `model.hidden_dropout` | `float` | `0.0` | Dropout on hidden states. |

### Embeddings and MLM head

| Override | Type | Default | Valid values / notes |
| --- | --- | --- | --- |
| `model.tie_word_embeddings` | `bool` | `false` | Tie the MLM decoder to the input embedding. |
| `model.mlm_head_activation` | `str` | `gelu` | `gelu`, `silu`, or `relu`. |

### Canon depthwise convolution (off by default)

| Override | Type | Default | Valid values / notes |
| --- | --- | --- | --- |
| `model.canon_enabled` | `bool` | `false` | Master switch for Canon conv sublayers. |
| `model.canon_positions` | `list[str]` | `[]` | Subset of `{A, B, C, D}`; required and non-empty when `canon_enabled=true`. |
| `model.canon_kernel_sizes` | `int \| list[int] \| dict` | `4` | `int` broadcasts to every layer; a `list[int]` must have length `num_hidden_layers`; a `dict` uses `schedule: linear` (`min`, `max`) or `schedule: constant` (`value`). All entries must be `≥ 2`. |
| `model.canon_activation` | `str` | `none` | `none`, `silu`, or `gelu`. |

### Fine-tuning heads

Consumed by `OplmForSequenceClassification` / `OplmForTokenClassification`.

| Override | Type | Default | Valid values / notes |
| --- | --- | --- | --- |
| `model.num_labels` | `int` | `2` | Output classes for classification heads. |
| `model.classifier_pool` | `str` | `mean` | `mean` or `cls`; sequence-level pooling. |
| `model.classifier_dropout` | `float` | `0.0` | Dropout before the classifier. |
| `model.pre_head_norm` | `bool` | `false` | Apply a norm before the classification head. |

### Runtime and special tokens

| Override | Type | Default | Valid values / notes |
| --- | --- | --- | --- |
| `model.initializer_range` | `float` | `0.02` | Truncated-normal init std. |
| `model.gradient_checkpointing` | `bool` | `false` | Activation checkpointing in the encoder. |
| `model.pad_token_id` | `int` | `1` | `<pad>`. |
| `model.bos_token_id` | `int` | `0` | `<cls>` (prepended to every sequence). |
| `model.eos_token_id` | `int` | `2` | `<eos>`. |
| `model.unk_token_id` | `int` | `3` | `<unk>`. |
| `model.mask_token_id` | `int` | `32` | `<mask>`. |

### Derived and validated fields

When omitted, derived fields resolve from the core dimensions:

- `head_dim = hidden_size // num_attention_heads`
- `intermediate_size = round_up_to(8/3 · hidden_size, 256)`
- `rope_dim = head_dim`, `nope_dim = 0` (full RoPE)

Config construction raises `ValueError` if any of these fail:

- `num_attention_heads > 0` and `hidden_size % num_attention_heads == 0`
- `head_dim · num_attention_heads == hidden_size`
- `rope_dim + nope_dim == head_dim`, both `≥ 0`, and `rope_dim` even
- enum fields (`norm_type`, `norm_strategy`, `residual_scaling`,
  `ffn_activation`, `mlm_head_activation`, `canon_activation`,
  `classifier_pool`) hold a valid value
- when `canon_enabled`, `canon_positions` is a non-empty, duplicate-free subset
  of `{A, B, C, D}`

## Train fields (`train.*`)

Backed by `oplm.config.TrainConfig`.

| Override | Type | Default | Valid values / notes |
| --- | --- | --- | --- |
| `train.max_steps` | `int` | `50_000` | Step budget; used when `max_epochs` is unset. |
| `train.max_epochs` | `int \| null` | `null` | Optional epoch-based stop. |
| `train.batch_size` | `int` | `32` | Per-process batch size, before accumulation. |
| `train.gradient_accumulation_steps` | `int` | `1` | Must be `≥ 1`. |
| `train.optimizer` | `str` | `adamw` | `adamw` or `muon` (hybrid Muon + AdamW). |
| `train.lr` | `float` | `1e-4` | Peak learning rate. |
| `train.min_lr` | `float` | `0.0` | Must be `≥ 0` and `≤ lr`. |
| `train.weight_decay` | `float` | `0.01` | Decoupled weight decay. |
| `train.adam_beta1` | `float` | `0.9` | AdamW β₁ (also Muon's auxiliary AdamW). |
| `train.adam_beta2` | `float` | `0.98` | AdamW β₂. |
| `train.adam_eps` | `float` | `1e-8` | AdamW ε. |
| `train.muon_adjust_lr_fn` | `str` | `match_rms_adamw` | `match_rms_adamw` or `original`. |
| `train.muon_momentum` | `float` | `0.95` | Must be `≥ 0`. |
| `train.muon_nesterov` | `bool` | `true` | Muon Nesterov momentum. |
| `train.muon_ns_steps` | `int` | `5` | Newton–Schulz steps; must be `≥ 1`. |
| `train.max_grad_norm` | `float` | `1.0` | Set `0` to disable clipping. |
| `train.scheduler` | `str` | `warmup_linear` | `warmup_linear`, `warmup_cosine`, `wsd_linear`, or `wsd_cosine`. |
| `train.warmup_steps` | `int` | `5_000` | Must be `≥ 0`. |
| `train.stable_steps` | `int` | `0` | WSD plateau length; must be `≥ 0`. |
| `train.log_every` | `int` | `10` | Train-metric logging cadence (optimizer steps). |
| `train.eval_every` | `dict \| null` | `null` | Default eval cadence for datasets that omit `every`. Exactly one of `{steps: N}` / `{tokens: N}`. `null` resolves to `{steps: 10_000}`. |
| `train.wandb_project` | `str` | `oplm` | W&B project. |
| `train.wandb_run_name` | `str \| null` | `null` | W&B run name (also settable via `--name`). |
| `train.wandb_enabled` | `bool` | `true` | Toggle W&B logging. |
| `train.save_every` | `int` | `10_000` | Checkpoint cadence (optimizer steps). |
| `train.save_total_limit` | `int` | `3` | Checkpoints to keep. |
| `train.resume_from` | `str \| null` | `null` | Path to an Accelerate checkpoint directory. |
| `train.seed` | `int` | `42` | Global seed. |
| `train.output_dir` | `str` | `outputs` | Base dir for logs and checkpoints. |
| `train.mixed_precision` | `str` | `bf16` | `bf16`, `fp16`, or `no`. |
| `train.config_path` | `str \| null` | `null` | Auto-populated from `--config` for provenance. |

## Data fields (`data.*`)

Backed by `oplm.config.DataConfig`.

| Override | Type | Default | Valid values / notes |
| --- | --- | --- | --- |
| `data.train` | `str \| dict \| null` | `null` | A single parquet path/directory, or a named map of `{path, fraction}` (see below). |
| `data.eval` | `dict \| null` | `null` | Named map of eval datasets (see [Eval datasets](#eval-datasets)). |
| `data.mask_prob` | `float` | `0.15` | Fraction of eligible positions selected for masking. |
| `data.mask_token_prob` | `float` | `0.8` | Of masked positions, fraction replaced with `<mask>`. In `[0, 1]`. |
| `data.random_token_prob` | `float` | `0.1` | Of masked positions, fraction replaced with a random amino acid. `mask_token_prob + random_token_prob ≤ 1`; the remainder is left unchanged. |
| `data.weighted_masking` | `bool` | `false` | Honor a per-row `masking_weights` column when selecting positions. |
| `data.num_workers` | `int` | `4` | DataLoader workers. |
| `data.pin_memory` | `bool` | `true` | Pin host memory for batches. |
| `data.prefetch_factor` | `int` | `4` | Batches prefetched per worker. |
| `data.shuffle_shards` | `bool` | `true` | Shuffle shard order (sharded parquet directories). |
| `data.shuffle_rows` | `bool` | `true` | Shuffle rows within each shard. |

### Training data format

Training and sequence-eval data are parquet with these columns:

| Column        | Type   | Required | Description |
|---------------|--------|----------|-------------|
| `sequence_id` | string | yes      | Unique identifier. |
| `sequence`    | string | yes      | Amino-acid sequence. |
| `masking_weights` | list/array | only if `weighted_masking=true` | Per-residue masking weights. |

`data.train` accepts a single `.parquet` file, a directory of shards, or a named
multi-dataset map with fractional sampling (fractions normalize to 1.0; omitted
fractions split the remainder evenly):

```yaml
data:
  train:
    uniref50: { path: /data/uniref50/, fraction: 0.6 }
    bfd:      { path: /data/bfd/,      fraction: 0.4 }
```

## Eval datasets

Each entry under `data.eval` is keyed by a name you choose:

```yaml
data:
  eval:
    heldout:
      path: /data/eval_sequences.parquet
      type: sequence
```

Shared keys on every entry:

- `path` — file or directory consumed by the task.
- `type` — one of `sequence`, `structure`, `proteingym`, `tape`, `proteinglue`,
  `everest`.
- `every` — optional per-dataset cadence: exactly one of `{steps: N}` /
  `{tokens: N}`, plus optional `at_start` (default `false`) and `at_end`
  (default `true`). Datasets that omit `every` inherit `train.eval_every`.
- `metrics` — optional list of metric names to keep.

Task-specific keys sit at the **same level** as `path` and `type` (not nested
under `extra:`). See [EVAL_HARNESS.md](EVAL_HARNESS.md) for the full harness
documentation.

### sequence

Masked-language-model metrics (loss, accuracy, perplexity) over the same parquet
schema as training data. No task-specific keys.

### structure

Unsupervised contact prediction — `precision_at_L{,_2,_5}` from the model's
categorical Jacobian over PDB/CIF structures.

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `contact_threshold` | `float` | `8.0` | Contact distance threshold (Å). |
| `min_seq_sep` | `int` | `6` | Minimum sequence separation for scoring. |
| `l_divisor` | `int` | `1` | Score the top `L / l_divisor` contacts. |
| `use_cbeta` | `bool` | `true` | Use virtual C-β distances. |
| `categorical_jacobian_sample_size` | `int \| null` | `null` | Optional deterministic subset of structures (the Jacobian costs `L × 20` forwards each). |
| `categorical_jacobian_sample_seed` | `int` | `42` | Seed for subset sampling. |
| `categorical_jacobian_mutation_batch_size` | `int` | `20` | Mutant sequences per Jacobian forward pass. |
| `max_structures` | `int \| null` | `null` | Optional cap on structures loaded. |

### proteingym, tape, proteinglue, everest

Registered but currently documented stubs; they consume no task-specific keys
yet. See [EVAL_HARNESS.md](EVAL_HARNESS.md) for status.

## Examples

### Minimal CPU smoke train

```yaml
model:
  hidden_size: 64
  num_hidden_layers: 2
  num_attention_heads: 4
  max_position_embeddings: 128

train:
  max_steps: 10
  batch_size: 4
  warmup_steps: 0
  wandb_enabled: false
  mixed_precision: no
  output_dir: outputs/smoke

data:
  train: tests/fixtures/training/test_sequences.parquet
  num_workers: 0
  pin_memory: false
```

### Train with eval

```yaml
train:
  eval_every: { steps: 500 }

data:
  train: /data/train_sequences.parquet
  eval:
    heldout:
      path: /data/eval_sequences.parquet
      type: sequence
    structures:
      path: /data/pdb
      type: structure
      every: { steps: 2_000 }
      contact_threshold: 8.0
      categorical_jacobian_sample_size: 12
```

### Checkpoint resume

```yaml
train:
  output_dir: outputs/base-run
  resume_from: outputs/base-run/checkpoint-10000
```

### Inference / embedding extraction

CLI:

```bash
oplm encode MKWVTFISLLLLFSSAYS MLPGLALLLLAAWTARA \
  --model outputs/base-run/checkpoint-10000 \
  --output embeddings.pt
```

Python:

```python
import torch
from oplm import OplmForMaskedLM, LogitsConfig

model = OplmForMaskedLM.from_pretrained("outputs/base-run/checkpoint-10000").eval()

with torch.no_grad():
    out = model.logits(
        ["MKWVTFISLLLLFSSAYS"],
        LogitsConfig(sequence=True, return_embeddings=True),
    )

out.sequence_logits   # (1, T, 33)
out.embeddings        # (1, T, hidden_size)
```

See the [README](../README.md#quick-start) for more inference examples.

# OPLM Config Reference

This is the canonical config reference for OPLM. The field tables in this file are
mechanically checked in `tests/test_config_docs.py` so new dataclass fields cannot land
without updating the docs.

## Merge Order

Config sources are merged in this order, with later sources taking priority:

1. In-package defaults from `src/oplm/config.py`
2. Optional preset from `--preset {small,medium,base,large,xlarge}`
3. Optional YAML file from `--config path.yaml`
4. Dotlist overrides such as `model.num_layers=24`

Notes:

- `model.head_dim`, `model.ffn_dim`, `model.rope_dim`, and `model.nope_dim` are derived
  unless you explicitly set them.
- `train.config_path` is populated automatically when `--config` is used.
- Runtime precision is controlled by `train.mixed_precision`. `model.dtype` is currently a
  reserved placeholder and is not consumed by the runtime path.
- `model.max_seq_len` is the canonical sequence-length setting for training, eval, and
  inference. `data.max_length` remains a deprecated compatibility alias.
- Eval task-specific keys live directly on each `data.eval.<name>` entry. Do not nest them
  under `extra:`.

## Override Syntax

Examples:

```bash
oplm train --preset medium \
  --override train.max_steps=1000 \
  --override data.train=/data/train.parquet
oplm info --config configs/my_run.yaml --override model.num_layers=16
accelerate launch -m oplm.train --config configs/my_run.yaml train.resume_from=outputs/checkpoint-1000
```

`oplm train`, `oplm info`, and `oplm encode` all use repeated `--override key=value`
flags. The lower-level distributed entry point keeps raw dotlist passthrough because
`accelerate launch -m oplm.train ...` forwards trailing arguments directly to
`load_config()`.

## Model Fields

| Override | Type | Default | Valid values / notes | Status |
| --- | --- | --- | --- | --- |
| `model.hidden_dim` | `int` | `768` | Must be divisible by `model.num_heads`. | active |
| `model.num_layers` | `int` | `12` | Positive layer count. | active |
| `model.num_heads` | `int` | `12` | Must divide `model.hidden_dim`. | active |
| `model.num_kv_heads` | `int` | `4` | Must divide `model.num_heads`. | active |
| `model.head_dim` | `int \| null` | derived | Auto-computed as `hidden_dim // num_heads` when omitted. | derived |
| `model.ffn_dim` | `int \| null` | derived | Auto-computed from `ffn_activation` when omitted. | derived |
| `model.ffn_activation` | `str` | `swiglu` | `swiglu`, `relu_squared`, or `gelu`. | active |
| `model.vocab_size` | `int` | `33` | Tokenizer vocabulary size. | active |
| `model.max_seq_len` | `int` | `512` | Canonical context length and initial RoPE cache size used by train, eval, and inference. | active |
| `model.shared_kv` | `bool` | `false` | Share K and V projections. | active |
| `model.qk_norm` | `bool` | `true` | Apply RMSNorm to Q and K before attention. | active |
| `model.output_gate` | `bool` | `false` | Enable attention output gating. | active |
| `model.query_dependent_gate` | `bool` | `false` | Only meaningful when `output_gate=true`. | active |
| `model.post_sdpa_norm` | `bool` | `false` | Apply RMSNorm after attention computation. | active |
| `model.rope_theta` | `float` | `10000.0` | RoPE base frequency. | active |
| `model.partial_rope` | `bool` | `false` | Split each head into NoPE and RoPE dimensions. | active |
| `model.nope_dim` | `int \| null` | derived | Auto-computed as `head_dim - rope_dim` when omitted. | derived |
| `model.rope_dim` | `int \| null` | derived | Auto-computed as `head_dim` or `32` when omitted. | derived |
| `model.value_residual` | `bool` | `false` | Enable cross-layer value residual mixing. | active |
| `model.value_residual_lambda_init` | `float` | `0.5` | Initial mixing bias for value residuals. | active |
| `model.num_value_embeds` | `int` | `0` | `0` disables value embeddings; positive values enable first/last N layers. | active |
| `model.value_embed_gate_dim` | `int` | `16` | Hidden size of the value-embedding gate. | active |
| `model.conv_positions` | `str` | `""` | Any combination of `A`, `C`, `D`. | active |
| `model.conv_kernel_size` | `int` | `7` | Must be odd; exact size in `static` mode and starting size in `block_step`. | active |
| `model.conv_kernel_schedule` | `str` | `static` | `static` or `block_step`. | active |
| `model.conv_kernel_increment` | `int` | `2` | Used by `block_step`; must be a non-negative even integer. | active |
| `model.conv_kernel_block_size` | `int` | `1` | Used by `block_step`; increment once every N layers. | active |
| `model.conv_kernel_max_size` | `int \| null` | `null` | Optional saturating clamp for `block_step`; must be odd when set. | active |
| `model.conv_activation` | `bool` | `true` | Apply SiLU inside depthwise convolutions. | active |
| `model.attn_residual` | `bool` | `false` | Enable block attention residuals. | active |
| `model.attn_residual_block_size` | `int` | `8` | Must divide `model.num_layers` when attention residuals are enabled. | active |
| `model.norm_eps` | `float` | `1e-6` | RMSNorm epsilon. | active |
| `model.post_embed_norm` | `bool` | `false` | Apply RMSNorm after token embedding lookup. | active |
| `model.gradient_checkpointing` | `bool` | `false` | Enables activation checkpointing in the encoder. | active |
| `model.tie_embeddings` | `bool` | `false` | Tie MLM projection weights to token embeddings. | active |
| `model.dtype` | `str` | `bfloat16` | Reserved placeholder; runtime precision is controlled by `train.mixed_precision`. | unused |

## Train Fields

| Override | Type | Default | Valid values / notes | Status |
| --- | --- | --- | --- | --- |
| `train.max_steps` | `int` | `50000` | Used when `train.max_epochs` is unset. | active |
| `train.max_epochs` | `int \| null` | `null` | Optional epoch-based stop condition. | active |
| `train.batch_size` | `int` | `32` | Per-process batch size before gradient accumulation. | active |
| `train.gradient_accumulation_steps` | `int` | `1` | Must be `>= 1`. | active |
| `train.optimizer` | `str` | `adamw` | `adamw` (default) or `muon`. Muon uses a hybrid Muon+AdamW partitioning. | active |
| `train.lr` | `float` | `1e-4` | Peak learning rate. | active |
| `train.min_lr` | `float` | `0.0` | Must be `>= 0` and `<= train.lr`. | active |
| `train.weight_decay` | `float` | `0.01` | Shared decoupled weight decay for AdamW and Muon. | active |
| `train.adam_beta1` | `float` | `0.9` | AdamW beta1. Reused by Muon's auxiliary AdamW optimizer. | active |
| `train.adam_beta2` | `float` | `0.98` | AdamW beta2. Reused by Muon's auxiliary AdamW optimizer. | active |
| `train.adam_eps` | `float` | `1e-8` | AdamW epsilon. Reused by Muon's auxiliary AdamW optimizer. | active |
| `train.muon_adjust_lr_fn` | `str` | `match_rms_adamw` | `match_rms_adamw` or `original`. | active |
| `train.muon_momentum` | `float` | `0.95` | Muon momentum. Must be `>= 0`. | active |
| `train.muon_nesterov` | `bool` | `true` | Enable Muon Nesterov momentum. | active |
| `train.muon_ns_steps` | `int` | `5` | Muon Newton-Schulz steps. Must be `>= 1`. | active |
| `train.max_grad_norm` | `float` | `1.0` | Set to `0` to disable clipping. | active |
| `train.scheduler` | `str` | `warmup_linear` | `warmup_linear`, `warmup_cosine`, `wsd_linear`, or `wsd_cosine`. | active |
| `train.warmup_steps` | `int` | `5000` | Must be `>= 0`. | active |
| `train.stable_fraction` | `float` | `0.0` | Must be in `[0, 1)`. | active |
| `train.log_every` | `int` | `10` | Train-metric logging cadence in optimizer steps. | active |
| `train.eval_every` | `int` | `10000` | Default eval cadence for datasets without a per-dataset override. | active |
| `train.wandb_project` | `str` | `oplm` | W&B project name. | active |
| `train.wandb_run_name` | `str \| null` | `null` | Optional W&B run name. | active |
| `train.wandb_enabled` | `bool` | `true` | Enable W&B logging through `accelerate`. | active |
| `train.save_every` | `int` | `10000` | Checkpoint cadence in optimizer steps. | active |
| `train.save_total_limit` | `int` | `3` | Number of checkpoints to keep. | active |
| `train.resume_from` | `str \| null` | `null` | Path to an `accelerate` checkpoint directory. | active |
| `train.seed` | `int` | `42` | Global random seed. | active |
| `train.output_dir` | `str` | `outputs` | Base directory for logs and checkpoints. | active |
| `train.config_path` | `str \| null` | `null` | Populated automatically from `--config` for provenance. | provenance |
| `train.mixed_precision` | `str` | `bf16` | `bf16`, `fp16`, or `no`. | active |

## Data Fields

| Override | Type | Default | Valid values / notes | Status |
| --- | --- | --- | --- | --- |
| `data.train` | `str \| dict \| null` | `null` | Single parquet path or named dataset map with optional fractions. | active |
| `data.eval` | `dict \| null` | `null` | Named eval dataset map. See [Eval Datasets](#eval-datasets). | active |
| `data.max_length` | `int` | `512` | Deprecated compatibility alias for `model.max_seq_len`. `load_config()` copies it into `model.max_seq_len` and warns. | deprecated |
| `data.mask_prob` | `float` | `0.15` | BERT-style MLM masking probability. | active |
| `data.num_workers` | `int` | `4` | PyTorch DataLoader workers. | active |
| `data.pin_memory` | `bool` | `true` | Pin host memory for DataLoader batches. | active |
| `data.prefetch_factor` | `int` | `4` | Batches prefetched per worker. | active |
| `data.shuffle_shards` | `bool` | `true` | Shuffle shard order for sharded parquet directories. | active |
| `data.shuffle_rows` | `bool` | `true` | Shuffle row order within each shard. | active |

## Eval Datasets

Each entry under `data.eval` is keyed by a dataset name of your choice:

```yaml
data:
  eval:
    heldout:
      path: /data/eval_sequences.parquet
      type: sequence
```

Shared keys on every eval dataset entry:

- `path`: file or directory path consumed by the task
- `type`: one of `sequence`, `structure`, `proteingym`, `tape`, `proteinglue`, `everest`
- `eval_every`: optional per-dataset cadence override
- `metrics`: optional list of metric names to keep

Task-specific keys go at the same level as `path` and `type`, not under `extra:`.
They are currently passed through to the task as `EvalDatasetEntry.extra`; typed
per-task config objects remain a future cleanup.

### Sequence

Sequence eval uses the same parquet schema as training data and currently consumes no
task-specific keys beyond the shared ones above.

### Structure

Structure eval loads PDB/CIF files and supports these task-specific keys:

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `contact_threshold` | `float` | `8.0` | Contact distance threshold in angstroms. |
| `min_seq_sep` | `int` | `6` | Minimum sequence separation for scoring. |
| `l_divisor` | `int` | `1` | Evaluate top `L / l_divisor` contacts. |
| `use_cbeta` | `bool` | `true` | Use virtual C-beta distances. |
| `use_logistic_regression` | `bool` | `true` | Fit the ESM-style L1 logistic regression probe when enough structures exist. |
| `logreg_n_train` | `int` | `20` | Structures reserved for probe training. |
| `logreg_n_iterations` | `int` | `5` | Cross-validation iterations for the probe. |
| `logreg_c` | `float` | `0.15` | Inverse regularization strength for the probe. |
| `use_categorical_jacobian` | `bool` | `false` | Enable unsupervised categorical-Jacobian P@L alongside the attention/logreg metrics. |
| `categorical_jacobian_sample_size` | `int \| null` | `null` | Optional deterministic subset size used only for the Jacobian path. |
| `categorical_jacobian_sample_seed` | `int` | `42` | Seed for Jacobian subset sampling. |
| `categorical_jacobian_mutation_batch_size` | `int` | `20` | Number of mutant sequences per Jacobian forward pass. |
| `max_structures` | `int \| null` | `null` | Optional cap on structures loaded from disk. |

### ProteinGym

ProteinGym is currently a documented stub. The planned task-specific keys are:

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `max_assays` | `int \| null` | `null` | Optional cost-control cap on assays. |
| `scoring` | `str` | `masked_marginals` | Planned values: `masked_marginals` or `wild_type_marginals`. |

### TAPE

TAPE is currently a stub. No task-specific keys are consumed yet.

### ProteinGlue

ProteinGlue is currently a stub. No task-specific keys are consumed yet.

### EVEREST

EVEREST is currently a stub. No task-specific keys are consumed yet.

## Examples

### Minimal CPU Smoke Train

```yaml
model:
  hidden_dim: 64
  num_layers: 2
  num_heads: 4
  num_kv_heads: 2
  max_seq_len: 128

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

### Train With Eval

```yaml
train:
  eval_every: 500

data:
  train: /data/train_sequences.parquet
  eval:
    heldout:
      path: /data/eval_sequences.parquet
      type: sequence
    structures:
      path: /data/pdb
      type: structure
      eval_every: 2000
      contact_threshold: 8.0
      use_logistic_regression: true
      use_categorical_jacobian: true
      categorical_jacobian_sample_size: 12
```

### Multi-Dataset Train Mix

```yaml
data:
  train:
    uniref50:
      path: /data/uniref50
      fraction: 0.7
    bfd:
      path: /data/bfd
      fraction: 0.3
```

### Checkpoint Resume

```yaml
train:
  output_dir: outputs/medium-run
  resume_from: outputs/medium-run/checkpoint-10000
```

### Inference / Embedding Extraction

CLI:

```bash
oplm encode MKWVTFISLLLLFSSAYS MLPGLALLLLAAWTARA \
  --model outputs/medium-run/checkpoint-10000 \
  --output embeddings.pt
```

Python:

```python
import torch

from oplm.data.tokenizer import ProteinTokenizer
from oplm.inference import load_model_for_inference, resolve_inference_config

checkpoint = "outputs/medium-run/checkpoint-10000"
cfg = resolve_inference_config(checkpoint)
model = load_model_for_inference(checkpoint, cfg)

tokenizer = ProteinTokenizer()
batch = tokenizer.batch_encode(["MKWVTFISLLLLFSSAYS"], max_length=cfg.model.max_seq_len)

with torch.no_grad():
    hidden = model.encoder(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )[0]
```

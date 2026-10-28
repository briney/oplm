# OPLM config reference

The canonical, field-by-field reference for configuring OPLM models and training
runs. For the *semantics* of the architecture toggles (what `norm_strategy`,
Canon convolutions, or partial RoPE actually do), see
[architecture.md](architecture.md). For the eval harness, see
[evaluation.md](evaluation.md). For the training loop, see
[training.md](training.md). For the data behind `data.*`, see [data.md](data.md).

## Two config namespaces

A run config has three top-level blocks, each mapping to a different object:

| Block     | Backing object                              | Notes |
|-----------|---------------------------------------------|-------|
| `model.*` | `oplm.model.OplmConfig` (a HuggingFace `PretrainedConfig`) | The model architecture. Serialized into `config.json`. |
| `train.*` | `oplm.config.TrainConfig` (dataclass)        | Optimizer, schedule, logging, checkpointing. |
| `data.*`  | `oplm.config.DataConfig` (dataclass)         | Datasets, masking, DataLoader. |

Structure-head runs add an optional fourth block, `fold.*`
(`oplm.config.FoldRunConfig`; see [Fold fields](#fold-fields-fold)), which is `null` for MLM runs.

Only the `model.*` block travels with a saved model. `train.*` and `data.*`
describe a *run* and are not part of the `hf/` export.

> **Unknown `model.*` keys are rejected.** `load_config()` validates `model.*`
> keys against the `OplmConfig` constructor signature (plus the HuggingFace
> metadata keys a serialized config carries), so a misspelled key such as
> `hidden_dimm` or `cannon_enabled` under `model:` raises at load time rather than
> being silently retained. `train.*` and `data.*` are likewise validated against
> their dataclasses. (`OplmConfig.from_pretrained()` stays permissive so saved
> checkpoints with extra metadata still load.)

## Merge order

Config sources are merged in this order; later sources win:

1. In-package defaults — `src/oplm/configs/{model,train,data}/base.yaml`
2. Optional size preset — `--preset {50M,170M,400M,800M,1B,3B,6B,12B}`
3. Optional YAML file — `--config path.yaml`
4. Dotlist overrides — e.g. `model.num_hidden_layers=24 train.lr=3e-4`

Notes:

- `model.head_dim`, `model.intermediate_size`, and `model.rope_dim` / `nope_dim`
  are **derived** from the core dimensions unless you set them explicitly (see
  [Derived fields](#derived-and-validated-fields)).
- `train.config_path` is populated automatically when `--config` is used.
- The `max_length` key under `data:` was **removed**. Use `model.max_position_embeddings` as the
  sequence-length setting; passing `max_length` under `data:` raises an error.

## Config assembly

`oplm.config.OplmConfig` is the run config, composing `{model, train, data}` (plus the optional
`fold`). `train` and `data` are OmegaConf-structured dataclasses. `model` is an untyped mapping so
OmegaConf can carry arbitrary HF field keys without a parallel schema. `load_config` merges the
sources in the order above, resolves the `model` subtree to a plain dict and instantiates
`oplm.model.OplmConfig(**model_dict)`, so the HF config owns derivation and validation. Derived
fields are simply omitted unless you set them. The root run config and the model config share a class
name, so import sites alias the model one: `from oplm.model import OplmConfig as OplmModelConfig`.
`train.eval_every` is the canonical global eval cadence.

## Override syntax

```bash
# bare key=value positional overrides (train, info)
oplm train --preset 170M \
  train.max_steps=100_000 \
  data.train=/data/train.parquet

oplm info --preset 1B model.num_hidden_layers=40

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

| Preset  | Parameters | Layers | Hidden | Heads | Head dim |
|---------|-----------:|-------:|-------:|------:|---------:|
| `50M`   |      ~50M   |     16 |    512 |     8 |       64 |
| `170M`  |     ~170M   |     24 |    768 |    12 |       64 |
| `400M`  |     ~400M   |     32 |   1024 |    16 |       64 |
| `800M`  |     ~800M   |     40 |   1280 |    20 |       64 |
| `1B`    |     ~1.6B   |     50 |   1600 |    25 |       64 |
| `3B`    |     ~3.3B   |     64 |   2048 |    32 |       64 |
| `6B`    |       ~6B   |     80 |   2560 |    40 |       64 |
| `12B`   |     ~12.5B  |    100 |   3200 |    50 |       64 |

The recipes set only the core dimensions. The `1B`/`3B`/`6B`/`12B` YAMLs also
carry a commented-out `gradient_checkpointing` block you can uncomment for the
larger sizes; no preset enables it automatically. Run
`oplm info --preset <name>` to print the resolved architecture and exact
parameter count.

## Model fields (`model.*`)

Backed by `oplm.model.OplmConfig`. Defaults shown are the effective defaults
from `src/oplm/configs/model/base.yaml` (the `400M` preset dimensions). The bare `OplmConfig()` fallback is 768 hidden, 12 layers, 12 heads.

### Core dimensions

| Override | Type | Default | Notes |
| --- | --- | --- | --- |
| `model.vocab_size` | `int` | `33` | Tokenizer size. Values other than `33` warn (custom vocabularies are not yet supported). |
| `model.hidden_size` | `int` | `1024` | Residual-stream width. Must be divisible by `num_attention_heads`. |
| `model.num_hidden_layers` | `int` | `32` | Number of transformer blocks. |
| `model.num_attention_heads` | `int` | `16` | Standard multi-head attention (no GQA). |
| `model.head_dim` | `int \| null` | derived | Defaults to `hidden_size // num_attention_heads`. |
| `model.intermediate_size` | `int \| null` | derived | FFN inner width. Defaults to `round_up_to(8/3 · hidden_size, 256)`. |
| `model.max_position_embeddings` | `int` | `1024` | Context length used by train, eval, and inference. |

### Shared-layer looping

`num_hidden_layers` counts unique blocks. The selected contiguous range repeats
while its prefix and suffix run once. One loop reproduces ordinary execution.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `model.num_loops` | `int` | `1` | Total executions of each selected layer, not extra repeats. |
| `model.loop_strategy` | `str` | `stack` | Repeat the complete region (`stack`) or each layer (`interleave`). |
| `model.loop_start` | `int` | `0` | Inclusive zero-based start. |
| `model.loop_end` | `int \| null` | `null` | Exclusive end; null means `num_hidden_layers`. |

Require positive integer loop counts and `0 <= loop_start < loop_end <=
num_hidden_layers` after resolving null. Booleans and fractional indices/counts
are rejected, including for one loop. Effective depth is
`L + (num_loops - 1) * (loop_end - loop_start)`.

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
| `model.norm_strategy` | `str` | `sandwich` | `pre`, `sandwich`, `hybrid`, or `post_sdpa` (run default `sandwich`; dataclass fallback `pre`). |
| `model.qk_norm` | `bool` | `true` | Normalize Q and K before attention. |
| `model.qk_norm_mode` | `str` | `channel` | `channel` (per-head LayerNorm/RMSNorm + fixed `1/√head_dim` scale, the current behavior), `l2` (fp32 L2-normalize over `head_dim` + a learned per-head scale), or `full` (ESM-C's: one bias-free norm over the full `hidden_size`, all heads jointly, + `1/√head_dim`). Inert when `qk_norm=false`. |
| `model.qk_norm_l2_scale_init` | `float \| null` | `null` | `l2` mode only: initial value of the learned per-head scale. `null` initializes to `√head_dim`. Must be positive when set. |
| `model.post_embed_norm` | `bool` | `false` | Apply a norm after the token-embedding lookup. |
| `model.mask_dropout` | `bool` | `false` | Zero every `<mask>` embedding row and rescale surviving embeddings by `(1 − reference_ratio) / (1 − observed_ratio)`. `input_ids` path only; the `inputs_embeds` path is unchanged. |
| `model.mask_dropout_reference_ratio` | `float` | `0.12` | Expected fraction of real tokens that are `<mask>` under the training masking policy (`mask_prob · mask_token_prob`), **not** a fraction of mask tokens to drop. Must satisfy `0 ≤ ratio < 1`. |
| `model.residual_scaling` | `str` | `sqrt_num_layers` | `sqrt_num_layers` (scale each sublayer's residual write by `1/√L`), `esmc` (ESM-C's `1/√(L/36)`), or `none`. |
| `model.residual_gate` | `str` | `none` | Learnable multiplicative gate refining each residual write on top of `residual_scaling`: `none` (no new params), `scalar` (one param per attention/FFN write), or `channel` (`(hidden_size,)` param per write). Run default `none`: under sandwich norm a `channel` gate duplicates the post-norm gain (Oct 2026 μP refresh dropped it at no loss cost). Dataclass fallback `none`. |
| `model.residual_gate_init` | `float` | `1.0` | Initial value for residual gate parameters. Must be finite. |
| `model.attn_output_gate` | `str` | `sigmoid` | Post-SDPA attention output gate (arXiv:2505.06708, G1): `none` (no new params), `sigmoid`, or `silu`. Adds a bias-free `(hidden_size, hidden_size)` `gate_proj` per layer; the merged attention output is multiplied elementwise by `act(gate_proj(x))` before `o_proj`. Run default `sigmoid`; dataclass fallback `none`. |
| `model.value_residual` | `str` | `learnable` | ResFormer value residual (arXiv:2410.17897): `none` (no new params), `fixed` (constant λ buffer, no new params), or `learnable` (one scalar λ per layer after the first). Layer 0 exposes its post-V-norm values `v₁`; every later layer blends `v' = λ·v + (1 − λ)·v₁` right after the V projection. Run default `learnable`; dataclass fallback `none`. |
| `model.value_residual_lambda_init` | `float` | `0.5` | Constant λ under `fixed`; initial value of the learnable λ under `learnable`. Must be finite. Inert when `value_residual=none`. |
| `model.init_scale_output_projections` | `bool` | `true` | Shrink residual-writing projections by `1/√(2L)` at init. |

### Feed-forward and dropout

| Override | Type | Default | Valid values / notes |
| --- | --- | --- | --- |
| `model.ffn_activation` | `str` | `swiglu` | `swiglu` / `geglu` (gated, 3 projections) or `relu2` (non-gated squared-ReLU, 2 projections; derived `intermediate_size` uses ~4·D instead of ~8/3·D for param parity). |
| `model.ffn_bias` | `bool` | `false` | Bias terms on FFN projections. |
| `model.attention_dropout` | `float` | `0.0` | Dropout on attention probabilities. |
| `model.hidden_dropout` | `float` | `0.0` | Dropout on hidden states. |

### Embeddings and MLM head

| Override | Type | Default | Valid values / notes |
| --- | --- | --- | --- |
| `model.tie_word_embeddings` | `bool` | `false` | Tie the MLM decoder to the input embedding. |
| `model.mlm_head_activation` | `str` | `gelu` | `gelu`, `silu`, or `relu`. |

### Canon depthwise convolution (on by default)

| Override | Type | Default | Valid values / notes |
| --- | --- | --- | --- |
| `model.canon_enabled` | `bool` | `true` | Master switch for Canon conv sublayers. Supported under `norm_strategy` in `{pre, sandwich, post_sdpa}`; `hybrid` raises at validation (no outer attention pre-norm for Canon-A). Run default `true`; dataclass fallback `false`. |
| `model.canon_residual` | `bool` | `true` | Use residual Canon updates (`z + Canon(z)`) by default. |
| `model.canon_positions` | `list[str]` | `[A, B, C, D]` | Subset of `{A, B, C, D}`; required and non-empty when `canon_enabled=true`. Position semantics are specified in [architecture.md, *Canon semantics*](architecture.md#canon-semantics). Run default all four positions; dataclass fallback `[]`. |
| `model.canon_kernel_sizes` | `int \| list[int] \| dict` | `7` | `int` broadcasts to every layer; a `list[int]` must have length `num_hidden_layers`; a `dict` uses `schedule: linear` (`min`, `max`) or `schedule: constant` (`value`). All entries must be `≥ 2`. Run default `7`; dataclass fallback `4`. |
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
| `model.gradient_checkpointing_mode` | `str` | `full` | `full` \| `selective`. `full` recomputes the whole block (max memory savings, ~+30% compute); `selective` keeps matmul/SDPA/FlexAttention outputs resident and recomputes only cheap ops (less memory savings, much less extra compute). Inert unless `gradient_checkpointing` is `true`. On multi-GPU with `train.compile`, `selective` makes the trainer auto-disable DDPOptimizer (`torch._dynamo.config.optimize_ddp = False`) so SAC isn't fragmented into a full recompute — see [training.md](training.md#what-a-training-step-does). |
| `model.packed_attention` | `str` | `sdpa` | Attention kernel for packed batches (`data.pack_sequences`). `sdpa` passes a dense block-diagonal mask, so cost grows with the row width. `flex` uses FlexAttention with a block mask built once per forward, skipping the 128×128 blocks between sequences; it needs `train.compile` for speed on GPU (with `train.compile_dynamic: false` it recompiles per new batch shape), has no attention dropout, and has no backward on CPU. Eval (unpacked) batches always use SDPA. |
| `model.pad_token_id` | `int` | `1` | `<pad>`. |
| `model.bos_token_id` | `int` | `0` | `<cls>` (prepended to every sequence). |
| `model.eos_token_id` | `int` | `2` | `<eos>`. |
| `model.unk_token_id` | `int` | `3` | `<unk>`. |
| `model.mask_token_id` | `int` | `32` | `<mask>`. |

### μP (maximal update parametrization)

μP makes a learning rate tuned on a small proxy model transfer to much larger
models without re-sweeping at every scale. It is **on by default** for `oplm train`
runs (with `train.optimizer=muon`, `train.muon_adjust_lr_fn=original`,
`train.lr=0.0063`), and a no-op at the base width. See [mup-lr.md](mup-lr.md) for the full
recipe and the tune-once-reuse-`train.lr` workflow, and for retuning the ESM-C comparison
recipe (`configs/train/esm-c.yaml`, AdamW-only μP).

> **Run defaults vs. dataclass fallbacks.** The `Default` column below (and in the
> Train table) is the **run default** from `configs/*/base.yaml` — what `oplm train`
> loads. Bare Python construction (`OplmConfig()` / `TrainConfig()`) and
> `from_pretrained` use conservative fallbacks instead (`mup_enable=false`,
> `optimizer=adamw`, `lr=1e-4`, `norm_strategy=pre`, `residual_gate=none`,
> `attn_output_gate=none`, `value_residual=none`, `canon_enabled=false`), so
> direct use and existing checkpoints are unaffected.

| Override | Type | Default | Valid values / notes |
| --- | --- | --- | --- |
| `model.mup_enable` | `bool` | `true` | Master switch (run default; dataclass fallback `false`). When `false`, init, forward multipliers, and per-group LRs are identical to non-μP runs. |
| `model.mup_base_width` | `int` | `768` | `hidden_size` of the proxy/base model where `m = hidden_size / mup_base_width` equals 1. Matches the production sweep's forced 768-wide (170M) anchor; dataclass fallback `512`. Must be `≥ 1`. |
| `model.mup_output_mult` | `float` | `1.0` | Tunable O(1) multiplier on the output logits (applied as `mup_output_mult / m` on the readout matmul path). Run default is the μP refresh winner (Oct 2026); dataclass fallback `1.0`. Must be `> 0`. |

### Derived and validated fields

When omitted, derived fields resolve from the core dimensions:

- `head_dim = hidden_size // num_attention_heads`
- `intermediate_size = round_up_to(8/3 · hidden_size, 256)`
- `rope_dim = head_dim`, `nope_dim = 0` (full RoPE)

Config construction raises `ValueError` if any of these fail:

- `num_attention_heads > 0` and `hidden_size % num_attention_heads == 0`
- `head_dim · num_attention_heads == hidden_size`
- `rope_dim + nope_dim == head_dim`, both `≥ 0`, and `rope_dim` even
- enum fields (`norm_type`, `norm_strategy`, `qk_norm_mode`, `residual_scaling`,
  `residual_gate`, `attn_output_gate`, `value_residual`, `ffn_activation`,
  `mlm_head_activation`, `canon_activation`, `classifier_pool`) hold a valid value
- `0 ≤ mask_dropout_reference_ratio < 1`
- `qk_norm_l2_scale_init`, when set, is positive; `residual_gate_init` and
  `value_residual_lambda_init` are finite
- when `canon_enabled`, `canon_positions` is a non-empty, duplicate-free subset
  of `{A, B, C, D}`
- `mup_base_width ≥ 1` and `mup_output_mult > 0`

## Train fields (`train.*`)

Backed by `oplm.config.TrainConfig`.

| Override | Type | Default | Valid values / notes |
| --- | --- | --- | --- |
| `train.max_steps` | `int` | `50_000` | Step budget; used when `max_epochs` is unset. |
| `train.max_epochs` | `int \| null` | `null` | Optional epoch-based stop. |
| `train.batch_size` | `int` | `32` | Per-process batch size, before accumulation (eval batch size in token mode). |
| `train.tokens_per_step` | `int \| null` | `null` | Token mode, with `train.batch_tokens`: global token capacity per optimizer step. Must be a multiple of `batch_tokens` × processes; the quotient becomes `gradient_accumulation_steps` at startup. Needs `max_steps` (not `max_epochs`) and `data.pack_sequences: true`. |
| `train.batch_tokens` | `int \| null` | `null` | Token mode: each process's micro-batch capacity, a multiple of `model.max_position_embeddings` (rows × width). Every micro-batch is exactly that shape. `train.batch_size` then only sizes eval batches. |
| `train.gradient_accumulation_steps` | `int` | `1` | Must be `≥ 1`. |
| `train.optimizer` | `str` | `muon` | `adamw` or `muon` (hybrid Muon + AdamW). Run default `muon` (dataclass fallback `adamw`). |
| `train.lr` | `float` | `0.0063` | Peak learning rate. With μP on (the default) this is the **μP base LR** (μP refresh winner, Oct 2026, at 4M tokens/step) — it transfers across width, so do not retune per size; retune if μP is disabled or the batch is very different. Dataclass fallback `1e-4`. |
| `train.min_lr` | `float` | `0.0` | Must be `≥ 0` and `≤ lr`. |
| `train.weight_decay` | `float` | `0.01` | Decoupled weight decay. |
| `train.adam_beta1` | `float` | `0.9` | AdamW β₁ (also Muon's auxiliary AdamW). |
| `train.adam_beta2` | `float` | `0.98` | AdamW β₂. |
| `train.adam_eps` | `float` | `1e-8` | AdamW ε. |
| `train.muon_adjust_lr_fn` | `str` | `original` | `match_rms_adamw` or `original`. Run default `original` (required by μP+Muon); dataclass fallback `match_rms_adamw`. |
| `train.muon_momentum` | `float` | `0.95` | Must be `≥ 0`. |
| `train.muon_nesterov` | `bool` | `true` | Muon Nesterov momentum. |
| `train.muon_ns_steps` | `int` | `5` | Newton–Schulz steps; must be `≥ 1`. |
| `train.mup_depth_lr_exponent` | `float` | `0.5` | Repeated-block LR exponent; finite and `>= 0`. Zero is a no-op; the run default `0.5` is the sweep-verified correction (no-op at the 24-layer reference depth). Dataclass fallback `0.0`. |
| `train.mup_depth_reference_layers` | `int` | `24` | Reference layer count in `(reference/L)^exponent`; must be `>= 1`. |
| `train.adamw_lr_mult` | `float` | `0.333` | Multiplier on the AdamW-side LR only (under Muon: embeddings, head, norms, biases, residual gates, Canon kernels; under `adamw`: every group). Muon matrices are untouched. Must be finite and `> 0`. Run default `0.333` (μP refresh); dataclass fallback `1.0`. |
| `train.embedding_weight_decay` | `float` | `0.01` | Decoupled weight decay on the token embedding, toward 0 (`0`: undecayed). Finite and `>= 0`. Run default `0.01`; dataclass fallback `0`. |
| `train.weight_decay_to_init` | `float` | `0.1` | Decoupled weight decay toward each parameter's initial value: norm gains toward 1, residual gates toward `model.residual_gate_init`, biases toward 0 (`0`: undecayed). Value-residual lambdas and L2 QK scales are never decayed. Finite and `>= 0`. Turning this or `embedding_weight_decay` on or off changes the AdamW param groups, so a resume or branch must keep the setting (on or off) its checkpoint was saved with; changing a nonzero value is fine. Run default `0.1`, which bounds the sandwich post-norm gains on long runs; dataclass fallback `0`. |
| `train.max_grad_norm` | `float` | `1.0` | Set `0` to disable clipping. |
| `train.scheduler` | `str` | `warmup_linear` | `warmup_linear`, `warmup_cosine`, `wsd_linear`, or `wsd_cosine`. |
| `train.warmup_steps` | `int` | `5_000` | Must be `≥ 0`. |
| `train.stable_steps` | `int` | `0` | WSD plateau length; must be `≥ 0`. |
| `train.log_every` | `int` | `10` | Train-metric logging cadence (optimizer steps). |
| `train.eval_every` | `dict \| null` | `null` | Default eval cadence for datasets that omit `every`. Exactly one of `{steps: N}` / `{tokens: N}`. `null` resolves to `{steps: 10_000}`. |
| `train.wandb_project` | `str` | `oplm` | W&B project. |
| `train.wandb_run_name` | `str \| null` | `null` | W&B run name (also settable via `--name`). |
| `train.wandb_enabled` | `bool` | `true` | Toggle W&B logging. |
| `train.save_final` | `bool` | `true` | Write a final checkpoint when training completes, even if the last step is not a multiple of `save_every`. Skipped when the last step already saved. |
| `train.save_every` | `int` | `10_000` | Checkpoint cadence (optimizer steps). |
| `train.save_total_limit` | `int` | `3` | Rolling checkpoints to keep. Permanent checkpoints (see `keep_every_n_steps` / `keep_every_n_hours`) are exempt from both the count and rotation. |
| `train.resume_from` | `str \| null` | `null` | Path to an Accelerate checkpoint directory. Explicit; always wins over `auto_resume` when both are set. |
| `train.branch_from` | `str \| null` | `null` | Branch a new run off a checkpoint directory: restores model, optimizer state, step counters, and (if `data.train` is unchanged) the data position like `resume_from`, but the LR, `adamw_lr_mult`, weight decay, and schedule come from **this** config. Needs a fresh `output_dir`; mutually exclusive with `resume_from`/`init_from`; a resolved `auto_resume` checkpoint in `output_dir` takes precedence (requeued branches resume themselves). See [training.md, *Branching a decay off a checkpoint*](training.md#branching-a-decay-off-a-checkpoint). |
| `train.save_every_minutes` | `int \| null` | `null` | Also checkpoint every N wall-clock minutes, in addition to (not instead of) the `save_every` step cadence. `null` disables the timer. Must be `> 0` when set. |
| `train.keep_every_n_steps` | `int \| null` | `null` | Checkpoints whose step is a multiple of this value are permanent — excluded from `save_total_limit` rotation. `null` disables the exemption. Must be `> 0` when set. |
| `train.keep_every_n_hours` | `float \| null` | `null` | Marks a checkpoint permanent at least this many wall-clock hours after the previous permanent one. Independent of `keep_every_n_steps`. `null` disables the exemption. Must be `> 0` when set. |
| `train.auto_resume` | `bool` | `false` | Resume from the newest committed checkpoint under `output_dir` automatically at trainer start, with no `resume_from` needed. `oplm slurm generate` injects `true` into every rendered job (see [slurm.md, *Requeue semantics*](slurm.md#requeue-semantics-drain-budget-and-the-no-progress-guard)). |
| `train.resume_data_position` | `bool` | `true` | **Data-exact resume.** When `true` and the checkpoint carries a data cursor, resume replays the exact row-level data position instead of restarting the current epoch from row 0 — no row re-seen or skipped, same order as an uninterrupted run. The cursor's recorded layout (world size, worker count, per-rank batch size, seed) must match the live run or resume raises. Set `false` to opt back into restart-from-row-0 (the escape hatch for a different layout, or an older checkpoint written before the data cursor existed). See [training.md, *Fault tolerance*](training.md#fault-tolerance). |
| `train.dist_timeout_minutes` | `int` | `15` | Timeout (minutes) passed to Accelerate's `InitProcessGroupKwargs`, bounding every NCCL/gloo collective's wait. Must be `> 0`. |
| `train.remote_checkpoint_uri` | `str \| null` | `null` | An fsspec URI (`s3://`, `gs://`, `file://`, ...) that every committed checkpoint is additionally mirrored to in the background, and that `auto_resume` also consults. `null` disables it entirely (zero behavior change). See [training.md, *Remote checkpoint mirror*](training.md#remote-checkpoint-mirror). |
| `train.ema_decay` | `float \| null` | `null` | Exponential moving average of the trainable weights, updated once per successful optimizer step (never on accumulation micro-steps or a skipped fp16 step). Every checkpoint then also writes `ema.pt` (tracker state + update count, restored on resume) and an `hf_ema/` export next to `hf/`; a later stage selects it with `train.init_from=<ckpt>/hf_ema`. Must be in `(0, 1)` when set. `ddp` only (refused under `hsdp`). See [training.md, *Fault tolerance*](training.md#fault-tolerance). |
| `train.parallelism` | `str` | `ddp` | `ddp` (one full model replica per rank, gradients all-reduced) or `hsdp` (FSDP2 `fully_shard` over a 2-D mesh: shard within a node, replicate across nodes; requires world size `> 1`). Checkpoints are parallelism-agnostic, so the same checkpoint resumes under either setting at any world size. `hsdp` refuses three combinations that would otherwise hang or silently diverge: configured `data.eval`, `mixed_precision=fp16`, and `stability_diagnostics` with `stability_probe_every > 0`. See [training.md, *Fault tolerance*](training.md#fault-tolerance) for the full memory/limitation picture. |
| `train.seed` | `int` | `42` | Global seed. |
| `train.output_dir` | `str` | `outputs` | Base dir for logs and checkpoints. |
| `train.mixed_precision` | `str` | `bf16` | `bf16`, `fp16`, or `no`. |
| `train.compile` | `bool` | `false` | Enable `torch.compile` (opt-in; adds first-step latency). |
| `train.compile_mode` | `str` | `default` | Compile mode: `default` \| `reduce-overhead` \| `max-autotune`. |
| `train.compile_dynamic` | `bool \| null` | `true` | `torch.compile(dynamic=...)`: `true` = one dynamic graph (default), `false` = static graph per shape (best with `data.pad_to_multiple_of` to bound shapes), `null` = Dynamo auto. When `false`, the trainer raises `torch._dynamo.config.cache_size_limit` to accommodate the expected number of shape buckets. |
| `train.weight_diag_every` | `int` | `0` | Every N optimizer steps, log per-group weight RMS (`diag/weight_rms/<group>`) and update-RMS / weight-RMS (`diag/update_ratio/<group>`); groups are `embed`, `head_dense`, `head_decoder`, `attn`, `mlp`, `norm_gain`, `residual_gate`, `canon`, `bias`, `other`. `0` disables. Use a multiple of `log_every`. `ddp` only (refused under `hsdp`). Backfill from checkpoints offline with `oplm weight-rms <ckpt>...`. See [training.md, *Logging and monitoring*](training.md#logging-and-monitoring). |
| `train.stability_diagnostics` | `bool` | `false` | Attach the stability-diagnostics callback: logs the pre-clip global grad norm every log, and every `stability_probe_every` logs runs one eager diagnostic forward on the unwrapped model recording per-depth residual-stream RMS, output-logit RMS and attention entropy under `diag/*`. For deep-model μP debugging. Refused with `hsdp` when `stability_probe_every > 0`. |
| `train.stability_probe_every` | `int` | `25` | Probe cadence in log emissions. `0` logs only the grad norm. Must be `≥ 0`; consulted only when `stability_diagnostics` is `true`. |
| `train.profile_start` | `int \| null` | `null` | Opt-in profiling: from this optimizer step, CUDA-sync-time each phase of `profile_steps` steps, then `torch.profiler`-trace the next `profile_steps` steps; results go to `<output_dir>/profile/`. Also times all-reduces at train start. `null` disables. Must be `≥ 1` when set. |
| `train.profile_steps` | `int` | `20` | Length of each profiling window; must be `≥ 1`. |
| `train.throughput_warmup_steps` | `int` | `50` | Exclude the first N optimizer steps from throughput accounting (covers compile/warmup latency). Set to `0` to include all steps. Must be `≥ 0`. |
| `train.peak_tflops` | `float \| null` | `null` | Device peak throughput in TFLOPs (e.g. `1979.0` for H100 BF16). When set, the trainer also logs `train/mfu` (achieved TFLOPs / peak). `null` → log only `train/achieved_tflops`. Must be `> 0` when set. |
| `train.config_path` | `str \| null` | `null` | Auto-populated with the source path passed to `--config`. |

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
| `data.pack_sequences` | `bool` | `false` | Training batches only: pack each batch's sequences back to back into as few rows of at most `model.max_position_embeddings` tokens as fit. The model keeps sequences apart (block-diagonal attention, Canon taps stop at boundaries), so the loss is unchanged. `train.batch_size` still counts sequences. Eval batches are never packed. See [data.md, *Truncation, padding, packing*](data.md#truncation-padding-packing). |
| `data.pad_to_multiple_of` | `int \| null` | `null` | Pad each batch to the smallest multiple of N ≥ batch-longest sequence. `null` = pad to batch-longest (default behavior). Must be `≥ 1` when set. **Divisibility requirement:** `model.max_position_embeddings % pad_to_multiple_of == 0` is enforced at config load to prevent position-id overflow. Pairs with `train.compile_dynamic=false` + large N (64/128/256) to bound the Dynamo shape space to a small fixed set of static graphs. |
| `data.num_workers` | `int` | `4` | DataLoader workers. |
| `data.pin_memory` | `bool` | `true` | Pin host memory for batches. |
| `data.prefetch_factor` | `int` | `4` | Batches prefetched per worker. |
| `data.shuffle_shards` | `bool` | `true` | Shuffle shard order (sharded parquet directories). |
| `data.shuffle_rows` | `bool` | `true` | Shuffle rows within each shard. |
| `data.reshuffle_on_refill` | `bool` | `true` | A mixed source sampled above its row share runs out mid-epoch and is read again: `true` gives each such pass its own order, `false` replays the epoch's order every pass (the pre-October-2026 behavior, for replaying older runs' data streams). |
| `data.drop_last` | `bool` | `true` | Give every (rank, worker) stream the same number of rows per epoch (dropping the remainder, fewer than `world_size × num_workers` rows) and drop each worker's partial last batch, so every rank runs the same number of full batches. `false` is the earlier behavior: up to one extra row per stream and short tail batches. |

### Training data format

Training and sequence-eval data are parquet with `sequence_id` and `sequence` columns, plus
`masking_weights` when `weighted_masking=true`; the column contract is in
[data.md, *Dataset formats*](data.md#dataset-formats).

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
- `type` — one of `sequence`, `structure`, `proteingym`, `proteingym_clinical`,
  `tape`, `proteinglue`, `everest`.
- `every` — optional per-dataset cadence: exactly one of `{steps: N}` /
  `{tokens: N}`, plus optional `at_start` (default `false`) and `at_end`
  (default `true`). Datasets that omit `every` inherit `train.eval_every`.
- `metrics` — optional list of metric names to keep.

Task-specific keys sit at the **same level** as `path` and `type` (not nested
under `extra:`). See [evaluation.md](evaluation.md) for the full harness
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
| `min_seq_sep` | `int` | `24` | Minimum sequence separation for scoring (long-range contacts, matching the reference paper). |
| `l_divisor` | `int` | `1` | Score the top `L / l_divisor` contacts. |
| `use_cbeta` | `bool` | `true` | Use virtual C-β distances. |
| `categorical_jacobian_sample_size` | `int \| null` | `null` | Optional deterministic subset of structures (the Jacobian costs `L × 20` forwards each). |
| `categorical_jacobian_sample_seed` | `int` | `42` | Seed for subset sampling. |
| `categorical_jacobian_mutation_batch_size` | `int` | `20` | Mutant sequences per Jacobian forward pass. |
| `max_structures` | `int \| null` | `null` | Optional cap on structures loaded. |

### proteingym

Zero-shot DMS variant-effect prediction — `spearman`, `auroc` (vs
`DMS_score_bin`), and `top_k_precision`, macro-averaged across assays. Reads a
directory of DMS substitution CSVs (`mutant`, `DMS_score`, `mutated_sequence`;
optional `DMS_score_bin`); the wild-type is reconstructed from `mutated_sequence`.

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `scoring` | `str` | `masked_marginals` | `masked_marginals` (mask each mutated position) or `wt_marginals` (single wild-type forward). |
| `mask_batch_size` | `int` | `64` | Masked sequences per forward pass (`masked_marginals` only); must be in `[1, 1024]`. |
| `top_k_fraction` | `float` | `0.1` | Top fraction (by `DMS_score`) for `top_k_precision`; must be in `(0, 1]`. |
| `max_assays` | `int \| null` | `null` | Optional cap on assays (CSV files) loaded. |

### proteingym_clinical

Zero-shot clinical-variant pathogenicity — `auroc` (Pathogenic vs Benign),
macro-averaged across per-protein assays. Reads a directory of one-protein
clinical-substitution CSVs (`protein_sequence`, `mutant`, `DMS_bin_score`).

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `scoring` | `str` | `masked_marginals` | `masked_marginals` (mask each mutated position) or `wt_marginals` (single wild-type forward). |
| `mask_batch_size` | `int` | `64` | Masked sequences per forward pass (`masked_marginals` only); must be in `[1, 1024]`. |
| `max_assays` | `int \| null` | `null` | Optional cap on assays (CSV files) loaded. |

### tape, proteinglue, everest

Registered but currently documented stubs; they consume no task-specific keys
yet. See [evaluation.md](evaluation.md), *Tasks and metrics* for status.

## Fold fields (`fold.*`)

Backed by `oplm.config.FoldRunConfig`; the structure-head training block
([FOLD.md, *Milestone 2b: losses and training*](FOLD.md#9-milestone-2b-losses-and-training-oplmfoldlosses-oplmfoldtraining)). The
block is absent (`null`) for MLM runs, so `oplm train` is unchanged; `oplm fold train`,
`python -m oplm.fold.train` and `oplm fold measure` require it and raise without it. Unknown
`fold.*` keys are rejected at load time. In a fold run every `data.train` entry is a structure
shard directory (see [FOLD.md, *Milestone 2a: training data*](FOLD.md#8-milestone-2a-training-data-oplmfolddata)) and the
top-level `model.*` block (the MLM architecture) is not used.

| Override | Type | Default | Valid values / notes |
| --- | --- | --- | --- |
| `fold.lm` | `str` | `esmc:esmc_600m` | The frozen language model: `esmc:<name>` (`esmc_300m`, `esmc_600m`, `esmc_6b`; needs `oplm[esmc]` on Python `≥ 3.12`; the `biohub/ESMC-*` weights are public, MIT-licensed and ungated), an OPLM checkpoint path, or a Hub id. Must be non-empty. |
| `fold.lm_revision` | `str \| null` | `null` | Hub revision of the LM weights; recorded in the saved head's config. |
| `fold.lm_max_chain_length` | `int` | `1024` | Longest full chain the data pipeline samples (the LM reads the whole chain, not the crop); `≥ 1`. |
| `fold.model` | `dict` | `{}` | `FoldConfig` overrides (`pair_width`, `trunk_blocks`, `inference_num_loops`, ...). Keys are `FoldConfig` constructor names, validated when the model is built (`build_fold_config` in `oplm.fold.training`): an unknown key raises, and the LM identity fields (`lm_name_or_path`, `lm_revision`, `lm_max_chain_length`) come from `fold.lm*` and may not be set here. Unset fields take the ESMFold2-Fast defaults; `lm_hidden_size` and `lm_num_hidden_states` are inferred from the loaded LM. |
| `fold.crop_size` | `int` | `384` | Tokens per crop; a positive multiple of 128. |
| `fold.contiguous_prob` | `float` | `0.5` | Probability a crop is a contiguous window rather than a spatial crop; in `[0, 1]`. |
| `fold.atoms_per_token_budget` | `int` | `24` | Atom-axis budget per token: batches pad atoms to `atoms_per_token_budget × crop_size` rounded up to `fold.model.atom_pad_multiple` (32), and an example over the budget is rejected, not truncated; `≥ 1`. |
| `fold.diffusion_samples` | `int` | `8` | Noised copies of each example denoised per step; `≥ 1`. |
| `fold.diffusion_microbatch` | `int \| null` | `null` | Samples per denoiser call. Below `diffusion_samples`, each chunk runs under activation checkpointing so one chunk's activations are live at a time; `null` (or `≥ diffusion_samples`) denoises all samples in one unchecked call, and activations scale with `diffusion_samples`. `≥ 1` when set. |
| `fold.confidence_rollout_steps` | `int` | `20` | Sampler steps of the no-grad rollout the confidence head is scored on; `≥ 1`. |
| `fold.loss_diffusion_mse` | `float` | `4.0` | Weight of the EDM-weighted diffusion MSE; `≥ 0`. |
| `fold.loss_smooth_lddt` | `float` | `1.0` | Weight of the smooth-lDDT term; `≥ 0`. |
| `fold.loss_distogram` | `float` | `0.03` | Weight of the distogram cross-entropy; `≥ 0`. |
| `fold.loss_confidence` | `float` | `1e-4` | Weight of the confidence composite (pLDDT + PDE + resolved + `loss_pae` × PAE); `≥ 0`. |
| `fold.loss_pae` | `float` | `1.0` | Weight of the PAE sub-term inside the confidence composite; `≥ 0`. |

Set `train.optimizer=adamw` explicitly in a fold run: the packaged `train` defaults are μP + Muon
(`optimizer: muon`, `muon_adjust_lr_fn: original`), tuned for the MLM, and the head trains with plain
AdamW. [`configs/fold/proof-w128.yaml`](../configs/fold/proof-w128.yaml) and
[`proof-w256.yaml`](../configs/fold/proof-w256.yaml) are the complete milestone-2 run configs.

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

See [inference.md](inference.md) for more inference examples.

`train.init_from` (default `null`) initializes a fresh training stage from a local
HF export or a checkpoint directory containing `hf/`. Only weights and persistent
model buffers transfer. Configure a new output directory and stage-local LR,
warmup, and step budget. A resolved full-state resume always takes precedence and
never reads the initialization source. See [the staged training example](training.md#training-a-separate-looped-stage).

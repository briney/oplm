# OPLM Trainer Overview

> Founding reference for the OPLM trainer — the loop that pretrains an
> `OplmForMaskedLM` with HuggingFace Accelerate, step/epoch/token-governed
> cadences, `wandb` logging, rich progress, callbacks, and checkpointing. This
> document specifies the **target** design for the rewritten trainer: it first
> describes how the *current* trainer works end to end, then defines the
> redesign that adapts it to the rewritten model, data tooling, and eval
> harness, streamlines redundant code, and fixes the bugs that went undetected
> in the prior trainer. It is the sole source of truth for the trainer design
> and is the `docs/TRAINER.md` referenced by [`EVAL_HARNESS.md`](EVAL_HARNESS.md) §1.2.
>
> The model and its forward pass live in [`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md);
> the data loaders, on-disk formats, tokenizer, and masking live in
> [`DATA_TOOLING.md`](DATA_TOOLING.md); the eval harness and the trainer↔eval
> contract live in [`EVAL_HARNESS.md`](EVAL_HARNESS.md). This document owns the
> training loop, optimizer/schedule construction, FLOPs accounting, config
> assembly, and the checkpoint format.

---

## 1. Scope and design philosophy

### 1.1 What this document covers

- The modules under `src/oplm/training/` — `Trainer`, optimizer/scheduler
  construction (`optim.py`), checkpointing (`checkpoint.py`), FLOPs accounting
  (`flops.py`), and the callback surface (`callbacks.py`).
- The **config assembly** path: how `oplm.config.load_config` composes
  defaults → preset → YAML → CLI overrides into the run config, and how the
  `model:` block becomes the HuggingFace `oplm.model.OplmConfig` the model
  consumes.
- The **entry points** (`oplm.train.main`, the `oplm.cli` `train`/`encode`/`info`
  subcommands) and the inference helpers in `oplm.inference` that share the
  config and model-construction path.
- The **checkpoint format**: resumable Accelerate state plus a
  `from_pretrained`-loadable HuggingFace export.

### 1.2 What this document does not cover

- The model architecture and forward signature. → [`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md).
- Data loaders, formats, tokenizer, masking. → [`DATA_TOOLING.md`](DATA_TOOLING.md).
- The eval harness internals and metrics. → [`EVAL_HARNESS.md`](EVAL_HARNESS.md).
  This document specifies only the *interface* the trainer uses to drive eval (§4.7).

### 1.3 Design principles

1. **One config schema per concern.** The model owns its hyperparameters via the
   HF `OplmConfig` (`PretrainedConfig`); the trainer and data tooling own theirs
   via the `TrainConfig` / `DataConfig` dataclasses. The run config composes the
   three; it does not re-describe the model.
2. **The trainer owns the clock; the harness owns the policy.** The trainer
   advances state and announces "we are at step N / token M"; the eval harness
   decides what is due. Per-task cadence never leaks into the loop (see
   [`EVAL_HARNESS.md`](EVAL_HARNESS.md) §1.3).
3. **Accelerate is the only distribution layer.** No manual `torch.distributed`
   wiring in the trainer; gradient accumulation, mixed precision, device
   placement, and state save/load go through the `Accelerator`.
4. **Steps are the native clock.** Training duration and every cadence are
   step-governed by default; epochs and tokens are derived overlays. Epoch-based
   *eval* cadence remains deferred (see [`EVAL_HARNESS.md`](EVAL_HARNESS.md) §4.6).
5. **Checkpoints are reproducible and portable.** Resume restores exact training
   state; a parallel HF export makes any checkpoint loadable with
   `OplmForMaskedLM.from_pretrained`.

---

## 2. The current trainer, end to end

This section documents the trainer as it exists today (`src/oplm/training/trainer.py`),
including the parts that are now broken against the rewritten model/data/eval
(flagged inline and consolidated in §3).

### 2.1 Entry points

- **`oplm.train.main(cfg=None)`** (`src/oplm/train.py:84`) — bootstraps the
  training environment (creates a writable `TRITON_CACHE_DIR`, disables DeepSpeed
  unless `OPLM_ENABLE_DEEPSPEED` is set), loads the config from `sys.argv[1:]`
  via `load_config` when `cfg` is `None`, then `Trainer(cfg).train()`. Launched
  directly (`python -m oplm.train --config …`) or distributed
  (`accelerate launch -m oplm.train --config … model.num_hidden_layers=32`).
- **`oplm.cli.train`** (`src/oplm/cli.py:46`) — a Typer wrapper that builds the
  same `argv` from `--config/--preset/--override`, prints a one-line model
  summary, and delegates to `oplm.train.main`.

### 2.2 `Trainer.__init__` — setup order

`src/oplm/training/trainer.py:33`. In order:

1. `set_seed(cfg.train.seed)`.
2. Build the `Accelerator` with `mixed_precision`, `gradient_accumulation_steps`,
   `log_with="wandb"` (when enabled), `project_dir=output_dir`,
   `DataLoaderConfiguration(dispatch_batches=False)`, and
   **`step_scheduler_with_optimizer=False`** (the trainer steps schedulers
   manually).
3. Initialize wandb trackers early (so the login prompt precedes slow setup),
   flattening the config via `_config_to_flat_dict`.
4. Build the `Evaluator` from `cfg` when `cfg.data.eval is not None`.
5. Build the model: `OplmForMaskedLM(cfg.model)`; enable gradient checkpointing
   if requested.
6. `build_optimizers(model, cfg.train)` and `build_train_dataloader(cfg)`.
7. `_compute_total_steps(cfg, dataloader)` then `build_schedulers(...)`.
8. `accelerator.prepare(model, *optimizers, dataloader, *schedulers)` and unpack.
9. Initialize state: `global_step=0`, `epoch=0`, `tokens_seen=0`,
   `_samples_seen=0`, `_epoch_at_last_opt_step=0`, `_step_local_tokens=0`,
   `flops_per_token=estimate_flops_per_token(cfg.model)`.
10. `_resume_from_checkpoint(cfg.train.resume_from)` when set.

### 2.3 The training loop

`src/oplm/training/trainer.py:138`. A single `while self.global_step < self.total_steps`
loop:

```python
with self.accelerator.accumulate(self.model):
    outputs = self.model(input_ids=…, attention_mask=…, labels=…)
    loss = outputs["loss"]
    self.accelerator.backward(loss)
    if cfg.max_grad_norm > 0 and self.accelerator.sync_gradients:
        self.accelerator.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
    for optimizer in self.optimizers: optimizer.step()
    for optimizer in self.optimizers: optimizer.zero_grad()

self._step_local_tokens += int(batch["attention_mask"].sum().item())
self._samples_seen += len(batch["input_ids"]) * self.accelerator.num_processes
if not self.accelerator.sync_gradients:
    continue                       # micro-step: skip cadence work

for scheduler in self.schedulers: scheduler.step()
self.global_step += 1
current_loss = loss.item()
# rank-reduce this step's tokens → tokens_delta / tokens_seen are rank-identical
…
if self.global_step % cfg.log_every == 0: self._log_step(current_loss)
eval_metrics = self._run_eval(tokens_delta)
if eval_metrics: …; self._log_metrics(eval_metrics); self._emit_eval_end(eval_metrics)
if self.global_step % cfg.save_every == 0: self._save_checkpoint()
```

Key invariants:

- **Accumulation boundary**: cadence work (scheduler step, `global_step++`,
  logging, eval, checkpointing) runs only when `accelerator.sync_gradients` is
  true — i.e. once per *optimizer* step, not per micro-batch.
- **Epoch rollover**: `StopIteration` from the iterator increments `epoch`, calls
  `_set_dataset_epoch(epoch)` (deterministic reshuffle), and re-creates the
  iterator.
- **Rank-identical token accounting**: local tokens accumulate across micro-steps,
  then `accelerator.reduce(..., "sum")` produces a `tokens_delta` that is summed
  into `tokens_seen`. Both are identical on every rank — the invariant the
  `EvalContext` relies on (§4.7).
- The loop saves a final checkpoint, stops the progress bar, emits `on_train_end`,
  and calls `accelerator.end_training()` in a `finally`.

### 2.4 Optimizer & parameter grouping

`src/oplm/training/optim.py`. `partition_optimizer_params` splits trainable
params into three groups:

- **no-decay** — `param.ndim <= 1 or "embed" in name` (biases, norms, embeddings).
- **Muon** — 2-D weights when `optimizer="muon"`, excluding the MLM head.
- **AdamW-decay** — everything else (2-D weights).

`build_optimizers` returns `[AdamW]` for `optimizer="adamw"` (two param groups:
decay with `weight_decay`, no-decay with `0.0`), or `[Muon, auxiliary AdamW]` for
`optimizer="muon"`. `torch.optim.Muon` is available (torch ≥ 2.11).

### 2.5 Learning-rate schedules

`get_schedule_fn` (`optim.py:163`) builds a three-phase multiplier — warmup
(linear 0→1) → optional stable plateau (WSD only) → decay (linear or cosine to
`min_lr/lr`). Wrapped in a `LambdaLR` per optimizer; stepped manually each
optimizer step. Schedulers: `warmup_linear`, `warmup_cosine`, `wsd_linear`,
`wsd_cosine`.

### 2.6 Checkpointing

`src/oplm/training/checkpoint.py`. `save_checkpoint` writes
`checkpoint-{step}/` containing `accelerator.save_state(...)` (model, optimizer(s),
scheduler(s), RNG), a `trainer_state.json` (`global_step`, `epoch`,
`samples_seen`, `tokens_seen`), and a frozen `config.yaml`. `_rotate_checkpoints`
keeps at most `save_total_limit`. `load_checkpoint` restores Accelerate state and
returns the trainer-state dict; resume re-seeds the dataset epoch and resets the
per-opt-step delta markers.

### 2.7 Logging, progress, callbacks

- **wandb** via `accelerator.log(metrics, step=global_step)`. Training metrics:
  `train/{loss,epoch,samples,tokens,flops,lr}`. Eval metrics arrive as
  `eval/<task>/<metric>` from the harness.
- **Rich progress bar** on the main process only.
- **Callbacks** (`callbacks.py`): `TrainerCallback` with `on_train_start`,
  `on_log`, `on_eval_end`, `on_checkpoint_saved`, `on_train_end`; all invoked on
  the main process only.

### 2.8 Eval integration

`_build_eval_context` constructs a frozen, rank-identical `EvalContext`
(`global_step`, `epoch`, `tokens_seen`, `steps_delta=1`, `tokens_delta`,
`epoch_delta`, `is_final`) and `_run_eval` delegates to
`Evaluator.run_due(ctx, model, accelerator)`, which unwraps the model, flips it to
eval and back, runs only the due tasks, and returns merged `eval/<task>/<metric>`
metrics. This contract is **already correct** and is retained unchanged (§4.7).

---

## 3. What changed underneath — why the trainer is broken now

The model, data tooling, and eval harness were rewritten on the `hf-compat`
branch. The trainer and its satellites were not, and now reference a model class,
config schema, and field names that no longer exist. The **central break** is the
config collision:

- The run config exposes `cfg.model` as the **dataclass** `oplm.config.ModelConfig`
  (old architecture: `hidden_dim`, `num_layers`, `num_heads`, `num_kv_heads`,
  `value_residual`, `output_gate`, `attn_residual`, `conv_positions` in `"ACD"`, …).
- `OplmForMaskedLM.__init__(config)` requires the **HuggingFace**
  `oplm.model.OplmConfig` (`PretrainedConfig`: `hidden_size`, `num_hidden_layers`,
  `num_attention_heads`, `intermediate_size`, `max_position_embeddings`,
  `norm_strategy`, `canon_positions` in `"ABCD"`, …).

The schemas diverge in names *and* feature set — the dataclass describes the
**old** model. Every site below must converge onto the HF config.

### 3.1 Seam inventory (current → new)

| Site | Current (broken/stale) | New |
|---|---|---|
| `trainer.py:93`, `inference.py:65`, `cli.py:119` | `OplmForMaskedLM(cfg.model)` — passes the dataclass `ModelConfig` | pass the HF `OplmConfig` (`cfg.model` *is* the HF config) |
| `trainer.py:95` | `model.encoder.gradient_checkpointing = True` — no `.encoder` | `model.gradient_checkpointing_enable()` (`modeling_oplm.py:119`) |
| `optim.py:66` | Muon excludes head via `name.startswith("mlm_head.")` | `name.startswith("lm_head.")` — the head is `self.lm_head` |
| `flops.py` | reads `hidden_dim / num_layers / num_kv_heads / ffn_dim` | `hidden_size / num_hidden_layers / intermediate_size` (no GQA) |
| `checkpoint.py:63` | `OmegaConf.structured(deepcopy(cfg))` — can't structure a `PretrainedConfig` | `cfg.model.to_dict()` + OmegaConf for `train`/`data`; plus HF export |
| `trainer.py:_config_to_flat_dict` (`:456`) | `dataclasses.asdict(cfg)` — fails on the HF config | merge `cfg.model.to_dict()` with `asdict` of `train`/`data` |
| `loaders.py:90,126` · `structure.py:216,323` | `cfg.model.max_seq_len` | `cfg.model.max_position_embeddings` |
| `cli.py:82,93,97` (`encode`) | `ProteinTokenizer` (deleted) + `model.encoder(...)` | `model.logits(seqs, LogitsConfig(return_embeddings=True)).embeddings` |
| `cli.py:57,144-185` (`train` print / `info` table) | old field names + removed-feature rows | HF field names |
| `config.py` | `ModelConfig`, `_DERIVED_MODEL_FIELDS`, model-field reset in `load_config` | delete; build the HF `OplmConfig` directly |
| `configs/model/base.yaml` + `presets/*.yaml` | old field names | HF field names |

### 3.2 What is reused unchanged

Document these; do **not** rewrite them:

- `EvalContext` / `Evaluator.run_due` (`eval/context.py`, `eval/evaluator.py`) and
  the rank-reduce-tokens invariant (`trainer.py:205-227`).
- `build_train_dataloader` (`data/sequence/loaders.py`) — only the `max_seq_len`
  field read changes.
- `get_schedule_fn` and the AdamW/Muon builders (`optim.py`) — only the head
  prefix changes.
- `_rotate_checkpoints` (`checkpoint.py`) and the `TrainerCallback` surface
  (`callbacks.py`).
- The loop's `accelerator.accumulate` + `sync_gradients` structure and epoch
  rollover.

---

## 4. The redesigned trainer

### 4.1 Config system

**Root config.** `oplm.config.OplmConfig` remains the dataclass that composes the
run: `{model, train, data}`. `train` (`TrainConfig`) and `data` (`DataConfig`)
stay OmegaConf-structured dataclasses, unchanged in shape. **`model` becomes an
untyped mapping** (`Any`, like `data.train`/`data.eval` already are) so OmegaConf
can carry arbitrary HF field keys without a parallel schema.

```python
@dataclass
class OplmConfig:
    model: Any = field(default_factory=dict)   # resolved into oplm.model.OplmConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
```

**`load_config` (`config.py:485`) rewrite.** Merge order is unchanged — defaults →
`--preset` → `--config` YAML → CLI dotlist overrides. Changes:

- Delete the dataclass `ModelConfig`, the `_DERIVED_MODEL_FIELDS` tuple, and the
  derived-field reset loop. The HF config owns derivation
  (`_resolve_derived_fields`) and validation (`_validate`), so derived fields
  (`head_dim`, `intermediate_size`, `rope_dim`, `nope_dim`) are simply **omitted**
  unless the user sets them, and resolve to `None` → derived.
- After merging, resolve the `model` subtree to a plain dict and instantiate the
  HF config:

  ```python
  from oplm.model import OplmConfig as OplmModelConfig
  model_dict = OmegaConf.to_container(base.model, resolve=True) or {}
  model_cfg = OplmModelConfig(**model_dict)     # HF validation + derivation here
  cfg = OplmConfig(model=model_cfg, train=…, data=…)
  ```

- Keep `cfg.train.config_path` provenance.
- Drop the old `train.eval_every` rejection — `train.eval_every` is now the
  canonical global-cadence field (a cadence mapping; the int form is caught by the
  schedule parser). **Update** the `data.max_length` rejection message — it currently
  points at the now-renamed `model.max_seq_len`; it should point at
  `model.max_position_embeddings`.

**No special handling for old/removed model field names (decision).** Unknown
`model.*` keys flow into the HF config's `**kwargs` and are absorbed by
`PretrainedConfig`. The doc and `model/base.yaml` document the new schema; the
caveat is explicit: **old or mistyped model keys do not raise** — they are
silently retained as config attributes and ignored by the model.

**Naming.** The root `oplm.config.OplmConfig` and the model `oplm.model.OplmConfig`
share a class name. The trainer keeps the root name (renaming it would ripple
through `data`/`eval`/tests for little gain) and uses the established alias at
import sites: `from oplm.model import OplmConfig as OplmModelConfig`.

**Config files.** Rewrite `configs/model/base.yaml` and `configs/model/presets/*.yaml`
with HF field names. Field reference (HF `OplmConfig`, defaults from
`configuration_oplm.py`):

| Field | Default | Notes |
|---|---|---|
| `vocab_size` | 33 | ESM-C vocab; warns if changed |
| `hidden_size` | 768 | was `hidden_dim` |
| `num_hidden_layers` | 12 | was `num_layers` |
| `num_attention_heads` | 12 | was `num_heads`; **no GQA** (`num_kv_heads` removed) |
| `head_dim` | `None`→`hidden_size//heads` | derived |
| `intermediate_size` | `None`→`round_up(8·H/3, 256)` | was `ffn_dim` |
| `max_position_embeddings` | 1024 | was `max_seq_len` |
| `rope_theta` / `rope_dim` / `nope_dim` | 10000 / `None`→full / 0 | `rope_dim+nope_dim==head_dim`, `rope_dim` even |
| `norm_type` | `layernorm` | `layernorm` \| `rmsnorm` |
| `norm_eps` | 1e-6 | |
| `norm_strategy` | `pre` | `pre` \| `sandwich` \| `hybrid` \| `post_sdpa` (replaces the old `pre_norm`/`post_norm`/`sandwich_norm`/`post_sdpa_norm` booleans) |
| `qk_norm` | `True` | |
| `post_embed_norm` | `False` | |
| `residual_scaling` | `sqrt_num_layers` | \| `none` |
| `init_scale_output_projections` | `True` | |
| `ffn_activation` | `swiglu` | `swiglu` \| `geglu` (both gated, 3 projections) |
| `ffn_bias` | `False` | |
| `attention_dropout` / `hidden_dropout` | 0.0 / 0.0 | |
| `tie_word_embeddings` | `False` | HF tying field (was `tie_embeddings`) |
| `mlm_head_activation` | `gelu` | `gelu` \| `silu` \| `relu` |
| `canon_enabled` | `False` | gates Canon depthwise conv |
| `canon_positions` | `[]` | subset of `{A,B,C,D}` (was `conv_positions` in `"ACD"`) |
| `canon_kernel_sizes` | 4 | int \| list \| per-layer dict |
| `canon_activation` | `none` | `none` \| `silu` \| `gelu` |
| `initializer_range` | 0.02 | |
| `classifier_pool` / `classifier_dropout` / `num_labels` / `pre_head_norm` | `mean` / 0.0 / 2 / `False` | task-head fields |
| `use_flex_attention` | `True` | flex-attention fast path on CUDA |
| `gradient_checkpointing` | `False` | |
| `pad/bos/eos/unk/mask_token_id` | 1 / 0 / 2 / 3 / 32 | |

> **Removed (no replacement)** — `num_kv_heads`/`shared_kv` (GQA), `value_residual`
> (+`value_residual_lambda_init`), `num_value_embeds`/`value_embed_gate_dim`,
> `output_gate`/`query_dependent_gate`, `attn_residual`/`attn_residual_block_size`,
> `conv_kernel_schedule`/`conv_kernel_increment`/`conv_kernel_block_size`/`conv_kernel_max_size`,
> `partial_rope` (now expressed via `rope_dim`/`nope_dim`), and `dtype`.

### 4.2 Model construction

```python
from oplm.model import OplmForMaskedLM
model = OplmForMaskedLM(cfg.model)              # cfg.model is the HF OplmConfig
if cfg.model.gradient_checkpointing:
    model.gradient_checkpointing_enable()       # propagates to every OplmBlock
```

`forward(input_ids, attention_mask, labels=…)` returns a `MaskedLMOutput`; loss is
computed inside the model (cross-entropy, `ignore_index=-100`). The loop keeps
reading `outputs["loss"]` (`MaskedLMOutput` supports item access). Setting
`config.gradient_checkpointing=True` already arms `OplmStack`/`OplmBlock` at init
(`transformer.py:204`), so the explicit `gradient_checkpointing_enable()` is the
HF-idiomatic belt-and-suspenders call and replaces the broken `model.encoder` line.

### 4.3 Optimizer & parameter grouping

Unchanged design, **one fix**: the Muon head-exclusion prefix becomes `lm_head.`
(the MLM head is `OplmForMaskedLM.lm_head`). Without it, `lm_head.dense.weight`
(a 2-D weight) leaks into Muon instead of staying on AdamW. The decoder weight is
tied to the embedding and already lands in no-decay via the `"embed"` name rule.
`optim.py` otherwise takes only `cfg.train` (a `TrainConfig`) and
`model.named_parameters()`, so no further changes.

### 4.4 FLOPs accounting

Rewrite `estimate_flops_per_token` to take the HF `OplmConfig` and its field
names. Simplifications versus the old version: there is **no GQA**, so Q/K/V each
project `hidden_size → hidden_size` (drop `num_kv_heads`); FFN is always gated
(`swiglu`/`geglu`) with **3 projections**. Per layer: attention projections
`2·H·(4H)` plus FFN `3·2·H·I`; head: `2·H·H + 2·H·V`; training ≈ `3×` forward.
Keep the documented caveat that attention-score FLOPs, norms, and embedding
lookups are omitted.

### 4.5 Training loop — kept structure, targeted cleanups

The loop shape (accumulate → backward → clip-on-sync → step/zero → cadence on
sync boundary) is retained. Cleanups:

- **Single optimizer iteration.** Fold the two `for optimizer in self.optimizers`
  loops (step, then zero-grad) into one.
- **Accumulation-aware loss for logging.** `current_loss = loss.item()` captures
  only the final micro-batch. Accumulate a detached running sum across micro-steps
  and divide by `gradient_accumulation_steps` so `train/loss` reflects the whole
  optimizer step (`gradient_accumulation_steps=1` is unchanged).
- **Final scheduler step.** Document that schedulers step exactly `total_steps`
  times; confirm the last step does not over-advance past the decay floor.

### 4.6 Cadence: steps, epochs, tokens

- **Duration** — `max_steps` (default) or `max_epochs`. `_compute_total_steps`
  derives steps-per-epoch from the resolved dataset length and the global
  effective batch (`batch_size · grad_accum · num_processes`); document the
  fallback when an iterable/sharded dataset reports no length.
- **Logging / checkpointing** — step-modulo (`log_every`, `save_every`).
- **Eval** — step or token cadence per dataset via the harness schedules
  (`every: {steps: N}` / `{tokens: N}`), defaulting to `train.eval_every`.
  **Epoch-based eval cadence stays deferred** ([`EVAL_HARNESS.md`](EVAL_HARNESS.md) §4.6);
  `epoch`/`epoch_delta` are still carried in the `EvalContext` for forward
  compatibility.

### 4.7 Eval integration (unchanged contract)

Retain `_build_eval_context` / `_run_eval` exactly: build a frozen, rank-identical
`EvalContext` once per optimizer step and call
`Evaluator.run_due(ctx, model, accelerator)`. The trainer keeps the unconditional
`accelerator.reduce` of per-step tokens so `tokens_seen`/`tokens_delta` are
rank-identical regardless of which task cadences are configured. Merged
`eval/<task>/<metric>` metrics are logged and forwarded to `on_eval_end`.

### 4.8 Checkpointing — Accelerate state + HF export

`save_checkpoint` writes `checkpoint-{step}/` with:

- **Accelerate state** at the top level (`accelerator.save_state(...)`): model,
  optimizer(s), scheduler(s), RNG — the resumable state.
- **`trainer_state.json`**: `global_step`, `epoch`, `samples_seen`, `tokens_seen`.
- **HF export** under `checkpoint-{step}/hf/` via the unwrapped model:
  `accelerator.unwrap_model(model).save_pretrained(.../hf)` (writes `config.json`
  + `model.safetensors`, honoring tied weights). Attach and save the tokenizer
  (`get_tokenizer().save_pretrained(.../hf)`) so
  `OplmForMaskedLM.from_pretrained(checkpoint-{step}/hf)` round-trips with its
  tokenizer.

This replaces the broken `OmegaConf.structured(deepcopy(cfg))` serialization. The
run config is still persisted for provenance via `cfg.model.to_dict()` plus
OmegaConf YAML for `train`/`data`. `_rotate_checkpoints` and the resume path are
unchanged; resume reads Accelerate state + `trainer_state.json` from the top level
(the `hf/` export is for downstream loading, not resume).

### 4.9 Logging / wandb

Fix `_config_to_flat_dict` so it no longer assumes the whole config is a nested
dataclass: flatten `cfg.model.to_dict()` under a `model/` prefix and `asdict`
`train`/`data` under their prefixes. Logged metric sets are unchanged
(`train/{loss,epoch,samples,tokens,flops,lr}`, `eval/<task>/<metric>`).

### 4.10 Callbacks

The `TrainerCallback` surface and main-process-only invocation are unchanged.

### 4.11 CLI & inference adaptation

- **`cli.py train`** — replace the `cfg.model.num_layers / hidden_dim` summary
  print with `num_hidden_layers / hidden_size`.
- **`cli.py info`** — rebuild the architecture/features tables on HF fields
  (`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `head_dim`,
  `intermediate_size`, `norm_strategy`, `qk_norm`, `canon_*`, `residual_scaling`,
  `gradient_checkpointing`, `tie_word_embeddings`); drop rows for removed features.
- **`cli.py encode`** — collapse the broken `ProteinTokenizer` + `model.encoder`
  path to the ESM-C convenience API. Note `model.encode(seqs)` returns padded
  **`input_ids`** (an ESM-C naming quirk), *not* embeddings; for the embedding
  matrix use `model.logits(seqs, LogitsConfig(return_embeddings=True)).embeddings`
  → `(B, L, hidden_size)` (post-final-norm hidden states). Requires a tokenizer
  attached (via `from_pretrained` of an `hf/` export, or an explicit
  `model.tokenizer = get_tokenizer()`).
- **`inference.py`** — `load_model_for_inference` builds `OplmForMaskedLM(cfg.model)`
  from the HF config (`cfg.model` is now the HF `OplmConfig`); prefer
  `OplmForMaskedLM.from_pretrained(<checkpoint>/hf)` when an HF export is present,
  falling back to the state-dict path for bare weights files.

---

## 5. Bugs diagnosed & fixed

1. **Model built from the wrong config type** — `OplmForMaskedLM(cfg.model)` passed
   the dataclass `ModelConfig`; converge on the HF `OplmConfig` (§4.1–4.2).
2. **`model.encoder` no longer exists** — gradient checkpointing crashed; use
   `gradient_checkpointing_enable()` (§4.2).
3. **Muon head exclusion checks the wrong prefix** — `mlm_head.` vs the actual
   `lm_head.`; the head's dense weight leaks into Muon (§4.3).
4. **FLOPs read removed fields** — `hidden_dim/num_kv_heads/ffn_dim` (§4.4).
5. **Checkpoint config serialization breaks on a `PretrainedConfig`** —
   `OmegaConf.structured(cfg)` (§4.8).
6. **wandb flat-dict breaks on a `PretrainedConfig`** — `asdict(cfg)` (§4.9).
7. **Stale `data.max_length` rejection message** — points at the renamed
   `model.max_seq_len` (§4.1).
8. **`cli encode` doubly broken** — `ProteinTokenizer` deleted and `model.encoder`
   gone (§4.11).
9. **Last-micro-batch loss logged as the step loss** — accumulation-aware loss
   (§4.5).
10. **Double optimizer iteration** — minor cleanup (§4.5).

---

## 6. File-by-file change map (for implementation)

| File | Change |
|---|---|
| `src/oplm/config.py` | delete `ModelConfig` + derived-field machinery; `OplmConfig.model: Any`; `load_config` builds HF `OplmConfig`; update `data.max_length` message |
| `src/oplm/configs/model/base.yaml`, `presets/{small,medium,base,large,xlarge}.yaml` | rewrite with HF field names |
| `src/oplm/training/trainer.py` | model build + `gradient_checkpointing_enable`; accumulation-aware loss; single optimizer loop; `_config_to_flat_dict` fix |
| `src/oplm/training/optim.py` | Muon head prefix `lm_head.` |
| `src/oplm/training/flops.py` | rewrite for HF fields; drop GQA |
| `src/oplm/training/checkpoint.py` | `cfg.model.to_dict()` serialization + HF export under `hf/` |
| `src/oplm/data/sequence/loaders.py`, `src/oplm/eval/tasks/structure.py` | `max_seq_len` → `max_position_embeddings` |
| `src/oplm/cli.py` | `train` print, `info` table, `encode` via `model.encode` |
| `src/oplm/inference.py` | HF-config construction; `from_pretrained` path |
| `tests/training/` (new), `tests/eval/test_trainer_integration.py` | new unit suite; un-skip integration test |

---

## 7. Test & verification strategy

- **New `tests/training/` suite** (mirrors `src/oplm/training/`):
  - `test_config.py` — `load_config` builds a valid HF `OplmConfig`; preset +
    CLI overrides (`model.num_hidden_layers=…`) apply; derived fields resolve;
    unknown `model.*` keys are absorbed (documented caveat).
  - `test_optim.py` — `partition_optimizer_params` puts `lm_head.*` on AdamW (not
    Muon); embeddings/norms/biases land in no-decay; partition covers all params.
  - `test_flops.py` — positive, finite estimate; scales with depth/width.
  - `test_checkpoint.py` — save → `from_pretrained(checkpoint/hf)` round-trips
    weights; resume restores `global_step/epoch/tokens_seen`; rotation respects
    `save_total_limit`.
- **Un-skip `tests/eval/test_trainer_integration.py`** and update its `_cfg` to
  build the HF `OplmConfig` (e.g. `OplmModelConfig(hidden_size=32, num_attention_heads=4,
  num_hidden_layers=2, max_position_embeddings=64)`). It already asserts the
  sequence eval fires on both step and token cadence over a 4-step CPU run.
- **Pilot CPU run** (à la `tests/model/test_pilot_train.py`, `@pytest.mark.slow`):
  a tiny model trains a few steps with `wandb_enabled=False`, `mixed_precision="no"`,
  fires one eval, writes a checkpoint, and resumes from it — asserting no shape
  errors, finite loss, and a correct post-resume `global_step`.
- **Gates**: `ruff check src/`, `ruff format src/`, `mypy src/`, and the full
  `pytest` suite (with `-m "not slow"` for fast iteration).

---

## 8. Open / deferred

- **Epoch-based eval cadence** — deferred ([`EVAL_HARNESS.md`](EVAL_HARNESS.md) §4.6);
  the `EvalContext` already carries `epoch`/`epoch_delta`.
- **Root/model `OplmConfig` name collision** — retained for now; revisit a rename
  to `RunConfig` when the cost of touching `data`/`eval`/tests is justified.
- **Muon** — available in torch ≥ 2.11; no behavioral change in this rewrite.

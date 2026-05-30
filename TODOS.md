# OPLM Trainer Rewrite — Implementation Plan

Standalone, task-list implementation plan for converging the training
infrastructure onto the rewritten HuggingFace model (`hf-compat` branch). This
plan is derived from `docs/TRAINER.md` and is self-contained: every change is
specified with file paths, before/after snippets, and acceptance checks so the
work can be completed without consulting any other document.

> **Status legend:** `[ ]` not started · `[~]` in progress · `[x]` done.
> Line numbers are approximate (as of writing) — locate by symbol/text, not by
> line.

---

## 0. Context & the central problem

The model, data tooling, and eval harness were rewritten to be HuggingFace-native.
The trainer and its satellites still reference an old architecture and a stale
config schema. The **root break** is a config-type collision:

- `oplm.config.OplmConfig` (a dataclass) currently composes `{model, train, data}`
  where `model` is the **old dataclass** `oplm.config.ModelConfig`
  (`hidden_dim`, `num_layers`, `num_heads`, `num_kv_heads`, `ffn_dim`,
  `max_seq_len`, `conv_positions`, `value_residual`, …).
- The model class `oplm.model.OplmForMaskedLM(config)` requires the **HuggingFace**
  `oplm.model.OplmConfig` (a `PretrainedConfig`: `hidden_size`,
  `num_hidden_layers`, `num_attention_heads`, `intermediate_size`,
  `max_position_embeddings`, `norm_strategy`, `canon_positions`, …).

The fix: make the run config's `model` field carry the **HF** `OplmConfig`
directly, delete the old `ModelConfig` dataclass, and update every site that
read old field names. There is **no `ModelConfig → OplmConfig` converter** — old
config files and old field names are intentionally unsupported (unknown
`model.*` keys are silently absorbed by `PretrainedConfig`, never raised).

### Naming convention (use everywhere)

The root config and the model config share the class name `OplmConfig`. At every
site that needs the model config, import it aliased:

```python
from oplm.model import OplmConfig as OplmModelConfig
```

The root config stays `oplm.config.OplmConfig`.

### Key facts confirmed in the codebase (rely on these)

- `OplmForMaskedLM(config)` takes the HF `OplmConfig`; `forward(input_ids,
  attention_mask, labels=…)` returns a `MaskedLMOutput` and computes the loss
  internally (`F.cross_entropy(..., ignore_index=-100)`). `outputs["loss"]` works
  (ModelOutput supports item access).
- The MLM head is `OplmForMaskedLM.lm_head` (an `OplmMLMHead` with `.dense`,
  `.norm`, `.decoder`). There is **no** `mlm_head` and **no** `model.encoder`.
- Gradient checkpointing: `model.gradient_checkpointing_enable()` (defined on
  `OplmPreTrainedModel`, propagates to every `OplmBlock`/`OplmStack`). Setting
  `config.gradient_checkpointing=True` also arms it at init.
- HF `OplmConfig` resolves derived fields in `__init__`
  (`head_dim`, `intermediate_size`, `rope_dim`/`nope_dim`) and validates there.
  After construction these are always concrete (never `None`).
- `oplm.data.get_tokenizer()` returns the shared `OplmTokenizerFast`; it has
  `.save_pretrained(dir)`.
- `import oplm` (top-level `src/oplm/__init__.py`) registers `OplmConfig`,
  `OplmForMaskedLM`, and `OplmTokenizerFast` with the HF Auto* classes
  in-process, so `OplmForMaskedLM.from_pretrained(<dir>)` and
  `AutoTokenizer.from_pretrained(<dir>)` work on a local export without
  `trust_remote_code`.
- ESM-C convenience API on every model: `model.tokenize(seqs)`,
  `model.encode(seqs)` (returns **input_ids**, not embeddings), and
  `model.logits(seqs, LogitsConfig(return_embeddings=True)).embeddings` →
  `(B, L, hidden_size)`. Requires `model.tokenizer` to be set.
  `LogitsConfig` is importable from `oplm.model`.

### Verification gates (run after each phase; all must pass at the end)

```bash
ruff check src/
ruff format src/
mypy src/
pytest -m "not slow"      # fast iteration
pytest                    # full suite, including slow pilot/integration runs
```

---

## Phase 1 — Config system & YAML defaults

**Files:** `src/oplm/config.py`, `src/oplm/configs/model/base.yaml`,
`src/oplm/configs/train/base.yaml` (**new**), `src/oplm/configs/data/base.yaml`,
`src/oplm/configs/train/__init__.py` (**new**),
`src/oplm/configs/model/presets/{small,medium,base,large,xlarge}.yaml`.

Everything else imports from here, so do this first.

> **Design decision — YAML defaults are authoritative.** Today the `base.yaml`
> files are *documentation mirrors* that `load_config` never reads; defaults come
> only from the Python dataclasses (`OmegaConf.structured(OplmConfig)`). This
> phase makes the per-concern `base.yaml` files the **loaded defaults layer**, so
> there is one human-editable home for defaults across model/train/data. New
> merge order:
>
> ```text
> structured(OplmConfig)         # schema + dataclass fallbacks
>   → configs/model/base.yaml    # authoritative model defaults  ┐
>   → configs/train/base.yaml    # authoritative train defaults  │ NEW base layer
>   → configs/data/base.yaml     # authoritative data defaults   ┘
>   → --preset YAML              # size preset (model only)
>   → --config YAML              # user run config
>   → CLI dotlist overrides
> ```
>
> The dataclass defaults remain (they keep `OmegaConf.structured` working, allow
> direct `TrainConfig()`/`DataConfig()` construction in tests, and run
> `__post_init__` validation), but for `load_config` the YAML wins. To neutralize
> the YAML↔Python drift hazard this introduces, Phase 8 adds a **consistency
> test** asserting each `base.yaml` agrees with its dataclass / HF-config fields.
> All three files must use the **section-wrapped convention** (`model:` / `train:`
> / `data:` at the top level) so they merge at the config root.

### 1.1 Delete the old `ModelConfig` and its machinery (`config.py`)

- [ ] Delete the entire `@dataclass class ModelConfig` block (≈ lines 25–172),
      including `__post_init__`, `conv_kernel_size_for_layer`, and all its
      validation.
- [ ] Delete now-unused module-level helpers that only served `ModelConfig`:
  - [ ] `round_multiple` (≈ line 20).
  - [ ] `_VALID_CONV_KERNEL_SCHEDULES` (≈ line 17).
  - [ ] `import math` at the top **if** nothing else in the file uses it (it is
        only used by `round_multiple`; confirm with a grep before removing).
- [ ] Delete the derived-field reset machinery:
  - [ ] `_DERIVED_MODEL_FIELDS` tuple (≈ line 444).
  - [ ] Keep `_NESTED_VALUE_MISSING` and `_lookup_nested_mapping_value` (still
        used by the rejection helpers below).

### 1.2 Make `OplmConfig.model` an untyped mapping (`config.py`)

- [ ] In `@dataclass class OplmConfig` change the `model` field:

  ```python
  # before
  model: ModelConfig = field(default_factory=ModelConfig)
  # after
  model: Any = field(default_factory=dict)   # resolved into oplm.model.OplmConfig
  ```

  `Any` is already imported. This mirrors how `DataConfig.train`/`.eval` are
  already `Any`, letting OmegaConf carry arbitrary HF field keys.

### 1.3 Update the removed-alias rejection message (`config.py`)

- [ ] In `_reject_removed_sequence_length_alias`, keep the rejection of
      `data.max_length` but update the message to point at the new field name:

  ```python
  raise ValueError(
      "`data.max_length` has been removed. Use `model.max_position_embeddings` "
      "as the sequence-length setting."
  )
  ```

- [ ] Leave `_reject_removed_eval_every_alias` unchanged.

### 1.4 Rewrite `load_config`: load YAML defaults + build the HF config (`config.py`)

Insert the authoritative `base.yaml` layer between the structured defaults and
the preset/config/CLI overrides, then replace the derived-field reset +
`OmegaConf.to_object` tail with HF-config construction.

- [ ] Add a packaged-YAML loader and a base-layer table near `get_preset_config`
      (reuse the existing `from importlib.resources import files` import):

  ```python
  # Per-concern default layers, merged at the config root in this order. Each
  # file is section-wrapped (top-level `model:` / `train:` / `data:`).
  _BASE_CONFIG_LAYERS = (
      ("oplm.configs.model", "base.yaml"),
      ("oplm.configs.train", "base.yaml"),
      ("oplm.configs.data", "base.yaml"),
  )


  def _load_packaged_yaml(package: str, filename: str) -> DictConfig:
      """Load a YAML resource shipped inside the package as a DictConfig."""
      text = files(package).joinpath(filename).read_text()
      return cast("DictConfig", OmegaConf.create(text))
  ```

- [ ] In `load_config`, after `base = OmegaConf.structured(OplmConfig)` and
      `OmegaConf.set_struct(base, False)`, merge the base layer **before** flag
      parsing / overrides:

  ```python
  base: DictConfig = OmegaConf.structured(OplmConfig)
  OmegaConf.set_struct(base, False)

  # Authoritative YAML defaults layer (model → train → data).
  for package, filename in _BASE_CONFIG_LAYERS:
      base = cast("DictConfig", OmegaConf.merge(base, _load_packaged_yaml(package, filename)))
  ```

  The two `_reject_removed_*` checks still run only on the user
  `override_dicts` (preset / `--config` / CLI), never on the trusted base layer.

- [ ] Remove the `explicit_model_keys` discovery loop and the
      `for fname in _DERIVED_MODEL_FIELDS: base.model[fname] = None` block.
- [ ] Replace the final conversion so `model` becomes an HF `OplmConfig`:

  ```python
  # ... after merging the base layer, then all user overrides into `base`,
  #     and after the two _reject_removed_* calls on override_dicts ...

  # Build train/data dataclasses (triggers their __post_init__ validation);
  # `model` is an Any field so it round-trips as a plain dict here.
  cfg: OplmConfig = OmegaConf.to_object(base)  # type: ignore[assignment]

  # Instantiate the HF model config from the merged `model` subtree. HF owns
  # derivation (head_dim, intermediate_size, rope_dim/nope_dim) and validation.
  from oplm.model import OplmConfig as OplmModelConfig

  model_dict = OmegaConf.to_container(base.model, resolve=True) or {}
  cfg.model = OplmModelConfig(**model_dict)  # type: ignore[arg-type]

  cfg.train.config_path = config_path
  return cfg
  ```

  Notes:
  - Derived fields are **omitted** from the model YAML / `model_dict` unless set,
    and resolve to `None` → derived inside `OplmModelConfig.__init__`.
  - Unknown / old / mistyped `model.*` keys flow into `**model_dict` →
    `PretrainedConfig` `**kwargs` and are silently retained (documented caveat —
    they do **not** raise). The Phase-8 base-layer test guards against typos in
    `model/base.yaml` itself.
  - `_load_packaged_yaml` needs `oplm.configs.train` to be an importable package,
    so create `src/oplm/configs/train/__init__.py` (empty — see 1.6).

### 1.5 Rewrite `configs/model/base.yaml` with HF field names

This file is now **loaded by `load_config`** (1.4) as the authoritative model
default layer, not just documentation. Keep it section-wrapped under `model:`.

- [ ] Replace the file contents entirely. Use HF names and the documented
      defaults (only fields worth surfacing; everything else inherits the HF
      `OplmConfig.__init__` defaults):

  ```yaml
  # OPLM default model configuration (HuggingFace OplmConfig field names).
  # Authoritative defaults: loaded by oplm.config.load_config. Any field omitted
  # here falls through to the OplmConfig.__init__ default. Derived fields
  # (head_dim, intermediate_size, rope_dim, nope_dim) are omitted so they resolve
  # from the source dimensions unless explicitly overridden.
  model:
    # Core dimensions
    vocab_size: 33
    hidden_size: 768
    num_hidden_layers: 12
    num_attention_heads: 12
    max_position_embeddings: 1024

    # Positional encoding (full RoPE by default: rope_dim resolves to head_dim)
    rope_theta: 10000.0
    nope_dim: 0

    # Normalization
    norm_type: layernorm        # layernorm | rmsnorm
    norm_eps: 1.0e-6
    norm_strategy: pre          # pre | sandwich | hybrid | post_sdpa
    qk_norm: true
    post_embed_norm: false
    residual_scaling: sqrt_num_layers   # sqrt_num_layers | none
    init_scale_output_projections: true

    # Feed-forward
    ffn_activation: swiglu      # swiglu | geglu (both gated, 3 projections)
    ffn_bias: false

    # Dropout
    attention_dropout: 0.0
    hidden_dropout: 0.0

    # Embeddings / head
    tie_word_embeddings: false
    mlm_head_activation: gelu   # gelu | silu | relu

    # Canon depthwise convolution (off by default)
    canon_enabled: false
    canon_positions: []         # subset of {A, B, C, D}
    canon_kernel_sizes: 4
    canon_activation: none      # none | silu | gelu

    # Init / attention kernel / checkpointing
    initializer_range: 0.02
    use_flex_attention: true
    gradient_checkpointing: false
  ```

### 1.6 Create `configs/train/base.yaml` (new authoritative train defaults)

- [ ] Create the package marker `src/oplm/configs/train/__init__.py` (empty file;
      no logic) so `files("oplm.configs.train")` resolves.
- [ ] Create `src/oplm/configs/train/base.yaml`, section-wrapped under `train:`,
      mirroring every `TrainConfig` field default **exactly** (the Phase-8
      consistency test enforces this). Omit `config_path` (auto-populated by
      `load_config`).

  ```yaml
  # OPLM default training configuration (mirrors oplm.config.TrainConfig).
  # Authoritative defaults: loaded by oplm.config.load_config. Values here MUST
  # equal the TrainConfig dataclass defaults (enforced by tests/training/test_config.py).
  train:
    # Duration
    max_steps: 50_000
    max_epochs: null

    # Batch
    batch_size: 32
    gradient_accumulation_steps: 1

    # Optimizer
    optimizer: adamw            # adamw | muon
    lr: 1.0e-4
    min_lr: 0.0
    weight_decay: 0.01
    adam_beta1: 0.9
    adam_beta2: 0.98
    adam_eps: 1.0e-8
    muon_adjust_lr_fn: match_rms_adamw   # match_rms_adamw | original
    muon_momentum: 0.95
    muon_nesterov: true
    muon_ns_steps: 5
    max_grad_norm: 1.0

    # Scheduler
    scheduler: warmup_linear    # warmup_linear | warmup_cosine | wsd_linear | wsd_cosine
    warmup_steps: 5_000
    stable_steps: 0             # WSD plateau length (WSD schedules only)

    # Logging
    log_every: 10
    eval_every: { steps: 10_000 }   # default cadence: {steps: N} | {tokens: N}
    wandb_project: oplm
    wandb_run_name: null
    wandb_enabled: true

    # Checkpointing
    save_every: 10_000
    save_total_limit: 3
    resume_from: null

    # Infrastructure
    seed: 42
    output_dir: outputs
    mixed_precision: bf16       # bf16 | fp16 | no
  ```

### 1.7 Rewrite `configs/data/base.yaml` to the section-wrapped convention

The existing `data/base.yaml` lists keys at the **top level** (not under `data:`),
which was harmless when it was documentation-only but is wrong now that the file
is merged at the config root. Re-nest everything under `data:` and keep the
helpful `train`/`eval` syntax comments.

- [ ] Rewrite the file, section-wrapped under `data:`, mirroring `DataConfig`
      defaults exactly (Phase-8 test enforces this for the scalar fields):

  ```yaml
  # OPLM default data configuration (mirrors oplm.config.DataConfig).
  # Authoritative defaults: loaded by oplm.config.load_config.
  data:
    # Training dataset(s). Single path (100% sampling):
    #   data: { train: /path/to/dataset }   (CLI: data.train=/path/to/dataset)
    # Multiple datasets with fractional sampling:
    #   data:
    #     train:
    #       uniref50: { path: /data/uniref50/, fraction: 0.6 }
    #       bfd:      { path: /data/bfd/,      fraction: 0.4 }
    # Notes: each path is a .parquet file or a directory of shards; fractions are
    # normalized to 1.0; omitted fractions share the remaining mass equally;
    # parquet must have columns sequence_id, sequence.
    train: null

    # Evaluation dataset(s). Named map; each entry needs `path` and `type`
    # (sequence, structure, proteingym, tape, proteinglue, everest). Per-dataset
    # cadence (`every`): exactly one of {steps: N} | {tokens: N}, optional
    # at_start (default false) / at_end (default true); datasets that omit `every`
    # use train.eval_every. Task-specific keys sit at the same level as
    # `path`/`type`. See docs/EVAL_HARNESS.md §9.
    #   eval:
    #     heldout:    { path: /data/eval.parquet, type: sequence, every: { tokens: 20_000_000 } }
    #     structures: { path: /data/pdb, type: structure, every: { steps: 20_000 },
    #                   categorical_jacobian_sample_size: 12 }
    eval: null

    # Sequence masking
    mask_prob: 0.15
    mask_token_prob: 0.8
    random_token_prob: 0.1
    weighted_masking: false     # set true to honor the masking_weights column

    # DataLoader settings
    num_workers: 4
    pin_memory: true
    prefetch_factor: 4

    # Shard iteration behavior (only affects sharded parquet directories)
    shuffle_shards: true
    shuffle_rows: true
  ```

### 1.8 Rewrite the size presets with HF field names

Drop `num_kv_heads` (no GQA). Confirm `hidden_size % num_attention_heads == 0`
for each (all listed below satisfy it).

- [ ] `presets/small.yaml`:

  ```yaml
  # ~25M parameters — fast ablation
  model:
    hidden_size: 256
    num_hidden_layers: 6
    num_attention_heads: 4
  ```

- [ ] `presets/medium.yaml`:

  ```yaml
  # ~150M parameters — standard ablation
  model:
    hidden_size: 768
    num_hidden_layers: 12
    num_attention_heads: 12
  ```

- [ ] `presets/base.yaml`:

  ```yaml
  # ~350M parameters — Proust-scale
  model:
    hidden_size: 1024
    num_hidden_layers: 24
    num_attention_heads: 16
  ```

- [ ] `presets/large.yaml`:

  ```yaml
  # ~3B parameters
  model:
    hidden_size: 2560
    num_hidden_layers: 32
    num_attention_heads: 32
    gradient_checkpointing: true
  ```

- [ ] `presets/xlarge.yaml`:

  ```yaml
  # ~15B parameters — production
  model:
    hidden_size: 5120
    num_hidden_layers: 40
    num_attention_heads: 40
    gradient_checkpointing: true
  ```

### 1.9 Phase-1 acceptance

- [ ] `python -c "from oplm.config import load_config; c=load_config(['--preset','small','model.num_hidden_layers=4']); print(type(c.model), c.model.num_hidden_layers, c.model.head_dim, c.model.intermediate_size)"`
      prints the HF `OplmConfig` type, `num_hidden_layers=4`, and resolved
      `head_dim`/`intermediate_size`.
- [ ] `load_config([])` actually reads the YAML layer: temporarily edit a value
      in `configs/train/base.yaml` (e.g. `seed: 99`) and confirm
      `load_config([]).train.seed == 99`, then revert.
- [ ] Override precedence holds: `load_config(['train.seed=7']).train.seed == 7`
      (CLI beats the YAML base layer).
- [ ] `load_config(['data.max_length=5'])` raises `ValueError` mentioning
      `model.max_position_embeddings`.
- [ ] `grep -rn "ModelConfig" src/` returns no hits except inside docstrings you
      intentionally keep (ideally none in `src/`).

---

## Phase 2 — FLOPs accounting

**File:** `src/oplm/training/flops.py`.

The model has **no GQA** (Q/K/V/O each project `hidden_size → hidden_size`) and
the FFN is always gated (3 projections).

- [x] Update the type import to the HF config:

  ```python
  if TYPE_CHECKING:
      from oplm.model import OplmConfig as OplmModelConfig
  ```

- [x] Rewrite `estimate_flops_per_token` for HF field names and drop GQA / the
      `ffn_dim`-derivation fallback (`intermediate_size` is always resolved):

  ```python
  def estimate_flops_per_token(config: OplmModelConfig) -> int:
      """Estimate training FLOPs per token (forward + backward ~= 3x forward).

      Counts the dominant linear projections (factor of 2 per matmul). Omits
      attention-score FLOPs, normalization, and embedding lookups by design.
      """
      h = config.hidden_size
      num_layers = config.num_hidden_layers
      vocab_size = config.vocab_size
      intermediate = config.intermediate_size  # always resolved post-__init__

      # Attention projections: Q, K, V, O each h -> h  => 2 * h * (4h)
      attn_proj_flops = 2 * h * (4 * h)
      # Gated FFN (swiglu/geglu): 3 projections of size h * intermediate
      ffn_flops = 3 * 2 * h * intermediate

      per_layer = attn_proj_flops + ffn_flops
      backbone_flops = num_layers * per_layer

      # MLM head: dense projection + vocab projection
      head_flops = 2 * h * h + 2 * h * vocab_size

      forward_flops = backbone_flops + head_flops
      return 3 * forward_flops
  ```

### Phase-2 acceptance

- [x] Estimate is positive and finite, and strictly increases when
      `num_hidden_layers` or `hidden_size` increase (covered by Phase 8 test).

---

## Phase 3 — Optimizer parameter grouping fix

**File:** `src/oplm/training/optim.py`.

- [x] In `partition_optimizer_params`, change the Muon head-exclusion prefix
      (≈ line 66) from the stale `mlm_head.` to the real head module name:

  ```python
  # before
  if cfg.optimizer == "muon" and param.ndim == 2 and not name.startswith("mlm_head."):
  # after
  if cfg.optimizer == "muon" and param.ndim == 2 and not name.startswith("lm_head."):
  ```

  Rationale: the MLM head is `OplmForMaskedLM.lm_head`. Without the fix,
  `lm_head.dense.weight` (a 2-D weight) leaks into Muon instead of AdamW. When
  weights are untied, `lm_head.decoder.weight` is also 2-D and must stay on
  AdamW. (The tied case routes the shared weight to no-decay via the existing
  `"embed"` name rule, since it appears under
  `oplm.backbone.embed_tokens.embed_tokens.weight`.)

- [x] Update any nearby comment that references `mlm_head` to `lm_head`.
      (None present — the changed line has no comment.)

### Phase-3 acceptance

- [x] With `optimizer="muon"`, `lm_head.dense.weight` (and untied
      `lm_head.decoder.weight`) land on AdamW-decay, not Muon (Phase 8 test).

---

## Phase 4 — Data loaders & structure-eval field rename

**Files:** `src/oplm/data/sequence/loaders.py`, `src/oplm/eval/tasks/structure.py`.

Rename every `cfg.model.max_seq_len` read to `cfg.model.max_position_embeddings`
(the HF field). These modules otherwise need no change.

- [x] `loaders.py:90` — `max_length=cfg.model.max_position_embeddings` (train builder).
- [x] `loaders.py:126` — `max_length=cfg.model.max_position_embeddings` (eval builder).
- [x] `loaders.py` docstrings at ≈ lines 53 and 112 — update the
      `cfg.model.max_seq_len` mentions to `cfg.model.max_position_embeddings`.
- [x] `structure.py:216` — `self.cfg.model.max_position_embeddings - 2` (log line).
- [x] `structure.py:323` — `len(struct.sequence) + 2 <= self.cfg.model.max_position_embeddings`.
      (Wrapped in `bool(...)` to resolve the pre-existing `Any`-return mypy error
      from Phase 1's `model: Any` change.)

### Phase-4 acceptance

- [~] `grep -rn "max_seq_len" src/` — clean in `loaders.py`/`structure.py`. Two
      hits remain in `cli.py` (lines 106, 163); those are rewritten in Phase 7,
      after which the grep is fully clean (re-verified in Phase 9).

---

## Phase 5 — Trainer loop changes

**File:** `src/oplm/training/trainer.py`.

### 5.1 Model construction + gradient checkpointing (`__init__`, ≈ lines 92–95)

- [x] Replace the broken `model.encoder` line:

  ```python
  # before
  model = OplmForMaskedLM(cfg.model)
  if cfg.model.gradient_checkpointing:
      model.encoder.gradient_checkpointing = True
  # after
  model = OplmForMaskedLM(cfg.model)              # cfg.model is the HF OplmConfig
  if cfg.model.gradient_checkpointing:
      model.gradient_checkpointing_enable()       # propagates to every OplmBlock
  ```

### 5.2 Single optimizer step/zero loop (inside `accelerator.accumulate`, ≈ 200–203)

- [x] Fold the two `for optimizer in self.optimizers` loops into one:

  ```python
  for optimizer in self.optimizers:
      optimizer.step()
      optimizer.zero_grad()
  ```

### 5.3 Accumulation-aware loss for logging (`train`, loop body)

The current code logs `current_loss = loss.item()`, capturing only the **final**
micro-batch. Accumulate a running sum across micro-steps and divide by
`gradient_accumulation_steps` at the optimizer-step boundary.
(`accelerator.backward` scales gradients by `1/grad_accum`; the per-micro-batch
`outputs["loss"]` value is unscaled mean-per-token, so the across-micro-step
mean is the correct step loss. For `grad_accum=1` behavior is unchanged.)

- [x] Before the `while` loop, alongside `current_loss = float("nan")`, add a
      step accumulator:

  ```python
  current_loss = float("nan")
  step_loss_sum = 0.0
  ```

- [x] Inside the `accelerator.accumulate` block, after `self.accelerator.backward(loss)`
      and the optimizer step/zero, accumulate the detached loss. Place the
      accumulation immediately after the existing token/sample tracking (it must
      run on **every** micro-step, before the `if not sync_gradients: continue`):

  ```python
  step_loss_sum += loss.detach().item()
  self._step_local_tokens += int(batch["attention_mask"].sum().item())
  self._samples_seen += len(batch["input_ids"]) * self.accelerator.num_processes

  if not self.accelerator.sync_gradients:
      continue
  ```

- [x] On the sync boundary, replace `current_loss = loss.item()` with the mean
      and reset the accumulator (keep this right after `self.global_step += 1`):

  ```python
  for scheduler in self.schedulers:
      scheduler.step()
  self.global_step += 1
  current_loss = step_loss_sum / cfg.gradient_accumulation_steps
  step_loss_sum = 0.0
  ```

  (`cfg` here is `self.cfg.train`, already bound at the top of `train()`.)

### 5.4 wandb flat-dict fix (`_config_to_flat_dict`, ≈ lines 456–467)

`asdict(cfg)` no longer works because `cfg.model` is a `PretrainedConfig`, not a
dataclass. Flatten the model via `to_dict()` and the dataclass sections via
`asdict`.

- [x] Rewrite the function:

  ```python
  def _config_to_flat_dict(cfg: OplmConfig) -> dict[str, Any]:
      """Flatten OplmConfig to a single-level dict for wandb init."""
      from dataclasses import asdict

      flat: dict[str, Any] = {}
      for key, value in cfg.model.to_dict().items():
          flat[f"model/{key}"] = value
      for section_name in ("train", "data"):
          section = asdict(getattr(cfg, section_name))
          for key, value in section.items():
              flat[f"{section_name}/{key}"] = value
      return flat
  ```

### 5.5 Pass the model to the checkpoint saver (`_save_checkpoint`, ≈ lines 350–365)

The checkpoint now writes an HF export, which needs the (unwrapped) model. Add a
`model=self.model` argument to the `save_checkpoint(...)` call (the new
parameter is added in Phase 6):

- [x] Update the call:

  ```python
  save_checkpoint(
      accelerator=self.accelerator,
      model=self.model,
      cfg=self.cfg,
      output_dir=self.cfg.train.output_dir,
      global_step=self.global_step,
      epoch=self.epoch,
      samples_seen=self._samples_seen,
      tokens_seen=self.tokens_seen,
      save_total_limit=self.cfg.train.save_total_limit,
  )
  ```

### 5.6 Eval integration — leave unchanged

- [x] Confirm (no edit) that `_build_eval_context`, `_run_eval`, and the
      unconditional `accelerator.reduce` of per-step tokens are untouched. This
      is the rank-identical-tokens contract the `EvalContext` relies on.

### Phase-5 acceptance

- [x] Trainer imports without error and `OplmForMaskedLM(cfg.model)` succeeds.
- [ ] Verified via Phase 8 integration/pilot tests. (Pending Phase 8; the
      `model=` kwarg on `save_checkpoint` also flags one `mypy` call-arg error
      until Phase 6 adds the parameter.)

---

## Phase 6 — Checkpointing: config serialization + HF export

**File:** `src/oplm/training/checkpoint.py`.

Replace the broken `OmegaConf.structured(deepcopy(cfg))` serialization and add a
`from_pretrained`-loadable HF export.

### 6.1 `save_checkpoint` signature + body

- [x] Add a `model` parameter (place it right after `accelerator`):

  ```python
  def save_checkpoint(
      accelerator: Accelerator,
      model: torch.nn.Module,
      cfg: OplmConfig,
      output_dir: str,
      global_step: int,
      epoch: int,
      samples_seen: int,
      tokens_seen: int,
      save_total_limit: int = 3,
  ) -> None:
  ```

  Add `import torch` (or only under `TYPE_CHECKING` and type the param as
  `"torch.nn.Module"`) — keep heavy imports lazy where practical.

- [x] Keep `accelerator.save_state(str(checkpoint_dir))` (the resumable state).

- [x] In the `if accelerator.is_main_process:` block:
  - [x] Keep `trainer_state.json` exactly as is (`global_step`, `epoch`,
        `samples_seen`, `tokens_seen`).
  - [x] Replace the frozen-config write with `to_dict()` + dataclass `asdict`:

    ```python
    from dataclasses import asdict

    config_dict = {
        "model": cfg.model.to_dict(),
        "train": asdict(cfg.train),
        "data": asdict(cfg.data),
    }
    (checkpoint_dir / "config.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create(config_dict))
    )
    ```

  - [x] Add the HF export under `hf/` and save the tokenizer alongside it:

    ```python
    from oplm.data import get_tokenizer

    hf_dir = checkpoint_dir / "hf"
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(hf_dir)          # config.json + model.safetensors
    get_tokenizer().save_pretrained(hf_dir)    # tokenizer files for round-trip
    ```

  - [x] Keep `_rotate_checkpoints(Path(output_dir), save_total_limit)` last.

- [x] Keep the trailing `accelerator.wait_for_everyone()`.

- [x] Remove the now-unused `from copy import deepcopy` import.

### 6.2 `load_checkpoint` — leave unchanged

- [x] Confirm (no edit) `load_checkpoint` still reads Accelerate state +
      `trainer_state.json` from the **top level**. The `hf/` export is for
      downstream loading, not resume.

### Phase-6 acceptance

- [x] After a save, `checkpoint-{step}/hf/` contains `config.json`,
      `model.safetensors`, and tokenizer files, and
      `OplmForMaskedLM.from_pretrained(checkpoint-{step}/hf)` reproduces the
      weights (verified via smoke test; Phase 8 adds the formal test).
- [x] `checkpoint-{step}/config.yaml` exists with `model`/`train`/`data` keys and
      is re-loadable by `load_config(["--config", <that file>])`.

---

## Phase 7 — CLI & inference adaptation

**Files:** `src/oplm/cli.py`, `src/oplm/inference.py`.

### 7.1 `cli.py` — `train` one-line summary (≈ line 57)

- [ ] Replace old field names:

  ```python
  console.print(
      f"[bold]Model:[/bold] {cfg.model.num_hidden_layers}L / {cfg.model.hidden_size}D"
  )
  ```

### 7.2 `cli.py` — `info` tables (≈ lines 104–187)

- [ ] Build the model on the meta device exactly as today
      (`OplmForMaskedLM(cfg.model)`), then rebuild both tables on HF fields and
      drop rows for removed features. Replace the Architecture and Features table
      bodies with:

  ```python
  # Architecture
  table.add_row("Parameters", f"{_fmt(total_params)} ({total_params:,})")
  table.add_row("Trainable", f"{_fmt(trainable_params)} ({trainable_params:,})")
  table.add_row("Hidden size", str(cfg.model.hidden_size))
  table.add_row("Layers", str(cfg.model.num_hidden_layers))
  table.add_row("Attention heads", str(cfg.model.num_attention_heads))
  table.add_row("Head dim", str(cfg.model.head_dim))
  table.add_row("Intermediate size", str(cfg.model.intermediate_size))
  table.add_row("FFN activation", cfg.model.ffn_activation)
  table.add_row("Vocab size", str(cfg.model.vocab_size))
  table.add_row("Max positions", str(cfg.model.max_position_embeddings))

  # Features
  features.add_row("Norm type", cfg.model.norm_type)
  features.add_row("Norm strategy", cfg.model.norm_strategy)
  features.add_row("Q/K norm", _status(cfg.model.qk_norm))
  features.add_row("Post-embed norm", _status(cfg.model.post_embed_norm))
  features.add_row("Residual scaling", cfg.model.residual_scaling)
  features.add_row("MLM head act", cfg.model.mlm_head_activation)
  features.add_row("Canon", _status(cfg.model.canon_enabled))
  canon_str = (
      ", ".join(cfg.model.canon_positions) if cfg.model.canon_positions else "[dim]none[/dim]"
  )
  features.add_row("Canon positions", canon_str)
  features.add_row("Flex attention", _status(cfg.model.use_flex_attention))
  features.add_row("Gradient ckpt", _status(cfg.model.gradient_checkpointing))
  features.add_row("Tied embeddings", _status(cfg.model.tie_word_embeddings))
  ```

  Keep the `_fmt` and `_status` helpers as they are.

### 7.3 `cli.py` — `encode` command (≈ lines 65–101)

The old path used `ProteinTokenizer` (deleted) and `model.encoder` (gone).
Collapse to the ESM-C API. `model.logits(...).embeddings` returns the
post-final-norm hidden states `(B, L, hidden_size)`. The state-dict load path
does **not** attach a tokenizer, so attach the shared one when missing.

- [ ] Replace the command body:

  ```python
  @app.command()
  def encode(
      sequences: Annotated[list[str], typer.Argument(help="Protein sequences to encode")],
      model_path: Annotated[
          str,
          typer.Option("--model", "-m", help="Path to model weights file or checkpoint directory"),
      ],
      output: Annotated[str, typer.Option("--output", "-o", help="Output file path")] = "embeddings.pt",
      config: ConfigOpt = None,
      preset: PresetOpt = None,
      overrides: OverridesOpt = None,
  ) -> None:
      """Encode protein sequences to per-residue embeddings."""
      import torch

      from oplm.data import get_tokenizer
      from oplm.model import LogitsConfig

      cfg = resolve_inference_config(
          model_path, config_path=config, preset=preset, overrides=overrides
      )
      model = load_model_for_inference(model_path, cfg)
      if getattr(model, "tokenizer", None) is None:
          model.tokenizer = get_tokenizer()

      with torch.no_grad():
          embeddings = model.logits(
              list(sequences), LogitsConfig(return_embeddings=True)
          ).embeddings

      out_path = Path(output)
      torch.save(embeddings, out_path)
      console.print(f"[green]Saved embeddings[/green] {tuple(embeddings.shape)} → {out_path}")
  ```

- [ ] Remove the stale `from oplm.data.tokenizer import ProteinTokenizer` import.

### 7.4 `inference.py` — prefer the HF export, fall back to state dict

- [ ] `load_model_for_inference` now builds from the HF config (`cfg.model` is the
      HF `OplmConfig`) and prefers `from_pretrained(<checkpoint>/hf)` when an HF
      export is present:

  ```python
  def load_model_for_inference(
      model_path: str | Path,
      cfg: OplmConfig,
      *,
      device: torch.device | str = "cpu",
  ) -> OplmForMaskedLM:
      """Load an inference-ready model.

      Prefers a HuggingFace export (``<dir>/hf`` or a directory that already
      contains ``config.json``); otherwise reconstructs from ``cfg.model`` and a
      bare state-dict file.
      """
      hf_dir = _find_hf_export(Path(model_path))
      if hf_dir is not None:
          model = OplmForMaskedLM.from_pretrained(str(hf_dir))
      else:
          state_dict = load_model_state_dict(model_path)
          model = OplmForMaskedLM(cfg.model)
          model.load_state_dict(state_dict)
      model.to(device)
      model.eval()
      return model


  def _find_hf_export(model_path: Path) -> Path | None:
      """Return a directory loadable by ``from_pretrained`` if one exists."""
      if model_path.is_dir():
          if (model_path / "hf" / "config.json").exists():
              return model_path / "hf"
          if (model_path / "config.json").exists():
              return model_path
      return None
  ```

- [ ] Leave `resolve_inference_config`, `load_model_state_dict`,
      `_find_associated_config`, and `_resolve_state_path` unchanged. (The
      checkpoint `config.yaml` written in Phase 6 still resolves through
      `load_config`; unknown HF keys from `to_dict()` are absorbed.)

### Phase-7 acceptance

- [ ] `oplm info --preset small` prints a table with HF fields and no traceback.
- [ ] `oplm encode <SEQ> --model <checkpoint-dir/hf>` writes an embeddings tensor
      of shape `(1, L, hidden_size)`.

---

## Phase 8 — Test migration & new training suite

### 8.1 Migrate existing tests off the deleted `ModelConfig`

Four tests build `OplmConfig(model=ModelConfig(...))` only to feed a sequence
length (and, in one case, tiny dims) into the data/eval path. Switch each to the
HF config via the `OplmModelConfig` alias. The data/eval code reads only
`cfg.model.max_position_embeddings`, so the minimal model config suffices.

- [ ] `tests/eval/test_sequence_task.py`
  - Replace `ModelConfig` in the `from oplm.config import ...` line; add
    `from oplm.model import OplmConfig as OplmModelConfig`.
  - Change `model=ModelConfig(max_seq_len=_MAX_SEQ_LEN)` →
    `model=OplmModelConfig(max_position_embeddings=_MAX_SEQ_LEN)`.

- [ ] `tests/eval/test_structure_task.py` — identical change to the above.

- [ ] `tests/data/sequence/test_loaders.py` — identical change (drop `ModelConfig`
      from the config import, add the `OplmModelConfig` alias import, swap the
      `model=` construction).

- [ ] `tests/data/test_e2e.py`
  - It already imports `from oplm.model import OplmConfig as OplmModelConfig`.
    Drop `ModelConfig` from the `oplm.config` import.
  - In `_make_data_config`, replace the `ModelConfig(hidden_dim=..., num_heads=...,
    num_kv_heads=..., num_layers=..., max_seq_len=...)` with:

    ```python
    model=OplmModelConfig(
        hidden_size=_HIDDEN,
        num_attention_heads=_HEADS,
        num_hidden_layers=_LAYERS,
        max_position_embeddings=_MAX_SEQ_LEN,
    ),
    ```

  - Leave `_make_model` as is (already uses `OplmModelConfig`).

### 8.2 Un-skip and fix the trainer↔eval integration test

- [ ] `tests/eval/test_trainer_integration.py`:
  - Remove the `pytest.mark.skip(...)` entry from `pytestmark` (keep
    `pytest.mark.slow`).
  - Replace `from oplm.config import DataConfig, ModelConfig, OplmConfig, TrainConfig`
    with `from oplm.config import DataConfig, OplmConfig, TrainConfig` and add
    `from oplm.model import OplmConfig as OplmModelConfig`.
  - In `_cfg`, change the model construction to:

    ```python
    model=OplmModelConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=64,
    ),
    ```

  - Leave the rest (4-step run, `every` parametrization over `{steps: 2}` /
    `{tokens: 1}`, the `_RecordingCallback` assertions) unchanged.
  - Update the module docstring/skip rationale if it still references the skip.

### 8.3 New `tests/training/` suite (mirrors `src/oplm/training/`)

- [ ] Create `tests/training/__init__.py` (empty).

- [ ] `tests/training/test_config.py`:
  - [ ] `load_config([])` yields `cfg.model` that is an instance of
        `oplm.model.OplmConfig` and passes its own validation (no exception).
  - [ ] Preset + CLI override apply:
        `load_config(["--preset","small","model.num_hidden_layers=4"])` →
        `cfg.model.num_hidden_layers == 4` and `cfg.model.hidden_size == 256`.
  - [ ] Derived fields resolve when omitted: with `hidden_size=256,
        num_attention_heads=4`, `cfg.model.head_dim == 64` and
        `cfg.model.intermediate_size` is a positive multiple of 256.
  - [ ] Unknown `model.*` keys are absorbed, not raised:
        `load_config(["model.bogus_key=1"])` succeeds and
        `cfg.model.bogus_key == 1` (documented caveat).
  - [ ] `load_config(["data.max_length=5"])` raises `ValueError` mentioning
        `model.max_position_embeddings`.
  - [ ] **Base-layer is loaded** (not just documentation): build the merged base
        layer directly and assert it is non-empty, e.g. load
        `configs/train/base.yaml` via `importlib.resources.files` and assert its
        `train.seed` equals `load_config([]).train.seed`.
  - [ ] **YAML↔dataclass consistency (drift guard):** for every key present in
        `configs/train/base.yaml` under `train:`, its value equals the
        corresponding `dataclasses.asdict(TrainConfig())[key]`; same for
        `configs/data/base.yaml` under `data:` vs `DataConfig()` (compare scalar
        fields; `train`/`eval` are both `None`). Parametrize over the keys so a
        mismatch names the offending field.
  - [ ] **No typos in `model/base.yaml` (drift guard):** every key under `model:`
        in `configs/model/base.yaml` is a recognized HF field — i.e. appears in
        `set(OplmModelConfig().to_dict())`. (Unknown keys are silently absorbed at
        runtime, so this test is the only thing that catches a misspelled model
        default.)

- [ ] `tests/training/test_optim.py` (build a tiny model with the HF config):

  ```python
  from oplm.model import OplmConfig as OplmModelConfig
  from oplm.model import OplmForMaskedLM
  from oplm.config import TrainConfig
  from oplm.training.optim import partition_optimizer_params

  def _model():
      return OplmForMaskedLM(OplmModelConfig(
          hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
          max_position_embeddings=64))
  ```

  - [ ] With `TrainConfig(optimizer="muon")`, no parameter named with prefix
        `lm_head.` appears in `groups.muon_params`; `lm_head.dense.weight` (and,
        for an untied model, `lm_head.decoder.weight`) appear in
        `adamw_decay_params`. (Map params→names via `id()` lookup over
        `model.named_parameters()`.)
  - [ ] Norms/biases (`ndim <= 1`) and any `"embed"`-named weight are in
        `adamw_no_decay_params`.
  - [ ] The partition covers every trainable parameter exactly once (the function
        already raises otherwise — assert it does **not** raise, and that the
        union of ids equals the set of trainable param ids).
  - [ ] With `optimizer="adamw"`, `muon_params` is empty.

- [ ] `tests/training/test_flops.py`:
  - [ ] `estimate_flops_per_token(OplmModelConfig(...))` is a positive, finite int.
  - [ ] Doubling `num_hidden_layers` increases the estimate; increasing
        `hidden_size` increases the estimate.

- [ ] `tests/training/test_checkpoint.py` (use the real `Accelerator`; mark `slow`
      if it is heavy on the CI box):
  - [ ] Build a tiny `OplmForMaskedLM`, prepare it with an `Accelerator`
        (`mixed_precision="no"`), call `save_checkpoint(accelerator=acc,
        model=model, cfg=cfg, output_dir=tmp, global_step=10, epoch=1,
        samples_seen=…, tokens_seen=…)`.
  - [ ] `tmp/checkpoint-10/hf/` exists; `OplmForMaskedLM.from_pretrained(that)`
        loads and its `lm_head.decoder.bias` (or any leaf weight) matches the
        original via `torch.allclose`.
  - [ ] `load_checkpoint(acc, tmp/checkpoint-10)` returns
        `{"global_step":10,"epoch":1,...}` and restores Accelerate state without
        error.
  - [ ] Rotation: saving more than `save_total_limit` checkpoints leaves exactly
        `save_total_limit` `checkpoint-*` directories (oldest removed).
  - Build `cfg` with `OplmConfig(model=OplmModelConfig(...), train=TrainConfig(...),
    data=DataConfig(...))`.

- [ ] `tests/training/test_pilot_train.py` (`@pytest.mark.slow`, CPU, uses the
      session-scoped `training_parquet` fixture from `tests/conftest.py`):
  - [ ] Build a tiny end-to-end `OplmConfig`:
        `model=OplmModelConfig(hidden_size=32, num_attention_heads=4,
        num_hidden_layers=2, max_position_embeddings=64)`;
        `train=TrainConfig(max_steps=4, batch_size=4, warmup_steps=0,
        wandb_enabled=False, mixed_precision="no", save_every=2,
        save_total_limit=2, output_dir=str(tmp))`;
        `data=DataConfig(train=str(training_parquet),
        eval={"hd": {"path": str(training_parquet), "type": "sequence",
        "every": {"steps": 2}}}, num_workers=0, pin_memory=False)`.
  - [ ] `Trainer(cfg).train()` completes with no shape errors; assert a
        checkpoint dir was written and (via a `_RecordingCallback`) eval fired at
        least once with finite metrics.
  - [ ] Resume: build a second `cfg` with `train.resume_from` pointing at the last
        checkpoint and a larger `max_steps` (e.g. 6); construct a fresh `Trainer`
        and assert `trainer.global_step` equals the resumed step right after
        construction, then `.train()` runs to the new `max_steps`.

### Phase-8 acceptance

- [ ] `grep -rn "ModelConfig" tests/` returns no hits (except intentional prose).
- [ ] `pytest -m "not slow"` is green.
- [ ] `pytest tests/training tests/eval/test_trainer_integration.py` (including
      slow) is green.

---

## Phase 9 — Full verification & cleanup

- [ ] `ruff check src/` — clean (no unused imports left behind, e.g. `deepcopy`,
      `math`, `ProteinTokenizer`).
- [ ] `ruff format src/` — no diffs.
- [ ] `mypy src/` — clean. Resolve any `Any`-related fallout from the
      `OplmConfig.model: Any` change with narrow, code-annotated ignores only if
      genuinely necessary.
- [ ] `pytest` — full suite green (fast + slow).
- [ ] Sanity: `grep -rn "max_seq_len\|num_kv_heads\|hidden_dim\|num_layers\b\|model.encoder\|mlm_head\.\|ProteinTokenizer" src/`
      returns no hits (all migrated to HF names / APIs).
- [ ] Confirm the LR schedule still steps exactly `total_steps` times and the
      decay multiplier is clamped (`progress = min(progress, 1.0)` in
      `get_schedule_fn`) so the final step does not over-advance past the
      `min_lr/lr` floor. (No code change expected — verification only.)

---

## File-by-file change summary

| File | Change | Phase |
|---|---|---|
| `src/oplm/config.py` | delete `ModelConfig` + derived machinery (`round_multiple`, `_VALID_CONV_KERNEL_SCHEDULES`, `_DERIVED_MODEL_FIELDS`); `OplmConfig.model: Any`; `load_config` merges the `base.yaml` layer + builds HF `OplmConfig`; add `_load_packaged_yaml` / `_BASE_CONFIG_LAYERS`; update `data.max_length` message | 1 |
| `src/oplm/configs/model/base.yaml` + `presets/*.yaml` | rewrite with HF field names; drop `num_kv_heads`; base.yaml now authoritative (loaded) | 1 |
| `src/oplm/configs/train/base.yaml` + `train/__init__.py` (**new**) | new authoritative train defaults, section-wrapped under `train:`, mirroring `TrainConfig` | 1 |
| `src/oplm/configs/data/base.yaml` | re-nest under `data:` (was top-level); now authoritative (loaded) | 1 |
| `src/oplm/training/flops.py` | rewrite for HF fields; drop GQA; FFN always 3 projections | 2 |
| `src/oplm/training/optim.py` | Muon head-exclusion prefix `mlm_head.` → `lm_head.` | 3 |
| `src/oplm/data/sequence/loaders.py` | `max_seq_len` → `max_position_embeddings` (x2 + docstrings) | 4 |
| `src/oplm/eval/tasks/structure.py` | `max_seq_len` → `max_position_embeddings` (x2) | 4 |
| `src/oplm/training/trainer.py` | model build + `gradient_checkpointing_enable`; single optimizer loop; accumulation-aware loss; `_config_to_flat_dict`; pass `model=` to `save_checkpoint` | 5 |
| `src/oplm/training/checkpoint.py` | add `model` param; `cfg.model.to_dict()` serialization; HF export + tokenizer under `hf/`; drop `deepcopy` | 6 |
| `src/oplm/cli.py` | `train` print, `info` tables, `encode` via ESM-C API | 7 |
| `src/oplm/inference.py` | HF-config construction; prefer `from_pretrained(<dir>/hf)` | 7 |
| `tests/eval/test_sequence_task.py`, `tests/eval/test_structure_task.py`, `tests/data/test_e2e.py`, `tests/data/sequence/test_loaders.py` | migrate `ModelConfig` → `OplmModelConfig` | 8 |
| `tests/eval/test_trainer_integration.py` | un-skip; build HF `OplmConfig` in `_cfg` | 8 |
| `tests/training/` (new) | `__init__.py`, `test_config.py`, `test_optim.py`, `test_flops.py`, `test_checkpoint.py`, `test_pilot_train.py` | 8 |

---

## Bugs this plan fixes (cross-reference)

1. Model built from the wrong config type (`OplmForMaskedLM(cfg.model)` got the
   dataclass) → Phase 1 + 5.1.
2. `model.encoder` no longer exists (gradient checkpointing crash) → Phase 5.1.
3. Muon head exclusion checks the wrong prefix (`mlm_head.` vs `lm_head.`) → Phase 3.
4. FLOPs read removed fields (`hidden_dim`/`num_kv_heads`/`ffn_dim`) → Phase 2.
5. Checkpoint config serialization breaks on a `PretrainedConfig`
   (`OmegaConf.structured(cfg)`) → Phase 6.
6. wandb flat-dict breaks on a `PretrainedConfig` (`asdict(cfg)`) → Phase 5.4.
7. Stale `data.max_length` rejection message points at the renamed field → Phase 1.3.
8. `cli encode` doubly broken (`ProteinTokenizer` deleted, `model.encoder` gone) → Phase 7.3.
9. Last-micro-batch loss logged as the step loss → Phase 5.3.
10. Double optimizer iteration → Phase 5.2.
</content>
</invoke>

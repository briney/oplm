# Training OPLM

A practical guide to training an OPLM model end to end: prepare data, write a
config, launch (single-GPU, multi-GPU, or DeepSpeed), monitor, checkpoint, and
resume.

This is the *how-to*. Related references:

- [CONFIG.md](CONFIG.md) — every `model.*` / `train.*` / `data.*` field.
- [EVAL_HARNESS.md](EVAL_HARNESS.md) — evaluation during training.
- [DATA_TOOLING.md](DATA_TOOLING.md) — dataset formats and the masking scheme.
- [TRAINER.md](TRAINER.md) — the trainer's internal design.
- [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) — the model itself.

> Training builds a **fresh, randomly initialized** `OplmForMaskedLM` from
> `model.*`. There is no "continue from a pretrained checkpoint" switch on the
> training CLI — `train.resume_from` resumes an *interrupted run of the same
> config*, not arbitrary pretrained weights.

---

## 1. Install

```bash
pip install -e ".[train]"
```

The `train` extra adds Accelerate, Weights & Biases, `datasets`, and BioPython
on top of the core dependency (`torch ≥ 2.10`). Muon (`torch.optim.Muon`) ships
with PyTorch, so no extra install is needed to use it.

---

## 2. Prepare training data

Training data is **parquet** with two required columns:

| Column        | Type   | Required | Description |
|---------------|--------|----------|-------------|
| `sequence_id` | string | yes      | Unique identifier. |
| `sequence`    | string | yes      | Amino-acid sequence (one-letter codes). |
| `masking_weights` | list/array | only if `data.weighted_masking=true` | Per-residue masking weights. |

`data.train` accepts:

- a single `.parquet` file,
- a **directory of shards** (many `.parquet` files), or
- a **named multi-dataset map** with fractional sampling — sources are
  interleaved by their fractions (normalized to 1.0; omitted fractions split the
  remainder evenly):

```yaml
data:
  train:
    uniref50: { path: /data/uniref50/, fraction: 0.6 }
    bfd:      { path: /data/bfd/,      fraction: 0.4 }
```

Masking is **dynamic** (re-drawn every epoch, RoBERTa-style): per sequence, a
fixed count `round(mask_prob · n_eligible)` of positions is selected, then the
BERT 80/10/10 split decides each masked token's replacement (`<mask>` / random
amino acid / unchanged). See [DATA_TOOLING.md](DATA_TOOLING.md) for the full
scheme and [CONFIG.md](CONFIG.md#data-fields-data) for the masking knobs
(`mask_prob`, `mask_token_prob`, `random_token_prob`, `weighted_masking`).

---

## 3. Quick start

The smallest useful run — a size preset, one parquet file, and a few overrides:

```bash
oplm train --preset small \
  data.train=/data/train.parquet \
  train.max_steps=10_000 \
  train.wandb_enabled=false
```

`oplm train` runs in a **single process** (one GPU, or CPU). It prints a one-line
model summary, then starts the loop with a live progress bar. For multiple GPUs,
see [§5](#5-launching).

---

## 4. Configuring a run

Most real runs use a YAML config (plus optional CLI overrides). A representative
masked-LM pretraining config:

```yaml
# my_run.yaml
model:
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  max_position_embeddings: 1024

train:
  max_steps: 100_000
  batch_size: 64
  gradient_accumulation_steps: 2
  optimizer: adamw
  lr: 4e-4
  min_lr: 4e-5
  weight_decay: 0.01
  scheduler: warmup_cosine
  warmup_steps: 5_000
  mixed_precision: bf16
  save_every: 10_000
  save_total_limit: 3
  wandb_project: oplm
  wandb_run_name: medium-uniref50

data:
  train: /data/uniref50/
  mask_prob: 0.15
  num_workers: 8
  eval:
    heldout:
      path: /data/heldout.parquet
      type: sequence
      every: { steps: 10_000 }
```

Launch it:

```bash
oplm train --config my_run.yaml
```

Config sources merge as **defaults → `--preset` → `--config` → CLI overrides**
(later wins). `model.*` fields map to the HuggingFace `OplmConfig`; `train.*` and
`data.*` map to the `TrainConfig` / `DataConfig` dataclasses. Every field, its
default, and its validation rule is documented in [CONFIG.md](CONFIG.md).

`oplm info --config my_run.yaml` prints the resolved architecture and exact
parameter count without training.

---

## 5. Launching

There are three entry points. They share the same argument parsing (`--config`,
`--preset`, `--name`, and bare `key=value` overrides):

| Command | Use it for |
|---------|-----------|
| `oplm train …` | Single process (1 GPU or CPU). Convenience wrapper. |
| `python -m oplm.train …` | Single process; identical, without the Typer wrapper. |
| `accelerate launch -m oplm.train …` | **Multi-GPU / multi-node** (DDP or FSDP via Accelerate). |

### Single GPU / CPU

```bash
oplm train --config my_run.yaml
# or, equivalently:
python -m oplm.train --config my_run.yaml
```

### Multiple GPUs

Configure Accelerate once (pick DDP or FSDP, number of GPUs, etc.):

```bash
accelerate config
```

Then launch through Accelerate — it sets up the distributed context, and the
trainer prepares the model, optimizers, dataloader, and schedulers accordingly:

```bash
accelerate launch -m oplm.train --config my_run.yaml
```

`batch_size` is **per process**, so the global batch is
`batch_size × num_processes × gradient_accumulation_steps`.

### DeepSpeed (opt-in)

DeepSpeed is **disabled by default** — even if your environment has it enabled
globally, the trainer clears the DeepSpeed Accelerate env vars at startup to
avoid spurious DeepSpeed/Triton initialization. Opt in explicitly:

```bash
OPLM_ENABLE_DEEPSPEED=1 accelerate launch -m oplm.train --config my_run.yaml
```

(Configure the DeepSpeed plugin itself via `accelerate config`.)

---

## 6. What a training step does

The trainer is a custom loop built on Accelerate (not the HuggingFace
`Trainer`). Each optimizer step:

1. Pulls a batch — `{input_ids, attention_mask, labels}` — from the masking
   collator; `labels` are the original token ids at masked positions and `-100`
   elsewhere.
2. Runs `OplmForMaskedLM` forward and gets the masked-LM cross-entropy loss.
3. Backpropagates inside `accelerator.accumulate(...)`, so gradients accumulate
   across `gradient_accumulation_steps` micro-batches before an optimizer step.
4. At the accumulation boundary: clips gradients to `train.max_grad_norm` (set
   `0` to disable), steps each optimizer, then steps each LR scheduler exactly
   once.

Mixed precision is set by `train.mixed_precision` (`bf16` default, `fp16`, or
`no`). Activation checkpointing is enabled when `model.gradient_checkpointing`
is true (preset `large` and `xlarge` set it). `model.gradient_checkpointing_mode`
picks the memory/compute tradeoff: `full` (default) recomputes the entire block
on backward for maximum memory savings (~+30% compute), while `selective` keeps
matmul/SDPA outputs resident and recomputes only cheap ops — less memory savings
for substantially less extra compute. The mode is inert unless
`gradient_checkpointing` is true, and both modes compose with `train.compile`.

On multi-GPU with `train.compile=true`, `selective` requires disabling PyTorch's
DDPOptimizer (it splits the compiled graph at gradient-bucket boundaries, which
fragments each block's activation-checkpoint op so the SAC policy is dropped and
`selective` silently collapses to `full` recompute). The trainer detects this
case and sets `torch._dynamo.config.optimize_ddp = False` automatically before
compiling; `full`/`none` are left untouched so they keep DDP comm/compute
overlap. The only cost is that lost overlap — a few ms of allreduce on a ~500 ms
step over NVLink, negligible next to the recompute SAC saves back. For internals,
see [TRAINER.md](TRAINER.md).

---

## 7. Optimizer and schedule

**Optimizer** — `train.optimizer`:

- `adamw` (default): a single AdamW. 2-D weights get `weight_decay`; biases,
  norms, and embeddings are no-decay.
- `muon`: a **hybrid** — eligible 2-D hidden weights are optimized with
  `torch.optim.Muon`, while embeddings, norms, biases, and the MLM head
  (`lm_head.*`) stay on an auxiliary AdamW. Tuned via `train.muon_*`.

**Schedule** — `train.scheduler`: `warmup_linear` (default), `warmup_cosine`,
`wsd_linear`, or `wsd_cosine`. All warm up linearly over `train.warmup_steps`;
WSD variants then hold peak LR for `train.stable_steps` before decaying. Decay
floors at `train.min_lr` (as a fraction `min_lr / lr` of peak). LR is stepped
once per optimizer step, reaching the floor at the final step.

See [CONFIG.md](CONFIG.md#train-fields-train) for every optimizer and scheduler
field.

---

## 8. Evaluation during training

Add datasets under `data.eval`; the trainer runs each on its own cadence and
logs the metrics. A dataset's `every` is exactly one of `{steps: N}` or
`{tokens: N}` (plus optional `at_start` / `at_end`); datasets that omit `every`
inherit `train.eval_every` (default `{steps: 10_000}`).

```yaml
data:
  eval:
    heldout:
      path: /data/heldout.parquet
      type: sequence            # MLM loss / accuracy / perplexity
      every: { steps: 10_000 }
    structures:
      path: /data/pdb
      type: structure           # unsupervised contact precision@L
      every: { tokens: 50_000_000 }
      categorical_jacobian_sample_size: 12
```

Available types: `sequence`, `structure`, `proteingym`, `tape`, `proteinglue`,
`everest`. See [EVAL_HARNESS.md](EVAL_HARNESS.md) for what each computes and
[CONFIG.md](CONFIG.md#eval-datasets) for the per-task keys.

---

## 9. Checkpointing and resume

The trainer writes a checkpoint every `train.save_every` optimizer steps (and
once at the end) under `train.output_dir`:

```
outputs/medium-uniref50/
└── checkpoint-10000/
    ├── trainer_state.json     # global_step, epoch, samples_seen, tokens_seen
    ├── config.yaml            # the full resolved run config (re-loadable)
    ├── hf/                    # HuggingFace export: config.json + model.safetensors + tokenizer
    └── <accelerate state>     # model, optimizer(s), scheduler(s), RNG — for resuming
```

- **`hf/`** is a standard HuggingFace model directory — load it for inference
  with `OplmForMaskedLM.from_pretrained("outputs/medium-uniref50/checkpoint-10000/hf")`
  (see the [README](../README.md#quick-start)).
- **Rotation:** only the most recent `train.save_total_limit` checkpoints are
  kept (oldest deleted by step number).

**Resume** an interrupted run by pointing `train.resume_from` at a checkpoint
directory. It restores model, optimizer, scheduler, RNG, and step counters and
continues:

```bash
oplm train --config my_run.yaml \
  train.resume_from=outputs/medium-uniref50/checkpoint-50000
```

Raise `train.max_steps` first if the original budget was already reached.

---

## 10. Logging and monitoring

**Console** — the main process shows a live `rich` progress bar with current
step, training loss, and the most recent eval loss.

**Weights & Biases** — enabled by default (`train.wandb_enabled`). At startup the
full flattened config is logged under `model/*`, `train/*`, `data/*`. Every
`train.log_every` optimizer steps the trainer logs `train/loss`, `train/lr`,
`train/epoch`, `train/samples`, `train/tokens`, and `train/flops` (a cumulative
estimate); eval passes add `eval/<dataset>/<metric>`. Set `train.wandb_project`
and `train.wandb_run_name` (or `--name`) to organize runs. Disable W&B entirely
with `train.wandb_enabled=false`.

---

## 11. Programmatic API

The CLI is a thin wrapper around `oplm.training.Trainer`. To drive training from
Python — e.g. to attach custom callbacks:

```python
from oplm.config import load_config
from oplm.training import Trainer, TrainerCallback


class PrintLoss(TrainerCallback):
    def on_log(self, trainer, metrics, step):
        if "train/loss" in metrics:
            print(f"step {step}: loss={metrics['train/loss']:.4f}")


cfg = load_config(["--config", "my_run.yaml", "train.max_steps=1000"])
Trainer(cfg, callbacks=[PrintLoss()]).train()
```

`TrainerCallback` hooks (all main-process only): `on_train_start`, `on_log`,
`on_eval_end`, `on_checkpoint_saved`, `on_train_end`.

---

## 12. Troubleshooting

- **Out of memory.** Lower `train.batch_size` and raise
  `train.gradient_accumulation_steps` to keep the global batch constant; enable
  `model.gradient_checkpointing=true`; prefer `mixed_precision: bf16`.
- **`fp16` loss instability.** Use `bf16` on Ampere/Hopper/Blackwell GPUs; it
  needs no loss scaling.
- **Slow or stalling data loading.** Increase `data.num_workers` and
  `data.prefetch_factor`; ensure sharded datasets have enough shards to feed all
  workers.
- **Muon error "requires at least one eligible 2D hidden weight".** The model is
  too small (no eligible 2-D hidden weights) — use `train.optimizer=adamw`.
- **Unexpected `model.*` value.** Unknown/misspelled `model.*` keys are silently
  absorbed by the HuggingFace config and never raise. Run `oplm info` to print
  the resolved architecture and confirm.
- **Triton cache noise / permissions.** The trainer sets `TRITON_CACHE_DIR`
  under `~/.cache/oplm` (falling back to a temp dir) automatically; set it
  yourself to override.

---

## 13. torch.compile

Pass `train.compile=true` to enable `torch.compile(model, dynamic=True)`.

The model is compiled before DDP wrapping (after gradient checkpointing is
enabled, if set), so the compiled graph includes the recompute wrapper and DDP
sees an `OptimizedModule` rather than a raw model — the standard ordering.
`dynamic=True` is mandatory: protein sequences are padded to the batch maximum,
so `seq_len` varies per batch; without dynamic shapes every unique length would
trigger a recompile.

**First-step latency:** compilation runs on the first forward pass and may take
several minutes for large models. Subsequent steps run the compiled graph.
Triton autotune artifacts are cached under `~/.cache/oplm/triton/autotune` so
repeated runs skip recompilation.

**Compile modes** (`train.compile_mode`):

| Mode | When to use |
|------|-------------|
| `default` | Balanced; safe for all hardware. |
| `reduce-overhead` | Uses CUDA graphs to reduce kernel-launch overhead; best for small batch sizes. |
| `max-autotune` | Tries more optimization strategies; longest compile time, best peak throughput on Blackwell. |

Recommended for all multi-GPU production runs. Use `compile_mode=max-autotune`
on Blackwell for best throughput at the cost of a longer initial compile.

```bash
# opt in via CLI override
torchrun --nproc_per_node=8 -m oplm.train --config my_run.yaml train.compile=true
```

---

## See also

- [CONFIG.md](CONFIG.md) — full configuration reference.
- [EVAL_HARNESS.md](EVAL_HARNESS.md) — evaluation tasks and metrics.
- [DATA_TOOLING.md](DATA_TOOLING.md) — data formats and masking.
- [TRAINER.md](TRAINER.md) — trainer internals and design rationale.
- [README](../README.md) — installation and inference.

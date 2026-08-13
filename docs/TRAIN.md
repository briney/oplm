# Training OPLM

A practical guide to training an OPLM model end to end: prepare data, write a
config, launch (single-GPU, multi-GPU, or DeepSpeed), monitor, checkpoint, and
resume.

This is the *how-to*. Related references:

- [CONFIG.md](CONFIG.md) — every `model.*` / `train.*` / `data.*` field.
- [MUP.md](MUP.md) — μP learning-rate transfer across width (tune `lr` once, reuse at scale).
- [SLURM.md](SLURM.md) — running training as a Slurm job (multi-node launch, arrays, resume).
- [EVAL_HARNESS.md](EVAL_HARNESS.md) — evaluation during training.
- [DATA_TOOLING.md](DATA_TOOLING.md) — dataset formats and the masking scheme.
- [OVERVIEW.md](OVERVIEW.md) — Part IV documents the trainer's internal design.
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
oplm train --preset 50M \
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

### Slurm clusters

For multi-node training on a Slurm cluster, `oplm.slurm` turns any training config plus a
`slurm:` block into ready-to-submit `sbatch` scripts (the same `accelerate launch` form as
above, wired up with `srun`, container mounts, and job dependencies) — see
[SLURM.md](SLURM.md) for the full schema and worked examples.

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
is true (the `1B`+ preset recipes carry a commented-out block you can enable).
`model.gradient_checkpointing_mode`
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
see [OVERVIEW.md](OVERVIEW.md) Part IV.

---

## 7. Optimizer and schedule

**Optimizer** — `train.optimizer`:

- `muon` (default): a **hybrid** — eligible 2-D hidden weights are optimized with
  `torch.optim.Muon`, while embeddings, norms, biases, and the MLM head
  (`lm_head.*`) stay on an auxiliary AdamW. Tuned via `train.muon_*`. The default
  pairs with μP (below) and `muon_adjust_lr_fn=original`.
- `adamw`: a single AdamW. 2-D weights get `weight_decay`; biases, norms, and
  embeddings are no-decay.

**Schedule** — `train.scheduler`: `warmup_linear` (default), `warmup_cosine`,
`wsd_linear`, or `wsd_cosine`. All warm up linearly over `train.warmup_steps`;
WSD variants then hold peak LR for `train.stable_steps` before decaying. Decay
floors at `train.min_lr` (as a fraction `min_lr / lr` of peak). LR is stepped
once per optimizer step, reaching the floor at the final step.

See [CONFIG.md](CONFIG.md#train-fields-train) for every optimizer and scheduler
field.

**μP learning-rate transfer** — μP + Muon is the **default**, so `train.lr` is the
μP base LR (`0.01`) and transfers across width: pick any preset and reuse the same
`lr` — no per-size retuning.

```bash
oplm train --preset 1B data.train=/data/uniref50/   # μP + Muon + lr 0.01, by default
```

To run the **vanilla ESM-C recipe** (μP off, AdamW, and the conventional pre-2026
architecture), apply the opt-out overlay (its `lr` is a plain AdamW LR you must
tune per size):

```bash
oplm train --preset 400M --config src/oplm/configs/train/vanilla_esm-c.yaml \
  data.train=/data/uniref50/ train.lr=<adamw-lr-for-this-size>
```

See [MUP.md](MUP.md) for the full recipe, the coord-check gate, and the pilot LR
sweep that produces the base LR.

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

Pass `train.compile=true` to enable `torch.compile`.

The model is compiled before DDP wrapping (after gradient checkpointing is
enabled, if set), so the compiled graph includes the recompute wrapper and DDP
sees an `OptimizedModule` rather than a raw model — the standard ordering.

**Dynamic mode** (`train.compile_dynamic`):

| Value | Behavior |
|-------|----------|
| `true` (default) | One dynamic graph; handles variable sequence lengths without recompilation. |
| `false` | Static graph per concrete shape; best throughput when combined with `data.pad_to_multiple_of` to bound the shape space to a small fixed set of buckets. |
| `null` | Dynamo auto-detects dynamism. |

With `compile_dynamic=false`, the trainer automatically raises
`torch._dynamo.config.cache_size_limit` to accommodate the expected number of
length buckets (computed from `max_position_embeddings / pad_to_multiple_of`). If
`pad_to_multiple_of` is unset with `compile_dynamic=false`, the trainer logs a
warning: batch-max padding yields near-continuous shape variation and will thrash
recompiles.

**First-step latency:** compilation runs on the first forward pass and may take
several minutes for large models. Subsequent steps run the compiled graph.
Triton autotune artifacts are cached under `~/.cache/oplm/triton/autotune` so
repeated runs skip recompilation.

**Compile modes** (`train.compile_mode`):

| Mode | When to use |
|------|-------------|
| `default` | Balanced; safe for all hardware. Required with gradient accumulation (see below). |
| `reduce-overhead` | Uses CUDA graphs to reduce kernel-launch overhead; best for small batch sizes. |
| `max-autotune` | Tries more optimization strategies; longest compile time, best peak throughput on Blackwell. |

Recommended for all multi-GPU production runs. Use `compile_mode=max-autotune`
on Blackwell for best throughput at the cost of a longer initial compile.

> **CUDA-graph modes are incompatible with gradient accumulation.**
> `reduce-overhead` and `max-autotune` enable CUDA graphs, which replay into a
> single set of static buffers. When `gradient_accumulation_steps > 1`, the
> trainer runs several forward/backward micro-steps per optimizer step, and a
> later micro-step's forward overwrites the static output buffer the previous
> micro-step's backward still needs — `RuntimeError: accessing tensor output of
> CUDAGraphs that has been overwritten by a subsequent run`. With
> `gradient_accumulation_steps > 1`, use `compile_mode=default` (Inductor/Triton
> fusion, no CUDA graphs) or `train.compile=false`. CUDA-graph modes are safe only
> at `gradient_accumulation_steps=1`.

```bash
# dynamic graph (default): one compiled graph handles variable-length batches
torchrun --nproc_per_node=8 -m oplm.train --config my_run.yaml train.compile=true

# static-bucket graph: collate to multiples of 128, compile one graph per bucket
torchrun --nproc_per_node=8 -m oplm.train --config my_run.yaml \
  train.compile=true train.compile_dynamic=false data.pad_to_multiple_of=128
```

---

## 14. Pad-to-multiple batching (`data.pad_to_multiple_of`)

By default every batch is padded to the longest sequence it contains
(`pad_to_multiple_of=null`). Setting `pad_to_multiple_of=N` instead pads each
batch to the smallest multiple of N that covers the longest sequence.

**Why it helps:**

- *Tensor-core alignment* (any compile mode, even `compile=false`): cuBLAS/cuDNN
  fast paths prefer leading matmul dimensions divisible by 8 (BF16) or 16 (FP8).
  N ∈ {8, 16} captures this benefit with minimal padding waste.
- *Static-shape bucketing* (`compile_dynamic=false`): each unique padded length
  triggers a separate compiled graph. A small N leaves the shape space
  near-continuous (pathological recompiles); a large N (64/128/256) collapses
  lengths into a small fixed set of buckets so a handful of static graphs are
  compiled once and reused — typically the bigger throughput win.

**Divisibility requirement:** `model.max_position_embeddings % pad_to_multiple_of
== 0` is enforced at config load. This prevents a padded length from exceeding
`max_position_embeddings` and causing position-id overflow. All built-in presets
(context length 1024) divide evenly by 8, 16, 32, 64, 128, and 256.

**Benchmark matrix:**

| Config | `compile_dynamic` | `pad_to_multiple_of` | Notes |
|--------|:-----------------:|:--------------------:|-------|
| A — baseline | `true` | `null` | Default behavior, one dynamic graph. |
| B — alignment | `true` | `8` or `16` | Tensor-core alignment only; minimal padding overhead. |
| C — static-bucket | `false` | `64`, `128`, or `256` | Static graphs per bucket; combine with `compile_mode=max-autotune` on Blackwell for best peak throughput. |

```yaml
# recipe B — dynamic compile + tensor-core alignment
train:
  compile: true
  compile_dynamic: true
data:
  pad_to_multiple_of: 16

# recipe C — static-bucket compile (pick N to suit your sequence-length distribution)
train:
  compile: true
  compile_dynamic: false
  compile_mode: max-autotune
data:
  pad_to_multiple_of: 128
```

---

## 15. Throughput logging

The trainer logs steady-state throughput metrics once per `train.log_every`
steps. The first `train.throughput_warmup_steps` optimizer steps (default `50`)
are excluded from the measurement window to avoid including compile/JIT warmup
latency. Eval and checkpoint steps are also excluded from wall-time accounting.

| Metric | Always logged | Notes |
|--------|:-------------:|-------|
| `train/tokens_per_sec` | yes | Tokens processed per second (headline metric). |
| `train/step_time_s` | yes | Average optimizer-step wall time (seconds). |
| `train/achieved_tflops` | yes | Estimated TFLOPs/s based on `estimate_flops_per_token`. Note: omits attention-score FLOPs; use `tokens_per_sec` for throughput comparison. |
| `train/mfu` | only when `peak_tflops` set | `achieved_tflops / peak_tflops`. Set `train.peak_tflops` to your device peak (e.g. `312.0` for an A100 SXM BF16, `989.5` for an H100 SXM BF16). |

```yaml
train:
  throughput_warmup_steps: 100   # exclude first 100 steps from tput window
  peak_tflops: 989.5             # H100 SXM BF16 dense peak → also logs train/mfu
```

---

## 16. Fault tolerance

Production runs are long enough that node failures, preemptions, and Slurm time limits are
routine, not exceptional. The trainer, the checkpoint layer, and (on Slurm) the requeue wrapper
cooperate so a failed or preempted run resumes automatically rather than losing progress. This
section is the reference for the knobs involved, what actually gets restored on resume, and the
end-to-end walkthrough. For the Slurm-side half (the requeue wrapper, `--signal`, the budget and
no-progress guard), see [SLURM.md §8](SLURM.md#8-requeue-semantics-drain-budget-and-the-no-progress-guard).

### Knobs (`train.*`)

| Field | Default | Meaning |
| --- | --- | --- |
| `save_every_minutes` | `null` | Also checkpoint every N wall-clock minutes, **in addition to** (not instead of) the `save_every` step cadence. `null` disables the timer. |
| `keep_every_n_steps` | `null` | Checkpoints on a step multiple of this value are permanent: excluded from `save_total_limit` rotation entirely. `null` disables the exemption. |
| `keep_every_n_hours` | `null` | Marks a checkpoint permanent at least this many wall-clock hours after the previous permanent one (independent of `keep_every_n_steps`). `null` disables the exemption. |
| `auto_resume` | `false` | Resume from the newest **committed** checkpoint under `output_dir` automatically at trainer start, with no `resume_from` needed. An explicit `resume_from` always wins over `auto_resume`. `oplm slurm generate` injects `train.auto_resume=true` into every job it renders (see [SLURM.md §8](SLURM.md#8-requeue-semantics-drain-budget-and-the-no-progress-guard)); `configs/scaling.yaml` also sets it explicitly, since that config only ever runs under Slurm. |
| `resume_data_position` | `true` | **Reserved, wired in a later phase (Phase 3).** The field exists and round-trips through config serialization, but nothing in the trainer reads it yet — a resume always restarts the dataloader from the beginning of the current epoch (see "What a resume restores" below). |
| `dist_timeout_minutes` | `15` | Timeout passed to Accelerate's `InitProcessGroupKwargs`, bounding every NCCL/gloo collective's wait. A genuine hang raises within this window instead of wedging until the Slurm time limit; a no-op on a single process. |
| `remote_checkpoint_uri` | `null` | An fsspec URI (`s3://`, `gs://`, `file://`, ...) that every committed checkpoint is additionally mirrored to in the background (Task 4.2) — durability beyond local/shared storage. `null` (default) disables it entirely: zero behavior change, no import of `oplm.training.remote`, no fsspec call. See "Remote checkpoint mirror" below. |
| `parallelism` | `ddp` | `ddp` (one full replica per rank, gradients all-reduced) or `hsdp` (FSDP2 `fully_shard` over a 2-D mesh: shard within a node, replicate across nodes). `hsdp` requires world size > 1 and is incompatible with `mixed_precision=fp16`. Checkpoints are parallelism-agnostic, so the same checkpoint resumes under either setting at any world size. |

`save_every`, `save_total_limit`, and `resume_from` are the pre-existing checkpointing knobs —
see [§9](#9-checkpointing-and-resume) and [CONFIG.md](CONFIG.md#train-fields-train).

### Remote checkpoint mirror

When `remote_checkpoint_uri` is set, each committed checkpoint (blocking or the deferred
commit of an async periodic save) is additionally uploaded, on a background daemon thread, to
that fsspec URI, following the same tmp-dir/manifest-last commit discipline as the local
checkpoint layout (`oplm.training.remote.RemoteStore`) — a `checkpoint-<step>/` directory
without a `manifest.json` remotely is uncommitted and invisible to discovery, exactly like a
local `.tmp` dir. Only one upload is ever in flight; a checkpoint that commits while a previous
one is still uploading is queued (at most one slot — a further commit before its turn drops the
queued one, since the newer checkpoint always supersedes an older, not-yet-started upload). On
multi-node runs, each node's `local_process_index == 0` process uploads only the DCP shard files
its own node's ranks wrote; the global main process additionally uploads the shared artifacts
(`.metadata`, `trainer_state.json`, `config.yaml`, `hf/`) — all of this over a dedicated GLOO
process group, never the trainer's own (typically NCCL) default group.

Finalizing the remote manifest is not a bare barrier: every node leader that just finished its
own upload exchanges its checkpoint identity (which step it just uploaded) with every other node
leader over that GLOO group. Only if every leader agrees on the step does the main process write
the manifest and rotate; a bare barrier can't detect two leaders having finished uploading
*different* checkpoints (a node whose own upload queue fell behind and coalesced onto a different
step than its peers), which would otherwise let a manifest get committed while silently missing
that node's files. On a disagreement, an operator will see an ERROR log naming the divergent
steps, and that checkpoint simply stays uncommitted remotely (no manifest.json — invisible to
`latest_committed`/rotation, exactly like a torn upload) rather than ever being finalized
incomplete. This self-heals: because the identity exchange is itself a blocking collective, every
node leader moves to the next round in lockstep, so the next round where every leader's upload
lands on the same step commits normally — genuine, *permanent* divergence needs a persistent
per-node backlog across round boundaries, not just one slow upload. **Storage hygiene note:** a
skipped or torn round leaves a manifest-less `checkpoint-<step>/` directory sitting on the remote
store; rotation never touches it (nothing without a manifest is ever counted or deleted), so these
are safe but do accumulate over time until an operator cleans them up manually.

The drain path (and the natural end of training) blocks on the upload, bounded by a 10-minute
timeout, before proceeding — the local checkpoint is always the fallback resume target
regardless of whether the remote mirror finishes (or finalizes) in time.

`auto_resume` also consults the remote mirror: if no local committed checkpoint validates, or a
remote one exists at a *higher* step than the best local candidate, the remote checkpoint is
downloaded to `output_dir` (becoming the local committed copy) before resuming — the mechanism a
requeued job on a fresh node with no local checkpoints (e.g. node-local NVMe lost on
reallocation) uses to recover purely from the remote mirror. **Multi-node caveat:** this download
happens once, on the main process, before the resume target is broadcast to every other rank —
correct as long as `output_dir` is a shared/network filesystem every rank can read (the common
case on a Slurm cluster, e.g. SUNK). A multi-node run with a **node-local** `output_dir` (e.g.
node-local NVMe) must not rely on this recovery path: only ranks on the main process's own node
would actually see the downloaded checkpoint.

### Drain: checkpoint-before-kill on preemption

The trainer installs a drain trigger (`oplm.training.signals.DrainSignal`) that goes true on
`SIGUSR1`, `SIGTERM`, or a wall-clock margin (600 s) computed from Slurm's `SLURM_JOB_END_TIME` —
whichever arrives first. Once true, the trainer finishes the in-flight optimizer step, saves a
checkpoint (with `tokens_seen`/`global_step` bookkeeping already consistent for that step), logs a
warning, and exits with a reserved exit code (`85`) distinct from `0` (finished `max_steps`) and
any other nonzero exit (a crash). On a plain workstation `SLURM_JOB_END_TIME` is unset, so only the
signals matter; on Slurm, `--signal=USR1@600` (rendered by `oplm slurm generate`) delivers exactly
this signal 600 s before the job's time limit, so the two paths normally agree.

### Async checkpointing

Periodic checkpoints (the `save_every` step cadence and the `save_every_minutes` timer) save via
`torch.distributed.checkpoint`'s asynchronous API by default: the write proceeds on a background
thread while training continues, and the commit itself — the atomic rename onto the checkpoint's
final name, the `latest` pointer update, and rotation — is deferred until every rank's write has
actually finished. This means saves only ever commit when **all** ranks finish writing; a crash
mid-write (this rank's or a peer's) never produces a torn *committed* checkpoint, only an invisible
`checkpoint-<step>.tmp/` staging directory that discovery and rotation ignore and that gets cleaned
up unconditionally the next time a `Trainer` starts against that `output_dir`. Drain and the final
end-of-training save always finish any still-pending async write first, so a preemption is never
blocked behind — or preceded by — a torn, uncommitted periodic save.

### What a resume restores

A resume (`resume_from` or `auto_resume`) restores, via Accelerate's `save_state`/`load_state`:

- model weights,
- **all** optimizer state — including Muon's, not just AdamW's,
- LR scheduler state,
- RNG state (Python, NumPy, CPU and CUDA generators), and
- the trainer's own step counters (`global_step`, `epoch`, `samples_seen`, `tokens_seen`, plus the
  `keep_every_n_hours` bookkeeping and the persisted W&B run id — see below).

**What it does not yet restore: dataloader position.** `resume_data_position` is reserved for a
later phase (Phase 3, data-exact resume); today a resume always restarts the data stream from the
beginning of the current epoch rather than the exact row it was on when interrupted. For a
shuffled, many-epoch pretraining run this is a minor efficiency cost (some rows get seen twice
sooner than they otherwise would), not a correctness issue — but it is real, and worth knowing
before assuming a resume is bit-exact.

### W&B continuity

The trainer persists its W&B run id (into `trainer_state.json`'s `wandb_run_id` key on every
checkpoint, and into an `output_dir/wandb_run_id` marker file as a fallback) the first time it
initializes tracking. On resume, it reads that id back and passes `id=<run_id>,
resume="allow"` to `wandb.init`, so a requeued/resumed run continues logging into the **same** W&B
run instead of starting a new one — one continuous loss curve across any number of
preemptions.

### Distributed hardening (NCCL / preflight / abort)

- **NCCL/dist env** (Slurm-rendered jobs): `NCCL_DEBUG` (default `WARN`, `slurm.nccl_debug`),
  `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` (turns a hung collective into a raised exception instead of
  wedging the job), a trace-buffer + dump-on-timeout pair, and `TORCH_FR_DUMP_TEMP_FILE` pointed
  at `slurm.log_dir` (so a stalled rank's flight-recorder trace is identifiable after the fact).
- **Preflight**: at trainer start, every rank allocates a small buffer and runs a matmul on its
  device, then all ranks participate in one health exchange (regardless of their own local
  pass/fail) so a sick node fails fast and attributably — naming the failing rank(s) — rather than
  hanging the healthy ranks in a later collective.
- **Non-finite-loss abort**: a NaN/inf loss on any rank raises on every rank, exiting nonzero
  (not the drain code). Combined with `auto_resume` and the Slurm requeue wrapper's no-progress
  guard, this acts as an automatic rollback — the next attempt resumes from the last checkpoint
  committed before the poisoned step.
- **Loss-spike warning**: the trainer tracks an EMA of the training loss and logs (never aborts)
  a warning when a step's loss exceeds 3× the EMA, once enough steps have logged for the EMA to
  have settled. This is diagnostic only — it does not affect training or trigger a requeue.

### The failure-recovery walkthrough

1. A node fails, is preempted, or the job hits its time limit → the training process exits
   nonzero (`85` on a clean drain; some other nonzero code on a crash).
2. Slurm requeues the job, subject to `slurm.max_requeues` and the requeue wrapper's no-progress
   (crash-loop) guard — see [SLURM.md §8](SLURM.md#8-requeue-semantics-drain-budget-and-the-no-progress-guard).
3. The new attempt's `Trainer` scans `output_dir` for the newest **committed** checkpoint
   (`auto_resume`, or an explicit `resume_from`) and resumes from it.
4. The same W&B run continues — no new run, no gap in the loss curve.

**Guarantee:** after this phase of work, a requeued production job resumes from the newest
committed checkpoint automatically, with no operator intervention.

---

## See also

- [CONFIG.md](CONFIG.md) — full configuration reference.
- [SLURM.md](SLURM.md) — Slurm job generation and submission.
- [EVAL_HARNESS.md](EVAL_HARNESS.md) — evaluation tasks and metrics.
- [DATA_TOOLING.md](DATA_TOOLING.md) — data formats and masking.
- [OVERVIEW.md](OVERVIEW.md) — Part IV: trainer internals and design rationale.
- [README](../README.md) — installation and inference.

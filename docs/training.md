# Training

How to train an OPLM model: write a config, launch it (single GPU, many GPUs, or Slurm),
monitor it, checkpoint it, resume it, and branch a decay off a published checkpoint. Code
lives in `src/oplm/training/`.

The recipe of record is [`configs/scaling.yaml`](../configs/scaling.yaml): token mode, a WSD
schedule, packed sequences, auto-resume. It is the config behind the 170M long run, and this page
points at it instead of copying its values. Its horizon, learning rate and data mix are
still marked `PENDING` in the file, with the issue each waits on.

Related pages:

- [config.md](config.md): every `model.*` / `train.*` / `data.*` field.
- [slurm.md](slurm.md): running training as a Slurm job (multi-node launch, arrays, requeue).
- [data.md](data.md): the training set, dataset formats and the masking scheme.
- [evaluation.md](evaluation.md): evaluation during training.
- [architecture.md](architecture.md): the model itself.
- [mup-lr.md](mup-lr.md): μP and the learning-rate method (tune `lr` once, reuse at every width).

> Training builds a **fresh, randomly initialized** `OplmForMaskedLM` from
> `model.*`. There is no "continue from a pretrained checkpoint" switch on the
> training CLI. `train.resume_from` resumes an *interrupted run of the same
> config*, and `train.branch_from` forks a new schedule off a checkpoint's full state
> ([Branching a decay off a checkpoint](#branching-a-decay-off-a-checkpoint)). Neither takes
> arbitrary pretrained weights. `train.init_from` does, for a new stage
> ([Training a separate looped stage](#training-a-separate-looped-stage)).

---

## Install

```bash
pip install -e ".[train]"
```

The `train` extra adds Accelerate, Weights & Biases, `datasets`, and BioPython
on top of the core dependency (`torch ≥ 2.10`). Muon (`torch.optim.Muon`) ships
with PyTorch, so no extra install is needed to use it.

---

## Training data

`data.train` takes a parquet file, a directory of parquet shards, or a named map of sources with
fractional sampling. The column contract and the loader are in
[data.md, *Dataset formats*](data.md#dataset-formats), the `data.*` knobs in
[config.md, *Data fields*](config.md#data-fields-data), and the training set itself in
[data.md, *Training data*](data.md#training-data).

Masking is **dynamic**: it is re-drawn every epoch, RoBERTa-style. The scheme (fixed count
`round(mask_prob · n_eligible)` per sequence, then the BERT 80/10/10 replacement split) is in
[data.md, *Masking*](data.md#masking).

---

## Quick start

The smallest useful run — a size preset, one parquet file, and a few overrides:

```bash
oplm train --preset 50M \
  data.train=/data/train.parquet \
  train.max_steps=10_000 \
  train.wandb_enabled=false
```

`oplm train` runs in a **single process** (one GPU, or CPU). It prints a one-line
model summary, then starts the loop with a live progress bar. For multiple GPUs,
see [Launching](#launching).

---

## Configuring a run

Most real runs use a YAML config (plus optional CLI overrides). The production recipe is
[`configs/scaling.yaml`](../configs/scaling.yaml). A small, generic masked-LM pretraining config
looks like this:

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
default, and its validation rule is documented in [config.md](config.md).

`oplm info --config my_run.yaml` prints the resolved architecture and exact
parameter count without training.

---

## Launching

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
[slurm.md](slurm.md) for the full schema and worked examples.

### DeepSpeed (opt-in)

DeepSpeed is **disabled by default** — even if your environment has it enabled
globally, the trainer clears the DeepSpeed Accelerate env vars at startup to
avoid spurious DeepSpeed/Triton initialization. Opt in explicitly:

```bash
OPLM_ENABLE_DEEPSPEED=1 accelerate launch -m oplm.train --config my_run.yaml
```

(Configure the DeepSpeed plugin itself via `accelerate config`.)

---

## What a training step does

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
step over NVLink, negligible next to the recompute SAC saves back.
### Trainer internals

- **Loop.** The loop is one `while self.global_step < self.total_steps`. Cadence work
  (scheduler step, `global_step` increment, logging, eval, checkpoint) runs once per
  **optimizer** step, not per micro-batch. Epoch rollover on `StopIteration` increments `epoch`,
  calls `set_epoch(epoch)` (a deterministic reshuffle) and re-creates the iterator. The logged
  loss is accumulation-aware: a detached running sum across micro-steps divided by
  `gradient_accumulation_steps`, not the last micro-batch alone. A final checkpoint, progress-bar
  stop, `on_train_end` and `accelerator.end_training()` run in a `finally`.
- **Duration.** `train.max_steps` (the default) or `train.max_epochs` (steps per epoch derived
  from the resolved dataset length and the global effective batch). Token mode needs `max_steps`.
  Steps are the native clock. Epochs and tokens are derived overlays, and epoch-based eval cadence
  is not supported ([evaluation.md, *Scheduling during training*](evaluation.md#scheduling-during-training)).
- **Optimizer groups** (`partition_optimizer_params` in `src/oplm/training/optim.py`).
  Trainable parameters split into three groups: no-decay (`ndim <= 1` or `"embed"` in the name:
  biases, norms, embeddings), Muon (2-D weights when `optimizer=muon`, excluding the head by its
  `lm_head.` prefix) and AdamW-decay (other 2-D weights). `build_optimizers` returns one AdamW
  (decay and no-decay groups) or `[Muon, auxiliary AdamW]`. The tied decoder weight lands in
  no-decay through the `"embed"` rule.
- **Schedulers.** A three-phase multiplier (`get_schedule_fn`): warmup (linear 0 to 1), an
  optional stable plateau (WSD only), then decay to `min_lr / lr`. It is wrapped in one `LambdaLR`
  per optimizer and stepped manually each optimizer step, because the `Accelerator` is built with
  `step_scheduler_with_optimizer=False`.
- **FLOPs** (`estimate_flops_per_token`). No GQA (Q/K/V each `hidden_size` to `hidden_size`) and
  an always-gated FFN (3 projections). Per layer: attention projections `2·H·(4H)` plus FFN
  `3·2·H·I`. The head adds `2·H·H + 2·H·V`, and training is about `3×` forward. Attention-score
  FLOPs, norms and embedding lookups are omitted.

---

## Optimizer and schedule

**Optimizer** — `train.optimizer`:

- `muon` (default): a **hybrid** — eligible 2-D hidden weights are optimized with
  `torch.optim.Muon`, while embeddings, norms, biases, and the MLM head
  (`lm_head.*`) stay on an auxiliary AdamW. Tuned via `train.muon_*`. The default
  pairs with μP (below) and `muon_adjust_lr_fn=original`.
- `adamw`: a single AdamW. 2-D weights get `weight_decay`; biases, norms, and
  embeddings are no-decay.

**Weight decay on the vector parameters and embeddings** (both off by default). With
`weight_decay` alone, the token embedding, norm gains, residual gates and biases are never
decayed. Under Adam's roughly fixed-size updates they random-walk, growing about linearly
with steps (in the 170M replication, norm-gain RMS grew from 1.5 to 7.8 over the run).
`train.embedding_weight_decay` pulls the embedding toward 0. `train.weight_decay_to_init`
pulls norm gains toward 1, residual gates toward `model.residual_gate_init` and biases toward
0, i.e. toward their initial values, so they stay free to move but no longer drift. Each step
moves a parameter `lr * wd` of the way to its target, with `lr` its AdamW group's LR (so
`adamw_lr_mult` scales the decay too). Under Adam noise of size about `lr` per step, this
holds a parameter within about `sqrt(lr / (2 * wd))` of its target.

**Schedule** (`train.scheduler`): `warmup_linear` (default), `warmup_cosine`,
`wsd_linear`, or `wsd_cosine`. The multiplier on the peak LR has three phases:

1. **Warmup**: linear from 0 to 1 over steps `0 .. warmup_steps`.
2. **Stable** (WSD only): held at 1 for the next `stable_steps` steps. `stable_steps` counts
   steps **after** warmup, so the plateau ends at step `warmup_steps + stable_steps`. It is ignored
   by the non-WSD schedulers.
3. **Decay**: linear or cosine from 1 to `min_lr / lr` over the remaining
   `decay_steps = max_steps - warmup_steps - stable_steps` steps (at least 1), reaching the floor at
   the final step.

LR is stepped once per optimizer step. Keep `warmup_steps + stable_steps` strictly below
`max_steps` so a decay phase exists. A WSD run is therefore a long plateau that can be cut into
a decay at any step, which is what [Branching a decay off a checkpoint](#branching-a-decay-off-a-checkpoint)
uses.

See [config.md](config.md#train-fields-train) for every optimizer and scheduler
field.

**μP learning-rate transfer** — μP + Muon is the **default**, so `train.lr` (in
`src/oplm/configs/train/base.yaml`) is the μP base LR and transfers across width: pick any
preset and reuse the same `lr` — no per-size retuning.

```bash
oplm train --preset 1B data.train=/data/uniref50/   # μP + Muon + the base LR, by default
```

To train the **ESM-C comparison model** (ESM-C's architecture with AdamW-only μP),
merge `configs/train/esm-c.yaml` over your run config and train on the result:

```bash
python -c "import sys; from omegaconf import OmegaConf as O; \
  O.save(O.merge(O.load(sys.argv[1]), O.load(sys.argv[2])), sys.argv[3])" \
  my-run.yaml src/oplm/configs/train/esm-c.yaml esmc-run.yaml
oplm train --preset 400M --config esmc-run.yaml
```

See [mup-lr.md](mup-lr.md) for the full recipe, the coordinate-check gate, and how the LR
was tuned.

---

## Evaluation during training

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
`everest`. See [evaluation.md](evaluation.md) for what each computes and
[config.md](config.md#eval-datasets) for the per-task keys.

---

## Checkpointing and resume

The trainer writes a checkpoint every `train.save_every` optimizer steps (and
once at the end) under `train.output_dir`:

```
outputs/medium-uniref50/
└── checkpoint-10000/
    ├── trainer_state.json     # global_step, epoch, samples_seen, tokens_seen, data cursor, W&B run id
    ├── config.yaml            # the full resolved run config (re-loadable)
    ├── hf/                    # HuggingFace export: config.json + model.safetensors + tokenizer
    ├── ema.pt                 # EMA tracker (averaged tensors + count); when `train.ema_decay` is set
    ├── hf_ema/                # HuggingFace export of the EMA weights; when `train.ema_decay` is set
    ├── .metadata + shard files  # model, optimizer(s), scheduler(s): torch.distributed.checkpoint
    └── rng_state_<rank>.pt    # per-rank RNG (Python, NumPy, torch CPU/CUDA), for resuming
```

Alongside the checkpoints, `output_dir` holds a `latest` pointer to the newest committed one.

- **`hf/`** is a standard HuggingFace model directory — load it for inference
  with `OplmForMaskedLM.from_pretrained("outputs/medium-uniref50/checkpoint-10000/hf")`
  (see [inference.md](inference.md)).
- **Rotation:** only the most recent `train.save_total_limit` checkpoints are
  kept (oldest deleted by step number). `train.keep_every_n_steps` and `train.keep_every_n_hours`
  mark checkpoints permanent, exempt from that rotation
  ([Fault tolerance](#fault-tolerance)).
- **Run config:** `config.yaml` is the resolved run config (`cfg.model.to_dict()` plus the
  OmegaConf YAML for `train` and `data`), so a checkpoint is self-describing and re-loadable.
  The `hf/` export is for downstream loading. Resume reads the DCP state and
  `trainer_state.json` from the top level.

**Resume** an interrupted run by pointing `train.resume_from` at a checkpoint
directory. It restores model, optimizer, scheduler, RNG, and step counters and
continues:

```bash
oplm train --config my_run.yaml \
  train.resume_from=outputs/medium-uniref50/checkpoint-50000
```

Raise `train.max_steps` first if the original budget was already reached.

---
## Published checkpoints

Intermediate checkpoints of the production run are published so a reader can start from a
stable-phase checkpoint instead of from step 0 (the entry
point of the [reproduce path](reproduce.md#starting-from-a-published-checkpoint)).

Each published checkpoint is a full checkpoint directory, not just weights:

| Part | What it is |
| --- | --- |
| `hf/` | HuggingFace export: config, safetensors weights, tokenizer. Enough for inference and for `train.init_from`. |
| `hf_ema/` and `ema.pt` | The EMA export and tracker, when the run set `train.ema_decay`. |
| DCP shards and `.metadata` | Model and optimizer state, including Muon's moments, and the LR scheduler. |
| `rng_state_<rank>.pt` | Per-rank RNG state. |
| `trainer_state.json` | Step, epoch, samples and tokens seen, and the **data cursor**: the epoch, batches consumed, and the layout it was recorded under (world size, worker count, per-rank batch size, seed). In token mode it also holds each rank's epoch, each worker's rows (and, for a mix, each source's pass and offset), and any reshard events from earlier layout changes. |
| `config.yaml` | The resolved run config the checkpoint was written under. |

The data cursor counts rows into the shard layout of the training set the run read, so it is
exact only against that same set. Each published checkpoint names the training-set tag its cursor
refers to (see [data.md, *Download the exact training set*](data.md#download-the-exact-training-set)).
Resume or branch with a different set and the stream restarts from row 0 of the current epoch
([Branching a decay off a checkpoint](#branching-a-decay-off-a-checkpoint)).

> **Pending:** the checkpoint list (step, phase, Hub repo, dataset tag) and cadence (waiting on #62).

---

## Branching a decay off a checkpoint

`train.resume_from` refuses a changed `lr` / `min_lr` / `scheduler` / `warmup_steps` /
`stable_steps` (and a shrunk `max_steps`) because a resume must continue the *same*
schedule. A **branch** is the opposite intent: start a new run from a checkpoint's full
state (weights, Muon/AdamW moments, step counters, data position) but under *this*
config's LR, `adamw_lr_mult`, weight decay, and schedule. This is the WSD "decay from the
stable phase" step, and it is also how an LR experiment is forked mid-run.

Three fields define a WSD branch: `train.branch_from` (the checkpoint directory),
`train.stable_steps` and `train.max_steps`. The schedule is
[defined by the code](#optimizer-and-schedule) as warmup, then `stable_steps` steps at peak, then
decay to `max_steps`, so with `W = train.warmup_steps`, branching off the checkpoint at step `N`
with a decay of `D` steps means:

```text
stable_steps = N - W        # the plateau ends exactly at step N (stable_steps counts steps after warmup)
max_steps    = N + D        # the decay runs from N to N + D
decay_steps  = max_steps - W - stable_steps = D     # check
```

The live schedule is evaluated at the checkpoint's step, so the first branch step is at peak LR
and the LR falls to `min_lr` at step `N + D`. Keep `warmup_steps`, `lr`, `min_lr` and
`scheduler` as in the main run.

### Worked example

> **Example only.** This uses a hypothetical recipe of 2.5M stable steps plus a 500k decay
> (`max_steps` 3,000,000), with `W = 2,000` as in `configs/scaling.yaml`. It is not a published run.
> `<run>` is the downloaded or local run directory, and `<fresh>` a new output directory.

Build each branch config from the checkpoint's own `config.yaml`, the resolved config of the run
that wrote it, and append your `slurm:` block (`config.yaml` has none). Don't start from
`configs/scaling.yaml`: its LR, data mix and horizon may differ from the run you branch off. This
is the same pattern as the refine-cell decays in
[mup-lr.md, *Decay every refine cell*](mup-lr.md#decay-every-refine-cell), which start from each
cell's `run.yaml`.

The main run's own schedule is `stable_steps = 2,500,000 - 2,000 = 2,498,000` and
`max_steps = 2,500,000 + 500,000 = 3,000,000`.

**A 50k decay off the 2.5M checkpoint** (`N = 2,500,000`, `D = 50,000`):

- `train.stable_steps = 2,500,000 - 2,000 = 2,498,000` (the same as the main run's own plateau)
- `train.max_steps = 2,500,000 + 50,000 = 2,550,000`
- check: `2,550,000 - 2,000 - 2,498,000 = 50,000` decay steps

```bash
# config.yaml has no slurm: block; append the one from your edited configs/scaling.yaml.
{ cat <run>/checkpoint-2500000/config.yaml; awk '/^[^ #]/{p=/^slurm:/} p' configs/scaling.yaml; } > decay-50k.yaml
oplm slurm generate --config decay-50k.yaml --preset 170M --out jobs/decay-50k \
  --name decay-50k train.branch_from=<run>/checkpoint-2500000 train.output_dir=<fresh> \
  train.stable_steps=2498000 train.max_steps=2550000
```

**A 20k decay off the 1.5M checkpoint** (`N = 1,500,000`, `D = 20,000`). Only the plateau end
and the horizon move:

- `train.stable_steps = 1,500,000 - 2,000 = 1,498,000`
- `train.max_steps = 1,500,000 + 20,000 = 1,520,000`
- check: `1,520,000 - 2,000 - 1,498,000 = 20,000` decay steps

```bash
{ cat <run>/checkpoint-1500000/config.yaml; awk '/^[^ #]/{p=/^slurm:/} p' configs/scaling.yaml; } > decay-20k.yaml
oplm slurm generate --config decay-20k.yaml --preset 170M --out jobs/decay-20k \
  --name decay-20k train.branch_from=<run>/checkpoint-1500000 train.output_dir=<fresh> \
  train.stable_steps=1498000 train.max_steps=1520000
```

The checkpoint at `N` must exist: only the permanent ones survive rotation
(`train.keep_every_n_steps`). The real long-run decay branches were built the same way, as
`train.branch_from=<ckpt-dir> train.max_steps=N+decay train.stable_steps=N-warmup` with a warmup of
2,000. Without Slurm, run the same overrides through `oplm train` or `accelerate launch -m oplm.train`.

For a constant-LR branch (an LR experiment) set `stable_steps = max_steps - warmup_steps`; the one
trailing decay step is harmless.

### Rules

- `branch_from` points at the checkpoint **directory** (not its `hf/` export). The
  optimizer state is what distinguishes a branch from `init_from`. It is mutually exclusive with
  `resume_from` and `init_from`.
- `output_dir` must be fresh. If it already holds a committed checkpoint and
  `auto_resume`/`resume_from` resolve it, that resume wins and `branch_from` is ignored.
  This is what lets a requeued branch job continue as a plain resume of itself.
- `max_steps` must exceed the checkpoint's step. All other config (model geometry, data
  layout) must match the checkpoint, exactly as for a resume.
- The data cursor is restored only when `data.train` equals the checkpoint's `data.train`
  exactly, as written: every source name, fraction and path, absolute paths included. The same
  data at a different path counts as different. For a downloaded checkpoint, put the training set
  at the paths its `config.yaml` names (mirror or symlink them). Any difference, including a
  different mix (the production decay-phase case), restarts the stream at row 0 of the current
  epoch with a warning. A kept cursor follows the same layout rules as a resume
  ([Resuming on different hardware](#resuming-on-different-hardware)).

The branch gets its own W&B run; `train/lr` (Muon) and `train/lr_adamw` show the live
schedule from the first logged step. The worked set of branch experiments for the
late-loss investigation (decay, AdamW-LR, all-LR, and weight-decay branches, plus the
WSD branch-point comparison) is in
[LATE_LOSS_INVESTIGATION.md](https://github.com/briney/oplm/blob/7db9c1bf0155fc9b4c05c8b358b3cf4cd6648f2c/docs/LATE_LOSS_INVESTIGATION.md).
The decay branches used to rank LR candidates are in
[mup-lr.md, *Decay every refine cell*](mup-lr.md#decay-every-refine-cell).

---

## Resuming on different hardware

Resume is **data-exact** by default: `train.resume_data_position=true` replays the exact
row-level position of the checkpoint, so no row is re-seen or skipped.

A **token-mode** cursor (the production recipe,
[Token-budget batching](#token-budget-batching-traintokens_per_step-trainbatch_tokens)) resumes
at **any world size and worker count**; only the seed must match. At a new layout the cursor
becomes a reshard event: the rows each old `(rank, worker)` stream had not yet consumed are
re-striped over the new streams, and every rank starts at the earliest epoch any old rank was in.
The event is kept in the cursor and replayed by later resumes until every rank is past it, so a
run can change layout more than once. For a `data.train` mix the resume is exact per source pass:
no source row is served twice in a pass, and the epoch keeps its total draw count, but the mix
order after the resume differs from an uninterrupted run. Resharding a mix needs each source's
position in the cursor (`worker_sources`). An older cursor without it resumes a mix only at its own
layout; at another layout it raises, naming `train.resume_data_position=false`.

A **sequence-mode** cursor still requires the **same world size, worker count, per-rank batch
size and seed**. `train.resume_data_position` validates this and **raises** on a mismatch, naming
every differing field. To resume one on other hardware, set `train.resume_data_position=false`.
That restarts the current epoch's data stream from row 0, so some rows are seen again; for a
shuffled, many-epoch run that is an efficiency cost, not a correctness problem. Everything else
(weights, optimizer, scheduler, counters) still restores, and the checkpoint itself is
parallelism-agnostic, so it loads at any world size.

One more constraint comes from the per-rank RNG files. A checkpoint saved by `N` ranks carries
`rng_state_0.pt` to `rng_state_<N-1>.pt`. Resuming at a *smaller* world size ignores the extra
files. At a *larger* one, each rank past the checkpoint's sidecar count is reseeded with
`train.seed + rank` and a warning is logged. The environment variable `OPLM_ALLOW_MISSING_RNG=1`
only applies to a sidecar missing below the count (or a checkpoint with none): it starts that
rank's RNG fresh instead of raising.

A branch has one more limit: it keeps the cursor only when `data.train` matches the checkpoint's
config as written, paths included ([Rules](#rules)). The check does not compare the data itself,
so the same training set at a new path loses the cursor. Mirror or symlink the original paths.

---

## Logging and monitoring

**Console** — the main process shows a live `rich` progress bar with current
step, training loss, and the most recent eval loss.

**Weights & Biases** — enabled by default (`train.wandb_enabled`). At startup the
full flattened config is logged under `model/*`, `train/*`, `data/*`. Every
`train.log_every` optimizer steps the trainer logs `train/loss`, `train/lr`,
`train/epoch`, `train/samples`, `train/tokens`, and, when the task provides a FLOP
estimate ([Training tasks](#training-tasks-traintask)), `train/flops` (a cumulative estimate);
eval passes add `eval/<dataset>/<metric>`. Every row is logged at the training step, so
W&B's `Step` axis equals `train/global_step` (also logged on every row). Set
`train.wandb_project` and `train.wandb_run_name` (or `--name`) to organize runs. Disable W&B
entirely with `train.wandb_enabled=false`.

Each train log also carries per-window diagnostics (the window is the steps since
the previous log): `train/loss_mean` (mean loss over the window, versus the
single-step `train/loss`), `train/grad_norm` / `train/grad_norm_max` (pre-clip
global gradient norm, mean and max) and `train/clip_frac` (fraction of steps where
clipping fired; all three only when `max_grad_norm > 0`), `train/mean_seq_len`
(tokens per sample), `train/batch_fill` (token mode: real tokens over capacity), `train/data_wait_s` (seconds per optimizer step the main
process spent blocked on the DataLoader; near 0 unless data loading starves the GPU),
and under Muon `train/lr_adamw` (the auxiliary AdamW optimizer's first group —
`train/lr` is the Muon LR).

**Weight-norm diagnostics** (`train.weight_diag_every=N`, off by default): every N
optimizer steps, `diag/weight_rms/<group>` (pooled RMS of the group's weights) and
`diag/update_ratio/<group>` (that step's update RMS divided by the weight RMS — the
effective relative step size) for the groups `embed`, `head_dense`, `head_decoder`,
`attn`, `mlp`, `norm_gain`, `residual_gate`, `canon`, `bias`, `other`. Main process
only, DDP only, one full parameter snapshot on the diagnostic step. The same numbers
can be backfilled from existing checkpoints' `hf/` exports:

```bash
oplm weight-rms /runs/170M/checkpoint-100000 /runs/170M/checkpoint-400000
```

**Profiling** (`train.profile_start=S`, off by default; `train.profile_steps=N`,
default 20). It writes everything to `<output_dir>/profile/`:

- **At train start:** all-reduces of 16, 128 and 512 MB on the run's process group, with
  bus bandwidth logged. Set `slurm.nccl_debug: INFO` to see which transport NCCL chose.
- **Steps `S` to `S+N−1`:** each phase of the step is timed with a CUDA synchronize on
  both sides: `data_wait`, `forward`, `backward` (which includes DDP's overlapped
  gradient all-reduce), `clip`, `optimizer` and `sync`. Each rank writes its medians to
  `summary_rank<r>.json`.
- **The next `N` steps:** these run under `torch.profiler` without synchronizing. Rank 0
  writes `trace_rank0.json` (open it in Perfetto or `chrome://tracing`) and the top
  operators by GPU and CPU time.

Comparing one node with two at the same per-GPU batch separates communication cost
from compute.

---

## Programmatic API

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

## Troubleshooting

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

## torch.compile

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

## Pad-to-multiple batching (`data.pad_to_multiple_of`)

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

### Sequence packing (`data.pack_sequences`)

Padding every sequence to the batch's longest wastes most of a step on short
proteins: at 512 tokens, a typical 128-sequence batch holds about 22.6k real
tokens in 65.5k slots. `data.pack_sequences: true` packs each training batch's
sequences back to back into as few rows of at most `max_position_embeddings`
tokens as fit (first-fit decreasing, after masking). The batch still holds
`train.batch_size` sequences, so the optimization is unchanged; only fewer pad
tokens get computed.

The model keeps sequences apart. Attention uses a block-diagonal mask, so a
token attends only within its own sequence. Canon taps that would reach into a
neighboring sequence contribute zero. RoPE positions need no reset, because
attention scores depend only on query–key offsets within a sequence. Each
sequence's logits, loss and gradients match its own padded row (the tests check
this).

Costs and limits:

- With the default `model.packed_attention: sdpa`, attention gets a dense block-diagonal
  `(rows, 1, T, T)` mask instead of a key-padding mask, and computes every score of a row: its
  cost grows with the row width, not with the sequences' lengths. (FlashAttention is out
  either way: it takes no mask at all.) If the result is slower than the padding saved, turn
  packing off: the off path is unchanged.
- `model.packed_attention: flex` runs packed batches through FlexAttention with a block mask
  built once per forward from `segment_ids`, skipping every 128×128 block between sequences.
  The intent is that attention cost follows each sequence's own length, so wide rows (2048)
  cost about what narrow ones do; [`docs/perf/attention_width_bench.py`](https://github.com/briney/oplm/blob/7db9c1bf0155fc9b4c05c8b358b3cf4cd6648f2c/docs/perf/attention_width_bench.py) measures that against
  `sdpa` by row width, and [`docs/perf/packing_check.py`](https://github.com/briney/oplm/blob/7db9c1bf0155fc9b4c05c8b358b3cf4cd6648f2c/docs/perf/packing_check.py) checks its numbers against padded rows
  at the bf16 floor. It is fast only under `train.compile` (the trainer warns on CUDA without
  it), has no attention dropout, and has no backward on CPU (the tests enable PyTorch's
  reference backward to check its gradients against `sdpa`). With sequence-count packing, rows
  and width vary per batch, so a compiled run with `train.compile_dynamic: false` recompiles
  flex for every new shape; the default `compile_dynamic: true` avoids that.
- It is incompatible with `model.mask_dropout`, which rescales by each row's
  masked fraction. The loader refuses that combination.
- Evaluation batches are never packed, so eval numbers stay comparable.
- `oplm stream-replay` replays a packed run unpacked: packing happens after
  masking, so it serves the same sequences and masks.

### Token-budget batching (`train.tokens_per_step`, `train.batch_tokens`)

Counting a batch in sequences makes its real size depend on the data: the same 8192 sequences
hold about 1.1M tokens at a deepclust-heavy mix and 1.9M at a uniref-heavy one, and far fewer
than `batch_size × max_position_embeddings` once packed. Token mode fixes the token count instead.

- `train.tokens_per_step` is the global capacity per optimizer step and `train.batch_tokens`
  each process's micro-batch capacity (`rows × model.max_position_embeddings`). Accumulation is
  derived: `tokens_per_step / (batch_tokens × processes)`, which must be a whole number. The same
  `tokens_per_step` therefore works at any node count and model size.
- Each data worker fills `rows` rows first-fit in stream order and closes a batch at the first
  sequence that fits nowhere; that sequence starts the next batch. Batches are exactly
  `(rows, width)`, so shapes never change. In simulation, real tokens fill about 97–99% of capacity
  for 2048-wide rows (more with more rows per GPU) and less for narrow, few-row shapes; `train/batch_fill`
  reports the actual value. Masking is unchanged.
- Each GPU rolls over to its next epoch on its own, since sequences per batch vary.
  `train/epoch` stays the global fractional epoch (samples seen across all GPUs over the dataset size);
  the integer epoch saved in checkpoint state is rank 0's.
- Resume is data-exact per worker: the checkpoint cursor records each rank's epoch and the rows
  each worker had emitted. Only the seed must match, so `tokens_per_step` and `batch_tokens`
  can change at a resume or branch without losing the position, and a changed world size or
  worker count re-stripes the unconsumed rows
  ([Resuming on different hardware](#resuming-on-different-hardware)). A sequence-mode checkpoint resumes or branches into token mode at its exact position
  (same world size, workers and seed); the reverse, token to sequence, needs
  `train.resume_data_position=false`.
- The worker-cycle save alignment is off in token mode; after a resume only the order in which
  a GPU's workers' batches interleave can shift within one cycle.
- Token mode often derives accumulation above 1, and the CUDA-graph compile modes
  (`train.compile_mode: max-autotune` or `reduce-overhead`) are incompatible with gradient
  accumulation (see the CUDA-graph note in [torch.compile](#torchcompile)); use `compile_mode: default` there.
- Requires `data.pack_sequences: true` and `train.max_steps`. `oplm stream-replay` does not
  support token-mode runs.

---

## Throughput logging

The trainer logs steady-state throughput metrics once per `train.log_every`
steps. The first `train.throughput_warmup_steps` optimizer steps (default `50`)
are excluded from the measurement window to avoid including compile/JIT warmup
latency. Eval and checkpoint steps are also excluded from wall-time accounting.

| Metric | Always logged | Notes |
|--------|:-------------:|-------|
| `train/tokens_per_sec` | yes | Tokens processed per second (headline metric). |
| `train/step_time_s` | yes | Average optimizer-step wall time (seconds). |
| `train/achieved_tflops` | only when the task provides a FLOP estimate ([Training tasks](#training-tasks-traintask)) | Estimated TFLOPs/s based on `estimate_flops_per_token`. Note: omits attention-score FLOPs; use `tokens_per_sec` for throughput comparison. |
| `train/mfu` | only when `peak_tflops` set | `achieved_tflops / peak_tflops`. Set `train.peak_tflops` to your device peak (e.g. `312.0` for an A100 SXM BF16, `989.5` for an H100 SXM BF16). |

```yaml
train:
  throughput_warmup_steps: 100   # exclude first 100 steps from tput window
  peak_tflops: 989.5             # H100 SXM BF16 dense peak → also logs train/mfu
```

---

## Fault tolerance

Production runs are long enough that node failures, preemptions, and Slurm time limits are
routine, not exceptional. The trainer, the checkpoint layer, and (on Slurm) the requeue wrapper
cooperate so a failed or preempted run resumes automatically rather than losing progress. This
section is the reference for the knobs involved, what actually gets restored on resume, and the
end-to-end walkthrough. For the Slurm-side half (the requeue wrapper, `--signal`, the budget and
no-progress guard), see [slurm.md, *Requeue semantics*](slurm.md#requeue-semantics-drain-budget-and-the-no-progress-guard).

### Knobs (`train.*`)

| Field | Default | Meaning |
| --- | --- | --- |
| `save_every_minutes` | `null` | Also checkpoint every N wall-clock minutes, **in addition to** (not instead of) the `save_every` step cadence. `null` disables the timer. |
| `keep_every_n_steps` | `null` | Checkpoints on a step multiple of this value are permanent: excluded from `save_total_limit` rotation entirely. `null` disables the exemption. |
| `keep_every_n_hours` | `null` | Marks a checkpoint permanent at least this many wall-clock hours after the previous permanent one (independent of `keep_every_n_steps`). `null` disables the exemption. |
| `auto_resume` | `false` | Resume from the newest **committed** checkpoint under `output_dir` automatically at trainer start, with no `resume_from` needed. An explicit `resume_from` always wins over `auto_resume`. `oplm slurm generate` injects `train.auto_resume=true` into every job it renders (see [slurm.md, *Requeue semantics*](slurm.md#requeue-semantics-drain-budget-and-the-no-progress-guard)); `configs/scaling.yaml` also sets it explicitly, since that config only ever runs under Slurm. |
| `resume_data_position` | `true` | **Data-exact resume.** When a checkpoint carries a data cursor and this is `true` (the default), a resume replays the exact row-level position the run was at when the checkpoint was taken — no row re-seen or skipped, same order as an uninterrupted run — instead of restarting the epoch's data stream from row 0. A token-mode cursor needs only the same seed and re-stripes onto any world size or worker count; a sequence-mode cursor's layout (`world_size`, `num_workers`, per-rank batch size, seed) is validated against the live run, and a mismatch raises, naming the escape hatch. Set `false` to restart the current epoch from row 0 — the escape hatch for a sequence-mode cursor on a different layout, or a checkpoint with no cursor at all. See "What a resume restores" below and [Resuming on different hardware](#resuming-on-different-hardware). |
| `dist_timeout_minutes` | `15` | Timeout passed to Accelerate's `InitProcessGroupKwargs`, bounding every NCCL/gloo collective's wait. A genuine hang raises within this window instead of wedging until the Slurm time limit; a no-op on a single process. |
| `remote_checkpoint_uri` | `null` | An fsspec URI (`s3://`, `gs://`, `file://`, ...) that every committed checkpoint is additionally mirrored to in the background — durability beyond local/shared storage. `null` (default) disables it entirely: zero behavior change, no import of `oplm.training.remote`, no fsspec call. See "Remote checkpoint mirror" below. |
| `ema_decay` | `null` | EMA of the trainable weights (fold stages use `0.999`). One update per optimizer step. Each checkpoint adds `ema.pt` + `hf_ema/`; resume restores the tracker; a pre-EMA checkpoint resumed with this set logs a warning and restarts the average from the live weights. `ddp` only. |
| `parallelism` | `ddp` | `ddp` (one full replica per rank, gradients all-reduced) or `hsdp` (FSDP2 `fully_shard` over a 2-D mesh: shard within a node, replicate across nodes). Checkpoints are parallelism-agnostic, so the same checkpoint resumes under either setting at any world size. `hsdp` requires world size > 1 and currently refuses three combinations, each of which would otherwise hang or silently diverge: **configured `data.eval`** (in-loop eval all-gathers on rank-striped forward counts and deadlocks — evaluate an HSDP run's checkpoints with a separate `ddp` job), `mixed_precision=fp16` (the GradScaler's inf-check is not shard-aware), and `stability_diagnostics` with `stability_probe_every > 0` (the probe's main-process-only forward all-gathers). See `oplm.training.parallel` for the full limitation list. |

`save_every`, `save_total_limit`, and `resume_from` are the pre-existing checkpointing knobs —
see [Checkpointing and resume](#checkpointing-and-resume) and [config.md](config.md#train-fields-train).

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
(`.metadata`, `trainer_state.json`, `config.yaml`, `ema.pt` and `hf_ema/` when EMA is on, `hf/`) — all of this over a dedicated GLOO
process group, never the trainer's own (typically NCCL) default group.

Finalizing the remote manifest is not a bare barrier: every node leader that just finished its
own upload exchanges its checkpoint identity (which step it just uploaded) *and* whether that
upload actually succeeded with every other node leader over that GLOO group. Only if every leader
reports success on the same step does the main process write the manifest and rotate; a bare
barrier can't detect two leaders having finished uploading
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

An upload that *fails* (a transient fsspec error, or local rotation deleting the source
`checkpoint-<step>/` while the background thread is still reading it) is caught, logged, and
reported into that same exchange as a failed round: nothing is finalized, and the leader still
makes its collective call, so the group never drifts out of lockstep. The next checkpoint's round
mirrors and commits normally. Local rotation is deliberately *not* coordinated with in-flight
uploads (no pinning) — a deleted source is simply a failed round, not a lost run.

The drain path (and the natural end of training) blocks on the upload before proceeding — the
local checkpoint is always the fallback resume target regardless of whether the remote mirror
finishes (or finalizes) in time. That wait is nested *inside* the 10-minute drain margin the
whole drain mechanism exists to beat, so its budget is derived from the wall clock rather than
being a second, independent 10 minutes: when `SLURM_JOB_END_TIME` is set, the upload wait is
`end_time - now - 60s` clamped to `[0s, 600s]`; without that env var (e.g. a workstation run) it
falls back to 180s.

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
warning, writes a `.drained` marker file into `output_dir`
(`oplm.training.signals.DRAIN_MARKER_NAME`), and exits with a reserved exit code (`85`) distinct
from `0` (finished `max_steps`) and any other nonzero exit (a crash). On a plain workstation
`SLURM_JOB_END_TIME` is unset, so only the signals matter; on Slurm, `--signal=USR1@600` (rendered
by `oplm slurm generate`) delivers exactly this signal 600 s before the job's time limit, so the
two paths normally agree.

The marker file, not the exit code, is what the Slurm requeue wrapper actually keys its drain
branch on: the rank processes do exit 85, but `accelerate launch --multi_gpu` reports any worker
failure as its *own* exit 1 (torchelastic's `ChildFailedError`), so on a real multi-GPU job the 85
never survives to the sbatch script. The wrapper consumes (deletes) the marker on every requeue,
and the trainer clears a stale one at startup (the batch node dying before the wrapper ran is the
only way one survives), so a marker can never misclassify a later genuine crash as a drain — see
[slurm.md, *Requeue semantics*](slurm.md#requeue-semantics-drain-budget-and-the-no-progress-guard).

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

A resume (`resume_from` or `auto_resume`) restores, via `torch.distributed.checkpoint` (DCP;
model/optimizer state) plus per-rank sidecars (RNG, and the fp16 `GradScaler` when in use):

- model weights,
- **all** optimizer state — including Muon's, not just AdamW's,
- LR scheduler state,
- the EMA tracker (`ema.pt`: averaged tensors and update count) when `train.ema_decay` is set,
- RNG state (Python, NumPy, CPU and CUDA generators), and
- the trainer's own step counters (`global_step`, `epoch`, `samples_seen`, `tokens_seen`, plus the
  `keep_every_n_hours` bookkeeping and the persisted W&B run id — see below).

**Dataloader position: data-exact resume.** Every checkpoint additionally carries a
data cursor (`{epoch, batches_consumed_this_epoch}` plus the layout it was recorded under —
world size, worker count, per-rank batch size, seed). With `resume_data_position=true` (the
default), a resume converts that cursor into a per-stream sample offset and skips there **by
arithmetic** — walking shard metadata (row counts) rather than reading and discarding rows — so
the resumed run sees exactly the same sequence of training examples, in the same order, as an
uninterrupted control run; no row is re-seen or skipped, and `tokens_seen` matches the
uninterrupted run exactly. A checkpoint taken exactly at an epoch boundary resumes into the next
epoch rather than replaying the finished one. Masking is unaffected by this: the collator's masking
RNG is untouched, so post-resume masks differ bit-for-bit from an uninterrupted run even though the
*data* stream is identical (data-exact, not bitwise). A token-mode cursor re-stripes onto a new
world size or worker count ([Resuming on different hardware](#resuming-on-different-hardware)). A
sequence-mode cursor whose recorded layout doesn't match the live run's (different world size,
worker count, batch size, or seed) raises rather than silently reproducing the wrong data order or
coverage; the fix is either to resume at the original layout or
set `resume_data_position=false` to restart the current
epoch's data stream from row 0 — a minor efficiency cost, not a correctness issue, for a shuffled,
many-epoch run). A checkpoint with no cursor at all always takes that same
restart-from-row-0 fallback, regardless of `resume_data_position`. When the striping remainder gives
ranks unequal batch counts, the single per-run cursor assumes lockstep ranks and the shorter rank's
resume is exact only to within `num_workers` batches per epoch.

### W&B continuity

The trainer persists its W&B run id (into `trainer_state.json`'s `wandb_run_id` key on every
checkpoint, and into an `output_dir/wandb_run_id` marker file as a fallback) the first time it
initializes tracking. On resume, it reads that id back and passes `id=<run_id>,
resume="allow"` to `wandb.init`, so a requeued/resumed run continues logging into the **same** W&B
run instead of starting a new one — one continuous loss curve across any number of
preemptions.

Rows are logged at the training step, and W&B never accepts a step below its latest. After a
crash, training resumes from the last committed checkpoint, which can be behind the steps the
dead attempt already logged. In that case the trainer first rewinds the run's history to the
checkpoint (`wandb.init(resume_from="<id>?_step=<checkpoint step + 1>")`), so the resumed
attempt's rows replace the dead attempt's and the history records only the run that produced
the model. W&B enables rewinding per entity (private preview, on request to W&B support).
Where it is not enabled, the trainer logs a warning and resumes without it: the rolled-back
steps keep the dead attempt's rows, and the resumed attempt's metrics for those steps are
dropped, including any (such as an eval) the dead attempt never got to log. A drain checkpoint never needs a rewind, because it is saved before its step is
logged. `tests/training/test_wandb_rewind_online.py` is the opt-in drill that checks the
server behaviour (`OPLM_WANDB_ONLINE_DRILL=1`).

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
   nonzero (a clean drain writes the `.drained` marker and exits `85` — flattened to `1` by
   `accelerate launch`, which is why the marker exists; a crash exits with whatever it raised).
2. Slurm requeues the job, subject to `slurm.max_requeues` and the requeue wrapper's no-progress
   (crash-loop) guard — see [slurm.md, *Requeue semantics*](slurm.md#requeue-semantics-drain-budget-and-the-no-progress-guard).
3. The new attempt's `Trainer` scans `output_dir` for the newest **committed** checkpoint
   (`auto_resume`, or an explicit `resume_from`) and resumes from it.
4. The same W&B run continues — no new run, no gap in the loss curve.

**Guarantee:** a requeued production job resumes from the newest
committed checkpoint automatically, with no operator intervention.

### Known limitations

Everything above is implemented and tested; these are the specific, deliberate gaps to know
about before relying on it in production:

- **Eval is refused under HSDP.** `train.parallelism=hsdp` together with any configured
  `data.eval` raises at config-load time (`oplm.config.validate_parallelism_compat`), not just
  a documentation warning. Mechanism: `fully_shard` mutates the model in place, so the evaluator
  gets handed the `FSDPModule` itself, and every eval forward all-gathers sharded parameters;
  eval tasks stripe their work across ranks (`[rank::world_size]`), so ranks issue *different
  numbers* of forwards, and the ranks that finish early never issue the matching all-gathers —
  the whole group wedges until `dist_timeout_minutes`, then crash-loops through the requeue.
  Follow-up work (not yet implemented): rank-padded forward counts, or gathering the model once
  into an unsharded eval copy. Until then, evaluate an HSDP run's checkpoints with a separate
  `ddp` job against the same checkpoint.
- **Muon's numerics on HSDP's sharded (DTensor) parameters are on the first-real-run watch
  list.** `torch.optim.Muon` has no DTensor-specific handling — its Newton–Schulz iteration runs
  as real distributed matmuls over sharded operands. An early spike established that this
  *runs* and checkpoints correctly (loss decreases, state round-trips), not that it is
  numerically identical to the unsharded computation; review measured ~3% divergence against a
  single-device oracle on a toy model. Nothing suggests it is wrong — distributed matmul is the
  mathematically correct way to compute it — but watch the first real Muon+HSDP production run's
  loss curve against a DDP control.
- **HSDP refuses `mixed_precision=fp16`.** `TrainConfig.__post_init__` raises on
  `parallelism=hsdp` + `mixed_precision=fp16`: under FSDP2 every gradient is a sharded DTensor,
  and `torch.amp.GradScaler`'s inf-check runs per rank over local shards with no cross-rank
  reduction, so ranks could disagree about skipping a step and silently desynchronize. Use
  `bf16` (default) or `no` under `hsdp`.
- **Gradient accumulation reduce-scatters every micro-batch under HSDP**, not just at the
  optimizer-step boundary. `Accelerator.no_sync` looks for a `no_sync` attribute that
  `FSDPModule` doesn't define, so it degrades to a null context — correct (gradients still
  accumulate into the sharded `.grad`) but extra communication on every micro-batch instead of
  just the last one. The proper fix (FSDP2's `set_requires_gradient_sync`/
  `set_is_last_backward`, driven per micro-batch) is left out rather than half-wired, since
  getting only one half of that pair right silently corrupts gradients. Throughput cost, not a
  correctness issue.
- **GPU/NCCL and real multi-node paths are verified structurally, not exercised in CI.** The
  test suite runs on CPU/gloo only (2-process `tests/training/test_e2e_hsdp.py`,
  `tests/training/test_fsdp2_muon_spike.py`; `tests/slurm/` renders and asserts sbatch script
  text without submitting to a scheduler) — there is no GPU or real-Slurm CI runner. Everything
  NCCL-specific (flight recorder paths, `TORCH_NCCL_ASYNC_ERROR_HANDLING`, actual multi-node
  rendezvous) is reviewed and unit-tested at the config/rendering level but has not run against
  real GPUs or a live multi-node Slurm allocation as part of this work.
- **Manifest-less remote checkpoint directories accumulate (storage hygiene, not correctness).**
  A skipped or torn upload-finalization round (see "Remote checkpoint mirror" above) leaves a
  `checkpoint-<step>/` directory with no `manifest.json` sitting in the remote store forever;
  rotation never touches it because nothing without a manifest is ever counted or deleted. Safe
  (never mistaken for a valid checkpoint) but requires an operator to clean up stale
  manifest-less directories periodically on long-running production buckets.

---

## Training tasks (`TrainTask`)

The Trainer owns acceleration, optimization, accumulation, eval cadence, fault
tolerance and checkpointing. The objective-specific third — building the model,
building the training dataloader, turning one micro-batch into a loss, and the
FLOP estimate — lives behind `oplm.training.task.TrainTask`:

| Method | Returns |
| --- | --- |
| `build_model(cfg, initialization_source)` | the trainable `nn.Module` (weights loaded from `train.init_from` when given) |
| `build_dataloader(cfg)` | the rank/worker-striped training `DataLoader` |
| `step(model, batch)` | `StepResult(loss, tokens, samples, metrics)` |
| `flops_per_token(cfg)` | an `int`, or `None` to omit `train/flops`, `train/achieved_tflops` and `train/mfu` |

`MLMTask` is the default (`Trainer(cfg)`), and reproduces the pre-seam behaviour
exactly. `StepResult.metrics` are logged as `train/<key>`, averaged over the log
window; the keys the trainer emits itself (`loss`, `lr`, `epoch`, `samples`,
`tokens`, `flops`, ...) are reserved.

`FoldTask` (`oplm.fold.training`) is the second implementation: `oplm fold train` /
`python -m oplm.fold.train` build `Trainer(cfg, task=FoldTask())`; batches are `FoldBatch` objects
moved through their own `.to`, and `train/<term>` metrics come from `StepResult.metrics`
([FOLD.md, *Milestone 2b: losses and training*](FOLD.md#9-milestone-2b-losses-and-training-oplmfoldlosses-oplmfoldtraining)).

---

## Training a separate looped stage

Keep the physical architecture and data settings in `parent.yaml`. Train the ordinary
model, then start a new run from its exported weights:

```bash
oplm train --config parent.yaml train.output_dir=outputs/parent train.max_steps=1000000
oplm train --config parent.yaml train.init_from=outputs/parent/checkpoint-1000000/hf train.output_dir=outputs/looped train.max_steps=500000 model.num_loops=2 model.loop_strategy=stack
```

`train.init_from` accepts a local Hugging Face export or a checkpoint directory with
an `hf/` export. It loads all model parameters and persistent buffers strictly;
non-loop model semantics must match. It starts fresh optimizers, LR schedules,
random state, data position, counters, and tracking identity. Set the new stage's
LR, warmup, and duration independently. Use a distinct output directory; the
export, its containing checkpoint, and its identifiable parent run directory
are rejected as outputs.

A resolved `train.resume_from` or `train.auto_resume` checkpoint takes precedence
and restores the current stage's complete state without accessing `init_from`.
An empty auto-resume directory starts from `init_from`; an invalid explicit resume
or unusable set of auto-resume checkpoints fails instead of starting over.

For interleaving, set `model.loop_strategy=interleave`. To loop a subset, also set
`model.loop_start` and `model.loop_end` (zero-based, end exclusive). Unselected
layers run once. Omitting `init_from` trains the configured looped model from
scratch with shared parameters from the first step.

Full-state resume checks physical depth and all loop settings before loading
model or optimizer state. `loop_end=null` and an explicit end equal to physical
depth are equivalent; other changes, including strategy changes at `num_loops=1`,
require a new stage with `init_from`. Older checkpoints missing loop fields use
ordinary defaults. If both saved model-config artifacts are absent, only an
ordinary default target may resume, with a warning that compatibility cannot be
verified.

Startup logs show physical depth, effective depth, loop range/strategy, and unique
parameter count. FLOPs and throughput-derived compute metrics count every block
execution, with the MLM head counted once. The estimator still omits attention
scores, normalization, and embedding lookups. Compare looping experiments at both
equal token budgets and equal estimated compute; shared weights leave parameter
and optimizer-state budgets unchanged, while additional executions increase
compute and saved activations (mitigated by activation checkpointing).

When compiling looped models with activation checkpointing, the trainer specializes
Python configuration floats (such as normalization epsilon) to avoid a PyTorch
2.10 tracing failure across repeated checkpoint calls. Tensor shapes remain dynamic
according to `train.compile_dynamic`. Direct library users combining these features
with `torch.compile` should set `torch._dynamo.config.specialize_float = True`
before the first compiled forward. This is a process-wide compiler setting;
changing configuration floats can cause recompilation.

---

## See also

- [config.md](config.md): full configuration reference.
- [slurm.md](slurm.md): Slurm job generation and submission.
- [evaluation.md](evaluation.md): evaluation tasks and metrics.
- [data.md](data.md): the training set, data formats and masking.
- [architecture.md](architecture.md): the model.
- [README](../README.md): installation and inference.

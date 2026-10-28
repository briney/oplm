# Reproduce OPLM training

The path from raw data to a trained, evaluated model, in order. Each step says what to do, gives
the command, and links to the doc that holds the details. This page does not repeat them. Every
step starts from the recipe config [`configs/scaling.yaml`](../configs/scaling.yaml); edit its
cluster paths, partition and container for your site first (see [slurm.md](slurm.md)). The LR sweep
in step 2 needs a derived copy of it, described there.

Full-scale training needs a multi-node GPU cluster. If you only want to try a different decay, skip
to [Starting from a published checkpoint](#starting-from-a-published-checkpoint), which is far
cheaper.

OPLMFold training will get its own path later. Its development notes are in [FOLD.md](FOLD.md).

## 1. Get the training and eval data

Download the published training set, or rebuild it from the primary sources. The eval sets are
published separately. Point `data.train` and `data.eval` in the config at the downloaded folders.

> **Pending:** the published training set and its Hub tag (waiting on #61).

- Download or rebuild the training set: [data.md, *Download the exact training set*](data.md#download-the-exact-training-set)
  and [*How the training set was built*](data.md#how-the-training-set-was-built).
- Eval sets: [data.md, *Evaluation data*](data.md#evaluation-data) and
  [evaluation.md, *Published structure eval sets*](evaluation.md#published-structure-eval-sets).

## 2. Determine the learning rate with μP

Tune `train.lr` once on the 170M proxy, rank candidates on loss after a decay, confirm at 800M, and
reuse the result at every width. Run the coordinate check first. The recipe config already carries
the learning rate from the last funnel, so this step is only needed to re-derive it (a different
batch, context, mix or optimizer). Make a sweep config from `configs/scaling.yaml` as in
[mup-lr.md, *Sweep config*](mup-lr.md#sweep-config), then set `CONFIG=...` and `SWEEP=...`
(absolute paths) as in [*Run the coordinate check*](mup-lr.md#run-the-coordinate-check).
The commands below are the first two of the funnel.

```bash
oplm sweep coord-check --config $CONFIG --scaling width --widths 384,768,1536 --depth 24 \
  --base-width 768 --out $SWEEP/coord-width
oplm sweep smoke --config $CONFIG --global-tokens 4194304 --out $SWEEP/smoke --submit
```

The full funnel (coarse, refine, decay every cell, replicate, confirm at 800M) is in
[mup-lr.md, *Commands*](mup-lr.md#commands). How the best LR falls with training length is in
[*LR vs training length*](mup-lr.md#lr-vs-training-length).

> **Pending:** the production LR from the long runs (waiting on #55).

## 3. Run the stable phase at production scale (or the whole schedule)

Generate and submit the sbatch for your preset. The job auto-resumes after preemption, so submitting
once is enough. The config's WSD schedule comes from `train.warmup_steps`, `train.stable_steps` and
`train.max_steps`, and it includes its own decay tail at the 170M long-run horizon, so rendering it
as-is runs the whole schedule. Splitting the stable phase from the decay is what the branching in
step 4 does.

> **Pending:** the production horizon (waiting on #55).

```bash
oplm slurm generate --config configs/scaling.yaml --preset 170M --out jobs/170M
oplm slurm submit jobs/170M
oplm slurm status jobs/170M
```

`generate` renders the config's `train.batch_tokens` (65536, the 170M micro-batch) for every
preset; it does not read `slurm.max_batch_tokens`. For any other preset, pass that preset's cap on
the command line, for example `train.batch_tokens=32768` for 400M. Without it, 400M runs at twice
its cap and 800M and 1B fail at launch. Whether those caps fit in memory has not been checked on
hardware.

```bash
oplm slurm generate --config configs/scaling.yaml --preset 400M --out jobs/400M train.batch_tokens=32768
```

- Launch, nodes and requeue behavior: [slurm.md, *`oplm slurm generate`, `submit` and `status`*](slurm.md#oplm-slurm-generate-submit-and-status).
- The training loop, schedule and checkpoints: [training.md, *Optimizer and schedule*](training.md#optimizer-and-schedule)
  and [*Checkpointing and resume*](training.md#checkpointing-and-resume).
- Watching a run: [training.md, *Logging and monitoring*](training.md#logging-and-monitoring).

## 4. Run the decay phase

The decay is a branch off a stable-phase checkpoint: it keeps the optimizer state and ramps the LR
down. Build the branch config from the checkpoint's own `config.yaml` plus your `slurm:` block, then
set `train.branch_from`, `train.stable_steps` and `train.max_steps`. The arithmetic is in the worked
example. The same commands are how you try a different decay off a published checkpoint.

```bash
# config.yaml has no slurm: block; append the one from your edited configs/scaling.yaml.
{ cat <run>/checkpoint-2500000/config.yaml; awk '/^[^ #]/{p=/^slurm:/} p' configs/scaling.yaml; } > decay-50k.yaml
oplm slurm generate --config decay-50k.yaml --preset 170M --out jobs/decay-50k \
  --name decay-50k train.branch_from=<run>/checkpoint-2500000 train.output_dir=<fresh> \
  train.stable_steps=2498000 train.max_steps=2550000
```

The data cursor carries over only if `data.train` in that config matches the checkpoint's exactly,
absolute paths included, so keep the training set at the paths the checkpoint names (mirror or
symlink them). Otherwise the branch restarts the data stream from row 0 of the epoch.

The example is hypothetical, not a published run. Mechanics and rules:
[training.md, *Branching a decay off a checkpoint*](training.md#branching-a-decay-off-a-checkpoint).
Decay shape, length and floor: [mup-lr.md, *Decay recipe*](mup-lr.md#decay-recipe).

> **Pending:** the production decay recipe (shape, length fraction, LR floor) (waiting on #55).

> **Pending:** the production data mix for the stable and decay phases (waiting on #58).

## 5. Evaluate

Evaluation runs inside training through `data.eval`, on its own cadence per dataset: held-out
sequence loss, structure contact precision@L and ProteinGym. Give the run the eval sets from
step 1, then read the logged metrics. What each task computes and how to schedule it:
[evaluation.md](evaluation.md#tasks-and-metrics) and
[*Scheduling during training*](evaluation.md#scheduling-during-training). Per-task keys are in
[config.md, *Eval datasets*](config.md#eval-datasets). To embed sequences with the result, see
[inference.md](inference.md).

> **Pending:** the expected loss and precision@L ranges for the reference run (waiting on #55).

Before you trust a changed setup, run the tests in [testing.md](testing.md).

## Starting from a published checkpoint

This is the same pipeline, entered at step 4 (or partway through step 3). Download a published
stable-phase checkpoint, then branch a different decay off it with `train.branch_from`. For
example, with a recipe of 2.5M stable steps plus a 500k decay, you can run a 50k decay off the 2.5M
checkpoint, or a 20k decay off the 1.5M checkpoint. Only the plateau end and the horizon change.

> **Pending:** the published checkpoints, their steps and Hub locations (waiting on #62).

1. Pick a checkpoint from [training.md, *Published checkpoints*](training.md#published-checkpoints).
   It lists what a checkpoint contains and which training-set tag its data cursor refers to.
2. Download that training-set tag (step 1) and put it at the absolute paths the checkpoint's
   `config.yaml` lists under `data.train` (mirror or symlink them). The cursor is kept only when
   `data.train` matches the checkpoint's exactly, paths included; any difference restarts the stream
   from row 0.
3. Branch the decay with the commands in step 4, building the config from the checkpoint's own
   `config.yaml`, not from `configs/scaling.yaml`, whose LR, mix and horizon are still pending and may
   differ from the published run. Point its `data.eval` paths at your copies of the eval sets
   (those paths do not affect the cursor). What must match the original run, and what can
   change freely (decay length, shape, LR floor, decay data mix), is in
   [training.md, *Branching a decay off a checkpoint*](training.md#branching-a-decay-off-a-checkpoint).
4. With fewer GPUs than the original run, data-exact resume is not possible. The limit and the
   workaround (`train.resume_data_position=false`) are in
   [training.md, *Resuming on different hardware*](training.md#resuming-on-different-hardware).
5. Evaluate as in step 5.

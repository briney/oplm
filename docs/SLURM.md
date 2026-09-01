# Running OPLM Training on Slurm

A practical guide to turning an ordinary oplm training config into Slurm job scripts and
submitting them, using `oplm.slurm` — a general layer with no μP or sweep concepts in it. It
targets a CoreWeave SUNK (Slurm-on-Kubernetes) cluster with Pyxis/enroot containers, but nothing
about the schema is CoreWeave-specific.

This is the *how-to* for the general layer. Related references:

- [TRAIN.md](TRAIN.md) — training itself (config, launch, checkpointing).
- [CONFIG.md](CONFIG.md) — every `model.*` / `train.*` / `data.*` field.
- [LR_SWEEP.md](LR_SWEEP.md) — the μP learning-rate sweep, a higher-level tool built on top of
  this layer (per-phase job arrays, ranking, resubmission). Nothing on this page requires it.

---

## 1. Overview

Add a `slurm:` block to any oplm training config, then run `oplm slurm generate` to write an
`sbatch` script for it. `oplm slurm submit` queues the scripts a `generate` call wrote, and
`oplm slurm status` reports which of them are still known to the scheduler.

The `slurm:` block is validated but otherwise inert to everything else in the config: `load_config`
accepts it as an unrecognized top-level key, and `serialize_config` (the writer used for a
resolved, re-loadable run config) omits it — so a `slurm:` block never leaks into a training run's
own saved config, and adding one to `configs/scaling.yaml` changes nothing about how that config
trains.

```bash
oplm slurm generate --config configs/scaling.yaml --preset 400M --out jobs/400M
```

writes `jobs/400M/oplm-400M.sbatch` plus a small `jobs.json` manifest, ready for `oplm slurm
submit jobs/400M` (§8 has the full worked example).

---

## 2. The `slurm:` block

Every field, its type, and its default (`SlurmConfig` in `src/oplm/slurm/config.py`):

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `partition` | string | *required* | Slurm partition (`#SBATCH --partition`). |
| `time_limit` | scalar or table | *required* | Wall-clock limit(s). Four accepted forms — §3. |
| `nodes` | scalar or table | *required* | Node count(s). Same four forms — §3. |
| `max_batch_size` | map of preset → int | `{}` | Per-preset memory cap on per-device batch, consumed by the batch-plan derivation (§4). |
| `log_dir` | path | *required* | Directory for `--output`/`--error` logs. |
| `env_file` | path | *required* | Shell script `source`d at job start; creates the job's node-local `$JOB_WORK_DIR` and exports object-storage / W&B credentials. |
| `container_image` | path | *required* | Enroot/Pyxis `.sqsh` container image (`--container-image`). |
| `container_mounts` | list of `host:container` strings | *required* | Extra bind mounts, **beyond** the automatic home-directory mount (see below). |
| `install` | string | *required* | Shell command run in-container before training (e.g. `pip install oplm[train]`) — no baked-in dependency beyond a working Python + CUDA stack. |
| `gpus_per_node` | positive int | `8` | GPUs requested per node (`--gres=gpu:N`). |
| `cpus_per_task` | positive int | `128` | `--cpus-per-task`. |
| `mem` | string | `"0"` | `--mem`; the conventional Slurm meaning of `"0"` is "all memory on the node." |
| `exclusive` | bool | `true` | Whether to pass `--exclusive`. |
| `max_concurrent` | positive int | `4` | Array-job throttle: rendered as `--array=0-N%max_concurrent` (§6). |
| `account` | string or null | `null` | Optional `--account` for billing/allocation; omitted from the header when unset. |
| `max_requeues` | positive int | `20` | Requeue budget consulted by the rendered requeue wrapper — §8. |
| `nccl_debug` | string | `"WARN"` | Value exported as `NCCL_DEBUG`. Set `INFO` for verbose rendezvous/topology logging; noisy for routine production runs. |

A minimal, real block — `configs/scaling.yaml`'s own `slurm:` section:

```yaml
slurm:
  partition: hpc-mid
  time_limit:
    default: "168:00:00"
  cpus_per_task: 128
  gpus_per_node: 8
  exclusive: true
  mem: "0"
  log_dir: /mnt/home/briney/logs
  env_file: /mnt/home/briney/.env
  container_image: /mnt/data/containers/deeplearning_v2026-05-26.sqsh
  container_mounts:
    - /mnt/data:/mnt/data
    - /tmp:/tmp
  install: pip install oplm[train]
  max_concurrent: 4
  max_requeues: 20
  nodes: {170M: 4, 400M: 8, 800M: 16, 1B: 16}
  max_batch_size: {170M: 256, 400M: 256, 800M: 256, 1B: 128}
```

`bool` values are rejected everywhere an int is expected (`gpus_per_node`, `cpus_per_task`,
`max_concurrent`, every `max_batch_size` entry) even though Python's `bool` is an `int` subclass —
YAML's bare `true`/`false` would otherwise silently coerce to `1`/`0`.

### The automatic home mount

Every job additionally mounts `/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER}` — the
submitting user's own home directory — ahead of whatever `container_mounts` lists. This is added
at render time (it is user-specific, so it cannot live in a shared config), not configured, and it
is why `container_mounts` only needs to list *shared* paths like a data or scratch volume.

---

## 3. `nodes` and `time_limit`: four accepted forms

Both fields share one small format (`PhaseTable` in `oplm/slurm/config.py`), because both need to
vary per model-size preset and, for tools that generate several related jobs, per phase:

```yaml
nodes: 4                                          # one value for every job
nodes: {170M: 1, 400M: 4}                         # per preset
nodes: {default: {170M: 1}, bridge: {170M: 4}}    # per phase, then per preset
time_limit: {default: "168:00:00", analyze: "1:00:00"}   # per phase, no preset dimension
```

The presence of a `default` key is what distinguishes the last two forms from the second — a
bare per-preset map with no `default` key is the second form, not a one-entry "phase" table.
Resolution looks in the requesting phase's own table first (exact preset match, then a `*`
wildcard for "any preset in this phase"), then falls back to the `default` table the same way. A
phase's own wildcard beats `default`'s exact-preset entry — the phase's table is exhausted
completely before `default` is ever consulted.

**`oplm slurm generate` never resolves by phase** — it has no `--phase` concept, so it always
resolves with `phase=None`, meaning only a bare scalar or the `default` table (or its own
top-level per-preset form) is reachable. The per-phase form exists for phase-aware callers built
on this layer — e.g. the μP sweep's phase generator, which resolves `nodes`/`time_limit` once per
named phase (`smoke`, `bridge`, `confirm`, …; see [LR_SWEEP.md](LR_SWEEP.md)). A plain training
config normally only needs the scalar or per-preset form.

Because `configs/scaling.yaml`'s `nodes` table has no `default`/wildcard entry, `--preset` is
required there — omitting it raises `nodes has no entry for phase=None preset=None` rather than
picking an arbitrary preset.

---

## 4. Per-device batch and gradient accumulation are derived, not configured

`oplm.slurm.config.resolve_batch_plan(*, global_examples, world_size, max_batch_size)` turns a
target **global** batch (in examples per optimizer step), a world size, and a per-preset memory
cap into a `BatchPlan(per_device_batch, gradient_accumulation_steps, world_size)`:

1. `base = global_examples // world_size` — the batch each process would train per accumulation
   cycle before any accumulation is applied.
2. Search `accum = 1, 2, 3, …` for the **smallest** value such that `base` divides evenly by it
   *and* `base // accum <= max_batch_size`.
3. That `accum` is `gradient_accumulation_steps`; `base // accum` is `per_device_batch`.

Node counts are chosen for wall time (§3); per-device batch is a *consequence* of that node count,
never configured directly — so a preset can move to more nodes without anyone hand-editing its
batch size. An **indivisible** configuration (the global batch does not divide evenly by the world
size at all) is a hard error, never a silently adjusted global batch:

```pycon
>>> from oplm.slurm.config import resolve_batch_plan
>>> resolve_batch_plan(global_examples=4096, world_size=24, max_batch_size=256)
Traceback (most recent call last):
    ...
ValueError: global batch 4096 is not divisible by world size 24
```

Worked examples against `configs/scaling.yaml`'s own `nodes`/`max_batch_size` table
(`gpus_per_node=8`, so `world_size = nodes * 8`), verified by calling the function directly:

| Preset | Nodes | World size | Global examples | Per-device batch | Accum |
| --- | ---: | ---: | ---: | ---: | ---: |
| `170M` | 4 | 32 | 4096 | 128 | 1 |
| `170M` | 4 | 32 | 16384 | 256 | 2 |
| `400M` | 8 | 64 | 4096 | 64 | 1 |
| `400M` | 8 | 64 | 16384 | 256 | 1 |
| `800M` | 16 | 128 | 4096 | 32 | 1 |
| `1B` | 16 | 128 | 16384 | 128 | 1 |

**`oplm slurm generate` itself does not call this.** It renders whatever `train.batch_size` /
`train.gradient_accumulation_steps` your config already specifies (the packaged defaults are
`batch_size=32`, `gradient_accumulation_steps=1`, unaffected by node count) — a plain training job
sets its own per-device batch directly, the ordinary way (see [TRAIN.md](TRAIN.md)).
`max_batch_size` and `resolve_batch_plan` are the mechanism a phase-aware tool built on this layer
uses to plan `train.batch_size`/`train.gradient_accumulation_steps` *before* writing each job's
config — the μP sweep's phase generator is the current example (see [LR_SWEEP.md](LR_SWEEP.md));
both fields are part of the general schema so any similar tool can reuse the same derivation.

---

## 5. One launcher form for every job

Every rendered job — one node or many — uses the same `srun` + `accelerate launch` body
(`oplm.slurm.render.render_job` / `accelerate_command`):

```bash
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
  --export=ALL \
  --container-image=<container_image> \
  --container-mounts=<mounts> \
  --container-workdir="$JOB_WORK_DIR" \
  --no-container-mount-home \
  bash -c '<install> && \
    accelerate launch \
    --multi_gpu \
    --mixed_precision bf16 \
    --num_machines $SLURM_NNODES \
    --num_processes $((SLURM_NNODES * <gpus_per_node>)) \
    --machine_rank $SLURM_PROCID \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port $MASTER_PORT \
    -m oplm.train \
    --config <config>'
```

`$SLURM_NNODES`, `$SLURM_PROCID`, and `$MASTER_ADDR` are left unexpanded at render time on
purpose — the whole training command is wrapped in *single* quotes so the container's own shell
resolves them (they reach it via `--export=ALL`), once per array/job task, on every node the
allocation actually has. That is also why there is no separate single-node code path: at
`SLURM_NNODES=1` the same `srun --nodes=$SLURM_NNODES` degrades to a single-process launch
correctly, with nothing to special-case.

Three lines above the `srun` set up distributed rendezvous once per job, on the rank-0 (batch)
node:

```bash
MASTER_ADDR=$(hostname --ip-address)
export MASTER_ADDR
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
```

`SLURM_JOB_ID` is unique per array task, so concurrent jobs (including different tasks of the same
array) cannot collide on a rendezvous port. The assignment and the `export` are deliberately two
statements, not one `export VAR=$(cmd)` line — bash would otherwise report `export`'s own (always
zero) exit status instead of the command substitution's, silently defeating `set -euo pipefail` if
the hostname lookup ever failed.

---

## 6. Job arrays and the homogeneous-resource constraint

A Slurm job array shares one `#SBATCH` header — including `--nodes` — across every task in it, so
an array can only ever hold jobs that need the *same* resources. `JobSpec` (`oplm.slurm.render`)
supports this directly: give it `array_size` (task count) and `array_index_file` (one identifier
per line, one per array task) and `render_job` adds the `--array=0-N%<max_concurrent>` header plus
a small lookup that maps `$SLURM_ARRAY_TASK_ID` to that line:

```bash
# --- array ---
#SBATCH --array=0-2%4

# Map this array index to its run. One run id per line; index = line number - 1.
BASE_DIR="<phase-output-dir>"
INDEX_FILE="<phase-output-dir>/jobs/<preset>.jobs"
RUN_ID=$(awk "NR==$((SLURM_ARRAY_TASK_ID + 1))" "$INDEX_FILE")
if [ -z "$RUN_ID" ]; then
  echo "no run for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi
export RUN_DIR="$BASE_DIR/runs/$RUN_ID"
```

`max_concurrent` (§2) throttles how many array tasks run at once (the `%4` above), independent of
how many tasks exist in total.

Consequence: a set of jobs that differ in node count (e.g. different presets on different node
counts) needs one array *per resource shape*, not one array across all of them. `oplm slurm
generate` only ever emits a single, non-array job per invocation; array-job generation is a
capability of the `oplm.slurm.render` layer for tools that need to fan a phase of work out across
many homogeneous cells — the μP sweep's phase generator is the concrete, in-repo example (one
array per preset, since node count varies by preset there; see [LR_SWEEP.md](LR_SWEEP.md)).

A job that needs no GPU at all (e.g. a lightweight post-processing step run after an array
finishes) can omit `--gres` entirely — `JobSpec.gres` defaults to `True` but can be set `False`.

---

## 7. Job dependencies: `afterany`, not `afterok`

`render_submit_script` renders a `submit.sh` that submits every job in order and threads job IDs
into `--dependency`:

```bash
A_170M=$(sbatch --parsable jobs/170M.sbatch)
echo "submitted jobs/170M.sbatch: $A_170M"
ANALYZE=$(sbatch --parsable --dependency=afterany:$A_170M jobs/analyze.sbatch)
echo "submitted jobs/analyze.sbatch: $ANALYZE"
```

Dependencies use `afterany`, never `afterok`. A downstream job that inspects the outputs of one or
more upstream jobs should run even if an upstream job failed or diverged — its own logic decides
what to do with a bad or missing result, and that failure should be *visible* there rather than
hidden. Under `afterok`, the first upstream job that exits non-zero would leave its dependent stuck
in `DependencyNeverSatisfied` forever, with no job ever running to report why.

---

## 8. Requeue semantics: drain, budget, and the no-progress guard

Every generated job carries `#SBATCH --requeue` unconditionally. This tells Slurm it may
automatically resubmit the job (same job ID) if it is preempted or the node otherwise fails —
without an operator noticing or intervening. Two more header lines cooperate with the training
process itself:

```bash
#SBATCH --signal=USR1@600
#SBATCH --open-mode=append
```

`--signal=USR1@600` asks Slurm to deliver `SIGUSR1` 600 seconds before the job's time limit, so
the trainer can drain (checkpoint and exit) instead of being `SIGTERM`'d mid-step by the scheduler.
`--open-mode=append` means a requeued job reuses the same `%j`/`%A_%a` log path and appends to it
rather than truncating it, so the pre-requeue tail — including whatever diagnosis led to the
requeue — survives across restarts.

### The trainer side: drain, the `.drained` marker, and exit 85

`oplm.train`'s `Trainer` installs a drain trigger (`oplm.training.signals.DrainSignal`) that goes
true on `SIGUSR1`, `SIGTERM`, **or** a wall-clock margin computed from Slurm's
`SLURM_JOB_END_TIME` (600 s before the time limit — the same margin the `--signal` header uses, so
either path can catch the job). Once true, the trainer finishes the in-flight optimizer step,
saves a checkpoint, logs a warning, writes a `.drained` marker file into `train.output_dir`
(`oplm.training.signals.DRAIN_MARKER_NAME`), and exits with `DRAIN_EXIT_CODE` (`85`) — a code
reserved exclusively for "drained cleanly, resume expected," distinct from `0` (reached
`max_steps`) and any other nonzero exit (a crash). See
[TRAIN.md §16](TRAIN.md#16-fault-tolerance) for the full knob table and what a resume restores.

The marker exists because the exit code does not survive the production launch stack: the rank
processes exit 85, but `accelerate launch --multi_gpu` reports any worker failure as its **own**
exit `1` (torchelastic's `ChildFailedError` is re-raised inside the launcher), so the sbatch
script's `$?` never sees the 85 on a real multi-GPU job. The wrapper below therefore keys its
drain branch on *marker-present OR exit 85* and **consumes** (deletes) the marker every time it
acts on one; the trainer additionally clears a stale marker at startup (possible only when the
batch node died before the wrapper ran and Slurm itself requeued the job), so a leftover marker
can never misclassify a later genuine crash as a drain.

Two more delivery details make the drain actually reach the trainer intact:

- `--signal=USR1@600` is delivered to **every** process in the step's cgroup — including the
  `bash -c` task wrapper and the `accelerate launch` parent, neither of which handles `USR1`
  (both would die instantly on it, tearing the step down mid-drain-checkpoint). The rendered
  command therefore starts with `trap "" USR1`: `SIG_IGN` is inherited across `exec`, shielding
  `accelerate launch` too, while the trainer ranks still drain because `signal.signal()`
  overrides an inherited ignore.
- `SIGTERM` is deliberately **not** ignored there — torchelastic installs its own `TERM` handler
  regardless, and `scancel`'s TERM→KILL teardown semantics should stay unchanged.

### The wrapper side: budget-capped, progress-aware requeue

After the training `srun` exits, every generated job script runs a requeue wrapper
(`oplm.slurm.render._requeue_wrapper`) that decides what happens next from the exit status and
the `.drained` marker:

| Outcome | Wrapper behavior |
| --- | --- |
| exit `0` | Clean finish (`max_steps` reached, or `save_final`'s final save completed). Logs `training complete` and exits `0` — never requeued. |
| drain (`.drained` marker present, or exit `85`) | Requeues **unconditionally**, as long as the requeue budget below is not exhausted. The no-progress guard is bypassed — a drain is not a crash, so there is nothing to diagnose. The marker is deleted as it is acted on. |
| any other nonzero | Requeues only if the budget is not exhausted **and** checkpoint progress was made since the previous restart (the no-progress guard below). |

**Requeue budget** (`slurm.max_requeues`, default `20`): the wrapper reads
`SLURM_RESTART_COUNT` (Slurm's own per-job restart counter) and refuses to requeue once it reaches
`max_requeues`, exiting with the training process's own status instead — an unbounded requeue loop
against a persistently broken node or config is not silently infinite.

**No-progress guard** (crash-loop detection, non-drain exits only): the wrapper scans the job's
output directory for `checkpoint-<step>` directories, keeps only the ones whose suffix is purely
numeric (so in-progress `.tmp` and `.old` directories are excluded — the same committed-only rule
the trainer's own discovery uses) and takes the highest step, then compares it to the step
recorded at the *previous non-drain failure* (`.last_requeue_step`, written into the same
directory). If the step has not advanced — i.e. this is already the second consecutive non-drain
failure with zero checkpoint progress between them — the wrapper treats it as a crash loop and
exits without requeueing, rather than repeatedly resubmitting a job that immediately dies again.
A first restart (`SLURM_RESTART_COUNT == 0`) always requeues on budget alone, since there is no
previous step to compare against yet. The drain path (marker or exit 85) neither reads nor writes
`.last_requeue_step`: a preemption is not a failure, so a drain must not arm the guard against
the *first* crash that happens to follow it. Jobs with no `progress_dir` (non-training jobs, e.g.
a post-processing step) skip the guard entirely and requeue on budget alone — they get no drain
marker check either.

### Auto-resume: how a requeued job actually continues

`--requeue` (and the wrapper's `scontrol requeue`) only get the job re-scheduled — whether the
*training process* picks up where it left off is a property of `train.auto_resume`.
`oplm slurm generate` injects `train.auto_resume=true` into the rendered command automatically, so
every job it produces resumes from the newest committed checkpoint under `train.output_dir` on
every restart, without an explicit `train.resume_from`. (`configs/scaling.yaml` also sets
`train.auto_resume: true` explicitly in the YAML itself — this config only ever runs under Slurm,
so there is no ambiguity about whether it should auto-resume.) An explicit `train.resume_from`
always wins if both are set. See [TRAIN.md §16](TRAIN.md#16-fault-tolerance) for the resolution
mechanics (main-rank scan + broadcast) and exactly what state a resume restores.

**The end-to-end failure-recovery walkthrough:** a node fails or is preempted → the training
process exits nonzero (drain: `.drained` marker + exit `85`, flattened to `1` by `accelerate
launch`; crash: whatever it raised) → Slurm requeues the job (subject to
`--requeue` and the wrapper's budget/no-progress checks above) → the new attempt's `Trainer` scans
`train.output_dir` for the newest committed checkpoint and resumes from it (model, optimizers,
schedulers, RNG, step counters) → the same W&B run continues (the persisted run id) rather than
starting a new one. (Tooling that runs many short cells back-to-back, like the μP sweep's runner,
implements its own analogous resume logic on top of this layer — see `oplm.sweep.run` — but that
is specific to that runner, not this general layer.)

---

## 9. `oplm slurm generate` / `submit` / `status`

### `generate`

```text
oplm slurm generate --config FILE --out DIRECTORY [--preset TEXT] [--nodes INT]
                     [--name TEXT] [--time-limit TEXT]
```

Writes one `<name>.sbatch` into `--out`, plus a `jobs.json` manifest `submit`/`status` read.
`--config` is resolved to an absolute path before rendering (the job runs from a job-scoped
working directory on a compute node, not wherever `generate` was invoked from, so a relative path
would silently resolve to the wrong file there). `--nodes`/`--time-limit` override whatever the
`slurm:` block's tables would otherwise resolve, and let you omit `--preset` entirely if you
supply both.

```bash
oplm slurm generate --config configs/scaling.yaml --preset 400M --out jobs/400M
# wrote jobs/400M/oplm-400M.sbatch (8 nodes, 168:00:00)
```

The rendered script matches `configs/scaling.yaml`'s `nodes: {..., 400M: 8, ...}` exactly
(`#SBATCH --nodes=8`), embeds the absolute `--config` path, and names the job `oplm-400M` by
default (override with `--name`). Every other preset in that file resolves the same way:
`170M` → 4 nodes, `800M` → 16 nodes, `1B` → 16 nodes.

### `submit`

```text
oplm slurm submit DIRECTORY
```

Submits every script `generate` wrote into `DIRECTORY` (`sbatch --parsable`), records each job ID
back into `jobs.json`, and prints them:

```bash
oplm slurm submit jobs/400M
# JOB_0: 812345
```

Requires a prior `generate` call in that directory (a clean error otherwise, not a traceback).

### `status`

```text
oplm slurm status DIRECTORY
```

Reports which of the job IDs recorded in `jobs.json` the scheduler still knows about
(`squeue --jobs`, 30 s timeout):

```bash
oplm slurm status jobs/400M
# JOB_0 (812345): active
```

Before any `submit` call it reports `no jobs submitted yet`. If `squeue` is unreachable (absent
from `PATH`, or a wedged controller past the query timeout), it says so explicitly and falls back
to just listing the recorded job IDs — it never reports "finished" for a job it simply could not
ask about, since that would look identical to the job having actually completed.

---

## See also

- [TRAIN.md](TRAIN.md) — training how-to: config, launch, checkpointing, resume.
- [CONFIG.md](CONFIG.md) — full `model.*` / `train.*` / `data.*` reference.
- [LR_SWEEP.md](LR_SWEEP.md) — the μP learning-rate sweep, built on this layer.

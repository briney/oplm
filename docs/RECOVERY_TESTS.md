# Real-cluster recovery drills

These six Python scripts exercise the real Trainer and the production Slurm renderer.
Run them from a login/controller host that survives the injected failure, with access to
`sbatch`, `squeue`, `scontrol`, `sacct`, and the shared training filesystem. They prepare
files by default; **only `--submit` submits a job and enables failure injection**.

| Python module under `oplm.slurm.recovery` | Injection and expected outcome |
| --- | --- |
| `graceful_drain` | Send SIGUSR1 to training rank zero after the first checkpoint. A fresh drain checkpoint commits; Slurm requeues; training resumes and finishes. |
| `rank_kill` | SIGKILL one exact training rank between checkpoints. The launcher fails; the shell wrapper requeues; training rolls back to the last committed checkpoint. |
| `checkpoint_write` | On the second asynchronous save, SIGKILL a rank after its DCP shard write and before acknowledging completion. The staging directory must not become a resume target. |
| `worker_node_loss` | Pause training after the first checkpoint and invoke an operator-supplied executable to take a non-batch node out of service. Require a replacement node at the original world size. |
| `batch_node_loss` | The same drill, targeting Slurm's actual `BatchHost`. Recovery must work without the original batch shell executing its requeue wrapper. |
| `crash_loop` | Kill a rank between checkpoints, then kill it immediately after checkpoint restoration on the next attempt. The shell's no-progress guard must end the job rather than requeue again. |

The injected failure is deliberately repeatable. No production training code is modified:
instrumentation and injections live in the drill-only `worker.py` subclass. The checkpoint
test interrupts a real asynchronous checkpoint transaction, not a fabricated `.tmp` directory;
it does not simulate partially written sectors or storage corruption. The graceful test sends
the signal to a training process; testing Slurm's wall-time signal delivery is a separate drill.

## Prepare and review

Install `oplm[train]==0.3.1` on the controller and in the compute containers; that PyPI
release includes the compiled distributed checkpoint fix. A source install is also supported.
Pin the package version or checkout for the entire drill: installation runs again on every restart.
Keep credentials in the existing environment setup, not in these scripts or the manifest.

Start from a **working training YAML with a `slurm:` block** (see [SLURM.md](SLURM.md)).
Use a small model first, but keep the production optimizer, precision, parallelism strategy,
container, and storage. `--out` must be a new absolute directory on storage available at the
same path to the controller and every compute container. Paths with shell metacharacters or
whitespace are rejected because the existing renderer cannot quote them in every context.

Example; replace the shared paths with those used on your cluster:

```bash
python -m oplm.slurm.recovery.rank_kill \
  --config configs/scaling.yaml --preset 170M \
  --out /mnt/home/briney/recovery/rank-kill-001 \
  --nodes 2 --save-every 16 --fail-step 24 --max-steps 64 \
  --install 'pip install "oplm[train]==0.3.1"'
```

Preparation writes `train.yaml`, `job.sbatch`, and `drill.json`, and creates `events/`,
`logs/`, and `training/`. Inspect the YAML and job script before submitting. Other modules
accept the same arguments; use a fresh `--out` for each drill.

`--install` is a preparation-time setting. Changing the source YAML or supplying
`--install` with `--submit` does not rewrite an already prepared `job.sbatch`. To change
the install command after a submitted drill fails, prepare a fresh output directory
(for example, replace `-001` with `-002`) with the desired `--install`, then submit it.

Defaults are two nodes, 64 training steps, checkpoints every 16 steps, injection at step 24,
and rank 1 as the SIGKILL target. `checkpoint_write` instead injects on the second periodic
save. Choose a checkpoint interval divisible by `data.num_workers` to avoid checkpoint
deferral obscuring the intended failure window. `--kill-rank 0` also supports a single-GPU
diagnostic run; it does not exercise multi-node recovery.

`--set key=value` can be repeated to change model/data/training settings before preparing.
The harness then fixes its test controls: a fresh output directory, automatic/data-cursor
resume, no explicit resume target, the requested short step horizon, checkpoint cadence,
per-step logging, and a dedicated W&B project/name. Warmup is capped at 10% of the horizon,
the stable phase at 50%; neither changes between attempts. Evaluation and remote checkpoint
mirroring are disabled. Source datasets, model, optimizer, batch size, accumulation, precision,
compile setting, and data workers are otherwise preserved. This tests **shared-filesystem
recovery**, not restoration from object storage or loss of the filesystem itself.

The generated script retains the production `srun`/container/Accelerate launch and requeue
logic, with a retry budget of three. `--time-limit` defaults to one hour per attempt; allow
enough time for initialization plus the renderer's ten-minute drain margin. A very short
wall-time limit would trigger an unintended drain before the planned injection.

## Supplied 400M scaling job

[`configs/recovery_400M.yaml`](../configs/recovery_400M.yaml) adapts the supplied scaling
job: `hpc-mid`, eight GPUs and 128 CPUs per node, the same container and mounts, bf16,
compilation, sequence length 512, no gradient checkpointing, and per-GPU batch 128.
Training data is **25% UniRef70 and 75% DeepClust70**, using the supplied `/mnt/data` paths.
Other model/optimizer defaults come from the installed package, just as omitted settings in the
original command come from the installed package. Use the same code version for both if
you need an exact production comparison.

Install the published release on the controller:

```bash
pip install 'oplm[train]==0.3.1'
```

The config's container install command pins that same PyPI release. No shared source
checkout is needed inside the containers. Run preparation from a repository checkout
containing `configs/recovery_400M.yaml`, or copy that YAML and pass its path to `--config`;
the example YAML and this guide are not installed by the wheel. For testing unpublished
changes, use `--install` to select a shared checkout and keep it unchanged between attempts.

Prepare all six drills on two nodes (this loop does **not** submit them):

```bash
for mode in graceful_drain rank_kill checkpoint_write worker_node_loss batch_node_loss crash_loop; do
  python -m "oplm.slurm.recovery.$mode" \
    --config configs/recovery_400M.yaml --preset 400M --nodes 2 \
    --out "/mnt/home/briney/recovery/400M-${mode}-001" \
    --max-steps 64 --save-every 16 --fail-step 24 --time-limit 01:00:00
done
```

The config uses four gradient accumulation steps, giving
**2 nodes × 8 GPUs × 128 sequences × 4 = 8,192 sequences per optimizer step**.
Matching production's global batch is optional for recovery testing; keeping world size
and effective batch unchanged across restart is what these checks require. Accumulation
also exercises checkpoint restoration with multiple microbatches per optimizer step.
For faster drills, use `--set train.gradient_accumulation_steps=1` (global batch 2,048).
Two nodes exercise multi-node recovery, but do not measure production-scale communication
or recovery latency. No eight-node run is required to use these drills.

The recovery CLI's node count comes from `--nodes` (default two), independently of the
YAML's Slurm node table. Use fresh output directories when changing a drill's settings;
already prepared directories retain their resolved configuration.

These are 64-step tests with checkpoints every 16 steps, a six-step warmup and 32-step
plateau followed by decay. The source job's million-step schedule (5,000 warmup + 995,000
stable), 500,000-step checkpoint interval, 30-day allocation, production W&B project, and
production output directory are intentionally replaced for the drills. Evaluation is disabled.
The generated jobs use the production renderer's requeue wrapper and signal protection;
the supplied shell script itself does not contain that wrapper or enable `train.auto_resume`.

Run one prepared drill at a time, for example:

```bash
python -m oplm.slurm.recovery.rank_kill \
  --out /mnt/home/briney/recovery/400M-rank_kill-001 --submit
```

Node-loss submissions additionally need the operator adapter described below. Add
`--reservation NAME` during preparation if a protected replacement pool is available;
a two-node drill needs at least three compatible healthy nodes in that pool to replace
one failed node. Increase `--time-limit` at preparation and `--timeout` at submission if
compilation, checkpoint I/O, or queue wait exceeds the short defaults. Inspect `result.json`
before moving on to the next drill.

## Submit and inspect the result

```bash
python -m oplm.slurm.recovery.rank_kill \
  --out /mnt/home/briney/recovery/rank-kill-001 --submit

# Repeat only the evidence/W&B checks later; this never submits or injects.
python -m oplm.slurm.recovery.rank_kill \
  --out /mnt/home/briney/recovery/rank-kill-001 --check
```

Submission records the job ID immediately. The controller polls every ten seconds and waits
through requeue to a terminal accounting state. Its `--timeout` defaults to 7,200 seconds,
including initial queue wait. On interruption, timeout, or a monitoring error, it attempts
to cancel only the job it submitted and preserves evidence. Keep this controller alive
(e.g. in tmux); a hard kill of the controller itself cannot perform cleanup. Never submit
the same prepared directory twice or run two controllers for the same directory.

The result is written to `result.json`. Exit codes: **0 passed, 1 failed, 2 incomplete**.
A missing online W&B check, including deliberate `--no-wandb`, yields incomplete rather
than a full pass. Preparation itself exits zero without claiming a recovery pass.

The checks require:

- Evidence of the intended injection and exactly one restarted attempt, with every rank
  recording startup and agreeing on the newest committed checkpoint.
- Restored optimizer step, sample/token counters, data cursor batch count, and learning
  rates matching that rank's save-time snapshot; unchanged world size and effective batch.
- Completion at the original planned step, or a failed job with the specific no-progress
  guard diagnosis for `crash_loop`.
- For node loss, a different replacement node, exclusion of the failed node, and the same
  total number of training nodes. The node adapter must actually induce node failure;
  a process kill plus manual requeue is not equivalent evidence.
- The same W&B entity/project/run ID/name, and unsampled server history containing the
  resumed attempt's logged steps. `recovery/attempt` and `recovery/step` distinguish replayed
  metrics from stale metrics left by the abandoned attempt.

W&B's internal history counter keeps increasing across attempts. Metrics use
`train/global_step` as their chart axis, so replayed training steps are retained instead
of being dropped as writes to past history. The abandoned attempt's metrics remain;
use `recovery/attempt` to distinguish them. Existing custom charts using W&B's default
`Step` axis should be switched to `train/global_step`.

After updating recovery or logging code, prepare fresh drill directories and ensure the
controller and container install the fixed version. For unpublished fixes, install the
updated checkout on the controller and use `--install` with a shared checkout path during
preparation. An already submitted script is not rewritten by changing the local file.
Rerun `rank_kill` and `checkpoint_write` first with `train.compile=true`, then the other
drills; a passing W&B chart alone does not replace a passing `--check` result.

The report includes before/after node lists and time from injection to the first restored
rank's startup observation and first resumed training-step log (assuming synchronized node
clocks). It is not a throughput benchmark. These checks do not compare
all model/optimizer tensors or promise bitwise RNG/masking equivalence to uninterrupted
training; checkpoint state round-trips have separate tests in `tests/training/`.

W&B must be reachable from both compute nodes (logging) and the controller (read API), with
the same account/entity settings. Online mode is forced for enabled tracking. The current
trainer's ordinary `resume="allow"` may drop metrics when a checkpoint rollback goes behind
the server's latest step: **that is a legitimate drill failure**, not something the harness
hides. A late upload can be checked again with `--check`. No production W&B runs are rewound
or deleted by these scripts.

## Actual node failures and spare capacity

Node-loss tests additionally require an executable supplied by your cluster operator:

```bash
python -m oplm.slurm.recovery.worker_node_loss \
  --out /mnt/home/briney/recovery/worker-loss-001 --submit \
  --node-failure-command /mnt/home/briney/bin/inject-test-node-failure
```

The adapter is invoked once as **`EXECUTABLE JOB_ID NODE`**, without a shell. It must validate
the test allocation and take exactly that node out of service through the cluster's supported
administrative mechanism. It must return zero after initiating the failure, leave the node
unavailable until replacement has been verified, and arrange operator restoration afterward.
It must not manually requeue the job: scheduler-driven recovery is what this test measures.
Merely marking a node DRAIN generally does not interrupt the running job. Power-off, SUNK pod
loss, and scheduler-declared DOWN exercise different detection paths; record the adapter's
actual mechanism with the results. The harness records its exact argv and output, but cannot
independently certify a physical hardware outage.

The controller selects the target from `scontrol show job`, checks allocation size and job ID,
and distinguishes `BatchHost` from other allocated nodes. No node command runs at preparation
time. Training pauses for at most `--node-wait-seconds` (default 300) waiting for the injection;
choose a larger value if the site's failure detection requires it. Missing or ineffective
injection cannot produce a passing node-replacement report.

Have the operator reserve three compatible healthy nodes for the initial two-node test.
`--reservation NAME` at preparation adds the reservation directive. The training job still
requests only two nodes; it must regain two after failure. At production scale the analogous
setup is 32+2 or 64+2 nodes in a protected pool. Simply requesting extra training nodes does
not make them spares. The harness does not create reservations or change cluster policy.

After the small drills pass, repeat representative rank loss, checkpoint-write loss, and
node replacement at production model/node size. Spare capacity does not remove failure
detection, launch, compilation, or checkpoint-load latency. Preserve the resolved configs,
append-mode Slurm logs, JSON events, W&B run, scheduler records, and adapter output as evidence.

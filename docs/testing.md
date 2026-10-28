# Testing

How the OPLM test suite is organized, how to run it, what the end-to-end (E2E) training tests
exercise, and the opt-in real-cluster recovery drills. The E2E test docstrings reference the
shared harness (*Shared harness*) and the scenario gallery (*End-to-end scenario gallery*).

## Commands

```bash
pip install -e ".[dev]"

.venv/bin/python -m pytest -m "not slow" -q   # fast inner loop: skips E2E and heavy integration
.venv/bin/python -m pytest                    # full suite, including the slow E2E Trainer scenarios
.venv/bin/python -m pytest tests/model/test_canon_semantics.py -q   # a focused subsystem
.venv/bin/python -m pytest --cov=oplm         # with coverage

ruff check src/         # lint
ruff format src/        # format
ty check src/           # type check (Astral's `ty`, not mypy)
```

Run tests through `python -m pytest`, not bare `pytest`: the standalone executable can resolve
outside the active environment (where `oplm` is not importable). The same applies to
`torchrun`, which becomes `python -m torch.distributed.run`. CUDA-only scenarios (mixed
precision, Muon, compiled training) skip automatically on CPU-only machines.

Tests prefer real data over synthetic data, mirror the source tree, and mark heavy tests
`@pytest.mark.slow`; large fixtures are session-scoped.

## Test layout

Tests mirror the source layout under `tests/`:

| Area | Covers |
| --- | --- |
| `tests/model/` | Architecture and components: attention (SDPA vs manual-softmax parity), RoPE, norms, FFN, Canon convs (`test_canon_semantics.py`), embeddings, tokenizer, config, outputs, save/load, auto-classes, push-to-hub. |
| `tests/training/` | The real `Trainer`: logging, checkpoint rotation/resume, gradient accumulation, epochs, mixed precision, `torch.compile`, optimizers/schedulers, grad clipping, eval scheduling, W&B. |
| `tests/eval/` | Eval harness: scheduling cadences, sequence/structure tasks, the registry, token accounting, the categorical Jacobian, trainer integration. |
| `tests/data/` | Data pipeline: dataloaders, collation, MLM masking, determinism, weighted-masking invariants, import hygiene. |
| `tests/fold/` | Structure head: kernels, the ESMFold2 port, prediction, and the data pipeline (`tests/fold/data/`: CCD table, `MolecularComplex`, mmCIF parsing on the real entries under `tests/fixtures/fold/`, crops, featurization with targets, collation, preprocessing and `StructureDataset` resume), and training: losses against the vendored Boltz-2 oracles (`tests/fold/oracles/`), the training branch, `FoldTask` through the real Trainer (the gradient test is fast; only the overfit and resume-with-EMA tests are `slow`), and a two-rank gloo DDP smoke (`slow`). The real ESM-C 300M states test (downloads the public weights) is `slow` and skips without `oplm[esmc]`. GPU parity tests are `slow` and skip without CUDA. |
| `tests/` (root) | Cross-module: CLI, training entrypoint, train→serve lifecycle. |
| `tests/fixtures/` | Real data: `training/test_sequences.parquet` (real protein sequences), `eval/structures/` (PDB fixtures), and `fold/` (real wwPDB mmCIF entries and a CCD subset for the structure head's data pipeline). The Boltz-2 loss oracles are test code, not data: they live in `tests/fold/oracles/` ([README](../tests/fold/oracles/README.md)). |

## Markers and selection

The only custom marker is `slow`, registered in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

All E2E training scenarios (and a few heavy model/eval tests) are tagged
`@pytest.mark.slow`. The fast inner-loop suite is everything else; the slow suite
is the full-`Trainer` and real-structure-eval coverage.

`OPLM_FOLD_FIXTURES=<dir>` enables `tests/fold/test_parity.py` (skipped with a reason otherwise).

## Shared harness

The slow training tests share one harness (`tests/training/conftest.py`):

- **`tiny_train_cfg(...)`** — builds a tiny end-to-end config (2 layers, 32
  hidden, 4 heads, 33-token vocab) over the real fixture data, with knobs for
  batch size, accumulation, optimizer, scheduler, precision, compile, and W&B.
  Defaults to a 6-step run with no W&B and no clipping.
- **`FullRecordingCallback`** — records every trainer event (train start/end,
  per-step logs, eval metrics, checkpoints) and exposes convenience views
  (`train_log_steps`, `eval_steps`, `checkpoint_steps`).
- **Isolation fixtures** — `_reset_accelerator_state` (autouse, resets the
  `AcceleratorState` singleton between tests), `reset_dynamo` (clears the
  `torch.compile` cache), and `restore_optimize_ddp` (saves/restores the
  process-global `torch._dynamo.config.optimize_ddp`).

Real-data fixtures are session-scoped (`tests/conftest.py`): `full_training_parquet`
(the full real parquet), `training_parquet` (first 256 rows), `tiny_training_parquet`
(16 rows, so `max_epochs=2` crosses an epoch boundary cheaply), `second_eval_parquet`
(a disjoint slice for multi-dataset eval), and `structure_fixtures_dir` (the PDB
fixtures). Tests skip when a fixture file is absent rather than failing.

## End-to-end scenario gallery (G1–G13)

Each scenario drives the real `Trainer` (or the full model graph) and asserts a
specific contract. The `G#` labels appear in the test-module docstrings.

| ID | File | Scenario & key assertions |
| --- | --- | --- |
| **G1** | `tests/training/test_e2e_logging.py` | Logging cadence. Train logs fire on multiples of `log_every`; every payload carries the full metric contract (`train/loss`, `train/lr`, `train/epoch`, `train/samples`, `train/tokens`, `train/flops`), all finite; FLOP accounting matches `estimate_flops_per_token`. |
| **G2** | `tests/training/test_e2e_eval.py` | Eval cadence. Eval fires exactly when due (step and token cadences), reconstructed from the half-open-interval schedule. Two datasets on different cadences fire independently and merge on shared steps, with isolated `eval/<name>/` namespaces. |
| **G3** | `tests/training/test_e2e_checkpoint.py` | Checkpoint rotation & resume. Saves fire on cadence; `save_total_limit` keeps only the newest dirs; each checkpoint has the full artifact set (`trainer_state.json`, `config.yaml`, `hf/`, tokenizer). Resume restores counters and continues the LR schedule with no loss discontinuity. `save_final` writes an off-cadence final checkpoint. |
| **G4** | `tests/training/test_e2e_accumulation.py` | Gradient accumulation. One optimizer step per `N` micro-batches; logged loss is the mean of the micro-batch losses; sample/token counters accumulate correctly and reduce once per step. |
| **G5** | `tests/training/test_e2e_precision.py` | Mixed precision (CUDA-only, bf16 & fp16). Losses stay finite under autocast with grad clipping; eval loss decreases over the run; the final HF export reloads and produces finite logits. |
| **G6** | `tests/training/test_e2e_optim.py` | Muon two-optimizer path (CUDA-only). Two optimizers and two schedulers exist and both accumulate state; loss is finite. |
| **G7** | `tests/training/test_e2e_optim.py` | LR schedules. For each of `warmup_linear`, `warmup_cosine`, `wsd_linear`, `wsd_cosine`, the logged `train/lr` matches the schedule function exactly; warmup ramps, tails decay to `min_lr`, WSD plateaus. |
| **G8** | `tests/training/test_e2e_epochs.py` | Epoch-bounded runs. `total_steps = ceil(rows / batch_size) · max_epochs`; the epoch counter advances at the dataset-exhaustion boundary and `train/epoch` is monotone across the crossing. |
| **G9** | `tests/training/test_e2e_gradckpt.py` | Gradient checkpointing. Both full and selective (SAC) checkpointing reproduce the plain-run loss trajectory within tolerance — checkpointing is mathematically transparent. |
| **G10** | `tests/model/test_e2e_attention_parity.py` | SDPA vs manual softmax parity. The fused SDPA compute path and the manual-softmax weights path yield the same MLM loss (`rtol=atol=1e-4`). |
| **G11** | `tests/test_train_entrypoint.py` | Training entrypoint. The distributed entrypoint bootstraps the environment and runs end-to-end (unit-level bootstrap coverage in `tests/training/test_train_bootstrap.py`). |
| **G12** | `tests/test_e2e_lifecycle.py` | Train → serve roundtrip. Train a few steps, save, reload via `OplmForMaskedLM.from_pretrained` and `AutoModelForMaskedLM` (`trust_remote_code`), and run inference — embeddings and logits are finite and correctly shaped. |
| **G13** | `tests/training/test_e2e_wandb.py` | W&B tracker path (optional). With `WANDB_MODE=offline`, `init_trackers` + `log` complete; the flat config dict is correctly `/`-namespaced. |

## Shared-layer looping

`test_e2e_looping.py` exercises ordinary-to-looped weights-only stage transitions,
AdamW/Muon changes, fresh warmup/counters/data/tracker identity, source-free child
resume, and rejection of output aliases or loop drift before state loading.
`test_loop_resume.py` checks legacy metadata defaults and semantic comparisons.

`test_e2e_looping_distributed.py` launches bounded two-rank CPU/gloo workers for
DDP and native HSDP. Full-stack and interleaved-subset schedules are compared to
unwrapped loss gradients and optimizer updates with learnable value residuals,
full/selective checkpointing, and accumulation. Separate launches exercise parent
initialization, resume after removing the parent, HF/DCP tensor equality, loop
mismatch on every rank, and guarded two-to-one-rank recovery. Per-rank JSON,
stdout/stderr logs, and initial full state are retained in pytest temporary dirs.

The compile suite compares AOT-eager loss and all gradients through the real
Trainer compile setup. CUDA Inductor/bf16 single- and two-rank cases are guarded
by device availability; CPU checks do not validate GPU kernels or throughput.
Recomputation tests include dropout-seeded parity in `tests/model/test_looping.py`.

## Cluster recovery drills

These six Python scripts exercise the real Trainer and the production Slurm renderer.
Run them from a login/controller host that survives the injected failure, with access to
`sbatch`, `squeue`, `scontrol`, `sacct`, and the shared training filesystem. They prepare
files by default; **only `--submit` submits a job and enables failure injection**. Commands in this section run
on the cluster controller, with its own `python`.

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

### Prepare and review

Install a published `oplm[train]` release on the controller and in the compute containers (the
release must include the compiled distributed checkpoint fix). A source install is also supported.
Pin the package version or checkout for the entire drill: installation runs again on every restart.
Keep credentials in the existing environment setup, not in these scripts or the manifest.

Start from a **working training YAML with a `slurm:` block** (see [slurm.md](slurm.md)).
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
  --install 'pip install "oplm[train]==<release>"'
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

### Supplied 400M scaling job

[`configs/recovery_400M.yaml`](../configs/recovery_400M.yaml) adapts the supplied scaling
job: `hpc-mid`, eight GPUs and 128 CPUs per node, the same container and mounts, bf16,
compilation, sequence length 512, no gradient checkpointing, and per-GPU batch 128.
Training data is **25% UniRef70 and 75% DeepClust70**, using the supplied `/mnt/data` paths.
Other model/optimizer defaults come from the installed package, just as omitted settings in the
original command come from the installed package. Use the same code version for both if
you need an exact production comparison.

Install the published release on the controller:

```bash
pip install "oplm[train]==<release>"
```

The config's container install command (`slurm.install`) pins a PyPI release, so every requeued attempt installs the same version. No shared source
checkout is needed inside the containers. Run preparation from a repository checkout
containing `configs/recovery_400M.yaml`, or copy that YAML and pass its path to `--config`;
the example YAML is not installed by the wheel. For testing unpublished
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

### Submit and inspect the result

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
- The same W&B entity/project/run ID/name, and unsampled server history in which every row's
  `Step` equals its training step (`recovery/step`), no step repeats, and every step the
  resumed attempt logged is present. Steps past the abandoned attempt's last logged step must
  be the resumed attempt's rows (`recovery/attempt` = 1).

The trainer logs every row at the training step and, when a resume lands behind the server's
history, rewinds the run to the checkpoint first ([training.md](training.md), *W&B continuity*). Inside
that replay window a row is therefore the resumed attempt's when W&B rewound the history,
and still the abandoned attempt's when the server refused the rewind (rewinding is a W&B
private-preview feature, enabled per entity). Both pass. The resumed trainer records what it
did, and `result.json` reports it as `wandb_rewind`: `applied`, `unavailable`, or `not needed`
when the server history was not ahead of the checkpoint (a drain). A reported `applied` with
any abandoned-attempt row left past the checkpoint fails the drill.

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
the same account/entity settings. Online mode is forced for enabled tracking. A missing
resumed step, a misaligned `Step`, or an abandoned-attempt row past the replay window is a
**legitimate drill failure**, not something the harness hides. A late upload can be checked
again with `--check`. The drill scripts never rewind or delete W&B runs themselves; the
trainer rewinds only the drill's own run, and only when the entity allows it.

### Actual node failures and spare capacity

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

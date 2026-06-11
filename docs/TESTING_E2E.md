# OPLM Testing & End-to-End Suite

How the OPLM test suite is organized, how to run it, and what the end-to-end
(E2E) training tests actually exercise. The E2E test docstrings reference the
shared harness in [§4](#4-shared-harness) and the scenario gallery in
[§5](#5-end-to-end-scenario-gallery-g1g13). For the broader testing strategy and
deferred coverage, see [OVERVIEW.md §34–35](OVERVIEW.md).

## 1. Test layout

Tests mirror the source layout under `tests/`:

| Area | Covers |
| --- | --- |
| `tests/model/` | Architecture and components: attention (SDPA vs manual-softmax parity), RoPE, norms, FFN, Canon convs (`test_canon_semantics.py`), embeddings, tokenizer, config, outputs, save/load, auto-classes, push-to-hub. |
| `tests/training/` | The real `Trainer`: logging, checkpoint rotation/resume, gradient accumulation, epochs, mixed precision, `torch.compile`, optimizers/schedulers, grad clipping, eval scheduling, W&B. |
| `tests/eval/` | Eval harness: scheduling cadences, sequence/structure tasks, the registry, token accounting, the categorical Jacobian, trainer integration. |
| `tests/data/` | Data pipeline: dataloaders, collation, MLM masking, determinism, weighted-masking invariants, import hygiene. |
| `tests/` (root) | Cross-module: CLI, training entrypoint, train→serve lifecycle. |
| `tests/fixtures/` | Real data: `training/test_sequences.parquet` (real protein sequences) and `eval/structures/` (PDB fixtures). |

## 2. Markers & selection

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

## 3. How to run

> **Use `python -m pytest`, not bare `pytest`.** In the current environment the
> standalone `pytest` executable resolves outside the active Conda environment
> (where `oplm` is not importable). Running through `python -m` guarantees the
> right interpreter. The same applies to `torchrun` → `python -m torch.distributed.run`.

```bash
# Fast inner loop — skips all E2E and heavy integration tests
python -m pytest -m "not slow"

# Full suite (includes the slow E2E Trainer scenarios)
python -m pytest

# A focused subsystem
python -m pytest tests/model/test_canon_semantics.py -q

# With coverage
python -m pytest --cov=oplm
```

CUDA-only scenarios (mixed precision, Muon, compiled training) skip automatically
on CPU-only machines.

## 4. Shared harness

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

## 5. End-to-end scenario gallery (G1–G13)

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

# OPLM End-to-End Test Plan

> Founding reference for OPLM's **end-to-end training tests** — multi-step runs
> of the live `Trainer` (and the real `python -m oplm.train` entrypoint) over a
> tiny model and real data, asserting the observable contracts of logging,
> evaluation cadence, checkpointing/resume, optimizers, schedules, mixed
> precision, and the train→serve lifecycle. This document specifies the **target**
> test suite in enough detail to implement directly. For the trainer design see
> [TRAINER.md](TRAINER.md); for the eval harness see [EVAL_HARNESS.md](EVAL_HARNESS.md);
> for config fields see [CONFIG.md](CONFIG.md).

The goal is not blanket coverage — unit tests already cover the modules. The goal
is to be **comprehensive about end-to-end behavior**: realistic, multi-step
training runs (with a very small model, but real CUDA, real Accelerate, real
data) that exercise logging intervals, eval scheduling, checkpoint rotation,
resume, accumulation, mixed precision, and optimizer/scheduler choices together.

All new tests in this plan are marked `@pytest.mark.slow` so per-commit CI
(`pytest -m "not slow"`) is unaffected. They are intended to be run on a GPU box
with `pytest -m slow` (see [§7 CI & gating](#7-ci--gating)).

---

## 1. Current state

Three "slow" tests drive a live training loop today. On a CUDA box they already
run on GPU (Accelerator resolves to `cuda`) and exercise the `flex_attention`
fast path — but they assert very little, run only in **fp32 / single optimizer /
no accumulation / single eval dataset**, and each tiny run is ~1–2.5 s on an
RTX A6000.

| Test | Drives | Asserts today |
|---|---|---|
| `tests/training/test_pilot_train.py` | `Trainer.train()`, 4–6 steps | a `checkpoint-*` dir exists; eval fired ≥1×; resume reaches new `max_steps` |
| `tests/eval/test_trainer_integration.py` | `Trainer.train()`, 4 steps | *some* `eval/hd/*` key appears (cadence `{steps:2}` / `{tokens:1}`) |
| `tests/data/test_e2e.py` | dataloader→model, 3 steps | finite loss; eval determinism; weighted-mask invariant |

Supporting slow tests exist for the model graph (`tests/model/test_pilot_train.py`),
checkpoint internals (`tests/training/test_checkpoint.py`), the structure eval
task in isolation (`tests/eval/test_structure_task.py`), and remote HF reload
(`tests/model/test_push_to_hub.py`).

## 2. Gap inventory

**Never exercised in any training run:**

- `gradient_accumulation_steps` — the sync-boundary logic in `trainer.py`
  (`accelerator.accumulate`, `sync_gradients`, mean-loss accumulation, once-per-
  opt-step token reduction) is core and untested e2e.
- `max_epochs` + epoch-boundary handling (`StopIteration → epoch++ → set_epoch`,
  `_compute_total_steps` epoch formula).
- `mixed_precision="bf16"` / `"fp16"` — every existing run uses `"no"`.
- `optimizer="muon"` end-to-end — only param partitioning is unit-tested; the
  two-optimizer / two-scheduler `prepare` path never runs.
- `max_grad_norm` clipping.
- `gradient_checkpointing=True` *through the `Trainer`* (only module-level today).
- The LR schedulers — `get_schedule_fn` / `build_scheduler` / `build_schedulers`
  have **no tests at all**, unit or e2e (all four: `warmup_linear`,
  `warmup_cosine`, `wsd_linear`, `wsd_cosine`).
- The `oplm.train.main()` entrypoint and the `oplm train` CLI command — always
  monkeypatched, so argv→config→bootstrap→train is never run.
- `_bootstrap_training_environment` (triton cache dir, DeepSpeed env scrub).
- The wandb tracker path (`wandb_enabled` is always `False`; `init_trackers` and
  `_config_to_flat_dict` never run).

**Weakly asserted:**

- Logging cadence (`log_every` → `on_log`) and the `train/{loss,lr,epoch,samples,
  tokens,flops}` payload values.
- Eval cadence *counts* and *positions*, `at_start` / `at_end` / `is_final`
  behavior, multi-eval-dataset merging, `_extract_eval_loss` / `_last_eval_loss`.
- Checkpoint rotation after a real run; resume *equivalence* (vs. merely
  "reaches the step").
- "The model actually learns" over a longer run.

**No full-lifecycle test:** train → `checkpoint-*/hf/` → `from_pretrained` →
`encode` / inference embeddings.

## 3. Design principles

1. **Drive the real surfaces.** Prefer `Trainer.train()`, `oplm.train.main(cfg)`,
   and a `python -m oplm.train` subprocess over reaching into internals. Assert
   *observable contracts*: callback event counts/positions, logged metric values,
   on-disk artifacts, LR trajectory.
2. **Slow, but bounded.** Every test `@pytest.mark.slow`. Keep the model tiny
   (2 layers, 32 hidden) and runs ≤ ~50 steps. Target total new slow wall-clock
   **< ~2 min on a single GPU**.
3. **Parametrize `(device, precision)`** and `skipif` CUDA when unavailable, so
   the suite is correct both on the GPU box and on CPU-only machines.
   `tests/conftest.py` already resets `AcceleratorState` between tests, so mixing
   precisions/devices in one process is safe.
4. **Real data.** Reuse `training_parquet` (256 real sequences) and the
   `1CRN.pdb` structure fixture; add the two small fixtures in [§6](#6-fixtures).
5. **Assert contracts, tolerate noise.** Where exact numeric equality is fragile
   (resume bit-exactness, cross-precision loss), assert the *contract* (no
   discontinuity, finite, decreasing) with a documented fallback.

## 4. Shared harness

New `tests/training/conftest.py`:

- **`FullRecordingCallback(TrainerCallback)`** — records all five events
  (`on_train_start`, `on_log`, `on_eval_end`, `on_checkpoint_saved`,
  `on_train_end`) with their `step` and a copy of each metrics dict, so a single
  run can be asserted against every cadence at once.
- **`tiny_train_cfg(...)`** — one factory for the knobs (steps/epochs, batch,
  accumulation, precision, optimizer, scheduler, save/log/eval cadence, eval
  datasets, grad-ckpt, clip, output_dir), returning an `OplmConfig` with
  `wandb_enabled=False`, `num_workers=0`, `pin_memory=False` by default.
- **`device_precision` param helper** — yields `[("cpu","no"), ("cuda","no"),
  ("cuda","bf16")]`, with CUDA entries `skipif`-skipped when
  `not torch.cuda.is_available()`. (Device is implicit via Accelerate; precision
  maps to `train.mixed_precision`. A `cpu` Accelerator is forced with
  `ACCELERATE_USE_CPU` / `Accelerator(cpu=True)` semantics as used in
  `tests/training/test_checkpoint.py`.)

## 5. Test groups

Mirror the source layout: trainer-feature tests under `tests/training/`. The two
genuinely cross-module flows (G11, G12) live at `tests/` top level with a
docstring noting the intentional deviation from the mirror convention.

### G1 — Logging cadence & metric contract — `tests/training/test_e2e_logging.py`
Run ~12 steps with `log_every=3`. Assert:
- `on_log` train-metric emissions occur exactly on multiples of 3;
- every train payload carries `train/{loss,lr,epoch,samples,tokens,flops}`, all
  finite;
- `train/tokens`, `train/samples` are monotonically non-decreasing and equal the
  hand-computed totals;
- `train/flops == flops_per_token * train/tokens` at each log.

### G2 — Eval cadence — `tests/training/test_e2e_eval.py`
Parametrize `{steps:N}` and `{tokens:N}`. Assert the *number* and *step
positions* of `on_eval_end` firings match `_crossed` semantics (`schedule.py`),
plus `at_start` / `at_end` / `is_final` behavior. Add a **two-dataset** config
(`hd` + a second eval set on a different cadence) and assert both
`eval/<name>/*` namespaces appear independently and merge correctly. Assert the
`_extract_eval_loss` → `_last_eval_loss` value is finite. *(Optional extension:
a structure eval task firing through the Trainer on its cadence, reusing
`1CRN.pdb`.)*

### G3 — Checkpointing, rotation & resume equivalence — `tests/training/test_e2e_checkpoint.py`
Real run with `save_every` + `save_total_limit`: assert exactly the expected
`checkpoint-*` set survives rotation, and each surviving dir contains
`trainer_state.json`, `config.yaml`, and `hf/` (config.json + model.safetensors +
tokenizer files). **Resume-equivalence:** run A uninterrupted to step S; run B to
S/2 then resume to S; assert final `global_step` and `tokens_seen` match and the
post-resume loss trajectory shows no discontinuity. *(Start with trajectory-
continuity + counter equality; tighten to bit-exact weight equality only if the
IterableDataset/RNG state cooperates — this is the riskiest assertion.)*

### G4 — Gradient accumulation — `tests/training/test_e2e_accumulation.py`
`gradient_accumulation_steps=4`: assert one optimizer step per 4 micro-batches
(`global_step` advances only on the sync boundary), the logged `train/loss` is
the **mean across micro-batches** (`step_loss_sum / gradient_accumulation_steps`,
not the last micro-batch), and token reduction happens once per opt-step.

### G5 — Mixed precision — `tests/training/test_e2e_precision.py` (CUDA-only)
bf16 (and fp16) runs complete with finite loss, write a loadable checkpoint, and
loss decreases over the run. Guards the Accelerate autocast / GradScaler path
that fp32 tests never touch.

### G6 — Optimizers & LR schedule — `tests/training/test_e2e_optim.py`
- **Muon e2e:** full run with `optimizer="muon"`, exercising the two-optimizer /
  two-scheduler `prepare` path on GPU; finite loss; both optimizers step.
- **LR trajectory:** for each of the four schedulers assert the logged `train/lr`
  series matches `get_schedule_fn` — warmup ramp, WSD stable plateau, decay
  toward `min_lr`.
- **Learns:** a ≥40-step run on real data with `at_start` eval asserts final eval
  loss < initial eval loss.

### G7 — Gradient clipping — `tests/training/test_e2e_optim.py`
A high-LR run with `max_grad_norm` set stays finite where an unclipped control
diverges — proves clipping is wired through `accelerator.clip_grad_norm_`.

### G8 — Epoch-bounded runs — `tests/training/test_e2e_epochs.py`
`max_epochs=2` over the **tiny** parquet fixture so an epoch boundary is actually
crossed cheaply: assert `_compute_total_steps == ceil(dataset/effective_batch) *
max_epochs`, the `StopIteration → epoch++ → set_epoch` path runs, and
`train/epoch` fractional accounting is monotone and lands near 2.0.

### G9 — Gradient checkpointing through Trainer — `tests/training/test_e2e_gradckpt.py`
`gradient_checkpointing=True` trains to a finite loss through the real loop; a
paired run (same seed, ckpt on vs off) yields matching loss within tolerance.

### G10 — Path parity (folded into G1/G5 parametrization)
The `(device, precision)` parametrization covers cpu/cuda × no/bf16. Add one
focused test that a single train step produces matching loss under flex
(`use_flex_attention=True`, CUDA) vs the manual fallback — extending the existing
module-level parity test (`tests/model/test_attention.py`) up to the full model +
MLM loss.

### G11 — Entrypoint e2e — `tests/test_train_entrypoint.py` (cross-cutting)
- `oplm.train.main(cfg)` runs a tiny real loop to completion.
- A `subprocess` `python -m oplm.train --preset small <overrides>` trains a few
  steps into a tmp `output_dir` and exits 0 — covering argv → `load_config` →
  `_bootstrap_training_environment` (triton cache created, DeepSpeed disabled) →
  `main` → `Trainer`.

Back this with a fast (non-slow) unit test `tests/training/test_train_bootstrap.py`
for `_bootstrap_training_environment` (DeepSpeed env scrub, triton dir creation
via injected `env`/`home_dir`/`tmp_dir`).

### G12 — Full lifecycle — `tests/test_e2e_lifecycle.py` (cross-module)
Train tiny model → load `checkpoint-*/hf/` via `OplmForMaskedLM.from_pretrained`
(and `AutoModelForMaskedLM`) → run `encode` / `model.logits(...)` → assert
embeddings have the right shape and are finite. Closes the train→serve loop.

### G13 — wandb tracker path (optional) — `tests/training/test_e2e_wandb.py`
Run with `wandb_enabled=True` under `WANDB_MODE=offline` (or a disabled service)
to exercise `init_trackers` + `_config_to_flat_dict` (config flattening is a real
serialization risk) without network.

## 6. Fixtures

Add to `tests/conftest.py` (session-scoped, derived from the existing full
parquet exactly like `training_parquet`):

- **`tiny_training_parquet`** — ~16 real rows, so `max_epochs=2` with
  `batch_size=4` crosses an epoch boundary in ~8 steps (G8).
- **`second_eval_parquet`** — a disjoint slice of the source parquet for the G2
  multi-dataset test.

Reuse: `training_parquet` (256 rows) for the main runs; `structure_fixtures_dir`
(`1CRN.pdb`) for the optional structure-eval-through-Trainer test.

## 7. CI & gating

**Decision: keep slow e2e tests local-only.** Per-commit CI continues to run
`pytest -m "not slow"`; the e2e suite is run on demand on a GPU box with:

```bash
pytest -m slow
```

A self-hosted GPU CI lane or nightly scheduled `-m slow` run is a possible future
addition but is intentionally out of scope here.

> Note (unrelated cleanup spotted while planning): `.github/workflows/ci.yaml`
> runs `mypy src/`, but [AGENTS.md](../AGENTS.md) mandates `ty check src/`. Worth
> reconciling separately.

## 8. Organization, markers & timing

- All new tests `@pytest.mark.slow` (module-level `pytestmark` where the whole
  file is slow).
- Trainer-feature files under `tests/training/`; cross-module flows (G11, G12) at
  `tests/` top level with a docstring noting the deviation.
- Budget: ~15–20 new slow tests × ~1–4 s each ≈ **under ~2 min on the A6000**; the
  CPU-parametrized subset adds little.

## 9. Implementation order (suggested)

1. Shared harness (`FullRecordingCallback`, `tiny_train_cfg`, `device_precision`)
   + fixtures (§4, §6).
2. Core matrix: G1, G2, G3, G4, G8 (logging, eval cadence, checkpoint/resume,
   accumulation, epochs).
3. G5, G6, G7, G9 (precision, optimizers/schedule, clipping, grad-ckpt).
4. G11 + bootstrap unit test, G12 (entrypoint + lifecycle).
5. G10 parity assertion; G13 wandb (optional).

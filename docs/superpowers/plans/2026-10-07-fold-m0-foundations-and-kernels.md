# Structure Head Milestone 0 — Foundations and Kernels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the pieces the folding head needs before any model code exists: a
`TrainTask` seam in the Trainer (default MLM behaviour unchanged), EMA weights
with checkpoint/resume support, the staged triangle-multiplication kernel with
explicit cuEquivariance dispatch, FlexAttention attention primitives with a
dense oracle, and the `oplm fold bench-kernels` measurement CLI.

**Architecture:** Two independent tracks. Track A (Tasks 1–2) touches the
Trainer only: the step/model/dataloader/FLOP decisions move behind a small
protocol, and an `AveragedModel` EMA is updated once per optimizer step and
saved as an `ema.pt` sidecar plus an `hf_ema/` export inside the atomic
checkpoint commit. Track B (Tasks 3–6) creates `src/oplm/fold/` with a
functional, three-stage triangle multiplication whose parameter names match
ESMFold2 exactly, a one-boundary fused-forward/reference-backward autograd
function, attention primitives, and the benchmark CLI. Task 7 is the on-GPU
acceptance gate.

**Tech Stack:** Python 3.11, PyTorch 2.11 (FlexAttention, DCP,
`torch.optim.swa_utils`), cuEquivariance ≥ 0.10 as the optional `fold` extra,
Accelerate, Transformers, typer, pytest, ruff, ty. No new core dependency.

**Spec:** [Structure prediction head design](../specs/2026-10-07-structure-prediction-head-design.md),
milestone 0 of §10. This plan implements §3 (package skeleton, extras,
attribution), §5.2 (trimul and dispatch), §5.3 (attention), §5.4 (precision
rules as they apply to the kernels), §6.4 (task seam), §6.5 (EMA), and the
`bench-kernels` tool from §5.2. Upstream source consulted:
`https://github.com/Biohub/esm`, `esm/models/esmfold2/layers.py`
(`TriangleMultiplicativeBlock`, `AttentionPairBias`, `SWA3DRoPEAttention`,
`PairUpdateBlock`); copies live in the planner's scratchpad only, re-fetch with
`curl -sL https://raw.githubusercontent.com/Biohub/esm/main/esm/models/esmfold2/layers.py`.

**Planning baseline:** `03441a3` on `main`. Start in an isolated worktree
(superpowers:using-git-worktrees). Read `AGENTS.md` and the spec first. The
development machine has no GPU and no cuEquivariance: every GPU test is
`@pytest.mark.slow` and skips cleanly; Task 7 runs them on the B200.

**Plan sequence (later plans, not this one):** M1 inference port (§4.5, §4.6,
§5.1, §5.7, §9 golden fixtures: `configuration_fold.py`, `modeling_fold.py`,
`lm_shim.py`, `pair.py`, `trunk.py`, `atoms.py`, `diffusion.py`,
`confidence.py`, `predict.py`), M2 pipeline proof (§4.2–4.4, §4.7, §6.1–6.3,
§7.1–7.3, §8: `data/`, `losses.py`, `training.py` with `FoldTask`, the `fold:`
run-config block, `eval/`), M3 v1, M4 Ab–Ag distillation. `FoldConfig` and the
`fold:` run-config block are deliberately deferred to M1/M2 so their fields are
fixed against the real upstream config and `FoldTask` rather than guessed now.

## Global Constraints

- `AGENTS.md` rules: Python 3.11+, `from __future__ import annotations` in every
  file, type hints on every signature, Google-style docstrings on public APIs,
  ruff line length 100, `pathlib.Path` only, no logic in `__init__.py`, tests
  mirror the source layout, `@pytest.mark.slow` for GPU/E2E work.
- Commands (from `AGENTS.md` and the repo's memory notes): run everything with
  the venv interpreter, `.venv/bin/python -m pytest ...`, `.venv/bin/ruff check src/`,
  `.venv/bin/ruff format <changed files only>` (formatting the whole tree churns
  unrelated tests), and `VIRTUAL_ENV=.venv .venv/bin/ty check src/`. All three
  must be clean before every commit.
- Dependencies: `torch>=2.10.0,<2.12`, `transformers>=4.45,<5.4` unchanged. New
  optional extra `fold = ["cuequivariance-torch>=0.10.0", "cuequivariance-ops-torch-cu13>=0.10.0"]`.
  `gemmi` and `DockQ` are added by the plans that use them (M2).
- Dependency direction (spec §3): `oplm.fold` may import `oplm.model`,
  `oplm.training`, `oplm.data`, `oplm.eval`; core packages import nothing from
  `oplm.fold` except the CLI registration in `src/oplm/cli.py`. `oplm.fold.cli`
  must not import torch at module level (the CLI import is eager).
- Spec §6.4: "The default MLM implementation preserves existing behavior"; there
  is no second training loop. Spec §6.4: "If FLOPs are unknown, omit FLOP/MFU
  estimates rather than reuse the LM formula."
- Spec §6.5: EMA "decay 0.999 using PyTorch's averaged-model utility. Update once
  after a successful optimizer step, never on accumulation microsteps or a skipped
  update." "Export ordinary and EMA HF directories together at each checkpoint."
  "Save EMA tensors and update count as checkpoint tensor state or a sidecar
  covered by the atomic checkpoint commit and remote mirror." "Resume must
  restore EMA." The config default is off (`ema_decay: null`) so MLM runs are
  byte-for-byte unchanged; fold stages set `0.999`.
- Spec §5.2: "Keep three explicit stages in the reference implementation: local
  projections, normalization and gates; triangular contraction; local
  normalization, projection and output gate. Chunk contraction output rows when
  necessary and accumulate in fp32." Dispatch order: cuEquivariance autograd →
  fused forward with reference backward → compiled reference; "CPU tests always
  have this path." "Implement it at one recomputation boundary."
- Parameter conventions match upstream exactly: `norm_start`, `norm_mix`
  (LayerNorm, `eps=1e-5`), `proj_bundle` (`D -> 4D`, no bias; rows `[0:2D]`
  signal, `[2D:4D]` gate logits), `proj_emit` (`D -> D`, no bias), `proj_gate`
  (`D -> D`, no bias). cuEquivariance mapping: `p_in_weight = proj_bundle.weight[:2D]`,
  `g_in_weight = proj_bundle.weight[2D:]`, `norm_in = norm_start`,
  `norm_out = norm_mix`, `p_out = proj_emit`, `g_out = proj_gate`. The kernel
  requires `D % 32 == 0`.
- Spec §5.3: "Retain a dense PyTorch attention formulation as the small-input
  oracle and unsupported-device path. Fully masked padded queries must produce
  finite, masked outputs." "Verify gradients into the pair-bias projection as well
  as Q, K, and V." "Preserve the upstream window indexing and reference-space
  semantics." FlexAttention has no CPU backward (verified on torch 2.11), so the
  CPU path is dense; flex-vs-dense gradient parity is a GPU test.
- Spec §5.2 benchmark: lengths 384, 768, 1024, 1536, 2048; widths 128 and 256;
  both directions; forward, backward, checkpointed execution; peak
  allocated/reserved memory; warm-up; synchronized timing; report hardware,
  versions, actual dispatch path, and numerical error.
- Spec §9: "Set tolerances from fp32-reference comparisons and publish them; do
  not assert bf16 bitwise identity."
- Spec §3 attribution: ported modules carry a header naming the Biohub source
  file and the modifications; `THIRD_PARTY_NOTICES.md` carries the Apache 2.0
  text. Upstream file headers say Apache-2.0 (`Copyright 2026 Biohub`); the
  repository's `LICENSE.md` says MIT. Preserve the per-file Apache notice.
- Spec §5.4: pair stream bf16, contractions accumulate in fp32, normalization
  uses fp32 internals. Fast attention paths never cast Q/K/V.

## Review Focus

1. **EMA under gradient accumulation.** With `gradient_accumulation_steps=2`,
   `n_averaged` must equal the optimizer-step count, not the micro-batch count
   (Task 2, `test_ema_counts_optimizer_steps_and_survives_resume`).
2. **Resume with `ema_decay` set from a checkpoint saved without EMA.** Must warn,
   start the tracker from the live weights, and write `ema.pt` at the next save;
   it must not crash (Task 2, `test_resume_into_ema_from_checkpoint_without_ema_warns`).
3. **Trimul with a chunk tail and a fully masked row.** `N` not divisible by
   `chunk_size` must equal the unchunked result; a row whose mask is all zero must
   contribute nothing to the contraction (Task 3, `test_chunked_equals_unchunked`,
   `test_masked_row_contributes_nothing`).
4. **Attention batch row with no valid key.** Dense and flex paths must return
   finite zeros for that row rather than NaN (Task 5, `test_pair_biased_zero_valid_keys_row_is_zero_and_finite`).
5. **Mixed fused-forward/reference-backward under activation checkpointing.** Every
   parameter gradient must equal the plain reference gradient (no double
   accumulation) and the reference forward must run exactly once per backward
   (Task 4, `test_mixed_path_under_checkpoint_recomputes_once`).

## File Map and Ownership

| File | Responsibility |
|---|---|
| New `src/oplm/training/task.py` | `StepResult`, `TrainTask` protocol, default `MLMTask` |
| `src/oplm/training/trainer.py` | Build model/dataloader/FLOPs via the task; `task.step` in the loop; per-task metrics; EMA update and wiring |
| `src/oplm/training/__init__.py` | Export `MLMTask`, `StepResult`, `TrainTask` |
| New `src/oplm/training/ema.py` | `build_ema`, `sync_ema_buffers`, sidecar/export names |
| `src/oplm/training/checkpoint.py` | `ema.pt` sidecar + `hf_ema/` export inside the commit; restore on load |
| `src/oplm/training/remote.py` | Mirror `ema.pt` and `hf_ema/` as shared artifacts |
| `src/oplm/config.py`, `src/oplm/configs/train/base.yaml` | `train.ema_decay` field, validation, documented default |
| New `src/oplm/fold/__init__.py` | Package docstring only |
| New `src/oplm/fold/trimul.py` | Staged reference, cuEquivariance dispatch, mixed autograd function, `TriangleMultiplication` |
| New `src/oplm/fold/attention.py` | `pair_biased_attention`, `sliding_window_attention` (dense + flex) |
| New `src/oplm/fold/cli.py` | `oplm fold bench-kernels` and `run_kernel_benchmark` |
| `src/oplm/cli.py` | Register the `fold` sub-app |
| `pyproject.toml` | `fold` extra |
| New `THIRD_PARTY_NOTICES.md` | Apache 2.0 attribution for ported ESMFold2 modules |
| New `docs/FOLD.md`; `docs/TRAIN.md`, `docs/CONFIG.md`, `docs/TESTING_E2E.md`, `AGENTS.md` | Contracts and field docs |
| `tests/training/conftest.py` | `ema_decay` knob on `tiny_train_cfg` |
| New `tests/training/test_task.py`, `tests/training/test_ema.py`, `tests/training/test_e2e_ema.py` | Track A tests |
| New `tests/fold/__init__.py`, `tests/fold/test_trimul.py`, `tests/fold/test_attention.py`, `tests/fold/test_cli.py` | Track B tests |

Tracks A (Tasks 1–2) and B (Tasks 3–6) are independent and may run in parallel
worktrees; Task 7 needs both. Within a track, execute in order.

---

### Task 1: `TrainTask` seam with an unchanged default MLM task

**Files:**
- Create: `src/oplm/training/task.py`
- Modify: `src/oplm/training/trainer.py` (`__init__` imports at 237–246, model build 522–551, dataloader 565–569, FLOPs 774–775, window state ~783–791; `train()` 861–899; `_log_step` 1306–1362)
- Modify: `src/oplm/training/__init__.py`
- Modify: `docs/TRAIN.md` (new §17 before `## See also`)
- Test: `tests/training/test_task.py`

**Interfaces:**
- Consumes: `OplmForMaskedLM`, `load_initial_model`, `build_train_dataloader`, `estimate_flops_per_token` (all existing).
- Produces: `StepResult(loss: Tensor, tokens: int, samples: int, metrics: dict[str, float])`;
  `TrainTask` protocol with `build_model(cfg, initialization_source) -> nn.Module`,
  `build_dataloader(cfg) -> DataLoader`, `step(model, batch) -> StepResult`,
  `flops_per_token(cfg) -> int | None`; `MLMTask`; `Trainer(cfg, callbacks=None, task=None)`;
  `Trainer.flops_per_token: int | None`. M2's `FoldTask` implements this protocol.

- [ ] **Step 1: Write the failing tests**

Create `tests/training/test_task.py`:

```python
"""TrainTask seam: MLMTask reproduces the pre-seam step; a custom task's metrics and
FLOP policy flow through the Trainer (docs/TRAIN.md §17)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest
import torch

from oplm.training.flops import estimate_flops_per_token
from oplm.training.task import MLMTask, StepResult, TrainTask
from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn


def test_step_result_rejects_reserved_metric_keys() -> None:
    with pytest.raises(ValueError, match="reserved"):
        StepResult(loss=torch.tensor(0.0), tokens=1, samples=1, metrics={"loss": 1.0})


def test_mlm_task_step_matches_direct_model_call(tmp_path: Path, training_parquet: Path) -> None:
    """MLMTask.step returns the model's own loss plus exact token/sample accounting."""
    cfg = tiny_train_cfg(tmp_path, training_parquet)
    task = MLMTask()
    assert isinstance(task, TrainTask)
    model = task.build_model(cfg, None)
    model.eval()  # no dropout: the two forward passes below must agree exactly
    batch = next(iter(task.build_dataloader(cfg)))

    result = task.step(model, batch)
    expected = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )["loss"]
    assert torch.equal(result.loss, expected)
    assert result.tokens == int(batch["attention_mask"].sum())
    assert result.samples == len(batch["input_ids"])
    assert result.metrics == {}
    assert task.flops_per_token(cfg) == estimate_flops_per_token(cfg.model)


class _AuxMetricTask(MLMTask):
    """MLMTask that also reports a per-micro-batch metric and declines to estimate FLOPs."""

    def step(self, model: nn.Module, batch: dict[str, Any]) -> StepResult:
        result = super().step(model, batch)
        result.metrics["aux"] = float(result.tokens)
        return result

    def flops_per_token(self, cfg: Any) -> int | None:
        return None


@pytest.mark.slow
def test_custom_task_metrics_and_flop_policy_flow_through_trainer(
    tmp_path: Path, training_parquet: Path
) -> None:
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(tmp_path, training_parquet, max_steps=4, log_every=2, batch_size=4)
    callback = FullRecordingCallback()
    Trainer(cfg, callbacks=[callback], task=_AuxMetricTask()).train()

    assert callback.train_log_steps == [2, 4]
    for _step, metrics in callback.train_logs:
        assert math.isfinite(metrics["train/aux"])
        assert "train/flops" not in metrics
        assert "train/achieved_tflops" not in metrics
        assert "train/mfu" not in metrics
    # train/aux is the window mean of a per-micro-batch token count: two optimizer
    # steps per log window at accumulation 1, so 2 * mean == the window's token delta.
    (_, first), (_, second) = callback.train_logs
    assert second["train/aux"] * 2 == pytest.approx(second["train/tokens"] - first["train/tokens"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/training/test_task.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.training.task'`.

- [ ] **Step 3: Create `src/oplm/training/task.py`**

```python
"""Training-task seam: what the Trainer needs from a model/data/step triple.

The Trainer owns acceleration, optimization, gradient accumulation, evaluation
cadence, fault tolerance, and checkpointing. A :class:`TrainTask` owns the three
things that differ between objectives: how the model is built, how the training
dataloader is built, and how one micro-batch becomes a loss. :class:`MLMTask` is
the default and reproduces the pre-seam masked-language-model behaviour exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from oplm.config import OplmConfig

__all__ = ["MLMTask", "StepResult", "TrainTask"]

# Metric keys the Trainer itself emits under ``train/``; a task may not reuse them.
_RESERVED_METRIC_KEYS = frozenset(
    {"loss", "loss_mean", "lr", "epoch", "samples", "tokens", "flops", "global_step"}
)


@dataclass
class StepResult:
    """What one micro-batch produced.

    Attributes:
        loss: Differentiable scalar the Trainer backpropagates. The Trainer averages
            the logged value across the accumulation window; return the plain loss.
        tokens: Valid (non-padding) tokens in this rank's micro-batch.
        samples: Examples in this rank's micro-batch.
        metrics: Detached scalars, logged as ``train/<key>`` (mean over the log window).
    """

    loss: torch.Tensor
    tokens: int
    samples: int
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        clash = _RESERVED_METRIC_KEYS & self.metrics.keys()
        if clash:
            raise ValueError(f"StepResult.metrics may not use reserved keys {sorted(clash)}")


@runtime_checkable
class TrainTask(Protocol):
    """The objective-specific third of a training run."""

    def build_model(self, cfg: OplmConfig, initialization_source: Path | None) -> nn.Module:
        """Construct the trainable model, loading weights from ``initialization_source`` if set."""
        ...

    def build_dataloader(self, cfg: OplmConfig) -> DataLoader[Any]:
        """Construct the rank/worker-striped training dataloader."""
        ...

    def step(self, model: nn.Module, batch: dict[str, Any]) -> StepResult:
        """Run one micro-batch forward and return its loss and accounting."""
        ...

    def flops_per_token(self, cfg: OplmConfig) -> int | None:
        """Training FLOPs per token, or ``None`` when no honest estimate exists."""
        ...


class MLMTask:
    """Default task: masked-language-model pretraining with :class:`OplmForMaskedLM`."""

    def build_model(self, cfg: OplmConfig, initialization_source: Path | None) -> nn.Module:
        from oplm.model import OplmForMaskedLM

        # transformers strips ``config.gradient_checkpointing`` during
        # ``PreTrainedModel.__init__``, so read it before constructing the model.
        gradient_checkpointing = getattr(cfg.model, "gradient_checkpointing", False)
        if initialization_source is None:
            model = OplmForMaskedLM(cfg.model)
        else:
            from oplm.training.initialization import load_initial_model

            model = load_initial_model(initialization_source, cfg.model)
            model.train()  # HF pretrained loading returns an evaluation-mode model.
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()  # propagates to every OplmBlock
        return model

    def build_dataloader(self, cfg: OplmConfig) -> DataLoader[Any]:
        from oplm.data import build_train_dataloader

        return build_train_dataloader(cfg)

    def step(self, model: nn.Module, batch: dict[str, Any]) -> StepResult:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return StepResult(
            loss=outputs["loss"],
            tokens=int(batch["attention_mask"].sum().item()),
            samples=len(batch["input_ids"]),
        )

    def flops_per_token(self, cfg: OplmConfig) -> int | None:
        from oplm.training.flops import estimate_flops_per_token

        return estimate_flops_per_token(cfg.model)
```

- [ ] **Step 4: Route the Trainer through the task**

In `src/oplm/training/trainer.py`:

(a) Add `from oplm.training.task import TrainTask` to the module's `if TYPE_CHECKING:` block.

(b) Change the constructor signature and the first lines of the body:

```python
    def __init__(
        self,
        cfg: OplmConfig,
        callbacks: Sequence[TrainerCallback] | None = None,
        task: TrainTask | None = None,
    ) -> None:
        from accelerate import Accelerator
        from accelerate.utils import DataLoaderConfiguration, InitProcessGroupKwargs, set_seed
        from rich.console import Console

        from oplm.config import validate_parallelism_compat
        from oplm.data import DeviceDataLoader
        from oplm.training.optim import build_optimizers, build_schedulers
        from oplm.training.preflight import run_preflight
        from oplm.training.task import MLMTask

        self.cfg = cfg
        self.callbacks = list(callbacks or [])
        # Objective-specific model/data/step (docs/TRAIN.md §17). The default
        # reproduces MLM training exactly; fold stages pass FoldTask (milestone 2).
        self.task: TrainTask = task if task is not None else MLMTask()
```

(delete the now-unused `OplmForMaskedLM`, `build_train_dataloader`, and
`estimate_flops_per_token` imports from this block).

(c) Replace the model-build block (the lines from `# Model` through the end of the
`if self.accelerator.is_main_process: logger.info("Model: ...")` call) with:

```python
        # Model
        _status("[dim]Building model...[/dim]")
        # Read before build_model: transformers strips the flag off the HF config during
        # PreTrainedModel.__init__; the compile block below still needs it.
        gradient_checkpointing = getattr(cfg.model, "gradient_checkpointing", False)
        model = self.task.build_model(cfg, initialization_source)

        if self.accelerator.is_main_process:
            logger.info(
                "Model: task=%s unique_parameters=%d",
                type(self.task).__name__,
                sum(parameter.numel() for parameter in model.parameters()),
            )
            if isinstance(self.task, MLMTask):
                logger.info(
                    "MLM backbone: physical_depth=%d effective_depth=%d loops=%d strategy=%s "
                    "range=[%d, %d)",
                    cfg.model.num_hidden_layers,
                    len(model.oplm.backbone.layer_execution_order),
                    cfg.model.num_loops,
                    cfg.model.loop_strategy,
                    cfg.model.loop_start,
                    cfg.model.num_hidden_layers
                    if cfg.model.loop_end is None
                    else cfg.model.loop_end,
                )
```

(d) Replace `dataloader = build_train_dataloader(cfg)` with
`dataloader = self.task.build_dataloader(cfg)`.

(e) Replace `self.flops_per_token = estimate_flops_per_token(cfg.model)` with:

```python
        # FLOP estimation; None when the task has no honest per-token formula (spec §6.4),
        # in which case train/flops, train/achieved_tflops and train/mfu are omitted.
        self.flops_per_token: int | None = self.task.flops_per_token(cfg)
```

(f) Next to the other per-log-window accumulators (`self._window_loss_sum = 0.0` ...), add:

```python
        # Task-reported per-micro-batch metrics, window-averaged in _log_step.
        self._window_metric_sums: dict[str, float] = {}
        self._window_metric_counts: dict[str, int] = {}
```

(g) In `train()`, replace the forward/backward and the accounting lines:

```python
                with self.accelerator.accumulate(self.model):
                    result = self.task.step(self.model, batch)
                    loss = result.loss
                    self.accelerator.backward(loss)
```

and

```python
                self._step_local_tokens += result.tokens
                self._samples_seen += result.samples * self.accelerator.num_processes
                self._batches_in_epoch += 1
                for key, value in result.metrics.items():
                    self._window_metric_sums[key] = self._window_metric_sums.get(key, 0.0) + value
                    self._window_metric_counts[key] = self._window_metric_counts.get(key, 0) + 1
```

(h) In `_log_step`, make FLOP metrics conditional and emit task metrics. Replace the
top of the method through the `metrics = {...}` literal with:

```python
        fractional_epoch = self._fractional_epoch()

        metrics = {
            "train/loss": loss,
            "train/epoch": fractional_epoch,
            "train/samples": self._samples_seen,
            "train/tokens": self.tokens_seen,
            "train/lr": self.scheduler.get_last_lr()[0],
        }
        if self.flops_per_token is not None:
            metrics["train/flops"] = self.flops_per_token * self.tokens_seen
        for key, total in self._window_metric_sums.items():
            metrics[f"train/{key}"] = total / self._window_metric_counts[key]
        self._window_metric_sums.clear()
        self._window_metric_counts.clear()
```

and in the throughput block replace the `achieved_tflops` lines with:

```python
            tokens_per_sec = self._tput_window_tokens / self._tput_window_seconds
            step_time_s = self._tput_window_seconds / self._tput_window_steps
            metrics["train/tokens_per_sec"] = tokens_per_sec
            metrics["train/step_time_s"] = step_time_s
            if self.flops_per_token is not None:
                achieved_tflops = (
                    self.flops_per_token
                    * self._tput_window_tokens
                    / self._tput_window_seconds
                    / 1e12
                )
                metrics["train/achieved_tflops"] = achieved_tflops
                if self.cfg.train.peak_tflops:
                    metrics["train/mfu"] = achieved_tflops / self.cfg.train.peak_tflops
```

(i) In `src/oplm/training/__init__.py` add
`from oplm.training.task import MLMTask, StepResult, TrainTask` and extend `__all__`.

- [ ] **Step 5: Run the new tests and the MLM regression gate**

Run: `.venv/bin/python -m pytest tests/training/test_task.py tests/training/test_e2e_logging.py tests/training/test_e2e_accumulation.py tests/training/test_e2e_checkpoint.py tests/training/test_trainer.py -v`
Expected: all PASS. `test_e2e_logging` still sees `train/flops` on every MLM payload.

- [ ] **Step 6: Document the seam**

Append to `docs/TRAIN.md` immediately before `## See also`:

```markdown
## 17. Training tasks (`TrainTask`)

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
`tokens`, `flops`, ...) are reserved. The folding head's `FoldTask`
(docs/FOLD.md) is the second implementation.
```

- [ ] **Step 7: Lint, type-check, commit**

Run: `.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/training/task.py src/oplm/training/trainer.py tests/training/test_task.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: clean.

```bash
git add src/oplm/training/task.py src/oplm/training/trainer.py src/oplm/training/__init__.py docs/TRAIN.md tests/training/test_task.py
git commit -m "feat(training): add TrainTask seam with MLMTask as the unchanged default"
```

---

### Task 2: EMA weights with checkpoint, resume, remote mirror, and `hf_ema/` export

**Files:**
- Create: `src/oplm/training/ema.py`
- Modify: `src/oplm/config.py` (TrainConfig fields ~L35–191, `__post_init__` ~L193–310)
- Modify: `src/oplm/configs/train/base.yaml`
- Modify: `src/oplm/training/checkpoint.py` (`save_checkpoint` 471–716, `load_checkpoint` 1015–1113)
- Modify: `src/oplm/training/remote.py` (L70 and `build_upload_job` ~L534–541)
- Modify: `src/oplm/training/trainer.py` (after schedulers ~L720; `train()` after `self.global_step += 1`; `_save_checkpoint`; `_checkpoint_extra_state`; `_resume_from_checkpoint`; `_branch_from_checkpoint`)
- Modify: `tests/training/conftest.py` (`tiny_train_cfg`)
- Modify: `docs/CONFIG.md` (train table), `docs/TRAIN.md` (§16 knobs table and "What a resume restores")
- Test: `tests/training/test_ema.py`, `tests/training/test_e2e_ema.py`, `tests/training/test_remote.py`, `tests/training/test_config.py` (existing parity test picks up the yaml key)

**Interfaces:**
- Consumes: `Trainer` from Task 1 (`self._unwrapped_model` is new here), `save_checkpoint`/`load_checkpoint`.
- Produces: `TrainConfig.ema_decay: float | None`; `build_ema(model, decay) -> AveragedModel`;
  `sync_ema_buffers(ema, model)`; constants `EMA_SIDECAR_NAME = "ema.pt"`,
  `EMA_HF_DIRNAME = "hf_ema"`; `save_checkpoint(..., ema: AveragedModel | None = None)`;
  `load_checkpoint(..., ema: AveragedModel | None = None)`; `Trainer._ema`;
  `trainer_state.json["ema"] = {"decay", "sidecar", "hf_dir", "n_averaged"}`.
  **M1/M2 requirement:** `build_ema` deep-copies the model, so the fold model
  must keep the frozen LM out of `copy.deepcopy` (e.g. a `__deepcopy__` that
  shares the LM reference) or the EMA would duplicate the LM.

- [ ] **Step 1: Write the failing unit tests**

Create `tests/training/test_ema.py`:

```python
"""EMA tracker semantics and config validation (docs/TRAIN.md §16, `train.ema_decay`)."""

from __future__ import annotations

import pytest
import torch

from oplm.config import TrainConfig
from oplm.training.ema import EMA_HF_DIRNAME, EMA_SIDECAR_NAME, build_ema, sync_ema_buffers


def test_build_ema_first_update_copies_then_lerps() -> None:
    torch.manual_seed(0)
    live = torch.nn.Linear(4, 2)
    ema = build_ema(live, decay=0.5)
    assert int(ema.n_averaged) == 0

    ema.update_parameters(live)
    assert int(ema.n_averaged) == 1
    assert torch.equal(ema.module.weight, live.weight)
    assert ema.module.weight.data_ptr() != live.weight.data_ptr()  # independent copy

    with torch.no_grad():
        live.weight.add_(1.0)
    ema.update_parameters(live)
    assert int(ema.n_averaged) == 2
    assert torch.allclose(ema.module.weight, live.weight - 0.5)

    assert set(ema.state_dict()) == {"n_averaged", "module.weight", "module.bias"}
    assert (EMA_SIDECAR_NAME, EMA_HF_DIRNAME) == ("ema.pt", "hf_ema")


def test_sync_ema_buffers_copies_live_buffers() -> None:
    live = torch.nn.BatchNorm1d(3)
    ema = build_ema(live, decay=0.9)
    with torch.no_grad():
        live.running_mean.fill_(7.0)
    sync_ema_buffers(ema, live)
    assert torch.equal(ema.module.running_mean, live.running_mean)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_ema_decay_must_be_in_open_unit_interval(bad: float) -> None:
    with pytest.raises(ValueError, match="ema_decay"):
        TrainConfig(ema_decay=bad)


def test_ema_refused_under_hsdp() -> None:
    with pytest.raises(ValueError, match="hsdp"):
        TrainConfig(parallelism="hsdp", ema_decay=0.999)
    TrainConfig(ema_decay=0.999)  # ddp default: accepted
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/training/test_ema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.training.ema'`.

- [ ] **Step 3: Config field, validation, yaml default**

In `src/oplm/config.py` `TrainConfig`, after `remote_checkpoint_uri: str | None = None`:

```python
    # Exponential moving average of the trainable weights (fold stages set 0.999).
    # None = off (MLM runs unchanged). Updated once per successful optimizer step;
    # every checkpoint then also writes ema.pt + hf_ema/ (docs/TRAIN.md §16).
    ema_decay: float | None = None
```

In `__post_init__`, after the `keep_every_n_hours` check:

```python
        if self.ema_decay is not None and not (0.0 < self.ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1) when set, got {self.ema_decay}")
```

and alongside the other `hsdp` refusals:

```python
        if self.parallelism == "hsdp" and self.ema_decay is not None:
            # The EMA copy and its ema.pt sidecar are built from full replicated (DDP)
            # parameters on every rank; under FSDP2 they are DTensor shards.
            raise ValueError(
                "parallelism='hsdp' is incompatible with ema_decay: the EMA tracker and its "
                "ema.pt sidecar are built from replicated (DDP) parameters, not FSDP2 shards."
            )
```

In `src/oplm/configs/train/base.yaml`, after `remote_checkpoint_uri: null`:

```yaml
  # Exponential moving average of the trainable weights, updated once per optimizer
  # step; null disables it. Fold stages use 0.999. Adds ema.pt + hf_ema/ to checkpoints.
  ema_decay: null
```

- [ ] **Step 4: Create `src/oplm/training/ema.py`**

```python
"""Exponential moving average of trainable weights (``train.ema_decay``; off by default)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

if TYPE_CHECKING:
    from torch import nn

__all__ = ["EMA_HF_DIRNAME", "EMA_SIDECAR_NAME", "build_ema", "sync_ema_buffers"]

EMA_SIDECAR_NAME = "ema.pt"
EMA_HF_DIRNAME = "hf_ema"


def build_ema(model: nn.Module, decay: float) -> AveragedModel:
    """Deep-copy ``model`` into an EMA tracker driven by ``update_parameters``.

    The first ``update_parameters`` call copies the live weights; every later call
    applies ``ema = decay * ema + (1 - decay) * live``. ``n_averaged`` counts the
    updates and is part of the tracker's ``state_dict``. Buffers are not averaged;
    see :func:`sync_ema_buffers`.

    Args:
        model: The unwrapped (no DDP/compile wrapper) live model.
        decay: EMA decay in ``(0, 1)``.
    """
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=False)


@torch.no_grad()
def sync_ema_buffers(ema: AveragedModel, model: nn.Module) -> None:
    """Copy the live model's buffers into the EMA copy before exporting it.

    OPLM's buffers (RoPE caches, residual ``alpha``) are constants, so this is a
    no-op in practice; it keeps an exported ``hf_ema/`` self-consistent for any
    buffer that does change during training.
    """
    for (name, ema_buffer), (live_name, live_buffer) in zip(
        ema.module.named_buffers(), model.named_buffers(), strict=True
    ):
        assert name == live_name, f"buffer order mismatch: {name} vs {live_name}"
        ema_buffer.copy_(live_buffer)
```

- [ ] **Step 5: Run the unit tests**

Run: `.venv/bin/python -m pytest tests/training/test_ema.py tests/training/test_config.py -v`
Expected: PASS (including `test_train_base_yaml_matches_dataclass[ema_decay]`).

- [ ] **Step 6: Write the failing checkpoint/E2E tests**

Add `ema_decay: float | None = None` to `tiny_train_cfg`'s keyword parameters in
`tests/training/conftest.py` and pass `ema_decay=ema_decay` into its `TrainConfig(...)`.

Create `tests/training/test_e2e_ema.py`:

```python
"""EMA through the real checkpoint layer and Trainer: sidecar round-trip, hf_ema/ export,
optimizer-step counting under accumulation, resume, and the pre-EMA-checkpoint case."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import pytest
import torch

from oplm.config import DataConfig, OplmConfig, TrainConfig
from oplm.model import OplmConfig as OplmModelConfig
from oplm.model import OplmForMaskedLM
from oplm.training.checkpoint import load_checkpoint, save_checkpoint
from oplm.training.ema import build_ema
from oplm.training.initialization import resolve_initialization_source
from tests.training.conftest import tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow


def _cfg(decay: float) -> OplmConfig:
    return OplmConfig(
        model=OplmModelConfig(
            hidden_size=32, num_attention_heads=4, num_hidden_layers=2, max_position_embeddings=64
        ),
        train=TrainConfig(wandb_enabled=False, mixed_precision="no", ema_decay=decay),
        data=DataConfig(num_workers=0, pin_memory=False),
    )


def _prepared(cfg: OplmConfig) -> tuple[Any, Any, Any, Any, Any]:
    """CPU accelerator, prepared tiny model + AdamW + LambdaLR, and a fresh EMA tracker."""
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model = OplmForMaskedLM(cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    model, optimizer = accelerator.prepare(model, optimizer)
    ema = build_ema(accelerator.unwrap_model(model), cfg.train.ema_decay)
    return accelerator, model, optimizer, scheduler, ema


def test_ema_sidecar_and_hf_ema_round_trip(tmp_path: Path) -> None:
    cfg = _cfg(0.5)
    accelerator, model, optimizer, scheduler, ema = _prepared(cfg)
    inputs = torch.randint(0, cfg.model.vocab_size, (2, 8))
    for _ in range(3):
        loss = model(input_ids=inputs, labels=inputs).loss
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        ema.update_parameters(accelerator.unwrap_model(model))
    assert int(ema.n_averaged) == 3

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizers=[optimizer],
        schedulers=[scheduler],
        cfg=cfg,
        output_dir=str(tmp_path),
        global_step=3,
        epoch=0,
        samples_seen=6,
        tokens_seen=48,
        ema=ema,
    )
    committed = tmp_path / "checkpoint-3"
    assert (committed / "ema.pt").is_file()
    assert (committed / "hf_ema" / "config.json").is_file()
    assert (committed / "hf" / "config.json").is_file()

    # hf_ema/ carries the EMA weights (not the live ones) and is a valid init_from target.
    reloaded = OplmForMaskedLM.from_pretrained(committed / "hf_ema", local_files_only=True)
    ema_state = ema.module.state_dict()
    live_state = accelerator.unwrap_model(model).state_dict()
    for name, tensor in reloaded.state_dict().items():
        assert torch.equal(tensor, ema_state[name]), name
    assert any(not torch.equal(ema_state[n], live_state[n]) for n in live_state)
    assert resolve_initialization_source(str(committed / "hf_ema")) == (
        committed / "hf_ema"
    ).resolve()

    # A fresh tracker restores the update count and every tensor.
    fresh_accelerator, fresh_model, fresh_optimizer, fresh_scheduler, fresh_ema = _prepared(cfg)
    load_checkpoint(
        fresh_accelerator,
        fresh_model,
        [fresh_optimizer],
        [fresh_scheduler],
        str(committed),
        cfg,
        ema=fresh_ema,
    )
    assert int(fresh_ema.n_averaged) == 3
    for name, tensor in fresh_ema.module.state_dict().items():
        assert torch.equal(tensor, ema_state[name]), name


def test_ema_counts_optimizer_steps_and_survives_resume(
    tmp_path: Path, training_parquet: Path
) -> None:
    from oplm.training.trainer import Trainer

    common = dict(
        batch_size=4, gradient_accumulation_steps=2, log_every=1, warmup_steps=0, ema_decay=0.5
    )
    cfg1 = tiny_train_cfg(tmp_path, training_parquet, max_steps=4, save_every=4, **common)
    first = Trainer(cfg1)
    first.train()
    assert first._ema is not None
    assert int(first._ema.n_averaged) == 4  # optimizer steps, not the 8 micro-batches

    ckpt = tmp_path / "checkpoint-4"
    state = json.loads((ckpt / "trainer_state.json").read_text())
    assert state["ema"] == {"decay": 0.5, "sidecar": "ema.pt", "hf_dir": "hf_ema", "n_averaged": 4}
    assert (ckpt / "ema.pt").is_file()
    assert (ckpt / "hf_ema" / "model.safetensors").is_file()
    saved = torch.load(ckpt / "ema.pt", map_location="cpu", weights_only=True)

    cfg2 = tiny_train_cfg(
        tmp_path, training_parquet, max_steps=8, save_every=8, resume_from=str(ckpt), **common
    )
    resumed = Trainer(cfg2)
    assert resumed._ema is not None
    assert int(resumed._ema.n_averaged) == 4  # restored, not restarted
    for name, tensor in resumed._ema.state_dict().items():
        assert torch.equal(tensor, saved[name]), name

    resumed.train()
    assert int(resumed._ema.n_averaged) == 8 == resumed.global_step
    live = resumed._unwrapped_model.state_dict()
    for name, tensor in resumed._ema.module.state_dict().items():
        assert torch.isfinite(tensor).all(), name
    assert any(not torch.equal(resumed._ema.module.state_dict()[n], live[n]) for n in live)


def test_resume_into_ema_from_checkpoint_without_ema_warns(
    tmp_path: Path, training_parquet: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from oplm.training.trainer import Trainer

    cfg1 = tiny_train_cfg(tmp_path, training_parquet, max_steps=2, save_every=2, batch_size=4)
    Trainer(cfg1).train()
    ckpt = tmp_path / "checkpoint-2"
    assert not (ckpt / "ema.pt").exists()
    assert not (ckpt / "hf_ema").exists()

    cfg2 = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=4,
        save_every=4,
        batch_size=4,
        resume_from=str(ckpt),
        ema_decay=0.5,
    )
    with caplog.at_level(logging.WARNING):
        resumed = Trainer(cfg2)
    assert "ema.pt" in caplog.text
    assert resumed._ema is not None
    assert int(resumed._ema.n_averaged) == 0

    resumed.train()
    assert int(resumed._ema.n_averaged) == 2
    assert (tmp_path / "checkpoint-4" / "ema.pt").is_file()
```

Run: `.venv/bin/python -m pytest tests/training/test_e2e_ema.py -v`
Expected: FAIL (`save_checkpoint() got an unexpected keyword argument 'ema'`).

- [ ] **Step 7: Checkpoint sidecar and export**

In `src/oplm/training/checkpoint.py`:

(a) Imports: add `from oplm.training.ema import EMA_HF_DIRNAME, EMA_SIDECAR_NAME, sync_ema_buffers`
at module level and `from torch.optim.swa_utils import AveragedModel` inside the
`if TYPE_CHECKING:` block.

(b) Add helpers next to `_write_scaler_sidecar` / `_restore_scaler_sidecar`:

```python
def _write_ema_artifacts(tmp_dir: Path, ema: AveragedModel, live_model: Any) -> None:
    """Write ``ema.pt`` (tracker state incl. ``n_averaged``) and ``hf_ema/``, main process only.

    Both land in ``tmp_dir`` before the commit rename, so they are covered by the
    atomic commit and (via ``remote._SHARED_ARTIFACT_NAMES`` / the ``hf_ema`` glob)
    by the remote mirror. DDP keeps every rank's tracker identical, so one copy suffices.
    """
    sync_ema_buffers(ema, live_model)
    torch.save(ema.state_dict(), tmp_dir / EMA_SIDECAR_NAME)
    ema_dir = tmp_dir / EMA_HF_DIRNAME
    ema_module: Any = ema.module  # a deep copy of the HF model: has save_pretrained
    ema_module.save_pretrained(ema_dir)
    get_tokenizer().save_pretrained(ema_dir)


def _restore_ema_sidecar(checkpoint_dir: Path, ema: AveragedModel | None) -> None:
    """Restore the EMA tracker from ``ema.pt`` on every rank; warn on a run/checkpoint mismatch."""
    sidecar_path = checkpoint_dir / EMA_SIDECAR_NAME
    if ema is not None and sidecar_path.is_file():
        ema.load_state_dict(torch.load(sidecar_path, map_location="cpu", weights_only=True))
    elif ema is not None:
        logger.warning(
            "train.ema_decay is set but checkpoint %s has no %s; the EMA tracker starts "
            "from the live weights at the next optimizer step.",
            checkpoint_dir,
            EMA_SIDECAR_NAME,
        )
    elif sidecar_path.is_file():
        logger.warning(
            "Checkpoint %s has EMA state (%s) but train.ema_decay is unset; not restored.",
            checkpoint_dir,
            EMA_SIDECAR_NAME,
        )
```

(c) `save_checkpoint`: add the keyword parameter `ema: AveragedModel | None = None`
(after `process_group`) and, inside the `if accelerator.is_main_process:` block right
after `get_tokenizer().save_pretrained(hf_dir)`:

```python
        if ema is not None:
            _write_ema_artifacts(tmp_dir, ema, unwrapped)
```

(d) `load_checkpoint`: add the keyword parameter `ema: AveragedModel | None = None`
and call `_restore_ema_sidecar(ckpt_path, ema)` right after `_restore_scaler_sidecar(ckpt_path, accelerator)`.

- [ ] **Step 8: Remote mirror**

In `src/oplm/training/remote.py` change the constant to

```python
_SHARED_ARTIFACT_NAMES = (
    ".metadata",
    "trainer_state.json",
    "config.yaml",
    "scaler.pt",
    "ema.pt",
    "KEEP",
)
```

and in `build_upload_job` replace the `hf_dir = checkpoint_dir / "hf"` block with:

```python
        for export_name in ("hf", "hf_ema"):
            export_dir = checkpoint_dir / export_name
            if export_dir.is_dir():
                shared_files.extend(
                    sorted(
                        p.relative_to(checkpoint_dir)
                        for p in export_dir.rglob("*")
                        if p.is_file()
                    )
                )
```

Extend the existing multi-process `build_upload_job` test in
`tests/training/test_remote.py` (find it with `grep -n shared_files tests/training/test_remote.py`):
create `ema.pt` and `hf_ema/model.safetensors` in its fake checkpoint directory and
assert `Path("ema.pt")` and `Path("hf_ema/model.safetensors")` are in the main rank's
`shared_files`, while a non-main rank's `shared_files` stays `None`.

- [ ] **Step 9: Trainer wiring**

In `src/oplm/training/trainer.py`:

(a) Add `from torch.optim.swa_utils import AveragedModel` to the `if TYPE_CHECKING:` block
and `from oplm.training.ema import EMA_HF_DIRNAME, EMA_SIDECAR_NAME` at module level.

(b) Right after `self.scheduler = self.schedulers[0]`:

```python
        # Unwrapped live model (no DDP/compile wrapper): the EMA source and hf/ export
        # target. keep_torch_compile=False for the reason given in checkpoint.save_checkpoint.
        self._unwrapped_model = self.accelerator.unwrap_model(
            self.model, keep_torch_compile=False
        )
        # EMA (train.ema_decay). Built after prepare/compile so the copy sits on the
        # right device, and before the resume below so a checkpoint's ema.pt lands in it.
        self._ema: AveragedModel | None = None
        if cfg.train.ema_decay is not None:
            from oplm.training.ema import build_ema

            self._ema = build_ema(self._unwrapped_model, cfg.train.ema_decay)
```

(c) In `train()`, directly after `self.global_step += 1`:

```python
                # EMA: once per real optimizer step (never on accumulation micro-steps --
                # we are past the sync_gradients gate -- and never on a step the fp16
                # scaler skipped).
                if self._ema is not None and not self.accelerator.optimizer_step_was_skipped:
                    self._ema.update_parameters(self._unwrapped_model)
```

(d) `_save_checkpoint`: pass `ema=self._ema` to `save_checkpoint(...)`.

(e) `_checkpoint_extra_state`: before `return extra_state` add

```python
        if self._ema is not None:
            extra_state["ema"] = {
                "decay": self.cfg.train.ema_decay,
                "sidecar": EMA_SIDECAR_NAME,
                "hf_dir": EMA_HF_DIRNAME,
                "n_averaged": int(self._ema.n_averaged),
            }
```

(f) `_resume_from_checkpoint` and `_branch_from_checkpoint`: pass `ema=self._ema` to
their `load_checkpoint(...)` calls.

- [ ] **Step 10: Run the EMA and checkpoint suites**

Run: `.venv/bin/python -m pytest tests/training/test_ema.py tests/training/test_e2e_ema.py tests/training/test_e2e_checkpoint.py tests/training/test_e2e_dcp.py tests/training/test_remote.py tests/training/test_e2e_accumulation.py -v`
Expected: all PASS. The skipped-step branch (`optimizer_step_was_skipped`) has no CPU
test: Accelerate builds no fp16 scaler on CPU. It is one guarded line; leave it.

- [ ] **Step 11: Docs**

`docs/CONFIG.md`, train table, after the `train.remote_checkpoint_uri` row:

```markdown
| `train.ema_decay` | `float \| null` | `null` | Exponential moving average of the trainable weights, updated once per successful optimizer step (never on accumulation micro-steps or a skipped fp16 step). Every checkpoint then also writes `ema.pt` (tracker state + update count, restored on resume) and an `hf_ema/` export next to `hf/`; a later stage selects it with `train.init_from=<ckpt>/hf_ema`. Must be in `(0, 1)` when set. `ddp` only (refused under `hsdp`). See [TRAIN.md §16](TRAIN.md#16-fault-tolerance). |
```

`docs/TRAIN.md` §16 knobs table, after the `remote_checkpoint_uri` row:

```markdown
| `ema_decay` | `null` | EMA of the trainable weights (fold stages use `0.999`). One update per optimizer step. Each checkpoint adds `ema.pt` + `hf_ema/`; resume restores the tracker; a pre-EMA checkpoint resumed with this set logs a warning and restarts the average from the live weights. `ddp` only. |
```

and in "What a resume restores" add the bullet
`- the EMA tracker (`ema.pt`: averaged tensors and update count) when `train.ema_decay` is set,`.

- [ ] **Step 12: Lint, type-check, commit**

Run: `.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/training/ema.py src/oplm/training/checkpoint.py src/oplm/training/remote.py src/oplm/training/trainer.py src/oplm/config.py tests/training/test_ema.py tests/training/test_e2e_ema.py tests/training/conftest.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: clean.

```bash
git add src/oplm/training/ema.py src/oplm/training/checkpoint.py src/oplm/training/remote.py src/oplm/training/trainer.py src/oplm/config.py src/oplm/configs/train/base.yaml tests/training/conftest.py tests/training/test_ema.py tests/training/test_e2e_ema.py tests/training/test_remote.py docs/CONFIG.md docs/TRAIN.md
git commit -m "feat(training): EMA weights with ema.pt sidecar, hf_ema export, and resume"
```

---

### Task 3: `oplm.fold` package, attribution, and the staged triangle-multiplication reference

**Files:**
- Create: `src/oplm/fold/__init__.py`, `src/oplm/fold/trimul.py`, `THIRD_PARTY_NOTICES.md`, `tests/fold/__init__.py`, `tests/fold/test_trimul.py`
- Modify: `pyproject.toml` (`[project.optional-dependencies]`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Direction = Literal["outgoing", "incoming"]`, `TRIMUL_EPS = 1e-5`,
  `trimul_pre(z, mask, norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, *, eps) -> (left, right, zn)`,
  `trimul_contract(left, right, direction, *, chunk_size) -> Tensor` (fp32),
  `trimul_post(contracted, zn, norm_out_weight, norm_out_bias, p_out_weight, g_out_weight, *, eps) -> Tensor`,
  `trimul_reference(z, direction, mask, *weights, eps, chunk_size) -> Tensor`,
  `TriangleMultiplication(width, direction, *, eps, chunk_size)` with `kernel_weights()`
  returning the 8-tuple `(norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight, norm_out_bias, p_out_weight, g_out_weight)`.
  Task 4 adds `backend` and the dispatching `forward`; M1's `trunk.py` wraps the module
  in the residual/dropout `PairUpdateBlock`.

- [ ] **Step 1: Package, extra, attribution**

`src/oplm/fold/__init__.py`:

```python
"""Structure prediction head for OPLM (docs/FOLD.md).

Milestone 0 ships the kernels and tooling only: :mod:`oplm.fold.trimul`,
:mod:`oplm.fold.attention`, and ``oplm fold bench-kernels``. Model, data,
losses and training land in later milestones of the design spec.
"""
```

`tests/fold/__init__.py`: empty file.

`pyproject.toml`, after the `train = [...]` extra:

```toml
fold = [
    # cuEquivariance fused triangle multiplication (oplm.fold.trimul dispatch). The
    # staged PyTorch reference runs without it; CPU tests never import it. The cu13
    # ops wheel matches the torch cu130 builds. Versions are re-pinned after the
    # B200 bench (docs/FOLD.md "Kernel benchmark").
    "cuequivariance-torch>=0.10.0",
    "cuequivariance-ops-torch-cu13>=0.10.0",
]
```

`THIRD_PARTY_NOTICES.md` (then append the license text with the command below):

```markdown
# Third-party notices

OPLM is MIT-licensed (see `LICENSE`). The components below are derived from
third-party work and keep their upstream notices.

## ESMFold2 (Chan Zuckerberg Biohub)

Portions of `src/oplm/fold/` are ported from Biohub's ESMFold2 implementation,
<https://github.com/Biohub/esm>. The upstream source files carry Apache License
2.0 headers (`Copyright 2026 Biohub. All rights reserved.`); the repository's
top-level `LICENSE.md` states MIT. The per-file Apache 2.0 notice is preserved
here. Each ported OPLM module names its upstream source and the modifications in
its module docstring.

| OPLM module | Upstream file | What was taken |
| --- | --- | --- |
| `src/oplm/fold/trimul.py` | `esm/models/esmfold2/layers.py` (`TriangleMultiplicativeBlock`) | Parameter conventions, forward equations, cuEquivariance weight mapping |
| `src/oplm/fold/attention.py` | `esm/models/esmfold2/layers.py` (`AttentionPairBias`, `SWA3DRoPEAttention`) | Pair-bias logit and key-mask semantics; rank-based sliding-window mask |

The Apache License, Version 2.0 follows.

```

Append the license text: `curl -sL https://www.apache.org/licenses/LICENSE-2.0.txt >> THIRD_PARTY_NOTICES.md`.

- [ ] **Step 2: Write the failing tests**

Create `tests/fold/test_trimul.py`:

```python
"""Triangle multiplication: staged reference vs the transcribed ESMFold2 equation, chunking,
masks, gradients, stage locality, and state-dict parity (docs/FOLD.md "Triangle multiplication")."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from oplm.fold.trimul import (
    TriangleMultiplication,
    trimul_contract,
    trimul_post,
    trimul_pre,
    trimul_reference,
)

DIRECTIONS = ("outgoing", "incoming")
_EINSUM = {"outgoing": "bikd,bjkd->bijd", "incoming": "bkid,bkjd->bijd"}


def _upstream_forward(
    z: torch.Tensor, mask: torch.Tensor | None, m: TriangleMultiplication
) -> torch.Tensor:
    """ESMFold2 ``TriangleMultiplicativeBlock.forward``, unchunked, transcribed verbatim."""
    if mask is None:
        mask = z.new_ones(z.shape[:-1])
    zn = F.layer_norm(z, (m.width,), m.norm_start.weight, m.norm_start.bias, 1e-5)
    bundled = F.linear(zn, m.proj_bundle.weight)
    signal, gate_logits = bundled.split(2 * m.width, dim=-1)
    routed = signal * torch.sigmoid(gate_logits)
    routed = routed * mask.unsqueeze(-1)
    left, right = routed.float().chunk(2, dim=-1)
    contracted = torch.einsum(_EINSUM[m.direction], left, right)
    mixed = F.linear(
        F.layer_norm(contracted, (m.width,), m.norm_mix.weight, m.norm_mix.bias, 1e-5),
        m.proj_emit.weight,
    )
    return mixed * torch.sigmoid(F.linear(zn, m.proj_gate.weight))


def _inputs(n: int = 7, width: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(2, n, n, width, generator=g)
    mask = (torch.rand(2, n, n, generator=g) > 0.3).float()
    return z, mask


def _module(direction: str, width: int = 32, seed: int = 0) -> TriangleMultiplication:
    torch.manual_seed(seed)
    m = TriangleMultiplication(width, direction)
    with torch.no_grad():  # non-trivial norms so a wrong eps/affine placement shows up
        for norm in (m.norm_start, m.norm_mix):
            norm.weight.uniform_(0.5, 1.5)
            norm.bias.uniform_(-0.5, 0.5)
    return m


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_matches_upstream_equation(direction: str) -> None:
    m = _module(direction)
    z, mask = _inputs()
    torch.testing.assert_close(m(z, mask), _upstream_forward(z, mask, m), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(m(z), _upstream_forward(z, None, m), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_chunked_equals_unchunked(direction: str) -> None:
    m = _module(direction)
    z, mask = _inputs(n=5)  # 5 rows, chunk 2 -> an uneven tail chunk
    m.chunk_size = 2
    chunked = m(z, mask)
    m.chunk_size = None
    torch.testing.assert_close(chunked, m(z, mask), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_masked_row_contributes_nothing(direction: str) -> None:
    m = _module(direction)
    z, _ = _inputs(n=6)
    mask = torch.ones(2, 6, 6)
    mask[:, 2, :] = 0.0  # row 2 fully masked
    left, right, _zn = trimul_pre(z, mask, *m.kernel_weights()[:4], eps=m.eps)
    contracted = trimul_contract(left, right, m.direction, chunk_size=2)
    if direction == "outgoing":  # out[i, j] = sum_k left[i, k] right[j, k]
        assert contracted[:, 2].abs().max() == 0 and contracted[:, :, 2].abs().max() == 0
    else:  # out[i, j] = sum_k left[k, i] right[k, j]: row 2 drops out of the k-sum
        unmasked = trimul_contract(
            *trimul_pre(z, None, *m.kernel_weights()[:4], eps=m.eps)[:2], m.direction, chunk_size=2
        )
        assert not torch.allclose(contracted, unmasked)
    assert torch.isfinite(m(z, mask)).all()


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_gradients_reach_input_and_every_parameter(direction: str) -> None:
    m = _module(direction)
    z, mask = _inputs()
    z.requires_grad_(True)
    m(z, mask).square().sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_pre_and_post_stages_are_pointwise_in_rows_and_columns() -> None:
    m = _module("outgoing")
    z, mask = _inputs(n=8)
    w = m.kernel_weights()
    full = trimul_pre(z, mask, *w[:4], eps=m.eps)
    block = trimul_pre(z[:, 2:5, 1:7], mask[:, 2:5, 1:7], *w[:4], eps=m.eps)
    for full_t, block_t in zip(full, block, strict=True):
        torch.testing.assert_close(block_t, full_t[:, 2:5, 1:7])
    contracted = trimul_contract(full[0], full[1], "outgoing", chunk_size=None)
    out_full = trimul_post(contracted, full[2], *w[4:], eps=m.eps)
    out_block = trimul_post(contracted[:, 2:5, 1:7], full[2][:, 2:5, 1:7], *w[4:], eps=m.eps)
    torch.testing.assert_close(out_block, out_full[:, 2:5, 1:7])


def test_functional_reference_equals_module() -> None:
    m = _module("incoming")
    z, mask = _inputs()
    torch.testing.assert_close(
        trimul_reference(z, "incoming", mask, *m.kernel_weights(), eps=m.eps, chunk_size=3),
        m(z, mask),
    )


def test_incoming_and_outgoing_differ() -> None:
    z, mask = _inputs()
    out = _module("outgoing")(z, mask)
    inc = _module("incoming")(z, mask)
    assert not torch.allclose(out, inc)


def test_state_dict_matches_upstream_names_and_shapes() -> None:
    m = TriangleMultiplication(64, "outgoing")
    shapes = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    assert shapes == {
        "norm_start.weight": (64,),
        "norm_start.bias": (64,),
        "norm_mix.weight": (64,),
        "norm_mix.bias": (64,),
        "proj_bundle.weight": (256, 64),
        "proj_emit.weight": (64, 64),
        "proj_gate.weight": (64, 64),
    }


def test_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        TriangleMultiplication(32, "sideways")  # type: ignore[arg-type]
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/fold/test_trimul.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.fold.trimul'`.

- [ ] **Step 4: Create `src/oplm/fold/trimul.py`**

```python
"""Triangle multiplicative update: staged reference and the ``TriangleMultiplication`` module.

Ported from Biohub's ESMFold2 ``TriangleMultiplicativeBlock``
(https://github.com/Biohub/esm, ``esm/models/esmfold2/layers.py``, Apache-2.0;
see THIRD_PARTY_NOTICES.md). Modifications: the forward is split into three
explicit stages -- local pre-projection, triangular contraction, local
post-projection -- so the contraction is the only cross-row/column operation
(design §5.2, §5.6); the contraction accumulates in fp32 chunked over output
rows; LayerNorms compute in fp32 and cast back (the ``OplmLayerNorm`` contract);
kernel dispatch is explicit instead of a try/except fallback.

Parameter names and shapes match upstream so released ESMFold2 weights load
without remapping: ``norm_start``/``norm_mix`` (LayerNorm, eps 1e-5),
``proj_bundle`` (D -> 4D, no bias; rows [0:2D] signal, [2D:4D] gate logits),
``proj_emit`` (D -> D, no bias), ``proj_gate`` (D -> D, no bias). The functional
API takes the eight tensors in cuEquivariance's order so the fused kernel is a
straight pass-through: ``p_in_weight = proj_bundle.weight[:2D]``,
``g_in_weight = proj_bundle.weight[2D:]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "TRIMUL_EPS",
    "Direction",
    "TriangleMultiplication",
    "trimul_contract",
    "trimul_post",
    "trimul_pre",
    "trimul_reference",
]

Direction = Literal["outgoing", "incoming"]
TRIMUL_EPS = 1e-5
_EINSUM: dict[str, str] = {"outgoing": "bikd,bjkd->bijd", "incoming": "bkid,bkjd->bijd"}


def _layer_norm_fp32(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    """LayerNorm with fp32 internals, cast back to ``x``'s dtype."""
    return F.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), eps).to(x.dtype)


def trimul_pre(
    z: Tensor,
    mask: Tensor | None,
    norm_in_weight: Tensor,
    norm_in_bias: Tensor,
    p_in_weight: Tensor,
    g_in_weight: Tensor,
    *,
    eps: float = TRIMUL_EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Stage 1 (pointwise): normalize, project, gate, mask.

    Args:
        z: Pair block ``(B, I, J, D)``.
        mask: Pair validity ``(B, I, J)`` (bool or float); ``None`` means all valid.
        norm_in_weight: ``norm_start`` affine weight ``(D,)``.
        norm_in_bias: ``norm_start`` affine bias ``(D,)``.
        p_in_weight: Signal projection ``(2D, D)`` (``proj_bundle.weight[:2D]``).
        g_in_weight: Gate-logit projection ``(2D, D)`` (``proj_bundle.weight[2D:]``).
        eps: LayerNorm epsilon.

    Returns:
        ``(left, right, zn)``: the two contraction operands ``(B, I, J, D)`` in ``z``'s
        dtype and the normalized input the output gate is computed from.
    """
    zn = _layer_norm_fp32(z, norm_in_weight, norm_in_bias, eps)
    routed = F.linear(zn, p_in_weight) * torch.sigmoid(F.linear(zn, g_in_weight))
    if mask is not None:
        routed = routed * mask.to(routed.dtype).unsqueeze(-1)
    left, right = routed.chunk(2, dim=-1)
    return left, right, zn


def trimul_contract(
    left: Tensor, right: Tensor, direction: Direction, *, chunk_size: int | None = 64
) -> Tensor:
    """Stage 2, the only cross-row/column op: the triangular contraction in fp32.

    ``outgoing``: ``out[b,i,j] = sum_k left[b,i,k] * right[b,j,k]``;
    ``incoming``: ``out[b,i,j] = sum_k left[b,k,i] * right[b,k,j]``.
    Output rows are chunked so one fp32 chunk of ``left`` at a time is live next to
    the fp32 copy of ``right``. The result is fp32.
    """
    # ponytail: keeps one full fp32 copy of `right` (4 GiB at L=2048, D=256); the fused
    # cuEquivariance path is the production kernel, this is the oracle/fallback.
    equation = _EINSUM[direction]
    right32 = right.float()
    n = left.shape[1]
    if chunk_size is None or n <= chunk_size:
        return torch.einsum(equation, left.float(), right32)
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        left_chunk = left[:, start:end] if direction == "outgoing" else left[:, :, start:end]
        chunks.append(torch.einsum(equation, left_chunk.float(), right32))
    return torch.cat(chunks, dim=1)


def trimul_post(
    contracted: Tensor,
    zn: Tensor,
    norm_out_weight: Tensor,
    norm_out_bias: Tensor,
    p_out_weight: Tensor,
    g_out_weight: Tensor,
    *,
    eps: float = TRIMUL_EPS,
) -> Tensor:
    """Stage 3 (pointwise): normalize the contraction in fp32, project out, apply the gate."""
    normed = _layer_norm_fp32(contracted, norm_out_weight, norm_out_bias, eps)
    mixed = F.linear(normed.to(zn.dtype), p_out_weight)
    return mixed * torch.sigmoid(F.linear(zn, g_out_weight))


def trimul_reference(
    z: Tensor,
    direction: Direction,
    mask: Tensor | None,
    norm_in_weight: Tensor,
    norm_in_bias: Tensor,
    p_in_weight: Tensor,
    g_in_weight: Tensor,
    norm_out_weight: Tensor,
    norm_out_bias: Tensor,
    p_out_weight: Tensor,
    g_out_weight: Tensor,
    *,
    eps: float = TRIMUL_EPS,
    chunk_size: int | None = 64,
) -> Tensor:
    """The differentiable staged reference (CPU path, backward oracle, fused-path oracle)."""
    left, right, zn = trimul_pre(
        z, mask, norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, eps=eps
    )
    contracted = trimul_contract(left, right, direction, chunk_size=chunk_size)
    return trimul_post(
        contracted, zn, norm_out_weight, norm_out_bias, p_out_weight, g_out_weight, eps=eps
    )


class TriangleMultiplication(nn.Module):
    """One triangle multiplicative update, returning the delta.

    The residual add and row-shared dropout live in the pair-update block that
    owns this module (milestone 1), exactly as upstream's ``PairUpdateBlock``.
    """

    def __init__(
        self,
        width: int,
        direction: Direction,
        *,
        eps: float = TRIMUL_EPS,
        chunk_size: int | None = 64,
    ) -> None:
        super().__init__()
        if direction not in _EINSUM:
            raise ValueError(f"direction must be 'outgoing' or 'incoming', got {direction!r}")
        self.width = width
        self.direction: Direction = direction
        self.eps = eps
        self.chunk_size = chunk_size
        # Plain nn.LayerNorm holders keep the upstream state-dict exactly; the fp32
        # math lives in _layer_norm_fp32, which is OplmLayerNorm's contract.
        self.norm_start = nn.LayerNorm(width, eps=eps)
        self.norm_mix = nn.LayerNorm(width, eps=eps)
        self.proj_bundle = nn.Linear(width, 4 * width, bias=False)
        self.proj_emit = nn.Linear(width, width, bias=False)
        self.proj_gate = nn.Linear(width, width, bias=False)

    def kernel_weights(self) -> tuple[Tensor, ...]:
        """The eight functional/cuEquivariance operands, ``proj_bundle`` split into signal/gate rows."""
        bundle = self.proj_bundle.weight
        return (
            self.norm_start.weight,
            self.norm_start.bias,
            bundle[: 2 * self.width],
            bundle[2 * self.width :],
            self.norm_mix.weight,
            self.norm_mix.bias,
            self.proj_emit.weight,
            self.proj_gate.weight,
        )

    def forward(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        return trimul_reference(
            z, self.direction, mask, *self.kernel_weights(), eps=self.eps, chunk_size=self.chunk_size
        )
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/fold/test_trimul.py -v`
Expected: all PASS.

- [ ] **Step 6: Lint, type-check, commit**

Run: `.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold tests/fold && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: clean.

```bash
git add pyproject.toml THIRD_PARTY_NOTICES.md src/oplm/fold/__init__.py src/oplm/fold/trimul.py tests/fold/__init__.py tests/fold/test_trimul.py
git commit -m "feat(fold): add the staged triangle-multiplication reference ported from ESMFold2"
```

---

### Task 4: Explicit trimul dispatch and the fused-forward/reference-backward boundary

**Files:**
- Modify: `src/oplm/fold/trimul.py`
- Test: `tests/fold/test_trimul.py` (append)

**Interfaces:**
- Consumes: everything from Task 3.
- Produces: `Backend = Literal["auto", "fused", "fused_forward_reference_backward", "reference"]`,
  `TriMulPath = Literal["fused_autograd", "fused_forward_reference_backward", "reference"]`,
  `cueq_available() -> bool`, `resolve_trimul_path(z, *, needs_grad, backend) -> TriMulPath`,
  `fused_trimul(z, direction, mask, *weights, eps) -> Tensor`,
  `trimul_mixed(z, direction, mask, *weights, eps, chunk_size, fused_fn) -> Tensor`,
  `TriangleMultiplication(..., backend: Backend = "auto")` with `.backend` settable.
  M1's `FoldConfig` carries `trimul_backend`; Task 6's bench maps path names to backends.

- [ ] **Step 1: Append the failing tests**

Add these imports to the top of `tests/fold/test_trimul.py` (merge into the existing
import block; ruff's isort rule orders them):

```python
import copy

from torch.utils.checkpoint import checkpoint

from oplm.fold import trimul as trimul_module
from oplm.fold.trimul import cueq_available, resolve_trimul_path, trimul_mixed
```

Then append:

```python
# --- dispatch and the mixed fused-forward/reference-backward path ---------------------


def _grads(m: TriangleMultiplication, z: torch.Tensor) -> dict[str, torch.Tensor]:
    grads = {name: p.grad.clone() for name, p in m.named_parameters() if p.grad is not None}
    assert z.grad is not None
    grads["z"] = z.grad.clone()
    m.zero_grad()
    z.grad = None
    return grads


def test_resolve_path_is_reference_on_cpu_regardless_of_backend() -> None:
    z = torch.randn(1, 4, 4, 32)
    for backend in ("auto", "fused", "fused_forward_reference_backward", "reference"):
        assert resolve_trimul_path(z, needs_grad=True, backend=backend) == "reference"
        assert resolve_trimul_path(z, needs_grad=False, backend=backend) == "reference"
    assert isinstance(cueq_available(), bool)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_mixed_path_matches_reference_gradients(direction: str) -> None:
    m = _module(direction)
    z, mask = _inputs(n=6)
    z.requires_grad_(True)
    calls = {"fused": 0}

    def fake_fused(
        z_: torch.Tensor, direction_: str, mask_: torch.Tensor | None, *weights: torch.Tensor, eps: float
    ) -> torch.Tensor:
        calls["fused"] += 1
        assert not torch.is_grad_enabled()  # the fused forward never builds a graph
        return trimul_reference(z_, direction_, mask_, *weights, eps=eps, chunk_size=None)

    out = trimul_mixed(
        z, direction, mask, *m.kernel_weights(), eps=m.eps, chunk_size=2, fused_fn=fake_fused
    )
    out.square().sum().backward()
    mixed = _grads(m, z)
    assert calls["fused"] == 1

    m(z, mask).square().sum().backward()
    reference = _grads(m, z)
    assert mixed.keys() == reference.keys()
    for name in reference:
        torch.testing.assert_close(mixed[name], reference[name], rtol=1e-5, atol=1e-6)


def test_mixed_path_under_checkpoint_recomputes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _module("outgoing")
    z, mask = _inputs(n=6)
    z.requires_grad_(True)
    calls = {"fused": 0, "reference": 0}
    real_reference = trimul_module.trimul_reference

    def counting_reference(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls["reference"] += 1
        return real_reference(*args, **kwargs)

    def fake_fused(
        z_: torch.Tensor, direction_: str, mask_: torch.Tensor | None, *weights: torch.Tensor, eps: float
    ) -> torch.Tensor:
        calls["fused"] += 1
        return real_reference(z_, direction_, mask_, *weights, eps=eps, chunk_size=None)

    monkeypatch.setattr(trimul_module, "trimul_reference", counting_reference)

    def block(z_: torch.Tensor) -> torch.Tensor:
        return trimul_mixed(
            z_, "outgoing", mask, *m.kernel_weights(), eps=m.eps, chunk_size=2, fused_fn=fake_fused
        )

    checkpoint(block, z, use_reentrant=False).square().sum().backward()
    assert calls == {"fused": 2, "reference": 1}  # forward + recompute; one backward pass
    mixed = _grads(m, z)

    monkeypatch.setattr(trimul_module, "trimul_reference", real_reference)
    m(z, mask).square().sum().backward()
    reference = _grads(m, z)
    for name in reference:
        torch.testing.assert_close(mixed[name], reference[name], rtol=1e-5, atol=1e-6)


def test_module_backend_knob_validates() -> None:
    with pytest.raises(ValueError, match="backend"):
        TriangleMultiplication(32, "outgoing", backend="triton")  # type: ignore[arg-type]
    assert TriangleMultiplication(32, "outgoing", backend="reference").backend == "reference"


# --- GPU parity (slow; skipped without CUDA + cuEquivariance) ---------------------------

_requires_cueq = pytest.mark.skipif(
    not (torch.cuda.is_available() and cueq_available()),
    reason="needs CUDA and the `fold` extra (cuequivariance-torch)",
)
# Tolerance policy (spec §9): the fused bf16 error against the fp32 reference may be at
# most 4x the bf16 *reference* error against the same fp32 oracle, plus a small floor.
_ERR_MULT = 4.0
_ERR_FLOOR = 1e-2


@pytest.mark.slow
@_requires_cueq
@pytest.mark.parametrize("width", [128, 256])
@pytest.mark.parametrize("length", [128, 512, 1024])
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("backend", ["fused", "fused_forward_reference_backward"])
def test_fused_paths_match_fp32_reference_within_bf16_tolerance(
    width: int, length: int, direction: str, backend: str
) -> None:
    torch.manual_seed(0)
    ref32 = TriangleMultiplication(width, direction, backend="reference").cuda()
    z32 = torch.randn(1, length, length, width, device="cuda", requires_grad=True)
    mask = (torch.rand(1, length, length, device="cuda") > 0.2).float()
    ref32(z32, mask).float().square().mean().backward()
    expected_out = ref32(z32, mask).detach()
    expected = _grads(ref32, z32)

    def run(backend_: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """bf16 forward/backward on a copy of ref32 under ``backend_``; returns fp32 views."""
        m = copy.deepcopy(ref32).bfloat16()
        m.backend = backend_
        z = z32.detach().bfloat16().requires_grad_(True)
        out = m(z, mask)
        out.float().square().mean().backward()
        return out.detach().float(), {k: v.float() for k, v in _grads(m, z).items()}

    ref_bf16_out, ref_bf16 = run("reference")  # the bf16 *reference* sets the tolerance
    assert resolve_trimul_path(z32.bfloat16(), needs_grad=True, backend=backend) != "reference"
    fused_out, fused = run(backend)

    baseline = (ref_bf16_out - expected_out).abs().max().item()
    assert (fused_out - expected_out).abs().max().item() <= _ERR_MULT * baseline + _ERR_FLOOR
    for name in expected:
        baseline = (ref_bf16[name] - expected[name]).abs().max().item()
        err = (fused[name] - expected[name]).abs().max().item()
        assert err <= _ERR_MULT * baseline + _ERR_FLOOR, (name, err, baseline)
```

`Any` is needed by `counting_reference`: add `from typing import Any` to the file's imports.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/fold/test_trimul.py -v -k "mixed or resolve or backend_knob"`
Expected: FAIL with `ImportError: cannot import name 'cueq_available'`.

- [ ] **Step 3: Add dispatch to `src/oplm/fold/trimul.py`**

Add after the `_EINSUM` constant:

```python
Backend = Literal["auto", "fused", "fused_forward_reference_backward", "reference"]
TriMulPath = Literal["fused_autograd", "fused_forward_reference_backward", "reference"]
_VALID_BACKENDS: tuple[str, ...] = ("auto", "fused", "fused_forward_reference_backward", "reference")
_FUSED_DTYPES = (torch.float32, torch.bfloat16, torch.float16)

try:
    from cuequivariance_torch import triangle_multiplicative_update as _cueq_trimul
except ImportError:  # the `fold` extra is optional; CPU/test environments never have it
    _cueq_trimul = None


def cueq_available() -> bool:
    """Whether ``cuequivariance_torch.triangle_multiplicative_update`` is importable."""
    return _cueq_trimul is not None


def resolve_trimul_path(z: Tensor, *, needs_grad: bool, backend: Backend) -> TriMulPath:
    """Pick the execution path for one call (design §5.2).

    ``"reference"`` whenever cuEquivariance is absent, ``z`` is not on CUDA, the width is
    not a multiple of 32, the dtype is unsupported, or the backend says so. Otherwise a
    call with no gradient requirement uses the fused forward directly; a gradient call
    uses the library's autograd unless ``backend`` selects the mixed path. ``"auto"``
    trusts the library's autograd until ``oplm fold bench-kernels`` says otherwise --
    the recorded choice then goes into the stage config.
    """
    if backend == "reference" or not cueq_available() or not z.is_cuda:
        return "reference"
    if z.shape[-1] % 32 or z.dtype not in _FUSED_DTYPES:
        return "reference"
    if not needs_grad or backend == "fused":
        return "fused_autograd"
    if backend == "fused_forward_reference_backward":
        return "fused_forward_reference_backward"
    return "fused_autograd"


def fused_trimul(
    z: Tensor, direction: Direction, mask: Tensor | None, *weights: Tensor, eps: float = TRIMUL_EPS
) -> Tensor:
    """cuEquivariance ``triangle_multiplicative_update`` with the port's weight mapping."""
    if _cueq_trimul is None:
        raise RuntimeError("cuequivariance_torch is not installed; pip install 'oplm[fold]'")
    (
        norm_in_weight,
        norm_in_bias,
        p_in_weight,
        g_in_weight,
        norm_out_weight,
        norm_out_bias,
        p_out_weight,
        g_out_weight,
    ) = weights
    return _cueq_trimul(
        z,
        direction=direction,
        mask=None if mask is None else mask.to(z.dtype),
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        p_in_weight=p_in_weight,
        g_in_weight=g_in_weight,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        g_out_weight=g_out_weight,
        eps=eps,
    )


class _FusedForwardReferenceBackward(torch.autograd.Function):
    """Fused inference kernel forward; the staged reference recomputed in backward.

    One recomputation boundary: the forward saves only its inputs (pair block, mask,
    weights -- what an activation checkpoint saves anyway); the backward rebuilds the
    staged reference graph under ``enable_grad`` and backpropagates through it, so
    reference activations exist only inside the backward call. Composes with
    ``torch.utils.checkpoint``: the outer checkpoint re-runs the cheap fused forward,
    then this backward runs the reference exactly once.
    """

    @staticmethod
    def forward(ctx: Any, z: Tensor, mask: Tensor | None, direction: Direction, eps: float, chunk_size: int | None, fused_fn: Callable[..., Tensor], *weights: Tensor) -> Tensor:
        ctx.direction, ctx.eps, ctx.chunk_size = direction, eps, chunk_size
        ctx.mask = mask  # not differentiated; may be None
        ctx.save_for_backward(z, *weights)
        with torch.no_grad():
            return fused_fn(z, direction, mask, *weights, eps=eps)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor | None, ...]:
        z, *weights = ctx.saved_tensors
        needs = ctx.needs_input_grad
        with torch.enable_grad():
            z_live = z.detach().requires_grad_(needs[0])
            weights_live = [
                w.detach().requires_grad_(need) for w, need in zip(weights, needs[6:], strict=True)
            ]
            out = trimul_reference(
                z_live, ctx.direction, ctx.mask, *weights_live, eps=ctx.eps, chunk_size=ctx.chunk_size
            )
            inputs = [t for t in (z_live, *weights_live) if t.requires_grad]
            grads = iter(torch.autograd.grad(out, inputs, grad_out, allow_unused=True))
        dz = next(grads) if z_live.requires_grad else None
        dweights = [next(grads) if w.requires_grad else None for w in weights_live]
        return (dz, None, None, None, None, None, *dweights)


def trimul_mixed(
    z: Tensor,
    direction: Direction,
    mask: Tensor | None,
    *weights: Tensor,
    eps: float = TRIMUL_EPS,
    chunk_size: int | None = 64,
    fused_fn: Callable[..., Tensor] = fused_trimul,
) -> Tensor:
    """Fused forward, reference backward. ``fused_fn`` is injectable so CPU tests can drive it."""
    return _FusedForwardReferenceBackward.apply(z, mask, direction, eps, chunk_size, fused_fn, *weights)
```

(`from collections.abc import Callable` goes in the `TYPE_CHECKING` block and `Any` joins
the `typing` import; `backward` references the module-level name `trimul_reference` so
tests can count calls.)

Add `backend` to the module and replace its `forward`:

```python
    def __init__(
        self,
        width: int,
        direction: Direction,
        *,
        eps: float = TRIMUL_EPS,
        chunk_size: int | None = 64,
        backend: Backend = "auto",
    ) -> None:
        super().__init__()
        if direction not in _EINSUM:
            raise ValueError(f"direction must be 'outgoing' or 'incoming', got {direction!r}")
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}")
        self.width = width
        self.direction: Direction = direction
        self.eps = eps
        self.chunk_size = chunk_size
        self.backend: Backend = backend
        ...  # the five holders exactly as in Task 3

    def forward(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        weights = self.kernel_weights()
        needs_grad = torch.is_grad_enabled() and (
            z.requires_grad or any(w.requires_grad for w in weights)
        )
        path = resolve_trimul_path(z, needs_grad=needs_grad, backend=self.backend)
        if path == "reference":
            return trimul_reference(
                z, self.direction, mask, *weights, eps=self.eps, chunk_size=self.chunk_size
            )
        if path == "fused_autograd":
            return fused_trimul(z, self.direction, mask, *weights, eps=self.eps)
        return trimul_mixed(
            z, self.direction, mask, *weights, eps=self.eps, chunk_size=self.chunk_size
        )
```

Extend `__all__` with `"Backend"`, `"TriMulPath"`, `"cueq_available"`, `"fused_trimul"`,
`"resolve_trimul_path"`, `"trimul_mixed"`. If `ty` flags the `autograd.Function`
overrides, suppress with a specific `# ty: ignore[<rule>]` and a one-line reason, as
the codebase does at framework boundaries.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/fold/test_trimul.py -v`
Expected: CPU tests PASS; the GPU parity test reports SKIPPED with the cuEquivariance reason.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold tests/fold && VIRTUAL_ENV=.venv .venv/bin/ty check src/`

```bash
git add src/oplm/fold/trimul.py tests/fold/test_trimul.py
git commit -m "feat(fold): explicit trimul dispatch with a fused-forward/reference-backward boundary"
```

---

### Task 5: Attention primitives — pair-biased and sliding-window, FlexAttention plus dense oracle

**Files:**
- Create: `src/oplm/fold/attention.py`, `tests/fold/test_attention.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `AttentionBackend = Literal["auto", "dense", "flex"]`,
  `resolve_attention_backend(x, backend) -> Literal["dense", "flex"]`,
  `pair_biased_attention(q, k, v, bias, key_mask=None, *, backend="auto") -> Tensor` with
  `q, k, v: (B, H, N, d)`, `bias: (B, H, N, N)` additive, `key_mask: (B, N)` bool,
  `sliding_window_attention(q, k, v, valid, half_window, *, backend="auto") -> Tensor`
  with `valid: (B, N)` bool. M1's `AttentionPairBias` (projections, adaLN, gates) and
  `SWA3DRoPEAttention` (qkv, RMS qk-norm, 3D RoPE, gate) call these.

- [ ] **Step 1: Write the failing tests**

Create `tests/fold/test_attention.py`:

```python
"""Attention primitives: dense oracle vs the transcribed ESMFold2 formulations, masked-row
finiteness, gradients, flex/dense forward parity on CPU, and GPU gradient parity."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from oplm.fold.attention import (
    pair_biased_attention,
    resolve_attention_backend,
    sliding_window_attention,
)


def _qkv(b: int = 2, h: int = 2, n: int = 48, d: int = 16, seed: int = 0) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(b, h, n, d, generator=g) for _ in range(3))


def _upstream_pair_bias(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, key_mask: torch.Tensor | None
) -> torch.Tensor:
    """ESMFold2 ``AttentionPairBias`` reference branch in (B, N, H, d) layout, transcribed."""
    qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))  # (B, N, H, d)
    logits = torch.einsum("... i h d, ... j h d -> ... i j h", qh, kh) * q.shape[-1] ** -0.5
    logits = logits + bias.permute(0, 2, 3, 1)  # (B, i, j, H)
    if key_mask is not None:
        min_val = torch.finfo(logits.dtype).min
        logits = logits + torch.where(key_mask.bool()[:, None, :, None], 0.0, min_val)
    attn = torch.softmax(logits, dim=-2)
    return torch.einsum("... i j h, ... j h d -> ... i h d", attn, vh).transpose(1, 2)


def _upstream_sliding_window(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid: torch.Tensor, half_window: int
) -> torch.Tensor:
    """ESMFold2 ``SWA3DRoPEAttention`` non-flash branch, transcribed."""
    n = q.shape[2]
    rank = torch.cumsum(valid, dim=1) - 1
    within = (rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= half_window
    allowed = within & valid.unsqueeze(1) & valid.unsqueeze(2)
    allowed |= torch.eye(n, dtype=torch.bool)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed.unsqueeze(1))
    return out * valid[:, None, :, None]


def test_backend_resolution() -> None:
    cpu = torch.zeros(1)
    assert resolve_attention_backend(cpu, "auto") == "dense"
    assert resolve_attention_backend(cpu, "flex") == "flex"
    assert resolve_attention_backend(cpu, "dense") == "dense"


def test_pair_biased_dense_matches_upstream_formulation() -> None:
    q, k, v = _qkv()
    bias = torch.randn(2, 2, 48, 48)
    key_mask = torch.ones(2, 48, dtype=torch.bool)
    key_mask[0, 40:] = False
    out = pair_biased_attention(q, k, v, bias, key_mask, backend="dense")
    torch.testing.assert_close(out, _upstream_pair_bias(q, k, v, bias, key_mask), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        pair_biased_attention(q, k, v, bias, backend="dense"),
        _upstream_pair_bias(q, k, v, bias, None),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", ["dense", "flex"])
def test_pair_biased_zero_valid_keys_row_is_zero_and_finite(backend: str) -> None:
    q, k, v = _qkv()
    bias = torch.randn(2, 2, 48, 48)
    key_mask = torch.ones(2, 48, dtype=torch.bool)
    key_mask[1] = False  # batch row 1: no valid key at all
    out = pair_biased_attention(q, k, v, bias, key_mask, backend=backend)
    assert torch.isfinite(out).all()
    assert out[1].abs().max() == 0
    assert out[0].abs().max() > 0


def test_pair_biased_dense_gradients_reach_bias_q_k_v() -> None:
    q, k, v = (t.requires_grad_(True) for t in _qkv())
    bias = torch.randn(2, 2, 48, 48, requires_grad=True)
    key_mask = torch.ones(2, 48, dtype=torch.bool)
    key_mask[0, 30:] = False
    pair_biased_attention(q, k, v, bias, key_mask, backend="dense").square().sum().backward()
    for t in (q, k, v, bias):
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
    assert bias.grad[0, :, :, 30:].abs().max() == 0  # masked keys receive no bias gradient


def test_flex_forward_matches_dense_on_cpu() -> None:
    q, k, v = _qkv(n=64)
    bias = torch.randn(2, 2, 64, 64)
    key_mask = torch.ones(2, 64, dtype=torch.bool)
    key_mask[0, 50:] = False
    with torch.no_grad():  # FlexAttention has no CPU backward; forward parity only here
        torch.testing.assert_close(
            pair_biased_attention(q, k, v, bias, key_mask, backend="flex"),
            pair_biased_attention(q, k, v, bias, key_mask, backend="dense"),
            rtol=1e-4,
            atol=1e-4,
        )
        valid = torch.ones(2, 64, dtype=torch.bool)
        valid[0, 40:] = False
        valid[1, 10:20] = False
        torch.testing.assert_close(
            sliding_window_attention(q, k, v, valid, 4, backend="flex"),
            sliding_window_attention(q, k, v, valid, 4, backend="dense"),
            rtol=1e-4,
            atol=1e-4,
        )


def test_sliding_window_dense_matches_upstream_and_zeroes_invalid() -> None:
    q, k, v = _qkv()
    valid = torch.ones(2, 48, dtype=torch.bool)
    valid[0, 36:] = False
    valid[1, 5:9] = False
    out = sliding_window_attention(q, k, v, valid, 3, backend="dense")
    torch.testing.assert_close(out, _upstream_sliding_window(q, k, v, valid, 3), rtol=1e-5, atol=1e-5)
    assert out[0, :, 36:].abs().max() == 0
    assert out[1, :, 5:9].abs().max() == 0


def test_sliding_window_counts_in_valid_atom_rank_not_padded_index() -> None:
    """Invalid atoms between two valid ones do not consume window budget (reference-space window)."""
    q, k, v = _qkv(n=16)
    valid = torch.zeros(2, 16, dtype=torch.bool)
    valid[:, 0] = True
    valid[:, 10] = True  # ranks 0 and 1 -> within half_window=1 despite index gap 10
    base = sliding_window_attention(q, k, v, valid, 1, backend="dense")
    v2 = v.clone()
    v2[:, :, 10] += 1.0
    moved = sliding_window_attention(q, k, v2, valid, 1, backend="dense")
    assert not torch.allclose(base[:, :, 0], moved[:, :, 0])  # atom 0 attends to atom 10


# --- GPU gradient parity (slow; skipped without CUDA) ----------------------------------

_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
_ERR_MULT = 4.0
_ERR_FLOOR = 1e-2


@pytest.mark.slow
@_requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n", [256, 512])
def test_flex_matches_dense_forward_and_backward_on_gpu(dtype: torch.dtype, n: int) -> None:
    torch.manual_seed(0)
    q32, k32, v32 = (torch.randn(2, 4, n, 32, device="cuda") for _ in range(3))
    bias32 = torch.randn(2, 4, n, n, device="cuda")
    key_mask = torch.ones(2, n, dtype=torch.bool, device="cuda")
    key_mask[0, n - 37 :] = False
    valid = key_mask.clone()
    valid[1, 100:140] = False

    def run(backend: str, dt: torch.dtype) -> list[torch.Tensor]:
        tensors = [t.detach().to(dt).requires_grad_(True) for t in (q32, k32, v32, bias32)]
        q, k, v, bias = tensors
        out = pair_biased_attention(q, k, v, bias, key_mask, backend=backend)
        out.float().square().sum().backward()
        grads = [t.grad.float() for t in tensors]
        tensors_sw = [t.detach().to(dt).requires_grad_(True) for t in (q32, k32, v32)]
        out_sw = sliding_window_attention(*tensors_sw, valid, 64, backend=backend)
        out_sw.float().square().sum().backward()
        return [out.float(), *grads, out_sw.float(), *[t.grad.float() for t in tensors_sw]]

    expected = run("dense", torch.float32)
    baseline = run("dense", dtype)
    flex = run("flex", dtype)
    for name, e, b, f in zip(
        ["out", "dq", "dk", "dv", "dbias", "out_sw", "dq_sw", "dk_sw", "dv_sw"], expected, baseline, flex, strict=True
    ):
        base_err = (b - e).abs().max().item()
        err = (f - e).abs().max().item()
        assert err <= _ERR_MULT * base_err + _ERR_FLOOR, (name, err, base_err)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/fold/test_attention.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.fold.attention'`.

- [ ] **Step 3: Create `src/oplm/fold/attention.py`**

```python
"""Pair-biased and sliding-window attention primitives: FlexAttention plus a dense oracle.

Semantics ported from Biohub's ESMFold2 ``AttentionPairBias`` (additive pair bias,
key padding mask) and ``SWA3DRoPEAttention`` (window measured in rank among valid
atoms, self always visible, invalid outputs zeroed) in
``esm/models/esmfold2/layers.py`` (Apache-2.0; see THIRD_PARTY_NOTICES.md).
Modifications: FlexAttention with block masks replaces dense-bias SDPA and the
FlashAttention dependency; the dense formulation is kept as the small-input oracle
and the CPU path (FlexAttention has no CPU backward); a batch row with no valid key
yields zeros rather than NaN. Q/K/V are never cast: the caller owns the precision
policy (design §5.3, §5.4).

Both functions take ``(B, H, N, d)`` tensors. Projections, gates, RoPE and
adaptive norms belong to the modules that call these (milestone 1).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Literal

import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor
    from torch.nn.attention.flex_attention import BlockMask

__all__ = [
    "AttentionBackend",
    "pair_biased_attention",
    "resolve_attention_backend",
    "sliding_window_attention",
]

AttentionBackend = Literal["auto", "dense", "flex"]


def resolve_attention_backend(x: Tensor, backend: AttentionBackend) -> Literal["dense", "flex"]:
    """``"auto"`` is flex on CUDA and dense elsewhere; explicit choices pass through."""
    if backend == "auto":
        return "flex" if x.is_cuda else "dense"
    return backend


@functools.cache
def _compiled_flex() -> Callable[..., Tensor]:
    # Uncompiled flex_attention materializes the full score matrix (it warns about it);
    # the fused kernel only exists through torch.compile. Compiled once per process.
    return torch.compile(flex_attention, dynamic=False)


def _flex(q: Tensor, k: Tensor, v: Tensor, score_mod: Callable[..., Tensor] | None, block_mask: BlockMask | None) -> Tensor:
    fn = _compiled_flex() if q.is_cuda else flex_attention  # CPU: eager, forward-only use
    return fn(q, k, v, score_mod=score_mod, block_mask=block_mask)


def pair_biased_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor,
    key_mask: Tensor | None = None,
    *,
    backend: AttentionBackend = "auto",
) -> Tensor:
    """``softmax_j(q_i.k_j / sqrt(d) + bias[b,h,i,j]) v_j`` over valid keys.

    Args:
        q: Queries ``(B, H, N, d)``.
        k: Keys ``(B, H, N, d)``.
        v: Values ``(B, H, N, d)``.
        bias: Additive pair bias ``(B, H, N, N)`` (any float dtype; read inside the kernel).
        key_mask: ``(B, N)`` bool, True for valid keys; ``None`` means all valid. A batch
            row with no valid key returns zeros.
        backend: ``"auto"``, ``"dense"`` or ``"flex"``.

    Returns:
        ``(B, H, N, d)`` in ``v``'s dtype.
    """
    if resolve_attention_backend(q, backend) == "dense":
        scale = q.shape[-1] ** -0.5
        logits = torch.matmul(q, k.transpose(-1, -2)).float() * scale + bias.float()
        if key_mask is not None:
            logits = logits.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        if key_mask is not None:  # all -inf rows softmax to NaN: define them as zero
            any_valid = key_mask.any(dim=-1)[:, None, None, None]
            attn = torch.where(any_valid, attn, torch.zeros_like(attn))
        return torch.matmul(attn.to(v.dtype), v)

    batch, _heads, n_q, _ = q.shape

    def score_mod(score: Tensor, b: Tensor, h: Tensor, qi: Tensor, ki: Tensor) -> Tensor:
        return score + bias[b, h, qi, ki]

    block_mask: BlockMask | None = None
    if key_mask is not None:

        def mask_mod(b: Tensor, h: Tensor, qi: Tensor, ki: Tensor) -> Tensor:
            return key_mask[b, ki]

        block_mask = create_block_mask(mask_mod, batch, None, n_q, k.shape[2], device=q.device)
    return _flex(q, k, v, score_mod, block_mask)


def sliding_window_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    valid: Tensor,
    half_window: int,
    *,
    backend: AttentionBackend = "auto",
) -> Tensor:
    """Local attention over the valid-atom rank with window ``[-half_window, half_window]``.

    ``allowed[b,i,j] = (valid[b,i] & valid[b,j] & |rank_i - rank_j| <= half_window) | (i == j)``
    with ``rank = cumsum(valid) - 1``; outputs at invalid positions are zero. This is
    ESMFold2's SDPA formulation and what its FlashAttention varlen path computes once
    padding is removed, so the window is measured in reference space, not padded index
    space. Self-attention is always allowed, so a padded query stays finite.

    Args:
        q: Queries ``(B, H, N, d)``.
        k: Keys ``(B, H, N, d)``.
        v: Values ``(B, H, N, d)``.
        valid: ``(B, N)`` bool atom validity.
        half_window: Window radius in valid-atom rank (upstream default 64 -> window 128).
        backend: ``"auto"``, ``"dense"`` or ``"flex"``.
    """
    batch, _heads, n, _ = q.shape
    rank = torch.cumsum(valid.to(torch.int64), dim=1) - 1
    if resolve_attention_backend(q, backend) == "dense":
        within = (rank[:, :, None] - rank[:, None, :]).abs() <= half_window
        allowed = within & valid[:, :, None] & valid[:, None, :]
        allowed = allowed | torch.eye(n, dtype=torch.bool, device=q.device)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None])
    else:

        def mask_mod(b: Tensor, h: Tensor, qi: Tensor, ki: Tensor) -> Tensor:
            in_window = (rank[b, qi] - rank[b, ki]).abs() <= half_window
            return (valid[b, qi] & valid[b, ki] & in_window) | (qi == ki)

        block_mask = create_block_mask(mask_mod, batch, None, n, n, device=q.device)
        out = _flex(q, k, v, None, block_mask)
    return out * valid[:, None, :, None].to(out.dtype)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/fold/test_attention.py -v`
Expected: CPU tests PASS (the eager CPU flex path prints a one-time "called without
torch.compile" warning; that is expected); GPU test SKIPPED.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold tests/fold && VIRTUAL_ENV=.venv .venv/bin/ty check src/`

```bash
git add src/oplm/fold/attention.py tests/fold/test_attention.py
git commit -m "feat(fold): pair-biased and sliding-window attention with FlexAttention and a dense oracle"
```

---

### Task 6: `oplm fold bench-kernels`, CLI registration, and docs

**Files:**
- Create: `src/oplm/fold/cli.py`, `tests/fold/test_cli.py`, `docs/FOLD.md`
- Modify: `src/oplm/cli.py` (imports L12–19 and `add_typer` lines), `AGENTS.md` (project structure tree), `docs/TESTING_E2E.md` (§1 table)

**Interfaces:**
- Consumes: `TriangleMultiplication`, `resolve_trimul_path` (Task 4).
- Produces: `run_kernel_benchmark(*, widths, lengths, directions, paths, device, dtype, iters, warmup, chunk_size, compile_reference=False, error_max_length=512) -> dict[str, Any]`
  and the `oplm fold bench-kernels` command. Report schema: `{"environment": {...}, "settings": {...}, "cases": [{"width", "length", "direction", "path", "actual_path", "status", "compiled", "forward", "forward_backward", "checkpointed", "max_abs_error_vs_fp32_reference"?}]}`
  where each timing entry is `{"status": "ok", "ms_per_iter", "peak_allocated_gib"?, "peak_reserved_gib"?}` or `{"status": "error", "error"}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/fold/test_cli.py`:

```python
"""`oplm fold` command group: registration and a tiny CPU run of bench-kernels."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from oplm.cli import app
from oplm.fold.cli import run_kernel_benchmark
from tests.cli_output import plain

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def test_fold_help_lists_bench_kernels() -> None:
    result = runner.invoke(app, ["fold", "--help"])
    assert result.exit_code == 0, result.output
    assert "bench-kernels" in plain(result.output)


def test_run_kernel_benchmark_cpu_reference_and_unavailable_fused() -> None:
    report = run_kernel_benchmark(
        widths=[32],
        lengths=[16],
        directions=["outgoing", "incoming"],
        paths=["reference", "fused_autograd"],
        device="cpu",
        dtype="fp32",
        iters=1,
        warmup=0,
        chunk_size=8,
    )
    assert report["environment"]["torch"] and report["environment"]["device"] == "cpu"
    assert report["settings"]["compile_reference"] is False
    by_key = {(c["direction"], c["path"]): c for c in report["cases"]}
    assert len(by_key) == 4
    for direction in ("outgoing", "incoming"):
        ref = by_key[(direction, "reference")]
        assert ref["status"] == "ok" and ref["actual_path"] == "reference"
        assert ref["compiled"] is False
        for phase in ("forward", "forward_backward", "checkpointed"):
            assert ref[phase]["status"] == "ok" and ref[phase]["ms_per_iter"] > 0
        assert ref["max_abs_error_vs_fp32_reference"] == 0.0
        fused = by_key[(direction, "fused_autograd")]
        assert fused["status"] == "unavailable" and fused["actual_path"] == "reference"


def test_bench_kernels_command_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    result = runner.invoke(
        app,
        [
            "fold", "bench-kernels", "--device", "cpu", "--dtype", "fp32", "--widths", "32",
            "--lengths", "16", "--paths", "reference", "--iters", "1", "--warmup", "0",
            "--out", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["cases"][0]["status"] == "ok"
    assert "reference" in plain(result.output)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/fold/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.fold.cli'`.

- [ ] **Step 3: Create `src/oplm/fold/cli.py`**

```python
"""``oplm fold`` command group. Milestone 0 ships ``bench-kernels`` (design §5.2).

Keep torch out of module scope: ``oplm.cli`` imports this module eagerly.
"""

from __future__ import annotations

import itertools
import json
import platform
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from oplm.fold.trimul import Backend

app = typer.Typer(name="fold", help="Structure prediction head", add_completion=False)
console = Console()

_PATH_TO_BACKEND = {
    "reference": "reference",
    "fused_autograd": "fused",
    "fused_forward_reference_backward": "fused_forward_reference_backward",
}
_DTYPES = {"fp32": "float32", "bf16": "bfloat16"}


def _environment(device: torch.device) -> dict[str, Any]:
    import torch

    try:
        cueq_version: str | None = metadata.version("cuequivariance-torch")
    except metadata.PackageNotFoundError:
        cueq_version = None
    try:
        git_rev: str | None = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_rev = None
    return {
        "host": platform.node(),
        "device": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuequivariance_torch": cueq_version,
        "git": git_rev,
    }


def _time(fn: Callable[[], None], device: torch.device, iters: int, warmup: int) -> dict[str, Any]:
    import torch

    try:
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result: dict[str, Any] = {
            "status": "ok",
            "ms_per_iter": (time.perf_counter() - start) * 1000.0 / iters,
        }
        if device.type == "cuda":
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            result["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
        return result
    except Exception as exc:  # the report records per-path failures instead of aborting
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def run_kernel_benchmark(
    *,
    widths: list[int],
    lengths: list[int],
    directions: list[str],
    paths: list[str],
    device: str,
    dtype: str,
    iters: int,
    warmup: int,
    chunk_size: int,
    compile_reference: bool = False,
    error_max_length: int = 512,
) -> dict[str, Any]:
    """Time forward, forward+backward and checkpointed forward+backward per (width, length, direction, path).

    A case whose requested path does not resolve on this machine (no CUDA, no
    cuEquivariance, unsupported width) is reported as ``"unavailable"`` with the path that
    would actually run. ``compile_reference`` wraps the ``reference`` path's module in
    ``torch.compile`` (design §5.2 path 3: the compiled reference). Numerical error against
    an fp32 reference is recorded for lengths up to ``error_max_length``.
    """
    import copy

    import torch
    from torch.utils.checkpoint import checkpoint

    from oplm.fold.trimul import TriangleMultiplication, resolve_trimul_path

    dev = torch.device(device)
    torch_dtype = getattr(torch, _DTYPES[dtype])
    report: dict[str, Any] = {
        "environment": _environment(dev),
        "settings": {
            "dtype": dtype,
            "iters": iters,
            "warmup": warmup,
            "chunk_size": chunk_size,
            "compile_reference": compile_reference,
        },
        "cases": [],
    }
    for width, length, direction, path in itertools.product(widths, lengths, directions, paths):
        case: dict[str, Any] = {
            "width": width, "length": length, "direction": direction, "path": path,
        }
        torch.manual_seed(0)
        backend = cast("Backend", _PATH_TO_BACKEND[path])
        module = TriangleMultiplication(
            width, direction, chunk_size=chunk_size, backend=backend
        ).to(dev, torch_dtype)
        z = torch.randn(1, length, length, width, device=dev, dtype=torch_dtype, requires_grad=True)
        mask = torch.ones(1, length, length, device=dev)
        actual = resolve_trimul_path(z, needs_grad=True, backend=module.backend)
        case["actual_path"] = actual
        if actual != path:
            case["status"] = "unavailable"
            report["cases"].append(case)
            continue
        case["status"] = "ok"
        case["compiled"] = compile_reference and path == "reference"
        runner: Callable[..., torch.Tensor] = module
        if case["compiled"]:
            runner = torch.compile(module, dynamic=False)

        def forward() -> None:
            with torch.no_grad():
                runner(z, mask)

        def forward_backward() -> None:
            runner(z, mask).float().sum().backward()
            z.grad = None

        def checkpointed() -> None:
            checkpoint(runner, z, mask, use_reentrant=False).float().sum().backward()
            z.grad = None

        for name, fn in (
            ("forward", forward),
            ("forward_backward", forward_backward),
            ("checkpointed", checkpointed),
        ):
            case[name] = _time(fn, dev, iters, warmup)
        if length <= error_max_length:
            with torch.no_grad():
                reference = copy.deepcopy(module).float()
                reference.backend = "reference"
                expected = reference(z.detach().float(), mask)
                case["max_abs_error_vs_fp32_reference"] = float(
                    (module(z, mask).float() - expected).abs().max()
                )
        report["cases"].append(case)
    return report


def _ints(csv: str) -> list[int]:
    return [int(x) for x in csv.split(",") if x]


def _strs(csv: str) -> list[str]:
    return [x.strip() for x in csv.split(",") if x.strip()]


def _ms(entry: dict[str, Any] | None) -> str:
    if not entry:
        return "-"
    return f"{entry['ms_per_iter']:.1f}" if entry["status"] == "ok" else "err"


@app.command("bench-kernels")
def bench_kernels(
    out: Annotated[Path, typer.Option(help="JSON report path")] = Path("bench-kernels.json"),
    widths: Annotated[str, typer.Option(help="Comma-separated pair widths")] = "128,256",
    lengths: Annotated[str, typer.Option(help="Comma-separated token lengths")] = "384,768,1024,1536,2048",
    directions: Annotated[str, typer.Option()] = "outgoing,incoming",
    paths: Annotated[str, typer.Option(help="reference | fused_autograd | fused_forward_reference_backward")] = "reference,fused_autograd,fused_forward_reference_backward",
    device: Annotated[str, typer.Option()] = "cuda",
    dtype: Annotated[str, typer.Option(help="fp32 | bf16")] = "bf16",
    iters: Annotated[int, typer.Option()] = 5,
    warmup: Annotated[int, typer.Option()] = 2,
    chunk_size: Annotated[int, typer.Option(help="Reference contraction row chunk")] = 64,
    compile_reference: Annotated[
        bool, typer.Option("--compile-reference", help="torch.compile the reference path")
    ] = False,
) -> None:
    """Time trimul paths and record peak memory, dispatch, versions, and numerical error."""
    unknown = set(_strs(paths)) - set(_PATH_TO_BACKEND)
    if unknown:
        raise typer.BadParameter(f"unknown paths {sorted(unknown)}; choose from {sorted(_PATH_TO_BACKEND)}")
    if dtype not in _DTYPES:
        raise typer.BadParameter(f"dtype must be one of {sorted(_DTYPES)}")
    report = run_kernel_benchmark(
        widths=_ints(widths),
        lengths=_ints(lengths),
        directions=_strs(directions),
        paths=_strs(paths),
        device=device,
        dtype=dtype,
        iters=iters,
        warmup=warmup,
        chunk_size=chunk_size,
        compile_reference=compile_reference,
    )
    out.write_text(json.dumps(report, indent=2))

    table = Table(title=f"trimul kernels ({report['environment']['device_name'] or device}, {dtype})")
    for column in ("width", "length", "dir", "path", "status", "fwd ms", "fwd+bwd ms", "ckpt ms", "peak GiB", "max err"):
        table.add_column(column)
    for case in report["cases"]:
        peak = case.get("forward_backward", {}).get("peak_allocated_gib")
        path_label = case["path"] + (" (compiled)" if case.get("compiled") else "")
        table.add_row(
            str(case["width"]), str(case["length"]), case["direction"], path_label, case["status"],
            _ms(case.get("forward")), _ms(case.get("forward_backward")), _ms(case.get("checkpointed")),
            f"{peak:.2f}" if peak is not None else "-",
            f"{case['max_abs_error_vs_fp32_reference']:.2e}" if "max_abs_error_vs_fp32_reference" in case else "-",
        )
    console.print(table)
    console.print(f"Wrote {out}")
```

- [ ] **Step 4: Register the sub-app**

In `src/oplm/cli.py` add `from oplm.fold.cli import app as fold_app` next to the other
sub-app imports and `app.add_typer(fold_app, name="fold", help="Structure prediction head")`
after the `slurm` registration.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/fold/test_cli.py tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 6: Write `docs/FOLD.md`**

(The outer fence is four backticks because the document contains its own fenced blocks.)

````markdown
# Structure Prediction Head (`oplm.fold`)

Architecture contract for the folding head, in the role `MODEL_ARCHITECTURE.md`
plays for the language model. The agreed design is
[`docs/superpowers/specs/2026-10-07-structure-prediction-head-design.md`](superpowers/specs/2026-10-07-structure-prediction-head-design.md);
this file records what is implemented. **Status: milestone 0 (foundations and
kernels).** Model, data, losses, and training land in later milestones.

## 1. Package layout (implemented so far)

```
src/oplm/fold/
├── trimul.py      # staged triangle multiplication + cuEquivariance dispatch
├── attention.py   # pair-biased / sliding-window attention: FlexAttention + dense oracle
└── cli.py         # oplm fold bench-kernels
tests/fold/        # mirrors the above; GPU parity tests are `slow` and skip without CUDA
```

Dependency direction: `oplm.fold` imports core packages; core imports nothing
from `oplm.fold` except the CLI registration in `oplm/cli.py`. Ported modules
name their Biohub source in the module docstring; `THIRD_PARTY_NOTICES.md`
carries the Apache 2.0 text.

## 2. Triangle multiplication (`oplm.fold.trimul`)

Parameters match ESMFold2's `TriangleMultiplicativeBlock` exactly:
`norm_start`, `norm_mix` (LayerNorm, eps 1e-5), `proj_bundle` (`D -> 4D`, rows
`[0:2D]` signal / `[2D:4D]` gate logits), `proj_emit`, `proj_gate` (no biases).
The forward is three explicit stages:

1. `trimul_pre` (pointwise): fp32-internal LayerNorm, signal and gate projections,
   sigmoid gate, pair mask → `left`, `right`, `zn`.
2. `trimul_contract` (the only cross-row/column op): outgoing
   `out[i,j] = Σ_k left[i,k]·right[j,k]`, incoming `out[i,j] = Σ_k left[k,i]·right[k,j]`,
   fp32 accumulation, chunked over output rows (`chunk_size`, default 64).
3. `trimul_post` (pointwise): fp32-internal LayerNorm, `proj_emit`, output gate.

Stages 1 and 3 are block-local by construction (tested on random sub-blocks), which
is the discipline later Fold-CP sharding relies on (spec §5.6).

Dispatch (`resolve_trimul_path`) is explicit, per call, from device/dtype/width/grad:

| Path | When | Backward |
| --- | --- | --- |
| `reference` | CPU, no cuEquivariance, width not a multiple of 32, or `backend="reference"` | autograd through the staged reference |
| `fused_autograd` | CUDA + cuEquivariance, `backend` in `auto`/`fused`, or any no-grad call | the library's own backward |
| `fused_forward_reference_backward` | `backend="fused_forward_reference_backward"` | fused forward saved as a single recomputation boundary; the reference graph is rebuilt inside `backward` and freed after it |

`TriangleMultiplication.backend` is the knob; `"auto"` trusts the library's
autograd. Choose per width from the benchmark below and record the choice in the
stage config. Residual add and row-shared dropout belong to the owning pair
block (milestone 1), as upstream.

## 3. Attention primitives (`oplm.fold.attention`)

- `pair_biased_attention(q, k, v, bias, key_mask)`: logits `q·k/√d + bias[b,h,i,j]`,
  keys masked by `key_mask`; a batch row with no valid key returns zeros.
- `sliding_window_attention(q, k, v, valid, half_window)`: window measured in rank
  among *valid* atoms (`rank = cumsum(valid) - 1`), self always visible, invalid
  outputs zeroed — ESMFold2's reference-space semantics, identical to its
  FlashAttention varlen path.

`backend="auto"` is FlexAttention (compiled, block masks) on CUDA and the dense
formulation on CPU; FlexAttention has no CPU backward. Q/K/V dtypes are never
changed by these functions. Gradient parity (incl. the bias) is a slow GPU test.

## 4. Training seam and EMA

The Trainer builds the model/dataloader and runs a step through a `TrainTask`
(`docs/TRAIN.md` §17); `FoldTask` lands in milestone 2. `train.ema_decay` keeps an
EMA of the trainable weights, updated once per optimizer step, saved as `ema.pt` and
exported as `hf_ema/` with every checkpoint (`docs/TRAIN.md` §16). The frozen LM
will be an unregistered attribute of the fold model and must be excluded from the
EMA's deep copy (milestone 1).

## 5. Kernel benchmark

```bash
oplm fold bench-kernels --out bench-kernels.json            # B200 defaults: widths 128,256; lengths 384..2048
oplm fold bench-kernels --paths reference --compile-reference --out bench-reference-compiled.json
oplm fold bench-kernels --device cpu --dtype fp32 --widths 32 --lengths 16 --paths reference
```

Per (width, length, direction, path) the report records forward, forward+backward
and checkpointed forward+backward ms/iter, peak allocated/reserved GiB, the path
that actually ran (`"unavailable"` when the request cannot resolve on the machine),
whether the reference was `torch.compile`d, and the max abs error against an fp32
reference (lengths ≤ 512). The environment block pins host, GPU,
torch/CUDA/cuEquivariance versions and git revision. Results for the target B200
are recorded in §6 once measured.

## 6. Measured results

_Pending: filled by the milestone-0 acceptance run (plan Task 7)._
````

- [ ] **Step 7: Update `AGENTS.md` and `docs/TESTING_E2E.md`**

In `AGENTS.md`'s project-structure tree, before the `slurm/` line, add:

```
├── fold/                       # structure prediction head: kernels + bench CLI now, model/data/training later (docs/FOLD.md)
```

In `docs/TESTING_E2E.md` §1 table, after the `tests/data/` row:

```markdown
| `tests/fold/` | Structure-head kernels: triangle-multiplication reference vs the upstream equation, dispatch and the mixed fused/reference path, attention primitives, `bench-kernels`. GPU parity tests are `slow` and skip without CUDA (+ cuEquivariance). |
```

- [ ] **Step 8: Lint, type-check, docs links, commit**

Run: `.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold src/oplm/cli.py tests/fold && VIRTUAL_ENV=.venv .venv/bin/ty check src/ && .venv/bin/python -m pytest tests/test_docs_links.py tests/fold -q`
Expected: clean; links resolve.

```bash
git add src/oplm/fold/cli.py src/oplm/cli.py tests/fold/test_cli.py docs/FOLD.md AGENTS.md docs/TESTING_E2E.md
git commit -m "feat(fold): add oplm fold bench-kernels and the FOLD.md contract"
```

---

### Task 7: Milestone-0 acceptance on the B200

**Files:**
- Modify: `docs/FOLD.md` §6, `pyproject.toml` (`fold` extra comment with the tested versions)
- Create: `docs/fold/bench-kernels-b200.json` (the committed report)

**Interfaces:** consumes Tasks 1–6. This task is operational; it needs a B200 node
with the `fold` extra installed (`pip install -e ".[dev,train,fold]"`).

- [ ] **Step 1: Full regression on the dev machine**

Run: `.venv/bin/python -m pytest -q`
Expected: everything green; `tests/fold` GPU cases and the CUDA rows of the
training matrix report SKIPPED.

- [ ] **Step 2: GPU parity on the B200**

Run: `python -m pytest tests/fold -v -m slow`
Expected: `test_fused_paths_match_fp32_reference_within_bf16_tolerance` passes for
both widths and both backends, and `test_flex_matches_dense_forward_and_backward_on_gpu`
passes. If a `fused` (library autograd) case errors for a width, that is the
measurement the spec asks for: keep the failing case's output, and record in
`docs/FOLD.md` §2 that the width needs `backend="fused_forward_reference_backward"`.

- [ ] **Step 3: Benchmark**

Run: `oplm fold bench-kernels --out docs/fold/bench-kernels-b200.json` and
`oplm fold bench-kernels --paths reference --compile-reference --out docs/fold/bench-kernels-b200-compiled-reference.json`.
Then paste a table into `docs/FOLD.md` §6 with, per width and direction, the
forward / forward+backward / checkpointed ms at 2048 tokens, the peak GiB, the
path that ran (eager and compiled reference as separate rows), and the max abs
error at 512, plus the `environment` block. Add the exact tested
`cuequivariance-torch` / `cuequivariance-ops-torch-cu13` versions to the `fold`
extra's comment in `pyproject.toml`.

- [ ] **Step 4: MLM regression on GPU**

Run: `python -m pytest tests/training/test_e2e_logging.py tests/training/test_e2e_precision.py tests/training/test_e2e_ema.py -v`
Expected: PASS on the `cuda-bf16` rows (EMA under autocast, FLOP contract unchanged).

- [ ] **Step 5: Commit**

```bash
git add docs/FOLD.md docs/fold/ pyproject.toml
git commit -m "docs(fold): record B200 trimul benchmark and tested cuEquivariance versions"
```

Milestone 0 acceptance (spec §10): MLM regressions clean (Steps 1, 4); forward/backward
parity on B200 (Step 2); published width-128/256 cost and memory results (Step 3).

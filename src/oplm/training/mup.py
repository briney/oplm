"""μP (maximal update parametrization) helpers for the training side.

The single scaling quantity of the μP recipe is the per-matrix fan-in multiplier
``m_W = fan_in / fan_in_base``, computed by :meth:`OplmConfig.mup_fanin_mult` —
the model-package single source of truth (so width-aware init stays
``trust_remote_code``-safe). This module wraps that quantity for training:

* :func:`mup_fanin_mult` / :func:`mup_lr_multiplier` — consumed by ``optim.py`` to
  scale per-group AdamW learning rates (Phase 4);
* :func:`coord_check` — the empirical correctness gate (per-activation RMS vs
  width);
* :class:`SweepMetricsCallback` — the LR-cell metric utility.

See ``docs/MUP.md`` for the recipe table and the coord-check pass/fail oracle.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from oplm.training.callbacks import TrainerCallback

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import pandas as pd

    from oplm.model.configuration_oplm import OplmConfig
    from oplm.training.trainer import Trainer

__all__ = [
    "StabilityDiagnosticsCallback",
    "SweepMetricsCallback",
    "coord_check",
    "mup_fanin_mult",
    "mup_lr_multiplier",
]

# Parameter-name suffixes that identify μP readouts (width-independent LR, ×1):
# the untied LM decoder and the fine-tuning classifier heads. Their `.bias`
# counterparts are 1-D and already covered by the ndim guard in
# `mup_lr_multiplier`. Mirrors the `_is_readout` flags consumed by width-aware
# init (Phase 2), which dispatch by module flag since init gets no name.
_READOUT_WEIGHT_SUFFIXES = ("lm_head.decoder.weight", "classifier.weight")


# ----------------------------------------------------------------------------
# Per-matrix multipliers (single source of truth for init + LR scaling)
# ----------------------------------------------------------------------------


def mup_fanin_mult(module_or_param: nn.Linear | torch.Tensor, config: OplmConfig) -> float:
    """Per-matrix μP fan-in multiplier ``m_W`` for a Linear module or weight tensor.

    Training-side wrapper over :meth:`OplmConfig.mup_fanin_mult` (the model-package
    single source of truth): pulls the matrix fan-in off ``module_or_param`` and
    delegates. Returns ``1.0`` for non-2-D tensors (biases, norms, conv kernels)
    and whenever μP is disabled, so callers can divide by it unconditionally.

    Args:
        module_or_param: An ``nn.Linear`` (fan-in = ``in_features``) or a weight
            tensor (fan-in = ``shape[1]`` for a 2-D weight).
        config: The model config carrying the μP knobs.
    """
    if isinstance(module_or_param, nn.Linear):
        fan_in = module_or_param.in_features
    elif isinstance(module_or_param, torch.Tensor):
        if module_or_param.ndim != 2:
            return 1.0
        fan_in = module_or_param.shape[1]
    else:
        raise TypeError(
            f"mup_fanin_mult expects nn.Linear or torch.Tensor, got {type(module_or_param)!r}"
        )
    return config.mup_fanin_mult(fan_in)


def mup_lr_multiplier(name: str, param: torch.Tensor, config: OplmConfig) -> float:
    """Per-parameter μP learning-rate multiplier for the AdamW path.

    Hidden weight matrices take ``1 / m_W``: ``1/m`` for hidden-fan-in matrices
    (``q/k/v/o_proj``, ``gate/up_proj``, ``lm_head.dense.weight``) and ``1/m_ffn``
    for ``down_proj`` (fan-in = ``intermediate_size``). Embeddings (μP input
    weights), μP readouts (``lm_head.decoder.weight``, ``classifier.weight``),
    norms, biases, and every non-2-D parameter take ``1.0``. Returns ``1.0``
    wholesale when μP is disabled, so param-group assembly is identical to the
    non-μP path. Mirrors the role dispatch of width-aware init (Phase 2).

    Args:
        name: The parameter's dotted name from ``model.named_parameters()``.
        param: The parameter tensor (its ndim/fan-in select the role).
        config: The model config carrying the μP knobs.
    """
    if not config.mup_enable:
        return 1.0
    if param.ndim != 2 or "embed" in name or name.endswith(_READOUT_WEIGHT_SUFFIXES):
        return 1.0
    return 1.0 / mup_fanin_mult(param, config)


# ----------------------------------------------------------------------------
# Coordinate check
# ----------------------------------------------------------------------------


def _make_rms_hook(
    width: int,
    names: dict[nn.Module, str],
    records: list[dict[str, object]],
    step_ref: dict[str, int],
) -> Callable[[nn.Module, object, object], None]:
    """Build a forward hook recording a module's output RMS into ``records``.

    ``step_ref["t"]`` is read at call time so one hook serves every step; only
    plain-tensor outputs are recorded (tuple-returning modules are skipped).
    """

    def hook(module: nn.Module, _inputs: object, output: object) -> None:
        if not isinstance(output, torch.Tensor):
            return
        rms = output.detach().float().pow(2).mean().sqrt().item()
        records.append({"width": width, "module": names[module], "step": step_ref["t"], "rms": rms})

    return hook


def coord_check(
    build_cfg_fn: Callable[[int], OplmConfig],
    widths: Sequence[int],
    batch: Mapping[str, torch.Tensor],
    *,
    steps: int = 3,
    optimizer: str = "muon",
    seed: int = 0,
    lr: float = 1e-2,
    scaling: str = "width",
    device: torch.device | str | None = None,
) -> pd.DataFrame:
    """Run a μP coordinate check: per-activation RMS vs width across train steps.

    For each width, build a model via ``build_cfg_fn(width)``, attach forward
    hooks on every ``nn.Linear``/``nn.Embedding`` submodule, run ``steps``
    optimizer steps on the fixed ``batch`` (which must carry ``labels``), and
    record each submodule's output RMS (``x.pow(2).mean().sqrt()``) at steps
    ``t = 0..steps``. ``t=0`` is the pre-step init state and is kept so the
    oracle can exclude it for the readout (allowed to shrink ``Θ(1/√m)`` at init).

    The pass/fail oracle is **one-sided**: with μP on, no module's RMS should grow
    with width; the ``--no-mup`` control fans out. See ``docs/MUP.md`` and
    ``scripts/mup_coord_check.py``.

    Args:
        build_cfg_fn: Maps a hidden width to a fully-formed ``OplmConfig``. The
            caller bakes in depth, μP settings, and ``head_dim`` — and, for the
            preset-ray mode, the depth-vs-width relationship.
        widths: Hidden sizes to sweep.
        batch: A fixed input batch (``input_ids``/``attention_mask``/``labels``),
            reused at every width and step.
        steps: Number of optimizer steps (and ``steps + 1`` forwards).
        optimizer: ``"muon"`` (uses the μP-correct ``adjust_lr_fn="original"``) or
            ``"adamw"``.
        seed: Re-seeded before each width so init is comparable across widths.
        lr: Base learning rate for the throwaway sweep optimizer.
        scaling: ``"width"`` (the μP gate: width varies, depth fixed) or
            ``"preset_ray"`` (width+depth co-scaled). Recorded in ``df.attrs``; the
            actual config geometry is owned by ``build_cfg_fn``.
        device: Where to run; defaults to CPU.

    Returns:
        A tidy frame with columns ``(width, module, step, rms)`` and
        ``df.attrs["scaling"]`` set.
    """
    import pandas as pd

    from oplm.config import TrainConfig
    from oplm.model import OplmForMaskedLM
    from oplm.training.optim import build_optimizers

    if optimizer not in ("muon", "adamw"):
        raise ValueError(f"optimizer must be 'muon' or 'adamw'; got {optimizer!r}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1; got {steps}")

    run_device = torch.device(device) if device is not None else torch.device("cpu")
    records: list[dict[str, object]] = []

    for width in widths:
        torch.manual_seed(seed)
        model = OplmForMaskedLM(build_cfg_fn(width)).to(run_device)
        model.train()

        # μP+Muon requires the aspect-ratio-only "original" adjust-lr fn (the
        # default "match_rms_adamw" grows like √width and does not transfer).
        train_cfg = TrainConfig(
            optimizer=optimizer,
            lr=lr,
            weight_decay=0.0,
            warmup_steps=0,
            max_steps=steps,
            muon_adjust_lr_fn="original",
            wandb_enabled=False,
        )
        optimizers = build_optimizers(model, train_cfg)

        names = {module: name for name, module in model.named_modules()}
        step_ref = {"t": 0}
        handles = [
            module.register_forward_hook(_make_rms_hook(width, names, records, step_ref))
            for module in model.modules()
            if isinstance(module, nn.Linear | nn.Embedding)
        ]
        batch_on_device = {k: v.to(run_device) for k, v in batch.items()}

        try:
            for t in range(steps + 1):
                step_ref["t"] = t
                outputs = model(**batch_on_device)
                if t == steps:
                    break
                if outputs.loss is None:
                    raise ValueError("coord_check requires a batch with 'labels' to compute a loss")
                outputs.loss.backward()
                for opt in optimizers:
                    opt.step()
                model.zero_grad(set_to_none=True)
        finally:
            for handle in handles:
                handle.remove()

    frame = pd.DataFrame.from_records(records, columns=["width", "module", "step", "rms"])
    frame.attrs["scaling"] = scaling
    return frame


# ----------------------------------------------------------------------------
# LR-sweep metric utilities
# ----------------------------------------------------------------------------


class SweepMetricsCallback(TrainerCallback):
    """Capture a sweep run's losses and write ``result.json`` at train end.

    ``trainer_state.json`` carries no loss, so the LR sweep relies on this
    callback. It EMA-smooths the train loss across ``on_log`` events, keeps the
    last value per ``eval/*`` key from ``on_eval_end``, and writes
    ``{final_train_loss, eval, lr, width, steps}`` on ``on_train_end``.
    """

    def __init__(self, path: str | Path, *, ema_decay: float = 0.9) -> None:
        self.path = Path(path)
        self.ema_decay = ema_decay
        self._ema_loss: float | None = None
        self._eval: dict[str, float] = {}

    def on_log(self, trainer: Trainer, metrics: dict[str, float], step: int) -> None:
        loss = metrics.get("train/loss")
        if loss is None:
            return
        loss = float(loss)
        if self._ema_loss is None:
            self._ema_loss = loss
        else:
            self._ema_loss = self.ema_decay * self._ema_loss + (1.0 - self.ema_decay) * loss

    def on_eval_end(self, trainer: Trainer, metrics: dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            if key.startswith("eval/"):
                self._eval[key] = float(value)

    def on_train_end(self, trainer: Trainer) -> None:
        # Record the global batch (batch_size × grad_accum × num_processes) for
        # provenance: μP transfers LR across width *at a fixed batch*, so a sweep
        # is only valid if every width shared one batch — this makes that auditable.
        global_batch = int(
            trainer.cfg.train.batch_size
            * trainer.cfg.train.gradient_accumulation_steps
            * trainer.accelerator.num_processes
        )
        payload = {
            "final_train_loss": self._ema_loss,
            "eval": dict(self._eval),
            "lr": trainer.cfg.train.lr,
            "width": trainer.cfg.model.hidden_size,
            "steps": getattr(trainer, "total_steps", trainer.cfg.train.max_steps),
            "global_batch": global_batch,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2))


# ----------------------------------------------------------------------------
# Training-stability diagnostics
# ----------------------------------------------------------------------------


def _rms(t: torch.Tensor) -> torch.Tensor:
    """Root-mean-square of a tensor as a detached 0-dim float tensor."""
    return t.detach().float().pow(2).mean().sqrt()


# Probe-batch caps: keep the diagnostic forward cheap. output_hidden_states uses
# the memory-efficient SDPA path (as training does), so a small slice suffices.
_PROBE_MAX_EXAMPLES = 4
_PROBE_MAX_TOKENS = 256


class StabilityDiagnosticsCallback(TrainerCallback):
    """Log training-stability signals for deep-model μP diagnosis — hook-free.

    On every training-loss log it emits the pre-clip global grad norm
    (``diag/grad_norm``, read from ``trainer._last_grad_norm``; absent when
    ``max_grad_norm <= 0``). This costs nothing and leaves the compiled training
    step untouched — the per-step tripwire for gradient spikes.

    Every ``probe_every`` training logs it additionally runs one eager diagnostic
    forward (``output_hidden_states``) on the *unwrapped* model over a small fixed
    probe batch and records, under the ``diag/`` prefix:

    * per-depth residual-stream RMS (``diag/residual_rms/{max,mean,argmax_layer,
      final_layer}``) — residual growth and the hidden-state index that drives it
      (index 0 is the post-embedding state; 1..L are block outputs);
    * output-logit RMS (``diag/logit_rms``).

    Distributed-safety (this used to hang DDP runs): the probe issues **no
    collectives**, runs **only on the main process** (weights are DDP-identical
    across ranks, so rank 0's shard is representative), uses the tested SDPA path
    (not the memory-heavy ``output_attentions`` manual path), and calls
    ``torch.cuda.synchronize()`` inside its guard so an async CUDA fault surfaces
    *there* — caught and the probe self-disabled — rather than corrupting the next
    NCCL collective and timing out the whole group. It also runs off the compiled
    training step (eager, unwrapped, ``no_grad``, ``eval`` restored afterward), so
    there are **no forward hooks** and ``torch.compile`` stays on for training.
    ``probe_every=0`` logs only the grad norm. Metrics emit through
    ``accelerator.log`` on the training-loss cadence (eval-only logs are skipped).
    """

    def __init__(self, *, probe_every: int = 25) -> None:
        if probe_every < 0:
            raise ValueError(f"probe_every must be >= 0, got {probe_every}")
        self.probe_every = probe_every
        self._probe_batch: dict[str, torch.Tensor] | None = None
        self._log_count = 0
        self._probe_disabled = False

    # -- lifecycle ----------------------------------------------------------

    def on_train_start(self, trainer: Trainer) -> None:
        # Probe on the main process only (see class docstring); other ranks skip
        # it entirely, so they never capture a batch or run the extra forward.
        if self.probe_every > 0 and trainer.accelerator.is_main_process:
            self._probe_batch = self._capture_probe_batch(trainer)

    def on_log(self, trainer: Trainer, metrics: dict[str, float], step: int) -> None:
        # Fire only on training-step logs, not eval-only emissions, so the diag
        # cadence tracks train logging.
        if "train/loss" not in metrics:
            return
        self._log_count += 1

        diag: dict[str, float] = {}
        grad_norm = getattr(trainer, "_last_grad_norm", None)
        if grad_norm is not None:
            diag["diag/grad_norm"] = float(grad_norm)

        if (
            self.probe_every > 0
            and not self._probe_disabled
            and self._log_count % self.probe_every == 0
        ):
            self._run_probe(trainer, diag)

        if diag:
            trainer.accelerator.log(diag, step=step)

    # -- probe --------------------------------------------------------------

    @staticmethod
    def _capture_probe_batch(trainer: Trainer) -> dict[str, torch.Tensor] | None:
        """Grab one small fixed batch (a fresh iterator; the train iterator is untouched)."""
        try:
            batch = next(iter(trainer.dataloader))
        except (StopIteration, TypeError, KeyError):
            logger.warning("stability probe: could not capture a batch; disabling it")
            return None
        device = trainer.accelerator.device
        probe = {
            key: batch[key][:_PROBE_MAX_EXAMPLES, :_PROBE_MAX_TOKENS].to(device)
            for key in ("input_ids", "attention_mask")
            if key in batch
        }
        return probe or None

    def _run_probe(self, trainer: Trainer, diag: dict[str, float]) -> None:
        """Eager, main-process-only diagnostic forward; fill residual + logit RMS."""
        # Main process only: the extra forward has no collectives, so other ranks
        # must NOT run it (that is what previously hung DDP — an async CUDA fault
        # in the probe corrupted a rank's context and timed out the next collective).
        if not trainer.accelerator.is_main_process:
            return
        if self._probe_batch is None or "input_ids" not in self._probe_batch:
            return
        model = trainer.accelerator.unwrap_model(trainer.model)
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=self._probe_batch["input_ids"],
                    attention_mask=self._probe_batch.get("attention_mask"),
                    output_hidden_states=True,
                )
            # Force any async CUDA fault to surface here, inside the guard, rather
            # than at the next collective (where it would hang the whole group).
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._emit_residual(diag, getattr(outputs, "hidden_states", None))
            logits = getattr(outputs, "logits", None)
            if isinstance(logits, torch.Tensor):
                diag["diag/logit_rms"] = float(_rms(logits))
        except Exception as exc:  # noqa: BLE001 - diagnostics must never kill training
            logger.warning("stability probe failed; disabling it: %s", exc)
            self._probe_disabled = True
        finally:
            if was_training:
                model.train()

    @staticmethod
    def _emit_residual(diag: dict[str, float], hidden_states: object) -> None:
        """Reduce the (L+1)-tuple of residual states to max/mean/argmax/final RMS."""
        if not hidden_states:
            return
        values = torch.stack(
            [_rms(h) for h in hidden_states if isinstance(h, torch.Tensor)]  # ty: ignore[not-iterable]
        )
        if values.numel() == 0:
            return
        diag["diag/residual_rms/max"] = float(values.max())
        diag["diag/residual_rms/mean"] = float(values.mean())
        diag["diag/residual_rms/argmax_layer"] = float(int(values.argmax()))
        diag["diag/residual_rms/final_layer"] = float(values[-1])

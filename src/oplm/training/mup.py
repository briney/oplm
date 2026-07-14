"""μP (maximal update parametrization) helpers for the training side.

The single scaling quantity of the μP recipe is the per-matrix fan-in multiplier
``m_W = fan_in / fan_in_base``, computed by :meth:`OplmConfig.mup_fanin_mult` —
the model-package single source of truth (so width-aware init stays
``trust_remote_code``-safe). This module wraps that quantity for training:

* :func:`mup_fanin_mult` / :func:`mup_lr_multiplier` — consumed by ``optim.py`` to
  scale per-group AdamW learning rates (Phase 4);
* :func:`coord_check` — the empirical correctness gate (per-activation RMS vs
  width);
* :class:`SweepMetricsCallback`, :func:`summarize_sweep`, :func:`best_lr_per_width`
  — the LR-sweep metric utilities.

See ``docs/MUP.md`` for the recipe table and the coord-check pass/fail oracle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from oplm.training.callbacks import TrainerCallback

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import pandas as pd

    from oplm.model.configuration_oplm import OplmConfig
    from oplm.training.trainer import Trainer

__all__ = [
    "MupTransferResult",
    "SweepMetricsCallback",
    "best_lr_per_width",
    "coord_check",
    "mup_fanin_mult",
    "mup_lr_multiplier",
    "summarize_sweep",
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


def summarize_sweep(run_dirs: Sequence[str | Path]) -> pd.DataFrame:
    """Load each run's ``metrics.json`` into a frame keyed by ``(width, lr)``.

    Args:
        run_dirs: Run directories (or direct paths to a ``metrics.json``) written
            by :class:`SweepMetricsCallback`.

    Returns:
        One row per run with ``width``, ``lr``, ``final_train_loss``, ``steps``,
        and one column per ``eval/*`` metric present.
    """
    import pandas as pd

    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        path = Path(run_dir)
        metrics_path = path if path.name == "metrics.json" else path / "metrics.json"
        data = json.loads(metrics_path.read_text())
        row: dict[str, object] = {
            "width": data["width"],
            "lr": data["lr"],
            "final_train_loss": data.get("final_train_loss"),
            "steps": data.get("steps"),
        }
        row.update(data.get("eval") or {})
        rows.append(row)
    return pd.DataFrame.from_records(rows)


@dataclass(frozen=True)
class MupTransferResult:
    """Outcome of an LR sweep: the best LR per width plus the μTransfer verdict.

    Attributes:
        best_lr: Argmin-loss base LR for each width.
        transferred: ``True`` iff at least two widths were swept and they all agree
            on the same argmin LR (the empirical width-transfer verdict).
    """

    best_lr: dict[int, float]
    transferred: bool


def best_lr_per_width(
    df: pd.DataFrame, *, loss_column: str = "final_train_loss"
) -> MupTransferResult:
    """Argmin-loss LR per width and whether the widths agree (the transfer verdict).

    Args:
        df: A sweep frame from :func:`summarize_sweep` (needs ``width``, ``lr``,
            and ``loss_column``).
        loss_column: Column whose minimum selects the best LR per width.

    Returns:
        A :class:`MupTransferResult`. ``transferred`` is ``True`` only when ≥2
        widths were swept and every width's argmin LR matches.
    """
    best: dict[int, float] = {}
    for _, group in df.groupby("width"):
        best_idx = group[loss_column].idxmin()
        best[int(group.loc[best_idx, "width"])] = float(group.loc[best_idx, "lr"])
    transferred = len(best) >= 2 and len(set(best.values())) == 1
    return MupTransferResult(best_lr=best, transferred=transferred)

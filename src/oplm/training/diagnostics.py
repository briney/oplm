"""Weight-norm diagnostics: per-group weight RMS and update-RMS / weight-RMS ratio.

Shared by :class:`WeightDiagnosticsCallback` (live, every ``train.weight_diag_every``
optimizer steps) and ``oplm weight-rms`` (offline, over a checkpoint's ``hf/`` export),
so the two agree on grouping and arithmetic.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import torch

from oplm.training.callbacks import TrainerCallback

if TYPE_CHECKING:
    from collections.abc import Iterable

    from oplm.training.trainer import Trainer

logger = logging.getLogger(__name__)

__all__ = ["WeightDiagnosticsCallback", "param_group_name", "weight_rms_by_group"]

GROUPS = (
    "embed",
    "head_dense",
    "head_decoder",
    "attn",
    "mlp",
    "norm_gain",
    "residual_gate",
    "canon",
    "bias",
    "other",
)


def param_group_name(name: str, ndim: int) -> str:
    """Diagnostic bucket for one named parameter (one of :data:`GROUPS`).

    Buckets follow the optimizer partition: the three Muon-free AdamW families that
    can drift (``embed``, the two head matrices, ``norm_gain``, ``residual_gate``),
    the two Muon matrix families (``attn`` = q/k/v/o/gate_proj, ``mlp`` =
    gate/up/down_proj), the Canon depthwise kernels, biases, and ``other`` (e.g. the
    scalar value-residual lambdas).
    """
    if "embed" in name:
        return "embed"
    if name.startswith("lm_head.dense.") and ndim == 2:
        return "head_dense"
    if name.startswith("lm_head.decoder.") and ndim == 2:
        return "head_decoder"
    if ".conv" in name:
        return "canon"
    if ndim == 2 and ".attention." in name:
        return "attn"
    if ndim == 2 and ".ffn." in name:
        return "mlp"
    if name.endswith(("attn_gate", "ffn_gate")):
        return "residual_gate"
    if name.endswith("norm.weight"):
        return "norm_gain"
    if name.endswith(".bias"):
        return "bias"
    return "other"


def weight_rms_by_group(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> dict[str, float]:
    """Pooled RMS (``sqrt(sum(x²) / numel)``) per diagnostic group, in fp32.

    Args:
        named_tensors: ``(name, tensor)`` pairs, e.g. ``model.named_parameters()`` or
            per-parameter update deltas keyed by the same names.

    Returns:
        ``{group: rms}`` for every group that had at least one tensor.
    """
    sumsq: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros((), dtype=torch.float64))
    numel: dict[str, int] = defaultdict(int)
    for name, tensor in named_tensors:
        group = param_group_name(name, tensor.ndim)
        sumsq[group] = sumsq[group] + tensor.detach().float().pow(2).sum().to("cpu", torch.float64)
        numel[group] += tensor.numel()
    return {g: float((sumsq[g] / numel[g]).sqrt()) for g in GROUPS if numel.get(g)}


class WeightDiagnosticsCallback(TrainerCallback):
    """Log ``diag/weight_rms/<group>`` and ``diag/update_ratio/<group>`` every ``every`` steps.

    Hook-free on the model: it registers a ``step_pre_hook`` on the first optimizer and a
    ``step_post_hook`` on the last, snapshots every parameter before the due step, and
    reduces ``(after - before)`` afterwards. ``update_ratio`` is the update RMS divided by
    the (post-step) weight RMS -- the effective relative step size. Metrics are stashed
    and emitted with the next training log (same step when ``every`` is a multiple of
    ``log_every``). Main process only, DDP only (weights are replicated; no collectives).
    """

    def __init__(self, *, every: int) -> None:
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        self.every = every
        self._trainer: Trainer | None = None
        self._named: list[tuple[str, torch.Tensor]] = []
        self._before: dict[str, torch.Tensor] | None = None
        self._pending: dict[str, float] = {}

    def on_train_start(self, trainer: Trainer) -> None:
        from oplm.training.checkpoint import _unwrap_optimizer

        self._trainer = trainer
        model = trainer.accelerator.unwrap_model(trainer.model, keep_torch_compile=False)
        self._named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        # The inner torch optimizers step exactly once per optimizer step (accelerate's
        # wrapper skips non-sync micro-steps), so the hooks fire once per global step.
        inner = [_unwrap_optimizer(o) for o in trainer.optimizers]
        inner[0].register_step_pre_hook(self._pre_step)
        inner[-1].register_step_post_hook(self._post_step)

    def _due(self) -> bool:
        # global_step is incremented after the optimizer step; the step in flight is +1.
        return self._trainer is not None and (self._trainer.global_step + 1) % self.every == 0

    def _pre_step(self, *_: object) -> None:
        if self._due():
            # ponytail: one full fp32 copy of the model on the diag step (0.7 GB at 170M,
            # 1.8 GB at 400M); move the snapshot to CPU if GPU memory gets tight.
            self._before = {n: p.detach().clone() for n, p in self._named}

    def _post_step(self, *_: object) -> None:
        trainer = self._trainer
        if trainer is None or self._before is None:
            return
        before, self._before = self._before, None
        weight = weight_rms_by_group(self._named)
        update = weight_rms_by_group((n, p.detach() - before[n]) for n, p in self._named)
        self._pending = {f"diag/weight_rms/{g}": w for g, w in weight.items()} | {
            f"diag/update_ratio/{g}": update[g] / w if w > 0 else 0.0 for g, w in weight.items()
        }
        self._pending["diag/weight_step"] = float(trainer.global_step + 1)

    def on_log(self, trainer: Trainer, metrics: dict[str, float], step: int) -> None:
        if self._pending and "train/loss" in metrics:
            trainer.accelerator.log(self._pending | {"train/global_step": step})
            self._pending = {}

"""Training-task seam: what the Trainer needs from a model/data/step triple.

The Trainer owns acceleration, optimization, gradient accumulation, evaluation
cadence, fault tolerance, and checkpointing. A :class:`TrainTask` owns the three
things that differ between objectives: how the model is built, how the training
dataloader is built, and how one micro-batch becomes a loss. :class:`MLMTask` is
the default and reproduces the pre-seam masked-language-model behaviour exactly.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from pathlib import Path

    from accelerate import Accelerator
    from torch import nn
    from torch.utils.data import DataLoader

    from oplm.config import DistillConfig, OplmConfig
    from oplm.training.contact_distill import AuxContactManager

__all__ = ["DistillMLMTask", "MLMTask", "StepResult", "TrainTask"]

# Metric keys the Trainer itself emits under ``train/``; a task may not reuse them.
_RESERVED_METRIC_KEYS = frozenset(
    {
        "loss",
        "loss_mean",
        "lr",
        "lr_adamw",
        "epoch",
        "samples",
        "tokens",
        "flops",
        "global_step",
        "grad_norm",
        "grad_norm_max",
        "clip_frac",
        "mean_seq_len",
        "tokens_per_sec",
        "step_time_s",
        "achieved_tflops",
        "mfu",
    }
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

    def step(self, model: nn.Module, batch: Any) -> StepResult:
        """Run one micro-batch forward and return its loss and accounting.

        ``batch`` is the dataloader's batch: a dict for MLM, a ``FoldBatch`` for folding.
        """
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

    def step(self, model: nn.Module, batch: Any) -> StepResult:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            segment_ids=batch.get("segment_ids"),  # packed batches only
        )
        return StepResult(
            loss=outputs["loss"],
            tokens=int(batch["attention_mask"].sum().item()),
            samples=batch.get("num_sequences", len(batch["input_ids"])),
        )

    def flops_per_token(self, cfg: OplmConfig) -> int | None:
        from oplm.training.flops import estimate_flops_per_token

        return estimate_flops_per_token(cfg.model)


class DistillMLMTask(MLMTask):
    """MLM task with optional 1D teacher-logit KL and 2D contact-pair distillation.

    Inherits model building, dataloader building, and FLOP accounting from
    :class:`MLMTask`. In :meth:`step`, each masked position gets standard
    cross-entropy when ``teacher_mask`` is ``False``, and
    ``(1 - alpha) * CE + alpha * T^2 * D(p_teacher_T, p_student_T)`` over the
    20 canonical amino-acid channels when ``teacher_mask`` is ``True``. When
    ``distill.contact_weight > 0``, adds a soft-BCE contact loss computed via a
    discarded training-only auxiliary head owned by this task.
    """

    def __init__(self, distill_cfg: DistillConfig) -> None:
        from oplm.data.tokenizer import get_tokenizer

        self._distill = distill_cfg
        tok = get_tokenizer()
        token_ids = [tok.convert_tokens_to_ids(aa) for aa in distill_cfg.teacher_alphabet]
        if any(not isinstance(tid, int) or tid < 0 for tid in token_ids):
            raise ValueError(
                f"teacher_alphabet {distill_cfg.teacher_alphabet!r} could not be mapped "
                "to valid tokenizer ids"
            )
        self._teacher_token_ids = torch.tensor(token_ids, dtype=torch.long)
        self._mask_token_id = int(tok.mask_token_id)
        self._device_token_ids: torch.Tensor | None = None
        self._aux: AuxContactManager | None = None
        self._qk_layers: tuple[int, ...] | None = None

    @property
    def has_contact_distill(self) -> bool:
        """Whether 2D contact-pair distillation is enabled."""
        return self._distill.contact_weight > 0.0

    def prepare_aux(
        self,
        model: nn.Module,
        accelerator: Accelerator,
        cfg: OplmConfig,
        total_steps: int,
    ) -> None:
        """Instantiate and broadcast the discarded 2D contact head when ``contact_weight > 0``."""
        if not self.has_contact_distill:
            return
        from oplm.training.contact_distill import AuxContactManager, build_contact_head

        head, qk_layers = build_contact_head(self._distill, model)
        head.to(accelerator.device)
        self._qk_layers = qk_layers
        self._aux = AuxContactManager(head, self._distill, cfg.train, total_steps)
        self._aux.broadcast_weights()

    def sync_aux_schedule(self, global_step: int, *, fresh_start: bool) -> None:
        """Align the auxiliary head's LR schedule after init, resume, or branch."""
        if self._aux is not None:
            self._aux.sync_schedule(global_step, fresh_start=fresh_start)

    def step_aux(self, accelerator: Accelerator, max_grad_norm: float, global_step: int) -> None:
        """Synchronize auxiliary contact-head gradients and take one optimizer step."""
        if self._aux is not None:
            self._aux.step(accelerator, max_grad_norm, global_step)

    def aux_state_dict(self) -> dict[str, Any] | None:
        """Return auxiliary head/optimizer state for ``distill_aux.pt``, or ``None``."""
        return self._aux.state_dict() if self._aux is not None else None

    def load_aux_state_dict(self, state: dict[str, Any], *, reset_hparams: bool = False) -> None:
        """Restore auxiliary head/optimizer state from ``distill_aux.pt``."""
        if self._aux is not None:
            self._aux.load_state_dict(state, reset_hparams=reset_hparams)

    def _token_ids_for(self, device: torch.device) -> torch.Tensor:
        if self._device_token_ids is None or self._device_token_ids.device != device:
            self._device_token_ids = self._teacher_token_ids.to(device)
        return self._device_token_ids

    def step(self, model: nn.Module, batch: Any) -> StepResult:
        use_contacts = self.has_contact_distill
        if use_contacts and self._aux is None:
            raise RuntimeError(
                "DistillMLMTask.prepare_aux() must be called before step() when "
                "distill.contact_weight > 0"
            )
        forward_kwargs: dict[str, Any] = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": None,
            "segment_ids": batch.get("segment_ids"),  # packed batches only
        }
        if use_contacts:
            if self._distill.contact_head == "attention":
                forward_kwargs["return_qk_layers"] = self._qk_layers
            else:
                forward_kwargs["output_hidden_states"] = True

        outputs = model(**forward_kwargs)
        logits = outputs["logits"].float()  # (R, T, V) fp32
        labels: torch.Tensor = batch["labels"]
        tokens = int(batch["attention_mask"].sum().item())
        samples = batch.get("num_sequences", len(batch["input_ids"]))

        masked = labels != -100
        n_masked = int(masked.sum().item())
        metrics: dict[str, float] = {}

        if n_masked > 0:
            has_1d = "teacher_mask" in batch and "teacher_logits" in batch
            if has_1d:
                teacher_mask: torch.Tensor = (
                    batch["teacher_mask"] & masked & (batch["input_ids"] == self._mask_token_id)
                )
                n_distill = int(teacher_mask.sum().item())
            else:
                teacher_mask = torch.zeros_like(masked)
                n_distill = 0

            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(labels)
            plain_ce_sum = ce[masked & ~teacher_mask].sum()
            distill_ce_sum = ce[teacher_mask].sum()

            alpha = self._distill.alpha
            if n_distill > 0:
                token_ids = self._token_ids_for(logits.device)
                kl_ctx = torch.no_grad() if alpha == 0.0 else nullcontext()
                with kl_ctx:
                    s20 = logits[teacher_mask].index_select(-1, token_ids)
                    t20 = batch["teacher_logits"][teacher_mask].float()
                    temp = self._distill.temperature
                    log_ps = F.log_softmax(s20 / temp, dim=-1)
                    log_pt = F.log_softmax(t20 / temp, dim=-1)
                    loss_type = self._distill.loss_type
                    if loss_type == "forward_kl":
                        div_per_token = F.kl_div(log_ps, log_pt.exp(), reduction="none").sum(dim=-1)
                    elif loss_type == "reverse_kl":
                        div_per_token = (log_ps.exp() * (log_ps - log_pt)).sum(dim=-1)
                    elif loss_type == "js":
                        log_m = torch.logaddexp(log_ps, log_pt) - math.log(2.0)
                        div_per_token = 0.5 * (
                            (log_pt.exp() * (log_pt - log_m)).sum(dim=-1)
                            + (log_ps.exp() * (log_ps - log_m)).sum(dim=-1)
                        )
                    else:
                        raise ValueError(f"Unsupported distill.loss_type: {loss_type!r}")
                    kl_per_token = div_per_token * (temp * temp)
                    kl_sum = kl_per_token.sum()
            else:
                kl_sum = logits.new_zeros(())

            if alpha == 0.0:
                loss = (plain_ce_sum + distill_ce_sum) / n_masked
            else:
                loss = (plain_ce_sum + (1.0 - alpha) * distill_ce_sum + alpha * kl_sum) / n_masked

            metrics["loss_ce"] = float((plain_ce_sum + distill_ce_sum).detach().item() / n_masked)
            metrics["distill_frac"] = float(n_distill / n_masked)
            if n_distill > 0:
                metrics["loss_kl"] = float(kl_sum.detach().item() / n_distill)
        else:
            loss = logits.sum() * 0.0

        if use_contacts:
            from oplm.training.contact_distill import compute_contact_distill_loss

            assert self._aux is not None
            contact_meta = batch.get("teacher_contact_meta") or []
            contact_values = batch.get("teacher_contact_values")
            metrics["contact_frac"] = float(len(contact_meta) / max(1, samples))
            if contact_meta and contact_values is not None:
                if self._distill.contact_head == "attention":
                    qk_states = outputs["qk_states"]
                    h_states = None
                else:
                    qk_states = None
                    h_states = outputs["hidden_states"][-self._distill.contact_layers :]
                loss_contact, pos_rate = compute_contact_distill_loss(
                    head=self._aux.head,
                    qk_states=qk_states,
                    hidden_states=h_states,
                    contact_meta=contact_meta,
                    contact_values=contact_values,
                    contact_min_sep=self._distill.contact_min_sep,
                    pos_weight=self._distill.contact_pos_weight,
                    max_chunk_pairs=self._distill.contact_max_chunk_pairs,
                )
                loss = loss + self._distill.contact_weight * loss_contact
                metrics["loss_contact"] = float(loss_contact.detach().item())
                metrics["contact_pos_rate"] = float(pos_rate)

        return StepResult(loss=loss, tokens=tokens, samples=samples, metrics=metrics)

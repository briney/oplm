"""MLM evaluation metrics: loss, masked accuracy, and perplexity."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import DataLoader

    from oplm.eval.context import DistContext
    from oplm.model import OplmForMaskedLM

_PERPLEXITY_CAP = 1000.0


def compute_mlm_metrics(
    model: OplmForMaskedLM,
    dataloader: DataLoader[dict[str, Tensor]],
    dist_ctx: DistContext,
) -> dict[str, float]:
    """Compute MLM metrics over an eval DataLoader.

    Runs the model on all batches, accumulating loss and accuracy across
    the full dataset. Handles distributed gathering so each rank contributes
    its shard and the final metrics reflect the complete dataset.

    The model should already be in eval mode when this is called.

    Args:
        model: The unwrapped model (already in eval mode).
        dataloader: Eval DataLoader yielding ``{input_ids, attention_mask, labels}``.
        dist_ctx: Distributed-runtime context for device + collective reduction.

    Returns:
        Dict with keys ``"loss"``, ``"accuracy"``, ``"perplexity"``.
    """
    device = dist_ctx.device

    total_loss = torch.zeros(1, device=device)
    total_correct = torch.zeros(1, device=device, dtype=torch.long)
    total_masked = torch.zeros(1, device=device, dtype=torch.long)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs["logits"]  # (B, T, V)

            # Mask of positions that have labels (not -100)
            mask = labels != -100  # (B, T)
            n_masked = mask.sum()

            if n_masked > 0:
                # Sum of per-token losses (not mean-reduced)
                # cross_entropy with reduction="none" gives per-position loss
                per_token_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),  # (B*T, V)
                    labels.view(-1),  # (B*T,)
                    ignore_index=-100,
                    reduction="sum",
                )
                total_loss += per_token_loss

                # Accuracy: count correct predictions at masked positions
                preds = logits.argmax(dim=-1)  # (B, T)
                correct = (preds == labels) & mask  # (B, T)
                total_correct += correct.sum()
                total_masked += n_masked

    # Gather across ranks (no-op for a single rank or an uninitialized group).
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_masked, op=dist.ReduceOp.SUM)

    n = total_masked.item()
    if n == 0:
        return {"loss": 0.0, "accuracy": 0.0, "perplexity": 1.0}

    loss = total_loss.item() / n
    accuracy = total_correct.item() / n
    perplexity = min(math.exp(loss), _PERPLEXITY_CAP)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "perplexity": perplexity,
    }

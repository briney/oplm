"""Shared model scoring for variant-effect eval tasks.

Per-position log-likelihood-ratio scoring (masked-marginal or wild-type-marginal)
shared by the ProteinGym DMS and clinical eval tasks, plus small helpers for
length eligibility and cross-rank gathering. Metric computation and the
score-direction (sign) convention live in the individual tasks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import torch

from oplm.data.tokenizer import mask_token_id
from oplm.data.variant.loader import parse_mutation

if TYPE_CHECKING:
    from accelerate import Accelerator

    from oplm.data import VariantAssay
    from oplm.model import OplmForMaskedLM, OplmTokenizerFast

logger = logging.getLogger(__name__)

T = TypeVar("T")

VALID_SCORING = ("masked_marginals", "wt_marginals")
_MULTI_MUTANT_SEP = ":"


def wildtype_fits(wildtype: str, max_position_embeddings: int) -> bool:
    """Return whether ``wildtype`` (plus BOS/EOS) fits the model context."""
    return len(wildtype) + 2 <= max_position_embeddings


def compute_variant_llrs(
    assay: VariantAssay,
    model: OplmForMaskedLM,
    tokenizer: OplmTokenizerFast,
    device: torch.device,
    *,
    scoring: str,
    mask_batch_size: int,
) -> np.ndarray:
    """Score each variant row by its summed per-position log-likelihood ratio.

    For a substitution ``(wt, pos, mut)`` the ratio is
    ``log P(mut | context) − log P(wt | context)`` read from a full-vocab
    ``log_softmax`` at that position; a row's score sums the ratios of its
    substitutions. Higher means the model finds the variant more tolerated. The
    caller decides the sign convention for its labels.

    Args:
        assay: The validated assay (its mutations were checked against the WT).
        model: The unwrapped model in eval mode.
        tokenizer: The canonical tokenizer.
        device: Device to run forward passes on.
        scoring: ``"masked_marginals"`` (mask each unique mutated position) or
            ``"wt_marginals"`` (one wild-type forward pass).
        mask_batch_size: Masked sequences per forward pass (``masked_marginals``).

    Returns:
        A ``(n_rows,)`` float64 array of per-variant log-likelihood ratios.

    Raises:
        ValueError: If a mutant amino acid is unknown to the tokenizer.
    """
    parsed_rows = [
        [parse_mutation(part) for part in token.split(_MULTI_MUTANT_SEP)]
        for token in assay.mutations
    ]
    positions = sorted({mutation.pos - 1 for row in parsed_rows for mutation in row})

    if scoring == "wt_marginals":
        pos_logprobs = _wt_marginal_logprobs(assay.wildtype, positions, model, tokenizer, device)
    else:
        pos_logprobs = _masked_marginal_logprobs(
            assay.wildtype, positions, model, tokenizer, device, mask_batch_size
        )
    return _sum_llrs(parsed_rows, pos_logprobs, tokenizer, assay.name)


def _sum_llrs(
    parsed_rows: list[list],
    pos_logprobs: dict[int, torch.Tensor],
    tokenizer: OplmTokenizerFast,
    assay_name: str,
) -> np.ndarray:
    """Sum per-position log-likelihood ratios into one score per variant row."""
    unk_id = tokenizer.unk_token_id
    aa_ids: dict[str, int] = {}

    def token_id(aa: str) -> int:
        if aa not in aa_ids:
            tid = tokenizer.convert_tokens_to_ids(aa)  # single token -> int
            if not isinstance(tid, int) or tid == unk_id:
                raise ValueError(f"[{assay_name}] amino acid {aa!r} maps to <unk>; cannot score")
            aa_ids[aa] = tid
        return aa_ids[aa]

    scores = np.empty(len(parsed_rows), dtype=np.float64)
    for k, row in enumerate(parsed_rows):
        total = 0.0
        for mutation in row:
            logprobs = pos_logprobs[mutation.pos - 1]
            total += float(logprobs[token_id(mutation.mut)] - logprobs[token_id(mutation.wt)])
        scores[k] = total
    return scores


def _wt_marginal_logprobs(
    wildtype: str,
    positions: list[int],
    model: OplmForMaskedLM,
    tokenizer: OplmTokenizerFast,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Log-softmax logits at each position from one wild-type forward pass.

    Returns a mapping ``{residue_index_0based: (V,) log-prob tensor}``.
    """
    input_ids = torch.tensor(tokenizer.encode(wildtype), dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
    logprobs = torch.log_softmax(logits[0].float(), dim=-1).cpu()  # (T, V)
    # Residue i (0-based) lives at token index i+1 (BOS at index 0).
    return {i: logprobs[i + 1] for i in positions}


def _masked_marginal_logprobs(
    wildtype: str,
    positions: list[int],
    model: OplmForMaskedLM,
    tokenizer: OplmTokenizerFast,
    device: torch.device,
    mask_batch_size: int,
) -> dict[int, torch.Tensor]:
    """Log-softmax logits at each position with that position masked.

    One masked forward per unique position, batched up to ``mask_batch_size``.
    Returns a mapping ``{residue_index_0based: (V,) log-prob tensor}``.
    """
    base = torch.tensor(tokenizer.encode(wildtype), dtype=torch.long)  # (T,)
    mask_id = mask_token_id(tokenizer)

    result: dict[int, torch.Tensor] = {}
    for start in range(0, len(positions), mask_batch_size):
        chunk = positions[start : start + mask_batch_size]
        batch = base.unsqueeze(0).repeat(len(chunk), 1)  # (B, T)
        for row_idx, pos in enumerate(chunk):
            batch[row_idx, pos + 1] = mask_id  # BOS offset
        input_ids = batch.to(device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
        for row_idx, pos in enumerate(chunk):
            result[pos] = torch.log_softmax(logits[row_idx, pos + 1].float(), dim=-1).cpu()
        del logits, input_ids, attention_mask, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def gather_objects(local_data: list[T], accelerator: Accelerator) -> list[T]:
    """Gather per-rank Python objects from all ranks into a flat list."""
    if accelerator.num_processes == 1:
        return local_data

    import torch.distributed as dist

    all_data_lists: list[list[T] | None] = [None] * accelerator.num_processes
    dist.all_gather_object(all_data_lists, local_data)

    gathered: list[T] = []
    for rank_data in all_data_lists:
        if rank_data is not None:
            gathered.extend(rank_data)
    return gathered

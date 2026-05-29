"""Tokenization, padding, and masked-language-model collation.

``tokenize_and_pad`` is the shared pad/tokenize primitive (no masking).
:class:`MLMCollator` layers fixed-``k`` Gumbel-top-k masking with BERT 80/10/10
replacement on top, supporting optional per-residue weighted masking.

The masking scheme (docs/DATA_TOOLING.md §4.5) is **dynamic** (RoBERTa-style):
masks are drawn fresh each call, so the same sequence is masked differently across
epochs. Evaluation freezes them via ``deterministic=True`` — a per-batch seeded
generator, *not* a separate collator class (§4.6). All token-id constants are
derived from the tokenizer (``data/tokenizer.py``), never hardcoded.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from oplm.data.tokenizer import (
    align_per_residue,
    canonical_amino_acid_ids,
    mask_token_id,
    non_maskable_ids,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from oplm.model import OplmTokenizerFast

logger = logging.getLogger(__name__)

# Cross-entropy ignore index for non-target positions (PyTorch default).
_IGNORE_INDEX = -100

# Per-residue masking-weight field carried on row dicts (see dataset.py).
_WEIGHTS_KEY = "masking_weights"

__all__ = ["MLMCollator", "tokenize_and_pad"]


def _sequence_of(item: Mapping[str, object] | str) -> str:
    """Extract the raw sequence string from a batch item.

    Args:
        item: Either a raw sequence ``str`` or a mapping with a ``"sequence"`` key.

    Returns:
        The raw one-letter amino-acid sequence.

    Raises:
        TypeError: If ``item`` is neither a ``str`` nor a mapping.
        KeyError: If a mapping item lacks a ``"sequence"`` key.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        if "sequence" not in item:
            raise KeyError("batch item mapping is missing required 'sequence' key")
        return str(item["sequence"])
    raise TypeError(f"batch items must be str or mapping, got {type(item).__name__}")


def tokenize_and_pad(
    batch: Sequence[Mapping[str, object] | str],
    tokenizer: OplmTokenizerFast,
    max_length: int,
    *,
    weights: Sequence[Sequence[float] | None] | None = None,
) -> dict[str, Tensor]:
    """Tokenize and pad a batch of sequences (no masking, no labels).

    Each raw sequence is truncated to ``max_length - 2`` residues (leaving room for
    ``<cls>``/``<eos>``), tokenized with the canonical tokenizer, and padded to the
    batch's longest member with ``pad_token_id``. This is the shared primitive used
    by the variant/structure/downstream consumers and, internally, by
    :class:`MLMCollator`.

    Args:
        batch: Items are raw sequence ``str`` s or mappings carrying a ``"sequence"``.
        tokenizer: The canonical :class:`~oplm.model.OplmTokenizerFast`.
        max_length: Maximum tokenized length ``T`` (including ``<cls>``/``<eos>``);
            raw sequences are clipped to ``max_length - 2`` residues.
        weights: Optional per-row raw per-residue vectors (``None`` for a row with
            no weights). When supplied, the output additionally carries an aligned
            ``"masking_weights"`` ``(B, T)`` tensor (specials/pad → ``0.0``,
            ``None`` row → uniform ``1.0``) via :func:`~oplm.data.tokenizer.align_per_residue`.
            Used internally by the MLM collator; the default output is the two keys
            below.

    Returns:
        ``{"input_ids": (B, T) long, "attention_mask": (B, T) long}``, plus
        ``"masking_weights": (B, T) float32`` when ``weights`` is supplied.
    """
    sequences = [_sequence_of(item) for item in batch]
    max_residues = max_length - 2  # room for <cls>/<eos>
    truncated = [seq[:max_residues] for seq in sequences]

    encoded = tokenizer(truncated, padding=True, return_tensors="pt")
    out: dict[str, Tensor] = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }

    if weights is not None:
        total_len = int(out["input_ids"].shape[1])
        # Raw (pre-truncation) lengths; align_per_residue re-applies truncation.
        lengths = [len(seq) for seq in sequences]
        out[_WEIGHTS_KEY] = align_per_residue(
            weights,
            lengths=lengths,
            total_len=total_len,
            fill_special=0.0,
            fill_pad=0.0,
        )
    return out


class MLMCollator:
    """Masked-language-model collator: tokenize/pad, then mask.

    Wraps :func:`tokenize_and_pad` and applies the OPLM masking scheme
    (docs/DATA_TOOLING.md §4.5): per row, a **fixed count**
    ``k = round(mask_prob * n_eligible)`` of eligible positions is sampled without
    replacement via **Gumbel-top-k**, then the BERT 80/10/10 split decides each
    masked token's replacement (``<mask>`` / random canonical AA / keep original).
    Uniform masking is the equal-weights special case of the same code path.

    Masking is dynamic by default (fresh each call). With ``deterministic=True`` a
    per-batch generator seeded by ``seed + batch_index`` makes a given batch index
    reproducible without disturbing the global RNG — the evaluation policy.

    Args:
        tokenizer: The canonical :class:`~oplm.model.OplmTokenizerFast`.
        max_length: Maximum tokenized length passed to :func:`tokenize_and_pad`.
        mask_prob: Fraction of eligible positions selected for masking per row.
        mask_token_prob: Of masked positions, fraction replaced with ``<mask>``.
        random_token_prob: Of masked positions, fraction replaced with a random
            canonical amino acid. The remainder keep their original id.
        weighted_masking: Honor each row's ``masking_weights`` to bias selection.
            When ``False`` the column is ignored and masking is uniform.
        deterministic: Freeze masks per batch index (evaluation policy).
        seed: Base seed for the deterministic per-batch generator.

    Raises:
        ValueError: If any probability is outside ``[0, 1]`` or
            ``mask_token_prob + random_token_prob > 1``.
    """

    def __init__(
        self,
        tokenizer: OplmTokenizerFast,
        max_length: int = 1024,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1,
        *,
        weighted_masking: bool = False,
        deterministic: bool = False,
        seed: int = 0,
    ) -> None:
        _validate_probs(mask_prob, mask_token_prob, random_token_prob)
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._mask_prob = mask_prob
        self._mask_token_prob = mask_token_prob
        self._random_token_prob = random_token_prob
        self._weighted_masking = weighted_masking
        self._deterministic = deterministic
        self._seed = seed

        # Derived id constants (computed once from the tokenizer, never hardcoded).
        self._mask_token_id = mask_token_id(tokenizer)
        self._canonical_ids = canonical_amino_acid_ids(tokenizer)  # (20,) long
        self._non_maskable = torch.tensor(sorted(non_maskable_ids(tokenizer)), dtype=torch.long)

        self._batch_idx = 0
        self._warned_missing_weights = False

    def __call__(self, batch: Sequence[Mapping[str, object] | str]) -> dict[str, Tensor]:
        """Collate ``batch`` into ``{input_ids, attention_mask, labels}``."""
        generator = self._next_generator()

        raw_weights = self._collect_weights(batch) if self._weighted_masking else None
        encoded = tokenize_and_pad(batch, self._tokenizer, self._max_length, weights=raw_weights)
        input_ids = encoded["input_ids"]  # (B, T) long
        attention_mask = encoded["attention_mask"]  # (B, T) long
        labels = torch.full_like(input_ids, _IGNORE_INDEX)

        # Eligibility (B, T): real, non-special tokens.
        eligible = (attention_mask == 1) & ~torch.isin(input_ids, self._non_maskable)

        # Selection weights (B, T): uniform unless weighted masking is on; either
        # way, non-eligible positions get weight 0 so they can never be selected.
        if self._weighted_masking:
            weights = encoded[_WEIGHTS_KEY]
        else:
            weights = torch.ones_like(input_ids, dtype=torch.float32)
        weights = weights.masked_fill(~eligible, 0.0)

        self._mask_batch(input_ids, labels, eligible, weights, generator)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def reset_batch_index(self) -> None:
        """Reset the per-batch counter so the next call is batch index 0 again.

        Only meaningful under ``deterministic=True``, where the per-batch RNG is
        seeded by ``seed + batch_index`` (§4.6): resetting before a fresh pass
        makes that pass reproduce the previous one exactly (identical masks),
        which is what makes MLM-eval metrics comparable across training steps. The
        sequence-eval dataloader calls this at the start of each iteration. The
        one-time missing-weights warning flag is intentionally left untouched so
        the warning does not re-fire every pass.
        """
        self._batch_idx = 0

    def _next_generator(self) -> torch.Generator | None:
        """Return this call's RNG and advance the batch counter.

        Deterministic mode returns a dedicated generator seeded by
        ``seed + batch_index`` (global RNG untouched). Dynamic mode returns
        ``None`` so sampling falls through to the global RNG and masks stay fresh.
        """
        if self._deterministic:
            generator = torch.Generator()
            generator.manual_seed(self._seed + self._batch_idx)
        else:
            generator = None
        self._batch_idx += 1
        return generator

    def _collect_weights(
        self, batch: Sequence[Mapping[str, object] | str]
    ) -> list[list[float] | None]:
        """Pull per-row ``masking_weights`` from the batch dicts.

        Rows without weights (non-mapping, or ``None``) surface as ``None`` and are
        treated as uniform by :func:`~oplm.data.tokenizer.align_per_residue`. If the
        whole batch lacks weights (column entirely absent), warn once and fall back
        to uniform masking (docs/DATA_TOOLING.md §4.5.1).
        """
        weights: list[list[float] | None] = []
        any_present = False
        for item in batch:
            value = item.get(_WEIGHTS_KEY) if isinstance(item, Mapping) else None
            if value is not None:
                any_present = True
            weights.append(value)  # type: ignore[arg-type]  # list[float] | None

        if not any_present and not self._warned_missing_weights:
            logger.warning(
                "weighted_masking=True but no '%s' present in batch; "
                "falling back to uniform masking.",
                _WEIGHTS_KEY,
            )
            self._warned_missing_weights = True
        return weights

    def _mask_batch(
        self,
        input_ids: Tensor,
        labels: Tensor,
        eligible: Tensor,
        weights: Tensor,
        generator: torch.Generator | None,
    ) -> None:
        """Select and replace masked positions in place, row by row.

        Selection is Gumbel-top-k: ``key = log(weight) + gumbel`` and the top
        ``k_b`` keys are masked. ``k_b`` is clamped to the count of positive-weight
        eligible positions, so uniform masking (all weights 1) is the special case.
        """
        # Gumbel noise g = -log(-log(u)), u ~ Uniform(0, 1). (B, T)
        u = torch.rand(input_ids.shape, generator=generator)
        gumbel = -torch.log(-torch.log(u))
        # key = log(w) + g; force -inf where weight == 0 (zero-weight + non-eligible)
        # so those positions are never in the top-k (and so NaNs from -inf+inf die).
        key = torch.log(weights) + gumbel
        key = key.masked_fill(weights == 0, float("-inf"))

        n_eligible = eligible.sum(dim=1)  # (B,)
        k_per_row = torch.round(self._mask_prob * n_eligible.float()).long()
        n_positive = (weights > 0).sum(dim=1)  # (B,)
        k_per_row = torch.minimum(k_per_row, n_positive)

        for b in range(input_ids.shape[0]):
            k_b = int(k_per_row[b].item())
            if k_b <= 0:
                continue
            masked_idx = torch.topk(key[b], k_b).indices  # (k_b,)
            labels[b, masked_idx] = input_ids[b, masked_idx]
            self._replace(input_ids, b, masked_idx, generator)

    def _replace(
        self,
        input_ids: Tensor,
        row: int,
        masked_idx: Tensor,
        generator: torch.Generator | None,
    ) -> None:
        """Apply the BERT 80/10/10 replacement split over a row's masked positions.

        ``mask_token_prob`` → ``<mask>``; ``random_token_prob`` → a uniform draw
        from the canonical amino acids; the remainder keep the original id.
        """
        n = int(masked_idx.numel())
        roll = torch.rand(n, generator=generator)  # (n,)
        to_mask = roll < self._mask_token_prob
        to_random = (roll >= self._mask_token_prob) & (
            roll < self._mask_token_prob + self._random_token_prob
        )

        input_ids[row, masked_idx[to_mask]] = self._mask_token_id

        random_idx = masked_idx[to_random]
        n_random = int(random_idx.numel())
        if n_random > 0:
            choice = torch.randint(
                0, int(self._canonical_ids.numel()), (n_random,), generator=generator
            )
            input_ids[row, random_idx] = self._canonical_ids[choice]


def _validate_probs(mask_prob: float, mask_token_prob: float, random_token_prob: float) -> None:
    """Validate the masking probabilities (mirrors ``DataConfig`` constraints)."""
    if not 0.0 <= mask_prob <= 1.0:
        raise ValueError(f"mask_prob must be in [0, 1], got {mask_prob}")
    if not 0.0 <= mask_token_prob <= 1.0:
        raise ValueError(f"mask_token_prob must be in [0, 1], got {mask_token_prob}")
    if not 0.0 <= random_token_prob <= 1.0:
        raise ValueError(f"random_token_prob must be in [0, 1], got {random_token_prob}")
    if mask_token_prob + random_token_prob > 1.0 + 1e-9:
        raise ValueError(
            "mask_token_prob + random_token_prob must be <= 1, got "
            f"{mask_token_prob} + {random_token_prob} = {mask_token_prob + random_token_prob}"
        )

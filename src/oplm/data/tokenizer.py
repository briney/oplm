"""Tokenizer access layer.

A thin accessor over :class:`oplm.model.OplmTokenizerFast` (the single source of
truth for the vocabulary). Defines no vocabulary of its own; provides the
tokenizer accessor, derived id constants computed from the tokenizer instance,
and per-residue vector alignment to tokenized ``input_ids``.

All id constants are *derived* from the live tokenizer rather than hardcoded, so
they can never drift from the model's embedding table (see docs/DATA_TOOLING.md
§3.3 and the legacy off-by-one vocabulary bug in §3.5). The constant accessors
default to the shared :func:`get_tokenizer` instance but accept an explicit
tokenizer so the collator can pass the one it already holds.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from oplm.model import OplmTokenizerFast

if TYPE_CHECKING:
    from collections.abc import Sequence

# The 20 standard amino acids in canonical (ESM-C) vocabulary order. This is the
# *order*, not a vocabulary definition — token ids are resolved through the
# tokenizer, which expects these to land on the contiguous block 4..23.
_CANONICAL_AA_ORDER = "LAGVSERTIDPKQNFYMHWC"

__all__ = [
    "align_per_residue",
    "canonical_amino_acid_ids",
    "get_tokenizer",
    "mask_token_id",
    "non_maskable_ids",
    "pad_token_id",
    "special_ids",
]


@cache
def get_tokenizer() -> OplmTokenizerFast:
    """Return the shared canonical tokenizer.

    Cached as a module-level singleton. The instance is treated as read-only;
    callers must not mutate it.

    Returns:
        The canonical :class:`~oplm.model.OplmTokenizerFast`.
    """
    return OplmTokenizerFast()


def _resolve(tok: OplmTokenizerFast | None) -> OplmTokenizerFast:
    """Return ``tok`` or the shared tokenizer when ``tok`` is ``None``."""
    return get_tokenizer() if tok is None else tok


def special_ids(tok: OplmTokenizerFast | None = None) -> set[int]:
    """Return the set of special-token ids (``{0, 1, 2, 3, 32}`` today).

    Args:
        tok: Tokenizer to query; defaults to the shared instance.

    Returns:
        ``set(tokenizer.all_special_ids)`` — ``<cls>``, ``<pad>``, ``<eos>``,
        ``<unk>``, ``<mask>``.
    """
    return set(_resolve(tok).all_special_ids)


def non_maskable_ids(tok: OplmTokenizerFast | None = None) -> set[int]:
    """Return ids that must never be selected as MLM targets.

    These coincide with the special-token ids today (cls/pad/eos/unk/mask); the
    helper exists as a distinct concept so the eligibility rule reads clearly and
    can diverge later without touching call sites.

    Args:
        tok: Tokenizer to query; defaults to the shared instance.

    Returns:
        The non-maskable id set.
    """
    return special_ids(tok)


def mask_token_id(tok: OplmTokenizerFast | None = None) -> int:
    """Return the ``<mask>`` token id (32 today)."""
    mask_id = _resolve(tok).mask_token_id
    if mask_id is None:
        raise ValueError("tokenizer has no <mask> token; cannot run MLM masking")
    return int(mask_id)


def pad_token_id(tok: OplmTokenizerFast | None = None) -> int:
    """Return the ``<pad>`` token id (1 today)."""
    pad_id = _resolve(tok).pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer has no <pad> token; cannot pad batches")
    return int(pad_id)


def canonical_amino_acid_ids(tok: OplmTokenizerFast | None = None) -> Tensor:
    """Return the token ids of the 20 standard amino acids.

    Built by mapping each character of the canonical order through the
    tokenizer's vocab (expected to yield the contiguous block ``range(4, 24)``).
    This is the shared sampling pool for random-token replacement in the MLM
    collator and is reused by eval metrics; deriving it from the tokenizer keeps
    it from drifting from the vocabulary.

    Args:
        tok: Tokenizer to query; defaults to the shared instance.

    Returns:
        A ``(20,)`` ``torch.long`` tensor of canonical amino-acid ids.
    """
    resolved = _resolve(tok)
    ids = [resolved.convert_tokens_to_ids(aa) for aa in _CANONICAL_AA_ORDER]
    return torch.tensor(ids, dtype=torch.long)


def align_per_residue(
    values: Sequence[Sequence[float] | None],
    *,
    lengths: Sequence[int],
    total_len: int,
    fill_special: float = 0.0,
    fill_pad: float = 0.0,
) -> Tensor:
    """Align raw per-residue vectors to tokenized ``input_ids``.

    Each raw vector carries one value per residue of a sequence. Tokenization
    inserts ``<cls>``/``<eos>``, truncates to ``max_length - 2`` residues, and
    pads, so the raw vector must undergo the same transform to stay positionally
    aligned with ``input_ids``. For row ``i`` (raw residue length ``lengths[i]``)
    the output row is laid out as::

        [ <cls> | residue values (truncated) | <eos> | padding ]
          fill_special   aligned to tokens     fill_special  fill_pad

    The truncation mirrors the token rule exactly: residues are clipped to
    ``min(lengths[i], total_len - 2)``. Because ``total_len`` is the batch's
    padded width (``max_j(clipped_len_j) + 2``), this reproduces the per-row
    clipped length without needing ``max_length`` here.

    Args:
        values: Per-row raw per-residue vectors, or ``None`` for a row with no
            values. ``len(values)`` must equal ``len(lengths)``.
        lengths: Per-row *raw* (pre-truncation) residue lengths,
            ``len(sequence_i)``.
        total_len: Output width ``T`` (the padded tokenized length).
        fill_special: Value written at ``<cls>`` and ``<eos>`` positions.
        fill_pad: Value written at trailing padding positions.

    Returns:
        A ``(B, total_len)`` ``torch.float32`` tensor aligned to ``input_ids``.

    Raises:
        ValueError: If ``len(values) != len(lengths)``, or a non-``None`` row's
            length disagrees with its ``lengths[i]`` (before truncation).
    """
    if len(values) != len(lengths):
        raise ValueError(f"values/lengths size mismatch: {len(values)} != {len(lengths)}")

    batch_size = len(values)
    # Start from the pad fill; cls/eos/residue positions are overwritten below.
    out = torch.full((batch_size, total_len), float(fill_pad), dtype=torch.float32)

    for i, (row_values, raw_len) in enumerate(zip(values, lengths, strict=True)):
        clipped_len = min(raw_len, total_len - 2)  # residues surviving truncation
        eos_pos = 1 + clipped_len
        out[i, 0] = fill_special  # <cls>
        out[i, eos_pos] = fill_special  # <eos>

        if row_values is None:
            # Missing weights -> uniform: every residue equally maskable.
            out[i, 1:eos_pos] = 1.0
            continue

        if len(row_values) != raw_len:
            raise ValueError(
                f"per-residue values length {len(row_values)} disagrees with "
                f"sequence length {raw_len} at row {i}"
            )
        if clipped_len > 0:
            clipped = torch.as_tensor(row_values[:clipped_len], dtype=torch.float32)
            out[i, 1:eos_pos] = clipped

    return out

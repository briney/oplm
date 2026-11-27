"""Tokenization, padding, and masked-language-model collation.

``tokenize_and_pad`` is the shared pad/tokenize primitive (no masking).
:class:`MLMCollator` layers fixed-``k`` Gumbel-top-k masking with BERT 80/10/10
replacement on top, supporting optional per-residue weighted masking.

The masking scheme (docs/data.md, Masking) is **dynamic** (RoBERTa-style):
masks are drawn fresh each call, so the same sequence is masked differently across
epochs. Evaluation freezes them via ``deterministic=True`` — a generator per row, seeded
from the sequence, *not* a separate collator class (docs/data.md, Deterministic eval
masking). All token-id constants are derived from the tokenizer (``data/tokenizer.py``),
never hardcoded.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from oplm.data.tokenizer import (
    align_per_residue,
    canonical_amino_acid_ids,
    mask_token_id,
    non_maskable_ids,
    pad_token_id,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from oplm.model import OplmTokenizerFast

logger = logging.getLogger(__name__)

# Cross-entropy ignore index for non-target positions (PyTorch default).
_IGNORE_INDEX = -100

# Per-residue masking-weight field carried on row dicts (see dataset.py).
_WEIGHTS_KEY = "masking_weights"
# Optional teacher-distillation binary payload fields carried on row dicts (see dataset.py).
_TEACHER_LOGITS_KEY = "teacher_logits"
_TEACHER_CONTACTS_KEY = "teacher_contacts"
_TEACHER_CHANNELS = 20
_TEACHER_BYTES_PER_RESIDUE = _TEACHER_CHANNELS * 2  # little-endian float16
CONTACT_FORMAT_VERSION = 1
_CONTACT_HEADER_STRUCT = struct.Struct("<BBH")  # (version, stored_min_sep, stored_len)

__all__ = [
    "CONTACT_FORMAT_VERSION",
    "MLMCollator",
    "RowFiller",
    "decode_teacher_contacts",
    "encode_teacher_contacts",
    "first_fit",
    "token_length",
    "tokenize_and_pad",
]


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


def _sequence_id_of(item: Mapping[str, object] | str) -> str:
    """Extract the ``sequence_id`` from a batch item (for ``keep_sequence_ids=True``).

    Args:
        item: A mapping with a ``"sequence_id"`` key.

    Returns:
        The row's ``sequence_id``.

    Raises:
        KeyError: If ``item`` is a raw ``str`` (which carries no id) or a mapping
            missing the ``"sequence_id"`` key.
    """
    if isinstance(item, str):
        raise KeyError("keep_sequence_ids=True requires mapping batch items, got a raw str")
    if isinstance(item, Mapping):
        if "sequence_id" not in item:
            raise KeyError("batch item mapping is missing required 'sequence_id' key")
        return str(item["sequence_id"])
    raise KeyError(
        f"keep_sequence_ids=True requires mapping batch items, got {type(item).__name__}"
    )


def tokenize_and_pad(
    batch: Sequence[Mapping[str, object] | str],
    tokenizer: OplmTokenizerFast,
    max_length: int,
    *,
    weights: Sequence[Sequence[float] | None] | None = None,
    pad_to_multiple_of: int | None = None,
) -> dict[str, Tensor]:
    """Tokenize and pad a batch of sequences (no masking, no labels).

    Each raw sequence is truncated to ``max_length - 2`` residues (leaving room for
    ``<cls>``/``<eos>``), tokenized with the canonical tokenizer, and padded to the
    smallest multiple of ``pad_to_multiple_of`` that is ≥ the batch's longest member
    (or exactly the batch's longest member when ``pad_to_multiple_of`` is ``None``).
    This is the shared primitive used by the variant/structure/downstream consumers
    and, internally, by :class:`MLMCollator`.

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
        pad_to_multiple_of: When set, pads sequences to the smallest multiple of this
            value that is ≥ the batch's longest member. ``None`` (default) pads to
            exactly the batch's longest member, reproducing today's behavior.

    Returns:
        ``{"input_ids": (B, T) long, "attention_mask": (B, T) long}``, plus
        ``"masking_weights": (B, T) float32`` when ``weights`` is supplied.
    """
    sequences = [_sequence_of(item) for item in batch]
    max_residues = max_length - 2  # room for <cls>/<eos>
    truncated = [seq[:max_residues] for seq in sequences]

    encoded = tokenizer(
        truncated, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt"
    )
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


class RowFiller:
    """First-fit placement of token lengths, in order, into ``rows`` rows of ``width`` tokens.

    The token batcher (:class:`~oplm.data.sequence.batching.TokenBatcher`) and the collator's
    fixed-shape packing both place sequences through this class, so they cannot disagree about
    where a batch closes or which row a sequence lands in.
    """

    def __init__(self, rows: int, width: int) -> None:
        self._free = [width] * rows
        self.placement: list[int] = []

    def place(self, length: int) -> bool:
        """Put ``length`` in the first row with room; return ``False``, placing nothing, if none."""
        for row, room in enumerate(self._free):
            if length <= room:
                self._free[row] -= length
                self.placement.append(row)
                return True
        return False


def first_fit(lengths: Sequence[int], rows: int, width: int) -> list[int] | None:
    """Each length's row under :class:`RowFiller`, or ``None`` if one fits in no row."""
    filler = RowFiller(rows, width)
    return filler.placement if all(filler.place(n) for n in lengths) else None


def token_length(sequence: str, width: int) -> int:
    """Tokens a sequence takes in a row of ``width``: up to ``width - 2`` residues + 2 specials."""
    return min(len(sequence), width - 2) + 2


class MLMCollator:
    """Masked-language-model collator: tokenize/pad, then mask.

    Wraps :func:`tokenize_and_pad` and applies the OPLM masking scheme
    (docs/data.md, Masking): per row, a **fixed count**
    ``k = round(mask_prob * n_eligible)`` of eligible positions is sampled without
    replacement via **Gumbel-top-k**, then the BERT 80/10/10 split decides each
    masked token's replacement (``<mask>`` / random canonical AA / keep original).
    Uniform masking is the equal-weights special case of the same code path.

    Masking is dynamic by default (fresh each call). With ``deterministic=True`` each
    row gets its own generator seeded from ``seed`` and the row's sequence, so a
    sequence's mask doesn't depend on which batch, worker or GPU it lands in, and the
    global RNG is left alone — the evaluation policy.

    Args:
        tokenizer: The canonical :class:`~oplm.model.OplmTokenizerFast`.
        max_length: Maximum tokenized length passed to :func:`tokenize_and_pad`.
        mask_prob: Fraction of eligible positions selected for masking per row.
        mask_token_prob: Of masked positions, fraction replaced with ``<mask>``.
        random_token_prob: Of masked positions, fraction replaced with a random
            canonical amino acid. The remainder keep their original id.
        weighted_masking: Honor each row's ``masking_weights`` to bias selection.
            When ``False`` the column is ignored and masking is uniform.
        deterministic: Freeze each sequence's mask (evaluation policy).
        seed: Base seed for the deterministic per-row generators.
        pad_to_multiple_of: Forwarded to :func:`tokenize_and_pad`. When set, the
            sequence axis is padded to the smallest multiple of this value ≥ the
            batch's longest member. ``None`` (default) pads to exactly the batch's
            longest member, reproducing today's behavior.
        keep_sequence_ids: Surface each row's ``sequence_id`` as a plain
            ``list[str]`` under ``"sequence_ids"`` in the returned batch dict
            (excluded from tensor collation). Off by default; used by tests and
            tooling that need to trace which rows a dataloader actually served
            (e.g. the double-sharding verification test) rather than by training.
        pack_sequences: After masking, pack the batch's sequences back to back into as
            few rows of at most ``max_length`` tokens as first-fit-decreasing finds (or
            exactly ``pack_rows`` rows, see below), and add ``"segment_ids"`` ``(rows, T)``
            (``1..n`` per sequence in a row, ``0`` at pads) and ``"num_sequences"`` (a plain
            ``int``, the batch size before packing). Masks and labels are those the
            unpacked batch would get.
        pack_rows: With ``pack_sequences``: pack into exactly this many rows of
            ``max_length`` (token mode), laid out by :func:`first_fit` in input order;
            raises if the sequences do not fit. ``None`` keeps first-fit-decreasing into
            as few rows as needed.
        load_teacher_logits: Decode each row's optional ``teacher_logits`` binary payload
            at masked canonical positions into ``"teacher_logits"`` ``(R, T, 20)`` float16
            and ``"teacher_mask"`` ``(R, T)`` bool batch tensors (packed alongside the
            other tensors when ``pack_sequences=True``). Off by default.

    Raises:
        ValueError: If any probability is outside ``[0, 1]``,
            ``mask_token_prob + random_token_prob > 1``, or ``pack_rows`` is set without
            ``pack_sequences`` or below 1.
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
        pad_to_multiple_of: int | None = None,
        keep_sequence_ids: bool = False,
        pack_sequences: bool = False,
        pack_rows: int | None = None,
        load_teacher_logits: bool = False,
        load_teacher_contacts: bool = False,
        contact_min_sep: int = 6,
    ) -> None:
        _validate_probs(mask_prob, mask_token_prob, random_token_prob)
        if pack_rows is not None and (not pack_sequences or pack_rows < 1):
            raise ValueError(
                f"pack_rows requires pack_sequences=True and pack_rows >= 1, got {pack_rows}"
            )
        if isinstance(contact_min_sep, bool) or contact_min_sep < 1:
            raise ValueError(f"contact_min_sep must be an int >= 1, got {contact_min_sep!r}")
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._mask_prob = mask_prob
        self._mask_token_prob = mask_token_prob
        self._random_token_prob = random_token_prob
        self._weighted_masking = weighted_masking
        self._deterministic = deterministic
        self._seed = seed
        self._pad_to_multiple_of = pad_to_multiple_of
        self._keep_sequence_ids = keep_sequence_ids
        self._pack_sequences = pack_sequences
        self._pack_rows = pack_rows
        self._load_teacher_logits = load_teacher_logits
        self._load_teacher_contacts = load_teacher_contacts
        self._contact_min_sep = contact_min_sep

        # Derived id constants (computed once from the tokenizer, never hardcoded).
        self._mask_token_id = mask_token_id(tokenizer)
        self._pad_token_id = pad_token_id(tokenizer)
        self._canonical_ids = canonical_amino_acid_ids(tokenizer)  # (20,) long
        self._non_maskable = torch.tensor(sorted(non_maskable_ids(tokenizer)), dtype=torch.long)

        self._warned_missing_weights = False

    def __call__(self, batch: Sequence[Mapping[str, object] | str]) -> dict[str, Any]:
        """Collate ``batch`` into ``{input_ids, attention_mask, labels}``.

        Also carries ``"sequence_ids"`` (a plain ``list[str]``, not collated into a
        tensor) when ``keep_sequence_ids=True``.
        """
        generators = self._row_generators(batch)

        raw_weights = self._collect_weights(batch) if self._weighted_masking else None
        encoded = tokenize_and_pad(
            batch,
            self._tokenizer,
            self._max_length,
            weights=raw_weights,
            pad_to_multiple_of=self._pad_to_multiple_of,
        )
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

        self._mask_batch(input_ids, labels, eligible, weights, generators, attention_mask)
        raw_lengths: list[int] | None = None
        if self._load_teacher_logits or self._load_teacher_contacts:
            raw_lengths = [len(_sequence_of(item)) for item in batch]
        teacher_logits: Tensor | None = None
        teacher_mask: Tensor | None = None
        if self._load_teacher_logits:
            assert raw_lengths is not None
            teacher_logits, teacher_mask = _collect_teacher_logits(
                batch, input_ids, labels, self._canonical_ids, self._mask_token_id, raw_lengths
            )
        # Per-input-row contact entries: row_idx -> (seg_res_len, contact_len, uint8_1d)
        row_contacts: dict[int, tuple[int, int, np.ndarray]] | None = None
        if self._load_teacher_contacts:
            assert raw_lengths is not None
            seq_tok_lens = attention_mask.sum(1).tolist()
            row_contacts = _collect_teacher_contacts(
                batch, raw_lengths, seq_tok_lens, self._contact_min_sep
            )
        out: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if teacher_logits is not None and teacher_mask is not None:
            out["teacher_logits"] = teacher_logits
            out["teacher_mask"] = teacher_mask
        if self._pack_sequences:
            if self._pack_rows is not None:
                expected = [token_length(_sequence_of(item), self._max_length) for item in batch]
                if attention_mask.sum(1).tolist() != expected:
                    raise ValueError(
                        "tokenized lengths differ from token_length(); the token batcher's "
                        "placement no longer matches the batch"
                    )
            out = self._pack(
                input_ids,
                attention_mask,
                labels,
                teacher_logits=teacher_logits,
                teacher_mask=teacher_mask,
                row_contacts=row_contacts,
            )
        elif row_contacts is not None:
            meta, values = _finalize_contact_batch(
                [
                    (row_idx, 0, seg_res_len, contact_len, arr)
                    for row_idx, (seg_res_len, contact_len, arr) in row_contacts.items()
                ]
            )
            out["teacher_contact_meta"] = meta
            out["teacher_contact_values"] = values
        if self._keep_sequence_ids:
            out["sequence_ids"] = self._collect_sequence_ids(batch)
        return out

    def _pack(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        *,
        teacher_logits: Tensor | None = None,
        teacher_mask: Tensor | None = None,
        row_contacts: dict[int, tuple[int, int, np.ndarray]] | None = None,
    ) -> dict[str, Any]:
        """Pack right-padded rows back to back into rows of at most ``max_length`` tokens.

        Without ``pack_rows``: first-fit-decreasing into as few rows as needed, trimmed to
        the longest row. With it: :func:`first_fit` in input order into exactly
        ``(pack_rows, max_length)``.
        """
        lengths = attention_mask.sum(1).tolist()
        rows: list[list[int]]  # batch indices per packed row
        if self._pack_rows is not None:
            placement = first_fit(lengths, self._pack_rows, self._max_length)
            if placement is None:
                raise ValueError(
                    f"{len(lengths)} sequences ({sum(lengths)} tokens) do not fit "
                    f"{self._pack_rows} rows of {self._max_length}"
                )
            rows = [
                [i for i, r in enumerate(placement) if r == row] for row in range(self._pack_rows)
            ]
            width = self._max_length
        else:
            rows = []
            free: list[int] = []
            for i in sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True):
                fit = next((r for r, room in enumerate(free) if lengths[i] <= room), None)
                if fit is None:
                    rows.append([i])
                    free.append(self._max_length - lengths[i])
                else:
                    rows[fit].append(i)
                    free[fit] -= lengths[i]
            width = self._max_length - min(free)
            if self._pad_to_multiple_of:
                width = -(-width // self._pad_to_multiple_of) * self._pad_to_multiple_of
        shape = (len(rows), width)
        packed: dict[str, Any] = {
            "input_ids": input_ids.new_full(shape, self._pad_token_id),
            "attention_mask": attention_mask.new_zeros(shape),
            "labels": labels.new_full(shape, _IGNORE_INDEX),
            "segment_ids": attention_mask.new_zeros(shape),
        }
        if teacher_logits is not None and teacher_mask is not None:
            packed["teacher_logits"] = teacher_logits.new_zeros((*shape, teacher_logits.shape[-1]))
            packed["teacher_mask"] = teacher_mask.new_zeros(shape)
        contact_entries: list[tuple[int, int, int, int, np.ndarray]] = []
        for r, members in enumerate(rows):
            start = 0
            for segment, i in enumerate(members, start=1):
                end = start + lengths[i]
                packed["input_ids"][r, start:end] = input_ids[i, : lengths[i]]
                packed["attention_mask"][r, start:end] = 1
                packed["labels"][r, start:end] = labels[i, : lengths[i]]
                packed["segment_ids"][r, start:end] = segment
                if teacher_logits is not None and teacher_mask is not None:
                    packed["teacher_logits"][r, start:end] = teacher_logits[i, : lengths[i]]
                    packed["teacher_mask"][r, start:end] = teacher_mask[i, : lengths[i]]
                if row_contacts is not None and i in row_contacts:
                    seg_res_len, contact_len, arr = row_contacts[i]
                    contact_entries.append((r, start, seg_res_len, contact_len, arr))
                start = end
        if row_contacts is not None:
            meta, values = _finalize_contact_batch(contact_entries)
            packed["teacher_contact_meta"] = meta
            packed["teacher_contact_values"] = values
        return {**packed, "num_sequences": len(lengths)}

    def _collect_sequence_ids(self, batch: Sequence[Mapping[str, object] | str]) -> list[str]:
        """Pull each row's ``sequence_id`` from the batch, in row order.

        Raises:
            KeyError: If a mapping item lacks a ``"sequence_id"`` key, or an item
                is a raw ``str`` (which carries no id).
        """
        return [_sequence_id_of(item) for item in batch]

    def _row_generators(
        self, batch: Sequence[Mapping[str, object] | str]
    ) -> list[torch.Generator] | None:
        """One generator per row in deterministic mode, seeded from ``seed`` and the sequence.

        Dynamic mode returns ``None`` so sampling falls through to the global RNG and masks
        stay fresh.
        """
        if not self._deterministic:
            return None
        generators = []
        for item in batch:
            key = f"{self._seed}:{_sequence_of(item)}".encode()
            generator = torch.Generator()
            generator.manual_seed(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest()))
            generators.append(generator)
        return generators

    def _collect_weights(
        self, batch: Sequence[Mapping[str, object] | str]
    ) -> list[list[float] | None]:
        """Pull per-row ``masking_weights`` from the batch dicts.

        Rows without weights (non-mapping, or ``None``) surface as ``None`` and are
        treated as uniform by :func:`~oplm.data.tokenizer.align_per_residue`. If the
        whole batch lacks weights (column entirely absent), warn once and fall back
        to uniform masking (docs/data.md, Masking).
        """
        weights: list[list[float] | None] = []
        any_present = False
        for item in batch:
            value = (
                item.get(_WEIGHTS_KEY)  # ty: ignore[invalid-argument-type]  # Mapping[str, object]
                if isinstance(item, Mapping)
                else None
            )
            if value is not None:
                any_present = True
            weights.append(value)  # ty: ignore[invalid-argument-type]  # list[float] | None

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
        generators: list[torch.Generator] | None,
        attention_mask: Tensor,
    ) -> None:
        """Select and replace masked positions in place, row by row.

        Selection is Gumbel-top-k: ``key = log(weight) + gumbel`` and the top
        ``k_b`` keys are masked. ``k_b`` is clamped to the count of positive-weight
        eligible positions, so uniform masking (all weights 1) is the special case.
        With per-row ``generators``, each row draws only over its own real tokens, so
        its mask doesn't depend on the batch's padded width.
        """
        # Gumbel noise g = -log(-log(u)), u ~ Uniform(0, 1). (B, T)
        if generators is None:
            u = torch.rand(input_ids.shape)
        else:
            u = torch.full(input_ids.shape, 0.5)  # padding: weight 0, never selected
            for b, n in enumerate(attention_mask.sum(dim=1).tolist()):
                u[b, :n] = torch.rand(n, generator=generators[b])
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
            self._replace(input_ids, b, masked_idx, None if generators is None else generators[b])

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


def _collect_teacher_logits(
    batch: Sequence[Mapping[str, object] | str],
    input_ids: Tensor,
    labels: Tensor,
    canonical_ids: Tensor,
    mask_token_id: int,
    raw_lengths: Sequence[int],
) -> tuple[Tensor, Tensor]:
    """Decode ``<mask>``-replaced canonical teacher logits into ``(B, T, 20)`` fp16 + bool mask.

    Each raw payload is little-endian float16 bytes of shape ``(raw_len, 20)`` where
    ``raw_len == len(sequence)``. Token index ``pos`` (1-based after ``<cls>``) maps to
    residue ``pos - 1`` in the raw payload, so truncated sequences stay aligned on their
    kept prefix. BERT kept-unchanged and random-AA replacement positions, non-canonical
    residues (e.g. ``X``), and any position with a non-finite teacher channel stay
    ``teacher_mask = False`` (plain CE) so query-masked teacher logits only supervise
    positions where the student actually sees ``<mask>``.
    """
    bsz, width = labels.shape
    teacher_logits = torch.zeros((bsz, width, _TEACHER_CHANNELS), dtype=torch.float16)
    teacher_mask = torch.zeros((bsz, width), dtype=torch.bool)
    eligible = (
        (input_ids == mask_token_id) & (labels != _IGNORE_INDEX) & torch.isin(labels, canonical_ids)
    )

    for row_idx, (item, raw_len) in enumerate(zip(batch, raw_lengths, strict=True)):
        if isinstance(item, str):
            continue
        raw = item.get(_TEACHER_LOGITS_KEY)
        if raw is None:
            continue
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise ValueError(f"teacher_logits must be bytes-like or None, got {type(raw).__name__}")
        nbytes = raw.nbytes if isinstance(raw, memoryview) else len(raw)
        expected_bytes = raw_len * _TEACHER_BYTES_PER_RESIDUE
        if nbytes != expected_bytes:
            raise ValueError(
                f"teacher_logits byte length ({nbytes}) disagrees with raw sequence length "
                f"{raw_len} (expected {expected_bytes} bytes for ({raw_len}, 20) float16)"
            )
        positions = eligible[row_idx].nonzero(as_tuple=False).squeeze(-1)
        if positions.numel() == 0:
            continue
        arr = np.frombuffer(raw, dtype="<f2").reshape(raw_len, _TEACHER_CHANNELS)
        res_idx = (positions - 1).numpy()
        picked = np.array(arr[res_idx], dtype=np.float16, copy=True)
        finite = np.isfinite(picked).all(axis=-1)
        if not finite.any():
            continue
        valid_pos = positions[torch.from_numpy(finite)]
        teacher_logits[row_idx, valid_pos] = torch.from_numpy(picked[finite])
        teacher_mask[row_idx, valid_pos] = True
    return teacher_logits, teacher_mask


def _expected_triu_pairs(length: int, min_sep: int) -> int:
    n = length - min_sep
    return 0 if n <= 0 else (n * (n + 1)) // 2


def encode_teacher_contacts(probs: np.ndarray, *, min_sep: int = 6) -> bytes:
    """Encode an ``(L, L)`` contact probability matrix into the v1 ``teacher_contacts`` payload.

    Layout: 4-byte little-endian header ``<BBH`` ``(version=1, stored_min_sep, stored_len)``
    followed by row-major ``uint8`` quantized upper-triangle entries at offset ``min_sep``
    (``round(clip(p, 0, 1) * 255)``).
    """
    arr = np.asarray(probs, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"probs must be a square 2D array, got shape {arr.shape}")
    if not (1 <= min_sep <= 255):
        raise ValueError(f"min_sep must be in [1, 255], got {min_sep}")
    length = int(arr.shape[0])
    if length > 65535:
        raise ValueError(f"sequence length {length} exceeds uint16 header limit (65535)")
    header = _CONTACT_HEADER_STRUCT.pack(CONTACT_FORMAT_VERSION, min_sep, length)
    if length <= min_sep:
        return header
    ii, jj = np.triu_indices(length, k=min_sep)
    quant = np.rint(np.clip(arr[ii, jj], 0.0, 1.0) * 255.0).astype(np.uint8)
    return header + quant.tobytes()


def decode_teacher_contacts(
    raw: bytes | bytearray | memoryview,
    raw_len: int,
    res_len: int,
    contact_min_sep: int,
) -> tuple[int, np.ndarray] | None:
    """Validate and decode a v1 ``teacher_contacts`` payload for ``res_len`` kept residues.

    Returns ``(contact_len, uint8_triu_values)`` matching ``np.triu_indices(contact_len,
    k=contact_min_sep)``, where ``contact_len = min(stored_len, res_len)``, or ``None``
    when ``contact_len <= contact_min_sep`` (zero scorable pairs).
    """
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise ValueError(f"teacher_contacts must be bytes-like or None, got {type(raw).__name__}")
    view = memoryview(raw)
    if view.nbytes < _CONTACT_HEADER_STRUCT.size:
        raise ValueError(f"teacher_contacts header truncated ({view.nbytes} bytes; expected >= 4)")
    version, stored_min_sep, stored_len = _CONTACT_HEADER_STRUCT.unpack_from(view, 0)
    if version != CONTACT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported teacher_contacts version {version}; expected {CONTACT_FORMAT_VERSION}"
        )
    if stored_min_sep < 1:
        raise ValueError(f"teacher_contacts stored min_sep must be >= 1, got {stored_min_sep}")
    if stored_min_sep > contact_min_sep:
        raise ValueError(
            f"teacher_contacts stored min_sep ({stored_min_sep}) exceeds requested "
            f"contact_min_sep ({contact_min_sep})"
        )
    if stored_len > raw_len:
        raise ValueError(
            f"teacher_contacts stored_len ({stored_len}) exceeds raw sequence length ({raw_len})"
        )
    expected_bytes = _CONTACT_HEADER_STRUCT.size + _expected_triu_pairs(stored_len, stored_min_sep)
    if view.nbytes != expected_bytes:
        raise ValueError(
            f"teacher_contacts byte length ({view.nbytes}) disagrees with header "
            f"(stored_len={stored_len}, stored_min_sep={stored_min_sep}, expected {expected_bytes})"
        )
    contact_len = min(stored_len, res_len)
    if contact_len <= contact_min_sep:
        return None
    body = np.frombuffer(view[_CONTACT_HEADER_STRUCT.size :], dtype=np.uint8)
    if contact_len == stored_len and contact_min_sep == stored_min_sep:
        return contact_len, np.array(body, dtype=np.uint8, copy=True)
    ii, jj = np.triu_indices(contact_len, k=contact_min_sep)
    span = stored_len - stored_min_sep
    flat_idx = ii * span - (ii * (ii - 1)) // 2 + (jj - ii - stored_min_sep)
    return contact_len, np.array(body[flat_idx], dtype=np.uint8, copy=True)


def _collect_teacher_contacts(
    batch: Sequence[Mapping[str, object] | str],
    raw_lengths: Sequence[int],
    seq_tok_lens: Sequence[int],
    contact_min_sep: int,
) -> dict[int, tuple[int, int, np.ndarray]]:
    """Decode each row's optional ``teacher_contacts`` payload keyed by batch row index."""
    out: dict[int, tuple[int, int, np.ndarray]] = {}
    for row_idx, (item, raw_len, tok_len) in enumerate(
        zip(batch, raw_lengths, seq_tok_lens, strict=True)
    ):
        if isinstance(item, str):
            continue
        raw = item.get(_TEACHER_CONTACTS_KEY)
        if raw is None:
            continue
        seg_res_len = max(0, tok_len - 2)
        decoded = decode_teacher_contacts(
            raw,  # ty: ignore[invalid-argument-type]  # validated inside decode_teacher_contacts
            raw_len=raw_len,
            res_len=seg_res_len,
            contact_min_sep=contact_min_sep,
        )
        if decoded is not None:
            contact_len, vals = decoded
            out[row_idx] = (seg_res_len, contact_len, vals)
    return out


def _finalize_contact_batch(
    entries: list[tuple[int, int, int, int, np.ndarray]],
) -> tuple[list[tuple[int, int, int, int, int]], Tensor]:
    """Sort contact segments by length ascending and concatenate into one ``uint8`` tensor."""
    if not entries:
        return [], torch.empty((0,), dtype=torch.uint8)
    entries.sort(key=lambda e: (e[3], e[2]))  # sort by (contact_len, seg_res_len) ascending
    meta: list[tuple[int, int, int, int, int]] = []
    chunks: list[np.ndarray] = []
    offset = 0
    for row_idx, start_pos, seg_res_len, contact_len, arr in entries:
        meta.append((row_idx, start_pos, seg_res_len, contact_len, offset))
        chunks.append(arr)
        offset += int(arr.size)
    combined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    return meta, torch.from_numpy(combined)


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

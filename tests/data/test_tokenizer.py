"""Tests for the tokenizer access layer (Phase 2).

The parity guard is the regression test that would have caught the legacy
off-by-one vocabulary bug (docs/DATA_TOOLING.md §3.5): every derived constant is
checked against the canonical :class:`~oplm.model.OplmTokenizerFast`. The
alignment tests pin :func:`~oplm.data.tokenizer.align_per_residue` to the
``<cls> … <eos>`` / truncation / padding rules it must mirror.
"""

from __future__ import annotations

import pytest
import torch

from oplm.data.tokenizer import (
    align_per_residue,
    canonical_amino_acid_ids,
    get_tokenizer,
    mask_token_id,
    non_maskable_ids,
    pad_token_id,
    special_ids,
)
from oplm.model import OplmTokenizerFast

# A battery of real human protein sequences (standard amino acids only).
_REAL_SEQUENCES = [
    "MEEPQ",  # p53 N-terminal fragment (the tokenizer's documented anchor)
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",  # ubiquitin
    "GIVEQCCTSICSLYQLENYCN",  # insulin A chain
    "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",  # insulin B chain
    "MALWMRLLPLLALLALWGPDPAAA",  # insulin signal peptide
]


# --------------------------------------------------------------------------- #
# Accessor
# --------------------------------------------------------------------------- #


def test_get_tokenizer_returns_canonical_singleton() -> None:
    """The accessor yields the canonical tokenizer and caches a singleton."""
    tok = get_tokenizer()
    assert isinstance(tok, OplmTokenizerFast)
    assert get_tokenizer() is tok  # cached


# --------------------------------------------------------------------------- #
# Parity guard (regression against vocabulary drift)
# --------------------------------------------------------------------------- #


def test_tokenizer_anchor_ids() -> None:
    """``MEEPQ`` tokenizes to the documented canonical ids."""
    assert get_tokenizer()("MEEPQ").input_ids == [0, 20, 9, 9, 14, 16, 2]


@pytest.mark.parametrize("seq", _REAL_SEQUENCES)
def test_ids_match_canonical_tokenizer(seq: str) -> None:
    """The accessor's ids equal a freshly constructed ``OplmTokenizerFast``'s."""
    assert get_tokenizer()(seq).input_ids == OplmTokenizerFast()(seq).input_ids


def test_special_and_mask_pad_ids() -> None:
    """Special-token constants match the canonical vocabulary."""
    assert special_ids() == {0, 1, 2, 3, 32}
    assert non_maskable_ids() == {0, 1, 2, 3, 32}
    assert non_maskable_ids() == special_ids()
    assert mask_token_id() == 32
    assert pad_token_id() == 1


def test_canonical_amino_acid_ids() -> None:
    """The 20 standard AAs occupy the contiguous block 4..23, as a long tensor."""
    ids = canonical_amino_acid_ids()
    assert ids.dtype == torch.long
    assert ids.tolist() == list(range(4, 24))


def test_constants_accept_explicit_tokenizer() -> None:
    """Constants honor an explicitly supplied tokenizer (collator's path)."""
    tok = OplmTokenizerFast()
    assert special_ids(tok) == {0, 1, 2, 3, 32}
    assert mask_token_id(tok) == 32
    assert pad_token_id(tok) == 1
    assert canonical_amino_acid_ids(tok).tolist() == list(range(4, 24))


# --------------------------------------------------------------------------- #
# align_per_residue
# --------------------------------------------------------------------------- #


def test_align_basic_layout() -> None:
    """A single full-length row is wrapped with fill_special at cls/eos."""
    out = align_per_residue([[1.0, 2.0, 3.0]], lengths=[3], total_len=5)
    assert out.dtype == torch.float32
    assert out.shape == (1, 5)
    assert out[0].tolist() == [0.0, 1.0, 2.0, 3.0, 0.0]


def test_align_padding_and_distinct_fills() -> None:
    """Specials get fill_special; trailing padding gets fill_pad."""
    out = align_per_residue(
        [[1.0, 2.0, 3.0], [5.0, 6.0]],
        lengths=[3, 2],
        total_len=5,
        fill_special=-1.0,
        fill_pad=-2.0,
    )
    assert out[0].tolist() == [-1.0, 1.0, 2.0, 3.0, -1.0]
    assert out[1].tolist() == [-1.0, 5.0, 6.0, -1.0, -2.0]


def test_align_lines_up_with_real_input_ids() -> None:
    """Weights land on residue positions; specials/pad of ``input_ids`` get fill."""
    tok = get_tokenizer()
    enc = tok(["MEEPQ", "AC"], padding=True)
    input_ids = enc.input_ids
    total_len = len(input_ids[0])  # 7

    weights = align_per_residue(
        [[10.0, 20.0, 30.0, 40.0, 50.0], [11.0, 22.0]],
        lengths=[5, 2],
        total_len=total_len,
        fill_special=-1.0,
        fill_pad=-2.0,
    )
    assert weights[0].tolist() == [-1.0, 10.0, 20.0, 30.0, 40.0, 50.0, -1.0]
    assert weights[1].tolist() == [-1.0, 11.0, 22.0, -1.0, -2.0, -2.0, -2.0]

    # Every special/pad position of input_ids carries a fill value; every real
    # residue carries its (non-negative) weight.
    specials = special_ids(tok)
    for row in range(len(input_ids)):
        for col in range(total_len):
            if input_ids[row][col] in specials:
                assert weights[row][col].item() in (-1.0, -2.0)
            else:
                assert weights[row][col].item() >= 0.0


def test_align_truncation_matches_token_rule() -> None:
    """Residues are clipped to ``total_len - 2`` (the token truncation rule)."""
    out = align_per_residue([[1.0, 2.0, 3.0, 4.0, 5.0]], lengths=[5], total_len=4)
    # clipped_len = min(5, 4 - 2) = 2; eos sits at position 3.
    assert out[0].tolist() == [0.0, 1.0, 2.0, 0.0]


def test_align_none_row_is_uniform() -> None:
    """A ``None`` row fills residue positions with uniform weight 1.0."""
    out = align_per_residue([None], lengths=[3], total_len=5)
    assert out[0].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]


def test_align_mixed_none_and_values() -> None:
    """``None`` and explicit-weight rows coexist in one batch."""
    out = align_per_residue(
        [None, [7.0, 8.0]],
        lengths=[3, 2],
        total_len=5,
        fill_special=-1.0,
        fill_pad=-2.0,
    )
    assert out[0].tolist() == [-1.0, 1.0, 1.0, 1.0, -1.0]
    assert out[1].tolist() == [-1.0, 7.0, 8.0, -1.0, -2.0]


def test_align_length_mismatch_raises() -> None:
    """A weight vector that disagrees with its sequence length is an error."""
    with pytest.raises(ValueError, match="disagrees with"):
        align_per_residue([[1.0, 2.0]], lengths=[3], total_len=5)


def test_align_batch_size_mismatch_raises() -> None:
    """``values`` and ``lengths`` must describe the same number of rows."""
    with pytest.raises(ValueError, match="size mismatch"):
        align_per_residue([[1.0, 2.0, 3.0]], lengths=[3, 3], total_len=5)

"""Tests for the pad/tokenize primitive and the MLM collator (Phase 4).

Covers the batch contract, fixed-``k`` Gumbel-top-k selection, the BERT 80/10/10
replacement split, RoBERTa-dynamic vs. evaluation-deterministic behavior, and the
optional per-residue weighted-masking path (docs/DATA_TOOLING.md §4.4-§4.6).

Sequences are real human proteins (standard amino acids only) so masked-position
ids always fall inside the canonical AA block; weighted-masking tests construct
synthetic per-residue weight vectors, which are a derived feature column rather
than biological data.
"""

from __future__ import annotations

import pytest
import torch

from oplm.data.sequence.collate import MLMCollator, tokenize_and_pad
from oplm.data.tokenizer import (
    canonical_amino_acid_ids,
    get_tokenizer,
    mask_token_id,
    non_maskable_ids,
)

# Real protein sequences (standard AAs only) of varied length.
_SEQUENCES = [
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP",  # p53 fragment, 60
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",  # ubiquitin, 76
    "GIVEQCCTSICSLYQLENYCN",  # insulin A chain, 21
    "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",  # insulin B chain, 30
]
_MASK_PROB = 0.15
_MAX_LENGTH = 128


def _batch(sequences: list[str]) -> list[dict[str, object]]:
    """Wrap raw sequences as dataset-style row dicts."""
    return [{"sequence_id": str(i), "sequence": s} for i, s in enumerate(sequences)]


def _expected_k(n_eligible: int, mask_prob: float = _MASK_PROB) -> int:
    """``k = round(mask_prob * n_eligible)`` — the per-row masked count."""
    return int(round(mask_prob * n_eligible))


@pytest.fixture(scope="module")
def tokenizer():  # noqa: ANN201 - pytest fixture
    """The canonical tokenizer (shared, read-only)."""
    return get_tokenizer()


# --------------------------------------------------------------------------- #
# tokenize_and_pad primitive
# --------------------------------------------------------------------------- #


def test_primitive_keys_shapes_dtypes(tokenizer) -> None:
    """The primitive returns exactly input_ids/attention_mask as (B,T) long."""
    out = tokenize_and_pad(_batch(_SEQUENCES), tokenizer, _MAX_LENGTH)
    assert set(out) == {"input_ids", "attention_mask"}
    b, t = out["input_ids"].shape
    assert b == len(_SEQUENCES)
    assert out["attention_mask"].shape == (b, t)
    assert out["input_ids"].dtype == torch.long
    assert out["attention_mask"].dtype == torch.long


def test_primitive_accepts_raw_strings(tokenizer) -> None:
    """A ``list[str]`` is accepted equivalently to row dicts."""
    from_dicts = tokenize_and_pad(_batch(_SEQUENCES), tokenizer, _MAX_LENGTH)
    from_strs = tokenize_and_pad(_SEQUENCES, tokenizer, _MAX_LENGTH)
    assert torch.equal(from_dicts["input_ids"], from_strs["input_ids"])


def test_primitive_truncates_to_max_length(tokenizer) -> None:
    """Sequences longer than ``max_length - 2`` are clipped before tokenizing."""
    long_seq = "A" * 500
    out = tokenize_and_pad([long_seq], tokenizer, max_length=64)
    assert out["input_ids"].shape[1] == 64  # <cls> + 62 residues + <eos>


def test_primitive_padding_matches_attention_mask(tokenizer) -> None:
    """Padding positions (id == pad) are exactly the zero attention-mask cells."""
    out = tokenize_and_pad(_batch(_SEQUENCES), tokenizer, _MAX_LENGTH)
    pad_id = tokenizer.pad_token_id
    is_pad = out["input_ids"] == pad_id
    assert torch.equal(is_pad, out["attention_mask"] == 0)


def test_primitive_optional_weights(tokenizer) -> None:
    """Supplying ``weights`` adds an aligned ``masking_weights`` (B,T) float tensor."""
    seqs = ["MEEP", "AC"]
    weights = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0]]
    out = tokenize_and_pad(seqs, tokenizer, _MAX_LENGTH, weights=weights)
    assert "masking_weights" in out
    mw = out["masking_weights"]
    assert mw.dtype == torch.float32
    assert mw.shape == out["input_ids"].shape
    # cls/eos/pad → 0.0; residues carry their weights.
    assert mw[0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 0.0]
    assert mw[1].tolist() == [0.0, 5.0, 6.0, 0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# Batch contract
# --------------------------------------------------------------------------- #


def test_batch_contract(tokenizer) -> None:
    """MLM batch has exactly the three documented keys, shapes, and dtypes."""
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, mask_prob=_MASK_PROB)
    out = collator(_batch(_SEQUENCES))
    assert set(out) == {"input_ids", "attention_mask", "labels"}
    b, t = out["input_ids"].shape
    assert b == len(_SEQUENCES)
    assert t <= _MAX_LENGTH
    for key in ("input_ids", "attention_mask", "labels"):
        assert out[key].shape == (b, t)
        assert out[key].dtype == torch.long
    # attention_mask still reflects padding.
    assert torch.equal(out["attention_mask"] == 0, out["input_ids"] == tokenizer.pad_token_id)


def test_no_masking_weights_key_emitted(tokenizer) -> None:
    """Even under weighted masking, masking_weights is consumed, not emitted."""
    weights = [[1.0] * len(s) for s in _SEQUENCES]
    batch = [
        {**row, "masking_weights": w} for row, w in zip(_batch(_SEQUENCES), weights, strict=True)
    ]
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, weighted_masking=True)
    out = collator(batch)
    assert "masking_weights" not in out


def test_keep_sequence_ids_off_by_default(tokenizer) -> None:
    """Without keep_sequence_ids, the batch dict has no sequence_ids key."""
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH)
    out = collator(_batch(_SEQUENCES))
    assert "sequence_ids" not in out


def test_keep_sequence_ids_surfaces_ids_in_row_order(tokenizer) -> None:
    """keep_sequence_ids=True adds a plain list of each row's sequence_id, in order."""
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, keep_sequence_ids=True)
    batch = _batch(_SEQUENCES)
    out = collator(batch)
    assert out["sequence_ids"] == [row["sequence_id"] for row in batch]
    assert isinstance(out["sequence_ids"], list)


def test_keep_sequence_ids_raises_on_raw_string_batch(tokenizer) -> None:
    """A raw-str batch item carries no id, so keep_sequence_ids=True raises."""
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, keep_sequence_ids=True)
    with pytest.raises(KeyError):
        collator(_SEQUENCES)


# --------------------------------------------------------------------------- #
# Fixed-k selection
# --------------------------------------------------------------------------- #


def test_fixed_k_per_row(tokenizer) -> None:
    """Exactly ``k = round(mask_prob * n_eligible)`` positions are masked per row."""
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, mask_prob=_MASK_PROB)
    out = collator(_batch(_SEQUENCES))
    # Eligibility is a property of the *pre-mask* ids; recomputing it from the
    # masked output would miscount (replaced <mask> tokens read as non-eligible).
    clean = tokenize_and_pad(_batch(_SEQUENCES), tokenizer, _MAX_LENGTH)
    nm = torch.tensor(sorted(non_maskable_ids(tokenizer)))
    for row in range(out["input_ids"].shape[0]):
        eligible = (clean["attention_mask"][row] == 1) & ~torch.isin(clean["input_ids"][row], nm)
        # labels mark masked positions; original ids recovered from labels.
        masked = out["labels"][row] != -100
        assert int(masked.sum()) == _expected_k(int(eligible.sum()))


def test_specials_and_pad_never_masked(tokenizer) -> None:
    """No special or padding position is ever selected as a target."""
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, mask_prob=0.5)
    nm = torch.tensor(sorted(non_maskable_ids(tokenizer)))
    for _ in range(20):
        out = collator(_batch(_SEQUENCES))
        masked = out["labels"] != -100
        # A masked label position must be a real, non-special original token.
        assert not torch.isin(out["labels"][masked], nm).any()
        assert (out["attention_mask"][masked] == 1).all()


def test_labels_minus_100_off_masked(tokenizer) -> None:
    """``labels`` is the original id at masked positions and -100 everywhere else."""
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, mask_prob=_MASK_PROB)
    # Reconstruct the pre-mask ids from a deterministic, unmasked tokenization.
    clean = tokenize_and_pad(_batch(_SEQUENCES), tokenizer, _MAX_LENGTH)["input_ids"]
    out = collator(_batch(_SEQUENCES))
    masked = out["labels"] != -100
    assert torch.equal(out["labels"][masked], clean[masked])
    assert (out["labels"][~masked] == -100).all()


# --------------------------------------------------------------------------- #
# 80/10/10 replacement split
# --------------------------------------------------------------------------- #


def test_replacement_split_proportions(tokenizer) -> None:
    """Over many batches, masked tokens split ~80/10/10 (mask/random/keep)."""
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, mask_prob=0.5, deterministic=False)
    mask_id = mask_token_id(tokenizer)
    n_mask = n_random = n_keep = 0
    clean = tokenize_and_pad(_batch(_SEQUENCES), tokenizer, _MAX_LENGTH)["input_ids"]
    for _ in range(200):
        out = collator(_batch(_SEQUENCES))
        masked = out["labels"] != -100
        ids = out["input_ids"][masked]
        original = clean[masked]
        n_mask += int((ids == mask_id).sum())
        kept = ids == original
        n_keep += int(kept.sum())
        n_random += int(((ids != mask_id) & ~kept).sum())
    total = n_mask + n_random + n_keep
    # Generous tolerances: random draws occasionally reproduce the original id,
    # inflating "keep" by ~1/20 of the random bucket.
    assert abs(n_mask / total - 0.8) < 0.03
    assert abs(n_random / total - 0.1) < 0.03
    assert abs(n_keep / total - 0.1) < 0.03


def test_random_replacements_are_canonical(tokenizer) -> None:
    """Every non-mask, non-kept masked token is a canonical amino-acid id."""
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, mask_prob=0.5)
    mask_id = mask_token_id(tokenizer)
    canonical = set(canonical_amino_acid_ids(tokenizer).tolist())
    for _ in range(50):
        out = collator(_batch(_SEQUENCES))
        masked = out["labels"] != -100
        ids = out["input_ids"][masked]
        # Drop <mask> tokens; the rest (random or kept) must be canonical AAs,
        # since the source sequences are standard AAs.
        non_mask = ids[ids != mask_id]
        assert set(non_mask.tolist()) <= canonical


# --------------------------------------------------------------------------- #
# Dynamic (RoBERTa) vs. deterministic (eval)
# --------------------------------------------------------------------------- #


def test_dynamic_masks_vary(tokenizer) -> None:
    """With ``deterministic=False`` the same sequence is masked differently."""
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, mask_prob=_MASK_PROB)
    seq_batch = _batch([_SEQUENCES[1]])  # one long sequence
    seen = set()
    for _ in range(10):
        out = collator(seq_batch)
        seen.add(tuple((out["labels"][0] != -100).nonzero().flatten().tolist()))
    assert len(seen) > 1  # masks are not frozen across calls


def test_deterministic_identical_for_same_batch_index(tokenizer) -> None:
    """Two deterministic collators yield identical masks for the same batch index."""
    a = MLMCollator(tokenizer, max_length=_MAX_LENGTH, deterministic=True, seed=7)
    b = MLMCollator(tokenizer, max_length=_MAX_LENGTH, deterministic=True, seed=7)
    out_a, out_b = a(_batch(_SEQUENCES)), b(_batch(_SEQUENCES))
    assert torch.equal(out_a["input_ids"], out_b["input_ids"])
    assert torch.equal(out_a["labels"], out_b["labels"])


def test_deterministic_advances_with_batch_index(tokenizer) -> None:
    """Successive deterministic calls differ (batch index advances the seed)."""
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, deterministic=True, seed=7)
    first = collator(_batch(_SEQUENCES))
    second = collator(_batch(_SEQUENCES))
    assert not torch.equal(first["input_ids"], second["input_ids"])


def test_deterministic_leaves_global_rng_untouched(tokenizer) -> None:
    """Deterministic masking does not disturb the global torch RNG."""
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, deterministic=True, seed=0)
    before = torch.random.get_rng_state()
    collator(_batch(_SEQUENCES))
    assert torch.equal(before, torch.random.get_rng_state())


# --------------------------------------------------------------------------- #
# Weighted masking
# --------------------------------------------------------------------------- #


def _weighted_batch(seq: str, weights: list[float]) -> list[dict[str, object]]:
    """A one-row batch carrying an explicit ``masking_weights`` vector."""
    return [{"sequence_id": "w", "sequence": seq, "masking_weights": weights}]


def test_weighted_keeps_fixed_count(tokenizer) -> None:
    """Weighted selection still masks exactly ``k`` positions."""
    seq = _SEQUENCES[0]
    weights = [1.0 + (i % 5) for i in range(len(seq))]
    collator = MLMCollator(
        tokenizer, max_length=_MAX_LENGTH, mask_prob=_MASK_PROB, weighted_masking=True
    )
    out = collator(_weighted_batch(seq, weights))
    assert int((out["labels"][0] != -100).sum()) == _expected_k(len(seq))


def test_weighted_inclusion_proportional_to_weight(tokenizer) -> None:
    """For ``k=1``, P(position selected) ∝ weight (exact Gumbel-max categorical)."""
    seq = "ACDEFGHIKL"  # 10 residues -> k = round(0.1 * 10) = 1
    weights = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, mask_prob=0.1, weighted_masking=True)
    counts = torch.zeros(len(seq))
    trials = 6000
    for _ in range(trials):
        out = collator(_weighted_batch(seq, weights))
        idx = (out["labels"][0] != -100).nonzero().flatten()
        assert idx.numel() == 1  # k = 1
        counts[int(idx) - 1] += 1  # token pos -> residue index
    freq = counts / trials
    expected = torch.tensor(weights) / sum(weights)
    # Monotone in weight and proportional (within sampling noise).
    assert torch.all(freq[2:] >= freq[:1])  # higher-weight residues masked more often
    assert torch.allclose(freq, expected, atol=0.02)


def test_weighted_scale_invariance(tokenizer) -> None:
    """Scaling all weights by a constant does not change the selection."""
    seq = _SEQUENCES[2]
    weights = [1.0 + (i % 4) for i in range(len(seq))]
    base = MLMCollator(
        tokenizer, max_length=_MAX_LENGTH, weighted_masking=True, deterministic=True, seed=3
    )
    scaled = MLMCollator(
        tokenizer, max_length=_MAX_LENGTH, weighted_masking=True, deterministic=True, seed=3
    )
    out_base = base(_weighted_batch(seq, weights))
    out_scaled = scaled(_weighted_batch(seq, [w * 17.0 for w in weights]))
    masked_base = out_base["labels"][0] != -100
    masked_scaled = out_scaled["labels"][0] != -100
    assert torch.equal(masked_base, masked_scaled)


def test_weighted_zero_weight_never_masked(tokenizer) -> None:
    """Zero-weight residues are never selected."""
    seq = _SEQUENCES[3]
    weights = [0.0] * len(seq)
    for i in (2, 7, 11):
        weights[i] = 1.0
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, mask_prob=0.5, weighted_masking=True)
    allowed = {i + 1 for i in (2, 7, 11)}  # residue index -> token position
    masked_positions: set[int] = set()
    for _ in range(50):
        out = collator(_weighted_batch(seq, weights))
        masked_positions.update((out["labels"][0] != -100).nonzero().flatten().tolist())
    assert masked_positions <= allowed


def test_weighted_positive_fewer_than_k_masks_all_positive(tokenizer) -> None:
    """When positive-weight positions < k, all of them (and only them) are masked."""
    seq = _SEQUENCES[0]  # len 60 -> k = round(0.5 * 60) = 30
    weights = [0.0] * len(seq)
    positive = {3, 10, 25, 40}  # only 4 positive positions, far below k
    for i in positive:
        weights[i] = 1.0
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, mask_prob=0.5, weighted_masking=True)
    out = collator(_weighted_batch(seq, weights))
    masked = set((out["labels"][0] != -100).nonzero().flatten().tolist())
    assert masked == {i + 1 for i in positive}


def test_weighted_masking_false_ignores_column(tokenizer) -> None:
    """With ``weighted_masking=False`` a present weight column is ignored."""
    seq = _SEQUENCES[2]
    weights = [0.0] * len(seq)
    weights[0] = 1.0  # would pin masking to one position if honored
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, mask_prob=0.5, weighted_masking=False)
    masked_positions: set[int] = set()
    for _ in range(30):
        out = collator(_weighted_batch(seq, weights))
        masked_positions.update((out["labels"][0] != -100).nonzero().flatten().tolist())
    # Uniform masking reaches many positions, not just the single positive weight.
    assert len(masked_positions) > 1


def test_weighted_absent_column_warns_once_and_uniform(
    tokenizer, caplog: pytest.LogCaptureFixture
) -> None:
    """Weighted masking with no weight column warns once and falls back to uniform."""
    collator = MLMCollator(
        tokenizer, max_length=_MAX_LENGTH, mask_prob=_MASK_PROB, weighted_masking=True
    )
    # Rows carry no masking_weights at all (column entirely absent).
    batch = _batch(_SEQUENCES)
    with caplog.at_level("WARNING"):
        out_first = collator(batch)
        collator(batch)
    warnings = [r for r in caplog.records if "masking_weights" in r.getMessage()]
    assert len(warnings) == 1  # warned once, not per batch
    # Still produces a valid fixed-k masked batch (uniform fallback). Eligibility
    # is read from the clean (pre-mask) tokenization to avoid miscounting <mask>.
    clean = tokenize_and_pad(batch, tokenizer, _MAX_LENGTH)
    nm = torch.tensor(sorted(non_maskable_ids(tokenizer)))
    eligible = (clean["attention_mask"][0] == 1) & ~torch.isin(clean["input_ids"][0], nm)
    masked = out_first["labels"][0] != -100
    assert int(masked.sum()) == _expected_k(int(eligible.sum()))


def test_weighted_length_mismatch_raises(tokenizer) -> None:
    """A weight vector whose length disagrees with the sequence is an error."""
    seq = _SEQUENCES[2]
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, weighted_masking=True)
    with pytest.raises(ValueError, match="disagrees with"):
        collator(_weighted_batch(seq, [1.0, 2.0]))  # too short


# --------------------------------------------------------------------------- #
# Construction validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("mask_token_prob", "random_token_prob"),
    [(1.2, 0.0), (0.0, -0.1), (0.7, 0.5)],
)
def test_invalid_probabilities_raise(tokenizer, mask_token_prob, random_token_prob) -> None:
    """Out-of-range or summing-over-one probabilities are rejected at construction."""
    with pytest.raises(ValueError):
        MLMCollator(
            tokenizer,
            mask_token_prob=mask_token_prob,
            random_token_prob=random_token_prob,
        )


# --------------------------------------------------------------------------- #
# pad_to_multiple_of
# --------------------------------------------------------------------------- #


def test_primitive_pad_to_multiple_of(tokenizer) -> None:
    """With pad_to_multiple_of=16, T is divisible by 16 and >= longest unpadded member."""
    multiple = 16
    out_unpadded = tokenize_and_pad(_batch(_SEQUENCES), tokenizer, _MAX_LENGTH)
    longest_unpadded = int(out_unpadded["input_ids"].shape[1])

    out = tokenize_and_pad(_batch(_SEQUENCES), tokenizer, _MAX_LENGTH, pad_to_multiple_of=multiple)
    t = out["input_ids"].shape[1]
    assert t % multiple == 0, f"T={t} is not divisible by {multiple}"
    assert t >= longest_unpadded, f"T={t} is shorter than unpadded longest {longest_unpadded}"


def test_primitive_pad_to_multiple_all_keys_share_t(tokenizer) -> None:
    """input_ids, attention_mask, and masking_weights all share the same T."""
    multiple = 16
    seqs = ["MEEP", "AC"]
    weights = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0]]
    out = tokenize_and_pad(seqs, tokenizer, _MAX_LENGTH, pad_to_multiple_of=multiple, weights=weights)
    t = out["input_ids"].shape[1]
    assert t % multiple == 0
    assert out["attention_mask"].shape[1] == t
    assert out["masking_weights"].shape[1] == t


def test_primitive_pad_to_multiple_none_reproduces_batch_max(tokenizer) -> None:
    """pad_to_multiple_of=None is byte-identical to today's batch-max behavior."""
    out_none = tokenize_and_pad(_batch(_SEQUENCES), tokenizer, _MAX_LENGTH, pad_to_multiple_of=None)
    out_default = tokenize_and_pad(_batch(_SEQUENCES), tokenizer, _MAX_LENGTH)
    assert torch.equal(out_none["input_ids"], out_default["input_ids"])
    assert torch.equal(out_none["attention_mask"], out_default["attention_mask"])


def test_collator_pad_to_multiple_of_shapes(tokenizer) -> None:
    """MLMCollator with pad_to_multiple_of=16: all output tensors share T divisible by 16."""
    multiple = 16
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, pad_to_multiple_of=multiple)
    out = collator(_batch(_SEQUENCES))
    t = out["input_ids"].shape[1]
    assert t % multiple == 0
    for key in ("input_ids", "attention_mask", "labels"):
        assert out[key].shape[1] == t, f"key={key} has shape {out[key].shape}, expected T={t}"


def test_collator_pad_to_multiple_of_with_weighted_masking(tokenizer) -> None:
    """With weighted masking + pad_to_multiple_of=16, masking_weights is consumed (not emitted);
    output tensors still share T divisible by 16."""
    multiple = 16
    weights = [[1.0] * len(s) for s in _SEQUENCES]
    batch = [
        {**row, "masking_weights": w} for row, w in zip(_batch(_SEQUENCES), weights, strict=True)
    ]
    collator = MLMCollator(
        tokenizer, max_length=_MAX_LENGTH, pad_to_multiple_of=multiple, weighted_masking=True
    )
    out = collator(batch)
    t = out["input_ids"].shape[1]
    assert t % multiple == 0
    assert "masking_weights" not in out
    for key in ("input_ids", "attention_mask", "labels"):
        assert out[key].shape[1] == t
    # Confirm masking actually fired on the weighted+pad path (not silently skipped).
    assert (out["labels"] != -100).any(), "Weighted+pad path must produce at least one masked position"


def test_collator_pad_masking_correctness(tokenizer) -> None:
    """Padded columns stay at ignore_index=-100 (labels) and real tokens have non-(-100) labels
    only at masked positions; the eligible-position contract holds on the non-pad region."""
    multiple = 16
    collator = MLMCollator(tokenizer, max_length=_MAX_LENGTH, pad_to_multiple_of=multiple)
    out = collator(_batch(_SEQUENCES))
    # Padded positions (attention_mask==0) must have labels==-100 (loss-neutral).
    pad_positions = out["attention_mask"] == 0
    assert (out["labels"][pad_positions] == -100).all(), "Padded positions must have ignore_index labels"
    # Every labeled (non-ignore_index) position must be a real, maskable token.
    # Concretely: labels != -100 implies attention_mask == 1 (real token, not pad)
    # AND the original token id (stored in labels by the collator) is not in the
    # non-maskable set (eligible for masking).
    # This catches masking leaking onto pad columns or special tokens.
    # Note: out["input_ids"] has been modified in-place (BERT 80/10/10 replacement),
    # so labels holds the original ids at masked positions — that's what we check.
    labeled = out["labels"] != -100
    non_maskable = collator._non_maskable
    original_ids_at_labeled = out["labels"][labeled]  # original token ids (not -100, not replaced)
    assert labeled.any(), "At least one position must be masked"
    assert (out["attention_mask"][labeled] == 1).all(), (
        "Labeled positions must be real tokens (attention_mask==1), not pad"
    )
    assert not torch.isin(original_ids_at_labeled, non_maskable).any(), (
        "Labeled positions must not be special/non-maskable tokens (eligible-position contract)"
    )


def test_collator_pad_to_multiple_none_matches_default(tokenizer) -> None:
    """MLMCollator(pad_to_multiple_of=None) is byte-identical to no argument."""
    collator_none = MLMCollator(
        tokenizer, max_length=_MAX_LENGTH, deterministic=True, seed=42, pad_to_multiple_of=None
    )
    collator_default = MLMCollator(
        tokenizer, max_length=_MAX_LENGTH, deterministic=True, seed=42
    )
    out_none = collator_none(_batch(_SEQUENCES))
    out_default = collator_default(_batch(_SEQUENCES))
    for key in ("input_ids", "attention_mask", "labels"):
        assert torch.equal(out_none[key], out_default[key]), f"key={key} differs"

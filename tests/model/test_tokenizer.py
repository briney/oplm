"""Tests for `oplm.model.tokenization_oplm` — OplmTokenizerFast."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from oplm.model import OplmTokenizerFast
from oplm.model.tokenization_oplm import VOCAB

_have_esm = importlib.util.find_spec("esm") is not None


@pytest.fixture(scope="module")
def tokenizer() -> OplmTokenizerFast:
    return OplmTokenizerFast()


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def test_vocab_size_and_special_ids():
    assert len(VOCAB) == 33
    assert VOCAB["<cls>"] == 0
    assert VOCAB["<pad>"] == 1
    assert VOCAB["<eos>"] == 2
    assert VOCAB["<unk>"] == 3
    assert VOCAB["<mask>"] == 32
    # id-31 is the chain-break token in ESM-C (was <null_1> in ESM-2).
    assert VOCAB["|"] == 31


def test_special_token_ids(tokenizer: OplmTokenizerFast):
    assert tokenizer.cls_token_id == 0
    assert tokenizer.pad_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.unk_token_id == 3
    assert tokenizer.mask_token_id == 32
    assert tokenizer.model_input_names == ["input_ids", "attention_mask"]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_canonical_sanity_check(tokenizer: OplmTokenizerFast):
    # The byte-identical-to-ESM-C example from §3.4 of the architecture doc.
    assert tokenizer("MEEPQ").input_ids == [0, 20, 9, 9, 14, 16, 2]


def test_per_character_split_no_merges(tokenizer: OplmTokenizerFast):
    # Every input character becomes exactly one token, wrapped by <cls>/<eos>.
    ids = tokenizer("MEEPQSDPSVEPPLSQ").input_ids
    assert ids[0] == 0 and ids[-1] == 2
    assert len(ids) == len("MEEPQSDPSVEPPLSQ") + 2


def test_unknown_chars_map_to_unk(tokenizer: OplmTokenizerFast):
    # Digits and symbols outside the AA alphabet -> <unk> (id 3).
    assert tokenizer("M*9").input_ids == [0, 20, 3, 3, 2]


def test_batch_padding(tokenizer: OplmTokenizerFast):
    out = tokenizer(["MEEPQ", "MEEPQSDPSV"], padding=True)
    lengths = {len(ids) for ids in out["input_ids"]}
    assert len(lengths) == 1  # all rows equal length after padding
    short = out["input_ids"][0]
    # The shorter sequence is right-padded with <pad> (id 1).
    assert short[-1] == tokenizer.pad_token_id
    assert out["attention_mask"][0][-1] == 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_pretrained_round_trip(tokenizer: OplmTokenizerFast, tmp_path: Path):
    tokenizer.save_pretrained(tmp_path)
    assert (tmp_path / "tokenizer.json").exists()
    config_path = tmp_path / "tokenizer_config.json"
    assert config_path.exists()
    # transformers >=5 folds special_tokens_map.json into tokenizer_config.json.
    config = json.loads(config_path.read_text())
    # HF persists the base class name and re-appends "Fast" on fast-load.
    assert config["tokenizer_class"] in {"OplmTokenizer", "OplmTokenizerFast"}

    reloaded = OplmTokenizerFast.from_pretrained(tmp_path)
    assert reloaded("MEEPQ").input_ids == [0, 20, 9, 9, 14, 16, 2]
    assert reloaded.get_vocab() == tokenizer.get_vocab()
    assert reloaded.mask_token_id == 32
    assert reloaded.cls_token_id == 0


# ---------------------------------------------------------------------------
# ESM-C parity (skipped when esm is unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _have_esm, reason="esm package not installed")
def test_esm_c_parity(tokenizer: OplmTokenizerFast):
    from esm.tokenization import EsmSequenceTokenizer

    esm_tok = EsmSequenceTokenizer()
    seqs = ["MEEPQSDPSVEPPLSQ", "GAGTRWPVQ"]
    for seq in seqs:
        assert tokenizer(seq).input_ids == esm_tok(seq)["input_ids"]

"""Tests for `oplm.model.tokenization_oplm` — OplmTokenizerFast."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from oplm.model import OplmTokenizerFast
from oplm.model.tokenization_oplm import VOCAB

if TYPE_CHECKING:
    from pathlib import Path

# The public ESM-2 tokenizer shares OPLM's exact vocabulary and ordering for
# every id except 31 (ESM-2 has `<null_1>` there; ESM-C — which OPLM follows —
# has the chain-break `|`). It loads through plain AutoTokenizer, so we can
# check parity without the heavy `esm` package. The true ESM-C tokenizer ships
# only inside `esm` and is not loadable from the hub via AutoTokenizer.
_ESM2_REPO = "facebook/esm2_t33_650M_UR50D"


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
# ESM parity (skipped when the hub tokenizer can't be downloaded, e.g. offline)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def esm2_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(_ESM2_REPO)
    except Exception as exc:  # network/hub unavailable -> skip, don't fail
        pytest.skip(f"could not load {_ESM2_REPO}: {exc}")


def test_input_ids_parity_with_esm(tokenizer: OplmTokenizerFast, esm2_tokenizer):
    # On real protein sequences (no `|`/`<null_1>`), OPLM is byte-identical to ESM.
    seqs = ["MEEPQ", "MEEPQSDPSVEPPLSQ", "GAGTRWPVQ"]
    for seq in seqs:
        assert tokenizer(seq).input_ids == esm2_tokenizer(seq)["input_ids"]


def test_vocab_parity_with_esm_except_chain_break(tokenizer: OplmTokenizerFast, esm2_tokenizer):
    esm_vocab = esm2_tokenizer.get_vocab()
    assert len(esm_vocab) == len(VOCAB) == 33
    differing = {i for tok, i in VOCAB.items() if esm_vocab.get(tok) != i}
    # The only intended divergence is id 31: ESM-C `|` vs ESM-2 `<null_1>`.
    assert differing == {31}
    assert VOCAB["|"] == 31
    assert esm_vocab["<null_1>"] == 31

"""Tests for the protein tokenizer."""

from __future__ import annotations

import pytest
import torch

from oplm.data.tokenizer import VOCAB, ProteinTokenizer


class TestProteinTokenizer:
    @pytest.fixture()
    def tokenizer(self) -> ProteinTokenizer:
        return ProteinTokenizer()

    def test_vocab_size(self, tokenizer: ProteinTokenizer) -> None:
        assert tokenizer.vocab_size == 32

    def test_special_token_ids(self, tokenizer: ProteinTokenizer) -> None:
        assert tokenizer.cls_token_id == 0
        assert tokenizer.pad_token_id == 1
        assert tokenizer.eos_token_id == 2
        assert tokenizer.mask_token_id == 4

    def test_encode_with_special_tokens(self, tokenizer: ProteinTokenizer) -> None:
        ids = tokenizer.encode("LAG")
        assert ids[0] == tokenizer.cls_token_id
        assert ids[-1] == tokenizer.eos_token_id
        assert len(ids) == 5  # cls + L + A + G + eos

    def test_encode_without_special_tokens(self, tokenizer: ProteinTokenizer) -> None:
        ids = tokenizer.encode("LAG", add_special_tokens=False)
        assert len(ids) == 3
        assert ids == [VOCAB["L"], VOCAB["A"], VOCAB["G"]]

    def test_unknown_character(self, tokenizer: ProteinTokenizer) -> None:
        ids = tokenizer.encode("L1A", add_special_tokens=False)
        assert ids[1] == VOCAB["<unk>"]

    def test_decode_strips_special_tokens(self, tokenizer: ProteinTokenizer) -> None:
        ids = tokenizer.encode("MKWV")
        decoded = tokenizer.decode(ids)
        assert decoded == "MKWV"

    def test_encode_decode_roundtrip(self, tokenizer: ProteinTokenizer) -> None:
        seq = "ACDEFGHIKLMNPQRSTVWY"
        decoded = tokenizer.decode(tokenizer.encode(seq))
        assert decoded == seq

    def test_decode_tensor(self, tokenizer: ProteinTokenizer) -> None:
        ids = torch.tensor(tokenizer.encode("LAG"))
        decoded = tokenizer.decode(ids)
        assert decoded == "LAG"

    def test_batch_encode_shapes(self, tokenizer: ProteinTokenizer) -> None:
        seqs = ["LAG", "MKWVTF"]
        result = tokenizer.batch_encode(seqs)
        assert result["input_ids"].shape == (2, 8)  # max is MKWVTF (6) + cls + eos = 8
        assert result["attention_mask"].shape == (2, 8)

    def test_batch_encode_padding(self, tokenizer: ProteinTokenizer) -> None:
        seqs = ["LA", "MKWV"]
        result = tokenizer.batch_encode(seqs)
        # Short sequence should be padded
        assert result["attention_mask"][0, -1].item() == 0  # padded position
        assert result["input_ids"][0, -1].item() == tokenizer.pad_token_id

    def test_batch_encode_max_length(self, tokenizer: ProteinTokenizer) -> None:
        seqs = ["ACDEFGHIKLMNPQRSTVWY"]
        result = tokenizer.batch_encode(seqs, max_length=5)
        assert result["input_ids"].shape[1] == 5

    def test_all_standard_amino_acids(self, tokenizer: ProteinTokenizer) -> None:
        """All 20 standard amino acids should encode without <unk>."""
        standard = "ACDEFGHIKLMNPQRSTVWY"
        ids = tokenizer.encode(standard, add_special_tokens=False)
        assert VOCAB["<unk>"] not in ids

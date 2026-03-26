"""ESM-compatible protein tokenizer."""

from __future__ import annotations

import torch
from torch import Tensor

# fmt: off
VOCAB: dict[str, int] = {
    "<cls>": 0,  "<pad>": 1,  "<eos>": 2,  "<unk>": 3,  "<mask>": 4,
    "L": 5,  "A": 6,  "G": 7,  "V": 8,  "S": 9,
    "E": 10, "R": 11, "T": 12, "I": 13, "D": 14,
    "P": 15, "K": 16, "Q": 17, "N": 18, "F": 19,
    "Y": 20, "M": 21, "H": 22, "W": 23, "C": 24,
    "B": 25, "U": 26, "Z": 27, "O": 28, "X": 29,
    ".": 30, "-": 31, "<null>": 32,
}
# fmt: on

ID_TO_TOKEN: dict[int, str] = {v: k for k, v in VOCAB.items()}


class ProteinTokenizer:
    """ESM-compatible protein tokenizer with 33 tokens.

    No external dependencies -- dict-based lookup, not sentencepiece/BPE.
    """

    def __init__(self) -> None:
        self._vocab = VOCAB
        self._id_to_token = ID_TO_TOKEN

    def encode(self, sequence: str, add_special_tokens: bool = True) -> list[int]:
        """Encode a protein sequence to token IDs.

        Args:
            sequence: Amino acid sequence (e.g. ``"MKWVTFISLLLLFSSAYS"``).
            add_special_tokens: Wrap with ``<cls>`` and ``<eos>``.

        Returns:
            List of integer token IDs.
        """
        unk_id = self._vocab["<unk>"]
        ids = [self._vocab.get(c, unk_id) for c in sequence]
        if add_special_tokens:
            ids = [self._vocab["<cls>"]] + ids + [self._vocab["<eos>"]]
        return ids

    def decode(self, token_ids: list[int] | Tensor) -> str:
        """Decode token IDs back to a sequence string.

        Special tokens (``<cls>``, ``<pad>``, ``<eos>``) are stripped.

        Args:
            token_ids: Integer token IDs.

        Returns:
            Decoded amino acid string.
        """
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.tolist()
        skip = {self._vocab["<cls>"], self._vocab["<pad>"], self._vocab["<eos>"]}
        return "".join(self._id_to_token.get(tid, "<unk>") for tid in token_ids if tid not in skip)

    def batch_encode(
        self,
        sequences: list[str],
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, Tensor]:
        """Encode a batch of sequences with padding.

        Args:
            sequences: List of amino acid strings.
            max_length: Truncate to this length (including special tokens).
            add_special_tokens: Wrap each sequence with ``<cls>`` and ``<eos>``.

        Returns:
            Dict with ``"input_ids"`` and ``"attention_mask"`` tensors,
            both of shape ``(B, T)``.
        """
        encoded = [self.encode(s, add_special_tokens) for s in sequences]
        if max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
        max_len = max(len(ids) for ids in encoded)

        input_ids = torch.full((len(encoded), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(len(encoded), max_len, dtype=torch.long)

        for i, ids in enumerate(encoded):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self._vocab)

    @property
    def pad_token_id(self) -> int:
        """Token ID for ``<pad>``."""
        return self._vocab["<pad>"]

    @property
    def mask_token_id(self) -> int:
        """Token ID for ``<mask>``."""
        return self._vocab["<mask>"]

    @property
    def cls_token_id(self) -> int:
        """Token ID for ``<cls>``."""
        return self._vocab["<cls>"]

    @property
    def eos_token_id(self) -> int:
        """Token ID for ``<eos>``."""
        return self._vocab["<eos>"]

"""OplmTokenizerFast — ESM-C-compatible 33-token WordLevel tokenizer.

OPLM uses the ESM-C 33-token vocabulary, in the same token order and with the
same special-token IDs (see docs/MODEL_ARCHITECTURE.md §3.1). A batch tokenized
for ESM-C is bit-for-bit identical to one tokenized for OPLM, so switching
between the two models is mechanical at the data layer.
"""

from __future__ import annotations

from typing import Any

from tokenizers import Regex, Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

__all__ = ["OplmTokenizerFast", "VOCAB"]


# The 33-token ESM-C vocabulary, in exact ID order (index == token id). The
# ordering and IDs must be bit-identical to ESM-C — see §3.1 of the model
# architecture doc. Note the id-31 slot is `|` (chain break) in ESM-C, where
# ESM-2 had `<null_1>`.
_VOCAB_TOKENS: tuple[str, ...] = (
    "<cls>",  # 0  special (BOS)
    "<pad>",  # 1  special
    "<eos>",  # 2  special
    "<unk>",  # 3  special
    "L",  # 4
    "A",  # 5
    "G",  # 6
    "V",  # 7
    "S",  # 8
    "E",  # 9
    "R",  # 10
    "T",  # 11
    "I",  # 12
    "D",  # 13
    "P",  # 14
    "K",  # 15
    "Q",  # 16
    "N",  # 17
    "F",  # 18
    "Y",  # 19
    "M",  # 20
    "H",  # 21
    "W",  # 22
    "C",  # 23
    "X",  # 24  ambiguous AA
    "B",  # 25  ambiguous AA (D/N)
    "U",  # 26  selenocysteine
    "Z",  # 27  ambiguous AA (E/Q)
    "O",  # 28  pyrrolysine
    ".",  # 29  gap marker
    "-",  # 30  alignment gap
    "|",  # 31  chain break
    "<mask>",  # 32  MLM mask
)

# Token -> id mapping (the WordLevel vocab).
VOCAB: dict[str, int] = {token: idx for idx, token in enumerate(_VOCAB_TOKENS)}

# Explicit special-token IDs from §3.1.
_CLS_TOKEN = "<cls>"
_PAD_TOKEN = "<pad>"
_EOS_TOKEN = "<eos>"
_UNK_TOKEN = "<unk>"
_MASK_TOKEN = "<mask>"


def _build_backend_tokenizer() -> Tokenizer:
    """Construct the backing fast ``tokenizers.Tokenizer`` programmatically.

    The tokenizer is a plain WordLevel model over the 33-token vocabulary with:

    * A ``Split`` pre-tokenizer using an empty regex with ``isolated`` behavior.
      This splits the raw string into individual characters — i.e. one token per
      input character, with no BPE merges and no SentencePiece. ``"MEEPQ"`` is
      pre-tokenized into ``["M", "E", "E", "P", "Q"]``.
    * A ``TemplateProcessing`` post-processor that wraps each sequence with
      ``<cls> ... <eos>`` (and the analogous wrapping for the pair input, which
      is not a production code path but is handled defensively).

    Returns:
        A configured ``tokenizers.Tokenizer`` ready to back a
        ``PreTrainedTokenizerFast``.
    """
    tokenizer = Tokenizer(WordLevel(vocab=dict(VOCAB), unk_token=_UNK_TOKEN))
    # Empty-regex isolated split == "split between every character". Unknown
    # single characters fall through to <unk> via the WordLevel unk_token.
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern=Regex(""), behavior="isolated")
    tokenizer.post_processor = TemplateProcessing(
        single=f"{_CLS_TOKEN} $A {_EOS_TOKEN}",
        pair=f"{_CLS_TOKEN} $A {_EOS_TOKEN} $B:1 {_EOS_TOKEN}:1",
        special_tokens=[(_CLS_TOKEN, VOCAB[_CLS_TOKEN]), (_EOS_TOKEN, VOCAB[_EOS_TOKEN])],
    )
    return tokenizer


class OplmTokenizerFast(PreTrainedTokenizerFast):
    """ESM-C-compatible per-character protein tokenizer.

    Wraps a programmatically constructed ``tokenizers.Tokenizer`` (WordLevel +
    per-character pre-tokenization + ``<cls> ... <eos>`` templating). When no
    serialized ``tokenizer_file`` is supplied, the backend is built from the
    33-token vocabulary defined in this module, so the class is usable with no
    on-disk artifacts:

        >>> tok = OplmTokenizerFast()
        >>> tok("MEEPQ").input_ids
        [0, 20, 9, 9, 14, 16, 2]
    """

    vocab_files_names = {"tokenizer_file": "tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str | None = None,
        tokenizer_file: str | None = None,
        *,
        cls_token: str = _CLS_TOKEN,
        pad_token: str = _PAD_TOKEN,
        eos_token: str = _EOS_TOKEN,
        unk_token: str = _UNK_TOKEN,
        mask_token: str = _MASK_TOKEN,
        **kwargs: Any,
    ) -> None:
        """Initialize the tokenizer.

        Args:
            vocab_file: Unused; accepted for HF API compatibility. The vocab is
                fixed and built into the package.
            tokenizer_file: Optional path to a serialized ``tokenizer.json``. When
                ``None`` (and no ``tokenizer_object`` is passed via ``kwargs``),
                the backend is constructed programmatically.
            cls_token: BOS / classification token (id 0).
            pad_token: Padding token (id 1).
            eos_token: End-of-sequence token (id 2).
            unk_token: Unknown token (id 3).
            mask_token: MLM mask token (id 32).
            **kwargs: Forwarded to ``PreTrainedTokenizerFast``.
        """
        if tokenizer_file is None and kwargs.get("tokenizer_object") is None:
            kwargs["tokenizer_object"] = _build_backend_tokenizer()

        super().__init__(
            tokenizer_file=tokenizer_file,
            cls_token=cls_token,
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
            mask_token=mask_token,
            **kwargs,
        )

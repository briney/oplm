"""ESM-C-compatible convenience API: tokenize / encode / logits (Phase 14.5)."""

from __future__ import annotations

import pytest
import torch

from oplm import LogitsConfig, OplmConfig, OplmForMaskedLM, OplmTokenizerFast

_VOCAB = 33


def _tiny_model() -> OplmForMaskedLM:
    config = OplmConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
    )
    return OplmForMaskedLM(config).eval()


@pytest.fixture
def model_with_tokenizer() -> OplmForMaskedLM:
    model = _tiny_model()
    model.tokenizer = OplmTokenizerFast()
    return model


def test_tokenize_without_tokenizer_raises_actionable() -> None:
    """A fresh model with no tokenizer raises a message naming both attach paths."""
    model = _tiny_model()
    assert model.tokenizer is None
    with pytest.raises(RuntimeError) as exc:
        model.tokenize(["MEEPQ"])
    msg = str(exc.value)
    assert "from_pretrained" in msg
    assert "model.tokenizer" in msg


def test_tokenize_returns_batch_encoding_on_device(model_with_tokenizer) -> None:
    model = model_with_tokenizer
    batch = model.tokenize(["MEEPQ", "GAGT"])
    assert "input_ids" in batch
    assert "attention_mask" in batch
    device = next(model.parameters()).device
    assert batch.input_ids.device == device
    assert batch.attention_mask.device == device
    # Padding to the longer of the two sequences.
    assert batch.input_ids.shape[0] == 2


def test_encode_returns_input_ids_only(model_with_tokenizer) -> None:
    """`encode` is the ESM-C footgun: input_ids with no mask.

    Parity against ESM-C token IDs is asserted in the tokenizer test (§11.3);
    here we only check that `encode` returns a bare `input_ids` tensor.
    """
    model = model_with_tokenizer
    ids = model.encode(["MEEPQ"])
    assert isinstance(ids, torch.Tensor)
    assert ids.shape[0] == 1
    assert ids.dim() == 2


def test_logits_carries_attention_mask(model_with_tokenizer) -> None:
    """`logits()` must match a direct masked forward bit-for-bit."""
    model = model_with_tokenizer
    seqs = ["MEEPQ", "GAGT"]
    out = model.logits(seqs, LogitsConfig(sequence=True, return_embeddings=True))

    batch = model.tokenize(seqs)
    t = batch.input_ids.shape[1]
    assert out.sequence_logits.shape == (2, t, _VOCAB)
    assert out.embeddings.shape == (2, t, model.config.hidden_size)

    with torch.no_grad():
        direct = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
    assert torch.equal(direct, out.sequence_logits)

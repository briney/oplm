"""End-to-end forward tests across all four public task classes (Phase 14.1).

These wire-level checks catch cross-component regressions that the per-module
tests miss: shape contracts, the hidden-states/attentions tuples, and loss
finiteness through the full graph.
"""

from __future__ import annotations

import pytest
import torch

from oplm import (
    OplmConfig,
    OplmForMaskedLM,
    OplmForSequenceClassification,
    OplmForTokenClassification,
    OplmModel,
)

_B, _T = 2, 16
_HIDDEN = 64
_LAYERS = 2
_HEADS = 4
_VOCAB = 33
_NUM_LABELS = 3


def _tiny_config(**overrides) -> OplmConfig:
    base = dict(
        hidden_size=_HIDDEN,
        num_hidden_layers=_LAYERS,
        num_attention_heads=_HEADS,
        max_position_embeddings=64,
        num_labels=_NUM_LABELS,
    )
    base.update(overrides)
    return OplmConfig(**base)


@pytest.fixture
def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    # Real-token IDs only (skip the special tokens 0..3) so masking is trivial.
    input_ids = torch.randint(4, _VOCAB, (_B, _T))
    attention_mask = torch.ones(_B, _T, dtype=torch.long)
    return input_ids, attention_mask


def test_backbone_forward_shape(inputs) -> None:
    input_ids, attention_mask = inputs
    model = OplmModel(_tiny_config()).eval()
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out.last_hidden_state.shape == (_B, _T, _HIDDEN)


def test_masked_lm_forward_shape(inputs) -> None:
    input_ids, attention_mask = inputs
    model = OplmForMaskedLM(_tiny_config()).eval()
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out.logits.shape == (_B, _T, _VOCAB)


def test_sequence_classification_forward_shape(inputs) -> None:
    input_ids, attention_mask = inputs
    model = OplmForSequenceClassification(_tiny_config()).eval()
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out.logits.shape == (_B, _NUM_LABELS)


def test_token_classification_forward_shape(inputs) -> None:
    input_ids, attention_mask = inputs
    model = OplmForTokenClassification(_tiny_config()).eval()
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out.logits.shape == (_B, _T, _NUM_LABELS)


def test_output_hidden_states_tuple_length(inputs) -> None:
    """`output_hidden_states=True` returns L + 1 states (post-embedding + per-block)."""
    input_ids, attention_mask = inputs
    model = OplmForMaskedLM(_tiny_config()).eval()
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    assert out.hidden_states is not None
    assert len(out.hidden_states) == _LAYERS + 1
    for hs in out.hidden_states:
        assert hs.shape == (_B, _T, _HIDDEN)


def test_output_attentions_fp32_and_normalized(inputs) -> None:
    """Each attention tensor is fp32 and rows sum to 1 over real positions."""
    input_ids, attention_mask = inputs
    model = OplmForMaskedLM(_tiny_config()).eval()
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
    )
    assert out.attentions is not None
    assert len(out.attentions) == _LAYERS
    for attn in out.attentions:
        assert attn.shape == (_B, _HEADS, _T, _T)
        assert attn.dtype == torch.float32
        sums = attn.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


@pytest.mark.parametrize(
    ("cls", "label_shape", "dtype"),
    [
        (OplmForMaskedLM, (_B, _T), torch.long),
        (OplmForSequenceClassification, (_B,), torch.long),
        (OplmForTokenClassification, (_B, _T), torch.long),
    ],
)
def test_loss_is_finite_with_labels(inputs, cls, label_shape, dtype) -> None:
    input_ids, attention_mask = inputs
    model = cls(_tiny_config()).eval()
    torch.manual_seed(1)
    labels = torch.randint(0, _NUM_LABELS, label_shape, dtype=dtype)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert out.loss is not None
    assert torch.isfinite(out.loss).item()

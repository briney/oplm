"""Tests for opt-in return_qk / return_qk_layers tap on OplmAttention..OplmForMaskedLM."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import MaskedLMOutput

from oplm.model import OplmConfig, OplmForMaskedLM
from oplm.model.modeling_oplm import OplmMaskedLMOutput


def _small_config(**overrides: object) -> OplmConfig:
    defaults: dict[str, object] = {
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_hidden_layers": 4,
        "intermediate_size": 64,
        "max_position_embeddings": 64,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
    }
    defaults.update(overrides)
    return OplmConfig(**defaults)  # ty: ignore[invalid-argument-type]


def test_return_qk_layers_parity_and_backward_compat() -> None:
    torch.manual_seed(0)
    model = OplmForMaskedLM(_small_config()).eval()
    input_ids = torch.randint(4, 24, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    segment_ids = torch.tensor(
        [[1] * 8 + [2] * 6 + [0] * 2, [1] * 10 + [2] * 6],
        dtype=torch.long,
    )

    base = model(input_ids=input_ids, attention_mask=attention_mask, segment_ids=segment_ids)
    assert type(base) is MaskedLMOutput

    tapped = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
        return_qk_layers=(2, 3),
    )
    assert isinstance(tapped, OplmMaskedLMOutput)
    assert isinstance(tapped, MaskedLMOutput)
    assert tapped.qk_states is not None
    assert len(tapped.qk_states) == 2
    for q, k in tapped.qk_states:
        assert q.shape == (2, 4, 16, 8)
        assert k.shape == (2, 4, 16, 8)
    torch.testing.assert_close(tapped.logits, base.logits, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("qk_norm_mode", ["channel", "l2"])
@pytest.mark.parametrize("canon_enabled", [False, True])
def test_return_qk_layers_matches_manual_attention_scores(
    qk_norm_mode: str, canon_enabled: bool
) -> None:
    torch.manual_seed(3)
    extra: dict[str, object] = {"qk_norm_mode": qk_norm_mode}
    if canon_enabled:
        extra.update(
            canon_enabled=True,
            canon_positions=["A", "B"],
            canon_kernel_sizes=[3, 3, 3, 3],
        )
    model = OplmForMaskedLM(_small_config(**extra)).eval()
    input_ids = torch.randint(4, 24, (1, 18))
    attention_mask = torch.ones_like(input_ids)
    segment_ids = torch.tensor([[1] * 7 + [2] * 9 + [0] * 2], dtype=torch.long)

    manual = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
        output_attentions=True,
    )
    tapped = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
        return_qk_layers=(1, 3),
    )
    assert manual.attentions is not None
    assert tapped.qk_states is not None
    # Segment 2 occupies positions [7:16]
    for qk_pair, layer_idx in zip(tapped.qk_states, (1, 3), strict=True):
        q, k = qk_pair
        scale = model.oplm.backbone.layers[layer_idx].attention.attn_scale
        q_seg = q[0, :, 7:16, :].float()
        k_seg = k[0, :, 7:16, :].float()
        attn_from_qk = F.softmax(torch.matmul(q_seg, k_seg.transpose(-2, -1)) * scale, dim=-1)
        attn_manual = manual.attentions[layer_idx][0, :, 7:16, 7:16].float()
        torch.testing.assert_close(attn_from_qk, attn_manual, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("ckpt_mode", ["full", "selective"])
@pytest.mark.parametrize("value_residual", ["none", "learnable"])
def test_return_qk_layers_gradients_under_checkpointing_and_loops(
    ckpt_mode: str, value_residual: str
) -> None:
    torch.manual_seed(7)
    cfg = _small_config(
        num_hidden_layers=4,
        num_loops=2,
        loop_strategy="stack",
        value_residual=value_residual,
        gradient_checkpointing=True,
        gradient_checkpointing_mode=ckpt_mode,
    )
    model = OplmForMaskedLM(cfg)
    model.gradient_checkpointing_enable({"mode": ckpt_mode})
    model.train()
    input_ids = torch.randint(4, 24, (2, 12))
    attention_mask = torch.ones_like(input_ids)

    # Effective depth = 8 (2 loops of 4 layers); tap final 2 executed passes (6, 7)
    out = model(input_ids=input_ids, attention_mask=attention_mask, return_qk_layers=(6, 7))
    assert out.qk_states is not None
    assert len(out.qk_states) == 2
    loss = sum((q.float().pow(2).mean() + k.float().pow(2).mean()) for q, k in out.qk_states)
    loss.backward()

    # Final two executed steps correspond to physical layers 2 and 3
    for idx in (0, 1, 2, 3):
        layer = model.oplm.backbone.layers[idx]
        assert layer.attention.q_proj.weight.grad is not None
        assert layer.attention.k_proj.weight.grad is not None
        assert torch.isfinite(layer.attention.q_proj.weight.grad).all()
        assert layer.attention.q_proj.weight.grad.abs().sum().item() > 0.0


def test_return_qk_layers_rejects_out_of_range_indices() -> None:
    cfg = _small_config(num_hidden_layers=2, num_loops=1)
    model = OplmForMaskedLM(cfg).eval()
    input_ids = torch.tensor([[0, 5, 6, 7, 2]])
    attention_mask = torch.ones_like(input_ids)
    with pytest.raises(ValueError, match="return_qk_layers"):
        model(input_ids=input_ids, attention_mask=attention_mask, return_qk_layers=(-1,))
    with pytest.raises(ValueError, match="return_qk_layers"):
        model(input_ids=input_ids, attention_mask=attention_mask, return_qk_layers=(2,))

"""Tests for `oplm.model.transformer` — OplmBlock and OplmStack."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from oplm.model.conv import CanonConv
from oplm.model.norm import OplmLayerNorm
from oplm.model.transformer import OplmBlock, OplmStack


def _config(
    *,
    hidden_size: int = 32,
    num_attention_heads: int = 4,
    head_dim: int | None = None,
    intermediate_size: int = 64,
    num_hidden_layers: int = 2,
    vocab_size: int = 33,
    max_position_embeddings: int = 64,
    rope_theta: float = 10000.0,
    rope_dim: int | None = None,
    norm_type: str = "layernorm",
    norm_eps: float = 1e-6,
    norm_strategy: str = "pre",
    qk_norm: bool = True,
    ffn_activation: str = "swiglu",
    ffn_bias: bool = False,
    attention_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    post_embed_norm: bool = False,
    residual_scaling: str = "sqrt_num_layers",
    gradient_checkpointing: bool = False,
    canon_enabled: bool = False,
    canon_positions: list[str] | None = None,
    canon_kernel_sizes: list[int] | None = None,
    canon_activation: str = "none",
) -> SimpleNamespace:
    head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
    rope_dim = rope_dim if rope_dim is not None else head_dim
    if canon_positions is None:
        canon_positions = []
    if canon_kernel_sizes is None:
        canon_kernel_sizes = [3] * num_hidden_layers
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        rope_dim=rope_dim,
        norm_type=norm_type,
        norm_eps=norm_eps,
        norm_strategy=norm_strategy,
        qk_norm=qk_norm,
        ffn_activation=ffn_activation,
        ffn_bias=ffn_bias,
        attention_dropout=attention_dropout,
        hidden_dropout=hidden_dropout,
        post_embed_norm=post_embed_norm,
        residual_scaling=residual_scaling,
        gradient_checkpointing=gradient_checkpointing,
        canon_enabled=canon_enabled,
        canon_positions=canon_positions,
        canon_kernel_sizes=canon_kernel_sizes,
        canon_activation=canon_activation,
    )


def _ones_mask(batch: int, seq: int) -> torch.Tensor:
    return torch.ones(batch, seq, dtype=torch.long)


# ---------------------------------------------------------------------------
# OplmBlock — construction
# ---------------------------------------------------------------------------


def test_block_alpha_uses_sqrt_num_layers():
    cfg = _config(num_hidden_layers=4, residual_scaling="sqrt_num_layers")
    block = OplmBlock(cfg, layer_idx=0)
    assert isinstance(block.alpha, torch.Tensor)
    assert block.alpha.item() == pytest.approx(1.0 / math.sqrt(4))


def test_block_alpha_is_one_under_no_residual_scaling():
    cfg = _config(residual_scaling="none")
    block = OplmBlock(cfg, layer_idx=0)
    assert isinstance(block.alpha, torch.Tensor)
    assert block.alpha.item() == pytest.approx(1.0)


def test_block_alpha_is_persistent_buffer():
    """alpha must be a persistent scalar tensor buffer, not a plain Python float.

    torch.compile + DDP (DDPOptimizer) lifts plain-float module attributes as
    graph inputs and may include them in subgraph outputs when partitioning;
    aot_autograd then fails with 'float has no attribute meta'.

    It must be persistent (saved in state_dict) so that HuggingFace's fast-init
    path (from_pretrained) restores the correct value: fast-init creates buffers
    with uninitialized memory and only overwrites persistent buffers from the
    checkpoint; a non-persistent alpha would survive as garbage (~0) after loading.
    """
    cfg = _config(num_hidden_layers=4, residual_scaling="sqrt_num_layers")
    block = OplmBlock(cfg, layer_idx=0)
    assert isinstance(block.alpha, torch.Tensor), "alpha must be a tensor buffer"
    assert block.alpha.ndim == 0, "alpha must be a scalar (0-dim tensor)"
    assert "alpha" in block.state_dict(), "alpha must be persistent (saved in state_dict)"


def test_block_rejects_unknown_residual_scaling():
    cfg = _config(residual_scaling="bogus")
    with pytest.raises(ValueError, match="residual_scaling"):
        OplmBlock(cfg, layer_idx=0)


def test_block_rejects_unknown_norm_strategy():
    cfg = _config(norm_strategy="zzz")
    with pytest.raises(ValueError, match="norm_strategy"):
        OplmBlock(cfg, layer_idx=0)


@pytest.mark.parametrize("strategy", ["pre", "sandwich", "post_sdpa"])
def test_block_has_attn_norm_for_non_hybrid_strategies(strategy: str):
    block = OplmBlock(_config(norm_strategy=strategy), layer_idx=0)
    assert isinstance(block.attn_norm, OplmLayerNorm)


def test_block_omits_attn_norm_under_hybrid():
    block = OplmBlock(_config(norm_strategy="hybrid"), layer_idx=0)
    assert not hasattr(block, "attn_norm")


def test_block_sandwich_adds_two_post_norms():
    block = OplmBlock(_config(norm_strategy="sandwich"), layer_idx=0)
    assert isinstance(block.attn_post_norm, OplmLayerNorm)
    assert isinstance(block.ffn_post_norm, OplmLayerNorm)


def test_block_post_sdpa_adds_attn_post_norm_only():
    block = OplmBlock(_config(norm_strategy="post_sdpa"), layer_idx=0)
    assert isinstance(block.attn_post_norm, OplmLayerNorm)
    assert not hasattr(block, "ffn_post_norm")


def test_block_pre_strategy_has_no_post_norms():
    block = OplmBlock(_config(norm_strategy="pre"), layer_idx=0)
    assert not hasattr(block, "attn_post_norm")
    assert not hasattr(block, "ffn_post_norm")


def test_block_hybrid_strategy_has_no_block_level_post_norms():
    block = OplmBlock(_config(norm_strategy="hybrid"), layer_idx=0)
    assert not hasattr(block, "attn_post_norm")
    assert not hasattr(block, "ffn_post_norm")


# ---------------------------------------------------------------------------
# OplmBlock — Canon wiring
# ---------------------------------------------------------------------------


def test_block_no_canon_when_disabled():
    cfg = _config(canon_enabled=False, canon_positions=["A", "B", "C", "D"])
    block = OplmBlock(cfg, layer_idx=0)
    for name in ("conv_a", "conv_b", "conv_c", "conv_d"):
        assert not hasattr(block, name)


@pytest.mark.parametrize("position", ["A", "B", "C", "D"])
def test_block_creates_only_requested_canon_position(position: str):
    cfg = _config(
        canon_enabled=True,
        canon_positions=[position],
        canon_kernel_sizes=[3, 3],
    )
    block = OplmBlock(cfg, layer_idx=0)
    name = f"conv_{position.lower()}"
    assert isinstance(getattr(block, name), CanonConv)
    for other in {"A", "B", "C", "D"} - {position}:
        assert not hasattr(block, f"conv_{other.lower()}")


def test_block_canon_kernel_size_comes_from_layer_idx():
    cfg = _config(
        num_hidden_layers=3,
        canon_enabled=True,
        canon_positions=["A"],
        canon_kernel_sizes=[2, 5, 7],
    )
    block_1 = OplmBlock(cfg, layer_idx=1)
    block_2 = OplmBlock(cfg, layer_idx=2)
    assert block_1.conv_a.kernel_size == 5
    assert block_2.conv_a.kernel_size == 7


def test_block_canon_rejects_bad_position():
    cfg = _config(canon_enabled=True, canon_positions=["Z"])
    with pytest.raises(ValueError, match="canon_positions"):
        OplmBlock(cfg, layer_idx=0)


def test_block_canon_rejects_unresolved_kernel_sizes():
    cfg = _config(
        num_hidden_layers=2,
        canon_enabled=True,
        canon_positions=["A"],
        canon_kernel_sizes=[3],  # length mismatch
    )
    with pytest.raises(ValueError, match="canon_kernel_sizes"):
        OplmBlock(cfg, layer_idx=0)


# ---------------------------------------------------------------------------
# OplmBlock — forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["pre", "sandwich", "hybrid", "post_sdpa"])
def test_block_forward_runs_under_every_norm_strategy(strategy: str):
    torch.manual_seed(0)
    cfg = _config(norm_strategy=strategy)
    block = OplmBlock(cfg, layer_idx=0)
    x = torch.randn(2, 5, cfg.hidden_size)
    out, attn = block(x, _ones_mask(2, 5))
    assert out.shape == x.shape
    # No output_attentions requested -> manual path still returns weights on CPU.
    assert attn is None or attn.shape == (2, cfg.num_attention_heads, 5, 5)


def test_block_forward_returns_attentions_when_requested():
    torch.manual_seed(1)
    cfg = _config()
    block = OplmBlock(cfg, layer_idx=0)
    x = torch.randn(2, 4, cfg.hidden_size)
    _, attn = block(x, _ones_mask(2, 4), output_attentions=True)
    assert attn is not None
    assert attn.shape == (2, cfg.num_attention_heads, 4, 4)


def test_block_forward_with_all_canon_positions_runs():
    torch.manual_seed(2)
    cfg = _config(
        canon_enabled=True,
        canon_positions=["A", "B", "C", "D"],
        canon_kernel_sizes=[3, 3],
    )
    block = OplmBlock(cfg, layer_idx=0)
    x = torch.randn(2, 6, cfg.hidden_size)
    out, _ = block(x, _ones_mask(2, 6))
    assert out.shape == x.shape


def test_block_forward_grad_flows_to_input():
    torch.manual_seed(3)
    cfg = _config()
    block = OplmBlock(cfg, layer_idx=0)
    x = torch.randn(2, 5, cfg.hidden_size, requires_grad=True)
    out, _ = block(x, _ones_mask(2, 5))
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# OplmBlock — gradient checkpointing
# ---------------------------------------------------------------------------


def test_block_gradient_checkpoint_matches_plain_forward():
    torch.manual_seed(0)
    cfg = _config(num_hidden_layers=2, gradient_checkpointing=False)
    block = OplmBlock(cfg, layer_idx=0)
    block.train()  # checkpointing only fires under .training

    x = torch.randn(2, 5, cfg.hidden_size, requires_grad=True)
    mask = _ones_mask(2, 5)

    block.gradient_checkpointing = False
    out_plain, _ = block(x, mask)

    block.gradient_checkpointing = True
    out_ckpt, _ = block(x, mask)

    assert torch.allclose(out_plain, out_ckpt, atol=1e-5)


def test_block_gradient_checkpoint_gradients_match_plain():
    torch.manual_seed(0)
    cfg = _config()
    block = OplmBlock(cfg, layer_idx=0)
    block.train()

    x = torch.randn(2, 5, cfg.hidden_size, requires_grad=True)
    mask = _ones_mask(2, 5)

    block.gradient_checkpointing = False
    out_plain, _ = block(x, mask)
    g_plain = torch.autograd.grad(out_plain.sum(), x, retain_graph=False)[0]

    block.gradient_checkpointing = True
    out_ckpt, _ = block(x, mask)
    g_ckpt = torch.autograd.grad(out_ckpt.sum(), x, retain_graph=False)[0]

    assert torch.allclose(g_plain, g_ckpt, atol=1e-5)


def test_block_gradient_checkpoint_skipped_under_eval():
    """Checkpointing only fires under `self.training` — eval-mode forward is plain."""
    cfg = _config()
    block = OplmBlock(cfg, layer_idx=0)
    block.eval()
    block.gradient_checkpointing = True
    x = torch.randn(1, 4, cfg.hidden_size, requires_grad=True)
    out, _ = block(x, _ones_mask(1, 4))
    # If the checkpointed path had fired, this would still work — but we just
    # confirm the forward succeeds and shape is preserved.
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# OplmStack — construction
# ---------------------------------------------------------------------------


def test_stack_layers_count_matches_config():
    cfg = _config(num_hidden_layers=3)
    stack = OplmStack(cfg)
    assert len(stack.layers) == 3
    for i, block in enumerate(stack.layers):
        assert block.layer_idx == i


def test_stack_has_final_norm():
    cfg = _config()
    stack = OplmStack(cfg)
    assert isinstance(stack.final_norm, OplmLayerNorm)


# ---------------------------------------------------------------------------
# OplmStack — forward
# ---------------------------------------------------------------------------


def test_stack_forward_shape():
    torch.manual_seed(0)
    cfg = _config(num_hidden_layers=2, hidden_size=32)
    stack = OplmStack(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 7))
    last_hidden, hidden_states, attentions = stack(input_ids)
    assert last_hidden.shape == (2, 7, cfg.hidden_size)
    assert hidden_states is None
    assert attentions is None


def test_stack_forward_with_none_mask_materializes_ones():
    """Passing attention_mask=None should run without error and match an all-ones mask."""
    torch.manual_seed(0)
    cfg = _config(num_hidden_layers=2)
    stack = OplmStack(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))

    with torch.no_grad():
        out_none, _, _ = stack(input_ids, attention_mask=None)
        out_ones, _, _ = stack(input_ids, attention_mask=_ones_mask(2, 5))
    assert torch.allclose(out_none, out_ones, atol=1e-6)


def test_stack_hidden_states_has_L_plus_1_entries():
    torch.manual_seed(0)
    cfg = _config(num_hidden_layers=4)
    stack = OplmStack(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
    _, hidden_states, _ = stack(input_ids, output_hidden_states=True)
    assert hidden_states is not None
    assert len(hidden_states) == cfg.num_hidden_layers + 1
    for h in hidden_states:
        assert h.shape == (2, 6, cfg.hidden_size)


def test_stack_first_hidden_state_is_post_embedding():
    """The first entry of hidden_states is the embedding output, not the final-norm output."""
    torch.manual_seed(0)
    cfg = _config(num_hidden_layers=2)
    stack = OplmStack(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))

    with torch.no_grad():
        emb = stack.embed_tokens(input_ids)
        _, hidden_states, _ = stack(input_ids, output_hidden_states=True)
    assert hidden_states is not None
    assert torch.allclose(hidden_states[0], emb)


def test_stack_attentions_has_L_entries():
    torch.manual_seed(0)
    cfg = _config(num_hidden_layers=3)
    stack = OplmStack(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
    _, _, attentions = stack(input_ids, output_attentions=True)
    assert attentions is not None
    assert len(attentions) == cfg.num_hidden_layers
    for a in attentions:
        assert a is not None
        assert a.shape == (2, cfg.num_attention_heads, 5, 5)


@pytest.mark.parametrize("strategy", ["pre", "sandwich", "hybrid", "post_sdpa"])
def test_stack_runs_under_every_norm_strategy(strategy: str):
    torch.manual_seed(0)
    cfg = _config(num_hidden_layers=2, norm_strategy=strategy)
    stack = OplmStack(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
    last_hidden, _, _ = stack(input_ids)
    assert last_hidden.shape == (2, 5, cfg.hidden_size)


def test_stack_runs_with_canon_enabled():
    torch.manual_seed(0)
    cfg = _config(
        num_hidden_layers=2,
        canon_enabled=True,
        canon_positions=["A", "D"],
        canon_kernel_sizes=[3, 3],
    )
    stack = OplmStack(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
    last_hidden, _, _ = stack(input_ids)
    assert last_hidden.shape == (2, 6, cfg.hidden_size)


# ---------------------------------------------------------------------------
# OplmStack — gradient checkpointing
# ---------------------------------------------------------------------------


def test_stack_set_gradient_checkpointing_propagates_to_blocks():
    cfg = _config(num_hidden_layers=3, gradient_checkpointing=False)
    stack = OplmStack(cfg)
    assert all(not b.gradient_checkpointing for b in stack.layers)
    stack.set_gradient_checkpointing(True)
    assert stack.gradient_checkpointing is True
    assert all(b.gradient_checkpointing for b in stack.layers)
    stack.set_gradient_checkpointing(False)
    assert all(not b.gradient_checkpointing for b in stack.layers)


def test_stack_gradient_checkpoint_matches_plain_forward():
    torch.manual_seed(0)
    cfg = _config(num_hidden_layers=2)
    stack = OplmStack(cfg)
    stack.train()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))

    stack.set_gradient_checkpointing(False)
    out_plain, _, _ = stack(input_ids)

    stack.set_gradient_checkpointing(True)
    out_ckpt, _, _ = stack(input_ids)

    assert torch.allclose(out_plain, out_ckpt, atol=1e-5)


# ---------------------------------------------------------------------------
# OplmStack — pad mask correctness
# ---------------------------------------------------------------------------


def test_stack_padded_inputs_match_unpadded_at_real_positions():
    """Doubling the seq with pad ids must not change the output at real positions."""
    torch.manual_seed(0)
    cfg = _config(num_hidden_layers=2)
    stack = OplmStack(cfg).eval()

    real_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    mask_real = torch.ones(1, 4, dtype=torch.long)

    pad_ids = torch.cat([real_ids, torch.randint(0, cfg.vocab_size, (1, 4))], dim=1)
    mask_pad = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        out_real, _, _ = stack(real_ids, attention_mask=mask_real)
        out_pad, _, _ = stack(pad_ids, attention_mask=mask_pad)
    assert torch.allclose(out_real, out_pad[:, :4, :], atol=1e-5)

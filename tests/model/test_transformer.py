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
    mask_dropout: bool = False,
    mask_dropout_reference_ratio: float = 0.12,
    mask_token_id: int = 32,
    residual_scaling: str = "sqrt_num_layers",
    residual_gate: str = "none",
    residual_gate_init: float = 1.0,
    gradient_checkpointing: bool = False,
    value_residual: str = "none",
    value_residual_lambda_init: float = 0.5,
    canon_enabled: bool = False,
    canon_residual: bool = True,
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
        mask_dropout=mask_dropout,
        mask_dropout_reference_ratio=mask_dropout_reference_ratio,
        mask_token_id=mask_token_id,
        residual_scaling=residual_scaling,
        residual_gate=residual_gate,
        residual_gate_init=residual_gate_init,
        gradient_checkpointing=gradient_checkpointing,
        value_residual=value_residual,
        value_residual_lambda_init=value_residual_lambda_init,
        canon_enabled=canon_enabled,
        canon_residual=canon_residual,
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


# ---------------------------------------------------------------------------
# OplmBlock — residual gates
# ---------------------------------------------------------------------------


def test_block_no_residual_gate_by_default():
    """residual_gate='none' (default) adds no gate parameters."""
    block = OplmBlock(_config(residual_gate="none"), layer_idx=0)
    assert block.residual_gate == "none"
    assert not hasattr(block, "attn_gate")
    assert not hasattr(block, "ffn_gate")
    # Exact top-level keys (SwiGLU's gate_proj.* must not be confused for these).
    assert "attn_gate" not in block.state_dict()
    assert "ffn_gate" not in block.state_dict()


def test_block_scalar_residual_gate_shapes():
    block = OplmBlock(_config(residual_gate="scalar"), layer_idx=0)
    assert isinstance(block.attn_gate, torch.nn.Parameter)
    assert isinstance(block.ffn_gate, torch.nn.Parameter)
    assert block.attn_gate.shape == (1,)
    assert block.ffn_gate.shape == (1,)


def test_block_channel_residual_gate_shapes():
    cfg = _config(hidden_size=32, residual_gate="channel")
    block = OplmBlock(cfg, layer_idx=0)
    assert block.attn_gate.shape == (32,)
    assert block.ffn_gate.shape == (32,)


@pytest.mark.parametrize("gate", ["scalar", "channel"])
def test_block_residual_gate_init_value(gate: str):
    block = OplmBlock(_config(residual_gate=gate, residual_gate_init=0.25), layer_idx=0)
    assert torch.allclose(block.attn_gate, torch.full_like(block.attn_gate, 0.25))
    assert torch.allclose(block.ffn_gate, torch.full_like(block.ffn_gate, 0.25))


@pytest.mark.parametrize("gate", ["scalar", "channel"])
def test_block_residual_gates_are_one_dimensional(gate: str):
    """Gate params must be 1D so optimizer grouping routes them to the no-decay group."""
    block = OplmBlock(_config(residual_gate=gate), layer_idx=0)
    assert block.attn_gate.ndim == 1
    assert block.ffn_gate.ndim == 1


@pytest.mark.parametrize("gate", ["scalar", "channel"])
def test_block_residual_gates_persist_in_state_dict(gate: str):
    block = OplmBlock(_config(residual_gate=gate), layer_idx=0)
    keys = block.state_dict()
    assert "attn_gate" in keys
    assert "ffn_gate" in keys


def test_block_residual_gate_init_one_matches_ungated_output():
    """A gate initialized to 1.0 is the identity refinement at init."""
    torch.manual_seed(0)
    cfg_gated = _config(num_hidden_layers=2, residual_gate="channel", residual_gate_init=1.0)
    cfg_plain = _config(num_hidden_layers=2, residual_gate="none")
    gated = OplmBlock(cfg_gated, layer_idx=0)
    plain = OplmBlock(cfg_plain, layer_idx=0)
    # Copy the shared parameters so only the residual gate differs (filter the
    # exact gate keys, not substring "gate" which also matches SwiGLU's gate_proj).
    plain.load_state_dict(
        {k: v for k, v in gated.state_dict().items() if k not in ("attn_gate", "ffn_gate")},
        strict=False,
    )
    x = torch.randn(2, 5, cfg_gated.hidden_size)
    mask = _ones_mask(2, 5)
    gated.eval()
    plain.eval()
    with torch.no_grad():
        out_gated, _ = gated(x, mask)
        out_plain, _ = plain(x, mask)
    assert torch.allclose(out_gated, out_plain, atol=1e-6)


@pytest.mark.parametrize("gate", ["scalar", "channel"])
def test_block_editing_residual_gate_changes_output(gate: str):
    """Mutating the gate parameter perturbs the block output."""
    torch.manual_seed(0)
    block = OplmBlock(_config(residual_gate=gate), layer_idx=0).eval()
    x = torch.randn(2, 5, 32)
    mask = _ones_mask(2, 5)
    with torch.no_grad():
        before, _ = block(x, mask)
        block.attn_gate.mul_(2.0)
        block.ffn_gate.mul_(2.0)
        after, _ = block(x, mask)
    assert not torch.allclose(before, after)


@pytest.mark.parametrize("gate", ["scalar", "channel"])
def test_block_residual_gates_receive_gradient(gate: str):
    torch.manual_seed(0)
    block = OplmBlock(_config(residual_gate=gate), layer_idx=0)
    x = torch.randn(2, 5, 32)
    out, _ = block(x, _ones_mask(2, 5))
    out.sum().backward()
    assert block.attn_gate.grad is not None
    assert block.attn_gate.grad.abs().sum() > 0
    assert block.ffn_gate.grad is not None
    assert block.ffn_gate.grad.abs().sum() > 0


def test_block_rejects_unknown_residual_gate():
    cfg = _config(residual_gate="bogus")
    with pytest.raises(ValueError, match="residual_gate"):
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
    # Canon-B lives on the attention module and Canon-D on the FFN; both off.
    assert not block.attention.canon_b_enabled
    assert block.ffn.conv_d is None


# Only A and C are block-level convs (conv_a/conv_c). Canon-B lives inside
# OplmAttention (conv_b_q/k/v); Canon-D lives inside the FFN (conv_d).
@pytest.mark.parametrize("position", ["A", "C"])
def test_block_creates_only_requested_canon_position(position: str):
    cfg = _config(
        canon_enabled=True,
        canon_positions=[position],
        canon_kernel_sizes=[3, 3],
    )
    block = OplmBlock(cfg, layer_idx=0)
    name = f"conv_{position.lower()}"
    assert isinstance(getattr(block, name), CanonConv)
    for other in {"A", "C"} - {position}:
        assert not hasattr(block, f"conv_{other.lower()}")
    # B is off (on attention), D is off (on FFN) for an A/C-only config.
    assert not hasattr(block, "conv_b")
    assert not hasattr(block, "conv_d")
    assert not block.attention.canon_b_enabled
    assert block.ffn.conv_d is None


def test_block_canon_b_lives_on_attention():
    cfg = _config(canon_enabled=True, canon_positions=["B"], canon_kernel_sizes=[3, 3])
    block = OplmBlock(cfg, layer_idx=0)
    assert not hasattr(block, "conv_b")
    assert block.attention.canon_b_enabled
    for name in ("conv_b_q", "conv_b_k", "conv_b_v"):
        assert isinstance(getattr(block.attention, name), CanonConv)
    # A/C (block-level) and D (FFN) were not requested.
    for name in ("conv_a", "conv_c"):
        assert not hasattr(block, name)
    assert block.ffn.conv_d is None


def test_block_canon_d_lives_on_ffn_at_intermediate_width():
    """Canon-D is a conv inside the FFN at intermediate_size, not on the block."""
    cfg = _config(canon_enabled=True, canon_positions=["D"], canon_kernel_sizes=[3, 3])
    block = OplmBlock(cfg, layer_idx=0)
    assert not hasattr(block, "conv_d")  # not block-level
    assert isinstance(block.ffn.conv_d, CanonConv)
    assert block.ffn.conv_d.channels == cfg.intermediate_size
    # A/C absent, B off.
    for name in ("conv_a", "conv_c"):
        assert not hasattr(block, name)
    assert not block.attention.canon_b_enabled


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


@pytest.mark.parametrize("strategy", ["pre", "sandwich", "post_sdpa"])
def test_block_canon_residual_zero_kernels_match_no_canon(strategy: str):
    """Residual Canon with zeroed convs is an identity at every position.

    A/C (block-level) and D (inside the FFN) all use ``z + conv(z)``; a zero
    kernel makes each a no-op, so the block output must match a no-canon block
    that shares its other weights. Proves no position destroys the identity path,
    under every norm strategy that supports Canon.
    """
    torch.manual_seed(0)
    cfg = _config(
        norm_strategy=strategy,
        canon_enabled=True,
        canon_positions=["A", "C", "D"],
        canon_kernel_sizes=[3, 3],
    )
    block = OplmBlock(cfg, layer_idx=0).eval()
    plain = OplmBlock(_config(norm_strategy=strategy), layer_idx=0).eval()
    block.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        block.conv_a.conv.weight.zero_()
        block.conv_c.conv.weight.zero_()
        block.ffn.conv_d.conv.weight.zero_()
    x = torch.randn(2, 7, cfg.hidden_size)
    mask = _ones_mask(2, 7)
    with torch.no_grad():
        out, _ = block(x, mask)
        out_plain, _ = plain(x, mask)
    assert torch.allclose(out, out_plain, rtol=1e-5, atol=1e-6)


def test_block_canon_c_feeds_ffn_input_not_residual_path():
    """Canon-C feeds the FFN pre-norm input, not the attention-output residual.

    With the FFN's ``down_proj`` zeroed the FFN contributes nothing, so the block
    output collapses to the attention-side residual ``h``. Under the paper-exact
    placement C only touches the FFN input, so ``h`` (and the output) is identical
    to a no-canon block even with a non-trivial C kernel. If C still modified
    ``attn_out`` (the old placement) it would change ``h`` and the test would fail.
    """
    torch.manual_seed(0)
    cfg_c = _config(canon_enabled=True, canon_positions=["C"], canon_kernel_sizes=[3, 3])
    block_c = OplmBlock(cfg_c, layer_idx=0).eval()
    plain = OplmBlock(_config(), layer_idx=0).eval()
    block_c.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        block_c.conv_c.conv.weight.normal_()  # non-identity: would perturb attn_out if misplaced
        block_c.ffn.down_proj.weight.zero_()
        plain.ffn.down_proj.weight.zero_()
    x = torch.randn(2, 7, cfg_c.hidden_size)
    mask = _ones_mask(2, 7)
    with torch.no_grad():
        out_c, _ = block_c(x, mask)
        out_plain, _ = plain(x, mask)
    assert torch.allclose(out_c, out_plain, atol=1e-6)


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
# OplmStack — mask dropout plumbing
# ---------------------------------------------------------------------------


def test_stack_mask_dropout_zeros_mask_positions_post_embedding():
    """With mask dropout on, the post-embedding hidden state zeros <mask> rows."""
    torch.manual_seed(0)
    cfg = _config(num_hidden_layers=2, mask_dropout=True, mask_token_id=32)
    stack = OplmStack(cfg).eval()
    input_ids = torch.tensor([[5, 32, 7, 9]])

    with torch.no_grad():
        _, hidden_states, _ = stack(input_ids, output_hidden_states=True)
    # First hidden state is the post-embedding tensor (mask dropout applied).
    assert hidden_states is not None
    assert torch.allclose(hidden_states[0][0, 1], torch.zeros(cfg.hidden_size))


def test_stack_inputs_embeds_bypasses_mask_dropout():
    """The inputs_embeds path must not apply mask dropout (IDs are gone)."""
    torch.manual_seed(0)
    cfg_on = _config(num_hidden_layers=2, mask_dropout=True, mask_token_id=32)
    cfg_off = _config(num_hidden_layers=2, mask_dropout=False, mask_token_id=32)
    stack_on = OplmStack(cfg_on).eval()
    stack_off = OplmStack(cfg_off).eval()
    # Mask dropout adds no parameters, so the state dicts are interchangeable.
    stack_off.load_state_dict(stack_on.state_dict())

    input_ids = torch.tensor([[5, 32, 7, 9]])
    raw_embeds = stack_on.embed_tokens.embed_tokens(input_ids)

    with torch.no_grad():
        out_embeds, _, _ = stack_on(inputs_embeds=raw_embeds)
        # mask_dropout=False fed input_ids reproduces the raw-embedding forward.
        out_ids_off, _, _ = stack_off(input_ids=input_ids)
    assert torch.allclose(out_embeds, out_ids_off, atol=1e-6)


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
# OplmStack — value residual (ResFormer, arXiv:2410.17897)
# ---------------------------------------------------------------------------


def test_stack_value_residual_lambda_count():
    """Learnable mode adds exactly one lambda per layer after the first."""
    cfg = _config(num_hidden_layers=4, value_residual="learnable")
    stack = OplmStack(cfg)
    lam_names = [n for n, _ in stack.named_parameters() if "value_residual_lambda" in n]
    assert len(lam_names) == cfg.num_hidden_layers - 1
    assert not any(n.startswith("layers.0.") for n in lam_names)


def test_stack_value_residual_fixed_lambda_one_matches_disabled():
    """Fixed lambda = 1.0 is a no-op blend: outputs match the disabled model."""
    torch.manual_seed(0)
    base = OplmStack(_config(num_hidden_layers=3)).eval()
    torch.manual_seed(0)
    blended = OplmStack(
        _config(num_hidden_layers=3, value_residual="fixed", value_residual_lambda_init=1.0)
    ).eval()

    input_ids = torch.randint(0, 33, (2, 6))
    with torch.no_grad():
        out_base, _, _ = base(input_ids)
        out_blended, _, _ = blended(input_ids)
    assert torch.allclose(out_base, out_blended, atol=1e-6)


def test_stack_value_residual_changes_output():
    """Fixed lambda < 1 actually changes the stack output vs disabled."""
    torch.manual_seed(0)
    base = OplmStack(_config(num_hidden_layers=3)).eval()
    torch.manual_seed(0)
    blended = OplmStack(
        _config(num_hidden_layers=3, value_residual="fixed", value_residual_lambda_init=0.5)
    ).eval()

    input_ids = torch.randint(0, 33, (2, 6))
    with torch.no_grad():
        out_base, _, _ = base(input_ids)
        out_blended, _, _ = blended(input_ids)
    assert not torch.allclose(out_base, out_blended)


def test_block_value_residual_threads_through_checkpointing():
    torch.manual_seed(0)
    cfg = _config(num_hidden_layers=2, value_residual="learnable", gradient_checkpointing=True)
    stack = OplmStack(cfg).train()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))

    out, _, _ = stack(input_ids)
    assert torch.isfinite(out).all()
    out.sum().backward()
    lam = stack.layers[1].attention.value_residual_lambda
    assert lam.grad is not None
    assert lam.grad.abs().sum() > 0


def test_stack_value_residual_checkpoint_matches_plain_forward():
    torch.manual_seed(0)
    cfg = _config(num_hidden_layers=2, value_residual="learnable")
    stack = OplmStack(cfg).train()
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

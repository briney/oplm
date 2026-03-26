"""Tests for the BlockAttentionResidual module."""

from __future__ import annotations

import torch

from oplm.config import ModelConfig
from oplm.model.residual import BlockAttentionResidual, BlockAttentionResidualState


def _make_config(**kwargs: object) -> ModelConfig:
    defaults: dict[str, object] = {
        "hidden_dim": 64,
        "num_heads": 4,
        "num_kv_heads": 2,
        "num_layers": 4,
        "max_seq_len": 32,
        "attn_residual": True,
        "attn_residual_block_size": 2,
    }
    defaults.update(kwargs)
    return ModelConfig(**defaults)


B, T, D = 2, 8, 64


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------


class TestBlockAttentionResidualState:
    """Test the state dataclass."""

    def test_initial_state(self) -> None:
        state = BlockAttentionResidualState()
        assert len(state.blocks) == 0
        assert state.partial_block is None
        assert state.step_count == 0


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------


class TestAccumulation:
    """Test sublayer accumulation and block boundary detection."""

    def test_first_accumulation_sets_partial(self) -> None:
        cfg = _make_config()
        attn_res = BlockAttentionResidual(cfg)
        state = BlockAttentionResidualState()
        sublayer = torch.randn(B, T, D)
        new_state = attn_res.accumulate(state, sublayer)
        assert new_state.partial_block is not None
        assert len(new_state.blocks) == 0
        assert new_state.step_count == 1

    def test_accumulation_adds_to_partial(self) -> None:
        cfg = _make_config()
        attn_res = BlockAttentionResidual(cfg)
        state = BlockAttentionResidualState()
        s1 = torch.randn(B, T, D)
        s2 = torch.randn(B, T, D)
        state = attn_res.accumulate(state, s1)
        state = attn_res.accumulate(state, s2)
        # Still within a block (block_size=2, so 4 steps per block)
        assert state.partial_block is not None
        assert len(state.blocks) == 0
        assert state.step_count == 2
        torch.testing.assert_close(state.partial_block, s1 + s2)

    def test_block_boundary(self) -> None:
        """After block_size * 2 steps, a block should be finalized."""
        cfg = _make_config(attn_residual_block_size=2)
        attn_res = BlockAttentionResidual(cfg)
        state = BlockAttentionResidualState()

        # block_size=2 -> steps_per_block = 2 * 2 = 4
        for _ in range(4):
            state = attn_res.accumulate(state, torch.randn(B, T, D))

        assert len(state.blocks) == 1
        assert state.partial_block is None
        assert state.step_count == 4

    def test_two_blocks(self) -> None:
        cfg = _make_config(attn_residual_block_size=2)
        attn_res = BlockAttentionResidual(cfg)
        state = BlockAttentionResidualState()

        for _ in range(8):
            state = attn_res.accumulate(state, torch.randn(B, T, D))

        assert len(state.blocks) == 2
        assert state.partial_block is None
        assert state.step_count == 8

    def test_partial_block_between_boundaries(self) -> None:
        """After 5 steps with steps_per_block=4, should have 1 block + partial."""
        cfg = _make_config(attn_residual_block_size=2)
        attn_res = BlockAttentionResidual(cfg)
        state = BlockAttentionResidualState()

        for _ in range(5):
            state = attn_res.accumulate(state, torch.randn(B, T, D))

        assert len(state.blocks) == 1
        assert state.partial_block is not None
        assert state.step_count == 5

    def test_state_immutability(self) -> None:
        """accumulate should return a new state, not mutate the old one."""
        cfg = _make_config()
        attn_res = BlockAttentionResidual(cfg)
        state = BlockAttentionResidualState()
        sublayer = torch.randn(B, T, D)
        new_state = attn_res.accumulate(state, sublayer)
        assert state.step_count == 0
        assert new_state.step_count == 1


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    """Test depth-wise attention aggregation."""

    def test_aggregate_single_block_output_shape(self) -> None:
        cfg = _make_config()
        attn_res = BlockAttentionResidual(cfg)
        block = torch.randn(B, T, D)
        state = BlockAttentionResidualState(blocks=[block])
        out = attn_res.aggregate(state, step_idx=0)
        assert out.shape == (B, T, D)

    def test_aggregate_multiple_blocks(self) -> None:
        cfg = _make_config()
        attn_res = BlockAttentionResidual(cfg)
        blocks = [torch.randn(B, T, D) for _ in range(3)]
        state = BlockAttentionResidualState(blocks=blocks)
        out = attn_res.aggregate(state, step_idx=0)
        assert out.shape == (B, T, D)

    def test_aggregate_includes_partial(self) -> None:
        """Aggregation should include the partial block when present."""
        cfg = _make_config()
        attn_res = BlockAttentionResidual(cfg)
        block = torch.randn(B, T, D)
        partial = torch.randn(B, T, D)
        state = BlockAttentionResidualState(blocks=[block], partial_block=partial)
        out = attn_res.aggregate(state, step_idx=0)
        assert out.shape == (B, T, D)

    def test_aggregate_weights_sum_to_one(self) -> None:
        """Softmax weights over depth should sum to 1."""
        cfg = _make_config()
        attn_res = BlockAttentionResidual(cfg)
        blocks = [torch.randn(B, T, D) for _ in range(3)]
        V = torch.stack(blocks)  # (N, B, T, D)
        K = attn_res.key_norms[0](V)
        w = attn_res.pseudo_queries[0]
        logits = torch.einsum("d, n b t d -> n b t", w, K)
        weights = logits.softmax(dim=0)
        sums = weights.sum(dim=0)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=0)

    def test_different_step_indices_use_different_params(self) -> None:
        """Different step_idx values should generally produce different outputs."""
        cfg = _make_config()
        attn_res = BlockAttentionResidual(cfg)
        blocks = [torch.randn(B, T, D) for _ in range(2)]
        state = BlockAttentionResidualState(blocks=blocks)
        out0 = attn_res.aggregate(state, step_idx=0)
        out1 = attn_res.aggregate(state, step_idx=1)
        assert not torch.allclose(out0, out1, atol=1e-5)


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------


class TestResidualParameters:
    """Verify parameter structure."""

    def test_num_pseudo_queries(self) -> None:
        cfg = _make_config(num_layers=4)
        attn_res = BlockAttentionResidual(cfg)
        assert len(attn_res.pseudo_queries) == 2 * 4

    def test_num_key_norms(self) -> None:
        cfg = _make_config(num_layers=4)
        attn_res = BlockAttentionResidual(cfg)
        assert len(attn_res.key_norms) == 2 * 4


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestResidualGradient:
    """Verify gradients flow through aggregation."""

    def test_gradient_flow_through_aggregate(self) -> None:
        cfg = _make_config()
        attn_res = BlockAttentionResidual(cfg)
        blocks = [torch.randn(B, T, D, requires_grad=True) for _ in range(2)]
        state = BlockAttentionResidualState(blocks=blocks)
        out = attn_res.aggregate(state, step_idx=0)
        loss = out.sum()
        loss.backward()
        # Gradients should reach the block tensors
        for i, block in enumerate(blocks):
            assert block.grad is not None, f"No gradient for block {i}"
        # Gradients should reach the pseudo_query and key_norm used (step_idx=0)
        assert attn_res.pseudo_queries[0].grad is not None
        assert attn_res.pseudo_queries[0].grad.abs().sum() > 0
        for param in attn_res.key_norms[0].parameters():
            assert param.grad is not None

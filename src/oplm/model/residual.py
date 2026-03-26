"""Block attention residuals for learned depth-wise aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from oplm.model.norm import RMSNorm

if TYPE_CHECKING:
    from oplm.config import ModelConfig


@dataclass
class BlockAttentionResidualState:
    """Tracks block boundaries and accumulated representations.

    Attributes:
        blocks: Completed block representations, each ``(B, T, D)``.
        partial_block: Current intra-block accumulation, ``(B, T, D)`` or None.
        step_count: Tracks sublayer steps for block boundary detection.
    """

    blocks: list[Tensor] = field(default_factory=list)
    partial_block: Tensor | None = None
    step_count: int = 0


class BlockAttentionResidual(nn.Module):
    """Learned depth-wise residual connections with block-level granularity.

    Replaces fixed residual connections (``x + sublayer(x)``) with learned,
    input-dependent softmax attention over depth. Based on the Kimi team's
    Attention Residuals paper, adapted to block-level granularity for memory
    efficiency.

    Each layer applies attention residuals **twice** (before attention and
    before FFN sublayers), so there are ``2 * num_layers`` pseudo-queries
    and key norms.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        n = 2 * config.num_layers
        self.block_size = config.attn_residual_block_size
        self.pseudo_queries = nn.ParameterList(
            [nn.Parameter(torch.randn(config.hidden_dim)) for _ in range(n)]
        )
        self.key_norms = nn.ModuleList(
            [RMSNorm(config.hidden_dim, config.norm_eps) for _ in range(n)]
        )

    def aggregate(
        self,
        state: BlockAttentionResidualState,
        step_idx: int,
    ) -> Tensor:
        """Compute depth-wise attention over block representations.

        Args:
            state: Current block accumulation state.
            step_idx: Index into pseudo_queries/key_norms (``0..2*num_layers-1``).

        Returns:
            Aggregated hidden state ``(B, T, D)``.
        """
        entries = state.blocks.copy()
        if state.partial_block is not None:
            entries.append(state.partial_block)

        V = torch.stack(entries)  # (N, B, T, D)
        K = self.key_norms[step_idx](V)  # (N, B, T, D)
        w = self.pseudo_queries[step_idx]  # (D,)
        logits = torch.einsum("d, n b t d -> n b t", w, K)  # (N, B, T)
        weights = logits.softmax(dim=0)  # (N, B, T)
        return torch.einsum("n b t, n b t d -> b t d", weights, V)  # (B, T, D)

    def accumulate(
        self,
        state: BlockAttentionResidualState,
        sublayer_out: Tensor,
    ) -> BlockAttentionResidualState:
        """Add sublayer output to partial block and check block boundary.

        Args:
            state: Current state.
            sublayer_out: Output from attention or FFN sublayer ``(B, T, D)``.

        Returns:
            Updated state (new object with updated fields).
        """
        if state.partial_block is None:
            partial = sublayer_out
        else:
            partial = state.partial_block + sublayer_out

        step_count = state.step_count + 1

        # Each block spans block_size layers × 2 sublayers per layer
        steps_per_block = self.block_size * 2
        if step_count % steps_per_block == 0:
            # Block boundary: finalize this block
            return BlockAttentionResidualState(
                blocks=state.blocks + [partial],
                partial_block=None,
                step_count=step_count,
            )

        return BlockAttentionResidualState(
            blocks=state.blocks,
            partial_block=partial,
            step_count=step_count,
        )

"""Transformer assembly: blocks, encoder, MLM head, and top-level model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from oplm.model.attention import Attention
from oplm.model.conv import BidirectionalDepthwiseConv
from oplm.model.embedding import TokenEmbedding, ValueEmbedding
from oplm.model.ffn import FFN
from oplm.model.masking import normalize_attention_mask
from oplm.model.norm import RMSNorm
from oplm.model.residual import BlockAttentionResidual, BlockAttentionResidualState

if TYPE_CHECKING:
    from oplm.config import ModelConfig


class TransformerBlock(nn.Module):
    """Single transformer layer with configurable normalization.

    Supports three normalization strategies via config:
    - Pre-norm (default): ``x + sublayer(norm(x))``
    - Post-norm: ``norm(x + sublayer(x))``
    - Sandwich norm: ``x + norm₂(sublayer(norm₁(x)))`` — overrides pre/post

    And two forward paths:
    - Standard residual: ``forward()``
    - Attention residuals: ``forward_with_attn_res()`` — learned depth-wise
      aggregation replaces fixed residual connections.

    Convolutions at positions A (pre-attention) and C (pre-FFN) are optionally
    inserted based on ``config.conv_positions``.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.conv_kernel_size = config.conv_kernel_size_for_layer(layer_idx)

        # Pre-sublayer norms (used by pre_norm and sandwich_norm)
        needs_pre = config.pre_norm or config.sandwich_norm
        self.attn_pre_norm = RMSNorm(config.hidden_dim, config.norm_eps) if needs_pre else None
        self.ffn_pre_norm = RMSNorm(config.hidden_dim, config.norm_eps) if needs_pre else None

        # Post-residual norms (classic post-norm, ignored when sandwich_norm)
        needs_post = config.post_norm and not config.sandwich_norm
        self.attn_post_norm = RMSNorm(config.hidden_dim, config.norm_eps) if needs_post else None
        self.ffn_post_norm = RMSNorm(config.hidden_dim, config.norm_eps) if needs_post else None

        # Post-sublayer norms (sandwich norm: norm after sublayer, before residual)
        self.attn_sandwich_norm = (
            RMSNorm(config.hidden_dim, config.norm_eps) if config.sandwich_norm else None
        )
        self.ffn_sandwich_norm = (
            RMSNorm(config.hidden_dim, config.norm_eps) if config.sandwich_norm else None
        )
        self.attention = Attention(config, layer_idx)
        self.ffn = FFN(config, conv_kernel_size=self.conv_kernel_size)
        self.conv_a = (
            BidirectionalDepthwiseConv(
                config.hidden_dim,
                self.conv_kernel_size,
                config.conv_activation,
            )
            if "A" in config.conv_positions
            else None
        )
        self.conv_c = (
            BidirectionalDepthwiseConv(
                config.hidden_dim,
                self.conv_kernel_size,
                config.conv_activation,
            )
            if "C" in config.conv_positions
            else None
        )

    def forward(
        self,
        x: Tensor,
        v_first: Tensor | None = None,
        attention_mask: Tensor | None = None,
        value_embed: Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Standard pre-norm residual forward pass.

        Args:
            x: Hidden state ``(B, T, D)``.
            v_first: First layer's V for cross-layer value residuals.
            attention_mask: Normalized 4D attention mask.
            value_embed: Value embedding ``(B, T, KV_H, D_head)`` or None.
            need_weights: If True, return per-head attention weights.

        Returns:
            Tuple of (output, v_first, attn_weights) where attn_weights is
            ``(B, H, T, T)`` if need_weights else None.
        """
        # Attention sublayer
        residual = x
        if self.conv_a is not None:
            x = self.conv_a(x)
        attn_input = self.attn_pre_norm(x) if self.attn_pre_norm is not None else x
        attn_out, v_first_out, attn_weights = self.attention(
            attn_input, v_first, attention_mask, value_embed, need_weights
        )
        if v_first_out is not None:
            v_first = v_first_out
        if self.attn_sandwich_norm is not None:
            attn_out = self.attn_sandwich_norm(attn_out)
        x = residual + attn_out
        if self.attn_post_norm is not None:
            x = self.attn_post_norm(x)

        # FFN sublayer
        residual = x
        if self.conv_c is not None:
            x = self.conv_c(x)
        ffn_input = self.ffn_pre_norm(x) if self.ffn_pre_norm is not None else x
        ffn_out = self.ffn(ffn_input)
        if self.ffn_sandwich_norm is not None:
            ffn_out = self.ffn_sandwich_norm(ffn_out)
        x = residual + ffn_out
        if self.ffn_post_norm is not None:
            x = self.ffn_post_norm(x)

        return x, v_first, attn_weights

    def forward_with_attn_res(
        self,
        v_first: Tensor | None,
        attention_mask: Tensor | None,
        value_embed: Tensor | None,
        attn_res: BlockAttentionResidual,
        state: BlockAttentionResidualState,
        materialize_output: bool = True,
    ) -> tuple[Tensor | None, Tensor | None, BlockAttentionResidualState]:
        """Attention residuals forward pass.

        Replaces fixed residual connections with learned depth-wise attention
        over block representations.

        Args:
            v_first: First layer's V for cross-layer value residuals.
            attention_mask: Normalized 4D attention mask.
            value_embed: Value embedding or None.
            attn_res: Block attention residual module.
            state: Current block accumulation state.
            materialize_output: Whether to materialize the post-FFN hidden
                state for downstream consumers.

        Returns:
            Tuple of (hidden, v_first, updated_state) where hidden is the
            post-FFN aggregated representation ``(B, T, D)`` when requested,
            else None.
        """
        # Attention sublayer
        h = attn_res.aggregate(state, 2 * self.layer_idx)
        if self.conv_a is not None:
            h = self.conv_a(h)
        attn_input = self.attn_pre_norm(h) if self.attn_pre_norm is not None else h
        attn_out, v_first_out, _ = self.attention(
            attn_input, v_first, attention_mask, value_embed
        )
        if v_first_out is not None:
            v_first = v_first_out
        state = attn_res.accumulate(state, attn_out)

        # FFN sublayer
        h = attn_res.aggregate(state, 2 * self.layer_idx + 1)
        if self.conv_c is not None:
            h = self.conv_c(h)
        ffn_input = self.ffn_pre_norm(h) if self.ffn_pre_norm is not None else h
        ffn_out = self.ffn(ffn_input)
        state = attn_res.accumulate(state, ffn_out)

        x: Tensor | None = None
        if materialize_output:
            # Re-aggregate over the updated state only when a downstream
            # consumer needs the layer output or the encoder needs its final
            # hidden state.
            x = attn_res.aggregate(state, 2 * self.layer_idx + 1)
        return x, v_first, state


class OplmEncoder(nn.Module):
    """OPLM encoder backbone.

    Stacks ``TransformerBlock`` layers with optional value embeddings,
    cross-layer value residuals, block attention residuals, and gradient
    checkpointing.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = TokenEmbedding(config)
        self.value_embedding = ValueEmbedding(config) if config.num_value_embeds > 0 else None
        self.blocks = nn.ModuleList([TransformerBlock(config, i) for i in range(config.num_layers)])
        self.attn_residual = BlockAttentionResidual(config) if config.attn_residual else None
        self.final_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.gradient_checkpointing = config.gradient_checkpointing

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[Tensor, list[Tensor] | None]:
        """Encode input token IDs.

        Args:
            input_ids: Token IDs ``(B, T)``.
            attention_mask: Public keep mask ``(B, T)`` or a pre-expanded 4D mask.
            need_weights: If True, collect per-layer attention weights.

        Returns:
            Tuple of (hidden_states, attn_weights) where hidden_states is
            ``(B, T, D)`` and attn_weights is a list of ``(B, H, T, T)``
            tensors (one per layer) or None.
        """
        x: Tensor | None = self.embedding(input_ids)  # (B, T, D)
        attention_mask = normalize_attention_mask(attention_mask)
        v_first: Tensor | None = None
        all_attn_weights: list[Tensor] = []

        # Initialize AttnRes state with token embedding as first "block"
        state: BlockAttentionResidualState | None = None
        if self.attn_residual is not None:
            assert x is not None
            state = BlockAttentionResidualState(blocks=[x], partial_block=None, step_count=0)

        for i, blk in enumerate(self.blocks):
            block: TransformerBlock = blk  # type: ignore[assignment]
            uses_value_embed = self.value_embedding is not None and self.value_embedding.uses_layer(
                i
            )
            ve: Tensor | None = None
            if uses_value_embed:
                assert self.value_embedding is not None
                assert x is not None
                ve = self.value_embedding(input_ids, x, i)

            if state is not None:
                assert self.attn_residual is not None
                materialize_output = i == len(self.blocks) - 1 or (
                    self.value_embedding is not None and self.value_embedding.uses_layer(i + 1)
                )
                if self.gradient_checkpointing and self.training:
                    x, v_first, state = torch_checkpoint(
                        block.forward_with_attn_res,
                        v_first,
                        attention_mask,
                        ve,
                        self.attn_residual,
                        state,
                        materialize_output,
                        use_reentrant=False,
                    )
                else:
                    x, v_first, state = block.forward_with_attn_res(
                        v_first,
                        attention_mask,
                        ve,
                        self.attn_residual,
                        state,
                        materialize_output=materialize_output,
                    )
            else:
                assert x is not None
                if self.gradient_checkpointing and self.training:
                    x, v_first, layer_weights = torch_checkpoint(
                        block.forward,
                        x,
                        v_first,
                        attention_mask,
                        ve,
                        need_weights,
                        use_reentrant=False,
                    )
                else:
                    x, v_first, layer_weights = block(x, v_first, attention_mask, ve, need_weights)
                if layer_weights is not None:
                    all_attn_weights.append(layer_weights)

        assert x is not None
        hidden = self.final_norm(x)  # (B, T, D)
        return hidden, all_attn_weights if need_weights else None


class MLMHead(nn.Module):
    """Masked language modeling head.

    Projects encoder hidden states to vocabulary logits via
    dense → norm → GELU → projection.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.activation = nn.GELU()
        self.projection = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Project hidden states to vocabulary logits.

        Args:
            hidden_states: Encoder output ``(B, T, D)``.

        Returns:
            Logits ``(B, T, V)``.
        """
        x = self.activation(self.norm(self.dense(hidden_states)))
        logits: Tensor = self.projection(x)
        return logits


class OplmForMLM(nn.Module):
    """OPLM model with masked language modeling head.

    Combines :class:`OplmEncoder` and :class:`MLMHead` with optional
    embedding weight tying and cross-entropy loss computation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = OplmEncoder(config)
        self.mlm_head = MLMHead(config)
        if config.tie_embeddings:
            self.mlm_head.projection.weight = self.encoder.embedding.embed.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        need_weights: bool = False,
    ) -> dict[str, Tensor | list[Tensor] | None]:
        """Forward pass with optional MLM loss.

        Args:
            input_ids: Token IDs ``(B, T)``.
            attention_mask: Public keep mask ``(B, T)`` or a pre-expanded 4D mask.
            labels: MLM targets ``(B, T)`` with ``-100`` for non-masked positions.
            need_weights: If True, include per-layer attention weights.

        Returns:
            Dict with keys ``"logits"`` ``(B, T, V)``, ``"loss"`` (scalar or
            None), and ``"attention_weights"`` (list of ``(B, H, T, T)`` or
            None).
        """
        hidden, attn_weights = self.encoder(input_ids, attention_mask, need_weights)
        logits = self.mlm_head(hidden)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        result: dict[str, Tensor | list[Tensor] | None] = {
            "logits": logits,
            "loss": loss,
        }
        if attn_weights is not None:
            result["attention_weights"] = attn_weights
        return result

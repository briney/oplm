"""Language-model shim: per-layer LayerNorm + projection, softmax layer mix, single-to-pair.

Ported from Biohub's ESMFold2 ``LanguageModelShim`` and ``SingleToPair``
(esm/models/esmfold2/layers.py, Apache-2.0; see THIRD_PARTY_NOTICES.md), with the released
checkpoint's names (``language_model.{layer_weights, pair_input_norm, pair_proj,
single_to_pair.{downproject, output_fc1, output_fc2}, pair_output_norm}``). Modifications: the
layer mix is an explicit einsum (upstream's ``weights @ x`` + ``squeeze(-2)`` breaks at
``L == 1``); the pair product is block-local (spec §4.6); the hidden states come from any
LM that exposes a ``(B, L, K, D)`` stack — the OPLM runner here feeds every chain as its own
batch row with BOS/EOS (spec §4.5) instead of one packed row per complex.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from oplm.fold.pair import outer_product_difference

if TYPE_CHECKING:
    from torch import Tensor

    from oplm.fold.configuration_fold import FoldConfig
    from oplm.fold.data.featurize import FoldFeatures

__all__ = ["LanguageModelShim", "SingleToPair", "frozen_lm_hidden_states", "lm_state_count"]


class SingleToPair(nn.Module):
    """``fc2(gelu(fc1(cat[x_i * x_j, x_i - x_j])))`` after a biased down-projection."""

    def __init__(self, width: int, hidden: int, out_width: int) -> None:
        super().__init__()
        self.downproject = nn.Linear(width, width, bias=True)
        self.output_fc1 = nn.Linear(2 * width, hidden, bias=True)
        self.output_fc2 = nn.Linear(hidden, out_width, bias=True)

    def forward(self, x: Tensor, rows: Tensor | None = None, cols: Tensor | None = None) -> Tensor:
        x = self.downproject(x)
        return self.output_fc2(F.gelu(self.output_fc1(outer_product_difference(x, rows, cols))))


class LanguageModelShim(nn.Module):
    """LM states ``(B, L, K, D)`` -> pair ``(B, I, J, pair)`` (checkpoint ``language_model``)."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(config.lm_num_hidden_states))
        self.pair_input_norm = nn.LayerNorm(config.lm_hidden_size, eps=config.layer_norm_eps)
        self.pair_proj = nn.Linear(config.lm_hidden_size, config.pair_width, bias=False)
        self.single_to_pair = SingleToPair(config.pair_width, config.pair_width, config.pair_width)
        self.pair_output_norm = nn.LayerNorm(config.pair_width, eps=config.layer_norm_eps)

    def mix(self, hidden_states: Tensor) -> Tensor:
        """Per-state LayerNorm + projection, then the softmax-weighted sum over states."""
        h = self.pair_proj(self.pair_input_norm(hidden_states))
        weights = torch.softmax(self.layer_weights.float(), dim=0).to(h.dtype)
        return torch.einsum("k,blkc->blc", weights, h)

    def forward(
        self, hidden_states: Tensor, rows: Tensor | None = None, cols: Tensor | None = None
    ) -> Tensor:
        return self.pair_output_norm(self.single_to_pair(self.mix(hidden_states), rows, cols))


def lm_state_count(lm: nn.Module) -> int:
    """Number of hidden states an ``OplmModel`` returns: executed blocks + the embedding output."""
    return len(lm.backbone.layer_execution_order) + 1  # ty: ignore[unresolved-attribute, invalid-argument-type]  # nn.Module attrs are Tensor | Module


def frozen_lm_hidden_states(
    lm: nn.Module, features: FoldFeatures, *, dtype: torch.dtype | None = None
) -> Tensor:
    """Run the frozen LM on the per-chain rows and gather every state per token: ``(1, L, K, D)``.

    The LM must already be in eval mode on the features' device (the fold model owns that).
    Padded tokens are zero. Runs under ``torch.no_grad`` (not inference mode: the OPLM RoPE
    cache re-assigns buffers lazily).
    """
    with torch.no_grad():
        out = lm(
            input_ids=features.lm_input_ids,
            attention_mask=features.lm_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    states = torch.stack(tuple(out.hidden_states), dim=2)  # (C, T, K, D)
    if dtype is not None:
        states = states.to(dtype)
    gathered = states[features.lm_rows[0], features.lm_positions[0]]  # (L, K, D)
    gathered = gathered * features.token_mask[0][:, None, None].to(gathered.dtype)
    return gathered[None]

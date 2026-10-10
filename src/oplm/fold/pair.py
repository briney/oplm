"""Block-local pair features (spec §4.6, §5.6).

Every function takes optional ``rows``/``cols`` index tensors and returns the corresponding
block of the all-by-all tensor; the full tensor is the call with both ``None``. No function
contains a hidden ``arange(L)``: indices come from the token features. Semantics follow
ESMFold2's ``ResIdxAsymIdSymIdEntityIdEncoding`` and ``SingleToPair`` (esm/models/esmfold2/
layers.py, Apache-2.0; see THIRD_PARTY_NOTICES.md); the relpos feature is the exact AF3-style
layout the released ``input_embedder.rel_pos.embed`` weight expects: ``[residue bins 66 |
token bins 66 | same entity 1 | chain bins 6]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "RelativePositionEncoding",
    "distance_bins",
    "outer_product_difference",
    "outer_sum",
    "pair_mask",
    "relpos_features",
    "select_cols",
    "select_rows",
    "token_bond_features",
]


def select_rows(x: Tensor, rows: Tensor | None) -> Tensor:
    """``x[:, rows]`` or ``x`` when ``rows`` is ``None``."""
    return x if rows is None else x[:, rows]


def select_cols(x: Tensor, cols: Tensor | None) -> Tensor:
    """``x[:, :, cols]`` (for pair tensors) or ``x`` when ``cols`` is ``None``."""
    return x if cols is None else x[:, :, cols]


def pair_mask(token_mask: Tensor, rows: Tensor | None = None, cols: Tensor | None = None) -> Tensor:
    """Outer product of the token mask, as float ``(B, I, J)``."""
    m = token_mask.float()
    return select_rows(m, rows)[:, :, None] * select_rows(m, cols)[:, None, :]


def outer_sum(row_vec: Tensor, col_vec: Tensor) -> Tensor:
    """``row_vec[:, i] + col_vec[:, j]`` -> ``(B, I, J, D)``; pass already-selected rows/cols."""
    return row_vec[:, :, None, :] + col_vec[:, None, :, :]


def outer_product_difference(
    x: Tensor, rows: Tensor | None = None, cols: Tensor | None = None
) -> Tensor:
    """``cat[x_i * x_j, x_i - x_j]`` -> ``(B, I, J, 2D)`` (ESMFold2 ``SingleToPair``)."""
    xi = select_rows(x, rows)[:, :, None, :]
    xj = select_rows(x, cols)[:, None, :, :]
    return torch.cat([xi * xj, xi - xj], dim=-1)


def token_bond_features(
    bonds: Tensor, rows: Tensor | None = None, cols: Tensor | None = None
) -> Tensor:
    """``(B, I, J, 1)`` float block of the symmetric token-bond matrix ``(B, L, L, 1)``."""
    return select_cols(select_rows(bonds, rows), cols).float()


def distance_bins(
    coords: Tensor, boundaries: Tensor, rows: Tensor | None = None, cols: Tensor | None = None
) -> Tensor:
    """Bin index ``sum(d > boundary)`` of pairwise distances between representative atoms.

    ``coords`` is ``(B, L, 3)``; the result is ``(B, I, J)`` long, values ``0..len(boundaries)``.
    Uses the exact (non-matmul) Euclidean distance like upstream's ``cdist(..., "donot_use_mm")``.
    """
    ci = select_rows(coords, rows).float()
    cj = select_rows(coords, cols).float()
    d = torch.sqrt(((ci[:, :, None, :] - cj[:, None, :, :]) ** 2).sum(-1))
    return (d[..., None] > boundaries.to(d.device)).sum(-1).long()


def relpos_features(
    residue_index: Tensor,
    asym_id: Tensor,
    sym_id: Tensor,
    entity_id: Tensor,
    token_index: Tensor,
    *,
    r_max: int,
    s_max: int,
    rows: Tensor | None = None,
    cols: Tensor | None = None,
) -> Tensor:
    """AF3/ESMFold2 relative-position one-hots, ``(B, I, J, 2*(2r+2) + 1 + (2s+2))``.

    Residue and token deltas are clipped to ``[0, 2r]`` after adding ``r_max`` and sent to the
    extra bin ``2r+1`` when the pair is not on the same chain (residue) or not on the same chain
    and residue (token). The chain delta uses ``sym_id`` and goes to bin ``2s+1`` on the same chain.
    """
    ri, rj = select_rows(residue_index, rows), select_rows(residue_index, cols)
    ai, aj = select_rows(asym_id, rows), select_rows(asym_id, cols)
    si, sj = select_rows(sym_id, rows), select_rows(sym_id, cols)
    ei, ej = select_rows(entity_id, rows), select_rows(entity_id, cols)
    ti, tj = select_rows(token_index, rows), select_rows(token_index, cols)
    same_chain = ai[:, :, None] == aj[:, None, :]
    same_residue = ri[:, :, None] == rj[:, None, :]
    same_entity = ei[:, :, None] == ej[:, None, :]
    d_res = (ri[:, :, None] - rj[:, None, :] + r_max).clamp(0, 2 * r_max)
    d_res = torch.where(same_chain, d_res, torch.full_like(d_res, 2 * r_max + 1))
    d_tok = (ti[:, :, None] - tj[:, None, :] + r_max).clamp(0, 2 * r_max)
    d_tok = torch.where(same_chain & same_residue, d_tok, torch.full_like(d_tok, 2 * r_max + 1))
    d_chain = (si[:, :, None] - sj[:, None, :] + s_max).clamp(0, 2 * s_max)
    d_chain = torch.where(same_chain, torch.full_like(d_chain, 2 * s_max + 1), d_chain)
    return torch.cat(
        [
            F.one_hot(d_res.long(), 2 * r_max + 2).float(),
            F.one_hot(d_tok.long(), 2 * r_max + 2).float(),
            same_entity.float()[..., None],
            F.one_hot(d_chain.long(), 2 * s_max + 2).float(),
        ],
        dim=-1,
    )


class RelativePositionEncoding(nn.Module):
    """Linear embedding of :func:`relpos_features` (checkpoint name ``rel_pos.embed``)."""

    def __init__(self, width: int, *, r_max: int, s_max: int) -> None:
        super().__init__()
        self.r_max, self.s_max = r_max, s_max
        feature_dim = 2 * (2 * r_max + 2) + 1 + (2 * s_max + 2)
        self.embed = nn.Linear(feature_dim, width, bias=False)

    def forward(
        self,
        residue_index: Tensor,
        asym_id: Tensor,
        sym_id: Tensor,
        entity_id: Tensor,
        token_index: Tensor,
        rows: Tensor | None = None,
        cols: Tensor | None = None,
    ) -> Tensor:
        feats = relpos_features(
            residue_index,
            asym_id,
            sym_id,
            entity_id,
            token_index,
            r_max=self.r_max,
            s_max=self.s_max,
            rows=rows,
            cols=cols,
        )
        return self.embed(feats.to(self.embed.weight.dtype))

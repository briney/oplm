"""Confidence head (pLDDT, PAE, PDE, resolved, pTM/ipTM) and the symmetrised distogram readout.

Ported from Biohub's ESMFold2 ``ConfidenceHead``, ``RowAttentionPooling``, ``_categorical_mean``
and the pTM/ipTM code (esm/models/esmfold2/{model,layers}.py, Apache-2.0; see
THIRD_PARTY_NOTICES.md), with the released checkpoint's names (``confidence_head.{boundaries,
dist_bin_pairwise_embed, input_embedder.*, folding_trunk, row_attention_pooling,
{plddt,pae,pde,resolved}_layernorm, plddt_weight, pae_head, pde_head, resolved_weight}``).
Modifications: the distance binning is the block-local ``distance_bins`` (spec §4.6); the
pair trunk's residual quirk (``pair + Trunk(pair)`` although ``Trunk`` already carries its own
residuals) is reproduced deliberately; unused upstream modules (``s_norm``,
``s_inputs_to_single``, ``s_input_to_s``) are not ported; interface pLDDT is deferred to
milestone 2. pLDDT is on the 0–1 scale, PAE/PDE in Å (64 bins over 0–32 Å by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from oplm.fold.atoms import gather_token_to_atom, intra_token_index, scatter_atom_to_token_mean
from oplm.fold.pair import distance_bins
from oplm.fold.trunk import PairStack, cuda_bf16_autocast, pair_stack_kwargs

if TYPE_CHECKING:
    from torch import Tensor

    from oplm.fold.configuration_fold import FoldConfig

__all__ = [
    "ConfidenceHead",
    "ConfidenceInputEmbedder",
    "ConfidenceOutput",
    "RowAttentionPooling",
    "categorical_mean",
    "symmetrized_distogram",
    "tm_scores",
]

_EPS = 1e-6


def symmetrized_distogram(head: nn.Linear, z: Tensor) -> Tensor:
    """``distogram_head(z + zᵀ)`` on the fp32 final pair (upstream reads the symmetrised pair)."""
    return head(z + z.transpose(1, 2))


def categorical_mean(logits: Tensor, start: float, end: float) -> Tensor:
    """Expected value over equal-width bins spanning ``[start, end]`` (bin centers)."""
    n_bins = logits.shape[-1]
    edges = torch.linspace(start, end, n_bins + 1, device=logits.device, dtype=torch.float32)
    centers = (edges[:-1] + edges[1:]) / 2
    return logits.float().softmax(dim=-1) @ centers


def tm_scores(
    pae_logits: Tensor, token_mask: Tensor, asym_id: Tensor, *, max_dist: float = 32.0
) -> tuple[Tensor, Tensor, Tensor]:
    """pTM, ipTM and per-chain-pair ipTM from PAE logits (upstream ``model.py:340-386``).

    ``d0 = 1.24 (max(N, 19) − 15)^(1/3) − 1.8`` with ``N`` the valid-token count; pTM is the
    max over frame rows of the masked mean of the expected TM term; ipTM restricts columns to
    other chains (0 for a single chain); ``pair_chains_iptm[c1, c2]`` is the max over rows in
    chain ``c2`` of the mean over columns in chain ``c1``.
    """
    n_bins = pae_logits.shape[-1]
    bin_width = max_dist / n_bins
    centers = torch.arange(0.5 * bin_width, max_dist, bin_width, device=pae_logits.device)
    mask_f = token_mask.float()
    n_res = mask_f.sum(dim=-1, keepdim=True)
    d0 = 1.24 * (n_res.clamp(min=19) - 15) ** (1 / 3) - 1.8
    tm_per_bin = 1 / (1 + (centers[None, :] / d0) ** 2)
    tm_expected = (pae_logits.float().softmax(dim=-1) * tm_per_bin[:, None, None, :]).sum(-1)
    pair = mask_f[:, :, None] * mask_f[:, None, :]
    ptm = ((tm_expected * pair).sum(-1) / (pair.sum(-1) + _EPS)).max(-1).values
    inter = (asym_id[:, :, None] != asym_id[:, None, :]).float() * pair
    iptm = ((tm_expected * inter).sum(-1) / (inter.sum(-1) + _EPS)).max(-1).values
    n_chains = int(asym_id.max().item()) + 1
    chains = torch.zeros(pae_logits.shape[0], n_chains, n_chains, device=pae_logits.device)
    for c1 in range(n_chains):
        cols = (asym_id == c1).float() * mask_f
        row_vals = (tm_expected * cols[:, None, :]).sum(-1) / (cols[:, None, :].sum(-1) + _EPS)
        for c2 in range(n_chains):
            rows = (asym_id == c2) & token_mask
            masked = row_vals.masked_fill(~rows, float("-inf")).max(-1).values
            chains[:, c1, c2] = masked.clamp(min=0.0)
    return ptm, iptm, chains


class ConfidenceInputEmbedder(nn.Module):
    """Checkpoint ``confidence_head.input_embedder``: the pair the confidence trunk starts from."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__()
        s_in, p, eps = config.single_inputs_width, config.pair_width, config.layer_norm_eps
        self.single_inputs_norm = nn.LayerNorm(s_in, eps=eps)
        self.pair_norm = nn.LayerNorm(p, eps=eps)
        self.single_to_pair = nn.Linear(s_in, p, bias=False)
        self.single_to_pair_transpose = nn.Linear(s_in, p, bias=False)
        self.single_to_pair_prod_in1 = nn.Linear(s_in, p, bias=False)
        self.single_to_pair_prod_in2 = nn.Linear(s_in, p, bias=False)
        self.single_to_pair_prod_out = nn.Linear(p, p, bias=False)

    def forward(self, s_inputs: Tensor, z: Tensor, relpos: Tensor, bonds: Tensor) -> Tensor:
        s = self.single_inputs_norm(s_inputs)
        z = self.pair_norm(z) + relpos + bonds
        z = z + self.single_to_pair(s)[:, :, None, :]
        z = z + self.single_to_pair_transpose(s)[:, None, :, :]
        prod = (
            self.single_to_pair_prod_in1(s)[:, :, None, :]
            * self.single_to_pair_prod_in2(s)[:, None, :, :]
        )
        return z + self.single_to_pair_prod_out(prod)


class RowAttentionPooling(nn.Module):
    """Softmax over columns of a learned score, masked at padded columns, then a projection."""

    def __init__(self, pair_width: int, single_width: int) -> None:
        super().__init__()
        self.attn_proj = nn.Linear(pair_width, 1, bias=False)
        self.out_proj = nn.Linear(pair_width, single_width, bias=False)

    def forward(self, z: Tensor, token_mask: Tensor) -> Tensor:
        scores = self.attn_proj(z).squeeze(-1) + torch.where(token_mask[:, None, :], 0.0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return self.out_proj(torch.einsum("bnm,bnmd->bnd", weights, z))


@dataclass
class ConfidenceOutput:
    """Per-sample confidence tensors; leading dim is ``B · num_samples``."""

    plddt_logits: Tensor  # (N, A, plddt_bins)
    plddt_per_atom: Tensor  # (N, A) in [0, 1]
    plddt: Tensor  # (N, L) masked mean over each token's atoms
    pae_logits: Tensor  # (N, L, L, pae_bins)
    pae: Tensor  # (N, L, L) Å
    pde_logits: Tensor
    pde: Tensor
    resolved_logits: Tensor  # (N, A, 2)
    ptm: Tensor  # (N,)
    iptm: Tensor  # (N,)
    pair_chains_iptm: Tensor  # (N, C, C)
    complex_plddt: Tensor  # (N,) mean over valid atoms


class ConfidenceHead(nn.Module):
    """Checkpoint ``confidence_head``."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__()
        self.config = config
        p, single, eps = config.pair_width, config.inputs_token_width, config.layer_norm_eps
        self.register_buffer(
            "boundaries",
            torch.linspace(
                config.confidence_min_dist,
                config.confidence_max_dist,
                config.confidence_dist_bins - 1,
            ),
        )
        self.dist_bin_pairwise_embed = nn.Embedding(config.confidence_dist_bins, p)
        self.input_embedder = ConfidenceInputEmbedder(config)
        self.folding_trunk = PairStack(config.confidence_blocks, p, **pair_stack_kwargs(config))
        self.row_attention_pooling = RowAttentionPooling(p, single)
        self.plddt_layernorm = nn.LayerNorm(single, eps=eps)
        self.plddt_weight = nn.Parameter(
            torch.zeros(config.max_atoms_per_token, single, config.plddt_bins)
        )
        self.pae_layernorm = nn.LayerNorm(p, eps=eps)
        self.pae_head = nn.Linear(p, config.pae_bins, bias=False)
        self.pde_layernorm = nn.LayerNorm(p, eps=eps)
        self.pde_head = nn.Linear(p, config.pde_bins, bias=False)
        self.resolved_layernorm = nn.LayerNorm(single, eps=eps)
        self.resolved_weight = nn.Parameter(torch.zeros(config.max_atoms_per_token, single, 2))

    def forward(
        self,
        *,
        s_inputs: Tensor,
        z: Tensor,
        relpos: Tensor,
        bonds: Tensor,
        coords: Tensor,
        distogram_atom_idx: Tensor,
        token_mask: Tensor,
        atom_to_token: Tensor,
        atom_mask: Tensor,
        asym_id: Tensor,
    ) -> ConfidenceOutput:
        """Score ``coords (B·S, A, 3)`` against the base-batch trunk tensors, per sample."""
        num_samples = coords.shape[0] // z.shape[0]

        def rep(t: Tensor) -> Tensor:
            return t.repeat_interleave(num_samples, dim=0)

        pair = rep(self.input_embedder(s_inputs, z.float(), relpos.float(), bonds.float()))
        token_mask, atom_to_token, atom_mask, asym_id = map(
            rep, (token_mask, atom_to_token, atom_mask, asym_id)
        )
        rep_idx = rep(distogram_atom_idx)
        rep_coords = torch.gather(coords.float(), 1, rep_idx[..., None].expand(-1, -1, 3))
        bins = distance_bins(
            rep_coords,
            self.boundaries,  # ty: ignore[invalid-argument-type]  # registered buffer typed Tensor | Module
        )
        pair = pair + self.dist_bin_pairwise_embed(bins)
        pair_mask = token_mask[:, :, None].float() * token_mask[:, None, :].float()
        with cuda_bf16_autocast(pair.is_cuda):
            delta = self.folding_trunk(pair, pair_mask)
        pair = pair + delta.float()  # upstream quirk: ``delta`` already includes the residual
        single = self.row_attention_pooling(pair, token_mask)
        pae_logits = self.pae_head(self.pae_layernorm(pair))
        pde_logits = self.pde_head(self.pde_layernorm(pair))
        s_atoms = gather_token_to_atom(single, atom_to_token)
        slot = intra_token_index(atom_to_token).clamp(max=self.plddt_weight.shape[0] - 1)
        plddt_logits = torch.einsum(
            "...c,...cb->...b", self.plddt_layernorm(s_atoms), self.plddt_weight[slot]
        )
        resolved_logits = torch.einsum(
            "...c,...cb->...b", self.resolved_layernorm(s_atoms), self.resolved_weight[slot]
        )
        plddt_per_atom = categorical_mean(plddt_logits, 0.0, 1.0)
        plddt = scatter_atom_to_token_mean(
            plddt_per_atom[..., None], atom_to_token, token_mask.shape[1], atom_mask
        )[..., 0]
        atom_f = atom_mask.float()
        complex_plddt = (plddt_per_atom * atom_f).sum(-1) / atom_f.sum(-1).clamp(min=1.0)
        ptm, iptm, pair_chains_iptm = tm_scores(
            pae_logits, token_mask, asym_id, max_dist=self.config.pae_max_dist
        )
        return ConfidenceOutput(
            plddt_logits=plddt_logits,
            plddt_per_atom=plddt_per_atom,
            plddt=plddt,
            pae_logits=pae_logits,
            pae=categorical_mean(pae_logits, 0.0, self.config.pae_max_dist),
            pde_logits=pde_logits,
            pde=categorical_mean(pde_logits, 0.0, self.config.pae_max_dist),
            resolved_logits=resolved_logits,
            ptm=ptm,
            iptm=iptm,
            pair_chains_iptm=pair_chains_iptm,
            complex_plddt=complex_plddt,
        )

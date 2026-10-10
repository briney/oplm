"""Confidence head against the transcribed upstream forward; pTM/ipTM edge cases; distogram."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from oplm.fold.atoms import gather_token_to_atom, intra_token_index
from oplm.fold.confidence import (
    ConfidenceHead,
    categorical_mean,
    symmetrized_distogram,
    tm_scores,
)
from oplm.fold.data.featurize import ChainSpec, featurize
from tests.fold.helpers import tiny_fold_config

if TYPE_CHECKING:
    from oplm.fold.configuration_fold import FoldConfig


def _cfg() -> FoldConfig:
    return tiny_fold_config(plddt_bins=10, pae_bins=8, pde_bins=8, confidence_dist_bins=5)


def test_names_and_shapes() -> None:
    cfg = _cfg()
    head = ConfidenceHead(cfg)
    names = set(head.state_dict())
    assert {
        "boundaries",
        "dist_bin_pairwise_embed.weight",
        "input_embedder.single_inputs_norm.weight",
        "input_embedder.pair_norm.bias",
        "input_embedder.single_to_pair.weight",
        "input_embedder.single_to_pair_transpose.weight",
        "input_embedder.single_to_pair_prod_in1.weight",
        "input_embedder.single_to_pair_prod_in2.weight",
        "input_embedder.single_to_pair_prod_out.weight",
        "folding_trunk.layers.0.tri_mul_out.proj_bundle.weight",
        "row_attention_pooling.attn_proj.weight",
        "row_attention_pooling.out_proj.weight",
        "plddt_layernorm.weight",
        "plddt_weight",
        "pae_layernorm.bias",
        "pae_head.weight",
        "pde_layernorm.weight",
        "pde_head.weight",
        "resolved_layernorm.bias",
        "resolved_weight",
    } <= names
    assert len(names) == 25 + 18  # 25 head tensors + one PairUpdateBlock (18)
    assert head.boundaries.shape == (4,)
    assert torch.equal(head.boundaries, torch.linspace(3.25, 50.75, 4))
    assert head.dist_bin_pairwise_embed.weight.shape == (5, 32)
    assert head.plddt_weight.shape == (23, 32, 10) and head.plddt_weight.abs().sum() == 0
    assert head.resolved_weight.shape == (23, 32, 2)
    assert head.row_attention_pooling.out_proj.weight.shape == (32, 32)  # token_width // 2
    assert head.pae_head.weight.shape == (8, 32) and head.pae_head.bias is None


def _run(
    head: ConfidenceHead,
    cfg: FoldConfig,
    chains: list[ChainSpec],
    num_samples: int,
    pad: int | None = None,
):
    f = featurize(chains, pad_tokens_to=pad)
    g = torch.Generator().manual_seed(0)
    L, A = f.num_tokens, f.num_atoms
    s_inputs = torch.randn(1, L, cfg.single_inputs_width, generator=g)
    z = torch.randn(1, L, L, cfg.pair_width, generator=g)
    relpos = torch.randn(1, L, L, cfg.pair_width, generator=g)
    bonds = torch.randn(1, L, L, cfg.pair_width, generator=g)
    coords = torch.randn(num_samples, A, 3, generator=g) * 5
    with torch.no_grad():
        out = head(
            s_inputs=s_inputs,
            z=z,
            relpos=relpos,
            bonds=bonds,
            coords=coords,
            distogram_atom_idx=f.distogram_atom_idx,
            token_mask=f.token_mask,
            atom_to_token=f.atom_to_token,
            atom_mask=f.atom_mask,
            asym_id=f.asym_id,
        )
    return f, (s_inputs, z, relpos, bonds, coords), out


def test_forward_matches_transcribed_upstream_confidence_head() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    head = ConfidenceHead(cfg).eval()
    with torch.no_grad():
        head.plddt_weight.normal_()
        head.resolved_weight.normal_()
    chains = [ChainSpec("MKV", "A"), ChainSpec("GG", "B")]
    f, (s_inputs, z, relpos, bonds, coords), out = _run(head, cfg, chains, 2)
    ie = head.input_embedder
    with torch.no_grad():
        # upstream ConfidenceHead.forward (model.py:179-300), transcribed
        s = ie.single_inputs_norm(s_inputs)
        z_base = ie.pair_norm(z) + relpos + bonds
        z_base = z_base + ie.single_to_pair(s).unsqueeze(2)
        z_base = z_base + ie.single_to_pair_transpose(s).unsqueeze(1)
        prod = (
            ie.single_to_pair_prod_in1(s)[:, :, None, :]
            * ie.single_to_pair_prod_in2(s)[:, None, :, :]
        )
        z_base = z_base + ie.single_to_pair_prod_out(prod)
        pair = z_base.repeat_interleave(2, 0)
        rep_idx = f.distogram_atom_idx.repeat_interleave(2, 0)
        rep = torch.gather(coords, 1, rep_idx[..., None].expand(-1, -1, 3))
        d = torch.cdist(rep, rep, compute_mode="donot_use_mm_for_euclid_dist")
        bins = (d.unsqueeze(-1) > head.boundaries).sum(-1).long()
        pair = pair + head.dist_bin_pairwise_embed(bins)
        mask = f.token_mask.repeat_interleave(2, 0)
        pair_mask = mask[:, :, None].float() * mask[:, None, :].float()
        pair = pair + head.folding_trunk(pair, pair_mask).float()  # the upstream residual quirk
        scores = head.row_attention_pooling.attn_proj(pair).squeeze(-1) + torch.where(
            mask[:, None, :], 0.0, -1e9
        )
        single = head.row_attention_pooling.out_proj(
            torch.einsum("bnm,bnmd->bnd", scores.softmax(-1), pair)
        )
        pae_logits = head.pae_head(head.pae_layernorm(pair))
        a2t = f.atom_to_token.repeat_interleave(2, 0)
        s_atoms = gather_token_to_atom(single, a2t)
        slot = intra_token_index(a2t).clamp(max=22)
        plddt_logits = torch.einsum(
            "...c,...cb->...b", head.plddt_layernorm(s_atoms), head.plddt_weight[slot]
        )
    torch.testing.assert_close(out.pae_logits, pae_logits)
    torch.testing.assert_close(out.plddt_logits, plddt_logits)
    torch.testing.assert_close(out.plddt_per_atom, categorical_mean(plddt_logits, 0.0, 1.0))
    torch.testing.assert_close(out.pae, categorical_mean(pae_logits, 0.0, 32.0))
    assert out.plddt.shape == (2, 5) and out.resolved_logits.shape == (2, f.num_atoms, 2)
    assert out.pair_chains_iptm.shape == (2, 2, 2) and out.complex_plddt.shape == (2,)
    assert (out.plddt >= 0).all() and (out.plddt <= 1).all()


def test_categorical_mean_bin_centers() -> None:
    logits = torch.full((1, 4), -1e9)
    logits[0, 2] = 0.0
    torch.testing.assert_close(categorical_mean(logits, 0.0, 1.0), torch.tensor([0.625]))
    torch.testing.assert_close(categorical_mean(logits, 0.0, 32.0), torch.tensor([20.0]))


def test_tm_scores_match_transcription_and_single_chain_iptm_is_zero() -> None:
    g = torch.Generator().manual_seed(0)
    pae_logits = torch.randn(2, 6, 6, 8, generator=g)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, 5] = False
    asym = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0]])
    ptm, iptm, chains = tm_scores(pae_logits, mask, asym, max_dist=32.0)
    # upstream (model.py:340-386), transcribed
    bw = 32.0 / 8
    centers = torch.arange(0.5 * bw, 32.0, bw)
    mask_f = mask.float()
    n_res = mask_f.sum(-1, keepdim=True)
    d0 = 1.24 * (n_res.clamp(min=19) - 15) ** (1 / 3) - 1.8
    tm_per_bin = 1 / (1 + (centers / d0) ** 2)
    tm_expected = (F.softmax(pae_logits, -1) * tm_per_bin[:, None, None, :]).sum(-1)
    pair = mask_f[..., None] * mask_f[:, None, :]
    ptm_ref = ((tm_expected * pair).sum(-1) / (pair.sum(-1) + 1e-6)).max(-1).values
    inter = (asym[..., None] != asym[:, None, :]).float() * pair
    iptm_ref = ((tm_expected * inter).sum(-1) / (inter.sum(-1) + 1e-6)).max(-1).values
    torch.testing.assert_close(ptm, ptm_ref)
    torch.testing.assert_close(iptm, iptm_ref)
    assert iptm[1] == 0.0  # Review Focus 3: single chain -> no inter-chain pairs -> 0, not NaN
    assert chains.shape == (2, 2, 2) and torch.isfinite(chains).all()
    assert chains[1, 1, 1] == 0.0 and chains[0, 0, 1] >= 0


def test_all_padding_row_is_finite() -> None:
    """Review Focus 2: a batch row with every token padded must not produce NaN anywhere."""
    cfg = _cfg()
    torch.manual_seed(0)
    head = ConfidenceHead(cfg).eval()
    f = featurize([ChainSpec("MK", "A")], pad_tokens_to=4)
    L, A = 4, f.num_atoms
    with torch.no_grad():
        out = head(
            s_inputs=torch.randn(2, L, cfg.single_inputs_width),
            z=torch.randn(2, L, L, 32),
            relpos=torch.zeros(2, L, L, 32),
            bonds=torch.zeros(2, L, L, 32),
            coords=torch.randn(2, A, 3),
            distogram_atom_idx=f.distogram_atom_idx.expand(2, L),
            token_mask=torch.stack([f.token_mask[0], torch.zeros(L, dtype=torch.bool)]),
            atom_to_token=f.atom_to_token.expand(2, A),
            atom_mask=torch.stack([f.atom_mask[0], torch.zeros(A, dtype=torch.bool)]),
            asym_id=f.asym_id.expand(2, L),
        )
    for name in ("plddt", "pae", "pde", "ptm", "iptm", "pair_chains_iptm", "complex_plddt"):
        assert torch.isfinite(getattr(out, name)).all(), name
    assert out.ptm[1] == 0.0 and out.complex_plddt[1] == 0.0


def test_symmetrized_distogram() -> None:
    torch.manual_seed(0)
    head = nn.Linear(32, 8)
    z = torch.randn(1, 5, 5, 32)
    out = symmetrized_distogram(head, z)
    torch.testing.assert_close(out, head(z + z.transpose(1, 2)))
    torch.testing.assert_close(out, out.transpose(1, 2))

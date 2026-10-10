"""Atom features, 3D RoPE, SWA atom attention/blocks, encoders and the inputs embedder."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from oplm.fold.atoms import (
    AtomAttention,
    AtomBlock,
    AtomDecoder,
    AtomEncoder,
    InputsEmbedder,
    apply_rotary_3d,
    atom_ffn_hidden,
    build_3d_rope,
    build_atom_features,
    gather_token_to_atom,
    intra_token_index,
    scatter_atom_to_token_mean,
)
from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.data.featurize import ChainSpec, featurize

_CFG = FoldConfig(trunk_blocks=1, lm_encoder_blocks=1, coda_blocks=1, attention_backend="dense")


def test_atom_feature_layout_matches_upstream_order() -> None:
    f = featurize([ChainSpec("MK", "A")])
    feats = build_atom_features(
        f.ref_pos,
        f.ref_charge,
        f.atom_mask,
        f.ref_element,
        f.ref_atom_name_chars,
        max_atomic_number=128,
        name_vocab=64,
    )
    assert feats.shape == (1, 32, 389)
    torch.testing.assert_close(feats[0, :, :3], f.ref_pos[0])
    assert feats[0, 8 + 8, 3] == 1.0  # LYS NZ charge
    assert feats[0, :17, 4].eq(1).all() and feats[0, 17:, 4].eq(0).all()  # mask channel
    assert feats[0, 0, 5 + 7] == 1.0 and feats[0, 0, 5:133].sum() == 1  # N: element one-hot
    assert feats[0, 1, 133 + 0 * 64 + 35] == 1.0 and feats[0, 1, 133 + 1 * 64 + 33] == 1.0  # "CA"
    assert feats[0, 17:].abs().sum() == 0  # padded atoms contribute nothing


def _upstream_rope(ref_pos: torch.Tensor, uid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ESMFold2 build_3d_rope (layers.py), transcribed with the released constants."""
    B, N = ref_pos.shape[:2]
    half_dim = 16
    sp = 1.0 / (20.0 ** (torch.arange(0, 2, dtype=torch.float32) / 2))
    ui = 1.0 / (10000.0 ** (torch.arange(0, 10, dtype=torch.float32) / 10))
    spatial = torch.einsum("bna,k->bnak", ref_pos.float(), sp).reshape(B, N, 6)
    uidf = torch.einsum("bn,k->bnk", uid.float(), ui)
    freqs = torch.cat([spatial, uidf], dim=-1)
    if freqs.shape[-1] < half_dim:
        freqs = torch.cat([freqs, torch.zeros(B, N, half_dim - freqs.shape[-1])], dim=-1)
    return freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)


def test_3d_rope_matches_transcription_and_neox_rotation() -> None:
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(2, 9, 3, generator=g) * 5
    uid = torch.randint(0, 40, (2, 9), generator=g)
    cos, sin = build_3d_rope(
        pos,
        uid,
        head_dim=32,
        spatial_pairs_per_axis=2,
        uid_pairs=10,
        spatial_base=20.0,
        uid_base=10000.0,
    )
    ref_cos, ref_sin = _upstream_rope(pos, uid)
    assert torch.equal(cos, ref_cos) and torch.equal(sin, ref_sin) and cos.dtype == torch.bfloat16
    x = torch.randn(2, 9, 4, 32, generator=g)
    x1, x2 = x.chunk(2, dim=-1)
    c = cos[:, :, None, :].repeat(1, 1, 1, 2)
    s = sin[:, :, None, :].repeat(1, 1, 1, 2)
    expected = x * c + torch.cat((-x2, x1), dim=-1) * s
    torch.testing.assert_close(apply_rotary_3d(x, cos, sin), expected)


def test_gather_scatter_and_intra_index() -> None:
    tok = torch.randn(1, 3, 4)
    a2t = torch.tensor([[0, 0, 1, 2, 2, 0, 0]])  # last two are pads mapped to 0
    mask = torch.tensor([[True, True, True, True, True, False, False]])
    torch.testing.assert_close(gather_token_to_atom(tok, a2t)[0, 2], tok[0, 1])
    atoms = torch.arange(7, dtype=torch.float32)[None, :, None].expand(1, 7, 2)
    pooled = scatter_atom_to_token_mean(atoms, a2t, 3, mask)
    torch.testing.assert_close(pooled[0, :, 0], torch.tensor([0.5, 2.0, 3.5]))
    assert intra_token_index(a2t)[0].tolist() == [0, 1, 0, 0, 1, 0, 1]


def test_scatter_uses_explicit_token_count() -> None:
    """Review Focus 1: a trailing token with no atoms must not shorten the token axis."""
    atoms = torch.ones(1, 4, 2)
    a2t = torch.tensor([[0, 0, 1, 1]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    pooled = scatter_atom_to_token_mean(atoms, a2t, 3, mask)
    assert pooled.shape == (1, 3, 2) and pooled[0, 2].abs().sum() == 0


def _upstream_swa_attention(attn: AtomAttention, x: torch.Tensor, cos, sin, valid) -> torch.Tensor:
    """ESMFold2 SWA3DRoPEAttention.forward non-flash branch, transcribed onto our projections."""
    B, N, _ = x.shape
    H, D = attn.heads, attn.head_dim
    q = attn.q_proj(x).view(B, N, H, D)
    k = attn.k_proj(x).view(B, N, H, D)
    v = attn.v_proj(x).view(B, N, H, D)
    q = F.rms_norm(q, (D,)).to(q.dtype)
    k = F.rms_norm(k, (D,)).to(k.dtype)
    q, k = apply_rotary_3d(q, cos, sin), apply_rotary_3d(k, cos, sin)
    in_dtype = q.dtype
    q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    rank = torch.cumsum(valid, dim=1) - 1
    within = (rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= attn.half_window
    allowed = within & valid.unsqueeze(1) & valid.unsqueeze(2)
    allowed |= torch.eye(N, dtype=torch.bool)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=allowed.unsqueeze(1),
        scale=D**-0.5,
    ).transpose(1, 2)
    out = out * valid.unsqueeze(-1).unsqueeze(-1)
    out = out.to(in_dtype).reshape(B, N, -1)
    out = out * torch.sigmoid(attn.gate_proj(x))
    return attn.o_proj(out)


def test_atom_attention_matches_upstream_transcription() -> None:
    torch.manual_seed(0)
    attn = AtomAttention(32, 2, 2, backend="dense")
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 10, 32, generator=g)
    pos = torch.randn(2, 10, 3, generator=g)
    uid = torch.arange(10)[None].expand(2, 10)
    valid = torch.ones(2, 10, dtype=torch.bool)
    valid[1, 7:] = False
    cos, sin = build_3d_rope(
        pos,
        uid,
        head_dim=16,
        spatial_pairs_per_axis=2,
        uid_pairs=2,
        spatial_base=20.0,
        uid_base=10000.0,
    )
    out = attn(x, cos, sin, valid)
    torch.testing.assert_close(
        out, _upstream_swa_attention(attn, x, cos, sin, valid), atol=1e-2, rtol=1e-2
    )
    assert out[1, 7:].abs().sum() == 0  # invalid atoms: attention output zero, gate*0 = 0


def test_atom_block_chunk_order_and_zero_init() -> None:
    torch.manual_seed(0)
    block = AtomBlock(32, 2, 4, backend="dense")
    assert block.adaln_linear.weight.abs().sum() == 0 and block.adaln_linear.bias is None
    assert block.mlp.gate_up_proj.weight.shape == (2 * atom_ffn_hidden(32), 32)
    x, c = torch.randn(1, 6, 32), torch.randn(1, 6, 32)
    pos, uid = torch.randn(1, 6, 3), torch.arange(6)[None]
    cos, sin = build_3d_rope(
        pos,
        uid,
        head_dim=16,
        spatial_pairs_per_axis=2,
        uid_pairs=2,
        spatial_base=20.0,
        uid_base=10000.0,
    )
    valid = torch.ones(1, 6, dtype=torch.bool)
    torch.testing.assert_close(block(x, c, cos, sin, valid), x)  # all gates zero -> identity
    # gate_a is chunk index 2 of [shift_a, scale_a, gate_a, shift_f, scale_f, gate_f]
    with torch.no_grad():
        block.adaln_linear.weight[2 * 32 : 3 * 32] = 0.1  # only gate_a is non-zero
    out = block(x, c, cos, sin, valid)
    gate_a = block.adaln_linear(F.silu(c)).chunk(6, dim=-1)[2]
    expected = x + gate_a * block.self_attn(F.rms_norm(x, (32,)), cos, sin, valid)
    torch.testing.assert_close(out, expected)


def test_encoder_decoder_names_and_shapes() -> None:
    enc = AtomEncoder(_CFG, out_width=_CFG.inputs_token_width, num_blocks=3, heads=4)
    dec = AtomDecoder(_CFG, num_blocks=3, heads=4)
    enc_names, dec_names = set(enc.state_dict()), set(dec.state_dict())
    assert {
        "atom_linear.weight",
        "atom_norm.weight",
        "atom_norm.bias",
        "atom_to_token_linear.weight",
        "layers.2.adaln_linear.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.gate_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.gate_up_proj.weight",
        "layers.0.mlp.down_proj.weight",
    } <= enc_names
    assert {
        "token_to_atom_linear.weight",
        "norm.weight",
        "output_linear.weight",
        "layers.2.mlp.down_proj.weight",
    } <= dec_names
    assert enc.atom_linear.weight.shape == (128, 389) and enc.atom_to_token_linear.weight.shape == (
        384,
        128,
    )
    assert enc.layers[0].adaln_linear.weight.shape == (768, 128) and enc.layers[
        0
    ].mlp.gate_up_proj.weight.shape == (512, 128)
    assert dec.token_to_atom_linear.weight.shape == (
        128,
        768,
    ) and dec.output_linear.weight.shape == (3, 128)
    # 4 non-block tensors + 8 per block (adaln, q/k/v/gate/o, gate_up, down) x 3 blocks
    assert len(enc_names) == 4 + 8 * 3 and len(dec_names) == 4 + 8 * 3


def test_inputs_embedder_shapes_padding_and_names() -> None:
    torch.manual_seed(0)
    emb = InputsEmbedder(_CFG).eval()
    f = featurize([ChainSpec("MKV", "A")], pad_tokens_to=8)
    with torch.no_grad():
        out = emb(f)
    assert out.s_inputs.shape == (1, 8, 451) and out.z_init.shape == (1, 8, 8, 256)
    assert out.relpos.shape == out.bonds.shape == (1, 8, 8, 256)
    # order: [atom aggregation 384 | res_type one-hot 33 | profile = one-hot 33 | deletion_mean 0]
    assert out.s_inputs[0, 0, 384 + 14] == 1 and out.s_inputs[0, 0, 384 + 33 + 14] == 1
    assert out.s_inputs[0, :, 450].abs().sum() == 0
    assert out.s_inputs[0, 3:].abs().sum() == 0  # padded tokens: no atoms, zeroed one-hots
    assert (out.s_inputs[0, :3, :384] >= 0).all()  # relu'd aggregation
    torch.testing.assert_close(
        out.z_init[0, 5, 6], out.relpos[0, 5, 6] + out.bonds[0, 5, 6]
    )  # bias-free inits
    assert {
        "atom_encoder.atom_linear.weight",
        "pair_init_1.weight",
        "pair_init_2.weight",
        "rel_pos.embed.weight",
        "token_bonds.weight",
    } <= set(emb.state_dict())
    assert emb.rel_pos.embed.weight.shape == (256, 139) and emb.token_bonds.weight.shape == (256, 1)

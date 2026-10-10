"""Atom-level machinery: reference features, 3D RoPE, SWA atom blocks, encoders, inputs embedder.

Ported from Biohub's ESMFold2 ``build_3d_rope``/``apply_rotary_emb_3d``, ``SWA3DRoPEAttention``,
``SWAAtomBlock``, ``EsmFold2AtomEncoder``/``Decoder`` and ``InputsEmbedder``
(esm/models/esmfold2/layers.py, Apache-2.0; see THIRD_PARTY_NOTICES.md). Parameter names
follow the released HF checkpoint (``layers.N.{adaln_linear, self_attn.{q,k,v,gate,o}_proj,
mlp.{gate_up_proj,down_proj}}``, ``atom_linear``, ``atom_norm``, ``atom_to_token_linear``,
``token_to_atom_linear``, ``norm``, ``output_linear``, ``pair_init_{1,2}``, ``rel_pos.embed``,
``token_bonds``). Modifications: the windowed attention is milestone 0's
``sliding_window_attention`` (FlexAttention / dense oracle) with a precomputed block mask;
the token count is explicit (upstream derives it from ``atom_to_token.max() + 1``); the
diffusion head's ``coords_linear`` offset enters the encoder through its ``q`` argument.
Precision: Q/K/V are cast to bf16 before attention and the RoPE tables are bf16, exactly as
upstream (parity requires it); everything else follows the caller's autocast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from oplm.fold.attention import sliding_window_attention
from oplm.fold.pair import RelativePositionEncoding
from oplm.fold.trunk import GatedMLP

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn.attention.flex_attention import BlockMask

    from oplm.fold.attention import AttentionBackend
    from oplm.fold.configuration_fold import FoldConfig
    from oplm.fold.data.featurize import FoldFeatures

__all__ = [
    "AtomAttention",
    "AtomBlock",
    "AtomDecoder",
    "AtomEncoder",
    "InputsEmbedder",
    "InputsEmbedding",
    "apply_rotary_3d",
    "atom_ffn_hidden",
    "build_3d_rope",
    "build_atom_features",
    "gather_token_to_atom",
    "intra_token_index",
    "scatter_atom_to_token_mean",
]


def build_atom_features(
    ref_pos: Tensor,
    ref_charge: Tensor,
    atom_mask: Tensor,
    ref_element: Tensor,
    ref_atom_name_chars: Tensor,
    *,
    max_atomic_number: int,
    name_vocab: int,
) -> Tensor:
    """``[pos 3 | charge 1 | mask 1 | element one-hot | name chars one-hot]``, zero on pads."""
    m = atom_mask.to(ref_pos.dtype)[..., None]
    element = F.one_hot(ref_element.long(), max_atomic_number).to(ref_pos.dtype) * m
    chars = F.one_hot(ref_atom_name_chars.long(), name_vocab).to(ref_pos.dtype) * m[..., None]
    return torch.cat(
        [ref_pos, ref_charge.to(ref_pos.dtype)[..., None], m, element, chars.flatten(-2)], dim=-1
    )


def build_3d_rope(
    ref_pos: Tensor,
    ref_space_uid: Tensor,
    *,
    head_dim: int,
    spatial_pairs_per_axis: int,
    uid_pairs: int,
    spatial_base: float,
    uid_base: float,
) -> tuple[Tensor, Tensor]:
    """bf16 ``(cos, sin)``, ``(B, A, head_dim // 2)``: 3 axes x spatial pairs, then uid pairs."""
    device = ref_pos.device
    half = head_dim // 2
    sp_inv = 1.0 / (
        spatial_base
        ** (
            torch.arange(spatial_pairs_per_axis, dtype=torch.float32, device=device)
            / spatial_pairs_per_axis
        )
    )
    uid_inv = 1.0 / (
        uid_base ** (torch.arange(uid_pairs, dtype=torch.float32, device=device) / uid_pairs)
    )
    spatial = torch.einsum("bna,k->bnak", ref_pos.float(), sp_inv).flatten(-2)
    uid = torch.einsum("bn,k->bnk", ref_space_uid.float(), uid_inv)
    freqs = torch.cat([spatial, uid], dim=-1)
    if freqs.shape[-1] < half:
        freqs = F.pad(freqs, (0, half - freqs.shape[-1]))
    return freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)


def apply_rotary_3d(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """NeoX half-split rotation of the first ``2 * cos.shape[-1]`` channels of ``x (B,A,H,D)``."""
    ro = cos.shape[-1] * 2
    c = cos[:, :, None, :].repeat(1, 1, 1, 2)
    s = sin[:, :, None, :].repeat(1, 1, 1, 2)
    xr, rest = x[..., :ro], x[..., ro:]
    x1, x2 = xr.chunk(2, dim=-1)
    rotated = xr * c + torch.cat((-x2, x1), dim=-1) * s
    return torch.cat([rotated, rest], dim=-1)


def gather_token_to_atom(token_features: Tensor, atom_to_token: Tensor) -> Tensor:
    """``(B, L, D), (B, A) -> (B, A, D)``."""
    idx = atom_to_token[..., None].expand(-1, -1, token_features.shape[-1])
    return torch.gather(token_features, 1, idx)


def scatter_atom_to_token_mean(
    atom_features: Tensor, atom_to_token: Tensor, n_tokens: int, atom_mask: Tensor
) -> Tensor:
    """Mean of each token's valid atoms, ``(B, A, D) -> (B, n_tokens, D)``; empty tokens are 0."""
    B, A, D = atom_features.shape
    idx = torch.where(atom_mask, atom_to_token, torch.full_like(atom_to_token, n_tokens))
    out = torch.zeros(B, n_tokens + 1, D, device=atom_features.device, dtype=atom_features.dtype)
    out.scatter_reduce_(
        1, idx[..., None].expand(B, A, D), atom_features, reduce="mean", include_self=False
    )
    return out[:, :n_tokens]


def intra_token_index(atom_to_token: Tensor) -> Tensor:
    """0-based slot of each atom inside its (contiguous) token, ``(B, A)``."""
    same_as_prev = F.pad(atom_to_token[:, 1:] == atom_to_token[:, :-1], (1, 0), value=False)
    cumsum = torch.cumsum(torch.ones_like(atom_to_token), dim=-1)
    group_start = torch.cummax(cumsum.masked_fill(same_as_prev, 0), dim=-1).values
    return cumsum - group_start


def atom_ffn_hidden(width: int, expansion: int = 2) -> int:
    """Upstream ``SwiGLUFFN`` hidden: ``((expansion * (width // 3) * 2) + 255) // 256 * 256``."""
    return ((expansion * (width // 3) * 2) + 255) // 256 * 256


class AtomAttention(nn.Module):
    """Sliding-window attention with qk-RMSNorm, 3D RoPE, bf16 Q/K/V and a sigmoid output gate."""

    def __init__(
        self, width: int, heads: int, half_window: int, *, backend: AttentionBackend = "auto"
    ) -> None:
        super().__init__()
        self.heads, self.head_dim, self.half_window = heads, width // heads, half_window
        self.backend = backend
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.gate_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        valid: Tensor,
        block_mask: BlockMask | None = None,
    ) -> Tensor:
        B, A, _ = x.shape
        q = self.q_proj(x).view(B, A, self.heads, self.head_dim)
        k = self.k_proj(x).view(B, A, self.heads, self.head_dim)
        v = self.v_proj(x).view(B, A, self.heads, self.head_dim)
        q = F.rms_norm(q, (self.head_dim,)).to(q.dtype)
        k = F.rms_norm(k, (self.head_dim,)).to(k.dtype)
        q, k = apply_rotary_3d(q, cos, sin), apply_rotary_3d(k, cos, sin)
        in_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
        out = sliding_window_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            valid,
            self.half_window,
            backend=self.backend,
            block_mask=block_mask,
        )
        out = out.transpose(1, 2).reshape(B, A, -1).to(in_dtype)
        out = out * torch.sigmoid(self.gate_proj(x))
        return self.o_proj(out)


class AtomBlock(nn.Module):
    """adaLN (``rms_norm(x) * (1 + scale) + shift``) + raw-gated residual attention and MLP.

    ``adaln_linear(silu(c))`` chunks to ``shift_a, scale_a, gate_a, shift_f, scale_f, gate_f``;
    it is zero-initialised so a fresh block is the identity (spec §5.4).
    """

    def __init__(
        self,
        width: int,
        heads: int,
        half_window: int,
        *,
        expansion: int = 2,
        backend: AttentionBackend = "auto",
    ) -> None:
        super().__init__()
        self.adaln_linear = nn.Linear(width, 6 * width, bias=False)
        self.adaln_linear._init_zero = True  # ty: ignore[unresolved-attribute]  # read by _init_weights
        nn.init.zeros_(self.adaln_linear.weight)
        self.self_attn = AtomAttention(width, heads, half_window, backend=backend)
        self.mlp = GatedMLP(width, atom_ffn_hidden(width, expansion))

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        cos: Tensor,
        sin: Tensor,
        valid: Tensor,
        block_mask: BlockMask | None = None,
    ) -> Tensor:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.adaln_linear(F.silu(c)).chunk(
            6, dim=-1
        )
        h = F.rms_norm(x, (x.shape[-1],)) * (1 + scale_a) + shift_a
        x = x + gate_a * self.self_attn(h, cos, sin, valid, block_mask)
        h = F.rms_norm(x, (x.shape[-1],)) * (1 + scale_f) + shift_f
        return x + gate_f * self.mlp(h)


def _atom_blocks(config: FoldConfig, num_blocks: int, heads: int) -> nn.ModuleList:
    return nn.ModuleList(
        [
            AtomBlock(
                config.atom_width,
                heads,
                config.atom_window // 2,
                backend=config.attention_backend,
            )
            for _ in range(num_blocks)
        ]
    )


class AtomEncoder(nn.Module):
    """Atom features -> per-atom conditioning ``c`` -> windowed blocks -> relu -> token mean."""

    def __init__(self, config: FoldConfig, *, out_width: int, num_blocks: int, heads: int) -> None:
        super().__init__()
        self.atom_linear = nn.Linear(config.atom_feature_dim, config.atom_width, bias=False)
        self.atom_norm = nn.LayerNorm(config.atom_width, eps=config.layer_norm_eps)
        self.layers = _atom_blocks(config, num_blocks, heads)
        self.atom_to_token_linear = nn.Linear(config.atom_width, out_width, bias=False)

    def embed(self, features: Tensor) -> Tensor:
        """``c = atom_norm(atom_linear(features))``: the per-atom conditioning and initial ``q``."""
        return self.atom_norm(self.atom_linear(features))

    def forward(
        self,
        q: Tensor,
        c: Tensor,
        cos: Tensor,
        sin: Tensor,
        valid: Tensor,
        atom_to_token: Tensor,
        n_tokens: int,
        *,
        block_mask: BlockMask | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns ``(a (B, n_tokens, out_width), q (B, A, atom_width))``; ``q`` is the skip."""
        for layer in self.layers:
            q = layer(q, c, cos, sin, valid, block_mask)
        a = scatter_atom_to_token_mean(
            F.relu(self.atom_to_token_linear(q)), atom_to_token, n_tokens, valid
        )
        return a, q


class AtomDecoder(nn.Module):
    """Token features broadcast to atoms, windowed blocks, LayerNorm, 3-D output projection."""

    def __init__(self, config: FoldConfig, *, num_blocks: int, heads: int) -> None:
        super().__init__()
        self.token_to_atom_linear = nn.Linear(config.token_width, config.atom_width, bias=False)
        self.layers = _atom_blocks(config, num_blocks, heads)
        self.norm = nn.LayerNorm(config.atom_width, eps=config.layer_norm_eps)
        self.output_linear = nn.Linear(config.atom_width, 3, bias=False)

    def forward(
        self,
        a: Tensor,
        q: Tensor,
        c: Tensor,
        cos: Tensor,
        sin: Tensor,
        valid: Tensor,
        atom_to_token: Tensor,
        *,
        block_mask: BlockMask | None = None,
    ) -> Tensor:
        q = q + gather_token_to_atom(self.token_to_atom_linear(a), atom_to_token)
        for layer in self.layers:
            q = layer(q, c, cos, sin, valid, block_mask)
        return self.output_linear(self.norm(q))


@dataclass
class InputsEmbedding:
    """What the inputs embedder hands the trunk and the heads."""

    s_inputs: Tensor  # (B, L, 451)
    z_init: Tensor  # (B, L, L, pair)
    relpos: Tensor  # (B, L, L, pair) rel_pos embedding (reused by the structure/confidence heads)
    bonds: Tensor  # (B, L, L, pair) token-bond embedding (reused by the confidence head)
    atom_features: Tensor  # (B, A, 389) (reused by the diffusion atom encoder)
    rope: tuple[Tensor, Tensor]  # bf16 cos/sin (reused by the diffusion atom encoder/decoder)


class InputsEmbedder(nn.Module):
    """Checkpoint ``input_embedder``: atom encoder, single-input assembly and the initial pair."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__()
        self.config = config
        self.atom_encoder = AtomEncoder(
            config,
            out_width=config.inputs_token_width,
            num_blocks=config.atom_encoder_blocks,
            heads=config.atom_encoder_heads,
        )
        self.pair_init_1 = nn.Linear(config.single_inputs_width, config.pair_width, bias=False)
        self.pair_init_2 = nn.Linear(config.single_inputs_width, config.pair_width, bias=False)
        self.rel_pos = RelativePositionEncoding(
            config.pair_width, r_max=config.relpos_r_max, s_max=config.relpos_s_max
        )
        self.token_bonds = nn.Linear(1, config.pair_width, bias=False)

    def rope(self, features: FoldFeatures) -> tuple[Tensor, Tensor]:
        cfg = self.config
        return build_3d_rope(
            features.ref_pos,
            features.ref_space_uid,
            head_dim=cfg.atom_width // cfg.atom_encoder_heads,
            spatial_pairs_per_axis=cfg.spatial_rope_pairs_per_axis,
            uid_pairs=cfg.uid_rope_pairs,
            spatial_base=cfg.spatial_rope_base,
            uid_base=cfg.uid_rope_base,
        )

    def forward(self, f: FoldFeatures, *, block_mask: BlockMask | None = None) -> InputsEmbedding:
        cfg = self.config
        feats = build_atom_features(
            f.ref_pos,
            f.ref_charge,
            f.atom_mask,
            f.ref_element,
            f.ref_atom_name_chars,
            max_atomic_number=cfg.max_atomic_number,
            name_vocab=cfg.atom_name_vocab,
        )
        rope = self.rope(f)
        c = self.atom_encoder.embed(feats)
        a, _ = self.atom_encoder(
            c, c, *rope, f.atom_mask, f.atom_to_token, f.num_tokens, block_mask=block_mask
        )
        res_oh = F.one_hot(f.res_type, cfg.num_res_types).float() * f.token_mask[..., None].float()
        # single-sequence mode: profile = the query one-hot, deletion_mean = 0 (upstream S2)
        s_inputs = torch.cat([a.float(), res_oh, res_oh, torch.zeros_like(res_oh[..., :1])], dim=-1)
        relpos = self.rel_pos(f.residue_index, f.asym_id, f.sym_id, f.entity_id, f.token_index)
        bonds = self.token_bonds(f.token_bonds.to(self.token_bonds.weight.dtype))
        z_init = (
            self.pair_init_1(s_inputs)[:, :, None, :] + self.pair_init_2(s_inputs)[:, None, :, :]
        )
        z_init = z_init + relpos + bonds
        return InputsEmbedding(s_inputs, z_init, relpos, bonds, feats, rope)

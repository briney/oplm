"""Diffusion structure head: conditioning, adaptive-norm pair-bias blocks, EDM denoiser, sampler.

Ported from Biohub's ESMFold2 ``FourierEmbedding``, ``AdaptiveLayerNorm``, ``AttentionPairBias``,
``ConditionedTransitionBlock``, ``DiffusionTransformer``, ``DiffusionConditioning``,
``DiffusionModule`` and ``DiffusionStructureHead`` (esm/models/esmfold2/layers.py, Apache-2.0;
see THIRD_PARTY_NOTICES.md). Names follow the released checkpoint: one
``token_transformer.layers.N`` holds the attention and transition halves of a block
(``input_layernorm``/``post_attention_layernorm`` adaptive norms, ``attn_gate``/``mlp_gate``
output gates, ``self_attn.{q,k,v,gate,o}_proj``, ``pair_norm``, ``pair_bias_proj``, ``mlp``).
Modifications: attention is milestone 0's ``pair_biased_attention`` (FlexAttention / dense
oracle) with a precomputed block mask, and padded query rows are zeroed; the denoiser takes the
conditioned pair and the atom conditioning as arguments so the sampler computes them once; the
sampler takes a ``torch.Generator``; the sigma cap truncates the schedule exactly as upstream.
Precision (spec §5.4): coordinates, preconditioning, alignment and the sampler are fp32; the
pair-conditioning transitions run under bf16 autocast on CUDA as upstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from oplm.fold.atoms import AtomDecoder, AtomEncoder
from oplm.fold.attention import (
    pair_bias_block_mask,
    pair_biased_attention,
    resolve_attention_backend,
    sliding_window_block_mask,
)
from oplm.fold.trunk import GatedMLP, Transition, cuda_bf16_autocast

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn.attention.flex_attention import BlockMask

    from oplm.fold.attention import AttentionBackend
    from oplm.fold.configuration_fold import FoldConfig

__all__ = [
    "AdaptiveLayerNorm",
    "DenoiserInputs",
    "DiffusionBlock",
    "DiffusionConditioning",
    "DiffusionTransformer",
    "FourierEmbedding",
    "PairBiasAttention",
    "StructureHead",
    "center_random_augmentation",
    "random_rotations",
    "weighted_rigid_align",
]


class FourierEmbedding(nn.Module):
    """``cos(2π (t · w + b))``; ``frequencies``/``phases`` are random persistent buffers (ckpt)."""

    frequencies: Tensor
    phases: Tensor

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.randn(dim))
        self.register_buffer("phases", torch.randn(dim))

    def forward(self, t: Tensor) -> Tensor:
        return torch.cos(2 * math.pi * (t[:, None] * self.frequencies + self.phases))


class AdaptiveLayerNorm(nn.Module):
    """``sigmoid(gate(LN_w(s))) · LN(a) + shift(LN_w(s))``; children are checkpoint-named."""

    def __init__(self, width: int, cond_width: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.cond_norm = nn.LayerNorm(cond_width, eps=eps, bias=False)
        self.gate_proj = nn.Linear(cond_width, width, bias=True)
        self.shift_proj = nn.Linear(cond_width, width, bias=False)

    def forward(self, a: Tensor, s: Tensor) -> Tensor:
        a_norm = F.layer_norm(a, (a.shape[-1],), eps=self.eps)
        s_norm = self.cond_norm(s)
        return torch.sigmoid(self.gate_proj(s_norm)) * a_norm + self.shift_proj(s_norm)


def _output_gate(width: int) -> nn.Linear:
    """Zero-weight, −2-bias gate (spec §5.4); tags keep ``_init_weights`` from overwriting it."""
    gate = nn.Linear(width, width, bias=True)
    gate._init_zero = True  # ty: ignore[unresolved-attribute]  # read by _init_weights
    gate._init_bias = -2.0  # ty: ignore[unresolved-attribute]  # read by _init_weights
    nn.init.zeros_(gate.weight)
    nn.init.constant_(gate.bias, -2.0)
    return gate


class PairBiasAttention(nn.Module):
    """Multi-head attention with an additive per-head pair bias and a sigmoid value gate."""

    def __init__(self, width: int, heads: int, *, backend: AttentionBackend = "auto") -> None:
        super().__init__()
        self.heads, self.head_dim, self.backend = heads, width // heads, backend
        self.q_proj = nn.Linear(width, width, bias=True)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.gate_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(
        self, x: Tensor, bias: Tensor, key_mask: Tensor, block_mask: BlockMask | None = None
    ) -> Tensor:
        B, L, _ = x.shape

        def heads(t: Tensor) -> Tensor:
            return t.view(B, L, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
        g = torch.sigmoid(heads(self.gate_proj(x)))
        ctx = pair_biased_attention(
            q, k, v, bias, key_mask, backend=self.backend, block_mask=block_mask
        )
        return self.o_proj((g * ctx).transpose(1, 2).reshape(B, L, -1))


class DiffusionBlock(nn.Module):
    """One ``token_transformer.layers.N``: gated pair-bias attention then a gated transition."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__()
        w, p, eps = config.token_width, config.pair_width, config.layer_norm_eps
        self.input_layernorm = AdaptiveLayerNorm(w, w, eps=eps)
        self.self_attn = PairBiasAttention(
            w, config.diffusion_heads, backend=config.attention_backend
        )
        self.pair_norm = nn.LayerNorm(p, eps=eps)
        self.pair_bias_proj = nn.Linear(p, config.diffusion_heads, bias=False)
        self.attn_gate = _output_gate(w)
        self.post_attention_layernorm = AdaptiveLayerNorm(w, w, eps=eps)
        self.mlp = GatedMLP(w, config.diffusion_transition_multiplier * w)
        self.mlp_gate = _output_gate(w)

    def forward(
        self,
        a: Tensor,
        s: Tensor,
        z: Tensor,
        token_mask: Tensor,
        block_mask: BlockMask | None = None,
    ) -> Tensor:
        bias = self.pair_bias_proj(self.pair_norm(z)).permute(0, 3, 1, 2)  # (B, H, L, L)
        if bias.shape[0] != a.shape[0]:
            bias = bias.repeat_interleave(a.shape[0] // bias.shape[0], dim=0)
        x = self.input_layernorm(a, s)
        attn = self.self_attn(x, bias, token_mask, block_mask) * token_mask[..., None].to(a.dtype)
        a = a + torch.sigmoid(self.attn_gate(s)) * attn
        x = self.post_attention_layernorm(a, s)
        return a + torch.sigmoid(self.mlp_gate(s)) * self.mlp(x)


class DiffusionTransformer(nn.Module):
    """Checkpoint ``token_transformer``."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DiffusionBlock(config) for _ in range(config.diffusion_blocks)]
        )

    def forward(
        self,
        a: Tensor,
        s: Tensor,
        z: Tensor,
        token_mask: Tensor,
        block_mask: BlockMask | None = None,
    ) -> Tensor:
        for layer in self.layers:
            a = layer(a, s, z, token_mask, block_mask)
        return a


class DiffusionConditioning(nn.Module):
    """Checkpoint ``conditioning``: pair conditioning (once per call) and noise-aware single."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__()
        p, w, s_in = config.pair_width, config.token_width, config.single_inputs_width
        mult, eps = config.diffusion_transition_multiplier, config.layer_norm_eps
        self.sigma_data = config.sigma_data
        self.fourier = FourierEmbedding(config.fourier_dim)
        self.pair_input_norm = nn.LayerNorm(2 * p, eps=eps)
        self.pair_proj = nn.Linear(2 * p, p, bias=False)
        self.pair_transition_0 = Transition(p, mult, eps=eps)
        self.pair_transition_1 = Transition(p, mult, eps=eps)
        self.single_input_norm = nn.LayerNorm(s_in, eps=eps)
        self.single_proj = nn.Linear(s_in, w, bias=False)
        self.single_transition_0 = Transition(w, mult, eps=eps)
        self.single_transition_1 = Transition(w, mult, eps=eps)
        self.noise_norm = nn.LayerNorm(config.fourier_dim, eps=eps)
        self.noise_proj = nn.Linear(config.fourier_dim, w, bias=False)

    def pair(self, z_trunk: Tensor, relpos: Tensor) -> Tensor:
        """``cat[z_trunk, relpos] -> LN -> proj -> two residual transitions`` (fp32 out)."""
        z = self.pair_proj(
            self.pair_input_norm(torch.cat([z_trunk.float(), relpos.float()], dim=-1))
        )
        with cuda_bf16_autocast(z.is_cuda):
            z = z + self.pair_transition_0(z)
            z = z + self.pair_transition_1(z)
        return z.float()

    def single(self, s_inputs: Tensor, t_hat: Tensor) -> Tensor:
        """``proj(LN(s_inputs)) + noise_proj(LN(fourier(0.25 log(t/σ_d))))``, two transitions."""
        s = self.single_proj(self.single_input_norm(s_inputs.float()))
        c_noise = 0.25 * torch.log((t_hat.float() / self.sigma_data).clamp(min=1e-20))
        n = self.noise_proj(self.noise_norm(self.fourier(c_noise)))
        s = s + n[:, None, :]
        s = s + self.single_transition_0(s)
        return s + self.single_transition_1(s)


@dataclass
class DenoiserInputs:
    """Everything that is constant across the denoising steps of one sampling call.

    All tensors except ``z`` are already expanded to ``B · num_samples`` rows.
    """

    s_inputs: Tensor
    z: Tensor  # (B, L, L, pair) conditioned pair, base batch
    c: Tensor  # (B·S, A, atom) atom conditioning
    rope: tuple[Tensor, Tensor]
    atom_mask: Tensor  # (B·S, A) bool
    atom_to_token: Tensor
    token_mask: Tensor  # (B·S, L) bool
    token_block_mask: BlockMask | None = None
    atom_block_mask: BlockMask | None = None


def random_rotations(
    n: int, *, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None
) -> Tensor:
    """``(n, 3, 3)`` rotations from sign-normalised random quaternions (upstream)."""
    q = torch.randn((n, 4), dtype=dtype, device=device, generator=generator)
    scale = torch.sqrt((q * q).sum(dim=1))
    signs = torch.where(q[:, 0] < 0, -scale, scale)
    q = q / signs[:, None]
    r, i, j, k = torch.unbind(q, dim=-1)
    two_s = 2.0 / (q * q).sum(dim=-1)
    return torch.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        dim=-1,
    ).reshape(n, 3, 3)


def center_random_augmentation(
    x: Tensor, atom_mask: Tensor, *, generator: torch.Generator | None = None
) -> Tensor:
    """Masked centering, a random rotation per sample, then a ``N(0, I)`` translation (Å)."""
    mask = atom_mask[..., None].to(x.dtype)
    x = x - (x * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1)
    rot = random_rotations(x.shape[0], device=x.device, dtype=x.dtype, generator=generator)
    x = torch.einsum("bmd,bds->bms", x, rot)
    return x + torch.randn((x.shape[0], 1, 3), dtype=x.dtype, device=x.device, generator=generator)


def weighted_rigid_align(x: Tensor, x_gt: Tensor, weights: Tensor) -> Tensor:
    """Weighted Kabsch in fp32: ``x`` superposed onto ``x_gt`` (upstream)."""
    w = weights[..., None].float()
    denom = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
    mu = (x.float() * w).sum(dim=1, keepdim=True) / denom
    mu_gt = (x_gt.float() * w).sum(dim=1, keepdim=True) / denom
    x_c, gt_c = x.float() - mu, x_gt.float() - mu_gt
    h = torch.einsum("bni,bnj->bij", w * gt_c, x_c)
    u, _, vh = torch.linalg.svd(h, driver="gesvd" if h.is_cuda else None)
    det = torch.linalg.det(u @ vh)
    d = torch.ones(h.shape[0], 3, device=h.device, dtype=h.dtype)
    d[:, 2] = det
    rot = u @ torch.diag_embed(d) @ vh
    return x_c @ rot.transpose(-1, -2) + mu_gt


class StructureHead(nn.Module):
    """Checkpoint ``structure_head``: the EDM denoiser and the Karras sampler."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__()
        self.config = config
        self.sigma_data = config.sigma_data
        eps = config.layer_norm_eps
        self.conditioning = DiffusionConditioning(config)
        self.coords_linear = nn.Linear(6, config.atom_width, bias=False)
        self.atom_encoder = AtomEncoder(
            config,
            out_width=config.token_width,
            num_blocks=config.diffusion_atom_blocks,
            heads=config.diffusion_atom_heads,
        )
        self.single_to_token = nn.Linear(config.token_width, config.token_width, bias=False)
        self.single_to_token._init_zero = True  # ty: ignore[unresolved-attribute]  # read by _init_weights
        nn.init.zeros_(self.single_to_token.weight)
        self.single_step_norm = nn.LayerNorm(config.token_width, eps=eps)
        self.token_transformer = DiffusionTransformer(config)
        self.token_norm = nn.LayerNorm(config.token_width, eps=eps)
        self.atom_decoder = AtomDecoder(
            config, num_blocks=config.diffusion_atom_blocks, heads=config.diffusion_atom_heads
        )

    def prepare(
        self,
        *,
        s_inputs: Tensor,
        z_trunk: Tensor,
        relpos: Tensor,
        atom_features: Tensor,
        rope: tuple[Tensor, Tensor],
        atom_mask: Tensor,
        atom_to_token: Tensor,
        token_mask: Tensor,
        num_samples: int,
    ) -> DenoiserInputs:
        """Condition the pair, embed the atoms and expand the masks once for ``num_samples``."""
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1; got {num_samples!r}")

        def rep(t: Tensor) -> Tensor:
            return t.repeat_interleave(num_samples, dim=0)

        z = self.conditioning.pair(z_trunk, relpos)
        c = self.atom_encoder.embed(atom_features)
        atom_mask_s, token_mask_s = rep(atom_mask), rep(token_mask)
        flex = resolve_attention_backend(z, self.config.attention_backend) == "flex"
        return DenoiserInputs(
            s_inputs=rep(s_inputs),
            z=z,
            c=rep(c),
            rope=(rep(rope[0]), rep(rope[1])),
            atom_mask=atom_mask_s,
            atom_to_token=rep(atom_to_token),
            token_mask=token_mask_s,
            token_block_mask=pair_bias_block_mask(token_mask_s) if flex else None,
            atom_block_mask=(
                sliding_window_block_mask(atom_mask_s, self.config.atom_window // 2)
                if flex
                else None
            ),
        )

    def denoise(self, x_noisy: Tensor, t_hat: Tensor, inp: DenoiserInputs) -> Tensor:
        """EDM-preconditioned denoiser: ``c_skip · x + c_out · F(c_in · x, c_noise)`` (fp32)."""
        sigma = self.sigma_data
        t = t_hat.float().reshape(-1)
        if t.numel() == 1:
            t = t.expand(x_noisy.shape[0])
        x_noisy = x_noisy.float()
        s = self.conditioning.single(inp.s_inputs, t)
        r_noisy = x_noisy / torch.sqrt(t * t + sigma * sigma)[:, None, None]
        coords = torch.cat(
            [r_noisy, torch.zeros_like(r_noisy)], dim=-1
        )  # upstream: pred_r1 is always 0
        q = inp.c + self.coords_linear(coords.to(inp.c.dtype))
        n_tokens = inp.token_mask.shape[1]
        a, q_skip = self.atom_encoder(
            q,
            inp.c,
            *inp.rope,
            inp.atom_mask,
            inp.atom_to_token,
            n_tokens,
            block_mask=inp.atom_block_mask,
        )
        a = a + self.single_to_token(self.single_step_norm(s))
        a = self.token_transformer(a, s, inp.z, inp.token_mask, inp.token_block_mask)
        a = self.token_norm(a)
        r_update = self.atom_decoder(
            a,
            q_skip,
            inp.c,
            *inp.rope,
            inp.atom_mask,
            inp.atom_to_token,
            block_mask=inp.atom_block_mask,
        )
        c_skip = (sigma * sigma / (sigma * sigma + t * t))[:, None, None]
        c_out = (sigma * t / torch.sqrt(sigma * sigma + t * t))[:, None, None]
        return c_skip * x_noisy + c_out * r_update.float()

    def noise_schedule(self, num_steps: int, device: torch.device) -> Tensor:
        """Karras ``σ_d · (s_max^(1/ρ) + k/(n-1) (s_min^(1/ρ) − s_max^(1/ρ)))^ρ``, trailing 0."""
        cfg = self.config
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1; got {num_steps!r}")
        if num_steps == 1:
            return torch.tensor([cfg.inference_sigma_max * self.sigma_data, 0.0], device=device)
        rho = cfg.inference_rho
        inv = 1.0 / rho
        k = torch.arange(num_steps, device=device, dtype=torch.float32)
        base = cfg.inference_sigma_max**inv + (k / (num_steps - 1)) * (
            cfg.inference_sigma_min**inv - cfg.inference_sigma_max**inv
        )
        return F.pad(self.sigma_data * base.pow(rho), (0, 1), value=0.0)

    def sample(
        self,
        inp: DenoiserInputs,
        *,
        num_steps: int | None = None,
        sigma_cap: float | None = None,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Upstream ``DiffusionStructureHead.sample``: churned Euler steps with per-step alignment.

        RNG draw order per upstream: initial noise; then per step a rotation quaternion, a
        translation and the churn noise (drawn even when ``noise_scale == 0``). The sigma cap
        truncates the schedule (``schedule[schedule <= cap]`` prefixed by ``cap``) without
        re-inflating the step count, and the churn gamma is keyed on the *next* sigma.
        """
        cfg = self.config
        steps = cfg.inference_num_steps if num_steps is None else num_steps
        cap = cfg.inference_sigma_cap if sigma_cap is None else sigma_cap
        lam = cfg.noise_scale if noise_scale is None else noise_scale
        eta = cfg.step_scale if step_scale is None else step_scale
        device = inp.z.device
        schedule = self.noise_schedule(steps, device)
        schedule = F.pad(schedule[schedule <= cap], (1, 0), value=cap)
        sigmas = schedule.tolist()
        gammas = [cfg.gamma_0 if s > cfg.gamma_min else 0.0 for s in sigmas]
        n, n_atoms = inp.atom_mask.shape
        atom_mask = inp.atom_mask.float()
        x = sigmas[0] * torch.randn(n, n_atoms, 3, device=device, generator=generator)
        for sigma_tm, sigma_t, gamma in zip(sigmas[:-1], sigmas[1:], gammas[1:], strict=True):
            x = center_random_augmentation(x, atom_mask, generator=generator)
            t_hat = sigma_tm * (1.0 + gamma)
            eps_std = lam * max(t_hat**2 - sigma_tm**2, 0.0) ** 0.5
            x_noisy = x + eps_std * torch.randn(x.shape, device=device, generator=generator)
            x_denoised = self.denoise(x_noisy, torch.full((n,), t_hat, device=device), inp)
            x_noisy = weighted_rigid_align(x_noisy, x_denoised, atom_mask)
            x = x_noisy + eta * (sigma_t - t_hat) * (x_noisy - x_denoised) / t_hat
        return x

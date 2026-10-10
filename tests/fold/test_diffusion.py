"""Diffusion head: adaLN, pair-bias blocks, conditioning, EDM denoiser, schedule and sampler."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from oplm.fold.atoms import InputsEmbedder
from oplm.fold.data.featurize import ChainSpec, featurize
from oplm.fold.diffusion import (
    AdaptiveLayerNorm,
    DiffusionBlock,
    DiffusionConditioning,
    PairBiasAttention,
    StructureHead,
    center_random_augmentation,
    random_rotations,
    weighted_rigid_align,
)
from tests.fold.helpers import tiny_fold_config

if TYPE_CHECKING:
    from oplm.fold.configuration_fold import FoldConfig


def test_adaptive_layer_norm_matches_transcription_and_names() -> None:
    torch.manual_seed(0)
    ada = AdaptiveLayerNorm(8, 6, eps=1e-5)
    assert set(ada.state_dict()) == {
        "cond_norm.weight",
        "gate_proj.weight",
        "gate_proj.bias",
        "shift_proj.weight",
    }
    a, s = torch.randn(2, 5, 8), torch.randn(2, 5, 6)
    a_norm = F.layer_norm(a, (8,), None, None, 1e-5)
    s_norm = F.layer_norm(s, (6,), ada.cond_norm.weight, None, 1e-5)
    expected = torch.sigmoid(ada.gate_proj(s_norm)) * a_norm + ada.shift_proj(s_norm)
    torch.testing.assert_close(ada(a, s), expected)


def test_pair_bias_attention_matches_upstream_transcription() -> None:
    torch.manual_seed(0)
    attn = PairBiasAttention(16, 2, backend="dense")
    x = torch.randn(2, 7, 16)
    bias = torch.randn(2, 2, 7, 7)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 5:] = False
    out = attn(x, bias, mask)
    q = attn.q_proj(x).view(2, 7, 2, 8)
    k = attn.k_proj(x).view(2, 7, 2, 8)
    v = attn.v_proj(x).view(2, 7, 2, 8)
    g = torch.sigmoid(attn.gate_proj(x)).view(2, 7, 2, 8)
    logits = torch.einsum("...ihd,...jhd->...ijh", q, k) * 8**-0.5 + bias.permute(0, 2, 3, 1)
    logits = logits + torch.where(mask[:, None, :, None], 0.0, torch.finfo(logits.dtype).min)
    ctx = torch.einsum("...ijh,...jhd->...ihd", logits.softmax(dim=-2), v)
    expected = attn.o_proj((g * ctx).reshape(2, 7, 16))
    torch.testing.assert_close(out, expected)
    assert attn.q_proj.bias is not None and attn.k_proj.bias is None


def test_diffusion_block_names_gate_init_and_padded_rows() -> None:
    cfg = tiny_fold_config()
    torch.manual_seed(0)
    block = DiffusionBlock(cfg)
    names = set(block.state_dict())
    assert names == {
        "input_layernorm.cond_norm.weight",
        "input_layernorm.gate_proj.weight",
        "input_layernorm.gate_proj.bias",
        "input_layernorm.shift_proj.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.gate_proj.weight",
        "self_attn.o_proj.weight",
        "pair_norm.weight",
        "pair_norm.bias",
        "pair_bias_proj.weight",
        "attn_gate.weight",
        "attn_gate.bias",
        "post_attention_layernorm.cond_norm.weight",
        "post_attention_layernorm.gate_proj.weight",
        "post_attention_layernorm.gate_proj.bias",
        "post_attention_layernorm.shift_proj.weight",
        "mlp.gate_up_proj.weight",
        "mlp.down_proj.weight",
        "mlp_gate.weight",
        "mlp_gate.bias",
    }
    for gate in (block.attn_gate, block.mlp_gate):
        assert gate.weight.abs().sum() == 0 and torch.equal(gate.bias, torch.full((64,), -2.0))
    assert block.pair_bias_proj.weight.shape == (4, 32)
    assert block.mlp.gate_up_proj.weight.shape == (256, 64)
    a, s = torch.randn(2, 6, 64), torch.randn(2, 6, 64)
    z = torch.randn(1, 6, 6, 32)  # base batch 1, two samples
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1] = False  # Review Focus 2: a fully padded batch row
    out = block(a, s, z, mask)
    assert out.shape == a.shape and torch.isfinite(out).all()
    # the fully padded row gets no attention output (zeroed), so it is a pure function of a and s
    residual = a + torch.sigmoid(block.attn_gate(s)) * 0.0
    x2 = block.post_attention_layernorm(residual, s)
    expected_row1 = residual[1] + torch.sigmoid(block.mlp_gate(s))[1] * block.mlp(x2)[1]
    torch.testing.assert_close(out[1], expected_row1)


def test_conditioning_names_and_noise_embedding() -> None:
    cfg = tiny_fold_config()
    torch.manual_seed(0)
    cond = DiffusionConditioning(cfg)
    names = set(cond.state_dict())
    assert {
        "fourier.frequencies",
        "fourier.phases",
        "pair_input_norm.weight",
        "pair_proj.weight",
        "pair_transition_0.mlp.gate_up_proj.weight",
        "pair_transition_1.norm.bias",
        "single_input_norm.weight",
        "single_proj.weight",
        "single_transition_0.mlp.down_proj.weight",
        "noise_norm.weight",
        "noise_proj.weight",
    } <= names
    assert cond.pair_proj.weight.shape == (
        32,
        64,
    ) and cond.pair_transition_0.mlp.gate_up_proj.weight.shape == (128, 32)
    assert cond.single_proj.weight.shape == (
        64,
        cfg.single_inputs_width,
    ) and cond.noise_proj.weight.shape == (64, 16)
    t = torch.tensor([16.0, 4.0])
    emb = cond.fourier(0.25 * torch.log(t / 16.0))
    expected = torch.cos(
        2
        * math.pi
        * ((0.25 * torch.log(t / 16.0))[:, None] * cond.fourier.frequencies + cond.fourier.phases)
    )
    torch.testing.assert_close(emb, expected)
    s_inputs = torch.randn(1, 5, cfg.single_inputs_width)
    s = cond.single(s_inputs.repeat_interleave(2, 0), t)
    assert s.shape == (2, 5, 64) and not torch.equal(s[0], s[1])  # different noise levels differ
    z = cond.pair(torch.randn(1, 5, 5, 32), torch.randn(1, 5, 5, 32))
    assert z.shape == (1, 5, 5, 32) and z.dtype == torch.float32


def _head_and_inputs(cfg: FoldConfig, num_samples: int):
    torch.manual_seed(0)
    emb = InputsEmbedder(cfg).eval()
    head = StructureHead(cfg).eval()
    f = featurize([ChainSpec("MKV", "A"), ChainSpec("GG", "B")], pad_tokens_to=8)
    with torch.no_grad():
        e = emb(f)
        z_trunk = torch.randn(1, 8, 8, cfg.pair_width)
        inp = head.prepare(
            s_inputs=e.s_inputs,
            z_trunk=z_trunk,
            relpos=e.relpos,
            atom_features=e.atom_features,
            rope=e.rope,
            atom_mask=f.atom_mask,
            atom_to_token=f.atom_to_token,
            token_mask=f.token_mask,
            num_samples=num_samples,
        )
    return head, inp, f


def test_denoiser_shapes_edm_limits_and_zero_init_names() -> None:
    cfg = tiny_fold_config()
    head, inp, f = _head_and_inputs(cfg, num_samples=2)
    assert head.single_to_token.weight.abs().sum() == 0 and head.coords_linear.weight.shape == (
        32,
        6,
    )
    assert {
        "conditioning.fourier.frequencies",
        "coords_linear.weight",
        "single_to_token.weight",
        "single_step_norm.weight",
        "token_norm.bias",
        "atom_encoder.atom_to_token_linear.weight",
        "atom_decoder.output_linear.weight",
        "token_transformer.layers.0.attn_gate.bias",
    } <= set(head.state_dict())
    assert head.atom_encoder.atom_to_token_linear.weight.shape == (64, 32)
    x = torch.randn(2, f.num_atoms, 3)
    with torch.no_grad():
        small = head.denoise(x, torch.full((2,), 1e-6), inp)
        big = head.denoise(x, torch.full((2,), 1e6), inp)
    assert small.shape == (2, f.num_atoms, 3)
    torch.testing.assert_close(small, x, atol=1e-3, rtol=0)  # c_skip -> 1, c_out -> 0 as t -> 0
    assert (
        torch.isfinite(big).all() and (big - x).abs().max() > 1e-3
    )  # c_skip -> 0: pure network output


def test_noise_schedule_matches_transcription_and_cap_truncates() -> None:
    cfg = tiny_fold_config(inference_num_steps=14)  # released sampler constants otherwise
    head = StructureHead(cfg)
    sched = head.noise_schedule(14, torch.device("cpu"))
    k = torch.arange(14, dtype=torch.float32)
    base = 160.0 ** (1 / 7) + (k / 13) * (4e-4 ** (1 / 7) - 160.0 ** (1 / 7))
    expected = F.pad(16.0 * base.pow(7.0), (0, 1), value=0.0)
    torch.testing.assert_close(sched, expected)
    assert sched[0] == 2560.0 and sched[-1] == 0.0
    capped = sched[sched <= 256.0]
    capped = F.pad(capped, (1, 0), value=256.0)
    assert capped.shape == (11,) and capped[0] == 256.0  # 10 steps actually run
    torch.testing.assert_close(capped[1], torch.tensor(165.6605), atol=1e-3, rtol=0)
    assert head.noise_schedule(1, torch.device("cpu")).tolist() == [160.0 * 16.0, 0.0]


def test_augmentation_and_kabsch_alignment() -> None:
    g = torch.Generator().manual_seed(0)
    rot = random_rotations(4, device=torch.device("cpu"), dtype=torch.float32, generator=g)
    torch.testing.assert_close(torch.linalg.det(rot), torch.ones(4))
    torch.testing.assert_close(
        rot @ rot.transpose(-1, -2), torch.eye(3).expand(4, 3, 3), atol=1e-5, rtol=0
    )
    gt = torch.randn(2, 10, 3, generator=g)
    mask = torch.ones(2, 10)
    mask[1, 8:] = 0.0
    moved = center_random_augmentation(gt, mask, generator=g)
    assert moved.shape == gt.shape
    aligned = weighted_rigid_align(moved, gt, mask)
    torch.testing.assert_close(aligned[0], gt[0], atol=1e-4, rtol=0)
    torch.testing.assert_close(aligned[1, :8], gt[1, :8], atol=1e-4, rtol=0)  # pads carry no weight


def test_sample_is_deterministic_under_a_generator() -> None:
    cfg = tiny_fold_config()
    head, inp, f = _head_and_inputs(cfg, num_samples=2)
    with torch.no_grad():
        a = head.sample(inp, generator=torch.Generator().manual_seed(7))
        b = head.sample(inp, generator=torch.Generator().manual_seed(7))
        c = head.sample(inp, generator=torch.Generator().manual_seed(8))
    assert a.shape == (2, f.num_atoms, 3) and torch.isfinite(a).all()
    assert torch.equal(a, b) and not torch.equal(a, c)

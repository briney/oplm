"""Pair-update blocks, the pair stack and the recurrence against transcribed upstream math."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.trunk import (
    GatedMLP,
    PairStack,
    PairUpdateBlock,
    Recurrence,
    Transition,
    pair_stack_kwargs,
)

_W = 32  # pair width (multiple of 32 for the trimul contract)


def _pair(b: int = 1, n: int = 6, w: int = _W, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, n, w, generator=g)


def test_gated_mlp_matches_upstream_swiglu_order() -> None:
    torch.manual_seed(0)
    mlp = GatedMLP(_W, 4 * _W)
    x = torch.randn(2, 5, _W)
    x1, x2 = mlp.gate_up_proj(x).chunk(
        2, dim=-1
    )  # upstream SwiGLUMLP: silu(FIRST half) * SECOND half
    torch.testing.assert_close(mlp(x), mlp.down_proj(F.silu(x1) * x2))
    assert mlp.gate_up_proj.bias is None and mlp.down_proj.bias is None


def test_transition_returns_delta_with_checkpoint_names() -> None:
    t = Transition(_W, 4)
    assert set(t.state_dict()) == {
        "norm.weight",
        "norm.bias",
        "mlp.gate_up_proj.weight",
        "mlp.down_proj.weight",
    }
    assert t.mlp.gate_up_proj.weight.shape == (8 * _W, _W)
    x = torch.randn(1, 3, 3, _W)
    torch.testing.assert_close(t(x), t.mlp(t.norm(x)))


def test_pair_update_block_is_the_upstream_residual_composition() -> None:
    torch.manual_seed(0)
    block = PairUpdateBlock(
        _W, expansion=4, dropout=0.25, eps=1e-5, chunk_size=64, trimul_backend="reference"
    )
    block.eval()  # dropout is a no-op in eval (and at p=0 in train)
    z = _pair()
    mask = torch.ones(1, 6, 6)
    mask[0, 4:, :] = mask[0, :, 4:] = 0.0
    expected = z + block.tri_mul_out(z, mask)
    expected = expected + block.tri_mul_in(expected, mask)
    expected = expected + block.pair_transition(expected)
    torch.testing.assert_close(block(z, mask), expected)
    names = set(block.state_dict())
    assert (
        "tri_mul_out.proj_bundle.weight" in names
        and "pair_transition.mlp.down_proj.weight" in names
    )
    assert not any("dropout" in n for n in names)


def test_row_shared_dropout_shares_the_mask_along_rows_and_rescales() -> None:
    torch.manual_seed(0)
    block = PairUpdateBlock(_W, dropout=0.5).train()
    delta = torch.ones(2, 7, 5, _W)
    out = block.dropout(delta)
    assert set(out.unique().tolist()) <= {0.0, 2.0}
    assert torch.equal(out[:, :1].expand_as(out), out)  # identical across the row axis


def test_pair_stack_checkpointing_matches_plain_gradients() -> None:
    torch.manual_seed(0)
    stack = PairStack(2, _W, trimul_backend="reference").train()
    z = _pair().requires_grad_(True)
    out = stack(z)
    (g_plain,) = torch.autograd.grad(out.square().sum(), z)
    stack.gradient_checkpointing = True
    (g_ckpt,) = torch.autograd.grad(stack(z).square().sum(), z)
    torch.testing.assert_close(g_ckpt, g_plain)
    assert [n for n, _ in stack.named_children()] == ["layers"] and len(stack.layers) == 2


def test_recurrence_dynamics_and_initial_values_match_upstream() -> None:
    rec = Recurrence(_W, coda_blocks=1, trimul_backend="reference")
    a, b = rec.dynamics()
    delta = F.softplus(rec.log_delta)
    torch.testing.assert_close(a, torch.exp(-delta * torch.exp(rec.log_state_decay)))
    torch.testing.assert_close(b, delta[:, None] * rec.input_matrix_continuous)
    # init: delta0 = 0.5 ln 5 -> a = sqrt(1/5); B = delta0 * I; out_proj = I
    torch.testing.assert_close(a, torch.full((_W,), math.sqrt(0.2)))
    torch.testing.assert_close(b, 0.5 * math.log(5.0) * torch.eye(_W))
    torch.testing.assert_close(rec.out_proj.weight, torch.eye(_W))
    assert set(rec.state_dict()) >= {
        "input_norm.weight",
        "log_delta",
        "log_state_decay",
        "input_matrix_continuous",
        "out_proj.weight",
        "output_stack.layers.0.tri_mul_out.norm_start.weight",
    }


def test_recurrence_init_state_is_truncated_normal_in_fp32_then_cast() -> None:
    rec = Recurrence(256, coda_blocks=1, trimul_backend="reference")
    like = torch.zeros(1, 40, 40, 256, dtype=torch.bfloat16)
    z0 = rec.init_state(like, generator=torch.Generator().manual_seed(0))
    std = math.sqrt(2.0 / (5.0 * 256))
    assert z0.dtype == torch.bfloat16 and z0.shape == like.shape
    assert (
        abs(z0.float().std().item() - std) < 0.1 * std and z0.float().abs().max() <= 3 * std + 1e-3
    )
    again = rec.init_state(like, generator=torch.Generator().manual_seed(0))
    assert torch.equal(z0, again)


def test_recurrence_run_matches_a_transcribed_loop_and_records_states() -> None:
    torch.manual_seed(0)
    rec = Recurrence(_W, coda_blocks=1, trimul_backend="reference").eval()
    trunk = PairStack(1, _W, trimul_backend="reference").eval()
    z_init, lm = _pair(seed=1), _pair(seed=2)
    mask = torch.ones(1, 6, 6)
    z0 = rec.init_state(z_init, generator=torch.Generator().manual_seed(3))

    with torch.no_grad():
        z, states = rec.run(
            trunk, lambda _t: z_init + lm, z0=z0, pair_mask=mask, num_loops=3, return_states=True
        )
        # upstream _run_one_loop, transcribed
        a, b_mat = rec.dynamics()
        a = a.view(1, 1, 1, -1)
        ref = z0
        for _ in range(3):
            injected = rec.input_norm(z_init + lm)
            ref = a * ref + F.linear(injected, b_mat)
            ref = trunk(ref, mask)
    torch.testing.assert_close(z, ref)
    assert len(states) == 3 and torch.equal(states[-1], z)


def test_recurrence_grad_loops_truncate_backpropagation() -> None:
    torch.manual_seed(0)
    rec = Recurrence(_W, coda_blocks=1, trimul_backend="reference").train()
    trunk = PairStack(1, _W, trimul_backend="reference").train()
    z_init = _pair(seed=1).requires_grad_(True)
    z0 = rec.init_state(z_init.detach()).requires_grad_(True)
    z, _ = rec.run(trunk, lambda _t: z_init, z0=z0, pair_mask=None, num_loops=3, grad_loops=1)
    z.sum().backward()
    assert z0.grad is None  # the first two loops ran under no_grad
    assert z_init.grad is not None and rec.log_delta.grad is not None


def test_pair_stack_kwargs_come_from_config() -> None:
    cfg = FoldConfig(trunk_dropout=0.1, trimul_backend="reference", trimul_chunk_size=None)
    kw = pair_stack_kwargs(cfg)
    assert kw == {
        "expansion": 4,
        "dropout": 0.1,
        "eps": 1e-5,
        "chunk_size": None,
        "trimul_backend": "reference",
    }
    stack = PairStack(cfg.trunk_blocks, cfg.pair_width, **kw)
    assert len(stack.layers) == 24
    assert stack.layers[0].pair_transition.mlp.gate_up_proj.weight.shape == (2048, 256)


@pytest.mark.parametrize("bad", [{"expansion": 0}, {"dropout": 1.0}])
def test_block_rejects_bad_arguments(bad: dict) -> None:
    with pytest.raises(ValueError):
        PairUpdateBlock(_W, **bad)

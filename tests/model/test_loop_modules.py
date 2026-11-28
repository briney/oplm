"""Unit tests for Step 2 core-boundary loop modules."""

from __future__ import annotations

import math

import pytest
import torch

from oplm.model.loop_modules import LoopCycleGate, LoopInputInjector, LoopStepConditioner


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_concat_proj_injector_shapes_and_grads(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    injector = LoopInputInjector("concat_proj", hidden_size=32, norm_type="layernorm", norm_eps=1e-6)
    injector = injector.to(dtype=dtype)
    z = torch.randn(2, 6, 32, dtype=dtype, requires_grad=True)
    e = torch.randn(2, 6, 32, dtype=dtype, requires_grad=True)

    out = injector(z, e)
    assert out.shape == (2, 6, 32)
    assert out.dtype == dtype
    out.float().pow(2).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert e.grad is not None and torch.isfinite(e.grad).all()
    assert injector.proj.weight.grad is not None and torch.isfinite(injector.proj.weight.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decayed_injector_matches_formula_and_init(dtype: torch.dtype) -> None:
    torch.manual_seed(1)
    injector = LoopInputInjector("decayed", hidden_size=32, norm_type="layernorm", norm_eps=1e-6)
    delta = torch.nn.functional.softplus(injector.log_delta)
    a = torch.exp(-delta * torch.exp(injector.log_state_decay))
    torch.testing.assert_close(a, torch.full((32,), 1.0 / math.sqrt(5.0)), rtol=1e-5, atol=1e-5)
    assert torch.equal(injector.input_matrix_continuous, torch.eye(32))

    z = torch.randn(2, 6, 32, dtype=dtype, requires_grad=True)
    e = torch.randn(2, 6, 32, dtype=dtype, requires_grad=True)
    out = injector(z, e)
    assert out.shape == (2, 6, 32)
    assert out.dtype == dtype

    expected = (
        a.to(dtype) * z
        + torch.nn.functional.linear(
            injector.input_norm(e).to(dtype),
            (delta[:, None] * injector.input_matrix_continuous).to(dtype),
        )
    )
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    out.float().pow(2).sum().backward()
    for tensor in (
        z,
        e,
        injector.log_delta,
        injector.log_state_decay,
        injector.input_matrix_continuous,
    ):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_orthogonal_injector_projects_out_e_and_handles_zero_norm_rows() -> None:
    torch.manual_seed(2)
    injector = LoopInputInjector("orthogonal", hidden_size=32, norm_type="layernorm", norm_eps=1e-6)
    z = torch.randn(2, 4, 32, requires_grad=True)
    e_data = torch.randn(2, 4, 32)
    e_data[0, 0] = 0.0  # zero-norm pad/masked position
    e = e_data.clone().requires_grad_(True)

    out = injector(z, e)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()

    # Non-zero rows: (out - scale * e) must be orthogonal to e.
    residual = out - injector.inject_scale * e
    dots = (residual * e).sum(dim=-1)
    assert dots[0, 1:].abs().max().item() < 1e-5
    assert dots[1].abs().max().item() < 1e-5

    out.pow(2).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert e.grad is not None and torch.isfinite(e.grad).all()
    assert injector.inject_scale.grad is not None and torch.isfinite(injector.inject_scale.grad).all()


@pytest.mark.parametrize("mode", ["scalar", "channel"])
def test_cycle_gate_parameter_list_is_1d_and_zero_init_is_exact_identity(mode: str) -> None:
    gate = LoopCycleGate(mode=mode, num_loops=3, hidden_size=32, init_val=0.0)
    assert len(gate.gates) == 2
    expected_shape = (1,) if mode == "scalar" else (32,)
    for p in gate.gates:
        assert p.ndim == 1
        assert p.shape == expected_shape

    torch.manual_seed(3)
    z_prev = torch.randn(2, 5, 32, requires_grad=True)
    core_out = torch.randn(2, 5, 32, requires_grad=True)
    out = gate(z_prev, core_out, 0)
    assert torch.equal(out, z_prev)

    out.pow(2).sum().backward()
    assert gate.gates[0].grad is not None
    assert gate.gates[0].grad.abs().sum().item() > 0.0
    assert gate.gates[1].grad is None


@pytest.mark.parametrize("hidden_size", [32, 31])
def test_step_conditioner_zero_init_identity_and_odd_hidden_size(hidden_size: int) -> None:
    torch.manual_seed(4)
    cond = LoopStepConditioner(hidden_size=hidden_size, norm_type="layernorm", norm_eps=1e-6)
    assert getattr(cond.fc2, "_init_zero", False) is True

    z = torch.randn(2, 5, hidden_size)
    out = cond(z, pass_idx=1, num_loops=3)
    assert torch.equal(out, z)

    with torch.no_grad():
        cond.fc2.weight.normal_(std=0.1)
    out1 = cond(z, pass_idx=1, num_loops=3)
    out2 = cond(z, pass_idx=2, num_loops=3)
    assert out1.shape == z.shape and out2.shape == z.shape
    assert not torch.allclose(out1, out2)

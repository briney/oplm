"""Semantic oracle tests for the paper-exact encoder Canon implementation.

These tests go beyond module-existence and shape checks (covered in
``test_conv.py`` / ``test_attention.py`` / ``test_transformer.py``): they prove
that each Canon position acts on the intended tensor stream, at the intended
point in the block dataflow, with the intended residual semantics, and that the
bidirectional convolution window has the pinned centered alignment.

Test categories:

1. Operation-order tests with forward hooks (A/B/C/D call order, B after
   projection, pinned ``projection -> QK/V norm -> Canon-B -> RoPE`` order).
2. Tiny numerical reference tests: block output vs a hand-coded reference for
   each Canon position, with an independent loop-based depthwise convolution.
3. Canon-D tensor-space tests: D sees ``(B, T, intermediate_size)`` at runtime.
4. Residual-vs-replacement tests: ``canon_residual=True`` preserves the
   identity path (zero kernels are a no-op); ``canon_residual=False`` does not.
5. Centered-conv impulse tests: exact receptive-field windows for odd/even
   kernels, anti-causal (bidirectional) influence, and pad isolation.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from oplm.model.configuration_oplm import OplmConfig
from oplm.model.conv import CanonConv
from oplm.model.transformer import OplmBlock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(**overrides) -> OplmConfig:
    """Tiny deterministic OplmConfig for semantic tests (pre-norm, no dropout)."""
    defaults = dict(
        hidden_size=16,
        num_attention_heads=2,
        intermediate_size=32,
        num_hidden_layers=2,
        max_position_embeddings=32,
        norm_strategy="pre",
        residual_scaling="none",
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    defaults.update(overrides)
    return OplmConfig(**defaults)


def _canon_config(
    positions: list[str],
    *,
    residual: bool = True,
    kernel_size: int = 3,
    **overrides,
) -> OplmConfig:
    return _config(
        canon_enabled=True,
        canon_residual=residual,
        canon_positions=positions,
        canon_kernel_sizes=kernel_size,
        canon_activation="none",
        **overrides,
    )


def _padded_inputs(
    batch: int = 2, seq_len: int = 6, hidden: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic `(x, attention_mask)` with trailing pads in the last row."""
    torch.manual_seed(7)
    x = torch.randn(batch, seq_len, hidden)
    mask = torch.ones(batch, seq_len, dtype=torch.long)
    mask[-1, -2:] = 0
    return x, mask


def _ref_depthwise_conv(
    x: torch.Tensor, weight: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Independent loop-based reference for `CanonConv` (no nn.Conv1d).

    Computes the cross-correlation ``out[t] = sum_j w[j] * x[t + j - k//2]``
    with out-of-range positions treated as zero — the centered window for odd
    `k` and the left-biased ``(k//2, k//2 - 1)`` window for even `k`. Pad
    positions are zeroed before the kernel runs.

    Args:
        x: `(B, T, C)` input.
        weight: `(C, 1, K)` depthwise kernel (from `CanonConv.conv.weight`).
        attention_mask: `(B, T)` mask with `1` at real tokens, `0` at pads.

    Returns:
        `(B, T, C)` convolved output.
    """
    x = x * attention_mask[..., None].to(x.dtype)
    _, seq_len, _ = x.shape
    kernel_size = weight.shape[-1]
    out = torch.zeros_like(x)
    for t in range(seq_len):
        for j in range(kernel_size):
            src = t + j - kernel_size // 2
            if 0 <= src < seq_len:
                out[:, t, :] += weight[:, 0, j] * x[:, src, :]
    return out


def _ref_apply_canon(
    z: torch.Tensor,
    conv: CanonConv,
    attention_mask: torch.Tensor,
    residual: bool,
) -> torch.Tensor:
    """Reference Canon application: `z + conv(z)` (residual) or `conv(z)`."""
    convolved = _ref_depthwise_conv(z, conv.conv.weight, attention_mask)
    return z + convolved if residual else convolved


def _ref_block_forward(
    block: OplmBlock,
    x: torch.Tensor,
    attention_mask: torch.Tensor,
    positions: set[str],
    residual: bool,
) -> torch.Tensor:
    """Hand-coded reference for one pre-norm block with Canon at `positions`.

    Encodes the paper-exact dataflow directly (A on the attention pre-norm
    stream, B on the flat projected Q/K/V, C on the FFN pre-norm stream, D on
    the SwiGLU gate branch before SiLU) using the block's leaf parameters and
    the loop-based conv reference. Assumes `norm_strategy="pre"`, SwiGLU,
    `qk_norm=False`, `rope_dim=0` (NoPE), no dropout, and `alpha == 1`.
    """
    attn = block.attention
    ffn = block.ffn
    batch, seq_len, hidden = x.shape
    heads, d_head = attn.num_attention_heads, attn.head_dim

    # Attention sublayer.
    a_base = block.attn_norm(x)
    a_in = a_base
    if "A" in positions:
        a_in = _ref_apply_canon(a_base, block.conv_a, attention_mask, residual)

    q_flat = F.linear(a_in, attn.q_proj.weight)  # (B, T, D)
    k_flat = F.linear(a_in, attn.k_proj.weight)
    v_flat = F.linear(a_in, attn.v_proj.weight)
    if "B" in positions:
        # The depthwise conv has no cross-channel mixing, so convolving the
        # flat (B, T, D) projections is the per-head operation's reference.
        q_flat = _ref_apply_canon(q_flat, attn.conv_b_q, attention_mask, residual)
        k_flat = _ref_apply_canon(k_flat, attn.conv_b_k, attention_mask, residual)
        v_flat = _ref_apply_canon(v_flat, attn.conv_b_v, attention_mask, residual)

    def split(t: torch.Tensor) -> torch.Tensor:
        return t.view(batch, seq_len, heads, d_head).transpose(1, 2)

    q, k, v = split(q_flat), split(k_flat), split(v_flat)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)  # (B, H, T, T)
    scores = scores.masked_fill((attention_mask == 0)[:, None, None, :], float("-inf"))
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
    attn_out = torch.matmul(probs, v)  # (B, H, T, d_head)
    merged = attn_out.transpose(1, 2).reshape(batch, seq_len, hidden)
    h = x + F.linear(merged, attn.o_proj.weight)  # alpha == 1

    # FFN sublayer (SwiGLU).
    m_base = block.ffn_norm(h)
    m_in = m_base
    if "C" in positions:
        m_in = _ref_apply_canon(m_base, block.conv_c, attention_mask, residual)
    gate = F.linear(m_in, ffn.gate_proj.weight)  # (B, T, F)
    if "D" in positions:
        gate = _ref_apply_canon(gate, ffn.conv_d, attention_mask, residual)
    up = F.linear(m_in, ffn.up_proj.weight)
    ffn_out = F.linear(F.silu(gate) * up, ffn.down_proj.weight)
    return h + ffn_out


# ---------------------------------------------------------------------------
# 1. Operation-order tests with hooks
# ---------------------------------------------------------------------------


def _record_hook(events: list[str], name: str):
    def hook(module: nn.Module, args: tuple, output: object) -> None:
        events.append(name)

    return hook


def test_canon_full_operation_order():
    """A/B/C/D fire at the paper-exact points in the block dataflow.

    Pins the full event sequence: attention pre-norm -> Canon-A -> Q/K/V
    projection -> QK/V norm -> Canon-B -> RoPE -> attention output -> FFN
    pre-norm -> Canon-C -> gate projection -> Canon-D -> activation/output
    projections.
    """
    torch.manual_seed(0)
    cfg = _canon_config(["A", "B", "C", "D"], qk_norm=True)
    block = OplmBlock(cfg, layer_idx=0).eval()
    attn = block.attention

    events: list[str] = []
    hooked = {
        "attn_norm": block.attn_norm,
        "conv_a": block.conv_a,
        "q_proj": attn.q_proj,
        "k_proj": attn.k_proj,
        "v_proj": attn.v_proj,
        "q_norm": attn.q_norm,
        "k_norm": attn.k_norm,
        "v_norm": attn.v_norm,
        "conv_b_q": attn.conv_b_q,
        "conv_b_k": attn.conv_b_k,
        "conv_b_v": attn.conv_b_v,
        "o_proj": attn.o_proj,
        "attention": attn,
        "ffn_norm": block.ffn_norm,
        "conv_c": block.conv_c,
        "gate_proj": block.ffn.gate_proj,
        "conv_d": block.ffn.conv_d,
        "up_proj": block.ffn.up_proj,
        "down_proj": block.ffn.down_proj,
        "ffn": block.ffn,
    }
    for name, module in hooked.items():
        module.register_forward_hook(_record_hook(events, name))

    # `apply_rotary` is called as a method (not via __call__), so wrap it to
    # record the RoPE event.
    original_apply_rotary = attn.rotary.apply_rotary

    def recording_apply_rotary(q: torch.Tensor, k: torch.Tensor):
        events.append("rope")
        return original_apply_rotary(q, k)

    attn.rotary.apply_rotary = recording_apply_rotary

    x, mask = _padded_inputs()
    with torch.no_grad():
        block(x, mask)

    assert events == [
        "attn_norm",
        "conv_a",
        "q_proj",
        "k_proj",
        "v_proj",
        "q_norm",
        "k_norm",
        "v_norm",
        "conv_b_q",
        "conv_b_k",
        "conv_b_v",
        "rope",
        "o_proj",
        "attention",
        "ffn_norm",
        "conv_c",
        "gate_proj",
        "conv_d",
        "up_proj",
        "down_proj",
        "ffn",
    ]


def test_canon_a_input_is_attention_pre_norm_output():
    """Canon-A acts on the attention pre-norm stream, not raw `x`."""
    torch.manual_seed(0)
    block = OplmBlock(_canon_config(["A"]), layer_idx=0).eval()

    captured: dict[str, torch.Tensor] = {}
    block.attn_norm.register_forward_hook(
        lambda m, args, out: captured.__setitem__("norm_out", out)
    )
    block.conv_a.register_forward_hook(
        lambda m, args, out: captured.__setitem__("conv_in", args[0])
    )

    x, mask = _padded_inputs()
    with torch.no_grad():
        block(x, mask)

    assert torch.equal(captured["conv_in"], captured["norm_out"])
    assert not torch.equal(captured["conv_in"], x)


def test_canon_c_input_is_ffn_pre_norm_output_not_attention_output():
    """Canon-C acts on the FFN pre-norm stream — never on the attention output."""
    torch.manual_seed(0)
    block = OplmBlock(_canon_config(["C"]), layer_idx=0).eval()

    captured: dict[str, torch.Tensor] = {}
    block.ffn_norm.register_forward_hook(lambda m, args, out: captured.__setitem__("norm_out", out))
    block.conv_c.register_forward_hook(
        lambda m, args, out: captured.__setitem__("conv_in", args[0])
    )
    block.attention.register_forward_hook(
        lambda m, args, out: captured.__setitem__("attn_out", out[0])
    )

    x, mask = _padded_inputs()
    with torch.no_grad():
        block(x, mask)

    assert torch.equal(captured["conv_in"], captured["norm_out"])
    assert not torch.equal(captured["conv_in"], captured["attn_out"])


def test_canon_b_inputs_are_projected_qkv():
    """Canon-B acts on the projected Q/K/V (after the Linear projections).

    With `qk_norm=False` the projection output reaches the conv unchanged, so
    each conv-B stream input must equal the corresponding projection output
    exactly.
    """
    torch.manual_seed(0)
    block = OplmBlock(_canon_config(["B"], qk_norm=False), layer_idx=0).eval()
    attn = block.attention

    captured: dict[str, torch.Tensor] = {}
    for name in ("q", "k", "v"):
        getattr(attn, f"{name}_proj").register_forward_hook(
            lambda m, args, out, name=name: captured.__setitem__(f"{name}_proj_out", out)
        )
        getattr(attn, f"conv_b_{name}").register_forward_hook(
            lambda m, args, out, name=name: captured.__setitem__(f"conv_b_{name}_in", args[0])
        )

    x, mask = _padded_inputs()
    with torch.no_grad():
        block(x, mask)

    for name in ("q", "k", "v"):
        assert torch.equal(captured[f"conv_b_{name}_in"], captured[f"{name}_proj_out"])


# ---------------------------------------------------------------------------
# 2. Tiny numerical reference tests (one oracle per position, plus combined)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("residual", [True, False], ids=["residual", "replacement"])
@pytest.mark.parametrize(
    "positions",
    [["A"], ["B"], ["C"], ["D"], ["A", "B", "C", "D"]],
    ids=["A", "B", "C", "D", "ABCD"],
)
def test_block_matches_hand_coded_reference(positions: list[str], residual: bool):
    """Block output equals the hand-coded paper-exact reference for each position.

    Uses NoPE (`rope_dim=0`) and `qk_norm=False` so the reference needs no RoPE
    or norm hand-coding; the Canon insertion points and residual semantics are
    fully exercised. This also proves residual-vs-replacement numerically: the
    reference applies `z + conv(z)` or `conv(z)` per the `residual` flag.
    """
    torch.manual_seed(11)
    cfg = _canon_config(positions, residual=residual, qk_norm=False, rope_dim=0, nope_dim=8)
    block = OplmBlock(cfg, layer_idx=0).eval()

    x, mask = _padded_inputs()
    with torch.no_grad():
        actual = block(x, mask)[0]
        expected = _ref_block_forward(block, x, mask, set(positions), residual)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    # The reference must be sensitive to the Canon path: with the conv outputs
    # removed entirely it must disagree (guards against a vacuous oracle).
    with torch.no_grad():
        no_canon_expected = _ref_block_forward(block, x, mask, set(), True)
    assert not torch.allclose(actual, no_canon_expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Canon-D tensor-space tests
# ---------------------------------------------------------------------------


def test_canon_d_sees_intermediate_width_at_runtime():
    """Canon-D operates on the `(B, T, intermediate_size)` gate branch."""
    torch.manual_seed(0)
    cfg = _canon_config(["D"], intermediate_size=40)  # != hidden_size (16)
    block = OplmBlock(cfg, layer_idx=0).eval()
    conv_d = block.ffn.conv_d

    assert conv_d.conv.in_channels == cfg.intermediate_size
    assert conv_d.conv.groups == cfg.intermediate_size

    captured: dict[str, torch.Tensor] = {}
    block.ffn.gate_proj.register_forward_hook(
        lambda m, args, out: captured.__setitem__("gate_out", out)
    )
    conv_d.register_forward_hook(lambda m, args, out: captured.__setitem__("conv_in", args[0]))

    x, mask = _padded_inputs()
    with torch.no_grad():
        block(x, mask)

    batch, seq_len, _ = x.shape
    assert captured["conv_in"].shape == (batch, seq_len, cfg.intermediate_size)
    # D acts on the pre-activation gate branch, not the FFN's hidden-size input.
    assert torch.equal(captured["conv_in"], captured["gate_out"])


# ---------------------------------------------------------------------------
# 4. Residual-vs-replacement tests (identity-path preservation)
# ---------------------------------------------------------------------------


def _zeroed_canon_block(positions: list[str], residual: bool) -> tuple[OplmBlock, OplmBlock]:
    """Build a no-Canon block and a Canon block sharing weights, convs zeroed."""
    torch.manual_seed(3)
    plain = OplmBlock(_config(), layer_idx=0).eval()
    canon = OplmBlock(_canon_config(positions, residual=residual), layer_idx=0).eval()
    # Share every non-conv parameter; missing keys are exactly the Canon convs.
    canon.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        for module in canon.modules():
            if isinstance(module, CanonConv):
                module.conv.weight.zero_()
    return plain, canon


@pytest.mark.parametrize(
    "positions",
    [["A"], ["B"], ["C"], ["D"], ["A", "B", "C", "D"]],
    ids=["A", "B", "C", "D", "ABCD"],
)
def test_residual_canon_with_zero_kernel_is_identity(positions: list[str]):
    """`canon_residual=True` preserves the identity path: zero kernels are a no-op."""
    plain, canon = _zeroed_canon_block(positions, residual=True)
    x, mask = _padded_inputs()
    with torch.no_grad():
        expected = plain(x, mask)[0]
        actual = canon(x, mask)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "positions",
    [["A"], ["B"], ["C"], ["D"], ["A", "B", "C", "D"]],
    ids=["A", "B", "C", "D", "ABCD"],
)
def test_replacement_canon_with_zero_kernel_destroys_identity(positions: list[str]):
    """`canon_residual=False` replaces the stream: zero kernels change the output."""
    plain, canon = _zeroed_canon_block(positions, residual=False)
    x, mask = _padded_inputs()
    with torch.no_grad():
        expected = plain(x, mask)[0]
        actual = canon(x, mask)[0]
    assert not torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Centered-conv impulse and alignment tests
# ---------------------------------------------------------------------------


def _impulse_conv(kernel: list[float], seq_len: int = 9, channels: int = 2) -> CanonConv:
    """CanonConv with every channel's kernel pinned to `kernel` (no activation)."""
    conv = CanonConv(channels, len(kernel)).eval()
    with torch.no_grad():
        conv.conv.weight.copy_(
            torch.tensor(kernel, dtype=torch.float32).view(1, 1, -1).expand(channels, 1, -1)
        )
    return conv


def _impulse_input(
    position: int, seq_len: int = 9, channels: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros(1, seq_len, channels)
    x[0, position, :] = 1.0
    mask = torch.ones(1, seq_len, dtype=torch.long)
    return x, mask


def test_impulse_odd_kernel_affects_centered_window():
    """k=3, all-ones kernel: an impulse at `p` reaches exactly `{p-1, p, p+1}`."""
    conv = _impulse_conv([1.0, 1.0, 1.0])
    x, mask = _impulse_input(position=4)
    with torch.no_grad():
        out = conv(x, mask)
    nonzero = sorted(torch.nonzero(out[0, :, 0]).flatten().tolist())
    assert nonzero == [3, 4, 5]


def test_impulse_even_kernel_affects_left_biased_window():
    """k=4 uses the documented `(k//2, k//2 - 1)` pad: out[t] sees x[t-2..t+1].

    An impulse at `p` therefore reaches exactly outputs `{p-1, p, p+1, p+2}` —
    the half-token-alignment encoder adaptation of the paper's k=4 window.
    """
    conv = _impulse_conv([1.0, 1.0, 1.0, 1.0])
    x, mask = _impulse_input(position=4)
    with torch.no_grad():
        out = conv(x, mask)
    nonzero = sorted(torch.nonzero(out[0, :, 0]).flatten().tolist())
    assert nonzero == [3, 4, 5, 6]


def test_impulse_odd_kernel_orientation_is_pinned():
    """k=3 kernel [1, 2, 4]: out[t] = 1*x[t-1] + 2*x[t] + 4*x[t+1] exactly."""
    conv = _impulse_conv([1.0, 2.0, 4.0])
    x, mask = _impulse_input(position=4)
    with torch.no_grad():
        out = conv(x, mask)
    expected = torch.zeros(9)
    expected[3] = 4.0  # x[4] enters out[3] via the j=2 (future) tap
    expected[4] = 2.0
    expected[5] = 1.0  # x[4] enters out[5] via the j=0 (past) tap
    torch.testing.assert_close(out[0, :, 0], expected)


def test_impulse_even_kernel_orientation_is_pinned():
    """k=4 kernel [1, 2, 4, 8]: out[t] = 1*x[t-2] + 2*x[t-1] + 4*x[t] + 8*x[t+1]."""
    conv = _impulse_conv([1.0, 2.0, 4.0, 8.0])
    x, mask = _impulse_input(position=4)
    with torch.no_grad():
        out = conv(x, mask)
    expected = torch.zeros(9)
    expected[3] = 8.0
    expected[4] = 4.0
    expected[5] = 2.0
    expected[6] = 1.0
    torch.testing.assert_close(out[0, :, 0], expected)


def test_conv_is_bidirectional_not_causal():
    """A future token influences a past output position (anti-causal proof)."""
    conv = _impulse_conv([1.0, 1.0, 1.0])
    x, mask = _impulse_input(position=5)
    with torch.no_grad():
        out = conv(x, mask)
    assert out[0, 4, 0].item() != 0.0  # position 4 sees the impulse at 5


def test_impulse_at_pad_position_does_not_reach_real_tokens():
    """Content at a pad position is zeroed before the kernel runs."""
    conv = _impulse_conv([1.0, 1.0, 1.0])
    x, mask = _impulse_input(position=6)
    mask[0, 6:] = 0  # the impulse sits on a pad position
    with torch.no_grad():
        out = conv(x, mask)
    real = out[0, :6, :]
    torch.testing.assert_close(real, torch.zeros_like(real))


@pytest.mark.parametrize("kernel_size", [2, 3, 4, 5], ids=lambda k: f"k{k}")
def test_canon_conv_matches_loop_reference(kernel_size: int):
    """CanonConv equals the independent loop reference for random weights/input."""
    torch.manual_seed(kernel_size)
    conv = CanonConv(channels=5, kernel_size=kernel_size).eval()
    x = torch.randn(2, 8, 5)
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[1, -3:] = 0
    with torch.no_grad():
        actual = conv(x, mask)
    expected = _ref_depthwise_conv(x, conv.conv.weight, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

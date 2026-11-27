"""Tests for 2D contact-pair distillation heads, masked APC, chunking, and aux optimizer."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from oplm.config import DistillConfig, TrainConfig
from oplm.data.sequence.collate import encode_teacher_contacts
from oplm.eval.metrics.contact import apply_apc
from oplm.training.contact_distill import (
    AttentionContactHead,
    AuxContactManager,
    PairContactHead,
    chunk_contact_segments,
    compute_contact_distill_loss,
    masked_apc,
)


def test_masked_apc_matches_eval_apply_apc() -> None:
    torch.manual_seed(0)
    lens = [8, 12, 5]
    l_max = max(lens)
    raw = torch.rand(3, 4, l_max, l_max)
    # Symmetrize & mask out padding
    valid = torch.zeros(3, l_max, dtype=torch.bool)
    for b, length in enumerate(lens):
        valid[b, :length] = True
    pair_mask = valid[:, :, None] & valid[:, None, :]
    mat = 0.5 * (raw + raw.transpose(-1, -2)) * pair_mask[:, None, :, :]

    out = masked_apc(mat, lens)
    for b, length in enumerate(lens):
        for h in range(4):
            expected = apply_apc(mat[b, h, :length, :length])
            torch.testing.assert_close(out[b, h, :length, :length], expected, atol=1e-5, rtol=1e-5)
        # Padding positions must be strictly zero
        if length < l_max:
            assert out[b, :, length:, :].abs().max().item() == 0.0
            assert out[b, :, :, length:].abs().max().item() == 0.0


def test_attention_contact_head_chunking_symmetry_and_scale() -> None:
    torch.manual_seed(2)
    # Packed row of length 60 holding 3 segments: L=10 (@0), L=12 (@12), L=16 (@26)
    num_layers, num_heads, head_dim = 2, 3, 8
    qk_states = tuple(
        (
            torch.randn(1, num_heads, 60, head_dim, requires_grad=True),
            torch.randn(1, num_heads, 60, head_dim, requires_grad=True),
        )
        for _ in range(num_layers)
    )
    attn_scales = (1.0 / (head_dim**0.5), 1.0)  # mixed channel + L2 scale check
    head = AttentionContactHead(num_layers=num_layers, num_heads=num_heads, attn_scales=attn_scales)
    assert head.proj.bias.item() == pytest.approx(-3.0)

    rng = np.random.default_rng(42)
    segments_info = [(0, 0, 10, 10), (0, 12, 12, 12), (0, 26, 16, 16)]
    chunks = []
    meta = []
    offset = 0
    for row, start, seg_len, c_len in segments_info:
        p = rng.uniform(0.0, 1.0, (c_len, c_len)).astype(np.float32)
        p = 0.5 * (p + p.T)
        raw = encode_teacher_contacts(p, min_sep=6)
        arr = np.frombuffer(raw[4:], dtype=np.uint8).copy()
        meta.append((row, start, seg_len, c_len, offset))
        chunks.append(arr)
        offset += len(arr)
    values = torch.from_numpy(np.concatenate(chunks))

    # Compare single large chunk vs forced 1-segment-per-chunk
    loss_batched, pos_batched = compute_contact_distill_loss(
        head=head,
        qk_states=qk_states,
        hidden_states=None,
        contact_meta=meta,
        contact_values=values,
        contact_min_sep=6,
        pos_weight=1.0,
        max_chunk_pairs=10_000_000,
    )
    loss_single, pos_single = compute_contact_distill_loss(
        head=head,
        qk_states=qk_states,
        hidden_states=None,
        contact_meta=meta,
        contact_values=values,
        contact_min_sep=6,
        pos_weight=1.0,
        max_chunk_pairs=1,
    )
    torch.testing.assert_close(loss_batched, loss_single, atol=1e-5, rtol=1e-5)
    assert pos_batched == pytest.approx(pos_single, abs=1e-6)

    # Check symmetry directly from head.forward_chunk
    chunk_logits = head.forward_chunk(qk_states, meta)
    torch.testing.assert_close(chunk_logits, chunk_logits.transpose(-1, -2), atol=1e-6, rtol=1e-6)

    # Gradients flow to q, k and head weights
    loss_batched.backward()
    for q, k in qk_states:
        assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
        assert k.grad is not None and torch.isfinite(k.grad).all() and k.grad.abs().sum() > 0
    assert head.proj.weight.grad is not None and head.proj.weight.grad.abs().sum() > 0


def test_pair_contact_head_symmetry_and_gradients() -> None:
    torch.manual_seed(4)
    num_layers, hidden_size, pair_dim = 2, 32, 16
    hidden_states = tuple(torch.randn(1, 40, hidden_size, requires_grad=True) for _ in range(3))
    head = PairContactHead(num_layers=num_layers, hidden_size=hidden_size, pair_dim=pair_dim)
    assert head.fc2.bias.item() == pytest.approx(-3.0)

    # Two segments (including one prefix-cropped: seg_res_len=14, contact_len=10)
    meta = [(0, 0, 14, 10, 0), (0, 16, 12, 12, 10)]
    logits = head.forward_chunk(hidden_states[-num_layers:], meta)
    assert logits.shape == (2, 12, 12)
    torch.testing.assert_close(logits, logits.transpose(-1, -2), atol=1e-6, rtol=1e-6)

    values = torch.full((10 + 21,), 128, dtype=torch.uint8)
    loss, pos_rate = compute_contact_distill_loss(
        head=head,
        qk_states=None,
        hidden_states=hidden_states[-num_layers:],
        contact_meta=meta,
        contact_values=values,
        contact_min_sep=6,
        pos_weight=1.0,
        max_chunk_pairs=1_000_000,
    )
    assert pos_rate == pytest.approx(128.0 / 255.0, abs=1e-5)
    loss.backward()
    assert hidden_states[-1].grad is not None and hidden_states[-1].grad.abs().sum() > 0
    assert hidden_states[-2].grad is not None and hidden_states[-2].grad.abs().sum() > 0
    assert head.fc2.weight.grad is not None


def test_soft_bce_per_sequence_averaging_and_pos_weight() -> None:
    # One short segment (L=10 -> 10 pairs, target=1) and one long (L=30 -> 300 pairs, target=0).
    # Per-sequence mean must weight both 50/50 rather than by pair count.
    class _ZeroHead(torch.nn.Module):
        def forward_chunk(self, inputs, chunk_meta):
            l_max = max(m[3] for m in chunk_meta)
            return torch.zeros(len(chunk_meta), l_max, l_max)

    n_short = (10 - 6) * (10 - 5) // 2  # 10
    n_long = (30 - 6) * (30 - 5) // 2  # 300
    meta = [(0, 0, 10, 10, 0), (0, 12, 30, 30, n_short)]
    vals = torch.cat(
        [
            torch.full((n_short,), 255, dtype=torch.uint8),
            torch.zeros((n_long,), dtype=torch.uint8),
        ]
    )
    loss_w2, _ = compute_contact_distill_loss(
        head=_ZeroHead(),
        qk_states=(),
        hidden_states=None,
        contact_meta=meta,
        contact_values=vals,
        contact_min_sep=6,
        pos_weight=3.0,
        max_chunk_pairs=1_000_000,
    )
    # At logit=0: bce(0, 1)=ln(2) * 3.0 and bce(0, 0)=ln(2) * 1.0 -> mean = 2.0 * ln(2).
    expected = 2.0 * float(F.softplus(torch.tensor(0.0)).item())
    assert loss_w2.item() == pytest.approx(expected, rel=1e-5)


def test_chunk_contact_segments_respects_budget() -> None:
    # Lengths 10, 12, 100 with budget 500: (10, 12) fits (2 * 14^2 = 392 <= 500), 100 splits alone.
    meta = [(0, 0, 10, 10, 0), (0, 12, 12, 12, 10), (0, 26, 100, 100, 31)]
    chunks = chunk_contact_segments(meta, max_chunk_pairs=500)
    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 1


def test_aux_contact_manager_branch_warmup_and_decay() -> None:
    head = AttentionContactHead(num_layers=2, num_heads=4, attn_scales=(0.25, 0.25))
    distill_cfg = DistillConfig(
        contact_weight=1.0,
        contact_head="attention",
        contact_layers=2,
        contact_head_lr=1e-3,
        contact_head_warmup_steps=200,
    )
    train_cfg = TrainConfig(
        scheduler="wsd_linear",
        lr=2e-3,
        min_lr=0.0,
        warmup_steps=2000,
        stable_steps=998_000,
        max_steps=1_025_000,
    )
    mgr = AuxContactManager(head, distill_cfg, train_cfg, total_steps=1_025_000)
    mgr.sync_schedule(global_step=1_000_000, fresh_start=True)
    assert mgr.aux_start_step == 1_000_000
    # Step 1,000,000: backbone mult=1.0, warmup ramp=1/200
    assert mgr.current_lr == pytest.approx(1e-3 / 200.0, rel=1e-5)
    # Step 1,000,199: ramp=200/200=1.0, backbone decay progress=199/25000
    lr_end_warmup = mgr.lr_for_step(1_000_199)
    assert lr_end_warmup == pytest.approx(1e-3 * (1.0 - 199.0 / 25_000.0), rel=1e-5)
    # Step 1,012,500: halfway through decay
    assert mgr.lr_for_step(1_012_500) == pytest.approx(0.5e-3, rel=1e-5)
    # Step 1,025,000: decayed cleanly to 0.0 (never stuck on stable_steps)
    assert mgr.lr_for_step(1_025_000) == pytest.approx(0.0, abs=1e-12)

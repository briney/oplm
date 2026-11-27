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


def test_contact_heads_emit_fp32_under_bf16_autocast_and_skip_zero_pair_segments() -> None:
    torch.manual_seed(4)
    attn_head = AttentionContactHead(num_layers=1, num_heads=2, attn_scales=(0.5,))
    pair_head = PairContactHead(num_layers=1, hidden_size=8, pair_dim=4)
    q = torch.randn(1, 2, 10, 4)
    k = torch.randn(1, 2, 10, 4)
    h = torch.randn(1, 10, 8)
    meta_ok = [(0, 0, 8, 8, 0)]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert attn_head.forward_chunk([(q, k)], meta_ok).dtype == torch.float32
        assert pair_head.forward_chunk([h], meta_ok).dtype == torch.float32

    # Direct call with contact_len <= contact_min_sep produces finite 0.0 (never NaN).
    loss_empty, pos_rate = compute_contact_distill_loss(
        head=attn_head,
        qk_states=[(q, k)],
        hidden_states=None,
        contact_meta=[(0, 0, 6, 6, 0)],
        contact_values=torch.empty((0,), dtype=torch.uint8),
        contact_min_sep=6,
        pos_weight=1.0,
        max_chunk_pairs=1_000_000,
    )
    assert torch.isfinite(loss_empty)
    assert loss_empty.item() == 0.0
    assert pos_rate == 0.0


def test_fast_contact_loss_matches_naive_reference_fwd_and_bwd() -> None:
    """Vectorized gather + single-pass APC + dense chunk BCE match naive reference fwd & bwd."""
    torch.manual_seed(31)
    bsz, n_heads, seq_len, head_dim, k_layers = 3, 4, 64, 8, 2
    d_model = n_heads * head_dim
    # Three segments across rows, including one prefix-cropped (seg_res_len=18, contact_len=14)
    # and forced multi-chunk splitting plus non-contiguous offsets (like _canon_self_check).
    seg_specs = [
        (0, 0, 10, 10),
        (2, 5, 18, 14),
        (1, 10, 20, 20),
    ]
    min_sep = 6
    meta: list[tuple[int, int, int, int, int]] = []
    val_parts: list[torch.Tensor] = []
    offset = 0
    for r, s, seg_l, c_l in seg_specs:
        # Insert a 7-byte gap between segments to also test non-contiguous offsets
        if offset > 0:
            val_parts.append(torch.full((7,), 123, dtype=torch.uint8))
            offset += 7
        n_pairs = (c_l - min_sep) * (c_l - min_sep + 1) // 2
        meta.append((r, s, seg_l, c_l, offset))
        val_parts.append(torch.randint(0, 256, (n_pairs,), dtype=torch.uint8))
        offset += n_pairs
    contact_values = torch.cat(val_parts)

    # --- Reference naive AttentionContactHead + per-segment triu BCE ---
    attn_head = AttentionContactHead(k_layers, n_heads, attn_scales=(0.35, 0.5))
    qks_fast = [
        (
            torch.randn(bsz, n_heads, seq_len, head_dim, requires_grad=True),
            torch.randn(bsz, n_heads, seq_len, head_dim, requires_grad=True),
        )
        for _ in range(k_layers)
    ]
    qks_ref = [
        (q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True))
        for q, k in qks_fast
    ]

    loss_fast, pos_fast = compute_contact_distill_loss(
        head=attn_head,
        qk_states=qks_fast,
        hidden_states=None,
        contact_meta=meta,
        contact_values=contact_values,
        contact_min_sep=min_sep,
        pos_weight=2.5,
        max_chunk_pairs=600,  # forces multiple chunks
    )
    loss_fast.backward()
    fast_w_grad = attn_head.proj.weight.grad.detach().clone()
    fast_b_grad = attn_head.proj.bias.grad.detach().clone()
    fast_qk_grads = [(q.grad.detach().clone(), k.grad.detach().clone()) for q, k in qks_fast]
    attn_head.zero_grad()

    # Naive reference: one segment at a time with explicit softmax, full masked_apc, triu_indices
    ref_losses: list[torch.Tensor] = []
    pos_sum = 0.0
    total_pairs = 0
    for r, s, seg_l, c_l, off in meta:
        tok_l = seg_l + 2
        maps = []
        for (q_all, k_all), sc in zip(qks_ref, attn_head.attn_scales, strict=True):
            q_s = q_all[r : r + 1, :, s : s + tok_l].float()
            k_s = k_all[r : r + 1, :, s : s + tok_l].float()
            att = F.softmax(torch.matmul(q_s, k_s.transpose(-2, -1)) * sc, dim=-1)
            res = att[:, :, 1 : c_l + 1, 1 : c_l + 1]
            sym = 0.5 * (res + res.transpose(-1, -2))
            maps.append(masked_apc(sym, [c_l]))
        stacked = torch.cat(maps, dim=1)
        logits = attn_head.proj(stacked.permute(0, 2, 3, 1)).squeeze(-1)[0]
        ii, jj = torch.triu_indices(c_l, c_l, offset=min_sep)
        n_p = int(ii.numel())
        tgt = contact_values[off : off + n_p].float() / 255.0
        bce = F.binary_cross_entropy_with_logits(logits[ii, jj], tgt, reduction="none")
        bce = bce * (1.0 + (2.5 - 1.0) * tgt)
        ref_losses.append(bce.mean())
        pos_sum += float(tgt.sum().item())
        total_pairs += n_p
    loss_ref = torch.stack(ref_losses).mean()
    loss_ref.backward()

    assert loss_fast.item() == pytest.approx(loss_ref.item(), rel=1e-5, abs=1e-6)
    assert pos_fast == pytest.approx(pos_sum / total_pairs, rel=1e-5, abs=1e-6)
    torch.testing.assert_close(fast_w_grad, attn_head.proj.weight.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fast_b_grad, attn_head.proj.bias.grad, atol=1e-5, rtol=1e-5)
    for (fq, fk), (rq, rk) in zip(fast_qk_grads, qks_ref, strict=True):
        torch.testing.assert_close(fq, rq.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(fk, rk.grad, atol=1e-5, rtol=1e-5)

    # --- Same parity check for PairContactHead ---
    pair_head = PairContactHead(k_layers, d_model, pair_dim=16)
    hs_fast = [torch.randn(bsz, seq_len, d_model, requires_grad=True) for _ in range(k_layers)]
    hs_ref = [h.detach().clone().requires_grad_(True) for h in hs_fast]

    ploss_fast, ppos_fast = compute_contact_distill_loss(
        head=pair_head,
        qk_states=None,
        hidden_states=hs_fast,
        contact_meta=meta,
        contact_values=contact_values,
        contact_min_sep=min_sep,
        pos_weight=2.5,
        max_chunk_pairs=600,
    )
    ploss_fast.backward()
    fast_p_grads = {n: p.grad.detach().clone() for n, p in pair_head.named_parameters()}
    fast_h_grads = [h.grad.detach().clone() for h in hs_fast]
    pair_head.zero_grad()

    pref_losses = []
    for r, s, _seg_l, c_l, off in meta:
        stacked = torch.stack([h[r : r + 1, s + 1 : s + 1 + c_l].float() for h in hs_ref], dim=2)
        proj = pair_head.proj(pair_head.input_norm(stacked))
        w = F.softmax(pair_head.layer_weights.float(), dim=0)
        u = torch.einsum("k,blkc->blc", w, proj)
        ui, uj = u[:, :, None, :], u[:, None, :, :]
        pair = torch.cat([ui * uj, (ui - uj).abs()], dim=-1)
        logits = pair_head.fc2(F.gelu(pair_head.fc1(pair))).squeeze(-1)[0]
        ii, jj = torch.triu_indices(c_l, c_l, offset=min_sep)
        n_p = int(ii.numel())
        tgt = contact_values[off : off + n_p].float() / 255.0
        bce = F.binary_cross_entropy_with_logits(logits[ii, jj], tgt, reduction="none")
        bce = bce * (1.0 + (2.5 - 1.0) * tgt)
        pref_losses.append(bce.mean())
    ploss_ref = torch.stack(pref_losses).mean()
    ploss_ref.backward()

    assert ploss_fast.item() == pytest.approx(ploss_ref.item(), rel=1e-5, abs=1e-6)
    assert ppos_fast == pytest.approx(pos_sum / total_pairs, rel=1e-5, abs=1e-6)
    for n, p in pair_head.named_parameters():
        torch.testing.assert_close(fast_p_grads[n], p.grad, atol=1e-5, rtol=1e-5)
    for fh, rh in zip(fast_h_grads, hs_ref, strict=True):
        torch.testing.assert_close(fh, rh.grad, atol=1e-5, rtol=1e-5)

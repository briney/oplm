"""Auxiliary 2D contact-pair distillation heads, soft-BCE loss, and distributed optimizer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from oplm.training.optim import get_schedule_fn

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from accelerate import Accelerator

    from oplm.config import DistillConfig, TrainConfig

__all__ = [
    "DISTILL_AUX_SIDECAR_NAME",
    "AttentionContactHead",
    "AuxContactManager",
    "PairContactHead",
    "build_contact_head",
    "chunk_contact_segments",
    "compute_contact_distill_loss",
    "masked_apc",
]

DISTILL_AUX_SIDECAR_NAME = "distill_aux.pt"

# (row_idx, start_pos, seg_res_len, contact_len, value_offset)
ContactMeta = tuple[int, int, int, int, int]


class _GatherPackedQK(torch.autograd.Function):
    """Gather packed ``(B, H, S, D)`` tokens at ``(row_ids, pos_ids)`` with 1 scatter backward.

    Replaces per-segment Python slice loops that otherwise spawn ``M * K * 2`` ``SliceBackward0``
    nodes, each allocating and adding a full ``(B, H, S, D)`` tensor during backward.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, row_ids: Tensor, pos_ids: Tensor) -> Tensor:
        ctx.save_for_backward(row_ids, pos_ids)
        ctx.x_shape = x.shape
        # permute(0, 2, 1, 3) is a zero-copy view (B, S, H, D)
        return x.permute(0, 2, 1, 3)[row_ids, pos_ids]  # (N, H, D)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, None, None]:
        (grad_out,) = grad_outputs
        row_ids, pos_ids = ctx.saved_tensors
        grad_x = grad_out.new_zeros(ctx.x_shape)
        grad_x.permute(0, 2, 1, 3).index_put_((row_ids, pos_ids), grad_out, accumulate=True)
        return grad_x, None, None


class _GatherPackedHidden(torch.autograd.Function):
    """Gather packed ``(B, S, D)`` tokens at ``(row_ids, pos_ids)`` with 1 scatter backward."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, row_ids: Tensor, pos_ids: Tensor) -> Tensor:
        ctx.save_for_backward(row_ids, pos_ids)
        ctx.x_shape = x.shape
        return x[row_ids, pos_ids]  # (N, D)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, None, None]:
        (grad_out,) = grad_outputs
        row_ids, pos_ids = ctx.saved_tensors
        grad_x = grad_out.new_zeros(ctx.x_shape)
        grad_x.index_put_((row_ids, pos_ids), grad_out, accumulate=True)
        return grad_x, None, None


@dataclass(frozen=True)
class _ChunkGatherPlan:
    """Precomputed GPU tensors for one length-sorted contact chunk."""

    c_size: int
    t_max: int
    l_max: int
    tok_start: int
    tok_end: int
    dst_idx: Tensor  # (N_chunk,) long into flattened (C * T_max,) or (C * L_max,)
    key_valid: Tensor | None  # (C, 1, 1, T_max) bool (attention head only)
    res_pair_mask: Tensor  # (C, 1, L_max, L_max) or (C, L_max, L_max) float32
    inv_lens: Tensor | None  # (C, 1, 1, 1) float32 (attention head only)
    c_lens_t: Tensor  # (C,) long


def _build_gather_plan(
    chunks: Sequence[Sequence[ContactMeta]],
    *,
    include_special: bool,
    device: torch.device,
) -> tuple[Tensor, Tensor, list[_ChunkGatherPlan]]:
    """Build one flat ``(row_ids, pos_ids)`` gather index across all chunks + per-chunk plans."""
    all_rows: list[np.ndarray] = []
    all_pos: list[np.ndarray] = []
    dst_arrays: list[np.ndarray] = []
    chunk_meta_info: list[tuple[int, int, int, int, int, list[int], list[int]]] = []
    tok_cursor = 0

    for chunk in chunks:
        c_size = len(chunk)
        tok_lens = [m[2] + 2 if include_special else m[3] for m in chunk]
        c_lens = [m[3] for m in chunk]
        t_max = max(tok_lens)
        l_max = max(c_lens)
        chunk_rows: list[np.ndarray] = []
        chunk_pos: list[np.ndarray] = []
        chunk_dst: list[np.ndarray] = []
        for idx, ((row_idx, start_pos, _, _, _), length) in enumerate(
            zip(chunk, tok_lens, strict=True)
        ):
            base_pos = start_pos if include_special else start_pos + 1
            offsets = np.arange(length, dtype=np.int64)
            chunk_rows.append(np.full(length, row_idx, dtype=np.int64))
            chunk_pos.append(base_pos + offsets)
            chunk_dst.append(idx * t_max + offsets)
        n_chunk = sum(tok_lens)
        all_rows.append(np.concatenate(chunk_rows))
        all_pos.append(np.concatenate(chunk_pos))
        dst_arrays.append(np.concatenate(chunk_dst))
        chunk_meta_info.append(
            (c_size, t_max, l_max, tok_cursor, tok_cursor + n_chunk, tok_lens, c_lens)
        )
        tok_cursor += n_chunk

    row_ids = torch.from_numpy(np.concatenate(all_rows)).to(device, non_blocking=True)
    pos_ids = torch.from_numpy(np.concatenate(all_pos)).to(device, non_blocking=True)

    plans: list[_ChunkGatherPlan] = []
    for (c_size, t_max, l_max, tok_start, tok_end, tok_lens, c_lens), dst_np in zip(
        chunk_meta_info, dst_arrays, strict=True
    ):
        dst_idx = torch.from_numpy(dst_np).to(device, non_blocking=True)
        c_lens_t = torch.tensor(c_lens, dtype=torch.long, device=device)
        res_pos = torch.arange(l_max, device=device)
        res_valid = res_pos[None, :] < c_lens_t[:, None]  # (C, L_max)
        if include_special:
            tok_pos = torch.arange(t_max, device=device)
            tok_lens_t = torch.tensor(tok_lens, dtype=torch.long, device=device)
            key_valid: Tensor | None = (tok_pos[None, :] < tok_lens_t[:, None])[:, None, None, :]
            res_pair_mask = (res_valid[:, None, :, None] & res_valid[:, None, None, :]).float()
            inv_lens: Tensor | None = (1.0 / c_lens_t.float()).view(c_size, 1, 1, 1)
        else:
            key_valid = None
            res_pair_mask = (res_valid[:, :, None] & res_valid[:, None, :]).float()
            inv_lens = None
        plans.append(
            _ChunkGatherPlan(
                c_size=c_size,
                t_max=t_max,
                l_max=l_max,
                tok_start=tok_start,
                tok_end=tok_end,
                dst_idx=dst_idx,
                key_valid=key_valid,
                res_pair_mask=res_pair_mask,
                inv_lens=inv_lens,
                c_lens_t=c_lens_t,
            )
        )
    return row_ids, pos_ids, plans


def masked_apc(mat: Tensor, lengths: Sequence[int]) -> Tensor:
    """Apply per-segment Average Product Correction to a padded ``(C, H, L_max, L_max)`` tensor.

    ``mat`` must already be zeroed outside each batch item's ``lengths[b] x lengths[b]`` square.
    Matches :func:`oplm.eval.metrics.contact.apply_apc` on each segment's valid region and keeps
    padding strictly zero.
    """
    bsz, _, l_max, _ = mat.shape
    device = mat.device
    lens = torch.tensor(lengths, dtype=torch.float32, device=device).view(bsz, 1, 1, 1)
    pos = torch.arange(l_max, device=device)
    valid_1d = pos[None, :] < torch.tensor(lengths, device=device)[:, None]  # (C, L_max)
    pair_mask = valid_1d[:, None, :, None] & valid_1d[:, None, None, :]  # (C, 1, L_max, L_max)

    row_mean = mat.sum(dim=-1, keepdim=True) / lens
    col_mean = mat.sum(dim=-2, keepdim=True) / lens
    global_mean = mat.sum(dim=(-2, -1), keepdim=True) / (lens * lens)

    near_zero = global_mean.abs() < 1e-8
    safe_global = torch.where(near_zero, torch.ones_like(global_mean), global_mean)
    corrected = torch.where(near_zero, mat, mat - (row_mean * col_mean) / safe_global)
    return corrected * pair_mask.to(corrected.dtype)


def _attn_layer_logits(
    q_chunk: Tensor,
    k_chunk: Tensor,
    scale: float,
    key_valid: Tensor,
    res_pair_mask: Tensor,
    inv_lens: Tensor,
    w_l: Tensor,
) -> Tensor:
    """Compute one layer's weighted APC-corrected contact logits ``(C, L_max, L_max)`` in fp32.

    Runs ``Q @ K^T`` in native dtype (BF16 Tensor Cores under bf16 training), then performs
    softmax, single-pass symmetric APC (reusing ``row_mean`` for both ``col_mean`` and
    ``global_mean``), and on-the-fly head projection in fp32 without materializing a
    ``(C, K*H, L_max, L_max)`` tensor.
    """
    scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)).float() * scale
    scores = scores.masked_fill(~key_valid, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    l_max = res_pair_mask.shape[-1]
    res_attn = attn[:, :, 1 : l_max + 1, 1 : l_max + 1] * res_pair_mask
    sym = 0.5 * (res_attn + res_attn.transpose(-1, -2))
    # For symmetric matrices zeroed outside L_c x L_c: col_mean == row_mean^T and
    # global_mean == (1/L_c) * sum_i row_mean_i (one O(L^2) reduction instead of three).
    row_mean = sym.sum(dim=-1, keepdim=True) * inv_lens
    global_mean = row_mean.sum(dim=-2, keepdim=True) * inv_lens
    near_zero = global_mean.abs() < 1e-8
    safe_global = torch.where(near_zero, torch.ones_like(global_mean), global_mean)
    apc = (
        torch.where(near_zero, sym, sym - (row_mean * row_mean.transpose(-1, -2)) / safe_global)
        * res_pair_mask
    )
    return (apc * w_l.float().view(1, -1, 1, 1)).sum(dim=1)


def _pair_mlp_chunk(
    u: Tensor,
    fc1_w: Tensor,
    fc1_b: Tensor,
    fc2_w: Tensor,
    fc2_b: Tensor,
    pair_mask: Tensor,
) -> Tensor:
    """Compute symmetric outer-interaction MLP contact logits ``(C, L_max, L_max)`` in fp32."""
    ui = u[:, :, None, :]
    uj = u[:, None, :, :]
    pair = torch.cat([ui * uj, (ui - uj).abs()], dim=-1)
    hidden = F.gelu(F.linear(pair, fc1_w, fc1_b))
    logits = F.linear(hidden, fc2_w, fc2_b).squeeze(-1)
    return logits.float() * pair_mask


def _dense_chunk_bce(
    pred_logits: Tensor,
    chunk_vals: Tensor,
    c_lens_t: Tensor,
    rel_offsets: Tensor,
    inv_n_pairs: Tensor,
    contact_min_sep: int,
    pos_weight: float,
) -> tuple[Tensor, Tensor]:
    """Zero-sync vectorized soft BCE over all upper-triangle pairs ``j - i >= min_sep`` in a chunk.

    Maps each valid ``(c, i, j)`` directly to its flat upper-triangle byte offset in ``chunk_vals``
    via the closed-form triangular index formula, avoiding per-segment ``torch.triu_indices``,
    boolean-index CUDA syncs, and backward scatter-adds.
    """
    l_max = pred_logits.shape[-1]
    pos = torch.arange(l_max, device=pred_logits.device)
    valid_1d = pos[None, :] < c_lens_t[:, None]  # (C, L_max)
    sep_mask = (pos[None, :] - pos[:, None]) >= contact_min_sep  # (L_max, L_max)
    triu_mask = sep_mask[None, :, :] & valid_1d[:, :, None] & valid_1d[:, None, :]

    i = pos[None, :, None]
    j = pos[None, None, :]
    span = (c_lens_t - contact_min_sep)[:, None, None]
    flat_idx = (
        rel_offsets[:, None, None] + i * span - (i * (i - 1)) // 2 + (j - i - contact_min_sep)
    )
    safe_idx = torch.where(triu_mask, flat_idx, 0)
    target_p = chunk_vals[safe_idx].float() * (1.0 / 255.0)

    bce = F.binary_cross_entropy_with_logits(pred_logits, target_p, reduction="none")
    if pos_weight != 1.0:
        bce = bce * (1.0 + (pos_weight - 1.0) * target_p)
    mask_f = triu_mask.float()
    chunk_loss = (bce * (mask_f * inv_n_pairs[:, None, None])).sum()
    chunk_pos_sum = (target_p.detach() * mask_f).sum()
    return chunk_loss, chunk_pos_sum


def chunk_contact_segments(
    meta: Sequence[ContactMeta],
    max_chunk_pairs: int,
) -> list[list[ContactMeta]]:
    """Group length-sorted segments into chunks bounded by ``len(chunk) * (max_L + 2)^2``."""
    chunks: list[list[ContactMeta]] = []
    cur: list[ContactMeta] = []
    cur_max_tok = 0
    for item in meta:
        seg_tok = item[2] + 2
        next_max_tok = max(cur_max_tok, seg_tok)
        next_cost = (len(cur) + 1) * next_max_tok * next_max_tok
        if cur and next_cost > max_chunk_pairs:
            chunks.append(cur)
            cur = [item]
            cur_max_tok = seg_tok
        else:
            cur.append(item)
            cur_max_tok = next_max_tok
    if cur:
        chunks.append(cur)
    return chunks


class AttentionContactHead(nn.Module):
    """Discarded ESM-style logistic regression head over upper-layer post-RoPE ``(Q, K)`` maps."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        attn_scales: Sequence[float],
    ) -> None:
        super().__init__()
        if len(attn_scales) != num_layers:
            raise ValueError(
                f"len(attn_scales) ({len(attn_scales)}) must match num_layers ({num_layers})"
            )
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attn_scales = tuple(float(s) for s in attn_scales)
        self.proj = nn.Linear(num_layers * num_heads, 1, bias=True)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=0.01, a=-0.02, b=0.02)
        nn.init.constant_(self.proj.bias, -3.0)
        self._layer_fn: Callable[..., Tensor] = _attn_layer_logits

    def compile_kernels(self) -> None:
        """Compile the per-layer softmax/APC/projection kernel with dynamic shapes."""
        self._layer_fn = torch.compile(_attn_layer_logits, dynamic=True)

    def forward_packed_chunk(
        self,
        q_packed_layers: Sequence[Tensor],
        k_packed_layers: Sequence[Tensor],
        plan: _ChunkGatherPlan,
    ) -> Tensor:
        """Compute symmetric ``(C, L_max, L_max)`` fp32 logits from gathered ``(N, H, D)`` QK."""
        if len(q_packed_layers) != self.num_layers or len(k_packed_layers) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} QK layer pairs, got {len(q_packed_layers)}"
            )
        assert plan.key_valid is not None and plan.inv_lens is not None
        weight_by_layer = self.proj.weight.view(self.num_layers, self.num_heads)
        n_heads = q_packed_layers[0].shape[1]
        head_dim = q_packed_layers[0].shape[2]
        logits: Tensor | None = None

        for layer_idx, (q_packed, k_packed, scale) in enumerate(
            zip(q_packed_layers, k_packed_layers, self.attn_scales, strict=True)
        ):
            q_buf = q_packed.new_zeros((plan.c_size * plan.t_max, n_heads, head_dim))
            k_buf = k_packed.new_zeros((plan.c_size * plan.t_max, n_heads, head_dim))
            q_buf.index_copy_(0, plan.dst_idx, q_packed[plan.tok_start : plan.tok_end])
            k_buf.index_copy_(0, plan.dst_idx, k_packed[plan.tok_start : plan.tok_end])
            q_chunk = q_buf.view(plan.c_size, plan.t_max, n_heads, head_dim).permute(0, 2, 1, 3)
            k_chunk = k_buf.view(plan.c_size, plan.t_max, n_heads, head_dim).permute(0, 2, 1, 3)
            layer_out = self._layer_fn(
                q_chunk,
                k_chunk,
                scale,
                plan.key_valid,
                plan.res_pair_mask,
                plan.inv_lens,
                weight_by_layer[layer_idx],
            )
            logits = layer_out if logits is None else logits + layer_out

        assert logits is not None
        return logits + self.proj.bias.float()

    def forward_chunk(
        self,
        qk_states: Sequence[tuple[Tensor, Tensor]],
        chunk_meta: Sequence[ContactMeta],
    ) -> Tensor:
        """Compute symmetric ``(C, L_max, L_max)`` fp32 contact logits for one segment chunk."""
        if len(qk_states) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} QK layer pairs, got {len(qk_states)}")
        device = qk_states[0][0].device
        row_ids, pos_ids, plans = _build_gather_plan(
            [chunk_meta], include_special=True, device=device
        )
        q_packed = [_GatherPackedQK.apply(q_all, row_ids, pos_ids) for q_all, _ in qk_states]
        k_packed = [_GatherPackedQK.apply(k_all, row_ids, pos_ids) for _, k_all in qk_states]
        return self.forward_packed_chunk(q_packed, k_packed, plans[0])


class PairContactHead(nn.Module):
    """Discarded symmetric outer-interaction MLP over a learned mix of top-K hidden states."""

    def __init__(self, num_layers: int, hidden_size: int, pair_dim: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.pair_dim = pair_dim
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.input_norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, pair_dim, bias=False)
        self.fc1 = nn.Linear(2 * pair_dim, pair_dim, bias=True)
        self.fc2 = nn.Linear(pair_dim, 1, bias=True)
        nn.init.constant_(self.fc2.bias, -3.0)
        self._pair_fn: Callable[..., Tensor] = _pair_mlp_chunk

    def compile_kernels(self) -> None:
        """Compile the outer-interaction MLP chunk kernel with dynamic shapes."""
        self._pair_fn = torch.compile(_pair_mlp_chunk, dynamic=True)

    def project_packed_tokens(self, h_packed_layers: Sequence[Tensor]) -> Tensor:
        """Project and layer-mix valid residue tokens ``(N, D)`` -> ``(N, pair_dim)``."""
        if len(h_packed_layers) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} hidden states, got {len(h_packed_layers)}"
            )
        stacked = torch.stack([h.float() for h in h_packed_layers], dim=1)  # (N, K, D)
        proj = self.proj(self.input_norm(stacked))  # (N, K, d_p)
        weights = F.softmax(self.layer_weights.float(), dim=0)
        return torch.einsum("k,nkc->nc", weights, proj)  # (N, d_p)

    def forward_packed_chunk(self, u_packed: Tensor, plan: _ChunkGatherPlan) -> Tensor:
        """Compute symmetric ``(C, L_max, L_max)`` fp32 logits from pre-projected ``(N, d_p)``."""
        u_buf = u_packed.new_zeros((plan.c_size * plan.l_max, self.pair_dim))
        u_buf.index_copy_(0, plan.dst_idx, u_packed[plan.tok_start : plan.tok_end])
        u_chunk = u_buf.view(plan.c_size, plan.l_max, self.pair_dim)
        return self._pair_fn(
            u_chunk,
            self.fc1.weight,
            self.fc1.bias,
            self.fc2.weight,
            self.fc2.bias,
            plan.res_pair_mask,
        )

    def forward_chunk(
        self,
        hidden_states: Sequence[Tensor],
        chunk_meta: Sequence[ContactMeta],
    ) -> Tensor:
        """Compute symmetric ``(C, L_max, L_max)`` fp32 contact logits for one segment chunk."""
        if len(hidden_states) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} hidden states, got {len(hidden_states)}")
        device = hidden_states[0].device
        row_ids, pos_ids, plans = _build_gather_plan(
            [chunk_meta], include_special=False, device=device
        )
        h_packed = [_GatherPackedHidden.apply(h_all, row_ids, pos_ids) for h_all in hidden_states]
        u_packed = self.project_packed_tokens(h_packed)
        return self.forward_packed_chunk(u_packed, plans[0])


def build_contact_head(
    distill_cfg: DistillConfig,
    model: nn.Module,
) -> tuple[AttentionContactHead | PairContactHead, tuple[int, ...] | None]:
    """Instantiate the configured 2D contact head and resolve its tapped execution indices.

    Returns ``(head_module, qk_layers_or_None)`` where ``qk_layers`` is a tuple of 0-based
    execution-step indices into ``backbone.layer_execution_order`` for ``"attention"`` and
    ``None`` for ``"pair"``.
    """
    lm: Any = model
    backbone: Any = lm.oplm.backbone
    exec_order: list[int] = list(backbone.layer_execution_order)
    effective_depth = len(exec_order)
    k = distill_cfg.contact_layers
    if k > effective_depth:
        raise ValueError(
            f"distill.contact_layers ({k}) exceeds model effective depth ({effective_depth})"
        )
    if distill_cfg.contact_head == "attention":
        qk_layers = tuple(range(effective_depth - k, effective_depth))
        attn_scales = tuple(
            float(backbone.layers[exec_order[idx]].attention.attn_scale) for idx in qk_layers
        )
        num_heads = int(lm.config.num_attention_heads)
        return AttentionContactHead(k, num_heads, attn_scales), qk_layers
    if distill_cfg.contact_head == "pair":
        hidden_size = int(lm.config.hidden_size)
        return PairContactHead(k, hidden_size, distill_cfg.contact_pair_dim), None
    raise ValueError(f"Unsupported distill.contact_head: {distill_cfg.contact_head!r}")


def _slice_chunk_values(
    chunk: Sequence[ContactMeta],
    contact_values: Tensor,
    contact_min_sep: int,
) -> tuple[Tensor, list[int], list[int]]:
    """Return ``(chunk_vals, rel_offsets, n_pairs_list)`` with zero-copy contiguous fast path."""
    n_pairs_list = [(m[3] - contact_min_sep) * (m[3] - contact_min_sep + 1) // 2 for m in chunk]
    base0 = chunk[0][4]
    expected = base0
    contiguous = True
    rel_offsets: list[int] = []
    rel = 0
    for m, n_p in zip(chunk, n_pairs_list, strict=True):
        if m[4] != expected:
            contiguous = False
        rel_offsets.append(rel)
        rel += n_p
        expected += n_p
    if contiguous:
        return contact_values[base0 : base0 + rel], rel_offsets, n_pairs_list
    parts = [contact_values[m[4] : m[4] + n_p] for m, n_p in zip(chunk, n_pairs_list, strict=True)]
    return torch.cat(parts, dim=0), rel_offsets, n_pairs_list


def compute_contact_distill_loss(
    *,
    head: Any,
    qk_states: Sequence[tuple[Tensor, Tensor]] | None,
    hidden_states: Sequence[Tensor] | None,
    contact_meta: Sequence[ContactMeta],
    contact_values: Tensor,
    contact_min_sep: int,
    pos_weight: float,
    max_chunk_pairs: int,
    bce_fn: Callable[..., tuple[Tensor, Tensor]] = _dense_chunk_bce,
) -> tuple[Tensor, float]:
    """Compute per-sequence averaged soft BCE over distilled segments in length-sorted chunks."""
    valid_meta = [m for m in contact_meta if m[3] > contact_min_sep]
    if not valid_meta:
        ref = (
            qk_states[0][0]
            if qk_states
            else (hidden_states[0] if hidden_states else contact_values)
        )
        return ref.new_zeros((), dtype=torch.float32), 0.0

    chunks = chunk_contact_segments(valid_meta, max_chunk_pairs)
    device = contact_values.device
    total_segs = float(len(valid_meta))
    total_loss: Tensor | None = None
    pos_sum_t = contact_values.new_zeros((), dtype=torch.float32)
    total_pairs = 0

    if isinstance(head, AttentionContactHead):
        assert qk_states is not None
        row_ids, pos_ids, plans = _build_gather_plan(chunks, include_special=True, device=device)
        q_packed: Any = [_GatherPackedQK.apply(q_all, row_ids, pos_ids) for q_all, _ in qk_states]
        k_packed: Any = [_GatherPackedQK.apply(k_all, row_ids, pos_ids) for _, k_all in qk_states]
        for chunk, plan in zip(chunks, plans, strict=True):
            pred_logits = head.forward_packed_chunk(q_packed, k_packed, plan)
            chunk_vals, rel_offsets, n_pairs_list = _slice_chunk_values(
                chunk, contact_values, contact_min_sep
            )
            rel_t = torch.tensor(rel_offsets, dtype=torch.long, device=device)
            inv_w = torch.tensor(
                [1.0 / (total_segs * n_p) for n_p in n_pairs_list],
                dtype=torch.float32,
                device=device,
            )
            c_loss, c_pos = bce_fn(
                pred_logits, chunk_vals, plan.c_lens_t, rel_t, inv_w, contact_min_sep, pos_weight
            )
            total_loss = c_loss if total_loss is None else total_loss + c_loss
            pos_sum_t = pos_sum_t + c_pos
            total_pairs += sum(n_pairs_list)
    elif isinstance(head, PairContactHead):
        assert hidden_states is not None
        row_ids, pos_ids, plans = _build_gather_plan(chunks, include_special=False, device=device)
        h_packed: Any = [
            _GatherPackedHidden.apply(h_all, row_ids, pos_ids) for h_all in hidden_states
        ]
        u_packed = head.project_packed_tokens(h_packed)
        for chunk, plan in zip(chunks, plans, strict=True):
            pred_logits = head.forward_packed_chunk(u_packed, plan)
            chunk_vals, rel_offsets, n_pairs_list = _slice_chunk_values(
                chunk, contact_values, contact_min_sep
            )
            rel_t = torch.tensor(rel_offsets, dtype=torch.long, device=device)
            inv_w = torch.tensor(
                [1.0 / (total_segs * n_p) for n_p in n_pairs_list],
                dtype=torch.float32,
                device=device,
            )
            c_loss, c_pos = bce_fn(
                pred_logits, chunk_vals, plan.c_lens_t, rel_t, inv_w, contact_min_sep, pos_weight
            )
            total_loss = c_loss if total_loss is None else total_loss + c_loss
            pos_sum_t = pos_sum_t + c_pos
            total_pairs += sum(n_pairs_list)
    else:
        head_inputs: Any = qk_states if qk_states is not None else hidden_states
        assert head_inputs is not None
        for chunk in chunks:
            pred_logits = head.forward_chunk(head_inputs, chunk)
            chunk_vals, rel_offsets, n_pairs_list = _slice_chunk_values(
                chunk, contact_values, contact_min_sep
            )
            c_lens_t = torch.tensor([m[3] for m in chunk], dtype=torch.long, device=device)
            rel_t = torch.tensor(rel_offsets, dtype=torch.long, device=device)
            inv_w = torch.tensor(
                [1.0 / (total_segs * n_p) for n_p in n_pairs_list],
                dtype=torch.float32,
                device=device,
            )
            c_loss, c_pos = bce_fn(
                pred_logits, chunk_vals, c_lens_t, rel_t, inv_w, contact_min_sep, pos_weight
            )
            total_loss = c_loss if total_loss is None else total_loss + c_loss
            pos_sum_t = pos_sum_t + c_pos
            total_pairs += sum(n_pairs_list)

    assert total_loss is not None
    mean_pos_rate = float(pos_sum_t.item()) / max(1, total_pairs)
    return total_loss, mean_pos_rate


class AuxContactManager:
    """Owns the discarded 2D contact head, its standalone AdamW optimizer, and branch-safe LR."""

    def __init__(
        self,
        head: nn.Module,
        distill_cfg: DistillConfig,
        train_cfg: TrainConfig,
        total_steps: int,
    ) -> None:
        self.head = head
        self.head_type = distill_cfg.contact_head
        self._base_lr = float(distill_cfg.contact_head_lr)
        self._warmup_steps = int(distill_cfg.contact_head_warmup_steps)
        self._weight_decay = 0.01
        min_ratio = train_cfg.min_lr / train_cfg.lr if train_cfg.lr > 0 else 0.0
        self._schedule_fn = get_schedule_fn(
            scheduler_name=train_cfg.scheduler,
            warmup_steps=train_cfg.warmup_steps,
            total_steps=total_steps,
            min_ratio=min_ratio,
            stable_steps=train_cfg.stable_steps,
        )
        self.aux_start_step: int = 0
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=self._base_lr,
            betas=(0.9, 0.98),
            weight_decay=self._weight_decay,
        )
        self.bce_fn: Callable[..., tuple[Tensor, Tensor]] = _dense_chunk_bce
        first_param = next(self.head.parameters(), None)
        if train_cfg.compile and first_param is not None and first_param.device.type == "cuda":
            compile_kernels = getattr(self.head, "compile_kernels", None)
            if callable(compile_kernels):
                compile_kernels()
            self.bce_fn = torch.compile(_dense_chunk_bce, dynamic=True)

    def broadcast_weights(self) -> None:
        """Broadcast rank-0 head weights so every rank starts bit-for-bit identical."""
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            for p in self.head.parameters():
                dist.broadcast(p.data, src=0)

    def lr_for_step(self, step: int) -> float:
        """Live LR: ``contact_head_lr * backbone_schedule_lambda(step) * warmup_ramp(step)``."""
        backbone_mult = float(self._schedule_fn(step))
        if self._warmup_steps == 0:
            ramp = 1.0
        else:
            elapsed = max(0, step - self.aux_start_step + 1)
            ramp = min(1.0, elapsed / self._warmup_steps)
        lr = self._base_lr * backbone_mult * ramp
        return lr if math.isfinite(lr) else 0.0

    @property
    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _apply_lr(self, step: int) -> None:
        lr = self.lr_for_step(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def sync_schedule(self, global_step: int, *, fresh_start: bool) -> None:
        """Align the auxiliary LR schedule after init, resume, or branch_from."""
        if fresh_start:
            self.aux_start_step = int(global_step)
        self._apply_lr(global_step)

    def step(self, accelerator: Accelerator, max_grad_norm: float, global_step: int) -> None:
        """All-reduce aux grads across ranks, clip, step AdamW, zero grads, and advance LR."""
        if accelerator.optimizer_step_was_skipped:
            self.optimizer.zero_grad(set_to_none=True)
            self._apply_lr(global_step + 1)
            return

        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        for p in self.head.parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            if distributed:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), max_grad_norm)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._apply_lr(global_step + 1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "head_type": self.head_type,
            "aux_start_step": self.aux_start_step,
            "head": {k: v.detach().cpu() for k, v in self.head.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any], *, reset_hparams: bool = False) -> None:
        saved_type = state.get("head_type")
        if saved_type != self.head_type:
            raise ValueError(
                f"distill_aux.pt head_type {saved_type!r} does not match configured "
                f"distill.contact_head {self.head_type!r}"
            )
        saved_head = state.get("head")
        if not isinstance(saved_head, dict):
            raise ValueError("distill_aux.pt is missing a valid 'head' state dict")
        expected_head = self.head.state_dict()
        if set(saved_head.keys()) != set(expected_head.keys()):
            raise ValueError(
                f"distill_aux.pt head keys {sorted(saved_head.keys())} do not match "
                f"expected keys {sorted(expected_head.keys())}"
            )
        for key, expected_tensor in expected_head.items():
            saved_tensor = saved_head[key]
            if not isinstance(saved_tensor, Tensor) or tuple(saved_tensor.shape) != tuple(
                expected_tensor.shape
            ):
                raise ValueError(
                    f"distill_aux.pt parameter {key!r} has incompatible shape "
                    f"{getattr(saved_tensor, 'shape', type(saved_tensor).__name__)} "
                    f"(expected {tuple(expected_tensor.shape)})"
                )
        head_backup = {k: v.detach().clone() for k, v in expected_head.items()}
        try:
            self.head.load_state_dict(saved_head)
            self.optimizer.load_state_dict(state["optimizer"])
        except Exception:
            self.head.load_state_dict(head_backup)
            raise
        self.aux_start_step = int(state.get("aux_start_step", 0))
        if reset_hparams:
            for group in self.optimizer.param_groups:
                group["lr"] = self._base_lr
                group["weight_decay"] = self._weight_decay

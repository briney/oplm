"""Auxiliary 2D contact-pair distillation heads, soft-BCE loss, and distributed optimizer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from oplm.training.optim import get_schedule_fn

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    def forward_chunk(
        self,
        qk_states: Sequence[tuple[Tensor, Tensor]],
        chunk_meta: Sequence[ContactMeta],
    ) -> Tensor:
        """Compute symmetric ``(C, L_max, L_max)`` fp32 contact logits for one segment chunk."""
        if len(qk_states) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} QK layer pairs, got {len(qk_states)}")
        c_size = len(chunk_meta)
        t_max = max(m[2] for m in chunk_meta) + 2
        l_max = max(m[3] for m in chunk_meta)
        device = qk_states[0][0].device
        contact_lens = [m[3] for m in chunk_meta]

        tok_pos = torch.arange(t_max, device=device)
        tok_lens = torch.tensor([m[2] + 2 for m in chunk_meta], device=device)
        key_valid = (tok_pos[None, :] < tok_lens[:, None])[:, None, None, :]  # (C, 1, 1, T_max)

        res_pos = torch.arange(l_max, device=device)
        c_lens_t = torch.tensor(contact_lens, device=device)
        res_valid = res_pos[None, :] < c_lens_t[:, None]  # (C, L_max)
        res_pair_mask = (res_valid[:, None, :, None] & res_valid[:, None, None, :]).float()

        head_maps: list[Tensor] = []
        for (q_all, k_all), scale in zip(qk_states, self.attn_scales, strict=True):
            _, n_heads, _, head_dim = q_all.shape
            q_chunk = q_all.new_zeros((c_size, n_heads, t_max, head_dim))
            k_chunk = k_all.new_zeros((c_size, n_heads, t_max, head_dim))
            for idx, (row_idx, start_pos, seg_res_len, _, _) in enumerate(chunk_meta):
                tok_len = seg_res_len + 2
                q_chunk[idx, :, :tok_len] = q_all[row_idx, :, start_pos : start_pos + tok_len]
                k_chunk[idx, :, :tok_len] = k_all[row_idx, :, start_pos : start_pos + tok_len]
            scores = torch.matmul(q_chunk.float(), k_chunk.float().transpose(-2, -1)) * scale
            scores = scores.masked_fill(~key_valid, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            # Slice residue-by-residue contact submatrix [1 : L_max + 1, 1 : L_max + 1]
            res_attn = attn[:, :, 1 : l_max + 1, 1 : l_max + 1] * res_pair_mask
            sym_attn = 0.5 * (res_attn + res_attn.transpose(-1, -2))
            head_maps.append(masked_apc(sym_attn, contact_lens))

        stacked = torch.cat(head_maps, dim=1)  # (C, K*H, L_max, L_max)
        logits = self.proj(stacked.permute(0, 2, 3, 1)).squeeze(-1)  # (C, L_max, L_max)
        return logits


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

    def forward_chunk(
        self,
        hidden_states: Sequence[Tensor],
        chunk_meta: Sequence[ContactMeta],
    ) -> Tensor:
        """Compute symmetric ``(C, L_max, L_max)`` fp32 contact logits for one segment chunk."""
        if len(hidden_states) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} hidden states, got {len(hidden_states)}")
        c_size = len(chunk_meta)
        l_max = max(m[3] for m in chunk_meta)
        device = hidden_states[0].device
        d_model = hidden_states[0].shape[-1]

        gathered_layers: list[Tensor] = []
        for h_all in hidden_states:
            h_chunk = h_all.new_zeros((c_size, l_max, d_model))
            for idx, (row_idx, start_pos, _, contact_len, _) in enumerate(chunk_meta):
                h_chunk[idx, :contact_len] = h_all[
                    row_idx, start_pos + 1 : start_pos + 1 + contact_len
                ]
            gathered_layers.append(h_chunk.float())

        stacked = torch.stack(gathered_layers, dim=2)  # (C, L_max, K, D)
        proj = self.proj(self.input_norm(stacked))  # (C, L_max, K, d_p)
        weights = F.softmax(self.layer_weights.float(), dim=0)
        u = torch.einsum("k,blkc->blc", weights, proj)  # (C, L_max, d_p)

        ui = u[:, :, None, :]
        uj = u[:, None, :, :]
        pair = torch.cat([ui * uj, (ui - uj).abs()], dim=-1)  # (C, L_max, L_max, 2*d_p)
        logits = self.fc2(F.gelu(self.fc1(pair))).squeeze(-1)  # (C, L_max, L_max)
        # Mask out padded residue rows/cols cleanly
        c_lens_t = torch.tensor([m[3] for m in chunk_meta], device=device)
        valid = torch.arange(l_max, device=device)[None, :] < c_lens_t[:, None]
        pair_mask = (valid[:, :, None] & valid[:, None, :]).float()
        return logits * pair_mask


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
) -> tuple[Tensor, float]:
    """Compute per-sequence averaged soft BCE over distilled segments in length-sorted chunks."""
    if not contact_meta:
        ref = (
            qk_states[0][0]
            if qk_states
            else (hidden_states[0] if hidden_states else contact_values)
        )
        return ref.new_zeros((), dtype=torch.float32), 0.0

    chunks = chunk_contact_segments(contact_meta, max_chunk_pairs)
    device = contact_values.device
    seq_losses: list[Tensor] = []
    pos_sum = 0.0
    total_pairs = 0
    triu_cache: dict[int, tuple[Tensor, Tensor]] = {}

    head_inputs: Any = qk_states if qk_states is not None else hidden_states
    assert head_inputs is not None

    for chunk in chunks:
        pred_logits: Tensor = head.forward_chunk(head_inputs, chunk)
        for idx, (_, _, _, contact_len, offset) in enumerate(chunk):
            if contact_len not in triu_cache:
                ii, jj = torch.triu_indices(
                    contact_len, contact_len, offset=contact_min_sep, device=device
                )
                triu_cache[contact_len] = (ii, jj)
            else:
                ii, jj = triu_cache[contact_len]
            n_pairs = int(ii.numel())
            pred_triu = pred_logits[idx, ii, jj]
            target_p = contact_values[offset : offset + n_pairs].float() / 255.0
            bce = F.binary_cross_entropy_with_logits(pred_triu, target_p, reduction="none")
            if pos_weight != 1.0:
                bce = bce * (1.0 + (pos_weight - 1.0) * target_p)
            seq_losses.append(bce.mean())
            pos_sum += float(target_p.detach().sum().item())
            total_pairs += n_pairs

    loss = torch.stack(seq_losses).mean()
    mean_pos_rate = pos_sum / max(1, total_pairs)
    return loss, mean_pos_rate


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
        self.head.load_state_dict(state["head"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.aux_start_step = int(state.get("aux_start_step", 0))
        if reset_hparams:
            for group in self.optimizer.param_groups:
                group["lr"] = self._base_lr
                group["weight_decay"] = self._weight_decay

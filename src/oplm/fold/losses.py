"""Training losses for the structure head (spec §6.2; addendum §5), checked against Boltz-2.

Every function runs in fp32 with any outer autocast disabled and returns one value per leading
row so the model can weight and reduce; masks are explicit and an empty mask gives a finite zero.
The conventions, recorded here because the oracle code (tests/fold/oracles/) encodes them
implicitly:

- Diffusion MSE: the reference is aligned onto the prediction (AF3 algorithm 28), detached, then
  ``Σ w |x − x_aligned|² / (3 Σ w)``; the EDM weight is ``(σ² + σ_d²) / (σ σ_d)²``.
- Smooth lDDT: pairs with reference distance below ``cutoff`` (15 Å, protein), both atoms masked
  in, ``i ≠ j``; ``1 − mean(¼ Σ_{t ∈ {0.5, 1, 2, 4}} sigmoid(t − |Δd|))``.
- Distogram: representative-atom distances bucketized with ``bins − 1`` boundaries evenly spaced
  on ``[min_dist, max_dist]``; masked mean cross-entropy over pairs whose representative atoms are
  both resolved, ``i ≠ j``.
- pLDDT target: per-atom lDDT (thresholds 0.5, 1, 2, 4 Å) against the representative atoms within
  ``cutoff`` in the reference, the atom's own representative excluded; ``floor(lddt · bins)``
  clamped to the last bin; atoms without a scored neighbour are masked out.
- PAE target: token ``j``'s CA expressed in token ``i``'s (N, CA, C) frame, reference minus
  prediction, Euclidean norm; ``floor(d · bins / max_dist)`` clamped. A frame row is valid when its
  three atoms are resolved and non-collinear in the reference (|cos| < 0.9063, bond norms > 0.01 Å)
  and non-collinear in the prediction; the column needs a resolved CA.
- PDE target: ``|d_ref − d_pred|`` between representative atoms, binned like PAE, both resolved.
- Resolved: two-class cross-entropy per non-padding atom; class 0 means resolved.

Rollout tensors (``pred``, logits) have ``N = B · k`` rows for ``k`` rollouts of each of the
``B`` base examples, in the order ``b · k + j``; base-batch tensors are repeated inside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from oplm.fold.diffusion import weighted_rigid_align
from oplm.fold.trunk import fp32_autocast_off

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "LossWeights",
    "diffusion_mse",
    "distogram_loss",
    "edm_weight",
    "express_in_frames",
    "masked_mean",
    "pae_loss",
    "pair_distances",
    "pde_loss",
    "plddt_loss",
    "resolved_loss",
    "smooth_lddt",
    "token_frames",
]

_LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class LossWeights:
    """Spec §6.2 starting weights; ``pae`` weights the PAE sub-term of the confidence composite."""

    diffusion_mse: float = 4.0
    smooth_lddt: float = 1.0
    distogram: float = 0.03
    confidence: float = 1e-4
    pae: float = 1.0


def edm_weight(sigma: Tensor, sigma_data: float) -> Tensor:
    """``(σ² + σ_d²) / (σ σ_d)²`` per sample."""
    return (sigma.float() ** 2 + sigma_data**2) / (sigma.float() * sigma_data) ** 2


def pair_distances(x: Tensor, eps: float = 1e-8) -> Tensor:
    """``(..., n, 3) -> (..., n, n)`` distances, with a finite gradient at zero distance."""
    return torch.sqrt(((x[..., :, None, :] - x[..., None, :, :]) ** 2).sum(-1) + eps)


def masked_mean(values: Tensor, mask: Tensor, dims: tuple[int, ...]) -> Tensor:
    """Mean of ``values`` over ``dims`` where ``mask``; exactly zero where the mask is empty."""
    m = mask.to(values.dtype)
    return (values * m).sum(dims) / m.sum(dims).clamp(min=1.0)


def _repeat(k: int, *tensors: Tensor) -> tuple[Tensor, ...]:
    return tuple(t.repeat_interleave(k, dim=0) for t in tensors)


def _gather_atoms(values: Tensor, index: Tensor) -> Tensor:
    """``values (N, A, ...)`` at ``index (N, L)`` -> ``(N, L, ...)``."""
    if values.ndim == 2:
        return torch.gather(values, 1, index)
    return torch.gather(values, 1, index[..., None].expand(-1, -1, values.shape[-1]))


def diffusion_mse(pred: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    """Weighted MSE / 3 per sample after aligning ``target`` onto ``pred`` (detached): ``(N,)``."""
    with fp32_autocast_off(pred.device):
        w = weights.float()
        aligned = weighted_rigid_align(target.float(), pred.detach().float(), w).detach()
        err = ((pred.float() - aligned) ** 2).sum(-1)
        return (err * w).sum(-1) / (3.0 * w.sum(-1)).clamp(min=1e-5)


def smooth_lddt(pred: Tensor, target: Tensor, mask: Tensor, *, cutoff: float = 15.0) -> Tensor:
    """``1 − smooth lDDT`` per sample ``(N,)``; 0 for a sample with fewer than two masked atoms.

    Each sample is compacted to its masked atoms before the ``(n, n)`` distance matrices are
    formed, so a padded 9216-atom crop costs its real atoms squared, not the budget squared.
    """
    with fp32_autocast_off(pred.device):
        out = []
        for n in range(pred.shape[0]):
            idx = mask[n].nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                out.append(pred.new_zeros(()))
                continue
            d_true = pair_distances(target[n, idx].float())
            d_pred = pair_distances(pred[n, idx].float())
            pair = (d_true < cutoff) & ~torch.eye(idx.numel(), dtype=torch.bool, device=pred.device)
            diff = (d_true - d_pred).abs()
            eps = sum(torch.sigmoid(t - diff) for t in _LDDT_THRESHOLDS) / len(_LDDT_THRESHOLDS)
            out.append(1.0 - (eps * pair).sum() / (pair.sum() + 1e-5))
        return torch.stack(out)


def distogram_loss(
    logits: Tensor,
    coords: Tensor,
    resolved: Tensor,
    token_mask: Tensor,
    rep_atom_idx: Tensor,
    *,
    min_dist: float,
    max_dist: float,
) -> Tensor:
    """Masked mean cross-entropy over representative-atom distance bins per example ``(B,)``."""
    with fp32_autocast_off(logits.device):
        B, L, bins = logits.shape[0], logits.shape[1], logits.shape[-1]
        reps = _gather_atoms(coords.float(), rep_atom_idx)
        rep_ok = _gather_atoms(resolved, rep_atom_idx) & token_mask
        boundaries = torch.linspace(min_dist, max_dist, bins - 1, device=logits.device)
        target = torch.bucketize(pair_distances(reps), boundaries)
        mask = rep_ok[:, :, None] & rep_ok[:, None, :]
        mask &= ~torch.eye(L, dtype=torch.bool, device=logits.device)
        ce = F.cross_entropy(logits.float().reshape(-1, bins), target.reshape(-1), reduction="none")
        return masked_mean(ce.reshape(B, L, L), mask, (-1, -2))


def token_frames(
    coords: Tensor, frame_atom_idx: Tensor, atom_valid: Tensor, token_mask: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-token frames ``(N, L, 3, 3)`` (rows e1, e2, e3), centers ``(N, L, 3)`` and validity.

    Boltz-2 basis: ``w1 = (N − CA)/|·|``, ``w2 = (C − CA)/|·|``, ``e1 ∝ w1 + w2``, ``e2 ∝ w2 − w1``,
    ``e3 = e1 × e2``. Valid where the token is real, its three atoms are ``atom_valid``,
    ``|cos(N−CA, C−CA)| < 0.9063`` and both bonds are longer than 0.01 Å.
    """
    with fp32_autocast_off(coords.device):
        a, b, c = (_gather_atoms(coords.float(), frame_atom_idx[..., k]) for k in range(3))
        a_ok, b_ok, c_ok = (_gather_atoms(atom_valid, frame_atom_idx[..., k]) for k in range(3))
        v1, v2 = a - b, c - b
        n1, n2 = v1.norm(dim=-1, keepdim=True), v2.norm(dim=-1, keepdim=True)
        w1, w2 = v1 / (n1 + 1e-5), v2 / (n2 + 1e-5)
        e1 = (w1 + w2) / ((w1 + w2).norm(dim=-1, keepdim=True) + 1e-5)
        e2 = (w2 - w1) / ((w2 - w1).norm(dim=-1, keepdim=True) + 1e-5)
        e3 = torch.linalg.cross(e1, e2)
        cos = ((v1 / (n1 + 1e-6)) * (v2 / (n2 + 1e-6))).sum(-1).abs()
        ok = (
            token_mask
            & a_ok
            & b_ok
            & c_ok
            & (cos < 0.9063)
            & (n1[..., 0] > 1e-2)
            & (n2[..., 0] > 1e-2)
        )
        return torch.stack([e1, e2, e3], dim=-2), b, ok


def express_in_frames(frames: Tensor, centers: Tensor) -> Tensor:
    """Token ``j``'s center in token ``i``'s frame: ``(N, L, L, 3)`` indexed ``[n, i, j]``."""
    d = centers[:, None, :, :] - centers[:, :, None, :]
    return torch.einsum("bijd,bikd->bijk", d, frames)


def _binned_cross_entropy(logits: Tensor, values: Tensor, scale: float) -> Tensor:
    """Cross-entropy of ``floor(values · scale)`` (clamped to the last bin) against ``logits``."""
    bins = logits.shape[-1]
    target = torch.floor(values * scale).long().clamp(min=0, max=bins - 1)
    ce = F.cross_entropy(logits.float().reshape(-1, bins), target.reshape(-1), reduction="none")
    return ce.reshape(values.shape)


def plddt_loss(
    logits: Tensor,
    pred: Tensor,
    target: Tensor,
    resolved: Tensor,
    atom_mask: Tensor,
    token_mask: Tensor,
    rep_atom_idx: Tensor,
    *,
    cutoff: float = 15.0,
) -> Tensor:
    """Per-rollout pLDDT cross-entropy ``(N,)`` (per-atom lDDT target, ``bins`` from the logits)."""
    with fp32_autocast_off(logits.device):
        k = pred.shape[0] // target.shape[0]
        t_coords, t_res, a_mask, tok, rep_idx = _repeat(
            k, target, resolved, atom_mask, token_mask, rep_atom_idx
        )
        A = pred.shape[1]
        t_reps, p_reps = (
            _gather_atoms(t_coords.float(), rep_idx),
            _gather_atoms(pred.float(), rep_idx),
        )
        rep_ok = _gather_atoms(t_res, rep_idx) & tok
        d_true = torch.sqrt(((t_coords.float()[:, :, None] - t_reps[:, None]) ** 2).sum(-1) + 1e-8)
        d_pred = torch.sqrt(((pred.float()[:, :, None] - p_reps[:, None]) ** 2).sum(-1) + 1e-8)
        own = rep_idx[:, None, :] == torch.arange(A, device=pred.device)[None, :, None]
        scored = t_res[:, :, None] & rep_ok[:, None, :] & ~own & (d_true < cutoff)
        l1 = (d_true - d_pred).abs()
        score = sum((l1 < t).float() for t in _LDDT_THRESHOLDS) / len(_LDDT_THRESHOLDS)
        n_scored = scored.float().sum(-1)
        # Multiply by the reciprocal as the oracle does: a quotient can land one ulp across a bin
        # edge (lddt = 15/25 floors to bin 5 or 6), which flips one atom's target class.
        lddt = (1e-10 + (score * scored).sum(-1)) * (1.0 / (1e-10 + n_scored))
        ce = _binned_cross_entropy(logits, lddt, float(logits.shape[-1]))
        return masked_mean(ce, t_res & a_mask & (n_scored > 0), (-1,))


def pae_loss(
    logits: Tensor,
    pred: Tensor,
    target: Tensor,
    resolved: Tensor,
    token_mask: Tensor,
    frame_atom_idx: Tensor,
    *,
    max_dist: float,
) -> Tensor:
    """Per-rollout PAE cross-entropy ``(N,)`` over valid (frame ``i``, resolved CA ``j``) pairs."""
    with fp32_autocast_off(logits.device):
        k = pred.shape[0] // target.shape[0]
        t_coords, t_res, tok, fidx = _repeat(k, target, resolved, token_mask, frame_atom_idx)
        true_frames, true_centers, true_ok = token_frames(t_coords, fidx, t_res, tok)
        pred_frames, pred_centers, pred_ok = token_frames(pred, fidx, torch.ones_like(t_res), tok)
        delta = express_in_frames(true_frames, true_centers) - express_in_frames(
            pred_frames, pred_centers
        )
        dist = torch.sqrt((delta**2).sum(-1) + 1e-8)
        ca_ok = _gather_atoms(t_res, fidx[..., 1]) & tok
        mask = (true_ok & pred_ok)[:, :, None] & ca_ok[:, None, :]
        return masked_mean(
            _binned_cross_entropy(logits, dist, logits.shape[-1] / max_dist), mask, (-1, -2)
        )


def pde_loss(
    logits: Tensor,
    pred: Tensor,
    target: Tensor,
    resolved: Tensor,
    token_mask: Tensor,
    rep_atom_idx: Tensor,
    *,
    max_dist: float,
) -> Tensor:
    """Per-rollout PDE cross-entropy ``(N,)`` over pairs of resolved representative atoms."""
    with fp32_autocast_off(logits.device):
        k = pred.shape[0] // target.shape[0]
        t_coords, t_res, tok, rep_idx = _repeat(k, target, resolved, token_mask, rep_atom_idx)
        rep_ok = _gather_atoms(t_res, rep_idx) & tok
        err = (
            pair_distances(_gather_atoms(t_coords.float(), rep_idx))
            - pair_distances(_gather_atoms(pred.float(), rep_idx))
        ).abs()
        mask = rep_ok[:, :, None] & rep_ok[:, None, :]
        return masked_mean(
            _binned_cross_entropy(logits, err, logits.shape[-1] / max_dist), mask, (-1, -2)
        )


def resolved_loss(logits: Tensor, resolved: Tensor, atom_mask: Tensor) -> Tensor:
    """Per-rollout two-class cross-entropy ``(N,)``; class 0 is "resolved"."""
    with fp32_autocast_off(logits.device):
        k = logits.shape[0] // resolved.shape[0]
        res, am = _repeat(k, resolved, atom_mask)
        ce = F.cross_entropy(
            logits.float().reshape(-1, 2), (~res).long().reshape(-1), reduction="none"
        )
        return masked_mean(ce.reshape(res.shape), am, (-1,))

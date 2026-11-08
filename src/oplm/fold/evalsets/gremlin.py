"""GREMLIN pseudolikelihood Potts fit and its APC-corrected contact map, in PyTorch.

A port of GREMLIN_TF v2.1 by Sergey Ovchinnikov and Peter Koo (``GREMLIN_TF.ipynb`` in
https://github.com/sokrypton/GREMLIN_CPP, formerly sokrypton/GREMLIN_TF, master as of
2022-10-17). The defaults are copied from it:

- ``W`` zero-initialized; symmetrized as ``W_sym = W_m + W_m^T`` where ``W_m`` is ``W`` with the
  ``i == j`` blocks zeroed. Regularization and the contact map use ``W_sym``.
- ``V`` initialized to ``log(sum_n w_n onehot_n + 0.01 * log(Neff))``, centered over states.
- Loss: ``(-sum_n w_n PLL_n + lam_v ||V||^2 + lam_w * 0.5 * (L - 1) * (A - 1) * ||W_sym||^2)
  / Neff``, ``Neff = sum_n w_n``, ``lam_v = lam_w = 0.01``, ``A = 21`` states.
- Optimizer: the reference's modified Adam (second moment is one scalar per tensor,
  ``sum(g^2)``; no bias correction; ``b1 = 0.9``, ``b2 = 0.999``, ``eps = 1e-8``),
  ``lr = 1.0``, 100 full-batch iterations.
- Contact map: Frobenius norm of each ``W_sym[i, :20, j, :20]`` block (gap excluded), then APC.

Departures from the reference, none affecting a real MSA: the pseudo-count floors Neff at 2,
because a query-only MSA (Neff = 1) gives ``log(0)`` in ``V``; and the APC-corrected map's
diagonal is set to 0 (the reference drops the diagonal). No explicit gauge fixing, as in the
reference: the L2 penalty fixes the gauge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from oplm.eval.metrics.contact import apply_apc

__all__ = ["GremlinConfig", "fit_gremlin", "full_length_map", "potts_contact_map"]

_STATES = 21  # 20 amino acids + gap, gap last
_B1, _B2, _EPS = 0.9, 0.999, 1e-8


@dataclass(frozen=True)
class GremlinConfig:
    """GREMLIN_TF v2.1 defaults (see the module docstring)."""

    lam_v: float = 0.01
    lam_w_scale: float = 0.01
    lr: float = 1.0
    iterations: int = 100


def _symmetrize(w: Tensor) -> Tensor:
    length = w.shape[0]
    w = w * (1 - torch.eye(length, device=w.device)).view(length, 1, length, 1)
    return w + w.permute(2, 3, 0, 1)


def fit_gremlin(
    msa: Tensor,
    weights: Tensor,
    cfg: GremlinConfig = GremlinConfig(),  # noqa: B008 (frozen, so a shared default is safe)
    device: str | torch.device = "cpu",
) -> Tensor:
    """Symmetric couplings ``W``, ``(L, 21, L, 21)`` float32 on CPU, fit to ``(N, L)`` ``msa``."""
    n, length = msa.shape
    la = length * _STATES
    target = msa.to(device).long()
    weights = weights.to(device, torch.float32)
    neff = float(weights.sum())
    onehot = torch.nn.functional.one_hot(target, _STATES).float().view(n, la)

    counts = (weights[:, None] * onehot).sum(0).view(length, _STATES)
    v = torch.log(counts + 0.01 * math.log(max(neff, 2.0)))
    v = (v - v.mean(-1, keepdim=True)).requires_grad_()
    w = torch.zeros(length, _STATES, length, _STATES, device=device, requires_grad=True)
    lam_w = cfg.lam_w_scale * 0.5 * (length - 1) * (_STATES - 1)

    params = (v, w)
    m = [torch.zeros_like(p) for p in params]
    s = [torch.zeros((), device=device) for _ in params]
    for _ in range(cfg.iterations):
        w_sym = _symmetrize(w)
        logits = (onehot @ w_sym.view(la, la)).view(n, length, _STATES) + v
        ce = torch.nn.functional.cross_entropy(
            logits.view(-1, _STATES), target.view(-1), reduction="none"
        ).view(n, length)
        loss = weights @ ce.sum(1) + cfg.lam_v * v.square().sum() + lam_w * w_sym.square().sum()
        grads = torch.autograd.grad(loss / neff, params)
        with torch.no_grad():
            for p, g, m_i, s_i in zip(params, grads, m, s, strict=True):
                m_i.mul_(_B1).add_(g, alpha=1 - _B1)
                s_i.mul_(_B2).add_((1 - _B2) * g.square().sum())
                p.sub_(cfg.lr / (s_i.sqrt() + _EPS) * m_i)
    with torch.no_grad():
        return _symmetrize(w).cpu()


def potts_contact_map(w: Tensor) -> Tensor:
    """``(L, L)`` APC-corrected Frobenius norms over the 20 amino-acid states, zero diagonal."""
    raw = w[:, :20, :, :20].square().sum((1, 3)).sqrt()
    raw = (raw + raw.T) / 2
    raw.fill_diagonal_(0)
    return apply_apc(raw).fill_diagonal_(0)


def full_length_map(contacts: Tensor, kept_columns: Tensor, length: int) -> Tensor:
    """``(length, length)`` map with ``contacts`` at ``kept_columns`` and NaN elsewhere."""
    out = torch.full((length, length), float("nan"), dtype=contacts.dtype)
    out[kept_columns[:, None], kept_columns[None, :]] = contacts
    return out

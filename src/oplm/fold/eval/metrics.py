"""Held-out structure metrics: TM-score and lDDT with explicit correspondence (addendum §6).

Both take predicted and reference coordinates in the same atom order (the correspondence is the
row index) and a reference mask: a row unresolved in the reference is excluded, never scored as
wrong. TM-score follows the TM-score program (Zhang & Skolnick 2004): normalized by the number of
kept reference rows, ``d0`` from that length, and the program's fragment-seeded iterative
superposition search rather than one RMSD-optimal fit. lDDT is the global score of Mariani et al.
2013: reference pairs closer than 15 Å, thresholds 0.5/1/2/4 Å and, given a residue map, no pairs
inside one residue; the OpenStructure tool's stereochemistry checks are not reproduced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = ["lddt", "tm_d0", "tm_score"]

_LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
_TM_ITERATIONS = 20  # TM-score's n_it


def tm_d0(length: int) -> float:
    """TM-score's distance scale in Å for a reference of ``length`` residues (at least 0.5)."""
    if length <= 21:
        return 0.5
    return max(1.24 * (length - 15) ** (1.0 / 3.0) - 1.8, 0.5)


def _fragment_lengths(n: int) -> list[int]:
    """TM-score's seed fragment lengths: n, n/2, n/4, ... down to min(4, n), at most six."""
    shortest = min(4, n)
    lengths: list[int] = []
    for i in range(5):
        length = int(n / 2**i)
        if length <= shortest:
            lengths.append(shortest)
            return lengths
        lengths.append(length)
    lengths.append(shortest)
    return lengths


def _superposed_sq_dist(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Squared distances (B, N) after superposing ``x`` onto ``y`` on each row set of ``w``.

    Batched Kabsch with 0/1 weights: ``x`` and ``y`` are (N, 3), ``w`` is (B, N).
    """
    wsum = w.sum(axis=1, keepdims=True)
    mx = (w @ x) / wsum
    my = (w @ y) / wsum
    xc = x[None] - mx[:, None]
    yc = y[None] - my[:, None]
    h = np.einsum("bn,bni,bnj->bij", w, xc, yc)
    u, _, vt = np.linalg.svd(h)
    v = vt.transpose(0, 2, 1)
    ut = u.transpose(0, 2, 1)
    d = np.sign(np.linalg.det(v @ ut))
    d[d == 0] = 1.0
    flip = np.ones((len(w), 3))
    flip[:, 2] = d
    rot = v @ (flip[:, :, None] * ut)  # xc @ rot.T best matches yc
    moved = xc @ rot.transpose(0, 2, 1) + my[:, None]
    return ((moved - y[None]) ** 2).sum(axis=-1)


def _select(sq_dist: np.ndarray, cutoff: float) -> np.ndarray:
    """Rows closer than ``cutoff``; a set of fewer than 3 relaxes it in 0.5 Å steps (TM-score)."""
    dist = np.sqrt(sq_dist)
    keep = dist < cutoff
    few = keep.sum(axis=1) < 3
    if few.any():
        if dist.shape[1] > 3:
            third = np.sort(dist[few], axis=1)[:, 2]
            steps = np.floor((third - cutoff) / 0.5) + 1.0
            keep[few] = dist[few] < (cutoff + 0.5 * steps)[:, None]
        else:  # ponytail: TM-score aligns on the 1-2 rows left; with 3 rows we use all three
            keep[few] = True
    return keep.astype(np.float64)


def tm_score(pred: ArrayLike, ref: ArrayLike, mask: ArrayLike | None = None) -> float:
    """TM-score of ``pred`` against ``ref`` over corresponding rows (e.g. one CA per residue).

    Args:
        pred: Predicted coordinates (N, 3) in Å.
        ref: Reference coordinates (N, 3) in Å; row ``i`` corresponds to ``pred[i]``.
        mask: Rows present in the reference (N,); the normalization length is their count.

    Raises:
        ValueError: The shapes differ or are not (N, 3), or fewer than 3 rows are kept.
    """
    x = np.asarray(torch.as_tensor(pred).detach().cpu(), dtype=np.float64)
    y = np.asarray(torch.as_tensor(ref).detach().cpu(), dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"pred and ref must both be (N, 3); got {x.shape} and {y.shape}")
    if mask is not None:
        keep = np.asarray(torch.as_tensor(mask).detach().cpu(), dtype=bool)
        x, y = x[keep], y[keep]
    n = len(x)
    if n < 3:
        raise ValueError(f"TM-score needs at least 3 kept rows, got {n}")
    d0 = tm_d0(n)
    d0_search = min(max(d0, 4.5), 8.0)
    positions = np.arange(n)
    best = 0.0
    for frag in _fragment_lengths(n):
        starts = np.arange(n - frag + 1)[:, None]
        w = ((positions >= starts) & (positions < starts + frag)).astype(np.float64)
        sq = _superposed_sq_dist(x, y, w)
        best = max(best, float((1.0 / (1.0 + sq / d0**2)).sum(axis=1).max()) / n)
        w = _select(sq, d0_search - 1.0)
        for _ in range(_TM_ITERATIONS):
            sq = _superposed_sq_dist(x, y, w)
            best = max(best, float((1.0 / (1.0 + sq / d0**2)).sum(axis=1).max()) / n)
            w_next = _select(sq, d0_search + 1.0)
            if np.array_equal(w_next, w):
                break
            w = w_next
    return best


def lddt(
    pred: ArrayLike,
    ref: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    residue_index: ArrayLike | None = None,
    cutoff: float = 15.0,
    chunk: int = 2048,
) -> float:
    """Global lDDT of ``pred`` against ``ref`` over corresponding rows (atoms).

    A pair (i, j) is scored when i != j, both rows are kept by ``mask``, the reference distance is
    below ``cutoff`` and, if ``residue_index`` is given, the rows belong to different residues. The
    score is the mean over scored pairs and the four thresholds of |d_pred - d_ref| < threshold.

    Args:
        pred: Predicted coordinates (N, 3) in Å.
        ref: Reference coordinates (N, 3) in Å.
        mask: Rows present in the reference (N,).
        residue_index: Residue of each row (N,); pairs inside one residue are skipped.
        cutoff: Inclusion radius in Å, on reference distances.
        chunk: Rows per block of the pairwise computation (memory is chunk × N).

    Raises:
        ValueError: The shapes differ or are not (N, 3), or no pair qualifies.
    """
    x = torch.as_tensor(pred).detach().float()
    y = torch.as_tensor(ref).detach().float().to(x.device)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(
            f"pred and ref must both be (N, 3); got {tuple(x.shape)} and {tuple(y.shape)}"
        )
    keep = torch.ones(len(x), dtype=torch.bool, device=x.device)
    if mask is not None:
        keep = torch.as_tensor(mask, device=x.device).bool()
    x, y = x[keep], y[keep]
    res = None
    if residue_index is not None:
        res = torch.as_tensor(residue_index, device=x.device)[keep]
    n = len(y)
    total = torch.zeros((), dtype=torch.float64, device=x.device)
    count = 0
    for start in range(0, n, chunk):
        rows = torch.arange(start, min(start + chunk, n), device=x.device)
        mode = "donot_use_mm_for_euclid_dist"  # exact distances: no matmul expansion error
        d_ref = torch.cdist(y[rows], y, compute_mode=mode)
        d_pred = torch.cdist(x[rows], x, compute_mode=mode)
        pair = d_ref < cutoff
        pair[torch.arange(len(rows), device=x.device), rows] = False
        if res is not None:
            pair &= res[rows, None] != res[None, :]
        err = (d_pred - d_ref).abs()
        preserved = sum((err < t).float() for t in _LDDT_THRESHOLDS) / len(_LDDT_THRESHOLDS)
        total += preserved[pair].double().sum()
        count += int(pair.sum())
    if count == 0:
        raise ValueError("lDDT has no reference pairs within the inclusion radius")
    return float(total / count)

"""Contact-map geometry, APC, and precision@L utilities.

These are the shared primitives for structure contact prediction: building a
binary ground-truth contact map from backbone coordinates, Average Product
Correction, and precision@L scoring. They back the categorical-Jacobian contact
metric (see :mod:`oplm.eval.metrics.categorical_jacobian`).
"""

from __future__ import annotations

import torch
from torch import Tensor

# Virtual Cβ coefficients from ideal backbone geometry.
# These place the virtual Cβ at the standard tetrahedral position
# relative to the backbone N, CA, C atoms.
_CBETA_A = -0.58273431
_CBETA_B = 0.56805504
_CBETA_C = -0.54927641


def compute_virtual_cbeta(coords: Tensor) -> Tensor:
    """Compute virtual Cβ coordinates from backbone N, CA, C atoms.

    Uses the standard geometric construction from ideal backbone geometry.
    Works for all residues including glycine (which has no real Cβ).

    Args:
        coords: Backbone coordinates ``(L, 3, 3)`` with ``[N, CA, C]`` ordering.

    Returns:
        Cβ coordinates ``(L, 3)``. Positions where any backbone atom is NaN
        will also be NaN.
    """
    n = coords[:, 0, :]  # (L, 3)
    ca = coords[:, 1, :]  # (L, 3)
    c = coords[:, 2, :]  # (L, 3)

    b = ca - n  # CA-N bond vector
    c_vec = c - ca  # C-CA bond vector
    a = torch.linalg.cross(b, c_vec)  # Normal to backbone plane

    # Coefficients assume un-normalized cross product (magnitude encodes
    # bond angle information needed for correct Cβ placement).
    cbeta: Tensor = _CBETA_A * a + _CBETA_B * b + _CBETA_C * c_vec + ca
    return cbeta


def compute_contact_map(
    coords: Tensor,
    threshold: float = 8.0,
    use_cbeta: bool = True,
) -> Tensor:
    """Compute binary contact map from backbone coordinates.

    Args:
        coords: Backbone coordinates ``(L, 3, 3)`` with ``[N, CA, C]`` ordering.
        threshold: Distance cutoff in angstroms. Default 8.0 (Cβ).
        use_cbeta: If True, compute virtual Cβ and use Cβ distances.
            If False, use Cα distances directly.

    Returns:
        Binary contact map ``(L, L)`` as float tensor. 1.0 where
        distance < threshold. Positions with NaN coordinates get 0.0.
    """
    positions = compute_virtual_cbeta(coords) if use_cbeta else coords[:, 1, :]

    # Pairwise distances
    dists = torch.cdist(positions, positions)  # (L, L)

    # Handle NaN: set to infinity so they don't form contacts
    nan_mask = torch.isnan(dists)
    dists = dists.masked_fill(nan_mask, float("inf"))

    return (dists < threshold).float()


def apply_apc(matrix: Tensor) -> Tensor:
    """Apply Average Product Correction to a contact score matrix.

    APC removes phylogenetic and systematic biases by subtracting the
    expected score under a null model of independent columns.

    Args:
        matrix: Score matrix ``(L, L)``.

    Returns:
        APC-corrected matrix ``(L, L)``. Returned unchanged if global mean
        is approximately zero.
    """
    row_mean = matrix.mean(dim=-1, keepdim=True)  # (L, 1)
    col_mean = matrix.mean(dim=-2, keepdim=True)  # (1, L)
    global_mean = matrix.mean()

    if global_mean.abs() < 1e-8:
        return matrix

    return matrix - (row_mean * col_mean) / global_mean


def compute_precision_at_l(
    pred_contacts: Tensor,
    true_contacts: Tensor,
    seq_len: int,
    min_seq_sep: int = 6,
    l_divisor: int = 1,
) -> float:
    """Compute precision@(L/divisor) for long-range contacts.

    Takes the top ``L/divisor`` predictions among residue pairs with
    sequence separation >= ``min_seq_sep`` and computes the fraction
    that are true contacts.

    Args:
        pred_contacts: Predicted contact scores ``(L, L)``.
        true_contacts: Binary ground-truth contacts ``(L, L)``.
        seq_len: Effective sequence length.
        min_seq_sep: Minimum ``|i - j|`` for long-range contacts.
        l_divisor: Denominator for L (1 -> L, 2 -> L/2, 5 -> L/5).

    Returns:
        Precision as a float in ``[0, 1]``. Returns 0.0 if no valid
        pairs exist or k is zero.
    """
    L = pred_contacts.shape[0]

    # Build mask: upper triangle, |i - j| >= min_seq_sep
    row_idx, col_idx = torch.triu_indices(L, L, offset=min_seq_sep)
    if len(row_idx) == 0:
        return 0.0

    pred_scores = pred_contacts[row_idx, col_idx]
    true_labels = true_contacts[row_idx, col_idx]

    k = max(1, seq_len // l_divisor)
    k = min(k, len(pred_scores))  # Don't exceed available pairs

    _, top_indices = torch.topk(pred_scores, k)
    precision = true_labels[top_indices].float().mean().item()

    return precision

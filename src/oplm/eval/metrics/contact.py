"""Contact prediction metrics for structure evaluation.

Implements the ESM-1b contact prediction protocol (Rives et al., 2021):
extract attention maps from all layers/heads, symmetrize, apply APC,
fit logistic regression to predict binary contacts (Cβ distance < 8Å).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Virtual Cβ coefficients from ideal backbone geometry.
# These place the virtual Cβ at the standard tetrahedral position
# relative to the backbone N, CA, C atoms.
_CBETA_A = -0.58273431
_CBETA_B = 0.56805504
_CBETA_C = -0.54927641


@dataclass
class StructureContactData:
    """Pre-extracted features and labels for one structure.

    Used as the unit of data for the logistic regression pipeline.

    Attributes:
        features: Attention features for valid residue pairs,
            shape ``(n_pairs, n_layers * n_heads)``.
        labels: Binary contact labels for the same pairs, shape ``(n_pairs,)``.
        seq_len: Effective sequence length (number of residues).
        pair_indices: ``(i_indices, j_indices)`` for reconstructing ``(L, L)``
            predictions from flat pair arrays.
    """

    features: Tensor  # (n_pairs, n_layers * n_heads)
    labels: Tensor  # (n_pairs,) binary
    seq_len: int
    pair_indices: tuple[Tensor, Tensor]  # (i_indices, j_indices)


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


def extract_attention_contacts(
    attention_weights: list[Tensor],
    layer: int | str = "all",
    head_aggregation: str | None = None,
    symmetrize: bool = True,
    apc: bool = True,
) -> Tensor:
    """Extract contact predictions from attention weight matrices.

    The default settings (``layer="all"``, ``head_aggregation=None``) match
    the ESM-1b protocol (Rives et al., 2021): all layers, all heads
    retained separately, each symmetrized and APC-corrected individually.

    Args:
        attention_weights: List of ``(H, L, L)`` CPU tensors, one per layer.
            Unbatched, with special tokens already stripped.
        layer: Which layers to use.
            ``"all"``: all layers (default).
            ``"last"``: alias for ``-1`` (final layer only).
            ``int``: specific layer index (supports negative indexing).
        head_aggregation: How to aggregate across heads.
            ``None``: keep heads separate (default).
            ``"mean"``: average across heads.
            ``"max"``: max across heads.
        symmetrize: If True, average ``A_ij`` and ``A_ji`` for each map.
        apc: If True, apply APC to each individual head map.

    Returns:
        Contact score tensor. Shape depends on arguments:

        - ``layer="all"``, ``head_aggregation=None``: ``(n_layers, n_heads, L, L)``
        - ``layer="all"``, ``head_aggregation="mean"/"max"``: ``(n_layers, L, L)``
        - single layer, ``head_aggregation=None``: ``(n_heads, L, L)``
        - single layer, ``head_aggregation="mean"/"max"``: ``(L, L)``
    """
    # Layer selection
    if layer == "all":
        selected = torch.stack(attention_weights)  # (n_layers, H, L, L)
        single_layer = False
    elif layer == "last":
        selected = attention_weights[-1].unsqueeze(0)  # (1, H, L, L)
        single_layer = True
    elif isinstance(layer, int):
        selected = attention_weights[layer].unsqueeze(0)  # (1, H, L, L)
        single_layer = True
    else:
        raise ValueError(f"Invalid layer value: {layer!r}. Expected 'all', 'last', or int.")

    # Symmetrize: (m + m^T) / 2, vectorized over layer and head dims
    if symmetrize:
        selected = (selected + selected.transpose(-1, -2)) / 2

    # APC: apply per individual head map
    if apc:
        n_layers, n_heads, seq_len, _ = selected.shape
        flat = selected.reshape(n_layers * n_heads, seq_len, seq_len)
        corrected = torch.stack([apply_apc(m) for m in flat])
        selected = corrected.reshape(n_layers, n_heads, seq_len, seq_len)

    # Head aggregation
    if head_aggregation == "mean":
        selected = selected.mean(dim=1)  # (n_layers, L, L)
    elif head_aggregation == "max":
        selected = selected.max(dim=1).values  # (n_layers, L, L)
    elif head_aggregation is not None:
        raise ValueError(
            f"Invalid head_aggregation: {head_aggregation!r}. Expected None, 'mean', or 'max'."
        )

    # Squeeze layer dim for single-layer selection
    if single_layer:
        selected = selected.squeeze(0)

    return selected


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


def build_structure_contact_data(
    attention_contacts: Tensor,
    true_contacts: Tensor,
    seq_len: int,
    min_seq_sep: int = 6,
) -> StructureContactData:
    """Build feature/label arrays for one structure for logreg.

    Extracts valid long-range residue pairs and constructs the feature
    matrix from the attention stack and the label vector from the
    ground-truth contact map.

    Args:
        attention_contacts: Per-layer per-head contact scores,
            shape ``(n_layers, n_heads, L, L)``.
        true_contacts: Binary contact map ``(L, L)``.
        seq_len: Effective sequence length.
        min_seq_sep: Minimum sequence separation for long-range contacts.

    Returns:
        :class:`StructureContactData` with features of shape
        ``(n_pairs, n_layers * n_heads)`` and labels of shape ``(n_pairs,)``.
    """
    L = true_contacts.shape[0]
    n_layers, n_heads = attention_contacts.shape[:2]

    # Valid pair mask: upper triangle, |i - j| >= min_seq_sep
    i_indices, j_indices = torch.triu_indices(L, L, offset=min_seq_sep)

    # Extract features: (n_layers, n_heads, n_pairs)
    features = attention_contacts[:, :, i_indices, j_indices]
    # Reshape to (n_pairs, n_layers * n_heads)
    features = features.permute(2, 0, 1).reshape(len(i_indices), n_layers * n_heads)

    labels = true_contacts[i_indices, j_indices]

    return StructureContactData(
        features=features,
        labels=labels,
        seq_len=seq_len,
        pair_indices=(i_indices, j_indices),
    )


def _fallback_mean_attention_precision(
    structures: list[StructureContactData],
    l_divisor: int = 1,
    min_seq_sep: int = 6,
) -> float:
    """Simple mean-attention P@L when logreg cannot be used.

    Averages features across the head dimension to produce a single score
    per pair, then computes P@L for each structure and returns the mean.
    """
    precisions: list[float] = []
    for s in structures:
        # Average features to get a single score per pair
        pair_scores = s.features.mean(dim=-1)  # (n_pairs,)

        # Reconstruct (L, L) score matrix
        pred = torch.zeros(s.seq_len, s.seq_len)
        pred[s.pair_indices[0], s.pair_indices[1]] = pair_scores

        true = torch.zeros(s.seq_len, s.seq_len)
        true[s.pair_indices[0], s.pair_indices[1]] = s.labels

        p = compute_precision_at_l(pred, true, s.seq_len, min_seq_sep, l_divisor)
        precisions.append(p)

    return float(np.mean(precisions)) if precisions else 0.0


def compute_logreg_precision_at_l(
    structures: list[StructureContactData],
    n_train: int = 20,
    n_iterations: int = 5,
    logreg_c: float = 0.15,
    l_divisor: int = 1,
    min_seq_sep: int = 6,
    seed: int = 42,
) -> float:
    """Compute P@L using logistic regression on attention features.

    Fits an L1-regularized logistic regression on attention head features
    from training structures, then evaluates precision on held-out test
    structures. Repeated over multiple random train/test splits.

    Falls back to mean-attention P@L if fewer than ``n_train + 1``
    structures are available.

    Args:
        structures: List of :class:`StructureContactData`, one per protein.
        n_train: Number of structures for training per iteration.
        n_iterations: Number of random train/test splits.
        logreg_c: Inverse regularization strength for logistic regression.
        l_divisor: L divisor for P@L (1 -> L, 2 -> L/2, 5 -> L/5).
        min_seq_sep: Minimum sequence separation for P@L computation.
        seed: Base random seed for deterministic splits.

    Returns:
        Mean P@L across all iterations and test structures.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:
        raise ImportError(
            "Logistic regression P@L requires scikit-learn. Install with: pip install oplm[train]"
        ) from e

    if len(structures) < n_train + 1:
        logger.warning(
            "Only %d structures available (need %d for logreg). "
            "Falling back to mean-attention P@L.",
            len(structures),
            n_train + 1,
        )
        return _fallback_mean_attention_precision(structures, l_divisor, min_seq_sep)

    all_precisions: list[float] = []

    for iteration in range(n_iterations):
        rng = np.random.RandomState(seed + iteration)
        indices = rng.permutation(len(structures))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        # Build training data
        train_features = torch.cat([structures[i].features for i in train_idx], dim=0)
        train_labels = torch.cat([structures[i].labels for i in train_idx], dim=0)

        X_train = train_features.numpy().astype(np.float32)
        y_train = train_labels.numpy().astype(np.float32)

        # Skip iteration if training data has only one class
        if len(np.unique(y_train)) < 2:
            continue

        # Fit logistic regression
        logreg = LogisticRegression(
            penalty="l1",
            C=logreg_c,
            solver="liblinear",
            max_iter=1000,
            random_state=seed,
        )
        logreg.fit(X_train, y_train)

        # Evaluate on each test structure
        for idx in test_idx:
            s = structures[idx]
            X_test = s.features.numpy().astype(np.float32)
            probs = logreg.predict_proba(X_test)[:, 1]  # P(contact)

            # Reconstruct (L, L) predicted score matrix
            pred = torch.zeros(s.seq_len, s.seq_len)
            pred[s.pair_indices[0], s.pair_indices[1]] = torch.from_numpy(probs).float()

            true = torch.zeros(s.seq_len, s.seq_len)
            true[s.pair_indices[0], s.pair_indices[1]] = s.labels

            p = compute_precision_at_l(pred, true, s.seq_len, min_seq_sep, l_divisor)
            all_precisions.append(p)

    return float(np.mean(all_precisions)) if all_precisions else 0.0

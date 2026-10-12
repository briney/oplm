"""Training crops of one chain (spec §7.3, addendum §2: single-chain examples).

``contiguous_crop`` is a random window; ``spatial_crop`` picks a random resolved token and
keeps the ``crop_size`` tokens nearest to it by center-atom distance (unresolved tokens are
infinitely far, so they fill a crop only after every resolved token, by index). Interface crops
arrive with multi-chain training (milestone 3). Randomness comes from the caller's
``torch.Generator`` so a resumed run restores the same crop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from oplm.fold.data.complex import MolecularComplex

__all__ = ["contiguous_crop", "crop_chain", "spatial_crop"]


def contiguous_crop(n: int, crop_size: int, generator: torch.Generator) -> np.ndarray:
    """Positions of a random ``crop_size`` window of ``range(n)``, or all of it when it fits."""
    if n <= crop_size:
        return np.arange(n)
    start = int(torch.randint(0, n - crop_size + 1, (1,), generator=generator))
    return np.arange(start, start + crop_size)


def spatial_crop(
    centers: np.ndarray, valid: np.ndarray, crop_size: int, generator: torch.Generator
) -> np.ndarray:
    """Sorted positions of the ``crop_size`` tokens nearest a random resolved center.

    Args:
        centers: ``(n, 3)`` center-atom coordinates.
        valid: ``(n,)`` whether each center atom is resolved.
        crop_size: Tokens to keep; everything when ``n <= crop_size``.
        generator: Draws the center (and the window when no center is resolved).
    """
    n = centers.shape[0]
    if n <= crop_size:
        return np.arange(n)
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        return contiguous_crop(n, crop_size, generator)
    pick = candidates[int(torch.randint(0, candidates.size, (1,), generator=generator))]
    dist = np.linalg.norm(centers - centers[pick], axis=1)
    dist = np.where(valid, dist, np.inf)
    order = np.lexsort((np.arange(n), dist))  # distance, then index
    return np.sort(order[:crop_size])


def crop_chain(
    complex: MolecularComplex,
    chain: int,
    crop_size: int,
    *,
    contiguous_prob: float,
    generator: torch.Generator,
) -> MolecularComplex:
    """Crop chain ``chain`` of ``complex`` to a one-chain complex of at most ``crop_size`` tokens.

    Draws the crop type first (contiguous with probability ``contiguous_prob``, else spatial)
    so the generator sequence is fixed per example.
    """
    tokens = complex.chain_tokens(chain)
    center_atoms = complex.token_center_atom[tokens]
    use_contiguous = bool(torch.rand((), generator=generator) < contiguous_prob)
    if use_contiguous:
        local = contiguous_crop(tokens.size, crop_size, generator)
    else:
        local = spatial_crop(
            complex.atom_coords[center_atoms],
            complex.atom_resolved[center_atoms],
            crop_size,
            generator,
        )
    return complex.select_tokens(tokens[local])

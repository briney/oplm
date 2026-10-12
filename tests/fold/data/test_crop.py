"""Single-chain contiguous and spatial crops (spec §7.3; addendum §2, §4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from oplm.fold.data.crop import contiguous_crop, crop_chain, spatial_crop

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.fold.data.complex import MolecularComplex


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_contiguous_crop_is_a_window_or_everything() -> None:
    assert contiguous_crop(10, 16, gen(0)).tolist() == list(range(10))
    starts = set()
    for seed in range(40):
        crop = contiguous_crop(46, 16, gen(seed))
        assert crop.shape == (16,) and np.array_equal(crop, np.arange(crop[0], crop[0] + 16))
        starts.add(int(crop[0]))
    assert len(starts) > 5 and min(starts) >= 0 and max(starts) <= 30


def test_spatial_crop_takes_the_nearest_tokens_to_a_resolved_center(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    c = parsed("1crn")
    tokens = c.chain_tokens(0)
    centers = c.atom_coords[c.token_center_atom[tokens]]
    valid = c.atom_resolved[c.token_center_atom[tokens]]
    crop = spatial_crop(centers, valid, 16, gen(3))
    assert crop.shape == (16,) and np.all(np.diff(crop) > 0)
    # Replay the draw to learn the center, then check the 16 nearest by CA distance.
    g = gen(3)
    pick = np.flatnonzero(valid)[int(torch.randint(0, int(valid.sum()), (1,), generator=g))]
    dist = np.linalg.norm(centers - centers[pick], axis=1)
    nearest = np.sort(np.lexsort((np.arange(len(dist)), dist))[:16])
    assert np.array_equal(crop, nearest) and pick in crop
    assert np.array_equal(crop, spatial_crop(centers, valid, 16, gen(3)))  # deterministic
    assert spatial_crop(centers, valid, 100, gen(0)).tolist() == list(range(46))


def test_spatial_crop_never_centers_on_unresolved_tokens(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    """2ds7: tokens 0-10 have no atoms; the center is always drawn from the 40 resolved tokens."""
    c = parsed("2ds7")
    tokens = c.chain_tokens(0)
    centers = c.atom_coords[c.token_center_atom[tokens]]
    valid = c.atom_resolved[c.token_center_atom[tokens]]
    assert valid.sum() == 40
    resolved = np.flatnonzero(valid)
    for seed in range(50):
        crop = spatial_crop(centers, valid, 8, gen(seed))
        pick = resolved[int(torch.randint(0, resolved.size, (1,), generator=gen(seed)))]
        dist = np.where(valid, np.linalg.norm(centers - centers[pick], axis=1), np.inf)
        nearest = np.sort(np.lexsort((np.arange(len(dist)), dist))[:8])
        assert pick in crop and np.array_equal(crop, nearest)


def test_spatial_crop_fills_with_unresolved_tokens_by_index(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    """Fewer resolved tokens than the crop: every resolved token, then unresolved by index."""
    c = parsed("2ds7")
    tokens = c.chain_tokens(0)
    centers = c.atom_coords[c.token_center_atom[tokens]]
    valid = c.atom_resolved[c.token_center_atom[tokens]]
    crop = spatial_crop(centers, valid, 48, gen(1))
    assert crop.shape == (48,)
    assert set(np.flatnonzero(valid)) <= set(crop.tolist())
    assert sorted(set(crop.tolist()) - set(np.flatnonzero(valid).tolist())) == list(range(8))


def test_spatial_crop_with_no_resolved_center_falls_back_to_contiguous() -> None:
    centers = np.zeros((20, 3), dtype=np.float32)
    valid = np.zeros(20, dtype=bool)
    crop = spatial_crop(centers, valid, 8, gen(0))
    assert np.array_equal(crop, contiguous_crop(20, 8, gen(0)))


@pytest.mark.parametrize("contiguous_prob", [0.0, 1.0])
def test_crop_chain_returns_a_one_chain_complex(
    parsed: Callable[[str], MolecularComplex], contiguous_prob: float
) -> None:
    c = parsed("4hhb")
    sub = crop_chain(c, 2, 16, contiguous_prob=contiguous_prob, generator=gen(5))
    sub.validate()
    assert sub.num_tokens == 16 and len(sub.chains) == 1 and sub.chains[0].chain_id == "C"
    assert sub.token_asym_id.tolist() == [0] * 16
    assert sub.token_entity_id.tolist() == [0] * 16 and sub.token_sym_id.tolist() == [1] * 16
    assert sub.chains[0].sequence == c.chains[2].sequence  # the full chain sequence survives
    residues = sub.token_residue_index
    if contiguous_prob == 1.0:
        assert np.array_equal(residues, np.arange(residues[0], residues[0] + 16))
    again = crop_chain(c, 2, 16, contiguous_prob=contiguous_prob, generator=gen(5))
    assert np.array_equal(again.token_residue_index, residues)
    other = crop_chain(c, 2, 16, contiguous_prob=contiguous_prob, generator=gen(6))
    assert not np.array_equal(other.token_residue_index, residues)


def test_crop_chain_short_chain_is_whole(parsed: Callable[[str], MolecularComplex]) -> None:
    c = parsed("1l2y")
    sub = crop_chain(c, 0, 384, contiguous_prob=0.5, generator=gen(0))
    assert sub.num_tokens == 20 and sub.num_atoms == c.num_atoms


def test_crop_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="crop_size"):
        contiguous_crop(10, 0, gen(0))
    with pytest.raises(ValueError, match="crop_size"):
        spatial_crop(np.zeros((10, 3), dtype=np.float32), np.ones(10, dtype=bool), -1, gen(0))

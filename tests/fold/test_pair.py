"""Block-local pair features (spec §4.6): block calls equal the sliced full call."""

from __future__ import annotations

import pytest
import torch

from oplm.fold.pair import (
    RelativePositionEncoding,
    distance_bins,
    outer_product_difference,
    outer_sum,
    pair_mask,
    relpos_features,
    token_bond_features,
)


def _indices(L: int = 12) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    asym = torch.tensor([[0] * 5 + [1] * 4 + [2] * 3])
    residue = torch.tensor([list(range(5)) + list(range(4)) + list(range(3))])
    entity = torch.tensor([[0] * 5 + [1] * 4 + [1] * 3])
    sym = torch.tensor([[0] * 5 + [0] * 4 + [1] * 3])
    token = torch.arange(L)[None]
    mask = torch.ones(1, L, dtype=torch.bool)
    mask[0, 10:] = False
    coords = torch.randn(1, L, 3, generator=g) * 10
    bonds = torch.zeros(1, L, L, 1)
    bonds[0, 2, 3, 0] = bonds[0, 3, 2, 0] = 1.0
    return {
        "asym": asym,
        "residue": residue,
        "entity": entity,
        "sym": sym,
        "token": token,
        "mask": mask,
        "coords": coords,
        "bonds": bonds,
    }


def test_relpos_feature_semantics() -> None:
    ix = _indices()
    f = relpos_features(
        ix["residue"], ix["asym"], ix["sym"], ix["entity"], ix["token"], r_max=32, s_max=2
    )
    assert f.shape == (1, 12, 12, 139) and f.dtype == torch.float32
    # same chain, residue delta +1 -> residue bin 33; token delta +1 -> token bin 65
    assert f[0, 1, 0, 33] == 1 and f[0, 1, 0, 66 + 65] == 1
    # different chains: residue bin 65 and token bin 65; chain delta via sym: 0-0+2 = 2 -> bin 2
    assert f[0, 0, 5, 65] == 1 and f[0, 0, 5, 66 + 65] == 1 and f[0, 0, 5, 133 + 2] == 1
    assert f[0, 0, 9, 133 + 1] == 1  # sym 0 vs 1: 0-1+2 = 1 -> bin 1
    # same chain -> chain bin 5; same entity flag at index 132
    assert f[0, 0, 1, 133 + 5] == 1 and f[0, 0, 1, 132] == 1 and f[0, 0, 5, 132] == 0
    assert f[0, 9, 10, 132] == 1  # entity 1 copies 0 and 1
    assert torch.equal(f.sum(-1), torch.full((1, 12, 12), 3.0) + f[..., 132])


@pytest.mark.parametrize("seed", [0, 1])
def test_every_pair_feature_is_block_local(seed: int) -> None:
    ix = _indices()
    g = torch.Generator().manual_seed(seed)
    rows = torch.randperm(12, generator=g)[:5]
    cols = torch.randperm(12, generator=g)[:7]
    torch.manual_seed(seed)
    relpos = RelativePositionEncoding(16, r_max=32, s_max=2)
    boundaries = torch.linspace(3.25, 50.75, 38)
    x = torch.randn(1, 12, 8)
    a = torch.randn(1, 12, 8)
    b = torch.randn(1, 12, 8)
    full_and_block = [
        (
            relpos(ix["residue"], ix["asym"], ix["sym"], ix["entity"], ix["token"]),
            relpos(
                ix["residue"],
                ix["asym"],
                ix["sym"],
                ix["entity"],
                ix["token"],
                rows=rows,
                cols=cols,
            ),
        ),
        (pair_mask(ix["mask"]), pair_mask(ix["mask"], rows=rows, cols=cols)),
        (outer_sum(a, b), outer_sum(a[:, rows], b[:, cols])),
        (outer_product_difference(x), outer_product_difference(x, rows=rows, cols=cols)),
        (token_bond_features(ix["bonds"]), token_bond_features(ix["bonds"], rows=rows, cols=cols)),
        (
            distance_bins(ix["coords"], boundaries),
            distance_bins(ix["coords"], boundaries, rows=rows, cols=cols),
        ),
    ]
    for full, block in full_and_block:
        torch.testing.assert_close(block, full[:, rows][:, :, cols])


def test_distance_bins_are_strict_upper_counts() -> None:
    coords = torch.tensor([[[0.0, 0, 0], [3.25, 0, 0], [4.0, 0, 0], [60.0, 0, 0]]])
    boundaries = torch.linspace(3.25, 50.75, 38)
    bins = distance_bins(coords, boundaries)
    assert bins[0, 0, 0] == 0 and bins[0, 0, 1] == 0  # d <= 3.25 -> bin 0 (strict >)
    assert bins[0, 0, 2] == 1 and bins[0, 0, 3] == 38

"""Contact primitives: which residues can be scored."""

from __future__ import annotations

import torch

from oplm.eval.metrics.contact import contact_positions_valid


def test_contact_positions_valid_flags_any_missing_backbone_atom() -> None:
    """Virtual Cβ needs N, CA and C; CA-only scoring needs just CA."""
    coords = torch.randn(5, 3, 3)
    coords[1, 0] = float("nan")  # N missing
    coords[3, 2] = float("nan")  # C missing
    assert contact_positions_valid(coords).tolist() == [True, False, True, False, True]
    assert contact_positions_valid(coords, use_cbeta=False).tolist() == [True] * 5

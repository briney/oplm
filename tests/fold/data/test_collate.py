"""Batching FoldExamples under the §4.7 atom budget."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from oplm.fold.data.collate import FoldBatch, atom_budget, collate_examples
from oplm.fold.data.featurize import featurize_complex
from oplm.model.tokenization_oplm import VOCAB

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.fold.data.complex import MolecularComplex


def test_atom_budget_rounds_up_to_the_block_multiple() -> None:
    assert atom_budget(384, 24, 32) == 9216
    assert (
        atom_budget(10, 3, 32) == 32
        and atom_budget(16, 2, 32) == 32
        and atom_budget(17, 2, 32) == 64
    )


def test_collate_pads_tokens_atoms_and_lm_rows(parsed: Callable[[str], MolecularComplex]) -> None:
    a = featurize_complex(parsed("1crn"))  # 46 tokens, one chain (row length 48)
    b = featurize_complex(parsed("1l2y"))  # 20 tokens, one chain (row length 22)
    batch = collate_examples([a, b], crop_size=64)
    assert isinstance(batch, FoldBatch) and batch.ids == ["1crn_A", "1l2y_A"]
    f, t = batch.features, batch.targets
    assert f.num_tokens == 64 and f.num_atoms == atom_budget(64, 24, 32) == 1536
    assert f.token_mask.sum(1).tolist() == [46, 20] and f.atom_mask.sum(1).tolist() == [
        int(a.features.atom_mask.sum()),
        int(b.features.atom_mask.sum()),
    ]
    assert f.token_bonds.shape == (2, 64, 64, 1) and f.ref_pos.shape == (2, 1536, 3)
    assert f.ref_atom_name_chars.shape == (2, 1536, 4)
    assert f.lm_input_ids.shape == (2, 48) and f.lm_attention_mask.shape == (2, 48)
    assert (
        f.lm_input_ids[1, 22:].eq(VOCAB["<pad>"]).all() and f.lm_attention_mask[1, 22:].eq(0).all()
    )
    assert torch.equal(f.lm_input_ids[0], a.features.lm_input_ids[0])
    assert f.lm_rows[1, :20].eq(1).all() and f.lm_rows[1, 20:].eq(0).all()  # offset into the stack
    assert f.lm_rows[0].eq(0).all()
    assert f.chain_ids == ["A", "A"] and f.num_chains == 2
    assert (
        torch.equal(f.res_type[0, :46], a.features.res_type[0]) and f.res_type[0, 46:].eq(0).all()
    )
    assert t.coords.shape == (2, 1536, 3) and t.resolved.shape == (2, 1536)
    assert t.frame_atom_idx.shape == (2, 64, 3) and not t.frame_atom_idx[1, 20:].any()
    assert torch.equal(t.coords[1, : b.features.num_atoms], b.targets.coords[0])


def test_collate_of_one_matches_padding_the_example(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    ex = featurize_complex(parsed("1l2y"))
    batch = collate_examples([ex], crop_size=32)
    padded = featurize_complex(parsed("1l2y"), pad_tokens_to=32)
    assert torch.equal(batch.features.residue_index, padded.features.residue_index)
    assert torch.equal(batch.features.ref_pos[:, : ex.features.num_atoms], ex.features.ref_pos)
    assert batch.features.num_atoms == atom_budget(32, 24, 32)


def test_multi_chain_example_rows_offset_by_chain_count(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    two = featurize_complex(parsed("1jb6"))  # chains A and B, 66 tokens
    one = featurize_complex(parsed("1l2y"))
    batch = collate_examples([two, one], crop_size=128)
    assert batch.features.lm_input_ids.shape[0] == 3
    assert batch.features.lm_rows[0, 33] == 1 and batch.features.lm_rows[1, :20].eq(2).all()


def test_over_budget_and_over_crop_examples_are_rejected(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    ex = featurize_complex(parsed("1crn"))
    with pytest.raises(ValueError, match="atom budget"):
        collate_examples([ex], crop_size=64, atoms_per_token_budget=1)
    with pytest.raises(ValueError, match="crop_size"):
        collate_examples([ex], crop_size=32)

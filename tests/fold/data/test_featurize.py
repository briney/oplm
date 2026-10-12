"""Sequences -> FoldFeatures: ids, atoms, padding, LM rows, the X quirk (spec §4.2, §4.5, §4.7)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest
import torch

from oplm.fold.data.ccd import LM_UNKNOWN_ID, UNK_RES_TYPE, ReferenceConformers
from oplm.fold.data.crop import crop_chain
from oplm.fold.data.featurize import (
    ChainSpec,
    FoldExample,
    FoldTargets,
    featurize,
    featurize_complex,
)
from oplm.model.tokenization_oplm import VOCAB

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.fold.data.complex import MolecularComplex

CRAMBIN = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"


def test_monomer_features() -> None:
    f = featurize([ChainSpec("MKV", "A")])
    assert f.num_tokens == 3 and f.num_chains == 1
    assert f.res_type.tolist() == [[14, 13, 21]]  # MET, LYS, VAL
    assert f.token_index.tolist() == [[0, 1, 2]] and f.residue_index.tolist() == [[0, 1, 2]]
    assert f.asym_id.tolist() == [[0, 0, 0]] and f.mol_type.tolist() == [[0, 0, 0]]
    assert f.token_mask.all()
    assert f.num_atoms == 32  # 8 + 9 + 7 = 24 real atoms padded to a multiple of 32
    assert f.atom_mask[0].sum() == 24 and not f.atom_mask[0, 24:].any()
    assert f.atom_to_token[0, :24].tolist() == [0] * 8 + [1] * 9 + [2] * 7
    assert f.atom_to_token[0, 24:].eq(0).all()
    assert f.ref_space_uid[0, :24].tolist() == f.atom_to_token[0, :24].tolist()
    assert f.distogram_atom_idx.tolist() == [[4, 8 + 4, 17 + 4]]  # CB of each residue
    assert f.ref_element[0, :8].tolist() == [7, 6, 6, 8, 6, 6, 16, 6]  # MET: N CA C O CB CG SD CE
    assert f.ref_charge[0, 8 + 8] == 1  # LYS NZ
    assert f.ref_atom_name_chars[0, 1].tolist() == [35, 33, 0, 0]  # "CA"
    assert torch.equal(f.ref_pos[0, 24:], torch.zeros(8, 3))
    assert f.token_bonds.shape == (1, 3, 3, 1) and not f.token_bonds.any()
    assert f.lm_input_ids.tolist() == [[0, VOCAB["M"], VOCAB["K"], VOCAB["V"], 2]]
    assert f.lm_attention_mask.all()
    assert f.lm_rows.tolist() == [[0, 0, 0]] and f.lm_positions.tolist() == [[1, 2, 3]]


def test_heterodimer_and_homodimer_numbering() -> None:
    f = featurize([ChainSpec("AG", "A"), ChainSpec("GG", "B", copies=2)])
    assert f.num_chains == 3 and f.num_tokens == 6
    assert f.asym_id.tolist() == [[0, 0, 1, 1, 2, 2]]
    assert f.entity_id.tolist() == [[0, 0, 1, 1, 1, 1]]
    assert f.sym_id.tolist() == [[0, 0, 0, 0, 1, 1]]
    assert f.residue_index.tolist() == [[0, 1, 0, 1, 0, 1]]
    assert f.chain_ids == ["A", "B", "B_2"]
    assert f.lm_input_ids.shape == (3, 4)  # three rows, [BOS a g EOS]
    assert f.lm_rows.tolist() == [[0, 0, 1, 1, 2, 2]] and f.lm_positions.tolist() == [
        [1, 2, 1, 2, 1, 2]
    ]


def test_lm_rows_pad_to_the_longest_chain() -> None:
    f = featurize([ChainSpec("MKVLA", "A"), ChainSpec("GG", "B")])
    assert f.lm_input_ids.shape == (2, 7)
    assert f.lm_input_ids[1].tolist() == [0, VOCAB["G"], VOCAB["G"], 2, 1, 1, 1]
    assert f.lm_attention_mask[1].tolist() == [1, 1, 1, 1, 0, 0, 0]


def test_unknown_residue_tokenization_and_bond_quirk() -> None:
    f = featurize([ChainSpec("AXG", "A")])
    assert f.res_type[0, 1] == UNK_RES_TYPE and f.lm_input_ids[0, 2] == LM_UNKNOWN_ID
    assert f.atom_to_token[0].tolist()[:13] == [0] * 5 + [1] * 4 + [2] * 4
    assert f.distogram_atom_idx[0, 1] == 5 + 1  # UNK has no CB -> CA
    bonds = f.token_bonds[0, :, :, 0]
    assert bonds[1, 0] == bonds[0, 1] == bonds[1, 2] == bonds[2, 1] == 1  # the upstream X quirk
    assert bonds[0, 2] == 0


def test_lowercase_and_unlisted_letters_are_unknown() -> None:
    f = featurize([ChainSpec("aBZ", "A")])
    assert (f.res_type == UNK_RES_TYPE).all()


def test_token_padding_and_atom_budget() -> None:
    f = featurize([ChainSpec("MKV", "A")], pad_tokens_to=8)
    assert f.num_tokens == 8 and f.token_mask[0].tolist() == [True] * 3 + [False] * 5
    assert f.res_type[0, 3:].eq(0).all() and f.asym_id[0, 3:].eq(0).all()
    assert f.lm_rows[0, 3:].eq(0).all()
    with pytest.raises(ValueError, match="pad_tokens_to"):
        featurize([ChainSpec("MKV", "A")], pad_tokens_to=2)


def test_to_device_and_custom_conformers() -> None:
    f = featurize([ChainSpec("G", "A")], conformers=ReferenceConformers.load()).to("cpu")
    assert f.ref_pos.device.type == "cpu" and f.num_atoms == 32


@pytest.mark.parametrize(
    "chains",
    [
        [ChainSpec("MKV", "B"), ChainSpec("GG", "B")],
        [ChainSpec("GG", "A", copies=2), ChainSpec("MKV", "A_2")],  # copy suffix collides
    ],
)
def test_duplicate_chain_ids_raise(chains: list[ChainSpec]) -> None:
    with pytest.raises(ValueError, match="duplicate chain id"):
        featurize(chains)


def test_full_complex_features_match_sequence_featurize(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    example = featurize_complex(parsed("1crn"))
    assert isinstance(example, FoldExample) and example.id == "1crn_A"
    expected = featurize([ChainSpec(CRAMBIN, "A")])
    for f in dataclasses.fields(expected):
        a, b = getattr(example.features, f.name), getattr(expected, f.name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), f.name
        else:
            assert a == b, f.name


def test_targets_line_up_with_the_atom_axis(parsed: Callable[[str], MolecularComplex]) -> None:
    c = parsed("2ds7")
    ex = featurize_complex(c)
    t = ex.targets
    n, A = c.num_atoms, ex.features.num_atoms
    assert isinstance(t, FoldTargets)
    assert t.coords.shape == (1, A, 3) and t.resolved.shape == (1, A) and t.bfactor.shape == (1, A)
    assert torch.equal(t.coords[0, :n], torch.from_numpy(c.atom_coords))
    assert torch.equal(t.resolved[0, :n], torch.from_numpy(c.atom_resolved))
    assert torch.equal(t.bfactor[0, :n], torch.from_numpy(c.atom_bfactor))
    assert not t.coords[0, n:].any() and not t.resolved[0, n:].any()
    assert t.frame_atom_idx.shape == (1, c.num_tokens, 3)
    starts = torch.from_numpy(c.token_atom_start).long()
    assert torch.equal(t.frame_atom_idx[0], torch.stack([starts, starts + 1, starts + 2], dim=1))
    assert not t.resolved[0, t.frame_atom_idx[0, 0]].any()  # unmodeled residue: no frame
    assert t.resolved[0, t.frame_atom_idx[0, 20]].all()
    moved = t.to("cpu")
    assert moved.coords.device.type == "cpu"


def test_conformer_rotations_are_per_token_rigid_and_seeded(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    c = parsed("1crn")
    raw = featurize_complex(c).features.ref_pos[0]
    g = torch.Generator().manual_seed(7)
    rot = featurize_complex(c, generator=g).features.ref_pos[0]
    assert not torch.allclose(raw, rot)
    for token in (0, 1, 19):
        start, count = int(c.token_atom_start[token]), int(c.token_atom_count[token])
        a, b = raw[start : start + count], rot[start : start + count]
        assert torch.allclose(torch.cdist(a, a), torch.cdist(b, b), atol=1e-4)  # rigid
        assert torch.allclose(b.mean(0), torch.zeros(3), atol=1e-4)  # centered
    thr1, thr2 = rot[:7], rot[7:14]  # two THR tokens get different rotations
    assert not torch.allclose(thr1, thr2, atol=1e-3)
    again = featurize_complex(c, generator=torch.Generator().manual_seed(7)).features.ref_pos[0]
    assert torch.equal(rot, again)
    assert torch.equal(featurize_complex(c).features.ref_pos[0], raw)  # no generator: raw


def test_cropped_chain_keeps_the_full_lm_row(parsed: Callable[[str], MolecularComplex]) -> None:
    c = parsed("2ds7")
    sub = crop_chain(c, 0, 16, contiguous_prob=1.0, generator=torch.Generator().manual_seed(0))
    ex = featurize_complex(sub)
    f = ex.features
    assert f.num_tokens == 16 and f.num_chains == 1 and f.chain_ids == ["A"]
    assert f.lm_input_ids.shape == (1, 53)  # BOS + 51 residues + EOS
    assert f.lm_input_ids[0, 1 + 42] == VOCAB["M"]  # MSE position reads as M
    assert torch.equal(f.lm_positions, f.residue_index + 1)
    assert f.asym_id.eq(0).all() and f.lm_rows.eq(0).all()
    assert ex.id == "2ds7_A"


def test_padding_and_multi_chain_ids(parsed: Callable[[str], MolecularComplex]) -> None:
    ex = featurize_complex(parsed("1jb6"), pad_tokens_to=128)
    assert ex.features.num_tokens == 128 and ex.features.token_mask[0].sum() == 66
    assert ex.features.lm_input_ids.shape == (2, 35) and ex.features.lm_rows[0, 33] == 1
    assert ex.targets.frame_atom_idx.shape == (1, 128, 3)
    assert not ex.targets.frame_atom_idx[0, 66:].any()
    assert ex.id == "1jb6_A_B"


def test_atom_names_must_match_the_template(parsed: Callable[[str], MolecularComplex]) -> None:
    c = dataclasses.replace(parsed("1crn"))
    c.atom_name = list(c.atom_name)
    c.atom_name[5] = "ZZ"
    with pytest.raises(ValueError, match="template"):
        featurize_complex(c)

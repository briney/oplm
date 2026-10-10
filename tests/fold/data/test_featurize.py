"""Sequences -> FoldFeatures: ids, atoms, padding, LM rows, the X quirk (spec §4.2, §4.5, §4.7)."""

from __future__ import annotations

import pytest
import torch

from oplm.fold.data.ccd import LM_UNKNOWN_ID, UNK_RES_TYPE, ReferenceConformers
from oplm.fold.data.featurize import ChainSpec, featurize
from oplm.model.tokenization_oplm import VOCAB


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

"""fold(): ranking, result layout, mmCIF round trip through gemmi; chain-argument parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from oplm.fold import ChainSpec, OplmForFolding, featurize
from oplm.fold.predict import FoldResult, fold, parse_chain_arg, rank_samples, write_mmcif
from tests.fold.helpers import tiny_fold_config, tiny_lm

if TYPE_CHECKING:
    from pathlib import Path

gemmi = pytest.importorskip("gemmi")


def _model() -> OplmForFolding:
    torch.manual_seed(0)
    cfg = tiny_fold_config(
        plddt_bins=10,
        pae_bins=8,
        pde_bins=8,
        confidence_dist_bins=5,
        distogram_bins=8,
        inference_num_steps=2,
        inference_num_loops=1,
        lm_hidden_size=32,
        lm_num_hidden_states=3,
    )
    model = OplmForFolding(cfg).eval()
    model.attach_lm(tiny_lm())
    return model


def test_parse_chain_arg() -> None:
    assert parse_chain_arg("MKV", "A") == ChainSpec("MKV", "A")
    assert parse_chain_arg("B:GG", "A") == ChainSpec("GG", "B")
    assert parse_chain_arg("B:GG*3", "A") == ChainSpec("GG", "B", copies=3)
    with pytest.raises(ValueError, match="copies"):
        parse_chain_arg("GG*0", "A")


def test_rank_samples_uses_iptm_for_complexes_and_ptm_for_monomers() -> None:
    model = _model()
    f = featurize([ChainSpec("MKV", "A"), ChainSpec("GG", "B")])
    with torch.no_grad():
        out = model(f, num_samples=3, generator=torch.Generator().manual_seed(0))
    out.confidence.iptm = torch.tensor([0.1, 0.9, 0.5])
    out.confidence.ptm = torch.tensor([0.9, 0.1, 0.5])
    assert rank_samples(out, num_chains=2).tolist() == [1, 2, 0]
    assert rank_samples(out, num_chains=1).tolist() == [0, 2, 1]


def test_fold_result_layout_and_determinism() -> None:
    model = _model()
    chains = [ChainSpec("MKV", "A"), ChainSpec("GG", "B", copies=2)]
    a = fold(model, chains, num_samples=2, seed=1)
    b = fold(model, chains, num_samples=2, seed=1)
    assert isinstance(a, FoldResult) and a.chain_ids == ["A", "B", "B_2"]
    assert a.coords.shape == (24 + 2 * 8, 3)  # MKV: 8+9+7 atoms; each GG copy: 2 x 4
    assert a.plddt_per_atom.shape == (40,) and a.plddt.shape == (7,)
    assert a.atom_names[:5] == ["N", "CA", "C", "O", "CB"]
    assert a.residue_names[:3] == ["MET", "LYS", "VAL"]
    assert a.atom_chain_index.tolist() == [0] * 24 + [1] * 8 + [2] * 8
    assert 0.0 <= a.ptm <= 1.0 and 0.0 <= a.iptm <= 1.0 and a.pae.shape == (7, 7)
    assert a.best_sample in (0, 1) and a.output.coords.shape[0] == 2
    torch.testing.assert_close(a.coords, b.coords)


def test_write_mmcif_round_trips_through_gemmi(tmp_path: Path) -> None:
    model = _model()
    result = fold(model, [ChainSpec("MKV", "A"), ChainSpec("GG", "B")], seed=0)
    path = write_mmcif(result, tmp_path / "pred.cif", name="unit")
    st = gemmi.read_structure(str(path))
    assert st.name.lower() == "unit" and len(st) == 1
    chains = {ch.name: ch for ch in st[0]}
    assert set(chains) == {"A", "B"}
    assert [r.name for r in chains["A"]] == ["MET", "LYS", "VAL"] and len(chains["B"]) == 2
    atoms = [a for ch in st[0] for r in ch for a in r]
    assert len(atoms) == 24 + 8
    assert all(0.0 <= a.b_iso <= 100.0 for a in atoms)
    assert atoms[0].name == "N" and atoms[0].element.name == "N"
    assert chains["A"][0].seqid.num == 1 and chains["B"][1].seqid.num == 2
    assert atoms[0].b_iso == pytest.approx(100 * float(result.plddt_per_atom[0]), abs=1e-3)


def test_fold_unpads_padded_tokens() -> None:
    model = _model()
    result = fold(model, [ChainSpec("MKV", "A"), ChainSpec("GG", "B")], pad_to_multiple=8, seed=0)
    assert result.features.num_tokens == 8
    assert result.coords.shape == (32, 3) and result.plddt_per_atom.shape == (32,)
    assert result.plddt.shape == (5,) and result.pae.shape == (5, 5)
    assert result.residue_names == ["MET", "LYS", "VAL", "GLY", "GLY"]


def test_fold_rejects_duplicate_chain_ids() -> None:
    with pytest.raises(ValueError, match="duplicate chain id"):
        fold(_model(), [ChainSpec("MKV", "B"), ChainSpec("GG", "B")])


def test_fold_requires_eval_mode() -> None:
    with pytest.raises(ValueError, match="eval-mode"):
        fold(_model().train(), [ChainSpec("MKV", "A")])

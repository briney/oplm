"""Tests for structure loading from PDB/CIF files."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

from oplm.eval.data.structure_loader import StructureData, load_structures


class TestLoadStructures:
    """Test load_structures() with real PDB fixtures."""

    def test_loads_pdb_files(self, structure_fixtures_dir: Path) -> None:
        structures = load_structures(structure_fixtures_dir)
        assert len(structures) >= 3

    def test_returns_structure_data(self, structure_fixtures_dir: Path) -> None:
        structures = load_structures(structure_fixtures_dir)
        for s in structures:
            assert isinstance(s, StructureData)

    def test_sequence_is_valid(self, structure_fixtures_dir: Path) -> None:
        valid_chars = set("ACDEFGHIKLMNPQRSTVWYX")
        structures = load_structures(structure_fixtures_dir)
        for s in structures:
            assert len(s.sequence) > 0
            assert all(c in valid_chars for c in s.sequence), (
                f"{s.name}: invalid chars in sequence: {set(s.sequence) - valid_chars}"
            )

    def test_coords_shape(self, structure_fixtures_dir: Path) -> None:
        structures = load_structures(structure_fixtures_dir)
        for s in structures:
            seq_len = len(s.sequence)
            assert s.coords.shape == (seq_len, 3, 3), (
                f"{s.name}: expected ({seq_len}, 3, 3), got {s.coords.shape}"
            )

    def test_coords_reasonable_bond_distances(self, structure_fixtures_dir: Path) -> None:
        """Backbone bond distances should be within physical bounds."""
        structures = load_structures(structure_fixtures_dir)
        for s in structures:
            n = s.coords[:, 0, :]  # N atoms
            ca = s.coords[:, 1, :]  # CA atoms
            c = s.coords[:, 2, :]  # C atoms

            # N-CA bond: ~1.47 Å, allow 1.2-1.7
            n_ca_dist = torch.norm(ca - n, dim=-1)
            valid = ~torch.isnan(n_ca_dist)
            if valid.any():
                assert (n_ca_dist[valid] > 1.2).all(), f"{s.name}: N-CA bond too short"
                assert (n_ca_dist[valid] < 1.7).all(), f"{s.name}: N-CA bond too long"

            # CA-C bond: ~1.52 Å, allow 1.3-1.8
            ca_c_dist = torch.norm(c - ca, dim=-1)
            valid = ~torch.isnan(ca_c_dist)
            if valid.any():
                assert (ca_c_dist[valid] > 1.3).all(), f"{s.name}: CA-C bond too short"
                assert (ca_c_dist[valid] < 1.8).all(), f"{s.name}: CA-C bond too long"

    def test_deterministic_ordering(self, structure_fixtures_dir: Path) -> None:
        s1 = load_structures(structure_fixtures_dir)
        s2 = load_structures(structure_fixtures_dir)
        assert [s.name for s in s1] == [s.name for s in s2]

    def test_max_structures(self, structure_fixtures_dir: Path) -> None:
        structures = load_structures(structure_fixtures_dir, max_structures=1)
        assert len(structures) == 1

    def test_chain_id_populated(self, structure_fixtures_dir: Path) -> None:
        structures = load_structures(structure_fixtures_dir)
        for s in structures:
            assert s.chain_id is not None

    def test_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_structures(tmp_path / "nonexistent")

    def test_empty_directory(self, tmp_path: Path) -> None:
        structures = load_structures(tmp_path)
        assert structures == []

    def test_skips_unparseable_files(self, structure_fixtures_dir: Path, tmp_path: Path) -> None:
        """A corrupt file should be skipped with a warning, not crash the loader."""
        # Copy real fixtures to temp dir
        test_dir = tmp_path / "structures"
        shutil.copytree(structure_fixtures_dir, test_dir)

        # Add a corrupt PDB file
        corrupt = test_dir / "CORRUPT.pdb"
        corrupt.write_text("this is not a valid PDB file\n")

        structures = load_structures(test_dir)
        names = {s.name for s in structures}
        assert "CORRUPT" not in names
        assert len(structures) >= 3  # Real structures still loaded


class TestKnownStructures:
    """Tests against known structure properties."""

    def test_crambin_1crn(self, structure_fixtures_dir: Path) -> None:
        """Crambin (1CRN) is 46 residues, well-resolved."""
        structures = load_structures(structure_fixtures_dir)
        by_name = {s.name: s for s in structures}
        assert "1CRN" in by_name
        crn = by_name["1CRN"]
        assert len(crn.sequence) == 46
        assert crn.sequence.startswith("TTCCPSI")

    def test_trp_cage_1l2y(self, structure_fixtures_dir: Path) -> None:
        """Trp-cage (1L2Y) is 20 residues, NMR structure."""
        structures = load_structures(structure_fixtures_dir)
        by_name = {s.name: s for s in structures}
        assert "1L2Y" in by_name
        l2y = by_name["1L2Y"]
        assert len(l2y.sequence) == 20

    def test_no_nan_in_well_resolved_structures(self, structure_fixtures_dir: Path) -> None:
        """Well-resolved test structures should have all backbone atoms."""
        structures = load_structures(structure_fixtures_dir)
        for s in structures:
            assert not torch.isnan(s.coords).any(), f"{s.name}: unexpected NaN in backbone coords"

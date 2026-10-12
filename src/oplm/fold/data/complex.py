"""``MolecularComplex``: the flat all-atom representation (spec §4.2) and its Parquet row.

Atoms and tokens are flat arrays, never a per-residue layout. The atoms of one token are
contiguous and in that token's template order (``ReferenceConformers``), so training
coordinates line up with the ``ref_*`` features by construction. Bonds are empty in v1 but
always present. ``complex_to_row``/``rows_to_table``/``row_to_complex`` are the shard contract:
one row per entry, ragged list columns, every array flattened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pyarrow as pa

__all__ = [
    "COMPLEX_SCHEMA",
    "ChainInfo",
    "ComplexMetadata",
    "MolecularComplex",
    "complex_to_row",
    "row_to_complex",
    "rows_to_table",
]


@dataclass
class ChainInfo:
    """One polymer chain: identity, entity and symmetry copy, and its full sequence."""

    chain_id: str  # label_asym_id
    auth_chain_id: str  # auth_asym_id
    entity_id: int  # index among the kept polymer entities of the entry
    sym_id: int  # copy index within the entity
    mol_type: int  # MOL_TYPE_* from ccd.py
    sequence: str  # the entity polymer sequence, one letter per token, X for unknown
    mean_plddt: float | None = None  # predicted structures only


@dataclass
class ComplexMetadata:
    """Entry-level provenance and the §4.4 dropped-entity flags."""

    id: str
    source: str
    method: str
    resolution: float | None
    release_date: str | None  # ISO date
    dropped: dict[str, int] = field(default_factory=dict)  # kind -> count of entities/chains


@dataclass
class MolecularComplex:
    """Flat atoms, flat tokens, bonds, chains and metadata (spec §4.2)."""

    atom_coords: np.ndarray  # (A, 3) float32; zeros where unresolved
    atom_resolved: np.ndarray  # (A,) bool
    atom_element: np.ndarray  # (A,) uint8 atomic number
    atom_name: list[str]  # (A,)
    atom_charge: np.ndarray  # (A,) int8 formal charge
    atom_bfactor: np.ndarray  # (A,) float32 B-factor, or pLDDT for predicted structures
    atom_to_token: np.ndarray  # (A,) int32
    token_res_type: np.ndarray  # (L,) int8 (PROTEIN_RES_TYPES values)
    token_mol_type: np.ndarray  # (L,) int8
    token_asym_id: np.ndarray  # (L,) int32 index into ``chains``
    token_entity_id: np.ndarray  # (L,) int32
    token_sym_id: np.ndarray  # (L,) int32
    token_residue_index: np.ndarray  # (L,) int32, 0-based within the chain sequence
    token_center_atom: np.ndarray  # (L,) int32 flat atom index (CA for protein)
    token_rep_atom: np.ndarray  # (L,) int32 flat atom index (CB; CA for GLY and UNK)
    token_atom_start: np.ndarray  # (L,) int32
    token_atom_count: np.ndarray  # (L,) int32
    bonds: np.ndarray  # (N, 2) int32 flat atom pairs; empty in v1
    chains: list[ChainInfo]
    metadata: ComplexMetadata

    @property
    def num_atoms(self) -> int:
        return int(self.atom_to_token.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.token_res_type.shape[0])

    def chain_tokens(self, chain: int) -> np.ndarray:
        """Token indices of chain ``chain`` (an index into ``chains``), ascending."""
        return np.flatnonzero(self.token_asym_id == chain)

    def validate(self) -> None:
        """Check the index invariants every consumer relies on.

        Raises:
            ValueError: An atom or token array has the wrong length, token atom ranges are not
                contiguous from 0, ``atom_to_token`` disagrees with the ranges, a center or
                representative atom lies outside its token's range, an ``asym_id`` is outside
                ``chains``, or ``bonds`` is not ``(N, 2)``.
        """
        n_atoms, n_tokens = self.num_atoms, self.num_tokens
        atom_arrays = (
            self.atom_coords,
            self.atom_resolved,
            self.atom_element,
            self.atom_charge,
            self.atom_bfactor,
        )
        if len(self.atom_name) != n_atoms or any(a.shape[0] != n_atoms for a in atom_arrays):
            raise ValueError("atom arrays disagree on the atom count")
        token_arrays = (
            self.token_mol_type,
            self.token_asym_id,
            self.token_entity_id,
            self.token_sym_id,
            self.token_residue_index,
            self.token_center_atom,
            self.token_rep_atom,
            self.token_atom_start,
            self.token_atom_count,
        )
        if any(a.shape[0] != n_tokens for a in token_arrays):
            raise ValueError("token arrays disagree on the token count")
        ends = self.token_atom_start + self.token_atom_count
        expected_starts = np.concatenate([[0], ends[:-1]]) if n_tokens else np.zeros(0)
        if not np.array_equal(self.token_atom_start, expected_starts) or (
            n_tokens and ends[-1] != n_atoms
        ):
            raise ValueError("token atom ranges are not contiguous over the atom axis")
        expected = np.repeat(np.arange(n_tokens, dtype=np.int32), self.token_atom_count)
        if not np.array_equal(self.atom_to_token, expected):
            raise ValueError("atom_to_token does not match the token atom ranges")
        for label, idx in (
            ("center", self.token_center_atom),
            ("representative", self.token_rep_atom),
        ):
            if n_tokens and np.any((idx < self.token_atom_start) | (idx >= ends)):
                raise ValueError(f"{label} atom outside its token's atom range")
        if n_tokens and (
            self.token_asym_id.min() < 0 or self.token_asym_id.max() >= len(self.chains)
        ):
            raise ValueError("token asym_id outside the chain list")
        if self.bonds.ndim != 2 or self.bonds.shape[1] != 2:
            raise ValueError("bonds must have shape (N, 2)")

    def select_tokens(self, tokens: np.ndarray) -> MolecularComplex:
        """The sub-complex of ``tokens``: atoms remapped, chains compacted to those present.

        Residue indices, entity ids, symmetry ids and chain identities are preserved;
        ``token_asym_id`` is renumbered densely over the chains that remain, in their original
        order. Bonds with an endpoint outside the selection are dropped.
        """
        tokens = np.unique(np.asarray(tokens, dtype=np.int64))
        counts = self.token_atom_count[tokens].astype(np.int32)
        atom_idx = np.concatenate(
            [
                np.arange(s, s + c)
                for s, c in zip(self.token_atom_start[tokens], counts, strict=True)
            ]
            or [np.zeros(0, dtype=np.int64)]
        ).astype(np.int64)
        starts = np.cumsum(counts, dtype=np.int32) - counts
        present = np.unique(self.token_asym_id[tokens])
        asym_map = np.full(len(self.chains), -1, dtype=np.int32)
        asym_map[present] = np.arange(present.size, dtype=np.int32)
        atom_map = np.full(self.num_atoms, -1, dtype=np.int64)
        atom_map[atom_idx] = np.arange(atom_idx.size)
        keep_bonds = (
            np.all(np.isin(self.bonds, atom_idx), axis=1)
            if self.bonds.size
            else np.zeros(0, dtype=bool)
        )
        return MolecularComplex(
            atom_coords=self.atom_coords[atom_idx],
            atom_resolved=self.atom_resolved[atom_idx],
            atom_element=self.atom_element[atom_idx],
            atom_name=[self.atom_name[i] for i in atom_idx],
            atom_charge=self.atom_charge[atom_idx],
            atom_bfactor=self.atom_bfactor[atom_idx],
            atom_to_token=np.repeat(np.arange(tokens.size, dtype=np.int32), counts),
            token_res_type=self.token_res_type[tokens],
            token_mol_type=self.token_mol_type[tokens],
            token_asym_id=asym_map[self.token_asym_id[tokens]],
            token_entity_id=self.token_entity_id[tokens],
            token_sym_id=self.token_sym_id[tokens],
            token_residue_index=self.token_residue_index[tokens],
            token_center_atom=atom_map[self.token_center_atom[tokens]].astype(np.int32),
            token_rep_atom=atom_map[self.token_rep_atom[tokens]].astype(np.int32),
            token_atom_start=starts,
            token_atom_count=counts,
            bonds=atom_map[self.bonds[keep_bonds]].astype(np.int32).reshape(-1, 2),
            chains=[self.chains[int(c)] for c in present],
            metadata=self.metadata,
        )


COMPLEX_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("source", pa.string()),
        ("method", pa.string()),
        ("resolution", pa.float32()),
        ("release_date", pa.string()),
        ("dropped_json", pa.string()),
        ("chain_ids", pa.list_(pa.string())),
        ("chain_auth_ids", pa.list_(pa.string())),
        ("chain_entity_ids", pa.list_(pa.int32())),
        ("chain_sym_ids", pa.list_(pa.int32())),
        ("chain_mol_types", pa.list_(pa.int8())),
        ("chain_sequences", pa.list_(pa.string())),
        ("chain_mean_plddt", pa.list_(pa.float32())),
        ("atom_coords", pa.list_(pa.float32())),  # flattened (A*3)
        ("atom_resolved", pa.list_(pa.bool_())),
        ("atom_element", pa.list_(pa.uint8())),
        ("atom_name", pa.list_(pa.string())),
        ("atom_charge", pa.list_(pa.int8())),
        ("atom_bfactor", pa.list_(pa.float32())),
        ("atom_to_token", pa.list_(pa.int32())),
        ("token_res_type", pa.list_(pa.int8())),
        ("token_mol_type", pa.list_(pa.int8())),
        ("token_asym_id", pa.list_(pa.int32())),
        ("token_entity_id", pa.list_(pa.int32())),
        ("token_sym_id", pa.list_(pa.int32())),
        ("token_residue_index", pa.list_(pa.int32())),
        ("token_center_atom", pa.list_(pa.int32())),
        ("token_rep_atom", pa.list_(pa.int32())),
        ("token_atom_start", pa.list_(pa.int32())),
        ("token_atom_count", pa.list_(pa.int32())),
        ("bond_atoms", pa.list_(pa.int32())),  # flattened (N*2)
    ]
)

_ARRAY_DTYPES: dict[str, type] = {
    "atom_resolved": np.bool_,
    "atom_element": np.uint8,
    "atom_charge": np.int8,
    "atom_bfactor": np.float32,
    "atom_to_token": np.int32,
    "token_res_type": np.int8,
    "token_mol_type": np.int8,
    "token_asym_id": np.int32,
    "token_entity_id": np.int32,
    "token_sym_id": np.int32,
    "token_residue_index": np.int32,
    "token_center_atom": np.int32,
    "token_rep_atom": np.int32,
    "token_atom_start": np.int32,
    "token_atom_count": np.int32,
}


def complex_to_row(c: MolecularComplex) -> dict[str, object]:
    """One shard row (column name -> value) for ``c``; arrays stay numpy until `rows_to_table`."""
    m = c.metadata
    row: dict[str, object] = {
        "id": m.id,
        "source": m.source,
        "method": m.method,
        "resolution": m.resolution,
        "release_date": m.release_date,
        "dropped_json": json.dumps(m.dropped, sort_keys=True),
        "chain_ids": [ch.chain_id for ch in c.chains],
        "chain_auth_ids": [ch.auth_chain_id for ch in c.chains],
        "chain_entity_ids": [ch.entity_id for ch in c.chains],
        "chain_sym_ids": [ch.sym_id for ch in c.chains],
        "chain_mol_types": [ch.mol_type for ch in c.chains],
        "chain_sequences": [ch.sequence for ch in c.chains],
        "chain_mean_plddt": [ch.mean_plddt for ch in c.chains],
        "atom_coords": c.atom_coords.astype(np.float32).reshape(-1),
        "atom_name": list(c.atom_name),
        "bond_atoms": c.bonds.astype(np.int32).reshape(-1),
    }
    for name, dtype in _ARRAY_DTYPES.items():
        row[name] = getattr(c, name).astype(dtype)
    return row


def rows_to_table(rows: list[dict[str, object]]) -> pa.Table:
    """Assemble rows from :func:`complex_to_row` into a table with :data:`COMPLEX_SCHEMA`."""
    columns = [pa.array([row[f.name] for row in rows], type=f.type) for f in COMPLEX_SCHEMA]
    return pa.Table.from_arrays(columns, schema=COMPLEX_SCHEMA)


def row_to_complex(row: dict[str, object]) -> MolecularComplex:
    """Rebuild a complex from one ``Table.to_pylist()`` row."""
    arrays = {name: np.asarray(row[name], dtype=dtype) for name, dtype in _ARRAY_DTYPES.items()}
    chains = [
        ChainInfo(cid, auth, int(ent), int(sym), int(mol), seq, plddt)
        for cid, auth, ent, sym, mol, seq, plddt in zip(
            cast("list[str]", row["chain_ids"]),
            cast("list[str]", row["chain_auth_ids"]),
            cast("list[int]", row["chain_entity_ids"]),
            cast("list[int]", row["chain_sym_ids"]),
            cast("list[int]", row["chain_mol_types"]),
            cast("list[str]", row["chain_sequences"]),
            cast("list[float | None]", row["chain_mean_plddt"]),
            strict=True,
        )
    ]
    resolution = cast("float | None", row["resolution"])
    return MolecularComplex(
        atom_coords=np.asarray(row["atom_coords"], dtype=np.float32).reshape(-1, 3),
        atom_name=cast("list[str]", row["atom_name"]),
        bonds=np.asarray(row["bond_atoms"], dtype=np.int32).reshape(-1, 2),
        chains=chains,
        metadata=ComplexMetadata(
            id=str(row["id"]),
            source=str(row["source"]),
            method=str(row["method"]),
            resolution=None if resolution is None else float(resolution),
            release_date=None if row["release_date"] is None else str(row["release_date"]),
            dropped=json.loads(str(row["dropped_json"])),
        ),
        **arrays,
    )

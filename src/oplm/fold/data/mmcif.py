"""wwPDB mmCIF -> :class:`MolecularComplex` with the §4.4 filters applied at parse time.

gemmi reads the (gzipped) file. Protein polymer chains of the asymmetric unit are kept;
waters, ligands, nucleic-acid and other polymer chains are dropped and counted in
``ComplexMetadata.dropped``. The token sequence is the entity polymer sequence, so unresolved
residues are tokens with ``atom_resolved`` false. First model only; per residue the
highest-occupancy altloc per atom name (ties keep the first); hydrogens dropped;
microheterogeneity keeps the entity sequence's first alternative. Non-standard residues
collapse to their CCD parent's token and atom set, matched by atom name, with AlphaFold's
selenomethionine convention (``MSE`` ``SE`` -> ``SD``); parent atoms absent from the modified
residue are unresolved. Bioassembly expansion is deferred (spec §4.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import gemmi
import numpy as np

from oplm.fold.data.ccd import (
    MOL_TYPE_PROTEIN,
    PROTEIN_3TO1,
    PROTEIN_RES_TYPES,
    ReferenceConformers,
)
from oplm.fold.data.complex import ChainInfo, ComplexMetadata, MolecularComplex

if TYPE_CHECKING:
    from oplm.fold.data.ccd import ChemCompTable

__all__ = ["MAX_RESOLUTION", "MIN_RESOLVED_RESIDUES", "PARSE_VERSION", "Rejection", "parse_mmcif"]

PARSE_VERSION = 1  # bump when the parse changes what a shard row contains
MAX_RESOLUTION = 9.0  # Å, spec §4.4
MIN_RESOLVED_RESIDUES = 4  # spec §4.4
_ATOM_NAME_REMAP = {("MSE", "SE"): "SD"}  # AlphaFold's selenomethionine convention
_DROPPED_KIND = {
    gemmi.EntityType.Water: "water",
    gemmi.EntityType.NonPolymer: "ligand",
    gemmi.EntityType.Branched: "ligand",
    gemmi.EntityType.Unknown: "unknown",
}
_NUCLEIC = {gemmi.PolymerType.Dna, gemmi.PolymerType.Rna, gemmi.PolymerType.DnaRnaHybrid}


@dataclass(frozen=True)
class Rejection:
    """Why an entry produced no complex; ``reason`` is counted by ``oplm fold preprocess``."""

    id: str
    reason: str  # parse_error | resolution | no_protein_chains | no_resolved_chain
    detail: str = ""


def parse_mmcif(
    path: Path,
    *,
    ccd: ChemCompTable,
    conformers: ReferenceConformers | None = None,
    max_resolution: float = MAX_RESOLUTION,
    min_resolved_residues: int = MIN_RESOLVED_RESIDUES,
) -> MolecularComplex | Rejection:
    """Parse one entry.

    Args:
        path: ``.cif`` or ``.cif.gz`` file.
        ccd: Chemical component table for the parent mapping of non-standard residues.
        conformers: Residue templates (atom order and elements); the packaged table when ``None``.
        max_resolution: Reject entries with a worse (larger) resolution; NMR has none and passes.
        min_resolved_residues: Drop chains with fewer residues that have at least one atom.

    Returns:
        The complex, or a :class:`Rejection` naming the reason.
    """
    table = conformers or ReferenceConformers.load()
    try:
        block = gemmi.cif.read(str(path)).sole_block()
        st = gemmi.make_structure_from_block(block)
        st.setup_entities()
        st.remove_hydrogens()
    except Exception as exc:  # gemmi raises RuntimeError/ValueError; nothing else to do with them
        return Rejection(Path(path).name, "parse_error", str(exc))

    entry_id = (st.name or block.name).lower()
    # gemmi's InfoMap supports ``in`` and ``[]`` but has no ``.get``.
    method = st.info["_exptl.method"] if "_exptl.method" in st.info else "UNKNOWN"  # noqa: SIM401
    resolution = float(st.resolution) if st.resolution > 0 else None
    if resolution is not None and resolution > max_resolution:
        return Rejection(entry_id, "resolution", f"{resolution} > {max_resolution}")
    dates = block.find_values("_pdbx_audit_revision_history.revision_date")
    # The earliest revision is the release; skip '?' rows so they never win the min.
    release_date = min((d for d in (gemmi.cif.as_string(v) for v in dates) if d), default=None)

    dropped: dict[str, int] = {}

    def drop(kind: str) -> None:
        dropped[kind] = dropped.get(kind, 0) + 1

    # Protein entities: entity name -> (entity index, parent codes, raw monomer list)
    protein: dict[str, tuple[int, list[str], list[str]]] = {}
    for ent in st.entities:
        if ent.entity_type != gemmi.EntityType.Polymer:
            drop(_DROPPED_KIND.get(ent.entity_type, "unknown"))
            continue
        if ent.polymer_type != gemmi.PolymerType.PeptideL:
            drop("nucleic_acid" if ent.polymer_type in _NUCLEIC else "other_polymer")
            continue
        monomers = list(ent.full_sequence)
        if not monomers:
            drop("empty_sequence")
            continue
        codes = [ccd.protein_parent(gemmi.Entity.first_mon(m)) for m in monomers]
        protein[ent.name] = (len(protein), codes, monomers)
    if not protein:
        return Rejection(entry_id, "no_protein_chains")

    chains: list[ChainInfo] = []
    chain_entity: list[str] = []
    chain_residues: list[dict[int, gemmi.Residue]] = []
    sym_seen: dict[int, int] = {}
    for chain in st[0]:  # first model
        for sub in chain.subchains():
            first = sub[0]
            if first.entity_type != gemmi.EntityType.Polymer or first.entity_id not in protein:
                continue
            ent_idx, codes, monomers = protein[first.entity_id]
            by_seq: dict[int, gemmi.Residue] = {}
            for res in sub:
                seq = res.label_seq
                if seq is None or not 1 <= seq <= len(codes):
                    continue
                want = gemmi.Entity.first_mon(monomers[seq - 1])
                if seq not in by_seq or (res.name == want and by_seq[seq].name != want):
                    by_seq[seq] = res  # microheterogeneity: keep the sequence's first alternative
            if sum(1 for r in by_seq.values() if len(r) > 0) < min_resolved_residues:
                drop("short_chain")
                continue
            sym = sym_seen.get(ent_idx, 0)
            sym_seen[ent_idx] = sym + 1
            chains.append(
                ChainInfo(
                    chain_id=sub.subchain_id(),
                    auth_chain_id=chain.name,
                    entity_id=ent_idx,
                    sym_id=sym,
                    mol_type=MOL_TYPE_PROTEIN,
                    sequence="".join(PROTEIN_3TO1[c] for c in codes),
                )
            )
            chain_entity.append(first.entity_id)
            chain_residues.append(by_seq)
    if not chains:
        return Rejection(entry_id, "no_resolved_chain")

    coords: list[tuple[float, float, float]] = []
    resolved: list[bool] = []
    elements: list[int] = []
    names: list[str] = []
    charges: list[int] = []
    bfactors: list[float] = []
    atom_tok: list[int] = []
    tok: dict[str, list[int]] = {
        k: []
        for k in ("res_type", "asym", "entity", "sym", "residue", "center", "rep", "start", "count")
    }
    token = 0
    for asym, (info, ent_name, by_seq) in enumerate(
        zip(chains, chain_entity, chain_residues, strict=True)
    ):
        codes = protein[ent_name][1]
        for pos, code in enumerate(codes):
            template = table[code]
            best: dict[str, gemmi.Atom] = {}
            res = by_seq.get(pos + 1)
            if res is not None:
                for atom in res:
                    name = _ATOM_NAME_REMAP.get((res.name, atom.name), atom.name)
                    current = best.get(name)
                    if current is None or atom.occ > current.occ:
                        best[name] = atom  # highest occupancy; ties keep the first
            start = len(names)
            for name, element, charge in zip(
                template.atoms, template.elements, template.charges, strict=True
            ):
                atom = best.get(name)
                if atom is None:
                    coords.append((0.0, 0.0, 0.0))
                    resolved.append(False)
                    bfactors.append(0.0)
                else:
                    coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
                    resolved.append(True)
                    bfactors.append(float(atom.b_iso))
                elements.append(element)
                names.append(name)
                charges.append(charge)
                atom_tok.append(token)
            tok["res_type"].append(PROTEIN_RES_TYPES[code])
            tok["asym"].append(asym)
            tok["entity"].append(info.entity_id)
            tok["sym"].append(info.sym_id)
            tok["residue"].append(pos)
            tok["center"].append(start + template.atoms.index("CA"))
            tok["rep"].append(start + template.representative_atom)
            tok["start"].append(start)
            tok["count"].append(len(template.atoms))
            token += 1

    def i32(values: list[int]) -> np.ndarray:
        return np.asarray(values, dtype=np.int32)

    c = MolecularComplex(
        atom_coords=np.asarray(coords, dtype=np.float32).reshape(-1, 3),
        atom_resolved=np.asarray(resolved, dtype=bool),
        atom_element=np.asarray(elements, dtype=np.uint8),
        atom_name=names,
        atom_charge=np.asarray(charges, dtype=np.int8),
        atom_bfactor=np.asarray(bfactors, dtype=np.float32),
        atom_to_token=i32(atom_tok),
        token_res_type=np.asarray(tok["res_type"], dtype=np.int8),
        token_mol_type=np.full(token, MOL_TYPE_PROTEIN, dtype=np.int8),
        token_asym_id=i32(tok["asym"]),
        token_entity_id=i32(tok["entity"]),
        token_sym_id=i32(tok["sym"]),
        token_residue_index=i32(tok["residue"]),
        token_center_atom=i32(tok["center"]),
        token_rep_atom=i32(tok["rep"]),
        token_atom_start=i32(tok["start"]),
        token_atom_count=i32(tok["count"]),
        bonds=np.zeros((0, 2), dtype=np.int32),
        chains=chains,
        metadata=ComplexMetadata(
            id=entry_id,
            source="pdb",
            method=str(method),
            resolution=resolution,
            release_date=release_date,
            dropped=dropped,
        ),
    )
    c.validate()
    return c

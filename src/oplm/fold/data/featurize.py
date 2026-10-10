"""Sequences -> model inputs (protein-only data on the general representation; spec §4).

Mirrors ESMFold2's builder path (esm/models/esmfold2/prepare_input.py, Apache-2.0; see
THIRD_PARTY_NOTICES.md): one token per residue, flat ragged atoms padded to a multiple of 32,
``ref_*`` atom features from the reference conformer table, AF3-style chain/entity/symmetry
indices, representative (CB, else CA) atom per token, and the empty token-bond matrix (always
present). Modifications: LM inputs are one row per chain with BOS/EOS (spec §4.5) instead of one
packed row per complex; optional token padding to a crop multiple. Reproduced quirk: an unknown
residue (res_type 22) gets peptide bonds to its chain neighbours in ``token_bonds``, as upstream's
atom-tokenized-residue logic does (prepare_input.py:1075-1110).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

from oplm.fold.data.ccd import (
    MOL_TYPE_PROTEIN,
    PROTEIN_1TO3,
    PROTEIN_RES_TYPES,
    UNK_RES_TYPE,
    ReferenceConformers,
    encode_atom_name,
)
from oplm.model.tokenization_oplm import VOCAB

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

__all__ = ["ChainSpec", "FoldFeatures", "featurize"]

_BOS, _EOS, _PAD, _X = VOCAB["<cls>"], VOCAB["<eos>"], VOCAB["<pad>"], VOCAB["X"]


@dataclass(frozen=True)
class ChainSpec:
    """One protein chain and how many identical copies of it the complex contains."""

    sequence: str
    chain_id: str
    copies: int = 1


@dataclass
class FoldFeatures:
    """Batched (batch size 1) model inputs: token tensors ``(1, L)``, atom tensors ``(1, A)``."""

    token_index: Tensor
    residue_index: Tensor
    asym_id: Tensor
    entity_id: Tensor
    sym_id: Tensor
    mol_type: Tensor
    res_type: Tensor
    token_mask: Tensor  # bool
    token_bonds: Tensor  # (1, L, L, 1) float
    distogram_atom_idx: Tensor  # (1, L) flat atom index of the representative atom
    ref_pos: Tensor  # (1, A, 3)
    ref_element: Tensor  # (1, A) atomic number, 0 = pad
    ref_charge: Tensor  # (1, A) float formal charge
    ref_atom_name_chars: Tensor  # (1, A, 4) long
    ref_space_uid: Tensor  # (1, A)
    atom_mask: Tensor  # (1, A) bool
    atom_to_token: Tensor  # (1, A) long, 0 on pads
    lm_input_ids: Tensor  # (C, T) one row per chain, BOS ... EOS, right-padded
    lm_attention_mask: Tensor  # (C, T)
    lm_rows: Tensor  # (1, L) chain row of each token (0 on pads)
    lm_positions: Tensor  # (1, L) position of each token inside its LM row (0 on pads)
    chain_ids: list[str]

    @property
    def num_tokens(self) -> int:
        return int(self.token_index.shape[1])

    @property
    def num_atoms(self) -> int:
        return int(self.atom_mask.shape[1])

    @property
    def num_chains(self) -> int:
        return int(self.lm_input_ids.shape[0])

    def to(self, device: torch.device | str) -> FoldFeatures:
        """Return a copy with every tensor on ``device``."""
        moved = {}
        for f in fields(self):
            value = getattr(self, f.name)
            moved[f.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return FoldFeatures(**moved)


def _expand_chains(chains: Sequence[ChainSpec]) -> list[tuple[str, str, int, int]]:
    """(chain_id, sequence, entity_id, sym_id) per chain copy; entities by first-seen sequence."""
    entities: dict[str, int] = {}
    copies_seen: dict[int, int] = {}
    seen_ids: set[str] = set()
    out = []
    for spec in chains:
        if spec.copies < 1:
            raise ValueError(f"copies must be >= 1 for chain {spec.chain_id!r}")
        entity = entities.setdefault(spec.sequence, len(entities))
        for k in range(spec.copies):
            sym = copies_seen.get(entity, 0)
            copies_seen[entity] = sym + 1
            chain_id = spec.chain_id if k == 0 else f"{spec.chain_id}_{k + 1}"
            if chain_id in seen_ids:
                raise ValueError(f"duplicate chain id {chain_id!r}")
            seen_ids.add(chain_id)
            out.append((chain_id, spec.sequence, entity, sym))
    return out


def featurize(
    chains: Sequence[ChainSpec],
    *,
    conformers: ReferenceConformers | None = None,
    pad_tokens_to: int | None = None,
    atom_pad_multiple: int = 32,
) -> FoldFeatures:
    """Build :class:`FoldFeatures` for protein chains (no coordinates; prediction inputs).

    Args:
        chains: Chains in order; ``copies`` expands homo-oligomers with increasing ``sym_id``.
        conformers: Reference geometry table; the packaged one when ``None``.
        pad_tokens_to: Pad the token axis to this length (crop/bucket size); ``None`` = no padding.
        atom_pad_multiple: Atom axis padding multiple (upstream: 32).

    Raises:
        ValueError: ``pad_tokens_to`` is smaller than the token count, a chain is empty, or
            two chains (after ``copies`` expansion to ``ID``, ``ID_2``, ...) share an id.
    """
    table = conformers or ReferenceConformers.load()
    expanded = _expand_chains(chains)
    if any(not seq for _, seq, _, _ in expanded):
        raise ValueError("empty chain sequence")

    tok_residue, tok_asym, tok_entity, tok_sym, tok_res_type = [], [], [], [], []
    tok_rep_atom, tok_lm_row, tok_lm_pos = [], [], []
    atom_pos, atom_elem, atom_charge, atom_chars, atom_uid, atom_tok = [], [], [], [], [], []
    lm_rows: list[list[int]] = []
    chain_ids: list[str] = []
    token = 0
    for asym, (chain_id, seq, entity, sym) in enumerate(expanded):
        chain_ids.append(chain_id)
        lm_rows.append(
            [_BOS] + [VOCAB.get(c, _X) if c in PROTEIN_1TO3 else _X for c in seq] + [_EOS]
        )
        for res_i, letter in enumerate(seq):
            code = PROTEIN_1TO3.get(letter, "UNK")
            template = table[code]
            tok_residue.append(res_i)
            tok_asym.append(asym)
            tok_entity.append(entity)
            tok_sym.append(sym)
            tok_res_type.append(PROTEIN_RES_TYPES[code])
            tok_rep_atom.append(len(atom_pos) + template.representative_atom)
            tok_lm_row.append(asym)
            tok_lm_pos.append(res_i + 1)
            for name, element, charge, pos in zip(
                template.atoms, template.elements, template.charges, template.positions, strict=True
            ):
                atom_pos.append(pos)
                atom_elem.append(element)
                atom_charge.append(float(charge))
                atom_chars.append(encode_atom_name(name))
                atom_uid.append(token)
                atom_tok.append(token)
            token += 1

    n_tokens = token
    L = n_tokens if pad_tokens_to is None else pad_tokens_to
    if n_tokens > L:
        raise ValueError(f"pad_tokens_to={pad_tokens_to} < token count {n_tokens}")
    n_atoms = len(atom_pos)
    A = -(-max(n_atoms, 1) // atom_pad_multiple) * atom_pad_multiple

    def tok(values: list[int], dtype: torch.dtype = torch.long) -> Tensor:
        t = torch.zeros(1, L, dtype=dtype)
        t[0, :n_tokens] = torch.tensor(values, dtype=dtype)
        return t

    res_type = tok(tok_res_type)
    token_bonds = torch.zeros(1, L, L, 1)
    asym_t = tok(tok_asym)
    for i in range(n_tokens):  # the upstream unknown-residue bond quirk (same-chain neighbours)
        if tok_res_type[i] != UNK_RES_TYPE:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < n_tokens and tok_asym[j] == tok_asym[i]:
                token_bonds[0, i, j, 0] = token_bonds[0, j, i, 0] = 1.0

    T = max(len(r) for r in lm_rows)
    lm_ids = torch.full((len(lm_rows), T), _PAD, dtype=torch.long)
    lm_mask = torch.zeros(len(lm_rows), T, dtype=torch.long)
    for r, row in enumerate(lm_rows):
        lm_ids[r, : len(row)] = torch.tensor(row)
        lm_mask[r, : len(row)] = 1

    def atoms(values: list, dtype: torch.dtype, trailing: tuple[int, ...] = ()) -> Tensor:
        t = torch.zeros(1, A, *trailing, dtype=dtype)
        if n_atoms:
            if isinstance(values[0], torch.Tensor):
                t[0, :n_atoms] = torch.stack(values)
            else:
                t[0, :n_atoms] = torch.tensor(values, dtype=dtype)
        return t

    token_mask = torch.zeros(1, L, dtype=torch.bool)
    token_mask[0, :n_tokens] = True
    atom_mask = torch.zeros(1, A, dtype=torch.bool)
    atom_mask[0, :n_atoms] = True
    return FoldFeatures(
        token_index=tok(list(range(n_tokens))),
        residue_index=tok(tok_residue),
        asym_id=asym_t,
        entity_id=tok(tok_entity),
        sym_id=tok(tok_sym),
        mol_type=tok([MOL_TYPE_PROTEIN] * n_tokens),
        res_type=res_type,
        token_mask=token_mask,
        token_bonds=token_bonds,
        distogram_atom_idx=tok(tok_rep_atom),
        ref_pos=atoms(atom_pos, torch.float32, (3,)),
        ref_element=atoms(atom_elem, torch.long),
        ref_charge=atoms(atom_charge, torch.float32),
        ref_atom_name_chars=atoms([list(c) for c in atom_chars], torch.long, (4,)),
        ref_space_uid=atoms(atom_uid, torch.long),
        atom_mask=atom_mask,
        atom_to_token=atoms(atom_tok, torch.long),
        lm_input_ids=lm_ids,
        lm_attention_mask=lm_mask,
        lm_rows=tok(tok_lm_row),
        lm_positions=tok(tok_lm_pos),
        chain_ids=chain_ids,
    )

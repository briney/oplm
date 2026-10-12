"""Sequences or complexes -> model inputs (protein-only data on the general representation).

Mirrors ESMFold2's builder path (esm/models/esmfold2/prepare_input.py, Apache-2.0; see
THIRD_PARTY_NOTICES.md): one token per residue, flat ragged atoms padded to a multiple of 32,
``ref_*`` atom features from the reference conformer table, AF3-style chain/entity/symmetry
indices, representative (CB, else CA) atom per token, and the empty token-bond matrix (always
present). Modifications: LM inputs are one row per chain with BOS/EOS (spec §4.5) instead of one
packed row per complex; optional token padding to a crop multiple. Reproduced quirk: an unknown
residue (res_type 22) gets peptide bonds to its chain neighbours in ``token_bonds``, as upstream's
atom-tokenized-residue logic does (prepare_input.py:1075-1110).

``featurize`` (sequences, prediction) and ``featurize_complex`` (a parsed, possibly cropped
``MolecularComplex``, training) share ``_assemble``. The training path also returns
``FoldTargets`` and, given a generator, applies spec §4.3's fresh per-token rotation of the
centered reference conformers; prediction keeps the raw conformers (M1 parity).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

import torch

from oplm.fold.data.ccd import (
    MOL_TYPE_PROTEIN,
    PROTEIN_1TO3,
    PROTEIN_RES_TYPES,
    RES_TYPE_TO_CODE,
    UNK_RES_TYPE,
    ReferenceConformers,
    encode_atom_name,
)
from oplm.model.tokenization_oplm import VOCAB

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from oplm.fold.data.complex import MolecularComplex

__all__ = [
    "ChainSpec",
    "FoldExample",
    "FoldFeatures",
    "FoldTargets",
    "featurize",
    "featurize_complex",
]

_BOS, _EOS, _PAD, _X = VOCAB["<cls>"], VOCAB["<eos>"], VOCAB["<pad>"], VOCAB["X"]


@dataclass(frozen=True)
class ChainSpec:
    """One protein chain and how many identical copies of it the complex contains."""

    sequence: str
    chain_id: str
    copies: int = 1


@dataclass
class FoldFeatures:
    """Batched model inputs: token tensors ``(B, L)``, atom tensors ``(B, A)``.

    ``featurize``/``featurize_complex`` produce ``B == 1``; ``collate_examples`` stacks examples.
    LM rows are shared across the batch: ``lm_input_ids`` holds every chain of every example and
    ``lm_rows`` indexes into that stack.
    """

    token_index: Tensor
    residue_index: Tensor
    asym_id: Tensor
    entity_id: Tensor
    sym_id: Tensor
    mol_type: Tensor
    res_type: Tensor
    token_mask: Tensor  # bool
    token_bonds: Tensor  # (B, L, L, 1) float
    distogram_atom_idx: Tensor  # (B, L) flat atom index of the representative atom
    ref_pos: Tensor  # (B, A, 3)
    ref_element: Tensor  # (B, A) atomic number, 0 = pad
    ref_charge: Tensor  # (B, A) float formal charge
    ref_atom_name_chars: Tensor  # (B, A, 4) long
    ref_space_uid: Tensor  # (B, A)
    atom_mask: Tensor  # (B, A) bool
    atom_to_token: Tensor  # (B, A) long, 0 on pads
    lm_input_ids: Tensor  # (C, T) one row per chain, BOS ... EOS, right-padded
    lm_attention_mask: Tensor  # (C, T)
    lm_rows: Tensor  # (B, L) row of each token in ``lm_input_ids`` (0 on pads)
    lm_positions: Tensor  # (B, L) position of each token inside its LM row (0 on pads)
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


@dataclass
class FoldTargets:
    """Supervision aligned with :class:`FoldFeatures`' atom and token axes.

    ``coords`` is meaningful only where ``resolved``; ``frame_atom_idx`` holds the flat N, CA, C
    atom indices of each token (0 on padded tokens), usable as a frame only where all three
    atoms are resolved.
    """

    coords: Tensor  # (B, A, 3) fp32 Å
    resolved: Tensor  # (B, A) bool
    bfactor: Tensor  # (B, A) fp32
    frame_atom_idx: Tensor  # (B, L, 3) long

    def to(self, device: torch.device | str) -> FoldTargets:
        """Return a copy with every tensor on ``device``."""
        return FoldTargets(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


@dataclass
class FoldExample:
    """One training example: ``<entry id>_<chain ids>``, its features and its targets."""

    id: str
    features: FoldFeatures
    targets: FoldTargets


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


def _lm_row(sequence: str) -> list[int]:
    return [_BOS] + [VOCAB.get(c, _X) if c in PROTEIN_1TO3 else _X for c in sequence] + [_EOS]


def _random_rotation(generator: torch.Generator) -> Tensor:
    """A uniformly random proper rotation (QR of a Gaussian matrix with sign fix)."""
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q = q.clone()
        q[:, 0] = -q[:, 0]
    return q


@dataclass
class _Lists:
    """Per-token and per-atom Python lists accumulated before tensor assembly."""

    tok_residue: list[int] = field(default_factory=list)
    tok_asym: list[int] = field(default_factory=list)
    tok_entity: list[int] = field(default_factory=list)
    tok_sym: list[int] = field(default_factory=list)
    tok_res_type: list[int] = field(default_factory=list)
    tok_rep_atom: list[int] = field(default_factory=list)
    tok_lm_row: list[int] = field(default_factory=list)
    tok_lm_pos: list[int] = field(default_factory=list)
    atom_pos: list[Tensor] = field(default_factory=list)
    atom_elem: list[int] = field(default_factory=list)
    atom_charge: list[float] = field(default_factory=list)
    atom_chars: list[tuple[int, int, int, int]] = field(default_factory=list)
    atom_uid: list[int] = field(default_factory=list)
    atom_tok: list[int] = field(default_factory=list)
    lm_rows: list[list[int]] = field(default_factory=list)
    chain_ids: list[str] = field(default_factory=list)


def _assemble(lists: _Lists, *, pad_tokens_to: int | None, atom_pad_multiple: int) -> FoldFeatures:
    """Pad and tensorize the accumulated lists (batch size 1)."""
    n_tokens = len(lists.tok_residue)
    L = n_tokens if pad_tokens_to is None else pad_tokens_to
    if n_tokens > L:
        raise ValueError(f"pad_tokens_to={pad_tokens_to} < token count {n_tokens}")
    n_atoms = len(lists.atom_pos)
    A = -(-max(n_atoms, 1) // atom_pad_multiple) * atom_pad_multiple

    def tok(values: list[int], dtype: torch.dtype = torch.long) -> Tensor:
        t = torch.zeros(1, L, dtype=dtype)
        t[0, :n_tokens] = torch.tensor(values, dtype=dtype)
        return t

    token_bonds = torch.zeros(1, L, L, 1)
    for i in range(n_tokens):  # the upstream unknown-residue bond quirk (same-chain neighbours)
        if lists.tok_res_type[i] != UNK_RES_TYPE:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < n_tokens and lists.tok_asym[j] == lists.tok_asym[i]:
                token_bonds[0, i, j, 0] = token_bonds[0, j, i, 0] = 1.0

    T = max(len(r) for r in lists.lm_rows)
    lm_ids = torch.full((len(lists.lm_rows), T), _PAD, dtype=torch.long)
    lm_mask = torch.zeros(len(lists.lm_rows), T, dtype=torch.long)
    for r, row in enumerate(lists.lm_rows):
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
        residue_index=tok(lists.tok_residue),
        asym_id=tok(lists.tok_asym),
        entity_id=tok(lists.tok_entity),
        sym_id=tok(lists.tok_sym),
        mol_type=tok([MOL_TYPE_PROTEIN] * n_tokens),
        res_type=tok(lists.tok_res_type),
        token_mask=token_mask,
        token_bonds=token_bonds,
        distogram_atom_idx=tok(lists.tok_rep_atom),
        ref_pos=atoms(lists.atom_pos, torch.float32, (3,)),
        ref_element=atoms(lists.atom_elem, torch.long),
        ref_charge=atoms(lists.atom_charge, torch.float32),
        ref_atom_name_chars=atoms([list(c) for c in lists.atom_chars], torch.long, (4,)),
        ref_space_uid=atoms(lists.atom_uid, torch.long),
        atom_mask=atom_mask,
        atom_to_token=atoms(lists.atom_tok, torch.long),
        lm_input_ids=lm_ids,
        lm_attention_mask=lm_mask,
        lm_rows=tok(lists.tok_lm_row),
        lm_positions=tok(lists.tok_lm_pos),
        chain_ids=lists.chain_ids,
    )


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
    lists = _Lists()
    token = 0
    for asym, (chain_id, seq, entity, sym) in enumerate(expanded):
        lists.chain_ids.append(chain_id)
        lists.lm_rows.append(_lm_row(seq))
        for res_i, letter in enumerate(seq):
            code = PROTEIN_1TO3.get(letter, "UNK")
            template = table[code]
            lists.tok_residue.append(res_i)
            lists.tok_asym.append(asym)
            lists.tok_entity.append(entity)
            lists.tok_sym.append(sym)
            lists.tok_res_type.append(PROTEIN_RES_TYPES[code])
            lists.tok_rep_atom.append(len(lists.atom_pos) + template.representative_atom)
            lists.tok_lm_row.append(asym)
            lists.tok_lm_pos.append(res_i + 1)
            for name, element, charge, pos in zip(
                template.atoms, template.elements, template.charges, template.positions, strict=True
            ):
                lists.atom_pos.append(pos)
                lists.atom_elem.append(element)
                lists.atom_charge.append(float(charge))
                lists.atom_chars.append(encode_atom_name(name))
                lists.atom_uid.append(token)
                lists.atom_tok.append(token)
            token += 1
    return _assemble(lists, pad_tokens_to=pad_tokens_to, atom_pad_multiple=atom_pad_multiple)


def featurize_complex(
    complex: MolecularComplex,
    *,
    conformers: ReferenceConformers | None = None,
    pad_tokens_to: int | None = None,
    atom_pad_multiple: int = 32,
    generator: torch.Generator | None = None,
) -> FoldExample:
    """Build features and targets for a parsed (and usually cropped) complex.

    Every chain in ``complex.chains`` gets one LM row holding its *full* sequence (spec §4.5), and
    each token's ``lm_positions`` is its residue index plus one, so a crop still reads the whole
    chain through the LM. With ``generator`` the reference conformer of every token is centered
    and rotated by a fresh random rotation (spec §4.3); without it the raw conformer is used.

    Raises:
        ValueError: An atom name disagrees with the token's template (the parser guarantees the
            template order; anything else means the row is not from this pipeline), or
            ``pad_tokens_to`` is smaller than the token count.
    """
    table = conformers or ReferenceConformers.load()
    lists = _Lists()
    lists.chain_ids = [ch.chain_id for ch in complex.chains]
    lists.lm_rows = [_lm_row(ch.sequence) for ch in complex.chains]
    starts = complex.token_atom_start
    counts = complex.token_atom_count
    frame_rows: list[list[int]] = []
    for token in range(complex.num_tokens):
        code = RES_TYPE_TO_CODE[int(complex.token_res_type[token])]
        template = table[code]
        start, count = int(starts[token]), int(counts[token])
        names = complex.atom_name[start : start + count]
        if tuple(names) != template.atoms:
            raise ValueError(
                f"token {token} ({code}) atoms {names} do not match the template {template.atoms}"
            )
        lists.tok_residue.append(int(complex.token_residue_index[token]))
        lists.tok_asym.append(int(complex.token_asym_id[token]))
        lists.tok_entity.append(int(complex.token_entity_id[token]))
        lists.tok_sym.append(int(complex.token_sym_id[token]))
        lists.tok_res_type.append(int(complex.token_res_type[token]))
        lists.tok_rep_atom.append(start + template.representative_atom)
        lists.tok_lm_row.append(int(complex.token_asym_id[token]))
        lists.tok_lm_pos.append(int(complex.token_residue_index[token]) + 1)
        positions = template.positions
        if generator is not None:
            positions = (positions - positions.mean(0)) @ _random_rotation(generator).T
        for name, element, charge, pos in zip(
            template.atoms, template.elements, template.charges, positions, strict=True
        ):
            lists.atom_pos.append(pos)
            lists.atom_elem.append(element)
            lists.atom_charge.append(float(charge))
            lists.atom_chars.append(encode_atom_name(name))
            lists.atom_uid.append(token)
            lists.atom_tok.append(token)
        frame_rows.append([start + template.atoms.index(n) for n in ("N", "CA", "C")])
    features = _assemble(lists, pad_tokens_to=pad_tokens_to, atom_pad_multiple=atom_pad_multiple)
    L, A, n_atoms = features.num_tokens, features.num_atoms, complex.num_atoms
    coords = torch.zeros(1, A, 3)
    resolved = torch.zeros(1, A, dtype=torch.bool)
    bfactor = torch.zeros(1, A)
    coords[0, :n_atoms] = torch.from_numpy(complex.atom_coords)
    resolved[0, :n_atoms] = torch.from_numpy(complex.atom_resolved)
    bfactor[0, :n_atoms] = torch.from_numpy(complex.atom_bfactor)
    frames = torch.zeros(1, L, 3, dtype=torch.long)
    if frame_rows:
        frames[0, : len(frame_rows)] = torch.tensor(frame_rows, dtype=torch.long)
    targets = FoldTargets(coords=coords, resolved=resolved, bfactor=bfactor, frame_atom_idx=frames)
    example_id = "_".join([complex.metadata.id, *lists.chain_ids])
    return FoldExample(id=example_id, features=features, targets=targets)

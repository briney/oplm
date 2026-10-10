"""Sequence(s) -> ranked structure -> mmCIF (spec §10 M1 "predict/mmCIF").

Ranking follows the ESMFold2 HF adapter's "best-ranked sample" convention: ipTM for complexes,
pTM for monomers (upstream's own ``fold()`` returns samples unranked). mmCIF is written with
gemmi; B-factors carry pLDDT × 100 (upstream's PDB writer leaves it on the 0–1 scale).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from oplm.fold.data.ccd import PROTEIN_RES_TYPES
from oplm.fold.data.featurize import ChainSpec, featurize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from oplm.fold.data.featurize import FoldFeatures
    from oplm.fold.modeling_fold import FoldOutput, OplmForFolding

__all__ = ["FoldResult", "fold", "parse_chain_arg", "rank_samples", "write_mmcif"]

_RES_NAMES = {v: k for k, v in PROTEIN_RES_TYPES.items()}
_ELEMENT_SYMBOLS = {6: "C", 7: "N", 8: "O", 16: "S"}


@dataclass
class FoldResult:
    """The best-ranked sample, unpadded, plus the raw model output."""

    chain_ids: list[str]
    sequences: list[str]
    coords: Tensor  # (A_real, 3) Å
    atom_names: list[str]
    atom_elements: list[int]
    atom_chain_index: Tensor  # (A_real,) index into chain_ids
    atom_residue_index: Tensor  # (A_real,) 0-based within the chain
    residue_names: list[str]  # per token
    plddt_per_atom: Tensor  # (A_real,) in [0, 1]
    plddt: Tensor  # (L,) per token
    pae: Tensor  # (L, L) Å
    ptm: float
    iptm: float
    best_sample: int
    features: FoldFeatures
    output: FoldOutput


def parse_chain_arg(text: str, default_id: str) -> ChainSpec:
    """``"SEQ"``, ``"ID:SEQ"`` or ``"ID:SEQ*N"`` -> :class:`ChainSpec`.

    Raises:
        ValueError: The sequence is empty or the copy count is below 1.
    """
    chain_id, _, rest = text.rpartition(":")
    seq, _, copies = rest.partition("*")
    n = int(copies) if copies else 1
    if n < 1:
        raise ValueError(f"copies must be >= 1 in {text!r}")
    if not seq:
        raise ValueError(f"empty sequence in {text!r}")
    return ChainSpec(seq.upper(), chain_id or default_id, copies=n)


def rank_samples(output: FoldOutput, num_chains: int) -> Tensor:
    """Sample indices from best to worst: by ipTM for complexes, pTM otherwise."""
    score = output.confidence.iptm if num_chains > 1 else output.confidence.ptm
    return torch.argsort(score.detach().cpu(), descending=True)


def fold(
    model: OplmForFolding,
    chains: Sequence[ChainSpec],
    *,
    lm_hidden_states: Tensor | None = None,
    num_samples: int = 1,
    num_loops: int | None = None,
    num_steps: int | None = None,
    seed: int | None = None,
    pad_to_multiple: int | None = None,
) -> FoldResult:
    """Featurize, run the model, rank the samples and unpad the best one.

    ``pad_to_multiple`` defaults to 128 on CUDA (FlexAttention shape buckets, docs/FOLD.md §3)
    and to no padding on CPU. ``seed`` makes the initial pair state and the sampler
    deterministic for a given device.
    """
    device = model.device
    if pad_to_multiple is None and device.type == "cuda":
        pad_to_multiple = 128
    n_tokens = sum(len(c.sequence) * c.copies for c in chains)
    pad_to = None if pad_to_multiple is None else -(-n_tokens // pad_to_multiple) * pad_to_multiple
    f = featurize(chains, pad_tokens_to=pad_to)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        output = model(
            f,
            lm_hidden_states=lm_hidden_states,
            num_loops=num_loops,
            num_samples=num_samples,
            num_steps=num_steps,
            generator=generator,
        )
    order = rank_samples(output, f.num_chains)
    best = int(order[0])
    real = f.atom_mask[0].cpu()
    tok = f.token_mask[0].cpu()
    atom_to_token = f.atom_to_token[0].cpu()[real]
    conf = output.confidence
    name_chars = f.ref_atom_name_chars[0].cpu()[real].tolist()
    return FoldResult(
        chain_ids=list(f.chain_ids),
        sequences=[c.sequence for c in chains for _ in range(c.copies)],
        coords=output.coords[best].detach().cpu()[real],
        atom_names=["".join(chr(int(c) + 32) for c in row).strip() for row in name_chars],
        atom_elements=f.ref_element[0].cpu()[real].tolist(),
        atom_chain_index=f.asym_id[0].cpu()[atom_to_token],
        atom_residue_index=f.residue_index[0].cpu()[atom_to_token],
        residue_names=[_RES_NAMES[int(r)] for r in f.res_type[0].cpu()[tok].tolist()],
        plddt_per_atom=conf.plddt_per_atom[best].detach().cpu()[real],
        plddt=conf.plddt[best].detach().cpu()[tok],
        pae=conf.pae[best].detach().cpu()[tok][:, tok],
        ptm=float(conf.ptm[best]),
        iptm=float(conf.iptm[best]),
        best_sample=best,
        features=f,
        output=output,
    )


def write_mmcif(result: FoldResult, path: Path, *, name: str = "pred") -> Path:
    """Write the best sample as mmCIF (one model; B-factor = pLDDT × 100)."""
    import gemmi

    structure = gemmi.Structure()
    structure.name = name
    chains = {chain_id: gemmi.Chain(chain_id) for chain_id in result.chain_ids}
    residues: dict[tuple[int, int], gemmi.Residue] = {}
    for i in range(result.coords.shape[0]):
        key = (int(result.atom_chain_index[i]), int(result.atom_residue_index[i]))
        if key not in residues:  # atoms are token-ordered, so the next residue is the next token
            res = gemmi.Residue()
            res.name = result.residue_names[len(residues)]
            res.seqid = gemmi.SeqId(key[1] + 1, " ")
            res.het_flag = "A"
            residues[key] = res
        atom = gemmi.Atom()
        atom.name = result.atom_names[i]
        atom.element = gemmi.Element(_ELEMENT_SYMBOLS.get(result.atom_elements[i], "X"))
        atom.pos = gemmi.Position(*result.coords[i].tolist())
        atom.occ = 1.0
        atom.b_iso = float(result.plddt_per_atom[i]) * 100.0
        residues[key].add_atom(atom)
    for (chain_idx, _), res in residues.items():
        chains[result.chain_ids[chain_idx]].add_residue(res)
    model = gemmi.Model(1)
    for chain in chains.values():
        model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    path = Path(path)
    structure.make_mmcif_document().write_file(str(path))
    return path

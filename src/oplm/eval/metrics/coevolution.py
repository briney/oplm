"""Coevolution metrics: Jacobian pair scores versus MSA Potts couplings."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor

from oplm.eval.metrics.ranking import spearman_corr

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from oplm.eval.metrics.categorical_jacobian import StructurePairScoreData

NEFF_BINS: tuple[str, ...] = ("low", "mid", "high")

_SUFFIXES: tuple[tuple[str, int], ...] = (("at_L", 1), ("at_L_2", 2), ("at_L_5", 5))

# Hit mask from (M, P, T): Jacobian top-k, Potts top-k, true contacts.
_HITS: dict[str, Callable[[Tensor, Tensor, Tensor], Tensor]] = {
    "coevo_precision_{s}": lambda m, p, t: m & p,
    "p_{s}_shared": lambda m, p, t: m & p & t,
    "p_{s}_beyond": lambda m, p, t: m & ~p & t,
    "p_{s}_potts_only": lambda m, p, t: p & ~m & t,
    "potts_precision_{s}": lambda m, p, t: p & t,
}
_METRIC_HITS = {tmpl.format(s=s): fn for tmpl, fn in _HITS.items() for s, _ in _SUFFIXES}

COEVOLUTION_METRIC_DIVISORS: dict[str, int] = {
    **{tmpl.format(s=s): d for tmpl in _HITS for s, d in _SUFFIXES},
    "coevo_spearman": 0,
}

DEFAULT_COEVOLUTION_METRICS: list[str] = [
    "coevo_precision_at_L",
    "p_at_L_shared",
    "p_at_L_beyond",
    "p_at_L_potts_only",
    "potts_precision_at_L",
    "coevo_spearman",
]

_HEADLINE_BINS = ("mid", "high")


def _spearman(a: Tensor, b: Tensor) -> float | None:
    try:
        return spearman_corr(a.double().numpy(), b.double().numpy())
    except ValueError:  # fewer than two points or a constant input
        return None


def has_coevolution(structure: StructurePairScoreData) -> bool:
    """True when the structure has at least one scorable pair with a finite Potts value."""
    return structure.potts_scores is not None and bool((~torch.isnan(structure.potts_scores)).any())


def _structure_values(
    structure: StructurePairScoreData, metrics: Collection[str]
) -> dict[str, float]:
    """Per-structure values for the requested metrics (Spearman omitted if undefined)."""
    potts = structure.potts_scores
    assert potts is not None
    n_pairs = structure.scores.numel()
    values: dict[str, float] = {}
    finite = ~torch.isnan(potts)
    for name in metrics:
        if name == "coevo_spearman":
            rho = _spearman(structure.scores[finite], potts[finite])
            if rho is not None:
                values[name] = rho
            continue
        if n_pairs == 0:
            values[name] = 0.0
            continue
        k = min(max(1, structure.seq_len // COEVOLUTION_METRIC_DIVISORS[name]), n_pairs)
        m = torch.zeros(n_pairs, dtype=torch.bool)
        p = torch.zeros(n_pairs, dtype=torch.bool)
        m[torch.topk(structure.scores, k).indices] = True
        p[torch.topk(potts.nan_to_num(nan=float("-inf")), k).indices] = True
        p &= finite  # fewer than k finite pairs: P holds only real Potts calls, k stays the divisor
        hits = _METRIC_HITS[name](m, p, structure.labels.bool())
        values[name] = hits.sum().item() / k
    return values


def _mean(
    per_structure: list[tuple[str | None, dict[str, float]]],
    name: str,
    bins: Collection[str] | None,
) -> float | None:
    vals = [v[name] for b, v in per_structure if name in v and (bins is None or b in bins)]
    return sum(vals) / len(vals) if vals else None


def compute_coevolution_metrics(
    structures: Sequence[StructurePairScoreData], metrics: Collection[str]
) -> dict[str, float]:
    """Compare Jacobian pair scores with MSA Potts couplings.

    Only structures with a finite Potts value on at least one scorable pair are used. For each
    requested metric the headline key is averaged over the ``mid`` and ``high`` Neff bins for
    ``coevo_*`` names (agreement needs a well-fit Potts model) and over all structures otherwise;
    ``{name}_neff_{bin}`` is emitted per populated bin. Empty groups are omitted.

    Args:
        structures: Per-structure pair scores; those without a finite Potts value are skipped.
        metrics: Base names from ``COEVOLUTION_METRIC_DIVISORS``.

    Returns:
        Mapping from metric key to mean value.
    """
    per_structure = [
        (s.neff_bin, _structure_values(s, metrics)) for s in structures if has_coevolution(s)
    ]
    out: dict[str, float] = {}
    for name in metrics:
        headline = _mean(per_structure, name, _HEADLINE_BINS if name.startswith("coevo_") else None)
        if headline is not None:
            out[name] = headline
        for b in NEFF_BINS:
            val = _mean(per_structure, name, (b,))
            if val is not None:
                out[f"{name}_neff_{b}"] = val
    return out


class CoevolutionReference:
    """Precomputed Potts contact maps keyed by sequence sha256.

    Layout: ``manifest.parquet`` (``sha256``, ``length``, ``neff_bin``) and
    ``couplings/<sha256>.npy`` (float16 ``(L, L)``, over the full sequence).
    """

    def __init__(self, root: Path, bins: dict[str, str]) -> None:
        self.root = root
        self._bins = bins

    @classmethod
    def load(cls, path: str | Path) -> CoevolutionReference:
        """Read the manifest under ``path``; maps load lazily in :meth:`lookup`."""
        root = Path(path)
        manifest = pq.read_table(root / "manifest.parquet", columns=["sha256", "neff_bin"])
        columns = manifest.to_pydict()
        # dict() keeps the last duplicate; rows sharing a sha256 share a bin
        return cls(root, dict(zip(columns["sha256"], columns["neff_bin"], strict=True)))

    def lookup(self, sequence: str) -> tuple[Tensor, str] | None:
        """Return the float32 ``(L, L)`` map and its Neff bin, or None if not in the manifest."""
        sha = hashlib.sha256(sequence.encode()).hexdigest()
        if sha not in self._bins:
            return None
        potts = torch.from_numpy(np.load(self.root / "couplings" / f"{sha}.npy")).float()
        n = len(sequence)
        if potts.shape != (n, n):
            raise ValueError(
                f"coevolution map {sha} has shape {tuple(potts.shape)}, expected {(n, n)}"
            )
        return potts, self._bins[sha]

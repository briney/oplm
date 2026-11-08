"""Coevolution metrics: Jacobian pair scores versus MSA Potts couplings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

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


def _ranks(x: Tensor) -> Tensor:
    # ponytail: ordinal ranks, ties not averaged; fp16 ties are rare
    return x.argsort().argsort().float()


def _spearman(a: Tensor, b: Tensor) -> float | None:
    if a.numel() < 2:
        return None
    ra, rb = _ranks(a) - (a.numel() - 1) / 2, _ranks(b) - (a.numel() - 1) / 2
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


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

    Only structures with ``potts_scores`` are used. For each requested metric the headline
    key is averaged over the ``mid`` and ``high`` Neff bins for ``coevo_*`` names (agreement
    needs a well-fit Potts model) and over all structures otherwise; ``{name}_neff_{bin}``
    is emitted per populated bin. Empty groups are omitted.

    Args:
        structures: Per-structure pair scores; those without ``potts_scores`` are skipped.
        metrics: Base names from ``COEVOLUTION_METRIC_DIVISORS``.

    Returns:
        Mapping from metric key to mean value.
    """
    per_structure = [
        (s.neff_bin, _structure_values(s, metrics))
        for s in structures
        if s.potts_scores is not None
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

"""Fold one held-out chain and score it; aggregate, store and compare score tables (addendum §6)."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from oplm.fold.data.featurize import ChainSpec
from oplm.fold.eval.metrics import lddt, tm_score
from oplm.fold.predict import fold

if TYPE_CHECKING:
    from collections.abc import Sequence

    from oplm.fold.eval.targets import EvalTarget
    from oplm.fold.modeling_fold import OplmForFolding
    from oplm.fold.predict import FoldResult

__all__ = [
    "METRICS",
    "SamplingSettings",
    "TargetScore",
    "compare_scores",
    "read_scores",
    "score_result",
    "score_target",
    "summarize",
    "write_scores",
]

METRICS = ("tm_score", "lddt_ca", "lddt", "plddt", "ptm")
_CASTS = {"str": str, "int": int, "float": float}


@dataclass(frozen=True)
class SamplingSettings:
    """How each target is folded; recorded next to every score table (spec §8)."""

    num_loops: int | None = None  # None: the checkpoint's inference_num_loops
    num_steps: int | None = None  # None: the checkpoint's inference_num_steps
    num_samples: int = 1  # the best sample by pTM is scored
    seed: int = 0


@dataclass
class TargetScore:
    """One row of a score table; the metrics are NaN unless ``status == "ok"``."""

    target_set: str
    entry_id: str
    chain_id: str
    auth_chain_id: str
    length: int
    resolved_residues: int
    status: str  # "ok" or "failed"
    error: str = ""
    seconds: float = 0.0
    tm_score: float = math.nan
    lddt_ca: float = math.nan
    lddt: float = math.nan
    plddt: float = math.nan
    ptm: float = math.nan

    @property
    def key(self) -> tuple[str, str, str]:
        """``(target_set, entry_id, chain_id)``: pairs rows across tables."""
        return (self.target_set, self.entry_id, self.chain_id)


def score_result(target: EvalTarget, result: FoldResult) -> dict[str, float]:
    """TM-score and CA lDDT over resolved CAs, all-atom lDDT, mean pLDDT (0-1) and pTM.

    Raises:
        ValueError: The prediction's atoms are not the reference's atoms in the same order.
    """
    if list(result.atom_names) != target.atom_names:
        raise ValueError(f"{target.key}: the predicted atom layout differs from the reference")
    pred = result.coords.detach().cpu().double().numpy()
    ca = target.ca_index
    ca_mask = target.resolved[ca]
    return {
        "tm_score": tm_score(pred[ca], target.coords[ca], ca_mask),
        "lddt_ca": lddt(pred[ca], target.coords[ca], ca_mask),
        "lddt": lddt(pred, target.coords, target.resolved, residue_index=target.residue_index),
        "plddt": float(result.plddt.float().mean()),
        "ptm": float(result.ptm),
    }


def score_target(
    model: OplmForFolding, target: EvalTarget, settings: SamplingSettings
) -> TargetScore:
    """Fold ``target.sequence`` alone (eval-mode ``model``) and score the best sample.

    Any exception, CUDA out-of-memory included, becomes a ``failed`` row instead of
    propagating: in training every rank must reach the gather, and offline one bad
    target must not end the run (spec §8 reports failures and coverage).
    """
    row = TargetScore(
        target_set=target.target_set,
        entry_id=target.entry_id,
        chain_id=target.chain_id,
        auth_chain_id=target.auth_chain_id,
        length=len(target.sequence),
        resolved_residues=int(target.resolved[target.ca_index].sum()),
        status="ok",
    )
    start = time.perf_counter()
    try:
        result = fold(
            model,
            [ChainSpec(target.sequence, "A")],
            num_samples=settings.num_samples,
            num_loops=settings.num_loops,
            num_steps=settings.num_steps,
            seed=settings.seed,
        )
        for name, value in score_result(target, result).items():
            setattr(row, name, value)
    except Exception as exc:
        row.status, row.error = "failed", f"{type(exc).__name__}: {exc}"[:500]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    row.seconds = time.perf_counter() - start
    return row


def summarize(scores: Sequence[TargetScore]) -> dict[str, float]:
    """Means of ``METRICS`` over ``ok`` rows (NaN when none) plus ``targets`` and ``failed``."""
    ok = [s for s in scores if s.status == "ok"]
    out = {m: float(np.mean([getattr(s, m) for s in ok])) if ok else math.nan for m in METRICS}
    out["targets"] = float(len(ok))
    out["failed"] = float(len(scores) - len(ok))
    return out


def write_scores(path: str | Path, scores: Sequence[TargetScore]) -> None:
    """Write ``scores`` as CSV, one row per target, columns in ``TargetScore`` field order."""
    with Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[f.name for f in fields(TargetScore)])
        writer.writeheader()
        for s in scores:
            writer.writerow(asdict(s))


def read_scores(path: str | Path) -> list[TargetScore]:
    """Read a table written by :func:`write_scores`."""
    types = {f.name: str(f.type) for f in fields(TargetScore)}
    with Path(path).open(newline="") as fh:
        result = []
        for raw in csv.DictReader(fh):
            kwargs: Any = {k: _CASTS[types[k]](v) for k, v in raw.items()}
            result.append(TargetScore(**kwargs))
        return result


def compare_scores(
    a: Sequence[TargetScore],
    b: Sequence[TargetScore],
    *,
    metric: str = "tm_score",
    samples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired ``b - a`` over targets scored ``ok`` in both, with a percentile bootstrap 95 % CI.

    Raises:
        ValueError: ``metric`` is not one of ``METRICS``, or no target is ``ok`` in both.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {METRICS}")
    left = {s.key: getattr(s, metric) for s in a if s.status == "ok"}
    right = {s.key: getattr(s, metric) for s in b if s.status == "ok"}
    keys = sorted(left.keys() & right.keys())
    if not keys:
        raise ValueError("no target is scored ok in both tables")
    x = np.array([left[k] for k in keys])
    y = np.array([right[k] for k in keys])
    diff = y - x
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(diff), size=(samples, len(diff)))].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "n": float(len(keys)),
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "mean_diff": float(diff.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "b_better_fraction": float((diff > 0).mean()),
    }

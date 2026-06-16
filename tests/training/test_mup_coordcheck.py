"""Slow μP coordinate-check oracle (Phases 5-6).

Runs :func:`oplm.training.mup.coord_check` over a few widths on real sequences and
asserts the **one-sided** oracle: with μP, no module's activation RMS grows with
width; the ``--no-mup`` control fans out. Growth is read at the final step
(``t = steps ≥ 1``), so the readout's init-time ``Θ(1/√m)`` shrink (``t=0``) is
excluded by construction — see ``docs/MUP.md`` and ``scripts/mup_coord_check.py``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from oplm.model import OplmConfig as OplmModelConfig
from oplm.training.mup import coord_check

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    import torch

pytestmark = pytest.mark.slow

_HEAD_DIM = 64  # held fixed across widths (the μP-invariant: only head count grows)
_PRESET_ASPECT_RATIO = 32  # hidden/layers of the preset ray (the 50M preset is 512/16)


def _build_cfg(
    width: int, *, mup: bool, depth: int, scaling: str, base_width: int
) -> OplmModelConfig:
    """Width → OplmConfig, mirroring the coord-check script's geometry."""
    layers = max(1, round(width / _PRESET_ASPECT_RATIO)) if scaling == "preset_ray" else depth
    return OplmModelConfig(
        hidden_size=width,
        num_hidden_layers=layers,
        num_attention_heads=width // _HEAD_DIM,
        head_dim=_HEAD_DIM,
        max_position_embeddings=64,
        mup_enable=mup,
        mup_base_width=base_width,
    )


def _make_batch(
    parquet: Path, *, n_seqs: int = 8, max_length: int = 64, seed: int = 0
) -> dict[str, torch.Tensor]:
    """One fixed, deterministically-masked batch from the real-data fixture."""
    import pyarrow.parquet as pq

    from oplm.data import MLMCollator, get_tokenizer

    rows = next(pq.ParquetFile(parquet).iter_batches(batch_size=n_seqs, columns=["sequence"]))
    sequences = rows.column("sequence").to_pylist()
    collator = MLMCollator(get_tokenizer(), max_length=max_length, deterministic=True, seed=seed)
    return collator(sequences)


def _worst_growth(df: pd.DataFrame) -> float:
    """Largest small-width→large-width RMS ratio over module types at the final step."""
    final = df[df["step"] == df["step"].max()].copy()
    final["mtype"] = final["module"].map(lambda n: re.sub(r"layers\.\d+", "layers.*", n))
    agg = final.groupby(["mtype", "width"], as_index=False)["rms"].mean()
    widths = sorted(agg["width"].unique())
    lo, hi = widths[0], widths[-1]
    growths = []
    for _, group in agg.groupby("mtype"):
        by_width = group.set_index("width")["rms"]
        rms_lo = float(by_width.loc[lo])
        growths.append(float(by_width.loc[hi]) / rms_lo if rms_lo > 0 else float("inf"))
    return max(growths)


def test_width_coord_check_one_sided_oracle(training_parquet: Path) -> None:
    """μP holds RMS flat across width; the μP-off control fans out past it."""
    batch = _make_batch(training_parquet)
    widths = [128, 256, 512]
    common = {"widths": widths, "batch": batch, "steps": 3, "optimizer": "muon"}

    mup_df = coord_check(
        lambda w: _build_cfg(w, mup=True, depth=4, scaling="width", base_width=128), **common
    )
    nomup_df = coord_check(
        lambda w: _build_cfg(w, mup=False, depth=4, scaling="width", base_width=128), **common
    )

    mup_worst = _worst_growth(mup_df)
    control_worst = _worst_growth(nomup_df)

    # One-sided: with μP, no module's RMS grows materially with width.
    assert mup_worst < 2.0, f"μP RMS grew {mup_worst:.2f}× with width (should not grow)"
    # The control is the contrast — it fans out, and by more than μP.
    assert control_worst > 2.0, f"control did not fan out (worst {control_worst:.2f}×)"
    assert control_worst > mup_worst


def test_preset_ray_smoke(training_parquet: Path) -> None:
    """``scaling='preset_ray'`` (depth co-scaled with width) runs and returns a tidy frame."""
    batch = _make_batch(training_parquet)
    df = coord_check(
        lambda w: _build_cfg(w, mup=True, depth=4, scaling="preset_ray", base_width=128),
        widths=[128, 256],
        batch=batch,
        steps=2,
        optimizer="muon",
        scaling="preset_ray",
    )
    assert not df.empty
    assert list(df.columns) == ["width", "module", "step", "rms"]
    assert df.attrs["scaling"] == "preset_ray"
    assert set(df["width"].unique()) == {128, 256}

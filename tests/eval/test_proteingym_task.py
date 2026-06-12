"""ProteinGym DMS task tests: config parsing plus a slow end-to-end run.

The slow test scores the committed real DMS fixture
(``tests/data/fixtures/variant/A0A2Z5U3Z0_9INFA_Wu_2014.csv``) end-to-end under
both scoring methods and checks the three leaderboard metrics land in range.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from oplm.config import (
    DataConfig,
    EvalDatasetEntry,
    OplmConfig,
    ScheduleSpec,
    TrainConfig,
)
from oplm.eval.tasks.proteingym import ProteinGymEvalTask, ProteinGymTaskConfig
from oplm.model import OplmConfig as OplmModelConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.model import OplmForMaskedLM

# The real DMS fixture's wild-type is 565 aa → tokenized length 567; the model
# context must exceed it so the assay is scored, not skipped.
_MAX_SEQ_LEN = 640

_DMS_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "variant"


# --- pure config-parse tests (no model / fixtures) -----------------------------


def test_config_defaults() -> None:
    """Defaults: masked_marginals, batch 64, no cap, top 10%."""
    cfg = ProteinGymTaskConfig.from_extra({})
    assert cfg.scoring == "masked_marginals"
    assert cfg.mask_batch_size == 64
    assert cfg.max_assays is None
    assert cfg.top_k_fraction == pytest.approx(0.1)


def test_config_rejects_unknown_scoring() -> None:
    """An unknown scoring method is rejected."""
    with pytest.raises(ValueError, match="scoring"):
        ProteinGymTaskConfig.from_extra({"scoring": "bogus"})


@pytest.mark.parametrize("bad_batch", [0, 99999])
def test_config_mask_batch_size_range(bad_batch: int) -> None:
    """``mask_batch_size`` must lie in ``[1, 1024]``."""
    with pytest.raises(ValueError, match="mask_batch_size"):
        ProteinGymTaskConfig.from_extra({"mask_batch_size": bad_batch})


@pytest.mark.parametrize("bad_fraction", [0.0, 1.5, -0.1])
def test_config_top_k_fraction_range(bad_fraction: float) -> None:
    """``top_k_fraction`` must lie in ``(0, 1]``."""
    with pytest.raises(ValueError, match="top_k_fraction"):
        ProteinGymTaskConfig.from_extra({"top_k_fraction": bad_fraction})


def test_config_negative_max_assays_rejected() -> None:
    """A negative ``max_assays`` is rejected."""
    with pytest.raises(ValueError, match="max_assays"):
        ProteinGymTaskConfig.from_extra({"max_assays": -1})


# --- slow end-to-end test ------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("scoring", ["masked_marginals", "wt_marginals"])
def test_dms_eval_metrics_in_range(
    scoring: str, make_model: Callable[..., OplmForMaskedLM]
) -> None:
    """Both scoring modes produce finite spearman/auroc/top_k_precision in range."""
    from accelerate import Accelerator

    assert _DMS_FIXTURE_DIR.is_dir(), (
        f"DMS fixture dir missing: {_DMS_FIXTURE_DIR} (see fixtures/README.md)"
    )

    cfg = OplmConfig(
        model=OplmModelConfig(max_position_embeddings=_MAX_SEQ_LEN),
        train=TrainConfig(wandb_enabled=False),
        data=DataConfig(num_workers=0, pin_memory=False),
    )
    model = make_model(max_position_embeddings=_MAX_SEQ_LEN)
    entry = EvalDatasetEntry(
        name="dms",
        path=str(_DMS_FIXTURE_DIR),
        type="proteingym",
        schedule=ScheduleSpec("steps", 1),
        extra={"scoring": scoring},
    )
    task = ProteinGymEvalTask(entry, cfg)
    accelerator = Accelerator(cpu=True)

    metrics = task.evaluate(model, accelerator)

    assert math.isfinite(metrics["spearman"]) and -1.0 <= metrics["spearman"] <= 1.0
    assert math.isfinite(metrics["auroc"]) and 0.0 <= metrics["auroc"] <= 1.0
    assert math.isfinite(metrics["top_k_precision"]) and 0.0 <= metrics["top_k_precision"] <= 1.0

    # Guard against a silent fallback: the real fixture must actually be scored
    # and yield all three metrics (it has DMS_score, DMS_score_bin, and >1 row).
    assert task._assays is not None and len(task._assays) == 1
    result = task._score_assay(task._assays[0], model, accelerator.device)
    assert result is not None
    assert set(result) == {"spearman", "auroc", "top_k_precision"}

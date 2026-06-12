"""ProteinGym clinical-variant task tests: config parsing plus a slow e2e run.

The slow test scores the committed real clinical fixture
(``tests/data/fixtures/variant_clinical/NP_000008.1.csv``) end-to-end under both
scoring methods and asserts the mean AUROC lands in ``[0, 1]``.
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
from oplm.eval.tasks.proteingym_clinical import (
    ProteinGymClinicalEvalTask,
    ProteinGymClinicalTaskConfig,
)
from oplm.model import OplmConfig as OplmModelConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.model import OplmForMaskedLM

# The real clinical fixture's wild-type is 412 aa → tokenized length 414; the
# model context must comfortably exceed it so the assay is scored, not skipped.
_MAX_SEQ_LEN = 512

_CLINICAL_FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "fixtures" / "variant_clinical"
)


# --- pure config-parse tests (no model / fixtures) -----------------------------


def test_config_defaults() -> None:
    """Defaults: masked_marginals scoring, batch size 64, no assay cap."""
    cfg = ProteinGymClinicalTaskConfig.from_extra({})
    assert cfg.scoring == "masked_marginals"
    assert cfg.mask_batch_size == 64
    assert cfg.max_assays is None


def test_config_rejects_unknown_scoring() -> None:
    """An unknown scoring method is rejected."""
    with pytest.raises(ValueError, match="scoring"):
        ProteinGymClinicalTaskConfig.from_extra({"scoring": "bogus"})


@pytest.mark.parametrize("bad_batch", [0, 99999])
def test_config_mask_batch_size_range(bad_batch: int) -> None:
    """``mask_batch_size`` must lie in ``[1, 1024]``."""
    with pytest.raises(ValueError, match="mask_batch_size"):
        ProteinGymClinicalTaskConfig.from_extra({"mask_batch_size": bad_batch})


def test_config_negative_max_assays_rejected() -> None:
    """A negative ``max_assays`` is rejected."""
    with pytest.raises(ValueError, match="max_assays"):
        ProteinGymClinicalTaskConfig.from_extra({"max_assays": -1})


def test_config_accepts_wt_marginals() -> None:
    """``wt_marginals`` is a valid scoring method."""
    cfg = ProteinGymClinicalTaskConfig.from_extra({"scoring": "wt_marginals"})
    assert cfg.scoring == "wt_marginals"


# --- slow end-to-end test ------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("scoring", ["masked_marginals", "wt_marginals"])
def test_clinical_eval_auroc_in_unit_interval(
    scoring: str, make_model: Callable[..., OplmForMaskedLM]
) -> None:
    """Both scoring modes produce a finite mean AUROC in ``[0, 1]`` on real data."""
    from accelerate import Accelerator

    assert _CLINICAL_FIXTURE_DIR.is_dir(), (
        f"clinical fixture dir missing: {_CLINICAL_FIXTURE_DIR} (see fixtures/README.md)"
    )

    cfg = OplmConfig(
        model=OplmModelConfig(max_position_embeddings=_MAX_SEQ_LEN),
        train=TrainConfig(wandb_enabled=False),
        data=DataConfig(num_workers=0, pin_memory=False),
    )
    model = make_model(max_position_embeddings=_MAX_SEQ_LEN)
    entry = EvalDatasetEntry(
        name="clinical",
        path=str(_CLINICAL_FIXTURE_DIR),
        type="proteingym_clinical",
        schedule=ScheduleSpec("steps", 1),
        extra={"scoring": scoring},
    )
    task = ProteinGymClinicalEvalTask(entry, cfg)
    accelerator = Accelerator(cpu=True)

    metrics = task.evaluate(model, accelerator)

    assert "auroc" in metrics
    auroc = metrics["auroc"]
    assert math.isfinite(auroc)
    assert 0.0 <= auroc <= 1.0

    # Guard against a silent fallback to 0.0: the real fixture (412-aa WT, both
    # classes present) must actually be scored, not skipped for length / one-class.
    assert task._assays is not None and len(task._assays) == 1
    assert task._score_assay(task._assays[0], model, accelerator.device) is not None

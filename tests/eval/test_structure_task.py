"""Structure-eval task tests: a slow end-to-end run plus pure config-parse tests.

The slow test exercises the canonical-tokenizer swap (TODOS.md §4.2) and the
full categorical-Jacobian P@L path — wildtype + per-mutation forward passes,
the ``(L, A, L, A)`` reduction, and precision@L scoring — over real structures.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from oplm.config import (
    DataConfig,
    EvalDatasetEntry,
    OplmConfig,
    ScheduleSpec,
    TrainConfig,
)
from oplm.eval.tasks.structure import StructureEvalTask, StructureTaskConfig
from oplm.model import OplmConfig as OplmModelConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from oplm.model import OplmForMaskedLM

_MAX_SEQ_LEN = 128


# --- pure config-parse test (no model / fixtures) ------------------------------


def test_structure_config_rejects_string_bool() -> None:
    """A YAML string like ``"false"`` must not coerce to ``True`` (issue #5)."""
    with pytest.raises(ValueError, match="use_cbeta"):
        StructureTaskConfig.from_extra({"use_cbeta": "false"})


def test_structure_config_real_bools_parse() -> None:
    """Actual bools are accepted verbatim."""
    cfg = StructureTaskConfig.from_extra({"use_cbeta": False})
    assert cfg.use_cbeta is False


def test_structure_config_jacobian_sample_size_floor() -> None:
    """``categorical_jacobian_sample_size`` must be >= 1 when provided."""
    with pytest.raises(ValueError, match="categorical_jacobian_sample_size"):
        StructureTaskConfig.from_extra({"categorical_jacobian_sample_size": 0})


@pytest.mark.parametrize("bad_batch", [0, 21])
def test_structure_config_jacobian_mutation_batch_size_range(bad_batch: int) -> None:
    """``categorical_jacobian_mutation_batch_size`` must lie in ``[1, 20]``."""
    with pytest.raises(ValueError, match="categorical_jacobian_mutation_batch_size"):
        StructureTaskConfig.from_extra({"categorical_jacobian_mutation_batch_size": bad_batch})


# --- slow end-to-end test ------------------------------------------------------


@pytest.mark.slow
def test_structure_eval_precision_in_unit_interval(
    structure_fixtures_dir: Path, make_model: Callable[..., OplmForMaskedLM]
) -> None:
    """Categorical-Jacobian P@L runs over a few real structures and lands in ``[0, 1]``."""
    import torch

    from oplm.eval.context import DistContext

    cfg = OplmConfig(
        model=OplmModelConfig(max_position_embeddings=_MAX_SEQ_LEN),
        train=TrainConfig(wandb_enabled=False),
        data=DataConfig(num_workers=0, pin_memory=False),
    )
    model = make_model(max_position_embeddings=_MAX_SEQ_LEN)
    entry = EvalDatasetEntry(
        name="struct",
        path=str(structure_fixtures_dir),
        type="structure",
        schedule=ScheduleSpec("steps", 1),
        # Cap to a couple of small structures to keep the Jacobian's L×20
        # forward passes cheap on CPU.
        extra={"max_structures": 2},
    )
    task = StructureEvalTask(entry, cfg)
    # Single-rank CPU context: _gather_data short-circuits at world_size=1, so the
    # categorical-Jacobian path runs without a live process group.
    dist_ctx = DistContext(rank=0, world_size=1, device=torch.device("cpu"), is_main=True)

    metrics = task.evaluate(model, dist_ctx)

    assert "precision_at_L" in metrics
    p = metrics["precision_at_L"]
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0

"""Structure-eval task tests: a slow end-to-end run plus a pure config-parse test.

The slow test exercises BOTH migration gates at once — the canonical-tokenizer
swap (TODOS.md §4.2) and the new ``output_attentions`` / ``.attentions`` forward
API. If the forward call were still on the old ``need_weights`` /
``"attention_weights"`` contract, the task would raise here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from oplm.config import (
    DataConfig,
    EvalDatasetEntry,
    ModelConfig,
    OplmConfig,
    ScheduleSpec,
    TrainConfig,
)
from oplm.eval.tasks.structure import StructureEvalTask, StructureTaskConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from oplm.model import OplmForMaskedLM

_MAX_SEQ_LEN = 128


# --- pure config-parse test (no model / fixtures) ------------------------------


def test_structure_config_rejects_string_bool() -> None:
    """A YAML string like ``"false"`` must not coerce to ``True`` (issue #5)."""
    with pytest.raises(ValueError, match="use_logistic_regression"):
        StructureTaskConfig.from_extra({"use_logistic_regression": "false"})


def test_structure_config_real_bools_parse() -> None:
    """Actual bools are accepted verbatim."""
    cfg = StructureTaskConfig.from_extra(
        {"use_logistic_regression": False, "use_categorical_jacobian": True}
    )
    assert cfg.use_logistic_regression is False
    assert cfg.use_categorical_jacobian is True


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
    """Mean-attention P@L runs over a few real structures and lands in ``[0, 1]``."""
    from accelerate import Accelerator

    cfg = OplmConfig(
        model=ModelConfig(max_seq_len=_MAX_SEQ_LEN),
        train=TrainConfig(wandb_enabled=False),
        data=DataConfig(num_workers=0, pin_memory=False),
    )
    model = make_model(max_position_embeddings=_MAX_SEQ_LEN)
    entry = EvalDatasetEntry(
        name="struct",
        path=str(structure_fixtures_dir),
        type="structure",
        schedule=ScheduleSpec("steps", 1),
        # Real Python bools — from_extra rejects the string "false". The
        # mean-attention path keeps the run cheap with only a couple structures.
        extra={"max_structures": 2, "use_logistic_regression": False},
    )
    task = StructureEvalTask(entry, cfg)
    accelerator = Accelerator(cpu=True)

    metrics = task.evaluate(model, accelerator)

    assert "precision_at_L" in metrics
    p = metrics["precision_at_L"]
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0

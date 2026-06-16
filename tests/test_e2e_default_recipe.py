"""G13 — the production ``base.yaml`` default recipe trains + evals end-to-end.

Cross-module flow (config -> model -> Trainer -> eval), kept at ``tests/`` top
level by the mirror convention. Unlike the dataclass-built ``tiny_train_cfg``
helper (which stays on the conservative library fallbacks), this drives the real
Trainer over a config resolved from ``configs/*/base.yaml`` with only the *size*
overridden — so it locks in that the **production default recipe** (μP + Muon,
sandwich norm, a sigmoid attention output gate, a learnable value residual, and
Canon at all four positions with ``k=7``) trains and evaluates without shape or
optimizer-grouping errors. A regression that breaks the default recipe fails here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from oplm.config import load_config
from tests.training.conftest import FullRecordingCallback

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow

_MAX_STEPS = 4


def test_base_yaml_default_recipe_trains_and_evals(
    training_parquet: Path, tmp_path: Path
) -> None:
    """A tiny model built from the base.yaml defaults trains a few steps and evaluates."""
    from oplm.training.trainer import Trainer

    # Only the size is overridden; every architecture/optimizer field falls through
    # to the production base.yaml defaults this test exists to guard.
    cfg = load_config(
        [
            "model.hidden_size=64",
            "model.num_hidden_layers=2",
            "model.num_attention_heads=4",
            "model.max_position_embeddings=64",
            f"train.max_steps={_MAX_STEPS}",
            "train.warmup_steps=0",
            "train.wandb_enabled=false",
            'train.mixed_precision="no"',  # quote: bare `no` parses as YAML boolean False
            f"train.save_every={_MAX_STEPS}",
            "train.log_every=1",
            f"train.output_dir={tmp_path}",
            f"data.train={training_parquet}",
            "data.num_workers=0",
            "data.pin_memory=false",
        ]
    )

    # Guard the production default recipe — the point of this test.
    assert cfg.model.norm_strategy == "sandwich"
    assert cfg.model.attn_output_gate == "sigmoid"
    assert cfg.model.value_residual == "learnable"
    assert cfg.model.canon_enabled is True
    assert cfg.model.canon_positions == ["A", "B", "C", "D"]
    assert cfg.model.canon_kernel_sizes == [7] * cfg.model.num_hidden_layers
    assert cfg.train.optimizer == "muon"
    assert cfg.model.mup_enable is True

    # Attach an eval dataset (DataConfig.eval is a free-form, runtime-parsed field).
    cfg.data.eval = {
        "hd": {"path": str(training_parquet), "type": "sequence", "every": {"steps": 2}}
    }

    callback = FullRecordingCallback()
    trainer = Trainer(cfg, callbacks=[callback])
    trainer.train()

    # Training reached the final step and wrote its checkpoint.
    assert (tmp_path / f"checkpoint-{_MAX_STEPS}").is_dir()

    # Eval fired under the new default recipe, with finite losses.
    assert callback.eval_steps  # non-vacuous
    for _, metrics in callback.evals:
        hd_values = [v for k, v in metrics.items() if k.startswith("eval/hd/")]
        assert hd_values
        assert all(math.isfinite(v) for v in hd_values)
    assert trainer._last_eval_loss is not None
    assert math.isfinite(trainer._last_eval_loss)

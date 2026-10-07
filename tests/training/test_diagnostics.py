"""Weight-norm diagnostics: parameter grouping, pooled RMS, and the live callback."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from oplm.model import OplmConfig as OplmModelConfig
from oplm.model import OplmForMaskedLM
from oplm.training.diagnostics import GROUPS, param_group_name, weight_rms_by_group

if TYPE_CHECKING:
    from pathlib import Path


def _tiny_model() -> OplmForMaskedLM:
    return OplmForMaskedLM(
        OplmModelConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=64,
            value_residual="learnable",
            canon_enabled=True,
            canon_positions=["A", "B", "C", "D"],
            residual_gate="channel",
            attn_output_gate="sigmoid",
        )
    )


def test_every_parameter_lands_in_a_named_group() -> None:
    """Only the scalar value-residual lambdas fall through to ``other``."""
    model = _tiny_model()
    groups = {n: param_group_name(n, p.ndim) for n, p in model.named_parameters()}
    other = {n for n, g in groups.items() if g == "other"}
    assert all(n.endswith("value_residual_lambda") for n in other), other
    assert set(groups.values()) >= {
        "embed",
        "head_dense",
        "head_decoder",
        "attn",
        "mlp",
        "norm_gain",
        "residual_gate",
        "canon",
        "bias",
    }
    assert groups["oplm.backbone.layers.0.attention.q_norm.weight"] == "norm_gain"
    assert groups["lm_head.norm.weight"] == "norm_gain"
    assert groups["oplm.backbone.layers.0.attention.gate_proj.weight"] == "attn"
    assert groups["oplm.backbone.layers.0.ffn.conv_d.conv.weight"] == "canon"


def test_weight_rms_pools_within_a_group() -> None:
    """RMS is pooled over all elements of a group, not averaged per tensor."""
    named = [
        ("oplm.backbone.layers.0.attention.q_proj.weight", torch.full((2, 2), 3.0)),
        ("oplm.backbone.layers.0.attention.k_proj.weight", torch.zeros((2, 6))),
    ]
    rms = weight_rms_by_group(named)
    assert set(rms) == {"attn"}
    # 4 elements of 3.0 and 12 zeros: sqrt(36 / 16) = 1.5.
    assert rms["attn"] == pytest.approx(1.5)
    assert [g for g in GROUPS if g in rms] == ["attn"]


@pytest.mark.slow
def test_callback_logs_weight_and_update_metrics(training_parquet: Path, tmp_path: Path) -> None:
    """Every ``weight_diag_every`` steps the callback emits weight RMS and update ratios."""
    from oplm.training.trainer import Trainer
    from tests.training.conftest import tiny_train_cfg

    cfg = tiny_train_cfg(
        tmp_path, training_parquet, max_steps=4, log_every=1, weight_diag_every=2, optimizer="muon"
    )
    trainer = Trainer(cfg)
    logged: list[dict[str, float]] = []
    trainer.accelerator.log = lambda values, **_kw: logged.append(dict(values))  # type: ignore[method-assign]
    trainer.train()

    diag = [m for m in logged if "diag/weight_rms/attn" in m]
    assert [m["train/global_step"] for m in diag] == [2, 4]
    for m in diag:
        assert m["diag/weight_rms/attn"] > 0
        assert m["diag/weight_rms/embed"] > 0
        assert m["diag/weight_rms/norm_gain"] == pytest.approx(1.0, abs=0.05)
        # Something moved on every group that has trainable weights.
        assert m["diag/update_ratio/attn"] > 0
        assert m["diag/update_ratio/embed"] > 0
        assert m["diag/weight_step"] == m["train/global_step"]

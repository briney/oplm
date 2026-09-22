"""Loop metadata is a resume invariant even when tensor shapes are unchanged."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from oplm.config import OplmConfig, serialize_config
from oplm.model import OplmConfig as ModelConfig
from oplm.training.checkpoint import validate_loop_resume_compat

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "field,changed",
    [
        ("num_loops", 3),
        ("loop_strategy", "interleave"),
        ("loop_start", 1),
        ("loop_end", 2),
        ("num_hidden_layers", 5),
    ],
)
@pytest.mark.parametrize("artifact", ["yaml", "hf"])
def test_resume_rejects_loop_drift(
    tmp_path: Path, field: str, changed: object, artifact: str
) -> None:
    saved = OplmConfig(model=ModelConfig(num_hidden_layers=4, num_loops=2))
    if artifact == "yaml":
        (tmp_path / "config.yaml").write_text(serialize_config(saved))
    else:
        saved.model.save_pretrained(tmp_path / "hf")
    live = OplmConfig(model=ModelConfig(num_hidden_layers=4, num_loops=2))
    setattr(live.model, field, changed)
    with pytest.raises(ValueError, match=field):
        validate_loop_resume_compat(tmp_path, live)


@pytest.mark.parametrize("loops,start,end", [(1, 0, None), (2, 1, 2)])
def test_identical_order_does_not_permit_strategy_drift(
    tmp_path: Path,
    loops: int,
    start: int,
    end: int | None,
) -> None:
    cfg = OplmConfig(
        model=ModelConfig(num_hidden_layers=4, num_loops=loops, loop_start=start, loop_end=end)
    )
    (tmp_path / "config.yaml").write_text(serialize_config(cfg))
    cfg.model.loop_strategy = "interleave"
    with pytest.raises(ValueError, match="loop_strategy"):
        validate_loop_resume_compat(tmp_path, cfg)


def test_full_end_equivalence(tmp_path: Path) -> None:
    cfg = OplmConfig(model=ModelConfig(num_hidden_layers=4, num_loops=2))
    (tmp_path / "config.yaml").write_text(serialize_config(cfg))
    cfg.model.loop_end = 4
    validate_loop_resume_compat(tmp_path, cfg)


@pytest.mark.parametrize("artifact", ["yaml", "hf"])
def test_legacy_absent_fields_mean_ordinary_defaults(tmp_path: Path, artifact: str) -> None:
    data = {"num_hidden_layers": 4}
    if artifact == "yaml":
        (tmp_path / "config.yaml").write_text("model:\n  num_hidden_layers: 4\n")
    else:
        (tmp_path / "hf").mkdir()
        (tmp_path / "hf" / "config.json").write_text(json.dumps(data))
    cfg = OplmConfig(model=ModelConfig(num_hidden_layers=4))
    validate_loop_resume_compat(tmp_path, cfg)
    cfg.model.num_loops = 2
    with pytest.raises(ValueError, match="num_loops"):
        validate_loop_resume_compat(tmp_path, cfg)


def test_missing_metadata_only_permits_ordinary_defaults(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = OplmConfig(model=ModelConfig(num_hidden_layers=4, loop_end=4))
    validate_loop_resume_compat(tmp_path, cfg)
    assert "cannot verify" in caplog.text
    cfg.model.loop_start = 1
    with pytest.raises(ValueError, match="init_from"):
        validate_loop_resume_compat(tmp_path, cfg)


@pytest.mark.parametrize("body", ["model: [1, 2]", "model:\n  num_loops: 0", "model: bad"])
def test_present_malformed_metadata_rejected(tmp_path: Path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(body)
    with pytest.raises(ValueError):
        validate_loop_resume_compat(tmp_path, OplmConfig(model=ModelConfig()))


def test_absent_yaml_model_falls_back_to_hf(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("train: {}")
    cfg = OplmConfig(model=ModelConfig(num_hidden_layers=4, num_loops=2))
    cfg.model.save_pretrained(tmp_path / "hf")
    validate_loop_resume_compat(tmp_path, cfg)

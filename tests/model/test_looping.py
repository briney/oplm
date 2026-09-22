"""Shared layer schedule and execution contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from oplm.model import OplmConfig
from oplm.model.looping import resolve_layer_execution_order
from oplm.model.transformer import OplmStack


@pytest.mark.parametrize(
    ("strategy", "start", "end", "expected"),
    [
        ("stack", 0, None, (0, 1, 2, 3, 0, 1, 2, 3)),
        ("interleave", 0, None, (0, 0, 1, 1, 2, 2, 3, 3)),
        ("stack", 1, 3, (0, 1, 2, 1, 2, 3)),
        ("interleave", 1, 3, (0, 1, 1, 2, 2, 3)),
        ("stack", 0, 1, (0, 0, 1, 2, 3)),
        ("interleave", 3, 4, (0, 1, 2, 3, 3)),
    ],
)
def test_execution_orders(
    strategy: str, start: int, end: int | None, expected: tuple[int, ...]
) -> None:
    assert (
        resolve_layer_execution_order(
            4, num_loops=2, loop_strategy=strategy, loop_start=start, loop_end=end
        )
        == expected
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_loops": True},
        {"num_loops": 1.5},
        {"num_loops": 0},
        {"loop_start": -1},
        {"loop_start": True},
        {"loop_start": 0.5},
        {"loop_end": False},
        {"loop_end": 2.5},
        {"loop_end": 0},
        {"loop_end": 5},
        {"loop_start": 3, "loop_end": 2},
        {"loop_strategy": "unknown"},
    ],
)
def test_invalid_configuration_and_postconstruction_mutation(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="loop"):
        OplmConfig(num_hidden_layers=4, **overrides)
    cfg = OplmConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=4)
    for name, value in overrides.items():
        setattr(cfg, name, value)
    with pytest.raises(ValueError, match="loop"):
        OplmStack(cfg)


@pytest.mark.parametrize("strategy", ["stack", "interleave"])
@pytest.mark.parametrize(("start", "end"), [(0, None), (0, 1), (1, 3), (3, 4)])
def test_one_loop_is_ordinary(strategy: str, start: int, end: int | None) -> None:
    assert resolve_layer_execution_order(
        4, loop_strategy=strategy, loop_start=start, loop_end=end
    ) == (0, 1, 2, 3)


def test_loop_config_round_trip(tmp_path: Path) -> None:
    from oplm.config import load_config, serialize_config

    cfg = load_config(
        [
            "model.num_hidden_layers=4",
            "model.num_loops=3",
            "model.loop_strategy=interleave",
            "model.loop_start=1",
            "model.loop_end=null",
        ]
    )
    cfg.model.save_pretrained(tmp_path / "hf")
    loaded_model = OplmConfig.from_pretrained(tmp_path / "hf")
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(serialize_config(cfg))
    loaded_run = load_config(["--config", str(yaml_path)])
    for model in (loaded_model, loaded_run.model):
        assert (model.num_loops, model.loop_strategy, model.loop_start, model.loop_end) == (
            3,
            "interleave",
            1,
            None,
        )
        assert "layer_execution_order" not in model.to_dict()


@pytest.mark.parametrize(
    "override",
    [
        "model.num_loops=true",
        "model.num_loops=1.5",
        "model.loop_start=true",
        "model.loop_end=2.5",
        "model.loop_end=false",
    ],
)
def test_cli_rejects_noninteger_loop_fields(override: str) -> None:
    from oplm.config import load_config

    with pytest.raises(ValueError, match="loop"):
        load_config([override])

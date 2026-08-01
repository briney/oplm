from __future__ import annotations

import pytest

from oplm.slurm.config import PhaseTable, SlurmConfig

RAW = {
    "partition": "hpc-mid",
    "time_limit": {"default": "168:00:00", "analyze": "01:00:00"},
    "cpus_per_task": 128,
    "gpus_per_node": 8,
    "exclusive": True,
    "mem": "0",
    "log_dir": "/mnt/home/briney/logs",
    "env_file": "/mnt/home/briney/.env",
    "container_image": "/mnt/data/containers/dl.sqsh",
    "container_mounts": ["/mnt/data:/mnt/data", "/tmp:/tmp"],
    "install": "pip install oplm[train]",
    "max_concurrent": 4,
    "nodes": {
        "default": {"170M": 1, "400M": 4, "800M": 8, "1B": 8},
        "bridge": {"170M": 4},
        "confirm": {"800M": 8},
    },
    "max_batch_size": {"170M": 256, "400M": 256, "800M": 256, "1B": 128},
}


def test_scalar_table_applies_everywhere() -> None:
    table = PhaseTable.from_value(4, name="nodes")
    assert table.resolve(phase=None, preset=None) == 4
    assert table.resolve(phase="bridge", preset="800M") == 4


def test_preset_table_without_default_key() -> None:
    table = PhaseTable.from_value({"170M": 1, "400M": 4}, name="nodes")
    assert table.resolve(phase=None, preset="400M") == 4
    with pytest.raises(KeyError, match="800M"):
        table.resolve(phase=None, preset="800M")


def test_phase_override_beats_default() -> None:
    cfg = SlurmConfig.from_mapping(RAW)
    assert cfg.nodes.resolve(phase="coarse", preset="170M") == 1
    assert cfg.nodes.resolve(phase="bridge", preset="170M") == 4
    assert cfg.nodes.resolve(phase="confirm", preset="800M") == 8
    # A phase override that omits a preset falls back to default for that preset.
    assert cfg.nodes.resolve(phase="bridge", preset="400M") == 4


def test_time_limit_resolves_per_phase() -> None:
    """`time_limit` entries are bare values, not preset maps: they apply to every preset."""
    cfg = SlurmConfig.from_mapping(RAW)
    assert cfg.time_limit.resolve(phase="coarse", preset="170M") == "168:00:00"
    assert cfg.time_limit.resolve(phase="analyze", preset=None) == "01:00:00"
    # A phase with no override falls back to default regardless of preset.
    assert cfg.time_limit.resolve(phase="transfer", preset="1B") == "168:00:00"


def test_exact_preset_beats_wildcard() -> None:
    table = PhaseTable.from_value({"default": 1, "bridge": {"170M": 4}}, name="nodes")
    assert table.resolve(phase="bridge", preset="170M") == 4
    # bridge has no 800M entry and no wildcard, so default's wildcard applies.
    assert table.resolve(phase="bridge", preset="800M") == 1
    assert table.resolve(phase="coarse", preset="170M") == 1


def test_phase_wildcard_beats_default_exact_preset() -> None:
    """A phase's own wildcard must win over `default`'s exact-preset entry.

    This pins the priority ordering: `resolve` must exhaust the phase's own table (exact, then
    wildcard) before ever consulting `default`. A regression that instead checked exact-preset
    matches across both tables before falling back to wildcards would return `default`'s `1`
    here instead of `coarse`'s wildcard `8`.
    """
    table = PhaseTable.from_value({"default": {"170M": 1}, "coarse": 8}, name="nodes")
    assert table.resolve(phase="coarse", preset="170M") == 8


def test_missing_required_field_raises() -> None:
    raw = {key: value for key, value in RAW.items() if key != "partition"}
    with pytest.raises(ValueError, match="partition"):
        SlurmConfig.from_mapping(raw)


@pytest.mark.parametrize("field", ["gpus_per_node", "cpus_per_task", "max_concurrent"])
def test_non_positive_ints_rejected(field: str) -> None:
    raw = {**RAW, field: 0}
    with pytest.raises(ValueError, match=field):
        SlurmConfig.from_mapping(raw)


def test_non_positive_max_batch_size_rejected() -> None:
    raw = {**RAW, "max_batch_size": {"170M": 0}}
    with pytest.raises(ValueError, match="max_batch_size"):
        SlurmConfig.from_mapping(raw)


def test_bool_max_batch_size_rejected() -> None:
    """`bool` is an `int` subclass; a bare `int(value)` coercion would silently accept it."""
    raw = {**RAW, "max_batch_size": {"170M": True}}
    with pytest.raises(ValueError, match=r"max_batch_size\['170M'\].*True"):
        SlurmConfig.from_mapping(raw)


def test_non_numeric_max_batch_size_rejected() -> None:
    """A non-numeric value must raise this module's actionable message, not a raw int() error."""
    raw = {**RAW, "max_batch_size": {"170M": "oops"}}
    with pytest.raises(ValueError, match=r"max_batch_size\['170M'\].*oops"):
        SlurmConfig.from_mapping(raw)


from oplm.slurm.config import BatchPlan, resolve_batch_plan


@pytest.mark.parametrize(
    ("global_examples", "nodes", "cap", "expected"),
    [
        # The spec's node table: every row resolves to accum == 1.
        (2048, 1, 256, BatchPlan(per_device_batch=256, gradient_accumulation_steps=1, world_size=8)),
        (2048, 4, 256, BatchPlan(per_device_batch=64, gradient_accumulation_steps=1, world_size=32)),
        (2048, 8, 256, BatchPlan(per_device_batch=32, gradient_accumulation_steps=1, world_size=64)),
        (2048, 8, 128, BatchPlan(per_device_batch=32, gradient_accumulation_steps=1, world_size=64)),
        (8192, 4, 256, BatchPlan(per_device_batch=256, gradient_accumulation_steps=1, world_size=32)),
        (8192, 8, 256, BatchPlan(per_device_batch=128, gradient_accumulation_steps=1, world_size=64)),
    ],
)
def test_batch_plan_matches_the_spec_table(
    global_examples: int, nodes: int, cap: int, expected: BatchPlan
) -> None:
    assert (
        resolve_batch_plan(
            global_examples=global_examples, world_size=nodes * 8, max_batch_size=cap
        )
        == expected
    )


def test_cap_forces_accumulation() -> None:
    # 2048 / 8 = 256 per device at accum 1, over a cap of 128, so accum must rise to 2.
    plan = resolve_batch_plan(global_examples=2048, world_size=8, max_batch_size=128)
    assert plan == BatchPlan(per_device_batch=128, gradient_accumulation_steps=2, world_size=8)


def test_local_world_size_still_lands_on_the_global_batch() -> None:
    """--local passes its actual process count, so a 400M cell keeps a 2048 global batch."""
    plan = resolve_batch_plan(global_examples=2048, world_size=8, max_batch_size=256)
    assert plan.per_device_batch * plan.gradient_accumulation_steps * plan.world_size == 2048


def test_indivisible_global_batch_raises() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        resolve_batch_plan(global_examples=2048, world_size=24, max_batch_size=256)


def test_global_batch_smaller_than_world_raises() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        resolve_batch_plan(global_examples=4, world_size=8, max_batch_size=256)

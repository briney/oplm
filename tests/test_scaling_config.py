"""``configs/scaling.yaml`` is a real production config -- it must actually decay.

Regression for a bug where ``warmup_steps + stable_steps == max_steps`` left the decay
branch of ``get_schedule_fn`` unreachable: the config was labelled ``wsd_linear`` but
trained at peak LR for its entire duration. This asserts the *property* (decay actually
happens before the run ends), not specific step/ratio numbers, so a future retune of the
decay-tail length does not fail this test spuriously.
"""

from __future__ import annotations

from pathlib import Path

from oplm.config import load_config
from oplm.training.optim import get_schedule_fn

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCALING_CONFIG = _REPO_ROOT / "configs" / "scaling.yaml"


def test_scaling_config_schedule_has_a_real_decay_phase() -> None:
    """The schedule built from configs/scaling.yaml must decay before max_steps."""
    cfg = load_config(["--config", str(_SCALING_CONFIG)])
    train = cfg.train

    # The decay phase must have nonzero length: this is exactly the condition Finding 1
    # violated (warmup_steps + stable_steps == max_steps => decay_steps floors to 1 step
    # of no practical effect at peak LR).
    assert train.warmup_steps + train.stable_steps < train.max_steps

    schedule_fn = get_schedule_fn(
        train.scheduler,
        warmup_steps=train.warmup_steps,
        total_steps=train.max_steps,
        min_ratio=0.0,
        stable_steps=train.stable_steps,
    )

    midpoint_of_stable_phase = train.warmup_steps + train.stable_steps // 2
    near_the_end = train.max_steps - 1

    assert schedule_fn(midpoint_of_stable_phase) == 1.0
    assert schedule_fn(near_the_end) < 1.0


def test_scaling_config_loads_the_fault_tolerance_knobs() -> None:
    """``load_config`` must accept the Task 1.9 fault-tolerance keys in scaling.yaml.

    Regression for the production config gaining ``save_every_minutes``,
    ``keep_every_n_steps``, and an explicit ``auto_resume: true`` (this config only ever
    runs under Slurm, via ``oplm slurm generate``) -- and for the ``slurm:`` block, which
    ``load_config`` tolerates as an unrecognized top-level key, still round-tripping
    alongside them.
    """
    cfg = load_config(["--config", str(_SCALING_CONFIG)])
    train = cfg.train

    assert train.save_every_minutes == 30
    assert train.keep_every_n_steps == 100_000
    assert train.auto_resume is True
    assert train.save_total_limit == 3
    # Reserved fields stay untouched by this config -- Phase 4 activates remote sync.
    assert train.remote_checkpoint_uri is None

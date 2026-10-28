"""``configs/scaling.yaml`` is a real production config -- it must actually decay.

Regression for a bug where ``warmup_steps + stable_steps == max_steps`` left the decay
branch of ``get_schedule_fn`` unreachable: the config was labelled ``wsd_linear`` but
trained at peak LR for its entire duration. This asserts the *property* (decay actually
happens before the run ends), not specific step/ratio numbers, so a future retune of the
decay-tail length does not fail this test spuriously.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from oplm.config import load_config, resolve_token_accumulation
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

    Regression for the production config gaining ``keep_every_n_steps`` and an explicit
    ``auto_resume: true`` (this config only ever runs under Slurm, via
    ``oplm slurm generate``) -- and for the ``slurm:`` block, which
    ``load_config`` tolerates as an unrecognized top-level key, still round-tripping
    alongside them.
    """
    cfg = load_config(["--config", str(_SCALING_CONFIG)])
    train = cfg.train

    assert train.keep_every_n_steps == 50_000
    assert train.auto_resume is True
    # Reserved fields stay untouched by this config -- Phase 4 activates remote sync.
    assert train.remote_checkpoint_uri is None


def test_scaling_config_is_the_current_recipe() -> None:
    """configs/scaling.yaml matches the 170M long run's recipe (W&B oplm-170M-long)."""
    cfg = load_config(["--config", str(_SCALING_CONFIG), "--preset", "170M"])
    assert cfg.train.tokens_per_step == 4_194_304
    assert cfg.data.pack_sequences is True
    assert cfg.model.max_position_embeddings == 2048
    assert cfg.model.residual_gate == "none"
    assert cfg.train.weight_decay_to_init == 0.1
    assert cfg.train.lr == 0.0063
    assert cfg.model.packed_attention == "flex"
    assert cfg.train.mup_depth_lr_exponent == 0.5


def test_every_preset_plans_with_its_batch_tokens_cap() -> None:
    """Each preset's token plan divides with ``train.batch_tokens`` set to its cap.

    ``oplm slurm generate`` renders ``train.batch_tokens`` as-is (it does not read
    ``slurm.max_batch_tokens``), so the docs tell users to pass the preset's cap as an
    override. That override must give a valid accumulation at the preset's node count.
    """
    slurm = OmegaConf.load(_SCALING_CONFIG).slurm
    for preset, nodes in slurm.nodes.items():
        cap = slurm.max_batch_tokens[preset]
        cfg = load_config(
            ["--config", str(_SCALING_CONFIG), "--preset", preset, f"train.batch_tokens={cap}"]
        )
        assert resolve_token_accumulation(cfg.train, nodes * slurm.gpus_per_node) >= 1, preset

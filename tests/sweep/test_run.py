from __future__ import annotations

import json
from importlib.metadata import version
from typing import TYPE_CHECKING

import pytest

from oplm.config import OplmConfig, TrainConfig, serialize_config
from oplm.sweep import run
from oplm.sweep.run import latest_checkpoint
from tests.training.conftest import tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.slow
def test_mup_run_writes_validation_result(training_parquet: Path, tmp_path: Path) -> None:
    cfg = tiny_train_cfg(
        tmp_path / "trainer-output",
        training_parquet,
        max_steps=2,
        batch_size=4,
        log_every=1,
    )
    cfg.data.eval = {
        "heldout": {
            "path": str(training_parquet),
            "type": "sequence",
            "every": {"steps": 1},
        }
    }
    run_yaml = tmp_path / "run.yaml"
    result_json = tmp_path / "result.json"
    run_yaml.write_text(serialize_config(cfg))

    run.main(config=run_yaml, result=result_json)

    payload = json.loads(result_json.read_text())
    assert payload["steps"] == 2
    assert payload["global_batch"] == 4
    assert payload["eval"]["eval/heldout/loss"] > 0
    assert payload["oplm_version"] == version("oplm")


@pytest.mark.slow
def test_mup_run_auto_resumes_without_explicit_resume_from(
    training_parquet: Path, tmp_path: Path
) -> None:
    """A requeued cell (no ``resume_from`` pinned) picks up from its own checkpoint.

    ``save_every=1`` with a generous ``save_total_limit`` means every step gets its
    own checkpoint and none are rotated away. ``checkpoint-1``'s mtime is then
    direct evidence of whether step 1 ever ran a *second* time: a properly resumed
    phase 2 starts at global_step 2 and never revisits it, while a restart-from-0
    bug would necessarily rewrite it on its way back up to step 4.
    """
    output_dir = tmp_path / "trainer-output"

    # Phase 1: train to step 2, checkpointing every step.
    cfg1 = tiny_train_cfg(
        output_dir,
        training_parquet,
        max_steps=2,
        batch_size=4,
        log_every=1,
        save_every=1,
        save_total_limit=10,
    )
    run_yaml = tmp_path / "run.yaml"
    result_json = tmp_path / "result.json"
    run_yaml.write_text(serialize_config(cfg1))
    run.main(config=run_yaml, result=result_json)

    checkpoint_1_state = output_dir / "checkpoint-1" / "trainer_state.json"
    assert checkpoint_1_state.exists()
    mtime_after_phase1 = checkpoint_1_state.stat().st_mtime_ns

    # Phase 2: same output_dir, higher target, resume_from left unset — simulating
    # a Slurm `--requeue` relaunch of the same cell.
    cfg2 = tiny_train_cfg(
        output_dir,
        training_parquet,
        max_steps=4,
        batch_size=4,
        log_every=1,
        save_every=1,
        save_total_limit=10,
    )
    assert cfg2.train.resume_from is None
    run_yaml.write_text(serialize_config(cfg2))
    run.main(config=run_yaml, result=result_json)

    # Progress was made past the phase-1 target...
    assert (output_dir / "checkpoint-3").exists()
    assert (output_dir / "checkpoint-4").exists()
    # ...but checkpoint-1 was never rewritten: phase 2 started at global_step 2
    # (the checkpoint phase 1 left behind), not 0.
    assert checkpoint_1_state.stat().st_mtime_ns == mtime_after_phase1


# ---------------------------------------------------------------------------
# latest_checkpoint — numeric-max checkpoint discovery, no torch/training needed
# ---------------------------------------------------------------------------


def test_latest_checkpoint_none_when_empty(tmp_path: Path) -> None:
    assert latest_checkpoint(tmp_path) is None


def test_latest_checkpoint_missing_dir(tmp_path: Path) -> None:
    assert latest_checkpoint(tmp_path / "nope") is None


def test_latest_checkpoint_picks_numeric_max(tmp_path: Path) -> None:
    """checkpoint-9000 must lose to checkpoint-10000: numeric, not lexicographic."""
    for step in (1000, 9000, 10000):
        (tmp_path / f"checkpoint-{step}").mkdir()
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-10000"


def test_latest_checkpoint_ignores_malformed_names(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-1000").mkdir()
    (tmp_path / "checkpoint-final").mkdir()
    (tmp_path / "checkpoint-").mkdir()
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-1000"


def test_latest_checkpoint_ignores_non_directory_matches(tmp_path: Path) -> None:
    """A file (not a directory) named like a checkpoint must not win the max."""
    (tmp_path / "checkpoint-1000").mkdir()
    (tmp_path / "checkpoint-99999").write_text("not a checkpoint dir")
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-1000"


# ---------------------------------------------------------------------------
# main(): auto-resume wiring (heavy imports stubbed out; no real training)
#
# The checkpoint scan itself lives in Trainer.__init__ (Task 1.4's single resume code
# path) -- main() only needs to opt every cell into it by setting auto_resume=True, so
# these tests assert that wiring rather than the scan (covered by
# tests/training/test_e2e_checkpoint.py's real end-to-end auto_resume tests, and by
# test_mup_run_auto_resumes_without_explicit_resume_from below through the real Trainer).
# ---------------------------------------------------------------------------


class _RecordingTrainer:
    """Stands in for ``Trainer``: records the cfg it was constructed with; trains nothing."""

    def __init__(self, cfg: OplmConfig, callbacks: list[object]) -> None:
        self.cfg = cfg
        self.callbacks = callbacks

    def train(self) -> None:
        pass


def _patch_main_dependencies(monkeypatch: pytest.MonkeyPatch, cfg: OplmConfig) -> None:
    """Stub every heavy dependency ``main()`` imports so it runs with no real training."""
    monkeypatch.setattr("oplm.train._bootstrap_training_environment", lambda: None)
    monkeypatch.setattr("oplm.config.load_config", lambda _argv: cfg)
    monkeypatch.setattr("oplm.training.mup.SweepMetricsCallback", lambda path: object())
    monkeypatch.setattr("oplm.training.trainer.Trainer", _RecordingTrainer)


def test_main_sets_auto_resume_and_leaves_resume_from_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every cell opts into auto_resume; main() no longer scans for a checkpoint itself."""
    output_dir = tmp_path / "cell"
    (output_dir / "checkpoint-9000").mkdir(parents=True)

    cfg = OplmConfig(train=TrainConfig(output_dir=str(output_dir), resume_from=None))
    _patch_main_dependencies(monkeypatch, cfg)

    config_path = tmp_path / "run.yaml"
    config_path.touch()
    run.main(config=config_path, result=tmp_path / "result.json")

    assert cfg.train.auto_resume is True
    assert cfg.train.resume_from is None


def test_main_preserves_explicit_resume_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator-pinned resume_from must survive: main() never mutates it, and setting
    auto_resume=True is harmless -- Trainer's explicit resume_from always wins over the scan."""
    output_dir = tmp_path / "cell"
    (output_dir / "checkpoint-9000").mkdir(parents=True)
    pinned = str(tmp_path / "elsewhere" / "checkpoint-42")

    cfg = OplmConfig(train=TrainConfig(output_dir=str(output_dir), resume_from=pinned))
    _patch_main_dependencies(monkeypatch, cfg)

    config_path = tmp_path / "run.yaml"
    config_path.touch()
    run.main(config=config_path, result=tmp_path / "result.json")

    assert cfg.train.resume_from == pinned
    assert cfg.train.auto_resume is True

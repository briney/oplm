"""G13 — wandb tracker path (docs/TESTING_E2E.md §5, optional).

Runs a tiny trainer with ``wandb_enabled=True`` under ``WANDB_MODE=offline`` so the
real ``init_trackers`` + ``_config_to_flat_dict`` + ``accelerator.log`` path runs
without a network or login. Config flattening is a genuine serialization risk
(HF config + dataclasses -> one flat dict), so it is also asserted directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow


def test_config_to_flat_dict_is_flat_and_namespaced(training_parquet: Path, tmp_path: Path) -> None:
    """``_config_to_flat_dict`` flattens model/train/data into one namespaced dict."""
    from oplm.training.trainer import _config_to_flat_dict

    cfg = tiny_train_cfg(tmp_path, training_parquet)
    flat = _config_to_flat_dict(cfg)

    assert isinstance(flat, dict)
    assert all("/" in key for key in flat)
    for prefix in ("model/", "train/", "data/"):
        assert any(key.startswith(prefix) for key in flat), f"no {prefix} keys"
    # Representative fields survive the flattening.
    assert flat["train/max_steps"] == cfg.train.max_steps
    assert flat["model/hidden_size"] == cfg.model.hidden_size


def test_wandb_offline_run_completes(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wandb-enabled run completes end to end in offline mode (init_trackers + log)."""
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=3,
        wandb_enabled=True,
        wandb_run_name="e2e-offline",
    )
    callback = FullRecordingCallback()
    Trainer(cfg, callbacks=[callback]).train()

    assert callback.train_start_count == 1
    assert callback.train_end_count == 1
    assert callback.train_log_steps  # metrics were logged through the wandb tracker

    # wandb logs are consolidated under output_dir (dir=output_dir), not ./wandb.
    assert (tmp_path / "wandb").is_dir()

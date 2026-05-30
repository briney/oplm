"""G11 — training entrypoint e2e (docs/TESTING_E2E.md §5).

Cross-cutting flow, so it lives at ``tests/`` top level rather than under
``tests/training/`` (the mirror convention). Two paths are exercised:

* ``oplm.train.main(cfg)`` with a pre-built config runs a tiny real loop to a
  checkpoint; and
* a ``python -m oplm.train --preset small <overrides>`` subprocess covers the
  full ``argv -> load_config -> _bootstrap_training_environment -> main ->
  Trainer`` chain and exits 0.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from tests.training.conftest import tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow


def test_main_runs_tiny_loop_to_checkpoint(training_parquet: Path, tmp_path: Path) -> None:
    """``oplm.train.main(cfg)`` drives a real 2-step run and writes a checkpoint."""
    from oplm.train import main

    cfg = tiny_train_cfg(tmp_path, training_parquet, max_steps=2, save_every=2)
    main(cfg)

    assert (tmp_path / "checkpoint-2").is_dir()
    assert (tmp_path / "checkpoint-2" / "hf" / "model.safetensors").exists()


def test_module_subprocess_trains_and_exits_zero(training_parquet: Path, tmp_path: Path) -> None:
    """``python -m oplm.train`` parses argv, bootstraps, and trains a few steps."""
    output_dir = tmp_path / "run"
    overrides = [
        "model.hidden_size=32",
        "model.num_hidden_layers=2",
        "model.num_attention_heads=4",
        "model.max_position_embeddings=64",
        "train.max_steps=2",
        "train.warmup_steps=0",
        "train.wandb_enabled=false",
        'train.mixed_precision="no"',  # quote: bare `no` is YAML boolean False
        "train.save_every=2",
        "train.log_every=1",
        f"train.output_dir={output_dir}",
        f"data.train={training_parquet}",
        "data.num_workers=0",
        "data.pin_memory=false",
    ]
    cmd = [sys.executable, "-m", "oplm.train", "--preset", "small", *overrides]
    # Force CPU in the child so the run is fast and free of flex-attention compile.
    env = {**os.environ, "ACCELERATE_USE_CPU": "true"}

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert (output_dir / "checkpoint-2").is_dir()

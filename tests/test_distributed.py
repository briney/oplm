"""Slow smoke tests for distributed training."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

_ACCELERATE_TEST_PORT = "29601"


def _write_distributed_config(tmp_path: Path, training_parquet: Path) -> Path:
    """Write a tiny CPU-only training config for accelerate smoke tests."""
    config_path = tmp_path / "distributed-train.yaml"
    output_dir = tmp_path / "outputs"
    config_path.write_text(
        textwrap.dedent(
            f"""
            model:
              hidden_dim: 32
              num_layers: 1
              num_heads: 2
              num_kv_heads: 2
              max_seq_len: 64
            train:
              max_steps: 2
              batch_size: 2
              lr: 0.001
              warmup_steps: 0
              log_every: 1
              eval_every: 100
              save_every: 2
              wandb_enabled: false
              mixed_precision: "no"
              output_dir: {output_dir}
            data:
              train: {training_parquet}
              max_length: 64
              num_workers: 0
              pin_memory: false
            """
        ).strip()
        + "\n"
    )
    return config_path


@pytest.mark.slow
def test_cpu_distributed_training_smoke(tmp_path: Path, training_parquet: Path) -> None:
    """A two-process CPU accelerate launch should complete and save a checkpoint."""
    accelerate_cli = shutil.which("accelerate")
    if accelerate_cli is None:
        pytest.skip("accelerate CLI not installed")

    config_path = _write_distributed_config(tmp_path, training_parquet)
    output_dir = tmp_path / "outputs"
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")

    command = [
        accelerate_cli,
        "launch",
        "--cpu",
        "--num_processes",
        "2",
        "--num_cpu_threads_per_process",
        "1",
        "--main_process_port",
        os.environ.get("OPLM_ACCELERATE_TEST_PORT", _ACCELERATE_TEST_PORT),
        "-m",
        "oplm.train",
        "--config",
        str(config_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    trainer_state_path = output_dir / "checkpoint-2" / "trainer_state.json"
    assert trainer_state_path.exists()

    trainer_state = json.loads(trainer_state_path.read_text())
    assert trainer_state["global_step"] == 2
    assert trainer_state["tokens_seen"] > 0

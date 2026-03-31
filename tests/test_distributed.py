"""Slow smoke tests for distributed training."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from typing import TYPE_CHECKING

import pytest

from oplm.train import _bootstrap_training_environment

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


def test_training_bootstrap_disables_accidental_deepspeed(tmp_path: Path) -> None:
    """Default training bootstrap should force plain Accelerate/DDP mode."""
    env = {
        "ACCELERATE_USE_DEEPSPEED": "true",
        "ACCELERATE_DEEPSPEED_CONFIG_FILE": "deepspeed.json",
        "ACCELERATE_DEEPSPEED_MOE_LAYER_CLS_NAMES": "MyLayer",
        "ACCELERATE_CONFIG_DS_FIELDS": "zero_stage",
    }

    cache_dir = _bootstrap_training_environment(
        env,
        home_dir=tmp_path / "home",
        tmp_dir=tmp_path / "tmp",
    )

    assert env["ACCELERATE_USE_DEEPSPEED"] == "false"
    assert "ACCELERATE_DEEPSPEED_CONFIG_FILE" not in env
    assert "ACCELERATE_DEEPSPEED_MOE_LAYER_CLS_NAMES" not in env
    assert "ACCELERATE_CONFIG_DS_FIELDS" not in env
    assert env["TRITON_CACHE_DIR"] == str(cache_dir)
    assert cache_dir.exists()


def test_training_bootstrap_preserves_explicit_deepspeed_opt_in(tmp_path: Path) -> None:
    """Explicit opt-in should preserve DeepSpeed launcher state."""
    env = {
        "OPLM_ENABLE_DEEPSPEED": "1",
        "ACCELERATE_USE_DEEPSPEED": "true",
        "ACCELERATE_DEEPSPEED_CONFIG_FILE": "deepspeed.json",
    }

    _bootstrap_training_environment(
        env,
        home_dir=tmp_path / "home",
        tmp_dir=tmp_path / "tmp",
    )

    assert env["ACCELERATE_USE_DEEPSPEED"] == "true"
    assert env["ACCELERATE_DEEPSPEED_CONFIG_FILE"] == "deepspeed.json"


def test_training_bootstrap_preserves_existing_triton_cache_dir(tmp_path: Path) -> None:
    """User-provided Triton cache dirs should not be replaced."""
    custom_cache_dir = tmp_path / "custom-triton-cache"
    custom_cache_dir.mkdir()
    env = {"TRITON_CACHE_DIR": str(custom_cache_dir)}

    cache_dir = _bootstrap_training_environment(
        env,
        home_dir=tmp_path / "home",
        tmp_dir=tmp_path / "tmp",
    )

    assert cache_dir == custom_cache_dir
    assert env["TRITON_CACHE_DIR"] == str(custom_cache_dir)


def test_training_bootstrap_falls_back_to_tmp_triton_cache_dir(tmp_path: Path) -> None:
    """Bootstrap should fall back to /tmp-style storage when home cache creation fails."""
    home_file = tmp_path / "home-file"
    home_file.write_text("not-a-directory")
    tmp_root = tmp_path / "tmp"
    env: dict[str, str] = {}

    cache_dir = _bootstrap_training_environment(
        env,
        home_dir=home_file,
        tmp_dir=tmp_root,
    )

    assert cache_dir == tmp_root / "oplm-triton-cache" / "autotune"
    assert env["TRITON_CACHE_DIR"] == str(cache_dir)
    assert cache_dir.exists()


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
    combined_output = result.stdout + "\n" + result.stderr
    assert "Setting ds_accelerator to" not in combined_output
    assert "TorchCheckpointEngine" not in combined_output
    assert ".triton/autotune" not in combined_output

    trainer_state_path = output_dir / "checkpoint-2" / "trainer_state.json"
    assert trainer_state_path.exists()

    trainer_state = json.loads(trainer_state_path.read_text())
    assert trainer_state["global_step"] == 2
    assert trainer_state["tokens_seen"] > 0

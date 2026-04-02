"""Slow smoke tests for distributed training."""

from __future__ import annotations

import importlib.util
import json
import logging
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


def _write_distributed_config(
    tmp_path: Path,
    training_parquet: Path,
    *,
    optimizer: str = "adamw",
    max_steps: int | None = 2,
    max_epochs: int | None = None,
    batch_size: int = 2,
    save_every: int = 2,
) -> Path:
    """Write a tiny CPU-only training config for accelerate smoke tests."""
    config_path = tmp_path / "distributed-train.yaml"
    output_dir = tmp_path / "outputs"
    duration_line = (
        f"max_epochs: {max_epochs}" if max_epochs is not None else f"max_steps: {max_steps}"
    )
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
              optimizer: {optimizer}
              {duration_line}
              batch_size: {batch_size}
              lr: 0.001
              warmup_steps: 0
              log_every: 1
              eval_every: 100
              save_every: {save_every}
              wandb_enabled: false
              mixed_precision: "no"
              output_dir: {output_dir}
            data:
              train: {training_parquet}
              num_workers: 0
              pin_memory: false
            """
        ).strip()
        + "\n"
    )
    return config_path


def _run_accelerate_training(tmp_path: Path, config_path: Path) -> subprocess.CompletedProcess[str]:
    """Launch the training entrypoint under a 2-process CPU accelerate run."""
    accelerate_cli = shutil.which("accelerate")
    if accelerate_cli is None:
        pytest.skip("accelerate CLI not installed")

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
    return subprocess.run(
        command,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
        timeout=180,
        check=False,
    )


def test_training_bootstrap_disables_accidental_deepspeed(tmp_path: Path) -> None:
    """Default training bootstrap should force plain Accelerate/DDP mode."""
    logging.getLogger("DeepSpeed").disabled = False
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
    assert logging.getLogger("DeepSpeed").disabled is True
    assert env["TRITON_CACHE_DIR"] == str(cache_dir)
    assert cache_dir.exists()


def test_training_bootstrap_preserves_explicit_deepspeed_opt_in(tmp_path: Path) -> None:
    """Explicit opt-in should preserve DeepSpeed launcher state."""
    logging.getLogger("DeepSpeed").disabled = True
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
    assert logging.getLogger("DeepSpeed").disabled is False


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


@pytest.mark.skipif(
    not importlib.util.find_spec("deepspeed"),
    reason="deepspeed not installed",
)
def test_training_bootstrap_suppresses_deepspeed_import_logs() -> None:
    """Bootstrap should silence DeepSpeed's import-time logger noise."""
    result = subprocess.run(
        [
            "python",
            "-c",
            (
                "from oplm.train import _bootstrap_training_environment; "
                "_bootstrap_training_environment(); "
                "import deepspeed; "
                "print('ok')"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    combined_output = result.stdout + "\n" + result.stderr
    assert "Setting accelerator to CPU" not in combined_output
    assert "Setting ds_accelerator to" not in combined_output
    assert "TorchCheckpointEngine" not in combined_output


@pytest.mark.slow
def test_cpu_distributed_training_smoke(tmp_path: Path, training_parquet: Path) -> None:
    """A two-process CPU accelerate launch should complete and save a checkpoint."""
    config_path = _write_distributed_config(tmp_path, training_parquet)
    output_dir = tmp_path / "outputs"
    result = _run_accelerate_training(tmp_path, config_path)

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


@pytest.mark.slow
def test_cpu_distributed_training_smoke_muon(tmp_path: Path, training_parquet: Path) -> None:
    """A two-process CPU accelerate launch should complete with Muon enabled."""
    config_path = _write_distributed_config(tmp_path, training_parquet, optimizer="muon")
    output_dir = tmp_path / "outputs"
    result = _run_accelerate_training(tmp_path, config_path)

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

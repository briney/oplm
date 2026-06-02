"""G11 — training entrypoint (docs/TESTING_E2E.md §5).

Cross-cutting flow, so it lives at ``tests/`` top level rather than under
``tests/training/`` (the mirror convention). Three paths are exercised:

* a fast unit test that mocks the Trainer and ``torch.distributed`` to prove
  ``main(cfg)`` delegates straight to ``Trainer(cfg).train()``;
* ``oplm.train.main(cfg)`` with a pre-built config running a tiny real loop to a
  checkpoint; and
* a ``python -m oplm.train --preset 50M <overrides>`` subprocess covering the
  full ``argv -> load_config -> _setup_triton_cache -> main -> Trainer`` chain
  (forced onto CPU/Gloo via ``CUDA_VISIBLE_DEVICES=""``) that exits 0.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from tests.training.conftest import tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path


def _free_port() -> int:
    """Return an ephemeral localhost port for the child's process-group rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_main_delegates_to_trainer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``main(cfg)`` builds a Trainer from the config and calls ``train()`` once.

    The Trainer and every ``torch.distributed`` entry point it would touch are
    mocked, so this is a pure wiring check: no process group is formed and no model
    is built. ``tiny_train_cfg`` only assembles paths/fields (the parquet is never
    read at config time), so a non-existent data path is fine here.
    """
    import torch.distributed as dist

    import oplm.training

    # Isolate from any real launcher / collective backend.
    monkeypatch.setattr(dist, "init_process_group", lambda *a, **k: None)
    monkeypatch.setattr(dist, "destroy_process_group", lambda *a, **k: None)
    monkeypatch.setattr(dist, "get_rank", lambda *a, **k: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 1)

    recorded: dict[str, object] = {}

    class _FakeTrainer:
        def __init__(self, cfg: object) -> None:
            recorded["cfg"] = cfg

        def train(self) -> None:
            recorded["trained"] = True

    monkeypatch.setattr(oplm.training, "Trainer", _FakeTrainer)

    from oplm.train import main

    cfg = tiny_train_cfg(tmp_path, tmp_path / "unused.parquet", max_steps=1)
    main(cfg)

    assert recorded["cfg"] is cfg
    assert recorded.get("trained") is True


@pytest.mark.slow
def test_main_runs_tiny_loop_to_checkpoint(training_parquet: Path, tmp_path: Path) -> None:
    """``oplm.train.main(cfg)`` drives a real 2-step run and writes a checkpoint."""
    from oplm.train import main

    cfg = tiny_train_cfg(tmp_path, training_parquet, max_steps=2, save_every=2)
    main(cfg)

    assert (tmp_path / "checkpoint-2").is_dir()
    assert (tmp_path / "checkpoint-2" / "hf" / "model.safetensors").exists()


@pytest.mark.slow
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
        "train.fsdp_sharding_strategy=none",  # single-rank, no real sharding
        "train.save_every=2",
        "train.log_every=1",
        f"train.output_dir={output_dir}",
        f"data.train={training_parquet}",
        "data.num_workers=0",
        "data.pin_memory=false",
    ]
    cmd = [sys.executable, "-m", "oplm.train", "--preset", "50M", *overrides]
    # Hide the GPU so the child runs on CPU/Gloo: torch.cuda.is_available() -> False,
    # which is fast, deterministic (SDPA math backend), and avoids GPU contention.
    # Override MASTER_PORT with a free port: an earlier in-process Trainer in this
    # session set it to the default 29500, which the child would otherwise inherit and
    # collide with (EADDRINUSE).
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MASTER_PORT": str(_free_port())}

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert (output_dir / "checkpoint-2").is_dir()

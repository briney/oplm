"""Two gloo ranks on CPU: DDP-wrapped fold head, checkpoint at step 2, exact resume to step 4."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from oplm.config import serialize_config
from tests.fold.helpers import tiny_fold_run_cfg, tiny_lm
from tests.training.test_e2e_hsdp import _child_env

pytestmark = pytest.mark.slow


def _launch(cfg: Any, result: Path) -> list[dict[str, Any]]:
    result.mkdir(parents=True)
    config = result / "run.yaml"
    config.write_text(serialize_config(cfg))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:0",
        str(Path(__file__).with_name("_train_worker.py")),
        str(config),
        str(result),
    ]
    completed = subprocess.run(
        command, env=_child_env(), capture_output=True, text=True, timeout=600
    )
    (result / "stdout.log").write_text(completed.stdout)
    (result / "stderr.log").write_text(completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return [json.loads((result / f"rank{rank}.json").read_text()) for rank in (0, 1)]


def test_two_rank_training_checkpoints_and_resumes_the_example_stream(
    tmp_path: Path, fixture_dataset: Path
) -> None:
    lm_dir = tmp_path / "lm"
    tiny_lm().save_pretrained(lm_dir)
    full = _launch(
        tiny_fold_run_cfg(
            tmp_path / "full", fixture_dataset, lm_dir, max_steps=4, batch_size=1, ema_decay=0.5
        ),
        tmp_path / "full-result",
    )
    first = _launch(
        tiny_fold_run_cfg(
            tmp_path / "split",
            fixture_dataset,
            lm_dir,
            max_steps=2,
            batch_size=1,
            save_every=2,
            ema_decay=0.5,
        ),
        tmp_path / "first-result",
    )
    ckpt = tmp_path / "split" / "checkpoint-2"
    assert (ckpt / "ema.pt").is_file()
    second = _launch(
        tiny_fold_run_cfg(
            tmp_path / "split",
            fixture_dataset,
            lm_dir,
            max_steps=4,
            batch_size=1,
            save_every=2,
            ema_decay=0.5,
            resume_from=str(ckpt),
        ),
        tmp_path / "second-result",
    )
    for rank in (0, 1):
        assert full[rank]["is_ddp"] and full[rank]["world_size"] == 2
        assert full[rank]["all_losses_finite"] and second[rank]["all_losses_finite"]
        assert second[rank]["resumed_from_step"] == 2 and second[rank]["global_step"] == 4
        assert first[rank]["ids"] + second[rank]["ids"] == full[rank]["ids"]
        assert second[rank]["ema_n_averaged"] == 4
    assert full[0]["ids"] != full[1]["ids"]  # the ranks see different stripes
    # In-model draws are per rank: FoldTask.build_model seeds the global RNG with seed + rank.
    # The seeds are the direct check (the draws also diverge from data-dependent consumption,
    # e.g. the LM-input mask shape); sigma is continuous, so equal per-step values on both ranks
    # would still mean identical RNG streams.
    assert full[1]["initial_seed"] == full[0]["initial_seed"] + 1
    assert len(full[0]["sigma_mean"]) == len(full[1]["sigma_mean"]) == 4
    assert full[0]["sigma_mean"] != full[1]["sigma_mean"]
    assert len(full[0]["loops"]) == len(full[1]["loops"]) == 4

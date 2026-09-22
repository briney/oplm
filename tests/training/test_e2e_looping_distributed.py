"""Two-rank shared-layer gradients and weights-only stage/recovery integration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch

from oplm.config import serialize_config
from oplm.model import OplmForMaskedLM
from tests.training.conftest import configure_accelerator_device, tiny_train_cfg
from tests.training.test_e2e_hsdp import _child_env

if TYPE_CHECKING:
    from oplm.config import OplmConfig

pytestmark = pytest.mark.slow


def _launch(
    cfg: OplmConfig,
    result: Path,
    *,
    parity: bool = False,
    expect_failure: bool = False,
    gpu: bool = False,
) -> list[dict[str, Any]]:
    result.mkdir(parents=True)
    config = result / "run.yaml"
    config.write_text(serialize_config(cfg))
    env = _child_env()
    if gpu:
        import os

        env.pop("ACCELERATE_USE_CPU", None)
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:0",
        str(Path(__file__).with_name("_looping_worker.py")),
        str(config),
        str(result),
    ]
    if parity:
        command.append("--parity")
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=240)
    (result / "stdout.log").write_text(completed.stdout)
    (result / "stderr.log").write_text(completed.stderr)
    if expect_failure:
        assert completed.returncode != 0
        for rank in (0, 1):
            assert "num_loops" in (result / f"rank{rank}.error").read_text()
        return []
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return [json.loads((result / f"rank{rank}.json").read_text()) for rank in (0, 1)]


@pytest.mark.parametrize("parallelism", ["ddp", "hsdp"])
@pytest.mark.parametrize("strategy", ["stack", "interleave"])
@pytest.mark.parametrize("mode", ["full", "selective"])
def test_two_rank_gradients_and_update_match_unwrapped_reference(
    tmp_path: Path,
    training_parquet: Path,
    parallelism: str,
    strategy: str,
    mode: str,
) -> None:
    cfg = tiny_train_cfg(
        tmp_path / "run",
        training_parquet,
        num_hidden_layers=3,
        num_loops=2,
        loop_strategy=strategy,
        loop_start=1 if strategy == "interleave" else 0,
        value_residual="learnable",
        parallelism=parallelism,
        gradient_checkpointing=True,
        gradient_checkpointing_mode=mode,
        gradient_accumulation_steps=2,
        max_steps=2,
    )
    results = _launch(cfg, tmp_path / "parity", parity=True)
    for result in results:
        assert result["is_sharded"] == (parallelism == "hsdp")
        assert result["effective_depth"] == (5 if strategy == "interleave" else 6)
        assert result["num_unique_layers"] == 3
        assert result["accumulation"] == 2
        assert result["unique_optimizer_parameters"] > 0
        assert result["max_gradient_difference"] < 1e-5


@pytest.mark.parametrize("parallelism", ["ddp", "hsdp"])
def test_distributed_new_stage_resumes_without_parent_and_reshards(
    tmp_path: Path,
    training_parquet: Path,
    monkeypatch: pytest.MonkeyPatch,
    parallelism: str,
) -> None:
    from oplm.training.trainer import Trainer

    source = tmp_path / "parent"
    cfg = tiny_train_cfg(
        tmp_path / "child",
        training_parquet,
        max_steps=2,
        num_hidden_layers=3,
        value_residual="learnable",
    )
    parent = OplmForMaskedLM(cfg.model)
    parent.save_pretrained(source)
    cfg.train.init_from = str(source)
    cfg.train.parallelism = parallelism
    cfg.model.num_loops = 2
    cfg.model.loop_strategy = "interleave"
    cfg.model.loop_start = 1
    cfg.model.gradient_checkpointing = True
    cfg.model.gradient_checkpointing_mode = "selective"
    first = _launch(cfg, tmp_path / "first")
    loaded = torch.load(tmp_path / "first" / "initial.pt", weights_only=True)
    for name, expected in parent.state_dict().items():
        torch.testing.assert_close(loaded[name], expected, rtol=0, atol=0)
    for result in first:
        assert result["resumed_from_step"] == 0
        assert result["global_step"] == 2
        assert result["num_unique_layers"] == 3 and result["effective_depth"] == 5
        assert result["all_losses_finite"]
    assert first[0]["loss_count"] == 2
    checkpoint = Path(cfg.train.output_dir) / "checkpoint-2"
    expected = OplmForMaskedLM.from_pretrained(checkpoint / "hf").state_dict()
    source.rename(tmp_path / "removed-parent")
    cfg.train.auto_resume = True
    cfg.train.max_steps = 3
    second = _launch(cfg, tmp_path / "second")
    loaded = torch.load(tmp_path / "second" / "initial.pt", weights_only=True)
    for name, value in expected.items():
        torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)
    for result in second:
        assert result["resumed_from_step"] == 2 and result["global_step"] == 3
        assert result["all_losses_finite"]
    assert second[0]["loss_count"] == 1
    expected_model = OplmForMaskedLM.from_pretrained(
        Path(cfg.train.output_dir) / "checkpoint-3" / "hf"
    )
    assert expected_model.config.num_loops == 2
    # Both ranks must reject semantic drift before any distributed state restore.
    cfg.train.max_steps = 4
    cfg.model.num_loops = 3
    _launch(cfg, tmp_path / "mismatch", expect_failure=True)
    cfg.model.num_loops = 2
    cfg.train.parallelism = "ddp"
    configure_accelerator_device("cpu", monkeypatch)
    with pytest.raises(ValueError, match="world_size"):
        Trainer(cfg)
    cfg.train.resume_data_position = False
    single = Trainer(cfg)
    assert single.global_step == 3
    actual = single.accelerator.unwrap_model(single.model).state_dict()
    for name, expected in expected_model.state_dict().items():
        torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)
    single.train()
    assert single.global_step == 4


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
@pytest.mark.parametrize("parallelism", ["ddp", "hsdp"])
@pytest.mark.parametrize("strategy,mode", [("stack", "full"), ("interleave", "selective")])
def test_gpu_inductor_bf16_looped_training(
    tmp_path: Path,
    training_parquet: Path,
    parallelism: str,
    strategy: str,
    mode: str,
) -> None:
    cfg = tiny_train_cfg(
        tmp_path / "run",
        training_parquet,
        max_steps=2,
        compile=True,
        mixed_precision="bf16",
        parallelism=parallelism,
        num_hidden_layers=3,
        num_loops=2,
        loop_strategy=strategy,
        value_residual="learnable",
        gradient_checkpointing=True,
        gradient_checkpointing_mode=mode,
    )
    results = _launch(cfg, tmp_path / "gpu", gpu=True)
    assert results[0]["loss_count"] == 2
    assert all(result["all_losses_finite"] for result in results)

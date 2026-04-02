"""CLI smoke tests for public workflows."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import torch
from accelerate import Accelerator
from typer.testing import CliRunner

from oplm.cli import app
from oplm.config import ModelConfig, OplmConfig, TrainConfig
from oplm.data.tokenizer import ProteinTokenizer
from oplm.model.transformer import OplmForMLM
from oplm.training.checkpoint import save_checkpoint

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _write_train_config(tmp_path: Path, training_parquet: Path) -> Path:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            model:
              hidden_dim: 64
              num_layers: 2
              num_heads: 4
              num_kv_heads: 2
              max_seq_len: 64
            train:
              max_steps: 1
              batch_size: 4
              lr: 0.001
              warmup_steps: 0
              log_every: 1
              eval_every: 100
              save_every: 100
              wandb_enabled: false
              mixed_precision: "no"
              output_dir: {tmp_path / "outputs"}
            data:
              train: {training_parquet}
              num_workers: 0
              pin_memory: false
            """
        ).strip()
        + "\n"
    )
    return config_path


def _create_inference_checkpoint(tmp_path: Path) -> tuple[Path, torch.Tensor]:
    torch.manual_seed(0)
    cfg = OplmConfig(
        model=ModelConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            max_seq_len=64,
        ),
        train=TrainConfig(output_dir=str(tmp_path), wandb_enabled=False, mixed_precision="no"),
    )

    model = OplmForMLM(cfg.model)
    model.eval()

    accelerator = Accelerator(cpu=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    save_checkpoint(
        accelerator=accelerator,
        cfg=cfg,
        output_dir=str(tmp_path),
        global_step=1,
        epoch=0,
        samples_seen=0,
        tokens_seen=0,
        save_total_limit=1,
    )

    tokenizer = ProteinTokenizer()
    batch = tokenizer.batch_encode(["MKWVTFISLLLLFSSAYS", "ACDEFGHIKLMNPQ"], max_length=64)
    with torch.no_grad():
        expected = accelerator.unwrap_model(model).encoder(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )[0]

    return tmp_path / "checkpoint-1", expected


class TestCli:
    """Smoke tests for the CLI entrypoints."""

    def test_info_command(self, tmp_path: Path, training_parquet: Path) -> None:
        config_path = _write_train_config(tmp_path, training_parquet)

        result = runner.invoke(app, ["info", "--config", str(config_path)])

        assert result.exit_code == 0, result.stdout
        assert "OPLM Model Info" in result.stdout
        assert "Hidden dim" in result.stdout

    def test_info_command_accepts_override_flags(
        self, tmp_path: Path, training_parquet: Path
    ) -> None:
        config_path = _write_train_config(tmp_path, training_parquet)

        result = runner.invoke(
            app,
            [
                "info",
                "--config",
                str(config_path),
                "--override",
                "model.hidden_dim=80",
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert "Hidden dim" in result.stdout
        assert "80" in result.stdout

    def test_info_command_shows_dynamic_conv_schedule(
        self, tmp_path: Path, training_parquet: Path
    ) -> None:
        config_path = _write_train_config(tmp_path, training_parquet)

        result = runner.invoke(
            app,
            [
                "info",
                "--config",
                str(config_path),
                "--override",
                "model.conv_positions=ACD",
                "--override",
                "model.conv_kernel_size=3",
                "--override",
                "model.conv_kernel_schedule=block_step",
                "--override",
                "model.conv_kernel_increment=2",
                "--override",
                "model.conv_kernel_block_size=2",
                "--override",
                "model.conv_kernel_max_size=7",
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert "Conv kernels" in result.stdout
        assert "+2 every 2 layer(s)" in result.stdout
        assert "max 7" in result.stdout

    def test_train_command(self, tmp_path: Path, training_parquet: Path) -> None:
        config_path = _write_train_config(tmp_path, training_parquet)
        output_dir = tmp_path / "outputs"

        result = runner.invoke(app, ["train", "--config", str(config_path)])

        assert result.exit_code == 0, result.stdout
        assert (output_dir / "checkpoint-1").exists()

    def test_train_command_accepts_override_flags(
        self, tmp_path: Path, training_parquet: Path
    ) -> None:
        config_path = _write_train_config(tmp_path, training_parquet)
        override_output_dir = tmp_path / "override-outputs"

        result = runner.invoke(
            app,
            [
                "train",
                "--config",
                str(config_path),
                "--override",
                f"train.output_dir={override_output_dir}",
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert (override_output_dir / "checkpoint-1").exists()

    def test_encode_command_loads_checkpoint_directory(self, tmp_path: Path) -> None:
        checkpoint_dir, expected = _create_inference_checkpoint(tmp_path)
        output_path = tmp_path / "embeddings.pt"

        result = runner.invoke(
            app,
            [
                "encode",
                "MKWVTFISLLLLFSSAYS",
                "ACDEFGHIKLMNPQ",
                "--model",
                str(checkpoint_dir),
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0, result.stdout
        actual = torch.load(output_path, map_location="cpu")
        torch.testing.assert_close(actual, expected)

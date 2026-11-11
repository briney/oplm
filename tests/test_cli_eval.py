"""``oplm eval``: score a checkpoint's ``hf/`` export on configured eval tasks, outside training.

Uses the committed real antibody fixture and a tiny model, so the whole path (config
resolution, hf load, task registry, JSONL output) runs on CPU in the fast suite.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from accelerate import Accelerator
from typer.testing import CliRunner

from oplm.cli import app
from oplm.config import DataConfig, EvalDatasetEntry, OplmConfig, ScheduleSpec, TrainConfig
from oplm.eval.tasks.antibody_cdr3 import AntibodyCdr3EvalTask
from oplm.model import OplmConfig as OplmModelConfig
from oplm.model import OplmForMaskedLM
from tests.cli_output import plain

_FIXTURE = Path(__file__).resolve().parent / "data/fixtures/antibody/eval_gte2_head.parquet"
runner = CliRunner()


@pytest.fixture
def ckpt(tmp_path: Path) -> Path:
    """A checkpoint directory holding only a tiny model's ``hf/`` export."""
    model = OplmForMaskedLM(
        OplmModelConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=256,
        )
    )
    model.save_pretrained(tmp_path / "ckpt" / "hf")
    return tmp_path / "ckpt"


@pytest.fixture
def eval_yaml(tmp_path: Path) -> Path:
    """An eval-only config: one ``antibody_cdr3`` task named ``ab``."""
    path = tmp_path / "eval.yaml"
    path.write_text(
        "data:\n"
        "  num_workers: 0\n"
        "  pin_memory: false\n"
        "  eval:\n"
        f"    ab: {{path: {_FIXTURE}, type: antibody_cdr3}}\n"
    )
    return path


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_eval_matches_direct_task_call(ckpt: Path, eval_yaml: Path, tmp_path: Path) -> None:
    """The CLI's record carries exactly what the task returns when called directly."""
    out = tmp_path / "out.jsonl"
    result = runner.invoke(
        app, ["eval", str(ckpt), "-c", str(eval_yaml), "--precision", "no", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.read_text().strip() in result.output  # the same line is printed to stdout
    (record,) = _records(out)
    assert record.pop("ckpt") == str(ckpt)
    assert record.pop("task") == "ab"
    assert isinstance(record.pop("seconds"), float)

    model = OplmForMaskedLM.from_pretrained(ckpt / "hf").eval()
    cfg = OplmConfig(
        model=model.config,
        train=TrainConfig(wandb_enabled=False),
        data=DataConfig(num_workers=0, pin_memory=False),
    )
    entry = EvalDatasetEntry(
        name="ab", path=str(_FIXTURE), type="antibody_cdr3", schedule=ScheduleSpec("steps", 1)
    )
    expected = AntibodyCdr3EvalTask(entry, cfg).evaluate(model, Accelerator(cpu=True))
    assert record == pytest.approx(expected, rel=1e-4)


def test_eval_appends_output(ckpt: Path, eval_yaml: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    argv = ["eval", str(ckpt), "-c", str(eval_yaml), "--precision", "no", "-o", str(out)]
    assert runner.invoke(app, argv).exit_code == 0
    assert runner.invoke(app, argv).exit_code == 0
    assert len(_records(out)) == 2


def test_eval_falls_back_to_checkpoint_config(ckpt: Path, eval_yaml: Path, tmp_path: Path) -> None:
    shutil.copy(eval_yaml, ckpt / "config.yaml")
    out = tmp_path / "out.jsonl"
    result = runner.invoke(app, ["eval", str(ckpt), "--precision", "no", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert [r["task"] for r in _records(out)] == ["ab"]


def test_eval_unknown_only_errors(ckpt: Path, eval_yaml: Path) -> None:
    result = runner.invoke(app, ["eval", str(ckpt), "-c", str(eval_yaml), "--only", "nope"])
    assert result.exit_code != 0
    assert "nope" in plain(result.output)


def test_eval_no_tasks_errors(ckpt: Path, tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("data:\n  num_workers: 0\n")
    result = runner.invoke(app, ["eval", str(ckpt), "-c", str(empty)])
    assert result.exit_code != 0
    assert "data.eval" in plain(result.output)


def test_eval_no_config_errors(ckpt: Path) -> None:
    result = runner.invoke(app, ["eval", str(ckpt)])
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert "config" in plain(result.output)

"""`oplm fold` command group: registration and a tiny CPU run of bench-kernels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from oplm.cli import app
from oplm.config import serialize_config
from oplm.fold.cli import run_kernel_benchmark
from tests.cli_output import plain
from tests.fold.helpers import tiny_fold_config, tiny_fold_run_cfg, tiny_lm

runner = CliRunner()


def test_fold_help_lists_predict_and_make_fixtures() -> None:
    result = runner.invoke(app, ["fold", "--help"])
    assert result.exit_code == 0, result.output
    out = plain(result.output)
    assert "bench-kernels" in out and "predict" in out and "make-fixtures" in out


def test_run_kernel_benchmark_cpu_reference_and_unavailable_fused() -> None:
    report = run_kernel_benchmark(
        widths=[32],
        lengths=[16],
        directions=["outgoing", "incoming"],
        paths=["reference", "fused_autograd"],
        device="cpu",
        dtype="fp32",
        iters=1,
        warmup=0,
        chunk_size=8,
    )
    assert report["environment"]["torch"] and report["environment"]["device"] == "cpu"
    assert report["settings"]["compile_reference"] is False
    by_key = {(c["direction"], c["path"]): c for c in report["cases"]}
    assert len(by_key) == 4
    for direction in ("outgoing", "incoming"):
        ref = by_key[(direction, "reference")]
        assert ref["status"] == "ok" and ref["actual_path"] == "reference"
        assert ref["compiled"] is False
        for phase in ("forward", "forward_backward", "checkpointed"):
            assert ref[phase]["status"] == "ok" and ref[phase]["ms_per_iter"] > 0
        assert ref["max_abs_error_vs_fp32_reference"] == 0.0
        fused = by_key[(direction, "fused_autograd")]
        assert fused["status"] == "unavailable" and fused["actual_path"] == "reference"


def _bench_args(out: Path, *, device: str, paths: str) -> list[str]:
    opts = {
        "--dtype": "fp32",
        "--widths": "32",
        "--lengths": "16",
        "--iters": "1",
        "--warmup": "0",
        "--device": device,
        "--paths": paths,
        "--out": str(out),
    }
    return ["fold", "bench-kernels", *(tok for kv in opts.items() for tok in kv)]


def test_run_kernel_benchmark_records_error_vs_fp32_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: object) -> float:
        raise RuntimeError("kernel boom")

    monkeypatch.setattr("oplm.fold.cli._error_vs_fp32", boom)
    report = run_kernel_benchmark(
        widths=[32],
        lengths=[16],
        directions=["outgoing"],
        paths=["reference"],
        device="cpu",
        dtype="fp32",
        iters=1,
        warmup=0,
        chunk_size=8,
    )
    (case,) = report["cases"]
    assert case["forward"]["status"] == "ok"
    assert case["max_abs_error_vs_fp32_reference"] == {
        "status": "error",
        "error": "RuntimeError: kernel boom",
    }


def test_bench_kernels_command_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    paths = "reference,fused_forward_reference_backward"
    result = runner.invoke(app, _bench_args(out, device="cpu", paths=paths))
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["cases"][0]["status"] == "ok"
    # Non-tty output defaults to 80 columns; the table must still show full path/status cells.
    text = plain(result.output)
    assert "fused_forward_reference_backward" in text
    assert "unavailable" in text


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-only check")
def test_bench_kernels_rejects_cuda_device_without_cuda(tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    result = runner.invoke(app, _bench_args(out, device="cuda", paths="reference"))
    assert result.exit_code != 0
    assert "CUDA is not available" in plain(result.output)
    assert not out.exists()


def test_predict_cli_writes_a_cif(tmp_path: Path) -> None:
    pytest.importorskip("gemmi")
    from oplm.fold import OplmForFolding

    torch.manual_seed(0)
    cfg = tiny_fold_config(
        plddt_bins=10,
        pae_bins=8,
        pde_bins=8,
        confidence_dist_bins=5,
        distogram_bins=8,
        inference_num_steps=2,
        inference_num_loops=1,
        lm_hidden_size=32,
        lm_num_hidden_states=3,
    )
    OplmForFolding(cfg).save_pretrained(tmp_path / "head")
    tiny_lm().save_pretrained(tmp_path / "lm")
    out = tmp_path / "pred.cif"
    args = ["fold", "predict", "MKV", "B:GG", "--model", str(tmp_path / "head")]
    args += ["--lm", str(tmp_path / "lm"), "--out", str(out), "--seed", "0", "--device", "cpu"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert out.exists() and "pTM" in plain(result.output)


def test_preprocess_command_runs_on_the_fixtures(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent.parent / "fixtures" / "fold"
    args = ["fold", "preprocess", "--archive", str(fixtures / "mmcif")]
    args += ["--ccd", str(fixtures / "ccd_subset.cif"), "--out", str(tmp_path / "ds")]
    args += ["--workers", "1", "--cutoff", "1990-01-01"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "accepted" in result.output and (tmp_path / "ds" / "manifest.json").exists()


def test_fold_train_runs_a_tiny_cpu_run_from_a_config(
    tmp_path: Path, fixture_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    lm_dir = tmp_path / "lm"
    tiny_lm().save_pretrained(lm_dir)
    cfg = tiny_fold_run_cfg(tmp_path / "run", fixture_dataset, lm_dir, max_steps=2, save_every=2)
    config = tmp_path / "run.yaml"
    config.write_text(serialize_config(cfg))
    result = runner.invoke(app, ["fold", "train", "--config", str(config), "train.max_steps=2"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "run" / "checkpoint-2" / "hf" / "config.json").is_file()
    assert "pair_width" in plain(result.output)


def test_fold_train_refuses_a_config_without_a_fold_block(tmp_path: Path) -> None:
    result = runner.invoke(app, ["fold", "train", "train.max_steps=1"])
    assert result.exit_code != 0 and "fold:" in plain(result.output)


def test_fold_measure_writes_a_report(
    tmp_path: Path, fixture_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    lm_dir = tmp_path / "lm"
    tiny_lm().save_pretrained(lm_dir)
    config = tmp_path / "run.yaml"
    config.write_text(
        serialize_config(tiny_fold_run_cfg(tmp_path / "run", fixture_dataset, lm_dir))
    )
    out = tmp_path / "measure.json"
    result = runner.invoke(
        app, ["fold", "measure", "--config", str(config), "--steps", "2", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["settings"]["steps"] == 2 and len(report["steps"]) == 2
    assert all(torch.isfinite(torch.tensor(s["loss"])) for s in report["steps"])
    assert report["environment"]["torch"] and "git" in report["environment"]
    assert report["settings"]["pair_width"] == 32 and report["settings"]["crop_size"] == 128
    assert report["peak_allocated_gib"] is None  # CPU run


@pytest.mark.parametrize("width", [128, 256])
def test_proof_configs_load(width: int) -> None:
    from oplm.config import load_config

    cfg = load_config(["--config", f"configs/fold/proof-w{width}.yaml"])
    assert cfg.fold is not None and cfg.fold.model["pair_width"] == width
    assert cfg.train.optimizer == "adamw" and cfg.train.ema_decay == 0.999
    assert cfg.fold.crop_size == 384

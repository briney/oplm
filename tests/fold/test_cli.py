"""`oplm fold` command group: registration and a tiny CPU run of bench-kernels."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch
from typer.testing import CliRunner

from oplm.cli import app
from oplm.fold.cli import run_kernel_benchmark
from tests.cli_output import plain
from tests.fold.helpers import tiny_fold_config, tiny_lm

if TYPE_CHECKING:
    from pathlib import Path

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

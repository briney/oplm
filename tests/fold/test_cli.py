"""`oplm fold` command group: registration and a tiny CPU run of bench-kernels."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from oplm.cli import app
from oplm.fold.cli import run_kernel_benchmark
from tests.cli_output import plain

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def test_fold_help_lists_bench_kernels() -> None:
    result = runner.invoke(app, ["fold", "--help"])
    assert result.exit_code == 0, result.output
    assert "bench-kernels" in plain(result.output)


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


def test_bench_kernels_command_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    result = runner.invoke(
        app,
        [
            "fold",
            "bench-kernels",
            "--device",
            "cpu",
            "--dtype",
            "fp32",
            "--widths",
            "32",
            "--lengths",
            "16",
            "--paths",
            "reference",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--out",
            str(out),
        ],
        # The 10-column table is wider than the default 80 columns, which would ellipsize
        # the path cell ("refe…"); rich reads COLUMNS lazily, so widen it for this run.
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["cases"][0]["status"] == "ok"
    assert "reference" in plain(result.output)

"""Opt-in training profiler: phase timing window, trace window, summary, data-wait metric."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from oplm.training.profiling import PHASES, StepProfiler, allreduce_benchmark

if TYPE_CHECKING:
    from pathlib import Path


def _accelerator() -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"), process_index=0, num_processes=1, is_main_process=True
    )


def test_windows_time_phases_then_trace(tmp_path: Path) -> None:
    prof = StepProfiler(_accelerator(), tmp_path, start=2, steps=2)  # time 2-3, trace 4-5
    timed: list[int] = []
    for global_step in range(6):
        prof.begin_step(global_step)
        for name in PHASES:
            with prof.phase(name):
                torch.ones(64, 64) @ torch.ones(64, 64)
        if prof._step in prof.timing:
            timed.append(prof._step)
        prof.end_step(tokens=100)
    assert timed == [2, 3]
    summary = json.loads((tmp_path / "profile" / "summary_rank0.json").read_text())
    assert summary["timing_steps"] == [2, 3] and summary["tokens_per_step_median"] == 100
    assert set(summary["phase_ms_median"]) == set(PHASES)
    assert summary["step_ms_median"] >= sum(summary["phase_ms_median"].values()) * 0.99
    assert summary["step_ms_unsynced"] > 0
    assert (tmp_path / "profile" / "trace_rank0.json").is_file()
    assert "Self CPU" in (tmp_path / "profile" / "top_ops_cpu.txt").read_text()


def test_allreduce_benchmark_is_empty_for_one_process() -> None:
    assert allreduce_benchmark(_accelerator()) == []  # ty: ignore[invalid-argument-type]


@pytest.mark.slow
def test_trainer_profiles_and_logs_data_wait(training_parquet: Path, tmp_path: Path) -> None:
    from oplm.training.trainer import Trainer
    from tests.training.conftest import tiny_train_cfg

    cfg = tiny_train_cfg(tmp_path, training_parquet, max_steps=6, log_every=1)
    cfg.train.profile_start, cfg.train.profile_steps = 2, 2
    trainer = Trainer(cfg)
    logged: list[dict[str, float]] = []
    trainer.accelerator.log = lambda values, **_kw: logged.append(dict(values))  # type: ignore[method-assign]
    trainer.train()
    waits = [m["train/data_wait_s"] for m in logged if "train/data_wait_s" in m]
    assert len(waits) == 6 and all(w >= 0 for w in waits)
    summary = json.loads((tmp_path / "profile" / "summary_rank0.json").read_text())
    assert set(summary["phase_ms_median"]) == set(PHASES)
    assert summary["phase_ms_median"]["forward"] > 0 and summary["phase_ms_median"]["optimizer"] > 0
    assert (tmp_path / "profile" / "trace_rank0.json").is_file()

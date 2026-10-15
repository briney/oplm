"""Opt-in training profiler (``train.profile_start``): where does a training step's time go?

Three measurements, all written to ``<output_dir>/profile/`` and logged by the main process:

1. **Communication.** At train start, all-reduces of a few sizes on the run's own process
   group, reported as algorithm and bus bandwidth (the nccl-tests convention). A multi-node
   run that falls back from InfiniBand to sockets shows up here as a bus bandwidth of a few
   GB/s instead of hundreds. Run with ``slurm.nccl_debug: INFO`` to log the transport chosen.
2. **Phase timing.** For ``profile_steps`` optimizer steps from ``profile_start``, each phase of
   the step (data wait, forward, backward, gradient clipping, optimizer, the per-step sync) is
   timed with a CUDA synchronize on both sides, so GPU work is charged to the phase that issued
   it. The syncs remove overlap, so the phase sum exceeds an unprofiled step; the attribution
   is the point.
3. **Trace.** The next ``profile_steps`` steps run unsynced under ``torch.profiler``; rank 0
   writes a Chrome trace and the top operators by GPU and CPU time.
"""

from __future__ import annotations

import contextlib
import json
import logging
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from accelerate import Accelerator

__all__ = ["StepProfiler", "allreduce_benchmark"]

logger = logging.getLogger(__name__)

PHASES = ("data_wait", "forward", "backward", "clip", "optimizer", "sync")


def _sync(device: torch.device) -> None:
    """Wait for queued GPU work (a no-op off CUDA, e.g. CPU tests)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def allreduce_benchmark(
    accelerator: Accelerator, sizes_mb: tuple[int, ...] = (16, 128, 512), iters: int = 5
) -> list[dict[str, float]]:
    """Time fp32 all-reduces on this run's process group; empty for a single process.

    Returns one row per size: ``size_mb``, ``ms`` (median of ``iters`` after 2 warmups),
    ``algbw_gbs`` (size / time) and ``busbw_gbs`` (algbw x 2(n-1)/n, comparable to link speed).
    """
    world = accelerator.num_processes
    if world == 1:
        return []
    rows = []
    for size_mb in sizes_mb:
        x = torch.ones(size_mb * 2**20 // 4, dtype=torch.float32, device=accelerator.device)
        times = []
        for i in range(iters + 2):
            torch.distributed.barrier()
            _sync(accelerator.device)
            t0 = time.perf_counter()
            torch.distributed.all_reduce(x)
            _sync(accelerator.device)
            if i >= 2:
                times.append(time.perf_counter() - t0)
        seconds = statistics.median(times)
        algbw = size_mb * 2**20 / seconds / 1e9
        rows.append(
            {
                "size_mb": float(size_mb),
                "ms": seconds * 1e3,
                "algbw_gbs": algbw,
                "busbw_gbs": algbw * 2 * (world - 1) / world,
            }
        )
        del x
    return rows


class StepProfiler:
    """Phase timing, then a ``torch.profiler`` trace, over two consecutive step windows.

    Args:
        accelerator: The run's accelerator (device, rank, world size).
        output_dir: The run's output directory; results go to ``output_dir/profile``.
        start: First optimizer step (1-based) of the phase-timing window.
        steps: Optimizer steps in each window.
    """

    def __init__(self, accelerator: Accelerator, output_dir: str | Path, start: int, steps: int):
        self.accelerator = accelerator
        self.dir = Path(output_dir) / "profile"
        self.timing = range(start, start + steps)
        self.tracing = range(start + steps, start + 2 * steps)
        self._step = 0  # the optimizer step in progress, 1-based
        self._phases: dict[str, list[float]] = defaultdict(list)
        self._current: dict[str, float] = defaultdict(float)
        self._step_seconds: list[float] = []
        self._step_t0 = 0.0
        self._tokens: list[int] = []
        self._trace: torch.profiler.profile | None = None
        self._trace_t0 = 0.0
        self._unsynced_ms: float | None = None
        self.comm: list[dict[str, float]] = []

    def begin_step(self, global_step: int) -> None:
        """Mark the start of optimizer step ``global_step + 1``."""
        self._step = global_step + 1
        if self._step in self.timing:
            _sync(self.accelerator.device)
            self._step_t0 = time.perf_counter()
        if self._step == self.tracing.start:
            _sync(self.accelerator.device)
            self._trace_t0 = time.perf_counter()
        if self._step == self.tracing.start and self.accelerator.is_main_process:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self._trace = torch.profiler.profile(activities=activities)
            self._trace.__enter__()

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time ``name`` (CUDA-synced) while in the timing window; a no-op otherwise."""
        if self._step not in self.timing:
            yield
            return
        _sync(self.accelerator.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.accelerator.device)
            self._current[name] += time.perf_counter() - t0

    def end_step(self, tokens: int) -> None:
        """Close the optimizer step; ``tokens`` is this rank's token count for it."""
        if self._step in self.timing:
            _sync(self.accelerator.device)
            self._step_seconds.append(time.perf_counter() - self._step_t0)
            for name in PHASES:
                self._phases[name].append(self._current.get(name, 0.0))
            self._current.clear()
            self._tokens.append(tokens)
            if self._step == self.timing.stop - 1:
                self._write_summary()
        if self._step == self.tracing.stop - 1:
            _sync(self.accelerator.device)
            self._unsynced_ms = (time.perf_counter() - self._trace_t0) / len(self.tracing) * 1e3
            if self._trace is not None:
                self._trace.__exit__(None, None, None)
                self._write_trace()
                self._trace = None
            self._write_summary()

    def _write_trace(self) -> None:
        assert self._trace is not None
        self.dir.mkdir(parents=True, exist_ok=True)
        self._trace.export_chrome_trace(str(self.dir / "trace_rank0.json"))
        averages = self._trace.key_averages()
        keys = [("self_cpu_time_total", "cpu")]
        if torch.cuda.is_available():
            keys.append(("self_cuda_time_total", "gpu"))
        for key, name in keys:
            table = averages.table(sort_by=key, row_limit=40)
            (self.dir / f"top_ops_{name}.txt").write_text(table)

    def _write_summary(self) -> None:
        if not self._step_seconds:
            return
        rank = self.accelerator.process_index
        summary: dict[str, Any] = {
            "rank": rank,
            "world_size": self.accelerator.num_processes,
            "timing_steps": [self.timing.start, self.timing.stop - 1],
            "step_ms_median": statistics.median(self._step_seconds) * 1e3,
            # Mean step over the trace window, no per-phase syncs (rank 0 also profiles):
            # the natural speed, comparable with an unprofiled run's train/step_time_s.
            "step_ms_unsynced": self._unsynced_ms,
            "phase_ms_median": {k: statistics.median(v) * 1e3 for k, v in self._phases.items()},
            "tokens_per_step_median": statistics.median(self._tokens),
            "allreduce": self.comm,
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"summary_rank{rank}.json").write_text(json.dumps(summary, indent=1))
        if self.accelerator.is_main_process:
            phases = ", ".join(f"{k} {v:.1f}" for k, v in summary["phase_ms_median"].items())
            logger.info(
                "profile: synced step %.1f ms (%s ms); %d tokens/step on rank 0",
                summary["step_ms_median"],
                phases,
                summary["tokens_per_step_median"],
            )

"""Unit tests for Trainer compile wiring and Dynamo recompile-budget logic.

These tests monkeypatch ``torch.compile`` to capture call kwargs without
triggering a real compilation (which would be slow on CPU). They follow the
same pattern as ``test_compile_ddp_optimizer.py``: build a real
``Trainer.__init__`` on CPU with a tiny config, inspect side-effects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.training.conftest import configure_accelerator_device, tiny_train_cfg


def test_train_end_callback_is_not_emitted_on_nonmain_process() -> None:
    from oplm.training.trainer import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.accelerator = SimpleNamespace(is_main_process=False)
    calls: list[Trainer] = []

    class Callback:
        def on_train_end(self, callback_trainer: Trainer) -> None:
            calls.append(callback_trainer)

    trainer.callbacks = [Callback()]
    trainer._emit_train_end()
    assert calls == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CompileCapture:
    """Accumulates every ``torch.compile(model, ...)`` call for later inspection."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, model, **kwargs):
        self.calls.append(kwargs)
        return model  # pass through the unwrapped model (no real compilation)


class _ListHandler(logging.Handler):
    """Collect log records emitted to a given logger."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


# ---------------------------------------------------------------------------
# compile_dynamic wiring: kwarg reaches torch.compile + cache_size_limit raised
# ---------------------------------------------------------------------------


def test_compile_dynamic_false_passed_to_torch_compile(
    tmp_path: Path,
    training_parquet: Path,
    reset_dynamo: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compile_dynamic=False must be forwarded as ``dynamic=False`` to torch.compile
    AND Dynamo cache_size_limit must be raised when pad_to_multiple_of is set.

    The test monkeypatches ``torch.compile`` so no actual compilation occurs and
    the Trainer can be built on CPU. The captured kwargs are then inspected.
    """
    from torch import _dynamo

    from oplm.training import trainer as trainer_module
    from oplm.training.trainer import Trainer

    configure_accelerator_device("cpu", monkeypatch)

    capture = _CompileCapture()
    monkeypatch.setattr(trainer_module.torch, "compile", capture)

    # pad_to_multiple_of=32 so the bucket-raise branch executes (not the warning branch)
    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=1,
        compile=True,
        compile_dynamic=False,
        max_position_embeddings=64,
        pad_to_multiple_of=32,
    )

    original_limit = _dynamo.config.cache_size_limit
    Trainer(cfg)

    # The compile call must have forwarded dynamic=False
    assert len(capture.calls) == 1, f"Expected 1 torch.compile call, got {len(capture.calls)}"
    assert capture.calls[0]["dynamic"] is False, (
        f"Expected dynamic=False, got dynamic={capture.calls[0]['dynamic']!r}"
    )

    # The Dynamo cache_size_limit must have been raised above its original value.
    # With max_position_embeddings=64 and pad_to_multiple_of=32: buckets=2, limit=max(orig,10)
    assert _dynamo.config.cache_size_limit > original_limit, (
        f"cache_size_limit was not raised: "
        f"original={original_limit}, current={_dynamo.config.cache_size_limit}"
    )


def test_compile_dynamic_true_passes_through(
    tmp_path: Path,
    training_parquet: Path,
    reset_dynamo: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compile_dynamic=True (default) must be forwarded as ``dynamic=True`` to torch.compile.

    The recompile-budget branch must NOT fire when compile_dynamic is True.
    """
    from torch import _dynamo

    from oplm.training import trainer as trainer_module
    from oplm.training.trainer import Trainer

    configure_accelerator_device("cpu", monkeypatch)

    capture = _CompileCapture()
    monkeypatch.setattr(trainer_module.torch, "compile", capture)

    original_limit = _dynamo.config.cache_size_limit
    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=1,
        compile=True,
        compile_dynamic=True,
    )
    Trainer(cfg)

    assert len(capture.calls) == 1
    assert capture.calls[0]["dynamic"] is True
    # cache_size_limit must be untouched when compile_dynamic=True
    assert _dynamo.config.cache_size_limit == original_limit, (
        "cache_size_limit was unexpectedly modified for compile_dynamic=True"
    )


# ---------------------------------------------------------------------------
# Throughput accumulator unit tests
# ---------------------------------------------------------------------------


def test_throughput_logging(
    tmp_path: Path,
    training_parquet: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Throughput metrics are emitted correctly when window data is present.

    Drive the throughput accumulators directly: set the window fields, capture
    _log_metrics via monkeypatch, call _log_step, and assert the emitted metrics.

    Checks:
    - train/tokens_per_sec, train/step_time_s, train/achieved_tflops are present
      and numerically correct.
    - train/mfu appears only when peak_tflops is set.
    - Window resets to zero after logging.
    """
    from oplm.training import trainer as trainer_module
    from oplm.training.trainer import Trainer

    configure_accelerator_device("cpu", monkeypatch)

    # Monkeypatch torch.compile to avoid real compilation
    monkeypatch.setattr(trainer_module.torch, "compile", lambda model, **kw: model)

    # Build trainer without peak_tflops set
    cfg_no_peak = tiny_train_cfg(
        tmp_path / "no_peak",
        training_parquet,
        max_steps=1,
        compile=False,
    )
    trainer = Trainer(cfg_no_peak)

    # Seed window with known values
    window_tokens = 1000
    window_seconds = 0.5
    window_steps = 2
    trainer._tput_window_tokens = window_tokens
    trainer._tput_window_seconds = window_seconds
    trainer._tput_window_steps = window_steps

    # Capture _log_metrics calls
    captured: list[dict[str, float]] = []

    def _capture_log_metrics(metrics: dict[str, float]) -> None:
        captured.append(dict(metrics))

    monkeypatch.setattr(trainer, "_log_metrics", _capture_log_metrics)

    # Call _log_step (loss value is irrelevant for this test)
    trainer._log_step(0.5)

    assert len(captured) == 1
    m = captured[0]

    # Numerically correct
    expected_tps = window_tokens / window_seconds  # 2000.0
    expected_step_time = window_seconds / window_steps  # 0.25
    expected_tflops = trainer.flops_per_token * window_tokens / window_seconds / 1e12

    assert "train/tokens_per_sec" in m, "train/tokens_per_sec missing"
    assert "train/step_time_s" in m, "train/step_time_s missing"
    assert "train/achieved_tflops" in m, "train/achieved_tflops missing"
    assert abs(m["train/tokens_per_sec"] - expected_tps) < 1e-6, (
        f"tokens_per_sec: got {m['train/tokens_per_sec']}, expected {expected_tps}"
    )
    assert abs(m["train/step_time_s"] - expected_step_time) < 1e-9, (
        f"step_time_s: got {m['train/step_time_s']}, expected {expected_step_time}"
    )
    assert abs(m["train/achieved_tflops"] - expected_tflops) < 1e-12, (
        f"achieved_tflops: got {m['train/achieved_tflops']}, expected {expected_tflops}"
    )

    # train/mfu must NOT appear when peak_tflops is not set
    assert "train/mfu" not in m, f"train/mfu should be absent when peak_tflops=None, got {m}"

    # Window must reset
    assert trainer._tput_window_tokens == 0
    assert trainer._tput_window_seconds == 0.0
    assert trainer._tput_window_steps == 0


def test_throughput_mfu_emitted_when_peak_tflops_set(
    tmp_path: Path,
    training_parquet: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train/mfu is emitted when peak_tflops is set and is numerically correct."""
    from oplm.training import trainer as trainer_module
    from oplm.training.trainer import Trainer

    configure_accelerator_device("cpu", monkeypatch)

    monkeypatch.setattr(trainer_module.torch, "compile", lambda model, **kw: model)

    peak = 312.0  # arbitrary TFLOPs (e.g. A100 BF16 peak)
    cfg_with_peak = tiny_train_cfg(
        tmp_path / "with_peak",
        training_parquet,
        max_steps=1,
        compile=False,
    )
    # Set peak_tflops directly on the config (conftest exposes it via tiny_train_cfg but compile=False skips the compile block)
    cfg_with_peak.train.peak_tflops = peak

    trainer = Trainer(cfg_with_peak)

    window_tokens = 2000
    window_seconds = 1.0
    trainer._tput_window_tokens = window_tokens
    trainer._tput_window_seconds = window_seconds
    trainer._tput_window_steps = 4

    captured: list[dict[str, float]] = []
    monkeypatch.setattr(trainer, "_log_metrics", lambda m: captured.append(dict(m)))

    trainer._log_step(0.5)

    assert len(captured) == 1
    m = captured[0]

    assert "train/mfu" in m, "train/mfu must be present when peak_tflops is set"
    expected_achieved = trainer.flops_per_token * window_tokens / window_seconds / 1e12
    expected_mfu = expected_achieved / peak
    assert abs(m["train/mfu"] - expected_mfu) < 1e-12, (
        f"mfu: got {m['train/mfu']}, expected {expected_mfu}"
    )


def test_throughput_empty_window_emits_no_tput_keys(
    tmp_path: Path,
    training_parquet: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No throughput keys appear when the window is empty (warmup exclusion)."""
    from oplm.training import trainer as trainer_module
    from oplm.training.trainer import Trainer

    configure_accelerator_device("cpu", monkeypatch)

    monkeypatch.setattr(trainer_module.torch, "compile", lambda model, **kw: model)

    cfg = tiny_train_cfg(
        tmp_path / "empty_window",
        training_parquet,
        max_steps=1,
        compile=False,
    )
    trainer = Trainer(cfg)

    # Window is at defaults (all zeros) — no throughput metrics should appear
    captured: list[dict[str, float]] = []
    monkeypatch.setattr(trainer, "_log_metrics", lambda m: captured.append(dict(m)))

    trainer._log_step(0.5)

    assert len(captured) == 1
    m = captured[0]
    for key in ("train/tokens_per_sec", "train/step_time_s", "train/achieved_tflops", "train/mfu"):
        assert key not in m, f"{key} should be absent when window is empty, got {m}"


def test_compile_dynamic_false_no_pad_emits_warning(
    tmp_path: Path,
    training_parquet: Path,
    reset_dynamo: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compile_dynamic=False + pad_to_multiple_of=None must emit a logger.warning.

    Without padding bucketing, non-dynamic compile sees every unique sequence
    length as a new shape, thrashing Dynamo recompiles.
    """
    from oplm.training import trainer as trainer_module
    from oplm.training.trainer import Trainer

    configure_accelerator_device("cpu", monkeypatch)

    capture = _CompileCapture()
    monkeypatch.setattr(trainer_module.torch, "compile", capture)

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=1,
        compile=True,
        compile_dynamic=False,
        pad_to_multiple_of=None,
    )

    handler = _ListHandler()
    handler.setLevel(logging.WARNING)
    trainer_module.logger.addHandler(handler)
    try:
        Trainer(cfg)
    finally:
        trainer_module.logger.removeHandler(handler)

    warning_msgs = [r.getMessage() for r in handler.records if r.levelno == logging.WARNING]
    assert any(
        "thrash" in m or "unbounded" in m for m in warning_msgs
    ), (
        "Expected a warning about unbounded shapes / recompile thrashing; "
        f"got warning messages: {warning_msgs!r}"
    )

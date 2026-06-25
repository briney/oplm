"""Unit tests for Trainer compile wiring and Dynamo recompile-budget logic.

These tests monkeypatch ``torch.compile`` to capture call kwargs without
triggering a real compilation (which would be slow on CPU). They follow the
same pattern as ``test_compile_ddp_optimizer.py``: build a real
``Trainer.__init__`` on CPU with a tiny config, inspect side-effects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from tests.training.conftest import configure_accelerator_device, tiny_train_cfg

if TYPE_CHECKING:
    pass


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
    import torch._dynamo

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

    original_limit = torch._dynamo.config.cache_size_limit
    Trainer(cfg)

    # The compile call must have forwarded dynamic=False
    assert len(capture.calls) == 1, f"Expected 1 torch.compile call, got {len(capture.calls)}"
    assert capture.calls[0]["dynamic"] is False, (
        f"Expected dynamic=False, got dynamic={capture.calls[0]['dynamic']!r}"
    )

    # The Dynamo cache_size_limit must have been raised above its original value.
    # With max_position_embeddings=64 and pad_to_multiple_of=32: buckets=2, limit=max(orig,10)
    assert torch._dynamo.config.cache_size_limit > original_limit, (
        f"cache_size_limit was not raised: "
        f"original={original_limit}, current={torch._dynamo.config.cache_size_limit}"
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
    import torch._dynamo

    from oplm.training import trainer as trainer_module
    from oplm.training.trainer import Trainer

    configure_accelerator_device("cpu", monkeypatch)

    capture = _CompileCapture()
    monkeypatch.setattr(trainer_module.torch, "compile", capture)

    original_limit = torch._dynamo.config.cache_size_limit
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
    assert torch._dynamo.config.cache_size_limit == original_limit, (
        "cache_size_limit was unexpectedly modified for compile_dynamic=True"
    )


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

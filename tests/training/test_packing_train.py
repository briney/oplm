"""data.pack_sequences end to end: loader wiring, sample counting, and a compiled train run."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.training.conftest import tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path


def test_packing_refuses_mask_dropout(training_parquet: Path, tmp_path: Path) -> None:
    from oplm.data import build_train_dataloader

    cfg = tiny_train_cfg(tmp_path, training_parquet)
    cfg.data.pack_sequences = True
    cfg.model.mask_dropout = True
    with pytest.raises(ValueError, match="mask_dropout"):
        build_train_dataloader(cfg)


def _sequences(batch: dict) -> list[tuple[list[int], list[int]]]:
    """Each sequence's (input ids, labels), whichever layout the batch uses."""
    seg = batch.get("segment_ids", batch["attention_mask"])
    out = []
    for r in range(seg.shape[0]):
        for s in seg[r].unique().tolist():
            if s:
                cols = seg[r] == s
                out.append(
                    (batch["input_ids"][r, cols].tolist(), batch["labels"][r, cols].tolist())
                )
    return sorted(out)


def test_packing_serves_the_same_masked_sequences(training_parquet: Path, tmp_path: Path) -> None:
    """Packing only re-lays-out each batch, so the data stream (and a resume) is mode-agnostic."""
    import torch

    from oplm.data import build_train_dataloader

    streams = []
    for pack in (False, True):
        cfg = tiny_train_cfg(tmp_path, training_parquet, batch_size=8, max_position_embeddings=512)
        cfg.data.pack_sequences, cfg.data.num_workers = pack, 0
        torch.manual_seed(0)  # masking draws from the global RNG in the main process
        loader = iter(build_train_dataloader(cfg))
        streams.append([_sequences(next(loader)) for _ in range(3)])
    assert streams[0] == streams[1]


@pytest.mark.slow
def test_compiled_packed_training_counts_sequences(training_parquet: Path, tmp_path: Path) -> None:
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=4,
        batch_size=8,
        compile=True,
        max_position_embeddings=512,
    )
    cfg.data.pack_sequences = True
    cfg.data.num_workers = 0
    trainer = Trainer(cfg)
    batch = next(iter(trainer.dataloader))
    assert "segment_ids" in batch and batch["num_sequences"] == 8
    assert batch["input_ids"].shape[0] < 8  # short fixture sequences share rows
    logged: list[dict[str, float]] = []
    trainer.accelerator.log = lambda values, **_kw: logged.append(dict(values))  # type: ignore[method-assign]
    trainer.train()
    losses = [m["train/loss"] for m in logged if "train/loss" in m]
    assert len(losses) == 4 and all(loss == loss for loss in losses)  # finite, not NaN
    assert trainer._samples_seen == 4 * 8


def _gate_kernel_on(monkeypatch: pytest.MonkeyPatch, gate_dtypes: list) -> None:
    """Dispatch predicate says "kernel" and the kernel is the reference; the trainer's gate call
    (an empty probe tensor) records the dtype it asks about."""
    from oplm.model import conv

    def use_kernel(x, out_dtype):  # noqa: ANN001, ANN202
        if x.numel() == 0:
            gate_dtypes.append(out_dtype)
        return True

    monkeypatch.setattr(conv, "_use_kernel", use_kernel)
    monkeypatch.setattr(conv, "canon_conv_kernel", conv.canon_conv_reference)


def _canon_cfg(tmp_path: Path, training_parquet: Path, **kwargs):  # noqa: ANN003, ANN202
    cfg = tiny_train_cfg(tmp_path, training_parquet, batch_size=16, **kwargs)
    cfg.data.pack_sequences = True
    cfg.model.canon_enabled, cfg.model.canon_positions = True, ["A", "B", "C", "D"]
    cfg.model.canon_kernel_sizes = [3, 4]
    return cfg


@pytest.mark.slow
def test_canon_self_check_runs_once_on_first_rows(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trainer checks the Canon kernel once, on the first micro-batch's first 4 rows.

    CPU stand-in for a bf16 CUDA run: the dispatch predicate says "kernel" and the kernel is the
    reference, so the check runs and training is unchanged. The gate asks about the autocast
    dtype, as the model's Canon calls will.
    """
    import torch

    from oplm.model import canon_kernel, conv
    from oplm.training.trainer import Trainer

    cfg = _canon_cfg(tmp_path, training_parquet, max_steps=3, mixed_precision="bf16")
    gate_dtypes: list = []
    _gate_kernel_on(monkeypatch, gate_dtypes)
    checked: list[int] = []
    real_check = canon_kernel.canon_self_check

    def check(model, run_forward):  # noqa: ANN001, ANN202
        checked.append(real_check(model, run_forward, kernel=conv.canon_conv_reference))
        return checked[-1]

    monkeypatch.setattr(canon_kernel, "canon_self_check", check)
    trainer = Trainer(cfg)
    rows: list[tuple[int, ...]] = []
    step = trainer.task.step

    def spy(model, batch):  # noqa: ANN001, ANN202
        rows.append(tuple(v.shape[0] for v in batch.values() if hasattr(v, "shape")))
        return step(model, batch)

    trainer.task.step = spy  # ty: ignore[invalid-assignment]
    trainer.train()
    assert gate_dtypes == [torch.bfloat16]
    assert checked == [2 * 6]  # 2 layers x (A, B-q, B-k, B-v, C, D), once
    assert rows[0] == (4, 4, 4, 4) and rows[1][0] > 4  # the check's slice, then the full batch
    assert len(rows) == 1 + 3


@pytest.mark.slow
def test_canon_self_check_raises_when_no_call_checked(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kernel is in use but the check saw no Canon call: refuse to train unchecked."""
    from oplm.model import canon_kernel
    from oplm.training.trainer import Trainer

    _gate_kernel_on(monkeypatch, [])
    monkeypatch.setattr(canon_kernel, "canon_self_check", lambda model, run_forward: 0)
    trainer = Trainer(_canon_cfg(tmp_path, training_parquet, max_steps=1))
    with pytest.raises(RuntimeError, match="no Canon calls"):
        trainer.train()

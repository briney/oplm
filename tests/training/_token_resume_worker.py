"""Worker for the multi-rank token-mode resume tests. Run under ``torch.distributed.run``.

Trains one phase of a token-mode run on CPU/gloo ranks and writes, per rank, every consumed
batch's sequence ids and the rank's epoch at the time to ``out_dir/<phase>-rank<r>.json``.
Phases: ``control`` (0 -> total), ``first`` (0 -> stop, one checkpoint at stop), ``second``
(auto-resume -> total). ``train_data`` is a parquet path or a JSON ``data.train`` mix mapping;
an optional last argument ``false`` sets ``data.drop_last=false``.

Also the home of the recording helpers the in-process token resume tests share.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pytest

    from oplm.training.trainer import Trainer


def patch_for_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep sequence ids in batches, and tag every mix row's id with its source pass (``@p``).

    A mix serves each source pass at most once per row, so the tag turns "exact per source pass"
    into "each tagged id at most once per epoch".
    """
    from oplm.data.sequence.dataset import InterleavedDataset
    from tests.training.test_e2e_data_exact import _force_keep_sequence_ids

    _force_keep_sequence_ids(monkeypatch)
    original = InterleavedDataset._open

    def _open(
        self: InterleavedDataset, source_idx: int, pass_index: int, offset: int = 0
    ) -> Iterator[dict[str, object]]:
        for row in original(self, source_idx, pass_index, offset):
            yield {**row, "sequence_id": f"{row['sequence_id']}@{pass_index}"}

    monkeypatch.setattr(InterleavedDataset, "_open", _open)


def record_batches(trainer: Trainer) -> list[dict[str, object]]:
    """Wrap ``trainer.dataloader`` to record each batch's ``{"epoch", "ids"}``; returns the list."""
    records: list[dict[str, object]] = []
    inner = trainer.dataloader

    class _Recorder:
        dataset = property(lambda self: inner.dataset)

        def set_epoch(self, epoch: int) -> None:
            inner.set_epoch(epoch)

        def __len__(self) -> int:
            return len(inner)

        def __iter__(self):  # noqa: ANN204
            for batch in inner:
                records.append({"epoch": trainer.epoch, "ids": list(batch["sequence_ids"])})
                yield batch

    trainer.dataloader = _Recorder()  # type: ignore[assignment]
    return records


def main(
    run_dir: str,
    train_data: str,
    out_dir: str,
    phase: str,
    total: int,
    stop: int,
    drop_last: bool = True,
) -> None:
    import pytest

    from oplm.training.trainer import Trainer
    from tests.training.conftest import tiny_train_cfg

    monkeypatch = pytest.MonkeyPatch()
    patch_for_recording(monkeypatch)
    # Every fixture sequence fills a 64-token row, which gives both ranks the same batch count
    # per epoch; 256-token rows pack each rank's sequences differently, so their epochs end at
    # different steps. Two rows per batch (at 2 ranks), accumulation 1.
    budget = dict(
        tokens_per_step=1024,
        batch_tokens=512,
        max_position_embeddings=256,
        pack_sequences=True,
        drop_last=drop_last,
    )
    mix = train_data.startswith("{")
    data = Path("mix" if mix else train_data)
    if phase == "first":
        cfg = tiny_train_cfg(
            Path(run_dir), data, max_steps=stop, save_every=stop, save_final=False, **budget
        )
    else:
        cfg = tiny_train_cfg(
            Path(run_dir),
            data,
            max_steps=total,
            save_final=False,
            auto_resume=phase == "second",
            **budget,
        )
    if mix:
        cfg.data.train = json.loads(train_data)
    trainer = Trainer(cfg, callbacks=[])
    records = record_batches(trainer)
    trainer.train()
    rank = trainer.accelerator.process_index
    (Path(out_dir) / f"{phase}-rank{rank}.json").write_text(json.dumps(records))
    monkeypatch.undo()


if __name__ == "__main__":
    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        int(sys.argv[5]),
        int(sys.argv[6]),
        drop_last=sys.argv[7:] != ["false"],
    )

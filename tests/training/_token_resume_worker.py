"""Worker for the two-rank token-mode resume test. Run under ``torch.distributed.run``.

Trains one phase of a token-mode run on 2 CPU/gloo ranks and writes, per rank, every consumed
batch's sequence ids and the rank's epoch at the time to ``out_dir/<phase>-rank<r>.json``.
Phases: ``control`` (0 -> total), ``first`` (0 -> stop, one checkpoint at stop), ``second``
(auto-resume -> total).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(run_dir: str, train_data: str, out_dir: str, phase: str, total: int, stop: int) -> None:
    import pytest

    from oplm.training.trainer import Trainer
    from tests.training.conftest import tiny_train_cfg
    from tests.training.test_e2e_data_exact import _force_keep_sequence_ids

    monkeypatch = pytest.MonkeyPatch()
    _force_keep_sequence_ids(monkeypatch)
    # Every fixture sequence fills a 64-token row, which gives both ranks the same batch count
    # per epoch; 256-token rows pack each rank's sequences differently, so their epochs end at
    # different steps. Two rows per batch, accumulation 1.
    budget = dict(
        tokens_per_step=1024, batch_tokens=512, max_position_embeddings=256, pack_sequences=True
    )
    if phase == "first":
        cfg = tiny_train_cfg(
            Path(run_dir),
            Path(train_data),
            max_steps=stop,
            save_every=stop,
            save_final=False,
            **budget,
        )
    else:
        cfg = tiny_train_cfg(
            Path(run_dir),
            Path(train_data),
            max_steps=total,
            save_final=False,
            auto_resume=phase == "second",
            **budget,
        )
    trainer = Trainer(cfg, callbacks=[])
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
    trainer.train()
    rank = trainer.accelerator.process_index
    (Path(out_dir) / f"{phase}-rank{rank}.json").write_text(json.dumps(records))
    monkeypatch.undo()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))

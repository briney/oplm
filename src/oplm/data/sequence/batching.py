"""Token-budget batching: fixed ``rows x width`` micro-batches filled from a sequence stream.

:class:`TokenBatcher` wraps the training dataset inside each DataLoader worker. It places
sequences in stream order into ``rows`` rows of ``width`` tokens (first fit) and closes a batch
at the first sequence that fits in no row; that sequence starts the next batch. Every batch is a
contiguous run of the worker's stream, so the number of rows emitted so far is an exact resume
position whatever the batch size (token-batching design spec, §5.3 and §6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import IterableDataset, get_worker_info

from oplm.data.sequence.collate import RowFiller, token_length
from oplm.data.sequence.dataset import POSITION_KEY

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from oplm.data.sequence.collate import MLMCollator
    from oplm.data.sequence.dataset import Reshard

__all__ = ["TokenBatcher"]


class TokenBatcher(IterableDataset[dict[str, object]]):
    """Yield fixed-shape packed, masked batches of whole sequences from ``dataset``.

    Args:
        dataset: The training stream (``InterleavedDataset`` or ``ShardedProteinDataset``); it
            stripes itself over ``(rank, worker)``.
        collator: An :class:`~oplm.data.sequence.collate.MLMCollator` built with
            ``pack_sequences=True, pack_rows=rows, max_length=width``.
        rows: Rows per batch (``train.batch_tokens // width``).
        width: Tokens per row (``model.max_position_embeddings``).
        drop_last: Drop the final partial batch of each worker's stream (``data.drop_last``).

    Each batch carries the collator's tensors (all ``(rows, width)``), ``num_sequences`` and
    ``stream_position = (worker_id, rows_emitted, tag)``: the rows of ``dataset``'s stream this
    worker has emitted this epoch, counting this batch, and the batch's last row's
    :data:`~oplm.data.sequence.dataset.POSITION_KEY` (popped from every row; ``None`` for a
    single source, whose rows carry none).
    """

    def __init__(
        self,
        dataset: IterableDataset[dict[str, object]],
        collator: MLMCollator,
        *,
        rows: int,
        width: int,
        drop_last: bool = True,
    ) -> None:
        super().__init__()
        if rows < 1 or width < 3:
            raise ValueError(f"TokenBatcher needs rows >= 1 and width >= 3, got {rows}, {width}")
        self._dataset = dataset
        self._collator = collator
        self._rows = rows
        self._width = width
        self._drop_last = drop_last
        self._resume_rows: list[int] | None = None

    @property
    def total_length(self) -> int:
        """Rows in one epoch of the wrapped stream (for the trainer's fractional epoch)."""
        return int(getattr(self._dataset, "total_length", 0))

    def set_epoch(self, epoch: int) -> None:
        """Forward the epoch to the wrapped dataset's shuffling."""
        self._dataset.set_epoch(epoch)  # ty: ignore[unresolved-attribute]  # duck-typed dataset

    def set_resume_rows(self, rows_per_worker: Sequence[int]) -> None:
        """Arm a one-epoch per-worker row skip (the wrapped dataset skips; counting resumes)."""
        self._resume_rows = list(rows_per_worker)
        self._dataset.set_resume_rows(rows_per_worker)  # ty: ignore[unresolved-attribute]

    def set_resume_sources(self, sources: Sequence[Sequence[Sequence[int]] | None]) -> None:
        """Forward per-worker, per-source positions to the wrapped dataset if it takes them."""
        forward = getattr(self._dataset, "set_resume_sources", None)
        if callable(forward):
            forward(sources)

    def set_reshards(self, reshards: Sequence[Reshard]) -> None:
        """Forward reshard events to the wrapped dataset if it takes them."""
        forward = getattr(self._dataset, "set_reshards", None)
        if callable(forward):
            forward(reshards)

    def clear_resume_skip(self) -> None:
        """Disarm the resume skip here and in the wrapped dataset."""
        self._resume_rows = None
        self._dataset.clear_resume_skip()  # ty: ignore[unresolved-attribute]

    def __iter__(self) -> Iterator[dict[str, object]]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        emitted = self._resume_rows[worker_id] if self._resume_rows is not None else 0
        filler, buffer, tag = RowFiller(self._rows, self._width), [], None
        for row in self._dataset:
            position = row.pop(POSITION_KEY, None)
            length = token_length(str(row["sequence"]), self._width)
            if not filler.place(length):
                emitted += len(buffer)
                yield self._batch(buffer, (worker_id, emitted, tag))
                filler, buffer = RowFiller(self._rows, self._width), []
                filler.place(length)  # empty rows take any sequence: length <= width
            buffer.append(row)
            tag = position
        if buffer and not self._drop_last:
            emitted += len(buffer)
            yield self._batch(buffer, (worker_id, emitted, tag))

    def _batch(self, rows: list[Mapping[str, object]], position: tuple) -> dict[str, object]:
        out: dict[str, object] = dict(self._collator(rows))
        out["stream_position"] = position
        return out

"""Downstream-task loading.

Loads labeled-sequence benchmarks (TAPE / ProteinGLUE) into :class:`DownstreamExample`
records and defines the label-collation contract (docs/DATA_TOOLING.md §7):

- **per-residue** tasks (secondary structure, per-residue contacts) → ``(B, T)``
  ``long`` label tensors aligned to non-special positions, padded with ``-100``;
- **sequence-level regression** (fluorescence, stability) → ``(B,)`` ``float``;
- **sequence-level classification** (fold, enzyme class, GO) → ``(B,)`` ``long``.

The downstream modality is **eval-only** and uses no MLM masking: the model is a
frozen embedder feeding a lightweight supervised head. Sequences themselves are
turned into tensors by the shared pad primitive
(:func:`~oplm.data.sequence.collate.tokenize_and_pad`); this module's job ends at
producing validated examples and collating their *labels*. Embedding extraction,
pooling, and the supervised head belong to the eval harness.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
import torch
from torch import Tensor

from oplm.data.tokenizer import align_per_residue

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyarrow import Table

logger = logging.getLogger(__name__)

__all__ = [
    "DownstreamExample",
    "DownstreamTaskType",
    "collate_downstream_labels",
    "load_downstream_dataset",
]

# Default column names (overridable per call). The sequence column matches the
# sequence modality's convention (docs/DATA_TOOLING.md §4.1).
_SEQUENCE_COLUMN = "sequence"
_LABEL_COLUMN = "label"

# Recognized on-disk suffixes (case-insensitive).
_PARQUET_SUFFIXES = frozenset({".parquet", ".parq", ".pq"})
_CSV_SUFFIX = ".csv"

# When a per-residue label list is stored as a single CSV string (e.g. "0 1 2"),
# split on any run of commas/whitespace.
_PER_RESIDUE_SPLIT_RE = re.compile(r"[,\s]+")

# Cross-entropy ignore index written at non-residue positions (<cls>/<eos>/pad).
_IGNORE_INDEX = -100


class DownstreamTaskType(StrEnum):
    """Label family of a downstream task — determines the collated label tensor.

    Attributes:
        PER_RESIDUE: One integer class per residue (e.g. SS3/SS8). Collates to a
            ``(B, T)`` ``long`` tensor aligned to non-special positions.
        REGRESSION: One real-valued target per sequence. Collates to ``(B,)`` ``float``.
        CLASSIFICATION: One integer class index per sequence. Collates to ``(B,)`` ``long``.
    """

    PER_RESIDUE = "per_residue"
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


@dataclass
class DownstreamExample:
    """One labeled sequence for a downstream benchmark.

    Attributes:
        sequence: Raw one-letter amino-acid sequence (no special tokens).
        label: Task label, shaped by task type — a ``list[int]`` of per-residue
            class indices (length == ``len(sequence)``) for
            :attr:`DownstreamTaskType.PER_RESIDUE`; a ``float`` for
            :attr:`~DownstreamTaskType.REGRESSION`; an ``int`` class index for
            :attr:`~DownstreamTaskType.CLASSIFICATION`.
    """

    sequence: str
    label: list[int] | float | int


def _read_table(path: Path, columns: list[str]) -> Table:
    """Read the requested columns from a parquet or CSV file.

    Args:
        path: Path to a ``.parquet``/``.parq``/``.pq`` or ``.csv`` file.
        columns: Column names that must be present and are returned.

    Returns:
        A pyarrow table restricted to ``columns``.

    Raises:
        ValueError: If the suffix is unsupported or a required column is missing.
    """
    suffix = path.suffix.lower()
    if suffix in _PARQUET_SUFFIXES:
        available = list(pq.ParquetFile(path).schema_arrow.names)
        _require_columns(path, columns, available)
        return pq.read_table(path, columns=columns)
    if suffix == _CSV_SUFFIX:
        table = pa_csv.read_csv(str(path))
        _require_columns(path, columns, table.column_names)
        return table.select(columns)
    raise ValueError(
        f"{path.name}: unsupported downstream file type {suffix!r}; "
        f"expected parquet ({sorted(_PARQUET_SUFFIXES)}) or '.csv'"
    )


def _require_columns(path: Path, required: list[str], available: list[str]) -> None:
    """Raise ``ValueError`` if any ``required`` column is absent from ``available``."""
    missing = [c for c in required if c not in available]
    if missing:
        raise ValueError(
            f"{path.name}: missing required column(s) {missing}; found {sorted(available)}"
        )


def _parse_per_residue_label(raw: Any) -> list[int]:
    """Coerce one per-residue label cell into a ``list[int]``.

    Parquet ``list<int>`` columns surface as Python lists/tuples; CSV stores the
    list as a single string (e.g. ``"0 1 2"``), split on commas/whitespace.

    Args:
        raw: The raw label cell (a sequence of ints, or a delimited string).

    Returns:
        The per-residue class indices.

    Raises:
        TypeError: If ``raw`` is neither a string nor a sequence of ints.
    """
    if isinstance(raw, str):
        tokens = [t for t in _PER_RESIDUE_SPLIT_RE.split(raw.strip()) if t]
        return [int(t) for t in tokens]
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    raise TypeError(
        f"per-residue label must be a delimited string or a sequence of ints, "
        f"got {type(raw).__name__}"
    )


def _coerce_label(
    raw: Any,
    task_type: DownstreamTaskType,
    *,
    sequence: str,
    row: int,
) -> list[int] | float | int:
    """Coerce one raw label cell to the typed label for ``task_type``.

    Args:
        raw: The raw label cell read from disk.
        task_type: The task's label family.
        sequence: The row's sequence (per-residue length validation).
        row: Row index (for error messages).

    Returns:
        A ``float`` (regression), ``int`` (classification), or ``list[int]``
        (per-residue) label.

    Raises:
        ValueError: If a per-residue label's length disagrees with ``sequence``.
    """
    if task_type is DownstreamTaskType.REGRESSION:
        return float(raw)
    if task_type is DownstreamTaskType.CLASSIFICATION:
        return int(raw)
    label = _parse_per_residue_label(raw)
    if len(label) != len(sequence):
        raise ValueError(
            f"row {row}: per-residue label length {len(label)} disagrees with "
            f"sequence length {len(sequence)}"
        )
    return label


def load_downstream_dataset(
    path: str | Path,
    task_type: DownstreamTaskType | str,
    *,
    sequence_column: str = _SEQUENCE_COLUMN,
    label_column: str = _LABEL_COLUMN,
) -> list[DownstreamExample]:
    """Load a labeled-sequence downstream dataset from parquet or CSV.

    The file must hold a sequence column (default ``"sequence"``) and a label
    column (default ``"label"``). Labels are coerced per ``task_type``:

    - :attr:`DownstreamTaskType.PER_RESIDUE`: a ``list<int>`` parquet column or a
      delimited CSV string (e.g. ``"0 1 2"``) → ``list[int]``, validated to have
      one label per residue.
    - :attr:`~DownstreamTaskType.REGRESSION`: → ``float``.
    - :attr:`~DownstreamTaskType.CLASSIFICATION`: → ``int`` class index. Mapping
      categorical class *names* to indices is the eval harness's job; this loader
      expects integer-coercible labels.

    Args:
        path: Path to a single ``.parquet``/``.parq``/``.pq`` or ``.csv`` file.
        task_type: The task's label family (a :class:`DownstreamTaskType` or its
            string value).
        sequence_column: Name of the sequence column.
        label_column: Name of the label column.

    Returns:
        One :class:`DownstreamExample` per row, in file order.

    Raises:
        FileNotFoundError: If ``path`` is not an existing file.
        ValueError: If the file type is unsupported, a required column is missing,
            or a per-residue label length disagrees with its sequence.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"downstream dataset file not found: {path}")
    task_type = DownstreamTaskType(task_type)

    table = _read_table(path, [sequence_column, label_column])
    sequences = [str(s) for s in table.column(sequence_column).to_pylist()]
    raw_labels = table.column(label_column).to_pylist()

    examples = [
        DownstreamExample(sequence=seq, label=_coerce_label(raw, task_type, sequence=seq, row=i))
        for i, (seq, raw) in enumerate(zip(sequences, raw_labels, strict=True))
    ]
    logger.info(
        "loaded %d downstream examples from %s (task_type=%s)", len(examples), path, task_type
    )
    return examples


def collate_downstream_labels(
    examples: Sequence[DownstreamExample],
    task_type: DownstreamTaskType | str,
    *,
    total_len: int | None = None,
) -> Tensor:
    """Collate a batch of example labels into a model-ready target tensor.

    Implements the label contract (docs/DATA_TOOLING.md §7.3):

    - per-residue → ``(B, T)`` ``long``, aligned to non-special positions via
      :func:`~oplm.data.tokenizer.align_per_residue`; ``<cls>``/``<eos>`` and
      padding are filled with ``-100`` (cross-entropy ignore index). Labels are
      clipped to ``total_len - 2`` residues, mirroring the token truncation in
      :func:`~oplm.data.sequence.collate.tokenize_and_pad`, so they stay aligned
      with ``input_ids``.
    - regression → ``(B,)`` ``float32``.
    - classification → ``(B,)`` ``long``.

    Args:
        examples: The batch's examples (their ``.label`` and, for per-residue,
            ``.sequence`` lengths are read).
        task_type: The task's label family.
        total_len: The padded tokenized width ``T`` (from ``tokenize_and_pad`` on
            the *same* batch). Required for per-residue tasks; ignored otherwise.

    Returns:
        The label tensor: ``(B, T)`` ``long`` (per-residue), or ``(B,)``
        ``float32`` / ``long`` (sequence-level regression / classification).

    Raises:
        ValueError: If ``task_type`` is per-residue and ``total_len`` is ``None``,
            or a per-residue label length disagrees with its sequence (raised by
            :func:`~oplm.data.tokenizer.align_per_residue`).
    """
    task_type = DownstreamTaskType(task_type)

    if task_type is DownstreamTaskType.PER_RESIDUE:
        if total_len is None:
            raise ValueError("total_len (the padded tokenized width T) is required for per-residue")
        # Per-residue labels are list[int]; the union widens them, so narrow back.
        values: list[Sequence[float] | None] = [
            cast("Sequence[float]", ex.label) for ex in examples
        ]
        lengths = [len(ex.sequence) for ex in examples]
        aligned = align_per_residue(
            values,
            lengths=lengths,
            total_len=total_len,
            fill_special=_IGNORE_INDEX,
            fill_pad=_IGNORE_INDEX,
        )  # (B, total_len) float32
        return aligned.to(torch.long)  # (B, T) long

    if task_type is DownstreamTaskType.REGRESSION:
        # Labels are pre-coerced floats; dtype casts any stray ints. (B,)
        return torch.tensor([ex.label for ex in examples], dtype=torch.float32)

    # Classification: pre-coerced int class indices. (B,)
    return torch.tensor([ex.label for ex in examples], dtype=torch.long)

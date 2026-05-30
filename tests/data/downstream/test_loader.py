"""Tests for the downstream loader (Phase 8, docs/DATA_TOOLING.md §7).

Covers loading labeled-sequence benchmarks from parquet/CSV (per-residue,
sequence-level regression, sequence-level classification) and the label-collation
contract: per-residue labels align to non-special positions and pad with ``-100``;
sequence-level labels collate to ``(B,)`` float/long.

Fixtures are built at test time from **real** protein sequences (the shared
``real_records`` fixture); the task *labels* are derived deterministically (they
are not biological data, only task targets) so the loader has real sequences to
validate against.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from oplm.data.downstream.loader import (
    DownstreamExample,
    DownstreamTaskType,
    collate_downstream_labels,
    load_downstream_dataset,
)
from oplm.data.sequence.collate import tokenize_and_pad
from oplm.data.tokenizer import get_tokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from tests.data.conftest import Record


# --------------------------------------------------------------------------- #
# Deterministic, derived labels (real sequences, synthetic task targets)
# --------------------------------------------------------------------------- #


def _per_residue_label(sequence: str) -> list[int]:
    """A deterministic 3-class per-residue label, one entry per residue."""
    return [ord(c) % 3 for c in sequence]


def _regression_label(sequence: str) -> float:
    """A deterministic real-valued sequence-level target."""
    return round(len(sequence) / 10.0, 3)


def _classification_label(sequence: str) -> int:
    """A deterministic integer class index per sequence."""
    return len(sequence) % 4


# --------------------------------------------------------------------------- #
# Fixture writers
# --------------------------------------------------------------------------- #


def _write_per_residue_parquet(path: Path, records: Sequence[Record]) -> Path:
    """Write a per-residue task parquet (``sequence`` + ``list<int>`` ``label``)."""
    seqs = [seq for _, seq in records]
    table = pa.table(
        {
            "sequence": pa.array(seqs, type=pa.large_string()),
            "label": pa.array([_per_residue_label(s) for s in seqs], type=pa.list_(pa.int64())),
        }
    )
    pq.write_table(table, path)
    return path


def _write_seq_csv(
    path: Path,
    records: Sequence[Record],
    *,
    label_fn,
) -> Path:
    """Write a sequence-level task CSV (``sequence`` + scalar ``label``)."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "label"])
        for _, seq in records:
            writer.writerow([seq, label_fn(seq)])
    return path


def _write_per_residue_csv(path: Path, records: Sequence[Record]) -> Path:
    """Write a per-residue task CSV with the label list as a space-joined string."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "label"])
        for _, seq in records:
            writer.writerow([seq, " ".join(str(x) for x in _per_residue_label(seq))])
    return path


# --------------------------------------------------------------------------- #
# Loading: parquet / CSV, per task family
# --------------------------------------------------------------------------- #


def test_load_per_residue_parquet(tmp_path: Path, real_records: list[Record]) -> None:
    """A per-residue parquet loads to per-residue ``list[int]`` labels, length-matched."""
    records = real_records[:6]
    path = _write_per_residue_parquet(tmp_path / "ss3.parquet", records)

    examples = load_downstream_dataset(path, DownstreamTaskType.PER_RESIDUE)

    assert len(examples) == len(records)
    for ex, (_, seq) in zip(examples, records, strict=True):
        assert isinstance(ex, DownstreamExample)
        assert ex.sequence == seq
        assert isinstance(ex.label, list)
        assert len(ex.label) == len(seq)
        assert all(isinstance(x, int) for x in ex.label)
        assert ex.label == _per_residue_label(seq)


def test_load_regression_csv(tmp_path: Path, real_records: list[Record]) -> None:
    """A sequence-level regression CSV loads to ``float`` labels."""
    records = real_records[:5]
    path = _write_seq_csv(tmp_path / "fluorescence.csv", records, label_fn=_regression_label)

    examples = load_downstream_dataset(path, "regression")

    assert [ex.label for ex in examples] == pytest.approx(
        [_regression_label(s) for _, s in records]
    )
    assert all(isinstance(ex.label, float) for ex in examples)


def test_load_classification_csv(tmp_path: Path, real_records: list[Record]) -> None:
    """A sequence-level classification CSV loads to ``int`` class indices."""
    records = real_records[:5]
    path = _write_seq_csv(tmp_path / "fold.csv", records, label_fn=_classification_label)

    examples = load_downstream_dataset(path, DownstreamTaskType.CLASSIFICATION)

    assert [ex.label for ex in examples] == [_classification_label(s) for _, s in records]
    assert all(isinstance(ex.label, int) for ex in examples)


def test_per_residue_label_from_csv_string(tmp_path: Path, real_records: list[Record]) -> None:
    """A per-residue label stored as a delimited CSV string parses to ``list[int]``."""
    records = real_records[:4]
    path = _write_per_residue_csv(tmp_path / "ss3.csv", records)

    examples = load_downstream_dataset(path, "per_residue")

    for ex, (_, seq) in zip(examples, records, strict=True):
        assert ex.label == _per_residue_label(seq)


def test_task_type_accepts_string_and_enum(tmp_path: Path, real_records: list[Record]) -> None:
    """``task_type`` accepts both the enum and its string value identically."""
    path = _write_seq_csv(tmp_path / "stab.csv", real_records[:3], label_fn=_regression_label)
    via_enum = load_downstream_dataset(path, DownstreamTaskType.REGRESSION)
    via_str = load_downstream_dataset(path, "regression")
    assert [e.label for e in via_enum] == [e.label for e in via_str]


def test_custom_column_names(tmp_path: Path, real_records: list[Record]) -> None:
    """Non-default sequence/label column names are honored."""
    records = real_records[:3]
    path = tmp_path / "custom.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seq", "y"])
        for _, seq in records:
            writer.writerow([seq, _classification_label(seq)])

    examples = load_downstream_dataset(
        path, "classification", sequence_column="seq", label_column="y"
    )
    assert [ex.sequence for ex in examples] == [s for _, s in records]
    assert [ex.label for ex in examples] == [_classification_label(s) for _, s in records]


# --------------------------------------------------------------------------- #
# Label collation contract (§7.3)
# --------------------------------------------------------------------------- #


def test_collate_per_residue_alignment(tmp_path: Path, real_records: list[Record]) -> None:
    """Per-residue labels collate to ``(B, T)`` long, ``-100`` at specials/pad."""
    records = real_records[:5]  # varied lengths -> real padding within the batch
    path = _write_per_residue_parquet(tmp_path / "ss3.parquet", records)
    examples = load_downstream_dataset(path, "per_residue")

    max_length = 1024
    encoded = tokenize_and_pad(
        [{"sequence": ex.sequence} for ex in examples], get_tokenizer(), max_length
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    total_len = int(input_ids.shape[1])

    labels = collate_downstream_labels(examples, "per_residue", total_len=total_len)

    assert labels.shape == input_ids.shape  # (B, T)
    assert labels.dtype == torch.long
    # <cls> at position 0 is always ignored.
    assert torch.all(labels[:, 0] == -100)

    for b, ex in enumerate(examples):
        length = min(len(ex.sequence), max_length - 2)
        eos_pos = 1 + length
        # Residue positions carry the per-residue labels (in order).
        assert labels[b, 1:eos_pos].tolist() == ex.label[:length]
        # <eos> and every trailing pad position are the ignore index.
        assert labels[b, eos_pos] == -100
        assert torch.all(labels[b, eos_pos + 1 :] == -100)
        # The ignore positions are exactly the non-attended ones plus <cls>/<eos>.
        ignored = labels[b] == -100
        attended = attention_mask[b] == 1
        # Every padded (non-attended) position is ignored.
        assert torch.all(ignored[~attended])


def test_collate_per_residue_truncation(tmp_path: Path, real_records: list[Record]) -> None:
    """Labels are clipped to ``total_len - 2`` residues, matching token truncation."""
    # Pick the longest real sequence and a small max_length so truncation bites.
    longest = max((r for r in real_records), key=lambda r: len(r[1]))
    max_length = 16
    path = _write_per_residue_parquet(tmp_path / "long.parquet", [longest])
    examples = load_downstream_dataset(path, "per_residue")

    encoded = tokenize_and_pad(
        [{"sequence": ex.sequence} for ex in examples], get_tokenizer(), max_length
    )
    total_len = int(encoded["input_ids"].shape[1])
    assert total_len == max_length  # sequence longer than max_length - 2

    labels = collate_downstream_labels(examples, "per_residue", total_len=total_len)
    keep = max_length - 2
    assert labels[0, 1 : 1 + keep].tolist() == examples[0].label[:keep]
    assert labels[0, 0] == -100  # <cls>
    assert labels[0, 1 + keep] == -100  # <eos>


def test_collate_regression_shape_dtype(tmp_path: Path, real_records: list[Record]) -> None:
    """Regression labels collate to a ``(B,)`` float tensor."""
    records = real_records[:5]
    path = _write_seq_csv(tmp_path / "reg.csv", records, label_fn=_regression_label)
    examples = load_downstream_dataset(path, "regression")

    labels = collate_downstream_labels(examples, "regression")
    assert labels.shape == (len(records),)
    assert labels.dtype == torch.float32
    assert labels.tolist() == pytest.approx([_regression_label(s) for _, s in records])


def test_collate_classification_shape_dtype(tmp_path: Path, real_records: list[Record]) -> None:
    """Classification labels collate to a ``(B,)`` long tensor."""
    records = real_records[:5]
    path = _write_seq_csv(tmp_path / "cls.csv", records, label_fn=_classification_label)
    examples = load_downstream_dataset(path, "classification")

    labels = collate_downstream_labels(examples, DownstreamTaskType.CLASSIFICATION)
    assert labels.shape == (len(records),)
    assert labels.dtype == torch.long
    assert labels.tolist() == [_classification_label(s) for _, s in records]


def test_collate_per_residue_requires_total_len(real_records: list[Record]) -> None:
    """Per-residue collation without ``total_len`` is an error."""
    ex = DownstreamExample(
        sequence=real_records[0][1], label=_per_residue_label(real_records[0][1])
    )
    with pytest.raises(ValueError, match="total_len"):
        collate_downstream_labels([ex], "per_residue")


def test_collate_per_residue_length_mismatch_raises() -> None:
    """A per-residue label whose length disagrees with the sequence raises."""
    bad = DownstreamExample(sequence="ACDEFG", label=[0, 1, 2])  # 3 labels, 6 residues
    with pytest.raises(ValueError, match="disagrees"):
        collate_downstream_labels([bad], "per_residue", total_len=10)


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("drop", ["sequence", "label"])
def test_missing_required_column_raises(
    tmp_path: Path, real_records: list[Record], drop: str
) -> None:
    """Dropping either required column is an error."""
    path = tmp_path / "missing.csv"
    kept = [c for c in ("sequence", "label") if c != drop]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(kept)
        writer.writerow([real_records[0][1] if c == "sequence" else 1 for c in kept])
    with pytest.raises(ValueError, match="missing required column"):
        load_downstream_dataset(path, "classification")


def test_per_residue_load_length_mismatch_raises(
    tmp_path: Path, real_records: list[Record]
) -> None:
    """A per-residue parquet with a wrong-length label list raises at load."""
    _, seq = real_records[0]
    path = tmp_path / "bad.parquet"
    table = pa.table(
        {
            "sequence": pa.array([seq], type=pa.large_string()),
            "label": pa.array([[0, 1, 2]], type=pa.list_(pa.int64())),  # too short
        }
    )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="disagrees"):
        load_downstream_dataset(path, "per_residue")


def test_unsupported_file_type_raises(tmp_path: Path) -> None:
    """A non-parquet/CSV file type raises a clear error."""
    path = tmp_path / "data.txt"
    path.write_text("sequence\tlabel\n")
    with pytest.raises(ValueError, match="unsupported downstream file type"):
        load_downstream_dataset(path, "regression")


def test_missing_file_raises(tmp_path: Path) -> None:
    """A nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_downstream_dataset(tmp_path / "nope.parquet", "regression")

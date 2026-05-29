"""Shared test fixtures for the oplm test suite."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
_FAST_TRAINING_ROWS = 256


@pytest.fixture(autouse=True)
def _reset_accelerator_state() -> None:
    """Reset accelerate singleton state between tests.

    AcceleratorState is a process-global singleton. Without resetting it,
    a test that creates an Accelerator with mixed_precision="no" prevents
    a later test from using mixed_precision="bf16" in the same process.
    """
    from accelerate.state import AcceleratorState

    AcceleratorState._reset_state(reset_partial_state=True)


@pytest.fixture(scope="session")
def full_training_parquet() -> Path:
    """Path to the full real training sequences parquet file."""
    path = FIXTURES_DIR / "training" / "test_sequences.parquet"
    if not path.exists():
        pytest.skip(f"Training fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def training_parquet(
    full_training_parquet: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Small real-data parquet fixture derived from the full training dataset."""
    path = tmp_path_factory.mktemp("fixtures") / "test_sequences_fast.parquet"
    parquet_file = pq.ParquetFile(full_training_parquet)
    first_batch = next(parquet_file.iter_batches(batch_size=_FAST_TRAINING_ROWS))
    pq.write_table(pa.Table.from_batches([first_batch]), path)
    return path


@pytest.fixture(scope="session")
def structure_fixtures_dir() -> Path:
    """Path to the directory containing PDB test fixtures."""
    path = FIXTURES_DIR / "eval" / "structures"
    if not path.exists():
        pytest.skip(f"Structure fixtures not found: {path}")
    return path

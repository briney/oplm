"""Shared test fixtures for the oplm test suite."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from collections.abc import Iterator

FIXTURES_DIR = Path(__file__).parent / "fixtures"
_FAST_TRAINING_ROWS = 256


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip ``@pytest.mark.blackwell`` tests on hardware without FP8 support.

    FP8 matmuls require sm90+ (Blackwell / H100+). Tests carrying the marker run
    only where the device reports compute capability >= 9; everywhere else
    (CPU-only CI, Ampere boxes) they are skipped rather than failing deep inside
    a torchao kernel.
    """
    if "blackwell" in item.keywords and (
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9
    ):
        pytest.skip("Test requires sm90+ (Blackwell / H100+) GPU")


@pytest.fixture(autouse=True)
def _cleanup_distributed() -> Iterator[None]:
    """Tear down any process group a test leaves behind.

    The native-FSDP2 Trainer initializes a ``torch.distributed`` process group in
    ``__init__`` and destroys it in ``train()``'s teardown, but a test that
    constructs a Trainer without running ``train()`` (or that errors mid-run) would
    leak the group into the next test — whose ``_init_distributed`` then sees
    ``dist.is_initialized()`` and skips re-init, inheriting the wrong backend. This
    fixture guarantees a clean slate after every test.
    """
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


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
def tiny_training_parquet(
    full_training_parquet: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """~16-row real-data parquet so ``max_epochs=2`` crosses an epoch boundary cheaply.

    With ``batch_size=4`` an epoch is exactly four optimizer steps, so a two-epoch
    run hits the ``StopIteration -> epoch++ -> set_epoch`` path after eight steps
    (see the G8 epoch-bounded e2e test).
    """
    path = tmp_path_factory.mktemp("fixtures") / "test_sequences_tiny.parquet"
    parquet_file = pq.ParquetFile(full_training_parquet)
    first_batch = next(parquet_file.iter_batches(batch_size=16))
    pq.write_table(pa.Table.from_batches([first_batch]), path)
    return path


@pytest.fixture(scope="session")
def second_eval_parquet(
    full_training_parquet: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """A disjoint 64-row slice of the source parquet for the G2 multi-dataset test.

    Drawn from rows ``[256, 320)`` so it does not overlap ``training_parquet``
    (the first 256 rows), letting the two eval namespaces be compared independently.
    """
    path = tmp_path_factory.mktemp("fixtures") / "second_eval.parquet"
    parquet_file = pq.ParquetFile(full_training_parquet)
    batches = parquet_file.iter_batches(batch_size=256)
    next(batches)  # skip the first 256 rows (the training_parquet slice)
    disjoint = next(batches).slice(0, 64)
    pq.write_table(pa.Table.from_batches([disjoint]), path)
    return path


@pytest.fixture(scope="session")
def structure_fixtures_dir() -> Path:
    """Path to the directory containing PDB test fixtures."""
    path = FIXTURES_DIR / "eval" / "structures"
    if not path.exists():
        pytest.skip(f"Structure fixtures not found: {path}")
    return path

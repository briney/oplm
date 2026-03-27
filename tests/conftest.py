"""Shared test fixtures for the oplm test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
def training_parquet() -> Path:
    """Path to the real training sequences parquet file."""
    path = FIXTURES_DIR / "training" / "test_sequences.parquet"
    if not path.exists():
        pytest.skip(f"Training fixture not found: {path}")
    return path

"""Prepare or run the rank kill recovery drill."""

from __future__ import annotations

from oplm.slurm.recovery.runner import main

if __name__ == "__main__":
    main("rank_kill")

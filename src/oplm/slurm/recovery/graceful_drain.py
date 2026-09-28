"""Prepare or run the graceful drain recovery drill."""

from __future__ import annotations

from oplm.slurm.recovery.runner import main

if __name__ == "__main__":
    main("graceful_drain")

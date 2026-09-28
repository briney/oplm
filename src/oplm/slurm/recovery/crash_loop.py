"""Prepare or run the crash loop recovery drill."""

from __future__ import annotations

from oplm.slurm.recovery.runner import main

if __name__ == "__main__":
    main("crash_loop")

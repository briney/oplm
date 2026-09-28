"""Prepare or run the batch node loss recovery drill."""

from __future__ import annotations

from oplm.slurm.recovery.runner import main

if __name__ == "__main__":
    main("batch_node_loss")

"""Turn an oplm training config into Slurm job scripts. Knows nothing about μP."""

from __future__ import annotations

from oplm.slurm.config import SlurmConfig, load_slurm_config

__all__ = ["SlurmConfig", "load_slurm_config"]

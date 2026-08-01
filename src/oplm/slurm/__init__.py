"""Turn an oplm training config into Slurm job scripts. Knows nothing about μP."""

from __future__ import annotations

from oplm.slurm.config import BatchPlan, SlurmConfig, load_slurm_config, resolve_batch_plan

__all__ = ["BatchPlan", "SlurmConfig", "load_slurm_config", "resolve_batch_plan"]

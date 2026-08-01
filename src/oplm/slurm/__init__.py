"""Turn an oplm training config into Slurm job scripts. Knows nothing about μP."""

from __future__ import annotations

from oplm.slurm.config import BatchPlan, SlurmConfig, load_slurm_config, resolve_batch_plan
from oplm.slurm.render import (
    JobSpec,
    SubmitEntry,
    accelerate_command,
    render_job,
    render_submit_script,
)
from oplm.slurm.submit import running_job_ids, submit_all, submit_job

__all__ = [
    "BatchPlan",
    "JobSpec",
    "SlurmConfig",
    "SubmitEntry",
    "accelerate_command",
    "load_slurm_config",
    "render_job",
    "render_submit_script",
    "resolve_batch_plan",
    "running_job_ids",
    "submit_all",
    "submit_job",
]

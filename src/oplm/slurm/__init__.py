"""Turn an oplm training config into Slurm job scripts. Knows nothing about μP."""

from __future__ import annotations

from oplm.slurm.config import (
    BatchPlan,
    SlurmConfig,
    TokenPlan,
    load_slurm_config,
    resolve_batch_plan,
    resolve_token_plan,
)
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
    "TokenPlan",
    "accelerate_command",
    "load_slurm_config",
    "render_job",
    "render_submit_script",
    "resolve_batch_plan",
    "resolve_token_plan",
    "running_job_ids",
    "submit_all",
    "submit_job",
]

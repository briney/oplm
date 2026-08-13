"""E2E drain test (Task 1.5): SIGUSR1 mid-run -> checkpoint -> exit 85 -> auto_resume.

Signals and threads don't mix (a signal delivered to a Python thread other than the
main one is deferred/lost in ways that make in-process testing unreliable), so this
drives a real ``python -m oplm.train`` **subprocess** and sends it a real SIGUSR1,
mirroring the ``python -m oplm.train`` subprocess pattern in
``tests/test_train_entrypoint.py`` but via a written YAML config (round-tripped
through ``serialize_config``/``load_config`` per
``tests/training/test_checkpoint.py::test_config_yaml_is_reloadable``) instead of CLI
dotlist overrides, since the config is assembled once and reused for the follow-up
resume run.

The signal is sent as soon as the child has written its top-level ``config.yaml`` --
early in ``Trainer.__init__``, well before model construction, dataloader prep, or the
training loop start. ``DrainSignal.install()`` runs even earlier (the first lines of
``__init__``), so by the time ``config.yaml`` exists the handlers are already armed and
the drain flag is guaranteed to be set well before *some* optimizer step's
control-bundle reduce observes it -- no race with whether the tiny model trains at all.
Exactly which step catches the flag depends on how much of model/dataloader
construction ran before the poll loop noticed ``config.yaml`` and fired the signal, so
the test discovers the drained step rather than assuming it is step 1.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import pytest

from oplm.config import serialize_config
from oplm.training.signals import DRAIN_EXIT_CODE
from tests.training.conftest import tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow

_CONFIG_WAIT_TIMEOUT_S = 60.0
_EXIT_WAIT_TIMEOUT_S = 120.0


def test_sigusr1_drains_to_checkpoint_and_exits_85_then_auto_resume_continues(
    training_parquet: Path, tmp_path: Path
) -> None:
    """A SIGUSR1 mid-run commits exactly one checkpoint, exits 85, and resumes past it."""
    run_dir = tmp_path / "run"

    # save_every=0 disables step-cadence checkpointing; save_final only fires if the loop
    # reaches max_steps normally, which a drain this early never does -- the signal is
    # sent as soon as config.yaml appears, long before even one training step completes,
    # so any max_steps comfortably above the (dynamically discovered) drained_step below
    # works. Kept small (rather than the very large value used pre-Task-2.2) because the
    # follow-up resume below must not *decrease* max_steps relative to this run's own
    # config -- validate_schedule_compat's asymmetric max_steps policy (Task 2.2 fix
    # round) raises on any decrease -- and a small original value keeps the follow-up
    # resume.train() call (which runs from drained_step up to original_max_steps + 2) fast.
    original_max_steps = 6
    cfg = tiny_train_cfg(
        run_dir,
        training_parquet,
        max_steps=original_max_steps,
        save_every=0,
        log_every=1,
    )
    config_path = tmp_path / "launch_config.yaml"
    config_path.write_text(serialize_config(cfg))

    cmd = [sys.executable, "-m", "oplm.train", "--config", str(config_path)]
    env = {**os.environ, "ACCELERATE_USE_CPU": "true"}

    stdout_path = tmp_path / "child.stdout.log"
    stderr_path = tmp_path / "child.stderr.log"
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        child = subprocess.Popen(cmd, stdout=out, stderr=err, env=env)

    try:
        # Poll for the top-level config.yaml the trainer writes early in __init__ --
        # our signal that DrainSignal.install() (which runs even earlier) has
        # definitely already armed the handlers.
        deadline = time.monotonic() + _CONFIG_WAIT_TIMEOUT_S
        written_config = run_dir / "config.yaml"
        while not written_config.exists():
            if child.poll() is not None:
                pytest.fail(
                    f"child exited early (rc={child.returncode}) before writing "
                    f"config.yaml.\nstdout:\n{stdout_path.read_text()}\n"
                    f"stderr:\n{stderr_path.read_text()}"
                )
            if time.monotonic() > deadline:
                pytest.fail("timed out waiting for the child to write config.yaml")
            time.sleep(0.05)

        child.send_signal(signal.SIGUSR1)
        returncode = child.wait(timeout=_EXIT_WAIT_TIMEOUT_S)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()

    assert returncode == DRAIN_EXIT_CODE == 85, (
        f"stdout:\n{stdout_path.read_text()}\nstderr:\n{stderr_path.read_text()}"
    )

    committed = [p for p in run_dir.glob("checkpoint-*") if p.is_dir()]
    assert len(committed) == 1, (
        f"expected exactly one committed checkpoint, got {sorted(p.name for p in committed)}\n"
        f"stderr:\n{stderr_path.read_text()}"
    )
    drained_step = int(committed[0].name.removeprefix("checkpoint-"))

    # A drain warning was logged (docs/DRAIN.md-style observability contract).
    assert "Drain requested" in stderr_path.read_text()

    # The key regression this test guards against (fix commit 69e70f3): the drain
    # branch must sit AFTER this step's tokens_seen/throughput bookkeeping, so the
    # checkpoint it writes carries a tokens_seen that actually covers the drained
    # step, not one step behind. Cross-check the checkpoint's tokens_seen against an
    # independent control run: the data pipeline's shuffling is a pure, seeded
    # function of (cfg.train.seed, epoch) (see oplm.data.sequence.dataset._epoch_seed
    # -- an explicit torch.Generator, not global RNG state), and tokens_seen depends
    # only on batch composition (attention_mask sums), not on masking or model
    # randomness. So a fresh, uninterrupted Trainer sharing the same seed/data/batch
    # config reproduces the exact same first `drained_step` batches -- in-process,
    # in a different subprocess, doesn't matter -- and its tokens_seen after exactly
    # `drained_step` steps is the ground truth for what the drained checkpoint should
    # have recorded. If the drain branch ever moves back above the tokens_seen/
    # throughput accounting, the checkpoint's tokens_seen will fall short of this
    # value by exactly one step's tokens, and this assertion catches it.
    checkpoint_state = json.loads(
        (run_dir / f"checkpoint-{drained_step}" / "trainer_state.json").read_text()
    )
    checkpoint_tokens_seen = checkpoint_state["tokens_seen"]

    from oplm.training.trainer import Trainer

    control_cfg = tiny_train_cfg(
        tmp_path / "control",
        training_parquet,
        max_steps=drained_step,
        save_every=0,
        save_final=False,
        log_every=1,
    )
    control = Trainer(control_cfg, callbacks=[])
    control.train()
    assert control.global_step == drained_step
    assert checkpoint_tokens_seen == control.tokens_seen > 0

    # Follow-up: a requeued run with auto_resume=true picks up past the drained step.
    # max_steps must not *decrease* relative to the drained checkpoint's own config
    # (original_max_steps, saved into its config.yaml) -- validate_schedule_compat's
    # asymmetric max_steps policy raises on any decrease (Task 2.2 fix round) -- so this
    # takes whichever of "past the drained step" or "past the original target" is
    # larger, guaranteeing a genuine increase (allowed, logs a warning) rather than a
    # decrease (rejected).
    resume_max_steps = max(drained_step + 2, original_max_steps + 2)
    resume_cfg = tiny_train_cfg(
        run_dir,
        training_parquet,
        max_steps=resume_max_steps,
        save_every=0,
        auto_resume=True,
        log_every=1,
    )
    assert resume_cfg.train.resume_from is None
    resumed = Trainer(resume_cfg, callbacks=[])
    assert resumed.global_step == drained_step
    assert resumed.tokens_seen == checkpoint_tokens_seen

    resumed.train()
    assert resumed.global_step == resume_max_steps

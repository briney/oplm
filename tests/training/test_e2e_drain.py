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
import logging
import os
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import pytest

from oplm.config import serialize_config
from oplm.training.signals import DRAIN_EXIT_CODE, DRAIN_MARKER_NAME
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
    # works.
    #
    # Sized at 200 to fix a 10-17% flake (Important review finding): at max_steps=6 the
    # whole training loop took ~70ms, so the signal-delivery race the docstring above
    # calls "no race" was in fact one -- the child could reach max_steps and exit 0
    # before the SIGUSR1 landed. 200 steps of this tiny CPU model is a ~1s window, three
    # orders of magnitude wider than signal delivery, while still finishing fast.
    # It must also satisfy the *other* constraint the previous small value was chosen
    # for: the follow-up resume must not *decrease* max_steps relative to this run's own
    # config (validate_schedule_compat's asymmetric max_steps policy, Task 2.2 fix
    # round), which resume_max_steps below handles by taking a value >= this one.
    original_max_steps = 200
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

    # The drain marker was written alongside the exit-85 (the signal the sbatch requeue
    # wrapper actually keys on: `accelerate launch` flattens a rank's exit 85 to its own
    # exit 1 on a real multi-GPU job, so the exit code alone never reaches the wrapper
    # there). Its content is the drained step, recorded for debugging.
    drain_marker = run_dir / DRAIN_MARKER_NAME
    assert drain_marker.is_file()
    assert drain_marker.read_text().strip() == str(drained_step)

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
    # decrease (rejected). With original_max_steps=200 this resume trains ~200 tiny CPU
    # steps in-process -- a couple of seconds, and the price of a drain window wide
    # enough not to race signal delivery.
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

    # Trainer start cleared the (here never-consumed -- no requeue wrapper ran) drain
    # marker, so it cannot misclassify a later genuine crash as a drain.
    assert not drain_marker.exists()

    resumed.train()
    assert resumed.global_step == resume_max_steps


def test_sigusr1_drain_with_remote_uri_mirrors_before_exiting_85(
    training_parquet: Path, tmp_path: Path
) -> None:
    """The drain path (Task 4.2) drains its remote upload before exiting 85.

    Same real-subprocess SIGUSR1 drive as the test above, but with
    ``train.remote_checkpoint_uri`` set to a ``file://`` store: ``Trainer.train()``'s
    drain branch calls ``_save_checkpoint()`` (the local commit) and then
    ``_drain_remote_uploads()`` -- bounded but, for this tiny/fast local-file upload,
    plenty of time to actually finish -- *before* raising ``SystemExit(85)``. This
    asserts the remote mirror is already committed with the drained step by the time
    the subprocess has exited, not merely that the local checkpoint exists.
    """
    from oplm.training.remote import RemoteStore

    run_dir = tmp_path / "run"
    remote_root = tmp_path / "remote"
    remote_uri = f"file://{remote_root}"

    # max_steps=200 for the same anti-flake reason as the test above: a 6-step run
    # finishes in ~70ms and can beat the SIGUSR1 to the finish line.
    cfg = tiny_train_cfg(
        run_dir,
        training_parquet,
        max_steps=200,
        save_every=0,
        log_every=1,
        remote_checkpoint_uri=remote_uri,
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

    result = RemoteStore(remote_uri).latest_committed()
    assert result is not None, (
        f"remote store has no committed checkpoint after the drain\n"
        f"stderr:\n{stderr_path.read_text()}"
    )
    name, manifest = result
    assert name == f"checkpoint-{drained_step}"
    assert "trainer_state.json" in manifest["files"]


# --- worker-cycle drain deferral (Task 3.3 controller addition, review fix round) ------


def test_drain_defers_checkpoint_and_skips_eval_until_worker_cycle_alignment(
    training_parquet: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A misaligned drain defers its save/exit -- and skips eval entirely while deferred.

    Drives the real ``train()`` loop in-process (no subprocess/real signal): a fake
    drain source reports ``requested=True`` from the very first optimizer step. With
    ``num_workers=4`` the drain is first observed at ``batches_in_epoch=1`` (misaligned)
    and must defer for three steps (batches_in_epoch=1,2,3, none a multiple of 4) before
    landing on ``batches_in_epoch=4`` -- exactly the scenario
    ``test_periodic_save_deferred_until_worker_cycle_alignment`` covers for periodic
    saves, but for drain, and with a wide-enough deferral window (3 steps) to also check
    the log-dedup fix below.

    ``_run_eval`` is patched to record every call (Critical review fix): with
    ``eval_every={"steps": 1}`` configured, an un-fixed trainer would run eval on every
    deferred step, burning wall-clock time the drain margin exists to avoid -- this
    asserts it is never called at all before the drain resolves. The "deferring
    checkpoint" info log is also asserted to appear exactly once (Minor review fix),
    not once per deferred step (``DrainSignal.requested`` is sticky, so an un-deduped
    log would otherwise repeat on steps 1, 2, and 3).
    """
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=6,
        batch_size=4,
        save_every=0,
        save_final=False,
        num_workers=4,
        log_every=1,
        eval={"hd": {"path": str(training_parquet), "type": "sequence", "every": {"steps": 1}}},
    )
    trainer = Trainer(cfg, callbacks=[])

    class _AlwaysRequestedDrain:
        """Stand-in for DrainSignal: reports requested=True from step 1 onward."""

        requested = True

    monkeypatch.setattr(trainer, "_drain_signal", _AlwaysRequestedDrain())

    eval_calls: list[int] = []

    def _recording_run_eval(tokens_delta: int) -> dict[str, float]:
        eval_calls.append(trainer.global_step)
        return {}

    monkeypatch.setattr(trainer, "_run_eval", _recording_run_eval)

    with (
        caplog.at_level(logging.INFO, logger="oplm.training.trainer"),
        pytest.raises(SystemExit) as exc_info,
    ):
        trainer.train()

    # (c) exit code 85 still propagates.
    assert exc_info.value.code == DRAIN_EXIT_CODE == 85

    # (a) the drain checkpoint lands on the aligned batch count (4), not the misaligned
    # trigger step (1) -- and nothing else was committed.
    committed = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-"))
    assert committed == ["checkpoint-4"]

    # (b) no eval ran at all: the drain was pending from step 1 through step 3 and
    # resolved at step 4, so every step this run reached had a pending drain save.
    assert eval_calls == []

    # Minor fix: the "deferring checkpoint" log appears exactly once across the
    # 3-step deferral window (steps 1, 2, 3), not once per deferred step.
    defer_logs = [r for r in caplog.records if "deferring checkpoint" in r.getMessage()]
    assert len(defer_logs) == 1

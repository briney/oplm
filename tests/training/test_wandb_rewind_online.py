"""Online drill: does the W&B server honour ``wandb.init(resume_from="<id>?_step=N")``?

The Trainer logs with ``step=global_step`` and, on a resume behind W&B's latest step,
rewinds the run's history to the checkpoint with ``resume_from``. That W&B feature is a
server-side private preview enabled per entity, so offline mode cannot test it. The rewind
probes pin the exact behaviour the implementation relies on, against the real server; they
skip, naming the reason, while the entity does not have the feature:

- the rewind keeps the run id, keeps history before step ``N``, removes history at and after
  ``N``, and accepts new rows from step ``N`` on (``test_rewind_truncates_and_replays``);
- rewinding to one past the last logged step is a no-op (the graceful-drain case);
- rewinding past the last logged step either works or fails loudly (W&B behind the
  checkpoint, e.g. a crash before the last rows were flushed);
- two consecutive rewinds work (a crash during the replay).

``test_trainer_resume_behind_the_server_history`` runs the real Trainer through a crash and
resume and passes either way: rewound when the entity has the feature, the documented
fallback when it does not.

Opt-in only: set ``OPLM_WANDB_ONLINE_DRILL=1`` and log in to W&B (``wandb login`` or
``WANDB_API_KEY``). Runs go to the project ``OPLM_WANDB_DRILL_PROJECT`` (default
``oplm-wandb-drill``) in your default entity and are deleted afterwards unless
``OPLM_WANDB_DRILL_KEEP=1``. Run with ``-s`` to see the observed values:

    OPLM_WANDB_ONLINE_DRILL=1 .venv/bin/python -m pytest -v -s \
        tests/training/test_wandb_rewind_online.py
"""

from __future__ import annotations

import os
import time
import uuid
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("OPLM_WANDB_ONLINE_DRILL") != "1",
        reason="online W&B drill; set OPLM_WANDB_ONLINE_DRILL=1 and log in to run it",
    ),
]

PROJECT: str = os.environ.get("OPLM_WANDB_DRILL_PROJECT", "oplm-wandb-drill")
_POLL_SECONDS = 5.0
_TIMEOUT_SECONDS = 180.0


@pytest.fixture
def wandb_mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    return wandb


@pytest.fixture
def created(wandb_mod: Any) -> Any:
    """Paths of runs a test created; deleted at teardown unless OPLM_WANDB_DRILL_KEEP=1."""
    paths: list[str] = []
    yield paths
    if os.environ.get("OPLM_WANDB_DRILL_KEEP") == "1":
        print(f"kept drill runs: {paths}")
        return
    api = wandb_mod.Api()
    for path in paths:
        try:
            api.run(path).delete()
        except Exception as exc:  # noqa: BLE001  # cleanup must not mask the test result
            print(f"could not delete {path}: {exc!r}")


def _new_run(wandb: Any, steps: range, created: list[str]) -> str:
    """Log ``x = step`` at every step of ``steps`` into a fresh run; return its path."""
    run = wandb.init(project=PROJECT, name=f"rewind-drill-{uuid.uuid4().hex[:8]}", config={"a": 1})
    path = f"{run.entity}/{run.project}/{run.id}"
    created.append(path)
    for step in steps:
        run.log({"x": float(step)}, step=step)
    run.finish()
    _history(wandb, path, until=lambda rows: len(rows) >= len(steps))
    return path


def _rewind(wandb: Any, path: str, step: int, values: dict[int, float]) -> dict[str, Any]:
    """``resume_from`` the run at ``step``, log ``values`` (step -> x), return observations."""
    run_id = path.rsplit("/", 1)[1]
    try:
        run = wandb.init(project=PROJECT, resume_from=f"{run_id}?_step={step}", config={"a": 1})
    except Exception as exc:
        if "private preview" in str(exc):
            pytest.skip("W&B rewind is not enabled for this entity (private preview)")
        raise
    seen = {"id": run.id, "step_after_init": run.step, "config_a": run.config.get("a")}
    for s, x in values.items():
        run.log({"x": x}, step=s)
    run.finish()
    return seen


def _history(wandb: Any, path: str, *, until: Any) -> list[tuple[int, float]]:
    """Unsampled ``(step, x)`` rows of ``path``, polling until ``until(rows)`` or timeout."""
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    while True:
        run = wandb.Api().run(path)
        rows = [(int(r["_step"]), float(r["x"])) for r in run.scan_history(keys=["x"])]
        if until(rows) or time.monotonic() > deadline:
            return sorted(rows)
        time.sleep(_POLL_SECONDS)


def _assert_unique(rows: list[tuple[int, float]]) -> dict[int, float]:
    steps = [s for s, _ in rows]
    assert len(steps) == len(set(steps)), f"duplicate steps in history: {rows}"
    return dict(rows)


def test_rewind_truncates_and_replays(wandb_mod: Any, created: list[str]) -> None:
    """The contract the Trainer relies on: keep < N, drop >= N, accept new rows from N."""
    path = _new_run(wandb_mod, range(30), created)
    seen = _rewind(wandb_mod, path, 20, {s: 1000.0 + s for s in range(20, 25)})
    print(f"after resume_from ?_step=20: {seen}")
    history = _assert_unique(
        _history(wandb_mod, path, until=lambda rows: dict(rows).get(24) == 1024.0)
    )
    print(f"history steps: {sorted(history)}")
    assert seen["id"] == path.rsplit("/", 1)[1], "resume_from must continue the same run id"
    assert seen["config_a"] == 1
    assert {s: history[s] for s in range(20)} == {s: float(s) for s in range(20)}, history
    assert {s: history.get(s) for s in range(20, 25)} == {s: 1000.0 + s for s in range(20, 25)}
    assert max(history) == 24, f"rows past the replay survived the rewind: {sorted(history)}"


def test_rewind_one_past_the_last_step_is_a_noop(wandb_mod: Any, created: list[str]) -> None:
    """Graceful drain: the checkpoint is at the last logged step, nothing to truncate."""
    path = _new_run(wandb_mod, range(10), created)
    seen = _rewind(wandb_mod, path, 10, {s: 1000.0 + s for s in range(10, 13)})
    print(f"after resume_from ?_step=10: {seen}")
    history = _assert_unique(
        _history(wandb_mod, path, until=lambda rows: dict(rows).get(12) == 1012.0)
    )
    assert {s: history[s] for s in range(10)} == {s: float(s) for s in range(10)}
    assert {s: history[s] for s in range(10, 13)} == {s: 1000.0 + s for s in range(10, 13)}


def test_rewind_past_the_last_step(wandb_mod: Any, created: list[str]) -> None:
    """W&B behind the checkpoint: record whether the server accepts it or raises."""
    path = _new_run(wandb_mod, range(10), created)
    try:
        seen = _rewind(wandb_mod, path, 15, {15: 1015.0, 16: 1016.0})
    except Exception as exc:  # noqa: BLE001  # recording the server's behaviour is the point
        print(f"resume_from past the last step raised {type(exc).__name__}: {exc}")
        pytest.skip("server rejects rewinding past the last step; the Trainer must fall back")
    print(f"after resume_from ?_step=15: {seen}")
    history = _assert_unique(
        _history(wandb_mod, path, until=lambda rows: dict(rows).get(16) == 1016.0)
    )
    assert {s: history[s] for s in range(10)} == {s: float(s) for s in range(10)}
    assert history.get(15) == 1015.0 and history.get(16) == 1016.0, sorted(history)


def test_trainer_resume_behind_the_server_history(
    wandb_mod: Any,
    created: list[str],
    training_parquet: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The real Trainer: Step equals the training step and each step is logged exactly once.

    Attempt 0 logs steps 1-6 but checkpoints only at 4 (a crash after step 6). Attempt 1
    resumes from checkpoint 4 and trains to 8. With W&B rewind enabled, steps 5-6 are the
    resumed attempt's rows; with it refused, the Trainer warns and steps 5-6 keep attempt
    0's rows. Steps 7-8 are always attempt 1's.
    """
    import logging

    from oplm.training.trainer import Trainer
    from tests.training.conftest import tiny_train_cfg

    class _TaggedTrainer(Trainer):
        attempt = 0

        def _log_metrics(self, metrics: dict[str, float]) -> None:
            if "train/loss" in metrics:
                metrics = metrics | {"drill/attempt": float(self.attempt)}
            super()._log_metrics(metrics)

    def cfg(max_steps: int, **overrides: Any) -> Any:
        return tiny_train_cfg(
            tmp_path,
            training_parquet,
            max_steps=max_steps,
            log_every=1,
            save_every=4,
            save_final=False,
            wandb_enabled=True,
            wandb_project=PROJECT,
            wandb_run_name=f"trainer-rewind-drill-{uuid.uuid4().hex[:8]}",
            **overrides,
        )

    first = _TaggedTrainer(cfg(6), callbacks=[])
    path = f"{wandb_mod.run.entity}/{PROJECT}/{wandb_mod.run.id}"
    created.append(path)
    first.train()
    _history_steps(wandb_mod, path, until=lambda rows: 6 in rows)

    second = _TaggedTrainer(cfg(8, auto_resume=True), callbacks=[])
    second.attempt = 1
    with caplog.at_level(logging.WARNING, logger="oplm.training.trainer"):
        second.train()
    refused = "W&B history rewind unavailable" in caplog.text
    print(f"rewind {'refused (fallback)' if refused else 'applied'}")

    rows = _history_steps(wandb_mod, path, until=lambda rows: 8 in rows)
    print(f"history (step: global_step, attempt): {rows}")
    assert sorted(rows) == list(range(1, 9)), "every training step exactly once"
    assert all(global_step == step for step, (global_step, _) in rows.items())
    window_attempt = 0.0 if refused else 1.0
    assert [rows[s][1] for s in range(1, 9)] == [0.0] * 4 + [window_attempt] * 2 + [1.0] * 2


def _history_steps(wandb: Any, path: str, *, until: Any) -> dict[int, tuple[int, float]]:
    """``{_step: (train/global_step, drill/attempt)}`` of a Trainer run, polled like _history."""
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    while True:
        run = wandb.Api().run(path)
        raw = list(run.scan_history(keys=["train/global_step", "drill/attempt"]))
        steps = [int(r["_step"]) for r in raw]
        assert len(steps) == len(set(steps)), f"duplicate steps in history: {sorted(steps)}"
        rows = {
            int(r["_step"]): (int(r["train/global_step"]), float(r["drill/attempt"])) for r in raw
        }
        if until(rows) or time.monotonic() > deadline:
            return rows
        time.sleep(_POLL_SECONDS)


def test_two_consecutive_rewinds(wandb_mod: Any, created: list[str]) -> None:
    """A crash during the replay: rewind again inside the replayed range."""
    path = _new_run(wandb_mod, range(30), created)
    _rewind(wandb_mod, path, 20, {s: 1000.0 + s for s in range(20, 25)})
    _history(wandb_mod, path, until=lambda rows: dict(rows).get(24) == 1024.0)
    seen = _rewind(wandb_mod, path, 22, {22: 2022.0, 23: 2023.0})
    print(f"after the second resume_from ?_step=22: {seen}")
    history = _assert_unique(
        _history(wandb_mod, path, until=lambda rows: dict(rows).get(23) == 2023.0)
    )
    assert {s: history[s] for s in range(20)} == {s: float(s) for s in range(20)}
    assert history[20] == 1020.0 and history[21] == 1021.0
    assert history[22] == 2022.0 and history[23] == 2023.0
    assert max(history) == 23, sorted(history)

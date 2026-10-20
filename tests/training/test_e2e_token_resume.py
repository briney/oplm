"""Token mode resumes data-exactly: same rows per worker, across workers, epochs and budgets."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.training.conftest import FullRecordingCallback, tiny_train_cfg
from tests.training.test_e2e_data_exact import _force_keep_sequence_ids, _wrap_recorder

pytestmark = pytest.mark.slow

TOKEN = dict(pack_sequences=True, log_every=1)


def _run(cfg, callbacks=None):  # noqa: ANN001, ANN202
    from oplm.training.trainer import Trainer

    trainer = Trainer(cfg, callbacks=callbacks or [])
    recorder = _wrap_recorder(trainer)
    trainer.train()
    return trainer, recorder


def _interrupt_and_resume(
    data: Path,
    tmp_path: Path,
    *,
    total: int,
    stop: int,
    first: dict,
    second: dict,
    num_workers: int = 0,
):  # noqa: ANN202
    common = dict(num_workers=num_workers, **TOKEN)
    control, control_rec = _run(
        tiny_train_cfg(
            tmp_path / "control", data, max_steps=total, save_final=False, **first, **common
        )
    )
    run_dir = tmp_path / "interrupt"
    callback = FullRecordingCallback()
    run1, rec1 = _run(
        tiny_train_cfg(
            run_dir, data, max_steps=stop, save_every=stop, save_final=False, **first, **common
        ),
        [callback],
    )
    assert callback.checkpoint_steps[-1] == run1.global_step == stop  # no deferral in token mode
    state = json.loads((run_dir / f"checkpoint-{stop}" / "trainer_state.json").read_text())
    run2, rec2 = _run(
        tiny_train_cfg(
            run_dir, data, max_steps=total, save_final=False, auto_resume=True, **second, **common
        )
    )
    return control, control_rec, run1, rec1, run2, rec2, state


@pytest.mark.parametrize("num_workers", [0, 2])
def test_token_resume_matches_control_mid_epoch(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, num_workers: int
) -> None:
    _force_keep_sequence_ids(monkeypatch)
    budget = dict(tokens_per_step=256, batch_tokens=256)
    control, control_rec, run1, rec1, run2, rec2, state = _interrupt_and_resume(
        training_parquet,
        tmp_path,
        total=12,
        stop=6,
        first=budget,
        second=budget,
        num_workers=num_workers,
    )
    cursor = state["cursor"]
    assert cursor["rank_epochs"] == [0] and run1.epoch == 0
    assert len(cursor["worker_rows"]) == 1 and len(cursor["worker_rows"][0]) == max(num_workers, 1)
    # Six batches is a whole number of worker cycles, so even the interleaving is exact.
    assert rec1.flat() + rec2.flat() == control_rec.flat()
    assert run2.tokens_seen == control.tokens_seen


def test_token_resume_keeps_position_when_the_budget_changes(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_keep_sequence_ids(monkeypatch)
    _, control_rec, _, rec1, _, rec2, _ = _interrupt_and_resume(
        training_parquet,
        tmp_path,
        total=12,
        stop=6,
        first=dict(tokens_per_step=256, batch_tokens=256),
        second=dict(tokens_per_step=192, batch_tokens=192),
    )
    combined = rec1.flat() + rec2.flat()
    # One worker, no epoch crossing: any budget serves the stream in order, so the resumed run
    # (smaller batches after step 6) is a prefix of the control's stream.
    assert combined == control_rec.flat()[: len(combined)]


def test_token_resume_matches_control_across_an_epoch(
    tiny_training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_keep_sequence_ids(monkeypatch)
    budget = dict(tokens_per_step=128, batch_tokens=128)
    _, control_rec, run1, rec1, _, rec2, state = _interrupt_and_resume(
        tiny_training_parquet,
        tmp_path,
        total=20,
        stop=10,
        first=budget,
        second=budget,
    )
    assert run1.epoch >= 1 and state["cursor"]["rank_epochs"] == [run1.epoch]
    assert rec1.flat() + rec2.flat() == control_rec.flat()


def test_resume_across_batching_modes_raises(training_parquet: Path, tmp_path: Path) -> None:
    from oplm.training.trainer import Trainer

    run_dir = tmp_path / "run"
    Trainer(tiny_train_cfg(run_dir, training_parquet, max_steps=2), callbacks=[]).train()
    resumed = tiny_train_cfg(
        run_dir,
        training_parquet,
        max_steps=4,
        auto_resume=True,
        tokens_per_step=256,
        batch_tokens=256,
        **TOKEN,
    )
    with pytest.raises(ValueError, match="resume_data_position=false"):
        Trainer(resumed, callbacks=[])


def test_token_resume_rejects_a_changed_worker_count(
    training_parquet: Path, tmp_path: Path
) -> None:
    from oplm.training.trainer import Trainer

    run_dir = tmp_path / "run"
    budget = dict(tokens_per_step=256, batch_tokens=256, **TOKEN)
    Trainer(tiny_train_cfg(run_dir, training_parquet, max_steps=2, **budget), callbacks=[]).train()
    resumed = tiny_train_cfg(
        run_dir, training_parquet, max_steps=4, auto_resume=True, num_workers=2, **budget
    )
    with pytest.raises(ValueError, match="num_workers"):
        Trainer(resumed, callbacks=[])


def _launch(worker: Path, *args: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join(
        p for p in (str(repo_root), str(repo_root / "src"), os.environ.get("PYTHONPATH", "")) if p
    )
    env = {
        **os.environ,
        "ACCELERATE_USE_CPU": "true",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": pythonpath,
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            str(worker),
            *args,
        ],
        check=True,
        timeout=300,
        env=env,
    )


def _rollover_steps(records: list[dict]) -> list[int]:  # type: ignore[type-arg]
    return [i for i in range(1, len(records)) if records[i]["epoch"] != records[i - 1]["epoch"]]


def test_two_rank_token_resume_with_uneven_rollovers(
    tiny_training_parquet: Path, tmp_path: Path
) -> None:
    worker = Path(__file__).with_name("_token_resume_worker.py")
    out = tmp_path / "out"
    out.mkdir()
    total, stop = 12, 6
    data = str(tiny_training_parquet)
    _launch(worker, str(tmp_path / "control"), data, str(out), "control", str(total), str(stop))
    for phase in ("first", "second"):
        _launch(worker, str(tmp_path / "interrupt"), data, str(out), phase, str(total), str(stop))

    def load(phase: str, rank: int) -> list[dict]:  # type: ignore[type-arg]
        return json.loads((out / f"{phase}-rank{rank}.json").read_text())

    control = [load("control", rank) for rank in range(2)]
    # The point of the test: the two ranks reach a new epoch at different steps, and are in
    # different epochs at the checkpoint, so each must restore its own.
    assert _rollover_steps(control[0]) != _rollover_steps(control[1])
    state = json.loads(
        (tmp_path / "interrupt" / f"checkpoint-{stop}" / "trainer_state.json").read_text()
    )
    rank_epochs = state["cursor"]["rank_epochs"]
    assert len(rank_epochs) == 2 and rank_epochs[0] != rank_epochs[1]
    for rank in range(2):
        resumed = [r["ids"] for r in load("first", rank) + load("second", rank)]
        assert resumed == [r["ids"] for r in control[rank]], rank

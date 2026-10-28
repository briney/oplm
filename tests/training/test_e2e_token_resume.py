"""Token mode resumes data-exactly: same rows per worker, across workers, epochs and budgets."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tests.data.conftest import write_sequence_shards
from tests.training._token_resume_worker import patch_for_recording, record_batches
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


Records = list[dict]  # type: ignore[type-arg]  # one {"epoch", "ids"} per batch


@pytest.fixture
def mix_train(training_parquet: Path, tmp_path: Path) -> dict[str, dict[str, object]]:
    """A 2-source ``data.train`` mix: 16 and 32 distinct real rows, half the draws each."""
    table = pq.read_table(training_parquet, columns=["sequence_id", "sequence"])
    records = list(zip(*(table.column(c).to_pylist() for c in table.column_names), strict=True))
    mix: dict[str, dict[str, object]] = {}
    for name, rows in (("a", records[:16]), ("b", records[16:48])):
        (tmp_path / name).mkdir()
        write_sequence_shards(tmp_path / name, rows, id_prefix=f"{name}:")
        mix[name] = {"path": str(tmp_path / name), "fraction": 0.5}
    return mix


def _cfg(run_dir: Path, data: Path | dict[str, object], **kwargs: object):  # noqa: ANN202
    """``tiny_train_cfg`` for a parquet path or a ``data.train`` mix mapping."""
    cfg = tiny_train_cfg(run_dir, data if isinstance(data, Path) else Path("mix"), **kwargs)  # type: ignore[arg-type]
    if not isinstance(data, Path):
        cfg.data.train = data
    return cfg


def _run_recorded(cfg):  # noqa: ANN001, ANN202
    from oplm.training.trainer import Trainer

    trainer = Trainer(cfg, callbacks=[])
    records = record_batches(trainer)
    trainer.train()
    return trainer, records


def _assert_finished_epochs_exact(phases: list[list[Records]], ids: set[str] | None = None) -> None:
    """Each epoch every rank of the last phase got past serves each id exactly once overall.

    ``phases`` holds, per phase, each rank's records. A mix's ids carry their source pass
    (``patch_for_recording``), so for a mix this is "exact per source pass": no row twice in one
    pass. With ``ids`` (a single source, ``drop_last=false``), every id is also served.
    """
    finished = min(max(r["epoch"] for r in records) for records in phases[-1])
    assert finished > 0, "no epoch finished"
    served: defaultdict[int, Counter[str]] = defaultdict(Counter)
    for ranks in phases:
        for records in ranks:
            for record in records:
                served[record["epoch"]].update(record["ids"])
    for epoch in range(finished):
        repeated = [sid for sid, n in served[epoch].items() if n > 1]
        assert not repeated, (epoch, repeated)
        if ids is not None:
            assert set(served[epoch]) == ids, epoch


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


def test_sequence_cursor_resumes_into_token_mode(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_keep_sequence_ids(monkeypatch)
    sequence = dict(batch_size=4, **TOKEN)
    _, control_rec = _run(
        tiny_train_cfg(
            tmp_path / "control", training_parquet, max_steps=24, save_final=False, **sequence
        )
    )
    run_dir = tmp_path / "run"
    run1, rec1 = _run(
        tiny_train_cfg(
            run_dir, training_parquet, max_steps=3, save_every=3, save_final=False, **sequence
        )
    )
    state = json.loads((run_dir / "checkpoint-3" / "trainer_state.json").read_text())
    assert state["cursor"].get("worker_rows") is None and run1.epoch == 0
    _, rec2 = _run(
        tiny_train_cfg(
            run_dir,
            training_parquet,
            max_steps=9,
            save_final=False,
            auto_resume=True,
            tokens_per_step=256,
            batch_tokens=256,
            **TOKEN,
        )
    )
    combined = rec1.flat() + rec2.flat()
    assert len(rec1.flat()) == 12 and rec2.flat()
    # One worker: any batching serves the stream in order, so the 12 rows sequence mode had
    # consumed plus the token-mode rows that follow are a prefix of the sequence-mode control.
    assert len(combined) <= len(control_rec.flat())
    assert combined == control_rec.flat()[: len(combined)]


def test_resume_token_cursor_into_sequence_mode_raises(
    training_parquet: Path, tmp_path: Path
) -> None:
    from oplm.training.trainer import Trainer

    run_dir = tmp_path / "run"
    budget = dict(tokens_per_step=256, batch_tokens=256, **TOKEN)
    Trainer(tiny_train_cfg(run_dir, training_parquet, max_steps=2, **budget), callbacks=[]).train()
    resumed = tiny_train_cfg(run_dir, training_parquet, max_steps=4, auto_resume=True)
    with pytest.raises(ValueError, match=r"token-mode run.*resume_data_position=false"):
        Trainer(resumed, callbacks=[])


def test_token_resume_restripes_a_changed_worker_count(
    mix_train: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_for_recording(monkeypatch)
    run_dir = tmp_path / "run"
    budget = dict(tokens_per_step=128, batch_tokens=128, drop_last=False, **TOKEN)
    _, rec1 = _run_recorded(
        _cfg(run_dir, mix_train, max_steps=10, save_every=10, save_final=False, **budget)
    )
    state = json.loads((run_dir / "checkpoint-10" / "trainer_state.json").read_text())
    assert state["cursor"]["num_workers"] == 1 and state["cursor"]["rank_epochs"] == [0]
    assert state["cursor"]["worker_sources"][0][0] is not None
    run2, rec2 = _run_recorded(
        _cfg(
            run_dir,
            mix_train,
            max_steps=60,
            save_final=False,
            auto_resume=True,
            num_workers=2,
            **budget,
        )
    )
    assert run2.epoch >= 2
    _assert_finished_epochs_exact([[rec1], [rec2]])


def test_same_layout_resume_after_a_reshard(
    mix_train: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_for_recording(monkeypatch)
    run_dir = tmp_path / "run"
    budget = dict(tokens_per_step=128, batch_tokens=128, drop_last=False, **TOKEN)
    _, rec1 = _run_recorded(
        _cfg(run_dir, mix_train, max_steps=10, save_every=10, save_final=False, **budget)
    )
    two = dict(num_workers=2, auto_resume=True, **budget)
    # Phase 2 (resharded to 2 workers) saves while still in epoch 0, so the event is kept.
    _, rec2 = _run_recorded(
        _cfg(run_dir, mix_train, max_steps=16, save_every=16, save_final=False, **two)
    )
    cursor = json.loads((run_dir / "checkpoint-16" / "trainer_state.json").read_text())["cursor"]
    assert cursor["rank_epochs"] == [0] and len(cursor["reshards"]) == 1
    assert cursor["reshards"][0]["stride"] == 1
    run3, rec3 = _run_recorded(_cfg(run_dir, mix_train, max_steps=60, save_final=True, **two))
    assert run3.epoch >= 2
    _assert_finished_epochs_exact([[rec1], [rec2], [rec3]])
    # Pruned once every rank is past the epochs it applies to.
    final = json.loads((run_dir / "checkpoint-60" / "trainer_state.json").read_text())["cursor"]
    assert final["reshards"] == []


def test_token_resume_rejects_a_changed_seed(training_parquet: Path, tmp_path: Path) -> None:
    from oplm.training.trainer import Trainer

    run_dir = tmp_path / "run"
    budget = dict(tokens_per_step=256, batch_tokens=256, **TOKEN)
    Trainer(tiny_train_cfg(run_dir, training_parquet, max_steps=2, **budget), callbacks=[]).train()
    resumed = tiny_train_cfg(
        run_dir, training_parquet, max_steps=4, auto_resume=True, seed=7, **budget
    )
    with pytest.raises(ValueError, match=r"seed.*resume_data_position=false"):
        Trainer(resumed, callbacks=[])


def test_reshard_of_a_mix_needs_worker_sources(mix_train: dict, tmp_path: Path) -> None:
    from oplm.training.trainer import Trainer

    run_dir = tmp_path / "run"
    budget = dict(tokens_per_step=128, batch_tokens=128, **TOKEN)
    Trainer(_cfg(run_dir, mix_train, max_steps=2, **budget), callbacks=[]).train()
    state_path = run_dir / "checkpoint-2" / "trainer_state.json"
    state = json.loads(state_path.read_text())
    del state["cursor"]["worker_sources"]  # a cursor from before per-source positions
    state_path.write_text(json.dumps(state))
    resumed = _cfg(run_dir, mix_train, max_steps=4, auto_resume=True, num_workers=2, **budget)
    with pytest.raises(ValueError, match="resume_data_position=false"):
        Trainer(resumed, callbacks=[])


def _launch(worker: Path, *args: str, nproc: int = 2) -> None:
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
            f"--nproc_per_node={nproc}",
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


# Checkpoint steps where the two ranks are in different epochs (one of them mid-epoch) at width
# 256 and drop_last=false: single source, rank 0 in epoch 4 and rank 1 in 5; mix, the reverse.
_STRADDLE_STOP = {"single": 19, "mix": 20}


@pytest.mark.parametrize(
    ("first_nproc", "second_nproc", "kind"),
    [
        (2, 1, "single"),
        (2, 1, "mix"),
        (1, 2, "single"),
    ],
)
def test_two_rank_token_reshard(
    tiny_training_parquet: Path,
    mix_train: dict,
    tmp_path: Path,
    first_nproc: int,
    second_nproc: int,
    kind: str,
) -> None:
    worker = Path(__file__).with_name("_token_resume_worker.py")
    out = tmp_path / "out"
    out.mkdir()
    total, stop = 40, _STRADDLE_STOP[kind]
    data = str(tiny_training_parquet) if kind == "single" else json.dumps(mix_train)
    run = str(tmp_path / "interrupt")
    for phase, nproc in (("first", first_nproc), ("second", second_nproc)):
        _launch(worker, run, data, str(out), phase, str(total), str(stop), "false", nproc=nproc)

    cursor = json.loads(
        (tmp_path / "interrupt" / f"checkpoint-{stop}" / "trainer_state.json").read_text()
    )["cursor"]
    if first_nproc == 2:
        # The straddle: the old ranks were in different epochs at the checkpoint.
        assert cursor["rank_epochs"][0] != cursor["rank_epochs"][1]
    phases = [
        [json.loads((out / f"{phase}-rank{r}.json").read_text()) for r in range(nproc)]
        for phase, nproc in (("first", first_nproc), ("second", second_nproc))
    ]
    ids = None
    if kind == "single":
        ids = set(pq.read_table(tiny_training_parquet).column("sequence_id").to_pylist())
    _assert_finished_epochs_exact(phases, ids)

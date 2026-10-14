"""fold_monomer: registration, config, CPU evaluate, failures, rank sharding, Trainer run."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import torch
from accelerate import Accelerator

import oplm.fold.eval  # noqa: F401  (registers fold_monomer)
from oplm.config import EvalDatasetEntry, OplmConfig, ScheduleSpec
from oplm.eval.registry import get_eval_task_class
from oplm.fold.eval import tasks
from oplm.fold.eval.monomer import METRICS, TargetScore
from oplm.fold.eval.tasks import FoldMonomerTask, FoldMonomerTaskConfig
from oplm.training import TrainerCallback
from tests.fold.helpers import tiny_fold_run_cfg, tiny_folder, tiny_lm

if TYPE_CHECKING:
    from pathlib import Path


def _task(path: Path, **extra: Any) -> FoldMonomerTask:
    entry = EvalDatasetEntry(
        name="valid",
        path=str(path),
        type="fold_monomer",
        schedule=ScheduleSpec("steps", 1),
        extra=extra,
    )
    return FoldMonomerTask(entry, OplmConfig())


def test_fold_monomer_registers_on_import() -> None:
    assert get_eval_task_class("fold_monomer") is FoldMonomerTask


def test_mlm_eval_tooling_never_imports_fold_code() -> None:
    code = (
        "import sys, oplm.eval, oplm.eval.tasks; "
        "bad = sorted(m for m in sys.modules if m.startswith('oplm.fold')); "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_task_config_rejects_bad_options() -> None:
    assert FoldMonomerTaskConfig.from_extra({}).target_set == "valid"
    with pytest.raises(ValueError, match="unknown fold_monomer option"):
        FoldMonomerTaskConfig.from_extra({"targets": "valid"})
    with pytest.raises(ValueError, match="num_samples"):
        FoldMonomerTaskConfig.from_extra({"num_samples": 0})
    with pytest.raises(ValueError, match="seed"):
        FoldMonomerTaskConfig.from_extra({"seed": "0"})
    entry = EvalDatasetEntry(
        name="v",
        path="x",
        type="fold_monomer",
        schedule=ScheduleSpec("steps", 1),
        metrics=["tm_score", "gdt_ts"],
    )
    with pytest.raises(ValueError, match="gdt_ts"):
        FoldMonomerTask(entry, OplmConfig())


def test_evaluate_scores_the_valid_split_on_cpu(fixture_dataset_early_cutoff: Path) -> None:
    metrics = _task(fixture_dataset_early_cutoff).evaluate(tiny_folder(), Accelerator(cpu=True))
    assert set(metrics) == {*METRICS, "targets", "failed"}
    assert metrics["targets"] == 1.0 and metrics["failed"] == 0.0
    assert 0.0 <= metrics["tm_score"] <= 1.0


def test_evaluate_counts_a_failed_target(
    fixture_dataset_early_cutoff: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr("oplm.fold.eval.monomer.fold", boom)
    metrics = _task(fixture_dataset_early_cutoff).evaluate(tiny_folder(), Accelerator(cpu=True))
    assert metrics["targets"] == 0.0 and metrics["failed"] == 1.0


def _ok_row(entry: str) -> TargetScore:
    return TargetScore(
        target_set="three",
        entry_id=entry,
        chain_id="A",
        auth_chain_id="A",
        length=1,
        resolved_residues=1,
        status="ok",
        tm_score=0.5,
        lddt_ca=0.5,
        lddt=0.5,
        plddt=0.5,
        ptm=0.5,
    )


def test_evaluate_shards_targets_across_ranks(
    fixture_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ds"
    shutil.copytree(fixture_dataset, root)
    items = [
        {"id": "1crn", "chain": "A"},
        {"id": "1jb6", "chain": None},
        {"id": "1l2y", "chain": "A"},
    ]
    (root / "targets" / "three.json").write_text(json.dumps(items))
    scored: list[str] = []

    def fake_score(model: object, target: Any, settings: object) -> TargetScore:
        scored.append(target.key)
        return _ok_row(target.entry_id)

    monkeypatch.setattr(tasks, "score_target", fake_score)
    # Rank 1 of 2: targets sorted 1crn_A, 1jb6_A, 1l2y_A -> this rank scores index 1 only; the
    # gather returns every rank's rows, so the aggregate covers all three exactly once.
    monkeypatch.setattr(
        tasks, "gather_objects", lambda local, acc: [_ok_row("1crn"), *local, _ok_row("1l2y")]
    )
    metrics = _task(root, target_set="three").evaluate(
        tiny_folder(), SimpleNamespace(process_index=1, num_processes=2)
    )
    assert scored == ["1jb6_A"] and metrics["targets"] == 3.0
    # More ranks than targets: rank 5 of 8 scores nothing but still reaches the gather.
    scored.clear()
    reached: list[int] = []
    monkeypatch.setattr(
        tasks, "gather_objects", lambda local, acc: reached.append(len(local)) or local
    )
    metrics = _task(root, target_set="three").evaluate(
        tiny_folder(), SimpleNamespace(process_index=5, num_processes=8)
    )
    assert scored == [] and reached == [0] and metrics["targets"] == 0.0


class _EvalRecorder(TrainerCallback):
    def __init__(self) -> None:
        self.calls: list[dict[str, float]] = []

    def on_eval_end(self, trainer: Any, metrics: dict[str, float], step: int) -> None:
        self.calls.append(dict(metrics))


@pytest.mark.slow
def test_trainer_runs_fold_monomer_eval(
    tmp_path: Path,
    fixture_dataset: Path,
    fixture_dataset_early_cutoff: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oplm.fold.training import FoldTask
    from oplm.training.trainer import Trainer
    from tests.training.conftest import configure_accelerator_device

    configure_accelerator_device("cpu", monkeypatch)
    lm_dir = tmp_path / "lm"
    tiny_lm().save_pretrained(lm_dir)
    cfg = tiny_fold_run_cfg(tmp_path / "run", fixture_dataset, lm_dir, max_steps=2)
    cfg.data.eval = {
        "valid": {
            "path": str(fixture_dataset_early_cutoff),
            "type": "fold_monomer",
            "every": {"steps": 2},
        }
    }
    recorder = _EvalRecorder()
    Trainer(cfg, task=FoldTask(), callbacks=[recorder]).train()
    keys = {k for call in recorder.calls for k in call}
    assert "eval/valid/tm_score" in keys and "eval/valid/targets" in keys
    assert all(0.0 <= call["eval/valid/tm_score"] <= 1.0 for call in recorder.calls)

"""Recovery drills must reject false positives, especially dropped W&B history."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def evidence() -> tuple[dict, list[dict], list[dict]]:
    snapshot = dict(
        step=10,
        epoch=0,
        samples=80,
        tokens=800,
        batches=10,
        lr=[[0.001]],
        world_size=2,
        global_batch=8,
    )
    identity = dict(id="abc", project="recovery", entity="team", name="rank-kill")
    events = []
    for rank in range(2):
        events += [
            dict(
                kind="start",
                attempt=0,
                rank=rank,
                step=0,
                world_size=2,
                global_batch=8,
                wandb=identity,
                node=f"node{rank}",
                time=1,
            ),
            dict(kind="save", attempt=0, rank=rank, **snapshot, time=2),
            dict(
                kind="start",
                attempt=1,
                rank=rank,
                **snapshot,
                wandb=identity,
                node=f"node{rank}",
                resume="/run/checkpoint-10",
                time=5,
            ),
        ]
    events += [
        dict(kind="commit", attempt=0, rank=0, step=10, time=3),
        dict(kind="inject", attempt=0, rank=1, step=15, time=4),
        dict(kind="log", attempt=1, rank=0, step=11, time=6),
        dict(kind="complete", attempt=1, rank=0, step=11, time=7),
    ]
    manifest = dict(
        mode="rank_kill",
        nodes=2,
        world_size=2,
        max_steps=11,
        wandb_enabled=True,
        terminal_state="COMPLETED",
    )
    history = [{"recovery/attempt": 1, "recovery/step": 11}]
    return manifest, events, history


def test_recovery_report_requires_restored_state_and_online_history() -> None:
    from oplm.slurm.recovery.checks import assess

    manifest, events, history = evidence()
    assert assess(manifest, events, history)["status"] == "passed"
    broken = copy.deepcopy(events)
    next(e for e in broken if e["kind"] == "start" and e["attempt"] == 1)["samples"] = 0
    assert assess(manifest, broken, history)["status"] == "failed"
    assert assess(manifest, events, [])["status"] == "failed"
    assert assess(manifest, events, None)["status"] == "incomplete"


@pytest.mark.parametrize(
    "mutation", ["no_injection", "no_rank", "wrong_checkpoint", "no_restart", "no_metrics"]
)
def test_recovery_report_does_not_confuse_completion_with_recovery(mutation: str) -> None:
    from oplm.slurm.recovery.checks import assess

    manifest, events, history = evidence()
    if mutation == "no_injection":
        events = [e for e in events if e["kind"] != "inject"]
    elif mutation == "no_rank":
        events = [e for e in events if not (e["kind"] == "start" and e["rank"] == 1)]
    elif mutation == "wrong_checkpoint":
        next(e for e in events if e["kind"] == "start" and e["attempt"] == 1)["resume"] = (
            "/run/checkpoint-9"
        )
    elif mutation == "no_metrics":
        events = [e for e in events if e["kind"] != "log"]
    else:
        events = [e for e in events if e["attempt"] == 0]
    assert assess(manifest, events, history)["status"] == "failed"


def test_node_drill_requires_a_replacement_node() -> None:
    from oplm.slurm.recovery.checks import assess

    manifest, events, history = evidence()
    manifest.update(mode="worker_node_loss", failed_node="node1")
    assert assess(manifest, events, history)["status"] == "failed"
    for event in events:
        if event["attempt"] == 1 and event["rank"] == 1:
            event["node"] = "node2"
    assert assess(manifest, events, history)["status"] == "passed"


@pytest.mark.parametrize("mode", ["graceful_drain", "rank_kill", "checkpoint_write", "crash_loop"])
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.slow
def test_real_worker_kills_then_restores_checkpoint(
    mode: str, world_size: int, training_parquet: Path, tmp_path: Path
) -> None:
    """Real CPU subprocesses prove the injection executes and the real Trainer restores state."""
    from oplm.config import serialize_config
    from oplm.slurm.recovery.checks import assess
    from oplm.slurm.recovery.runner import read_events
    from tests.training.conftest import tiny_train_cfg

    (tmp_path / "events").mkdir()
    cfg = tiny_train_cfg(
        tmp_path / "training", training_parquet, max_steps=8, save_every=2, auto_resume=True
    )
    (tmp_path / "train.yaml").write_text(serialize_config(cfg))
    manifest = dict(
        mode=mode,
        nodes=1,
        world_size=world_size,
        max_steps=8,
        save_every=2,
        fail_step=3,
        kill_rank=world_size - 1,
        wandb_enabled=False,
        node_wait_seconds=5,
    )
    (tmp_path / "drill.json").write_text(json.dumps(manifest))
    for attempt in (0, 1):
        env = dict(
            os.environ,
            ACCELERATE_USE_CPU="true",
            SLURM_RESTART_COUNT=str(attempt),
            OMP_NUM_THREADS="1",
            HF_HUB_OFFLINE="1",
            TRITON_CACHE_DIR=str(tmp_path / "triton"),
        )
        command = [sys.executable]
        if world_size == 2:
            command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
        command += ["-m", "oplm.slurm.recovery.worker", "--directory", str(tmp_path)]
        result = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        (tmp_path / f"attempt-{attempt}.log").write_text(result.stdout + result.stderr)
        expected = 85 if mode == "graceful_drain" else -9
        if world_size == 2:
            expected = 1  # torchrun flattens worker failures, like Accelerate does.
        if attempt == 1 and mode != "crash_loop":
            expected = 0
        assert result.returncode == expected, result.stdout + result.stderr
    events = read_events(tmp_path)
    manifest.update(
        terminal_state="FAILED" if mode == "crash_loop" else "COMPLETED",
        guard_stopped=mode == "crash_loop",
    )
    # This test supplies scheduler evidence only; production shell guard has its own Bash tests.
    report = assess(manifest, events, None)
    assert report["errors"] == [], report
    assert report["status"] == "incomplete"  # No live W&B or Slurm was exercised here.


def test_preparation_is_isolated_and_cannot_overwrite_an_existing_drill(tmp_path: Path) -> None:
    from oplm.config import load_config

    destination = tmp_path / "drill"
    args = [
        sys.executable,
        "-m",
        "oplm.slurm.recovery.rank_kill",
        "--config",
        "configs/scaling.yaml",
        "--preset",
        "170M",
        "--out",
        str(destination),
        "--set",
        "data.num_workers=0",
        "--no-wandb",
    ]
    first = subprocess.run(args, capture_output=True, text=True, timeout=30)
    assert first.returncode == 0, first.stderr
    cfg = load_config(["--config", str(destination / "train.yaml")])
    assert cfg.train.output_dir == str(destination / "training")
    assert cfg.train.auto_resume and cfg.train.resume_from is None
    assert cfg.train.remote_checkpoint_uri is None
    assert cfg.train.max_steps == 64
    script = destination / "job.sbatch"
    subprocess.run(["bash", "-n", str(script)], check=True)
    before = script.read_bytes()
    second = subprocess.run(args, capture_output=True, text=True, timeout=30)
    assert second.returncode != 0
    assert script.read_bytes() == before


@pytest.mark.parametrize("mode,target", [("worker_node_loss", "n2"), ("batch_node_loss", "n1")])
def test_node_command_targets_only_the_submitted_allocation(
    mode: str, target: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oplm.slurm.recovery import runner

    (tmp_path / "events").mkdir()
    adapter = tmp_path / "adapter"
    adapter.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    adapter.chmod(0o755)
    manifest = dict(mode=mode, nodes=2, fail_step=24, job_id="123", node_wait_seconds=10)
    starts = [dict(kind="start", attempt=0, node=node, time=1) for node in ("n1", "n2")]
    (tmp_path / "events" / "workers.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in starts)
    )

    def scheduler(argv: list[str]) -> str:
        if argv == ["scontrol", "show", "job", "--oneliner", "123"]:
            return "JobId=123 NodeList=n[1-2] BatchHost=n1 JobState=RUNNING Restarts=0"
        assert argv == ["scontrol", "show", "hostnames", "n[1-2]"]
        return "n1\nn2\n"

    monkeypatch.setattr(runner, "_command", scheduler)
    runner._inject_node(tmp_path, manifest, adapter)
    assert (tmp_path / "node-command.log").read_text().splitlines() == ["123", target]
    assert manifest["failed_node"] == target
    assert any(e["kind"] == "inject" for e in runner.read_events(tmp_path))

    manifest["nodes"] = 3
    with pytest.raises(RuntimeError, match="does not match"):
        runner._inject_node(tmp_path, manifest, adapter)
    manifest["nodes"] = 2
    (tmp_path / "events" / "restart.jsonl").write_text(
        json.dumps(dict(kind="start", attempt=1, node="n3", time=2)) + "\n"
    )
    with pytest.raises(RuntimeError, match="does not match"):
        runner._inject_node(tmp_path, manifest, adapter)


def test_post_submission_storage_failure_cancels_known_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oplm.slurm.recovery import runner

    monkeypatch.setattr(runner, "submit_job", lambda script: "123")
    cancelled = []
    monkeypatch.setattr(runner, "_command", lambda argv: cancelled.append(argv))

    def unavailable(path: Path, value: object) -> None:
        raise OSError("shared storage unavailable")

    monkeypatch.setattr(runner, "write_json", unavailable)
    with pytest.raises(OSError, match="shared storage"):
        runner.execute(
            tmp_path,
            {"mode": "rank_kill"},
            argparse.Namespace(node_failure_command=None, timeout=60),
        )
    assert cancelled == [["scancel", "123"]]

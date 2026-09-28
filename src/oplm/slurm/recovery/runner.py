"""Prepare, submit, monitor, and assess one isolated Slurm recovery drill."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from oplm.slurm.config import load_slurm_config
from oplm.slurm.recovery.checks import assess
from oplm.slurm.render import JobSpec, accelerate_command, render_job
from oplm.slurm.submit import running_job_ids, submit_job

NODE_MODES = {"worker_node_loss", "batch_node_loss"}


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a small evidence file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_events(directory: Path) -> list[dict[str, Any]]:
    """Read per-rank journals, ignoring only a last line torn by a killed writer."""
    events = []
    for path in sorted((directory / "events").glob("*.jsonl")):
        lines = path.read_text().splitlines(keepends=True)
        for line in lines:
            if line.endswith("\n"):
                events.append(json.loads(line))
    return sorted(events, key=lambda event: event["time"])


def _safe_path(path: Path) -> str:
    # The existing production renderer embeds paths in several shell contexts.
    value = str(path)
    if not re.fullmatch(r"[\w/.:@+=,-]+", value, flags=re.ASCII):
        raise ValueError(
            f"Slurm drill paths must not contain whitespace or shell metacharacters: {path}"
        )
    return value


def prepare(mode: str, args: argparse.Namespace) -> dict[str, Any]:
    """Write a fresh drill config and production-rendered sbatch script without submitting."""
    from oplm.config import load_config, serialize_config

    if args.config is None:
        raise ValueError("--config is required when preparing a new drill")
    if args.nodes < 1 or args.save_every < 1 or args.max_steps <= args.fail_step:
        raise ValueError("nodes/save-every must be positive and max-steps must exceed fail-step")
    if not args.save_every < args.fail_step < 2 * args.save_every:
        raise ValueError("fail-step must fall strictly between the first two checkpoint steps")
    if args.max_steps <= 2 * args.save_every:
        raise ValueError("max-steps must leave training after the second checkpoint")
    if args.timeout <= 0 or args.node_wait_seconds <= 0:
        raise ValueError("timeouts must be positive")
    source = args.config.resolve()
    directory = args.out.resolve()
    _safe_path(directory)
    slurm = load_slurm_config(source)
    world_size = args.nodes * slurm.gpus_per_node
    if not 0 <= args.kill_rank < world_size:
        raise ValueError(f"kill-rank must be in [0, {world_size})")
    if mode in NODE_MODES and args.nodes < 2:
        raise ValueError("node replacement drills require at least two training nodes")
    argv = ["--config", str(source)]
    if args.preset:
        argv += ["--preset", args.preset]
    cfg = load_config(argv + args.set)
    cfg.train.output_dir = str(directory / "training")
    cfg.train.resume_from = None
    cfg.train.auto_resume = True
    cfg.train.resume_data_position = True
    cfg.train.max_steps = args.max_steps
    cfg.train.max_epochs = None
    cfg.train.warmup_steps = min(cfg.train.warmup_steps, args.max_steps // 10)
    cfg.train.stable_steps = min(cfg.train.stable_steps, args.max_steps // 2)
    cfg.train.save_every = args.save_every
    cfg.train.save_every_minutes = None
    cfg.train.save_final = True
    cfg.train.log_every = 1
    cfg.train.wandb_enabled = not args.no_wandb
    cfg.train.wandb_project = args.wandb_project
    cfg.train.wandb_run_name = f"{mode}-{directory.name}"
    # Recovery drills must never read or rotate the source run's remote checkpoints.
    cfg.train.remote_checkpoint_uri = None
    cfg.data.eval = None
    if args.save_every % max(1, cfg.data.num_workers) != 0:
        raise ValueError(
            "save-every must be a multiple of data.num_workers (or use --set data.num_workers=0)"
        )
    if cfg.data.num_workers > args.save_every:
        raise ValueError("use a save interval at least as large as the data worker count")
    if args.install:
        slurm = replace(slurm, install=args.install)
    if "'" in slurm.install or "\n" in slurm.install:
        raise ValueError(
            "install command cannot contain single quotes/newlines in the current renderer"
        )
    slurm = replace(slurm, log_dir=directory / "logs", max_requeues=3)
    for path in (slurm.env_file, slurm.container_image):
        _safe_path(path)
    command = accelerate_command(
        module="oplm.slurm.recovery.worker",
        gpus_per_node=slurm.gpus_per_node,
        args=f"--directory {directory}",
        mixed_precision=cfg.train.mixed_precision,
    )
    spec = JobSpec(
        name=f"recovery-{mode}",
        nodes=args.nodes,
        time_limit=args.time_limit,
        command=command,
        progress_dir=cfg.train.output_dir,
    )
    script = render_job(spec, slurm)
    if args.reservation:
        if not re.fullmatch(r"[\w.-]+", args.reservation, flags=re.ASCII):
            raise ValueError("invalid reservation name")
        script = script.replace(
            "#!/bin/bash\n", f"#!/bin/bash\n#SBATCH --reservation={args.reservation}\n", 1
        )
    # Keep online tracking explicit: inherited WANDB_MODE=offline cannot silently weaken a drill.
    script = script.replace(
        "export OMP_NUM_THREADS=1", "export OMP_NUM_THREADS=1\nexport WANDB_MODE=online"
    )
    manifest = dict(
        mode=mode,
        nodes=args.nodes,
        world_size=world_size,
        max_steps=args.max_steps,
        save_every=args.save_every,
        fail_step=args.fail_step,
        kill_rank=args.kill_rank,
        wandb_enabled=cfg.train.wandb_enabled,
        node_wait_seconds=args.node_wait_seconds,
        source_config=str(source),
    )
    directory.mkdir(parents=True, exist_ok=False)
    for child in ("logs", "events", "training"):
        (directory / child).mkdir()
    (directory / "train.yaml").write_text(serialize_config(cfg))
    (directory / "job.sbatch").write_text(script)
    write_json(directory / "drill.json", manifest)
    return manifest


def _command(argv: list[str], *, timeout: float = 30) -> str:
    return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=timeout).stdout


def _inject_node(directory: Path, manifest: dict[str, Any], executable: Path) -> None:
    job_id = manifest["job_id"]
    job = _command(["scontrol", "show", "job", "--oneliner", job_id])
    fields = dict(re.findall(r"(\w+)=(\S+)", job))
    nodes = _command(["scontrol", "show", "hostnames", fields["NodeList"]]).splitlines()
    batch_host = fields["BatchHost"]
    events = read_events(directory)
    observed_nodes = {e["node"] for e in events if e["kind"] == "start" and e["attempt"] == 0}
    if (
        fields.get("JobId") != job_id
        or fields.get("JobState") != "RUNNING"
        or fields.get("Restarts") != "0"
        or batch_host not in nodes
        or len(nodes) != manifest["nodes"]
        or set(nodes) != observed_nodes
        or any(e["kind"] == "start" and e["attempt"] != 0 for e in events)
    ):
        raise RuntimeError("scheduler allocation does not match the drill")
    target = (
        batch_host
        if manifest["mode"] == "batch_node_loss"
        else next(node for node in nodes if node != batch_host)
    )
    manifest.update(failed_node=target, allocation_before=nodes, batch_host=batch_host)
    write_json(directory / "drill.json", manifest)
    write_json(directory / "node-command.json", [str(executable), job_id, target])
    event = dict(kind="inject", attempt=0, rank=0, step=manifest["fail_step"], time=time.time())
    (directory / "events" / "controller.jsonl").write_text(json.dumps(event) + "\n")
    # The executable is an explicit operator-supplied adapter, never an interpolated shell command.
    with (directory / "node-command.log").open("w") as stream:
        subprocess.run(
            [str(executable), job_id, target],
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=manifest["node_wait_seconds"],
        )


def execute(directory: Path, manifest: dict[str, Any], args: argparse.Namespace) -> None:
    """Submit once, inject node loss when armed, and monitor through requeue to a terminal state."""
    if manifest.get("job_id"):
        raise ValueError("this drill has already been submitted; use --check or a fresh --out")
    executable = args.node_failure_command
    if manifest["mode"] in NODE_MODES:
        if executable is None or not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("node-loss --submit requires an executable --node-failure-command")
        executable = executable.resolve()
    # An interrupted/ambiguous sbatch response must not invite a duplicate submission.
    with (directory / "submission-started").open("x") as stream:
        stream.write("Do not resubmit this directory, even if sbatch lost its response.\n")
    try:
        manifest["job_id"] = submit_job(directory / "job.sbatch")
        write_json(directory / "drill.json", manifest)
        print(f"Submitted {manifest['job_id']}; evidence: {directory}", flush=True)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if (
                executable is not None
                and manifest["mode"] in NODE_MODES
                and not manifest.get("failed_node")
                and any(e["kind"] == "node_ready" for e in read_events(directory))
            ):
                _inject_node(directory, manifest, executable)
            query = running_job_ids([manifest["job_id"]])
            if not query.reachable:
                raise RuntimeError("scheduler unreachable; cannot verify recovery")
            if not query.ids:
                rows = _command(
                    [
                        "sacct",
                        "--noheader",
                        "--parsable2",
                        "--jobs",
                        manifest["job_id"],
                        "--format=JobIDRaw,State",
                    ]
                ).splitlines()
                states = [
                    row.split("|")[1].split()[0].rstrip("+")
                    for row in rows
                    if row.split("|")[0] == manifest["job_id"]
                ]
                if states and states[-1] in {
                    "COMPLETED",
                    "FAILED",
                    "CANCELLED",
                    "TIMEOUT",
                    "OUT_OF_MEMORY",
                    "NODE_FAIL",
                    "BOOT_FAIL",
                    "DEADLINE",
                }:
                    manifest["terminal_state"] = states[-1]
                    break
            time.sleep(10)
        else:
            raise TimeoutError("drill exceeded --timeout, including queue wait")
    except BaseException:
        # Only cancel the exact job created above; preserve all evidence for --check.
        try:
            if manifest.get("job_id"):
                _command(["scancel", manifest["job_id"]])
        finally:
            manifest["terminal_state"] = "INTERRUPTED"
            write_json(directory / "drill.json", manifest)
        raise
    write_json(directory / "drill.json", manifest)


def report(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Save local evidence checks plus an unsampled online W&B history check."""
    events = read_events(directory)
    logs = "\n".join(p.read_text(errors="replace") for p in (directory / "logs").glob("*.err"))
    manifest["guard_stopped"] = "crash loop -- not requeueing" in logs
    history = None
    online_error = None
    if manifest["wandb_enabled"]:
        identity = next(
            (
                e.get("wandb")
                for e in events
                if e["kind"] == "start" and e["rank"] == 0 and e.get("wandb")
            ),
            None,
        )
        if identity:
            try:
                import wandb

                path = "/".join(identity[key] for key in ("entity", "project", "id"))
                api = wandb.Api(timeout=30)
                # Allow the final upload to become queryable; --check can be repeated later.
                for attempt in range(3):
                    run = api.run(path)
                    history = list(run.scan_history(keys=["recovery/attempt", "recovery/step"]))
                    expected = {
                        (e["attempt"], e["step"])
                        for e in events
                        if e["kind"] == "log" and e["attempt"] == 1
                    }
                    if expected <= {(r["recovery/attempt"], r["recovery/step"]) for r in history}:
                        break
                    if attempt < 2:
                        time.sleep(10)
                        api.flush()
                if run.name != identity["name"]:
                    online_error = "online W&B run name differs from recorded identity"
            except Exception as exc:  # noqa: BLE001 -- missing online evidence is reported, never passed
                history = None
                online_error = f"{type(exc).__name__}: {exc}"
    result = assess(manifest, events, history)
    if online_error:
        result["online_error"] = online_error
        if result["status"] == "passed":
            result["status"] = "incomplete"
    write_json(directory / "result.json", result)
    return result


def main(mode: str) -> None:
    """Common CLI for the six separately runnable failure-injection scripts."""
    parser = argparse.ArgumentParser(
        description=f"Slurm recovery drill: {mode}. Prepare only by default."
    )
    parser.add_argument("--out", type=Path, required=True, help="new shared-filesystem directory")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preset")
    parser.add_argument(
        "--set", action="append", default=[], help="training config key=value override"
    )
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=16)
    parser.add_argument("--fail-step", type=int, default=24)
    parser.add_argument("--kill-rank", type=int, default=1)
    parser.add_argument("--time-limit", default="01:00:00")
    parser.add_argument(
        "--timeout",
        type=float,
        default=7200,
        help="controller deadline in seconds, including queue wait",
    )
    parser.add_argument("--node-wait-seconds", type=int, default=300)
    parser.add_argument(
        "--node-failure-command",
        type=Path,
        help="executable accepting JOB_ID NODE; node tests only",
    )
    parser.add_argument("--install", help="container install command for this checkout/version")
    parser.add_argument("--reservation")
    parser.add_argument("--wandb-project", default="oplm-recovery-tests")
    parser.add_argument(
        "--no-wandb", action="store_true", help="local checks only; result remains incomplete"
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--submit", action="store_true")
    action.add_argument(
        "--check", action="store_true", help="recheck evidence without submitting or injecting"
    )
    args = parser.parse_args()
    directory = args.out.resolve()
    try:
        if args.timeout <= 0:
            raise ValueError("timeout must be positive")
        if (directory / "drill.json").exists():
            manifest = json.loads((directory / "drill.json").read_text())
            if manifest["mode"] != mode or args.config is not None:
                raise ValueError("existing drill: use its matching module and omit --config")
            if not (args.submit or args.check):
                raise ValueError("drill already prepared; use --submit or --check")
        elif args.check:
            raise ValueError("no prepared drill at --out")
        else:
            manifest = prepare(mode, args)
        if args.submit:
            execute(directory, manifest, args)
        if args.submit or args.check:
            result = report(directory, manifest)
            print(json.dumps(result, indent=2))
            raise SystemExit(
                0 if result["status"] == "passed" else 1 if result["status"] == "failed" else 2
            )
        print(
            f"Prepared {directory / 'job.sbatch'}; inspect it, "
            f"then rerun with --out {directory} --submit"
        )
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")

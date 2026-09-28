"""Evaluate recorded evidence without mistaking an ordinary finish for recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def assess(
    manifest: dict[str, Any],
    events: list[dict[str, Any]],
    history: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Check rank agreement, restored state, node replacement, and online W&B history.

    Args:
        manifest: Prepared drill settings and observed scheduler outcome.
        events: Worker observations, including every rank's restored state.
        history: Unsampled online W&B history, or None when unavailable/disabled.

    Returns:
        A JSON-serializable report. Missing online evidence never yields a full pass.
    """
    errors: list[str] = []
    starts = [e for e in events if e["kind"] == "start"]
    injections = [e for e in events if e["kind"] == "inject"]
    attempts = {e["attempt"] for e in starts}
    if attempts != {0, 1}:
        errors.append(f"expected exactly attempts 0 and 1, observed {sorted(attempts)}")
    expected_injections = {0, 1} if manifest["mode"] == "crash_loop" else {0}
    if {e["attempt"] for e in injections} != expected_injections:
        errors.append("missing or unexpected failure injection")
    for attempt in (0, 1):
        ranks = {e["rank"] for e in starts if e["attempt"] == attempt}
        if ranks != set(range(manifest["world_size"])):
            errors.append(f"attempt {attempt}: not all ranks recorded startup")
        nodes = {e["node"] for e in starts if e["attempt"] == attempt}
        if len(nodes) != manifest["nodes"]:
            errors.append(f"attempt {attempt}: wrong node count")
    for event in starts:
        if event["world_size"] != manifest["world_size"]:
            errors.append("world size changed")
    batches = {e["global_batch"] for e in starts}
    if len(batches) != 1:
        errors.append("global batch changed")

    initial_injection = next((e for e in injections if e["attempt"] == 0), None)
    commits = [e for e in events if e["kind"] == "commit" and e["attempt"] == 0]
    if manifest["mode"] != "graceful_drain" and initial_injection is not None:
        # Rank clocks can drift; checkpoint step order is the relevant invariant.
        commits = [e for e in commits if e["step"] < initial_injection["step"]]
    checkpoint_step = max((e["step"] for e in commits), default=None)
    if checkpoint_step is None:
        errors.append("no committed checkpoint before restart")
    for event in starts:
        if event["attempt"] != 1:
            continue
        if event["step"] != checkpoint_step or Path(event.get("resume") or "").name != (
            f"checkpoint-{checkpoint_step}"
        ):
            errors.append(f"rank {event['rank']}: did not resume newest committed checkpoint")
        saved = next(
            (
                e
                for e in events
                if e["kind"] == "save"
                and e["attempt"] == 0
                and e["rank"] == event["rank"]
                and e["step"] == checkpoint_step
            ),
            None,
        )
        if saved is None:
            errors.append(f"rank {event['rank']}: no checkpoint snapshot to compare")
        else:
            for key in ("epoch", "samples", "tokens", "batches", "lr", "global_batch"):
                if event[key] != saved[key]:
                    errors.append(f"rank {event['rank']}: restored {key} differs from checkpoint")

    if manifest["mode"] == "checkpoint_write":
        if initial_injection is None or not initial_injection.get("shards_written"):
            errors.append("no evidence of real shard writes before interrupted commit")
        elif checkpoint_step is None or initial_injection["step"] <= checkpoint_step:
            errors.append("interrupted checkpoint was selected for resume")
    if manifest["mode"] == "graceful_drain" and (
        initial_injection is None
        or checkpoint_step is None
        or checkpoint_step <= initial_injection["step"]
    ):
        errors.append("drain did not commit a fresh checkpoint after the signal")
    if manifest["mode"] in {"worker_node_loss", "batch_node_loss"}:
        old = {e["node"] for e in starts if e["attempt"] == 0}
        new = {e["node"] for e in starts if e["attempt"] == 1}
        failed = manifest.get("failed_node")
        if failed not in old or failed in new or not new - old:
            errors.append("failed node was not replaced by a different node")
        if len(new) != manifest["nodes"]:
            errors.append("replacement allocation has the wrong node count")

    if manifest["mode"] == "crash_loop":
        if manifest.get("terminal_state") != "FAILED" or not manifest.get("guard_stopped"):
            errors.append("job did not stop with FAILED and the no-progress guard diagnosis")
        if any(e["kind"] == "log" and e["attempt"] == 1 for e in events):
            errors.append("second crash advanced training instead of testing no progress")
    elif manifest.get("terminal_state") != "COMPLETED" or not any(
        e["kind"] == "complete" and e["attempt"] == 1 and e["step"] == manifest["max_steps"]
        for e in events
    ):
        errors.append("restarted job did not complete its planned training steps")

    resumed_logs = [
        e for e in events if e["kind"] == "log" and e["attempt"] == 1 and e["rank"] == 0
    ]
    if (
        manifest["mode"] != "crash_loop"
        and checkpoint_step is not None
        and {e["step"] for e in resumed_logs}
        != set(range(checkpoint_step + 1, manifest["max_steps"] + 1))
    ):
        errors.append("local observations do not cover every resumed training step")

    identities = [e.get("wandb") for e in starts if e["rank"] == 0]
    if manifest["wandb_enabled"]:
        if (
            not identities
            or any(not identity for identity in identities)
            or any(identity != identities[0] for identity in identities)
        ):
            errors.append("W&B entity/project/id/name missing or changed")
        if history is not None:
            observed = {(e.get("recovery/attempt"), e.get("recovery/step")) for e in history}
            expected = {
                (e["attempt"], e["step"])
                for e in events
                if e["kind"] == "log" and e["rank"] == 0 and e["attempt"] == 1
            }
            missing = sorted(expected - observed)
            if missing:
                errors.append(f"W&B dropped resumed metrics at {missing}")
    online_checked = manifest["wandb_enabled"] and history is not None
    status = "failed" if errors else ("passed" if online_checked else "incomplete")
    resumed = [e["time"] for e in starts if e["attempt"] == 1]
    return {
        "status": status,
        "errors": errors,
        "checkpoint_step": checkpoint_step,
        "wandb": "checked" if online_checked else "not checked",
        "restart_seconds": min(resumed) - initial_injection["time"]
        if resumed and initial_injection
        else None,
        "first_resumed_step_seconds": min(e["time"] for e in resumed_logs)
        - initial_injection["time"]
        if resumed_logs and initial_injection
        else None,
        "nodes_before": sorted({e["node"] for e in starts if e["attempt"] == 0}),
        "nodes_after": sorted({e["node"] for e in starts if e["attempt"] == 1}),
    }

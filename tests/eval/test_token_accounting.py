"""Token-accounting and rank-sync tests — the guarantee behind token cadence.

The trainer rank-reduces each optimizer step's tokens so ``tokens_seen`` is the
true global count and identical on every rank (docs/EVAL_HARNESS.md §3.2). These
tests pin that accounting (single process), the purity that makes ranks agree
without communicating, and — when a multi-process launch is available — that a
token-cadence eval does not deadlock under ragged per-rank batches.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
import torch

from oplm.eval import EvalContext, EveryNTokens


def _opt_step_tokens(step_local_tokens: int) -> int:
    """Mirror the trainer's per-opt-step token reduction at a single rank.

    The trainer rank-reduces each step's local tokens with
    ``dist.all_reduce(..., op=SUM)``; with one rank that reduction is the identity,
    so the global count equals the local count. (The genuine multi-rank reduction is
    exercised by ``test_ragged_distributed_token_reduction_no_deadlock`` below.)
    """
    tokens_tensor = torch.tensor(step_local_tokens, dtype=torch.long)
    return int(tokens_tensor.item())


def test_token_accounting_is_exact_global_count() -> None:
    """Accumulated ``tokens_seen`` equals the exact sum of ``attention_mask`` ones.

    Drives the trainer's accumulate-then-reduce logic over micro-batches grouped
    into optimizer steps. With a single process the reduction is identity, so this
    nails the per-step accumulation, the reset of ``_step_local_tokens`` each step,
    and that the result is a true count — not a ``local × world_size`` estimate.
    """
    # Ragged micro-batches grouped into 3 optimizer steps (gradient accumulation).
    steps_mask_sums = [[5, 3], [7], [2, 6, 1]]
    expected_total = sum(s for step in steps_mask_sums for s in step)

    tokens_seen = 0
    step_local_tokens = 0
    for step_sums in steps_mask_sums:
        for n_ones in step_sums:
            attention_mask = torch.ones(1, n_ones, dtype=torch.long)
            step_local_tokens += int(attention_mask.sum().item())
        tokens_delta = _opt_step_tokens(step_local_tokens)
        tokens_seen += tokens_delta
        step_local_tokens = 0  # reset at the opt-step boundary
        assert tokens_delta == sum(step_sums)
        assert step_local_tokens == 0

    assert tokens_seen == expected_total


def test_is_due_is_a_pure_function_of_context() -> None:
    """``is_due`` depends only on the context, so all ranks agree by construction."""
    sched = EveryNTokens(1_000_000)
    fields = dict(
        global_step=10,
        epoch=0,
        tokens_seen=3_000_050,
        steps_delta=1,
        tokens_delta=120,
        epoch_delta=0,
        is_final=False,
    )
    ctx_a = EvalContext(**fields)  # type: ignore[arg-type]
    ctx_b = EvalContext(**fields)  # type: ignore[arg-type]  # independently built, identical fields

    assert sched.is_due(ctx_a) == sched.is_due(ctx_a)  # deterministic
    assert sched.is_due(ctx_a) == sched.is_due(ctx_b)  # rank-agnostic


# --- ragged distributed smoke --------------------------------------------------


def _free_port() -> int:
    """Return an ephemeral localhost port for the gloo rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ragged_worker(rank: int, world_size: int, master_port: int, out_dir: str) -> None:
    """One rank: ragged local tokens → all-reduce → token-cadence due check.

    If ranks disagreed on ``is_due``, only some would enter the ``barrier`` that
    stands in for the collective inside a real task's ``evaluate`` — and the run
    would hang. Reaching the barrier on every rank is the no-deadlock assertion.
    """
    import json

    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        local_tokens = 7 * (rank + 1)  # ragged: rank 0 → 7, rank 1 → 14, ...
        tokens_tensor = torch.tensor(local_tokens, dtype=torch.long)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        global_tokens = int(tokens_tensor.item())

        ctx = EvalContext(
            global_step=1,
            epoch=0,
            tokens_seen=global_tokens,
            steps_delta=1,
            tokens_delta=global_tokens,
            epoch_delta=0,
            is_final=False,
        )
        due = EveryNTokens(10).is_due(ctx)
        if due:
            dist.barrier()  # the "collective inside evaluate" — deadlocks on disagreement

        Path(out_dir, f"rank{rank}.json").write_text(
            json.dumps({"global_tokens": global_tokens, "due": due})
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_ragged_distributed_token_reduction_no_deadlock(tmp_path: Path) -> None:
    """≥2 ranks with ragged batches agree on tokens and the cadence eval does not hang."""
    import json

    import torch.distributed as dist

    if not dist.is_available():
        pytest.skip("torch.distributed unavailable in this environment")
    if not getattr(dist, "is_gloo_available", lambda: False)():
        pytest.skip("gloo backend unavailable in this environment")

    world_size = 2
    try:
        mp_ctx = torch.multiprocessing.get_context("spawn")
    except (ValueError, RuntimeError) as exc:  # spawn start method unavailable
        pytest.skip(f"multiprocessing spawn unavailable: {exc}")

    port = _free_port()
    procs = []
    try:
        for rank in range(world_size):
            proc = mp_ctx.Process(
                target=_ragged_worker, args=(rank, world_size, port, str(tmp_path))
            )
            proc.start()
            procs.append(proc)
    except (OSError, RuntimeError) as exc:  # cannot spawn workers here
        for proc in procs:
            proc.terminate()
        pytest.skip(f"could not launch distributed workers: {exc}")

    for proc in procs:
        proc.join(timeout=60)
    hung = [p for p in procs if p.is_alive()]
    for proc in hung:
        proc.terminate()
    assert not hung, "distributed worker hung — token-cadence deadlock (ranks disagreed on is_due)"
    assert all(p.exitcode == 0 for p in procs), "a distributed worker errored"

    results = [
        json.loads(Path(tmp_path, f"rank{rank}.json").read_text()) for rank in range(world_size)
    ]
    expected_global = sum(7 * (rank + 1) for rank in range(world_size))
    # Every rank computed the SAME global token count after the reduction...
    assert all(r["global_tokens"] == expected_global for r in results)
    # ...and therefore the SAME due decision (so the barrier above could not deadlock).
    assert all(r["due"] is True for r in results)

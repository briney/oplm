"""DDPOptimizer gating for selective activation checkpointing + ``torch.compile``.

Under DDP+compile the default ``torch._dynamo.config.optimize_ddp`` ("ddp_optimizer")
splits the compiled graph at gradient-bucket boundaries, fragmenting each block's
activation-checkpoint HOP so AOT autograd drops the SAC ``MUST_SAVE`` policy and
``selective`` silently collapses to full recompute. The trainer disables that
graph-splitting for ``selective`` only; ``full``/``none`` keep the comm/compute
overlap. These tests cover both the wiring (CPU, flag check) and the physical
memory effect (multi-GPU, peak ordering).
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
import torch

from tests.training.conftest import configure_accelerator_device, tiny_train_cfg

# ---------------------------------------------------------------------------
# CPU wiring guards: the trainer flips optimize_ddp only for selective + compile
# ---------------------------------------------------------------------------


def test_compile_selective_disables_ddp_optimizer(
    tmp_path: Path,
    training_parquet: Path,
    reset_dynamo: None,
    restore_optimize_ddp: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compile=True + mode='selective': the trainer sets optimize_ddp=False.

    CPU-safe — ``torch.compile`` wrapping is lazy, so ``Trainer.__init__`` flips the
    flag without triggering a real compilation. No ``train()`` call is needed.
    """
    import torch._dynamo

    from oplm.training.trainer import Trainer

    configure_accelerator_device("cpu", monkeypatch)

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=1,
        compile=True,
        gradient_checkpointing=True,
        gradient_checkpointing_mode="selective",
    )
    Trainer(cfg)

    assert torch._dynamo.config.optimize_ddp is False


def test_compile_full_mode_leaves_ddp_optimizer_default(
    tmp_path: Path,
    training_parquet: Path,
    reset_dynamo: None,
    restore_optimize_ddp: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compile=True + mode='full': optimize_ddp is left at its default (overlap kept)."""
    import torch._dynamo

    from oplm.training.trainer import Trainer

    configure_accelerator_device("cpu", monkeypatch)

    default = torch._dynamo.config.optimize_ddp
    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=1,
        compile=True,
        gradient_checkpointing=True,
        gradient_checkpointing_mode="full",
    )
    Trainer(cfg)

    assert torch._dynamo.config.optimize_ddp == default


# ---------------------------------------------------------------------------
# Multi-GPU peak-memory ordering under DDP + compile (the physical fix)
# ---------------------------------------------------------------------------

_PEAK_PROBE_DIMS = dict(
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    max_position_embeddings=512,
)
_PEAK_PROBE_BATCH = 8
_PEAK_PROBE_SEQ = 512
_VOCAB = 33


def _free_port() -> int:
    """Return an ephemeral localhost port for the NCCL rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ddp_sac_peak_worker(
    rank: int, world_size: int, master_port: int, mode: str, out_dir: str
) -> None:
    """One DDP rank: build → compile → measure backward peak under ``mode``.

    Replicates the trainer's gating (``optimize_ddp=False`` for ``selective``) and
    compiles the DDP-wrapped model — the exact configuration that triggers
    DDPOptimizer graph-splitting. The measured backward peak is written per rank so
    the parent can assert the selective/full ordering.
    """
    import json

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from oplm.model import OplmConfig, OplmForMaskedLM

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        # Mirror the trainer: SAC is incompatible with the default DDPOptimizer.
        # Use `from torch import _dynamo` so `torch` isn't rebound as a function-local.
        if mode == "selective":
            from torch import _dynamo

            _dynamo.config.optimize_ddp = False

        config = OplmConfig(
            **_PEAK_PROBE_DIMS,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        model = OplmForMaskedLM(config).cuda(rank).train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"mode": mode})
        ddp = DDP(model, device_ids=[rank])
        compiled = torch.compile(ddp, dynamic=True)

        torch.manual_seed(123 + rank)
        input_ids = torch.randint(4, _VOCAB, (_PEAK_PROBE_BATCH, _PEAK_PROBE_SEQ), device=rank)
        attention_mask = torch.ones(
            _PEAK_PROBE_BATCH, _PEAK_PROBE_SEQ, dtype=torch.long, device=rank
        )
        labels = torch.randint(0, _VOCAB, (_PEAK_PROBE_BATCH, _PEAK_PROBE_SEQ), device=rank)

        def _step() -> None:
            out = compiled(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out.loss.backward()

        _step()  # warm up: trigger (re)compilation before the measured step
        torch.cuda.synchronize(rank)
        torch.cuda.reset_peak_memory_stats(rank)
        _step()
        torch.cuda.synchronize(rank)
        peak = int(torch.cuda.max_memory_allocated(rank))

        Path(out_dir, f"rank{rank}_{mode}.json").write_text(json.dumps({"peak": peak}))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_ddp_peak(world_size: int, mode: str, out_dir: Path) -> int:
    """Spawn ``world_size`` ranks for ``mode`` and return rank-0's measured peak."""
    import json

    mp_ctx = torch.multiprocessing.get_context("spawn")
    port = _free_port()
    procs = []
    try:
        for rank in range(world_size):
            proc = mp_ctx.Process(
                target=_ddp_sac_peak_worker, args=(rank, world_size, port, mode, str(out_dir))
            )
            proc.start()
            procs.append(proc)
    except (OSError, RuntimeError) as exc:  # cannot spawn workers here
        for proc in procs:
            proc.terminate()
        pytest.skip(f"could not launch distributed workers: {exc}")

    for proc in procs:
        proc.join(timeout=600)
    alive = [p for p in procs if p.is_alive()]
    for proc in alive:
        proc.terminate()
    assert not alive, f"a distributed {mode} worker hung"
    assert all(p.exitcode == 0 for p in procs), f"a distributed {mode} worker errored"

    return int(json.loads(Path(out_dir, f"rank0_{mode}.json").read_text())["peak"])


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="DDP peak-memory probe needs >=2 CUDA devices",
)
def test_ddp_compile_selective_peak_exceeds_full(tmp_path: Path) -> None:
    """Under DDP+compile, selective peak must strictly exceed full peak.

    If DDPOptimizer fragmented the SAC HOP, selective would collapse to full
    recompute and the two peaks would match. A clearly higher selective peak proves
    the per-block matmul/SDPA outputs stay resident — i.e. SAC survives DDP+compile.
    The CPU guard tests above confirm the trainer actually flips the flag this relies on.
    """
    import torch.distributed as dist

    if not dist.is_available() or not getattr(dist, "is_nccl_available", lambda: False)():
        pytest.skip("NCCL backend unavailable in this environment")

    world_size = 2
    peak_selective = _run_ddp_peak(world_size, "selective", tmp_path)
    peak_full = _run_ddp_peak(world_size, "full", tmp_path)

    # Collapse-to-full would make these ~equal; require a clear margin above allocator
    # noise. SAC keeps every block's matmul/SDPA activations resident, so the gap is large.
    assert peak_selective > peak_full * 1.1, (
        f"selective peak {peak_selective} not clearly above full peak {peak_full} — "
        "SAC likely collapsed to full recompute under DDPOptimizer"
    )

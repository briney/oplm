"""HSDP end-to-end pilot (Task 5.1): ``train.parallelism=hsdp`` over an FSDP2 mesh.

``parallelism="hsdp"`` builds a 2-D ``init_device_mesh`` (``("replicate", "shard")``,
sized from ``WORLD_SIZE``/``LOCAL_WORLD_SIZE``) and applies ``fully_shard`` per
``OplmBlock`` plus the root, instead of letting Accelerate wrap the model in DDP. The
checkpoint format is unchanged -- ``get_state_dict``/``set_state_dict`` (Task 2.1) are
parallelism-agnostic -- so a checkpoint written by a sharded run must load into a plain
``ddp`` run and vice versa. This module is that gate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import safetensors.torch
import torch

from tests.training.conftest import configure_accelerator_device, tiny_train_cfg

pytestmark = pytest.mark.slow

_HSDP_STEPS = 3
_HSDP_RESUME_STEPS = 5
_DDP_RESUME_STEPS = 7


def _child_env() -> dict[str, str]:
    """Environment for the subprocess ranks: CPU-only, repo root on ``PYTHONPATH``.

    Mirrors ``test_e2e_dcp.py``'s reshard worker env. ``CUDA_VISIBLE_DEVICES=""`` is
    required (not merely tidy): Accelerate's ``wait_for_everyone`` calls
    ``torch.distributed.barrier(device_ids=[local_process_index])`` even for a gloo
    process group, which maps the local index onto a CUDA ordinal and fails once it
    exceeds the visible GPU count. The worker imports ``tests.training.conftest``, so the
    repo root -- not just ``src/`` -- must be on the child's ``PYTHONPATH``.
    """
    repo_root = Path(__file__).resolve().parents[2]
    existing = os.environ.get("PYTHONPATH", "")
    child_pythonpath = os.pathsep.join(
        p for p in (str(repo_root), str(repo_root / "src"), existing) if p
    )
    return {
        **os.environ,
        "ACCELERATE_USE_CPU": "true",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": child_pythonpath,
    }


def _run_hsdp_pilot(
    run_dir: Path,
    training_parquet: Path,
    out_dir: Path,
    *,
    max_steps: int,
    auto_resume: bool,
    mixed_precision: str,
) -> list[dict[str, object]]:
    """Launch the 2-rank HSDP worker and return both ranks' recorded payloads."""
    worker = Path(__file__).with_name("_hsdp_worker.py")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            str(worker),
            str(run_dir),
            str(training_parquet),
            str(out_dir),
            str(max_steps),
            "true" if auto_resume else "false",
            mixed_precision,
        ],
        check=True,
        timeout=600,
        env=_child_env(),
    )
    return [json.loads((out_dir / f"rank{rank}.json").read_text()) for rank in (0, 1)]


@pytest.mark.parametrize("mixed_precision", ["no", "bf16"])
def test_hsdp_pilot_trains_resumes_and_its_checkpoint_loads_under_ddp(
    mixed_precision: str,
    training_parquet: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 2-rank HSDP run trains, auto-resumes itself, and hands its checkpoint to ddp.

    Three phases:

    1. 2 CPU/gloo ranks with ``parallelism=hsdp`` train ``_HSDP_STEPS`` steps and commit
       ``checkpoint-3``. Both ranks confirm the model is genuinely FSDP2-sharded (params
       are ``DTensor``s over a ``("replicate", "shard")`` mesh) -- without that check this
       whole test would pass unchanged under plain DDP.
    2. A second 2-rank launch with ``auto_resume=True`` picks up ``checkpoint-3``,
       trains to ``_HSDP_RESUME_STEPS``, and commits ``checkpoint-5``. Same world size and
       layout, so the Phase-3 data cursor resumes normally (no ``resume_data_position``
       opt-out needed).
    3. This process (single rank, default ``parallelism="ddp"``) auto-resumes
       ``checkpoint-5``, restoring the exact gathered weight the sharded run held, and
       trains 2 more steps. This is the DDP<->HSDP interop claim: the on-disk format
       carries no parallelism assumption.

    The world-size change in phase 3 (2 -> 1) requires ``resume_data_position=False``,
    exactly as the Phase-3 cursor layout guard demands -- the *parallelism* change is
    irrelevant to data striping (``ShardedProteinDataset`` stripes by rank count, which
    HSDP does not alter), only the world-size change is.

    Parametrized over ``mixed_precision``: ``"no"`` (pure fp32) and ``"bf16"``, which is
    the production default and therefore the configuration a real HSDP run will actually
    use -- it exercises ``MixedPrecisionPolicy`` (bf16 all-gather, fp32 reduce-scatter)
    rather than ``fully_shard``'s default policy. Phase 3 deliberately resumes in fp32
    regardless: the checkpoint stores fp32 master weights either way, so the bit-exact
    weight comparison also proves an HSDP-bf16 checkpoint restores exactly into an fp32
    DDP run.
    """
    from oplm.training.trainer import Trainer

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Phase 1: fresh 2-rank HSDP run.
    results = _run_hsdp_pilot(
        run_dir,
        training_parquet,
        out_dir,
        max_steps=_HSDP_STEPS,
        auto_resume=False,
        mixed_precision=mixed_precision,
    )
    for result in results:
        assert result["resumed_from_step"] == 0
        assert result["global_step"] == _HSDP_STEPS
        assert result["is_dtensor"], "model params are not DTensors -- fully_shard did not apply"
        assert result["mesh_shape"] == [1, 2]
        assert result["mesh_dim_names"] == ["replicate", "shard"]
        assert result["placements"] == ["Replicate", "Shard"]
        # The MixedPrecisionPolicy really reached fully_shard (bf16 compute dtype), and
        # the sharded parameters themselves stay fp32 master weights either way.
        expected_param_dtype = "torch.bfloat16" if mixed_precision == "bf16" else None
        assert result["mp_param_dtype"] == expected_param_dtype
        assert result["master_weight_dtype"] == "torch.float32"

    committed = run_dir / f"checkpoint-{_HSDP_STEPS}"
    assert committed.is_dir()
    assert (committed / ".metadata").exists()
    assert (committed / "rng_state_0.pt").exists()
    assert (committed / "rng_state_1.pt").exists()
    # The HF export must survive sharding too: under FSDP2 the raw state dict holds
    # DTensors, which safetensors cannot serialize -- save_checkpoint has to gather.
    assert (committed / "hf" / "model.safetensors").exists()

    # Phase 2: same topology, auto-resume, train to _HSDP_RESUME_STEPS.
    results = _run_hsdp_pilot(
        run_dir,
        training_parquet,
        out_dir,
        max_steps=_HSDP_RESUME_STEPS,
        auto_resume=True,
        mixed_precision=mixed_precision,
    )
    for result in results:
        assert result["resumed_from_step"] == _HSDP_STEPS
        assert result["global_step"] == _HSDP_RESUME_STEPS
    resumed_checkpoint = run_dir / f"checkpoint-{_HSDP_RESUME_STEPS}"
    assert resumed_checkpoint.is_dir()

    # The sharded run's own gathered HF export is the reference. It cannot come from the
    # worker process: Accelerate's end_training (in Trainer.train's finally) destroys the
    # process group, so a DTensor.full_tensor() gather after training would fail -- and
    # taking the reference from the export instead additionally proves the gathered
    # safetensors and the DCP shards agree bit-for-bit.
    reference_weight = safetensors.torch.load_file(resumed_checkpoint / "hf" / "model.safetensors")[
        "lm_head.decoder.bias"
    ]

    # Phase 3: single-process DDP resume of the sharded run's checkpoint.
    configure_accelerator_device("cpu", monkeypatch)
    resume_cfg = tiny_train_cfg(
        run_dir,
        training_parquet,
        max_steps=_DDP_RESUME_STEPS,
        save_every=_DDP_RESUME_STEPS,
        auto_resume=True,
        log_every=1,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        # World size 2 -> 1 changes the data striping (not the parallelism change --
        # see this test's docstring), which the cursor layout guard rejects by design.
        resume_data_position=False,
    )
    assert resume_cfg.train.parallelism == "ddp"

    resumed = Trainer(resume_cfg, callbacks=[])
    assert resumed.global_step == _HSDP_RESUME_STEPS

    resumed_weight = resumed.accelerator.unwrap_model(resumed.model).lm_head.decoder.bias.detach()
    # Bit-exact, not merely close: nothing trains between the sharded run's commit and
    # this load, so any drift would be a real precision bug in the gather/load path.
    assert torch.equal(resumed_weight, reference_weight)

    resumed.train()
    assert resumed.global_step == _DDP_RESUME_STEPS


def test_hsdp_with_configured_eval_refuses_at_trainer_init(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The eval-deadlock guard is wired into ``Trainer.__init__``, not only ``load_config``.

    A config built directly (tests, sweeps, notebooks) never passes through
    ``load_config``, so the trainer re-checks it -- before the Accelerator exists, so the
    refusal costs nothing and cannot itself desynchronize ranks.
    """
    from oplm.training.trainer import Trainer

    configure_accelerator_device("cpu", monkeypatch)
    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=1,
        parallelism="hsdp",
        eval={"heldout": str(training_parquet)},
    )
    with pytest.raises(ValueError, match="data.eval"):
        Trainer(cfg, callbacks=[])


def test_hsdp_on_a_single_process_refuses_with_an_actionable_error(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``parallelism=hsdp`` on one process raises rather than silently sharding nothing."""
    from oplm.training.trainer import Trainer

    configure_accelerator_device("cpu", monkeypatch)
    cfg = tiny_train_cfg(tmp_path, training_parquet, max_steps=1, parallelism="hsdp")
    with pytest.raises(ValueError, match="ddp"):
        Trainer(cfg, callbacks=[])

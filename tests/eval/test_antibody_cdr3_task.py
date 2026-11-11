"""Antibody CDR3 task: batched, padded scoring matches a one-residue-per-forward reference.

Uses the committed real fixture ``tests/data/fixtures/antibody/eval_gte2_head.parquet``
(10 chains from ``brineylab/oplm-antibody-sequences`` ``eval_gte2``: 4 IGH, 3 IGK, 3 IGL).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest
import torch

from oplm.config import DataConfig, EvalDatasetEntry, OplmConfig, ScheduleSpec, TrainConfig
from oplm.data import get_tokenizer
from oplm.data.tokenizer import mask_token_id
from oplm.eval.tasks.antibody_cdr3 import AntibodyCdr3EvalTask
from oplm.model import OplmConfig as OplmModelConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.model import OplmForMaskedLM

_FIXTURE = Path(__file__).resolve().parents[1] / "data/fixtures/antibody/eval_gte2_head.parquet"


def _make_task(**extra: object) -> AntibodyCdr3EvalTask:
    cfg = OplmConfig(
        model=OplmModelConfig(max_position_embeddings=256),
        train=TrainConfig(wandb_enabled=False),
        data=DataConfig(num_workers=0, pin_memory=False),
    )
    entry = EvalDatasetEntry(
        name="ab", path=str(_FIXTURE), type="antibody_cdr3",
        schedule=ScheduleSpec("steps", 1), extra=extra,
    )  # fmt: skip
    return AntibodyCdr3EvalTask(entry, cfg)


def _reference(model: OplmForMaskedLM) -> dict[str, float]:
    """One unpadded forward per CDR3 residue; loss/accuracy per heavy/light/all."""
    tok = get_tokenizer()
    sums = {g: [0.0, 0, 0] for g in ("heavy", "light")}
    for row in pq.read_table(_FIXTURE).to_pylist():
        ids = torch.tensor(tok.encode(row["sequence"]))
        group = "heavy" if row["locus"] == "IGH" else "light"
        for i, digit in enumerate(row["cdr_mask_aa"]):
            if digit != "3":
                continue
            masked = ids.clone()
            masked[i + 1] = mask_token_id(tok)
            with torch.no_grad():
                logits = model(input_ids=masked[None])["logits"][0, i + 1]
            sums[group][0] += torch.nn.functional.cross_entropy(logits, ids[i + 1]).item()
            sums[group][1] += int(logits.argmax().item() == ids[i + 1].item())
            sums[group][2] += 1
    sums["all"] = [a + b for a, b in zip(sums["heavy"], sums["light"], strict=True)]
    return {
        f"{m}_{g}": (s[0] / s[2] if m == "loss" else s[1] / s[2])
        for g, s in sums.items()
        for m in ("loss", "accuracy")
    }


def test_matches_reference(make_model: Callable[..., OplmForMaskedLM]) -> None:
    """Mixed-length batches that split chains (batch 7) reproduce the reference."""
    from accelerate import Accelerator

    torch.manual_seed(0)
    model = make_model(max_position_embeddings=256)
    metrics = _make_task(mask_batch_size=7).evaluate(model, Accelerator(cpu=True))

    expected = _reference(model)
    assert set(metrics) == {
        f"{m}_{g}" for m in ("loss", "accuracy", "perplexity") for g in ("heavy", "light", "all")
    }
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value, rel=1e-4, abs=1e-6), key
    assert metrics["perplexity_all"] == pytest.approx(math.exp(expected["loss_all"]), rel=1e-4)


def test_rejects_bad_mask_batch_size() -> None:
    with pytest.raises(ValueError, match="mask_batch_size"):
        _make_task(mask_batch_size=0)

"""TrainTask seam: MLMTask reproduces the pre-seam step; a custom task's metrics and
FLOP policy flow through the Trainer (docs/training.md, "Training tasks")."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn.functional as F

from oplm.config import DistillConfig, FoldRunConfig, load_config
from oplm.data.sequence.collate import MLMCollator
from oplm.data.tokenizer import get_tokenizer
from oplm.training.flops import estimate_flops_per_token
from oplm.training.task import DistillMLMTask, MLMTask, StepResult, TrainTask
from tests.training.conftest import (
    FullRecordingCallback,
    configure_accelerator_device,
    tiny_train_cfg,
)

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn


@pytest.mark.parametrize("key", ["loss", "mfu", "tokens_per_sec", "grad_norm"])
def test_step_result_rejects_reserved_metric_keys(key: str) -> None:
    with pytest.raises(ValueError, match="reserved"):
        StepResult(loss=torch.tensor(0.0), tokens=1, samples=1, metrics={key: 1.0})


def test_mlm_task_step_matches_direct_model_call(tmp_path: Path, training_parquet: Path) -> None:
    """MLMTask.step returns the model's own loss plus exact token/sample accounting."""
    cfg = tiny_train_cfg(tmp_path, training_parquet)
    task = MLMTask()
    assert isinstance(task, TrainTask)
    model = task.build_model(cfg, None)
    model.eval()  # no dropout: the two forward passes below must agree exactly
    batch = next(iter(task.build_dataloader(cfg)))

    result = task.step(model, batch)
    expected = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )["loss"]
    assert torch.equal(result.loss, expected)
    assert result.tokens == int(batch["attention_mask"].sum())
    assert result.samples == len(batch["input_ids"])
    assert result.metrics == {}
    assert task.flops_per_token(cfg) == estimate_flops_per_token(cfg.model)


class _AuxMetricTask(MLMTask):
    """MLMTask that also reports a per-micro-batch metric and declines to estimate FLOPs."""

    def step(self, model: nn.Module, batch: dict[str, Any]) -> StepResult:
        result = super().step(model, batch)
        result.metrics["aux"] = float(result.tokens)
        return result

    def flops_per_token(self, cfg: Any) -> int | None:
        return None


@pytest.mark.slow
def test_custom_task_metrics_and_flop_policy_flow_through_trainer(
    tmp_path: Path, training_parquet: Path
) -> None:
    from oplm.training.trainer import Trainer

    # Zero warmup and a peak so the throughput window fires and, given a FLOP estimate,
    # would emit achieved_tflops and mfu: their absence below is then the task's doing.
    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=4,
        log_every=2,
        batch_size=4,
        throughput_warmup_steps=0,
        peak_tflops=1.0,
    )
    callback = FullRecordingCallback()
    Trainer(cfg, callbacks=[callback], task=_AuxMetricTask()).train()

    assert callback.train_log_steps == [2, 4]
    for _step, metrics in callback.train_logs:
        assert math.isfinite(metrics["train/aux"])
        assert "train/tokens_per_sec" in metrics
        assert "train/step_time_s" in metrics
        assert "train/flops" not in metrics
        assert "train/achieved_tflops" not in metrics
        assert "train/mfu" not in metrics
    # train/aux is the window mean of a per-micro-batch token count: two optimizer
    # steps per log window at accumulation 1, so 2 * mean == the window's token delta.
    (_, first), (_, second) = callback.train_logs
    assert second["train/aux"] * 2 == pytest.approx(second["train/tokens"] - first["train/tokens"])


# --------------------------------------------------------------------------- #
# DistillMLMTask and Trainer integration
# --------------------------------------------------------------------------- #


class _FixedLogitsModel(torch.nn.Module):
    """Minimal stub returning a fixed logits tensor without an underlying transformer."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self._logits = logits

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        assert labels is None
        return {"logits": self._logits}


def _write_mixed_distill_parquet(path: Path) -> Path:
    rng = np.random.default_rng(0)
    seqs = [
        "ACDEFGHIKLMNPQRSTVWY",
        "MKTAYIAKQRQISFVKSHFSRQ",
        "GIVEQCCTSICSLYQLENYCN",
        "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    ]
    payloads = [
        rng.normal(size=(len(s), 20)).astype("<f2").tobytes() if i < 2 else None
        for i, s in enumerate(seqs)
    ]
    neff_sqrt_l = [6.0, 6.0, None, None]
    pq.write_table(
        pa.table(
            {
                "sequence_id": [f"s{i}" for i in range(len(seqs))],
                "sequence": seqs,
                "teacher_logits": pa.array(payloads, type=pa.binary()),
                "teacher_neff": pa.array([40.0, 40.0, None, None], type=pa.float32()),
                "teacher_neff_sqrt_l": pa.array(neff_sqrt_l, type=pa.float32()),
            }
        ),
        path,
    )
    return path


def test_distill_task_alpha_zero_matches_mlm_loss_and_grads(tmp_path: Path) -> None:
    cfg = tiny_train_cfg(
        tmp_path,
        _write_mixed_distill_parquet(tmp_path / "m.parquet"),
        batch_size=4,
        mask_prob=0.5,
        drop_last=False,
    )
    cfg.distill = DistillConfig(alpha=0.0, temperature=1.5, min_neff_sqrt_l=2.0)
    mlm_task = MLMTask()
    distill_task = DistillMLMTask(cfg.distill)
    assert isinstance(distill_task, MLMTask)
    assert isinstance(distill_task, TrainTask)

    model_a = mlm_task.build_model(cfg, None)
    model_b = distill_task.build_model(cfg, None)
    model_b.load_state_dict(model_a.state_dict())
    model_a.eval()
    model_b.eval()

    batch = next(iter(distill_task.build_dataloader(cfg)))
    r_mlm = mlm_task.step(model_a, batch)
    r_dist = distill_task.step(model_b, batch)

    assert torch.allclose(r_dist.loss, r_mlm.loss, atol=1e-6)
    r_mlm.loss.backward()
    r_dist.loss.backward()
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters(), strict=True):
        if p_a.grad is not None:
            assert p_b.grad is not None
            assert torch.allclose(p_b.grad, p_a.grad, atol=1e-5)
    assert r_dist.metrics["loss_ce"] == pytest.approx(r_mlm.loss.item(), abs=1e-6)
    assert "loss_kl" in r_dist.metrics
    assert math.isfinite(r_dist.metrics["loss_kl"])
    assert 0.0 < r_dist.metrics["distill_frac"] < 1.0


def test_distill_task_channel_order_maps_teacher_alphabet_to_tokenizer_ids() -> None:
    # 'A' is channel 0 in ARNDCEQGHILKMFPSTWYV, token id 5 in OplmTokenizerFast ('L' is id 4).
    tok = get_tokenizer()
    assert tok.convert_tokens_to_ids("L") == 4
    assert tok.convert_tokens_to_ids("A") == 5

    logits = torch.zeros((1, 3, tok.vocab_size), dtype=torch.float32, requires_grad=True)
    teacher = torch.zeros((1, 3, 20), dtype=torch.float16)
    teacher[0, 1, 0] = 10.0  # strong spike on channel 0 ('A')
    batch = {
        "input_ids": torch.tensor([[0, 32, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": torch.tensor([[-100, 5, -100]]),
        "teacher_logits": teacher,
        "teacher_mask": torch.tensor([[False, True, False]]),
    }
    stub = _FixedLogitsModel(logits)
    result = DistillMLMTask(DistillConfig(alpha=1.0, temperature=1.0)).step(stub, batch)
    result.loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 1, 5].item() < -0.5  # pushes 'A' (id 5) UP
    assert logits.grad[0, 1, 4].item() > 0.0  # pushes 'L' (id 4) DOWN
    assert logits.grad[0, 1, 0].item() == 0.0  # non-canonical token untouched at alpha=1.0

    # With ESM-C alphabet order ('L' at channel 0), channel 0 pushes 'L' (id 4) UP instead.
    logits2 = torch.zeros((1, 3, tok.vocab_size), dtype=torch.float32, requires_grad=True)
    result2 = DistillMLMTask(
        DistillConfig(alpha=1.0, temperature=1.0, teacher_alphabet="LAGVSERTIDPKQNFYMHWC")
    ).step(_FixedLogitsModel(logits2), batch)
    result2.loss.backward()
    assert logits2.grad is not None
    assert logits2.grad[0, 1, 4].item() < -0.5  # pushes 'L' (id 4) UP
    assert logits2.grad[0, 1, 5].item() > 0.0  # pushes 'A' (id 5) DOWN


def test_distill_task_blended_loss_formula_and_temperature_scaling() -> None:
    tok = get_tokenizer()
    torch.manual_seed(0)
    logits = torch.randn((1, 4, tok.vocab_size), dtype=torch.float32, requires_grad=True)
    teacher = torch.randn((1, 4, 20), dtype=torch.float16)
    labels = torch.tensor([[-100, 5, 4, -100]])  # pos 1 distilled ('A'), pos 2 plain ('L')
    teacher_mask = torch.tensor([[False, True, False, False]])
    batch = {
        "input_ids": torch.tensor([[0, 32, 32, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "labels": labels,
        "teacher_logits": teacher,
        "teacher_mask": teacher_mask,
    }
    cfg = DistillConfig(alpha=0.6, temperature=2.0)
    result = DistillMLMTask(cfg).step(_FixedLogitsModel(logits), batch)

    ce1 = F.cross_entropy(logits[0, 1:2], labels[0, 1:2])
    ce2 = F.cross_entropy(logits[0, 2:3], labels[0, 2:3])
    token_ids = torch.tensor(
        [tok.convert_tokens_to_ids(aa) for aa in cfg.teacher_alphabet], dtype=torch.long
    )
    s20 = logits[0, 1, token_ids]
    t20 = teacher[0, 1].float()
    temp = cfg.temperature
    kl1 = (
        F.kl_div(
            F.log_softmax(s20 / temp, dim=-1),
            F.softmax(t20 / temp, dim=-1),
            reduction="sum",
        )
        * temp
        * temp
    )
    expected_loss = (ce2 + (1.0 - cfg.alpha) * ce1 + cfg.alpha * kl1) / 2.0
    assert torch.allclose(result.loss, expected_loss, atol=1e-6)
    assert result.metrics["loss_ce"] == pytest.approx(((ce1 + ce2) / 2.0).item(), abs=1e-6)
    assert result.metrics["loss_kl"] == pytest.approx(kl1.item(), abs=1e-6)
    assert result.metrics["distill_frac"] == pytest.approx(0.5)


def test_distill_task_ignores_teacher_mask_on_kept_and_random_inputs() -> None:
    """Query-masked teacher KL must only apply where input_ids == <mask>, not kept/rand tokens."""
    tok = get_tokenizer()
    torch.manual_seed(1)
    logits = torch.randn((1, 5, tok.vocab_size), dtype=torch.float32, requires_grad=True)
    teacher = torch.randn((1, 5, 20), dtype=torch.float16)
    # pos 1: <mask> (id 32), pos 2: kept-unchanged ('A'=5), pos 3: random-AA ('L'=4, label='A'=5)
    labels = torch.tensor([[-100, 5, 5, 5, -100]])
    batch = {
        "input_ids": torch.tensor([[0, tok.mask_token_id, 5, 4, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        "labels": labels,
        "teacher_logits": teacher,
        "teacher_mask": torch.tensor([[False, True, True, True, False]]),
    }
    cfg = DistillConfig(alpha=0.5, temperature=1.0)
    result = DistillMLMTask(cfg).step(_FixedLogitsModel(logits), batch)

    ce1 = F.cross_entropy(logits[0, 1:2], labels[0, 1:2])
    ce2 = F.cross_entropy(logits[0, 2:3], labels[0, 2:3])
    ce3 = F.cross_entropy(logits[0, 3:4], labels[0, 3:4])
    token_ids = torch.tensor(
        [tok.convert_tokens_to_ids(aa) for aa in cfg.teacher_alphabet], dtype=torch.long
    )
    kl1 = F.kl_div(
        F.log_softmax(logits[0, 1, token_ids], dim=-1),
        F.softmax(teacher[0, 1].float(), dim=-1),
        reduction="sum",
    )
    expected = (ce2 + ce3 + (1.0 - cfg.alpha) * ce1 + cfg.alpha * kl1) / 3.0
    assert torch.allclose(result.loss, expected, atol=1e-6)
    assert result.metrics["distill_frac"] == pytest.approx(1.0 / 3.0)
    assert result.metrics["loss_kl"] == pytest.approx(kl1.item(), abs=1e-6)


@pytest.mark.parametrize("loss_type", ["reverse_kl", "js"])
def test_distill_task_reverse_kl_and_js_formulas_and_grads(loss_type: str) -> None:
    tok = get_tokenizer()
    torch.manual_seed(7)
    logits = torch.randn((1, 3, tok.vocab_size), dtype=torch.float32, requires_grad=True)
    teacher = torch.randn((1, 3, 20), dtype=torch.float16)
    labels = torch.tensor([[-100, 5, -100]])
    batch = {
        "input_ids": torch.tensor([[0, tok.mask_token_id, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": labels,
        "teacher_logits": teacher,
        "teacher_mask": torch.tensor([[False, True, False]]),
    }
    temp = 1.25
    cfg = DistillConfig(alpha=1.0, temperature=temp, loss_type=loss_type)
    result = DistillMLMTask(cfg).step(_FixedLogitsModel(logits), batch)

    token_ids = torch.tensor(
        [tok.convert_tokens_to_ids(aa) for aa in cfg.teacher_alphabet], dtype=torch.long
    )
    s20 = logits[0, 1, token_ids]
    t20 = teacher[0, 1].float()
    ps = F.softmax(s20 / temp, dim=-1)
    pt = F.softmax(t20 / temp, dim=-1)
    if loss_type == "reverse_kl":
        expected = (ps * (ps.log() - pt.log())).sum() * (temp * temp)
    else:
        m = 0.5 * (ps + pt)
        expected = (
            0.5
            * ((pt * (pt.log() - m.log())).sum() + (ps * (ps.log() - m.log())).sum())
            * (temp * temp)
        )
    assert torch.allclose(result.loss, expected, atol=1e-5)
    assert result.metrics["loss_kl"] == pytest.approx(expected.item(), abs=1e-5)
    result.loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1, token_ids].abs().sum().item() > 0.0


def test_distill_task_packed_matches_unpacked_loss(tmp_path: Path) -> None:
    tok = get_tokenizer()
    seqs = ["ACDEFGHIKLMNPQRSTVWY", "MKTAYIAKQRQISFVKSHFSRQ"]
    rng = np.random.default_rng(1)
    items = [
        {
            "sequence_id": "0",
            "sequence": seqs[0],
            "teacher_logits": rng.normal(size=(len(seqs[0]), 20)).astype("<f2").tobytes(),
        },
        {"sequence_id": "1", "sequence": seqs[1], "teacher_logits": None},
    ]
    unpacked = MLMCollator(
        tok, max_length=64, mask_prob=0.5, deterministic=True, seed=3, load_teacher_logits=True
    )(items)
    packed = MLMCollator(
        tok,
        max_length=64,
        mask_prob=0.5,
        deterministic=True,
        seed=3,
        pack_sequences=True,
        pack_rows=1,
        load_teacher_logits=True,
    )(items)

    cfg = tiny_train_cfg(tmp_path, tmp_path / "unused.parquet")
    task = DistillMLMTask(DistillConfig(alpha=0.5, temperature=1.5))
    model = task.build_model(cfg, None)
    model.eval()

    unpacked_result = task.step(model, unpacked)
    packed_result = task.step(model, packed)
    assert torch.allclose(packed_result.loss, unpacked_result.loss, atol=1e-5)
    assert packed_result.metrics["loss_ce"] == pytest.approx(
        unpacked_result.metrics["loss_ce"], abs=1e-5
    )
    assert packed_result.metrics["loss_kl"] == pytest.approx(
        unpacked_result.metrics["loss_kl"], abs=1e-5
    )
    assert packed_result.metrics["distill_frac"] == pytest.approx(
        unpacked_result.metrics["distill_frac"]
    )


def test_distill_task_handles_zero_distilled_and_zero_masked_batches() -> None:
    tok = get_tokenizer()
    task = DistillMLMTask(DistillConfig(alpha=0.5, temperature=1.5))

    # 1) teacher_mask all False (mirrors _canon_self_check [:4] slicing with no distilled rows)
    logits = torch.randn((4, 6, tok.vocab_size), dtype=torch.float32, requires_grad=True)
    labels = torch.full((4, 6), -100, dtype=torch.long)
    labels[:, 2] = 5
    plain_batch = {
        "input_ids": torch.ones((4, 6), dtype=torch.long),
        "attention_mask": torch.ones((4, 6), dtype=torch.long),
        "labels": labels,
        "teacher_logits": torch.zeros((4, 6, 20), dtype=torch.float16),
        "teacher_mask": torch.zeros((4, 6), dtype=torch.bool),
    }
    r_plain = task.step(_FixedLogitsModel(logits), plain_batch)
    expected_ce = F.cross_entropy(logits.view(-1, tok.vocab_size), labels.view(-1))
    assert torch.allclose(r_plain.loss, expected_ce, atol=1e-6)
    assert "loss_kl" not in r_plain.metrics
    assert r_plain.metrics["loss_ce"] == pytest.approx(expected_ce.item(), abs=1e-6)
    assert r_plain.metrics["distill_frac"] == 0.0
    r_plain.loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    # 2) labels all -100 (zero masked tokens)
    logits_empty = torch.randn((2, 4, tok.vocab_size), dtype=torch.float32, requires_grad=True)
    empty_batch = {
        "input_ids": torch.ones((2, 4), dtype=torch.long),
        "attention_mask": torch.ones((2, 4), dtype=torch.long),
        "labels": torch.full((2, 4), -100, dtype=torch.long),
        "teacher_logits": torch.zeros((2, 4, 20), dtype=torch.float16),
        "teacher_mask": torch.zeros((2, 4), dtype=torch.bool),
    }
    r_empty = task.step(_FixedLogitsModel(logits_empty), empty_batch)
    assert r_empty.loss.item() == 0.0
    assert r_empty.loss.requires_grad
    r_empty.loss.backward()
    assert logits_empty.grad is not None and torch.isfinite(logits_empty.grad).all()
    assert r_empty.metrics == {}


def test_trainer_distill_compat_and_mixed_packed_branch_from_plain_mlm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oplm.training.trainer import Trainer

    configure_accelerator_device("cpu", monkeypatch)

    # 1) Trainer.__init__ rejects fold + distill even when bypassing load_config
    plain_shard = tmp_path / "plain.parquet"
    seqs = [
        "ACDEFGHIKLMNPQRSTVWY",
        "MKTAYIAKQRQISFVKSHFSRQ",
        "GIVEQCCTSICSLYQLENYCN",
        "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    ] * 4
    pq.write_table(
        pa.table({"sequence_id": [f"p{i}" for i in range(len(seqs))], "sequence": seqs}),
        plain_shard,
    )
    bad_cfg = tiny_train_cfg(tmp_path / "bad", plain_shard)
    bad_cfg.fold = FoldRunConfig()
    bad_cfg.distill = DistillConfig()
    with pytest.raises(ValueError, match="distill"):
        Trainer(bad_cfg)

    # 2) Stage 1: plain MLM for 2 steps -> checkpoint-2
    stage1_dir = tmp_path / "stage1"
    stage1_cfg = tiny_train_cfg(
        stage1_dir,
        plain_shard,
        max_steps=2,
        log_every=1,
        save_final=True,
        pack_sequences=True,
        tokens_per_step=128,
        batch_tokens=128,
        mask_prob=0.5,
    )
    Trainer(stage1_cfg).train()
    ckpt2 = stage1_dir / "checkpoint-2"
    assert ckpt2.is_dir()

    # 3) Stage 2: branch_from checkpoint-2 into mixed packed distill anneal
    rng = np.random.default_rng(7)
    distilled_shard = tmp_path / "distilled.parquet"
    pq.write_table(
        pa.table(
            {
                "sequence_id": [f"d{i}" for i in range(len(seqs))],
                "sequence": seqs,
                "teacher_logits": pa.array(
                    [rng.normal(size=(len(s), 20)).astype("<f2").tobytes() for s in seqs],
                    type=pa.binary(),
                ),
                "teacher_neff": pa.array([40.0] * len(seqs), type=pa.float32()),
                "teacher_neff_sqrt_l": pa.array([6.0] * len(seqs), type=pa.float32()),
            }
        ),
        distilled_shard,
    )
    members_shard = tmp_path / "members.parquet"
    pq.write_table(
        pa.table(
            {
                "sequence_id": [f"m{i}" for i in range(len(seqs))],
                "sequence": seqs,
                "member_ids": [[f"m{i}_a", f"m{i}_b"] for i in range(len(seqs))],
                "member_sequences": [[s, s] for s in seqs],
            }
        ),
        members_shard,
    )

    stage2_dir = tmp_path / "stage2"
    stage2_cfg = tiny_train_cfg(
        stage2_dir,
        plain_shard,
        max_steps=4,
        log_every=1,
        save_final=True,
        branch_from=str(ckpt2),
        pack_sequences=True,
        tokens_per_step=128,
        batch_tokens=128,
        mask_prob=0.5,
    )
    stage2_cfg.data.train = {
        "distilled": {"path": str(distilled_shard), "fraction": 0.5},
        "plain": {"path": str(plain_shard), "fraction": 0.25},
        "members": {"path": str(members_shard), "fraction": 0.25, "sample_members": True},
    }
    stage2_cfg.distill = DistillConfig(alpha=0.5, temperature=1.5, min_neff_sqrt_l=2.0)

    cb = FullRecordingCallback()
    trainer = Trainer(stage2_cfg, callbacks=[cb])
    assert isinstance(trainer.task, DistillMLMTask)
    assert isinstance(trainer.task, MLMTask)
    trainer.train()

    assert cb.train_log_steps == [3, 4]
    for _, m in cb.train_logs:
        assert math.isfinite(m["train/loss_ce"])
        assert 0.0 <= m["train/distill_frac"] <= 1.0
    assert any("train/loss_kl" in m and math.isfinite(m["train/loss_kl"]) for _, m in cb.train_logs)

    reloaded = load_config(["--config", str(stage2_dir / "checkpoint-4" / "config.yaml")])
    assert reloaded.distill == stage2_cfg.distill


@pytest.mark.parametrize("contact_head", ["attention", "pair"])
@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_distill_mlm_task_2d_step_and_empty_contacts(
    tmp_path: Path, contact_head: str, alpha: float
) -> None:
    from accelerate import Accelerator

    from oplm.data.sequence.collate import MLMCollator, encode_teacher_contacts
    from oplm.data.tokenizer import get_tokenizer

    torch.manual_seed(11)
    seqs = ["ACDEFGHIKLMNPQRSTVWY", "MKTAYIAKQRQISFVKSHFSRQ"]
    rng = np.random.default_rng(11)
    items = []
    for i, s in enumerate(seqs):
        mat = rng.uniform(0.0, 1.0, (len(s), len(s))).astype(np.float32)
        mat = 0.5 * (mat + mat.T)
        row: dict[str, object] = {
            "sequence_id": f"s{i}",
            "sequence": s,
            "teacher_contacts": encode_teacher_contacts(mat, min_sep=6),
        }
        if alpha > 0.0:
            row["teacher_logits"] = rng.normal(size=(len(s), 20)).astype("<f2").tobytes()
        items.append(row)

    collator = MLMCollator(
        get_tokenizer(),
        max_length=64,
        mask_prob=0.5,
        deterministic=True,
        seed=0,
        pack_sequences=True,
        pack_rows=1,
        load_teacher_logits=alpha > 0.0,
        load_teacher_contacts=True,
        contact_min_sep=6,
    )
    batch = collator(items)

    cfg = tiny_train_cfg(tmp_path, tmp_path / "unused.parquet", max_steps=10)
    cfg.distill = DistillConfig(
        alpha=alpha,
        temperature=1.0,
        contact_weight=1.0,
        contact_head=contact_head,
        contact_layers=2,
        contact_min_sep=6,
        contact_pair_dim=16,
    )
    task = DistillMLMTask(cfg.distill)
    model = task.build_model(cfg, initialization_source=None)
    acc = Accelerator(cpu=True)
    task.prepare_aux(model, acc, cfg, total_steps=10)
    task.sync_aux_schedule(0, fresh_start=True)

    result = task.step(model, batch)
    assert math.isfinite(result.loss.item())
    assert "loss_contact" in result.metrics and math.isfinite(result.metrics["loss_contact"])
    assert result.metrics["contact_frac"] == pytest.approx(1.0)
    assert 0.0 <= result.metrics["contact_pos_rate"] <= 1.0
    if alpha > 0.0:
        assert "loss_kl" in result.metrics
    else:
        assert "loss_kl" not in result.metrics

    result.loss.backward()
    task.step_aux(acc, max_grad_norm=1.0, global_step=0)

    # Empty-contact batch (M == 0): zero 2D loss, step_aux zero-fills grads without error
    model.zero_grad()
    empty_batch = collator([{"sequence_id": "e0", "sequence": seqs[0], "teacher_contacts": None}])
    empty_res = task.step(model, empty_batch)
    assert math.isfinite(empty_res.loss.item())
    assert empty_res.metrics["contact_frac"] == 0.0
    assert "loss_contact" not in empty_res.metrics
    empty_res.loss.backward()
    task.step_aux(acc, max_grad_norm=1.0, global_step=1)


@pytest.mark.parametrize("contact_head", ["attention", "pair"])
def test_trainer_2d_contact_distill_branch_resume_and_clean_hf_export(
    tmp_path: Path, contact_head: str
) -> None:
    from oplm.data.sequence.collate import encode_teacher_contacts
    from oplm.model import OplmForMaskedLM
    from oplm.training.contact_distill import DISTILL_AUX_SIDECAR_NAME
    from oplm.training.trainer import Trainer

    seqs = [
        "ACDEFGHIKLMNPQRSTVWY",
        "MKTAYIAKQRQISFVKSHFSRQ",
        "GIVEQCCTSICSLYQLENYCN",
        "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    ] * 4
    plain_shard = tmp_path / "plain.parquet"
    pq.write_table(
        pa.table({"sequence_id": [f"p{i}" for i in range(len(seqs))], "sequence": seqs}),
        plain_shard,
    )

    # 1) Plain MLM for 2 steps -> checkpoint-2 (no distill_aux.pt)
    stage1_dir = tmp_path / "stage1"
    stage1_cfg = tiny_train_cfg(
        stage1_dir,
        plain_shard,
        max_steps=2,
        log_every=1,
        save_final=True,
        pack_sequences=True,
        tokens_per_step=128,
        batch_tokens=128,
        mask_prob=0.5,
    )
    Trainer(stage1_cfg).train()
    ckpt2 = stage1_dir / "checkpoint-2"
    assert not (ckpt2 / DISTILL_AUX_SIDECAR_NAME).exists()

    # 2) Build contact-bearing shard and branch from checkpoint-2 into 2D distill
    rng = np.random.default_rng(13)
    contacts = []
    for s in seqs:
        m = rng.uniform(0.0, 1.0, (len(s), len(s))).astype(np.float32)
        contacts.append(encode_teacher_contacts(0.5 * (m + m.T), min_sep=6))
    distill_shard = tmp_path / "distill_2d.parquet"
    pq.write_table(
        pa.table(
            {
                "sequence_id": [f"d{i}" for i in range(len(seqs))],
                "sequence": seqs,
                "teacher_contacts": pa.array(contacts, type=pa.large_binary()),
                "teacher_neff_sqrt_l": pa.array([5.0] * len(seqs), type=pa.float32()),
            }
        ),
        distill_shard,
    )

    stage2_dir = tmp_path / "stage2"
    stage2_cfg = tiny_train_cfg(
        stage2_dir,
        plain_shard,
        max_steps=4,
        log_every=1,
        save_final=True,
        branch_from=str(ckpt2),
        scheduler="wsd_linear",
        warmup_steps=1,
        stable_steps=1,
        pack_sequences=True,
        tokens_per_step=128,
        batch_tokens=128,
        mask_prob=0.5,
    )
    stage2_cfg.data.train = {
        "distilled": {"path": str(distill_shard), "fraction": 0.5},
        "plain": {"path": str(plain_shard), "fraction": 0.5},
    }
    stage2_cfg.distill = DistillConfig(
        alpha=0.0,
        contact_weight=1.0,
        contact_head=contact_head,
        contact_layers=2,
        contact_min_sep=6,
        contact_pair_dim=16,
        contact_head_warmup_steps=2,
        min_neff_sqrt_l=2.0,
    )

    cb = FullRecordingCallback()
    trainer = Trainer(stage2_cfg, callbacks=[cb])
    assert isinstance(trainer.task, DistillMLMTask)
    assert trainer.task._aux is not None
    assert trainer.task._aux.aux_start_step == 2
    trainer.train()

    assert cb.train_log_steps == [3, 4]
    assert any(
        "train/loss_contact" in m and math.isfinite(m["train/loss_contact"])
        for _, m in cb.train_logs
    )
    ckpt4 = stage2_dir / "checkpoint-4"
    assert (ckpt4 / DISTILL_AUX_SIDECAR_NAME).is_file()

    # Clean HF inference checkpoint: loads cleanly with zero missing/unexpected keys
    reloaded, info = OplmForMaskedLM.from_pretrained(str(ckpt4 / "hf"), output_loading_info=True)
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    assert reloaded is not None

    # 3) Auto-resume from checkpoint-4 to step 6 restores distill_aux.pt & aux_start_step == 2
    stage2_cfg.train.max_steps = 6
    stage2_cfg.train.auto_resume = True
    cb_resume = FullRecordingCallback()
    trainer_resume = Trainer(stage2_cfg, callbacks=[cb_resume])
    assert isinstance(trainer_resume.task, DistillMLMTask)
    assert trainer_resume.task._aux is not None
    assert trainer_resume.task._aux.aux_start_step == 2
    trainer_resume.train()
    assert cb_resume.train_log_steps == [5, 6]


def test_canon_self_check_slices_2d_contact_batch_past_row_four(tmp_path: Path) -> None:
    """_canon_self_check trims batches to 4 rows without indexing OOB in teacher_contact_meta."""
    from oplm.data.sequence.collate import encode_teacher_contacts
    from oplm.training.trainer import Trainer

    seqs = ["ACDEFGHIKLMNPQRSTVWY"] * 6
    rng = np.random.default_rng(7)
    contacts = [
        encode_teacher_contacts(
            0.5 * (m + m.T),
            min_sep=6,
        )
        for m in (rng.uniform(0.0, 1.0, (len(s), len(s))).astype(np.float32) for s in seqs)
    ]
    shard = tmp_path / "canon_2d.parquet"
    pq.write_table(
        pa.table(
            {
                "sequence_id": [f"c{i}" for i in range(len(seqs))],
                "sequence": seqs,
                "teacher_contacts": pa.array(contacts, type=pa.large_binary()),
                "teacher_neff_sqrt_l": pa.array([5.0] * len(seqs), type=pa.float32()),
            }
        ),
        shard,
    )
    cfg = tiny_train_cfg(
        tmp_path / "run",
        shard,
        max_steps=1,
        batch_size=6,
        pack_sequences=False,
    )
    cfg.model.canon_enabled = True
    cfg.model.canon_positions = ["A", "B", "C", "D"]
    cfg.model.canon_kernel_sizes = [4] * cfg.model.num_hidden_layers
    cfg.distill = DistillConfig(
        alpha=0.0,
        contact_weight=1.0,
        contact_head="attention",
        contact_layers=2,
        contact_min_sep=6,
        min_neff_sqrt_l=2.0,
    )
    trainer = Trainer(cfg)
    batch = next(iter(trainer.dataloader))
    assert batch["input_ids"].shape[0] == 6
    assert any(m[0] >= 4 for m in batch["teacher_contact_meta"])
    # Exercise the exact slicing helper; CanonConv.forward runs on CPU (no Triton kernel),
    # so patch canon_self_check to just execute run_forward and verify no IndexError occurs.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("oplm.model.canon_kernel.canon_self_check", lambda _m, fn: (fn(), 1)[1])
        trainer._canon_self_check(batch)

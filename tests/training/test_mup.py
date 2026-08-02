"""Fast unit tests for μP width transfer (Phases 2-5).

These exercise the μP recipe without training: the per-matrix fan-in multiplier
and LR multiplier, width-aware init, the readout output multiplier, per-group
learning-rate assembly under both optimizers, the Muon+μP guard, and the
no-op-when-off / no-op-at-base-width invariants. See ``docs/MUP.md`` for the
recipe table these mirror.

Width-aware init is checked *exactly*, not statistically: ``trunc_normal_`` scales
a fixed sequence of draws by ``std``, so with one shared seed a μP-on model's
weights equal the μP-off model's weights times ``std_on / std_off`` element-wise.
That ratio is ``1/√m_W`` for hidden matrices and ``1.0`` for readouts/embeddings.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F

from oplm.config import OplmConfig, TrainConfig
from oplm.model import OplmConfig as OplmModelConfig
from oplm.model import OplmForMaskedLM, OplmForSequenceClassification
from oplm.training.mup import SweepMetricsCallback, mup_fanin_mult, mup_lr_multiplier
from oplm.training.optim import build_optimizers, partition_optimizer_params

if TYPE_CHECKING:
    from pathlib import Path

# Base width 64 with width 256 gives m = 4 and (via the 256-rounded FFN sizes
# i0 = 256, i = 768) m_ffn = 3 — so the down_proj multiplier is genuinely
# distinct from the hidden multiplier, exercising the per-matrix fan-in rule.
_BASE_WIDTH = 64
_WIDTH = 256
_HEADS = _WIDTH // 64  # head_dim held at 64, the μP-invariant
_LAYERS = 2

# Role-tagged parameter-name suffixes used to classify the multiplier a 2-D
# weight should receive (mirrors the recipe's owner-by-role split).
_HIDDEN_FANIN_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "lm_head.dense.weight",
)


def _mlm_model(
    width: int = _WIDTH,
    *,
    mup: bool = True,
    base_width: int = _BASE_WIDTH,
    output_mult: float = 1.0,
    seed: int = 0,
) -> OplmForMaskedLM:
    """Build a tiny untied ``OplmForMaskedLM`` at a fixed seed (head_dim = 64)."""
    torch.manual_seed(seed)
    return OplmForMaskedLM(
        OplmModelConfig(
            hidden_size=width,
            num_attention_heads=width // 64,
            num_hidden_layers=_LAYERS,
            max_position_embeddings=64,
            mup_enable=mup,
            mup_base_width=base_width,
            mup_output_mult=output_mult,
        )
    )


def _mults(model: OplmForMaskedLM) -> tuple[float, float]:
    """Return ``(m, m_ffn)`` for a model's config (computed, not hard-coded)."""
    cfg = model.config
    m = cfg.mup_width_mult()
    m_ffn = cfg.intermediate_size / cfg.mup_base_intermediate_size()
    return m, m_ffn


def _group_lr(name: str, model: OplmForMaskedLM, optimizers: list) -> tuple[str, float]:
    """Find the ``(optimizer_type, group_lr)`` holding the named parameter."""
    target = dict(model.named_parameters())[name]
    for opt in optimizers:
        for group in opt.param_groups:
            if any(p is target for p in group["params"]):
                return type(opt).__name__, group["lr"]
    raise AssertionError(f"{name} not found in any optimizer group")


# ---------------------------------------------------------------------------
# mup_fanin_mult / mup_lr_multiplier
# ---------------------------------------------------------------------------


def test_lr_multiplier_per_role() -> None:
    """``mup_lr_multiplier`` → 1/m hidden, 1/m_ffn down_proj, 1.0 embed/readout/1-D."""
    model = _mlm_model()
    m, m_ffn = _mults(model)
    assert (m, m_ffn) == (4.0, 3.0)  # guards the chosen widths against silent drift

    for name, param in model.named_parameters():
        mult = mup_lr_multiplier(name, param, model.config)
        if param.ndim != 2 or "embed" in name or name.endswith("lm_head.decoder.weight"):
            assert mult == 1.0, f"{name} should be ×1"
        elif name.endswith("down_proj.weight"):
            assert mult == pytest.approx(1.0 / m_ffn), f"{name} should be 1/m_ffn"
        elif name.endswith(_HIDDEN_FANIN_SUFFIXES):
            assert mult == pytest.approx(1.0 / m), f"{name} should be 1/m"


def test_lr_multiplier_is_one_when_disabled() -> None:
    """With μP off, every parameter's LR multiplier is exactly 1.0."""
    model = _mlm_model(mup=False)
    for name, param in model.named_parameters():
        assert mup_lr_multiplier(name, param, model.config) == 1.0


def test_fanin_mult_uses_intermediate_for_down_proj() -> None:
    """``mup_fanin_mult`` keys off true fan-in: m for hidden, m_ffn for down_proj."""
    model = _mlm_model()
    m, m_ffn = _mults(model)
    params = dict(model.named_parameters())
    q = params["oplm.backbone.layers.0.attention.q_proj.weight"]
    down = params["oplm.backbone.layers.0.ffn.down_proj.weight"]
    assert mup_fanin_mult(q, model.config) == pytest.approx(m)
    assert mup_fanin_mult(down, model.config) == pytest.approx(m_ffn)
    # Non-2-D tensors (biases, norms) and μP-off both collapse to 1.0.
    assert mup_fanin_mult(torch.zeros(8), model.config) == 1.0
    assert mup_fanin_mult(q, _mlm_model(mup=False).config) == 1.0


# ---------------------------------------------------------------------------
# Width-aware init (exact element-wise ratio vs the μP-off model)
# ---------------------------------------------------------------------------


def test_init_std_ratios_per_role() -> None:
    """μP-on weights = μP-off weights × (std_on/std_off): 1/√m hidden, 1/√m_ffn ffn, 1× readout."""
    on = _mlm_model(mup=True)
    off = _mlm_model(mup=False)  # same seed, same shapes → same underlying draws
    m, m_ffn = _mults(on)
    on_p, off_p = dict(on.named_parameters()), dict(off.named_parameters())

    def assert_ratio(name: str, factor: float) -> None:
        assert torch.allclose(on_p[name], off_p[name] * factor, rtol=1e-5, atol=1e-7), name

    # Hidden-fan-in matrices (incl. the residual writer o_proj — the 1/√(2L)
    # writer factor is identical at a fixed width, so it cancels in the ratio).
    for suffix in ("attention.q_proj.weight", "attention.o_proj.weight", "ffn.gate_proj.weight"):
        assert_ratio(f"oplm.backbone.layers.0.{suffix}", 1.0 / math.sqrt(m))
    assert_ratio("lm_head.dense.weight", 1.0 / math.sqrt(m))
    # down_proj scales by the intermediate fan-in, not the hidden one.
    assert_ratio("oplm.backbone.layers.0.ffn.down_proj.weight", 1.0 / math.sqrt(m_ffn))
    # Embedding (μP input) and the untied readout keep width-independent init.
    assert_ratio("oplm.backbone.embed_tokens.embed_tokens.weight", 1.0)
    assert_ratio("lm_head.decoder.weight", 1.0)


def test_classifier_head_init_is_width_independent() -> None:
    """The fine-tuning ``classifier`` head is a μP readout: init unchanged by m."""

    def build(mup: bool) -> OplmForSequenceClassification:
        torch.manual_seed(0)
        return OplmForSequenceClassification(
            OplmModelConfig(
                hidden_size=_WIDTH,
                num_attention_heads=_HEADS,
                num_hidden_layers=_LAYERS,
                max_position_embeddings=64,
                mup_enable=mup,
                mup_base_width=_BASE_WIDTH,
                num_labels=5,
            )
        )

    on, off = build(True), build(False)
    assert torch.allclose(on.classifier.weight, off.classifier.weight, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Readout output multiplier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output_mult", [1.0, 2.0])
def test_readout_output_mult_value(output_mult: float) -> None:
    """``lm_head.output_mult`` == mup_output_mult / m."""
    model = _mlm_model(output_mult=output_mult)
    m, _ = _mults(model)
    assert model.lm_head.output_mult == pytest.approx(output_mult / m)


def test_readout_scales_matmul_path_not_bias() -> None:
    """``output_mult`` multiplies the decoder *input*, leaving ``decoder.bias`` unscaled."""
    model = _mlm_model()
    head = model.lm_head
    head.eval()
    with torch.no_grad():
        head.decoder.bias.normal_()  # a nonzero bias makes the "unscaled" claim testable
        bias = head.decoder.bias.clone()
        x = torch.randn(2, 5, model.config.hidden_size)
        pre = head.norm(head.act(head.dense(x)))
        weight_path = F.linear(pre, head.decoder.weight)  # matmul only, no bias

        head.output_mult = 1.0
        y1 = head(x)
        head.output_mult = 3.0
        y3 = head(x)

    # The matmul path scales with output_mult; the bias is added once, unscaled.
    assert torch.allclose(y1, weight_path + bias, atol=1e-5)
    assert torch.allclose(y3, 3.0 * weight_path + bias, atol=1e-5)
    assert torch.allclose(y3 - y1, 2.0 * weight_path, atol=1e-5)  # bias cancels


# ---------------------------------------------------------------------------
# Per-group learning-rate assembly
# ---------------------------------------------------------------------------


def test_param_groups_muon_mode() -> None:
    """Muon mode: backbone on Muon @ lr; lm_head.dense @ lr/m; readout/embed @ lr."""
    model = _mlm_model()
    m, _ = _mults(model)
    lr = 1e-2
    optimizers = build_optimizers(
        model, TrainConfig(optimizer="muon", muon_adjust_lr_fn="original", lr=lr, weight_decay=0.01)
    )
    muon, _adamw = optimizers
    assert type(muon).__name__ == "Muon"
    # Every Muon group runs at the constant base LR (μP transfer rides on init + the
    # aspect-ratio-only "original" factor, not a per-width Muon LR).
    assert all(g["lr"] == pytest.approx(lr) for g in muon.param_groups)

    # Backbone hidden weights (incl. down_proj) are on Muon.
    assert _group_lr("oplm.backbone.layers.0.attention.q_proj.weight", model, optimizers) == (
        "Muon",
        pytest.approx(lr),
    )
    assert _group_lr("oplm.backbone.layers.0.ffn.down_proj.weight", model, optimizers)[0] == "Muon"
    # lm_head.dense stays on AdamW and takes lr/m; the readout + embedding take ×1.
    assert _group_lr("lm_head.dense.weight", model, optimizers) == ("AdamW", pytest.approx(lr / m))
    assert _group_lr("lm_head.decoder.weight", model, optimizers) == ("AdamW", pytest.approx(lr))
    assert _group_lr("oplm.backbone.embed_tokens.embed_tokens.weight", model, optimizers) == (
        "AdamW",
        pytest.approx(lr),
    )


def test_param_groups_adamw_mode() -> None:
    """AdamW-only mode: hidden @ lr/m, down_proj @ lr/m_ffn (distinct group), readout @ lr."""
    model = _mlm_model()
    m, m_ffn = _mults(model)
    lr = 1e-2
    optimizers = build_optimizers(model, TrainConfig(optimizer="adamw", lr=lr, weight_decay=0.01))
    assert len(optimizers) == 1

    assert _group_lr("oplm.backbone.layers.0.attention.q_proj.weight", model, optimizers) == (
        "AdamW",
        pytest.approx(lr / m),
    )
    assert _group_lr("lm_head.dense.weight", model, optimizers) == ("AdamW", pytest.approx(lr / m))
    # down_proj uses the intermediate fan-in → a genuinely separate LR (lr/m_ffn).
    assert _group_lr("oplm.backbone.layers.0.ffn.down_proj.weight", model, optimizers) == (
        "AdamW",
        pytest.approx(lr / m_ffn),
    )
    assert _group_lr("lm_head.decoder.weight", model, optimizers) == ("AdamW", pytest.approx(lr))


def test_depth_lr_multiplier_composes_with_muon_groups() -> None:
    model = _mlm_model()
    m, _ = _mults(model)
    lr = 1e-2
    depth_mult = (1 / _LAYERS) ** 0.5
    cfg = TrainConfig(
        optimizer="muon",
        muon_adjust_lr_fn="original",
        lr=lr,
        weight_decay=0.01,
        mup_depth_lr_exponent=0.5,
        mup_depth_reference_layers=1,
    )
    optimizers = build_optimizers(model, cfg)

    assert _group_lr("oplm.backbone.layers.0.attention.q_proj.weight", model, optimizers) == (
        "Muon",
        pytest.approx(lr * depth_mult),
    )
    assert _group_lr("lm_head.dense.weight", model, optimizers) == (
        "AdamW",
        pytest.approx(lr / m),
    )


def test_depth_lr_multiplier_composes_with_adamw_width_groups() -> None:
    model = _mlm_model()
    m, _ = _mults(model)
    lr = 1e-2
    depth_mult = (1 / _LAYERS) ** 0.5
    cfg = TrainConfig(
        optimizer="adamw",
        lr=lr,
        weight_decay=0.01,
        mup_depth_lr_exponent=0.5,
        mup_depth_reference_layers=1,
    )
    optimizers = build_optimizers(model, cfg)

    assert _group_lr("oplm.backbone.layers.0.attention.q_proj.weight", model, optimizers) == (
        "AdamW",
        pytest.approx(lr * depth_mult / m),
    )
    assert _group_lr("lm_head.dense.weight", model, optimizers) == (
        "AdamW",
        pytest.approx(lr / m),
    )
    assert _group_lr("oplm.backbone.embed_tokens.embed_tokens.weight", model, optimizers) == (
        "AdamW",
        pytest.approx(lr),
    )


# ---------------------------------------------------------------------------
# Guard + no-op invariants
# ---------------------------------------------------------------------------


def test_muon_match_rms_adamw_guard() -> None:
    """μP + Muon + ``match_rms_adamw`` raises (it scales like √width, breaking transfer)."""
    model = _mlm_model()
    with pytest.raises(ValueError, match="muon_adjust_lr_fn='original'"):
        build_optimizers(model, TrainConfig(optimizer="muon", muon_adjust_lr_fn="match_rms_adamw"))
    # The μP-correct factor builds fine.
    build_optimizers(model, TrainConfig(optimizer="muon", muon_adjust_lr_fn="original"))


def test_disabled_mup_is_a_noop_for_param_groups() -> None:
    """With μP off, AdamW groups carry only the usual decay/no-decay split (all ×1)."""
    model = _mlm_model(mup=False)
    for optimizer in ("adamw", "muon"):
        groups = partition_optimizer_params(model, TrainConfig(optimizer=optimizer)).adamw_groups
        assert all(g.lr_mult == 1.0 for g in groups)
        assert len(groups) <= 2  # at most (decay, no-decay) — no per-width sub-groups


def test_mup_is_noop_at_base_width() -> None:
    """μP-on at hidden == mup_base_width (m = 1) is bit-identical to μP-off."""
    on = _mlm_model(width=_BASE_WIDTH, mup=True, base_width=_BASE_WIDTH)
    off = _mlm_model(width=_BASE_WIDTH, mup=False)
    on_p, off_p = dict(on.named_parameters()), dict(off.named_parameters())
    assert on_p.keys() == off_p.keys()
    for name in on_p:
        assert torch.equal(on_p[name], off_p[name]), name
    assert on.lm_head.output_mult == 1.0


# ---------------------------------------------------------------------------
# SweepMetricsCallback.on_train_end — result.json version provenance
# ---------------------------------------------------------------------------


class _StubAccelerator:
    """Exposes only what ``on_train_end`` reads off the accelerator: ``num_processes``."""

    def __init__(self, num_processes: int = 1) -> None:
        self.num_processes = num_processes


class _StubTrainer:
    """Minimal trainer stand-in for :meth:`SweepMetricsCallback.on_train_end`.

    Exposes exactly the attributes that method reads (``src/oplm/training/mup.py``):
    ``cfg.train.{batch_size,gradient_accumulation_steps,lr}``, ``cfg.model.hidden_size``,
    ``accelerator.num_processes``, and ``total_steps`` (read via ``getattr(..., default=
    cfg.train.max_steps)``, so the real ``Trainer`` — which always sets it in
    ``__init__`` — takes the ``total_steps`` branch). Notably, ``on_train_end`` never
    reads ``global_step``.
    """

    def __init__(
        self,
        *,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 2,
        lr: float = 0.01,
        hidden_size: int = 128,
        num_attention_heads: int = 2,
        num_processes: int = 1,
        total_steps: int = 7,
    ) -> None:
        self.cfg = OplmConfig(
            model=OplmModelConfig(hidden_size=hidden_size, num_attention_heads=num_attention_heads),
            train=TrainConfig(
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                lr=lr,
            ),
        )
        self.accelerator = _StubAccelerator(num_processes)
        self.total_steps = total_steps


def test_result_json_records_installed_version(tmp_path: Path) -> None:
    """oplm.__version__ is stale (0.0.1); result.json must carry the installed dist version."""
    from importlib.metadata import version

    callback = SweepMetricsCallback(tmp_path / "result.json")
    callback.on_train_end(_StubTrainer())

    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["oplm_version"] == version("oplm")


def test_result_json_version_is_none_when_package_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare checkout (no installed distribution) records ``oplm_version: None``, not a crash."""
    import oplm.training.mup as mup_module

    def _raise(_name: str) -> str:
        raise mup_module.PackageNotFoundError

    monkeypatch.setattr(mup_module, "version", _raise)

    callback = SweepMetricsCallback(tmp_path / "result.json")
    callback.on_train_end(_StubTrainer())

    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["oplm_version"] is None

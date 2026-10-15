"""Tests for optimizer parameter partitioning (Phase 3 head-exclusion fix).

The MLM head is ``OplmForMaskedLM.lm_head`` (an ``OplmMLMHead``). Muon must never
claim its 2-D weights (``lm_head.dense.weight`` and, when untied,
``lm_head.decoder.weight``) — those belong on AdamW-with-decay. Norms, biases, and
any ``embed``-named weight go to AdamW-no-decay.
"""

from __future__ import annotations

import pytest
import torch

from oplm.config import TrainConfig
from oplm.model import OplmConfig as OplmModelConfig
from oplm.model import OplmForMaskedLM
from oplm.training.optim import OptimizerParamGroups, partition_optimizer_params


def _model() -> OplmForMaskedLM:
    """Build a tiny untied ``OplmForMaskedLM`` (default ``tie_word_embeddings=False``)."""
    return OplmForMaskedLM(
        OplmModelConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=64,
        )
    )


def _mup_model() -> OplmForMaskedLM:
    """A μP-enabled model at width 256 over base width 64 (m = 4, m_ffn = 3)."""
    return OplmForMaskedLM(
        OplmModelConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=64,
            mup_enable=True,
            mup_base_width=64,
        )
    )


def _names_of(params: list[torch.nn.Parameter], id_to_name: dict[int, str]) -> set[str]:
    """Map a parameter list back to its parameter names via ``id()`` lookup."""
    return {id_to_name[id(p)] for p in params}


def _muon_params(groups: OptimizerParamGroups) -> list[torch.nn.Parameter]:
    return [param for group in groups.muon_groups for param in group.params]


def test_muon_excludes_lm_head_weights() -> None:
    """Under Muon, no ``lm_head.*`` weight leaks into Muon; both 2-D head weights go to AdamW."""
    model = _model()
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    groups = partition_optimizer_params(model, TrainConfig(optimizer="muon"))

    muon_names = _names_of(_muon_params(groups), id_to_name)
    assert not any(name.startswith("lm_head.") for name in muon_names)

    decay_names = _names_of(groups.adamw_decay_params, id_to_name)
    assert "lm_head.dense.weight" in decay_names
    assert "lm_head.decoder.weight" in decay_names  # present because the model is untied

    # Muon still claims the backbone's 2-D hidden weights (otherwise the run errors).
    assert muon_names, "Muon received no eligible parameters"


def test_no_decay_holds_norms_biases_and_embeddings() -> None:
    """1-D params (norms/biases) and any ``embed``-named weight land on AdamW-no-decay."""
    model = _model()
    groups = partition_optimizer_params(model, TrainConfig(optimizer="muon"))

    no_decay_ids = {id(p) for p in groups.adamw_no_decay_params}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "embed" in name:
            assert id(param) in no_decay_ids, f"{name} should be AdamW-no-decay"


def test_partition_covers_every_trainable_param_once() -> None:
    """The three groups tile the trainable parameter set exactly (no gaps, no dupes)."""
    model = _model()
    groups = partition_optimizer_params(model, TrainConfig(optimizer="muon"))

    grouped = [
        *_muon_params(groups),
        *groups.adamw_decay_params,
        *groups.adamw_no_decay_params,
    ]
    grouped_ids = [id(p) for p in grouped]
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}

    assert len(grouped_ids) == len(set(grouped_ids))  # no parameter counted twice
    assert set(grouped_ids) == trainable_ids  # full coverage


def test_adamw_leaves_muon_group_empty() -> None:
    """With the default AdamW optimizer, nothing is routed to Muon."""
    model = _model()
    groups = partition_optimizer_params(model, TrainConfig(optimizer="adamw"))
    assert _muon_params(groups) == []


def test_mup_partition_tags_scaled_adamw_groups() -> None:
    """Under μP, AdamW groups carry per-matrix LR multipliers (the new μP groups).

    Complements ``test_mup.py`` (which asserts the built optimizers' LRs) by
    checking the partition-level contract: ``lm_head.dense`` lands in a ``1/m``
    group, the untied readout in a ``×1`` group, and the backbone hidden weights
    stay on Muon (never in an AdamW μP group).
    """
    model = _mup_model()
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    groups = partition_optimizer_params(model, TrainConfig(optimizer="muon"))
    m = model.config.mup_width_mult()

    def lr_mult_of(name: str) -> float:
        return next(
            g.lr_mult
            for g in groups.adamw_groups
            if any(id_to_name[id(p)] == name for p in g.params)
        )

    assert lr_mult_of("lm_head.dense.weight") == pytest.approx(1.0 / m)
    assert lr_mult_of("lm_head.decoder.weight") == 1.0  # μP readout: width-independent

    muon_names = _names_of(_muon_params(groups), id_to_name)
    assert "oplm.backbone.layers.0.attention.q_proj.weight" in muon_names
    assert not any(name.startswith("lm_head.") for name in muon_names)


def test_mup_off_partition_lr_mults_all_one() -> None:
    """With μP off, every AdamW group keeps ``lr_mult == 1.0`` (no behavior change)."""
    model = _model()  # μP disabled by default
    for optimizer in ("adamw", "muon"):
        groups = partition_optimizer_params(model, TrainConfig(optimizer=optimizer))
        assert all(g.lr_mult == 1.0 for g in groups.adamw_groups)
        assert len(groups.adamw_groups) <= 2  # only the usual decay / no-decay split


def test_zero_depth_exponent_keeps_one_muon_lr_group() -> None:
    model = _model()
    groups = partition_optimizer_params(
        model,
        TrainConfig(
            optimizer="muon",
            mup_depth_lr_exponent=0.0,
            mup_depth_reference_layers=24,
        ),
    )
    assert len(groups.muon_groups) == 1
    assert groups.muon_groups[0].lr_mult == 1.0


def test_residual_gates_land_in_no_decay_group() -> None:
    """1-D residual-gate params route to AdamW-no-decay under both adamw and muon.

    Under Muon the no-decay group is consumed by the auxiliary AdamW optimizer, so
    this also confirms the gates never leak into Muon (which claims only 2-D weights).
    """
    model = OplmForMaskedLM(
        OplmModelConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=64,
            residual_gate="channel",
        )
    )
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    gate_names = {n for n in id_to_name.values() if n.endswith(("attn_gate", "ffn_gate"))}
    assert len(gate_names) == 4, "expected attn_gate + ffn_gate per layer (2 layers)"

    for optimizer in ("adamw", "muon"):
        groups = partition_optimizer_params(model, TrainConfig(optimizer=optimizer))
        no_decay = _names_of(groups.adamw_no_decay_params, id_to_name)
        muon = _names_of(_muon_params(groups), id_to_name)
        assert gate_names <= no_decay, f"gates missing from no-decay under {optimizer}"
        assert not (gate_names & muon), f"gates leaked into Muon under {optimizer}"


def test_adamw_lr_mult_scales_only_the_adamw_groups() -> None:
    """``adamw_lr_mult`` rescales every AdamW group's LR; Muon groups are untouched."""
    from oplm.training.optim import build_optimizers

    model = _model()
    base = TrainConfig(optimizer="muon", muon_adjust_lr_fn="original", lr=1e-2)
    scaled = TrainConfig(optimizer="muon", muon_adjust_lr_fn="original", lr=1e-2, adamw_lr_mult=0.1)
    muon_base, adamw_base = build_optimizers(model, base)
    muon_scaled, adamw_scaled = build_optimizers(model, scaled)

    assert [g["lr"] for g in muon_scaled.param_groups] == [g["lr"] for g in muon_base.param_groups]
    for g_scaled, g_base in zip(adamw_scaled.param_groups, adamw_base.param_groups, strict=True):
        assert g_scaled["lr"] == pytest.approx(0.1 * g_base["lr"])
        assert g_scaled["weight_decay"] == g_base["weight_decay"]


def _decay_model() -> OplmForMaskedLM:
    """LayerNorm (gains + biases), channel gates at a non-1 init, value residual, L2 QK scales."""
    return OplmForMaskedLM(
        OplmModelConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=64,
            norm_type="layernorm",
            residual_gate="channel",
            residual_gate_init=0.5,
            value_residual="learnable",
            qk_norm_mode="l2",
        )
    )


def _decay_of(groups: OptimizerParamGroups, id_to_name: dict[int, str]) -> dict[str, tuple]:
    return {
        id_to_name[id(p)]: (g.weight_decay, g.decay_target)
        for g in groups.adamw_groups
        for p in g.params
    }


def test_decay_knobs_off_keep_the_default_groups() -> None:
    """Both knobs at 0 (the default): embeddings, norms, biases and gates stay undecayed."""
    model = _decay_model()
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    groups = partition_optimizer_params(model, TrainConfig(optimizer="muon"))
    assert all(g.decay_target == 0.0 for g in groups.adamw_groups)
    assert sorted(g.weight_decay for g in groups.adamw_groups) == [0.0, 0.01]  # the usual two
    decay = _decay_of(groups, id_to_name)
    assert all(
        decay[n] == (0.0, 0.0) for n in decay if "embed" in n or n.endswith(("bias", "gate"))
    )


def test_decay_knobs_route_each_param_to_its_target() -> None:
    model = _decay_model()
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    cfg = TrainConfig(optimizer="muon", embedding_weight_decay=0.02, weight_decay_to_init=0.05)
    decay = _decay_of(partition_optimizer_params(model, cfg), id_to_name)
    assert decay["oplm.backbone.embed_tokens.embed_tokens.weight"] == (0.02, 0.0)
    for name, wd_target in decay.items():
        if name.endswith("norm.weight"):
            assert wd_target == (0.05, 1.0), name
        elif name.endswith(("attn_gate", "ffn_gate")):
            assert wd_target == (0.05, 0.5), name  # model.residual_gate_init
        elif name.endswith(".bias"):
            assert wd_target == (0.05, 0.0), name
        elif name.endswith(("value_residual_lambda", "qk_l2_scale")):
            assert wd_target == (0.0, 0.0), name  # never decayed
    assert decay["lm_head.dense.weight"] == (0.01, 0.0)  # weight_decay, unchanged


def test_decay_pulls_each_param_toward_its_target() -> None:
    """With zero gradients Adam's update is 0, so one step is exactly the decay."""
    from oplm.training.optim import build_optimizers

    model = _decay_model()
    cfg = TrainConfig(
        optimizer="muon",
        muon_adjust_lr_fn="original",
        lr=0.1,
        embedding_weight_decay=0.5,
        weight_decay_to_init=0.5,
    )
    _, adamw = build_optimizers(model, cfg)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for p in params.values():
            p.fill_(3.0)
            p.grad = torch.zeros_like(p)
    adamw.step()
    shrink = 1 - 0.1 * 0.5
    expected = {
        "oplm.backbone.embed_tokens.embed_tokens.weight": 3.0 * shrink,
        "oplm.backbone.layers.0.attn_norm.weight": 1.0 + 2.0 * shrink,
        "oplm.backbone.layers.0.attn_norm.bias": 3.0 * shrink,
        "oplm.backbone.layers.0.attn_gate": 0.5 + 2.5 * shrink,
        "oplm.backbone.layers.1.attention.value_residual_lambda": 3.0,
    }
    for name, value in expected.items():
        torch.testing.assert_close(params[name], torch.full_like(params[name], value), msg=name)


def test_optimizer_state_from_before_decay_targets_loads() -> None:
    """Checkpoints saved before ``decay_target`` existed have no such key in their groups."""
    from oplm.training.optim import build_optimizers

    model = _decay_model()
    cfg = TrainConfig(optimizer="muon", muon_adjust_lr_fn="original")
    _, adamw = build_optimizers(model, cfg)
    state = adamw.state_dict()
    for group in state["param_groups"]:
        del group["decay_target"]
    _, fresh = build_optimizers(model, cfg)
    fresh.load_state_dict(state)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    fresh.step()


@pytest.mark.parametrize("name", ["embedding_weight_decay", "weight_decay_to_init"])
def test_decay_knobs_reject_negative_values(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        TrainConfig(**{name: -0.1})


def test_group_structure_depends_on_which_knobs_are_on_not_their_values() -> None:
    """Equal decay values must not merge groups: a branch that changes one value would then
    change the group structure and no longer load the checkpoint's optimizer state."""
    model = _decay_model()
    shapes = []
    for value in (0.01, 0.03):  # 0.01 equals weight_decay
        cfg = TrainConfig(
            optimizer="muon", embedding_weight_decay=value, weight_decay_to_init=value
        )
        groups = partition_optimizer_params(model, cfg).adamw_groups
        shapes.append([(len(g.params), g.decay_target) for g in groups])
    assert shapes[0] == shapes[1]

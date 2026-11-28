"""Shared layer schedule and execution contracts."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import pytest
import torch

from oplm.model import OplmConfig
from oplm.model.looping import resolve_layer_execution_order
from oplm.model.transformer import OplmStack

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("strategy", "start", "end", "expected"),
    [
        ("stack", 0, None, (0, 1, 2, 3, 0, 1, 2, 3)),
        ("interleave", 0, None, (0, 0, 1, 1, 2, 2, 3, 3)),
        ("stack", 1, 3, (0, 1, 2, 1, 2, 3)),
        ("interleave", 1, 3, (0, 1, 1, 2, 2, 3)),
        ("stack", 0, 1, (0, 0, 1, 2, 3)),
        ("interleave", 3, 4, (0, 1, 2, 3, 3)),
    ],
)
def test_execution_orders(
    strategy: str, start: int, end: int | None, expected: tuple[int, ...]
) -> None:
    assert (
        resolve_layer_execution_order(
            4, num_loops=2, loop_strategy=strategy, loop_start=start, loop_end=end
        )
        == expected
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_loops": True},
        {"num_loops": 1.5},
        {"num_loops": 0},
        {"loop_start": -1},
        {"loop_start": True},
        {"loop_start": 0.5},
        {"loop_end": False},
        {"loop_end": 2.5},
        {"loop_end": 0},
        {"loop_end": 5},
        {"loop_start": 3, "loop_end": 2},
        {"loop_strategy": "unknown"},
    ],
)
def test_invalid_configuration_and_postconstruction_mutation(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="loop"):
        OplmConfig(num_hidden_layers=4, **overrides)
    cfg = OplmConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=4)
    for name, value in overrides.items():
        setattr(cfg, name, value)
    with pytest.raises(ValueError, match="loop"):
        OplmStack(cfg)


@pytest.mark.parametrize("strategy", ["stack", "interleave"])
@pytest.mark.parametrize(("start", "end"), [(0, None), (0, 1), (1, 3), (3, 4)])
def test_one_loop_is_ordinary(strategy: str, start: int, end: int | None) -> None:
    assert resolve_layer_execution_order(
        4, loop_strategy=strategy, loop_start=start, loop_end=end
    ) == (0, 1, 2, 3)


def test_loop_config_round_trip(tmp_path: Path) -> None:
    from oplm.config import load_config, serialize_config

    cfg = load_config(
        [
            "model.num_hidden_layers=4",
            "model.num_loops=3",
            "model.loop_strategy=interleave",
            "model.loop_start=1",
            "model.loop_end=null",
        ]
    )
    cfg.model.save_pretrained(tmp_path / "hf")
    loaded_model = OplmConfig.from_pretrained(tmp_path / "hf")
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(serialize_config(cfg))
    loaded_run = load_config(["--config", str(yaml_path)])
    for model in (loaded_model, loaded_run.model):
        assert (model.num_loops, model.loop_strategy, model.loop_start, model.loop_end) == (
            3,
            "interleave",
            1,
            None,
        )
        assert "layer_execution_order" not in model.to_dict()


@pytest.mark.parametrize(
    "override",
    [
        "model.num_loops=true",
        "model.num_loops=1.5",
        "model.loop_start=true",
        "model.loop_end=2.5",
        "model.loop_end=false",
    ],
)
def test_cli_rejects_noninteger_loop_fields(override: str) -> None:
    from oplm.config import load_config

    with pytest.raises(ValueError, match="loop"):
        load_config([override])


@pytest.mark.parametrize(
    ("strategy", "start", "end", "order"),
    [
        ("stack", 0, None, (0, 1, 2, 0, 1, 2)),
        ("interleave", 0, None, (0, 0, 1, 1, 2, 2)),
        ("stack", 1, 3, (0, 1, 2, 1, 2)),
        ("interleave", 1, 3, (0, 1, 1, 2, 2)),
    ],
)
@pytest.mark.parametrize("value_mode", ["none", "fixed", "learnable"])
@pytest.mark.parametrize("variant", ["pre", "sandwich", "hybrid", "post_sdpa", "canon"])
def test_forward_and_gradients_match_unrolled_reference(
    strategy: str,
    start: int,
    end: int | None,
    order: tuple[int, ...],
    value_mode: str,
    variant: str,
) -> None:
    cfg = OplmConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=3,
        num_loops=2,
        loop_strategy=strategy,
        loop_start=start,
        loop_end=end,
        value_residual=value_mode,
    )
    if variant == "canon":
        cfg = OplmConfig(
            **(
                cfg.to_dict()
                | {
                    "canon_enabled": True,
                    "canon_positions": ["A", "B", "C", "D"],
                    "canon_kernel_sizes": [3, 5, 3],
                    "residual_gate": "channel",
                }
            )
        )
    else:
        cfg.norm_strategy = variant
        cfg.residual_gate = "scalar"
    actual = OplmStack(cfg).train()
    reference = copy.deepcopy(actual)
    ids = torch.tensor([[0, 20, 9, 9, 14, 16, 2, 1]])  # MEEPQ plus pad
    mask = ids.ne(1).long()
    expected = reference.embed_tokens(ids, mask)
    anchor = None
    for index in order:
        result = reference.layers[index](
            expected,
            mask,
            output_attentions=True,
            value_residual=None if index == 0 else anchor,
        )
        expected = result[0]
        if value_mode != "none" and anchor is None:
            anchor = result[2]
    expected = reference.final_norm(expected)
    got, states, attentions = actual(ids, mask, output_hidden_states=True, output_attentions=True)
    assert states is not None and len(states) == len(order) + 1
    assert attentions is not None and len(attentions) == len(order)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
    probe = torch.randn_like(got)
    (got * probe).sum().backward()
    (expected * probe).sum().backward()
    for (name, param), (ref_name, ref_param) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == ref_name
        assert (param.grad is None) == (ref_param.grad is None)
        if param.grad is not None:
            torch.testing.assert_close(param.grad, ref_param.grad, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("strategy", ["stack", "interleave"])
@pytest.mark.parametrize("mode", ["full", "selective"])
def test_checkpointing_preserves_dropout_loss_and_gradients(strategy: str, mode: str) -> None:
    from oplm.model import OplmForMaskedLM

    cfg = OplmConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=3,
        num_loops=2,
        loop_strategy=strategy,
        value_residual="learnable",
        hidden_dropout=0.15,
        attention_dropout=0.1,
    )
    plain = OplmForMaskedLM(cfg).train()
    checkpointed = copy.deepcopy(plain)
    checkpointed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"mode": mode})
    ids = torch.tensor([[0, 20, 9, 32, 14, 16, 2, 1]])
    labels = torch.full_like(ids, -100)
    labels[0, 3] = 9
    mask = ids.ne(1).long()
    torch.manual_seed(2026)
    a = plain(input_ids=ids, attention_mask=mask, labels=labels).loss
    a.backward()
    torch.manual_seed(2026)
    b = checkpointed(input_ids=ids, attention_mask=mask, labels=labels).loss
    b.backward()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for name, param in plain.named_parameters():
        other = checkpointed.get_parameter(name)
        assert (param.grad is None) == (other.grad is None)
        if param.grad is not None:
            torch.testing.assert_close(param.grad, other.grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_looping_does_not_change_initialized_state_or_optimizer_parameters(optimizer: str) -> None:
    from oplm.config import TrainConfig
    from oplm.model import OplmForMaskedLM
    from oplm.training.optim import build_optimizers

    cfg = OplmConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=3)
    torch.manual_seed(7)
    ordinary = OplmForMaskedLM(cfg)
    cfg = copy.deepcopy(cfg)
    cfg.num_loops = 2
    torch.manual_seed(7)
    looped = OplmForMaskedLM(cfg)
    assert ordinary.state_dict().keys() == looped.state_dict().keys()
    for name, value in ordinary.state_dict().items():
        assert torch.equal(value, looped.state_dict()[name]), name
    opts = build_optimizers(looped, TrainConfig(optimizer=optimizer))
    identities = [id(p) for opt in opts for group in opt.param_groups for p in group["params"]]
    assert len(identities) == len(set(identities))
    assert set(identities) == {id(p) for p in looped.parameters() if p.requires_grad}
    assert len(looped.oplm.backbone.layers) == 3


@pytest.mark.parametrize("strategy", ["stack", "interleave"])
def test_one_loop_matches_original_traversal_exactly(strategy: str) -> None:
    cfg = OplmConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=3,
        loop_strategy=strategy,
        loop_start=1,
        loop_end=2,
    )
    stack = OplmStack(cfg).eval()
    ids = torch.tensor([[0, 20, 9, 9, 14, 16, 2, 1]])
    mask = ids.ne(1).long()
    expected = stack.embed_tokens(ids, mask)
    for block in stack.layers:
        expected = block(expected, mask)[0]
    expected = stack.final_norm(expected)
    got = stack(input_ids=ids, attention_mask=mask)[0]
    assert torch.equal(got, expected)
    embedded = stack.embed_tokens(ids, mask)
    assert torch.equal(stack(inputs_embeds=embedded, attention_mask=mask)[0], got)


def test_resolve_loop_schedule_partitions_prelude_core_coda() -> None:
    from oplm.model.looping import LoopSchedule, resolve_loop_schedule

    sched = resolve_loop_schedule(6, num_loops=3, loop_strategy="stack", loop_start=1, loop_end=5)
    assert isinstance(sched, LoopSchedule)
    assert sched.prelude == (0,)
    assert sched.core == (1, 2, 3, 4)
    assert sched.coda == (5,)
    assert sched.num_loops == 3
    assert sched.loop_strategy == "stack"
    assert sched.execution_order == (0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5)


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_loops": 2, "loop_input_injection": "bad"},
        {"num_loops": 2, "loop_gate": "bad"},
        {"num_loops": 2, "loop_step_conditioning": "bad"},
        {"num_loops": 2, "loop_gate": "scalar", "loop_gate_init": float("nan")},
        {"num_loops": 2, "loop_gate": "scalar", "loop_gate_init": float("inf")},
        {"num_loops": 2, "loop_untie_norms": 1},
        {"num_loops": 1, "loop_input_injection": "decayed"},
        {"num_loops": 1, "loop_gate": "scalar"},
        {"num_loops": 1, "loop_untie_norms": True},
        {"num_loops": 1, "loop_step_conditioning": "sinusoidal"},
        {"num_loops": 2, "loop_strategy": "interleave", "loop_input_injection": "orthogonal"},
        {"num_loops": 2, "loop_strategy": "interleave", "loop_gate": "channel"},
        {"num_loops": 2, "loop_strategy": "interleave", "loop_untie_norms": True},
        {"num_loops": 2, "loop_strategy": "interleave", "loop_step_conditioning": "sinusoidal"},
    ],
)
def test_loop_extension_validation_rejects_invalid_values_and_combos(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="loop"):
        OplmConfig(num_hidden_layers=4, **overrides)


def test_loop_extension_config_round_trip(tmp_path: Path) -> None:
    from oplm.config import load_config, serialize_config

    cfg = load_config(
        [
            "model.num_hidden_layers=4",
            "model.num_loops=2",
            "model.loop_strategy=stack",
            "model.loop_start=1",
            "model.loop_end=3",
            "model.loop_input_injection=decayed",
            "model.loop_gate=channel",
            "model.loop_gate_init=0.0",
            "model.loop_untie_norms=true",
            "model.loop_step_conditioning=sinusoidal",
        ]
    )
    assert cfg.model.has_loop_extensions is True
    cfg.model.save_pretrained(tmp_path / "hf")
    loaded_model = OplmConfig.from_pretrained(tmp_path / "hf")
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(serialize_config(cfg))
    loaded_run = load_config(["--config", str(yaml_path)])
    for model in (loaded_model, loaded_run.model):
        assert model.loop_input_injection == "decayed"
        assert model.loop_gate == "channel"
        assert model.loop_gate_init == 0.0
        assert model.loop_untie_norms is True
        assert model.loop_step_conditioning == "sinusoidal"
        assert model.has_loop_extensions is True


def test_mup_fanin_mult_concat_proj_width() -> None:
    base = OplmConfig(hidden_size=768, num_attention_heads=12, mup_enable=True, mup_base_width=768)
    wide = OplmConfig(hidden_size=1536, num_attention_heads=24, mup_enable=True, mup_base_width=768)
    assert base.mup_fanin_mult(1536) == pytest.approx(1.0)
    assert wide.mup_fanin_mult(3072) == pytest.approx(2.0)


def test_default_loop_extensions_are_bit_identical_to_vanilla() -> None:
    cfg = OplmConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=4,
        num_loops=2,
        loop_start=1,
        loop_end=3,
        norm_strategy="sandwich",
        value_residual="learnable",
    )
    torch.manual_seed(11)
    stack = OplmStack(cfg).eval()
    keys = set(stack.state_dict())
    assert not any(
        tag in k
        for k in keys
        for tag in ("loop_injector", "loop_gate_module", "loop_step_cond", "loop_attn_norms")
    )
    assert stack.loop_injector is None
    assert stack.loop_gate_module is None
    assert stack.loop_step_cond is None


@pytest.mark.parametrize("mode", ["none", "concat_proj", "decayed", "orthogonal"])
def test_zero_init_loop_gate_matches_unlooped_forward_exactly(mode: str) -> None:
    from oplm.model import OplmForMaskedLM

    base_cfg = OplmConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=4,
        norm_strategy="sandwich",
        value_residual="learnable",
    )
    torch.manual_seed(12)
    unlooped = OplmForMaskedLM(base_cfg).eval()

    ext_cfg = OplmConfig(
        **(
            base_cfg.to_dict()
            | {
                "num_loops": 3,
                "loop_strategy": "stack",
                "loop_start": 1,
                "loop_end": 3,
                "loop_input_injection": mode,
                "loop_gate": "channel",
                "loop_gate_init": 0.0,
                "loop_untie_norms": True,
                "loop_step_conditioning": "sinusoidal",
            }
        )
    )
    looped = OplmForMaskedLM(ext_cfg).eval()
    looped.load_state_dict(unlooped.state_dict(), strict=False)

    ids = torch.tensor([[0, 20, 9, 32, 14, 16, 2, 1]])
    mask = ids.ne(1).long()
    with torch.no_grad():
        logits_unlooped = unlooped(input_ids=ids, attention_mask=mask).logits
        logits_looped = looped(input_ids=ids, attention_mask=mask).logits
    assert torch.equal(logits_looped, logits_unlooped)


def test_untied_norms_preserve_pass0_keys_and_skip_prelude_coda() -> None:
    cfg = OplmConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=4,
        num_loops=3,
        loop_strategy="stack",
        loop_start=1,
        loop_end=3,
        norm_strategy="sandwich",
        value_residual="learnable",
        loop_untie_norms=True,
    )
    stack = OplmStack(cfg)
    for idx in (0, 3):  # prelude and coda
        blk = stack.layers[idx]
        assert not hasattr(blk, "loop_attn_norms")
        assert not hasattr(blk, "loop_ffn_norms")
        assert not hasattr(blk, "loop_attn_post_norms")
        assert not hasattr(blk, "loop_ffn_post_norms")
        assert not hasattr(blk.attention, "loop_value_residual_lambdas")
    for idx in (1, 2):  # core
        blk = stack.layers[idx]
        assert hasattr(blk, "attn_norm") and len(blk.loop_attn_norms) == 2
        assert hasattr(blk, "ffn_norm") and len(blk.loop_ffn_norms) == 2
        assert hasattr(blk, "attn_post_norm") and len(blk.loop_attn_post_norms) == 2
        assert hasattr(blk, "ffn_post_norm") and len(blk.loop_ffn_post_norms) == 2
        assert hasattr(blk.attention, "value_residual_lambda")
        assert len(blk.attention.loop_value_residual_lambdas) == 2


@pytest.mark.parametrize("inj", ["none", "concat_proj", "decayed", "orthogonal"])
@pytest.mark.parametrize("gate", ["none", "scalar", "channel"])
@pytest.mark.parametrize("untie", [False, True])
@pytest.mark.parametrize("step_cond", ["none", "sinusoidal"])
def test_step2_extensions_forward_and_gradients_match_manual_reference(
    inj: str, gate: str, untie: bool, step_cond: str
) -> None:
    cfg = OplmConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=4,
        num_loops=3,
        loop_strategy="stack",
        loop_start=1,
        loop_end=3,
        norm_strategy="sandwich",
        value_residual="learnable",
        loop_input_injection=inj,
        loop_gate=gate,
        loop_gate_init=0.7,
        loop_untie_norms=untie,
        loop_step_conditioning=step_cond,
    )
    torch.manual_seed(13)
    actual = OplmStack(cfg).train()
    if actual.loop_step_cond is not None:
        with torch.no_grad():
            actual.loop_step_cond.fc2.weight.normal_(std=0.05)
    reference = copy.deepcopy(actual)

    ids = torch.tensor([[0, 20, 9, 14, 16, 2, 1]])
    mask = ids.ne(1).long()

    # Manual reference unroll
    x = reference.embed_tokens(ids, mask)
    expected_states = [x]
    v1 = None
    # Prelude: layer 0
    x, _, v1 = reference.layers[0](x, mask, output_attentions=False, value_residual=None)
    expected_states.append(x)
    e = x
    # Core passes 0, 1, 2 over layers 1, 2
    for pass_idx in range(3):
        z_prev = x
        if pass_idx >= 1:
            if reference.loop_injector is not None:
                x = reference.loop_injector(z_prev, e)
            if reference.loop_step_cond is not None:
                x = reference.loop_step_cond(x, pass_idx, 3)
        for core_pos, layer_idx in enumerate((1, 2)):
            x, _, _ = reference.layers[layer_idx](
                x, mask, output_attentions=False, value_residual=v1, pass_idx=pass_idx
            )
            if pass_idx >= 1 and core_pos == 1 and reference.loop_gate_module is not None:
                x = reference.loop_gate_module(z_prev, x, pass_idx - 1)
            expected_states.append(x)
    # Coda: layer 3
    x, _, _ = reference.layers[3](x, mask, output_attentions=False, value_residual=v1)
    expected_states.append(x)
    expected_last = reference.final_norm(x)

    got_last, got_states, _ = actual(ids, mask, output_hidden_states=True)
    assert got_states is not None
    assert len(got_states) == len(actual.layer_execution_order) + 1 == len(expected_states)
    torch.testing.assert_close(got_last, expected_last, rtol=1e-5, atol=1e-6)
    for got_h, exp_h in zip(got_states, expected_states, strict=True):
        torch.testing.assert_close(got_h, exp_h, rtol=1e-5, atol=1e-6)

    probe = torch.randn_like(got_last)
    (got_last * probe).sum().backward()
    (expected_last * probe).sum().backward()
    for (name, param), (ref_name, ref_param) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == ref_name
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert ref_param.grad is not None
        torch.testing.assert_close(param.grad, ref_param.grad, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("mode", ["full", "selective"])
def test_step2_extensions_checkpointing_parity(mode: str) -> None:
    from oplm.model import OplmForMaskedLM

    cfg = OplmConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=4,
        num_loops=3,
        loop_strategy="stack",
        loop_start=1,
        loop_end=3,
        norm_strategy="sandwich",
        value_residual="learnable",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        loop_input_injection="decayed",
        loop_gate="channel",
        loop_gate_init=0.6,
        loop_untie_norms=True,
        loop_step_conditioning="sinusoidal",
    )
    torch.manual_seed(14)
    plain = OplmForMaskedLM(cfg).train()
    with torch.no_grad():
        plain.oplm.backbone.loop_step_cond.fc2.weight.normal_(std=0.05)
    ckpt = copy.deepcopy(plain)
    ckpt.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"mode": mode})

    ids = torch.tensor([[0, 20, 9, 32, 14, 16, 2, 1]])
    labels = torch.full_like(ids, -100)
    labels[0, 3] = 9
    mask = ids.ne(1).long()

    torch.manual_seed(2026)
    out_a = plain(
        input_ids=ids,
        attention_mask=mask,
        labels=labels,
        return_hidden_layers=(2, 4, 6),
        return_qk_layers=(2, 4),
    )
    (out_a.loss + sum(h.pow(2).mean() for h in out_a.hidden_states)).backward()

    torch.manual_seed(2026)
    out_b = ckpt(
        input_ids=ids,
        attention_mask=mask,
        labels=labels,
        return_hidden_layers=(2, 4, 6),
        return_qk_layers=(2, 4),
    )
    (out_b.loss + sum(h.pow(2).mean() for h in out_b.hidden_states)).backward()

    torch.testing.assert_close(out_a.loss, out_b.loss, rtol=0, atol=0)
    for ha, hb in zip(out_a.hidden_states, out_b.hidden_states, strict=True):
        torch.testing.assert_close(ha, hb, rtol=0, atol=0)
    for (qa, ka), (qb, kb) in zip(out_a.qk_states, out_b.qk_states, strict=True):
        torch.testing.assert_close(qa, qb, rtol=0, atol=0)
        torch.testing.assert_close(ka, kb, rtol=0, atol=0)
    for name, param in plain.named_parameters():
        other = ckpt.get_parameter(name)
        assert (param.grad is None) == (other.grad is None), name
        if param.grad is not None:
            torch.testing.assert_close(param.grad, other.grad, rtol=1e-4, atol=1e-6)

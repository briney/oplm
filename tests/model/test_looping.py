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

"""Strict local pretrained initialization preserves all model state."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from oplm.model import OplmConfig, OplmForMaskedLM
from oplm.training.initialization import (
    load_initial_model,
    resolve_initialization_source,
    validate_initialization_config,
)

if TYPE_CHECKING:
    from pathlib import Path


def _config(**overrides: Any) -> OplmConfig:
    return OplmConfig(
        **(
            dict(hidden_size=32, intermediate_size=64, num_attention_heads=4, num_hidden_layers=3)
            | overrides
        )
    )


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("strategy", ["stack", "interleave"])
@pytest.mark.parametrize("sharded", [False, True])
def test_strict_load_preserves_all_tensors(
    tmp_path: Path, tied: bool, strategy: str, sharded: bool
) -> None:
    cfg = _config(tie_word_embeddings=tied, value_residual="fixed", residual_gate="channel")
    source = OplmForMaskedLM(cfg)
    source.save_pretrained(tmp_path, max_shard_size="10KB" if sharded else "5GB")
    target = copy.deepcopy(cfg)
    target.num_loops = 2
    target.loop_strategy = strategy
    target.loop_start = 1
    loaded = load_initial_model(tmp_path, target)
    assert len(loaded.oplm.backbone.layer_execution_order) == 5
    for name, value in source.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name
    assert (loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight) == tied


@pytest.mark.parametrize(
    "missing_key",
    [
        "oplm.backbone.layers.0.alpha",
        "oplm.backbone.layers.1.ffn.down_proj.weight",
    ],
)
def test_missing_independent_tensor_fails(tmp_path: Path, missing_key: str) -> None:
    cfg = _config()
    OplmForMaskedLM(cfg).save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    del tensors[missing_key]
    save_file(tensors, path, metadata={"format": "pt"})
    with pytest.raises(ValueError, match=missing_key.replace(".", r"\.")):
        load_initial_model(tmp_path, cfg)


@pytest.mark.parametrize("corruption", ["unexpected", "shape", "both_tied_aliases"])
def test_corrupt_checkpoint_rejected(tmp_path: Path, corruption: str) -> None:
    cfg = _config(tie_word_embeddings=True)
    OplmForMaskedLM(cfg).save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    if corruption == "unexpected":
        tensors["bogus.weight"] = torch.zeros(2)
    elif corruption == "shape":
        tensors["oplm.backbone.layers.1.ffn.down_proj.weight"] = torch.zeros(2, 2)
    else:
        tensors.pop("lm_head.decoder.weight", None)
        tensors.pop("oplm.backbone.embed_tokens.embed_tokens.weight", None)
    save_file(tensors, path, metadata={"format": "pt"})
    with pytest.raises(ValueError):
        load_initial_model(tmp_path, cfg)


@pytest.mark.parametrize(
    "change",
    [
        {"num_attention_heads": 8, "head_dim": 4, "rope_dim": 4},
        {"norm_strategy": "sandwich"},
        {"rope_theta": 20000.0},
        {"residual_scaling": "none"},
        {"hidden_dropout": 0.1},
        {"tie_word_embeddings": True},
        {"norm_eps": 1e-4},
    ],
)
def test_same_shape_semantic_mismatches_fail(change: dict[str, Any]) -> None:
    source = _config()
    target = _config(**change)
    with pytest.raises(ValueError, match=next(iter(change))):
        validate_initialization_config(source, target)


@pytest.mark.parametrize(
    "field,value", [("mup_output_mult", 2.0), ("canon_kernel_sizes", [3, 5, 3])]
)
def test_active_optional_semantics_are_compared(field: str, value: Any) -> None:
    source = _config(
        mup_enable=True, canon_enabled=True, canon_positions=["A"], canon_kernel_sizes=3
    )
    target = copy.deepcopy(source)
    setattr(target, field, value)
    with pytest.raises(ValueError, match=field):
        validate_initialization_config(source, target)


def test_ignored_and_inactive_settings_do_not_block_loading() -> None:
    source = _config(qk_norm=False)
    target = _config(
        qk_norm=False,
        qk_norm_mode="l2",
        num_loops=2,
        loop_strategy="interleave",
        gradient_checkpointing=True,
        gradient_checkpointing_mode="selective",
        initializer_range=0.01,
        residual_gate_init=0.5,
        value_residual_lambda_init=0.3,
        qk_norm_l2_scale_init=0.4,
        init_scale_output_projections=False,
        canon_kernel_sizes=7,
        canon_positions=["C"],
        mup_base_width=1024,
        mup_output_mult=2.0,
        mask_dropout_reference_ratio=0.2,
        classifier_dropout=0.2,
        classifier_pool="cls",
        num_labels=3,
        pre_head_norm=True,
    )
    validate_initialization_config(source, target)


def test_packed_attention_kernel_choice_is_ignored_for_initialization() -> None:
    """packed_attention selects the execution kernel, not model semantics."""
    source = _config(packed_attention="sdpa")
    target = _config(packed_attention="flex")
    validate_initialization_config(source, target)


def test_packed_attention_reverse_kernel_choice_is_ignored() -> None:
    """Verify the reverse direction: flex source, sdpa target."""
    source = _config(packed_attention="flex")
    target = _config(packed_attention="sdpa")
    validate_initialization_config(source, target)


def test_active_canon_position_order_is_semantically_equivalent() -> None:
    a = _config(canon_enabled=True, canon_positions=["A", "C"], canon_kernel_sizes=3)
    b = _config(canon_enabled=True, canon_positions=["C", "A"], canon_kernel_sizes=[3, 3, 3])
    validate_initialization_config(a, b)


def test_source_resolves_export_or_checkpoint_root(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint-2"
    export = root / "hf"
    export.mkdir(parents=True)
    _config().save_pretrained(export)
    assert resolve_initialization_source(str(export)) == export.resolve()
    assert resolve_initialization_source(str(root)) == export.resolve()
    _config().save_pretrained(root)
    assert resolve_initialization_source(str(root)) == root.resolve()


def test_missing_local_source_does_not_fall_back_to_hub(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_initialization_source(str(tmp_path / "organization" / "model"))


@pytest.mark.parametrize("conflicting", [False, True])
def test_explicit_tied_aliases_preserve_tying_or_reject_conflicting_weights(
    tmp_path: Path,
    conflicting: bool,
) -> None:
    cfg = _config(tie_word_embeddings=True)
    OplmForMaskedLM(cfg).save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    embedding = tensors["oplm.backbone.embed_tokens.embed_tokens.weight"]
    tensors["lm_head.decoder.weight"] = embedding.clone() + (1 if conflicting else 0)
    save_file(tensors, path, metadata={"format": "pt"})
    if conflicting:
        with pytest.raises(ValueError, match="tied"):
            load_initial_model(tmp_path, cfg)
    else:
        model = load_initial_model(tmp_path, cfg)
        assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
        torch.testing.assert_close(model.get_input_embeddings().weight, embedding, rtol=0, atol=0)


def test_warm_start_unlooped_to_step2_extensions_succeeds_and_copies_norms(
    tmp_path: Path,
) -> None:
    """Warm-starting into Step 2 copies pass-0 norms/lambdas and preserves step-0 output."""
    torch.manual_seed(11)
    src_cfg = _config(
        num_hidden_layers=4,
        norm_strategy="sandwich",
        norm_type="layernorm",
        value_residual="learnable",
    )
    source = OplmForMaskedLM(src_cfg).eval()
    # Mutate base norms and value_residual_lambda away from init so copy-from-pass-0 is observable.
    with torch.no_grad():
        for layer_idx in (1, 2):
            block = source.oplm.backbone.layers[layer_idx]
            block.attn_norm.weight.fill_(1.2 + 0.1 * layer_idx)
            block.attn_norm.bias.fill_(0.05 * layer_idx)
            block.ffn_norm.weight.fill_(0.9 + 0.1 * layer_idx)
            block.ffn_norm.bias.fill_(-0.04 * layer_idx)
            block.attn_post_norm.weight.fill_(1.1 + 0.1 * layer_idx)
            block.ffn_post_norm.weight.fill_(0.85 + 0.1 * layer_idx)
            block.attention.value_residual_lambda.fill_(0.37 + 0.1 * layer_idx)
    source.save_pretrained(tmp_path)

    target_cfg = _config(
        num_hidden_layers=4,
        norm_strategy="sandwich",
        norm_type="layernorm",
        value_residual="learnable",
        num_loops=3,
        loop_strategy="stack",
        loop_start=1,
        loop_end=3,
        loop_input_injection="decayed",
        loop_gate="channel",
        loop_gate_init=0.0,
        loop_untie_norms=True,
        loop_step_conditioning="sinusoidal",
    )
    loaded = load_initial_model(tmp_path, target_cfg).eval()

    # Every base weight matches source bit-for-bit.
    src_sd = source.state_dict()
    loaded_sd = loaded.state_dict()
    for name, value in src_sd.items():
        assert torch.equal(value, loaded_sd[name]), name

    # Newly created decayed injector & zero-init gates are initialized cleanly (no meta garbage).
    backbone = loaded.oplm.backbone
    assert torch.isfinite(backbone.loop_injector.log_delta).all()
    assert torch.equal(
        backbone.loop_injector.log_state_decay,
        torch.zeros_like(backbone.loop_injector.log_state_decay),
    )
    assert torch.equal(
        backbone.loop_injector.input_matrix_continuous,
        torch.eye(target_cfg.hidden_size),
    )
    for gate in backbone.loop_gate_module.gates:
        assert torch.equal(gate, torch.zeros_like(gate))
    assert torch.equal(
        backbone.loop_step_cond.fc2.weight,
        torch.zeros_like(backbone.loop_step_cond.fc2.weight),
    )
    assert torch.equal(
        backbone.loop_step_cond.fc2.bias,
        torch.zeros_like(backbone.loop_step_cond.fc2.bias),
    )

    # Every untied norm and value-residual lambda equals its pass-0 counterpart.
    for layer_idx in (1, 2):
        block = backbone.layers[layer_idx]
        for attr, loop_attr in (
            ("attn_norm", "loop_attn_norms"),
            ("ffn_norm", "loop_ffn_norms"),
            ("attn_post_norm", "loop_attn_post_norms"),
            ("ffn_post_norm", "loop_ffn_post_norms"),
        ):
            base_sd = getattr(block, attr).state_dict()
            for copy_mod in getattr(block, loop_attr):
                for k, v in base_sd.items():
                    assert torch.equal(copy_mod.state_dict()[k], v)
        for lam in block.attention.loop_value_residual_lambdas:
            assert torch.equal(lam, block.attention.value_residual_lambda)

    # With loop_gate_init=0.0, step-0 logits match the unlooped source checkpoint.
    input_ids = torch.tensor([[0, 5, 6, 7, 8, 2], [0, 9, 10, 11, 2, 1]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]])
    with torch.no_grad():
        src_logits = source(input_ids=input_ids, attention_mask=attention_mask).logits
        loaded_logits = loaded(input_ids=input_ids, attention_mask=attention_mask).logits
    torch.testing.assert_close(loaded_logits, src_logits, atol=1e-6, rtol=1e-6)


def test_warm_start_rejects_corrupt_missing_base_key_even_with_step2_extensions(
    tmp_path: Path,
) -> None:
    """Allowing newly enabled Step 2 keys must never forgive a missing base checkpoint weight."""
    src_cfg = _config(num_hidden_layers=4)
    OplmForMaskedLM(src_cfg).save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    del tensors["oplm.backbone.layers.1.ffn.down_proj.weight"]
    save_file(tensors, path, metadata={"format": "pt"})

    target_cfg = _config(
        num_hidden_layers=4,
        num_loops=3,
        loop_strategy="stack",
        loop_start=1,
        loop_end=3,
        loop_input_injection="decayed",
        loop_gate="channel",
        loop_gate_init=0.0,
        loop_untie_norms=True,
        loop_step_conditioning="sinusoidal",
    )
    with pytest.raises(ValueError, match=r"down_proj\.weight"):
        load_initial_model(tmp_path, target_cfg)


@pytest.mark.parametrize(
    ("source_overrides", "target_overrides"),
    [
        # Mode swap on active injection
        (
            {"num_loops": 2, "loop_start": 1, "loop_end": 3, "loop_input_injection": "decayed"},
            {"num_loops": 2, "loop_start": 1, "loop_end": 3, "loop_input_injection": "concat_proj"},
        ),
        # Disabling active injection
        (
            {"num_loops": 2, "loop_start": 1, "loop_end": 3, "loop_input_injection": "decayed"},
            {"num_loops": 2, "loop_start": 1, "loop_end": 3, "loop_input_injection": "none"},
        ),
        # Disabling active untied norms
        (
            {"num_loops": 2, "loop_start": 1, "loop_end": 3, "loop_untie_norms": True},
            {"num_loops": 2, "loop_start": 1, "loop_end": 3, "loop_untie_norms": False},
        ),
        # Disabling active step conditioning
        (
            {
                "num_loops": 2,
                "loop_start": 1,
                "loop_end": 3,
                "loop_step_conditioning": "sinusoidal",
            },
            {"num_loops": 2, "loop_start": 1, "loop_end": 3, "loop_step_conditioning": "none"},
        ),
        # Changing num_loops when source already has loop_gate
        (
            {"num_loops": 2, "loop_start": 1, "loop_end": 3, "loop_gate": "scalar"},
            {"num_loops": 3, "loop_start": 1, "loop_end": 3, "loop_gate": "scalar"},
        ),
        # Changing loop_start when source already has loop_gate
        (
            {"num_loops": 2, "loop_start": 1, "loop_end": 3, "loop_gate": "scalar"},
            {"num_loops": 2, "loop_start": 0, "loop_end": 3, "loop_gate": "scalar"},
        ),
        # Changing num_loops when source already has loop_untie_norms
        (
            {"num_loops": 2, "loop_start": 1, "loop_end": 3, "loop_untie_norms": True},
            {"num_loops": 3, "loop_start": 1, "loop_end": 3, "loop_untie_norms": True},
        ),
        # Changing loop_start when source already has loop_untie_norms
        (
            {"num_loops": 2, "loop_start": 1, "loop_end": 3, "loop_untie_norms": True},
            {"num_loops": 2, "loop_start": 0, "loop_end": 3, "loop_untie_norms": True},
        ),
    ],
)
def test_warm_start_rejects_invalid_step2_transitions(
    source_overrides: dict[str, Any], target_overrides: dict[str, Any]
) -> None:
    source = _config(num_hidden_layers=4, **source_overrides)
    target = _config(num_hidden_layers=4, **target_overrides)
    with pytest.raises(ValueError):
        validate_initialization_config(source, target)

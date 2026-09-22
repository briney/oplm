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

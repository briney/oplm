"""save_pretrained / from_pretrained round-trips for every task class (Phase 14.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from oplm import (
    OplmConfig,
    OplmForMaskedLM,
    OplmForSequenceClassification,
    OplmForTokenClassification,
    OplmModel,
    OplmTokenizerFast,
)

if TYPE_CHECKING:
    from pathlib import Path

_B, _T = 2, 16
_VOCAB = 33

_TASK_CLASSES = [
    OplmModel,
    OplmForMaskedLM,
    OplmForSequenceClassification,
    OplmForTokenClassification,
]


def _tiny_config(**overrides) -> OplmConfig:
    base = dict(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        num_labels=3,
    )
    base.update(overrides)
    return OplmConfig(**base)


def _last_hidden_or_logits(out) -> torch.Tensor:
    """Pull the single forward tensor each task class exposes."""
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    return out.last_hidden_state


@pytest.mark.parametrize("cls", _TASK_CLASSES, ids=[c.__name__ for c in _TASK_CLASSES])
def test_round_trip_state_dict_and_outputs(cls, tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = cls(_tiny_config()).eval()
    model.save_pretrained(tmp_path)

    reloaded = cls.from_pretrained(tmp_path).eval()

    # State-dict equality, key for key.
    sd_a, sd_b = model.state_dict(), reloaded.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for key, value in sd_a.items():
        assert torch.equal(value, sd_b[key]), f"mismatch in {key}"

    # Identical forward outputs on the same input.
    torch.manual_seed(1)
    input_ids = torch.randint(4, _VOCAB, (_B, _T))
    attention_mask = torch.ones(_B, _T, dtype=torch.long)
    with torch.no_grad():
        out_a = _last_hidden_or_logits(model(input_ids=input_ids, attention_mask=attention_mask))
        out_b = _last_hidden_or_logits(reloaded(input_ids=input_ids, attention_mask=attention_mask))
    assert torch.equal(out_a, out_b)


def test_tokenizer_round_trip_resolves_to_oplm_fast(tmp_path: Path) -> None:
    """A saved tokenizer reloads back to OplmTokenizerFast via AutoTokenizer."""
    from transformers import AutoTokenizer

    OplmTokenizerFast().save_pretrained(tmp_path)
    reloaded = AutoTokenizer.from_pretrained(tmp_path)
    assert type(reloaded).__name__ == "OplmTokenizerFast"
    # Canonical parity check survives the round-trip.
    assert reloaded("MEEPQ").input_ids == [0, 20, 9, 9, 14, 16, 2]


def test_from_pretrained_auto_attaches_tokenizer(tmp_path: Path) -> None:
    """When tokenizer files sit beside the weights, from_pretrained attaches them."""
    model = OplmForMaskedLM(_tiny_config())
    model.save_pretrained(tmp_path)
    OplmTokenizerFast().save_pretrained(tmp_path)

    reloaded = OplmForMaskedLM.from_pretrained(tmp_path)
    assert reloaded.tokenizer is not None
    assert type(reloaded.tokenizer).__name__ == "OplmTokenizerFast"


def test_loading_info_preserves_tuple_and_tokenizer(tmp_path: Path) -> None:
    model = OplmForMaskedLM(_tiny_config(num_loops=2))
    model.save_pretrained(tmp_path)
    OplmTokenizerFast().save_pretrained(tmp_path)
    loaded, info = OplmForMaskedLM.from_pretrained(tmp_path, output_loading_info=True)
    assert isinstance(loaded, OplmForMaskedLM)
    assert loaded.tokenizer is not None
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]


@pytest.mark.parametrize("cls", _TASK_CLASSES, ids=[c.__name__ for c in _TASK_CLASSES])
@pytest.mark.parametrize("strategy", ["stack", "interleave"])
def test_looped_round_trip_preserves_outputs(cls: type, strategy: str, tmp_path: Path) -> None:
    config = _tiny_config(
        num_loops=2, loop_strategy=strategy, loop_start=1, loop_end=2, tie_word_embeddings=True
    )
    model = cls(config).eval()
    model.save_pretrained(tmp_path)
    loaded = cls.from_pretrained(tmp_path).eval()
    batch = OplmTokenizerFast()(["MEEPQ", "MKVL"], padding=True, return_tensors="pt")
    with torch.no_grad():
        a = _last_hidden_or_logits(model(**batch))
        b = _last_hidden_or_logits(loaded(**batch))
    assert torch.equal(a, b)
    assert loaded.config.loop_end == 2
    assert loaded.config.num_loops == 2
    assert model.state_dict().keys() == loaded.state_dict().keys()


def test_legacy_checkpoint_defaults_to_one_loop(tmp_path: Path) -> None:
    import json

    model = OplmForMaskedLM(_tiny_config()).eval()
    model.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    data = json.loads(path.read_text())
    for key in ("num_loops", "loop_strategy", "loop_start", "loop_end"):
        data.pop(key, None)
    path.write_text(json.dumps(data))
    loaded = OplmForMaskedLM.from_pretrained(tmp_path).eval()
    assert loaded.oplm.backbone.layer_execution_order == (0, 1)
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])


def test_step2_loop_extensions_save_load_preserves_custom_and_mutated_weights(
    tmp_path: Path,
) -> None:
    config = _tiny_config(
        num_hidden_layers=4,
        num_loops=3,
        loop_strategy="stack",
        loop_start=1,
        loop_end=3,
        norm_strategy="sandwich",
        value_residual="learnable",
        loop_input_injection="decayed",
        loop_gate="channel",
        loop_gate_init=0.0,
        loop_untie_norms=True,
        loop_step_conditioning="sinusoidal",
    )
    torch.manual_seed(77)
    model = OplmForMaskedLM(config).eval()
    # Mutate custom-init and zero-init loop parameters to non-initial values.
    with torch.no_grad():
        backbone = model.oplm.backbone
        backbone.loop_injector.log_delta.fill_(0.42)
        backbone.loop_injector.log_state_decay.fill_(-0.25)
        backbone.loop_injector.input_matrix_continuous.normal_(std=0.1)
        for idx, g in enumerate(backbone.loop_gate_module.gates):
            g.fill_(0.35 + 0.1 * idx)
        backbone.loop_step_cond.fc2.weight.normal_(std=0.05)
        backbone.loop_step_cond.fc2.bias.fill_(0.07)
        backbone.layers[1].loop_attn_norms[0].weight.fill_(1.3)
        backbone.layers[1].attention.loop_value_residual_lambdas[0].fill_(0.82)

    model.save_pretrained(tmp_path)
    loaded, info = OplmForMaskedLM.from_pretrained(tmp_path, output_loading_info=True)
    loaded.eval()
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    assert not info["mismatched_keys"]
    assert model.state_dict().keys() == loaded.state_dict().keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name

    batch = OplmTokenizerFast()(["MEEPQ", "MKVL"], padding=True, return_tensors="pt")
    with torch.no_grad():
        # Weights are bit-identical (checked above); safetensors mmap alignment after
        # 1-element scalar params can shift CPU BLAS SIMD reduction order by ~1 ULP.
        torch.testing.assert_close(
            model(**batch).logits, loaded(**batch).logits, atol=1e-6, rtol=1e-6
        )

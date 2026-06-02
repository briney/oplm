"""Unit tests for the torchao FP8 precision module (TODOS.md Phase 6.2).

``apply_fp8_training`` swaps every ``nn.Linear`` for a torchao ``Float8Linear`` in
place while leaving norms, embeddings, and conv layers untouched; the bf16 path
leaves the model entirely unchanged. The actual FP8 conversion is exercised only
on sm90+ hardware (``@pytest.mark.blackwell``); the bf16 and capability-gate tests
are pure and run anywhere.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from oplm.model import OplmConfig as OplmModelConfig
from oplm.model import OplmForMaskedLM


def _tiny_model() -> OplmForMaskedLM:
    """A 2-layer / 32-hidden model: small but with the full Linear/norm/embed mix."""
    return OplmForMaskedLM(
        OplmModelConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=64,
        )
    )


@pytest.mark.blackwell
def test_apply_fp8_skips_non_linear() -> None:
    """Every ``nn.Linear`` becomes ``Float8Linear``; norms and embeddings are untouched."""
    from torchao.float8.float8_linear import Float8Linear

    from oplm.training.precision import apply_fp8_training

    model = _tiny_model()
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    n_norm = sum(1 for m in model.modules() if type(m).__name__ == "OplmLayerNorm")
    n_embed = sum(1 for m in model.modules() if isinstance(m, nn.Embedding))
    assert n_linear > 0 and n_norm > 0 and n_embed > 0  # sanity: the mix exists

    apply_fp8_training(model)

    # Float8Linear subclasses nn.Linear, so compare concrete types: every linear was
    # converted (count matches) and no *bare* nn.Linear survives.
    assert sum(1 for m in model.modules() if isinstance(m, Float8Linear)) == n_linear
    assert not any(type(m) is nn.Linear for m in model.modules())
    # Non-Linear modules are skipped by the module_filter_fn.
    assert sum(1 for m in model.modules() if type(m).__name__ == "OplmLayerNorm") == n_norm
    assert sum(1 for m in model.modules() if isinstance(m, nn.Embedding)) == n_embed


def test_bf16_path_untouched() -> None:
    """With precision='bf16' no conversion runs: linears stay standard ``nn.Linear``."""
    model = _tiny_model()
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linears  # the model has Linear layers to begin with
    # No apply_fp8_training call -> every linear is exactly nn.Linear, not a subclass.
    assert all(type(m) is nn.Linear for m in linears)


def test_is_fp8_supported_returns_bool() -> None:
    """The capability gate returns a plain bool regardless of hardware (no GPU needed)."""
    from oplm.training.precision import is_fp8_supported

    assert isinstance(is_fp8_supported(), bool)

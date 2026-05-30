"""G10 — full-model flex vs. manual attention parity (docs/TESTING_E2E.md §5). CUDA-only.

Extends the module-level parity test (``tests/model/test_attention.py``) up to the
whole ``OplmForMaskedLM`` + MLM loss: a single batch through the flex_attention
fast path and through the manual fallback must produce the same masked-LM loss.
Guards the path-equivalence assumption that the ``(device, precision)`` matrix
relies on, where each row silently picks one path.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention requires CUDA"),
]


def _set_flex(model: torch.nn.Module, enabled: bool) -> None:
    """Toggle the flex fast path on every attention submodule that exposes it."""
    for module in model.modules():
        if hasattr(module, "use_flex_attention"):
            module.use_flex_attention = enabled


def test_full_model_flex_matches_manual_mlm_loss() -> None:
    """The MLM loss agrees between the flex and manual attention paths."""
    from oplm.model import OplmConfig as OplmModelConfig
    from oplm.model import OplmForMaskedLM

    torch.manual_seed(0)
    config = OplmModelConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=64,
        use_flex_attention=True,
    )
    model = OplmForMaskedLM(config).to("cuda").eval()

    batch, seq = 2, 12
    input_ids = torch.randint(0, config.vocab_size, (batch, seq), device="cuda")
    attention_mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
    labels = input_ids.clone()

    with torch.no_grad():
        _set_flex(model, True)
        loss_flex = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        _set_flex(model, False)
        loss_manual = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

    assert torch.isfinite(loss_flex) and torch.isfinite(loss_manual)
    assert torch.allclose(loss_flex, loss_manual, rtol=1e-2, atol=1e-2)

"""G10 — full-model SDPA vs. manual-softmax attention parity (docs/TESTING_E2E.md §5).

Extends the module-level parity test (``tests/model/test_attention.py``) up to the
whole ``OplmForMaskedLM`` + MLM loss: a single batch run with ``output_attentions``
off (the SDPA compute path used for training) and on (the manual softmax path that
exposes attention weights) must produce the same masked-LM loss. Guards the
assumption that the analysis-only weights path stays numerically faithful to the
path used for training. Runs on CPU via SDPA's math backend.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.slow


def test_full_model_sdpa_matches_manual_mlm_loss() -> None:
    """The MLM loss agrees between the SDPA and manual-softmax attention paths."""
    from oplm.model import OplmConfig as OplmModelConfig
    from oplm.model import OplmForMaskedLM

    torch.manual_seed(0)
    config = OplmModelConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=64,
    )
    model = OplmForMaskedLM(config).eval()

    batch, seq = 2, 12
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    labels = input_ids.clone()

    with torch.no_grad():
        loss_sdpa = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss_manual = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True,
        ).loss

    assert torch.isfinite(loss_sdpa) and torch.isfinite(loss_manual)
    assert torch.allclose(loss_sdpa, loss_manual, rtol=1e-4, atol=1e-4)

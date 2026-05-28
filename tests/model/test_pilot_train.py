"""Pilot-scale training smoke test: the full graph learns end-to-end (Phase 14.6)."""

from __future__ import annotations

import pytest
import torch

from oplm import OplmConfig, OplmForMaskedLM

_VOCAB = 33


@pytest.mark.slow
def test_pilot_mlm_trains_five_steps() -> None:
    """A 4-layer MLM overfits a fixed synthetic batch over 5 AdamW steps.

    Training repeatedly on the same batch guarantees a monotone-ish loss drop if
    the graph is wired correctly; the assertion is the loose "final < initial,
    nothing NaNs" contract from the plan.
    """
    torch.manual_seed(0)
    config = OplmConfig(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=64,
    )
    model = OplmForMaskedLM(config).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch_size, seq_len = 4, 24
    input_ids = torch.randint(4, _VOCAB, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Mask ~15% of positions for the MLM objective; the rest are ignored (-100).
    labels = torch.full((batch_size, seq_len), -100)
    mask_positions = torch.rand(batch_size, seq_len) < 0.15
    labels[mask_positions] = input_ids[mask_positions]
    # Guarantee at least one supervised position.
    labels[0, 0] = input_ids[0, 0]

    losses: list[float] = []
    for _ in range(5):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert torch.isfinite(out.loss).item()
        out.loss.backward()
        optimizer.step()
        losses.append(out.loss.item())

    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"

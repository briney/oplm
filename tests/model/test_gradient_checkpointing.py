"""Gradient checkpointing is numerically transparent (Phase 14.7).

With and without `gradient_checkpointing`, a fixed-seed forward + backward must
produce matching outputs and matching parameter grads. Checkpointing only fires
in training mode, so the model is kept in `train()`; dropout defaults to 0 so the
forward is deterministic.
"""

from __future__ import annotations

import torch

from oplm import OplmConfig, OplmForMaskedLM

_B, _T = 2, 16
_VOCAB = 33


def _tiny_model() -> OplmForMaskedLM:
    config = OplmConfig(
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        max_position_embeddings=64,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    return OplmForMaskedLM(config)


def _forward_backward(model: OplmForMaskedLM) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(123)
    input_ids = torch.randint(4, _VOCAB, (_B, _T))
    attention_mask = torch.ones(_B, _T, dtype=torch.long)
    labels = torch.randint(0, _VOCAB, (_B, _T))

    model.zero_grad(set_to_none=True)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    out.loss.backward()
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    return out.logits.detach().clone(), grads


def test_checkpointing_matches_plain_forward_and_grads() -> None:
    model = _tiny_model().train()

    model.gradient_checkpointing_disable()
    logits_plain, grads_plain = _forward_backward(model)

    model.gradient_checkpointing_enable()
    logits_ckpt, grads_ckpt = _forward_backward(model)

    assert torch.allclose(logits_plain, logits_ckpt, atol=1e-5, rtol=1e-4)
    assert grads_plain.keys() == grads_ckpt.keys()
    for name in grads_plain:
        assert torch.allclose(grads_plain[name], grads_ckpt[name], atol=1e-5, rtol=1e-4), (
            f"grad mismatch in {name}"
        )

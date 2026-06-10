"""Gradient checkpointing is numerically transparent (Phase 14.7).

With and without `gradient_checkpointing`, a fixed-seed forward + backward must
produce matching outputs and matching parameter grads. Checkpointing only fires
in training mode, so the model is kept in `train()`; dropout defaults to 0 so the
forward is deterministic.
"""

from __future__ import annotations

import pytest
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


def _assert_transparent(
    logits_plain: torch.Tensor,
    grads_plain: dict[str, torch.Tensor],
    logits_ckpt: torch.Tensor,
    grads_ckpt: dict[str, torch.Tensor],
) -> None:
    assert torch.allclose(logits_plain, logits_ckpt, atol=1e-5, rtol=1e-4)
    assert grads_plain.keys() == grads_ckpt.keys()
    for name in grads_plain:
        assert torch.allclose(grads_plain[name], grads_ckpt[name], atol=1e-5, rtol=1e-4), (
            f"grad mismatch in {name}"
        )


def test_checkpointing_matches_plain_forward_and_grads() -> None:
    model = _tiny_model().train()

    model.gradient_checkpointing_disable()
    logits_plain, grads_plain = _forward_backward(model)

    model.gradient_checkpointing_enable()
    logits_ckpt, grads_ckpt = _forward_backward(model)

    _assert_transparent(logits_plain, grads_plain, logits_ckpt, grads_ckpt)


def test_selective_checkpointing_matches_plain_forward_and_grads() -> None:
    """Selective activation checkpointing must be numerically transparent too."""
    model = _tiny_model().train()

    model.gradient_checkpointing_disable()
    logits_plain, grads_plain = _forward_backward(model)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"mode": "selective"})
    logits_ckpt, grads_ckpt = _forward_backward(model)

    _assert_transparent(logits_plain, grads_plain, logits_ckpt, grads_ckpt)


def test_selective_mode_propagates_to_blocks() -> None:
    """Enabling with mode='selective' flips the flag on every block and the stack."""
    model = _tiny_model().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"mode": "selective"})

    stack = model.oplm.backbone
    assert stack.gradient_checkpointing_mode == "selective"
    assert all(block.gradient_checkpointing_mode == "selective" for block in stack.layers)
    assert all(block.gradient_checkpointing for block in stack.layers)


def test_invalid_checkpointing_mode_raises() -> None:
    """An unknown mode in gradient_checkpointing_kwargs is rejected early."""
    model = _tiny_model().train()
    with pytest.raises(ValueError, match="gradient_checkpointing mode"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"mode": "bogus"})


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="peak-memory probe needs CUDA")
def test_sac_peak_memory_between_none_and_full() -> None:
    """Selective checkpointing should sit between no-checkpointing and full on memory.

    Peak activation memory ordering: full <= selective <= none. Measured on CUDA
    via max_memory_allocated; CPU has no comparable peak probe so this is skipped
    off-GPU. A moderately sized model is used so the activation footprint dominates
    fixed parameter/optimizer memory enough for the ordering to be observable.
    """
    config = OplmConfig(
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        max_position_embeddings=512,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )

    def _peak(mode: str | None) -> int:
        model = OplmForMaskedLM(config).cuda().train()
        if mode is None:
            model.gradient_checkpointing_disable()
        else:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"mode": mode})
        torch.manual_seed(123)
        input_ids = torch.randint(4, _VOCAB, (4, 256), device="cuda")
        attention_mask = torch.ones(4, 256, dtype=torch.long, device="cuda")
        labels = torch.randint(0, _VOCAB, (4, 256), device="cuda")
        torch.cuda.reset_peak_memory_stats()
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        return torch.cuda.max_memory_allocated()

    peak_none = _peak(None)
    peak_selective = _peak("selective")
    peak_full = _peak("full")

    # Allow a small slack for allocator granularity / fixed overheads.
    slack = 1.05
    assert peak_full <= peak_selective * slack
    assert peak_selective <= peak_none * slack

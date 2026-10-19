"""Sequence packing: a packed row must give each sequence exactly what its own padded row gives."""

from __future__ import annotations

import pytest
import torch
import torch.nn.attention.flex_attention as flex_mod

import oplm.model.transformer as transformer_mod
from oplm import OplmConfig, OplmForMaskedLM
from oplm.model.masking import segment_attention_mask

LENGTHS = (5, 7, 3)


def test_segment_mask_is_block_diagonal_with_no_empty_rows() -> None:
    mask = segment_attention_mask(torch.tensor([[1, 1, 1, 2, 2, 3, 0, 0]]))[0, 0]
    assert mask.shape == (8, 8) and bool(mask.any(-1).all())
    assert mask[3].tolist() == [False, False, False, True, True, False, False, False]
    assert mask[7].tolist() == [False] * 6 + [True, True]


def _model(checkpointing: str | None, packed_attention: str = "sdpa") -> OplmForMaskedLM:
    torch.manual_seed(0)
    cfg = OplmConfig(
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
        num_hidden_layers=3,
        max_position_embeddings=64,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        rope_dim=4,  # partial RoPE: half of each 8-channel head rotates
        nope_dim=4,
        value_residual="learnable",
        canon_enabled=True,
        canon_positions=["A", "B", "C", "D"],
        canon_kernel_sizes=[3, 4, 7],  # odd and even kernels, one wider than a sequence
        packed_attention=packed_attention,
    )
    model = OplmForMaskedLM(cfg)
    if checkpointing is not None:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"mode": checkpointing})
    return model.train()


def _batches() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The same three labeled sequences, one per padded row and packed into one row."""
    gen = torch.Generator().manual_seed(1)
    seqs = [torch.randint(5, 25, (n,), generator=gen) for n in LENGTHS]
    labels = [
        torch.where(torch.rand(n, generator=gen) < 0.5, s, -100)
        for s, n in zip(seqs, LENGTHS, strict=True)
    ]
    width = max(LENGTHS) + 1
    padded = {
        "input_ids": torch.ones(len(seqs), width, dtype=torch.long),
        "attention_mask": torch.zeros(len(seqs), width, dtype=torch.long),
        "labels": torch.full((len(seqs), width), -100),
    }
    for row, (s, lab) in enumerate(zip(seqs, labels, strict=True)):
        padded["input_ids"][row, : len(s)] = s
        padded["attention_mask"][row, : len(s)] = 1
        padded["labels"][row, : len(s)] = lab
    total = sum(LENGTHS)
    pad = 2
    packed = {
        "input_ids": torch.cat([*seqs, torch.ones(pad, dtype=torch.long)])[None],
        "attention_mask": torch.cat([torch.ones(total), torch.zeros(pad)]).long()[None],
        "labels": torch.cat([*labels, torch.full((pad,), -100)])[None],
        "segment_ids": torch.cat(
            [torch.full((n,), i + 1) for i, n in enumerate(LENGTHS)]
            + [torch.zeros(pad, dtype=torch.long)]
        )[None],
    }
    return padded, packed


def _unpack(packed_logits: torch.Tensor) -> list[torch.Tensor]:
    starts = [sum(LENGTHS[:i]) for i in range(len(LENGTHS))]
    return [packed_logits[0, s : s + n] for s, n in zip(starts, LENGTHS, strict=True)]


@pytest.mark.parametrize("output_attentions", [False, True], ids=["sdpa", "manual"])
def test_packed_logits_match_padded_rows(output_attentions: bool) -> None:
    model = _model(None)
    padded, packed = _batches()
    ref = model(**padded, output_attentions=output_attentions)
    out = model(**packed, output_attentions=output_attentions)
    for row, got in enumerate(_unpack(out.logits)):
        torch.testing.assert_close(got, ref.logits[row, : LENGTHS[row]], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out.loss, ref.loss, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("checkpointing", [None, "full", "selective"])
def test_packed_gradients_match_padded_rows(checkpointing: str | None) -> None:
    padded, packed = _batches()
    grads = []
    for batch in (padded, packed):
        model = _model(checkpointing)
        model(**batch).loss.backward()
        grads.append({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    for name, ref in grads[0].items():
        torch.testing.assert_close(grads[1][name], ref, rtol=1e-4, atol=1e-6, msg=name)


def test_without_segment_ids_a_single_sequence_row_is_unchanged() -> None:
    """One sequence per row, packed or not, is today's computation: same ids, same output."""
    model = _model(None).eval()
    padded, _ = _batches()
    seg = padded["attention_mask"].clone()  # one segment per row, pads 0
    with torch.no_grad():
        ref = model(**padded).logits
        got = model(**padded, segment_ids=seg).logits
    mask = padded["attention_mask"].bool()
    torch.testing.assert_close(got[mask], ref[mask], rtol=1e-5, atol=1e-6)


def _spy_flex_kernel(monkeypatch: pytest.MonkeyPatch) -> list[tuple[torch.dtype, ...]]:
    """Record the q/k/v dtypes of every real FlexAttention kernel call (one per block).

    Parity alone cannot tell flex from a silent SDPA fallback: both compute the same thing.
    """
    calls: list[tuple[torch.dtype, ...]] = []
    real = flex_mod.flex_attention

    def spy(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.append((q.dtype, k.dtype, v.dtype))
        return real(q, k, v, *args, **kwargs)

    monkeypatch.setattr(flex_mod, "flex_attention", spy)
    return calls


@pytest.mark.parametrize("checkpointing", [None, "full", "selective"])
def test_flex_packed_logits_match_sdpa(
    checkpointing: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same weights, same packed batch: flex and the dense-mask SDPA path agree (forward only:
    FlexAttention has no CPU backward). Train mode so checkpointing wraps each block. The spy
    proves the flex kernel really ran (once per block) through each checkpointing wrapper."""
    calls = _spy_flex_kernel(monkeypatch)
    _, packed = _batches()
    with torch.no_grad():
        ref = _model(checkpointing, "sdpa")(**packed).logits
        assert calls == []
        flex_model = _model(checkpointing, "flex")
        got = flex_model(**packed).logits
    assert len(calls) == flex_model.config.num_hidden_layers
    mask = packed["attention_mask"].bool()
    torch.testing.assert_close(got[mask], ref[mask], rtol=1e-4, atol=1e-5)


def test_flex_packed_logits_match_padded_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy_flex_kernel(monkeypatch)
    padded, packed = _batches()
    model = _model(None, "flex").eval()
    with torch.no_grad():
        ref = model(**padded).logits
        assert calls == []  # no segment_ids: the padded batch never takes the flex path
        out = model(**packed).logits
    assert len(calls) == model.config.num_hidden_layers
    for row, got in enumerate(_unpack(out)):
        torch.testing.assert_close(got, ref[row, : LENGTHS[row]], rtol=1e-4, atol=1e-5)


def _count_block_masks(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = [0]
    real = transformer_mod.segment_block_mask

    def counting(segment_ids: torch.Tensor):  # noqa: ANN202 — returns BlockMask
        calls[0] += 1
        return real(segment_ids)

    monkeypatch.setattr(transformer_mod, "segment_block_mask", counting)
    return calls


def test_flex_block_mask_is_built_once_per_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_block_masks(monkeypatch)
    _, packed = _batches()
    with torch.no_grad():
        _model(None, "flex").eval()(**packed)  # 3 layers
    assert calls[0] == 1


def test_flex_config_without_segment_ids_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_block_masks(monkeypatch)
    flex_calls = _spy_flex_kernel(monkeypatch)
    padded, _ = _batches()
    with torch.no_grad():
        ref = _model(None, "sdpa").eval()(**padded).logits
        got = _model(None, "flex").eval()(**padded).logits
    assert calls[0] == 0
    assert flex_calls == []
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_flex_config_keeps_manual_path_for_attention_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _count_block_masks(monkeypatch)
    flex_calls = _spy_flex_kernel(monkeypatch)
    _, packed = _batches()
    with torch.no_grad():
        out = _model(None, "flex").eval()(**packed, output_attentions=True)
    assert calls[0] == 0
    assert flex_calls == []
    assert out.attentions is not None and all(a is not None for a in out.attentions)


def test_flex_matches_sdpa_under_bf16_autocast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Channel QK-norm returns fp32 q/k under autocast while v is bf16; flex must cast like SDPA."""
    calls = _spy_flex_kernel(monkeypatch)
    _, packed = _batches()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        ref = _model(None, "sdpa").eval()(**packed).logits
        assert calls == []
        flex_model = _model(None, "flex").eval()
        got = flex_model(**packed).logits
    # The kernel ran once per block and saw q, k and v all cast to bf16.
    assert calls == [(torch.bfloat16,) * 3] * flex_model.config.num_hidden_layers
    mask = packed["attention_mask"].bool()
    assert torch.isfinite(got[mask]).all()
    torch.testing.assert_close(got[mask], ref[mask], rtol=5e-2, atol=5e-2)

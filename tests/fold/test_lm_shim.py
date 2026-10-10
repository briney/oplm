"""LanguageModelShim against the transcribed upstream math; the per-chain frozen-LM runner."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.data.featurize import ChainSpec, featurize
from oplm.fold.lm_shim import (
    LanguageModelShim,
    SingleToPair,
    frozen_lm_hidden_states,
    lm_state_count,
)
from tests.fold.helpers import tiny_lm


def test_single_to_pair_matches_upstream_transcription() -> None:
    torch.manual_seed(0)
    stp = SingleToPair(16, 16, 16)
    x = torch.randn(2, 5, 16)
    h = stp.downproject(x)
    pair = torch.cat([h.unsqueeze(2) * h.unsqueeze(1), h.unsqueeze(2) - h.unsqueeze(1)], dim=3)
    expected = stp.output_fc2(F.gelu(stp.output_fc1(pair)))
    torch.testing.assert_close(stp(x), expected)
    assert stp.output_fc1.weight.shape == (16, 32) and stp.downproject.bias is not None
    rows, cols = torch.tensor([4, 0]), torch.tensor([1, 3, 2])
    torch.testing.assert_close(stp(x, rows=rows, cols=cols), expected[:, rows][:, :, cols])


def test_shim_mixes_states_with_softmax_weights_and_names() -> None:
    cfg = FoldConfig(lm_hidden_size=24, lm_num_hidden_states=4, pair_width=32)
    torch.manual_seed(0)
    shim = LanguageModelShim(cfg)
    assert set(shim.state_dict()) == {
        "layer_weights",
        "pair_input_norm.weight",
        "pair_input_norm.bias",
        "pair_proj.weight",
        "single_to_pair.downproject.weight",
        "single_to_pair.downproject.bias",
        "single_to_pair.output_fc1.weight",
        "single_to_pair.output_fc1.bias",
        "single_to_pair.output_fc2.weight",
        "single_to_pair.output_fc2.bias",
        "pair_output_norm.weight",
        "pair_output_norm.bias",
    }
    assert shim.layer_weights.shape == (4,) and shim.pair_proj.weight.shape == (32, 24)
    hs = torch.randn(1, 6, 4, 24)
    with torch.no_grad():
        shim.layer_weights.copy_(torch.tensor([0.0, 50.0, 0.0, 0.0]))  # ~one-hot on state 1
        mixed = shim.mix(hs)
        expected = shim.pair_proj(shim.pair_input_norm(hs[:, :, 1]))
        torch.testing.assert_close(mixed, expected, atol=1e-5, rtol=1e-5)
        out = shim(hs)
        torch.testing.assert_close(out, shim.pair_output_norm(shim.single_to_pair(mixed)))
    assert out.shape == (1, 6, 6, 32)


def test_shim_handles_a_single_token() -> None:
    cfg = FoldConfig(lm_hidden_size=8, lm_num_hidden_states=3, pair_width=32)
    out = LanguageModelShim(cfg)(torch.randn(1, 1, 3, 8))
    assert out.shape == (1, 1, 1, 32) and torch.isfinite(out).all()


def test_frozen_lm_hidden_states_gathers_per_chain_rows() -> None:
    lm = tiny_lm()
    f = featurize([ChainSpec("MKVLA", "A"), ChainSpec("GG", "B")], pad_tokens_to=8)
    assert lm_state_count(lm) == 3
    hs = frozen_lm_hidden_states(lm, f, dtype=torch.float32)
    assert hs.shape == (1, 8, 3, 32) and not hs.requires_grad
    with torch.no_grad():
        direct = lm(
            input_ids=f.lm_input_ids,
            attention_mask=f.lm_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    stacked = torch.stack(direct.hidden_states, dim=2)  # (C, T, K, D)
    torch.testing.assert_close(hs[0, 0], stacked[0, 1])  # chain A, residue 1 (after BOS)
    torch.testing.assert_close(hs[0, 6], stacked[1, 2])  # chain B, residue 2
    assert hs[0, 7:].abs().sum() == 0  # padded tokens are zero


def test_state_count_follows_the_executed_depth() -> None:
    assert lm_state_count(tiny_lm(num_loops=2)) == 5  # 2 layers x 2 loops + embedding

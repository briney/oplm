"""Cartesian sweep of architecture toggles, one forward + backward each (Phase 14.4).

Every combination must construct, run a forward, and back-propagate a finite
loss with a grad on every parameter. Models are kept tiny (2 layers, 64 hidden)
so the full matrix runs fast.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from oplm import OplmConfig, OplmForMaskedLM
from oplm.config import TrainConfig
from oplm.training.optim import partition_optimizer_params

_HIDDEN = 64
_HEADS = 4
_HEAD_DIM = _HIDDEN // _HEADS  # 16
_LAYERS = 2
_VOCAB = 33
_B, _T = 2, 16

_NORM_TYPES = ("layernorm", "rmsnorm")
_NORM_STRATEGIES = ("pre", "sandwich", "hybrid", "post_sdpa")
_CANON = (False, True)
_ROPE_DIMS = (_HEAD_DIM, _HEAD_DIM // 2)  # full RoPE + partial RoPE
_QK_NORM = (True, False)
_RESIDUAL_SCALING = ("sqrt_num_layers", "none")
_TIE = (False, True)
_POST_EMBED_NORM = (False, True)

_COMBOS = list(
    itertools.product(
        _NORM_TYPES,
        _NORM_STRATEGIES,
        _CANON,
        _ROPE_DIMS,
        _QK_NORM,
        _RESIDUAL_SCALING,
        _TIE,
        _POST_EMBED_NORM,
    )
)


def _combo_id(combo) -> str:
    norm_type, strat, canon, rope_dim, qk, resid, tie, post = combo
    return (
        f"{norm_type}-{strat}-canon{int(canon)}-rope{rope_dim}-"
        f"qk{int(qk)}-{resid}-tie{int(tie)}-pe{int(post)}"
    )


@pytest.mark.parametrize("combo", _COMBOS, ids=[_combo_id(c) for c in _COMBOS])
def test_toggle_combination_trains_one_step(combo) -> None:
    norm_type, norm_strategy, canon, rope_dim, qk_norm, residual_scaling, tie, post_embed = combo
    torch.manual_seed(0)

    config = OplmConfig(
        hidden_size=_HIDDEN,
        num_hidden_layers=_LAYERS,
        num_attention_heads=_HEADS,
        max_position_embeddings=64,
        norm_type=norm_type,
        norm_strategy=norm_strategy,
        canon_enabled=canon,
        canon_positions=["A", "C", "D"] if canon else None,
        rope_dim=rope_dim,
        nope_dim=_HEAD_DIM - rope_dim,
        qk_norm=qk_norm,
        residual_scaling=residual_scaling,
        tie_word_embeddings=tie,
        post_embed_norm=post_embed,
    )
    model = OplmForMaskedLM(config).train()

    input_ids = torch.randint(4, _VOCAB, (_B, _T))
    attention_mask = torch.ones(_B, _T, dtype=torch.long)
    labels = torch.randint(0, _VOCAB, (_B, _T))

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert torch.isfinite(out.loss).item()

    out.loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"params without grad: {missing}"
    for name, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


# ---------------------------------------------------------------------------
# Ablation-feature integration (Phases 2-5 + gated attention): mask dropout +
# L2 QKNorm + residual gates + GEGLU + attention output gate + sandwich norm
# exercised together, plus optimizer partitioning of the new params. Kept to a
# few representative combos rather than folded into the matrix above (which
# would multiply its size).
# ---------------------------------------------------------------------------

_INTEGRATION_COMBOS = [
    ("scalar", "sigmoid", "adamw"),
    ("scalar", "silu", "muon"),
    ("channel", "silu", "adamw"),
    ("channel", "sigmoid", "muon"),
]


@pytest.mark.parametrize("residual_gate,attn_output_gate,optimizer", _INTEGRATION_COMBOS)
def test_ablation_features_train_and_partition_together(
    residual_gate, attn_output_gate, optimizer
) -> None:
    """All ablation toggles on at once: train one step, then check optim grouping."""
    torch.manual_seed(0)
    config = OplmConfig(
        hidden_size=_HIDDEN,
        num_hidden_layers=_LAYERS,
        num_attention_heads=_HEADS,
        max_position_embeddings=64,
        norm_strategy="sandwich",
        mask_dropout=True,
        qk_norm=True,
        qk_norm_mode="l2",
        residual_gate=residual_gate,
        attn_output_gate=attn_output_gate,
        ffn_activation="geglu",
    )
    model = OplmForMaskedLM(config).train()

    # Inputs carry <mask> tokens and pads so mask dropout's per-row real-token
    # accounting runs on a non-trivial mask (real tokens stay in [4, mask_id)).
    input_ids = torch.randint(4, config.mask_token_id, (_B, _T))
    input_ids[0, 1] = config.mask_token_id
    input_ids[1, 2] = config.mask_token_id
    attention_mask = torch.ones(_B, _T, dtype=torch.long)
    attention_mask[0, -2:] = 0  # two pad positions on the first row
    labels = torch.randint(0, _VOCAB, (_B, _T))

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert torch.isfinite(out.loss).item()

    out.loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"params without grad: {missing}"
    for name, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"

    # The new 1-D params (qk_l2_scale + residual gates) must land in
    # AdamW-no-decay and never leak into Muon, and the partition must still tile
    # the trainable set exactly once.
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    groups = partition_optimizer_params(model, TrainConfig(optimizer=optimizer))
    no_decay_ids = {id(p) for p in groups.adamw_no_decay_params}
    muon_ids = {id(p) for p in groups.muon_params}

    new_1d = {
        n for n in id_to_name.values() if n.endswith(("qk_l2_scale", "attn_gate", "ffn_gate"))
    }
    assert new_1d, "expected qk_l2_scale and residual-gate params to exist"
    for name, param in model.named_parameters():
        if name in new_1d:
            assert id(param) in no_decay_ids, f"{name} should be AdamW-no-decay"
            assert id(param) not in muon_ids, f"{name} leaked into Muon"

    # The attention output-gate projection is a regular 2-D hidden matrix: under
    # Muon it must join the Muon group, never the no-decay group.
    gate_proj_names = {n for n in id_to_name.values() if "gate_proj" in n and "attention" in n}
    assert gate_proj_names, "expected attention gate_proj params to exist"
    if optimizer == "muon":
        for name, param in model.named_parameters():
            if name in gate_proj_names:
                assert id(param) in muon_ids, f"{name} should be in Muon"
                assert id(param) not in no_decay_ids, f"{name} should not be AdamW-no-decay"

    grouped_ids = [
        id(p)
        for p in (*groups.muon_params, *groups.adamw_decay_params, *groups.adamw_no_decay_params)
    ]
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(grouped_ids) == len(set(grouped_ids))  # no param counted twice
    assert set(grouped_ids) == trainable_ids  # full coverage

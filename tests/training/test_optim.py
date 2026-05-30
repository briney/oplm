"""Tests for optimizer parameter partitioning (Phase 3 head-exclusion fix).

The MLM head is ``OplmForMaskedLM.lm_head`` (an ``OplmMLMHead``). Muon must never
claim its 2-D weights (``lm_head.dense.weight`` and, when untied,
``lm_head.decoder.weight``) — those belong on AdamW-with-decay. Norms, biases, and
any ``embed``-named weight go to AdamW-no-decay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from oplm.config import TrainConfig
from oplm.model import OplmConfig as OplmModelConfig
from oplm.model import OplmForMaskedLM
from oplm.training.optim import partition_optimizer_params

if TYPE_CHECKING:
    import torch


def _model() -> OplmForMaskedLM:
    """Build a tiny untied ``OplmForMaskedLM`` (default ``tie_word_embeddings=False``)."""
    return OplmForMaskedLM(
        OplmModelConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=64,
        )
    )


def _names_of(params: list[torch.nn.Parameter], id_to_name: dict[int, str]) -> set[str]:
    """Map a parameter list back to its parameter names via ``id()`` lookup."""
    return {id_to_name[id(p)] for p in params}


def test_muon_excludes_lm_head_weights() -> None:
    """Under Muon, no ``lm_head.*`` weight leaks into Muon; both 2-D head weights go to AdamW."""
    model = _model()
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    groups = partition_optimizer_params(model, TrainConfig(optimizer="muon"))

    muon_names = _names_of(groups.muon_params, id_to_name)
    assert not any(name.startswith("lm_head.") for name in muon_names)

    decay_names = _names_of(groups.adamw_decay_params, id_to_name)
    assert "lm_head.dense.weight" in decay_names
    assert "lm_head.decoder.weight" in decay_names  # present because the model is untied

    # Muon still claims the backbone's 2-D hidden weights (otherwise the run errors).
    assert muon_names, "Muon received no eligible parameters"


def test_no_decay_holds_norms_biases_and_embeddings() -> None:
    """1-D params (norms/biases) and any ``embed``-named weight land on AdamW-no-decay."""
    model = _model()
    groups = partition_optimizer_params(model, TrainConfig(optimizer="muon"))

    no_decay_ids = {id(p) for p in groups.adamw_no_decay_params}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "embed" in name:
            assert id(param) in no_decay_ids, f"{name} should be AdamW-no-decay"


def test_partition_covers_every_trainable_param_once() -> None:
    """The three groups tile the trainable parameter set exactly (no gaps, no dupes)."""
    model = _model()
    groups = partition_optimizer_params(model, TrainConfig(optimizer="muon"))

    grouped = [
        *groups.muon_params,
        *groups.adamw_decay_params,
        *groups.adamw_no_decay_params,
    ]
    grouped_ids = [id(p) for p in grouped]
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}

    assert len(grouped_ids) == len(set(grouped_ids))  # no parameter counted twice
    assert set(grouped_ids) == trainable_ids  # full coverage


def test_adamw_leaves_muon_group_empty() -> None:
    """With the default AdamW optimizer, nothing is routed to Muon."""
    model = _model()
    groups = partition_optimizer_params(model, TrainConfig(optimizer="adamw"))
    assert groups.muon_params == []

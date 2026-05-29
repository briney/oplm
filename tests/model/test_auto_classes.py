"""Auto-class resolution without trust_remote_code (Phase 14.3).

`import oplm` registers every Oplm* class with the HF Auto registry, so a plain
`AutoModelForMaskedLM.from_pretrained(<dir>)` must resolve to the concrete class
with no `trust_remote_code` flag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

import oplm  # noqa: F401  (import triggers auto-class registration)
from oplm import (
    OplmConfig,
    OplmForMaskedLM,
    OplmForSequenceClassification,
    OplmForTokenClassification,
    OplmModel,
)

if TYPE_CHECKING:
    from pathlib import Path


def _tiny_config(**overrides) -> OplmConfig:
    base = dict(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        num_labels=3,
    )
    base.update(overrides)
    return OplmConfig(**base)


def test_auto_config_resolves(tmp_path: Path) -> None:
    _tiny_config().save_pretrained(tmp_path)
    config = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(config, OplmConfig)


@pytest.mark.parametrize(
    ("model_cls", "auto_cls"),
    [
        (OplmModel, AutoModel),
        (OplmForMaskedLM, AutoModelForMaskedLM),
        (OplmForSequenceClassification, AutoModelForSequenceClassification),
        (OplmForTokenClassification, AutoModelForTokenClassification),
    ],
    ids=["AutoModel", "MaskedLM", "SeqCls", "TokCls"],
)
def test_auto_model_resolves_concrete_class(model_cls, auto_cls, tmp_path: Path) -> None:
    model_cls(_tiny_config()).save_pretrained(tmp_path)
    loaded = auto_cls.from_pretrained(tmp_path)
    assert type(loaded) is model_cls

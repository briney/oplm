"""G12 — full train -> serve lifecycle (docs/TESTING_E2E.md §5).

Cross-module flow (kept at ``tests/`` top level by design): train a tiny model
through the real Trainer, reload its ``checkpoint-*/hf/`` export via both
``OplmForMaskedLM.from_pretrained`` and ``AutoModelForMaskedLM`` (trust_remote_code),
then run the ESM-C-style ``encode`` / ``logits`` API and assert the inference
embeddings are well-shaped and finite. Closes the train->serve loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from tests.training.conftest import tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow

_SEQS = ["MEEPQSDPSV", "GAGTAARELK"]
_HIDDEN = 32


def test_train_then_serve_roundtrip(training_parquet: Path, tmp_path: Path) -> None:
    """A trained checkpoint reloads and serves finite, correctly-shaped embeddings."""
    from oplm.data import get_tokenizer
    from oplm.model import LogitsConfig, OplmForMaskedLM
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(tmp_path, training_parquet, max_steps=4, save_every=4, hidden_size=_HIDDEN)
    Trainer(cfg).train()

    hf_dir = tmp_path / "checkpoint-4" / "hf"
    assert (hf_dir / "config.json").exists()

    # Direct reload via the concrete class.
    model = OplmForMaskedLM.from_pretrained(str(hf_dir)).eval()
    if getattr(model, "tokenizer", None) is None:
        model.tokenizer = get_tokenizer()

    out = model.logits(_SEQS, LogitsConfig(return_embeddings=True))
    assert out.embeddings is not None
    assert out.embeddings.shape[0] == len(_SEQS)
    assert out.embeddings.shape[2] == _HIDDEN
    assert torch.isfinite(out.embeddings).all()

    # encode() returns bare padded input ids for the same batch.
    ids = model.encode(_SEQS)
    assert ids.shape[0] == len(_SEQS)
    assert ids.dim() == 2

    # Auto-class reload closes the from_pretrained loop and yields a finite forward.
    from transformers import AutoModelForMaskedLM

    auto = AutoModelForMaskedLM.from_pretrained(str(hf_dir), trust_remote_code=True).eval()
    assert type(auto).__name__ == "OplmForMaskedLM"
    with torch.no_grad():
        input_ids = torch.randint(0, auto.config.vocab_size, (2, 8))
        logits = auto(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
    assert logits.shape == (2, 8, auto.config.vocab_size)
    assert torch.isfinite(logits).all()

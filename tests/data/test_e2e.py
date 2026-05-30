"""Pilot end-to-end test for the ``oplm.data`` pipeline (Phase 10.2).

Wires the public data API to the real model and proves the whole path works:

* ``build_train_dataloader`` → ``OplmForMaskedLM`` for a handful of train steps
  (forward + backward + optimizer step) with a finite loss and no shape errors.
* ``build_sequence_eval_dataloader`` → a finite eval loss, and two passes that
  yield byte-identical batches (the evaluation determinism contract).
* The weighted-masking path trains end-to-end and honors the
  ``masking_weights`` column: with zero-weight residues at every even residue
  index (``sequence_shards_zero_weighted``), masking can only ever land on the
  positive-weight positions.

The model is kept tiny (2 layers, 32 hidden) and runs on CPU via the manual
attention fallback, so the whole file stays fast despite being marked ``slow``.
Sequences are the real human proteins from the shared ``sequence_shards``
fixtures (``tests/data/conftest.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from oplm.config import DataConfig, OplmConfig, TrainConfig
from oplm.data import build_sequence_eval_dataloader, build_train_dataloader
from oplm.model import OplmConfig as OplmModelConfig
from oplm.model import OplmForMaskedLM

if TYPE_CHECKING:
    from pathlib import Path

    from torch.utils.data import DataLoader

pytestmark = pytest.mark.slow

_IGNORE_INDEX = -100

# Tiny pilot model — small enough to construct and train on CPU in well under a
# second, while keeping vocab_size at its 33-token default (the data path feeds
# canonical token ids straight into the embedding table).
_HIDDEN = 32
_HEADS = 4
_LAYERS = 2
_MAX_SEQ_LEN = 64
_BATCH_SIZE = 4
_N_STEPS = 3


def _make_data_config(train: object, *, weighted_masking: bool = False) -> OplmConfig:
    """Build a tiny data/training config wired to a fixture dataset."""
    return OplmConfig(
        model=OplmModelConfig(
            hidden_size=_HIDDEN,
            num_attention_heads=_HEADS,
            num_hidden_layers=_LAYERS,
            max_position_embeddings=_MAX_SEQ_LEN,
        ),
        train=TrainConfig(batch_size=_BATCH_SIZE, seed=7),
        data=DataConfig(
            train=train,
            num_workers=0,
            pin_memory=False,
            weighted_masking=weighted_masking,
            mask_prob=0.15,
        ),
    )


def _make_model() -> OplmForMaskedLM:
    """Instantiate the matching tiny ``OplmForMaskedLM`` (default 33-token vocab)."""
    model_cfg = OplmModelConfig(
        hidden_size=_HIDDEN,
        num_attention_heads=_HEADS,
        num_hidden_layers=_LAYERS,
        max_position_embeddings=_MAX_SEQ_LEN,
    )
    return OplmForMaskedLM(model_cfg)


def _forward_backward_step(
    model: OplmForMaskedLM,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run one forward + backward + optimizer step; return the (finite) loss."""
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    assert out.loss is not None
    assert torch.isfinite(out.loss).item(), "training loss is not finite"
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()
    return out.loss.detach()


def test_pilot_train_runs_end_to_end(sequence_shards: Path) -> None:
    """A tiny model trains for a few steps over the train dataloader without error."""
    torch.manual_seed(0)
    cfg = _make_data_config(str(sequence_shards))
    model = _make_model().train()
    loader = build_train_dataloader(cfg)
    loader.dataset.set_epoch(0)  # type: ignore[attr-defined]  # IterableDataset carries set_epoch
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    steps = 0
    for _, batch in zip(range(_N_STEPS), loader, strict=False):
        assert set(batch) == {"input_ids", "attention_mask", "labels"}
        assert batch["input_ids"].shape[1] <= _MAX_SEQ_LEN  # T never exceeds max_seq_len
        _forward_backward_step(model, optimizer, batch)
        steps += 1

    assert steps == _N_STEPS  # the fixture has enough rows to serve every step


def test_sequence_eval_is_finite_and_deterministic(sequence_shards: Path) -> None:
    """Eval yields a finite loss, and two passes produce byte-identical batches."""
    torch.manual_seed(0)
    cfg = _make_data_config(str(sequence_shards))
    model = _make_model().eval()
    loader: DataLoader[dict[str, torch.Tensor]] = build_sequence_eval_dataloader(
        str(sequence_shards), cfg
    )

    first = list(loader)
    second = list(loader)
    assert first  # the fixture has rows
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert set(a) == set(b)
        for key in a:
            assert torch.equal(a[key], b[key]), f"eval batch differs on {key!r} across passes"

    with torch.no_grad():
        out = model(
            input_ids=first[0]["input_ids"],
            attention_mask=first[0]["attention_mask"],
            labels=first[0]["labels"],
        )
    assert out.loss is not None
    assert torch.isfinite(out.loss).item(), "eval loss is not finite"


def test_weighted_masking_trains_and_respects_weights(
    sequence_shards_zero_weighted: Path,
) -> None:
    """Weighted masking trains end-to-end and never masks a zero-weight residue.

    The fixture zeroes every even residue index, which (residue ``j`` → token
    column ``j + 1``) puts the zero-weight positions on the *odd* token columns.
    The Gumbel-top-k masker forces ``key = -inf`` at zero weight, so every masked
    position must fall on an *even* column — a hard, draw-independent invariant.
    """
    torch.manual_seed(0)
    cfg = _make_data_config(str(sequence_shards_zero_weighted), weighted_masking=True)
    assert cfg.data.weighted_masking
    model = _make_model().train()
    loader = build_train_dataloader(cfg)
    loader.dataset.set_epoch(0)  # type: ignore[attr-defined]  # IterableDataset carries set_epoch
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    total_masked = 0
    for _, batch in zip(range(_N_STEPS), loader, strict=False):
        masked_cols = (batch["labels"] != _IGNORE_INDEX).nonzero(as_tuple=False)[:, 1]  # (N,)
        assert torch.all(masked_cols % 2 == 0), "masked a zero-weight (odd-column) position"
        total_masked += int(masked_cols.numel())
        _forward_backward_step(model, optimizer, batch)

    # Non-vacuous: positive-weight positions are still masked, so the invariant
    # above is meaningful rather than trivially satisfied by an empty mask.
    assert total_masked > 0

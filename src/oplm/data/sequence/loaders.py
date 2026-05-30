"""Sequence dataloader builders.

``build_train_dataloader`` and ``build_sequence_eval_dataloader`` assemble the
datasets, MLM collator, and :class:`torch.utils.data.DataLoader` from an
:class:`oplm.config.OplmConfig`, applying train vs. eval shuffling/determinism
policy.

The two builders share the same machinery — :class:`ShardedProteinDataset`
(optionally wrapped in :class:`InterleavedDataset`), an :class:`MLMCollator`, and
a :class:`~torch.utils.data.DataLoader`. They differ only in *policy*
parameters: training shuffles and draws fresh masks each epoch, while sequence
evaluation freezes shuffling and masking for reproducibility (docs/DATA_TOOLING.md
§9). There is deliberately no separate "eval dataset"/"eval collator" class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import DataLoader

from oplm.data.config import parse_train_configs
from oplm.data.sequence.collate import MLMCollator
from oplm.data.sequence.dataset import InterleavedDataset, ShardedProteinDataset
from oplm.data.tokenizer import get_tokenizer

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import IterableDataset

    from oplm.config import OplmConfig

__all__ = ["build_sequence_eval_dataloader", "build_train_dataloader"]

# Fixed seed for sequence evaluation. Shuffling is off and masks are frozen, so
# the seed only needs to be stable run-to-run; decoupling it from
# ``train.seed`` keeps eval batches identical even when the training seed
# changes (docs/DATA_TOOLING.md §9.2).
_EVAL_SEED = 42


def build_train_dataloader(cfg: OplmConfig) -> DataLoader[dict[str, Tensor]]:
    """Build the training dataloader from an :class:`~oplm.config.OplmConfig`.

    Applies the **training policy**: shard/row shuffling per ``cfg.data`` and
    dynamic (RoBERTa-style) masking that is re-drawn each epoch. With more than
    one training dataset the sources are mixed by their parsed fractions via
    :class:`InterleavedDataset`.

    Args:
        cfg: Fully resolved configuration. Reads ``cfg.data.train`` (the dataset
            spec), the masking knobs, the shard-iteration flags, the DataLoader
            settings, and ``cfg.model.max_position_embeddings`` / ``cfg.train``
            (seed, batch size).

    Returns:
        A :class:`~torch.utils.data.DataLoader` yielding the §4.7 batch contract
        (``input_ids`` / ``attention_mask`` / ``labels``).

    Raises:
        ValueError: If ``cfg.data.train`` resolves to no datasets.
    """
    entries = parse_train_configs(cfg.data.train)
    if not entries:
        raise ValueError("cfg.data.train resolved to no datasets; nothing to train on")

    sources = [
        ShardedProteinDataset(
            entry.path,
            shuffle_shards=cfg.data.shuffle_shards,
            shuffle_rows=cfg.data.shuffle_rows,
            seed=cfg.train.seed,
            load_masking_weights=cfg.data.weighted_masking,
        )
        for entry in entries
    ]

    dataset: IterableDataset[dict[str, object]]
    if len(sources) > 1:
        dataset = InterleavedDataset(
            sources,
            [entry.fraction for entry in entries],
            seed=cfg.train.seed,
        )
    else:
        dataset = sources[0]

    collator = MLMCollator(
        get_tokenizer(),
        max_length=cfg.model.max_position_embeddings,
        mask_prob=cfg.data.mask_prob,
        mask_token_prob=cfg.data.mask_token_prob,
        random_token_prob=cfg.data.random_token_prob,
        weighted_masking=cfg.data.weighted_masking,
        deterministic=False,
        seed=cfg.train.seed,
    )
    return _build_dataloader(dataset, collator, cfg)


def build_sequence_eval_dataloader(path: str, cfg: OplmConfig) -> DataLoader[dict[str, Tensor]]:
    """Build a deterministic sequence-evaluation dataloader for one parquet path.

    Applies the **evaluation policy**: shard/row shuffling disabled and masking
    frozen via ``deterministic=True`` so repeated passes yield byte-identical
    batches. Masking probabilities and DataLoader settings are otherwise shared
    with training (docs/DATA_TOOLING.md §9.2).

    Args:
        path: A single ``.parquet`` file or directory of shards to evaluate on.
        cfg: Fully resolved configuration; shares masking knobs, DataLoader
            settings, and ``cfg.model.max_position_embeddings`` with the
            training builder.

    Returns:
        A :class:`~torch.utils.data.DataLoader` yielding the §4.7 batch contract.
    """
    dataset = ShardedProteinDataset(
        path,
        shuffle_shards=False,
        shuffle_rows=False,
        seed=_EVAL_SEED,
        load_masking_weights=cfg.data.weighted_masking,
    )
    collator = MLMCollator(
        get_tokenizer(),
        max_length=cfg.model.max_position_embeddings,
        mask_prob=cfg.data.mask_prob,
        mask_token_prob=cfg.data.mask_token_prob,
        random_token_prob=cfg.data.random_token_prob,
        weighted_masking=cfg.data.weighted_masking,
        deterministic=True,
        seed=_EVAL_SEED,
    )
    return _build_dataloader(dataset, collator, cfg, loader_cls=_ResettingDataLoader)


class _ResettingDataLoader(DataLoader):  # generic param set at call sites
    """DataLoader that rewinds a deterministic collator at the start of each pass.

    The deterministic :class:`MLMCollator` seeds its per-batch RNG from a running
    batch counter (§4.6); without a rewind, a second pass over the *same* loader
    would continue the counter and draw different masks. Resetting in
    ``__iter__`` — before any worker processes are spawned, so the rewound state
    is what gets pickled to them — makes every pass reproduce the first, which is
    the evaluation determinism contract. Training never uses this wrapper.
    """

    def __iter__(self):  # delegates to DataLoader.__iter__
        reset = getattr(self.collate_fn, "reset_batch_index", None)
        if callable(reset):
            reset()
        return super().__iter__()


def _build_dataloader(
    dataset: IterableDataset[dict[str, object]],
    collator: MLMCollator,
    cfg: OplmConfig,
    *,
    loader_cls: type[DataLoader] = DataLoader,  # generic param set by return type
) -> DataLoader[dict[str, Tensor]]:
    """Wrap a dataset + collator in a DataLoader using ``cfg``'s worker settings.

    ``prefetch_factor`` must be ``None`` when ``num_workers == 0`` (PyTorch rejects
    a positive value with no worker processes), so it is gated accordingly.
    ``loader_cls`` lets the eval builder substitute a determinism-preserving
    subclass while training uses the plain :class:`~torch.utils.data.DataLoader`.
    """
    num_workers = cfg.data.num_workers
    return loader_cls(
        dataset,
        batch_size=cfg.train.batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor if num_workers > 0 else None,
    )

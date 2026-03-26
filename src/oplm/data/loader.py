"""Training dataloader construction from config."""

from __future__ import annotations

from torch.utils.data import DataLoader

from oplm.config import OplmConfig, parse_train_configs
from oplm.data.collate import MLMCollator
from oplm.data.dataset import InterleavedDataset, ShardedProteinDataset
from oplm.data.tokenizer import ProteinTokenizer


def build_train_dataloader(cfg: OplmConfig) -> DataLoader[dict[str, str]]:
    """Build a training DataLoader from config.

    Parses ``cfg.data.train`` to determine dataset path(s) and sampling
    fractions, instantiates the appropriate dataset(s), and wraps them in
    a :class:`~torch.utils.data.DataLoader` with an :class:`MLMCollator`.

    Args:
        cfg: Full OPLM configuration.

    Returns:
        A DataLoader yielding batches of ``{input_ids, attention_mask, labels}``.

    Raises:
        ValueError: If no training datasets are configured.
    """
    entries = parse_train_configs(cfg.data.train)
    if not entries:
        raise ValueError(
            "No training datasets configured. Set data.train in your config "
            "or via CLI: data.train=/path/to/dataset"
        )

    seed = cfg.train.seed

    datasets = [
        ShardedProteinDataset(
            entry.path,
            shuffle_shards=cfg.data.shuffle_shards,
            shuffle_rows=cfg.data.shuffle_rows,
            seed=seed,
        )
        for entry in entries
    ]

    dataset: ShardedProteinDataset | InterleavedDataset
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = InterleavedDataset(
            datasets,
            [entry.fraction for entry in entries],
            seed=seed,
        )

    tokenizer = ProteinTokenizer()
    collator = MLMCollator(
        tokenizer,
        max_length=cfg.data.max_length,
        mask_prob=cfg.data.mask_prob,
    )

    num_workers = cfg.data.num_workers
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor if num_workers > 0 else None,
    )

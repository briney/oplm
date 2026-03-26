"""Integration tests for the training dataloader builder."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from oplm.config import DataConfig, OplmConfig, TrainConfig
from oplm.data.loader import build_train_dataloader

SEQUENCES = [
    ("sp|P68871|HBB_HUMAN", "MVHLTPEEKSAVTALWGKVNVDEVGGEALG"),
    ("sp|P69905|HBA_HUMAN", "MVLSPADKTNVKAAWGKVGAHAGEYGAEAL"),
    ("sp|P01308|INS_HUMAN", "MALWMRLLPLLALLALWGPDPAAAFVNQHL"),
    ("sp|P10636|TAU_HUMAN", "MAEPRQEFEVMEDHAGTYGLGDRKDQGGYT"),
    ("sp|P04637|P53_HUMAN", "MEEPQSDPSVEPPLSQETFSDLWKLLPENN"),
    ("sp|P38398|BRCA1_HUMAN", "MDLSALRVEEVQNVINAMQKILECPICLEE"),
    ("sp|P01375|TNFA_HUMAN", "MSTESMIRDVELAEEALPKKTGGPQGSRRC"),
    ("sp|P21359|NF1_HUMAN", "MAAHRPVEWVQAVVSRFDEQLPIKTGQQNT"),
]


def _write_parquet(path: Path, sequences: list[tuple[str, str]]) -> None:
    table = pa.table(
        {
            "sequence_id": [s[0] for s in sequences],
            "sequence": [s[1] for s in sequences],
        }
    )
    pq.write_table(table, path)


def _make_config(train_data: str | dict, batch_size: int = 4) -> OplmConfig:
    """Build a minimal config for testing."""
    return OplmConfig(
        train=TrainConfig(batch_size=batch_size, seed=42),
        data=DataConfig(
            train=train_data,
            max_length=64,
            mask_prob=0.15,
            num_workers=0,
            pin_memory=False,
        ),
    )


class TestBuildTrainDataloader:
    def test_single_dataset(self, tmp_path: Path) -> None:
        path = tmp_path / "train.parquet"
        _write_parquet(path, SEQUENCES)

        cfg = _make_config(str(path))
        loader = build_train_dataloader(cfg)

        batch = next(iter(loader))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] == 4  # batch_size

    def test_sharded_dataset(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_parquet(shard_dir / "shard_00.parquet", SEQUENCES[:4])
        _write_parquet(shard_dir / "shard_01.parquet", SEQUENCES[4:])

        cfg = _make_config(str(shard_dir))
        loader = build_train_dataloader(cfg)

        batch = next(iter(loader))
        assert batch["input_ids"].shape[0] == 4

    def test_multi_dataset(self, tmp_path: Path) -> None:
        path_a = tmp_path / "ds_a.parquet"
        path_b = tmp_path / "ds_b.parquet"
        _write_parquet(path_a, SEQUENCES[:4])
        _write_parquet(path_b, SEQUENCES[4:])

        train_config = {
            "ds_a": {"path": str(path_a), "fraction": 0.6},
            "ds_b": {"path": str(path_b), "fraction": 0.4},
        }
        cfg = _make_config(train_config)
        loader = build_train_dataloader(cfg)

        batch = next(iter(loader))
        assert batch["input_ids"].shape[0] == 4

    def test_no_dataset_raises(self) -> None:
        cfg = _make_config(None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="No training datasets"):
            build_train_dataloader(cfg)

    def test_batch_feeds_into_model(self, tmp_path: Path) -> None:
        """Verify batch format is compatible with OplmForMLM.forward()."""
        path = tmp_path / "train.parquet"
        _write_parquet(path, SEQUENCES)

        cfg = _make_config(str(path), batch_size=2)
        loader = build_train_dataloader(cfg)
        batch = next(iter(loader))

        # Check shapes and dtypes match model expectations
        B, T = batch["input_ids"].shape
        assert B == 2
        assert T <= 64
        assert batch["input_ids"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.long
        assert batch["labels"].dtype == torch.long
        assert batch["attention_mask"].shape == (B, T)
        assert batch["labels"].shape == (B, T)

    def test_iterate_full_epoch(self, tmp_path: Path) -> None:
        path = tmp_path / "train.parquet"
        _write_parquet(path, SEQUENCES)

        cfg = _make_config(str(path), batch_size=4)
        loader = build_train_dataloader(cfg)

        total_samples = 0
        for batch in loader:
            total_samples += batch["input_ids"].shape[0]
        assert total_samples > 0

"""Tests for ``--data`` loading in the μP coord-check CLI.

``_load_sequences`` must accept the same inputs as ``data.train``: built-ins when
omitted, a single ``.parquet`` file, **and a directory of shards** (the case that
raised ``OSError: ... is a directory`` before discovery was wired through the
training loader's ``ShardedProteinDataset._discover_shards``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq
from scripts._mup_common import Scaling
from scripts.mup_coord_check import _DEFAULT_SEQUENCES, _build_cfg_fn, _load_sequences

if TYPE_CHECKING:
    from pathlib import Path


def test_load_sequences_builtin_when_no_data() -> None:
    """With no ``--data``, the built-in sequences are used (capped at ``n_seqs``)."""
    assert _load_sequences(None, 3) == _DEFAULT_SEQUENCES[:3]


def test_load_sequences_single_file(training_parquet: Path) -> None:
    """A single ``.parquet`` file yields ``n_seqs`` real sequences."""
    seqs = _load_sequences(training_parquet, 5)
    assert len(seqs) == 5
    assert all(isinstance(s, str) and s for s in seqs)


def test_load_sequences_shard_directory(training_parquet: Path, tmp_path: Path) -> None:
    """A directory of shards is discovered and read across shards up to ``n_seqs``."""
    table = pq.read_table(training_parquet, columns=["sequence"])
    half = table.num_rows // 2
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    # Two shards, each smaller than the request, so reading must cross shard files.
    pq.write_table(table.slice(0, half), shard_dir / "shard_00.parquet")
    pq.write_table(table.slice(half), shard_dir / "shard_01.parquet")

    n_seqs = half + 2  # forces a read past the first shard
    seqs = _load_sequences(shard_dir, n_seqs)
    assert len(seqs) == n_seqs
    assert all(isinstance(s, str) and s for s in seqs)


def test_coord_check_resolves_production_model_features(tmp_path: Path) -> None:
    config = tmp_path / "base.yaml"
    config.write_text(
        """
model:
  norm_strategy: sandwich
  canon_enabled: true
  canon_positions: [A, B, C, D]
  residual_gate: channel
  value_residual: learnable
""".lstrip()
    )
    build = _build_cfg_fn(
        config=config,
        depth=24,
        mup=True,
        scaling=Scaling.preset_ray,
        base_width=768,
        output_mult=1.0,
    )

    cfg = build(1280)

    assert cfg.hidden_size == 1280
    assert cfg.num_hidden_layers == 40
    assert cfg.num_attention_heads == 20
    assert cfg.head_dim == 64
    assert cfg.norm_strategy == "sandwich"
    assert cfg.canon_positions == ["A", "B", "C", "D"]
    assert cfg.residual_gate == "channel"
    assert cfg.value_residual == "learnable"
    assert cfg.mup_base_width == 768

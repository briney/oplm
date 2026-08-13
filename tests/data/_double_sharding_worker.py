"""Worker for the double-sharding verification test. Run under torch.distributed.run.

Builds the real train dataloader over a real parquet fixture exactly as
``Trainer.__init__`` does post-Task-0.2: ``Accelerator(dataloader_config=
DataLoaderConfiguration(dispatch_batches=False))``, but the dataloader is kept OUT
of ``accelerator.prepare`` and wrapped in
:class:`oplm.data.sequence.loaders.DeviceDataLoader` instead (see
``src/oplm/training/trainer.py``). Iterates one full epoch and writes each rank's
consumed ``sequence_id``s to ``out_dir/rank<i>.json`` so the parent test process
can check coverage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration


def main(fixture_path: str, out_dir: str) -> None:
    """Run one epoch of the DeviceDataLoader-wrapped dataloader, dump sequence ids.

    Args:
        fixture_path: Parquet file or directory of shards to stream from.
        out_dir: Directory to write ``rank<i>.json`` into (must already exist).
    """
    accelerator = Accelerator(
        cpu=True, dataloader_config=DataLoaderConfiguration(dispatch_batches=False)
    )
    from torch.utils.data import DataLoader

    from oplm.data.sequence.collate import MLMCollator
    from oplm.data.sequence.dataset import ShardedProteinDataset
    from oplm.data.sequence.loaders import DeviceDataLoader
    from oplm.data.tokenizer import get_tokenizer

    dataset = ShardedProteinDataset(fixture_path, seed=0)
    collator = MLMCollator(get_tokenizer(), max_length=64, keep_sequence_ids=True)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator, num_workers=2)
    dataloader = DeviceDataLoader(dataloader, accelerator.device)

    seen: list[str] = []
    for batch in dataloader:
        seen.extend(batch["sequence_ids"])
    rank = accelerator.process_index
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(seen))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

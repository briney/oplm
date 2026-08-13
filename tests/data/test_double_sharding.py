"""Double-sharding verification test (Task 0.1).

``accelerator.prepare(dataloader)`` may wrap an ``IterableDataset`` in
``accelerate``'s own ``IterableDatasetShard`` (per-rank striping) *on top of*
``ShardedProteinDataset``'s own explicit rank/worker striping (over the joint
``(rank, worker)`` index — see ``src/oplm/data/sequence/dataset.py``). If both
layers are active at once, each rank's already-striped stream gets re-striped,
and rows are silently dropped.

This test launches the real ``Trainer``-style dataloader construction
(``Accelerator(dataloader_config=DataLoaderConfiguration(dispatch_batches=False))``
then ``accelerator.prepare(dataloader)``) under two CPU/gloo processes via
``torch.distributed.run``, has each rank consume one full epoch, and checks
whether the union of what both ranks saw reproduces the full fixture with no
duplicates. See ``tests/data/_double_sharding_worker.py`` for the subprocess
entry point.

The fixture is deliberately larger than the repo's tiny 14-row ``sequence_shards``
fixture: with ``batch_size=4``/``world_size=2``, ``IterableDatasetShard`` only
drops rows once a rank's per-(rank, worker) row count exceeds
``batch_size * world_size`` (its internal ``real_batch_size``) — below that
every row happens to survive by coincidence (verified empirically: 14 rows
loses none, 40 rows loses 16). 40 real rows across 2 shards reliably exceeds
that threshold, so the test is a faithful, non-coincidental check.

VERDICT (see ``.superpowers/sdd/TODOS/task-0.1-report.md`` for full numbers):
double-sharding is confirmed — rows are lost (16/40 in the empirical run,
0 duplicated across ranks). The test below is marked ``xfail(strict=True)``
until Task 0.2 removes the extra striping layer.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def double_sharding_fixture_dir(
    tmp_path_factory: pytest.TempPathFactory,
    real_records: list[tuple[str, str]],
    make_sequence_shards: Callable[..., Path],
) -> Path:
    """A 2-shard directory of 40 real sequences, large enough to expose row loss.

    Reuses the repo's real-data fixtures (``tests/data/conftest.py``): the
    session-scoped pool of 40 real ``(sequence_id, sequence)`` records and the
    ``write_sequence_shards`` factory.
    """
    directory = tmp_path_factory.mktemp("double_sharding_fixture")
    return make_sequence_shards(directory, real_records, n_shards=2)


def _fixture_ids(shard_dir: Path) -> set[str]:
    """Ground-truth set of every ``sequence_id`` across all parquet shards in a directory."""
    ids: set[str] = set()
    for shard in sorted(shard_dir.iterdir()):
        ids.update(pq.read_table(shard, columns=["sequence_id"]).column("sequence_id").to_pylist())
    return ids


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "Task 0.1 audit: accelerate's IterableDatasetShard re-stripes "
        "ShardedProteinDataset's already rank/worker-striped stream, dropping rows. "
        "Fixed by Task 0.2 (removes the double striping)."
    ),
)
def test_prepared_dataloader_covers_dataset_exactly_once(
    tmp_path: Path, double_sharding_fixture_dir: Path
) -> None:
    """Union of what both ranks consume from the prepared dataloader covers the fixture exactly.

    Launches ``tests/data/_double_sharding_worker.py`` under
    ``torch.distributed.run --nproc_per_node=2`` (CPU/gloo), which builds the
    dataloader exactly as ``Trainer.__init__`` does and dumps each rank's consumed
    ``sequence_id``s. Asserts the invariant we *want*: no rows lost, none
    duplicated across ranks.
    """
    worker = Path(__file__).with_name("_double_sharding_worker.py")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            str(worker),
            str(double_sharding_fixture_dir),
            str(tmp_path),
        ],
        check=True,
        timeout=300,
    )
    rank0 = set(json.loads((tmp_path / "rank0.json").read_text()))
    rank1 = set(json.loads((tmp_path / "rank1.json").read_text()))
    all_ids = _fixture_ids(double_sharding_fixture_dir)

    assert not (rank0 & rank1), "rows duplicated across ranks"
    assert rank0 | rank1 == all_ids, "rows lost: double-sharding is real"

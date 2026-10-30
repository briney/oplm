"""Dataset manifest: every file's relative path, rows (parquet), bytes and sha256."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from pathlib import Path

MAX_FILES_PER_FOLDER = 10_000
MANIFEST_NAME = "manifest.parquet"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(dataset: Path) -> Path:
    """Write ``dataset/manifest.parquet`` (``path``, ``rows``, ``bytes``, ``sha256``).

    ``rows`` is null for non-parquet files. The manifest does not list itself.

    Raises:
        ValueError: a folder holds more than 10,000 files.
    """
    files = sorted(p for p in dataset.rglob("*") if p.is_file())
    files = [p for p in files if p != dataset / MANIFEST_NAME and not p.name.endswith(".tmp")]
    counts: dict[Path, int] = {}
    for p in files:
        counts[p.parent] = counts.get(p.parent, 0) + 1
    for folder, n in counts.items():
        if n > MAX_FILES_PER_FOLDER:
            raise ValueError(f"{folder} holds {n} files (> {MAX_FILES_PER_FOLDER})")
    cols: dict[str, list] = {"path": [], "rows": [], "bytes": [], "sha256": []}
    for p in files:
        cols["path"].append(p.relative_to(dataset).as_posix())
        cols["rows"].append(pq.ParquetFile(p).metadata.num_rows if p.suffix == ".parquet" else None)
        cols["bytes"].append(p.stat().st_size)
        cols["sha256"].append(sha256_file(p))
    schema = pa.schema(
        [
            ("path", pa.string()),
            ("rows", pa.int64()),
            ("bytes", pa.int64()),
            ("sha256", pa.string()),
        ]
    )
    out = dataset / MANIFEST_NAME
    tmp = dataset / f"{MANIFEST_NAME}.tmp"
    pq.write_table(pa.table(cols, schema=schema), tmp)
    tmp.rename(out)
    return out

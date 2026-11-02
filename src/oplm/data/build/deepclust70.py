"""deepclust70 held-out re-cut: screen the copied eval/test reps against the whole training
mix (deepclust70 train/ and every uniref70 train member) and keep ``n_eval`` / ``n_test``
survivors per stratum."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as _pc
import pyarrow.parquet as pq

from oplm.data.build.mmseqs import mmseqs_version
from oplm.data.build.reshard import _COMPRESSION
from oplm.data.build.uniref70 import SCREEN_ARGS, STRATA, screen_against
from oplm.data.sequence.dataset import MEMBER_IDS_COLUMN, MEMBER_SEQUENCES_COLUMN

if TYPE_CHECKING:
    from pathlib import Path

pc: Any = _pc  # the pyarrow stubs omit the generated compute kernels

__all__ = ["screen_heldout"]


def screen_heldout(
    src: Path,
    out: Path,
    *,
    deepclust70: Path,
    uniref70: Path,
    workdir: Path,
    mmseqs: str,
    threads: int,
    n_eval: int,
    n_test: int,
    chunk_rows: int = 200_000_000,
) -> dict:
    """Write ``out/{eval,test}_{singletons,gte2}.parquet`` and ``out/heldout.json``.

    Per stratum the candidate pool is ``src/eval_<stratum>.parquet`` then
    ``src/test_<stratum>.parquet``, in file order. A candidate with any ``SCREEN_ARGS`` hit
    in ``deepclust70`` (train shards) or among the members of ``uniref70`` (train shards)
    is dropped; the first ``n_eval`` survivors are eval and the next ``n_test`` are test.
    Screen hits files stay in ``workdir``, so a rerun resumes.

    Raises:
        ValueError: ``src`` is ``out``, or a stratum has fewer than ``n_eval + n_test``
            survivors (stats are in the message).
    """
    from oplm import __version__

    if src.resolve() == out.resolve():
        raise ValueError("src and out must differ: the source files are the screen input")
    pools = {
        k: pa.concat_tables([pq.read_table(src / f"{s}_{k}.parquet") for s in ("eval", "test")])
        for k in STRATA
    }
    cands = [
        {"sequence_id": i, MEMBER_IDS_COLUMN: [i], MEMBER_SEQUENCES_COLUMN: [s]}
        for t in pools.values()
        for i, s in zip(t["sequence_id"].to_pylist(), t["sequence"].to_pylist(), strict=True)
    ]
    kw: dict[str, Any] = {"mmseqs": mmseqs, "threads": threads, "chunk_rows": chunk_rows}
    deep = screen_against(cands, deepclust70, workdir / "screen-deepclust70", **kw)
    u70 = screen_against(cands, uniref70, workdir / "screen-uniref70", members=True, **kw)
    hits = pa.array(sorted(deep | u70), pa.large_string())

    stats: dict[str, Any] = {"hits_deepclust70": len(deep), "hits_uniref70": len(u70)}
    kept = {}
    for k, t in pools.items():
        kept[k] = t.filter(pc.invert(pc.is_in(t["sequence_id"], value_set=hits)))
        stats[k] = {
            "pool": t.num_rows,
            "survivors": kept[k].num_rows,
            "survival_rate": kept[k].num_rows / t.num_rows,
        }
    short = {k: v for k, v in stats.items() if k in STRATA and v["survivors"] < n_eval + n_test}
    if short:
        raise ValueError(f"fewer than n_eval + n_test = {n_eval + n_test} survivors: {stats}")

    out.mkdir(parents=True, exist_ok=True)
    for k, t in kept.items():
        for name, part in (("eval", t.slice(0, n_eval)), ("test", t.slice(n_eval, n_test))):
            pq.write_table(part, out / f"{name}_{k}.parquet", compression=_COMPRESSION)
    summary = {
        "params": {
            "src": str(src),
            "deepclust70": str(deepclust70),
            "uniref70": str(uniref70),
            "n_eval": n_eval,
            "n_test": n_test,
            "chunk_rows": chunk_rows,
            "screen_args": list(SCREEN_ARGS),
        },
        "stats": stats,
        "mmseqs_version": mmseqs_version(mmseqs),
        "oplm_version": __version__,
    }
    (out / "heldout.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary

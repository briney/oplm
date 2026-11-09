"""``oplm fold eval-coevolution``: MSAs and GREMLIN Potts maps for the structure eval sets.

Spec: docs/superpowers/specs/2026-11-07-coevolution-eval-design.md. The output folder is what
``oplm.eval.metrics.coevolution.CoevolutionReference`` reads. A rerun reuses every file already
written (each is renamed into place, so an interrupted run leaves no partial file).
"""

from __future__ import annotations

import dataclasses
import datetime
import gzip
import hashlib
import json
import logging
import math
import os
import subprocess
import textwrap
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from oplm.eval.metrics.coevolution import NEFF_BINS
from oplm.fold.evalsets.casp import sha256sum
from oplm.fold.evalsets.gremlin import (
    GremlinConfig,
    fit_gremlin,
    full_length_map,
    potts_contact_map,
)
from oplm.fold.evalsets.msa import (
    PUBLIC_COLABFOLD_HOST,
    Fetch,
    encode_msa,
    fetch_msas,
    filter_gap_columns,
    neff,
    parse_a3m,
)

__all__ = ["CoevolutionBuildConfig", "build_coevolution", "render_readme"]

logger = logging.getLogger(__name__)

_MSA_MODE = "env"
_LOG_EVERY = 25


@dataclass
class CoevolutionBuildConfig:
    """Inputs and settings; the defaults are the spec values."""

    sets_root: Path
    out: Path
    sets: tuple[str, ...] = ("casp15_16", "cameo_256", "pdb40_256", "pdb40_2048")
    host: str = PUBLIC_COLABFOLD_HOST
    msa_batch_size: int = 100
    device: str = "cpu"
    max_seqs: int = 20_000
    max_gap_fraction: float = 0.25
    identity: float = 0.8
    bin_edges: tuple[float, float] | None = None
    fit: bool = True
    fetch: Fetch | None = None
    colabfold_db: str = "unrecorded"


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _sha(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def _fit_settings(cfg: CoevolutionBuildConfig) -> dict[str, Any]:
    """The settings a stored fit depends on (``device`` is recorded but not compared)."""
    return {
        "max_seqs": cfg.max_seqs,
        "max_gap_fraction": cfg.max_gap_fraction,
        "identity": cfg.identity,
        "gremlin": dataclasses.asdict(GremlinConfig()),
    }


def _fit_one(cfg: CoevolutionBuildConfig, sequence: str, sha: str) -> None:
    """Fit one sequence from its stored MSA; writes ``couplings/`` then ``stats/``."""

    rows = parse_a3m(gzip.decompress((cfg.out / "msas" / f"{sha}.a3m.gz").read_bytes()).decode())
    if rows[0] != sequence:
        raise RuntimeError(f"sequence {sha}: MSA query differs from the sequence")
    msa, kept = filter_gap_columns(encode_msa(rows, cfg.max_seqs), cfg.max_gap_fraction)
    n_eff, weights = neff(msa.to(cfg.device), cfg.identity)
    contacts = potts_contact_map(fit_gremlin(msa, weights, GremlinConfig(), cfg.device))
    full = full_length_map(contacts, kept, len(sequence)).numpy().astype(np.float16)
    path = cfg.out / "couplings" / f"{sha}.npy"
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, full)
    os.replace(tmp, path)
    stats = {
        "depth": len(rows),
        "depth_filtered": int(msa.shape[0]),
        "kept_columns": int(kept.numel()),
        "neff": n_eff,
        "settings": _fit_settings(cfg),
        "device": cfg.device,
    }
    _write_atomic(cfg.out / "stats" / f"{sha}.json", json.dumps(stats).encode())


def _git_commit() -> str | None:
    """HEAD of the repository holding this package, or None outside a git checkout."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip() or None


def _bin(value: float, edges: tuple[float, float]) -> str:
    return NEFF_BINS[0 if value < edges[0] else 1 if value < edges[1] else 2]


def _validate(out: Path, rows: list[dict[str, Any]]) -> None:
    """Every row's files exist, match their checksums, and every map is ``(L, L)``.

    Raises:
        RuntimeError: A file is missing, differs from its recorded checksum, or has a wrong shape.
    """
    for row in rows:
        sha = row["sha256"]
        for path, digest in (
            (out / "msas" / f"{sha}.a3m.gz", row["msa_sha256"]),
            (out / "couplings" / f"{sha}.npy", row["couplings_sha256"]),
        ):
            if not path.is_file() or sha256sum(path) != digest:
                raise RuntimeError(f"{path}: missing or does not match the manifest checksum")
        shape = np.load(out / "couplings" / f"{sha}.npy", mmap_mode="r").shape
        if shape != (row["length"], row["length"]):
            raise RuntimeError(f"{row['set']}/{row['name']}: map shape {shape}")


def render_readme(summary: dict[str, Any]) -> str:
    """The dataset card: layout, how the maps were fit, and the counts per set and Neff bin."""
    lines = "\n".join(
        f"| `{name}` | {' | '.join(str(bins[b]) for b in NEFF_BINS)} |"
        for name, bins in summary["bins"].items()
    )
    config = textwrap.indent(json.dumps(summary["config"], indent=2, sort_keys=True), "    ")
    gremlin = textwrap.indent(json.dumps(summary["gremlin"], indent=2, sort_keys=True), "    ")
    edges = ", ".join(f"{e:.4g}" for e in summary["neff_bin_edges"])
    return (
        "# OPLM coevolution reference\n\n"
        f"Built {summary['built']} by `oplm fold eval-coevolution` (oplm {summary['oplm']}, "
        f"commit {summary['commit']}). "
        "Design: `docs/superpowers/specs/2026-11-07-coevolution-eval-design.md` in the oplm "
        "repository.\n\n"
        f"MSAs come from the ColabFold server at `{summary['host']}` (mode `{summary['mode']}`; "
        f"ColabFold databases: {summary['colabfold_db']}). `couplings/<sha256>.npy` is a float16 "
        "(L, L) APC-corrected GREMLIN Potts contact map over the full sequence (NaN on columns "
        "dropped for gaps), keyed by the sha256 of the sequence. `stats/<sha256>.json` holds "
        "the MSA depth, kept columns and Neff. `manifest.parquet` has one row per structure.\n\n"
        f"Neff bins use Neff / sqrt(L) with edges {edges} (low below the first, mid below the "
        "second, high otherwise).\n\n"
        f"| set | {' | '.join(NEFF_BINS)} |\n|---|---|---|---|\n{lines}\n\n"
        f"## Build settings\n\n{config}\n\n"
        "## GREMLIN settings\n\n"
        "The reference implementation and values are recorded in "
        "`src/oplm/fold/evalsets/gremlin.py`.\n\n"
        f"{gremlin}\n"
    )


def build_coevolution(cfg: CoevolutionBuildConfig) -> dict[str, Any]:
    """Fetch MSAs, fit Potts maps and write the manifest, summary and README under ``cfg.out``.

    Reruns reuse existing MSAs and maps. With ``cfg.fit`` false it stops after the MSAs.

    Returns:
        The summary also written to ``summary.json`` (empty of bins when ``fit`` is false).

    Raises:
        FileNotFoundError: A set directory is missing.
        ValueError: A stored fit used different settings than this run.
        RuntimeError: The server reports ERROR for a batch (message carries its sha256s), or
            the written files don't validate.
    """
    from oplm.data import load_structures

    structures: dict[str, list[tuple[str, str]]] = {}
    for name in cfg.sets:
        structures[name] = [(s.name, s.sequence) for s in load_structures(cfg.sets_root / name)]
    sequences = {_sha(seq): seq for items in structures.values() for _, seq in items}
    order = sorted(sequences)

    missing = [sha for sha in order if not (cfg.out / "msas" / f"{sha}.a3m.gz").exists()]
    for start in range(0, len(missing), cfg.msa_batch_size):
        batch = missing[start : start + cfg.msa_batch_size]
        try:
            texts = fetch_msas(
                [sequences[sha] for sha in batch], cfg.host, fetch=cfg.fetch, mode=_MSA_MODE
            )
        except Exception as exc:
            raise RuntimeError(f"sequences {', '.join(batch)}: {exc!r}") from exc
        for sha, text in zip(batch, texts, strict=True):
            _write_atomic(cfg.out / "msas" / f"{sha}.a3m.gz", gzip.compress(text.encode(), mtime=0))
        logger.info("msas %d/%d fetched", start + len(batch), len(missing))
    fetched = len(missing)
    if not cfg.fit:
        logger.info("%d MSAs in %s; stopping before the fit (--no-fit)", len(order), cfg.out)
        return {"sequences": len(order), "fetched": fetched}

    for i, sha in enumerate(order, 1):
        stats_path = cfg.out / "stats" / f"{sha}.json"
        if stats_path.exists() and (cfg.out / "couplings" / f"{sha}.npy").exists():
            stored = json.loads(stats_path.read_text()).get("settings", {})
            current = _fit_settings(cfg)
            differ = sorted(
                k for k in current.keys() | stored.keys() if stored.get(k) != current.get(k)
            )
            if differ:
                raise ValueError(
                    f"sequence {sha}: stored fit used different settings ({', '.join(differ)}); "
                    f"delete {cfg.out}/couplings and {cfg.out}/stats to refit"
                )
        else:
            _fit_one(cfg, sequences[sha], sha)
        if i % _LOG_EVERY == 0 or i == len(order):
            logger.info("fits %d/%d", i, len(order))

    stats = {sha: json.loads((cfg.out / "stats" / f"{sha}.json").read_text()) for sha in order}
    per_sqrt_l = {sha: stats[sha]["neff"] / math.sqrt(len(sequences[sha])) for sha in order}
    if cfg.bin_edges is not None:
        edges = cfg.bin_edges
    else:
        q = np.quantile(list(per_sqrt_l.values()), [1 / 3, 2 / 3])
        edges = (float(q[0]), float(q[1]))

    rows: list[dict[str, Any]] = []
    bins: dict[str, dict[str, int]] = {}
    for set_name, items in structures.items():
        bins[set_name] = dict.fromkeys(NEFF_BINS, 0)
        for name, seq in sorted(items):
            sha = _sha(seq)
            bin_ = _bin(per_sqrt_l[sha], edges)
            bins[set_name][bin_] += 1
            rows.append(
                {
                    "set": set_name,
                    "name": name,
                    "sha256": sha,
                    "length": len(seq),
                    "msa_depth": stats[sha]["depth"],
                    "msa_depth_filtered": stats[sha]["depth_filtered"],
                    "neff": stats[sha]["neff"],
                    "neff_per_sqrt_l": per_sqrt_l[sha],
                    "neff_bin": bin_,
                    "msa_sha256": sha256sum(cfg.out / "msas" / f"{sha}.a3m.gz"),
                    "couplings_sha256": sha256sum(cfg.out / "couplings" / f"{sha}.npy"),
                }
            )
    _validate(cfg.out, rows)
    for set_name, counts in bins.items():
        for bin_, count in counts.items():
            if count == 0:
                logger.warning("%s: no structures in the %s Neff bin", set_name, bin_)

    skip = {"sets_root", "out", "fetch"}
    summary: dict[str, Any] = {
        "built": datetime.date.today().isoformat(),
        "oplm": metadata.version("oplm"),
        "commit": _git_commit(),
        "host": cfg.host,
        "mode": _MSA_MODE,
        "colabfold_db": cfg.colabfold_db,
        "config": {k: v for k, v in dataclasses.asdict(cfg).items() if k not in skip},
        "gremlin": dataclasses.asdict(GremlinConfig()),
        "neff_bin_edges": list(edges),
        "bins": bins,
        "sequences": len(order),
    }
    pq.write_table(pa.Table.from_pylist(rows), cfg.out / "manifest.parquet")
    (cfg.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (cfg.out / "README.md").write_text(render_readme(summary))
    return summary

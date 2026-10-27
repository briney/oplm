"""CAMEO and PDB40 chains from a preprocessed fold dataset (``oplm fold preprocess``)."""

from __future__ import annotations

import subprocess
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq

from oplm.fold.eval.targets import load_targets, read_targets
from oplm.fold.evalsets.chain import STANDARD_AMINO_ACIDS, EvalChain, from_eval_target, rejection

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "CLUSTER_MIN_LENGTH",
    "MMSEQS_ARGS",
    "QUALITY_LIMITS",
    "cameo_chains",
    "mmseqs_version",
    "passes_quality",
    "pdb40_chains",
    "read_clusters",
    "run_mmseqs",
    "write_cluster_fasta",
]

QUALITY_LIMITS: dict[str, float] = {"X-RAY DIFFRACTION": 2.5, "ELECTRON MICROSCOPY": 3.0}
CLUSTER_MIN_LENGTH = 30
MMSEQS_ARGS = ("--min-seq-id", "0.4", "-c", "0.8", "--cov-mode", "0", "-s", "7.5")
_META_COLUMNS = ["id", "chain_id", "auth_chain_id", "method", "resolution", "release_date"]
_CANDIDATE_COLUMNS = [
    *_META_COLUMNS,
    "chain_index",
    "sequence",
    "length",
    "resolved_count",
    "shard",
    "row",
]


def passes_quality(method: str | None, resolution: float | None) -> bool:
    """X-ray at 2.5 Å or better, or cryo-EM at 3.0 Å or better; anything else fails."""
    limit = QUALITY_LIMITS.get(method or "")
    return limit is not None and resolution is not None and resolution <= limit


def _meta(row: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "source": source,
        "entry": row["id"],
        "chain": row["auth_chain_id"],
        "method": row["method"],
        "resolution": row["resolution"],
        "release_date": row["release_date"],
    }


def cameo_chains(root: Path, target_set: str = "cameo") -> tuple[list[EvalChain], dict[str, int]]:
    """The listed CAMEO chains that pass the quality filter.

    Counts: ``listed``, ``missing`` (not in the snapshot), ``skipped`` (fewer than three
    resolved CA atoms, per :func:`load_targets`) and ``quality``.
    """
    targets, coverage = load_targets(root, target_set)
    index = pq.read_table(root / "index.parquet", columns=_META_COLUMNS).to_pylist()
    meta = {(r["id"], r["chain_id"]): r for r in index}
    stats = {
        "listed": coverage.listed,
        "missing": len(coverage.missing),
        "skipped": len(coverage.skipped),
        "quality": 0,
    }
    chains: list[EvalChain] = []
    for target in targets:
        row = meta[(target.entry_id, target.chain_id)]
        if passes_quality(row["method"], row["resolution"]):
            chains.append(from_eval_target(target, meta=_meta(row, "cameo")))
        else:
            stats["quality"] += 1
    return chains, stats


def write_cluster_fasta(root: Path, path: Path) -> dict[str, str]:
    """Write the dataset's distinct protein sequences of length >= 30 as FASTA.

    Returns ``{fasta id: sequence}``. Ids are ``s000000``, ``s000001``, ... in sorted-sequence
    order, so a rerun on the same snapshot writes the same file.
    """
    column = pq.read_table(root / "index.parquet", columns=["sequence"]).column("sequence")
    distinct = sorted({s for s in column.to_pylist() if len(s) >= CLUSTER_MIN_LENGTH})
    ids = {f"s{i:06d}": sequence for i, sequence in enumerate(distinct)}
    path.write_text("".join(f">{name}\n{sequence}\n" for name, sequence in ids.items()))
    return ids


def run_mmseqs(fasta: Path, workdir: Path, *, mmseqs: str = "mmseqs", threads: int = 1) -> Path:
    """Cluster ``fasta`` at 40% identity and 80% coverage; return the cluster TSV.

    Raises:
        RuntimeError: ``mmseqs`` is missing or exits non-zero (the message carries stderr).
    """
    prefix = workdir / "pdb40"
    command = [
        mmseqs,
        "easy-cluster",
        str(fasta),
        str(prefix),
        str(workdir / "mmseqs_tmp"),
        *MMSEQS_ARGS,
        "--threads",
        str(threads),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"MMseqs2 executable not found: {mmseqs}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{mmseqs} easy-cluster failed:\n{exc.stderr}") from exc
    return workdir / "pdb40_cluster.tsv"


def mmseqs_version(mmseqs: str = "mmseqs") -> str:
    """The ``mmseqs version`` string, recorded in the README."""
    return subprocess.run(
        [mmseqs, "version"], check=True, capture_output=True, text=True
    ).stdout.strip()


def read_clusters(tsv: Path) -> dict[str, str]:
    """``{member id: representative id}`` from an MMseqs2 cluster TSV."""
    clusters: dict[str, str] = {}
    for line in tsv.read_text().splitlines():
        representative, member = line.split("\t")
        clusters[member] = representative
    return clusters


def pdb40_chains(
    root: Path,
    sequence_cluster: dict[str, str],
    *,
    min_length: int,
    max_length: int,
    min_modeled_fraction: float,
    max_rounds: int = 5,
) -> tuple[list[EvalChain], dict[str, int]]:
    """One representative chain per cluster.

    Candidates are chains whose sequence is clustered, that pass the quality filter, with a
    length in ``[min_length, max_length]``, only standard amino acids, and a resolved fraction
    of at least ``min_modeled_fraction``. Each cluster's candidates are tried best first
    (resolution, then resolved fraction, then entry and chain index) against
    :func:`rejection`, one round per attempt, reading each round's chains in one pass over the
    shards; the first that passes represents the cluster.

    Counts: ``clusters``; ``no_candidate``; ``exhausted`` (every candidate failed);
    ``max_rounds`` (candidates left untried after ``max_rounds`` attempts); and one count per
    rejection reason over all attempts.
    """
    queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(root / "index.parquet", columns=_CANDIDATE_COLUMNS).to_pylist():
        cluster = sequence_cluster.get(row["sequence"])
        if (
            cluster is None
            or not passes_quality(row["method"], row["resolution"])
            or not min_length <= row["length"] <= max_length
            or set(row["sequence"]) - STANDARD_AMINO_ACIDS
            or row["resolved_count"] < min_modeled_fraction * row["length"]
        ):
            continue
        queues[cluster].append(row)
    for queue in queues.values():
        queue.sort(
            key=lambda r: (
                r["resolution"],
                -r["resolved_count"] / r["length"],
                r["id"],
                r["chain_index"],
            )
        )
    stats: dict[str, int] = defaultdict(int)
    stats["clusters"] = len(set(sequence_cluster.values()))
    stats["no_candidate"] = stats["clusters"] - len(queues)
    stats["exhausted"] = 0
    chosen: dict[str, EvalChain] = {}
    pending = dict.fromkeys(queues, 0)
    # ponytail: one process; parallelize the shard reads if the full build gets slow.
    for _ in range(max_rounds):
        if not pending:
            break
        rows = {cluster: queues[cluster][k] for cluster, k in pending.items()}
        by_chain = {(row["id"], row["chain_id"]): cluster for cluster, row in rows.items()}
        following: dict[str, int] = {}
        for target in read_targets(root, list(rows.values()), "pdb40"):
            cluster = by_chain[(target.entry_id, target.chain_id)]
            meta = _meta(rows[cluster], "pdb40") | {"cluster": cluster}
            chain = from_eval_target(target, meta=meta)
            reason = rejection(
                chain,
                min_length=min_length,
                max_length=max_length,
                min_modeled_fraction=min_modeled_fraction,
            )
            if reason is None:
                chosen[cluster] = chain
            elif pending[cluster] + 1 < len(queues[cluster]):
                stats[reason] += 1
                following[cluster] = pending[cluster] + 1
            else:
                stats[reason] += 1
                stats["exhausted"] += 1
        pending = following
    stats["max_rounds"] = len(pending)
    return sorted(chosen.values(), key=lambda c: c.name), dict(stats)

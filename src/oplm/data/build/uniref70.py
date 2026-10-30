"""UniRef90 -> 70% identity linclust stage and a streaming parser for its cluster output."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from oplm.data.build.mmseqs import run_mmseqs_cmd

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

__all__ = ["Cluster", "cluster_uniref90", "iter_clusters"]


class Cluster(NamedTuple):
    """One cluster; the representative's own entry is first in the member lists."""

    rep_id: str
    member_ids: list[str]
    member_sequences: list[str]


def cluster_uniref90(
    fasta: Path, workdir: Path, *, mmseqs: str, threads: int, extra_args: Sequence[str] = ()
) -> Path:
    """Run ``easy-linclust`` at 70% identity / 80% coverage; return ``*_all_seqs.fasta``.

    Raises:
        RuntimeError: ``mmseqs`` is missing or exits non-zero.
    """
    prefix = workdir / "uniref70"
    run_mmseqs_cmd(
        mmseqs,
        [
            "easy-linclust",
            str(fasta),
            str(prefix),
            str(workdir / "tmp"),
            *("--min-seq-id", "0.7", "-c", "0.8", "--cov-mode", "0", "--dbtype", "1"),
            "--threads",
            str(threads),
            *extra_args,
        ],
    )
    return workdir / "uniref70_all_seqs.fasta"


def iter_clusters(all_seqs_fasta: Path) -> Iterator[Cluster]:
    """Stream clusters from an ``easy-linclust`` all-seqs FASTA.

    A header directly followed by another header opens a cluster (it carries only the rep
    id); header+sequence pairs then list the members, rep first. Sequences may be wrapped.
    """
    rep: str | None = None
    ids: list[str] = []
    seqs: list[str] = []
    parts: list[str] = []
    pending_header: str | None = None  # header seen, no sequence line yet

    def flush_member() -> None:
        if ids and len(seqs) < len(ids):
            seqs.append("".join(parts))
        parts.clear()

    with all_seqs_fasta.open() as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith(">"):
                if pending_header is not None:
                    # two headers in a row: the earlier one opens a new cluster
                    if rep is not None:
                        yield Cluster(rep, ids, seqs)
                    rep, ids, seqs = pending_header, [], []
                else:
                    flush_member()
                pending_header = line[1:].split(None, 1)[0]
                continue
            if pending_header is not None:
                ids.append(pending_header)
                pending_header = None
            parts.append(line.strip())
    flush_member()
    if rep is not None:
        yield Cluster(rep, ids, seqs)

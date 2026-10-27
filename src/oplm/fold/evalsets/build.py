"""``oplm fold eval-sets``: build, validate and describe the structure eval sets.

Spec: docs/superpowers/specs/2026-10-27-structure-eval-sets-design.md.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import random
import shutil
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from oplm.fold.evalsets.casp import (
    CASP_EDITIONS,
    casp_domain_chains,
    read_fasta,
    sha256sum,
    verify_casp_sources,
)
from oplm.fold.evalsets.chain import EvalChain, count_long_range_contacts, rejection, write_pdb
from oplm.fold.evalsets.pdb import (
    cameo_chains,
    mmseqs_version,
    pdb40_chains,
    read_clusters,
    run_mmseqs,
    write_cluster_fasta,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = ["CASP_SET", "BuildConfig", "build", "nested_subsets", "render_readme"]

CASP_SET = "casp15_16"
_MANIFEST_FIELDS = (
    "source",
    "entry",
    "chain",
    "cluster",
    "method",
    "resolution",
    "release_date",
    "source_file",
)


@dataclass
class BuildConfig:
    """Inputs and thresholds; the defaults are the approved spec values."""

    fold_data: Path
    casp_sources: Path
    out: Path
    mmseqs: str = "mmseqs"
    threads: int = 1
    seed: int = 42
    casp_length: tuple[int, int] = (50, 1024)
    cameo_length: tuple[int, int] = (50, 512)
    pdb40_length: tuple[int, int] = (50, 1024)
    subset_max_length: int = 512
    min_modeled_fraction: float = 0.7
    cameo_subsets: tuple[int, ...] = (256,)
    pdb40_subsets: tuple[int, ...] = (256, 2048)


def nested_subsets(names: list[str], sizes: Sequence[int], seed: int) -> dict[int, list[str]]:
    """Seeded subsets of ``names``, one per size, each contained in every larger one."""
    order = sorted(names)
    random.Random(seed).shuffle(order)
    return {size: sorted(order[:size]) for size in sizes}


def _curate(
    chains: list[EvalChain], length: tuple[int, int], min_modeled_fraction: float
) -> tuple[list[EvalChain], dict[str, int]]:
    kept: list[EvalChain] = []
    reasons: dict[str, int] = defaultdict(int)
    for chain in chains:
        reason = rejection(
            chain,
            min_length=length[0],
            max_length=length[1],
            min_modeled_fraction=min_modeled_fraction,
        )
        if reason is None:
            kept.append(chain)
        else:
            reasons[reason] += 1
    return sorted(kept, key=lambda c: c.name), dict(reasons)


def _write_set(directory: Path, chains: list[EvalChain]) -> None:
    directory.mkdir(parents=True)
    for chain in chains:
        write_pdb(chain, directory / f"{chain.name}.pdb")


def _validate(out: Path, built: dict[str, list[EvalChain]]) -> None:
    """Every set directory loads back as built: names, sequences, modeled residues, contacts.

    Raises:
        RuntimeError: A directory or file doesn't round-trip.
    """
    from oplm.data import load_structures
    from oplm.eval.metrics.contact import contact_positions_valid

    for name, chains in built.items():
        expected = {chain.name: chain for chain in chains}
        loaded = load_structures(out / name)
        if sorted(s.name for s in loaded) != sorted(expected):
            raise RuntimeError(f"{name}: loaded {len(loaded)} structures, built {len(expected)}")
        for structure in loaded:
            chain = expected[structure.name]
            built_valid = np.isfinite(chain.backbone()).all(axis=(1, 2))
            if (
                structure.sequence != chain.sequence
                or not np.array_equal(
                    contact_positions_valid(structure.coords).numpy(), built_valid
                )
                or count_long_range_contacts(structure.coords.numpy())
                != chain.meta["long_range_contacts"]
            ):
                raise RuntimeError(f"{name}/{structure.name}: does not load back as built")


def _write_manifest(path: Path, out: Path, built: dict[str, list[EvalChain]]) -> None:
    rows = [
        {
            "set": name,
            "name": chain.name,
            **{key: chain.meta.get(key) for key in _MANIFEST_FIELDS},
            "sequence": chain.sequence,
            "length": len(chain.sequence),
            "modeled": chain.modeled,
            "long_range_contacts": chain.meta["long_range_contacts"],
            "sha256": sha256sum(out / name / f"{chain.name}.pdb"),
        }
        for name, chains in built.items()
        for chain in chains
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def render_readme(summary: dict[str, Any]) -> str:
    """The dataset card: what each directory holds, how it was built, and the counts."""
    sets = "\n".join(f"| `{name}` | {count} |" for name, count in summary["exported"].items())
    sources = "\n".join(
        f"| `{name}` | `{digest}` |" for name, digest in sorted(summary["sources"].items())
    )
    config = textwrap.indent(json.dumps(summary["config"], indent=2, sort_keys=True), "    ")
    counts = {"sets": summary["sets"], "overlap": summary["overlap"]}
    stages = textwrap.indent(json.dumps(counts, indent=2, sort_keys=True), "    ")
    return (
        "# OPLM structure eval sets\n\n"
        f"Built {summary['built']} by `oplm fold eval-sets` (oplm {summary['oplm']}, "
        f"MMseqs2 {summary['mmseqs']}). Design and curation rules: "
        "`docs/superpowers/specs/2026-10-27-structure-eval-sets-design.md` in the oplm "
        "repository.\n\n"
        "Each directory holds one-chain PDB files for the `structure` eval type. SEQRES carries "
        "the full sequence and residues are numbered by sequence position, so unmodeled "
        "residues load as NaN rows and are left out of scoring. `manifest.parquet` has one row "
        "per file; `summary.json` has the numbers below.\n\n"
        f"| directory | structures |\n|---|---|\n{sets}\n\n"
        f"## Thresholds\n\n{config}\n\n"
        f"## Counts by stage\n\n{stages}\n\n"
        f"## Sources\n\n| file | sha256 |\n|---|---|\n{sources}\n"
    )


def build(cfg: BuildConfig) -> dict[str, Any]:
    """Build every set under ``cfg.out``, validate it, and write the manifest and README.

    Returns:
        The summary also written to ``summary.json`` and rendered into ``README.md``.

    Raises:
        FileExistsError: ``cfg.out`` exists and is not empty.
        RuntimeError: MMseqs2 failed, or a written file doesn't load back as built.
    """
    if cfg.out.exists() and any(cfg.out.iterdir()):
        raise FileExistsError(f"{cfg.out} exists and is not empty")
    sources = cfg.out / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    skip = {"fold_data", "casp_sources", "out", "mmseqs", "threads"}
    summary: dict[str, Any] = {
        "built": datetime.date.today().isoformat(),
        "oplm": metadata.version("oplm"),
        "mmseqs": mmseqs_version(cfg.mmseqs),
        "config": {k: v for k, v in dataclasses.asdict(cfg).items() if k not in skip},
        "sources": {},
        "sets": {},
    }
    for path in verify_casp_sources(cfg.casp_sources):
        shutil.copy2(path, sources / path.name)
        summary["sources"][path.name] = sha256sum(path)
    for name, path in (
        ("cameo.json", cfg.fold_data / "targets" / "cameo.json"),
        ("fold_manifest.json", cfg.fold_data / "manifest.json"),
    ):
        shutil.copy2(path, sources / name)
        summary["sources"][name] = sha256sum(path)

    built: dict[str, list[EvalChain]] = {}
    casp_chains: list[EvalChain] = []
    casp_stats: dict[str, Any] = {}
    for edition, (tarball, fasta) in CASP_EDITIONS.items():
        chains, stats = casp_domain_chains(sources / tarball, read_fasta(sources / fasta), edition)
        casp_chains += chains
        casp_stats[f"casp{edition}"] = stats
    built[CASP_SET], casp_stats["rules"] = _curate(casp_chains, cfg.casp_length, 0.0)
    summary["sets"][CASP_SET] = casp_stats

    cameo, cameo_stats = cameo_chains(cfg.fold_data)
    built["cameo"], cameo_rules = _curate(cameo, cfg.cameo_length, cfg.min_modeled_fraction)
    summary["sets"]["cameo"] = {**cameo_stats, "rules": cameo_rules}

    ids = write_cluster_fasta(cfg.fold_data, sources / "pdb40.fasta")
    tsv = run_mmseqs(sources / "pdb40.fasta", sources, mmseqs=cfg.mmseqs, threads=cfg.threads)
    shutil.rmtree(sources / "mmseqs_tmp", ignore_errors=True)
    for extra in ("pdb40_rep_seq.fasta", "pdb40_all_seqs.fasta"):
        (sources / extra).unlink(missing_ok=True)
    sequence_cluster = {ids[member]: rep for member, rep in read_clusters(tsv).items()}
    built["pdb40"], summary["sets"]["pdb40"] = pdb40_chains(
        cfg.fold_data,
        sequence_cluster,
        min_length=cfg.pdb40_length[0],
        max_length=cfg.pdb40_length[1],
        min_modeled_fraction=cfg.min_modeled_fraction,
    )
    summary["sets"]["pdb40"]["sequences"] = len(ids)

    for chains in built.values():
        for chain in chains:
            chain.meta["long_range_contacts"] = count_long_range_contacts(chain.backbone())
    for name, chains in list(built.items()):
        _write_set(cfg.out / name, chains)
    for parent, sizes in (("cameo", cfg.cameo_subsets), ("pdb40", cfg.pdb40_subsets)):
        by_name = {chain.name: chain for chain in built[parent]}
        eligible = [n for n, c in by_name.items() if len(c.sequence) <= cfg.subset_max_length]
        for size, names in nested_subsets(eligible, sizes, cfg.seed).items():
            subset = f"{parent}_{size}"
            (cfg.out / subset).mkdir()
            for n in names:
                shutil.copy2(cfg.out / parent / f"{n}.pdb", cfg.out / subset / f"{n}.pdb")
            built[subset] = [by_name[n] for n in names]

    cameo_clusters = {sequence_cluster.get(c.sequence) for c in built["cameo"]} - {None}
    pdb40_clusters = {c.meta["cluster"] for c in built["pdb40"]}
    summary["overlap"] = {
        "cameo_chains_in_pdb40": len(
            {c.name for c in built["cameo"]} & {c.name for c in built["pdb40"]}
        ),
        "cameo_clusters_in_pdb40": len(cameo_clusters & pdb40_clusters),
    }

    _validate(cfg.out, built)
    _write_manifest(cfg.out / "manifest.parquet", cfg.out, built)
    summary["exported"] = {name: len(chains) for name, chains in built.items()}
    (cfg.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (cfg.out / "README.md").write_text(render_readme(summary))
    return summary

"""Structure shards: ``oplm fold preprocess`` and the resumable ``StructureDataset``.

Preprocessing (addendum §4): parse the archive in worker processes, write Parquet shards of
about ``shard_bytes`` (one row per entry, small row groups so a training worker reads one
entry without the whole shard), a per-chain index whose ``split`` column carries the date
cutoff and held-out rules, the held-out target manifests, the CCD table, input checksums,
``manifest.json`` and the dataset card. The output directory is the Hugging Face dataset repo.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from oplm.data.sequence.dataset import ShardedProteinDataset
from oplm.fold.data.ccd import ChemCompTable, ReferenceConformers
from oplm.fold.data.complex import complex_to_row, row_to_complex, rows_to_table
from oplm.fold.data.crop import crop_chain
from oplm.fold.data.featurize import featurize_complex

if TYPE_CHECKING:
    from collections.abc import Iterator

    from oplm.fold.data.complex import MolecularComplex

logger = logging.getLogger(__name__)

__all__ = [
    "PreprocessConfig",
    "StructureDataset",
    "assign_splits",
    "discover_entries",
    "preprocess",
    "read_target_list",
]

SCHEMA_VERSION = 1
_CIF_SUFFIXES = (".cif.gz", ".cif")
_INDEX_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("chain_id", pa.string()),
        ("auth_chain_id", pa.string()),
        ("chain_index", pa.int32()),
        ("entity_id", pa.int32()),
        ("sym_id", pa.int32()),
        ("sequence", pa.string()),
        ("length", pa.int32()),
        ("resolved_count", pa.int32()),
        ("n_chains", pa.int32()),
        ("resolution", pa.float32()),
        ("release_date", pa.string()),
        ("method", pa.string()),
        ("shard", pa.string()),
        ("row", pa.int32()),
        ("split", pa.string()),
    ]
)


@dataclass
class PreprocessConfig:
    """Everything ``oplm fold preprocess`` needs; recorded verbatim in the manifest."""

    archive: Path  # root of the mmCIF tree (searched recursively)
    ccd: Path  # components.cif(.gz), or a ccd_table.json written by a previous run
    out: Path
    workers: int = max(1, (os.cpu_count() or 2) - 1)
    shard_bytes: int = 1 << 30
    row_group_size: int = 16
    cutoff: str = "2021-09-30"
    valid_size: int = 64
    valid_seed: int = 0
    heldout: dict[str, Path] = field(default_factory=dict)  # split name -> target list file
    ids: Path | None = None  # optional subset: one entry id per line
    source: str = "pdb"
    max_resolution: float = 9.0
    min_resolved_residues: int = 4


def _entry_id(path: Path) -> str:
    name = path.name
    for suffix in _CIF_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)].lower()
    raise ValueError(f"not an mmCIF path: {path}")


def discover_entries(archive: Path, ids: Path | None = None) -> list[Path]:
    """Every ``.cif``/``.cif.gz`` under ``archive``, sorted by entry id.

    Args:
        archive: Root of the mmCIF tree, searched recursively.
        ids: Optional file with one entry id per line (case-insensitive) restricting the result.

    Raises:
        ValueError: Two files map to the same entry id (case-insensitive).
    """
    wanted = None
    if ids is not None:
        wanted = {
            line.strip().lower() for line in Path(ids).read_text().splitlines() if line.strip()
        }
    found: dict[str, Path] = {}
    for path in sorted(p for p in Path(archive).rglob("*") if p.name.endswith(_CIF_SUFFIXES)):
        entry = _entry_id(path)
        if wanted is not None and entry not in wanted:
            continue
        if entry in found:
            raise ValueError(f"entry {entry!r} appears twice: {found[entry]} and {path}")
        found[entry] = path
    return [found[k] for k in sorted(found)]


def read_target_list(path: Path) -> list[tuple[str, str | None]]:
    """Parse a held-out target list: one ``PDBID`` or ``PDBID_CHAIN`` per line; ``#`` comments."""
    targets: list[tuple[str, str | None]] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        token = line.replace(",", " ").split()[0]
        entry, _, chain = token.partition("_")
        targets.append((entry.lower(), chain or None))
    return targets


def _load_ccd(path: Path) -> ChemCompTable:
    """A ``ccd_table.json`` cache when ``path`` is JSON, else parse ``components.cif(.gz)``."""
    return ChemCompTable.load(path) if path.suffix == ".json" else ChemCompTable.parse(path)


# ----------------------------------------------------------------------------- workers

_WORKER: dict[str, Any] = {}


def _init_worker(ccd: str, max_resolution: float, min_resolved_residues: int, archive: str) -> None:
    _WORKER.update(
        archive=Path(archive),
        ccd=_load_ccd(Path(ccd)),
        conformers=ReferenceConformers.load(),
        max_resolution=max_resolution,
        min_resolved_residues=min_resolved_residues,
    )


def _chain_stats(c: MolecularComplex) -> list[dict[str, Any]]:
    stats = []
    for k, ch in enumerate(c.chains):
        tokens = c.chain_tokens(k)
        resolved = 0
        for t in tokens:
            s = int(c.token_atom_start[t])
            if c.atom_resolved[s : s + int(c.token_atom_count[t])].any():
                resolved += 1
        stats.append(
            {
                "chain_id": ch.chain_id,
                "auth_chain_id": ch.auth_chain_id,
                "chain_index": k,
                "entity_id": ch.entity_id,
                "sym_id": ch.sym_id,
                "sequence": ch.sequence,
                "length": len(ch.sequence),
                "resolved_count": resolved,
            }
        )
    return stats


def _row_bytes(c: MolecularComplex) -> int:
    """Rough in-memory size of a shard row (array bytes plus an allowance per atom name)."""
    arrays = (
        c.atom_coords,
        c.atom_resolved,
        c.atom_element,
        c.atom_charge,
        c.atom_bfactor,
        c.atom_to_token,
        c.token_res_type,
        c.token_mol_type,
        c.token_asym_id,
        c.token_entity_id,
        c.token_sym_id,
        c.token_residue_index,
        c.token_center_atom,
        c.token_rep_atom,
        c.token_atom_start,
        c.token_atom_count,
        c.bonds,
    )
    return sum(int(a.nbytes) for a in arrays) + 8 * c.num_atoms


def _parse_one(path_str: str) -> dict[str, Any]:
    from oplm.fold.data.mmcif import Rejection, parse_mmcif

    path = Path(path_str)
    # ``path`` is relative to the archive root so ``sha256sum -c checksums.txt`` works there.
    out: dict[str, Any] = {"path": str(path.relative_to(_WORKER["archive"])), "sha256": ""}
    try:
        out["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result = parse_mmcif(
            path,
            ccd=_WORKER["ccd"],
            conformers=_WORKER["conformers"],
            max_resolution=_WORKER["max_resolution"],
            min_resolved_residues=_WORKER["min_resolved_residues"],
        )
        if isinstance(result, Rejection):
            out["rejection"] = {"id": result.id, "reason": result.reason, "detail": result.detail}
            return out
        out.update(
            row=complex_to_row(result),
            nbytes=_row_bytes(result),
            chains=_chain_stats(result),
            metadata={
                "id": result.metadata.id,
                "resolution": result.metadata.resolution,
                "release_date": result.metadata.release_date,
                "method": result.metadata.method,
                "dropped": result.metadata.dropped,
            },
        )
    except Exception as exc:  # one bad entry must not abort an hours-long run: count it instead
        out["rejection"] = {
            "id": path.name,
            "reason": "internal_error",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    return out


def _parse_all(
    cfg: PreprocessConfig, paths: list[Path], ccd_json: Path
) -> Iterator[dict[str, Any]]:
    # Workers load the small saved JSON, not the multi-hundred-MB components.cif.gz.
    init_args = (str(ccd_json), cfg.max_resolution, cfg.min_resolved_residues, str(cfg.archive))
    if cfg.workers <= 1:
        _init_worker(*init_args)
        yield from (_parse_one(str(p)) for p in paths)
        return
    with ProcessPoolExecutor(
        max_workers=cfg.workers,
        mp_context=get_context("spawn"),
        initializer=_init_worker,
        initargs=init_args,
    ) as pool:
        yield from pool.map(_parse_one, [str(p) for p in paths], chunksize=8)


# ----------------------------------------------------------------------------- shards


class _ShardWriter:
    """Buffer rows and write a shard whenever the buffered size reaches ``shard_bytes``."""

    def __init__(self, directory: Path, shard_bytes: int, row_group_size: int) -> None:
        self._dir = directory
        self._shard_bytes = shard_bytes
        self._row_group_size = row_group_size
        self._rows: list[dict[str, object]] = []
        self._bytes = 0
        self.shards: list[dict[str, Any]] = []
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def current(self) -> tuple[str, int]:
        """``(shard file name, row index)`` the next appended row will get."""
        return f"shard-{len(self.shards):05d}.parquet", len(self._rows)

    def append(self, row: dict[str, object], nbytes: int) -> None:
        self._rows.append(row)
        self._bytes += nbytes
        if self._bytes >= self._shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        name, _ = self.current
        path = self._dir / name
        pq.write_table(rows_to_table(self._rows), path, row_group_size=self._row_group_size)
        self.shards.append({"file": name, "rows": len(self._rows), "bytes": path.stat().st_size})
        self._rows, self._bytes = [], 0


# ----------------------------------------------------------------------------- splits


def assign_splits(
    index_rows: list[dict[str, Any]],
    *,
    cutoff: str,
    heldout: dict[str, set[str]],
    valid_size: int,
    valid_seed: int,
) -> None:
    """Fill ``row["split"]`` for every chain row (entry-level decisions; addendum §4).

    Held-out entries take their list's name (first match in ``heldout`` order). A pre-cutoff
    entry sharing any chain sequence with a held-out entry is ``excluded_heldout_sequence``;
    other pre-cutoff entries are ``train``; entries without a date are ``undated``. Post-cutoff
    entries are ``post_cutoff`` except the seeded ``valid`` choice of ``valid_size`` monomers
    (one chain, resolution ≤ 2.5 Å, length 50–400).
    """
    by_entry: dict[str, list[dict[str, Any]]] = {}
    for row in index_rows:
        by_entry.setdefault(row["id"], []).append(row)
    heldout_sequences: set[str] = set()
    entry_split: dict[str, str] = {}
    for entry, rows in by_entry.items():
        for name, ids in heldout.items():
            if entry in ids:
                entry_split[entry] = name
                heldout_sequences.update(r["sequence"] for r in rows)
                break
    candidates: list[str] = []
    for entry, rows in by_entry.items():
        if entry in entry_split:
            continue
        date = rows[0]["release_date"]
        if not date:  # None, or '' from a '?' in the mmCIF
            entry_split[entry] = "undated"
        elif date <= cutoff:  # ISO dates compare lexicographically
            excluded = any(r["sequence"] in heldout_sequences for r in rows)
            entry_split[entry] = "excluded_heldout_sequence" if excluded else "train"
        else:
            entry_split[entry] = "post_cutoff"
            first = rows[0]
            if (
                first["n_chains"] == 1
                and first["resolution"] is not None
                and first["resolution"] <= 2.5
                and 50 <= first["length"] <= 400
            ):
                candidates.append(entry)
    candidates.sort()
    if candidates and valid_size > 0:
        g = torch.Generator().manual_seed(valid_seed)
        order = torch.randperm(len(candidates), generator=g).tolist()
        for i in order[:valid_size]:
            entry_split[candidates[i]] = "valid"
    for row in index_rows:
        row["split"] = entry_split[row["id"]]


# ----------------------------------------------------------------------------- driver

_CARD = """---
license: cc0-1.0
pretty_name: {pretty_json}
tags:
- protein-structure
- wwpdb
---

# {pretty}

Protein chains of the wwPDB archive parsed into OPLM's flat all-atom representation
(`oplm.fold.data.complex.MolecularComplex`) for training the OPLM structure prediction head.
Snapshot of {snapshot}; {accepted} entries accepted, {chains} chains; CCD {ccd}.
Source data: wwPDB (CC0 1.0) and the wwPDB Chemical Component Dictionary (CC0 1.0).

Layout: `shards/*.parquet` (one row per entry; see `manifest.json` for the schema version and
`oplm.fold.data.complex.COMPLEX_SCHEMA` for the columns), `index.parquet` (one row per chain
with the `split` column), `targets/*.json` (held-out target lists), `checksums.txt`
(sha256 of every archive entry file; the CCD's sha256 is in `manifest.json` under
`ccd.sha256`), `manifest.json` (provenance, configuration, counts).

Splits: {splits}

Load with `oplm.fold.data.dataset.StructureDataset(path, split="train", crop_size=384)`;
see `docs/FOLD.md` in https://github.com/briney/oplm.
"""


def preprocess(cfg: PreprocessConfig) -> dict[str, Any]:
    """Run the whole pipeline and return the manifest (also written to ``out/manifest.json``)."""
    cutoff = dt.date.fromisoformat(cfg.cutoff).isoformat()  # ValueError on a malformed cutoff
    out = Path(cfg.out)
    paths = discover_entries(cfg.archive, cfg.ids)
    if not paths:
        raise ValueError(f"no mmCIF files found under {cfg.archive}")
    for stale in ("shards", "targets"):  # ``out`` is the published repo: never mix two runs
        if (out / stale).is_dir() and any((out / stale).iterdir()):
            raise FileExistsError(f"{out / stale} is not empty; preprocess into a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    (out / "targets").mkdir(exist_ok=True)
    ccd_path = Path(cfg.ccd)
    ccd = _load_ccd(ccd_path)
    ccd.save(out / "ccd_table.json")
    heldout: dict[str, set[str]] = {}
    for name, list_path in cfg.heldout.items():
        targets = read_target_list(list_path)
        (out / "targets" / f"{name}.json").write_text(
            json.dumps([{"id": e, "chain": c} for e, c in targets], indent=1)
        )
        heldout[name] = {e for e, _ in targets}

    writer = _ShardWriter(out / "shards", cfg.shard_bytes, cfg.row_group_size)
    index_rows: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    rejections: list[dict[str, str]] = []
    dropped: dict[str, int] = {}
    checksums: list[str] = []
    accepted = 0
    for i, result in enumerate(_parse_all(cfg, paths, out / "ccd_table.json")):
        if result["sha256"]:  # '' when the file could not be read: no line for sha256sum -c
            checksums.append(f"{result['sha256']}  {result['path']}")
        if "rejection" in result:
            rej = result["rejection"]
            rejected[rej["reason"]] = rejected.get(rej["reason"], 0) + 1
            rejections.append(rej)
        else:
            accepted += 1
            meta = result["metadata"]
            for kind, n in meta["dropped"].items():
                dropped[kind] = dropped.get(kind, 0) + n
            shard, row_idx = writer.current
            for chain in result["chains"]:
                index_rows.append(
                    {
                        "id": meta["id"],
                        **chain,
                        "n_chains": len(result["chains"]),
                        "resolution": meta["resolution"],
                        "release_date": meta["release_date"],
                        "method": meta["method"],
                        "shard": shard,
                        "row": row_idx,
                        "split": "",
                    }
                )
            writer.append(result["row"], result["nbytes"])
        if (i + 1) % 1000 == 0 or i + 1 == len(paths):
            logger.info(
                "preprocess: %d/%d parsed, %d accepted, %d rejected, %d shards written",
                i + 1,
                len(paths),
                accepted,
                sum(rejected.values()),
                len(writer.shards),
            )
    writer.flush()

    assign_splits(
        index_rows,
        cutoff=cutoff,
        heldout=heldout,
        valid_size=cfg.valid_size,
        valid_seed=cfg.valid_seed,
    )
    present = {r["id"] for r in index_rows}
    heldout_missing = {name: sorted(ids - present) for name, ids in heldout.items()}
    pq.write_table(rows_to_index_table(index_rows), out / "index.parquet")
    (out / "checksums.txt").write_text("\n".join(checksums) + ("\n" if checksums else ""))

    entry_splits: dict[str, str] = {r["id"]: r["split"] for r in index_rows}
    split_counts: dict[str, int] = {}
    for split in entry_splits.values():
        split_counts[split] = split_counts.get(split, 0) + 1
    snapshot = dt.date.today().isoformat()
    # Local paths do not belong in a published manifest: keep the names only.
    config = {k: (v.name if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    config["heldout"] = {k: Path(v).name for k, v in cfg.heldout.items()}
    config["cutoff"] = cutoff
    from oplm.fold.data.mmcif import PARSE_VERSION

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "parse_version": PARSE_VERSION,
        "source": cfg.source,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "snapshot_date": snapshot,
        "archive": Path(cfg.archive).name,
        "ccd": {
            "path": ccd_path.name,
            "sha256": hashlib.sha256(ccd_path.read_bytes()).hexdigest(),
            "components": len(ccd),
        },
        "inputs": len(paths),
        "accepted": accepted,
        "rejected": rejected,
        "rejections": rejections,
        "dropped": dropped,
        "chains": len(index_rows),
        "splits": dict(sorted(split_counts.items())),
        "heldout_missing": heldout_missing,
        "row_group_size": cfg.row_group_size,
        "shards": writer.shards,
        "config": config,
        "licenses": {"wwPDB": "CC0 1.0", "CCD": "CC0 1.0"},
        "index_columns": _INDEX_SCHEMA.names,
        "note": "paths are basenames; the archive snapshot is identified by snapshot_date "
        "and checksums.txt",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    pretty = f"OPLM fold data: {cfg.source} {snapshot}"
    (out / "README.md").write_text(
        _CARD.format(
            pretty=pretty,
            pretty_json=json.dumps(pretty),
            snapshot=snapshot,
            accepted=accepted,
            chains=len(index_rows),
            ccd=ccd.source,
            splits=", ".join(f"{k} {v}" for k, v in manifest["splits"].items()) or "none",
        )
    )
    return manifest


def rows_to_index_table(index_rows: list[dict[str, Any]]) -> pa.Table:
    """The per-chain index as a pyarrow table with explicit nullable types."""
    columns = [pa.array([r[f.name] for r in index_rows], type=f.type) for f in _INDEX_SCHEMA]
    return pa.Table.from_arrays(columns, schema=_INDEX_SCHEMA)


# ----------------------------------------------------------------------------- runtime

_MASK32 = 0xFFFF_FFFF


def _example_seed(seed: int, epoch: int, shard: int, row: int) -> int:
    """Seed for one example's chain choice, crop and conformer rotations."""
    mixed = (
        (seed * 0x9E37_79B1) ^ (epoch * 0x85EB_CA77) ^ (shard * 0xC2B2_AE3D) ^ (row * 0x27D4_EB2F)
    )
    return (mixed + 0x1656_67B1) & _MASK32


class StructureDataset(ShardedProteinDataset):
    """Single-chain training examples from a preprocessed structure dataset directory.

    Subclasses :class:`ShardedProteinDataset` so rank/worker striping, per-epoch shuffling,
    :class:`InterleavedDataset` mixing and the data-exact resume contract are inherited. Only
    the row reader differs: this dataset strides over the *eligible* rows of each shard (entries
    with at least one chain in ``split`` that is at most ``max_chain_length`` long and has at
    least ``min_resolved_residues`` resolved residues), and every eligible row yields exactly one
    example, so the base class's skip arithmetic holds. Per row, a generator seeded from
    ``(seed, epoch, shard, row)`` picks one eligible chain uniformly, crops it and featurizes it
    with fresh conformer rotations; a resumed run therefore restores the same example.

    Rows are ``{"id": "<entry>_<chain>", "example": FoldExample}``; collate with
    :func:`oplm.fold.data.collate.collate_rows`.

    Args:
        path: Dataset directory written by ``oplm fold preprocess``.
        split: Index ``split`` value to train on; ``None`` means every chain.
        crop_size: Tokens per example (a multiple of 128 in training configs).
        contiguous_prob: Probability of a contiguous crop; otherwise spatial.
        max_chain_length: Chains longer than this are never sampled (addendum §3).
        min_resolved_residues: Chains with fewer resolved residues are never sampled.
        seed: Base seed (shared with the inherited shuffling).
        shuffle_shards: Shuffle shard order each epoch.
        shuffle_rows: Shuffle eligible-row order within a shard each epoch.
        conformers: Reference conformer table; the packaged one when ``None``.
        atom_pad_multiple: Per-example atom padding multiple (the collator re-pads to the budget).

    Raises:
        FileNotFoundError: ``path`` lacks ``shards/``, ``index.parquet`` or ``manifest.json``, or
            the index names a shard file that is not on disk.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        split: str | None = "train",
        crop_size: int = 384,
        contiguous_prob: float = 0.5,
        max_chain_length: int = 1024,
        min_resolved_residues: int = 4,
        seed: int = 0,
        shuffle_shards: bool = True,
        shuffle_rows: bool = True,
        conformers: ReferenceConformers | None = None,
        atom_pad_multiple: int = 32,
    ) -> None:
        root = Path(path)
        for required in ("shards", "index.parquet", "manifest.json"):
            if not (root / required).exists():
                raise FileNotFoundError(f"{root} has no {required}; run `oplm fold preprocess`")
        super().__init__(
            root / "shards", shuffle_shards=shuffle_shards, shuffle_rows=shuffle_rows, seed=seed
        )
        manifest = json.loads((root / "manifest.json").read_text())
        self._row_group_size = int(manifest["row_group_size"])
        self._crop_size = crop_size
        self._contiguous_prob = contiguous_prob
        self._conformers = conformers
        self._atom_pad_multiple = atom_pad_multiple

        index = pq.read_table(
            root / "index.parquet",
            columns=["shard", "row", "chain_index", "split", "length", "resolved_count"],
        ).to_pylist()
        shard_pos = {p.name: i for i, p in enumerate(self._shards)}
        missing = sorted({r["shard"] for r in index} - shard_pos.keys())
        if missing:
            raise FileNotFoundError(f"index names shards missing under {root}: {missing}")
        chains_by_shard: list[dict[int, list[int]]] = [{} for _ in self._shards]
        for r in index:
            if split is not None and r["split"] != split:
                continue
            if r["length"] > max_chain_length or r["resolved_count"] < min_resolved_residues:
                continue
            chains_by_shard[shard_pos[r["shard"]]].setdefault(int(r["row"]), []).append(
                int(r["chain_index"])
            )
        self._eligible_chains = chains_by_shard
        self._eligible_rows = [sorted(d) for d in chains_by_shard]
        # The base class strides over these counts; positions map to real rows in _read_rows.
        self._rows_per_shard = [len(rows) for rows in self._eligible_rows]
        self._total_rows = sum(self._rows_per_shard)

    def _read_rows(
        self, shard_idx: int, row_indices: list[int], epoch_seed: int
    ) -> Iterator[dict[str, object]]:
        """Yield one example per eligible-row position, reading one row group at a time.

        ponytail: random access with a one-group cache; small row groups (manifest
        ``row_group_size``, default 16) bound the read amplification. Group-blocked row
        orders would cut it further if I/O ever dominates.
        """
        real_rows = [self._eligible_rows[shard_idx][i] for i in row_indices]
        pf = pq.ParquetFile(self._shards[shard_idx])
        size = self._row_group_size
        cached: tuple[int, Any] | None = None
        table = self._conformers or ReferenceConformers.load()
        self._conformers = table
        for real in real_rows:
            group = real // size
            if cached is None or cached[0] != group:
                cached = (group, pf.read_row_group(group))
            row = cached[1].slice(real - group * size, 1).to_pylist()[0]
            complex = row_to_complex(row)
            generator = torch.Generator().manual_seed(
                _example_seed(self._seed, self._epoch, shard_idx, real)
            )
            chains = self._eligible_chains[shard_idx][real]
            chain = chains[int(torch.randint(0, len(chains), (1,), generator=generator))]
            cropped = crop_chain(
                complex,
                chain,
                self._crop_size,
                contiguous_prob=self._contiguous_prob,
                generator=generator,
            )
            example = featurize_complex(
                cropped,
                conformers=table,
                atom_pad_multiple=self._atom_pad_multiple,
                generator=generator,
            )
            yield {"id": example.id, "example": example}

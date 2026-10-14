"""Held-out targets for monomer evaluation (addendum §6): full chains and their ground truth.

A target set is a ``targets/<name>.json`` list written by ``oplm fold preprocess`` (``cameo``,
``casp14``) or, when no such list exists, a split of the index (``valid``). Lists are read from
the file rather than the split column because the first matching list wins the split: three
CASP14 entries sit in the ``cameo`` split (docs/fold/m2a/README.md). A list item with a chain
names an author chain. An item without one (every CASP14 item) contributes one chain per distinct
entity, the lowest chain index, so homo-oligomer copies are not scored as separate targets.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq

from oplm.fold.data.complex import row_to_complex

if TYPE_CHECKING:
    from collections.abc import Iterator

    from oplm.fold.data.complex import MolecularComplex

__all__ = ["EvalTarget", "TargetCoverage", "load_targets"]

_INDEX_COLUMNS = [
    "id",
    "chain_id",
    "auth_chain_id",
    "entity_id",
    "chain_index",
    "split",
    "length",
    "shard",
    "row",
]


@dataclass(frozen=True)
class EvalTarget:
    """One full chain to fold and score: its sequence and its reference atoms in fold() order."""

    target_set: str
    entry_id: str
    chain_id: str  # label_asym_id
    auth_chain_id: str
    sequence: str  # one letter per residue, X for unknown
    coords: np.ndarray  # (A, 3) float32 Å; zeros where unresolved
    resolved: np.ndarray  # (A,) bool
    atom_names: list[str]
    residue_index: np.ndarray  # (A,) int64: residue of each atom, 0-based
    ca_index: np.ndarray  # (L,) int64: flat atom index of each residue's CA

    @property
    def key(self) -> str:
        """``<entry>_<label chain>``, unique within a target set."""
        return f"{self.entry_id}_{self.chain_id}"


@dataclass
class TargetCoverage:
    """What a target set listed, what the snapshot lacks, and what was skipped and why."""

    target_set: str
    listed: int
    missing: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    selected: int = 0


def load_targets(
    root: str | Path,
    target_set: str,
    *,
    max_length: int | None = None,
    limit: int | None = None,
) -> tuple[list[EvalTarget], TargetCoverage]:
    """The chains of ``target_set`` in a preprocessed dataset, with their coverage record.

    Args:
        root: Dataset directory (``index.parquet``, ``manifest.json``, ``shards/``, ``targets/``).
        target_set: A ``targets/<name>.json`` list name, or else an index split name.
        max_length: Skip chains longer than this many residues (recorded in ``skipped``).
        limit: Keep only the first ``limit`` eligible chains in ``(entry, chain)`` order.

    Raises:
        ValueError: ``target_set`` is neither a list nor a split, or a chain's token count
            differs from its sequence length.
    """
    root = Path(root)
    index = pq.read_table(root / "index.parquet", columns=_INDEX_COLUMNS).to_pylist()
    list_path = root / "targets" / f"{target_set}.json"
    if list_path.exists():
        items = json.loads(list_path.read_text())
        rows, missing = _rows_for_list(index, items)
        coverage = TargetCoverage(target_set, listed=len(items), missing=missing)
    else:
        splits = sorted({str(r["split"]) for r in index})
        if target_set not in splits:
            raise ValueError(
                f"unknown target set {target_set!r}: no {list_path} and no such split "
                f"(splits: {', '.join(splits)})"
            )
        rows = [r for r in index if r["split"] == target_set]
        coverage = TargetCoverage(target_set, listed=len(rows))
    eligible: list[dict[str, Any]] = []
    for r in sorted(rows, key=lambda r: (r["id"], r["chain_id"])):
        if max_length is not None and int(r["length"]) > max_length:
            coverage.skipped[f"{r['id']}_{r['chain_id']}"] = (
                f"length {r['length']} > max_length {max_length}"
            )
        else:
            eligible.append(r)
    if limit is not None:
        eligible = eligible[:limit]
    targets: list[EvalTarget] = []
    for target in _read_targets(root, eligible, target_set):
        if int(target.resolved[target.ca_index].sum()) < 3:
            coverage.skipped[target.key] = "fewer than 3 resolved CA atoms"
        else:
            targets.append(target)
    targets.sort(key=lambda t: (t.entry_id, t.chain_id))
    coverage.selected = len(targets)
    return targets, coverage


def _rows_for_list(
    index: list[dict[str, Any]], items: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Index rows for the list items (author chain, or one chain per entity) and the misses."""
    by_entry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in index:
        by_entry[r["id"]].append(r)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: set[tuple[str, int]] = set()
    for item in items:
        entry, chain = item["id"], item["chain"]
        candidates = sorted(by_entry.get(entry, []), key=lambda r: r["chain_index"])
        if chain is not None:
            picked = [r for r in candidates if r["auth_chain_id"] == chain][:1]
        else:
            first: dict[int, dict[str, Any]] = {}
            for r in candidates:
                first.setdefault(int(r["entity_id"]), r)
            picked = list(first.values())
        if not picked:
            missing.append(entry if chain is None else f"{entry}_{chain}")
        for r in picked:
            if (r["id"], int(r["chain_index"])) not in seen:
                seen.add((r["id"], int(r["chain_index"])))
                rows.append(r)
    return rows, missing


def _read_targets(root: Path, rows: list[dict[str, Any]], target_set: str) -> Iterator[EvalTarget]:
    """Read each row's complex, one row group at a time, and cut out its chain."""
    size = int(json.loads((root / "manifest.json").read_text())["row_group_size"])
    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_shard[str(r["shard"])].append(r)
    for shard in sorted(by_shard):
        pf = pq.ParquetFile(root / "shards" / shard)
        cached: tuple[int, Any] | None = None
        for r in sorted(by_shard[shard], key=lambda r: int(r["row"])):
            row = int(r["row"])
            group = row // size
            if cached is None or cached[0] != group:
                cached = (group, pf.read_row_group(group))
            c = row_to_complex(cached[1].slice(row - group * size, 1).to_pylist()[0])
            yield _chain_target(c, int(r["chain_index"]), target_set)


def _chain_target(c: MolecularComplex, chain: int, target_set: str) -> EvalTarget:
    sub = c.select_tokens(c.chain_tokens(chain))
    info = c.chains[chain]
    if len(sub.token_res_type) != len(info.sequence):
        raise ValueError(
            f"{c.metadata.id}_{info.chain_id}: {len(sub.token_res_type)} tokens for a "
            f"{len(info.sequence)}-residue sequence"
        )
    return EvalTarget(
        target_set=target_set,
        entry_id=c.metadata.id,
        chain_id=info.chain_id,
        auth_chain_id=info.auth_chain_id,
        sequence=info.sequence,
        coords=sub.atom_coords.astype(np.float32),
        resolved=sub.atom_resolved.astype(bool),
        atom_names=list(sub.atom_name),
        residue_index=sub.atom_to_token.astype(np.int64),
        ca_index=sub.token_center_atom.astype(np.int64),
    )

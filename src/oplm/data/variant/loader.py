"""Variant-effect loading.

Loads zero-shot variant-effect assays (e.g. ProteinGym) into
:class:`VariantAssay`, parsing and validating mutations against the wild-type
sequence. Scoring itself lives in the eval harness; this module only loads and
validates (docs/DATA_TOOLING.md §6).

The variant modality is **eval-only** and uses no MLM masking: scoring is
position-specific (masked-marginal) and is the eval harness's job. This module's
contract ends at producing validated :class:`VariantAssay` objects.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pyarrow.csv as pa_csv

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pyarrow import Table

logger = logging.getLogger(__name__)

__all__ = [
    "Mutation",
    "VariantAssay",
    "load_variant_assays",
    "parse_mutation",
]

# Required/optional CSV columns (ProteinGym substitution-benchmark convention).
_MUTANT_COLUMN = "mutant"
_SCORE_COLUMN = "DMS_score"
_WILDTYPE_COLUMN = "wildtype"

# Multi-mutant strings join single substitutions with a colon, e.g. "A42T:G50S".
_MULTI_MUTANT_SEP = ":"

# A single substitution token: <wt-aa><1-based-pos><mut-aa>, e.g. "A42T".
_MUTATION_RE = re.compile(r"([A-Za-z])([0-9]+)([A-Za-z])")


class Mutation(NamedTuple):
    """A single parsed point substitution (1-based position).

    Attributes:
        wt: Wild-type amino acid (one-letter, upper-cased).
        pos: 1-based residue position in the wild-type sequence.
        mut: Mutant amino acid (one-letter, upper-cased).
    """

    wt: str
    pos: int
    mut: str


@dataclass
class VariantAssay:
    """A single zero-shot variant-effect assay (one CSV file).

    Attributes:
        name: Assay id (the CSV filename stem).
        wildtype: One-letter wild-type sequence the mutations are defined against.
        mutations: Raw ``mutant`` strings, one per CSV row (``:``-joined for
            multi-mutants). Kept verbatim; the eval harness parses them at
            scoring time, but every one is validated against ``wildtype`` here.
        labels: ``DMS_score`` per row, aligned to ``mutations``.
    """

    name: str
    wildtype: str
    mutations: list[str]
    labels: list[float]


def parse_mutation(token: str) -> Mutation:
    """Parse a single substitution token like ``"A42T"`` into a :class:`Mutation`.

    Args:
        token: One substitution in ``<wt><pos><mut>`` form (1-based position),
            e.g. ``"A42T"``. Surrounding whitespace is ignored; amino-acid codes
            are upper-cased.

    Returns:
        The parsed :class:`Mutation` (``wt``, ``pos``, ``mut``).

    Raises:
        ValueError: If ``token`` is not a well-formed single substitution.
    """
    match = _MUTATION_RE.fullmatch(token.strip())
    if match is None:
        raise ValueError(
            f"malformed mutation token {token!r}; expected '<wt><pos><mut>' like 'A42T'"
        )
    wt, pos_str, mut = match.groups()
    pos = int(pos_str)
    if pos < 1:
        raise ValueError(f"mutation position must be 1-based and >= 1, got {pos} in {token!r}")
    return Mutation(wt=wt.upper(), pos=pos, mut=mut.upper())


def _split_mutant(mutant: str) -> list[str]:
    """Split a (possibly multi-) mutant string on ``:`` into single tokens."""
    return mutant.split(_MULTI_MUTANT_SEP)


def _validate_mutation(wildtype: str, mutation: Mutation, *, assay: str, token: str) -> None:
    """Validate one parsed substitution against the wild-type sequence.

    Args:
        wildtype: One-letter wild-type sequence.
        mutation: The parsed substitution to check.
        assay: Assay name (for error messages).
        token: The raw ``mutant`` string the substitution came from (for errors).

    Raises:
        ValueError: If the position is out of range, or the wild-type amino acid
            at ``mutation.pos`` disagrees with ``mutation.wt``.
    """
    idx = mutation.pos - 1  # 1-based -> 0-based index into wildtype
    if not 0 <= idx < len(wildtype):
        raise ValueError(
            f"[{assay}] mutation {token!r} position {mutation.pos} is out of range for a "
            f"wild-type of length {len(wildtype)}"
        )
    actual = wildtype[idx]
    if actual != mutation.wt:
        raise ValueError(
            f"[{assay}] mutation {token!r} expects wild-type {mutation.wt!r} at position "
            f"{mutation.pos}, but the sequence has {actual!r}"
        )


def _resolve_wildtype(
    name: str,
    columns: list[str],
    table: Table,
    override: str | None,
) -> str:
    """Resolve the wild-type sequence for one assay.

    Resolution order (the documented accepted sources, §6.1):

    1. An explicit ``override`` (e.g. plumbed from
       ``EvalDatasetEntry.extra["wildtype"]`` / a sidecar) takes precedence.
    2. Otherwise a ``wildtype`` column in the CSV, which must hold a single
       constant sequence repeated across rows.

    Args:
        name: Assay name (for error messages).
        columns: The CSV's column names.
        table: The parsed pyarrow table (read for the ``wildtype`` column).
        override: Caller-supplied wild-type sequence, or ``None``.

    Returns:
        The resolved one-letter wild-type sequence (upper-cased).

    Raises:
        ValueError: If no source supplies a wild-type, or a ``wildtype`` column
            is present but not constant across rows.
    """
    if override is not None:
        return override.strip().upper()

    if _WILDTYPE_COLUMN in columns:
        values = {str(v).strip() for v in table.column(_WILDTYPE_COLUMN).to_pylist()}
        if len(values) != 1:
            raise ValueError(
                f"[{name}] {_WILDTYPE_COLUMN!r} column must hold a single constant sequence, "
                f"found {len(values)} distinct values"
            )
        return values.pop().upper()

    raise ValueError(
        f"[{name}] no wild-type sequence available: pass it via the `wildtypes` mapping "
        f"(e.g. EvalDatasetEntry.extra['wildtype']) or add a {_WILDTYPE_COLUMN!r} column"
    )


def _load_single_assay(path: Path, wildtype_override: str | None) -> VariantAssay:
    """Parse and validate one assay CSV into a :class:`VariantAssay`.

    Args:
        path: Path to the assay CSV.
        wildtype_override: Wild-type sequence supplied by the caller, or ``None``
            to resolve from a ``wildtype`` column.

    Returns:
        The parsed, WT-validated assay.

    Raises:
        ValueError: If required columns are missing, no wild-type is resolvable,
            or any mutation fails validation against the wild-type.
    """
    name = path.stem
    table = pa_csv.read_csv(str(path))
    columns = table.column_names

    missing = [c for c in (_MUTANT_COLUMN, _SCORE_COLUMN) if c not in columns]
    if missing:
        raise ValueError(f"[{name}] CSV missing required column(s) {missing}; found {columns}")

    mutations = [str(m) for m in table.column(_MUTANT_COLUMN).to_pylist()]
    labels = [float(s) for s in table.column(_SCORE_COLUMN).to_pylist()]
    wildtype = _resolve_wildtype(name, columns, table, wildtype_override)

    # Validate every (possibly multi-) mutant against the wild-type up front so
    # the eval harness can trust the loaded assay. Scoring is done downstream.
    for token in mutations:
        for mutation in (parse_mutation(part) for part in _split_mutant(token)):
            _validate_mutation(wildtype, mutation, assay=name, token=token)

    return VariantAssay(name=name, wildtype=wildtype, mutations=mutations, labels=labels)


def _discover_variant_files(directory: Path) -> list[Path]:
    """Return ``*.csv`` files in ``directory``, sorted by filename for determinism."""
    files: Iterable[Path] = (p for p in directory.iterdir() if p.suffix.lower() == ".csv")
    return sorted(files, key=lambda p: p.name)


def load_variant_assays(
    directory: str | Path,
    *,
    wildtypes: Mapping[str, str] | None = None,
    max_assays: int | None = None,
) -> list[VariantAssay]:
    """Load every variant-effect assay CSV in a directory.

    One :class:`VariantAssay` is produced per CSV. Each CSV must have a
    ``mutant`` column (a single substitution like ``"A42T"`` or ``:``-joined
    multi-mutants) and a ``DMS_score`` column (float). The wild-type sequence is
    resolved per assay from, in order of precedence:

    1. The ``wildtypes`` mapping keyed by assay name (the CSV filename stem) —
       the route for ``EvalDatasetEntry.extra["wildtype"]`` or a sidecar.
    2. A ``wildtype`` column in the CSV (a single constant sequence per file).

    Every mutation is parsed and validated against the wild-type at load time;
    a position out of range or a wild-type-residue mismatch raises ``ValueError``.

    Args:
        directory: Directory of assay CSV files.
        wildtypes: Optional mapping ``{assay_name: wildtype_sequence}`` (assay
            name = CSV filename stem). Takes precedence over a ``wildtype`` column.
        max_assays: Optional cap on the number of assays returned (applied after
            sorting by filename). ``None`` loads all.

    Returns:
        A list of validated :class:`VariantAssay`, ordered by filename.

    Raises:
        FileNotFoundError: If ``directory`` is not an existing directory.
        ValueError: If any assay is missing required columns, lacks a resolvable
            wild-type, or contains a mutation inconsistent with its wild-type.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"variant directory not found: {directory}")

    files = _discover_variant_files(directory)
    if max_assays is not None:
        files = files[:max_assays]

    overrides = wildtypes or {}
    assays = [_load_single_assay(p, overrides.get(p.stem)) for p in files]
    logger.info("loaded %d variant assays from %s", len(assays), directory)
    return assays

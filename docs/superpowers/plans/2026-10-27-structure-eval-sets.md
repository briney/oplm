# Structure Eval Sets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `oplm fold eval-sets`, which writes the CASP15+16, CAMEO and PDB40 contact-eval sets (with nested PDB40/CAMEO subsets), and switch the `structure` eval to the ESM convention: full sequence in, unresolved residues unscored.

**Architecture:** Two small changes to the existing eval come first. The loader places residues on SEQRES; the metric leaves pairs with missing coordinates out of the ranking and uses L = modeled residues. Then a new `oplm.fold.evalsets` package does the build:
- `chain.py`: the `EvalChain` type, PDB writer and curation rules.
- `casp.py`: CASP domain tarballs.
- `pdb.py`: CAMEO and PDB40 from the preprocessed fold dataset.
- `build.py`: orchestration, subsets, manifest, validation and README.
- A thin `oplm fold eval-sets` command in `fold/cli.py`.

**Tech Stack:** Python 3.11+, numpy, torch (the contact code), pyarrow, biopython (loader), MMseqs2 (external binary), typer, pytest.

**Spec:** `docs/superpowers/specs/2026-10-27-structure-eval-sets-design.md`

## Global Constraints

- Python 3.11+; `from __future__ import annotations` in every file; type hints on all signatures; Google-style docstrings on public APIs; ruff, line length 100.
- `pathlib.Path`, never `os.path`; no logic in `__init__.py`; no `# type: ignore` without an error code.
- Keep torch out of `src/oplm/fold/cli.py` module scope (import inside the command).
- No new Python dependencies. MMseqs2 is an external binary, called by path; its version is recorded.
- Tests mirror `src/` under `tests/`; prefer real data (crambin `1CRN.pdb`, the preprocessed `fixture_dataset`); mark slow tests `@pytest.mark.slow`.
- Tooling: `.venv/bin/python -m pytest` (no `-n`; xdist isn't installed), `.venv/bin/ruff check`, `.venv/bin/ruff format`, `VIRTUAL_ENV=$PWD/.venv .venv/bin/ty check src/`.
- Quality filter (CAMEO, PDB40): X-ray ≤ 2.5 Å or cryo-EM ≤ 3.0 Å; no NMR.
- Modeled residue = has N, CA and C. Floor: modeled ≥ 70% of the sequence (CAMEO, PDB40; not CASP).
- Contact floor: at least L/5 long-range contacts, with L = modeled residues; a contact is virtual Cβ < 8 Å with |i − j| ≥ 24 by sequence position.
- Lengths (full sequence): CASP target 50–1024, CAMEO 50–512, PDB40 50–1024; subsets draw from chains ≤ 512.
- Subsets: seed 42; `cameo_256`; `pdb40_256` ⊂ `pdb40_2048`.
- PDB40 clustering: distinct protein sequences of length ≥ 30, `mmseqs easy-cluster --min-seq-id 0.4 -c 0.8 --cov-mode 0 -s 7.5`.
- Exported files: one chain, PDB format, chain id `A`, SEQRES = full sequence, ATOM records only, residues numbered by sequence position (1-based).
- File stems: `casp<edition>_<target>-D<n>` (CASP) and `<entry>_<label chain id>` (CAMEO, PDB40; label ids are upper-case, so names can't collide on case-insensitive filesystems, and they match the fold head's `EvalTarget.key`).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Real PDB files numbered by author convention (offset numbering, insertion codes) must still load as before. Pinned by Task 1's `test_author_numbered_seqres_file_falls_back` (real `1VII.pdb`, numbered from 41).
2. A structure in which no residue has a full backbone must be skipped before any model forward, not crash or score 0/0. Pinned by Task 2's `test_structure_without_modeled_residues_is_skipped`.
3. CASP domain files that carry water/ligand HETATM records or alternate locations must still place cleanly. Pinned by Task 4's `test_hetatm_and_altloc_lines_are_ignored`.
4. A missing or failing `mmseqs` binary must raise a clear `RuntimeError` naming the binary, not a bare traceback. Pinned by Task 5's `test_run_mmseqs_reports_a_missing_binary`.
5. Output names must be unique even on case-insensitive filesystems. Pinned by Task 6's end-to-end test asserting lower-cased names are unique per set.

---

### Task 1: Loader places residues on SEQRES

**Files:**
- Modify: `src/oplm/data/structure/loader.py`
- Test: `tests/data/structure/test_loader.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `load_structures(directory)` / `_parse_single_structure(path)` return, for a PDB file whose SEQRES numbering fits, `StructureData(sequence=<full SEQRES sequence>, coords=(L, 3, 3) with NaN rows for unmodeled residues)`. Otherwise they keep today's behavior. Later tasks rely on this to round-trip exported files.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/structure/test_loader.py`:

```python
# --------------------------------------------------------------------------- #
# SEQRES placement (ESM convention: full sequence in, unmodeled residues NaN)
# --------------------------------------------------------------------------- #

_GAP = range(20, 25)  # crambin residues 20-24 (1-based)


def _crambin_variant(
    tmp_path: Path, drop: range = range(0), *, renumber: int = 0, keep_seqres: bool = True
) -> Path:
    """Crambin with residues ``drop`` removed, optionally renumbered or without SEQRES."""
    kept: list[str] = []
    for line in _require_crambin().read_text().splitlines(keepends=True):
        if line.startswith("SEQRES") and not keep_seqres:
            continue
        if line.startswith(("ATOM", "ANISOU", "TER")) and line[22:26].strip():
            resseq = int(line[22:26])
            if resseq in drop:
                continue
            line = f"{line[:22]}{resseq + renumber:>4}{line[26:]}"
        kept.append(line)
    path = tmp_path / "crambin_variant.pdb"
    path.write_text("".join(kept))
    return path


def test_seqres_keeps_unmodeled_residues_as_nan_rows(tmp_path: Path) -> None:
    """Residues missing from the coordinates stay in the sequence as all-NaN rows."""
    parsed = _parse_single_structure(_crambin_variant(tmp_path, _GAP))
    assert parsed is not None
    assert parsed.sequence == _CRAMBIN_SEQUENCE
    assert parsed.coords.shape == (46, 3, 3)
    gap = torch.tensor([i - 1 for i in _GAP])
    assert torch.isnan(parsed.coords[gap]).all()
    keep = torch.ones(46, dtype=torch.bool)
    keep[gap] = False
    assert torch.isfinite(parsed.coords[keep]).all()


def test_seqres_numbering_offset_falls_back_to_modeled_residues(tmp_path: Path) -> None:
    """Numbering that doesn't fit SEQRES keeps the coordinates-only sequence (old behavior)."""
    parsed = _parse_single_structure(_crambin_variant(tmp_path, _GAP, renumber=100))
    assert parsed is not None
    expected = "".join(aa for i, aa in enumerate(_CRAMBIN_SEQUENCE, 1) if i not in _GAP)
    assert parsed.sequence == expected
    assert torch.isfinite(parsed.coords).all()


def test_file_without_seqres_keeps_modeled_residues_only(tmp_path: Path) -> None:
    """No SEQRES: today's behavior, unmodeled residues are absent."""
    parsed = _parse_single_structure(_crambin_variant(tmp_path, _GAP, keep_seqres=False))
    assert parsed is not None
    assert len(parsed.sequence) == 41


def test_author_numbered_seqres_file_falls_back() -> None:
    """Real villin headpiece (1VII) is numbered from 41 against a 36-residue SEQRES."""
    path = Path(__file__).resolve().parents[2] / "fixtures" / "eval" / "structures" / "1VII.pdb"
    if not path.is_file():
        pytest.skip(f"structure fixture missing: {path}")
    parsed = _parse_single_structure(path)
    assert parsed is not None
    assert len(parsed.sequence) == 36
    assert torch.isfinite(parsed.coords).all()
```

- [ ] **Step 2: Run them to verify the first one fails**

Run: `.venv/bin/python -m pytest tests/data/structure/test_loader.py -q`
Expected: `test_seqres_keeps_unmodeled_residues_as_nan_rows` FAILS (the sequence is 41 residues, not 46). The other three new tests pass already; they pin today's behavior so the change can't break it.

- [ ] **Step 3: Implement SEQRES placement**

In `src/oplm/data/structure/loader.py`:

Replace the `TYPE_CHECKING` block with:

```python
if TYPE_CHECKING:
    from collections.abc import Iterable

    from Bio.PDB.Chain import Chain
```

Add after `_residue_to_one_letter`:

```python
def _read_seqres(path: Path) -> dict[str, list[str]]:
    """SEQRES residue names per chain id from a PDB-format file; empty when it has none.

    Reading stops at the first coordinate record, since SEQRES belongs to the header.
    """
    seqres: dict[str, list[str]] = {}
    with path.open() as handle:
        for line in handle:
            if line.startswith("SEQRES"):
                seqres.setdefault(line[11], []).extend(line[19:].split())
            elif line.startswith(("ATOM", "HETATM", "MODEL")):
                break
    return seqres


def _place_on_seqres(chain: Chain, seqres: list[str]) -> tuple[str, Tensor] | None:
    """Coordinates on the full SEQRES sequence, or ``None`` when the numbering doesn't fit.

    Residue ``resSeq`` goes to index ``resSeq - 1``; indices with no residue (unmodeled) stay
    ``NaN``. Placement fits only if every kept residue lands inside the sequence, once, with
    no insertion code, and with the one-letter code SEQRES has at that index. Files numbered
    by author convention (offsets, insertion codes) don't fit; the caller then falls back to
    the coordinates-only sequence.
    """
    sequence = "".join(_residue_to_one_letter(name) for name in seqres)
    coords = torch.full((len(sequence), len(_BACKBONE_ATOMS), 3), float("nan"))
    placed: set[int] = set()
    for residue in chain.get_residues():
        hetflag, resseq, icode = residue.id
        resname = residue.resname.strip()
        if hetflag != " " and resname not in _MODIFIED_RESIDUE_MAP:
            continue
        index = resseq - 1
        if (
            icode.strip()
            or not 0 <= index < len(sequence)
            or index in placed
            or _residue_to_one_letter(resname) != sequence[index]
        ):
            return None
        placed.add(index)
        for k, atom in enumerate(_BACKBONE_ATOMS):
            if atom in residue:
                coords[index, k] = torch.tensor(residue[atom].get_coord().tolist())
    return sequence, coords
```

In `_parse_single_structure`, insert directly after the `if chain is None: ... return None` block:

```python
    if path.suffix.lower() not in _MMCIF_SUFFIXES:
        seqres = _read_seqres(path).get(chain.id)
        placed = _place_on_seqres(chain, seqres) if seqres else None
        if placed is not None:
            sequence, coords = placed
            return StructureData(
                name=path.stem, sequence=sequence, coords=coords, chain_id=chain.id
            )
        if seqres:
            logger.debug("%s: numbering doesn't fit SEQRES; using modeled residues only", path)
```

Docstrings:
- `_parse_single_structure`: append this paragraph: "A PDB file whose residue numbers fit its SEQRES (see :func:`_place_on_seqres`) yields the full SEQRES sequence with NaN rows for unmodeled residues, so the structure eval sees the deposited sequence and leaves those residues unscored. Other files yield the residues that have coordinates."
- Module docstring: after the first paragraph, add: "PDB files whose residue numbers fit their SEQRES records load on the full sequence, with unmodeled residues as NaN rows (the ESM-C convention; ``oplm fold eval-sets`` writes files this way)."

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/structure/test_loader.py tests/eval/test_structure_task.py -q`
Expected: PASS (all loader tests, including the existing crambin tests, which now go through the placement path, plus the slow end-to-end structure test).

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/ruff format src/oplm/data/structure/loader.py tests/data/structure/test_loader.py && .venv/bin/ruff check src/oplm/data tests/data/structure && VIRTUAL_ENV=$PWD/.venv .venv/bin/ty check src/`
Expected: no errors.

```bash
git add src/oplm/data/structure/loader.py tests/data/structure/test_loader.py
git commit -m "feat(data): place structure residues on SEQRES, unmodeled residues as NaN rows

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Contact metric leaves unknown pairs unscored; L = modeled residues

**Files:**
- Modify: `src/oplm/eval/metrics/contact.py` (add `contact_positions_valid`, delete `compute_precision_at_l`)
- Modify: `src/oplm/eval/metrics/categorical_jacobian.py:198-240`
- Modify: `src/oplm/eval/tasks/structure.py` (imports, `evaluate`, `_process_single_structure`)
- Modify: `docs/EVAL_HARNESS.md` (structure section), `docs/OVERVIEW.md` (contact primitives bullet)
- Test: `tests/eval/test_contact.py` (new), `tests/eval/test_categorical_jacobian.py`, `tests/eval/test_structure_task.py`

**Interfaces:**
- Consumes: Task 1's NaN rows for unmodeled residues.
- Produces:
  - `contact_positions_valid(coords: Tensor, use_cbeta: bool = True) -> Tensor`: `(L,)` bool, used by Tasks 3 and 6.
  - `build_structure_pair_score_data(pred_contacts, true_contacts, seq_len, min_seq_sep=24, valid: Tensor | None = None)`.
  - `compute_mean_pair_score_precision_at_l(structures, l_divisor=1) -> float`, which no longer takes `min_seq_sep`.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_contact.py`:

```python
"""Contact primitives: which residues can be scored."""

from __future__ import annotations

import torch

from oplm.eval.metrics.contact import contact_positions_valid


def test_contact_positions_valid_flags_any_missing_backbone_atom() -> None:
    """Virtual Cβ needs N, CA and C; CA-only scoring needs just CA."""
    coords = torch.randn(5, 3, 3)
    coords[1, 0] = float("nan")  # N missing
    coords[3, 2] = float("nan")  # C missing
    assert contact_positions_valid(coords).tolist() == [True, False, True, False, True]
    assert contact_positions_valid(coords, use_cbeta=False).tolist() == [True] * 5
```

In `tests/eval/test_categorical_jacobian.py`, change the existing call in `test_mean_pair_score_precision_is_one_for_perfect_ranking` to:

```python
    precision = compute_mean_pair_score_precision_at_l([data], l_divisor=1)
```

and append:

```python
def _bands(seq_len: int) -> torch.Tensor:
    """Ground truth with contacts on the 6th, 7th and 8th off-diagonals."""
    true_contacts = torch.zeros(seq_len, seq_len)
    for offset in (6, 7, 8):
        idx = torch.arange(seq_len - offset)
        true_contacts[idx, idx + offset] = 1.0
    return true_contacts


def test_pair_data_drops_pairs_touching_unmodeled_residues() -> None:
    """With a validity mask, no kept pair involves an invalid residue."""
    seq_len, min_seq_sep = 30, 6
    valid = torch.ones(seq_len, dtype=torch.bool)
    valid[[3, 17]] = False
    data = build_structure_pair_score_data(
        torch.rand(seq_len, seq_len), _bands(seq_len), 28, min_seq_sep, valid=valid
    )
    i, j = data.pair_indices
    bad = torch.tensor([3, 17])
    assert not torch.isin(i, bad).any() and not torch.isin(j, bad).any()
    expected = sum(
        1
        for a in range(seq_len)
        for b in range(a + min_seq_sep, seq_len)
        if a not in (3, 17) and b not in (3, 17)
    )
    assert len(data.scores) == expected
    assert data.seq_len == 28


def test_unknown_pairs_are_not_false_positives() -> None:
    """The top-scoring pair touches an unmodeled residue: masked, P@L stays 1.0."""
    seq_len, min_seq_sep = 24, 6
    true_contacts = _bands(seq_len)
    pred = true_contacts.clone()
    pred[0, 23] = 10.0  # highest score, not a contact, residue 0 unmodeled
    valid = torch.ones(seq_len, dtype=torch.bool)
    valid[0] = False
    masked = build_structure_pair_score_data(
        pred, true_contacts, seq_len - 1, min_seq_sep, valid=valid
    )
    unmasked = build_structure_pair_score_data(pred, true_contacts, seq_len, min_seq_sep)
    assert compute_mean_pair_score_precision_at_l([masked]) == pytest.approx(1.0)
    assert compute_mean_pair_score_precision_at_l([unmasked]) < 1.0
```

In `tests/eval/test_structure_task.py`, add the imports:

```python
import torch

from oplm.data import StructureData, get_tokenizer, load_structures
from oplm.eval.metrics.categorical_jacobian import get_canonical_amino_acid_token_ids
```

and append:

```python
def _task() -> StructureEvalTask:
    """A structure task with its tokenizer state initialized, as ``evaluate`` would."""
    cfg = OplmConfig(
        model=OplmModelConfig(max_position_embeddings=_MAX_SEQ_LEN),
        train=TrainConfig(wandb_enabled=False),
        data=DataConfig(num_workers=0, pin_memory=False),
    )
    entry = EvalDatasetEntry(
        name="struct", path=".", type="structure", schedule=ScheduleSpec("steps", 1)
    )
    task = StructureEvalTask(entry, cfg)
    task._tokenizer = get_tokenizer()
    task._canonical_aa_token_ids = get_canonical_amino_acid_token_ids(task._tokenizer)
    return task


def test_structure_without_modeled_residues_is_skipped(
    make_model: Callable[..., OplmForMaskedLM],
) -> None:
    """No residue with a full backbone: skipped before any forward pass."""
    struct = StructureData(
        name="empty",
        sequence="ACDEFGHIK",
        coords=torch.full((9, 3, 3), float("nan")),
        chain_id="A",
    )
    model = make_model(max_position_embeddings=_MAX_SEQ_LEN)
    assert _task()._process_single_structure(struct, model, torch.device("cpu")) is None


@pytest.mark.slow
def test_unmodeled_residues_are_unscored(
    structure_fixtures_dir: Path,
    tmp_path: Path,
    make_model: Callable[..., OplmForMaskedLM],
) -> None:
    """Five unmodeled residues: the model reads all 46, L counts the 41 modeled."""
    lines = [
        line
        for line in (structure_fixtures_dir / "1CRN.pdb").read_text().splitlines(keepends=True)
        if not (line.startswith(("ATOM", "ANISOU")) and 20 <= int(line[22:26]) <= 24)
    ]
    (tmp_path / "1CRN_gap.pdb").write_text("".join(lines))
    (struct,) = load_structures(tmp_path)
    model = make_model(max_position_embeddings=_MAX_SEQ_LEN)
    data = _task()._process_single_structure(struct, model, torch.device("cpu"))
    assert data is not None
    assert len(struct.sequence) == 46 and data.seq_len == 41
    gap = torch.arange(19, 24)
    assert not torch.isin(data.pair_indices[0], gap).any()
    assert not torch.isin(data.pair_indices[1], gap).any()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_contact.py tests/eval/test_categorical_jacobian.py tests/eval/test_structure_task.py -q`
Expected: FAIL. `contact_positions_valid` can't be imported, `build_structure_pair_score_data` has no `valid` argument, and the skip test fails because the forward pass still runs.

- [ ] **Step 3: Implement**

`src/oplm/eval/metrics/contact.py`: add after `compute_contact_map`:

```python
def contact_positions_valid(coords: Tensor, use_cbeta: bool = True) -> Tensor:
    """Residues whose contact position (virtual Cβ, or CA) is finite, shape ``(L,)`` bool.

    A pair involving a residue without coordinates is unknown, not a non-contact: the
    structure eval leaves it out of the ranking (the ESM-C convention).

    Args:
        coords: Backbone coordinates ``(L, 3, 3)`` with ``[N, CA, C]`` ordering.
        use_cbeta: Score by virtual Cβ (needs N, CA and C) rather than CA alone.
    """
    positions = compute_virtual_cbeta(coords) if use_cbeta else coords[:, 1, :]
    return torch.isfinite(positions).all(dim=-1)
```

Delete `compute_precision_at_l` (its only caller is replaced below), and change the module docstring's first sentence to: "Contact-map geometry and APC utilities." In the docstring's second sentence, replace "Average Product Correction, and precision@L scoring." with "Average Product Correction, and which residues can be scored."

`src/oplm/eval/metrics/categorical_jacobian.py`: change the import to `from oplm.eval.metrics.contact import apply_apc`. In `StructurePairScoreData`'s docstring, change `seq_len` to "L for precision@L: the number of residues with coordinates." Replace both functions at the end of the file with:

```python
def build_structure_pair_score_data(
    pred_contacts: Tensor,
    true_contacts: Tensor,
    seq_len: int,
    min_seq_sep: int = 24,
    valid: Tensor | None = None,
) -> StructurePairScoreData:
    """Flatten a contact score matrix into the scorable long-range residue pairs.

    Args:
        pred_contacts: Predicted contact scores ``(L, L)``.
        true_contacts: Binary ground-truth contacts ``(L, L)``.
        seq_len: L for precision@L (the number of residues with coordinates).
        min_seq_sep: Minimum ``|i - j|`` of a long-range pair.
        valid: Optional ``(L,)`` bool mask of residues with coordinates. Pairs touching an
            invalid residue are dropped: unknown, not non-contacts.
    """
    n = true_contacts.shape[0]
    i_indices, j_indices = torch.triu_indices(n, n, offset=min_seq_sep)
    if valid is not None:
        keep = valid[i_indices] & valid[j_indices]
        i_indices, j_indices = i_indices[keep], j_indices[keep]
    return StructurePairScoreData(
        scores=pred_contacts[i_indices, j_indices],
        labels=true_contacts[i_indices, j_indices],
        seq_len=seq_len,
        pair_indices=(i_indices, j_indices),
    )


def compute_mean_pair_score_precision_at_l(
    structures: list[StructurePairScoreData],
    l_divisor: int = 1,
) -> float:
    """Mean precision@(L/l_divisor) over structures: the fraction of each structure's top
    ``seq_len // l_divisor`` scorable pairs that are true contacts (0.0 with no pairs)."""
    precisions: list[float] = []
    for structure in structures:
        n_pairs = structure.scores.numel()
        if n_pairs == 0:
            precisions.append(0.0)
            continue
        k = min(max(1, structure.seq_len // l_divisor), n_pairs)
        top = torch.topk(structure.scores, k).indices
        precisions.append(structure.labels[top].float().mean().item())
    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)
```

`src/oplm/eval/tasks/structure.py`: change the contact import to `from oplm.eval.metrics.contact import compute_contact_map, contact_positions_valid`. In `evaluate`, change the metric call to:

```python
            results[metric_name] = compute_mean_pair_score_precision_at_l(
                all_pair_score_data,
                l_divisor=divisor,
            )
```

In `_process_single_structure`, insert right after the eligibility `return None` block:

```python
        valid = contact_positions_valid(struct.coords, use_cbeta=self.tcfg.use_cbeta)
        modeled = int(valid.sum())
        if modeled == 0:
            logger.debug("Skipping %s: no residue has backbone coordinates", struct.name)
            return None
```

and change the `build_structure_pair_score_data` call to:

```python
        pair_score_data = build_structure_pair_score_data(
            jacobian_contacts,
            true_contacts,
            modeled,
            self.tcfg.min_seq_sep,
            valid=valid,
        )
```

Docs:
- In `docs/EVAL_HARNESS.md` (both phrases below span line breaks in the file), step 4: replace "and compute precision over the top `L / l_divisor` predicted pairs." with "drop pairs that involve a residue without coordinates (unknown, not non-contacts), and compute precision over the top `L / l_divisor` predicted pairs, where L is the number of residues with coordinates."
- In the paragraph after it, replace "NaN backbone coordinates are treated as infinite distance (no contact); with no valid pairs" with "The model reads the full sequence. For a PDB file whose numbering fits its SEQRES, that includes unmodeled residues, which load as NaN rows (the ESM-C convention). Pairs involving a residue with a NaN backbone atom are left out of the ranking. A structure with no residue having a full backbone is skipped; with no valid pairs".
- In `docs/OVERVIEW.md`'s contact primitives bullet, replace "**precision@L** — rank long-range pairs by score, take top `L/l_divisor`, report fraction that are true contacts." with "**scorable residues** — `contact_positions_valid` marks residues whose virtual Cβ is finite. Precision@L (in `categorical_jacobian.py`) ranks only long-range pairs between such residues, takes the top `L/l_divisor` with L = their count, and reports the fraction that are true contacts."

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval tests/data/structure -q`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/ruff format src/oplm/eval tests/eval/test_contact.py tests/eval/test_categorical_jacobian.py tests/eval/test_structure_task.py && .venv/bin/ruff check src/oplm/eval tests/eval && VIRTUAL_ENV=$PWD/.venv .venv/bin/ty check src/`
Expected: no errors.

```bash
git add src/oplm/eval tests/eval docs/EVAL_HARNESS.md docs/OVERVIEW.md
git commit -m "feat(eval): leave pairs with unmodeled residues unscored; L = modeled residues

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `EvalChain`, the PDB writer and the curation rules

**Files:**
- Modify: `src/oplm/fold/eval/targets.py` (rename `_read_targets` to `read_targets`, export it)
- Create: `src/oplm/fold/evalsets/__init__.py`, `src/oplm/fold/evalsets/chain.py`
- Test: `tests/fold/evalsets/__init__.py` (empty), `tests/fold/evalsets/test_chain.py`

**Interfaces:**
- Consumes:
  - Task 1's SEQRES placement, for the round-trip.
  - Task 2's `contact_positions_valid(coords, use_cbeta=True) -> Tensor`.
  - `oplm.fold.eval.targets.load_targets(root, target_set) -> (list[EvalTarget], TargetCoverage)`.
  - `oplm.fold.data.ccd.PROTEIN_1TO3: dict[str, str]`.
- Produces:
  - `EvalChain(name: str, sequence: str, atoms: list[dict[str, np.ndarray]], meta: dict[str, Any])`, with `.modeled -> int` and `.backbone() -> np.ndarray (L, 3, 3)`.
  - `STANDARD_AMINO_ACIDS: frozenset[str]`.
  - `from_eval_target(target: EvalTarget, *, meta: dict[str, Any]) -> EvalChain`, named `target.key`.
  - `count_long_range_contacts(backbone: np.ndarray, *, threshold: float = 8.0, min_sep: int = 24) -> int`.
  - `rejection(chain, *, min_length: int, max_length: int, min_modeled_fraction: float) -> str | None`, with reasons `"nonstandard" | "length" | "modeled" | "contacts"`.
  - `write_pdb(chain: EvalChain, path: Path) -> None`.
  - `read_targets(root: Path, rows: list[dict[str, Any]], target_set: str) -> Iterator[EvalTarget]`, public in `oplm.fold.eval.targets`.

- [ ] **Step 1: Write the failing tests**

Create `tests/fold/evalsets/__init__.py` (empty) and `tests/fold/evalsets/test_chain.py`:

```python
"""EvalChain: fold-dataset chains to eval PDB files, and the shared curation rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from oplm.data import load_structures
from oplm.eval.metrics.contact import contact_positions_valid
from oplm.fold.eval.targets import load_targets
from oplm.fold.evalsets.chain import (
    EvalChain,
    count_long_range_contacts,
    from_eval_target,
    rejection,
    write_pdb,
)

if TYPE_CHECKING:
    from pathlib import Path

_RULES = {"min_length": 20, "max_length": 1024, "min_modeled_fraction": 0.7}


def _chain(dataset: Path, split: str, key: str) -> EvalChain:
    targets, _ = load_targets(dataset, split)
    (target,) = [t for t in targets if t.key == key]
    return from_eval_target(target, meta={})


def test_from_eval_target_keeps_unmodeled_residues(fixture_dataset_early_cutoff: Path) -> None:
    """2ds7_A: 51 residues in the sequence, 11 without atoms, 40 modeled."""
    chain = _chain(fixture_dataset_early_cutoff, "valid", "2ds7_A")
    assert chain.name == "2ds7_A" and len(chain.sequence) == 51
    assert chain.modeled == 40
    assert sum(not residue for residue in chain.atoms) == 11
    assert np.isnan(chain.backbone()).any(axis=(1, 2)).sum() == 11


def test_write_pdb_round_trips_through_the_loader(
    fixture_dataset_early_cutoff: Path, tmp_path: Path
) -> None:
    """The written file loads as the full sequence with NaN rows exactly where unmodeled."""
    chain = _chain(fixture_dataset_early_cutoff, "valid", "2ds7_A")
    write_pdb(chain, tmp_path / f"{chain.name}.pdb")
    (loaded,) = load_structures(tmp_path)
    assert loaded.name == "2ds7_A" and loaded.sequence == chain.sequence
    built = chain.backbone()
    assert np.array_equal(np.isnan(loaded.coords.numpy()), np.isnan(built))
    assert np.allclose(loaded.coords.numpy(), built, atol=1e-3, equal_nan=True)
    assert contact_positions_valid(loaded.coords).sum().item() == chain.modeled
    assert count_long_range_contacts(loaded.coords.numpy()) == count_long_range_contacts(built)


def test_count_long_range_contacts(fixture_dataset: Path) -> None:
    """A lone helix has none; a globin has well over a hundred."""
    assert count_long_range_contacts(_chain(fixture_dataset, "casp14", "1jb6_A").backbone()) == 0
    assert count_long_range_contacts(_chain(fixture_dataset, "train", "4hhb_A").backbone()) > 100


@pytest.mark.parametrize(
    ("split", "key", "overrides", "reason"),
    [
        ("train", "4hhb_A", {}, None),
        ("cameo", "1crn_A", {"min_length": 50}, "length"),
        ("casp14", "1jb6_A", {}, "contacts"),
    ],
)
def test_rejection_reasons(
    fixture_dataset: Path, split: str, key: str, overrides: dict, reason: str | None
) -> None:
    assert rejection(_chain(fixture_dataset, split, key), **(_RULES | overrides)) == reason


def test_rejection_modeled_fraction(fixture_dataset_early_cutoff: Path) -> None:
    """2ds7_A is 40/51 = 78% modeled: fails a 0.8 floor."""
    chain = _chain(fixture_dataset_early_cutoff, "valid", "2ds7_A")
    assert rejection(chain, **(_RULES | {"min_modeled_fraction": 0.8})) == "modeled"


def test_nonstandard_residues_are_rejected_and_not_written(tmp_path: Path) -> None:
    chain = EvalChain(name="x", sequence="AXA", atoms=[{}, {}, {}])
    assert rejection(chain, **_RULES) == "nonstandard"
    with pytest.raises(ValueError, match="non-standard"):
        write_pdb(chain, tmp_path / "x.pdb")


def test_atoms_must_match_the_sequence() -> None:
    with pytest.raises(ValueError, match="atoms for 1 residues"):
        EvalChain(name="x", sequence="AA", atoms=[{}])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/evalsets/test_chain.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.fold.evalsets'`.

- [ ] **Step 3: Implement**

`src/oplm/fold/eval/targets.py`: rename `_read_targets` to `read_targets` (definition and the call in `load_targets`), add `"read_targets"` to `__all__`, and give it this docstring: "Read each row's complex, one row group at a time, and cut out its chain. ``rows`` are index rows (``id``, ``chain_id``, ``chain_index``, ``shard``, ``row``); targets come out in shard order, not input order."

`src/oplm/fold/evalsets/__init__.py`:

```python
"""Structure eval sets built from CASP, CAMEO and the PDB (``oplm fold eval-sets``)."""

from __future__ import annotations
```

`src/oplm/fold/evalsets/chain.py`:

```python
"""One eval structure: full sequence, modeled atoms, PDB output, and the curation rules.

The ``structure`` eval reads these files with :func:`oplm.data.load_structures`. SEQRES carries
the full sequence and residues are numbered by sequence position, so unmodeled residues load as
NaN rows and are left out of scoring
(docs/superpowers/specs/2026-10-27-structure-eval-sets-design.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from oplm.eval.metrics.contact import compute_contact_map, contact_positions_valid
from oplm.fold.data.ccd import PROTEIN_1TO3

if TYPE_CHECKING:
    from pathlib import Path

    from oplm.fold.eval.targets import EvalTarget

__all__ = [
    "STANDARD_AMINO_ACIDS",
    "EvalChain",
    "count_long_range_contacts",
    "from_eval_target",
    "rejection",
    "write_pdb",
]

STANDARD_AMINO_ACIDS = frozenset(PROTEIN_1TO3) - {"X"}
_BACKBONE = ("N", "CA", "C")
_SEQRES_PER_LINE = 13


@dataclass(frozen=True)
class EvalChain:
    """A curated chain: what the model reads (``sequence``) and what is scored (``atoms``).

    Attributes:
        name: File stem, e.g. ``casp16_T1201-D1`` or ``4hhb_A`` (entry and label chain id).
        sequence: Full input sequence, one letter per residue.
        atoms: Per residue, modeled atom name -> ``(3,)`` float32 coordinates; ``{}`` when the
            residue is unmodeled.
        meta: Manifest fields (``source``, ``entry``, ``chain``, ``method``, ...).
    """

    name: str
    sequence: str
    atoms: list[dict[str, np.ndarray]]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.atoms) != len(self.sequence):
            raise ValueError(
                f"{self.name}: atoms for {len(self.atoms)} residues, "
                f"sequence of {len(self.sequence)}"
            )

    @property
    def modeled(self) -> int:
        """Residues with N, CA and C: the L of precision@L."""
        return sum(all(atom in residue for atom in _BACKBONE) for residue in self.atoms)

    def backbone(self) -> np.ndarray:
        """``(L, 3, 3)`` N/CA/C coordinates, NaN where an atom is unmodeled."""
        out = np.full((len(self.sequence), len(_BACKBONE), 3), np.nan, dtype=np.float32)
        for i, residue in enumerate(self.atoms):
            for k, atom in enumerate(_BACKBONE):
                if atom in residue:
                    out[i, k] = residue[atom]
        return out


def from_eval_target(target: EvalTarget, *, meta: dict[str, Any]) -> EvalChain:
    """An :class:`EvalChain` named ``target.key`` from a fold-dataset chain: every residue of
    the entity sequence, with the atoms that are resolved."""
    atoms: list[dict[str, np.ndarray]] = [{} for _ in target.sequence]
    for k, atom in enumerate(target.atom_names):
        if target.resolved[k]:
            atoms[int(target.residue_index[k])][atom] = target.coords[k].astype(np.float32)
    return EvalChain(name=target.key, sequence=target.sequence, atoms=atoms, meta=meta)


def count_long_range_contacts(
    backbone: np.ndarray, *, threshold: float = 8.0, min_sep: int = 24
) -> int:
    """Contacts (virtual Cβ closer than ``threshold`` Å) between modeled residues at least
    ``min_sep`` apart in sequence, counted with the ``structure`` eval's own contact code."""
    coords = torch.as_tensor(backbone, dtype=torch.float32)
    contacts = compute_contact_map(coords, threshold=threshold)
    valid = contact_positions_valid(coords)
    i, j = torch.triu_indices(len(coords), len(coords), offset=min_sep)
    keep = valid[i] & valid[j]
    return int(contacts[i[keep], j[keep]].sum().item())


def rejection(
    chain: EvalChain, *, min_length: int, max_length: int, min_modeled_fraction: float
) -> str | None:
    """Why ``chain`` fails the shared curation rules, or ``None`` when it passes.

    Checked in order: only standard amino acids (``"nonstandard"``); sequence length within
    ``[min_length, max_length]`` (``"length"``); at least one modeled residue and a modeled
    fraction of at least ``min_modeled_fraction`` (``"modeled"``); at least ``modeled / 5``
    long-range contacts (``"contacts"``).
    """
    if set(chain.sequence) - STANDARD_AMINO_ACIDS:
        return "nonstandard"
    if not min_length <= len(chain.sequence) <= max_length:
        return "length"
    modeled = chain.modeled
    if modeled == 0 or modeled < min_modeled_fraction * len(chain.sequence):
        return "modeled"
    if count_long_range_contacts(chain.backbone()) < modeled / 5:
        return "contacts"
    return None


def write_pdb(chain: EvalChain, path: Path) -> None:
    """Write ``chain`` as a one-chain PDB file (chain id ``A``).

    SEQRES holds the full sequence; ATOM records hold the modeled atoms, numbered by sequence
    position (1-based), with occupancy 1 and B-factor 0.

    Raises:
        ValueError: The sequence has a residue outside the twenty standard amino acids.
    """
    unknown = sorted(set(chain.sequence) - STANDARD_AMINO_ACIDS)
    if unknown:
        raise ValueError(f"{chain.name}: non-standard residues {unknown}")
    names = [PROTEIN_1TO3[aa] for aa in chain.sequence]
    lines = [
        f"SEQRES {n // _SEQRES_PER_LINE + 1:>3} A {len(names):>4}  "
        + " ".join(names[n : n + _SEQRES_PER_LINE])
        for n in range(0, len(names), _SEQRES_PER_LINE)
    ]
    serial = 0
    for i, residue in enumerate(chain.atoms):
        for atom, (x, y, z) in residue.items():
            serial += 1
            label = atom if len(atom) == 4 else f" {atom:<3}"
            lines.append(
                f"ATOM  {serial:>5} {label}{names[i]:>4} A{i + 1:>4}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{0.0:>6.2f}          {atom[0]:>2}"
            )
    lines += ["TER", "END"]
    path.write_text("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/fold/evalsets/test_chain.py tests/fold/eval/test_targets.py -q`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/ruff format src/oplm/fold/evalsets src/oplm/fold/eval/targets.py tests/fold/evalsets && .venv/bin/ruff check src/oplm/fold tests/fold/evalsets && VIRTUAL_ENV=$PWD/.venv .venv/bin/ty check src/`
Expected: no errors.

```bash
git add src/oplm/fold/eval/targets.py src/oplm/fold/evalsets tests/fold/evalsets
git commit -m "feat(fold): EvalChain, eval PDB writer and the shared curation rules

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: CASP domain targets

**Files:**
- Create: `src/oplm/fold/evalsets/casp.py`
- Create: `tests/fold/evalsets/conftest.py`
- Test: `tests/fold/evalsets/test_casp.py`

**Interfaces:**
- Consumes: Task 3's `EvalChain`, `STANDARD_AMINO_ACIDS`; `PROTEIN_1TO3`.
- Produces:
  - `CASP_EDITIONS: dict[str, tuple[str, str]]` (edition → tarball name, sequence FASTA name).
  - `CASP_SHA256: dict[str, str]`.
  - `sha256sum(path: Path) -> str`.
  - `verify_casp_sources(directory: Path) -> list[Path]`.
  - `read_fasta(path: Path) -> dict[str, str]`.
  - `canonical_target(stem: str) -> str`.
  - `casp_domain_chains(tarball: Path, sequences: dict[str, str], edition: str) -> tuple[list[EvalChain], dict[str, int]]`.
  - Test helpers in `tests/fold/evalsets/conftest.py`: `CRAMBIN`, `GAP`, `crambin_domain(drop=frozenset(), renumber=0) -> bytes`, `make_tarball(path, files) -> Path`, and fixture `casp_sources -> Path`.

- [ ] **Step 1: Write the shared test helpers and the failing tests**

Create `tests/fold/evalsets/conftest.py`:

```python
"""Shared fixtures for the eval-set builder tests: CASP sources built from real crambin."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

CRAMBIN_PDB = Path(__file__).parents[2] / "data" / "fixtures" / "structures" / "1CRN.pdb"
CRAMBIN = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"
GAP = frozenset(range(20, 25))  # residues 20-24 (1-based) left unmodeled


def crambin_domain(drop: frozenset[int] = frozenset(), renumber: int = 0) -> bytes:
    """Crambin's ATOM records as a CASP domain file, minus residues ``drop`` (1-based)."""
    out: list[str] = []
    for line in CRAMBIN_PDB.read_text().splitlines():
        if line.startswith("ATOM"):
            resseq = int(line[22:26])
            if resseq not in drop:
                out.append(f"{line[:22]}{resseq + renumber:>4}{line[26:]}")
    return ("\n".join(out) + "\nEND\n").encode()


def make_tarball(path: Path, files: dict[str, bytes]) -> Path:
    """A gzipped tarball holding ``files`` (member name -> bytes)."""
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture
def casp_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The four CASP source files, made from crambin, with their hashes pinned for the test.

    CASP15 holds one full domain; CASP16 holds a gapped domain and its phase-2 copy.
    """
    from oplm.fold.evalsets import casp

    directory = tmp_path / "casp"
    directory.mkdir()
    (tar15, seq15), (tar16, seq16) = casp.CASP_EDITIONS["15"], casp.CASP_EDITIONS["16"]
    make_tarball(directory / tar15, {"./T1104-D1.pdb": crambin_domain()})
    (directory / seq15).write_text(f">T1104 crambin\n{CRAMBIN}\n")
    gapped = crambin_domain(GAP)
    make_tarball(directory / tar16, {"T0201-D1.pdb": gapped, "T2201-D1.pdb": gapped})
    (directory / seq16).write_text(f">T1201 crambin, 46 residues\n{CRAMBIN}\n")
    pinned = {path.name: casp.sha256sum(path) for path in directory.iterdir()}
    monkeypatch.setattr(casp, "CASP_SHA256", pinned)
    return directory
```

Create `tests/fold/evalsets/test_casp.py`:

```python
"""CASP domain tarballs -> eval chains on the full target sequence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from oplm.fold.evalsets import casp
from tests.fold.evalsets.conftest import CRAMBIN, GAP, crambin_domain, make_tarball

if TYPE_CHECKING:
    from pathlib import Path


def test_domains_sit_on_the_full_target_sequence(tmp_path: Path) -> None:
    """A gapped domain keeps all 46 residues; the five unmodeled ones have no atoms."""
    tarball = make_tarball(tmp_path / "d.tgz", {"T0201-D1.pdb": crambin_domain(GAP)})
    (chain,), stats = casp.casp_domain_chains(tarball, {"T1201": CRAMBIN}, "16")
    assert chain.name == "casp16_T1201-D1" and chain.sequence == CRAMBIN
    assert chain.modeled == 41
    assert [i for i, residue in enumerate(chain.atoms, 1) if not residue] == sorted(GAP)
    assert chain.meta == {
        "source": "casp16",
        "entry": "T1201",
        "chain": "D1",
        "source_file": "T0201-D1.pdb",
    }
    assert stats["files"] == 1


def test_duplicates_versions_combined_units_and_mismatches_are_skipped(tmp_path: Path) -> None:
    domain = crambin_domain()
    files = {
        "T0201-D1.pdb": domain,
        "T2201-D1.pdb": domain,  # phase-2 copy of T1201-D1
        "T1228v1-D1.pdb": domain,
        "T1228v2-D1.pdb": domain,  # later version of the same domain
        "T1300-D12.pdb": domain,  # combined unit
        "T1301-D1.pdb": domain,  # no sequence
        "T1302-D1.pdb": crambin_domain(renumber=100),  # numbering outside the sequence
    }
    sequences = {name: CRAMBIN for name in ("T1201", "T1228v1", "T1300", "T1302")}
    chains, stats = casp.casp_domain_chains(
        make_tarball(tmp_path / "d.tgz", files), sequences, "16"
    )
    assert [c.name for c in chains] == ["casp16_T1201-D1", "casp16_T1228v1-D1"]
    assert stats == {
        "files": 7,
        "multi_domain": 1,
        "duplicate": 2,
        "no_sequence": 1,
        "numbering_mismatch": 1,
    }


def test_hetatm_and_altloc_lines_are_ignored(tmp_path: Path) -> None:
    """Waters and a second alternate location don't break placement or add atoms."""
    lines = crambin_domain().decode().splitlines()
    first_ca = next(line for line in lines if line[12:16] == " CA ")
    altloc_b = f"{first_ca[:16]}B{first_ca[17:30]}{99.0:>8.3f}{first_ca[38:]}"
    water = "HETATM  999  O   HOH A 101      10.000  10.000  10.000  1.00 20.00           O"
    data = "\n".join([*lines[:-1], altloc_b, water, "END"]).encode()
    tarball = make_tarball(tmp_path / "d.tgz", {"T1104-D1.pdb": data})
    (chain,), _ = casp.casp_domain_chains(tarball, {"T1104": CRAMBIN}, "15")
    assert chain.modeled == 46
    assert chain.atoms[0]["CA"][0] != pytest.approx(99.0)


def test_member_paths_are_flattened(tmp_path: Path) -> None:
    """CASP15 members are stored as ``./T1104-D1.pdb``."""
    tarball = make_tarball(tmp_path / "d.tgz", {"./T1104-D1.pdb": crambin_domain()})
    (chain,), _ = casp.casp_domain_chains(tarball, {"T1104": CRAMBIN}, "15")
    assert chain.name == "casp15_T1104-D1"


def test_canonical_target() -> None:
    assert casp.canonical_target("T0206") == "T1206"
    assert casp.canonical_target("T2201") == "T1201"
    assert casp.canonical_target("T1228v2") == "T1228v2"
    assert casp.canonical_target("T1104") == "T1104"


def test_read_fasta(tmp_path: Path) -> None:
    path = tmp_path / "s.fasta"
    path.write_text(">T1201 Q9GZX9, Human, 8 residues\nACDE\nFGHI\n>T1202\nKL\n")
    assert casp.read_fasta(path) == {"T1201": "ACDEFGHI", "T1202": "KL"}


def test_verify_casp_sources(casp_sources: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert len(casp.verify_casp_sources(casp_sources)) == 4
    name = next(iter(casp.CASP_SHA256))
    monkeypatch.setitem(casp.CASP_SHA256, name, "0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        casp.verify_casp_sources(casp_sources)
    (casp_sources / name).unlink()
    with pytest.raises(FileNotFoundError, match="missing CASP source"):
        casp.verify_casp_sources(casp_sources)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/evalsets/test_casp.py -q`
Expected: FAIL with `ImportError: cannot import name 'casp' from 'oplm.fold.evalsets'`.

- [ ] **Step 3: Implement**

Create `src/oplm/fold/evalsets/casp.py`:

```python
"""CASP domain targets: the assessed, domain-trimmed coordinates on the full target sequence.

Sources are the Prediction Center's domain tarballs and target sequence files, pinned by sha256
(docs/superpowers/specs/2026-10-27-structure-eval-sets-design.md, "CASP").
"""

from __future__ import annotations

import hashlib
import re
import tarfile
from pathlib import Path

import numpy as np

from oplm.fold.data.ccd import PROTEIN_1TO3
from oplm.fold.evalsets.chain import EvalChain

__all__ = [
    "CASP_EDITIONS",
    "CASP_SHA256",
    "canonical_target",
    "casp_domain_chains",
    "read_fasta",
    "sha256sum",
    "verify_casp_sources",
]

# edition -> (domain tarball, target sequences); both under predictioncenter.org/download_area/
# CASP<edition>/targets/ and CASP<edition>/sequences/.
CASP_EDITIONS: dict[str, tuple[str, str]] = {
    "15": ("casp15.targets.TS-domains.public_12.20.2022.tar.gz", "casp15.seq.txt"),
    "16": ("casp16.targets_monomer_trimmed2domains.tgz", "casp16.T1.seq.txt"),
}
CASP_SHA256: dict[str, str] = {
    "casp15.targets.TS-domains.public_12.20.2022.tar.gz": (
        "ac159c284a1506b460143ef6e22832272c5da1b4078fdb5fec8292b373b6239b"
    ),
    "casp15.seq.txt": "84dd04c1fda2a550af73e77ad25be94da23f16bd7ff6632f2c22c1978b9ee0a5",
    "casp16.targets_monomer_trimmed2domains.tgz": (
        "30e6bf992baed08695ebebe481711adea867b483015d30e18d170a0eee66ceb9"
    ),
    "casp16.T1.seq.txt": "77d0b5c4ffda52ca1be7563a92eabae7ca59a3aab814e79a377ddd962783e286",
}

_DOMAIN_FILE = re.compile(r"^(?P<target>T\d{4}[^-]*)-D(?P<domain>\d+)\.pdb$")
_ONE_LETTER = {three: one for one, three in PROTEIN_1TO3.items()} | {"MSE": "M"}


def sha256sum(path: Path) -> str:
    """Hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_casp_sources(directory: Path) -> list[Path]:
    """The CASP source files in ``directory``, each checked against its pinned sha256.

    Raises:
        FileNotFoundError: A pinned file is missing.
        ValueError: A file's sha256 differs from the pinned value.
    """
    paths: list[Path] = []
    for name, expected in CASP_SHA256.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"missing CASP source {path}")
        actual = sha256sum(path)
        if actual != expected:
            raise ValueError(f"{path}: sha256 {actual} != pinned {expected}")
        paths.append(path)
    return paths


def read_fasta(path: Path) -> dict[str, str]:
    """``{id: sequence}`` from a FASTA file; the id is the header's first word."""
    sequences: dict[str, str] = {}
    name: str | None = None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            sequences[name] = ""
        elif name is not None:
            sequences[name] += line.strip()
    return sequences


def canonical_target(stem: str) -> str:
    """The ``T1xxx`` id behind a CASP16 file stem. Monomer files name target ``T1xxx`` as
    ``T0xxx`` and phase-2 copies as ``T2xxx``; other stems are returned unchanged."""
    return re.sub(r"^T[02]", "T1", stem)


def _domain_atoms(text: str, sequence: str) -> list[dict[str, np.ndarray]] | None:
    """Heavy atoms per target residue from a domain file, or ``None`` when a residue falls
    outside ``sequence`` or disagrees with it. Water and ligand HETATM records, hydrogens, and
    alternate locations other than the first are ignored."""
    atoms: list[dict[str, np.ndarray]] = [{} for _ in sequence]
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or line[16] not in " A":
            continue
        resname = line[17:20].strip()
        if resname not in _ONE_LETTER:
            if line.startswith("HETATM"):
                continue
            return None
        name = line[12:16].strip()
        if (line[76:78].strip() or name[0]) in ("H", "D"):
            continue
        index = int(line[22:26]) - 1
        if (
            line[26].strip()
            or not 0 <= index < len(sequence)
            or _ONE_LETTER[resname] != sequence[index]
        ):
            return None
        if resname == "MSE" and name == "SE":
            name = "SD"
        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        atoms[index].setdefault(name, np.array(xyz, dtype=np.float32))
    return atoms


def casp_domain_chains(
    tarball: Path, sequences: dict[str, str], edition: str
) -> tuple[list[EvalChain], dict[str, int]]:
    """One :class:`EvalChain` per distinct single-domain unit in a CASP domain tarball.

    A chain's sequence is the full target sequence; only the domain's residues have atoms,
    placed by target numbering (``resSeq - 1``). Files are skipped and counted under
    ``multi_domain`` (combined units such as ``D12``), ``duplicate`` (a ``T0``/``T2`` copy of a
    domain already taken, or a later ``vN`` version), ``no_sequence`` and
    ``numbering_mismatch``; ``files`` counts every domain file seen.
    """
    stats = dict.fromkeys(
        ("files", "multi_domain", "duplicate", "no_sequence", "numbering_mismatch"), 0
    )
    groups: dict[tuple[str, str], list[tuple[str, str, bytes]]] = {}
    with tarfile.open(tarball) as archive:
        for member in archive.getmembers():
            filename = Path(member.name).name
            match = _DOMAIN_FILE.match(filename)
            handle = archive.extractfile(member) if member.isfile() and match else None
            if match is None or handle is None:
                continue
            stats["files"] += 1
            if len(match["domain"]) > 1:
                stats["multi_domain"] += 1
                continue
            target = canonical_target(match["target"])
            key = (re.sub(r"v\d+$", "", target), match["domain"])
            groups.setdefault(key, []).append((target, filename, handle.read()))
    chains: list[EvalChain] = []
    for (base, domain), candidates in sorted(groups.items()):
        stats["duplicate"] += len(candidates) - 1
        target, filename, data = min(candidates)
        sequence = sequences.get(target) or sequences.get(base)
        if sequence is None:
            stats["no_sequence"] += 1
            continue
        atoms = _domain_atoms(data.decode(), sequence)
        if atoms is None:
            stats["numbering_mismatch"] += 1
            continue
        meta = {
            "source": f"casp{edition}",
            "entry": target,
            "chain": f"D{domain}",
            "source_file": filename,
        }
        chains.append(
            EvalChain(
                name=f"casp{edition}_{target}-D{domain}",
                sequence=sequence,
                atoms=atoms,
                meta=meta,
            )
        )
    return chains, stats
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/fold/evalsets -q`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/ruff format src/oplm/fold/evalsets tests/fold/evalsets && .venv/bin/ruff check src/oplm/fold/evalsets tests/fold/evalsets && VIRTUAL_ENV=$PWD/.venv .venv/bin/ty check src/`
Expected: no errors.

```bash
git add src/oplm/fold/evalsets/casp.py tests/fold/evalsets
git commit -m "feat(fold): CASP15/16 domain targets on the full target sequence

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: CAMEO and PDB40 from the fold dataset

**Files:**
- Create: `src/oplm/fold/evalsets/pdb.py`
- Modify: `tests/fold/evalsets/conftest.py` (add the `fake_mmseqs` fixture)
- Test: `tests/fold/evalsets/test_pdb.py`

**Interfaces:**
- Consumes:
  - Task 3's `EvalChain`, `STANDARD_AMINO_ACIDS`, `from_eval_target`, `rejection`.
  - `oplm.fold.eval.targets.load_targets` and `read_targets(root, rows, target_set)`.
- Produces:
  - `QUALITY_LIMITS: dict[str, float]` and `passes_quality(method, resolution) -> bool`.
  - `cameo_chains(root: Path, target_set: str = "cameo") -> tuple[list[EvalChain], dict[str, int]]`.
  - `write_cluster_fasta(root: Path, path: Path) -> dict[str, str]` (FASTA id → sequence).
  - `run_mmseqs(fasta: Path, workdir: Path, *, mmseqs: str = "mmseqs", threads: int = 1) -> Path` (cluster TSV).
  - `mmseqs_version(mmseqs: str = "mmseqs") -> str`.
  - `read_clusters(tsv: Path) -> dict[str, str]` (member id → representative id).
  - `pdb40_chains(root, sequence_cluster: dict[str, str], *, min_length, max_length, min_modeled_fraction, max_rounds=5) -> tuple[list[EvalChain], dict[str, int]]`.
  - Fixture `fake_mmseqs -> Path`.

- [ ] **Step 1: Add the fake MMseqs2 fixture and write the failing tests**

Append to `tests/fold/evalsets/conftest.py`. Add `import sys` to the imports at the top of the file.

```python
_FAKE_MMSEQS = '''
import sys
from pathlib import Path

args = sys.argv[1:]
if args[0] == "version":
    print("fake-mmseqs")
    raise SystemExit(0)
assert args[0] == "easy-cluster" and "--min-seq-id" in args, args
fasta, prefix = Path(args[1]), args[2]
lines = fasta.read_text().splitlines()
reps = {}
rows = []
for header, sequence in zip(lines[::2], lines[1::2]):
    member = header[1:]
    rep = reps.setdefault(sequence[:10], member)  # one cluster per ten-residue prefix
    rows.append(f"{rep}\\t{member}\\n")
Path(f"{prefix}_cluster.tsv").write_text("".join(rows))
'''


@pytest.fixture(scope="session")
def fake_mmseqs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An ``mmseqs`` stand-in: ``version``, and ``easy-cluster`` by ten-residue prefix (so the
    two crambins, 1crn and 1ejg, share a cluster)."""
    path = tmp_path_factory.mktemp("bin") / "mmseqs"
    path.write_text(f"#!{sys.executable}{_FAKE_MMSEQS}")
    path.chmod(0o755)
    return path
```

Create `tests/fold/evalsets/test_pdb.py`:

```python
"""CAMEO and PDB40 chains from the preprocessed fixture dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

from oplm.fold.evalsets import pdb

if TYPE_CHECKING:
    from pathlib import Path

_RULES = {"min_length": 20, "max_length": 1024, "min_modeled_fraction": 0.7}


def _sequence(dataset: Path, entry: str, chain: str) -> str:
    rows = pq.read_table(dataset / "index.parquet", columns=["id", "chain_id", "sequence"])
    (row,) = [r for r in rows.to_pylist() if (r["id"], r["chain_id"]) == (entry, chain)]
    return row["sequence"]


def _clusters(dataset: Path) -> dict[str, str]:
    """Sequence -> cluster: the crambins together, everything else alone."""
    groups = {
        "crambin": [("1crn", "A"), ("1ejg", "A")],
        "zipper": [("1jb6", "A")],
        "2ds7": [("2ds7", "A")],
        "alpha": [("4hhb", "A")],
        "beta": [("4hhb", "B")],
    }
    return {
        _sequence(dataset, entry, chain): cluster
        for cluster, members in groups.items()
        for entry, chain in members
    }


@pytest.mark.parametrize(
    ("method", "resolution", "expected"),
    [
        ("X-RAY DIFFRACTION", 2.5, True),
        ("X-RAY DIFFRACTION", 2.51, False),
        ("ELECTRON MICROSCOPY", 3.0, True),
        ("ELECTRON MICROSCOPY", 3.2, False),
        ("SOLUTION NMR", None, False),
        ("X-RAY DIFFRACTION", None, False),
    ],
)
def test_passes_quality(method: str, resolution: float | None, expected: bool) -> None:
    assert pdb.passes_quality(method, resolution) is expected


def test_cameo_chains(fixture_dataset: Path) -> None:
    chains, stats = pdb.cameo_chains(fixture_dataset)
    assert [c.name for c in chains] == ["1crn_A"]
    assert stats == {"listed": 1, "missing": 0, "skipped": 0, "quality": 0}
    assert chains[0].meta["method"] == "X-RAY DIFFRACTION"
    assert chains[0].meta["resolution"] == pytest.approx(1.5)
    assert chains[0].meta["chain"] == "A"


def test_write_cluster_fasta(fixture_dataset: Path, tmp_path: Path) -> None:
    """Six distinct sequences of length >= 30 (trp-cage is too short), sorted, stable ids."""
    ids = pdb.write_cluster_fasta(fixture_dataset, tmp_path / "pdb40.fasta")
    assert list(ids) == [f"s{i:06d}" for i in range(6)]
    assert list(ids.values()) == sorted(ids.values())
    assert min(len(s) for s in ids.values()) >= pdb.CLUSTER_MIN_LENGTH
    text = (tmp_path / "pdb40.fasta").read_text()
    assert text.startswith(">s000000\n") and text.count(">") == 6


def test_read_clusters(tmp_path: Path) -> None:
    tsv = tmp_path / "c.tsv"
    tsv.write_text("s0\ts0\ns0\ts2\ns1\ts1\n")
    assert pdb.read_clusters(tsv) == {"s0": "s0", "s2": "s0", "s1": "s1"}


def test_pdb40_takes_the_best_passing_member(fixture_dataset: Path) -> None:
    """1ejg (0.54 Å) beats 1crn (1.5 Å); the zipper and 2ds7 clusters fail on contacts."""
    chains, stats = pdb.pdb40_chains(fixture_dataset, _clusters(fixture_dataset), **_RULES)
    assert [c.name for c in chains] == ["1ejg_A", "4hhb_A", "4hhb_B"]
    assert stats == {
        "clusters": 5,
        "no_candidate": 0,
        "contacts": 3,
        "exhausted": 2,
        "max_rounds": 0,
    }
    assert chains[0].meta["cluster"] == "crambin"


def test_pdb40_stops_after_max_rounds(fixture_dataset: Path) -> None:
    """With one round, the zipper's second chain is never tried."""
    _, stats = pdb.pdb40_chains(
        fixture_dataset, _clusters(fixture_dataset), **_RULES, max_rounds=1
    )
    assert stats["max_rounds"] == 1 and stats["exhausted"] == 1 and stats["contacts"] == 2


def test_run_mmseqs_with_a_stand_in(fake_mmseqs: Path, tmp_path: Path) -> None:
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">s0\nTTCCPSIVARSNF\n>s1\nTTCCPSIVARQQQ\n>s2\nVLSPADKTNVKAA\n")
    tsv = pdb.run_mmseqs(fasta, tmp_path, mmseqs=str(fake_mmseqs))
    assert pdb.read_clusters(tsv) == {"s0": "s0", "s1": "s0", "s2": "s2"}
    assert pdb.mmseqs_version(str(fake_mmseqs)) == "fake-mmseqs"


def test_run_mmseqs_reports_a_missing_binary(tmp_path: Path) -> None:
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">s0\nACDE\n")
    with pytest.raises(RuntimeError, match="no-such-mmseqs"):
        pdb.run_mmseqs(fasta, tmp_path, mmseqs=str(tmp_path / "no-such-mmseqs"))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/evalsets/test_pdb.py -q`
Expected: FAIL with `ImportError: cannot import name 'pdb' from 'oplm.fold.evalsets'`.

- [ ] **Step 3: Implement**

Create `src/oplm/fold/evalsets/pdb.py`:

```python
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
```


- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/fold/evalsets -q`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/ruff format src/oplm/fold/evalsets tests/fold/evalsets && .venv/bin/ruff check src/oplm/fold/evalsets tests/fold/evalsets && VIRTUAL_ENV=$PWD/.venv .venv/bin/ty check src/`
Expected: no errors.

```bash
git add src/oplm/fold/evalsets/pdb.py tests/fold/evalsets
git commit -m "feat(fold): CAMEO and PDB40 eval chains from the fold dataset

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The build, `oplm fold eval-sets`, and docs

**Files:**
- Create: `src/oplm/fold/evalsets/build.py`
- Modify: `src/oplm/fold/cli.py` (module docstring and the new command)
- Modify: `docs/FOLD.md` (package layout, new §11), `AGENTS.md` (fold line in the project tree)
- Test: `tests/fold/evalsets/test_build.py`

**Interfaces:**
- Consumes: everything from Tasks 3–5; `oplm.data.load_structures`; Task 2's `contact_positions_valid`.
- Produces:
  - `BuildConfig` (fields below).
  - `build(cfg: BuildConfig) -> dict[str, Any]`, a summary with keys `built`, `oplm`, `mmseqs`, `config`, `sources`, `sets`, `overlap`, `exported`.
  - `nested_subsets(names: list[str], sizes: Sequence[int], seed: int) -> dict[int, list[str]]`.
  - `render_readme(summary) -> str`.
  - The CLI command `oplm fold eval-sets --fold-data --casp-sources --out [--mmseqs --threads --seed]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/fold/evalsets/test_build.py`:

```python
"""End-to-end build of the eval sets from the fixture dataset and crambin CASP sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest
import torch
from typer.testing import CliRunner

from oplm.data import load_structures
from oplm.fold.cli import app
from oplm.fold.evalsets.build import BuildConfig, build, nested_subsets

if TYPE_CHECKING:
    from pathlib import Path

_SMALL = {
    "casp_length": (20, 1024),
    "cameo_length": (20, 512),
    "pdb40_length": (20, 1024),
    "cameo_subsets": (1,),
    "pdb40_subsets": (1, 2),
}


def test_build_end_to_end(
    fixture_dataset: Path, casp_sources: Path, fake_mmseqs: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    cfg = BuildConfig(
        fold_data=fixture_dataset,
        casp_sources=casp_sources,
        out=out,
        mmseqs=str(fake_mmseqs),
        **_SMALL,
    )
    summary = build(cfg)

    assert summary["exported"] == {
        "casp15_16": 2,
        "cameo": 1,
        "pdb40": 3,
        "cameo_1": 1,
        "pdb40_1": 1,
        "pdb40_2": 2,
    }
    names = {d: sorted(p.stem for p in (out / d).glob("*.pdb")) for d in summary["exported"]}
    assert names["casp15_16"] == ["casp15_T1104-D1", "casp16_T1201-D1"]
    assert names["pdb40"] == ["1ejg_A", "4hhb_A", "4hhb_B"]
    assert set(names["pdb40_1"]) < set(names["pdb40_2"]) <= set(names["pdb40"])
    for files in names.values():
        assert len({n.lower() for n in files}) == len(files)
    assert summary["sets"]["casp15_16"]["casp16"]["duplicate"] == 1
    assert summary["overlap"] == {"cameo_chains_in_pdb40": 0, "cameo_clusters_in_pdb40": 1}

    (gapped,) = [s for s in load_structures(out / "casp15_16") if s.name == "casp16_T1201-D1"]
    assert len(gapped.sequence) == 46
    assert torch.isnan(gapped.coords[19:24]).all()

    manifest = pq.read_table(out / "manifest.parquet").to_pylist()
    assert len(manifest) == sum(summary["exported"].values())
    row = next(r for r in manifest if r["set"] == "casp15_16" and r["name"] == "casp16_T1201-D1")
    assert row["modeled"] == 41 and row["length"] == 46 and len(row["sha256"]) == 64
    assert row["source_file"] == "T0201-D1.pdb"

    sources = {p.name for p in (out / "sources").iterdir()}
    assert {"cameo.json", "pdb40.fasta", "pdb40_cluster.tsv"} <= sources
    assert {p.name for p in casp_sources.iterdir()} <= sources
    readme = (out / "README.md").read_text()
    assert "casp15_16" in readme and "fake-mmseqs" in readme


def test_build_refuses_a_non_empty_output(
    fixture_dataset: Path, casp_sources: Path, fake_mmseqs: Path, tmp_path: Path
) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "stale.pdb").write_text("")
    cfg = BuildConfig(
        fold_data=fixture_dataset,
        casp_sources=casp_sources,
        out=tmp_path / "out",
        mmseqs=str(fake_mmseqs),
    )
    with pytest.raises(FileExistsError, match="not empty"):
        build(cfg)


def test_nested_subsets_are_nested_and_deterministic() -> None:
    names = [f"c{i}" for i in range(10)]
    subsets = nested_subsets(names, (2, 5, 50), seed=42)
    assert len(subsets[2]) == 2 and len(subsets[5]) == 5 and subsets[50] == sorted(names)
    assert set(subsets[2]) < set(subsets[5])
    assert nested_subsets(list(reversed(names)), (2, 5, 50), seed=42) == subsets


def test_cli_eval_sets(
    fixture_dataset: Path, casp_sources: Path, fake_mmseqs: Path, tmp_path: Path
) -> None:
    """Spec defaults: the 46-residue crambins fall below 50, so only the globins remain."""
    out = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        [
            "eval-sets",
            "--fold-data",
            str(fixture_dataset),
            "--casp-sources",
            str(casp_sources),
            "--out",
            str(out),
            "--mmseqs",
            str(fake_mmseqs),
            "--threads",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert sorted(p.stem for p in (out / "pdb40").glob("*.pdb")) == ["4hhb_A", "4hhb_B"]
    assert "pdb40_2048" in result.output
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/evalsets/test_build.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.fold.evalsets.build'`.

- [ ] **Step 3: Implement the build**

Create `src/oplm/fold/evalsets/build.py`:

```python
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
from pathlib import Path
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
                or not np.array_equal(contact_positions_valid(structure.coords).numpy(), built_valid)
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
```

Two notes on the test's overlap value. `cameo_clusters_in_pdb40 == 1` holds because 1crn (CAMEO) shares the crambin cluster with the PDB40 representative 1ejg. `cameo_chains_in_pdb40 == 0` holds because 1ejg, not 1crn, represents that cluster.

- [ ] **Step 4: Add the CLI command**

In `src/oplm/fold/cli.py`, change the module docstring's first line to:

```python
"""``oplm fold`` command group: ``bench-kernels``, ``compare``, ``eval``, ``eval-sets``,
``make-fixtures``, ``measure``, ``predict`` (design §5.2), ``preprocess``, and ``train``.
```

and add after `preprocess_cmd`:

```python
@app.command("eval-sets")
def eval_sets_cmd(
    fold_data: Annotated[
        Path, typer.Option("--fold-data", help="Preprocessed fold dataset (oplm fold preprocess)")
    ],
    casp_sources: Annotated[
        Path,
        typer.Option("--casp-sources", help="Directory with the pinned CASP15/16 source files"),
    ],
    out: Annotated[Path, typer.Option("--out", help="Output directory (absent or empty)")],
    mmseqs: Annotated[str, typer.Option(help="MMseqs2 executable")] = "mmseqs",
    threads: Annotated[int, typer.Option(help="MMseqs2 threads")] = max(
        1, (os.cpu_count() or 2) - 1
    ),
    seed: Annotated[int, typer.Option(help="Seed for the nested subsets")] = 42,
) -> None:
    """Build the structure eval sets: CASP15+16 domains, CAMEO, PDB40 and its subsets."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from oplm.fold.evalsets.build import BuildConfig, build

    summary = build(
        BuildConfig(
            fold_data=fold_data,
            casp_sources=casp_sources,
            out=out,
            mmseqs=mmseqs,
            threads=threads,
            seed=seed,
        )
    )
    table = Table(title=f"eval-sets -> {out}")
    table.add_column("set")
    table.add_column("structures", justify="right")
    for name, count in summary["exported"].items():
        table.add_row(name, str(count))
    console.print(table)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/fold/evalsets tests/fold/test_cli.py -q`
Expected: PASS.

- [ ] **Step 6: Docs**

In `docs/FOLD.md` §1, add `eval-sets` to the `cli.py` line (`bench-kernels | compare | eval | eval-sets | make-fixtures | ...`) and add this block after the `eval/` block, before `tests/fold/`:

```
└── evalsets/
    ├── chain.py            # EvalChain, eval PDB writer (SEQRES + sequence numbering), curation rules
    ├── casp.py             # CASP15/16 domain tarballs on the full target sequence (pinned sha256)
    ├── pdb.py              # CAMEO list and PDB40 (MMseqs2 40% clusters) from the fold dataset
    └── build.py            # oplm fold eval-sets: subsets, validation, manifest, README
```

(change the `eval/` block's leading `└──` to `├──`). Append a new section at the end of `docs/FOLD.md`:

````markdown
## 11. Structure eval sets (`oplm fold eval-sets`)

Builds the unsupervised contact-precision sets for the LM's `structure` eval. Design and
curation rules are in `docs/superpowers/specs/2026-10-27-structure-eval-sets-design.md`:
full sequence in, unresolved residues unscored; X-ray ≤ 2.5 Å or cryo-EM ≤ 3.0 Å; at least
70% modeled; at least L/5 long-range contacts.

```bash
# MMseqs2 (static build); the version is recorded in the output README
curl -L https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz | tar xz -C $OPLM_DATA/tmp
# CASP sources (sha256-pinned in oplm.fold.evalsets.casp)
mkdir -p $OPLM_DATA/casp && cd $OPLM_DATA/casp
curl -O https://predictioncenter.org/download_area/CASP15/targets/casp15.targets.TS-domains.public_12.20.2022.tar.gz
curl -O https://predictioncenter.org/download_area/CASP15/sequences/casp15.seq.txt
curl -O https://predictioncenter.org/download_area/CASP16/targets/casp16.targets_monomer_trimmed2domains.tgz
curl -O https://predictioncenter.org/download_area/CASP16/sequences/casp16.T1.seq.txt
oplm fold eval-sets --fold-data $OPLM_DATA/hf/wwpdb/2026-10-12 --casp-sources $OPLM_DATA/casp \
    --out $OPLM_DATA/hf/structure-evals/2026-10-12 --mmseqs $OPLM_DATA/tmp/mmseqs/bin/mmseqs
```

| directory | contents |
| --- | --- |
| `casp15_16/` | CASP15+16 single-domain units: full target sequence, coordinates only for the domain |
| `cameo/`, `cameo_256/` | the fold head's CAMEO list (2021-10 .. 2022-09), filtered; seed-42 subset |
| `pdb40/`, `pdb40_2048/`, `pdb40_256/` | one representative per MMseqs2 40% cluster; nested seed-42 subsets of chains ≤ 512 |
| `manifest.parquet`, `summary.json`, `README.md`, `sources/` | per-file metadata, counts by stage, sources and their sha256 |

Every file is one chain with SEQRES and residues numbered by sequence position, so
`load_structures` returns the full sequence with NaN rows for unmodeled residues, and the metric
leaves pairs that involve them out of the ranking. Names are `<entry>_<label chain>`, matching
`EvalTarget.key`, or `casp<edition>_<target>-D<n>`. The build validates every directory by
loading it back. In-training use:

```yaml
data:
  eval:
    pdb40:      {path: .../pdb40_256,  type: structure, every: {steps: 5_000}}
    pdb40_2048: {path: .../pdb40_2048, type: structure, every: {steps: 50_000}}
    cameo:      {path: .../cameo_256,  type: structure, every: {steps: 5_000}}
    casp:       {path: .../casp15_16,  type: structure, every: {steps: 5_000}}
```
````

In `AGENTS.md`, change the `fold/` line of the project tree to end with: `..., eval/ (held-out metrics, `fold_monomer` task), evalsets/ (`oplm fold eval-sets`: CASP/CAMEO/PDB40 contact-eval sets); parity tests need OPLM_FOLD_FIXTURES (docs/FOLD.md §7–§9)`.

- [ ] **Step 7: Full fast suite, lint, type-check, commit**

Run: `.venv/bin/python -m pytest -m "not slow" -q > /tmp/claude-fast.txt 2>&1; tail -3 /tmp/claude-fast.txt`
Expected: all pass (the earlier baseline was 2885 passed, 38 skipped; expect more passes now).

Run: `.venv/bin/ruff format src/oplm/fold tests/fold/evalsets && .venv/bin/ruff check src/ tests/fold/evalsets && VIRTUAL_ENV=$PWD/.venv .venv/bin/ty check src/`
Expected: no errors.

```bash
git add src/oplm/fold tests/fold/evalsets docs/FOLD.md AGENTS.md
git commit -m "feat(fold): oplm fold eval-sets builds, validates and describes the eval sets

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Build the real sets and write the data manifest (operational)

**Files:**
- Create: `docs/DATA_MANIFEST.md` (where and when each source was obtained and how it was processed; the user will use it to write the methods section)
- Modify: `docs/superpowers/specs/2026-10-27-structure-eval-sets-design.md` (measured counts)

- [ ] **Step 1: Install MMseqs2 and fetch the CASP sources**

```bash
export OPLM_DATA=/home/bryanbriney/data
curl -L https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz | tar xz -C $OPLM_DATA/tmp
$OPLM_DATA/tmp/mmseqs/bin/mmseqs version
mkdir -p $OPLM_DATA/casp && cd $OPLM_DATA/casp
curl -O https://predictioncenter.org/download_area/CASP15/targets/casp15.targets.TS-domains.public_12.20.2022.tar.gz
curl -O https://predictioncenter.org/download_area/CASP15/sequences/casp15.seq.txt
curl -O https://predictioncenter.org/download_area/CASP16/targets/casp16.targets_monomer_trimmed2domains.tgz
curl -O https://predictioncenter.org/download_area/CASP16/sequences/casp16.T1.seq.txt
date -u +%FT%TZ > DOWNLOADED_AT
sha256sum *.gz *.tgz *.txt
```

Expected: a version hash from MMseqs2, and the four sha256 values equal to `CASP_SHA256`. `DOWNLOADED_AT` records the download time for the manifest; the builder ignores files it doesn't pin.

- [ ] **Step 2: Run the build**

```bash
cd /home/bryanbriney/git/oplm
git rev-parse --short HEAD > $OPLM_DATA/tmp/eval-sets.commit
.venv/bin/oplm fold eval-sets --fold-data $OPLM_DATA/hf/wwpdb/2026-10-12 \
    --casp-sources $OPLM_DATA/casp --out $OPLM_DATA/hf/structure-evals/2026-10-12 \
    --mmseqs $OPLM_DATA/tmp/mmseqs/bin/mmseqs --threads 28 > $OPLM_DATA/tmp/eval-sets.log 2>&1
tail -15 $OPLM_DATA/tmp/eval-sets.log
```

Expected (the spec's measured values and estimates):

| set | expected |
|---|---|
| `casp15_16` | 101 |
| `cameo` | about 420 (at most 439) |
| `pdb40` | about 29k |
| `cameo_256`, `pdb40_256`, `pdb40_2048` | 256, 256, 2048 |

Wall time is well under an hour (clustering 10–30 min). A count far off these (for example `pdb40` under 20k, or `casp15_16` ≠ 101) means a filter misbehaves. Stop and investigate with `summary.json`'s per-stage counts before going further.

- [ ] **Step 3: Save the CASP domain summary pages into the sources**

```bash
cd $OPLM_DATA/hf/structure-evals/2026-10-12/sources
curl -o casp15_domains_summary.html https://predictioncenter.org/casp15/domains_summary.cgi
curl -o casp16_domains_summary.html https://predictioncenter.org/casp16/domains_summary.cgi
```

- [ ] **Step 4: Record the measured counts in the spec**

Replace the spec's estimated sizes (Goal table; the CAMEO and PDB40 sections) with the counts from `summary.json`.

- [ ] **Step 5: Write `docs/DATA_MANIFEST.md`**

Create the file from the skeleton below. Each `«...»` names the artifact its value comes from. Replace every one with that value, and copy the tables' numbers from `summary.json` (`$OPLM_DATA/hf/structure-evals/2026-10-12/summary.json`). Values that are already filled come from `docs/fold/m2a/README.md` and `$OPLM_DATA/pdb/targets/sources/README.md`.

````markdown
# Data manifest

Where and when each dataset used by oplm was obtained, and how it was processed. Written to
support the methods section: every value is copied from the build artifact named beside it.

## Structure evaluation sets (`structure-evals/2026-10-12`)

Built «summary.json `built`» with `oplm fold eval-sets` at commit
«`$OPLM_DATA/tmp/eval-sets.commit`» (oplm «summary.json `oplm`», MMseqs2
«summary.json `mmseqs`»). Output: `$OPLM_DATA/hf/structure-evals/2026-10-12/`, with a
`README.md`, `summary.json` (per-stage counts) and `manifest.parquet` (one row per structure,
with its sha256). Design and rationale:
`docs/superpowers/specs/2026-10-27-structure-eval-sets-design.md`.

### Sources

| source | origin | obtained | version, size, checksum |
|---|---|---|---|
| wwPDB mmCIF archive | `rsync.rcsb.org::ftp_data/structures/divided/mmCIF/` (port 33444) | 2026-10-12, 13:10–16:49 UTC | snapshot 2026-10-12; 260,626 entries, 92.3 GB |
| Chemical Component Dictionary | `files.wwpdb.org` `components.cif.gz` | 2026-10-12 | 51,389 components; sha256 `405da7e5d06527d45292db9400378d9788c024830cad4e3b43890b6c6a0dd4c3` |
| Preprocessed PDB (fold dataset) | `oplm fold preprocess`, code `e0ea2fd` (PR #34) | 2026-10-12 | 1,080,220 chains; HF `brineylab/oplmfold-training-data`, tag `wwpdb-2026-10-12` |
| CAMEO target list | CAMEO one-year exports: Wayback capture (2023-08-07) of `to_date=2022-01-15`, and the Biodataset4AI mirror of `to_date=2022-12-31` (CAMEO's archive returned HTTP 503) | 2026-10-12 | 755 chains, weekly releases 2021-10-02 .. 2022-09-24 |
| CASP15 domain targets | `predictioncenter.org/download_area/CASP15/targets/casp15.targets.TS-domains.public_12.20.2022.tar.gz` | «`$OPLM_DATA/casp/DOWNLOADED_AT`» | sha256 `ac159c284a1506b460143ef6e22832272c5da1b4078fdb5fec8292b373b6239b` |
| CASP15 target sequences | `predictioncenter.org/download_area/CASP15/sequences/casp15.seq.txt` | «`DOWNLOADED_AT`» | sha256 `84dd04c1fda2a550af73e77ad25be94da23f16bd7ff6632f2c22c1978b9ee0a5` |
| CASP16 domain targets | `predictioncenter.org/download_area/CASP16/targets/casp16.targets_monomer_trimmed2domains.tgz` | «`DOWNLOADED_AT`» | sha256 `30e6bf992baed08695ebebe481711adea867b483015d30e18d170a0eee66ceb9` |
| CASP16 target sequences | `predictioncenter.org/download_area/CASP16/sequences/casp16.T1.seq.txt` | «`DOWNLOADED_AT`» | sha256 `77d0b5c4ffda52ca1be7563a92eabae7ca59a3aab814e79a377ddd962783e286` |
| CASP domain classifications | `predictioncenter.org/casp15/domains_summary.cgi`, `/casp16/domains_summary.cgi` (saved HTML) | «date of Step 3» | FM/TBM labels, not used for filtering |

### Processing

1. **Structure parsing** (the preprocessed PDB). gemmi mmCIF parsing of L-peptide polymer
   entities in the asymmetric unit; first model; highest-occupancy alternate location per
   atom; hydrogens removed. Non-standard residues are mapped to their CCD parent when the
   component is a non-D peptide, otherwise to unknown. Entries with resolution > 9 Å and
   chains with fewer than 4 resolved residues are dropped. Each chain keeps its full entity
   sequence, with a per-atom resolved mask.
2. **Input convention.** Each structure is one chain. The model receives the full sequence:
   the deposited entity sequence (CAMEO, PDB40) or the full target sequence (CASP).
   Residues without coordinates stay in the sequence and are not scored: any residue pair
   that involves one is excluded from evaluation.
3. **Quality filter** (CAMEO, PDB40): X-ray diffraction at ≤ 2.5 Å or cryo-EM at ≤ 3.0 Å;
   NMR structures are excluded.
4. **Curation rules** (all sets). Only the 20 standard amino acids. Full-sequence length
   50–1024 (CASP, PDB40) or 50–512 (CAMEO). At least 70% of residues with backbone N, Cα and
   C (CAMEO, PDB40). At least L/5 long-range contacts, where L is the number of modeled
   residues and a contact is virtual Cβ–Cβ < 8 Å with sequence separation ≥ 24.
5. **CASP15 and CASP16.** Official domain-trimmed target coordinates. Combined
   (multi-domain) evaluation units are excluded. CASP16 phase-2 copies (`T2xxx`) and
   monomer-file aliases (`T0xxx`) are mapped to their `T1xxx` target. For versioned targets,
   the lowest version is kept. One structure per (target, domain), with only that domain's
   residues carrying coordinates.
6. **CAMEO.** The listed chains that pass the quality filter and the curation rules.
7. **PDB40.** The «summary.json `sets.pdb40.sequences`» distinct protein entity sequences of
   length ≥ 30 in the snapshot were clustered with MMseqs2 (`easy-cluster --min-seq-id 0.4
   -c 0.8 --cov-mode 0 -s 7.5`) into «`sets.pdb40.clusters`» clusters. For each cluster, candidate chains passing the quality
   filter were ranked by resolution, then resolved fraction, then entry and chain id. The
   first candidate passing the curation rules became the cluster's representative.
8. **Subsets.** Seed-42 random subsets of chains ≤ 512 residues: `cameo_256`, `pdb40_2048`
   and `pdb40_256`, with `pdb40_256` contained in `pdb40_2048`.
9. **Scoring** (the `structure` eval type). Unsupervised contact precision from the model's
   categorical Jacobian (L × 20 single-substitution forward passes, centered, symmetrized,
   APC-corrected), reported as precision at L, L/2 and L/5 over long-range pairs.

### Counts

| set | candidates | removed (reason: count) | exported |
|---|---|---|---|
| `casp15_16` | «domain files, distinct domains» | «`sets.casp15_16`: per-edition skips and `rules`» | «`exported.casp15_16`» |
| `cameo` | «`sets.cameo.listed`» | «`missing`, `skipped`, `quality`, `rules`» | «`exported.cameo`» |
| `pdb40` | «`sets.pdb40.clusters`» clusters | «`no_candidate`, `exhausted`, `max_rounds`» | «`exported.pdb40`» |
| `cameo_256`, `pdb40_2048`, `pdb40_256` | | | «`exported`» |

Overlap: «`overlap.cameo_chains_in_pdb40`» CAMEO chains are also PDB40 representatives, and
«`overlap.cameo_clusters_in_pdb40`» CAMEO chains fall in a cluster with a PDB40
representative. Both are reported, not removed.
````

Check that nothing was left unfilled:

Run: `grep -c "«" docs/DATA_MANIFEST.md`
Expected: `0`

- [ ] **Step 6: Commit**

```bash
git add docs/DATA_MANIFEST.md docs/superpowers/specs/2026-10-27-structure-eval-sets-design.md
git commit -m "docs: data manifest for the structure eval sets; measured set sizes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Stop before publishing**

Publishing `brineylab/oplm-structure-evals` to the Hugging Face Hub is outward-facing. Report the counts and the README to the user and wait for the go-ahead. When approved, publish with `hf upload-large-folder` as in docs/FOLD.md §8, then tag the dataset `2026-10-12`.

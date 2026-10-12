# Structure Head Milestone 2a — Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the wwPDB mmCIF archive into published, resumable training data for
the folding head: a gemmi parser to the general `MolecularComplex` representation,
Parquet shards with a per-chain index and provenance manifest, single-chain crops,
training featurization with targets, atom-budget collation, and a `StructureDataset`
that inherits the Trainer's data-exact resume contract.

**Architecture:** Parsing (`mmcif.py`) and the on-disk representation (`complex.py`)
are general per spec §4.2; every filter of §4.4 is applied at parse time with its
reason recorded. `oplm fold preprocess` parses in parallel into ~1 GB shards (one row
per entry, small row groups for random access), writes a per-chain index whose
`split` column carries the date-cutoff and held-out rules, and emits the manifest and
dataset card that make the directory a Hugging Face dataset repo. At runtime
`StructureDataset` subclasses `ShardedProteinDataset`, overriding only the row
reader, so striping, shuffling, `InterleavedDataset` and resume are inherited; each
eligible row yields exactly one single-chain example (seeded chain choice, crop,
rotated conformers) so resume arithmetic holds.

**Tech Stack:** Python 3.11, gemmi 0.7 (mmCIF and CCD parsing; already in the
`fold` and `dev` extras), pyarrow (core dependency; Parquet shards and index), numpy,
torch (featurization tensors), `huggingface_hub` for publishing (documented commands,
no code).

**Spec:** [M2 pipeline-proof addendum](../specs/2026-10-11-fold-m2-pipeline-proof-design.md)
§4 (M2a), §2 (decisions), §3 (chain-length bound), over the
[structure prediction head design](../specs/2026-10-07-structure-prediction-head-design.md)
§4.2–4.4, §4.7, §7.1–7.3, §9 (data row), §11.

**Planning baseline:** `0989317` on branch `docs/fold-m2-spec` (the addendum commit on
top of `main` 7b78452, milestone 1 accepted). Work on branch `feat/fold-m2a` from
that commit, in an isolated worktree with its own `.venv`
(`uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev,train]"`).
No GPU is needed anywhere in this plan. Task 10 is the user-run acceptance on the
development machine (32 cores, 1.6 TB free).

## Global Constraints

- `AGENTS.md` rules: Python 3.11+, `from __future__ import annotations` in every file, type hints on every signature, Google-style docstrings on public APIs, ruff line length 100, `pathlib.Path` only, no logic in `__init__.py` beyond imports, tests mirror the source layout, `@pytest.mark.slow` for anything over a few seconds.
- Commands: `.venv/bin/python -m pytest ...`, `.venv/bin/ruff check src/`, `.venv/bin/ruff format <changed files only>`, `VIRTUAL_ENV=.venv .venv/bin/ty check src/`. All three clean before every commit. Framework-boundary diagnostics only via a specific `# ty: ignore[<rule>]` with a reason.
- Dependencies: no new ones. gemmi stays in the `fold` and `dev` extras (`gemmi>=0.7`); `mmcif.py` and the CCD parser import it at module scope and are imported only by preprocessing and tests, never by `oplm.fold.data.__init__`, `featurize.py`, `complex.py`, `crop.py`, `collate.py` or `dataset.py`.
- Dependency direction (spec §3): `oplm.fold` imports core packages; core imports nothing from `oplm.fold` except `src/oplm/cli.py`. `oplm.fold.cli` keeps torch out of module scope.
- Addendum §2: single-chain training examples; contiguous and spatial crops only; whole archive, protein chains only; publishing in M2a; uniform chain sampling; exact-sequence exclusion of held-out targets; MMseqs2 clustering and homology exclusions are M3.
- Spec §4.2: flat atom and token arrays, never a per-residue layout; bonds "empty in v1, always present"; metadata id, source, resolution, release date, method, dropped-entity flags, per-chain mean pLDDT.
- Spec §4.4, applied at parse time with the rejection reason recorded: protein polymer chains of the asymmetric unit; waters, ions, ligands and nucleic-acid chains dropped and flagged; token sequence from the entity polymer sequence (unresolved residues are tokens with masked coordinates); highest-occupancy altloc; first model; hydrogens dropped; resolution 9 Å or better; chains with fewer than 4 resolved residues dropped; bioassembly expansion deferred.
- Spec §4.1 / addendum §4: non-standard residues with a CCD parent become the parent token and atom set; residues without a single parent become `UNK`. Atom matching is by name with the one AlphaFold convention `MSE` `SE` → `SD`; parent atoms absent from the modified residue are unresolved.
- Spec §4.3: training featurization applies fresh per-token random rotations to the centered reference conformers; inference keeps the raw conformers (M1 parity).
- Spec §4.7: tokens pad to the crop; atoms pad to `atoms_per_token_budget` (default 24) × crop, rounded up to the attention block multiple (32); an over-budget example is rejected before collation, never truncated; token padding, atom padding, reference-geometry validity and resolved masks stay distinct. Crop sizes are multiples of 128.
- Spec §6.4: the dataset supports `__len__`/`total_length`, `set_epoch`, `set_resume_skip`, `clear_resume_skip` and `_arm_sample_skip` with the sequence pipeline's rank/worker striping; crop and augmentation randomness derives from seed, epoch, shard and row so a skipped example is restored exactly.
- Addendum §3: `lm_max_chain_length` (proof default 1024) bounds eligible chains; over-length chains are excluded from sampling by the index filter and never truncated.
- Addendum §4 splits: `train` = release date ≤ 2021-09-30 minus exact-sequence matches of held-out chains; `valid` = 64 seeded post-cutoff monomer chains (one protein chain, resolution ≤ 2.5 Å, length 50–400); `cameo` and `casp14` from target-list files resolved against the snapshot; splits are entry-level.
- Spec §11: wwPDB and CCD are CC0 1.0; the manifest records snapshot date, mirror, per-file checksums, CCD version and checksum, preprocessing configuration, licenses and split rules; the published directory is the training dataset.
- Spec §9: real structure fixtures with provenance notes; every fixture-backed test runs on CPU in the fast suite.

## Review Focus

1. **A chain whose residues are modeled without CA atoms** (backbone fragments only). The center atom is unresolved; spatial crops must never center on such a token and must still fill the crop deterministically (Task 4, `test_spatial_crop_never_centers_on_unresolved_tokens`).
2. **A chain with fewer resolved tokens than the crop size** (2ds7 cropped at 48 has 40 resolved). The spatial crop must fill from unresolved tokens by index, not drop them or raise (Task 4, `test_spatial_crop_fills_with_unresolved_tokens_by_index`).
3. **The same entry id arriving twice** (a `.cif` next to its `.cif.gz`, or a re-run over a merged tree). Preprocess must raise naming both paths rather than silently double-count (Task 7, `test_duplicate_entry_ids_raise`).
4. **Upper-case ids in held-out target lists** (CAMEO and CASP lists are upper-case; the archive is lower-case). Split assignment must normalize (Task 7, `test_heldout_ids_are_case_insensitive`).
5. **An index that names a shard file missing on disk**, or a split with no eligible rows in some shard. The dataset must raise at construction for the first and silently skip the shard for the second, keeping `len()` exact (Task 8, `test_missing_shard_raises_and_empty_shards_are_skipped`).

## File Map and Ownership

| File | Responsibility |
|---|---|
| `src/oplm/fold/data/ccd.py` | + `ChemComp`, `ChemCompTable` (parse `components.cif.gz`, parent mapping, JSON cache), `PROTEIN_3TO1`, `RES_TYPE_TO_CODE` |
| New `src/oplm/fold/data/complex.py` | `ChainInfo`, `ComplexMetadata`, `MolecularComplex` (+ `validate`, `chain_tokens`, `select_tokens`), `COMPLEX_SCHEMA`, `complex_to_row`, `rows_to_table`, `row_to_complex` |
| New `src/oplm/fold/data/mmcif.py` | `parse_mmcif` (gemmi) → `MolecularComplex | Rejection` |
| New `src/oplm/fold/data/crop.py` | `contiguous_crop`, `spatial_crop`, `crop_chain` |
| `src/oplm/fold/data/featurize.py` | + `FoldTargets`, `FoldExample`, `featurize_complex`; shared `_assemble` |
| New `src/oplm/fold/data/collate.py` | `FoldBatch`, `collate_examples` (atom budget) |
| New `src/oplm/fold/data/dataset.py` | `PreprocessConfig`, `preprocess`, `assign_splits`, `read_target_list`, `StructureDataset` |
| `src/oplm/fold/cli.py` | + `oplm fold preprocess` |
| `src/oplm/fold/data/__init__.py` | docstring only (unchanged contract) |
| New `tests/fixtures/fold/` | `mmcif/{1crn,1ejg,4hhb,2ds7,1jb6,1l2y}.cif.gz` (the archive root for tests), `ccd_subset.cif`, `README.md` |
| New `tests/fold/data/conftest.py` | fixture paths, parsed complexes, a preprocessed fixture dataset |
| New tests `tests/fold/data/test_{complex,mmcif,crop,collate,dataset}.py`; extended `test_ccd.py`, `test_featurize.py`; `tests/fold/test_cli.py` | |
| `docs/FOLD.md`, `docs/TESTING_E2E.md`, `tests/fixtures/README.md` | contract and fixture provenance |
| New `docs/fold/m2a/` | acceptance records (Task 10) |

Execute Tasks 1–9 in order; Task 10 is the user-run acceptance.

---

### Task 1: Chemical component table

**Files:**
- Modify: `src/oplm/fold/data/ccd.py`
- Create: `tests/fixtures/fold/ccd_subset.cif`
- Test: `tests/fold/data/test_ccd.py` (append)

**Interfaces:**
- Consumes: `PROTEIN_RES_TYPES`, `PROTEIN_1TO3` (existing).
- Produces: `ChemComp(code, type, parent, one_letter)`, `ChemCompTable.parse(path) -> ChemCompTable`, `.save(path)`, `.load(path)`, `.protein_parent(code) -> str` (a standard three-letter code or `"UNK"`), `.is_peptide(code) -> bool`, `__contains__`, `__getitem__`, `__len__`, `.source`; `PROTEIN_3TO1: dict[str, str]` (includes `UNK -> X`); `RES_TYPE_TO_CODE: dict[int, str]`. Task 3 calls `protein_parent` per sequence position; Task 5 uses `RES_TYPE_TO_CODE`; Task 7 parses or loads the table once per worker.

- [ ] **Step 1: Create the CCD subset fixture**

Download the dictionary once (about 120 MB; it is not committed) and write a trimmed copy holding only the `_chem_comp` category of 45 components:

```bash
mkdir -p /tmp/ccd && curl -sS -o /tmp/ccd/components.cif.gz https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
.venv/bin/python - <<'EOF'
import gemmi
keep = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO",
        "SER","THR","TRP","TYR","VAL","UNK","MSE","SEP","TPO","PTR","NLE","HYP","PCA","CSO","HOH",
        "HEM","ATP","DA","DC","DG","DT","A","C","G","U","0YG","175","ACE","NH2","DAL"]
doc = gemmi.cif.read("/tmp/ccd/components.cif.gz")
out = gemmi.cif.Document()
for block in doc:
    if block.name in keep:
        nb = out.add_new_block(block.name)
        for item in block:
            if item.pair is not None and item.pair[0].startswith("_chem_comp."):
                nb.set_pair(item.pair[0], item.pair[1])
assert len(out) == len(keep), len(out)
out.write_file("tests/fixtures/fold/ccd_subset.cif")
EOF
ls -la tests/fixtures/fold/ccd_subset.cif   # about 40 KB
```

Expected parents in the fixture (verified 2026-10-11): `MSE→MET`, `SEP→SER`, `TPO→THR`,
`PTR→TYR`, `NLE→LEU`, `HYP→PRO`, `PCA→GLN`, `CSO→CYS`; `0YG` has `TYR,GLY` and `175` has
`"ALA, SER, GLY"` (multi-parent); `DAL` is `D-PEPTIDE LINKING` with no parent; `HOH`,
`HEM`, `ATP`, `ACE`, `NH2` are `NON-POLYMER`.

- [ ] **Step 2: Append the failing tests**

```python
# tests/fold/data/test_ccd.py (append)
from pathlib import Path

from oplm.fold.data.ccd import PROTEIN_3TO1, RES_TYPE_TO_CODE, ChemCompTable

CCD_SUBSET = Path(__file__).parents[2] / "fixtures" / "fold" / "ccd_subset.cif"


def test_chem_comp_table_parses_parents_types_and_letters() -> None:
    table = ChemCompTable.parse(CCD_SUBSET)
    assert len(table) == 45 and "MSE" in table and "ZZZ" not in table
    assert table["MSE"].parent == "MET" and table["MSE"].one_letter == "M"
    assert table["MSE"].type == "L-PEPTIDE LINKING"
    assert table["0YG"].type == "L-PEPTIDE NH3 AMINO TERMINUS"  # upper-cased
    assert table["0YG"].parent is None and table["175"].parent is None  # multi-parent
    assert table["ALA"].parent is None and table["HOH"].parent is None
    assert table["HOH"].one_letter is None
    assert table.source.startswith("ccd_subset.cif")


@pytest.mark.parametrize(
    ("code", "parent"),
    [
        ("ALA", "ALA"),  # standard: itself
        ("UNK", "UNK"),
        ("MSE", "MET"),
        ("SEP", "SER"),
        ("NLE", "LEU"),
        ("PCA", "GLN"),
        ("HYP", "PRO"),
        ("0YG", "UNK"),  # multi-parent
        ("DAL", "UNK"),  # D-amino acid, no parent
        ("HOH", "UNK"),  # not a peptide at all
        ("ZZZ", "UNK"),  # unknown code
    ],
)
def test_protein_parent(code: str, parent: str) -> None:
    assert ChemCompTable.parse(CCD_SUBSET).protein_parent(code) == parent


def test_is_peptide() -> None:
    table = ChemCompTable.parse(CCD_SUBSET)
    assert table.is_peptide("MSE") and table.is_peptide("DAL") and table.is_peptide("GLY")
    assert not table.is_peptide("HOH") and not table.is_peptide("DA") and not table.is_peptide("ZZZ")


def test_chem_comp_table_json_round_trip(tmp_path: Path) -> None:
    table = ChemCompTable.parse(CCD_SUBSET)
    table.save(tmp_path / "ccd.json")
    loaded = ChemCompTable.load(tmp_path / "ccd.json")
    assert len(loaded) == len(table) and loaded.source == table.source
    assert all(loaded[c] == table[c] for c in ("MSE", "0YG", "HOH", "ALA"))


def test_inverse_tables() -> None:
    assert PROTEIN_3TO1["MET"] == "M" and PROTEIN_3TO1["UNK"] == "X" and len(PROTEIN_3TO1) == 21
    assert RES_TYPE_TO_CODE[2] == "ALA" and RES_TYPE_TO_CODE[22] == "UNK" and len(RES_TYPE_TO_CODE) == 21
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/data/test_ccd.py -q`
Expected: FAIL with `ImportError: cannot import name 'ChemCompTable'`.

- [ ] **Step 4: Implement the table**

Append to `src/oplm/fold/data/ccd.py` (add `from typing import TYPE_CHECKING`; keep gemmi inside `parse` so the module stays importable in the parity venv without touching gemmi at import time):

```python
PROTEIN_3TO1: dict[str, str] = {code: letter for letter, code in PROTEIN_1TO3.items()}
RES_TYPE_TO_CODE: dict[int, str] = {res_type: code for code, res_type in PROTEIN_RES_TYPES.items()}


def _cif_value(raw: str | None) -> str | None:
    """A ``_chem_comp`` value with CIF quoting removed; ``None`` for absent, ``?`` and ``.``."""
    if raw is None or raw in ("?", "."):
        return None
    import gemmi

    return gemmi.cif.as_string(raw)


@dataclass(frozen=True)
class ChemComp:
    """One ``_chem_comp`` row of the wwPDB Chemical Component Dictionary."""

    code: str
    type: str  # ``_chem_comp.type`` upper-cased, e.g. "L-PEPTIDE LINKING"
    parent: str | None  # ``mon_nstd_parent_comp_id`` when it names exactly one component
    one_letter: str | None


class ChemCompTable:
    """``_chem_comp`` rows keyed by code, from ``components.cif.gz`` or a saved JSON cache."""

    def __init__(self, comps: dict[str, ChemComp], source: str) -> None:
        self._comps = comps
        self.source = source

    @classmethod
    def parse(cls, components_cif: Path) -> ChemCompTable:
        """Parse the dictionary (gzipped or plain mmCIF); about 3 s for the full 51k-block file."""
        import gemmi

        doc = gemmi.cif.read(str(components_cif))
        comps: dict[str, ChemComp] = {}
        for block in doc:
            parent = _cif_value(block.find_value("_chem_comp.mon_nstd_parent_comp_id"))
            if parent is not None and "," in parent:
                parent = None  # multi-parent components (e.g. "ALA, SER, GLY") have no single parent
            comps[block.name] = ChemComp(
                code=block.name,
                type=(_cif_value(block.find_value("_chem_comp.type")) or "").upper(),
                parent=parent,
                one_letter=_cif_value(block.find_value("_chem_comp.one_letter_code")),
            )
        return cls(comps, f"{Path(components_cif).name} ({len(comps)} components)")

    def save(self, path: Path) -> None:
        """Write the table as JSON (``{"source": ..., "components": {code: {...}}}``)."""
        payload = {
            "source": self.source,
            "components": {
                code: {"type": c.type, "parent": c.parent, "one_letter": c.one_letter}
                for code, c in self._comps.items()
            },
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ChemCompTable:
        """Read a table written by :meth:`save`."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        comps = {
            code: ChemComp(code, entry["type"], entry["parent"], entry["one_letter"])
            for code, entry in raw["components"].items()
        }
        return cls(comps, raw["source"])

    def __len__(self) -> int:
        return len(self._comps)

    def __contains__(self, code: str) -> bool:
        return code in self._comps

    def __getitem__(self, code: str) -> ChemComp:
        return self._comps[code]

    def is_peptide(self, code: str) -> bool:
        """Whether the component's type names a peptide (any chirality or linkage)."""
        comp = self._comps.get(code)
        return comp is not None and "PEPTIDE" in comp.type

    def protein_parent(self, code: str) -> str:
        """The standard amino acid a polymer residue collapses to (spec §4.1), else ``"UNK"``.

        A standard residue is its own parent; a modified residue resolves to its single
        ``mon_nstd_parent_comp_id`` when that is a standard amino acid; everything else,
        including multi-parent components and unknown codes, becomes ``UNK``.
        """
        if code in PROTEIN_RES_TYPES:
            return code
        comp = self._comps.get(code)
        if comp is not None and comp.parent in PROTEIN_RES_TYPES and comp.parent != "UNK":
            return comp.parent
        return "UNK"
```

Add `"ChemComp"`, `"ChemCompTable"`, `"PROTEIN_3TO1"`, `"RES_TYPE_TO_CODE"` to `__all__`. Update the module docstring's last sentence: the gemmi-parsed CCD table now exists for the parent mapping; ligand and nucleic-acid atom tables remain milestone 5.

- [ ] **Step 5: Run the tests, lint, types**

Run: `.venv/bin/python -m pytest tests/fold/data/test_ccd.py -q && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/data/ccd.py tests/fold/data/test_ccd.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: all pass, no diagnostics.

- [ ] **Step 6: Commit**

```bash
git add src/oplm/fold/data/ccd.py tests/fold/data/test_ccd.py tests/fixtures/fold/ccd_subset.cif
git commit -m "feat(fold): CCD chemical component table with parent mapping"
```

---

### Task 2: `MolecularComplex` and its Arrow row

**Files:**
- Create: `src/oplm/fold/data/complex.py`
- Test: `tests/fold/data/test_complex.py`

**Interfaces:**
- Consumes: nothing from this plan.
- Produces: `ChainInfo(chain_id, auth_chain_id, entity_id, sym_id, mol_type, sequence, mean_plddt=None)`, `ComplexMetadata(id, source, method, resolution, release_date, dropped)`, `MolecularComplex` with the fields listed in the code below plus `num_atoms`, `num_tokens`, `chain_tokens(chain) -> np.ndarray`, `validate() -> None`, `select_tokens(tokens) -> MolecularComplex`; `COMPLEX_SCHEMA: pa.Schema`; `complex_to_row(c) -> dict[str, object]`; `rows_to_table(rows) -> pa.Table`; `row_to_complex(row) -> MolecularComplex`. Task 3 builds complexes; Task 4 calls `chain_tokens`/`select_tokens`; Task 7 writes rows; Task 8 reads them.

- [ ] **Step 1: Write the failing tests**

```python
# tests/fold/data/test_complex.py
"""The flat all-atom representation and its Parquet row (spec §4.2, §4.7)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from oplm.fold.data.complex import (
    COMPLEX_SCHEMA,
    ChainInfo,
    ComplexMetadata,
    MolecularComplex,
    complex_to_row,
    row_to_complex,
    rows_to_table,
)


def two_chain_complex() -> MolecularComplex:
    """Chain A: GLY (4 atoms), ALA (5 atoms); chain B: GLY (4 atoms). 13 atoms, 3 tokens."""
    counts = np.array([4, 5, 4], dtype=np.int32)
    starts = np.array([0, 4, 9], dtype=np.int32)
    names = ["N", "CA", "C", "O", "N", "CA", "C", "O", "CB", "N", "CA", "C", "O"]
    return MolecularComplex(
        atom_coords=np.arange(39, dtype=np.float32).reshape(13, 3),
        atom_resolved=np.array([True] * 8 + [False] + [True] * 4),
        atom_element=np.array([7, 6, 6, 8, 7, 6, 6, 8, 6, 7, 6, 6, 8], dtype=np.uint8),
        atom_name=names,
        atom_charge=np.zeros(13, dtype=np.int8),
        atom_bfactor=np.full(13, 20.0, dtype=np.float32),
        atom_to_token=np.array([0] * 4 + [1] * 5 + [2] * 4, dtype=np.int32),
        token_res_type=np.array([9, 2, 9], dtype=np.int8),
        token_mol_type=np.zeros(3, dtype=np.int8),
        token_asym_id=np.array([0, 0, 1], dtype=np.int32),
        token_entity_id=np.array([0, 0, 1], dtype=np.int32),
        token_sym_id=np.zeros(3, dtype=np.int32),
        token_residue_index=np.array([0, 1, 0], dtype=np.int32),
        token_center_atom=starts + 1,
        token_rep_atom=np.array([1, 8, 10], dtype=np.int32),
        token_atom_start=starts,
        token_atom_count=counts,
        bonds=np.zeros((0, 2), dtype=np.int32),
        chains=[
            ChainInfo("A", "A", 0, 0, 0, "GA"),
            ChainInfo("B", "B", 1, 0, 0, "G", mean_plddt=0.5),
        ],
        metadata=ComplexMetadata("test", "pdb", "X-RAY DIFFRACTION", 1.5, "2000-01-01", {"water": 1}),
    )


def assert_complexes_equal(a: MolecularComplex, b: MolecularComplex) -> None:
    for f in dataclasses.fields(MolecularComplex):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype, f.name
            assert np.array_equal(x, y), f.name
        else:
            assert x == y, f.name


def test_validate_accepts_a_consistent_complex() -> None:
    c = two_chain_complex()
    c.validate()
    assert c.num_atoms == 13 and c.num_tokens == 3
    assert c.chain_tokens(0).tolist() == [0, 1] and c.chain_tokens(1).tolist() == [2]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("atom_to_token", np.array([0] * 13, dtype=np.int32), "atom_to_token"),
        ("token_atom_start", np.array([0, 5, 9], dtype=np.int32), "contiguous"),
        ("token_rep_atom", np.array([1, 8, 3], dtype=np.int32), "representative"),
        ("token_asym_id", np.array([0, 0, 2], dtype=np.int32), "asym_id"),
        ("atom_name", ["N"] * 12, "atom count"),
    ],
)
def test_validate_rejects_inconsistent_indices(field: str, value: object, match: str) -> None:
    c = dataclasses.replace(two_chain_complex(), **{field: value})
    with pytest.raises(ValueError, match=match):
        c.validate()


def test_select_tokens_remaps_atoms_and_compacts_chains() -> None:
    c = two_chain_complex()
    sub = c.select_tokens(np.array([2, 1]))  # unsorted input: result is in token order
    sub.validate()
    assert sub.num_tokens == 2 and sub.num_atoms == 9
    assert sub.token_atom_start.tolist() == [0, 5] and sub.token_atom_count.tolist() == [5, 4]
    assert sub.atom_name == c.atom_name[4:13]
    assert np.array_equal(sub.atom_coords, c.atom_coords[4:13])
    assert sub.token_center_atom.tolist() == [1, 6] and sub.token_rep_atom.tolist() == [4, 6]
    assert sub.token_asym_id.tolist() == [0, 1]  # both chains still present
    assert sub.token_residue_index.tolist() == [1, 0]  # original identities preserved
    only_b = c.select_tokens(np.array([2]))
    assert only_b.token_asym_id.tolist() == [0] and [ch.chain_id for ch in only_b.chains] == ["B"]
    assert only_b.token_entity_id.tolist() == [1] and only_b.chains[0].entity_id == 1
    assert only_b.metadata is c.metadata


def test_arrow_row_round_trip(tmp_path: Path) -> None:
    c = two_chain_complex()
    table = rows_to_table([complex_to_row(c), complex_to_row(c.select_tokens(np.array([2])))])
    assert table.schema.equals(COMPLEX_SCHEMA) and table.num_rows == 2
    pq.write_table(table, tmp_path / "shard.parquet", row_group_size=1)
    back = pq.read_table(tmp_path / "shard.parquet")
    assert_complexes_equal(row_to_complex(back.slice(0, 1).to_pylist()[0]), c)
    assert_complexes_equal(
        row_to_complex(back.slice(1, 1).to_pylist()[0]), c.select_tokens(np.array([2]))
    )


def test_round_trip_keeps_none_resolution_and_plddt() -> None:
    c = two_chain_complex()
    c.metadata = ComplexMetadata("nmr", "pdb", "SOLUTION NMR", None, None, {})
    back = row_to_complex(rows_to_table([complex_to_row(c)]).to_pylist()[0])
    assert back.metadata.resolution is None and back.metadata.release_date is None
    assert back.chains[0].mean_plddt is None and back.chains[1].mean_plddt == 0.5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/data/test_complex.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.fold.data.complex'`.

- [ ] **Step 3: Implement the module**

```python
# src/oplm/fold/data/complex.py
"""``MolecularComplex``: the flat all-atom representation (spec §4.2) and its Parquet row.

Atoms and tokens are flat arrays, never a per-residue layout. The atoms of one token are
contiguous and in that token's template order (``ReferenceConformers``), so training
coordinates line up with the ``ref_*`` features by construction. Bonds are empty in v1 but
always present. ``complex_to_row``/``rows_to_table``/``row_to_complex`` are the shard contract:
one row per entry, ragged list columns, every array flattened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa

__all__ = [
    "COMPLEX_SCHEMA",
    "ChainInfo",
    "ComplexMetadata",
    "MolecularComplex",
    "complex_to_row",
    "row_to_complex",
    "rows_to_table",
]


@dataclass
class ChainInfo:
    """One polymer chain: identity, entity and symmetry copy, and its full sequence."""

    chain_id: str  # label_asym_id
    auth_chain_id: str  # auth_asym_id
    entity_id: int  # index among the kept polymer entities of the entry
    sym_id: int  # copy index within the entity
    mol_type: int  # MOL_TYPE_* from ccd.py
    sequence: str  # the entity polymer sequence, one letter per token, X for unknown
    mean_plddt: float | None = None  # predicted structures only


@dataclass
class ComplexMetadata:
    """Entry-level provenance and the §4.4 dropped-entity flags."""

    id: str
    source: str
    method: str
    resolution: float | None
    release_date: str | None  # ISO date
    dropped: dict[str, int] = field(default_factory=dict)  # kind -> count of entities/chains


@dataclass
class MolecularComplex:
    """Flat atoms, flat tokens, bonds, chains and metadata (spec §4.2)."""

    atom_coords: np.ndarray  # (A, 3) float32; zeros where unresolved
    atom_resolved: np.ndarray  # (A,) bool
    atom_element: np.ndarray  # (A,) uint8 atomic number
    atom_name: list[str]  # (A,)
    atom_charge: np.ndarray  # (A,) int8 formal charge
    atom_bfactor: np.ndarray  # (A,) float32 B-factor, or pLDDT for predicted structures
    atom_to_token: np.ndarray  # (A,) int32
    token_res_type: np.ndarray  # (L,) int8 (PROTEIN_RES_TYPES values)
    token_mol_type: np.ndarray  # (L,) int8
    token_asym_id: np.ndarray  # (L,) int32 index into ``chains``
    token_entity_id: np.ndarray  # (L,) int32
    token_sym_id: np.ndarray  # (L,) int32
    token_residue_index: np.ndarray  # (L,) int32, 0-based within the chain sequence
    token_center_atom: np.ndarray  # (L,) int32 flat atom index (CA for protein)
    token_rep_atom: np.ndarray  # (L,) int32 flat atom index (CB; CA for GLY and UNK)
    token_atom_start: np.ndarray  # (L,) int32
    token_atom_count: np.ndarray  # (L,) int32
    bonds: np.ndarray  # (N, 2) int32 flat atom pairs; empty in v1
    chains: list[ChainInfo]
    metadata: ComplexMetadata

    @property
    def num_atoms(self) -> int:
        return int(self.atom_to_token.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.token_res_type.shape[0])

    def chain_tokens(self, chain: int) -> np.ndarray:
        """Token indices of chain ``chain`` (an index into ``chains``), ascending."""
        return np.flatnonzero(self.token_asym_id == chain)

    def validate(self) -> None:
        """Check the index invariants every consumer relies on.

        Raises:
            ValueError: An atom or token array has the wrong length, token atom ranges are not
                contiguous from 0, ``atom_to_token`` disagrees with the ranges, a center or
                representative atom lies outside its token's range, an ``asym_id`` is outside
                ``chains``, or ``bonds`` is not ``(N, 2)``.
        """
        n_atoms, n_tokens = self.num_atoms, self.num_tokens
        atom_arrays = (
            self.atom_coords,
            self.atom_resolved,
            self.atom_element,
            self.atom_charge,
            self.atom_bfactor,
        )
        if len(self.atom_name) != n_atoms or any(a.shape[0] != n_atoms for a in atom_arrays):
            raise ValueError("atom arrays disagree on the atom count")
        token_arrays = (
            self.token_mol_type,
            self.token_asym_id,
            self.token_entity_id,
            self.token_sym_id,
            self.token_residue_index,
            self.token_center_atom,
            self.token_rep_atom,
            self.token_atom_start,
            self.token_atom_count,
        )
        if any(a.shape[0] != n_tokens for a in token_arrays):
            raise ValueError("token arrays disagree on the token count")
        ends = self.token_atom_start + self.token_atom_count
        expected_starts = np.concatenate([[0], ends[:-1]]) if n_tokens else np.zeros(0)
        if not np.array_equal(self.token_atom_start, expected_starts) or (
            n_tokens and ends[-1] != n_atoms
        ):
            raise ValueError("token atom ranges are not contiguous over the atom axis")
        expected = np.repeat(np.arange(n_tokens, dtype=np.int32), self.token_atom_count)
        if not np.array_equal(self.atom_to_token, expected):
            raise ValueError("atom_to_token does not match the token atom ranges")
        for label, idx in (("center", self.token_center_atom), ("representative", self.token_rep_atom)):
            if n_tokens and np.any((idx < self.token_atom_start) | (idx >= ends)):
                raise ValueError(f"{label} atom outside its token's atom range")
        if n_tokens and (self.token_asym_id.min() < 0 or self.token_asym_id.max() >= len(self.chains)):
            raise ValueError("token asym_id outside the chain list")
        if self.bonds.ndim != 2 or self.bonds.shape[1] != 2:
            raise ValueError("bonds must have shape (N, 2)")

    def select_tokens(self, tokens: np.ndarray) -> MolecularComplex:
        """The sub-complex of ``tokens``: atoms remapped, chains compacted to those present.

        Residue indices, entity ids, symmetry ids and chain identities are preserved;
        ``token_asym_id`` is renumbered densely over the chains that remain, in their original
        order. Bonds with an endpoint outside the selection are dropped.
        """
        tokens = np.unique(np.asarray(tokens, dtype=np.int64))
        counts = self.token_atom_count[tokens].astype(np.int32)
        atom_idx = np.concatenate(
            [np.arange(s, s + c) for s, c in zip(self.token_atom_start[tokens], counts, strict=True)]
            or [np.zeros(0, dtype=np.int64)]
        ).astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int32)
        present = np.unique(self.token_asym_id[tokens])
        asym_map = np.full(len(self.chains), -1, dtype=np.int32)
        asym_map[present] = np.arange(present.size, dtype=np.int32)
        atom_map = np.full(self.num_atoms, -1, dtype=np.int64)
        atom_map[atom_idx] = np.arange(atom_idx.size)
        keep_bonds = (
            np.all(np.isin(self.bonds, atom_idx), axis=1)
            if self.bonds.size
            else np.zeros(0, dtype=bool)
        )
        return MolecularComplex(
            atom_coords=self.atom_coords[atom_idx],
            atom_resolved=self.atom_resolved[atom_idx],
            atom_element=self.atom_element[atom_idx],
            atom_name=[self.atom_name[i] for i in atom_idx],
            atom_charge=self.atom_charge[atom_idx],
            atom_bfactor=self.atom_bfactor[atom_idx],
            atom_to_token=np.repeat(np.arange(tokens.size, dtype=np.int32), counts),
            token_res_type=self.token_res_type[tokens],
            token_mol_type=self.token_mol_type[tokens],
            token_asym_id=asym_map[self.token_asym_id[tokens]],
            token_entity_id=self.token_entity_id[tokens],
            token_sym_id=self.token_sym_id[tokens],
            token_residue_index=self.token_residue_index[tokens],
            token_center_atom=atom_map[self.token_center_atom[tokens]].astype(np.int32),
            token_rep_atom=atom_map[self.token_rep_atom[tokens]].astype(np.int32),
            token_atom_start=starts,
            token_atom_count=counts,
            bonds=atom_map[self.bonds[keep_bonds]].astype(np.int32).reshape(-1, 2),
            chains=[self.chains[int(c)] for c in present],
            metadata=self.metadata,
        )


COMPLEX_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("source", pa.string()),
        ("method", pa.string()),
        ("resolution", pa.float32()),
        ("release_date", pa.string()),
        ("dropped_json", pa.string()),
        ("chain_ids", pa.list_(pa.string())),
        ("chain_auth_ids", pa.list_(pa.string())),
        ("chain_entity_ids", pa.list_(pa.int32())),
        ("chain_sym_ids", pa.list_(pa.int32())),
        ("chain_mol_types", pa.list_(pa.int8())),
        ("chain_sequences", pa.list_(pa.string())),
        ("chain_mean_plddt", pa.list_(pa.float32())),
        ("atom_coords", pa.list_(pa.float32())),  # flattened (A*3)
        ("atom_resolved", pa.list_(pa.bool_())),
        ("atom_element", pa.list_(pa.uint8())),
        ("atom_name", pa.list_(pa.string())),
        ("atom_charge", pa.list_(pa.int8())),
        ("atom_bfactor", pa.list_(pa.float32())),
        ("atom_to_token", pa.list_(pa.int32())),
        ("token_res_type", pa.list_(pa.int8())),
        ("token_mol_type", pa.list_(pa.int8())),
        ("token_asym_id", pa.list_(pa.int32())),
        ("token_entity_id", pa.list_(pa.int32())),
        ("token_sym_id", pa.list_(pa.int32())),
        ("token_residue_index", pa.list_(pa.int32())),
        ("token_center_atom", pa.list_(pa.int32())),
        ("token_rep_atom", pa.list_(pa.int32())),
        ("token_atom_start", pa.list_(pa.int32())),
        ("token_atom_count", pa.list_(pa.int32())),
        ("bond_atoms", pa.list_(pa.int32())),  # flattened (N*2)
    ]
)

_TOKEN_FIELDS = (
    "token_res_type",
    "token_mol_type",
    "token_asym_id",
    "token_entity_id",
    "token_sym_id",
    "token_residue_index",
    "token_center_atom",
    "token_rep_atom",
    "token_atom_start",
    "token_atom_count",
)
_ARRAY_DTYPES: dict[str, type] = {
    "atom_resolved": np.bool_,
    "atom_element": np.uint8,
    "atom_charge": np.int8,
    "atom_bfactor": np.float32,
    "atom_to_token": np.int32,
    "token_res_type": np.int8,
    "token_mol_type": np.int8,
    "token_asym_id": np.int32,
    "token_entity_id": np.int32,
    "token_sym_id": np.int32,
    "token_residue_index": np.int32,
    "token_center_atom": np.int32,
    "token_rep_atom": np.int32,
    "token_atom_start": np.int32,
    "token_atom_count": np.int32,
}


def complex_to_row(c: MolecularComplex) -> dict[str, object]:
    """One shard row (column name -> value) for ``c``; arrays stay numpy until `rows_to_table`."""
    m = c.metadata
    row: dict[str, object] = {
        "id": m.id,
        "source": m.source,
        "method": m.method,
        "resolution": m.resolution,
        "release_date": m.release_date,
        "dropped_json": json.dumps(m.dropped, sort_keys=True),
        "chain_ids": [ch.chain_id for ch in c.chains],
        "chain_auth_ids": [ch.auth_chain_id for ch in c.chains],
        "chain_entity_ids": [ch.entity_id for ch in c.chains],
        "chain_sym_ids": [ch.sym_id for ch in c.chains],
        "chain_mol_types": [ch.mol_type for ch in c.chains],
        "chain_sequences": [ch.sequence for ch in c.chains],
        "chain_mean_plddt": [ch.mean_plddt for ch in c.chains],
        "atom_coords": c.atom_coords.astype(np.float32).reshape(-1),
        "atom_name": list(c.atom_name),
        "bond_atoms": c.bonds.astype(np.int32).reshape(-1),
    }
    for name, dtype in _ARRAY_DTYPES.items():
        row[name] = getattr(c, name).astype(dtype)
    return row


def rows_to_table(rows: list[dict[str, object]]) -> pa.Table:
    """Assemble rows from :func:`complex_to_row` into a table with :data:`COMPLEX_SCHEMA`."""
    columns = [
        pa.array([row[f.name] for row in rows], type=f.type) for f in COMPLEX_SCHEMA
    ]
    return pa.Table.from_arrays(columns, schema=COMPLEX_SCHEMA)


def row_to_complex(row: dict[str, object]) -> MolecularComplex:
    """Rebuild a complex from one ``Table.to_pylist()`` row."""
    arrays = {name: np.asarray(row[name], dtype=dtype) for name, dtype in _ARRAY_DTYPES.items()}
    chains = [
        ChainInfo(cid, auth, int(ent), int(sym), int(mol), seq, plddt)
        for cid, auth, ent, sym, mol, seq, plddt in zip(
            row["chain_ids"],  # ty: ignore[not-iterable]  # pylist rows are untyped
            row["chain_auth_ids"],  # ty: ignore[not-iterable]
            row["chain_entity_ids"],  # ty: ignore[not-iterable]
            row["chain_sym_ids"],  # ty: ignore[not-iterable]
            row["chain_mol_types"],  # ty: ignore[not-iterable]
            row["chain_sequences"],  # ty: ignore[not-iterable]
            row["chain_mean_plddt"],  # ty: ignore[not-iterable]
            strict=True,
        )
    ]
    resolution = row["resolution"]
    return MolecularComplex(
        atom_coords=np.asarray(row["atom_coords"], dtype=np.float32).reshape(-1, 3),
        atom_name=list(row["atom_name"]),  # ty: ignore[invalid-argument-type]
        bonds=np.asarray(row["bond_atoms"], dtype=np.int32).reshape(-1, 2),
        chains=chains,
        metadata=ComplexMetadata(
            id=str(row["id"]),
            source=str(row["source"]),
            method=str(row["method"]),
            resolution=None if resolution is None else float(resolution),  # ty: ignore[invalid-argument-type]
            release_date=None if row["release_date"] is None else str(row["release_date"]),
            dropped=json.loads(str(row["dropped_json"])),
        ),
        **arrays,
    )
```

If `ty` accepts the pylist row values without the inline ignores, drop them; keep only the
ignores it actually needs, each with its reason.

- [ ] **Step 4: Run the tests, lint, types**

Run: `.venv/bin/python -m pytest tests/fold/data/test_complex.py -q && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/data/complex.py tests/fold/data/test_complex.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: all pass, no diagnostics.

- [ ] **Step 5: Commit**

```bash
git add src/oplm/fold/data/complex.py tests/fold/data/test_complex.py
git commit -m "feat(fold): MolecularComplex representation and its Parquet row"
```

---

### Task 3: mmCIF parsing on real fixtures

**Files:**
- Create: `src/oplm/fold/data/mmcif.py`
- Create: `tests/fixtures/fold/{1crn,1ejg,4hhb,2ds7,1jb6,1l2y}.cif.gz`, `tests/fixtures/fold/README.md`
- Create: `tests/fold/data/conftest.py`
- Test: `tests/fold/data/test_mmcif.py`

**Interfaces:**
- Consumes: `ChemCompTable.protein_parent` (Task 1), `PROTEIN_3TO1`, `PROTEIN_RES_TYPES`, `MOL_TYPE_PROTEIN`, `ReferenceConformers` (existing), `MolecularComplex`, `ChainInfo`, `ComplexMetadata` (Task 2).
- Produces: `parse_mmcif(path, *, ccd, conformers=None, max_resolution=9.0, min_resolved_residues=4) -> MolecularComplex | Rejection`, `Rejection(id, reason, detail="")` with reasons `"parse_error" | "resolution" | "no_protein_chains" | "no_resolved_chain"`, constants `PARSE_VERSION`, `MAX_RESOLUTION`, `MIN_RESOLVED_RESIDUES`. Task 7 calls `parse_mmcif` in worker processes and counts `Rejection.reason`.

- [ ] **Step 1: Add the structure fixtures with pinned checksums**

```bash
mkdir -p tests/fixtures/fold/mmcif && cd tests/fixtures/fold/mmcif
for id in 1crn 1ejg 4hhb 2ds7 1jb6 1l2y; do curl -sS -o $id.cif.gz https://files.rcsb.org/download/$id.cif.gz; done
sha256sum -c <<'EOF'
16602cc74cd75747fd48dca2c5f8a674e009bdb50ad521913469769c2bea3c86  1crn.cif.gz
871e6f26a3d68e6f33516ea0c7c5964b8eecf663f3d2a156f603052875426e11  1ejg.cif.gz
6f41c480c041388976352af8212da8c6fb6fad0c406fa2b84db12118892eb19b  4hhb.cif.gz
82011b37f70f09f649d01ca252079253034e97ca5b9daeca08ad0a8a6fc14939  2ds7.cif.gz
ab402e4f080679c81893ed0b31ebf5b3a0b095a1882ae132e0c8a1d65be48944  1jb6.cif.gz
b8a2bd7e12c26fa920277703c6cfd1404ee189425f9f4b28658780e10106715e  1l2y.cif.gz
EOF
cd -
```

If a checksum differs, the archive file was re-released after 2026-10-11; keep the new
file, record its checksum in the README below, and expect the version-dependent
assertions (release date, atom counts) to need updating.

Write `tests/fixtures/fold/README.md`:

```markdown
# Fold data fixtures

Real wwPDB entries (`mmcif/`: mmCIF, gzipped, downloaded from https://files.rcsb.org/download/ on
2026-10-11; wwPDB data is CC0 1.0) and a trimmed Chemical Component Dictionary. Checksums
are pinned in `docs/superpowers/plans/2026-10-11-fold-m2a-data.md` Task 3.

| File | Why it is here |
| --- | --- |
| `mmcif/1crn.cif.gz` | Crambin, 46-residue X-ray monomer at 1.5 Å, every residue resolved. |
| `mmcif/1ejg.cif.gz` | Crambin at 0.54 Å: altlocs (381 atoms), hydrogens, microheterogeneity (`PRO,SER` at 22, `LEU,ILE` at 25; the parsed sequence therefore has L at 25 where 1crn has I). |
| `mmcif/4hhb.cif.gz` | Hemoglobin: two entities × two copies (symmetry ids), hemes and phosphate (ligands dropped), waters dropped. |
| `mmcif/2ds7.cif.gz` | Se-Met monomer with residues 1–11 unmodeled (tokens with masked coordinates); `MSE` → `MET` with `SE` → `SD`. |
| `mmcif/1jb6.cif.gz` | Homodimer with `MSE` and `NLE` (norleucine → `LEU`, two atoms unmatched); chain B unmodeled at 1–2 and 31–33. |
| `mmcif/1l2y.cif.gz` | Trp-cage, solution NMR, 38 models (first model only, no resolution). |
| `ccd_subset.cif` | `_chem_comp` category of 45 components from `components.cif.gz` (wwPDB CCD, CC0 1.0, 2026-10-11): the 20 amino acids, `UNK`, modified residues with parents, multi-parent and D-amino-acid cases, nucleotides, water and ligands. |
```

- [ ] **Step 2: Write the shared conftest**

```python
# tests/fold/data/conftest.py
"""Real structure fixtures for the fold data tests (tests/fixtures/fold/README.md)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from oplm.fold.data.ccd import ChemCompTable, ReferenceConformers

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.fold.data.complex import MolecularComplex

FOLD_FIXTURES = Path(__file__).parents[2] / "fixtures" / "fold"
MMCIF_DIR = FOLD_FIXTURES / "mmcif"  # the archive root for preprocessing tests
FIXTURE_IDS = ("1crn", "1ejg", "4hhb", "2ds7", "1jb6", "1l2y")


@pytest.fixture(scope="session")
def ccd_table() -> ChemCompTable:
    return ChemCompTable.parse(FOLD_FIXTURES / "ccd_subset.cif")


@pytest.fixture(scope="session")
def conformers() -> ReferenceConformers:
    return ReferenceConformers.load()


@pytest.fixture(scope="session")
def parsed(
    ccd_table: ChemCompTable, conformers: ReferenceConformers
) -> Callable[[str], MolecularComplex]:
    """``parsed("1crn")`` -> the parsed fixture complex (cached per session)."""
    from oplm.fold.data.mmcif import Rejection, parse_mmcif

    cache: dict[str, MolecularComplex] = {}

    def _parsed(entry_id: str) -> MolecularComplex:
        if entry_id not in cache:
            result = parse_mmcif(
                MMCIF_DIR / f"{entry_id}.cif.gz", ccd=ccd_table, conformers=conformers
            )
            assert not isinstance(result, Rejection), result
            cache[entry_id] = result
        return cache[entry_id]

    return _parsed
```

- [ ] **Step 3: Write the failing tests**

```python
# tests/fold/data/test_mmcif.py
"""wwPDB mmCIF -> MolecularComplex on real entries (spec §4.2, §4.4; addendum §4)."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import TYPE_CHECKING

import gemmi
import numpy as np
import pytest

from oplm.fold.data.ccd import PROTEIN_RES_TYPES
from oplm.fold.data.mmcif import Rejection, parse_mmcif
from tests.fold.data.conftest import MMCIF_DIR

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.fold.data.ccd import ChemCompTable, ReferenceConformers
    from oplm.fold.data.complex import MolecularComplex

CRAMBIN = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"


def token_atoms(c: MolecularComplex, token: int) -> slice:
    start = int(c.token_atom_start[token])
    return slice(start, start + int(c.token_atom_count[token]))


def test_monomer_1crn(parsed: Callable[[str], MolecularComplex]) -> None:
    c = parsed("1crn")
    c.validate()
    m = c.metadata
    assert m.id == "1crn" and m.source == "pdb" and m.method == "X-RAY DIFFRACTION"
    assert m.resolution == pytest.approx(1.5) and m.release_date == "1981-07-28"
    assert m.dropped == {}
    assert len(c.chains) == 1 and c.chains[0].chain_id == "A" and c.chains[0].auth_chain_id == "A"
    assert c.chains[0].sequence == CRAMBIN and c.chains[0].entity_id == 0 and c.chains[0].sym_id == 0
    assert c.num_tokens == 46 and c.atom_resolved.all() and c.token_mol_type.tolist() == [0] * 46
    assert c.token_residue_index.tolist() == list(range(46))
    assert c.atom_name[:4] == ["N", "CA", "C", "O"] and c.token_center_atom[0] == 1
    assert np.allclose(c.atom_coords[0], [17.047, 14.099, 3.625], atol=1e-3)  # THR 1 N
    assert c.token_rep_atom[0] == 4  # THR CB
    assert c.token_rep_atom[19] == c.token_atom_start[19] + 1  # GLY 20 -> CA
    assert c.token_res_type[19] == PROTEIN_RES_TYPES["GLY"] and c.token_atom_count[19] == 4
    assert c.bonds.shape == (0, 2) and c.atom_bfactor[0] > 0
    assert c.atom_element[c.token_atom_start[2] + 5] == 16  # CYS 3 SG


def test_altlocs_hydrogens_and_microheterogeneity_1ejg(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    c = parsed("1ejg")
    c.validate()
    assert c.chains[0].sequence == CRAMBIN and c.metadata.resolution == pytest.approx(0.54)
    assert not np.any(c.atom_element == 1)  # hydrogens dropped
    # Highest-occupancy altloc per atom: THR 1 N has A (0.82) and B (0.18).
    st = gemmi.read_structure(str(MMCIF_DIR / "1ejg.cif.gz"))
    thr1 = st[0]["A"][0]
    alt_a = next(a for a in thr1 if a.name == "N" and a.altloc == "A")
    assert np.allclose(c.atom_coords[0], [alt_a.pos.x, alt_a.pos.y, alt_a.pos.z], atol=1e-4)
    assert c.atom_bfactor[0] == pytest.approx(alt_a.b_iso)
    # Microheterogeneity keeps the entity sequence's first alternative.
    assert c.token_res_type[21] == PROTEIN_RES_TYPES["PRO"] and c.token_atom_count[21] == 7
    assert c.token_res_type[24] == PROTEIN_RES_TYPES["LEU"] and c.token_atom_count[24] == 8
    assert c.atom_resolved.all()


def test_entities_copies_and_dropped_4hhb(parsed: Callable[[str], MolecularComplex]) -> None:
    c = parsed("4hhb")
    c.validate()
    assert [ch.chain_id for ch in c.chains] == ["A", "B", "C", "D"]
    assert [ch.entity_id for ch in c.chains] == [0, 1, 0, 1]
    assert [ch.sym_id for ch in c.chains] == [0, 0, 1, 1]
    assert [len(ch.sequence) for ch in c.chains] == [141, 146, 141, 146]
    assert c.chains[0].sequence.startswith("VLSPADKTNV") and c.chains[1].sequence.startswith("VHLTPEEKSA")
    assert c.metadata.dropped == {"ligand": 2, "water": 1}  # HEM, PO4 entities; water
    assert c.num_tokens == 574 and c.metadata.resolution == pytest.approx(1.74)
    assert c.chain_tokens(2).tolist() == list(range(287, 428))
    assert np.array_equal(c.token_entity_id[c.chain_tokens(2)], np.zeros(141, dtype=np.int32))
    assert np.array_equal(c.token_sym_id[c.chain_tokens(2)], np.ones(141, dtype=np.int32))


def test_unresolved_residues_and_selenomethionine_2ds7(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    c = parsed("2ds7")
    c.validate()
    assert len(c.chains) == 1 and len(c.chains[0].sequence) == 51 and c.num_tokens == 51
    for token in range(11):  # residues 1-11 are not modeled: tokens exist, coordinates masked
        atoms = token_atoms(c, token)
        assert not c.atom_resolved[atoms].any()
        assert not c.atom_coords[atoms].any() and not c.atom_bfactor[atoms].any()
    assert c.atom_resolved[token_atoms(c, 11)].any()
    mse = 42  # label_seq 43
    assert c.token_res_type[mse] == PROTEIN_RES_TYPES["MET"] and c.chains[0].sequence[mse] == "M"
    names = c.atom_name[token_atoms(c, mse)]
    assert names == ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"] and "SE" not in names
    assert c.atom_resolved[token_atoms(c, mse)].all()  # SE matched SD
    assert c.atom_element[c.token_atom_start[mse] + 6] == 16  # template element (S), not Se


def test_homodimer_norleucine_and_partial_chain_1jb6(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    c = parsed("1jb6")
    c.validate()
    assert [ch.chain_id for ch in c.chains] == ["A", "B"]
    assert [ch.entity_id for ch in c.chains] == [0, 0] and [ch.sym_id for ch in c.chains] == [0, 1]
    assert c.chains[0].sequence == c.chains[1].sequence and len(c.chains[0].sequence) == 33
    # NLE collapses to LEU: CB and CG match by name, CD1/CD2 do not exist in norleucine.
    st = gemmi.read_structure(str(MMCIF_DIR / "1jb6.cif.gz"))
    nle_seq = next(r.label_seq for r in st[0]["A"].subchains()[0] if r.name == "NLE")
    nle = int(np.flatnonzero((c.token_asym_id == 0) & (c.token_residue_index == nle_seq - 1))[0])
    assert c.token_res_type[nle] == PROTEIN_RES_TYPES["LEU"] and c.chains[0].sequence[nle_seq - 1] == "L"
    names = c.atom_name[token_atoms(c, nle)]
    resolved = c.atom_resolved[token_atoms(c, nle)].tolist()
    assert dict(zip(names, resolved, strict=True)) == {
        "N": True, "CA": True, "C": True, "O": True, "CB": True, "CG": True, "CD1": False, "CD2": False,
    }
    b = c.chain_tokens(1)
    unresolved_b = [int(c.token_residue_index[t]) for t in b if not c.atom_resolved[token_atoms(c, t)].any()]
    assert unresolved_b == [0, 1, 30, 31, 32]


def test_nmr_first_model_1l2y(parsed: Callable[[str], MolecularComplex]) -> None:
    c = parsed("1l2y")
    c.validate()
    assert c.metadata.method == "SOLUTION NMR" and c.metadata.resolution is None
    assert c.num_tokens == 20 and c.atom_resolved.all()
    assert c.num_atoms == int(c.token_atom_count.sum())  # one model's worth of atoms


def test_resolution_and_short_chain_filters(
    ccd_table: ChemCompTable, conformers: ReferenceConformers
) -> None:
    hhb = MMCIF_DIR / "4hhb.cif.gz"
    rej = parse_mmcif(hhb, ccd=ccd_table, conformers=conformers, max_resolution=1.0)
    assert isinstance(rej, Rejection) and rej.id == "4hhb" and rej.reason == "resolution"
    c = parse_mmcif(
        MMCIF_DIR / "1jb6.cif.gz", ccd=ccd_table, conformers=conformers, min_resolved_residues=30
    )
    assert not isinstance(c, Rejection)
    assert [ch.chain_id for ch in c.chains] == ["A"] and c.metadata.dropped == {"short_chain": 1}
    rej = parse_mmcif(
        MMCIF_DIR / "1crn.cif.gz", ccd=ccd_table, conformers=conformers, min_resolved_residues=1000
    )
    assert isinstance(rej, Rejection) and rej.reason == "no_resolved_chain"


def test_no_protein_chains_and_parse_error(
    tmp_path: Path, ccd_table: ChemCompTable, conformers: ReferenceConformers
) -> None:
    st = gemmi.read_structure(str(MMCIF_DIR / "1crn.cif.gz"))
    st.setup_entities()
    st.entities[0].polymer_type = gemmi.PolymerType.Dna
    dna = tmp_path / "dna.cif"
    st.make_mmcif_document().write_file(str(dna))
    rej = parse_mmcif(dna, ccd=ccd_table, conformers=conformers)
    assert isinstance(rej, Rejection) and rej.reason == "no_protein_chains"
    bad = tmp_path / "bad.cif.gz"
    with gzip.open(bad, "wt") as fh:
        fh.write("data_bad\n_entry.id bad\nloop_ broken\n")
    rej = parse_mmcif(bad, ccd=ccd_table, conformers=conformers)
    assert isinstance(rej, Rejection) and rej.reason == "parse_error" and rej.id == "bad.cif.gz"
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/data/test_mmcif.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.fold.data.mmcif'`.

- [ ] **Step 5: Implement the parser**

```python
# src/oplm/fold/data/mmcif.py
"""wwPDB mmCIF -> :class:`MolecularComplex` with the §4.4 filters applied at parse time.

gemmi reads the (gzipped) file. Protein polymer chains of the asymmetric unit are kept;
waters, ligands, nucleic-acid and other polymer chains are dropped and counted in
``ComplexMetadata.dropped``. The token sequence is the entity polymer sequence, so unresolved
residues are tokens with ``atom_resolved`` false. First model only; per residue the
highest-occupancy altloc per atom name (ties keep the first); hydrogens dropped;
microheterogeneity keeps the entity sequence's first alternative. Non-standard residues
collapse to their CCD parent's token and atom set, matched by atom name, with AlphaFold's
selenomethionine convention (``MSE`` ``SE`` -> ``SD``); parent atoms absent from the modified
residue are unresolved. Bioassembly expansion is deferred (spec §4.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import gemmi
import numpy as np

from oplm.fold.data.ccd import (
    MOL_TYPE_PROTEIN,
    PROTEIN_3TO1,
    PROTEIN_RES_TYPES,
    ReferenceConformers,
)
from oplm.fold.data.complex import ChainInfo, ComplexMetadata, MolecularComplex

if TYPE_CHECKING:
    from oplm.fold.data.ccd import ChemCompTable

__all__ = ["MAX_RESOLUTION", "MIN_RESOLVED_RESIDUES", "PARSE_VERSION", "Rejection", "parse_mmcif"]

PARSE_VERSION = 1  # bump when the parse changes what a shard row contains
MAX_RESOLUTION = 9.0  # Å, spec §4.4
MIN_RESOLVED_RESIDUES = 4  # spec §4.4
_ATOM_NAME_REMAP = {("MSE", "SE"): "SD"}  # AlphaFold's selenomethionine convention
_DROPPED_KIND = {
    gemmi.EntityType.Water: "water",
    gemmi.EntityType.NonPolymer: "ligand",
    gemmi.EntityType.Branched: "ligand",
    gemmi.EntityType.Unknown: "unknown",
}
_NUCLEIC = {gemmi.PolymerType.Dna, gemmi.PolymerType.Rna, gemmi.PolymerType.DnaRnaHybrid}


@dataclass(frozen=True)
class Rejection:
    """Why an entry produced no complex; ``reason`` is counted by ``oplm fold preprocess``."""

    id: str
    reason: str  # parse_error | resolution | no_protein_chains | no_resolved_chain
    detail: str = ""


def parse_mmcif(
    path: Path,
    *,
    ccd: ChemCompTable,
    conformers: ReferenceConformers | None = None,
    max_resolution: float = MAX_RESOLUTION,
    min_resolved_residues: int = MIN_RESOLVED_RESIDUES,
) -> MolecularComplex | Rejection:
    """Parse one entry.

    Args:
        path: ``.cif`` or ``.cif.gz`` file.
        ccd: Chemical component table for the parent mapping of non-standard residues.
        conformers: Residue templates (atom order and elements); the packaged table when ``None``.
        max_resolution: Reject entries with a worse (larger) resolution; NMR has none and passes.
        min_resolved_residues: Drop chains with fewer residues that have at least one atom.

    Returns:
        The complex, or a :class:`Rejection` naming the reason.
    """
    table = conformers or ReferenceConformers.load()
    try:
        block = gemmi.cif.read(str(path)).sole_block()
        st = gemmi.make_structure_from_block(block)
        st.setup_entities()
        st.remove_hydrogens()
    except Exception as exc:  # gemmi raises RuntimeError/ValueError; nothing else to do with them
        return Rejection(Path(path).name, "parse_error", str(exc))

    entry_id = (st.name or block.name).lower()
    method = st.info["_exptl.method"] if "_exptl.method" in st.info else "UNKNOWN"
    resolution = float(st.resolution) if st.resolution > 0 else None
    if resolution is not None and resolution > max_resolution:
        return Rejection(entry_id, "resolution", f"{resolution} > {max_resolution}")
    dates = block.find_values("_pdbx_audit_revision_history.revision_date")
    release_date = gemmi.cif.as_string(dates[0]) if len(dates) else None

    dropped: dict[str, int] = {}

    def drop(kind: str) -> None:
        dropped[kind] = dropped.get(kind, 0) + 1

    # Protein entities: entity name -> (entity index, parent codes, raw monomer list)
    protein: dict[str, tuple[int, list[str], list[str]]] = {}
    for ent in st.entities:
        if ent.entity_type != gemmi.EntityType.Polymer:
            drop(_DROPPED_KIND.get(ent.entity_type, "unknown"))
            continue
        if ent.polymer_type != gemmi.PolymerType.PeptideL:
            drop("nucleic_acid" if ent.polymer_type in _NUCLEIC else "other_polymer")
            continue
        monomers = list(ent.full_sequence)
        if not monomers:
            drop("empty_sequence")
            continue
        codes = [ccd.protein_parent(gemmi.Entity.first_mon(m)) for m in monomers]
        protein[ent.name] = (len(protein), codes, monomers)
    if not protein:
        return Rejection(entry_id, "no_protein_chains")

    chains: list[ChainInfo] = []
    chain_entity: list[str] = []
    chain_residues: list[dict[int, gemmi.Residue]] = []
    sym_seen: dict[int, int] = {}
    for chain in st[0]:  # first model
        for sub in chain.subchains():
            first = sub[0]
            if first.entity_type != gemmi.EntityType.Polymer or first.entity_id not in protein:
                continue
            ent_idx, codes, monomers = protein[first.entity_id]
            by_seq: dict[int, gemmi.Residue] = {}
            for res in sub:
                seq = res.label_seq
                if seq is None or not 1 <= seq <= len(codes):
                    continue
                want = gemmi.Entity.first_mon(monomers[seq - 1])
                if seq not in by_seq or (res.name == want and by_seq[seq].name != want):
                    by_seq[seq] = res  # microheterogeneity: keep the sequence's first alternative
            if sum(1 for r in by_seq.values() if len(r) > 0) < min_resolved_residues:
                drop("short_chain")
                continue
            sym = sym_seen.get(ent_idx, 0)
            sym_seen[ent_idx] = sym + 1
            chains.append(
                ChainInfo(
                    chain_id=sub.subchain_id(),
                    auth_chain_id=chain.name,
                    entity_id=ent_idx,
                    sym_id=sym,
                    mol_type=MOL_TYPE_PROTEIN,
                    sequence="".join(PROTEIN_3TO1[c] for c in codes),
                )
            )
            chain_entity.append(first.entity_id)
            chain_residues.append(by_seq)
    if not chains:
        return Rejection(entry_id, "no_resolved_chain")

    coords: list[tuple[float, float, float]] = []
    resolved: list[bool] = []
    elements: list[int] = []
    names: list[str] = []
    charges: list[int] = []
    bfactors: list[float] = []
    atom_tok: list[int] = []
    tok: dict[str, list[int]] = {k: [] for k in ("res_type", "asym", "entity", "sym", "residue", "center", "rep", "start", "count")}
    token = 0
    for asym, (info, ent_name, by_seq) in enumerate(zip(chains, chain_entity, chain_residues, strict=True)):
        codes = protein[ent_name][1]
        for pos, code in enumerate(codes):
            template = table[code]
            best: dict[str, gemmi.Atom] = {}
            res = by_seq.get(pos + 1)
            if res is not None:
                for atom in res:
                    name = _ATOM_NAME_REMAP.get((res.name, atom.name), atom.name)
                    current = best.get(name)
                    if current is None or atom.occ > current.occ:
                        best[name] = atom  # highest occupancy; ties keep the first
            start = len(names)
            for name, element, charge in zip(template.atoms, template.elements, template.charges, strict=True):
                atom = best.get(name)
                if atom is None:
                    coords.append((0.0, 0.0, 0.0))
                    resolved.append(False)
                    bfactors.append(0.0)
                else:
                    coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
                    resolved.append(True)
                    bfactors.append(float(atom.b_iso))
                elements.append(element)
                names.append(name)
                charges.append(charge)
                atom_tok.append(token)
            tok["res_type"].append(PROTEIN_RES_TYPES[code])
            tok["asym"].append(asym)
            tok["entity"].append(info.entity_id)
            tok["sym"].append(info.sym_id)
            tok["residue"].append(pos)
            tok["center"].append(start + template.atoms.index("CA"))
            tok["rep"].append(start + template.representative_atom)
            tok["start"].append(start)
            tok["count"].append(len(template.atoms))
            token += 1

    def i32(values: list[int]) -> np.ndarray:
        return np.asarray(values, dtype=np.int32)

    c = MolecularComplex(
        atom_coords=np.asarray(coords, dtype=np.float32).reshape(-1, 3),
        atom_resolved=np.asarray(resolved, dtype=bool),
        atom_element=np.asarray(elements, dtype=np.uint8),
        atom_name=names,
        atom_charge=np.asarray(charges, dtype=np.int8),
        atom_bfactor=np.asarray(bfactors, dtype=np.float32),
        atom_to_token=i32(atom_tok),
        token_res_type=np.asarray(tok["res_type"], dtype=np.int8),
        token_mol_type=np.full(token, MOL_TYPE_PROTEIN, dtype=np.int8),
        token_asym_id=i32(tok["asym"]),
        token_entity_id=i32(tok["entity"]),
        token_sym_id=i32(tok["sym"]),
        token_residue_index=i32(tok["residue"]),
        token_center_atom=i32(tok["center"]),
        token_rep_atom=i32(tok["rep"]),
        token_atom_start=i32(tok["start"]),
        token_atom_count=i32(tok["count"]),
        bonds=np.zeros((0, 2), dtype=np.int32),
        chains=chains,
        metadata=ComplexMetadata(
            id=entry_id,
            source="pdb",
            method=str(method),
            resolution=resolution,
            release_date=release_date,
            dropped=dropped,
        ),
    )
    c.validate()
    return c
```

The `Exception` catch is deliberate: gemmi raises `RuntimeError` for unreadable files and
`ValueError` for malformed CIF, and either means the same thing here (a rejection carrying the
message).

- [ ] **Step 6: Run the tests, lint, types**

Run: `.venv/bin/python -m pytest tests/fold/data/test_mmcif.py -q && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/data/mmcif.py tests/fold/data/test_mmcif.py tests/fold/data/conftest.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: all pass. If `ty` cannot resolve gemmi attributes (`gemmi` ships no stubs), add one `# ty: ignore[unresolved-attribute]` per flagged line with the reason "gemmi has no type stubs", not a module-wide ignore.

- [ ] **Step 7: Commit**

```bash
git add src/oplm/fold/data/mmcif.py tests/fold/data/conftest.py tests/fold/data/test_mmcif.py tests/fixtures/fold/
git commit -m "feat(fold): gemmi mmCIF parser to MolecularComplex with the §4.4 filters and real fixtures"
```

---

### Task 4: Contiguous and spatial crops

**Files:**
- Create: `src/oplm/fold/data/crop.py`
- Test: `tests/fold/data/test_crop.py`

**Interfaces:**
- Consumes: `MolecularComplex.chain_tokens`, `.select_tokens`, atom arrays (Task 2).
- Produces: `contiguous_crop(n, crop_size, generator) -> np.ndarray`, `spatial_crop(centers, valid, crop_size, generator) -> np.ndarray` (sorted token positions), `crop_chain(complex, chain, crop_size, *, contiguous_prob, generator) -> MolecularComplex` (a one-chain complex). Task 8 calls `crop_chain` per example.

- [ ] **Step 1: Write the failing tests**

```python
# tests/fold/data/test_crop.py
"""Single-chain contiguous and spatial crops (spec §7.3; addendum §2, §4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from oplm.fold.data.crop import contiguous_crop, crop_chain, spatial_crop

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.fold.data.complex import MolecularComplex


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_contiguous_crop_is_a_window_or_everything() -> None:
    assert contiguous_crop(10, 16, gen(0)).tolist() == list(range(10))
    starts = set()
    for seed in range(40):
        crop = contiguous_crop(46, 16, gen(seed))
        assert crop.shape == (16,) and np.array_equal(crop, np.arange(crop[0], crop[0] + 16))
        starts.add(int(crop[0]))
    assert len(starts) > 5 and min(starts) >= 0 and max(starts) <= 30


def test_spatial_crop_takes_the_nearest_tokens_to_a_resolved_center(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    c = parsed("1crn")
    tokens = c.chain_tokens(0)
    centers = c.atom_coords[c.token_center_atom[tokens]]
    valid = c.atom_resolved[c.token_center_atom[tokens]]
    crop = spatial_crop(centers, valid, 16, gen(3))
    assert crop.shape == (16,) and np.all(np.diff(crop) > 0)
    # Replay the draw to learn the center, then check the 16 nearest by CA distance.
    g = gen(3)
    pick = np.flatnonzero(valid)[int(torch.randint(0, int(valid.sum()), (1,), generator=g))]
    dist = np.linalg.norm(centers - centers[pick], axis=1)
    nearest = np.sort(np.lexsort((np.arange(len(dist)), dist))[:16])
    assert np.array_equal(crop, nearest) and pick in crop
    assert np.array_equal(crop, spatial_crop(centers, valid, 16, gen(3)))  # deterministic
    assert spatial_crop(centers, valid, 100, gen(0)).tolist() == list(range(46))


def test_spatial_crop_never_centers_on_unresolved_tokens(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    """2ds7: tokens 0-10 have no atoms; with 40 resolved tokens a crop of 8 is all resolved."""
    c = parsed("2ds7")
    tokens = c.chain_tokens(0)
    centers = c.atom_coords[c.token_center_atom[tokens]]
    valid = c.atom_resolved[c.token_center_atom[tokens]]
    assert valid.sum() == 40
    for seed in range(50):
        crop = spatial_crop(centers, valid, 8, gen(seed))
        assert valid[crop].all()


def test_spatial_crop_fills_with_unresolved_tokens_by_index(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    """Fewer resolved tokens than the crop: every resolved token, then unresolved by index."""
    c = parsed("2ds7")
    tokens = c.chain_tokens(0)
    centers = c.atom_coords[c.token_center_atom[tokens]]
    valid = c.atom_resolved[c.token_center_atom[tokens]]
    crop = spatial_crop(centers, valid, 48, gen(1))
    assert crop.shape == (48,)
    assert set(np.flatnonzero(valid)) <= set(crop.tolist())
    assert sorted(set(crop.tolist()) - set(np.flatnonzero(valid).tolist())) == list(range(8))


def test_spatial_crop_with_no_resolved_center_falls_back_to_contiguous() -> None:
    centers = np.zeros((20, 3), dtype=np.float32)
    valid = np.zeros(20, dtype=bool)
    crop = spatial_crop(centers, valid, 8, gen(0))
    assert np.array_equal(crop, contiguous_crop(20, 8, gen(0)))


@pytest.mark.parametrize("contiguous_prob", [0.0, 1.0])
def test_crop_chain_returns_a_one_chain_complex(
    parsed: Callable[[str], MolecularComplex], contiguous_prob: float
) -> None:
    c = parsed("4hhb")
    sub = crop_chain(c, 2, 16, contiguous_prob=contiguous_prob, generator=gen(5))
    sub.validate()
    assert sub.num_tokens == 16 and len(sub.chains) == 1 and sub.chains[0].chain_id == "C"
    assert sub.token_asym_id.tolist() == [0] * 16
    assert sub.token_entity_id.tolist() == [0] * 16 and sub.token_sym_id.tolist() == [1] * 16
    assert sub.chains[0].sequence == c.chains[2].sequence  # the full chain sequence survives
    residues = sub.token_residue_index
    if contiguous_prob == 1.0:
        assert np.array_equal(residues, np.arange(residues[0], residues[0] + 16))
    again = crop_chain(c, 2, 16, contiguous_prob=contiguous_prob, generator=gen(5))
    assert np.array_equal(again.token_residue_index, residues)
    other = crop_chain(c, 2, 16, contiguous_prob=contiguous_prob, generator=gen(6))
    assert not np.array_equal(other.token_residue_index, residues)


def test_crop_chain_short_chain_is_whole(parsed: Callable[[str], MolecularComplex]) -> None:
    c = parsed("1l2y")
    sub = crop_chain(c, 0, 384, contiguous_prob=0.5, generator=gen(0))
    assert sub.num_tokens == 20 and sub.num_atoms == c.num_atoms
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/data/test_crop.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.fold.data.crop'`.

- [ ] **Step 3: Implement the crops**

```python
# src/oplm/fold/data/crop.py
"""Training crops of one chain (spec §7.3, addendum §2: single-chain examples).

``contiguous_crop`` is a random window; ``spatial_crop`` picks a random resolved token and
keeps the ``crop_size`` tokens nearest to it by center-atom distance (unresolved tokens are
infinitely far, so they fill a crop only after every resolved token, by index). Interface crops
arrive with multi-chain training (milestone 3). Randomness comes from the caller's
``torch.Generator`` so a resumed run restores the same crop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from oplm.fold.data.complex import MolecularComplex

__all__ = ["contiguous_crop", "crop_chain", "spatial_crop"]


def contiguous_crop(n: int, crop_size: int, generator: torch.Generator) -> np.ndarray:
    """Positions of a random ``crop_size`` window of ``range(n)``, or all of it when it fits."""
    if n <= crop_size:
        return np.arange(n)
    start = int(torch.randint(0, n - crop_size + 1, (1,), generator=generator))
    return np.arange(start, start + crop_size)


def spatial_crop(
    centers: np.ndarray, valid: np.ndarray, crop_size: int, generator: torch.Generator
) -> np.ndarray:
    """Sorted positions of the ``crop_size`` tokens nearest a random resolved center.

    Args:
        centers: ``(n, 3)`` center-atom coordinates.
        valid: ``(n,)`` whether each center atom is resolved.
        crop_size: Tokens to keep; everything when ``n <= crop_size``.
        generator: Draws the center (and the window when no center is resolved).
    """
    n = centers.shape[0]
    if n <= crop_size:
        return np.arange(n)
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        return contiguous_crop(n, crop_size, generator)
    pick = candidates[int(torch.randint(0, candidates.size, (1,), generator=generator))]
    dist = np.linalg.norm(centers - centers[pick], axis=1)
    dist = np.where(valid, dist, np.inf)
    order = np.lexsort((np.arange(n), dist))  # distance, then index
    return np.sort(order[:crop_size])


def crop_chain(
    complex: MolecularComplex,
    chain: int,
    crop_size: int,
    *,
    contiguous_prob: float,
    generator: torch.Generator,
) -> MolecularComplex:
    """Crop chain ``chain`` of ``complex`` to a one-chain complex of at most ``crop_size`` tokens.

    Draws the crop type first (contiguous with probability ``contiguous_prob``, else spatial)
    so the generator sequence is fixed per example.
    """
    tokens = complex.chain_tokens(chain)
    center_atoms = complex.token_center_atom[tokens]
    use_contiguous = bool(torch.rand((), generator=generator) < contiguous_prob)
    if use_contiguous:
        local = contiguous_crop(tokens.size, crop_size, generator)
    else:
        local = spatial_crop(
            complex.atom_coords[center_atoms], complex.atom_resolved[center_atoms], crop_size, generator
        )
    return complex.select_tokens(tokens[local])
```

- [ ] **Step 4: Run the tests, lint, types**

Run: `.venv/bin/python -m pytest tests/fold/data/test_crop.py -q && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/data/crop.py tests/fold/data/test_crop.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/oplm/fold/data/crop.py tests/fold/data/test_crop.py
git commit -m "feat(fold): contiguous and spatial single-chain crops"
```

---

### Task 5: Training featurization with targets and conformer rotations

**Files:**
- Modify: `src/oplm/fold/data/featurize.py`
- Test: `tests/fold/data/test_featurize.py` (append)

**Interfaces:**
- Consumes: `MolecularComplex` (Task 2), `RES_TYPE_TO_CODE` (Task 1), `ReferenceConformers`, `encode_atom_name`, `VOCAB` (existing).
- Produces: `FoldTargets(coords, resolved, bfactor, frame_atom_idx)` with `.to(device)`, `FoldExample(id, features, targets)`, `featurize_complex(complex, *, conformers=None, pad_tokens_to=None, atom_pad_multiple=32, generator=None) -> FoldExample`. The existing `featurize(chains)` keeps its signature and behaviour (M1 parity tests stay green). Task 6 batches `FoldExample`s; Task 8 produces them; M2b's losses read `FoldTargets`.

Contract for M2b: `targets.coords[b, a]` is meaningful only where `targets.resolved[b, a]`; `frame_atom_idx[b, l]` holds the flat indices of N, CA, C of token `l` (0 on padded tokens); a frame is usable only where all three are resolved.

- [ ] **Step 1: Append the failing tests**

```python
# tests/fold/data/test_featurize.py (append)
import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from oplm.fold.data.crop import crop_chain
from oplm.fold.data.featurize import FoldExample, FoldTargets, featurize_complex

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.fold.data.complex import MolecularComplex

CRAMBIN = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"


def test_full_complex_features_match_sequence_featurize(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    example = featurize_complex(parsed("1crn"))
    assert isinstance(example, FoldExample) and example.id == "1crn_A"
    expected = featurize([ChainSpec(CRAMBIN, "A")])
    for f in dataclasses.fields(expected):
        a, b = getattr(example.features, f.name), getattr(expected, f.name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), f.name
        else:
            assert a == b, f.name


def test_targets_line_up_with_the_atom_axis(parsed: Callable[[str], MolecularComplex]) -> None:
    c = parsed("2ds7")
    ex = featurize_complex(c)
    t = ex.targets
    n, A = c.num_atoms, ex.features.num_atoms
    assert isinstance(t, FoldTargets)
    assert t.coords.shape == (1, A, 3) and t.resolved.shape == (1, A) and t.bfactor.shape == (1, A)
    assert torch.equal(t.coords[0, :n], torch.from_numpy(c.atom_coords))
    assert torch.equal(t.resolved[0, :n], torch.from_numpy(c.atom_resolved))
    assert torch.equal(t.bfactor[0, :n], torch.from_numpy(c.atom_bfactor))
    assert not t.coords[0, n:].any() and not t.resolved[0, n:].any()
    assert t.frame_atom_idx.shape == (1, c.num_tokens, 3)
    starts = torch.from_numpy(c.token_atom_start).long()
    assert torch.equal(t.frame_atom_idx[0], torch.stack([starts, starts + 1, starts + 2], dim=1))
    assert not t.resolved[0, t.frame_atom_idx[0, 0]].any()  # unmodeled residue: no frame
    assert t.resolved[0, t.frame_atom_idx[0, 20]].all()
    moved = t.to("cpu")
    assert moved.coords.device.type == "cpu"


def test_conformer_rotations_are_per_token_rigid_and_seeded(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    c = parsed("1crn")
    raw = featurize_complex(c).features.ref_pos[0]
    g = torch.Generator().manual_seed(7)
    rot = featurize_complex(c, generator=g).features.ref_pos[0]
    assert not torch.allclose(raw, rot)
    for token in (0, 1, 19):
        start, count = int(c.token_atom_start[token]), int(c.token_atom_count[token])
        a, b = raw[start : start + count], rot[start : start + count]
        assert torch.allclose(torch.cdist(a, a), torch.cdist(b, b), atol=1e-4)  # rigid
        assert torch.allclose(b.mean(0), torch.zeros(3), atol=1e-4)  # centered
    thr1, thr2 = rot[:7], rot[7:14]  # two THR tokens get different rotations
    assert not torch.allclose(thr1, thr2, atol=1e-3)
    again = featurize_complex(c, generator=torch.Generator().manual_seed(7)).features.ref_pos[0]
    assert torch.equal(rot, again)
    assert torch.equal(featurize_complex(c).features.ref_pos[0], raw)  # no generator: raw


def test_cropped_chain_keeps_the_full_lm_row(parsed: Callable[[str], MolecularComplex]) -> None:
    c = parsed("2ds7")
    sub = crop_chain(c, 0, 16, contiguous_prob=1.0, generator=torch.Generator().manual_seed(0))
    ex = featurize_complex(sub)
    f = ex.features
    assert f.num_tokens == 16 and f.num_chains == 1 and f.chain_ids == ["A"]
    assert f.lm_input_ids.shape == (1, 53)  # BOS + 51 residues + EOS
    assert f.lm_input_ids[0, 1 + 42] == VOCAB["M"]  # MSE position reads as M
    assert torch.equal(f.lm_positions, f.residue_index + 1)
    assert f.asym_id.eq(0).all() and f.lm_rows.eq(0).all()
    assert ex.id == "2ds7_A"


def test_padding_and_multi_chain_ids(parsed: Callable[[str], MolecularComplex]) -> None:
    ex = featurize_complex(parsed("1jb6"), pad_tokens_to=128)
    assert ex.features.num_tokens == 128 and ex.features.token_mask[0].sum() == 66
    assert ex.features.lm_input_ids.shape == (2, 35) and ex.features.lm_rows[0, 33] == 1
    assert ex.targets.frame_atom_idx.shape == (1, 128, 3) and not ex.targets.frame_atom_idx[0, 66:].any()
    assert ex.id == "1jb6_A_B"


def test_atom_names_must_match_the_template(parsed: Callable[[str], MolecularComplex]) -> None:
    c = dataclasses.replace(parsed("1crn"))
    c.atom_name = list(c.atom_name)
    c.atom_name[5] = "ZZ"
    with pytest.raises(ValueError, match="template"):
        featurize_complex(c)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/data/test_featurize.py -q`
Expected: FAIL with `ImportError: cannot import name 'FoldExample'`.

- [ ] **Step 3: Rewrite `featurize.py` around a shared assembly step**

Replace the file with the following. `featurize` keeps every line of its current behaviour
(the loop is unchanged; only the tensor assembly moved into `_assemble`).

```python
# src/oplm/fold/data/featurize.py
"""Sequences or complexes -> model inputs (protein-only data on the general representation).

Mirrors ESMFold2's builder path (esm/models/esmfold2/prepare_input.py, Apache-2.0; see
THIRD_PARTY_NOTICES.md): one token per residue, flat ragged atoms padded to a multiple of 32,
``ref_*`` atom features from the reference conformer table, AF3-style chain/entity/symmetry
indices, representative (CB, else CA) atom per token, and the empty token-bond matrix (always
present). Modifications: LM inputs are one row per chain with BOS/EOS (spec §4.5) instead of one
packed row per complex; optional token padding to a crop multiple. Reproduced quirk: an unknown
residue (res_type 22) gets peptide bonds to its chain neighbours in ``token_bonds``, as upstream's
atom-tokenized-residue logic does (prepare_input.py:1075-1110).

``featurize`` (sequences, prediction) and ``featurize_complex`` (a parsed, possibly cropped
``MolecularComplex``, training) share ``_assemble``. The training path also returns
``FoldTargets`` and, given a generator, applies spec §4.3's fresh per-token rotation of the
centered reference conformers; prediction keeps the raw conformers (M1 parity).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

import torch

from oplm.fold.data.ccd import (
    MOL_TYPE_PROTEIN,
    PROTEIN_1TO3,
    PROTEIN_RES_TYPES,
    RES_TYPE_TO_CODE,
    UNK_RES_TYPE,
    ReferenceConformers,
    encode_atom_name,
)
from oplm.model.tokenization_oplm import VOCAB

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from oplm.fold.data.complex import MolecularComplex

__all__ = ["ChainSpec", "FoldExample", "FoldFeatures", "FoldTargets", "featurize", "featurize_complex"]

_BOS, _EOS, _PAD, _X = VOCAB["<cls>"], VOCAB["<eos>"], VOCAB["<pad>"], VOCAB["X"]


@dataclass(frozen=True)
class ChainSpec:
    """One protein chain and how many identical copies of it the complex contains."""

    sequence: str
    chain_id: str
    copies: int = 1


@dataclass
class FoldFeatures:
    """Batched model inputs: token tensors ``(B, L)``, atom tensors ``(B, A)``.

    ``featurize``/``featurize_complex`` produce ``B == 1``; ``collate_examples`` stacks examples.
    LM rows are shared across the batch: ``lm_input_ids`` holds every chain of every example and
    ``lm_rows`` indexes into that stack.
    """

    token_index: Tensor
    residue_index: Tensor
    asym_id: Tensor
    entity_id: Tensor
    sym_id: Tensor
    mol_type: Tensor
    res_type: Tensor
    token_mask: Tensor  # bool
    token_bonds: Tensor  # (B, L, L, 1) float
    distogram_atom_idx: Tensor  # (B, L) flat atom index of the representative atom
    ref_pos: Tensor  # (B, A, 3)
    ref_element: Tensor  # (B, A) atomic number, 0 = pad
    ref_charge: Tensor  # (B, A) float formal charge
    ref_atom_name_chars: Tensor  # (B, A, 4) long
    ref_space_uid: Tensor  # (B, A)
    atom_mask: Tensor  # (B, A) bool
    atom_to_token: Tensor  # (B, A) long, 0 on pads
    lm_input_ids: Tensor  # (C, T) one row per chain, BOS ... EOS, right-padded
    lm_attention_mask: Tensor  # (C, T)
    lm_rows: Tensor  # (B, L) row of each token in ``lm_input_ids`` (0 on pads)
    lm_positions: Tensor  # (B, L) position of each token inside its LM row (0 on pads)
    chain_ids: list[str]

    @property
    def num_tokens(self) -> int:
        return int(self.token_index.shape[1])

    @property
    def num_atoms(self) -> int:
        return int(self.atom_mask.shape[1])

    @property
    def num_chains(self) -> int:
        return int(self.lm_input_ids.shape[0])

    def to(self, device: torch.device | str) -> FoldFeatures:
        """Return a copy with every tensor on ``device``."""
        moved = {}
        for f in fields(self):
            value = getattr(self, f.name)
            moved[f.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return FoldFeatures(**moved)


@dataclass
class FoldTargets:
    """Supervision aligned with :class:`FoldFeatures`' atom and token axes.

    ``coords`` is meaningful only where ``resolved``; ``frame_atom_idx`` holds the flat N, CA, C
    atom indices of each token (0 on padded tokens), usable as a frame only where all three
    atoms are resolved.
    """

    coords: Tensor  # (B, A, 3) fp32 Å
    resolved: Tensor  # (B, A) bool
    bfactor: Tensor  # (B, A) fp32
    frame_atom_idx: Tensor  # (B, L, 3) long

    def to(self, device: torch.device | str) -> FoldTargets:
        """Return a copy with every tensor on ``device``."""
        return FoldTargets(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


@dataclass
class FoldExample:
    """One training example: ``<entry id>_<chain ids>``, its features and its targets."""

    id: str
    features: FoldFeatures
    targets: FoldTargets


def _expand_chains(chains: Sequence[ChainSpec]) -> list[tuple[str, str, int, int]]:
    """(chain_id, sequence, entity_id, sym_id) per chain copy; entities by first-seen sequence."""
    entities: dict[str, int] = {}
    copies_seen: dict[int, int] = {}
    seen_ids: set[str] = set()
    out = []
    for spec in chains:
        if spec.copies < 1:
            raise ValueError(f"copies must be >= 1 for chain {spec.chain_id!r}")
        entity = entities.setdefault(spec.sequence, len(entities))
        for k in range(spec.copies):
            sym = copies_seen.get(entity, 0)
            copies_seen[entity] = sym + 1
            chain_id = spec.chain_id if k == 0 else f"{spec.chain_id}_{k + 1}"
            if chain_id in seen_ids:
                raise ValueError(f"duplicate chain id {chain_id!r}")
            seen_ids.add(chain_id)
            out.append((chain_id, spec.sequence, entity, sym))
    return out


def _lm_row(sequence: str) -> list[int]:
    return [_BOS] + [VOCAB.get(c, _X) if c in PROTEIN_1TO3 else _X for c in sequence] + [_EOS]


def _random_rotation(generator: torch.Generator) -> Tensor:
    """A uniformly random proper rotation (QR of a Gaussian matrix with sign fix)."""
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q = q.clone()
        q[:, 0] = -q[:, 0]
    return q


@dataclass
class _Lists:
    """Per-token and per-atom Python lists accumulated before tensor assembly."""

    tok_residue: list[int] = field(default_factory=list)
    tok_asym: list[int] = field(default_factory=list)
    tok_entity: list[int] = field(default_factory=list)
    tok_sym: list[int] = field(default_factory=list)
    tok_res_type: list[int] = field(default_factory=list)
    tok_rep_atom: list[int] = field(default_factory=list)
    tok_lm_row: list[int] = field(default_factory=list)
    tok_lm_pos: list[int] = field(default_factory=list)
    atom_pos: list[Tensor] = field(default_factory=list)
    atom_elem: list[int] = field(default_factory=list)
    atom_charge: list[float] = field(default_factory=list)
    atom_chars: list[tuple[int, int, int, int]] = field(default_factory=list)
    atom_uid: list[int] = field(default_factory=list)
    atom_tok: list[int] = field(default_factory=list)
    lm_rows: list[list[int]] = field(default_factory=list)
    chain_ids: list[str] = field(default_factory=list)


def _assemble(
    lists: _Lists, *, pad_tokens_to: int | None, atom_pad_multiple: int
) -> FoldFeatures:
    """Pad and tensorize the accumulated lists (batch size 1)."""
    n_tokens = len(lists.tok_residue)
    L = n_tokens if pad_tokens_to is None else pad_tokens_to
    if n_tokens > L:
        raise ValueError(f"pad_tokens_to={pad_tokens_to} < token count {n_tokens}")
    n_atoms = len(lists.atom_pos)
    A = -(-max(n_atoms, 1) // atom_pad_multiple) * atom_pad_multiple

    def tok(values: list[int], dtype: torch.dtype = torch.long) -> Tensor:
        t = torch.zeros(1, L, dtype=dtype)
        t[0, :n_tokens] = torch.tensor(values, dtype=dtype)
        return t

    token_bonds = torch.zeros(1, L, L, 1)
    for i in range(n_tokens):  # the upstream unknown-residue bond quirk (same-chain neighbours)
        if lists.tok_res_type[i] != UNK_RES_TYPE:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < n_tokens and lists.tok_asym[j] == lists.tok_asym[i]:
                token_bonds[0, i, j, 0] = token_bonds[0, j, i, 0] = 1.0

    T = max(len(r) for r in lists.lm_rows)
    lm_ids = torch.full((len(lists.lm_rows), T), _PAD, dtype=torch.long)
    lm_mask = torch.zeros(len(lists.lm_rows), T, dtype=torch.long)
    for r, row in enumerate(lists.lm_rows):
        lm_ids[r, : len(row)] = torch.tensor(row)
        lm_mask[r, : len(row)] = 1

    def atoms(values: list, dtype: torch.dtype, trailing: tuple[int, ...] = ()) -> Tensor:
        t = torch.zeros(1, A, *trailing, dtype=dtype)
        if n_atoms:
            if isinstance(values[0], torch.Tensor):
                t[0, :n_atoms] = torch.stack(values)
            else:
                t[0, :n_atoms] = torch.tensor(values, dtype=dtype)
        return t

    token_mask = torch.zeros(1, L, dtype=torch.bool)
    token_mask[0, :n_tokens] = True
    atom_mask = torch.zeros(1, A, dtype=torch.bool)
    atom_mask[0, :n_atoms] = True
    return FoldFeatures(
        token_index=tok(list(range(n_tokens))),
        residue_index=tok(lists.tok_residue),
        asym_id=tok(lists.tok_asym),
        entity_id=tok(lists.tok_entity),
        sym_id=tok(lists.tok_sym),
        mol_type=tok([MOL_TYPE_PROTEIN] * n_tokens),
        res_type=tok(lists.tok_res_type),
        token_mask=token_mask,
        token_bonds=token_bonds,
        distogram_atom_idx=tok(lists.tok_rep_atom),
        ref_pos=atoms(lists.atom_pos, torch.float32, (3,)),
        ref_element=atoms(lists.atom_elem, torch.long),
        ref_charge=atoms(lists.atom_charge, torch.float32),
        ref_atom_name_chars=atoms([list(c) for c in lists.atom_chars], torch.long, (4,)),
        ref_space_uid=atoms(lists.atom_uid, torch.long),
        atom_mask=atom_mask,
        atom_to_token=atoms(lists.atom_tok, torch.long),
        lm_input_ids=lm_ids,
        lm_attention_mask=lm_mask,
        lm_rows=tok(lists.tok_lm_row),
        lm_positions=tok(lists.tok_lm_pos),
        chain_ids=lists.chain_ids,
    )


def featurize(
    chains: Sequence[ChainSpec],
    *,
    conformers: ReferenceConformers | None = None,
    pad_tokens_to: int | None = None,
    atom_pad_multiple: int = 32,
) -> FoldFeatures:
    """Build :class:`FoldFeatures` for protein chains (no coordinates; prediction inputs).

    Args:
        chains: Chains in order; ``copies`` expands homo-oligomers with increasing ``sym_id``.
        conformers: Reference geometry table; the packaged one when ``None``.
        pad_tokens_to: Pad the token axis to this length (crop/bucket size); ``None`` = no padding.
        atom_pad_multiple: Atom axis padding multiple (upstream: 32).

    Raises:
        ValueError: ``pad_tokens_to`` is smaller than the token count, a chain is empty, or
            two chains (after ``copies`` expansion to ``ID``, ``ID_2``, ...) share an id.
    """
    table = conformers or ReferenceConformers.load()
    expanded = _expand_chains(chains)
    if any(not seq for _, seq, _, _ in expanded):
        raise ValueError("empty chain sequence")
    lists = _Lists()
    token = 0
    for asym, (chain_id, seq, entity, sym) in enumerate(expanded):
        lists.chain_ids.append(chain_id)
        lists.lm_rows.append(_lm_row(seq))
        for res_i, letter in enumerate(seq):
            code = PROTEIN_1TO3.get(letter, "UNK")
            template = table[code]
            lists.tok_residue.append(res_i)
            lists.tok_asym.append(asym)
            lists.tok_entity.append(entity)
            lists.tok_sym.append(sym)
            lists.tok_res_type.append(PROTEIN_RES_TYPES[code])
            lists.tok_rep_atom.append(len(lists.atom_pos) + template.representative_atom)
            lists.tok_lm_row.append(asym)
            lists.tok_lm_pos.append(res_i + 1)
            for name, element, charge, pos in zip(
                template.atoms, template.elements, template.charges, template.positions, strict=True
            ):
                lists.atom_pos.append(pos)
                lists.atom_elem.append(element)
                lists.atom_charge.append(float(charge))
                lists.atom_chars.append(encode_atom_name(name))
                lists.atom_uid.append(token)
                lists.atom_tok.append(token)
            token += 1
    return _assemble(lists, pad_tokens_to=pad_tokens_to, atom_pad_multiple=atom_pad_multiple)


def featurize_complex(
    complex: MolecularComplex,
    *,
    conformers: ReferenceConformers | None = None,
    pad_tokens_to: int | None = None,
    atom_pad_multiple: int = 32,
    generator: torch.Generator | None = None,
) -> FoldExample:
    """Build features and targets for a parsed (and usually cropped) complex.

    Every chain in ``complex.chains`` gets one LM row holding its *full* sequence (spec §4.5), and
    each token's ``lm_positions`` is its residue index plus one, so a crop still reads the whole
    chain through the LM. With ``generator`` the reference conformer of every token is centered
    and rotated by a fresh random rotation (spec §4.3); without it the raw conformer is used.

    Raises:
        ValueError: An atom name disagrees with the token's template (the parser guarantees the
            template order; anything else means the row is not from this pipeline), or
            ``pad_tokens_to`` is smaller than the token count.
    """
    table = conformers or ReferenceConformers.load()
    lists = _Lists()
    lists.chain_ids = [ch.chain_id for ch in complex.chains]
    lists.lm_rows = [_lm_row(ch.sequence) for ch in complex.chains]
    starts = complex.token_atom_start
    counts = complex.token_atom_count
    frame_rows: list[list[int]] = []
    for token in range(complex.num_tokens):
        code = RES_TYPE_TO_CODE[int(complex.token_res_type[token])]
        template = table[code]
        start, count = int(starts[token]), int(counts[token])
        names = complex.atom_name[start : start + count]
        if tuple(names) != template.atoms:
            raise ValueError(
                f"token {token} ({code}) atoms {names} do not match the template {template.atoms}"
            )
        lists.tok_residue.append(int(complex.token_residue_index[token]))
        lists.tok_asym.append(int(complex.token_asym_id[token]))
        lists.tok_entity.append(int(complex.token_entity_id[token]))
        lists.tok_sym.append(int(complex.token_sym_id[token]))
        lists.tok_res_type.append(int(complex.token_res_type[token]))
        lists.tok_rep_atom.append(start + template.representative_atom)
        lists.tok_lm_row.append(int(complex.token_asym_id[token]))
        lists.tok_lm_pos.append(int(complex.token_residue_index[token]) + 1)
        positions = template.positions
        if generator is not None:
            positions = (positions - positions.mean(0)) @ _random_rotation(generator).T
        for name, element, charge, pos in zip(
            template.atoms, template.elements, template.charges, positions, strict=True
        ):
            lists.atom_pos.append(pos)
            lists.atom_elem.append(element)
            lists.atom_charge.append(float(charge))
            lists.atom_chars.append(encode_atom_name(name))
            lists.atom_uid.append(token)
            lists.atom_tok.append(token)
        frame_rows.append([start + template.atoms.index(n) for n in ("N", "CA", "C")])
    features = _assemble(lists, pad_tokens_to=pad_tokens_to, atom_pad_multiple=atom_pad_multiple)
    L, A, n_atoms = features.num_tokens, features.num_atoms, complex.num_atoms
    coords = torch.zeros(1, A, 3)
    resolved = torch.zeros(1, A, dtype=torch.bool)
    bfactor = torch.zeros(1, A)
    coords[0, :n_atoms] = torch.from_numpy(complex.atom_coords)
    resolved[0, :n_atoms] = torch.from_numpy(complex.atom_resolved)
    bfactor[0, :n_atoms] = torch.from_numpy(complex.atom_bfactor)
    frames = torch.zeros(1, L, 3, dtype=torch.long)
    if frame_rows:
        frames[0, : len(frame_rows)] = torch.tensor(frame_rows, dtype=torch.long)
    targets = FoldTargets(coords=coords, resolved=resolved, bfactor=bfactor, frame_atom_idx=frames)
    example_id = "_".join([complex.metadata.id, *lists.chain_ids])
    return FoldExample(id=example_id, features=features, targets=targets)
```

- [ ] **Step 4: Run the new and the existing featurize/parity-adjacent tests**

Run: `.venv/bin/python -m pytest tests/fold/data/test_featurize.py tests/fold/test_predict.py tests/fold/test_modeling.py -q && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/data/featurize.py tests/fold/data/test_featurize.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: all pass (the M1 tests prove `featurize` is unchanged). If `OPLM_FOLD_FIXTURES` points at `~/fold-fixtures/25479`, also run `tests/fold/test_parity.py -q`: 37 pass.

- [ ] **Step 5: Commit**

```bash
git add src/oplm/fold/data/featurize.py tests/fold/data/test_featurize.py
git commit -m "feat(fold): featurize_complex with targets, full-chain LM rows and per-token conformer rotations"
```

---

### Task 6: Collation with the atom budget

**Files:**
- Create: `src/oplm/fold/data/collate.py`
- Test: `tests/fold/data/test_collate.py`

**Interfaces:**
- Consumes: `FoldExample`, `FoldFeatures`, `FoldTargets` (Task 5).
- Produces: `FoldBatch(ids, features, targets)`, `collate_examples(examples, *, crop_size, atoms_per_token_budget=24, atom_pad_multiple=32) -> FoldBatch`, `atom_budget(crop_size, atoms_per_token_budget, atom_pad_multiple) -> int`. Task 8's tests and M2b's `FoldTask.build_dataloader` pass `collate_examples` (wrapped to read the dataset's row dicts) as the DataLoader's `collate_fn`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/fold/data/test_collate.py
"""Batching FoldExamples under the §4.7 atom budget."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from oplm.fold.data.collate import FoldBatch, atom_budget, collate_examples
from oplm.fold.data.featurize import featurize_complex
from oplm.model.tokenization_oplm import VOCAB

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.fold.data.complex import MolecularComplex


def test_atom_budget_rounds_up_to_the_block_multiple() -> None:
    assert atom_budget(384, 24, 32) == 9216
    assert atom_budget(10, 3, 32) == 32 and atom_budget(16, 2, 32) == 32 and atom_budget(17, 2, 32) == 64


def test_collate_pads_tokens_atoms_and_lm_rows(parsed: Callable[[str], MolecularComplex]) -> None:
    a = featurize_complex(parsed("1crn"))  # 46 tokens, one chain (row length 48)
    b = featurize_complex(parsed("1l2y"))  # 20 tokens, one chain (row length 22)
    batch = collate_examples([a, b], crop_size=64)
    assert isinstance(batch, FoldBatch) and batch.ids == ["1crn_A", "1l2y_A"]
    f, t = batch.features, batch.targets
    assert f.num_tokens == 64 and f.num_atoms == atom_budget(64, 24, 32) == 1536
    assert f.token_mask.sum(1).tolist() == [46, 20] and f.atom_mask.sum(1).tolist() == [
        int(a.features.atom_mask.sum()),
        int(b.features.atom_mask.sum()),
    ]
    assert f.token_bonds.shape == (2, 64, 64, 1) and f.ref_pos.shape == (2, 1536, 3)
    assert f.ref_atom_name_chars.shape == (2, 1536, 4)
    assert f.lm_input_ids.shape == (2, 48) and f.lm_attention_mask.shape == (2, 48)
    assert f.lm_input_ids[1, 22:].eq(VOCAB["<pad>"]).all() and f.lm_attention_mask[1, 22:].eq(0).all()
    assert torch.equal(f.lm_input_ids[0], a.features.lm_input_ids[0])
    assert f.lm_rows[1, :20].eq(1).all() and f.lm_rows[1, 20:].eq(0).all()  # offset into the stack
    assert f.lm_rows[0].eq(0).all()
    assert f.chain_ids == ["A", "A"] and f.num_chains == 2
    assert torch.equal(f.res_type[0, :46], a.features.res_type[0]) and f.res_type[0, 46:].eq(0).all()
    assert t.coords.shape == (2, 1536, 3) and t.resolved.shape == (2, 1536)
    assert t.frame_atom_idx.shape == (2, 64, 3) and not t.frame_atom_idx[1, 20:].any()
    assert torch.equal(t.coords[1, : b.features.num_atoms], b.targets.coords[0])


def test_collate_of_one_matches_padding_the_example(parsed: Callable[[str], MolecularComplex]) -> None:
    ex = featurize_complex(parsed("1l2y"))
    batch = collate_examples([ex], crop_size=32)
    padded = featurize_complex(parsed("1l2y"), pad_tokens_to=32)
    assert torch.equal(batch.features.residue_index, padded.features.residue_index)
    assert torch.equal(batch.features.ref_pos[:, : ex.features.num_atoms], ex.features.ref_pos)
    assert batch.features.num_atoms == atom_budget(32, 24, 32)


def test_multi_chain_example_rows_offset_by_chain_count(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    two = featurize_complex(parsed("1jb6"))  # chains A and B, 66 tokens
    one = featurize_complex(parsed("1l2y"))
    batch = collate_examples([two, one], crop_size=128)
    assert batch.features.lm_input_ids.shape[0] == 3
    assert batch.features.lm_rows[0, 33] == 1 and batch.features.lm_rows[1, :20].eq(2).all()


def test_over_budget_and_over_crop_examples_are_rejected(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    ex = featurize_complex(parsed("1crn"))
    with pytest.raises(ValueError, match="atom budget"):
        collate_examples([ex], crop_size=64, atoms_per_token_budget=1)
    with pytest.raises(ValueError, match="crop_size"):
        collate_examples([ex], crop_size=32)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/data/test_collate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.fold.data.collate'`.

- [ ] **Step 3: Implement the collator**

```python
# src/oplm/fold/data/collate.py
"""Batch ``FoldExample``s to the crop and the atom budget (spec §4.7).

Tokens pad to ``crop_size``; atoms pad to ``atoms_per_token_budget × crop_size`` rounded up to
the attention block multiple. An example over either limit is rejected, never truncated. LM rows
of all examples are stacked and ``lm_rows`` is offset so each token still names its own row.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

from oplm.fold.data.featurize import FoldFeatures, FoldTargets
from oplm.model.tokenization_oplm import VOCAB

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from oplm.fold.data.featurize import FoldExample

__all__ = ["FoldBatch", "atom_budget", "collate_examples"]

_PAD = VOCAB["<pad>"]
_TOKEN_FIELDS = (
    "token_index",
    "residue_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "mol_type",
    "res_type",
    "token_mask",
    "distogram_atom_idx",
    "lm_rows",
    "lm_positions",
)
_ATOM_FIELDS = (
    "ref_pos",
    "ref_element",
    "ref_charge",
    "ref_atom_name_chars",
    "ref_space_uid",
    "atom_mask",
    "atom_to_token",
)


@dataclass
class FoldBatch:
    """A collated batch: ids, features ``(B, ...)`` and targets ``(B, ...)``."""

    ids: list[str]
    features: FoldFeatures
    targets: FoldTargets


def atom_budget(crop_size: int, atoms_per_token_budget: int, atom_pad_multiple: int) -> int:
    """Padded atom-axis length for a crop: budget × crop rounded up to the block multiple."""
    return -(-atoms_per_token_budget * crop_size // atom_pad_multiple) * atom_pad_multiple


def _pad(t: Tensor, dim: int, size: int, value: float = 0) -> Tensor:
    if t.shape[dim] == size:
        return t
    shape = list(t.shape)
    shape[dim] = size - t.shape[dim]
    return torch.cat([t, torch.full(shape, value, dtype=t.dtype)], dim=dim)


def collate_examples(
    examples: Sequence[FoldExample],
    *,
    crop_size: int,
    atoms_per_token_budget: int = 24,
    atom_pad_multiple: int = 32,
) -> FoldBatch:
    """Stack examples into one :class:`FoldBatch`.

    Raises:
        ValueError: ``examples`` is empty, an example has more tokens than ``crop_size``, or more
            real atoms than the atom budget.
    """
    if not examples:
        raise ValueError("collate_examples needs at least one example")
    L = crop_size
    A = atom_budget(crop_size, atoms_per_token_budget, atom_pad_multiple)
    for ex in examples:
        n_tokens = int(ex.features.token_mask.sum())
        n_atoms = int(ex.features.atom_mask.sum())
        if n_tokens > L:
            raise ValueError(f"{ex.id}: {n_tokens} tokens exceed crop_size={crop_size}")
        if n_atoms > A:
            raise ValueError(
                f"{ex.id}: {n_atoms} atoms exceed the atom budget {A} "
                f"({atoms_per_token_budget} per token × {crop_size}); recrop instead of truncating"
            )
    T = max(ex.features.lm_input_ids.shape[1] for ex in examples)
    cols: dict[str, list[Tensor]] = {name: [] for name in (*_TOKEN_FIELDS, *_ATOM_FIELDS)}
    bonds, lm_ids, lm_mask, chain_ids = [], [], [], []
    row_offset = 0
    for ex in examples:
        f = ex.features
        for name in _TOKEN_FIELDS:
            value = getattr(f, name)
            if name == "lm_rows":
                value = value + row_offset * f.token_mask.long()
            cols[name].append(_pad(value, 1, L))
        for name in _ATOM_FIELDS:
            cols[name].append(_pad(getattr(f, name), 1, A))
        bonds.append(_pad(_pad(f.token_bonds, 1, L), 2, L))
        lm_ids.append(_pad(f.lm_input_ids, 1, T, _PAD))
        lm_mask.append(_pad(f.lm_attention_mask, 1, T))
        chain_ids.extend(f.chain_ids)
        row_offset += f.num_chains
    features = FoldFeatures(
        **{name: torch.cat(cols[name], dim=0) for name in (*_TOKEN_FIELDS, *_ATOM_FIELDS)},
        token_bonds=torch.cat(bonds, dim=0),
        lm_input_ids=torch.cat(lm_ids, dim=0),
        lm_attention_mask=torch.cat(lm_mask, dim=0),
        chain_ids=chain_ids,
    )
    targets = FoldTargets(
        **{
            fld.name: torch.cat(
                [
                    _pad(getattr(ex.targets, fld.name), 1, L if fld.name == "frame_atom_idx" else A)
                    for ex in examples
                ],
                dim=0,
            )
            for fld in fields(FoldTargets)
        }
    )
    return FoldBatch(ids=[ex.id for ex in examples], features=features, targets=targets)
```

- [ ] **Step 4: Run the tests, lint, types**

Run: `.venv/bin/python -m pytest tests/fold/data/test_collate.py -q && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/data/collate.py tests/fold/data/test_collate.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/oplm/fold/data/collate.py tests/fold/data/test_collate.py
git commit -m "feat(fold): collate FoldExamples under the atom budget"
```

---

### Task 7: `oplm fold preprocess`: shards, index, splits, manifest, dataset card

**Files:**
- Create: `src/oplm/fold/data/dataset.py` (preprocessing half; Task 8 adds `StructureDataset`)
- Modify: `src/oplm/fold/cli.py`
- Modify: `tests/fold/data/conftest.py` (session fixture dataset)
- Test: `tests/fold/data/test_dataset.py`, `tests/fold/test_cli.py` (append)

**Interfaces:**
- Consumes: `parse_mmcif`, `Rejection`, `PARSE_VERSION`, `MAX_RESOLUTION`, `MIN_RESOLVED_RESIDUES` (Task 3), `ChemCompTable` (Task 1), `complex_to_row`, `rows_to_table`, `COMPLEX_SCHEMA` (Task 2), `ReferenceConformers`.
- Produces: `PreprocessConfig`, `preprocess(cfg) -> dict[str, Any]` (the manifest), `assign_splits(index_rows, *, cutoff, heldout, valid_size, valid_seed) -> None`, `read_target_list(path) -> list[tuple[str, str | None]]`, `discover_entries(archive, ids=None) -> list[Path]`, the on-disk layout below, and the CLI command. Task 8 reads `manifest.json` (`row_group_size`), `index.parquet` and `shards/`; M2c reads `targets/<name>.json`.

On-disk layout of `<out>/`:

```
shards/shard-00000.parquet ...   # COMPLEX_SCHEMA rows, row groups of `row_group_size`
index.parquet                    # one row per chain (columns below)
targets/<name>.json              # [{"id": "1abc", "chain": "A" | null}, ...] per held-out list
ccd_table.json                   # ChemCompTable.save of the parsed CCD
checksums.txt                    # sha256sum format, one line per input file
manifest.json                    # provenance, config, counts, shard list
README.md                        # dataset card (YAML front matter + description)
```

Index columns: `id`, `chain_id`, `auth_chain_id`, `chain_index` (position in the entry's
`chains`), `entity_id`, `sym_id`, `sequence`, `length`, `resolved_count` (tokens with at least
one resolved atom), `n_chains` (protein chains in the entry), `resolution` (nullable),
`release_date` (nullable), `method`, `shard` (file name), `row`, `split`.

Split values: `train`, `valid`, one per held-out list name (`cameo`, `casp14`),
`excluded_heldout_sequence` (pre-cutoff entry sharing a chain sequence with a held-out
entry), `post_cutoff` (after the cutoff, not chosen for `valid`), `undated` (no release date).

- [ ] **Step 1: Write the failing tests**

Add to `tests/fold/data/conftest.py`:

```python
from oplm.fold.data.dataset import PreprocessConfig, preprocess


def _preprocess_fixtures(out: Path, cutoff: str) -> Path:
    targets = out / "targets-in"
    targets.mkdir(parents=True)
    (targets / "cameo.txt").write_text("1CRN_A\n")
    (targets / "casp14.txt").write_text("# CASP14 subset\n1jb6_A\n")
    preprocess(
        PreprocessConfig(
            archive=MMCIF_DIR,
            ccd=FOLD_FIXTURES / "ccd_subset.cif",
            out=out / "dataset",
            workers=2,
            shard_bytes=150_000,
            row_group_size=2,
            cutoff=cutoff,
            valid_size=64,
            heldout={"cameo": targets / "cameo.txt", "casp14": targets / "casp14.txt"},
        )
    )
    return out / "dataset"


@pytest.fixture(scope="session")
def fixture_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The six fixtures preprocessed with the real cutoff: train = 1ejg, 4hhb, 2ds7, 1l2y."""
    return _preprocess_fixtures(tmp_path_factory.mktemp("fold_ds"), cutoff="2021-09-30")


@pytest.fixture(scope="session")
def fixture_dataset_early_cutoff(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Cutoff 1990-01-01: 4hhb trains, 2ds7 becomes valid, 1ejg and 1l2y are post_cutoff."""
    return _preprocess_fixtures(tmp_path_factory.mktemp("fold_ds_early"), cutoff="1990-01-01")
```

```python
# tests/fold/data/test_dataset.py
"""Preprocessing to shards, index, splits and manifest (addendum §4; spec §4.7, §7.2, §11)."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

from oplm.fold.data.complex import COMPLEX_SCHEMA, row_to_complex
from oplm.fold.data.dataset import (
    PreprocessConfig,
    assign_splits,
    discover_entries,
    preprocess,
    read_target_list,
)
from tests.fold.data.conftest import FIXTURE_IDS, FOLD_FIXTURES, MMCIF_DIR

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.fold.data.complex import MolecularComplex


def index_by_id(dataset: Path) -> dict[str, list[dict[str, object]]]:
    rows = pq.read_table(dataset / "index.parquet").to_pylist()
    out: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        out.setdefault(str(row["id"]), []).append(row)
    return out


def test_preprocess_layout_counts_and_schema(
    fixture_dataset: Path, parsed: Callable[[str], MolecularComplex]
) -> None:
    manifest = json.loads((fixture_dataset / "manifest.json").read_text())
    assert manifest["schema_version"] == 1 and manifest["parse_version"] == 1
    assert manifest["accepted"] == 6 and manifest["rejected"] == {}
    expected_dropped: dict[str, int] = {}
    for eid in FIXTURE_IDS:  # 4hhb: 2 ligand entities + water; 2ds7: 1 + water; 1jb6: water
        for kind, n in parsed(eid).metadata.dropped.items():
            expected_dropped[kind] = expected_dropped.get(kind, 0) + n
    assert manifest["dropped"] == expected_dropped == {"ligand": 3, "water": 3}
    assert manifest["row_group_size"] == 2 and manifest["config"]["cutoff"] == "2021-09-30"
    assert manifest["licenses"] == {"wwPDB": "CC0 1.0", "CCD": "CC0 1.0"}
    assert manifest["ccd"]["components"] == 45 and len(manifest["ccd"]["sha256"]) == 64
    shards = sorted((fixture_dataset / "shards").glob("shard-*.parquet"))
    assert len(shards) >= 2 and [s["file"] for s in manifest["shards"]] == [s.name for s in shards]
    assert sum(s["rows"] for s in manifest["shards"]) == 6
    for shard in shards:
        pf = pq.ParquetFile(shard)
        assert pf.schema_arrow.equals(COMPLEX_SCHEMA)
        assert all(pf.metadata.row_group(i).num_rows <= 2 for i in range(pf.metadata.num_row_groups))
    checksums = (fixture_dataset / "checksums.txt").read_text().splitlines()
    assert len(checksums) == 6 and all(len(line.split()[0]) == 64 for line in checksums)
    assert (fixture_dataset / "ccd_table.json").exists()
    card = (fixture_dataset / "README.md").read_text()
    assert card.startswith("---\nlicense: cc0-1.0") and "6 entries" in card
    assert json.loads((fixture_dataset / "targets" / "cameo.json").read_text()) == [
        {"id": "1crn", "chain": "A"}
    ]


def test_index_rows_point_at_their_shard_rows(fixture_dataset: Path) -> None:
    by_id = index_by_id(fixture_dataset)
    assert sorted(by_id) == ["1crn", "1ejg", "1jb6", "1l2y", "2ds7", "4hhb"]
    assert len(by_id["4hhb"]) == 4 and [r["chain_index"] for r in by_id["4hhb"]] == [0, 1, 2, 3]
    row = by_id["2ds7"][0]
    assert row["length"] == 51 and row["resolved_count"] == 40 and row["n_chains"] == 1
    assert row["resolution"] == pytest.approx(2.5) and row["method"] == "X-RAY DIFFRACTION"
    table = pq.read_table(fixture_dataset / "shards" / str(row["shard"]))
    c = row_to_complex(table.slice(int(row["row"]), 1).to_pylist()[0])
    assert c.metadata.id == "2ds7" and c.chains[int(row["chain_index"])].chain_id == row["chain_id"]
    assert by_id["1l2y"][0]["resolution"] is None


def test_splits_with_the_real_cutoff(fixture_dataset: Path) -> None:
    split = {eid: {r["split"] for r in rows} for eid, rows in index_by_id(fixture_dataset).items()}
    assert split == {
        "1crn": {"cameo"},
        "1jb6": {"casp14"},
        "1ejg": {"train"},  # crambin too, but L at residue 25 (vs I in 1crn): no exact match
        "4hhb": {"train"},
        "2ds7": {"train"},
        "1l2y": {"train"},
    }


def test_splits_with_an_early_cutoff_pick_valid(fixture_dataset_early_cutoff: Path) -> None:
    split = {
        eid: {r["split"] for r in rows} for eid, rows in index_by_id(fixture_dataset_early_cutoff).items()
    }
    assert split["4hhb"] == {"train"} and split["2ds7"] == {"valid"}
    assert split["1ejg"] == {"post_cutoff"} and split["1l2y"] == {"post_cutoff"}
    manifest = json.loads((fixture_dataset_early_cutoff / "manifest.json").read_text())
    assert manifest["splits"] == {"cameo": 1, "casp14": 1, "train": 1, "valid": 1, "post_cutoff": 2}


def test_assign_splits_valid_is_seeded_and_capped() -> None:
    def entry(eid: str, date: str, res: float | None = 2.0, n: int = 1, length: int = 100) -> dict:
        return {
            "id": eid, "release_date": date, "resolution": res, "n_chains": n,
            "length": length, "sequence": eid * 10, "split": "",
        }

    rows = [entry(f"p{i:03d}", "2022-01-01") for i in range(10)] + [
        entry("old1", "2020-01-01"),
        entry("big1", "2022-01-01", n=2),
        entry("low1", "2022-01-01", res=3.0),
        entry("nmr1", "2022-01-01", res=None),
        entry("short", "2022-01-01", length=20),
        entry("none1", None),
    ]
    assign_splits(rows, cutoff="2021-09-30", heldout={}, valid_size=4, valid_seed=0)
    chosen = sorted(r["id"] for r in rows if r["split"] == "valid")
    assert len(chosen) == 4 and all(c.startswith("p") for c in chosen)
    assert {r["id"]: r["split"] for r in rows if not r["id"].startswith("p")} == {
        "old1": "train", "big1": "post_cutoff", "low1": "post_cutoff",
        "nmr1": "post_cutoff", "short": "post_cutoff", "none1": "undated",
    }
    again = [dict(r, split="") for r in rows]
    assign_splits(again, cutoff="2021-09-30", heldout={}, valid_size=4, valid_seed=0)
    assert sorted(r["id"] for r in again if r["split"] == "valid") == chosen
    other = [dict(r, split="") for r in rows]
    assign_splits(other, cutoff="2021-09-30", heldout={}, valid_size=4, valid_seed=1)
    assert sorted(r["id"] for r in other if r["split"] == "valid") != chosen


def test_heldout_ids_are_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "t.txt").write_text("4HHB_B\n1L2Y\n\n# comment\n")
    assert read_target_list(tmp_path / "t.txt") == [("4hhb", "B"), ("1l2y", None)]
    rows = [
        {"id": "4hhb", "release_date": "1984-07-17", "resolution": 1.74, "n_chains": 4,
         "length": 141, "sequence": "VLSP", "split": ""},
        {"id": "abcd", "release_date": "1990-01-01", "resolution": 2.0, "n_chains": 1,
         "length": 100, "sequence": "VLSP", "split": ""},  # same sequence as a held-out chain
    ]
    assign_splits(rows, cutoff="2021-09-30", heldout={"cameo": {"4hhb"}}, valid_size=0, valid_seed=0)
    assert rows[0]["split"] == "cameo" and rows[1]["split"] == "excluded_heldout_sequence"


def test_duplicate_entry_ids_raise(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    (archive / "a").mkdir(parents=True)
    (archive / "b").mkdir()
    shutil.copy(MMCIF_DIR / "1crn.cif.gz", archive / "a" / "1crn.cif.gz")
    shutil.copy(MMCIF_DIR / "1crn.cif.gz", archive / "b" / "1CRN.cif.gz")
    with pytest.raises(ValueError, match="1crn"):
        discover_entries(archive)
    assert [p.name for p in discover_entries(MMCIF_DIR)] == [
        "1crn.cif.gz", "1ejg.cif.gz", "1jb6.cif.gz", "1l2y.cif.gz", "2ds7.cif.gz", "4hhb.cif.gz"
    ]
    (tmp_path / "ids.txt").write_text("4HHB\n1l2y\n")
    assert [p.name for p in discover_entries(MMCIF_DIR, ids=tmp_path / "ids.txt")] == [
        "1l2y.cif.gz", "4hhb.cif.gz"
    ]


def test_rejections_are_counted_not_fatal(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    for name in ("1crn.cif.gz", "1l2y.cif.gz"):
        shutil.copy(MMCIF_DIR / name, archive / name)
    with gzip.open(archive / "zzzz.cif.gz", "wt") as fh:
        fh.write("not a cif\n")
    manifest = preprocess(
        PreprocessConfig(
            archive=archive, ccd=FOLD_FIXTURES / "ccd_subset.cif", out=tmp_path / "out", workers=1
        )
    )
    assert manifest["accepted"] == 2 and manifest["rejected"] == {"parse_error": 1}
    assert manifest["rejections"] == [{"id": "zzzz.cif.gz", "reason": "parse_error"}]
```

Append to `tests/fold/test_cli.py`:

```python
def test_preprocess_command_runs_on_the_fixtures(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent.parent / "fixtures" / "fold"
    result = runner.invoke(
        app,
        [
            "fold", "preprocess",
            "--archive", str(fixtures / "mmcif"), "--ccd", str(fixtures / "ccd_subset.cif"),
            "--out", str(tmp_path / "ds"), "--workers", "1", "--cutoff", "1990-01-01",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "accepted" in result.output and (tmp_path / "ds" / "manifest.json").exists()
```

(`tests/fold/test_cli.py` already has the module-level `runner` and `app`; add `from pathlib import Path`
to its imports if it is only under `TYPE_CHECKING`.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/data/test_dataset.py tests/fold/test_cli.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.fold.data.dataset'`.

- [ ] **Step 3: Implement preprocessing**

```python
# src/oplm/fold/data/dataset.py
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
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
import torch

from oplm.fold.data.ccd import ChemCompTable, ReferenceConformers
from oplm.fold.data.complex import complex_to_row, rows_to_table

if TYPE_CHECKING:
    from collections.abc import Iterator

    from oplm.fold.data.complex import MolecularComplex

__all__ = [
    "PreprocessConfig",
    "assign_splits",
    "discover_entries",
    "preprocess",
    "read_target_list",
]

SCHEMA_VERSION = 1
_CIF_SUFFIXES = (".cif.gz", ".cif")
_INDEX_COLUMNS = (
    "id", "chain_id", "auth_chain_id", "chain_index", "entity_id", "sym_id", "sequence", "length",
    "resolved_count", "n_chains", "resolution", "release_date", "method", "shard", "row", "split",
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
    """Every ``.cif``/``.cif.gz`` under ``archive``, sorted by entry id, optionally restricted to ``ids``.

    Raises:
        ValueError: Two files map to the same entry id (case-insensitive).
    """
    wanted = None
    if ids is not None:
        wanted = {line.strip().lower() for line in Path(ids).read_text().splitlines() if line.strip()}
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


# ----------------------------------------------------------------------------- workers

_WORKER: dict[str, Any] = {}


def _init_worker(ccd: str, max_resolution: float, min_resolved_residues: int) -> None:
    table = ChemCompTable.load(Path(ccd)) if ccd.endswith(".json") else ChemCompTable.parse(Path(ccd))
    _WORKER.update(
        ccd=table,
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
        c.atom_coords, c.atom_resolved, c.atom_element, c.atom_charge, c.atom_bfactor,
        c.atom_to_token, c.token_res_type, c.token_mol_type, c.token_asym_id,
        c.token_entity_id, c.token_sym_id, c.token_residue_index, c.token_center_atom,
        c.token_rep_atom, c.token_atom_start, c.token_atom_count, c.bonds,
    )
    return sum(int(a.nbytes) for a in arrays) + 8 * c.num_atoms


def _parse_one(path_str: str) -> dict[str, Any]:
    from oplm.fold.data.mmcif import Rejection, parse_mmcif

    path = Path(path_str)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = parse_mmcif(
        path,
        ccd=_WORKER["ccd"],
        conformers=_WORKER["conformers"],
        max_resolution=_WORKER["max_resolution"],
        min_resolved_residues=_WORKER["min_resolved_residues"],
    )
    out: dict[str, Any] = {"path": path.name, "sha256": digest}
    if isinstance(result, Rejection):
        out["rejection"] = {"id": result.id, "reason": result.reason}
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
    return out


def _parse_all(cfg: PreprocessConfig, paths: list[Path]) -> Iterator[dict[str, Any]]:
    init_args = (str(cfg.ccd), cfg.max_resolution, cfg.min_resolved_residues)
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
        if date is None:
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
pretty_name: {pretty}
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
(sha256 of every input file), `manifest.json` (provenance, configuration, counts).

Splits: {splits}

Load with `oplm.fold.data.dataset.StructureDataset(path, split="train", crop_size=384)`;
see `docs/FOLD.md` in https://github.com/briney/oplm.
"""


def preprocess(cfg: PreprocessConfig) -> dict[str, Any]:
    """Run the whole pipeline and return the manifest (also written to ``out/manifest.json``)."""
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "targets").mkdir(exist_ok=True)
    paths = discover_entries(cfg.archive, cfg.ids)
    ccd_path = Path(cfg.ccd)
    ccd = ChemCompTable.load(ccd_path) if ccd_path.suffix == ".json" else ChemCompTable.parse(ccd_path)
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
    for result in _parse_all(cfg, paths):
        checksums.append(f"{result['sha256']}  {result['path']}")
        if "rejection" in result:
            rej = result["rejection"]
            rejected[rej["reason"]] = rejected.get(rej["reason"], 0) + 1
            rejections.append(rej)
            continue
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
    writer.flush()

    assign_splits(
        index_rows,
        cutoff=cfg.cutoff,
        heldout=heldout,
        valid_size=cfg.valid_size,
        valid_seed=cfg.valid_seed,
    )
    index_table = rows_to_index_table(index_rows)
    pq.write_table(index_table, out / "index.parquet")
    (out / "checksums.txt").write_text("\n".join(checksums) + ("\n" if checksums else ""))

    entry_splits: dict[str, str] = {r["id"]: r["split"] for r in index_rows}
    split_counts: dict[str, int] = {}
    for split in entry_splits.values():
        split_counts[split] = split_counts.get(split, 0) + 1
    snapshot = dt.date.today().isoformat()
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    config["heldout"] = {k: str(v) for k, v in cfg.heldout.items()}
    from oplm.fold.data.mmcif import PARSE_VERSION

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "parse_version": PARSE_VERSION,
        "source": cfg.source,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "snapshot_date": snapshot,
        "archive": str(cfg.archive),
        "ccd": {
            "path": str(ccd_path),
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
        "row_group_size": cfg.row_group_size,
        "shards": writer.shards,
        "config": config,
        "licenses": {"wwPDB": "CC0 1.0", "CCD": "CC0 1.0"},
        "index_columns": list(_INDEX_COLUMNS),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "README.md").write_text(
        _CARD.format(
            pretty=f"OPLM fold data: {cfg.source} {snapshot}",
            snapshot=snapshot,
            accepted=accepted,
            chains=len(index_rows),
            ccd=ccd.source,
            splits=", ".join(f"{k} {v}" for k, v in manifest["splits"].items()) or "none",
        )
    )
    return manifest


def rows_to_index_table(index_rows: list[dict[str, Any]]) -> Any:
    """The per-chain index as a pyarrow table with explicit nullable types."""
    import pyarrow as pa

    schema = pa.schema(
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
    columns = [pa.array([r[f.name] for r in index_rows], type=f.type) for f in schema]
    return pa.Table.from_arrays(columns, schema=schema)
```

Note `manifest["splits"]` counts entries, not chains. The `test_splits_with_an_early_cutoff`
expectation `{"cameo": 1, "casp14": 1, "train": 1, "valid": 1, "post_cutoff": 2}` is in that
unit (4hhb train, 2ds7 valid, 1ejg and 1l2y post-cutoff, 1crn cameo, 1jb6 casp14).

Add the CLI command to `src/oplm/fold/cli.py` (after `predict`):

```python
@app.command("preprocess")
def preprocess_cmd(
    archive: Annotated[Path, typer.Option("--archive", help="Root of the mmCIF tree")],
    ccd: Annotated[Path, typer.Option("--ccd", help="components.cif.gz or a ccd_table.json")],
    out: Annotated[Path, typer.Option("--out", help="Dataset directory to create")],
    workers: Annotated[int, typer.Option(help="Parser processes")] = max(1, (os.cpu_count() or 2) - 1),
    shard_bytes: Annotated[int, typer.Option(help="Target bytes per shard")] = 1 << 30,
    row_group_size: Annotated[int, typer.Option(help="Rows per Parquet row group")] = 16,
    cutoff: Annotated[str, typer.Option(help="Training release-date cutoff (ISO)")] = "2021-09-30",
    valid_size: Annotated[int, typer.Option(help="Post-cutoff monomer chains held for valid")] = 64,
    valid_seed: Annotated[int, typer.Option()] = 0,
    heldout: Annotated[
        list[str] | None, typer.Option("--heldout", help="name=path of a target list; repeatable")
    ] = None,
    ids: Annotated[Path | None, typer.Option(help="Restrict to the entry ids in this file")] = None,
) -> None:
    """Parse an mmCIF archive into structure shards, index, splits and manifest."""
    from oplm.fold.data.dataset import PreprocessConfig, preprocess

    lists: dict[str, Path] = {}
    for item in heldout or []:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise typer.BadParameter(f"--heldout expects name=path, got {item!r}")
        lists[name] = Path(path)
    manifest = preprocess(
        PreprocessConfig(
            archive=archive,
            ccd=ccd,
            out=out,
            workers=workers,
            shard_bytes=shard_bytes,
            row_group_size=row_group_size,
            cutoff=cutoff,
            valid_size=valid_size,
            valid_seed=valid_seed,
            heldout=lists,
            ids=ids,
        )
    )
    table = Table(title=f"preprocess -> {out}")
    table.add_column("key")
    table.add_column("value")
    table.add_row("inputs", str(manifest["inputs"]))
    table.add_row("accepted", str(manifest["accepted"]))
    table.add_row("rejected", json.dumps(manifest["rejected"]))
    table.add_row("dropped", json.dumps(manifest["dropped"]))
    table.add_row("chains", str(manifest["chains"]))
    table.add_row("splits", json.dumps(manifest["splits"]))
    table.add_row("shards", str(len(manifest["shards"])))
    console.print(table)
```

Add `import os` to the CLI module's imports and update its module docstring to list
`preprocess`.

- [ ] **Step 4: Run the tests, lint, types**

Run: `.venv/bin/python -m pytest tests/fold/data/test_dataset.py tests/fold/test_cli.py -q && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/data/dataset.py src/oplm/fold/cli.py tests/fold/data/test_dataset.py tests/fold/data/conftest.py tests/fold/test_cli.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: all pass. The two-worker session fixture uses the spawn context and takes a few
seconds; if it exceeds ten, mark the fixture-backed tests `slow` and keep a `workers=1`
smoke in the fast suite.

- [ ] **Step 5: Commit**

```bash
git add src/oplm/fold/data/dataset.py src/oplm/fold/cli.py tests/fold/data/test_dataset.py tests/fold/data/conftest.py tests/fold/test_cli.py
git commit -m "feat(fold): oplm fold preprocess: shards, per-chain index, splits, manifest and dataset card"
```

---

### Task 8: `StructureDataset` with the inherited resume contract

**Files:**
- Modify: `src/oplm/fold/data/dataset.py`
- Modify: `src/oplm/fold/data/collate.py` (+ `collate_rows`)
- Test: `tests/fold/data/test_dataset.py` (append), `tests/fold/data/test_collate.py` (append)

**Interfaces:**
- Consumes: `ShardedProteinDataset` (`oplm.data.sequence.dataset`), `row_to_complex` (Task 2), `crop_chain` (Task 4), `featurize_complex` (Task 5), `collate_examples` (Task 6), the Task 7 layout.
- Produces: `StructureDataset(path, *, split="train", crop_size=384, contiguous_prob=0.5, max_chain_length=1024, min_resolved_residues=4, seed=0, shuffle_shards=True, shuffle_rows=True, conformers=None, atom_pad_multiple=32)` yielding `{"id": str, "example": FoldExample}` rows, with `__len__`, `total_length`, `set_epoch`, `set_resume_skip`, `clear_resume_skip`, `_arm_sample_skip` inherited; `collate_rows(rows, *, crop_size, atoms_per_token_budget=24, atom_pad_multiple=32) -> FoldBatch`. M2b's `FoldTask.build_dataloader` builds one `StructureDataset` per `data.train` entry and passes `functools.partial(collate_rows, crop_size=...)` to the DataLoader; `split=None` means every chain regardless of split (eval tooling, tests).

- [ ] **Step 1: Append the failing tests**

```python
# tests/fold/data/test_dataset.py (append)
import dataclasses
from functools import partial

import torch
from torch.utils.data import DataLoader

import oplm.data.sequence.dataset as seq_dataset
from oplm.data.sequence.dataset import InterleavedDataset
from oplm.fold.data.collate import collate_rows
from oplm.fold.data.dataset import StructureDataset


@pytest.fixture
def single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seq_dataset, "_resolve_distributed_context", lambda: (0, 1, 0, 1))


def stream(ds: StructureDataset | InterleavedDataset) -> list[tuple[str, list[int]]]:
    return [(str(r["id"]), r["example"].features.residue_index[0].tolist()) for r in ds]


def test_eligibility_filters_and_len(fixture_dataset: Path, single_process: None) -> None:
    train = StructureDataset(fixture_dataset, split="train", crop_size=16)
    assert len(train) == 4 and train.total_length == 4  # 1ejg, 4hhb, 2ds7, 1l2y
    assert len(StructureDataset(fixture_dataset, split="train", max_chain_length=50)) == 2  # 1ejg, 1l2y
    assert len(StructureDataset(fixture_dataset, split="train", min_resolved_residues=45)) == 2  # 1ejg, 4hhb
    assert len(StructureDataset(fixture_dataset, split="casp14")) == 1
    assert len(StructureDataset(fixture_dataset, split="cameo")) == 1
    assert len(StructureDataset(fixture_dataset, split=None)) == 6
    assert len(StructureDataset(fixture_dataset, split="nope")) == 0


def test_examples_are_seeded_single_chain_crops(fixture_dataset: Path, single_process: None) -> None:
    ds = StructureDataset(fixture_dataset, split="train", crop_size=16, seed=0)
    ds.set_epoch(0)
    rows = list(ds)
    assert len(rows) == 4
    assert sorted(str(r["id"]).split("_")[0] for r in rows) == ["1ejg", "1l2y", "2ds7", "4hhb"]
    by_id = index_by_id(fixture_dataset)
    for r in rows:
        ex = r["example"]
        f = ex.features
        entry, chain_id = ex.id.split("_")
        sequence = next(c["sequence"] for c in by_id[entry] if c["chain_id"] == chain_id)
        assert f.num_tokens <= 16 and f.num_chains == 1 and f.asym_id.eq(0).all()
        assert f.lm_input_ids.shape[1] == len(sequence) + 2  # the LM row is the full chain
        assert ex.targets.coords.shape[1] == f.num_atoms
        starts = f.atom_to_token[0, f.atom_mask[0]]
        for token in range(f.num_tokens):  # rotated, centered conformers (spec §4.3)
            atoms = f.ref_pos[0, f.atom_mask[0]][starts == token]
            assert torch.allclose(atoms.mean(0), torch.zeros(3), atol=1e-4)
    hhb = next(r for r in rows if str(r["id"]).startswith("4hhb"))
    assert hhb["example"].features.lm_input_ids.shape[1] in (143, 148)  # a full α or β chain
    again = StructureDataset(fixture_dataset, split="train", crop_size=16, seed=0)
    again.set_epoch(0)
    assert stream(again) == stream(ds)
    ds.set_epoch(1)
    assert stream(ds) != stream(again)  # a new epoch gives new chains/crops
    other = StructureDataset(fixture_dataset, split="train", crop_size=16, seed=1)
    other.set_epoch(0)
    assert stream(other) != stream(again)


@pytest.mark.parametrize("skip", [0, 1, 2, 5])
def test_resumed_stream_is_baseline_suffix(fixture_dataset: Path, single_process: None, skip: int) -> None:
    baseline = StructureDataset(fixture_dataset, split=None, crop_size=16, seed=3)
    baseline.set_epoch(2)
    expected = stream(baseline)
    assert len(expected) == 6
    resumed = StructureDataset(fixture_dataset, split=None, crop_size=16, seed=3)
    resumed.set_epoch(2)
    resumed.set_resume_skip(batches_in_epoch=skip, per_rank_batch=1, num_workers=1)
    assert stream(resumed) == expected[skip:]
    resumed.clear_resume_skip()
    assert stream(resumed) == expected


def test_resume_with_two_dataloader_workers_matches_batch_suffix(fixture_dataset: Path) -> None:
    collate = partial(collate_rows, crop_size=16)

    def loader(ds: StructureDataset) -> DataLoader:
        return DataLoader(ds, batch_size=1, collate_fn=collate, num_workers=2, persistent_workers=False)

    def batches(ds: StructureDataset) -> list[tuple[list[str], list[int]]]:
        return [(b.ids, b.features.residue_index[0].tolist()) for b in loader(ds)]

    baseline = StructureDataset(fixture_dataset, split=None, crop_size=16, seed=0)
    baseline.set_epoch(0)
    expected = batches(baseline)
    assert len(expected) == 6
    k = 2  # a full worker-cycle boundary (k % num_workers == 0)
    resumed = StructureDataset(fixture_dataset, split=None, crop_size=16, seed=0)
    resumed.set_epoch(0)
    resumed.set_resume_skip(batches_in_epoch=k, per_rank_batch=1, num_workers=2)
    assert batches(resumed) == expected[k:]


def test_interleaved_sources_resume(fixture_dataset: Path, single_process: None) -> None:
    def mix() -> InterleavedDataset:
        a = StructureDataset(fixture_dataset, split="train", crop_size=16, seed=0)
        b = StructureDataset(fixture_dataset, split="casp14", crop_size=16, seed=0)
        return InterleavedDataset([a, b], [0.7, 0.3], num_samples=8, seed=0)

    baseline = mix()
    baseline.set_epoch(0)
    expected = stream(baseline)
    assert len(expected) == 8
    resumed = mix()
    resumed.set_epoch(0)
    resumed.set_resume_skip(batches_in_epoch=3, per_rank_batch=1, num_workers=1)
    assert stream(resumed) == expected[3:]


def test_missing_shard_raises_and_empty_shards_are_skipped(
    fixture_dataset: Path, tmp_path: Path, single_process: None
) -> None:
    cameo = StructureDataset(fixture_dataset, split="cameo", crop_size=16)
    assert len(cameo) == 1 and [r["id"] for r in cameo] == ["1crn_A"]  # other shards have no rows
    broken = tmp_path / "broken"
    shutil.copytree(fixture_dataset, broken)
    victim = sorted((broken / "shards").glob("shard-*.parquet"))[-1]
    victim.unlink()
    with pytest.raises(FileNotFoundError, match=victim.name):
        StructureDataset(broken, split=None)
```

Append to `tests/fold/data/test_collate.py`:

```python
def test_collate_rows_unwraps_dataset_rows(parsed: Callable[[str], MolecularComplex]) -> None:
    from oplm.fold.data.collate import collate_rows

    ex = featurize_complex(parsed("1l2y"))
    batch = collate_rows([{"id": ex.id, "example": ex}], crop_size=32)
    assert batch.ids == ["1l2y_A"] and batch.features.num_tokens == 32
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/data/test_dataset.py tests/fold/data/test_collate.py -q`
Expected: FAIL with `ImportError: cannot import name 'StructureDataset'` / `'collate_rows'`.

- [ ] **Step 3: Implement the dataset and `collate_rows`**

Append to `src/oplm/fold/data/collate.py`:

```python
def collate_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    crop_size: int,
    atoms_per_token_budget: int = 24,
    atom_pad_multiple: int = 32,
) -> FoldBatch:
    """``collate_examples`` over ``StructureDataset`` rows (``{"id", "example"}`` dicts)."""
    examples = [row["example"] for row in rows]
    return collate_examples(
        examples,  # ty: ignore[invalid-argument-type]  # rows are untyped dicts by the dataset contract
        crop_size=crop_size,
        atoms_per_token_budget=atoms_per_token_budget,
        atom_pad_multiple=atom_pad_multiple,
    )
```

(add `Mapping` to the `collections.abc` type-checking import and `"collate_rows"` to `__all__`).

Add these to the import block at the top of `src/oplm/fold/data/dataset.py` (the
`complex` import extends the existing line):

```python
from oplm.data.sequence.dataset import ShardedProteinDataset
from oplm.fold.data.complex import complex_to_row, row_to_complex, rows_to_table
from oplm.fold.data.crop import crop_chain
from oplm.fold.data.featurize import featurize_complex
```

Then append:

```python
# ----------------------------------------------------------------------------- runtime

_MASK32 = 0xFFFF_FFFF


def _example_seed(seed: int, epoch: int, shard: int, row: int) -> int:
    """Seed for one example's chain choice, crop and conformer rotations."""
    mixed = (seed * 0x9E37_79B1) ^ (epoch * 0x85EB_CA77) ^ (shard * 0xC2B2_AE3D) ^ (row * 0x27D4_EB2F)
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
        super().__init__(root / "shards", shuffle_shards=shuffle_shards, shuffle_rows=shuffle_rows, seed=seed)
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
        chains_by_shard: list[dict[int, list[int]]] = [{} for _ in self._shards]
        for r in index:
            if split is not None and r["split"] != split:
                continue
            if r["length"] > max_chain_length or r["resolved_count"] < min_resolved_residues:
                continue
            if r["shard"] not in shard_pos:
                raise FileNotFoundError(f"index names shard {r['shard']} which is missing under {root}")
            chains_by_shard[shard_pos[r["shard"]]].setdefault(int(r["row"]), []).append(int(r["chain_index"]))
        self._eligible_chains = chains_by_shard
        self._eligible_rows = [sorted(d) for d in chains_by_shard]
        # The base class strides over these counts; positions map to real rows in _read_rows.
        self._rows_per_shard = [len(rows) for rows in self._eligible_rows]
        self._total_rows = sum(self._rows_per_shard)

    def _read_rows(self, shard_idx: int, row_indices: list[int]) -> Iterator[dict[str, object]]:
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
```

Add `"StructureDataset"` to `__all__`.

- [ ] **Step 4: Run the tests, lint, types**

Run: `.venv/bin/python -m pytest tests/fold/data -q && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/data/dataset.py src/oplm/fold/data/collate.py tests/fold/data/test_dataset.py tests/fold/data/test_collate.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: all pass.

- [ ] **Step 5: Run the whole fast suite**

Run: `.venv/bin/python -m pytest -m "not slow" -q -x`
Expected: everything green (M1's 2233 plus the new tests; parity tests skip without
`OPLM_FOLD_FIXTURES`).

- [ ] **Step 6: Commit**

```bash
git add src/oplm/fold/data/dataset.py src/oplm/fold/data/collate.py tests/fold/data/test_dataset.py tests/fold/data/test_collate.py
git commit -m "feat(fold): StructureDataset with inherited striping, shuffling and data-exact resume"
```

---

### Task 9: Documentation

**Files:**
- Modify: `docs/FOLD.md` (status line, §1 layout, new §8)
- Modify: `docs/TESTING_E2E.md` (the `tests/fold/` row)
- Modify: `tests/fixtures/README.md`

**Interfaces:** none; this task records the contract Tasks 1–8 implemented.

- [ ] **Step 1: Update the FOLD.md status line and §1 layout**

Replace the bold status sentence at the top with:

```markdown
**Status: milestone 2a (data) implemented; milestone 1 (inference port) accepted 2026-10-11.**
Kernels, the model, prediction and the training data pipeline exist (§7 holds the M1 acceptance
evidence, §8 the data contract); losses and training land in milestone 2b.
```

Replace the §1 tree with the current package (M1 left it at the M0 files):

```
src/oplm/fold/
├── configuration_fold.py   # FoldConfig (released ESMFold2-Fast defaults)
├── modeling_fold.py        # OplmFoldPreTrainedModel, OplmForFolding, frozen-LM ownership
├── lm_shim.py              # per-layer LN + proj, softmax layer mix, LM -> pair
├── pair.py                 # block-local pair features
├── trimul.py               # staged triangle multiplication + cuEquivariance dispatch
├── trunk.py                # pair update block, pair stack, contractive recurrence
├── atoms.py                # atom encoder/decoder, 3D RoPE, inputs embedder
├── attention.py            # pair-biased / sliding-window attention: FlexAttention + dense oracle
├── diffusion.py            # DiT, EDM schedule, sampler
├── confidence.py           # confidence + distogram heads, pTM/ipTM
├── predict.py              # fold(), mmCIF writer
├── fixtures.py             # ESMFold2 golden-fixture generator (runs in the esm venv)
├── cli.py                  # oplm fold bench-kernels | make-fixtures | predict | preprocess
└── data/
    ├── ccd.py              # residue vocabulary, reference conformers, ChemCompTable (CCD parents)
    ├── complex.py          # MolecularComplex (flat atoms/tokens/bonds) + Parquet row
    ├── mmcif.py            # gemmi mmCIF -> MolecularComplex with the §4.4 filters
    ├── crop.py             # contiguous / spatial single-chain crops
    ├── featurize.py        # ChainSpec or MolecularComplex -> FoldFeatures (+ FoldTargets)
    ├── collate.py          # FoldBatch under the atom budget
    └── dataset.py          # oplm fold preprocess + StructureDataset (resume contract)
tests/fold/                 # mirrors the above; tests/fixtures/fold/ holds real entries
```

- [ ] **Step 2: Append §8 to FOLD.md**

```markdown
## 8. Milestone 2a: training data (`oplm.fold.data`)

Implements spec §4.2–4.4, §4.7, §7.1–7.3 and §11 for the PDB source under the
[M2 addendum](superpowers/specs/2026-10-11-fold-m2-pipeline-proof-design.md) §4: whole
archive, protein chains only, single-chain training examples, published to Hugging Face.

### Representation and parsing

`MolecularComplex` (`data/complex.py`) is the flat layout of spec §4.2: atom arrays
(coordinates, resolved, element, name, charge, B-factor, atom→token), token arrays (residue
type, molecule type, asym/entity/symmetry ids, residue index, center and representative atom,
atom range), an always-present bond list (empty in v1), per-chain `ChainInfo` (label and
auth chain ids, entity, copy, full sequence, mean pLDDT for predicted data) and
`ComplexMetadata` (id, source, method, resolution, release date, dropped-entity counts). The
atoms of a token are contiguous and in the packaged template order, so training coordinates
line up with the `ref_*` features by construction (`featurize_complex` verifies the names).

`parse_mmcif` (`data/mmcif.py`, gemmi) applies §4.4 at parse time and records why an entry
was rejected (`parse_error`, `resolution`, `no_protein_chains`, `no_resolved_chain`) or what
was dropped (`water`, `ligand`, `nucleic_acid`, `other_polymer`, `short_chain`,
`empty_sequence`): L-peptide polymer entities of the asymmetric unit only; the token sequence
is the entity polymer sequence (unmodeled residues are tokens with `atom_resolved` false);
first model; highest-occupancy altloc per atom name; hydrogens dropped; microheterogeneity
keeps the sequence's first alternative; resolution ≤ 9 Å (NMR has none); chains with fewer
than 4 resolved residues dropped. Non-standard residues collapse to their CCD parent
(`ChemCompTable.protein_parent`, from `_chem_comp.mon_nstd_parent_comp_id`); atoms match the
parent template by name, with AlphaFold's `MSE` `SE → SD` convention, and parent atoms the
modified residue lacks are unresolved (norleucine loses `CD1`/`CD2`). Multi-parent and
parent-less components become `UNK`. Release date is the first
`_pdbx_audit_revision_history.revision_date`.

### Shards, index and splits (`oplm fold preprocess`)

```bash
oplm fold preprocess --archive <mmCIF root> --ccd components.cif.gz --out <dataset dir> \
    --heldout cameo=targets/cameo.txt --heldout casp14=targets/casp14.txt [--workers N] [--ids ids.txt]
```

| Output | Contents |
| --- | --- |
| `shards/shard-NNNNN.parquet` | one row per entry (`COMPLEX_SCHEMA`), ~1 GB each, row groups of 16 rows for random access |
| `index.parquet` | one row per chain: id, chain ids, chain index, entity/sym, sequence, length, resolved count, chain count, resolution, release date, method, shard, row, `split` |
| `targets/<name>.json` | the held-out target lists as `[{"id", "chain"}]` |
| `ccd_table.json`, `checksums.txt`, `manifest.json`, `README.md` | the parsed CCD table, sha256 of every input, provenance/config/counts, dataset card |

Splits are entry-level: `cameo`/`casp14` (from the target lists, ids case-insensitive),
`excluded_heldout_sequence` (pre-cutoff entries sharing a chain sequence with a held-out
entry), `train` (release date ≤ 2021-09-30), `valid` (64 seeded post-cutoff monomer chains:
one protein chain, resolution ≤ 2.5 Å, length 50–400), `post_cutoff`, `undated`. Clustering
and homology exclusions are milestone 3 (a dataset revision).

### Runtime

`StructureDataset(path, split="train", crop_size=384, contiguous_prob=0.5,
max_chain_length=1024)` subclasses `ShardedProteinDataset`: rank/worker striping, per-epoch
shuffling, `InterleavedDataset` mixing and the data-exact resume contract (`set_resume_skip`,
`_arm_sample_skip`) are inherited. It strides over the index-eligible rows of each shard and
yields exactly one example per row: a generator seeded from `(seed, epoch, shard, row)` picks
one eligible chain uniformly, crops it (`crop_chain`: contiguous window or spatial crop around
a resolved center; unresolved tokens fill a spatial crop last, by index) and featurizes it
with the full chain as the LM row (`lm_positions = residue_index + 1`) and fresh per-token
rotations of the centered reference conformers. Rows are `{"id": "<entry>_<chain>",
"example": FoldExample(features, targets)}`; `collate_rows(rows, crop_size=...)` pads tokens
to the crop and atoms to `24 × crop` rounded up to 32, stacks the LM rows of the batch and
offsets `lm_rows`, and rejects an over-budget example instead of truncating it.
`FoldTargets` carries coordinates, resolved mask, B-factor and the N/CA/C atom indices of
each token for M2b's frame-based losses.

### Acquisition, publishing, download

```bash
# wwPDB mmCIF archive (about 60 GB gzipped; read in place, never decompressed)
rsync -rlpt -v -z --delete --port=33444 rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ $OPLM_DATA/pdb/mmCIF/
# Chemical Component Dictionary
curl -o $OPLM_DATA/pdb/components.cif.gz https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
# publish (resumable) and pull on a cluster node
hf upload-large-folder <namespace>/<dataset> <dataset dir> --repo-type dataset
hf download <namespace>/<dataset> --repo-type dataset --local-dir $OPLM_DATA/fold/<dataset>
```

The manifest's `snapshot_date`, `checksums.txt` and the CCD checksum pin the inputs; the
repo revision pins the artifact a run trained on. Acceptance evidence lives in
[`docs/fold/m2a/`](fold/m2a/).
```

- [ ] **Step 3: Update TESTING_E2E.md and the fixtures README**

In `docs/TESTING_E2E.md` §1, replace the `tests/fold/` row's description with:
"Structure head: kernels, the ESMFold2 port, prediction, and the data pipeline
(`tests/fold/data/`: CCD table, `MolecularComplex`, mmCIF parsing on the real entries under
`tests/fixtures/fold/`, crops, featurization with targets, collation, preprocessing and
`StructureDataset` resume). GPU parity tests are `slow` and skip without CUDA."

Write `tests/fixtures/README.md` (currently empty):

```markdown
# Test fixtures

| Directory | Contents |
| --- | --- |
| `training/test_sequences.parquet` | Real protein sequences for the MLM data and trainer tests. |
| `eval/structures/` | PDB files for the categorical-Jacobian structure eval. |
| `fold/` | Real wwPDB mmCIF entries and a CCD subset for the structure head's data pipeline; see `fold/README.md` for provenance and why each entry is there. |
```

- [ ] **Step 4: Check rendering and commit**

Run: `.venv/bin/python -m pytest tests/fold/data -q` (nothing changed in code; confirms the tree is still green) and skim `docs/FOLD.md` for broken table rows.

```bash
git add docs/FOLD.md docs/TESTING_E2E.md tests/fixtures/README.md
git commit -m "docs(fold): milestone-2a data contract, package layout and fixture provenance"
```

---

### Task 10: Acceptance on the development machine (user-run)

**Files:**
- Create: `docs/fold/m2a/` (`manifest.json`, `preprocess.log`, `publish.txt`, `clean-env.log`, `README.md`)
- Modify: `docs/FOLD.md` §8 (one status line with the dataset revision)

**Interfaces:** consumes Tasks 1–9; produces the published dataset revision M2b and M2c train from.

This task runs outside the test suite, on the development machine (32 cores, 123 GB RAM,
1.6 TB free). Record wall-clock times for each step in `docs/fold/m2a/README.md`.

- [ ] **Step 1: Choose a data root and fetch the inputs**

```bash
export OPLM_DATA=$HOME/data            # or wherever the 1.6 TB volume is mounted
mkdir -p $OPLM_DATA/pdb $OPLM_DATA/fold
date -u +%Y-%m-%d > $OPLM_DATA/pdb/snapshot-date.txt
rsync -rlpt -v -z --delete --port=33444 rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ $OPLM_DATA/pdb/mmCIF/ 2>&1 | tail -3
# PDBe mirror if RCSB is slow: rsync -rlpt -v -z --delete rsync.ebi.ac.uk::pub/databases/pdb/data/structures/divided/mmCIF/ $OPLM_DATA/pdb/mmCIF/
curl -sS -o $OPLM_DATA/pdb/components.cif.gz https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
du -sh $OPLM_DATA/pdb/mmCIF; find $OPLM_DATA/pdb/mmCIF -name '*.cif.gz' | wc -l; sha256sum $OPLM_DATA/pdb/components.cif.gz
```

rsync is resumable: rerun the same command after an interruption. Expect about 60 GB and a
count in the low hundreds of thousands.

- [ ] **Step 2: Build the held-out target lists**

Write `$OPLM_DATA/pdb/targets/cameo.txt` and `$OPLM_DATA/pdb/targets/casp14.txt`, one
`PDBID` or `PDBID_CHAIN` per line, `#` comments allowed (`read_target_list` lower-cases ids).

- CAMEO: from the CAMEO modeling target archive (https://www.cameo3d.org/modeling/targets/),
  export the weekly target lists covering 2021-10-01 through 2022-09-30 and write one
  `PDBID_CHAIN` per target. Keep the exported files next to the list and note the export
  date in `docs/fold/m2a/README.md`.
- CASP14: from the Prediction Center target list (https://predictioncenter.org/casp14/targetlist.cgi),
  take every target with a released PDB id and write one `PDBID` per line (chain is resolved
  in M2c's eval from the target's description; the split only needs the entry).

Targets whose entry is not in the snapshot are reported by `preprocess` as missing from the
split counts (compare `targets/<name>.json` with `index.parquet` after the run); record the
count.

- [ ] **Step 3: Run preprocessing**

```bash
cd ~/git/oplm && git checkout feat/fold-m2a
time .venv/bin/oplm fold preprocess \
    --archive $OPLM_DATA/pdb/mmCIF --ccd $OPLM_DATA/pdb/components.cif.gz \
    --out $OPLM_DATA/fold/pdb-$(cat $OPLM_DATA/pdb/snapshot-date.txt) \
    --heldout cameo=$OPLM_DATA/pdb/targets/cameo.txt --heldout casp14=$OPLM_DATA/pdb/targets/casp14.txt \
    --workers 28 2>&1 | tee docs/fold/m2a/preprocess.log
```

Expected: hours, not days (gemmi parses a typical entry in tens of milliseconds; 28 workers).
Sanity-check the printed table: `accepted` close to the input count minus `rejected`
(`resolution` and `no_protein_chains` dominate), `dropped.water` close to the number of
X-ray entries, `splits.train` the large majority, `splits.valid` 64, held-out counts equal to
the number of list entries present in the snapshot. Copy `manifest.json` into
`docs/fold/m2a/` and `du -sh` the dataset directory.

- [ ] **Step 4: Verify a batch loads locally**

```bash
.venv/bin/python - <<'EOF'
import os, time, torch
from functools import partial
from torch.utils.data import DataLoader
from oplm.fold.data.collate import collate_rows
from oplm.fold.data.dataset import StructureDataset
root = f"{os.environ['OPLM_DATA']}/fold/pdb-{open(os.environ['OPLM_DATA'] + '/pdb/snapshot-date.txt').read().strip()}"
ds = StructureDataset(root, split="train", crop_size=384, seed=0)
print("train entries", len(ds), "valid chains", len(StructureDataset(root, split="valid")))
loader = DataLoader(ds, batch_size=8, num_workers=4, collate_fn=partial(collate_rows, crop_size=384))
t0 = time.time()
for i, batch in enumerate(loader):
    if i == 20: break
print("20 batches of 8 in", round(time.time() - t0, 1), "s;", batch.features.res_type.shape, batch.features.ref_pos.shape, batch.targets.coords.shape)
EOF
```

Record the throughput line in `docs/fold/m2a/README.md`.

- [ ] **Step 5: Publish**

```bash
hf auth login            # once; a write token
hf upload-large-folder <namespace>/<dataset> $OPLM_DATA/fold/pdb-<snapshot> --repo-type dataset 2>&1 | tail -5
hf repo tag create <namespace>/<dataset> v1-<snapshot> --repo-type dataset
```

The upload is resumable; rerun the same command after an interruption. Write the repo id,
tag and resulting revision sha (`hf repo info <repo> --repo-type dataset` or the web UI) to
`docs/fold/m2a/publish.txt`.

- [ ] **Step 6: Clean-environment load**

```bash
uv venv --python 3.11 /tmp/oplm-clean && uv pip install --python /tmp/oplm-clean/bin/python "oplm[fold] @ git+https://github.com/briney/oplm@$(git rev-parse HEAD)"
hf download <namespace>/<dataset> --repo-type dataset --revision v1-<snapshot> --local-dir /tmp/pdb-check 2>&1 | tail -2
/tmp/oplm-clean/bin/python - <<'EOF' 2>&1 | tee docs/fold/m2a/clean-env.log
from functools import partial
from torch.utils.data import DataLoader
from oplm.fold.data.collate import collate_rows
from oplm.fold.data.dataset import StructureDataset
ds = StructureDataset("/tmp/pdb-check", split="train", crop_size=384)
batch = next(iter(DataLoader(ds, batch_size=4, collate_fn=partial(collate_rows, crop_size=384))))
print(len(ds), batch.ids, batch.features.res_type.shape, batch.targets.resolved.sum().item())
EOF
```

Expected: the entry count matches the local run and one batch prints finite shapes.

- [ ] **Step 7: Record and commit**

Write `docs/fold/m2a/README.md` with: machine, dates, snapshot date, archive size and file
count, CCD checksum, preprocess wall time and the counts table, dataset size, local
throughput line, repo id/tag/revision, clean-env result, and the count of held-out targets
missing from the snapshot. Add one line to FOLD.md §8 naming the published revision.

```bash
git add docs/fold/m2a docs/FOLD.md
git commit -m "docs(fold): record the milestone-2a preprocessing run and the published dataset revision"
```

Then open the PR for `feat/fold-m2a` against `main` (the addendum commit on
`docs/fold-m2-spec` rides along as its base, or merge that branch first).

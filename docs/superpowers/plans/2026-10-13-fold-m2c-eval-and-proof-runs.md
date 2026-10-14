# Structure Head Milestone 2c — Evaluation and the Proof Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close milestone 2: held-out monomer metrics (TM-score, CA and all-atom lDDT), a
`fold_monomer` eval task for in-training validation, `oplm fold eval` for offline scoring of
saved checkpoints on CAMEO and CASP14, Slurm rendering for fold runs. Then run the two 8× B200
proof runs (pair width 128 and 256) with a forced resume, evaluate every kept checkpoint, and
record the width decision.

**Architecture:** A new `oplm.fold.eval` package holds four layers:

- **`metrics.py`:** pure metric math. It is checked against TM-align (`tmtools`) and the
  vendored Boltz-2 lDDT.
- **`targets.py`:** reads full chains and their ground truth from the published dataset's
  index, shards and target lists.
- **`monomer.py`:** folds one target with `fold()`, scores it, records failures, and
  aggregates, stores and compares score tables.
- **`tasks.py`:** registers `fold_monomer` in the existing eval registry.

`EvalTask` becomes generic over the model class, so the MLM tasks keep `OplmForMaskedLM` and the
fold task takes `OplmForFolding`. `oplm.fold.training` imports the package, so every fold entry
point can name the type, and MLM tooling never imports fold code.

The other pieces:
- **`oplm fold eval`** scores a checkpoint's EMA (or raw) weights offline and writes a
  per-target CSV plus a JSON summary. The summary carries coverage, failures and the sampling
  settings.
- **`oplm fold compare`** gives the paired, bootstrapped difference between two such tables.
- **`oplm slurm generate`** launches `oplm.fold.train` for a config with a `fold:` block, and
  refuses a relative `output_dir`.
- **The proof configs** gain a `slurm:` block that builds a Python 3.12 venv with `esm`, plus the
  in-training eval entry.

**Tech Stack:** Python 3.11 dev venv, torch 2.11, numpy, pyarrow, accelerate, typer. `tmtools`
0.3.0 (GPLv3) serves as an optional test oracle only: it is not listed in `pyproject.toml`, and
its test skips when it is absent. Cluster jobs use a Python 3.12 venv with
`oplm[train,fold,esmc]`.

**Spec:** [M2 pipeline-proof addendum](../specs/2026-10-11-fold-m2-pipeline-proof-design.md)
§6 (M2c), §2 (decisions), §8 (gates), over the
[structure prediction head design](../specs/2026-10-07-structure-prediction-head-design.md)
§6.4 (lazy eval registration, generalized annotation), §8 (evaluation), §9 (metrics row),
§10 (milestone 2 acceptance and decision gates). Measured inputs:
[docs/fold/m2/README.md](../../fold/m2/README.md) (milestone-2c budget).

**Planning baseline:** `8238a23` on `main` (milestone 2b merged as PR #35). Work on branch
`feat/fold-m2c` in the worktree `.claude/worktrees/fold-m2c`, which already has its own `.venv`
(`uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev,train]"`;
`tmtools` installed by hand for the oracle test). Tasks 1–7 need no GPU and form one PR.
Tasks 8–9 run on the cluster after that PR merges, and their evidence goes in on a second
branch.

## Global Constraints

- `AGENTS.md` rules:
  - Python 3.11+, with `from __future__ import annotations` in every file.
  - Type hints on every signature and Google-style docstrings on public APIs.
  - Ruff line length 100; `pathlib.Path` only.
  - No logic in `__init__.py` beyond imports.
  - Tests mirror the source layout; `@pytest.mark.slow` for anything over a few seconds.
- Commands, all clean before every commit:
  - `.venv/bin/python -m pytest ...`
  - `.venv/bin/ruff check src/`
  - `.venv/bin/ruff format <changed files only>`
  - `VIRTUAL_ENV=.venv .venv/bin/ty check src/`

  Silence framework-boundary diagnostics only with a specific `# ty: ignore[<rule>]` and a reason.
- No new dependency in `pyproject.toml`. `tmtools` is GPLv3 and this repo is MIT, so it stays a
  hand-installed test oracle (`pytest.importorskip("tmtools")`). Nothing under `src/` imports it.
- Dependency direction (base spec §3, §6.4):
  - Core packages (`oplm.eval`, `oplm.training`, `oplm.config`, `oplm.slurm`) import nothing
    from `oplm.fold`.
  - `oplm.fold.eval` imports core packages.
  - `oplm.fold.cli` keeps torch out of module scope.
  - Importing `oplm.eval` or `oplm.eval.tasks` must not import any `oplm.fold` module (Task 4
    pins this).
- Addendum §6 metric conventions:
  - **Correspondence:** residue `i` of the prediction is residue `i` of the reference. Both come
    from the same entity sequence and the same `ReferenceConformers` atom templates.
  - **Masking:** an atom unresolved in the reference is excluded, never scored.
  - **TM-score:** over resolved CA atoms. Normalized by their count. `d0 = 1.24·(L−15)^(1/3) − 1.8`
    for L > 21, else 0.5, and at least 0.5. Uses the TM-score program's fragment-seeded iterative
    search: seed lengths L, L/2, … down to 4 (at most six), selection cutoff `d0_search − 1` then
    `d0_search + 1`, `d0_search = clamp(d0, 4.5, 8)`, and up to 20 iterations.
  - **lDDT:** global form. Reference pairs within 15 Å, thresholds 0.5/1/2/4 Å. CA-lDDT over CA
    atoms; all-atom lDDT over all resolved atoms, excluding pairs inside one residue.
  - **pLDDT:** the mean over residues, on the 0–1 scale. pTM is that of the best sample.
- Spec §8 and addendum §6 reporting:
  - Full-chain scores only; CASP domain scoring is not reproduced.
  - Per-target rows, aggregates, coverage (listed, missing, skipped with reasons), failures and
    the sampling metadata (loops, steps, samples, seed, model path, LM, device, versions) are
    written for every offline evaluation.
- Target rules (addendum §6, docs/fold/m2a/README.md):
  - A target set is `targets/<name>.json` when that file exists, otherwise a split of
    `index.parquet`.
  - A list item with a chain names an **author** chain (`auth_chain_id`).
  - An item without a chain (every CASP14 item) contributes one chain per distinct entity: the
    lowest `chain_index`.
  - Chains longer than `max_length` are skipped. The default is the model's
    `lm_max_chain_length`, 1024 in the proof runs. Chains with fewer than 3 resolved CA atoms
    are skipped. Both skips are reported.
- Training-time evaluation:
  - `fold_monomer` scores the `valid` split (64 single-chain entries, 75–391 residues) with the
    live model.
  - The Trainer never passes the EMA copy. EMA weights are scored offline from each kept
    checkpoint's `hf_ema/`.
  - One failing target (an exception, CUDA OOM included) becomes a `failed` row. It never
    raises, so every rank reaches the gather.
- Run shape (addendum §2, docs/fold/m2/README.md):
  - One node, 8× B200, DDP, global batch 128, 50k steps, EMA 0.999.
  - Width 128 at batch 8 × accumulation 2; width 256 at batch 4 × accumulation 4.
  - Checkpoints every 2000 steps; permanent keeps every 10000; in-training eval every 2000
    steps.
  - At least one resume per run forced through the drain path (`scancel --signal=USR1`) and
    verified from the logs.

## Review Focus

1. **A reference chain with unresolved residues.**
   - Scores use only resolved atoms, and TM-score normalizes by the resolved CA count, not the
     sequence length.
   - The fold still runs on the full sequence.
   - Pinned by Task 2 `test_split_targets_keep_unresolved_residues` and Task 3
     `test_score_result_ignores_unresolved_reference_atoms`.
2. **A target that cannot be scored.** Examples: longer than `max_length`, fewer than 3 resolved
   CA atoms, or listed but absent from the snapshot. It is reported in coverage with a reason;
   it never crashes the run or silently vanishes. Pinned by Task 2
   `test_entry_items_take_one_chain_per_entity_and_report_coverage` and
   `test_max_length_and_limit`.
3. **`fold()` raising for one target mid-evaluation** (OOM, layout mismatch). The row is
   recorded as failed with the error, the remaining targets are scored, and the aggregate counts
   the failure. Pinned by Task 3 `test_score_target_records_failures` and Task 4
   `test_evaluate_counts_a_failed_target`.
4. **More ranks than targets, or an uneven split.** Each rank scores
   `targets[rank::world]` (possibly none), every rank calls the gather, and the aggregate covers
   all targets exactly once. Pinned by Task 4 `test_evaluate_shards_targets_across_ranks`.
5. **A Slurm job whose `output_dir` is relative.** Inside the container it would resolve under
   the job's node-local `$JOB_WORK_DIR`, so checkpoints vanish at job end and auto-resume finds
   nothing. `generate` must refuse it. Pinned by Task 6
   `test_generate_rejects_a_relative_output_dir`.

## Decisions taken in this plan

| Topic | Decision | Why |
|---|---|---|
| Generalizing `EvalTask.evaluate`'s model annotation (base spec §6.4) | `EvalTask(ABC, Generic[ModelT])`, `ModelT` bound to `nn.Module`; each MLM task subclasses `EvalTask["OplmForMaskedLM"]`, the fold task `EvalTask["OplmForFolding"]` | Annotating the base with `nn.Module` makes `ty` report all seven MLM overrides as Liskov violations (checked while planning); the generic form is clean under `ty` and changes one line per subclass |
| CASP14 items have no chain | One chain per distinct entity, lowest `chain_index`; seq-partial entries included | 131 present entries hold 524 chains but 248 entities; homo-oligomer copies would score the identical prediction several times. Every listed entry is held out of training, so all are valid monomer targets; coverage reports the counts |
| Which weights are scored offline | `hf_ema/` of every permanent checkpoint (10k … 50k) | EMA is the stage-boundary artifact (base spec §6.5); the live model's curve comes from the in-training `valid` eval every 2000 steps |
| Sampling for eval | The checkpoint's `inference_num_loops` (10) and `inference_num_steps` (14), 1 sample, seed 0 | Fixed and recorded; cheap enough for ~1000 targets per checkpoint |
| `tmtools` | Hand-installed oracle, never a declared dependency | GPLv3 against the repo's MIT license; on identity alignments it matched the planned TM-score to machine precision (about 1e-16) across three fixtures and three noise levels |
| Cluster install | Per-run Python 3.12 venv on `/mnt/data`, `oplm[train,fold,esmc]` from the immutable tag `fold-m2c-runs`, one resolve, then `uv pip check` | esm needs Python ≥ 3.12 and transformers < 5; a single resolve keeps transformers in range (milestone-2b run 1, failure 3); a tag keeps every requeue on identical code |
| Width decision rule (preregistered) | Choose 256 only if, at 50k steps on the pooled CAMEO + CASP14 targets, the paired EMA TM-score gain of 256 over 128 has a bootstrap 95 % CI entirely above 0.01; otherwise 128 | 256 costs 1.69× per optimizer step on a B200 (17.31 s against 10.22 s) and more in the M0 table; a gain must be real and non-trivial to pay for it |

## File Map

| File | Responsibility |
|---|---|
| New `src/oplm/fold/eval/__init__.py` | Imports `tasks` (registers `fold_monomer`) |
| New `src/oplm/fold/eval/metrics.py` | `tm_d0`, `tm_score`, `lddt` |
| New `src/oplm/fold/eval/targets.py` | `EvalTarget`, `TargetCoverage`, `load_targets` |
| New `src/oplm/fold/eval/monomer.py` | `METRICS`, `SamplingSettings`, `TargetScore`, `score_result`, `score_target`, `summarize`, `write_scores`, `read_scores`, `compare_scores` |
| New `src/oplm/fold/eval/tasks.py` | `FoldMonomerTaskConfig`, `FoldMonomerTask` (`fold_monomer`) |
| `src/oplm/eval/tasks/base.py` | `EvalTask` generic over `ModelT` |
| `src/oplm/eval/tasks/{sequence,structure,proteingym,proteingym_clinical,tape,proteinglue,everest}.py` | Subclass `EvalTask["OplmForMaskedLM"]` |
| `src/oplm/fold/training.py` | `import oplm.fold.eval` (registration) |
| `src/oplm/fold/cli.py` | + `eval`, `compare` commands; `measure` drops `data.eval` |
| `src/oplm/slurm/cli.py` | `generate`: fold entry module, absolute `output_dir` |
| `configs/fold/proof-w{128,256}.yaml` | Absolute `output_dir`, `wandb_run_name`, `data.eval.valid`, `slurm:` block |
| New `docs/fold/b200-m2c-eval.sbatch` | Offline eval of every kept checkpoint of both runs |
| `docs/FOLD.md` | New §10 (evaluation and the proof runs); status line |
| `docs/SLURM.md` | Fold entry, absolute `output_dir`, forcing a drain |
| New tests | `tests/fold/eval/{__init__,test_metrics,test_targets,test_monomer,test_tasks}.py` |
| Modified tests | `tests/fold/test_cli.py`, `tests/slurm/test_cli.py` |
| Task 8–9 evidence | `docs/fold/m2/runs/`, `docs/fold/m2/eval/`, `docs/fold/m2/README.md` |

---

### Task 1: Metrics — TM-score and lDDT with explicit correspondence

**Files:**
- Create: `src/oplm/fold/eval/__init__.py`, `src/oplm/fold/eval/metrics.py`
- Test: `tests/fold/eval/__init__.py` (empty), `tests/fold/eval/test_metrics.py`

**Interfaces:**
- Produces:
  - `tm_d0(length: int) -> float`
  - `tm_score(pred: ArrayLike, ref: ArrayLike, mask: ArrayLike | None = None) -> float`
  - `lddt(pred, ref, mask=None, *, residue_index=None, cutoff=15.0, chunk=2048) -> float`
- Inputs are (N, 3) in Å as numpy arrays or tensors. Both functions raise `ValueError` on a
  shape mismatch, on fewer than 3 kept rows (TM-score), or when no pair qualifies (lDDT).

In this task `__init__.py` holds only the module docstring and `from __future__ import
annotations`. Task 4 adds the registration import.

- [ ] **Step 1: Write the failing tests**

`tests/fold/eval/test_metrics.py`:

```python
"""TM-score and lDDT against identity, rigid motion, a hinge, TM-align and Boltz-2 (spec §9)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from oplm.fold.eval.metrics import lddt, tm_d0, tm_score
from tests.fold.oracles.boltz2_confidence import lddt_dist

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.fold.data.complex import MolecularComplex


def _chain(entry: Callable[[str], MolecularComplex], name: str):
    c = entry(name)
    return c.select_tokens(c.chain_tokens(0)), c.chains[0].sequence


def _random_walk(n: int, seed: int) -> np.ndarray:
    steps = np.random.default_rng(seed).normal(size=(n, 3))
    return np.cumsum(3.8 * steps / np.linalg.norm(steps, axis=1, keepdims=True), axis=0)


def _rigid(x: np.ndarray, seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    return x @ q.T + np.array([5.0, -3.0, 12.0])


def test_tm_d0_follows_the_tm_score_program() -> None:
    assert tm_d0(10) == 0.5 and tm_d0(21) == 0.5
    assert tm_d0(100) == pytest.approx(1.24 * 85 ** (1 / 3) - 1.8)
    assert tm_d0(22) == pytest.approx(max(1.24 * 7 ** (1 / 3) - 1.8, 0.5))


def test_identity_and_rigid_motion_score_one() -> None:
    x = _random_walk(60, seed=0)
    moved = _rigid(x, seed=1)
    assert tm_score(x, x) == pytest.approx(1.0, abs=1e-12)
    assert tm_score(moved, x) == pytest.approx(1.0, abs=1e-9)
    assert lddt(x, x) == 1.0 and lddt(moved, x) == pytest.approx(1.0)


def test_iterative_search_beats_one_global_superposition() -> None:
    chain = _random_walk(90, seed=3)
    theta = np.pi / 2
    rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    hinged = chain.copy()
    hinged[60:] = (chain[60:] - chain[59]) @ rz.T + chain[59]
    # One RMSD-optimal fit over all 90 rows, scored with the same d0.
    mc, mh = chain.mean(0), hinged.mean(0)
    u, _, vt = np.linalg.svd((hinged - mh).T @ (chain - mc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    fitted = (hinged - mh) @ rot.T + mc
    single = float((1 / (1 + ((fitted - chain) ** 2).sum(1) / tm_d0(90) ** 2)).sum()) / 90
    ours = tm_score(hinged, chain)
    assert ours > single + 0.2  # 0.70 against 0.32 when planned
    assert ours > 60 / 90 - 0.05


def test_mask_drops_rows_and_sets_the_normalization_length() -> None:
    x = _random_walk(50, seed=4)
    pred = x + np.random.default_rng(5).normal(scale=1.0, size=x.shape)
    pred[30:] += 40.0  # unresolved in the reference: must not count at all
    mask = np.arange(50) < 30
    assert tm_score(pred, x, mask) == pytest.approx(tm_score(pred[:30], x[:30]))
    assert lddt(pred, x, mask) == pytest.approx(lddt(pred[:30], x[:30]))


def test_tm_score_matches_tm_align_on_identity_alignments(parsed) -> None:
    tmtools = pytest.importorskip("tmtools")
    for name in ("1crn", "2ds7", "4hhb"):
        sub, seq = _chain(parsed, name)
        ca = sub.token_center_atom
        ref = sub.atom_coords[ca].astype(np.float64)
        keep = sub.atom_resolved[ca]
        kept_seq = "".join(s for s, k in zip(seq, keep, strict=True) if k)
        for i, sigma in enumerate((0.3, 0.8, 1.5)):
            pred = ref + np.random.default_rng(i).normal(scale=sigma, size=ref.shape)
            result = tmtools.tm_align(pred[keep], ref[keep], kept_seq, kept_seq)
            assert "-" not in result.seqxA and "-" not in result.seqyA  # identity alignment
            assert tm_score(pred, ref, keep) == pytest.approx(result.tm_norm_chain2, abs=1e-6)


def test_lddt_matches_boltz2_lddt_dist(parsed) -> None:
    for name in ("1crn", "2ds7"):
        sub, _ = _chain(parsed, name)
        ref = torch.tensor(sub.atom_coords, dtype=torch.float32)
        keep = torch.tensor(sub.atom_resolved)
        res = torch.tensor(sub.atom_to_token)
        noise = torch.randn(ref.shape, generator=torch.Generator().manual_seed(1))
        pred = ref + 0.7 * noise
        pair = keep[:, None] & keep[None, :] & (res[:, None] != res[None, :])
        oracle, _ = lddt_dist(torch.cdist(pred, pred), torch.cdist(ref, ref), pair.float())
        assert lddt(pred, ref, keep, residue_index=res) == pytest.approx(float(oracle), abs=1e-6)
        assert lddt(pred, ref, keep, residue_index=res, chunk=37) == pytest.approx(float(oracle), abs=1e-6)
        ca = torch.as_tensor(sub.token_center_atom, dtype=torch.long)
        pair_ca = keep[ca][:, None] & keep[ca][None, :] & ~torch.eye(len(ca), dtype=torch.bool)
        oracle_ca, _ = lddt_dist(
            torch.cdist(pred[ca], pred[ca]), torch.cdist(ref[ca], ref[ca]), pair_ca.float()
        )
        assert lddt(pred[ca], ref[ca], keep[ca]) == pytest.approx(float(oracle_ca), abs=1e-6)


def test_invalid_inputs_raise() -> None:
    x = _random_walk(10, seed=6)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        tm_score(x, x[:9])
    with pytest.raises(ValueError, match="at least 3"):
        tm_score(x, x, np.arange(10) < 2)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        lddt(x[:, :2], x[:, :2])
    far = np.array([[0.0, 0, 0], [100.0, 0, 0], [0, 100.0, 0]])
    with pytest.raises(ValueError, match="no reference pairs"):
        lddt(far, far)
```

`parsed` is the session fixture in `tests/fold/conftest.py`. The `tests/fold/eval/` package
needs an empty `__init__.py`, as `tests/fold/data/` has.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/eval/test_metrics.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'oplm.fold.eval'`.

- [ ] **Step 3: Implement `metrics.py`**

`src/oplm/fold/eval/__init__.py`:

```python
"""Held-out monomer evaluation for the structure head (addendum §6, spec §8; docs/FOLD.md §10)."""

from __future__ import annotations
```

`src/oplm/fold/eval/metrics.py`. Planning ran this code against TM-align and Boltz-2, with the
results quoted in the tests:

```python
"""Held-out structure metrics: TM-score and lDDT with explicit correspondence (addendum §6).

Both take predicted and reference coordinates in the same atom order (the correspondence is the
row index) and a reference mask: a row unresolved in the reference is excluded, never scored as
wrong. TM-score follows the TM-score program (Zhang & Skolnick 2004): normalized by the number of
kept reference rows, ``d0`` from that length, and the program's fragment-seeded iterative
superposition search rather than one RMSD-optimal fit. lDDT is the global score of Mariani et al.
2013: reference pairs closer than 15 Å, thresholds 0.5/1/2/4 Å and, given a residue map, no pairs
inside one residue; the OpenStructure tool's stereochemistry checks are not reproduced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = ["lddt", "tm_d0", "tm_score"]

_LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
_TM_ITERATIONS = 20  # TM-score's n_it


def tm_d0(length: int) -> float:
    """TM-score's distance scale in Å for a reference of ``length`` residues (at least 0.5)."""
    if length <= 21:
        return 0.5
    return max(1.24 * (length - 15) ** (1.0 / 3.0) - 1.8, 0.5)


def _fragment_lengths(n: int) -> list[int]:
    """TM-score's seed fragment lengths: n, n/2, n/4, ... down to min(4, n), at most six."""
    shortest = min(4, n)
    lengths: list[int] = []
    for i in range(5):
        length = int(n / 2**i)
        if length <= shortest:
            lengths.append(shortest)
            return lengths
        lengths.append(length)
    lengths.append(shortest)
    return lengths


def _superposed_sq_dist(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Squared distances (B, N) after superposing ``x`` onto ``y`` on each row set of ``w``.

    Batched Kabsch with 0/1 weights: ``x`` and ``y`` are (N, 3), ``w`` is (B, N).
    """
    wsum = w.sum(axis=1, keepdims=True)
    mx = (w @ x) / wsum
    my = (w @ y) / wsum
    xc = x[None] - mx[:, None]
    yc = y[None] - my[:, None]
    h = np.einsum("bn,bni,bnj->bij", w, xc, yc)
    u, _, vt = np.linalg.svd(h)
    v = vt.transpose(0, 2, 1)
    ut = u.transpose(0, 2, 1)
    d = np.sign(np.linalg.det(v @ ut))
    d[d == 0] = 1.0
    flip = np.ones((len(w), 3))
    flip[:, 2] = d
    rot = v @ (flip[:, :, None] * ut)  # xc @ rot.T best matches yc
    moved = xc @ rot.transpose(0, 2, 1) + my[:, None]
    return ((moved - y[None]) ** 2).sum(axis=-1)


def _select(sq_dist: np.ndarray, cutoff: float) -> np.ndarray:
    """Rows closer than ``cutoff``; a set of fewer than 3 relaxes it in 0.5 Å steps (TM-score)."""
    dist = np.sqrt(sq_dist)
    keep = dist < cutoff
    few = keep.sum(axis=1) < 3
    if few.any():
        if dist.shape[1] > 3:
            third = np.sort(dist[few], axis=1)[:, 2]
            steps = np.floor((third - cutoff) / 0.5) + 1.0
            keep[few] = dist[few] < (cutoff + 0.5 * steps)[:, None]
        else:  # ponytail: TM-score aligns on the 1-2 rows left; with 3 rows we use all three
            keep[few] = True
    return keep.astype(np.float64)


def tm_score(pred: ArrayLike, ref: ArrayLike, mask: ArrayLike | None = None) -> float:
    """TM-score of ``pred`` against ``ref`` over corresponding rows (e.g. one CA per residue).

    Args:
        pred: Predicted coordinates (N, 3) in Å.
        ref: Reference coordinates (N, 3) in Å; row ``i`` corresponds to ``pred[i]``.
        mask: Rows present in the reference (N,); the normalization length is their count.

    Raises:
        ValueError: The shapes differ or are not (N, 3), or fewer than 3 rows are kept.
    """
    x = np.asarray(torch.as_tensor(pred).detach().cpu(), dtype=np.float64)
    y = np.asarray(torch.as_tensor(ref).detach().cpu(), dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"pred and ref must both be (N, 3); got {x.shape} and {y.shape}")
    if mask is not None:
        keep = np.asarray(torch.as_tensor(mask).detach().cpu(), dtype=bool)
        x, y = x[keep], y[keep]
    n = len(x)
    if n < 3:
        raise ValueError(f"TM-score needs at least 3 kept rows, got {n}")
    d0 = tm_d0(n)
    d0_search = min(max(d0, 4.5), 8.0)
    positions = np.arange(n)
    best = 0.0
    for frag in _fragment_lengths(n):
        starts = np.arange(n - frag + 1)[:, None]
        w = ((positions >= starts) & (positions < starts + frag)).astype(np.float64)
        sq = _superposed_sq_dist(x, y, w)
        best = max(best, float((1.0 / (1.0 + sq / d0**2)).sum(axis=1).max()) / n)
        w = _select(sq, d0_search - 1.0)
        for _ in range(_TM_ITERATIONS):
            sq = _superposed_sq_dist(x, y, w)
            best = max(best, float((1.0 / (1.0 + sq / d0**2)).sum(axis=1).max()) / n)
            w_next = _select(sq, d0_search + 1.0)
            if np.array_equal(w_next, w):
                break
            w = w_next
    return best


def lddt(
    pred: ArrayLike,
    ref: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    residue_index: ArrayLike | None = None,
    cutoff: float = 15.0,
    chunk: int = 2048,
) -> float:
    """Global lDDT of ``pred`` against ``ref`` over corresponding rows (atoms).

    A pair (i, j) is scored when i != j, both rows are kept by ``mask``, the reference distance is
    below ``cutoff`` and, if ``residue_index`` is given, the rows belong to different residues. The
    score is the mean over scored pairs and the four thresholds of |d_pred - d_ref| < threshold.

    Args:
        pred: Predicted coordinates (N, 3) in Å.
        ref: Reference coordinates (N, 3) in Å.
        mask: Rows present in the reference (N,).
        residue_index: Residue of each row (N,); pairs inside one residue are skipped.
        cutoff: Inclusion radius in Å, on reference distances.
        chunk: Rows per block of the pairwise computation (memory is chunk × N).

    Raises:
        ValueError: The shapes differ or are not (N, 3), or no pair qualifies.
    """
    x = torch.as_tensor(pred).detach().float()
    y = torch.as_tensor(ref).detach().float().to(x.device)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(
            f"pred and ref must both be (N, 3); got {tuple(x.shape)} and {tuple(y.shape)}"
        )
    keep = torch.ones(len(x), dtype=torch.bool, device=x.device)
    if mask is not None:
        keep = torch.as_tensor(mask, device=x.device).bool()
    x, y = x[keep], y[keep]
    res = None
    if residue_index is not None:
        res = torch.as_tensor(residue_index, device=x.device)[keep]
    n = len(y)
    total = torch.zeros((), dtype=torch.float64, device=x.device)
    count = 0
    for start in range(0, n, chunk):
        rows = torch.arange(start, min(start + chunk, n), device=x.device)
        mode = "donot_use_mm_for_euclid_dist"  # exact distances: no matmul expansion error
        d_ref = torch.cdist(y[rows], y, compute_mode=mode)
        d_pred = torch.cdist(x[rows], x, compute_mode=mode)
        pair = d_ref < cutoff
        pair[torch.arange(len(rows), device=x.device), rows] = False
        if res is not None:
            pair &= res[rows, None] != res[None, :]
        err = (d_pred - d_ref).abs()
        preserved = sum((err < t).float() for t in _LDDT_THRESHOLDS) / len(_LDDT_THRESHOLDS)
        total += preserved[pair].double().sum()
        count += int(pair.sum())
    if count == 0:
        raise ValueError("lDDT has no reference pairs within the inclusion radius")
    return float(total / count)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/fold/eval/test_metrics.py -q`
Expected: 7 passed. `test_tm_score_matches_tm_align_on_identity_alignments` skips if `tmtools`
is absent; install it with `uv pip install --python .venv/bin/python tmtools` and re-run, and it
must pass. A run of 1024 residues takes about 3.5 s for TM-score and 0.4 s for all-atom lDDT on
CPU.

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/eval tests/fold/eval
VIRTUAL_ENV=.venv .venv/bin/ty check src/
git add src/oplm/fold/eval tests/fold/eval
git commit -m "feat(fold): TM-score and lDDT with explicit correspondence"
```

---

### Task 2: Held-out targets from the published dataset

**Files:**
- Create: `src/oplm/fold/eval/targets.py`
- Test: `tests/fold/eval/test_targets.py`

**Interfaces:**
- Consumes:
  - `row_to_complex`, plus `MolecularComplex.select_tokens` and `.chain_tokens`
    (`oplm.fold.data.complex`);
  - the dataset layout `index.parquet`, `manifest.json` (`row_group_size`), `shards/` and
    `targets/<name>.json`, whose items look like `[{"id": "<lower>", "chain": "<auth id>" | null}]`.
- Produces:
  - `EvalTarget`, a frozen dataclass with fields `target_set`, `entry_id`, `chain_id`,
    `auth_chain_id`, `sequence`, `coords (A,3) float32`, `resolved (A,) bool`,
    `atom_names list[str]`, `residue_index (A,) int64` and `ca_index (L,) int64`, plus a
    property `key -> "<entry>_<chain>"`.
  - `TargetCoverage`, a dataclass with fields `target_set`, `listed`, `missing: list[str]`,
    `skipped: dict[str, str]` and `selected`.
  - `load_targets(root: str | Path, target_set: str, *, max_length: int | None = None,
    limit: int | None = None) -> tuple[list[EvalTarget], TargetCoverage]`. Targets are sorted by
    `(entry_id, chain_id)`. Raises `ValueError` for an unknown set (naming the splits), or when a
    chain's token count differs from its sequence length.

- [ ] **Step 1: Write the failing tests**

`tests/fold/eval/test_targets.py`:

```python
"""Target sets from the preprocessed fixtures: lists, splits, entity rule, coverage (addendum §6)."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import numpy as np
import pytest

from oplm.fold.eval.targets import load_targets

if TYPE_CHECKING:
    from pathlib import Path


def _with_list(dataset: Path, tmp_path: Path, name: str, items: list[dict]) -> Path:
    root = tmp_path / "ds"
    shutil.copytree(dataset, root)
    (root / "targets" / f"{name}.json").write_text(json.dumps(items))
    return root


def test_list_targets_pick_the_named_author_chain(fixture_dataset: Path) -> None:
    targets, coverage = load_targets(fixture_dataset, "cameo")
    assert [t.key for t in targets] == ["1crn_A"]
    t = targets[0]
    assert t.target_set == "cameo" and t.auth_chain_id == "A" and len(t.sequence) == 46
    assert t.coords.shape == (len(t.atom_names), 3) and t.coords.dtype == np.float32
    assert [t.atom_names[i] for i in t.ca_index] == ["CA"] * 46
    assert t.resolved.all() and int(t.residue_index[-1]) == 45
    assert (coverage.listed, coverage.selected, coverage.missing, coverage.skipped) == (1, 1, [], {})


def test_split_targets_keep_unresolved_residues(fixture_dataset_early_cutoff: Path) -> None:
    targets, coverage = load_targets(fixture_dataset_early_cutoff, "valid")
    (t,) = targets
    assert t.key == "2ds7_A" and len(t.sequence) == 51 and len(t.ca_index) == 51
    assert int(t.resolved[t.ca_index].sum()) == 40  # 11 residues unmodeled, still tokens
    assert np.all(t.coords[~t.resolved] == 0.0)
    assert coverage.listed == 1 and coverage.selected == 1


def test_entry_items_take_one_chain_per_entity_and_report_coverage(
    fixture_dataset: Path, tmp_path: Path
) -> None:
    items = [
        {"id": "1jb6", "chain": None},  # homodimer: one entity -> chain A only
        {"id": "4hhb", "chain": None},  # two entities -> A (alpha) and B (beta)
        {"id": "9zzz", "chain": None},  # not in the snapshot
        {"id": "1crn", "chain": "Q"},  # no such author chain
    ]
    targets, coverage = load_targets(_with_list(fixture_dataset, tmp_path, "entries", items), "entries")
    assert [t.key for t in targets] == ["1jb6_A", "4hhb_A", "4hhb_B"]
    assert coverage.listed == 4 and coverage.selected == 3
    assert coverage.missing == ["9zzz", "1crn_Q"]


def test_max_length_and_limit(fixture_dataset: Path, tmp_path: Path) -> None:
    items = [{"id": "1jb6", "chain": None}, {"id": "4hhb", "chain": None}, {"id": "1crn", "chain": "A"}]
    root = _with_list(fixture_dataset, tmp_path, "entries", items)
    targets, coverage = load_targets(root, "entries", max_length=100)
    assert [t.key for t in targets] == ["1crn_A", "1jb6_A"]
    assert coverage.skipped == {
        "4hhb_A": "length 141 > max_length 100",
        "4hhb_B": "length 146 > max_length 100",
    }
    limited, _ = load_targets(root, "entries", max_length=100, limit=1)
    assert [t.key for t in limited] == ["1crn_A"]


def test_unknown_target_set_names_the_splits(fixture_dataset: Path) -> None:
    with pytest.raises(ValueError, match=r"unknown target set 'cmaeo'.*train"):
        load_targets(fixture_dataset, "cmaeo")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/eval/test_targets.py -q`
Expected: `ModuleNotFoundError: No module named 'oplm.fold.eval.targets'`.

- [ ] **Step 3: Implement `targets.py`**

```python
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
    "id", "chain_id", "auth_chain_id", "entity_id", "chain_index", "split", "length", "shard", "row"
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


def _read_targets(
    root: Path, rows: list[dict[str, Any]], target_set: str
) -> Iterator[EvalTarget]:
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
```

`ruff format` will expand `_INDEX_COLUMNS` to one item per line. Keep whatever it produces.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/fold/eval/test_targets.py -q`
Expected: 5 passed.

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/eval/targets.py tests/fold/eval/test_targets.py
VIRTUAL_ENV=.venv .venv/bin/ty check src/
git add src/oplm/fold/eval/targets.py tests/fold/eval/test_targets.py
git commit -m "feat(fold): held-out target sets from the published dataset"
```

---

### Task 3: Fold, score, aggregate, store and compare

**Files:**
- Create: `src/oplm/fold/eval/monomer.py`
- Test: `tests/fold/eval/test_monomer.py`

**Interfaces:**
- Consumes:
  - `fold(model, chains, *, num_samples, num_loops, num_steps, seed) -> FoldResult`
    (`oplm.fold.predict`). The result fields used are `coords (A,3)`, `atom_names`, `plddt (L,)`
    on the 0–1 scale, and `ptm`.
  - `ChainSpec(sequence, chain_id)`;
  - `tm_score` and `lddt` (Task 1); `EvalTarget` (Task 2).
- Produces:
  - `METRICS = ("tm_score", "lddt_ca", "lddt", "plddt", "ptm")`.
  - `SamplingSettings(num_loops: int | None = None, num_steps: int | None = None,
    num_samples: int = 1, seed: int = 0)`, frozen.
  - `TargetScore`, a dataclass with fields `target_set`, `entry_id`, `chain_id`,
    `auth_chain_id`, `length`, `resolved_residues`, `status` (`"ok"` or `"failed"`), `error`,
    `seconds` and the five metric fields (NaN unless ok). Property `key -> (target_set,
    entry_id, chain_id)`.
  - `score_result(target, result) -> dict[str, float]`, which raises `ValueError` on a
    different atom layout.
  - `score_target(model, target, settings) -> TargetScore`, which never raises.
  - `summarize(scores) -> dict[str, float]`: the `METRICS` means plus `targets` and `failed`.
  - `write_scores(path, scores) -> None` and `read_scores(path) -> list[TargetScore]` (CSV).
  - `compare_scores(a, b, *, metric="tm_score", samples=10_000, seed=0) -> dict[str, float]`,
    with keys `n`, `mean_a`, `mean_b`, `mean_diff` (b − a), `ci_low`, `ci_high` and
    `b_better_fraction`.

- [ ] **Step 1: Write the failing tests**

`tests/fold/eval/test_monomer.py`:

```python
"""score_result / score_target / summarize / CSV round trip / paired comparison (addendum §6)."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from oplm.fold.eval import monomer
from oplm.fold.eval.monomer import (
    METRICS,
    SamplingSettings,
    TargetScore,
    compare_scores,
    read_scores,
    score_result,
    score_target,
    summarize,
    write_scores,
)
from oplm.fold.eval.targets import load_targets
from oplm.fold.modeling_fold import OplmForFolding
from tests.fold.helpers import tiny_fold_config, tiny_lm

if TYPE_CHECKING:
    from pathlib import Path


def _tiny_model() -> OplmForFolding:
    torch.manual_seed(0)
    cfg = tiny_fold_config(
        plddt_bins=10,
        pae_bins=8,
        pde_bins=8,
        confidence_dist_bins=5,
        distogram_bins=8,
        inference_num_steps=2,
        inference_num_loops=1,
        lm_hidden_size=32,
        lm_num_hidden_states=3,
    )
    model = OplmForFolding(cfg).eval()
    model.attach_lm(tiny_lm())
    return model


def _result(coords: np.ndarray, names: list[str], n_res: int) -> SimpleNamespace:
    return SimpleNamespace(
        coords=torch.tensor(coords), atom_names=names, plddt=torch.full((n_res,), 0.8), ptm=0.6
    )


def _row(key: str, tm: float, status: str = "ok") -> TargetScore:
    return TargetScore(
        target_set="cameo", entry_id=key, chain_id="A", auth_chain_id="A", length=50,
        resolved_residues=50, status=status, tm_score=tm, lddt_ca=tm, lddt=tm, plddt=0.5, ptm=0.5,
    )


def test_score_result_of_the_reference_itself_is_perfect(fixture_dataset: Path) -> None:
    (t,), _ = load_targets(fixture_dataset, "cameo")
    scores = score_result(t, _result(t.coords, t.atom_names, len(t.sequence)))
    assert scores["tm_score"] == pytest.approx(1.0) and scores["lddt"] == pytest.approx(1.0)
    assert scores["lddt_ca"] == pytest.approx(1.0)
    assert scores["plddt"] == pytest.approx(0.8) and scores["ptm"] == pytest.approx(0.6)


def test_score_result_ignores_unresolved_reference_atoms(fixture_dataset_early_cutoff: Path) -> None:
    (t,), _ = load_targets(fixture_dataset_early_cutoff, "valid")  # 2ds7: 40 of 51 CA resolved
    pred = t.coords.copy()
    pred[~t.resolved] = 999.0  # wherever the reference has nothing, the prediction is absurd
    scores = score_result(t, _result(pred, t.atom_names, len(t.sequence)))
    assert scores["tm_score"] == pytest.approx(1.0) and scores["lddt"] == pytest.approx(1.0)


def test_score_result_rejects_a_different_atom_layout(fixture_dataset: Path) -> None:
    (t,), _ = load_targets(fixture_dataset, "cameo")
    names = list(t.atom_names)
    names[1] = "CB"
    with pytest.raises(ValueError, match="atom layout"):
        score_result(t, _result(t.coords, names, len(t.sequence)))


def test_score_target_folds_and_scores_with_a_tiny_model(fixture_dataset: Path) -> None:
    (t,), _ = load_targets(fixture_dataset, "cameo")
    row = score_target(_tiny_model(), t, SamplingSettings(seed=0))
    assert row.status == "ok", row.error
    assert (row.entry_id, row.chain_id, row.length, row.resolved_residues) == ("1crn", "A", 46, 46)
    for name in METRICS:
        assert 0.0 <= getattr(row, name) <= 1.0, name
    assert row.seconds > 0


def test_score_target_records_failures(fixture_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (t,), _ = load_targets(fixture_dataset, "cameo")

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(monomer, "fold", boom)
    row = score_target(_tiny_model(), t, SamplingSettings())
    assert row.status == "failed" and row.error == "RuntimeError: boom"
    assert all(math.isnan(getattr(row, name)) for name in METRICS)


def test_summarize_averages_ok_rows_and_counts_failures() -> None:
    summary = summarize([_row("a", 0.4), _row("b", 0.6), _row("c", math.nan, status="failed")])
    assert summary["tm_score"] == pytest.approx(0.5)
    assert summary["targets"] == 2.0 and summary["failed"] == 1.0
    empty = summarize([])
    assert math.isnan(empty["tm_score"]) and empty["targets"] == 0.0


def test_score_table_round_trip(tmp_path: Path) -> None:
    rows = [_row("a", 0.4), _row("b", math.nan, status="failed")]
    rows[1].error = "RuntimeError: boom, with a comma"
    write_scores(tmp_path / "targets.csv", rows)
    back = read_scores(tmp_path / "targets.csv")
    assert back[0] == rows[0]
    assert back[1].error == rows[1].error and math.isnan(back[1].tm_score)


def test_compare_scores_is_paired_and_reproducible() -> None:
    a = [_row(k, v) for k, v in [("a", 0.40), ("b", 0.50), ("c", 0.60), ("x", 0.9)]]
    b = [_row(k, v) for k, v in [("a", 0.50), ("b", 0.62), ("c", 0.68), ("y", 0.1)]]
    b.append(_row("d", math.nan, status="failed"))
    out = compare_scores(a, b, samples=2000, seed=0)
    assert out["n"] == 3.0
    assert out["mean_diff"] == pytest.approx(0.1) and out["b_better_fraction"] == 1.0
    assert out["ci_low"] <= 0.1 <= out["ci_high"] and out["ci_low"] > 0.0
    assert compare_scores(a, b, samples=2000, seed=0) == out
    with pytest.raises(ValueError, match="metric"):
        compare_scores(a, b, metric="rmsd")
    with pytest.raises(ValueError, match="no target"):
        compare_scores(a[:1], b[1:2])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/eval/test_monomer.py -q`
Expected: `ModuleNotFoundError: No module named 'oplm.fold.eval.monomer'`.

- [ ] **Step 3: Implement `monomer.py`**

```python
"""Fold one held-out chain and score it; aggregate, store and compare score tables (addendum §6)."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from oplm.fold.data.featurize import ChainSpec
from oplm.fold.eval.metrics import lddt, tm_score
from oplm.fold.predict import fold

if TYPE_CHECKING:
    from collections.abc import Sequence

    from oplm.fold.eval.targets import EvalTarget
    from oplm.fold.modeling_fold import OplmForFolding
    from oplm.fold.predict import FoldResult

__all__ = [
    "METRICS",
    "SamplingSettings",
    "TargetScore",
    "compare_scores",
    "read_scores",
    "score_result",
    "score_target",
    "summarize",
    "write_scores",
]

METRICS = ("tm_score", "lddt_ca", "lddt", "plddt", "ptm")
_CASTS = {"str": str, "int": int, "float": float}


@dataclass(frozen=True)
class SamplingSettings:
    """How each target is folded; recorded next to every score table (spec §8)."""

    num_loops: int | None = None  # None: the checkpoint's inference_num_loops
    num_steps: int | None = None  # None: the checkpoint's inference_num_steps
    num_samples: int = 1  # the best sample by pTM is scored
    seed: int = 0


@dataclass
class TargetScore:
    """One row of a score table; the metrics are NaN unless ``status == "ok"``."""

    target_set: str
    entry_id: str
    chain_id: str
    auth_chain_id: str
    length: int
    resolved_residues: int
    status: str  # "ok" or "failed"
    error: str = ""
    seconds: float = 0.0
    tm_score: float = math.nan
    lddt_ca: float = math.nan
    lddt: float = math.nan
    plddt: float = math.nan
    ptm: float = math.nan

    @property
    def key(self) -> tuple[str, str, str]:
        """``(target_set, entry_id, chain_id)``: pairs rows across tables."""
        return (self.target_set, self.entry_id, self.chain_id)


def score_result(target: EvalTarget, result: FoldResult) -> dict[str, float]:
    """TM-score and CA lDDT over resolved CAs, all-atom lDDT, mean pLDDT (0-1) and pTM.

    Raises:
        ValueError: The prediction's atoms are not the reference's atoms in the same order.
    """
    if list(result.atom_names) != target.atom_names:
        raise ValueError(f"{target.key}: the predicted atom layout differs from the reference")
    pred = result.coords.detach().cpu().double().numpy()
    ca = target.ca_index
    ca_mask = target.resolved[ca]
    return {
        "tm_score": tm_score(pred[ca], target.coords[ca], ca_mask),
        "lddt_ca": lddt(pred[ca], target.coords[ca], ca_mask),
        "lddt": lddt(pred, target.coords, target.resolved, residue_index=target.residue_index),
        "plddt": float(result.plddt.float().mean()),
        "ptm": float(result.ptm),
    }


def score_target(
    model: OplmForFolding, target: EvalTarget, settings: SamplingSettings
) -> TargetScore:
    """Fold ``target.sequence`` alone (eval-mode ``model``) and score the best sample.

    Any exception, CUDA out-of-memory included, becomes a ``failed`` row instead of propagating:
    in training every rank must reach the gather, and offline one bad target must not end the run
    (spec §8 reports failures and coverage).
    """
    row = TargetScore(
        target_set=target.target_set,
        entry_id=target.entry_id,
        chain_id=target.chain_id,
        auth_chain_id=target.auth_chain_id,
        length=len(target.sequence),
        resolved_residues=int(target.resolved[target.ca_index].sum()),
        status="ok",
    )
    start = time.perf_counter()
    try:
        result = fold(
            model,
            [ChainSpec(target.sequence, "A")],
            num_samples=settings.num_samples,
            num_loops=settings.num_loops,
            num_steps=settings.num_steps,
            seed=settings.seed,
        )
        for name, value in score_result(target, result).items():
            setattr(row, name, value)
    except Exception as exc:
        row.status, row.error = "failed", f"{type(exc).__name__}: {exc}"[:500]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    row.seconds = time.perf_counter() - start
    return row


def summarize(scores: Sequence[TargetScore]) -> dict[str, float]:
    """Means of ``METRICS`` over ``ok`` rows (NaN when none) plus ``targets`` and ``failed``."""
    ok = [s for s in scores if s.status == "ok"]
    out = {m: float(np.mean([getattr(s, m) for s in ok])) if ok else math.nan for m in METRICS}
    out["targets"] = float(len(ok))
    out["failed"] = float(len(scores) - len(ok))
    return out


def write_scores(path: str | Path, scores: Sequence[TargetScore]) -> None:
    """Write ``scores`` as CSV, one row per target, columns in ``TargetScore`` field order."""
    with Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[f.name for f in fields(TargetScore)])
        writer.writeheader()
        for s in scores:
            writer.writerow(asdict(s))


def read_scores(path: str | Path) -> list[TargetScore]:
    """Read a table written by :func:`write_scores`."""
    types = {f.name: str(f.type) for f in fields(TargetScore)}
    with Path(path).open(newline="") as fh:
        return [
            TargetScore(**{k: _CASTS[types[k]](v) for k, v in raw.items()})
            for raw in csv.DictReader(fh)
        ]


def compare_scores(
    a: Sequence[TargetScore],
    b: Sequence[TargetScore],
    *,
    metric: str = "tm_score",
    samples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired ``b - a`` over targets scored ``ok`` in both, with a percentile bootstrap 95 % CI.

    Raises:
        ValueError: ``metric`` is not one of ``METRICS``, or no target is ``ok`` in both.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {METRICS}")
    left = {s.key: getattr(s, metric) for s in a if s.status == "ok"}
    right = {s.key: getattr(s, metric) for s in b if s.status == "ok"}
    keys = sorted(left.keys() & right.keys())
    if not keys:
        raise ValueError("no target is scored ok in both tables")
    x = np.array([left[k] for k in keys])
    y = np.array([right[k] for k in keys])
    diff = y - x
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(diff), size=(samples, len(diff)))].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "n": float(len(keys)),
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "mean_diff": float(diff.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "b_better_fraction": float((diff > 0).mean()),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/fold/eval/test_monomer.py -q`
Expected: 8 passed.

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/eval/monomer.py tests/fold/eval/test_monomer.py
VIRTUAL_ENV=.venv .venv/bin/ty check src/
git add src/oplm/fold/eval/monomer.py tests/fold/eval/test_monomer.py
git commit -m "feat(fold): score held-out chains and compare score tables"
```

---

### Task 4: The `fold_monomer` eval task, the generic `EvalTask`, and registration

**Files:**
- Create: `src/oplm/fold/eval/tasks.py`
- Modify:
  - `src/oplm/fold/eval/__init__.py` (registration import)
  - `src/oplm/fold/training.py` (import the package)
  - `src/oplm/eval/tasks/base.py` (generic)
  - the seven MLM task modules (subclass line)
- Test: `tests/fold/eval/test_tasks.py`

**Interfaces:**
- Consumes:
  - `register_eval_task` (`oplm.eval.registry`), `EvalTask` (`oplm.eval.tasks.base`),
    `gather_objects(local, accelerator)` (`oplm.eval.tasks.variant_scoring`);
  - `EvalDatasetEntry(name, path, type, schedule, metrics, extra)`;
  - Task 2's `load_targets`; Task 3's `METRICS`, `SamplingSettings`, `score_target` and
    `summarize`.
- Produces:
  - `EvalTask(ABC, Generic[ModelT])` with `ModelT = TypeVar("ModelT", bound="nn.Module")`, and
    `evaluate(self, model: ModelT, accelerator) -> dict[str, float]`.
  - `FoldMonomerTaskConfig.from_extra(extra)`, with keys `target_set` (default `"valid"`),
    `max_targets`, `max_length`, `num_loops`, `num_steps`, `num_samples` (1) and `seed` (0). It
    has a `sampling` property.
  - `FoldMonomerTask`, registered as `"fold_monomer"`, with
    `default_metrics = [*METRICS, "targets", "failed"]`.

- [ ] **Step 1: Write the failing tests**

`tests/fold/eval/test_tasks.py`:

```python
"""fold_monomer: registration, config, CPU evaluate, failures, rank sharding, Trainer run."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import torch
from accelerate import Accelerator

import oplm.fold.eval  # noqa: F401  (registers fold_monomer)
from oplm.config import EvalDatasetEntry, OplmConfig, ScheduleSpec
from oplm.eval.registry import get_eval_task_class
from oplm.fold.eval import tasks
from oplm.fold.eval.monomer import METRICS, TargetScore
from oplm.fold.eval.tasks import FoldMonomerTask, FoldMonomerTaskConfig
from oplm.fold.modeling_fold import OplmForFolding
from oplm.training import TrainerCallback
from tests.fold.helpers import tiny_fold_config, tiny_fold_run_cfg, tiny_lm

if TYPE_CHECKING:
    from pathlib import Path


def _tiny_model() -> OplmForFolding:
    torch.manual_seed(0)
    cfg = tiny_fold_config(
        plddt_bins=10,
        pae_bins=8,
        pde_bins=8,
        confidence_dist_bins=5,
        distogram_bins=8,
        inference_num_steps=2,
        inference_num_loops=1,
        lm_hidden_size=32,
        lm_num_hidden_states=3,
    )
    model = OplmForFolding(cfg).eval()
    model.attach_lm(tiny_lm())
    return model


def _task(path: Path, **extra: Any) -> FoldMonomerTask:
    entry = EvalDatasetEntry(
        name="valid", path=str(path), type="fold_monomer", schedule=ScheduleSpec("steps", 1),
        extra=extra,
    )
    return FoldMonomerTask(entry, OplmConfig())


def test_fold_monomer_registers_on_import() -> None:
    assert get_eval_task_class("fold_monomer") is FoldMonomerTask


def test_mlm_eval_tooling_never_imports_fold_code() -> None:
    code = (
        "import sys, oplm.eval, oplm.eval.tasks; "
        "bad = sorted(m for m in sys.modules if m.startswith('oplm.fold')); "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_task_config_rejects_bad_options() -> None:
    assert FoldMonomerTaskConfig.from_extra({}).target_set == "valid"
    with pytest.raises(ValueError, match="unknown fold_monomer option"):
        FoldMonomerTaskConfig.from_extra({"targets": "valid"})
    with pytest.raises(ValueError, match="num_samples"):
        FoldMonomerTaskConfig.from_extra({"num_samples": 0})
    with pytest.raises(ValueError, match="seed"):
        FoldMonomerTaskConfig.from_extra({"seed": "0"})
    entry = EvalDatasetEntry(
        name="v", path="x", type="fold_monomer", schedule=ScheduleSpec("steps", 1),
        metrics=["tm_score", "gdt_ts"],
    )
    with pytest.raises(ValueError, match="gdt_ts"):
        FoldMonomerTask(entry, OplmConfig())


def test_evaluate_scores_the_valid_split_on_cpu(fixture_dataset_early_cutoff: Path) -> None:
    metrics = _task(fixture_dataset_early_cutoff).evaluate(_tiny_model(), Accelerator(cpu=True))
    assert set(metrics) == {*METRICS, "targets", "failed"}
    assert metrics["targets"] == 1.0 and metrics["failed"] == 0.0
    assert 0.0 <= metrics["tm_score"] <= 1.0


def test_evaluate_counts_a_failed_target(
    fixture_dataset_early_cutoff: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr("oplm.fold.eval.monomer.fold", boom)
    metrics = _task(fixture_dataset_early_cutoff).evaluate(_tiny_model(), Accelerator(cpu=True))
    assert metrics["targets"] == 0.0 and metrics["failed"] == 1.0


def _ok_row(entry: str) -> TargetScore:
    return TargetScore(
        target_set="three", entry_id=entry, chain_id="A", auth_chain_id="A", length=1,
        resolved_residues=1, status="ok", tm_score=0.5, lddt_ca=0.5, lddt=0.5, plddt=0.5, ptm=0.5,
    )


def test_evaluate_shards_targets_across_ranks(
    fixture_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ds"
    shutil.copytree(fixture_dataset, root)
    items = [{"id": "1crn", "chain": "A"}, {"id": "1jb6", "chain": None}, {"id": "1l2y", "chain": "A"}]
    (root / "targets" / "three.json").write_text(json.dumps(items))
    scored: list[str] = []

    def fake_score(model: object, target: Any, settings: object) -> TargetScore:
        scored.append(target.key)
        return _ok_row(target.entry_id)

    monkeypatch.setattr(tasks, "score_target", fake_score)
    # Rank 1 of 2: targets sorted 1crn_A, 1jb6_A, 1l2y_A -> this rank scores index 1 only; the
    # gather returns every rank's rows, so the aggregate covers all three exactly once.
    monkeypatch.setattr(tasks, "gather_objects", lambda local, acc: [_ok_row("1crn"), *local, _ok_row("1l2y")])
    metrics = _task(root, target_set="three").evaluate(
        _tiny_model(), SimpleNamespace(process_index=1, num_processes=2)
    )
    assert scored == ["1jb6_A"] and metrics["targets"] == 3.0
    # More ranks than targets: rank 5 of 8 scores nothing but still reaches the gather.
    scored.clear()
    reached: list[int] = []
    monkeypatch.setattr(tasks, "gather_objects", lambda local, acc: reached.append(len(local)) or local)
    metrics = _task(root, target_set="three").evaluate(
        _tiny_model(), SimpleNamespace(process_index=5, num_processes=8)
    )
    assert scored == [] and reached == [0] and metrics["targets"] == 0.0


class _EvalRecorder(TrainerCallback):
    def __init__(self) -> None:
        self.calls: list[dict[str, float]] = []

    def on_eval_end(self, trainer: Any, metrics: dict[str, float], step: int) -> None:
        self.calls.append(dict(metrics))


@pytest.mark.slow
def test_trainer_runs_fold_monomer_eval(
    tmp_path: Path,
    fixture_dataset: Path,
    fixture_dataset_early_cutoff: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oplm.fold.training import FoldTask
    from oplm.training.trainer import Trainer
    from tests.training.conftest import configure_accelerator_device

    configure_accelerator_device("cpu", monkeypatch)
    lm_dir = tmp_path / "lm"
    tiny_lm().save_pretrained(lm_dir)
    cfg = tiny_fold_run_cfg(tmp_path / "run", fixture_dataset, lm_dir, max_steps=2)
    cfg.data.eval = {
        "valid": {"path": str(fixture_dataset_early_cutoff), "type": "fold_monomer", "every": {"steps": 2}}
    }
    recorder = _EvalRecorder()
    Trainer(cfg, task=FoldTask(), callbacks=[recorder]).train()
    keys = {k for call in recorder.calls for k in call}
    assert "eval/valid/tm_score" in keys and "eval/valid/targets" in keys
    assert all(0.0 <= call["eval/valid/tm_score"] <= 1.0 for call in recorder.calls)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/eval/test_tasks.py -q -m "not slow"`
Expected: `ModuleNotFoundError: No module named 'oplm.fold.eval.tasks'`.

- [ ] **Step 3: Make `EvalTask` generic**

In `src/oplm/eval/tasks/base.py`:
- Change the typing import to `from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar`.
- Make the `TYPE_CHECKING` block read exactly as follows. `OplmForMaskedLM` is no longer
  imported here.

```python
if TYPE_CHECKING:
    from accelerate import Accelerator
    from torch import nn

    from oplm.config import EvalDatasetEntry, OplmConfig
    from oplm.eval.schedule import Schedule

ModelT = TypeVar("ModelT", bound="nn.Module")
```

Then change the class line and `evaluate`:

```python
class EvalTask(ABC, Generic[ModelT]):
    """Abstract base class for evaluation tasks.

    Each subclass handles one eval dataset type: loading data, computing metrics,
    and returning results. Subclasses are registered via @register_eval_task.
    ``ModelT`` is the model class a task scores (spec §6.4: the annotation generalizes to the
    common module interface): the MLM tasks bind ``OplmForMaskedLM``, ``fold_monomer`` binds
    ``OplmForFolding``.
    """
```

```python
    @abstractmethod
    def evaluate(
        self,
        model: ModelT,
        accelerator: Accelerator,
    ) -> dict[str, float]:
```

In each of `sequence.py`, `structure.py`, `proteingym.py`, `proteingym_clinical.py`, `tape.py`,
`proteinglue.py` and `everest.py` under `src/oplm/eval/tasks/`, change the one class line from
`class XEvalTask(EvalTask):` to `class XEvalTask(EvalTask["OplmForMaskedLM"]):`. Each module
already imports `OplmForMaskedLM` under `TYPE_CHECKING`, and the string subscript is a forward
reference at runtime. Planning checked this: `ty check src/` is clean and `tests/eval` passes.

- [ ] **Step 4: Implement `tasks.py`, the registration import and the training import**

`src/oplm/fold/eval/tasks.py`:

```python
"""``fold_monomer``: held-out monomer accuracy as an eval task (addendum §6, spec §8).

In training the Trainer passes the live model (never the EMA copy) every ``every`` steps;
``oplm fold eval`` scores saved EMA weights offline with the same scoring code.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, ClassVar

from oplm.eval.registry import register_eval_task
from oplm.eval.tasks.base import EvalTask
from oplm.eval.tasks.variant_scoring import gather_objects
from oplm.fold.eval.monomer import METRICS, SamplingSettings, score_target, summarize
from oplm.fold.eval.targets import load_targets

if TYPE_CHECKING:
    from accelerate import Accelerator

    from oplm.config import EvalDatasetEntry, OplmConfig
    from oplm.fold.eval.targets import EvalTarget
    from oplm.fold.modeling_fold import OplmForFolding

__all__ = ["FoldMonomerTask", "FoldMonomerTaskConfig"]

_POSITIVE = ("max_targets", "max_length", "num_loops", "num_steps", "num_samples")


@dataclass(frozen=True)
class FoldMonomerTaskConfig:
    """The extra keys of a ``type: fold_monomer`` entry under ``data.eval``."""

    target_set: str = "valid"
    max_targets: int | None = None
    max_length: int | None = None  # None: the model's lm_max_chain_length
    num_loops: int | None = None  # None: the checkpoint's inference default
    num_steps: int | None = None
    num_samples: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.target_set, str) or not self.target_set:
            raise ValueError(f"fold_monomer target_set must be a name, got {self.target_set!r}")
        for name in _POSITIVE:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"fold_monomer {name} must be a positive integer, got {value!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"fold_monomer seed must be an integer, got {self.seed!r}")

    @classmethod
    def from_extra(cls, extra: dict[str, Any]) -> FoldMonomerTaskConfig:
        """Build from an eval entry's ``extra`` mapping.

        Raises:
            ValueError: An unknown key, or a value of the wrong type or range.
        """
        known = sorted(f.name for f in fields(cls))
        unknown = sorted(set(extra) - set(known))
        if unknown:
            raise ValueError(f"unknown fold_monomer option(s) {unknown}; known: {known}")
        return cls(**extra)

    @property
    def sampling(self) -> SamplingSettings:
        """The fold settings for every target."""
        return SamplingSettings(
            num_loops=self.num_loops,
            num_steps=self.num_steps,
            num_samples=self.num_samples,
            seed=self.seed,
        )


@register_eval_task("fold_monomer")
class FoldMonomerTask(EvalTask["OplmForFolding"]):
    """Fold every target of a set, sharded across ranks, and report the means (spec §8).

    Metrics: TM-score, CA and all-atom lDDT, mean pLDDT (0-1) and pTM over targets scored ``ok``,
    plus the ``targets`` and ``failed`` counts. The entry's ``path`` is a preprocessed dataset.
    """

    default_metrics: ClassVar[list[str]] = [*METRICS, "targets", "failed"]

    def __init__(self, entry: EvalDatasetEntry, cfg: OplmConfig) -> None:
        super().__init__(entry, cfg)
        self.tcfg = FoldMonomerTaskConfig.from_extra(entry.extra)
        unknown = sorted(set(self.metrics) - set(self.default_metrics))
        if unknown:
            raise ValueError(
                f"fold_monomer cannot report {unknown}; available: {self.default_metrics}"
            )
        self._targets: list[EvalTarget] | None = None

    def evaluate(self, model: OplmForFolding, accelerator: Accelerator) -> dict[str, float]:
        """Score this rank's share of the targets, gather every row, return the means."""
        if self._targets is None:
            max_length = self.tcfg.max_length or model.config.lm_max_chain_length
            self._targets, _ = load_targets(
                self.path, self.tcfg.target_set, max_length=max_length, limit=self.tcfg.max_targets
            )
        share = self._targets[accelerator.process_index :: accelerator.num_processes]
        local = [score_target(model, target, self.tcfg.sampling) for target in share]
        summary = summarize(gather_objects(local, accelerator))
        return {k: v for k, v in summary.items() if k in self.metrics}
```

`src/oplm/fold/eval/__init__.py` becomes:

```python
"""Held-out monomer evaluation for the structure head (addendum §6, spec §8; docs/FOLD.md §10).

Importing this package registers the ``fold_monomer`` eval type. ``oplm.fold.training`` imports
it, so every fold run can name the type under ``data.eval``; MLM tooling never imports fold code.
"""

from __future__ import annotations

from oplm.fold.eval import tasks  # noqa: F401  (registers fold_monomer)
```

In `src/oplm/fold/training.py`, add the following with the other first-party imports at module
top:

```python
import oplm.fold.eval  # noqa: F401  (registers the fold_monomer eval type, docs/FOLD.md §10)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/fold/eval -q -m "not slow" && .venv/bin/python -m pytest tests/fold/eval/test_tasks.py -q -m slow`
Expected: all pass. The slow test runs 2 CPU steps and one `valid` eval.

Then run the MLM eval and training suites, which must be unchanged:
`.venv/bin/python -m pytest tests/ev[a]l tests/training -q -m "not slow"` (the `[a]` glob only
dodges a shell-guard false positive on the word "eval").

- [ ] **Step 6: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/eval src/oplm/eval/tasks src/oplm/fold/training.py tests/fold/eval
VIRTUAL_ENV=.venv .venv/bin/ty check src/
git add src/oplm/fold/eval src/oplm/eval/tasks src/oplm/fold/training.py tests/fold/eval
git commit -m "feat(fold): fold_monomer eval task; EvalTask generic over the model class"
```

If `ruff format` reformats lines in the seven MLM task modules beyond the class line, keep only
the class-line change (`git checkout -p`) and note it in the report.

---

### Task 5: `oplm fold eval` and `oplm fold compare`

**Files:**
- Modify: `src/oplm/fold/cli.py`
- Test: `tests/fold/test_cli.py`

**Interfaces:**
- Consumes:
  - `OplmForFolding.from_pretrained(path, lm_name_or_path=...)`;
  - Task 2's `load_targets`; Task 3's `SamplingSettings`, `score_target`, `summarize`,
    `write_scores`, `read_scores` and `compare_scores`.
- Produces:
  - `oplm fold eval --model PATH --data DIR --target-set NAME [--target-set NAME ...]
    --out DIR [--weights ema|raw] [--lm X] [--loops N] [--steps N] [--samples N] [--seed N]
    [--max-length N] [--limit N] [--device auto|cpu|cuda]`. It writes `OUT/targets.csv` (every
    set) and `OUT/summary.json`.
  - `oplm fold compare A B [--metric tm_score] [--target-set NAME] [--samples N] [--seed N]`,
    where A and B are eval output directories or CSV paths. It prints JSON.

`summary.json` layout:

```json
{
  "model": "<resolved HF dir>", "weights": "ema", "lm": "<lm_name_or_path>",
  "data": "<dataset dir>", "sampling": {"num_loops": null, "num_steps": null, "num_samples": 1, "seed": 0},
  "resolved_sampling": {"num_loops": 10, "num_steps": 14},
  "device": "cuda", "versions": {"oplm": "...", "torch": "..."},
  "started": "<ISO time>", "seconds": 1234.5,
  "target_sets": {"cameo": {"coverage": {"listed": 755, "missing": [], "skipped": {}, "selected": 752},
                            "metrics": {"tm_score": 0.5, "...": 0.0, "targets": 752.0, "failed": 0.0}}}
}
```

- [ ] **Step 1: Write the failing tests** (append to `tests/fold/test_cli.py`)

```python
def _saved_head(root: Path) -> tuple[Path, Path]:
    from oplm.fold import OplmForFolding

    torch.manual_seed(0)
    cfg = tiny_fold_config(
        plddt_bins=10, pae_bins=8, pde_bins=8, confidence_dist_bins=5, distogram_bins=8,
        inference_num_steps=2, inference_num_loops=1, lm_hidden_size=32, lm_num_hidden_states=3,
    )
    OplmForFolding(cfg).save_pretrained(root / "checkpoint-10" / "hf_ema")
    tiny_lm().save_pretrained(root / "lm")
    return root / "checkpoint-10", root / "lm"


def test_eval_cli_writes_scores_and_summary(tmp_path: Path, fixture_dataset: Path) -> None:
    ckpt, lm = _saved_head(tmp_path)
    out = tmp_path / "eval"
    args = ["fold", "eval", "--model", str(ckpt), "--lm", str(lm), "--data", str(fixture_dataset)]
    args += ["--target-set", "cameo", "--target-set", "casp14", "--out", str(out), "--device", "cpu"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    from oplm.fold.eval.monomer import read_scores

    rows = read_scores(out / "targets.csv")
    assert [(r.target_set, r.entry_id, r.chain_id, r.status) for r in rows] == [
        ("cameo", "1crn", "A", "ok"),
        ("casp14", "1jb6", "A", "ok"),
    ]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["model"].endswith("hf_ema") and summary["weights"] == "ema"
    assert summary["sampling"] == {"num_loops": None, "num_steps": None, "num_samples": 1, "seed": 0}
    assert summary["resolved_sampling"] == {"num_loops": 1, "num_steps": 2}
    cameo = summary["target_sets"]["cameo"]
    assert cameo["coverage"] == {"listed": 1, "missing": [], "skipped": {}, "selected": 1}
    assert cameo["metrics"]["targets"] == 1.0 and 0.0 <= cameo["metrics"]["tm_score"] <= 1.0


def test_eval_cli_resolves_and_rejects_weights(tmp_path: Path, fixture_dataset: Path) -> None:
    ckpt, lm = _saved_head(tmp_path)
    base = ["fold", "eval", "--lm", str(lm), "--data", str(fixture_dataset), "--target-set", "cameo"]
    base += ["--device", "cpu", "--out", str(tmp_path / "o")]
    ok = runner.invoke(app, [*base, "--model", str(ckpt / "hf_ema")])  # an HF dir directly
    assert ok.exit_code == 0, ok.output
    raw = runner.invoke(app, [*base, "--model", str(ckpt), "--weights", "raw"])
    assert raw.exit_code != 0 and "hf" in plain(raw.output)
    bad = runner.invoke(app, [*base, "--model", str(ckpt), "--weights", "best"])
    assert bad.exit_code != 0


def test_compare_cli_prints_the_paired_difference(tmp_path: Path) -> None:
    from oplm.fold.eval.monomer import TargetScore, write_scores

    def rows(tms: list[float]) -> list[TargetScore]:
        return [
            TargetScore(target_set="cameo", entry_id=f"e{i}", chain_id="A", auth_chain_id="A",
                        length=50, resolved_residues=50, status="ok", tm_score=v, lddt_ca=v,
                        lddt=v, plddt=v, ptm=v)
            for i, v in enumerate(tms)
        ]

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    write_scores(tmp_path / "a" / "targets.csv", rows([0.4, 0.5, 0.6]))
    write_scores(tmp_path / "b" / "targets.csv", rows([0.5, 0.6, 0.7]))
    result = runner.invoke(app, ["fold", "compare", str(tmp_path / "a"), str(tmp_path / "b")])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["n"] == 3.0 and out["mean_diff"] == pytest.approx(0.1)
```

`tests/fold/test_cli.py` already imports `runner`, `app`, `plain`, `tiny_fold_config`,
`tiny_lm`, `torch`, `json` and `pytest`. Add any that are missing. `fixture_dataset` comes from
`tests/fold/conftest.py`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/test_cli.py -q -k "eval_cli or compare_cli"`
Expected: exit code 2 (`No such command 'eval'`).

- [ ] **Step 3: Implement the two commands** (in `src/oplm/fold/cli.py`, after `predict`)

```python
_WEIGHT_DIRS = {"ema": "hf_ema", "raw": "hf"}


def _resolve_model_dir(model: Path, weights: str) -> Path:
    """An HF directory as given, or a checkpoint directory's ``hf_ema/`` / ``hf/``."""
    if (model / "config.json").is_file():
        return model
    if weights not in _WEIGHT_DIRS:
        raise typer.BadParameter(f"--weights must be ema or raw, got {weights!r}")
    path = model / _WEIGHT_DIRS[weights]
    if not (path / "config.json").is_file():
        raise typer.BadParameter(f"{model} has no {_WEIGHT_DIRS[weights]}/ model directory")
    return path


@app.command("eval")
def eval_(
    model: Annotated[
        Path, typer.Option("--model", help="A checkpoint directory (see --weights) or HF dir")
    ],
    data: Annotated[
        Path, typer.Option("--data", help="Preprocessed dataset (index.parquet, shards/, targets/)")
    ],
    target_sets: Annotated[
        list[str], typer.Option("--target-set", help="cameo, casp14, valid, ... (repeatable)")
    ],
    out: Annotated[Path, typer.Option("--out", help="Output directory")],
    weights: Annotated[str, typer.Option(help="ema or raw, for a checkpoint directory")] = "ema",
    lm: Annotated[
        str | None,
        typer.Option("--lm", help="Frozen LM path or Hub id (overrides the checkpoint's)"),
    ] = None,
    loops: Annotated[int | None, typer.Option(help="Recurrence iterations (default: config)")] = None,
    steps: Annotated[int | None, typer.Option(help="Sampler steps (default: config)")] = None,
    samples: Annotated[int, typer.Option(help="Diffusion samples; the best by pTM is scored")] = 1,
    seed: Annotated[int, typer.Option(help="Seed for every target")] = 0,
    max_length: Annotated[
        int | None, typer.Option(help="Skip longer chains (default: lm_max_chain_length)")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Score only the first N per set")] = None,
    device: Annotated[str, typer.Option(help="cuda, cpu or auto")] = "auto",
) -> None:
    """Score a checkpoint on held-out target sets: per-target CSV plus a JSON summary.

    Full-chain TM-score, CA and all-atom lDDT, mean pLDDT and pTM (docs/FOLD.md §10).
    """
    import datetime
    import time

    import torch

    import oplm
    from oplm.fold.eval.monomer import SamplingSettings, score_target, summarize, write_scores
    from oplm.fold.eval.targets import load_targets
    from oplm.fold.modeling_fold import OplmForFolding

    hf_dir = _resolve_model_dir(model, weights)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    folder = OplmForFolding.from_pretrained(hf_dir, lm_name_or_path=lm).to(dev).eval()
    if folder.lm is None:
        raise typer.BadParameter(
            "no language model: pass --lm or save the checkpoint with lm_name_or_path"
        )
    folder.lm.to(dev)
    if dev.type == "cuda":
        folder.lm.to(torch.bfloat16)  # spec §5.4: the frozen LM runs in bf16 on CUDA
    settings = SamplingSettings(num_loops=loops, num_steps=steps, num_samples=samples, seed=seed)
    cap = max_length or folder.config.lm_max_chain_length
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    clock = time.perf_counter()
    rows = []
    sets: dict[str, Any] = {}
    for name in target_sets:
        targets, coverage = load_targets(data, name, max_length=cap, limit=limit)
        scores = []
        for i, target in enumerate(targets, 1):
            scores.append(score_target(folder, target, settings))
            console.print(
                f"[{name} {i}/{len(targets)}] {target.key} {scores[-1].status} "
                f"TM {scores[-1].tm_score:.3f}"
            )
        rows.extend(scores)
        sets[name] = {
            "coverage": {
                "listed": coverage.listed,
                "missing": coverage.missing,
                "skipped": coverage.skipped,
                "selected": coverage.selected,
            },
            "metrics": summarize(scores),
        }
    out.mkdir(parents=True, exist_ok=True)
    write_scores(out / "targets.csv", rows)
    summary = {
        "model": str(hf_dir),
        "weights": weights,
        "lm": folder.config.lm_name_or_path,
        "data": str(data),
        "sampling": dataclasses.asdict(settings),
        "resolved_sampling": {
            "num_loops": loops or folder.config.inference_num_loops,
            "num_steps": steps or folder.config.inference_num_steps,
        },
        "device": str(dev),
        "versions": {"oplm": oplm.__version__, "torch": torch.__version__},
        "started": started,
        "seconds": time.perf_counter() - clock,
        "target_sets": sets,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for name, block in sets.items():
        m = block["metrics"]
        console.print(
            f"{name}: {int(m['targets'])} scored, {int(m['failed'])} failed, "
            f"{len(block['coverage']['skipped'])} skipped  TM {m['tm_score']:.3f}  "
            f"lDDT {m['lddt']:.3f}  pLDDT {m['plddt']:.3f}"
        )
    console.print(f"wrote {out / 'targets.csv'} and {out / 'summary.json'}")


@app.command("compare")
def compare(
    a: Annotated[Path, typer.Argument(help="Baseline eval directory or targets.csv")],
    b: Annotated[Path, typer.Argument(help="Candidate eval directory or targets.csv")],
    metric: Annotated[str, typer.Option(help="tm_score, lddt_ca, lddt, plddt or ptm")] = "tm_score",
    target_set: Annotated[
        str | None, typer.Option("--target-set", help="Only this set (default: all, pooled)")
    ] = None,
    samples: Annotated[int, typer.Option(help="Bootstrap resamples")] = 10_000,
    seed: Annotated[int, typer.Option(help="Bootstrap seed")] = 0,
) -> None:
    """Paired B − A over targets scored in both, with a bootstrap 95 % CI, as JSON."""
    from oplm.fold.eval.monomer import compare_scores, read_scores

    def table(path: Path) -> list[Any]:
        rows = read_scores(path / "targets.csv" if path.is_dir() else path)
        return [r for r in rows if target_set is None or r.target_set == target_set]

    try:
        result = compare_scores(table(a), table(b), metric=metric, samples=samples, seed=seed)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=1))
```

`compare` prints with `typer.echo` rather than `console.print` so its output stays plain JSON.
The module already imports `json`, `Any`, `Annotated`, `Path`, `typer` and `console` at top
level; add `import dataclasses` to those top-level imports. Torch stays out of module scope.
Both `inference_num_loops` and `inference_num_steps` exist on `FoldConfig`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/fold/test_cli.py -q`
Expected: all pass (the existing CLI tests plus 3 new ones).

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/cli.py
VIRTUAL_ENV=.venv .venv/bin/ty check src/
git add src/oplm/fold/cli.py tests/fold/test_cli.py
git commit -m "feat(fold): oplm fold eval and oplm fold compare"
```

Format only `src/oplm/fold/cli.py`. Do not reformat `tests/fold/test_cli.py` as a whole, because
it carries pre-existing drift.

---

### Task 6: `oplm slurm generate` for fold runs

**Files:**
- Modify: `src/oplm/slurm/cli.py`, `docs/SLURM.md`
- Test: `tests/slurm/test_cli.py`

**Interfaces:**
- Consumes: `load_config(argv).fold`, which is `None` for an MLM run; and
  `accelerate_command(module=..., gpus_per_node=..., args=...)`.
- Produces: `generate` renders `-m oplm.fold.train` when the resolved run config has a `fold:`
  block, and `-m oplm.train` otherwise. It raises `typer.BadParameter` when the resolved
  `train.output_dir` is not absolute.

- [ ] **Step 1: Write the failing tests**

In `tests/slurm/test_cli.py`, append a `train:` block to the existing `_SLURM_BLOCK` text so
every fixture config has an absolute `output_dir`:

```python
      max_batch_size: {170M: 256, 400M: 256}
    train:
      output_dir: /runs/test
    """
)
```

Then add:

```python
def test_generate_launches_the_fold_entry_for_a_fold_config(tmp_path: Path) -> None:
    config = tmp_path / "fold.yaml"
    config.write_text(_SLURM_BLOCK + "fold:\n  lm: esmc:esmc_600m\n")
    result = runner.invoke(
        app, ["slurm", "generate", "--config", str(config), "--out", str(tmp_path / "out"), "--nodes", "1"]
    )
    assert result.exit_code == 0, result.stdout
    text = (tmp_path / "out" / "oplm-run.sbatch").read_text()
    assert "-m oplm.fold.train" in text and "-m oplm.train " not in text


def test_generate_keeps_the_mlm_entry_without_a_fold_block(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "mlm.yaml")
    result = runner.invoke(
        app, ["slurm", "generate", "--config", str(config), "--out", str(tmp_path / "out"), "--nodes", "1"]
    )
    assert result.exit_code == 0, result.stdout
    text = (tmp_path / "out" / "oplm-run.sbatch").read_text()
    assert "-m oplm.train" in text and "oplm.fold.train" not in text


def test_generate_rejects_a_relative_output_dir(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "rel.yaml")
    result = runner.invoke(
        app,
        ["slurm", "generate", "--config", str(config), "--out", str(tmp_path / "out"),
         "--nodes", "1", "train.output_dir=outputs/run"],
    )
    assert result.exit_code != 0
    assert "absolute" in result.output
```

Check how the existing tests assert on rejected parameters: they read `result.output` or
`result.stdout`, possibly through a `plain()` helper that strips ANSI. Match their style in the
last assertion.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/slurm/test_cli.py -q`
Expected:
- the fold-entry test fails (the script contains `-m oplm.train`);
- the relative-dir test fails (exit code 0);
- the MLM-entry test passes;
- every pre-existing test still passes with the new `train:` block.

- [ ] **Step 3: Implement**

In `generate`, replace `progress_dir = str(load_config(train_argv).train.output_dir)` with:

```python
        run_cfg = load_config(train_argv)
        progress_dir = str(run_cfg.train.output_dir)
        if not Path(progress_dir).is_absolute():
            raise ValueError(
                f"train.output_dir must be an absolute path on shared storage, got "
                f"{progress_dir!r}: the job runs from a node-local $JOB_WORK_DIR, where a "
                "relative path would hold the checkpoints only until the job ends"
            )
        # A run config with a fold: block trains the structure head (docs/FOLD.md §10).
        module = "oplm.fold.train" if run_cfg.fold is not None else "oplm.train"
```

Pass `module=module` to `accelerate_command`. Check that `Path` is imported at module scope;
add `from pathlib import Path` if the module only imports it under `TYPE_CHECKING`.

- [ ] **Step 4: Document** (in `docs/SLURM.md`)

In §9, after the paragraph describing what `generate` writes, add:

```markdown
`generate` launches `oplm.fold.train` instead of `oplm.train` when the resolved run config has a
`fold:` block (docs/FOLD.md §10), and refuses a relative `train.output_dir`: the job runs from a
node-local `$JOB_WORK_DIR`, so a relative path would keep checkpoints only until the job ends and
auto-resume would find nothing.
```

In §8, after the drain triggers, add:

```markdown
**Forcing a drain by hand** (for example, to exercise resume once per run): `scancel
--signal=USR1 <jobid>`. Without `--batch`, Slurm signals only the job steps, so the batch
wrapper survives to requeue. The trainer ranks checkpoint, write `.drained` and exit 85;
`accelerate launch` reports that as `exit=1`. The wrapper then logs `requeueing (exit=1,
drained=1, ...)` and requeues, and the next attempt logs `restart_count=1` and `Auto-resuming
from checkpoint-<N>`.
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/slurm -q`
Expected: all pass.

- [ ] **Step 6: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/slurm/cli.py
VIRTUAL_ENV=.venv .venv/bin/ty check src/
git add src/oplm/slurm/cli.py tests/slurm/test_cli.py docs/SLURM.md
git commit -m "feat(slurm): launch the fold entry for fold configs; require an absolute output_dir"
```

---

### Task 7: Proof-run configs, offline eval job, and the operating docs

**Files:**
- Modify:
  - `configs/fold/proof-w128.yaml`, `configs/fold/proof-w256.yaml`
  - `src/oplm/fold/cli.py` (`measure` drops `data.eval`)
  - `docs/FOLD.md` (new §10, status line)
- Create: `docs/fold/b200-m2c-eval.sbatch`
- Test: `tests/fold/test_cli.py`

**Interfaces:**
- Consumes:
  - Task 4's `fold_monomer`; Task 5's `oplm fold eval`;
  - Task 6's fold entry in `generate`;
  - `load_slurm_config(path) -> SlurmConfig` (fields `gpus_per_node`, `install`, `nodes`, and
    `time_limit` as a `PhaseTable` with `.resolve(phase=None, preset=None)`).
- Produces: the two run configs that Task 8 submits, and the eval job that Task 9 submits.

- [ ] **Step 1: Write the failing tests** (in `tests/fold/test_cli.py`)

Replace `test_proof_configs_load` with:

```python
@pytest.mark.parametrize("width", [128, 256])
def test_proof_configs_load(width: int) -> None:
    from oplm.config import load_config
    from oplm.slurm.config import load_slurm_config

    path = Path("configs/fold") / f"proof-w{width}.yaml"
    cfg = load_config(["--config", str(path)])
    assert cfg.fold is not None and cfg.fold.model["pair_width"] == width
    assert cfg.train.optimizer == "adamw" and cfg.train.ema_decay == 0.999
    assert cfg.fold.crop_size == 384
    assert 8 * cfg.train.batch_size * cfg.train.gradient_accumulation_steps == 128  # 8 x B200
    assert Path(cfg.train.output_dir).is_absolute()
    assert cfg.train.output_dir.endswith(f"proof-w{width}")
    assert cfg.train.wandb_run_name == f"fold-proof-w{width}"
    valid = cfg.data.eval["valid"]
    assert valid["type"] == "fold_monomer" and valid["target_set"] == "valid"
    assert valid["path"] == cfg.data.train and valid["every"] == {"steps": 2000}
    slurm = load_slurm_config(path)
    assert slurm.gpus_per_node == 8 and slurm.nodes.resolve(phase=None, preset=None) == 1
    assert "@fold-m2c-runs" in slurm.install and "esmc" in slurm.install
    assert f"venv-w{width}" in slurm.install and "uv pip check" in slurm.install
    assert "'" not in slurm.install  # rendered inside bash -c '...'


def test_slurm_generate_renders_a_proof_run(tmp_path: Path) -> None:
    args = ["slurm", "generate", "--config", "configs/fold/proof-w128.yaml"]
    args += ["--name", "fold-proof-w128", "--out", str(tmp_path)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    text = (tmp_path / "fold-proof-w128.sbatch").read_text()
    assert "-m oplm.fold.train" in text and "train.auto_resume=true" in text
    assert "/mnt/data/briney/fold-m2c/venv-w128/bin/activate" in text
    assert "/mnt/data/briney/oplm-runs/fold-m2c/proof-w128/.last_requeue_step" in text


def test_measure_ignores_data_eval(
    tmp_path: Path, fixture_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    lm_dir = tmp_path / "lm"
    tiny_lm().save_pretrained(lm_dir)
    cfg = tiny_fold_run_cfg(tmp_path / "run", fixture_dataset, lm_dir)
    cfg.data.eval = {"nope": {"path": "x", "type": "no_such_eval_type"}}  # fatal if built
    config = tmp_path / "run.yaml"
    config.write_text(serialize_config(cfg))
    out = tmp_path / "measure.json"
    result = runner.invoke(
        app, ["fold", "measure", "--config", str(config), "--steps", "1", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
```

`serialize_config`, `tiny_fold_run_cfg` and `tiny_lm` are already imported in this file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/test_cli.py -q -k "proof or measure_ignores"`
Expected: the config tests fail on `output_dir` and the missing `slurm:` block, and the measure
test fails with `Unknown eval type 'no_such_eval_type'`.

- [ ] **Step 3: `measure` drops `data.eval`**

In `measure`, next to the other forced settings, add:

```python
    cfg.data.eval = None  # a cost measurement times training steps only
```

- [ ] **Step 4: The run configs**

In `configs/fold/proof-w128.yaml`, replace the header comment, set the `train` paths and run
name, add `data.eval` and append the `slurm:` block. The `fold:` block and every other `train`
value stay as they are.

```yaml
# Milestone-2 pipeline proof, pair width 128 (addendum §6; docs/FOLD.md §9–§10). Train from the
# published dataset: hf download brineylab/oplmfold-training-data --repo-type dataset \
#   --revision wwpdb-2026-10-12 --include "wwpdb/2026-10-12/*" --local-dir $OPLM_DATA/hf
# and point data.train and data.eval.valid.path at $OPLM_DATA/hf/wwpdb/2026-10-12.
# train.batch_size and gradient_accumulation_steps come from the milestone-2b B200 batch sweep
# (docs/fold/m2/README.md); max_steps is spec §6.3's stage-1 budget. On the cluster:
#   oplm slurm generate --config configs/fold/proof-w128.yaml --name fold-proof-w128 --out <dir>
# The job installs oplm at the tag named in slurm.install (docs/FOLD.md §10).
```

```yaml
  output_dir: /mnt/data/briney/oplm-runs/fold-m2c/proof-w128
  wandb_run_name: fold-proof-w128
```

```yaml
data:
  train: /mnt/data/briney/oplm-data/hf/wwpdb/2026-10-12
  eval:
    valid:                    # in-training eval of the live model (docs/FOLD.md §10)
      path: /mnt/data/briney/oplm-data/hf/wwpdb/2026-10-12
      type: fold_monomer
      target_set: valid       # 64 single-chain entries, 75-391 residues
      every: {steps: 2000}
  num_workers: 4
  pin_memory: true
  prefetch_factor: 4
slurm:
  partition: hpc-mid
  time_limit: "168:00:00"
  nodes: 1
  gpus_per_node: 8
  cpus_per_task: 128
  exclusive: true
  mem: "0"
  log_dir: /mnt/home/briney/logs
  env_file: /mnt/home/briney/.env
  container_image: /mnt/data/containers/deeplearning_v2026-05-26.sqsh
  container_mounts:
    - /mnt/data:/mnt/data
    - /tmp:/tmp
  max_requeues: 20
  # One shell line inside the container, so no single quotes. A Python 3.12 venv per run (esm
  # needs >= 3.12 and transformers < 5) resolved in ONE install and checked with `uv pip check`
  # (docs/fold/m2/README.md, run 1, failure 3); oplm from the immutable tag so every requeue runs
  # identical code; ESM-C weights cached on shared storage.
  install: >-
    export HF_HOME=/mnt/data/briney/hf-cache &&
    python -m pip install --quiet uv &&
    python -m uv venv --allow-existing --seed --python 3.12 /mnt/data/briney/fold-m2c/venv-w128 &&
    python -m uv pip install --quiet --python /mnt/data/briney/fold-m2c/venv-w128/bin/python
    "oplm[train,fold,esmc] @ git+https://github.com/briney/oplm@fold-m2c-runs" &&
    python -m uv pip check --python /mnt/data/briney/fold-m2c/venv-w128/bin/python &&
    source /mnt/data/briney/fold-m2c/venv-w128/bin/activate
```

`configs/fold/proof-w256.yaml` gets the same edits with `w256` in place of `w128` everywhere:
`output_dir`, `wandb_run_name`, the `generate` line in the header, and all four venv paths. Then
`diff` the two configs: they must differ only in `pair_width`, the batch/accumulation lines,
those names, and the header's width.

Check `uv venv --allow-existing` against the container's uv. `python -m pip install uv` installs
the current release, and `--allow-existing` has been available since uv 0.4. Re-running the
install on a requeue is then a no-op.

- [ ] **Step 5: The offline eval job** (`docs/fold/b200-m2c-eval.sbatch`)

```bash
#!/bin/bash
# Milestone-2c offline evaluation (docs/FOLD.md §10; docs/fold/m2/README.md): the EMA weights of
# every permanent checkpoint (step % 10000 == 0) of both proof runs, scored with `oplm fold eval`
# on the cameo and casp14 target sets, one checkpoint per GPU, eight at a time. One-shot job; a
# failing evaluation is recorded in status.txt and the others still run.
#
# Submit from a login node after both runs finish:
#   sbatch docs/fold/b200-m2c-eval.sbatch
# Overrides (environment variables, all optional):
#   OPLM_GIT_REF=fold-m2c-runs                        tag or sha to install
#   OPLM_RUNS=/mnt/data/<you>/oplm-runs/fold-m2c      parent of proof-w128/ and proof-w256/
#   OPLM_DATASET=/mnt/data/<you>/oplm-data/hf/wwpdb/2026-10-12
#   OPLM_VENV=/mnt/data/<you>/fold-m2c/venv-eval
#   OPLM_RESULTS=/mnt/home/<you>/fold-m2c/eval/<jobid>
#   OPLM_KEEP_EVERY=10000
#
#SBATCH --job-name=fold-m2c-eval
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/home/briney/logs/%x_%j.out
#SBATCH --error=/mnt/home/briney/logs/%x_%j.err

set -euo pipefail

ENV_FILE="${OPLM_ENV_FILE:-/mnt/home/${SLURM_JOB_USER}/.env}"
CONTAINER="${OPLM_CONTAINER:-/mnt/data/containers/deeplearning_v2026-05-26.sqsh}"
GIT_REF="${OPLM_GIT_REF:-fold-m2c-runs}"
RUNS="${OPLM_RUNS:-/mnt/data/${SLURM_JOB_USER}/oplm-runs/fold-m2c}"
DATASET="${OPLM_DATASET:-/mnt/data/${SLURM_JOB_USER}/oplm-data/hf/wwpdb/2026-10-12}"
VENV="${OPLM_VENV:-/mnt/data/${SLURM_JOB_USER}/fold-m2c/venv-eval}"
RESULTS="${OPLM_RESULTS:-/mnt/home/${SLURM_JOB_USER}/fold-m2c/eval/${SLURM_JOB_ID}}"
KEEP_EVERY="${OPLM_KEEP_EVERY:-10000}"
export GIT_REF RUNS DATASET VENV RESULTS KEEP_EVERY

source "$ENV_FILE"  # creates JOB_WORK_DIR
mkdir -p "$RESULTS" "$(dirname "$VENV")" "$JOB_WORK_DIR"
echo "=== $(date -Is) start; job=$SLURM_JOB_ID node=$(hostname) results=$RESULTS ==="

INNER="$RESULTS/run-inner.sh"
cat > "$INNER" <<'EOF'
set -uo pipefail
export HF_HOME="/mnt/data/${SLURM_JOB_USER}/hf-cache"
export OMP_NUM_THREADS=8
: > "$RESULTS/status.txt"
python -m pip install --quiet uv
python -m uv venv --allow-existing --seed --python 3.12 "$VENV"
python -m uv pip install --quiet --python "$VENV/bin/python" \
  "oplm[train,fold,esmc] @ git+https://github.com/briney/oplm@$GIT_REF"
python -m uv pip check --python "$VENV/bin/python" || { echo "install broken" >> "$RESULTS/status.txt"; exit 1; }
source "$VENV/bin/activate"
python -m uv pip freeze | grep -i -E '^(oplm|esm|torch|transformers|accelerate|cuequivariance)' > "$RESULTS/pip-freeze.txt"
nvidia-smi > "$RESULTS/nvidia-smi.txt" 2>&1 || true

jobs=()
for run in proof-w128 proof-w256; do
  for ckpt in "$RUNS/$run"/checkpoint-*; do
    step="${ckpt##*-}"
    [[ "$step" =~ ^[0-9]+$ && -d "$ckpt/hf_ema" ]] || continue
    (( step % KEEP_EVERY == 0 )) && jobs+=("$run $step $ckpt")
  done
done
echo "evaluating ${#jobs[@]} checkpoints"

gpu=0
for job in "${jobs[@]}"; do
  read -r run step ckpt <<< "$job"
  mkdir -p "$RESULTS/$run"
  (
    CUDA_VISIBLE_DEVICES=$gpu oplm fold eval --model "$ckpt" --weights ema --data "$DATASET" \
      --target-set cameo --target-set casp14 --out "$RESULTS/$run/step-$step" \
      > "$RESULTS/$run/step-$step.log" 2>&1
    echo "$run step-$step $?" >> "$RESULTS/status.txt"
  ) &
  gpu=$(( (gpu + 1) % 8 ))
  if (( gpu == 0 )); then wait; fi
done
wait
echo "=== status ==="; cat "$RESULTS/status.txt"
EOF

srun --nodes=1 --ntasks-per-node=1 --export=ALL \
  --container-image="$CONTAINER" \
  --container-mounts="/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},/mnt/data:/mnt/data,/tmp:/tmp" \
  --container-workdir="$JOB_WORK_DIR" \
  --no-container-mount-home \
  bash "$INNER"

echo "=== $(date -Is) done; copy $RESULTS into docs/fold/m2/eval/ (docs/fold/m2/README.md) ==="
```

Check it with `bash -n docs/fold/b200-m2c-eval.sbatch`.

- [ ] **Step 6: `docs/FOLD.md`**

Change the status line to: "**Status: milestone 2c (evaluation and proof runs) implemented, proof
runs pending; milestone 2b (losses and training) accepted 2026-10-13 …**". Keep the rest of the
line.

Insert a new section after §9:

```markdown
## 10. Milestone 2c: evaluation and the proof runs (`oplm.fold.eval`)

**Metrics** (`metrics.py`, addendum §6). Predicted and reference atoms correspond by index: both
come from the same entity sequence and the same reference-conformer templates. Reference atoms
that are unresolved are excluded, never scored.

- **TM-score** runs over resolved CA atoms and is normalized by their count. `d0 = 1.24·(L−15)^⅓ −
  1.8` (0.5 for L ≤ 21). It uses the TM-score program's fragment-seeded iterative superposition
  rather than one RMSD fit. It matches TM-align (`tmtools`) to machine precision on identity alignments.
- **lDDT** is the global score: reference pairs within 15 Å, thresholds 0.5/1/2/4 Å, and no pairs
  inside one residue for the all-atom score. It matches Boltz-2's `lddt_dist`.
- **pLDDT** is the mean over residues (0–1). **pTM** is that of the best sample.
- Scores are full-chain; CASP domain scoring is not reproduced.

**Targets** (`targets.py`). A target set is `targets/<name>.json` when that file exists, else an
index split:

- An item with a chain names an author chain.
- An item without one (CASP14) contributes one chain per distinct entity.
- Chains longer than the model's `lm_max_chain_length` (1024) or with fewer than 3 resolved CA
  atoms are skipped, and the skip is reported.
- On the published snapshot: CAMEO has 752 of its 755 listed chains present. CASP14 has 131 of
  133 listed entries present, which hold 248 distinct entities.

**In training.** A `data.eval` entry of `type: fold_monomer` scores a target set with the live
model, sharded across ranks and gathered. It logs `eval/<name>/{tm_score, lddt_ca, lddt, plddt,
ptm, targets, failed}`. Its options are `target_set` (default `valid`), `max_targets`,
`max_length`, `num_loops`, `num_steps`, `num_samples` (1) and `seed` (0). A target that raises
(OOM included) counts as `failed`. `oplm.fold.training` imports `oplm.fold.eval`, which registers
the type; MLM tooling never imports it.

**Offline.**

    oplm fold eval --model <run>/checkpoint-50000 --weights ema \
      --data $OPLM_DATA/hf/wwpdb/2026-10-12 --target-set cameo --target-set casp14 --out <dir>

It writes `targets.csv` (one row per target: status, error, seconds and the five metrics) and
`summary.json` (coverage, failures, aggregates, sampling settings, versions).
`oplm fold compare A B` prints the paired B − A difference over targets scored in both, with a
bootstrap 95 % CI.

**The proof runs.** `configs/fold/proof-w128.yaml` and `proof-w256.yaml` set:

- one node of 8 B200s, DDP, global batch 128, 50k steps, EMA 0.999;
- checkpoints every 2000 steps with permanent keeps every 10000;
- `valid` eval every 2000 steps.

`oplm slurm generate` launches `oplm.fold.train` for them (docs/SLURM.md §9). Each attempt builds
or reuses a per-run Python 3.12 venv with `oplm[train,fold,esmc]` from the tag `fold-m2c-runs`.
A resume is forced once per run with `scancel --signal=USR1 <jobid>` (docs/SLURM.md §8).
`docs/fold/b200-m2c-eval.sbatch` then scores every permanent checkpoint's `hf_ema/` on CAMEO and
CASP14. Results, the width decision and the reproduction commands are in
[`docs/fold/m2/README.md`](fold/m2/README.md).
```

Renumber nothing else. If later sections reference "§10" for something else, run
`grep -n "§10\|§11" docs/FOLD.md` and fix them.

- [ ] **Step 7: Run the tests, lint, commit**

```bash
.venv/bin/python -m pytest tests/fold/test_cli.py tests/slurm -q
.venv/bin/python -m pytest tests/fold -q -m "not slow"
bash -n docs/fold/b200-m2c-eval.sbatch
.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/cli.py
VIRTUAL_ENV=.venv .venv/bin/ty check src/
git add configs/fold src/oplm/fold/cli.py docs/FOLD.md docs/fold/b200-m2c-eval.sbatch tests/fold/test_cli.py
git commit -m "feat(fold): proof-run configs with slurm and eval blocks; offline eval job; FOLD.md §10"
```

After Task 7, the branch gets the whole-branch review and goes up as one PR against `main`. Run
the full fast suite (`.venv/bin/python -m pytest -m "not slow"`) and the fold slow suite first.
Tasks 8–9 start only after that PR merges.

---

### Task 8: The two proof runs, with one forced resume each (user-run, on the cluster)

**Files:**
- Create (on branch `docs/fold-m2c-results` off the merged `main`):
  - `docs/fold/m2/runs/proof-w128.md` and `docs/fold/m2/runs/proof-w256.md` (log excerpts);
  - `docs/fold/m2/runs/*.out.txt` (wrapper logs, trimmed).

- [ ] **Step 1: Tag the code the runs install**

After the Task 1–7 PR merges, with `main` fast-forwarded locally:

```bash
git tag fold-m2c-runs <merge commit sha>
git push origin fold-m2c-runs
```

The tag never moves. A code change mid-run means a new tag, a config edit and a fresh run.

- [ ] **Step 2: Generate and submit both runs** (cluster login node, in a checkout of
`fold-m2c-runs` with oplm installed for `generate`)

```bash
oplm slurm generate --config configs/fold/proof-w128.yaml --name fold-proof-w128 --out ~/fold-m2c/jobs/w128
oplm slurm generate --config configs/fold/proof-w256.yaml --name fold-proof-w256 --out ~/fold-m2c/jobs/w256
sbatch ~/fold-m2c/jobs/w128/fold-proof-w128.sbatch
sbatch ~/fold-m2c/jobs/w256/fold-proof-w256.sbatch
```

The runs can share a node pool. Each needs one 8-GPU node, for about 142 h (w128) and 240 h
(w256), plus requeue overhead.

- [ ] **Step 3: Check the first eval and the first checkpoint**

Right after launch, then again after step 2000 (about 6 h for w128, 10 h for w256):
- W&B project `oplm-fold` shows `eval/valid/tm_score` with `targets` = 64 and `failed` = 0. The
  config sets `at_start`, so the first point arrives right after step 1, and the next at step 2000.
- `checkpoint-2000/` has both `hf/` and `hf_ema/`.
- Loss curves look like the M2b measurement's first steps (distogram falling).

If `failed` > 0, read the training log: each failed target is logged as a warning with its error.

- [ ] **Step 4: Force one resume per run**

Between steps 2000 and 4000, for each job:

```bash
scancel --signal=USR1 <jobid>
```

Then confirm from `/mnt/home/briney/logs/fold-proof-w<width>_<jobid>.out` and `.err`:
1. `.err` shows `Drain requested: checkpoint saved at step N; exiting 85`.
2. `.out` shows `requeueing (exit=1, drained=1, restarts=0, step=N)`.
3. `.out` shows `start; restart_count=1`, then `Auto-resuming from checkpoint-N`.
4. W&B continues the same run id: `train/global_step` goes on from N+1 with no gap or restart,
   and the next `eval/valid/*` point arrives at step 4000.

- [ ] **Step 5: Record and commit the evidence**

Copy the four excerpts (with timestamps) and the W&B run URL into
`docs/fold/m2/runs/proof-w<width>.md`. Note any unplanned requeues, for example node failures
or wall-time drains, with their steps. Commit on `docs/fold-m2c-results`:

```bash
git add docs/fold/m2/runs
git commit -m "docs(fold): milestone-2c proof runs launched; forced resume verified for both widths"
```

---

### Task 9: Offline evaluation, the width decision, and milestone-2 acceptance

**Files:**
- Create:
  - `docs/fold/m2/eval/<run>/step-<N>/{targets.csv,summary.json}`
  - `docs/fold/m2/eval/{status.txt,pip-freeze.txt}`
  - `docs/fold/m2/eval/valid-curves.csv`
- Modify: `docs/fold/m2/README.md`, `docs/FOLD.md`

- [ ] **Step 1: Run the offline evaluation** (after both runs finish 50k steps)

```bash
sbatch docs/fold/b200-m2c-eval.sbatch
```

The job scores 10 checkpoints, steps 10k through 50k for each width, at about 1 h per checkpoint
per GPU, in two waves. Copy `$RESULTS` (its path is printed in the job log) to the dev box, then
into `docs/fold/m2/eval/`. Check `status.txt`: every line must end in ` 0`.

- [ ] **Step 2: Export the in-training `valid` curves**

```python
import wandb, csv
api = wandb.Api()
with open("docs/fold/m2/eval/valid-curves.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["run", "step", "tm_score", "lddt", "plddt", "ptm"])
    for name in ("fold-proof-w128", "fold-proof-w256"):
        (run,) = api.runs("thebrineylab/oplm-fold", filters={"display_name": name})
        keys = ["train/global_step", "eval/valid/tm_score", "eval/valid/lddt", "eval/valid/plddt", "eval/valid/ptm"]
        for row in run.scan_history(keys=keys):
            w.writerow([name] + [row[k] for k in keys])
```

- [ ] **Step 3: Build the results tables**

```python
import json, pathlib
root = pathlib.Path("docs/fold/m2/eval")
print("| run | step | set | scored | failed | skipped | TM | lDDT-CA | lDDT | pLDDT | pTM |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for summary in sorted(root.glob("proof-w*/step-*/summary.json"), key=lambda p: (p.parts[-3], int(p.parts[-2][5:]))):
    s = json.loads(summary.read_text())
    for name, block in s["target_sets"].items():
        m, c = block["metrics"], block["coverage"]
        print(f"| {summary.parts[-3]} | {summary.parts[-2][5:]} | {name} | {int(m['targets'])} | "
              f"{int(m['failed'])} | {len(c['skipped'])} | {m['tm_score']:.3f} | {m['lddt_ca']:.3f} | "
              f"{m['lddt']:.3f} | {m['plddt']:.3f} | {m['ptm']:.3f} |")
```

Then compare the widths at matched steps, pooled and per set:

```bash
for step in 10000 20000 30000 40000 50000; do
  oplm fold compare docs/fold/m2/eval/proof-w128/step-$step docs/fold/m2/eval/proof-w256/step-$step
done
oplm fold compare docs/fold/m2/eval/proof-w128/step-50000 docs/fold/m2/eval/proof-w256/step-50000 --target-set cameo
oplm fold compare docs/fold/m2/eval/proof-w128/step-50000 docs/fold/m2/eval/proof-w256/step-50000 --target-set casp14
```

- [ ] **Step 4: Check acceptance (addendum §6, base spec §10)**

1. **Held-out TM-score improves across checkpoints for both runs.** On CAMEO and on CASP14, the
   step-50000 mean exceeds the step-10000 mean. Also check `oplm fold compare` between those two
   steps of the same run: its CI must lie above 0. If a curve is non-monotone in between, state
   so; that is not a failure.
2. **Resume was exercised.** Task 8's evidence for both runs.
3. **The width decision follows the preregistered rule.** Choose 256 only if the pooled step-50000
   `oplm fold compare proof-w128 proof-w256` has `ci_low > 0.01`; otherwise choose 128. Record
   the numbers, the cost ratio (17.31 s against 10.22 s per optimizer step on a B200;
   docs/FOLD.md §6 for the M0 table) and the chosen width.
4. **Gates.** Spec §10 decision gates owned by M2 (pair width; LR and global batch for the proof):
   record the width from item 3. Record LR 1e-3 with 1000 warmup steps, global batch 128 and
   whether the curves suggest changing either for milestone 3, from the `train/loss` and
   `valid` curves.

If item 1 fails, milestone 2 is not accepted. Write the README section anyway, with the
diagnosis, and stop.

- [ ] **Step 5: Write `docs/fold/m2/README.md` and `docs/FOLD.md`**

Add a "Milestone 2c: proof runs and the width decision" section to `docs/fold/m2/README.md`
covering:
- the two runs: job ids, W&B URLs, wall time, requeues, and the forced resume with a pointer to
  `runs/`;
- the eval job: id, node, `pip-freeze.txt`;
- the Step 3 table;
- the `valid` curves summarized at 10k-step marks from `valid-curves.csv`;
- the matched-step comparisons;
- the acceptance checklist with each item's evidence;
- the decision;
- the reproduction commands (Task 8 Steps 1–2, Task 9 Steps 1–3).

In `docs/FOLD.md` §10, add a "Results" paragraph: the 50k numbers for both widths, the decision,
and a link. Then change the status line to "**Status: milestone 2 (pipeline proof) accepted
<date>: width <W> chosen (docs/fold/m2/); …**".

- [ ] **Step 6: Commit and open the results PR**

```bash
git add docs/fold/m2 docs/FOLD.md
git commit -m "docs(fold): milestone-2c evaluation, the width decision, milestone 2 accepted"
git push -u origin docs/fold-m2c-results
gh pr create --title "docs(fold): milestone 2 accepted — proof runs, evaluation, width decision" \
  --body-file /tmp/m2c-results-pr.md
```

The PR body summarizes the README section: the 50k table rows, the decision with its CI, the
acceptance checklist, and links to `docs/fold/m2/`. It ends with the repository's standard
attribution line.

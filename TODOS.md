# μP Sweep on SUNK/Slurm — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the phased μP learning-rate sweep on CoreWeave SUNK/Slurm instead of one local
8×B200 node, behind a general-purpose job-generation layer usable by any oplm training run, and
re-center the learning-rate grid on the region the 170M coarse sweep identified.

**Architecture:** Sweep tooling moves out of the unpackaged `scripts/` directory into the wheel and
splits in two. `oplm.slurm` turns any training config plus a `slurm:` block into sbatch scripts,
submits them, and reports status — it knows nothing about μP. `oplm.sweep` keeps the phase funnel,
ranking, and selection, and delegates all rendering and submission to `oplm.slurm`. Each phase
emits one job array per preset plus a CPU-only analyze job wired with `--dependency=afterany`.

**Tech Stack:** Python 3.11+, Typer, OmegaConf, pytest, Slurm (SUNK) with Pyxis/enroot containers,
HuggingFace Accelerate.

**Spec:** [docs/superpowers/specs/2026-07-31-mup-sweep-slurm-design.md](docs/superpowers/specs/2026-07-31-mup-sweep-slurm-design.md)

## Global Constraints

- Python 3.11+; `from __future__ import annotations` in every module.
- Max line length 100. Formatter `ruff format`; linter `ruff check`; type checker `ty` (pinned
  `0.0.40`).
- CI gates run on `src/` only: `ruff check src/`, `ruff format --check src/`, `ty check src/`,
  `pytest -m "not slow" --cov=oplm`. Run all four before declaring any task done.
- Run tests as `python -m pytest` (a bare `pytest` resolves to a linuxbrew Python without oplm).
- Type hints on every function signature. Google-style docstrings on public classes/functions.
- `subprocess.run` with a list of args. Never `shell=True`, never `os.system`.
- `pathlib.Path` throughout; `str()` only at IO boundaries.
- `oplm.slurm` MUST NOT import `oplm.sweep`. The reverse is required.
- Preserve existing μP behavior: parameterization, ranking rules, phase ordering, and the
  selection protocol are unchanged except where a task says otherwise.
- Never use `oplm.__version__` (stale at `0.0.1` vs `pyproject` `0.1.6`). Use
  `importlib.metadata.version("oplm")`.

## File Structure

**Created:**

| Path | Responsibility |
| --- | --- |
| `src/oplm/slurm/__init__.py` | Public exports for the general layer |
| `src/oplm/slurm/config.py` | `SlurmConfig` schema, phase/preset table resolution, batch planning |
| `src/oplm/slurm/render.py` | sbatch/srun script text for array and single jobs |
| `src/oplm/slurm/submit.py` | `sbatch --parsable`, dependency wiring, `squeue` status |
| `src/oplm/slurm/cli.py` | `oplm slurm generate\|submit\|status` |
| `src/oplm/sweep/__init__.py` | Public exports for the sweep layer |
| `src/oplm/sweep/cli.py` | `oplm sweep <phase>\|analyze\|status\|coord-check` |
| `configs/scaling.yaml` | Runnable production scaling config with its own `slurm:` block |
| `docs/SLURM.md` | General-layer operator docs (no μP content) |
| `tests/slurm/*` | Tests for the general layer, using plain training configs |

**Moved:**

| From | To |
| --- | --- |
| `scripts/_mup_common.py` | `src/oplm/sweep/common.py` |
| `scripts/mup_sweep.py` | `src/oplm/sweep/phases.py` |
| `scripts/mup_run.py` | `src/oplm/sweep/run.py` |
| `scripts/mup_coord_check.py` | `src/oplm/sweep/coord_check.py` |
| `tests/scripts/test_mup_common.py` | `tests/sweep/test_common.py` |
| `tests/scripts/test_mup_sweep.py` | `tests/sweep/test_phases.py` |
| `tests/scripts/test_mup_run.py` | `tests/sweep/test_run.py` |
| `tests/scripts/test_mup_coord_check_data.py` | `tests/sweep/test_coord_check.py` |

**Modified:** `src/oplm/cli.py` (wire both sub-apps), `src/oplm/training/mup.py` (record version),
`docs/LR_SWEEP.md`, `docs/MUP.md`, `docs/TRAIN.md`, `AGENTS.md`.

**Deleted:** `scripts/` entirely.

---

## Phase 1 — Move the tooling into the package

No behavior changes in this phase. The goal is that `pip install oplm[train]` provides everything,
and that the moved code passes the `src/`-scoped CI gates it was never subject to before.

### Task 1: Move sweep modules into `src/oplm/sweep/`

**Files:**
- Create: `src/oplm/sweep/__init__.py`
- Move: `scripts/_mup_common.py` → `src/oplm/sweep/common.py`
- Move: `scripts/mup_sweep.py` → `src/oplm/sweep/phases.py`
- Move: `scripts/mup_run.py` → `src/oplm/sweep/run.py`
- Move: `scripts/mup_coord_check.py` → `src/oplm/sweep/coord_check.py`
- Move: `tests/scripts/*` → `tests/sweep/*` (see File Structure table)
- Create: `tests/sweep/__init__.py`
- Delete: `scripts/`

**Interfaces:**
- Produces: `oplm.sweep.common` exporting `Params`, `RunSpec`, `PhaseManifest`, `Scaling`,
  `Optimizer`, `parse_widths`, `parse_floats`, `parse_candidates`, `num_layers_for`,
  `gradient_accumulation_steps`, `relative_path`, `write_phase`, `load_phase`, `result_metric`,
  `accelerate_argv`, `HEAD_DIM`, `PRESET_ASPECT_RATIO` — all names unchanged from
  `scripts/_mup_common.py`.
- Produces: `oplm.sweep.phases` exporting `app`, `analyze_phase`, `SMOKE_LRS`, `COARSE_LRS`,
  `OUTPUT_MULTS`.

- [ ] **Step 1: Move files with git so history follows**

```bash
mkdir -p src/oplm/sweep tests/sweep
git mv scripts/_mup_common.py src/oplm/sweep/common.py
git mv scripts/mup_sweep.py src/oplm/sweep/phases.py
git mv scripts/mup_run.py src/oplm/sweep/run.py
git mv scripts/mup_coord_check.py src/oplm/sweep/coord_check.py
git mv tests/scripts/test_mup_common.py tests/sweep/test_common.py
git mv tests/scripts/test_mup_sweep.py tests/sweep/test_phases.py
git mv tests/scripts/test_mup_run.py tests/sweep/test_run.py
git mv tests/scripts/test_mup_coord_check_data.py tests/sweep/test_coord_check.py
rm -rf scripts tests/scripts
```

- [ ] **Step 2: Create the package init**

`src/oplm/sweep/__init__.py`:

```python
"""μP learning-rate sweep tooling: phase generation, ranking, and selection."""

from __future__ import annotations

from oplm.sweep.common import (
    Params,
    PhaseManifest,
    RunSpec,
    load_phase,
    result_metric,
    write_phase,
)

__all__ = [
    "Params",
    "PhaseManifest",
    "RunSpec",
    "load_phase",
    "result_metric",
    "write_phase",
]
```

Create an empty `tests/sweep/__init__.py` (matching `tests/eval/__init__.py` etc.).

- [ ] **Step 3: Rewrite imports**

In `src/oplm/sweep/phases.py` and `src/oplm/sweep/coord_check.py`, replace
`from scripts._mup_common import (...)` with `from oplm.sweep.common import (...)`. The imported
names do not change.

In `src/oplm/sweep/common.py`, change `accelerate_argv` to launch the packaged module:

```python
    argv.extend(
        [
            "--num_processes",
            str(num_processes),
            "-m",
            "oplm.sweep.run",
            "--config",
            str(config),
            "--result",
            str(result),
        ]
    )
```

In the moved tests, replace `from scripts import mup_sweep` with `from oplm.sweep import phases`
(and update every `mup_sweep.` reference to `phases.`), and
`from scripts._mup_common import ...` with `from oplm.sweep.common import ...`. Do the same for
`scripts.mup_run` → `oplm.sweep.run` and `scripts.mup_coord_check` → `oplm.sweep.coord_check`.

- [ ] **Step 4: Run the moved tests**

Run: `python -m pytest tests/sweep/ -q`
Expected: PASS, same test count as before the move. Any failure here is an import path that was
missed, not a behavior change.

- [ ] **Step 5: Verify no `scripts.` references survive**

Run: `grep -rn "scripts\._mup_common\|scripts\.mup_\|from scripts import" --include=*.py --include=*.md . | grep -v "^./.venv\|docs/superpowers"`
Expected: no output. (`docs/superpowers/` specs and plans are historical records and are excluded
deliberately; `docs/LR_SWEEP.md` and `docs/MUP.md` are updated in Phase 10.)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: move sweep tooling into the oplm package"
```

### Task 2: Bring the moved code under the `src/` CI gates

The moved modules were previously outside `ruff check src/` and `ty check src/`. This task fixes
whatever those now report, and nothing else.

**Files:**
- Modify: `src/oplm/sweep/*.py` (as needed by the gates)

- [ ] **Step 1: Run the linter and see what the move surfaced**

Run: `ruff check src/`
Expected: possible new findings in `src/oplm/sweep/*`. Record them.

- [ ] **Step 2: Run the type checker**

Run: `ty check src/`
Expected: possible new findings in `src/oplm/sweep/*`. Record them.

- [ ] **Step 3: Fix the findings**

Fix each finding in place. Do not change runtime behavior. Two rules:
- Do not add `# type: ignore` without a specific error code and a comment explaining why.
- If a fix would change behavior, stop and leave a note in the commit body instead of guessing.

The most likely findings are `TC003`-style typing-only import placement (already suppressed with
`# noqa: TC003` at `phases.py` and `run.py` for Typer's runtime annotation resolution — keep those
suppressions, they are load-bearing) and `ty` complaints about `dict[str, object]` indexing in the
ranking helpers.

- [ ] **Step 4: Verify all four gates**

```bash
ruff check src/ && ruff format --check src/ && ty check src/ && python -m pytest tests/sweep/ -q
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A src/oplm/sweep
git commit -m "style: satisfy src CI gates for moved sweep tooling"
```

### Task 3: Wire `oplm sweep` into the CLI

**Files:**
- Create: `src/oplm/sweep/cli.py`
- Modify: `src/oplm/cli.py`
- Test: `tests/sweep/test_cli.py`

**Interfaces:**
- Produces: `oplm.sweep.cli.app` — a `typer.Typer` carrying every phase command plus `analyze` and
  `coord-check`. Mounted at `oplm sweep`.

- [ ] **Step 1: Write the failing test**

`tests/sweep/test_cli.py`:

```python
from __future__ import annotations

from typer.testing import CliRunner

from oplm.cli import app

runner = CliRunner()


def test_sweep_subcommand_is_registered() -> None:
    result = runner.invoke(app, ["sweep", "--help"])
    assert result.exit_code == 0
    for command in ("smoke", "coarse", "refine", "replicate", "transfer", "bridge", "confirm"):
        assert command in result.stdout


def test_sweep_analyze_is_registered() -> None:
    result = runner.invoke(app, ["sweep", "analyze", "--help"])
    assert result.exit_code == 0


def test_sweep_coord_check_is_registered() -> None:
    result = runner.invoke(app, ["sweep", "coord-check", "--help"])
    assert result.exit_code == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/sweep/test_cli.py -q`
Expected: FAIL — `oplm` has no `sweep` command, exit code 2.

- [ ] **Step 3: Create the sub-app**

`src/oplm/sweep/cli.py`:

```python
"""`oplm sweep` command surface."""

from __future__ import annotations

from oplm.sweep.coord_check import main as coord_check_main
from oplm.sweep.phases import app

app.command("coord-check")(coord_check_main)

__all__ = ["app"]
```

`coord_check.py` currently declares its own `typer.Typer` named `mup-coord-check` with a single
`main` command; reuse that function rather than duplicating its signature. Delete the now-unused
`app = typer.Typer(...)` line and the `if __name__ == "__main__"` block from `coord_check.py`, and
likewise delete the `if __name__ == "__main__"` block from `phases.py`.

- [ ] **Step 4: Mount it in the root CLI**

In `src/oplm/cli.py`, after the `app = typer.Typer(...)` line, add:

```python
app.add_typer(sweep_app, name="sweep", help="μP learning-rate sweep phases")
```

with `from oplm.sweep.cli import app as sweep_app` placed with the other `oplm` imports at the top
of the file, not inline.

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/sweep/test_cli.py -q`
Expected: PASS.

- [ ] **Step 6: Verify the module entrypoint still works**

Run: `python -m oplm.sweep.run --help`
Expected: exit 0, usage text for the cell runner.

- [ ] **Step 7: Run all gates and commit**

```bash
ruff check src/ && ruff format --check src/ && ty check src/ && python -m pytest tests/sweep/ -q
git add -A src/oplm tests/sweep
git commit -m "feat: expose sweep phases as oplm sweep"
```

---

## Phase 2 — Learning-rate grid and manifest-derived gates

Independent of Slurm. Do it before the Slurm work so the grid change is reviewable on its own.

### Task 4: Shift the grid and derive the smoke gates from the manifest

**Files:**
- Modify: `src/oplm/sweep/phases.py:30-32` (constants), `:417-428` (`_smoke_gated_lrs`),
  `:573-576` (smoke branch of `analyze_phase`), and the `--lrs` defaults on the `smoke` and
  `coarse` commands
- Test: `tests/sweep/test_phases.py`

**Interfaces:**
- Produces: `SMOKE_LRS = (0.0004, 0.0016, 0.0063)`,
  `COARSE_LRS = (0.0004, 0.00063, 0.001, 0.0016, 0.0025, 0.004, 0.0063)`.
- Produces: `_smoke_gated_lrs(source: Path, lrs: list[float]) -> list[float]` — signature
  unchanged, behavior now derived from the source manifest's own LR set.

- [ ] **Step 1: Write the failing tests**

Add to `tests/sweep/test_phases.py`. These use the existing helpers in that file for building a
phase directory; if it has a fixture that writes a manifest plus `result.json` files, reuse it
rather than writing a new one.

```python
def test_grid_constants_are_recentered() -> None:
    assert phases.COARSE_LRS == (0.0004, 0.00063, 0.001, 0.0016, 0.0025, 0.004, 0.0063)
    assert phases.SMOKE_LRS == (0.0004, 0.0016, 0.0063)
    # Smoke probes the coarse grid's endpoints and midpoint.
    assert phases.SMOKE_LRS[0] == phases.COARSE_LRS[0]
    assert phases.SMOKE_LRS[-1] == phases.COARSE_LRS[-1]
    assert phases.SMOKE_LRS[1] == phases.COARSE_LRS[len(phases.COARSE_LRS) // 2]


def test_no_phase_logic_hardcodes_a_learning_rate() -> None:
    """The old gates tested scores.get(0.0025) / scores.get(0.01) literally."""
    source = Path(phases.__file__).read_text()
    body = source.split("COARSE_LRS", 1)[1].split("\n", 1)[1]
    for literal in ("0.0025", "0.01", "0.0016", "0.0004"):
        assert literal not in body, f"phase logic still references the literal {literal}"


def test_smoke_gate_follows_a_custom_grid(tmp_path: Path) -> None:
    """Gates must track --lrs, not a module constant."""
    source = _write_smoke_phase(
        tmp_path,
        # Custom grid, nothing to do with SMOKE_LRS.
        results={0.002: 3.1, 0.008: 3.0, 0.032: None},
    )
    # The highest LR diverged, so it is dropped from the downstream coarse grid.
    assert phases._smoke_gated_lrs(source, [0.002, 0.008, 0.032]) == [0.002, 0.008]


def test_smoke_gate_raises_when_a_low_lr_is_non_finite(tmp_path: Path) -> None:
    source = _write_smoke_phase(
        tmp_path,
        results={0.002: 3.1, 0.008: None, 0.032: 3.5},
    )
    with pytest.raises(ValueError, match="0.008"):
        phases._smoke_gated_lrs(source, [0.002, 0.008, 0.032])


def test_analyze_smoke_requires_two_lowest_finite(tmp_path: Path) -> None:
    source = _write_smoke_phase(tmp_path, results={0.002: 3.1, 0.008: None, 0.032: 3.5})
    with pytest.raises(ValueError, match="two lowest"):
        phases.analyze_phase(source)
```

Add the helper to the same file:

```python
def _write_smoke_phase(tmp_path: Path, *, results: dict[float, float | None]) -> Path:
    """Write a minimal smoke phase directory with one cell per learning rate."""
    out = tmp_path / "smoke"
    (out / "runs").mkdir(parents=True)
    runs = []
    for lr, value in results.items():
        run_id = f"170M-lr{lr:g}"
        run_dir = out / "runs" / run_id
        run_dir.mkdir()
        if value is not None:
            (run_dir / "result.json").write_text(json.dumps({"eval": {"eval/heldout/loss": value}}))
        runs.append(
            RunSpec(
                run_id,
                f"runs/{run_id}/run.yaml",
                f"runs/{run_id}/result.json",
                {"preset": "170M", "lr": lr, "output_mult": 1.0, "depth_exponent": 0.0, "seed": 42},
            )
        )
    path = out / "phase.json"
    write_phase(
        path,
        PhaseManifest(
            version=1,
            phase="smoke",
            metric="eval/heldout/loss",
            source=None,
            runs=runs,
            ranking=[],
            selected=[],
        ),
    )
    return path
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/sweep/test_phases.py -q -k "grid or smoke_gate or hardcode or two_lowest"`
Expected: FAIL — constants are the old values and the gates reference `SMOKE_LRS` and literal
learning rates.

- [ ] **Step 3: Update the constants and the CLI defaults**

In `src/oplm/sweep/phases.py`:

```python
# Grid re-centered after the 170M coarse sweep ranked 0.0025 ~ 0.004 >> 0.0063 > 0.01 > 0.016,
# putting the winner on the old grid's lower boundary. Same 1.6x spacing, shifted one half-decade
# down, so both observed winners sit interior with headroom below and 0.0063 remains an upper
# guard. See docs/LR_SWEEP.md.
SMOKE_LRS = (0.0004, 0.0016, 0.0063)
COARSE_LRS = (0.0004, 0.00063, 0.001, 0.0016, 0.0025, 0.004, 0.0063)
OUTPUT_MULTS = (0.5, 1.0, 2.0)


def _grid_default(values: tuple[float, ...]) -> str:
    """Render an LR grid as the comma-separated string a Typer default expects."""
    return ",".join(f"{value:g}" for value in values)
```

Change the `--lrs` defaults so they cannot drift from the constants:

```python
    lrs: Annotated[str, typer.Option("--lrs")] = _grid_default(SMOKE_LRS),
```

in `smoke`, and

```python
    lrs: Annotated[str, typer.Option("--lrs")] = _grid_default(COARSE_LRS),
```

in `coarse`.

- [ ] **Step 4: Rewrite `_smoke_gated_lrs` to read the manifest**

Replace the body of `_smoke_gated_lrs` entirely:

```python
def _smoke_gated_lrs(source: Path, lrs: list[float]) -> list[float]:
    """Drop the smoke phase's highest LR from ``lrs`` when it failed to produce a finite metric.

    The gate is derived from the source manifest's own learning rates rather than from a module
    constant, so changing ``--lrs`` moves the gate with it.

    Args:
        source: Path to the completed smoke ``phase.json``.
        lrs: Candidate coarse grid.

    Returns:
        ``lrs``, minus the smoke phase's highest learning rate if that cell diverged.

    Raises:
        ValueError: If the smoke phase has fewer than two cells, or either of its two lowest
            learning rates lacks a finite metric.
    """
    phase = load_phase(source)
    phase_dir = source.resolve().parent
    runs_by_lr = {float(run.params["lr"]): run for run in phase.runs}
    phase_lrs = sorted(runs_by_lr)
    if len(phase_lrs) < 2:
        raise ValueError(f"smoke phase {source} must have at least two learning rates")
    for lr in phase_lrs[:2]:
        if result_metric(phase_dir, runs_by_lr[lr], phase.metric) is None:
            raise ValueError(f"smoke lr={lr:g} lacks a finite {phase.metric}")
    highest = phase_lrs[-1]
    if result_metric(phase_dir, runs_by_lr[highest], phase.metric) is None:
        return [lr for lr in lrs if lr != highest]
    return lrs
```

- [ ] **Step 5: Rewrite the smoke branch of `analyze_phase`**

Replace:

```python
        if phase.phase == "smoke":
            scores = {float(entry["params"]["lr"]): entry["score"] for entry in phase.ranking}
            if scores.get(0.0025) is None or scores.get(0.01) is None:
                raise ValueError("smoke requires finite validation loss at LR 0.0025 and 0.01")
```

with:

```python
        if phase.phase == "smoke":
            scores = {float(entry["params"]["lr"]): entry["score"] for entry in phase.ranking}
            ordered = sorted(scores)
            if len(ordered) < 2:
                raise ValueError("smoke requires at least two learning rates")
            missing = [lr for lr in ordered[:2] if scores[lr] is None]
            if missing:
                listed = ", ".join(f"{lr:g}" for lr in missing)
                raise ValueError(
                    "smoke requires finite validation loss at the two lowest learning rates; "
                    f"missing: {listed}"
                )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/sweep/test_phases.py -q`
Expected: PASS. Existing gate tests written against the old grid will fail — update their LR
values to the new grid; do not weaken the assertions.

- [ ] **Step 7: Run all gates and commit**

```bash
ruff check src/ && ruff format --check src/ && ty check src/ && python -m pytest tests/sweep -q
git add -A src/oplm/sweep tests/sweep
git commit -m "feat: recenter LR grid and derive smoke gates from the manifest"
```

---

## Phase 3 — `oplm.slurm` configuration layer

### Task 5: `SlurmConfig` schema and table resolution

**Files:**
- Create: `src/oplm/slurm/__init__.py`, `src/oplm/slurm/config.py`
- Create: `tests/slurm/__init__.py`, `tests/slurm/test_config.py`

**Interfaces:**
- Produces: `SlurmConfig` (frozen dataclass) with fields `partition: str`,
  `time_limit: PhaseTable[str]`, `nodes: PhaseTable[int]`, `max_batch_size: dict[str, int]`,
  `gpus_per_node: int`, `cpus_per_task: int`, `mem: str`, `exclusive: bool`, `log_dir: Path`,
  `env_file: Path`, `container_image: Path`, `container_mounts: tuple[str, ...]`, `install: str`,
  `max_concurrent: int`, `account: str | None`.
- Produces: `SlurmConfig.from_mapping(raw: Mapping[str, object]) -> SlurmConfig`.
- Produces: `load_slurm_config(config_path: Path) -> SlurmConfig`.
- Produces: `PhaseTable[T].resolve(*, phase: str | None, preset: str | None) -> T`.

- [ ] **Step 1: Write the failing tests**

`tests/slurm/test_config.py`:

```python
from __future__ import annotations

import pytest

from oplm.slurm.config import PhaseTable, SlurmConfig

RAW = {
    "partition": "hpc-mid",
    "time_limit": {"default": "168:00:00", "analyze": "01:00:00"},
    "cpus_per_task": 128,
    "gpus_per_node": 8,
    "exclusive": True,
    "mem": "0",
    "log_dir": "/mnt/home/briney/logs",
    "env_file": "/mnt/home/briney/.env",
    "container_image": "/mnt/data/containers/dl.sqsh",
    "container_mounts": ["/mnt/data:/mnt/data", "/tmp:/tmp"],
    "install": "pip install oplm[train]",
    "max_concurrent": 4,
    "nodes": {
        "default": {"170M": 1, "400M": 4, "800M": 8, "1B": 8},
        "bridge": {"170M": 4},
        "confirm": {"800M": 8},
    },
    "max_batch_size": {"170M": 256, "400M": 256, "800M": 256, "1B": 128},
}


def test_scalar_table_applies_everywhere() -> None:
    table = PhaseTable.from_value(4, name="nodes")
    assert table.resolve(phase=None, preset=None) == 4
    assert table.resolve(phase="bridge", preset="800M") == 4


def test_preset_table_without_default_key() -> None:
    table = PhaseTable.from_value({"170M": 1, "400M": 4}, name="nodes")
    assert table.resolve(phase=None, preset="400M") == 4
    with pytest.raises(KeyError, match="800M"):
        table.resolve(phase=None, preset="800M")


def test_phase_override_beats_default() -> None:
    cfg = SlurmConfig.from_mapping(RAW)
    assert cfg.nodes.resolve(phase="coarse", preset="170M") == 1
    assert cfg.nodes.resolve(phase="bridge", preset="170M") == 4
    assert cfg.nodes.resolve(phase="confirm", preset="800M") == 8
    # A phase override that omits a preset falls back to default for that preset.
    assert cfg.nodes.resolve(phase="bridge", preset="400M") == 4


def test_time_limit_resolves_per_phase() -> None:
    """`time_limit` entries are bare values, not preset maps: they apply to every preset."""
    cfg = SlurmConfig.from_mapping(RAW)
    assert cfg.time_limit.resolve(phase="coarse", preset="170M") == "168:00:00"
    assert cfg.time_limit.resolve(phase="analyze", preset=None) == "01:00:00"
    # A phase with no override falls back to default regardless of preset.
    assert cfg.time_limit.resolve(phase="transfer", preset="1B") == "168:00:00"


def test_exact_preset_beats_wildcard() -> None:
    table = PhaseTable.from_value({"default": 1, "bridge": {"170M": 4}}, name="nodes")
    assert table.resolve(phase="bridge", preset="170M") == 4
    # bridge has no 800M entry and no wildcard, so default's wildcard applies.
    assert table.resolve(phase="bridge", preset="800M") == 1
    assert table.resolve(phase="coarse", preset="170M") == 1


def test_missing_required_field_raises() -> None:
    raw = {key: value for key, value in RAW.items() if key != "partition"}
    with pytest.raises(ValueError, match="partition"):
        SlurmConfig.from_mapping(raw)


@pytest.mark.parametrize("field", ["gpus_per_node", "cpus_per_task", "max_concurrent"])
def test_non_positive_ints_rejected(field: str) -> None:
    raw = {**RAW, field: 0}
    with pytest.raises(ValueError, match=field):
        SlurmConfig.from_mapping(raw)


def test_non_positive_max_batch_size_rejected() -> None:
    raw = {**RAW, "max_batch_size": {"170M": 0}}
    with pytest.raises(ValueError, match="max_batch_size"):
        SlurmConfig.from_mapping(raw)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/slurm/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'oplm.slurm'`.

- [ ] **Step 3: Implement `PhaseTable` and `SlurmConfig`**

`src/oplm/slurm/config.py`:

```python
"""Schema and resolution for the ``slurm:`` block of an oplm training config.

The block is general: it describes how to turn a training config into Slurm job scripts and
carries no μP or sweep concepts. ``load_config`` tolerates it as an unknown top-level key
(``OmegaConf.set_struct(base, False)``), and ``serialize_config`` omits it, so a generated
per-cell ``run.yaml`` never carries cluster settings.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_KEY = "default"
# Stands in for "any preset" when a phase entry carries a bare value rather than a preset map,
# e.g. `time_limit: {default: "168:00:00", analyze: "01:00:00"}`.
_ANY_KEY = "*"


@dataclass(frozen=True)
class PhaseTable[T]:
    """A setting that may be a scalar, per-preset, or per-phase-and-preset.

    Four accepted YAML forms::

        nodes: 4                                          # one value for every job
        nodes: {170M: 1, 400M: 4}                         # per preset
        nodes: {default: {170M: 1}, bridge: {170M: 4}}    # per phase, then per preset
        time_limit: {default: "168:00:00", analyze: "1:00:00"}   # per phase, no preset dimension

    The presence of a ``default`` key is what distinguishes the last two forms from the second.
    A phase entry that is not itself a mapping applies to every preset in that phase.
    """

    name: str
    scalar: T | None
    tables: dict[str, dict[str, T]]

    @classmethod
    def from_value(cls, value: object, *, name: str) -> PhaseTable[T]:
        """Build a table from any of the four accepted forms."""
        if not isinstance(value, Mapping):
            return cls(name=name, scalar=value, tables={})  # ty: ignore[invalid-argument-type]
        if _DEFAULT_KEY in value:
            tables: dict[str, dict[str, T]] = {}
            for phase, sub in value.items():
                if isinstance(sub, Mapping):
                    tables[str(phase)] = {str(preset): entry for preset, entry in sub.items()}
                else:
                    tables[str(phase)] = {_ANY_KEY: sub}
            return cls(name=name, scalar=None, tables=tables)
        return cls(
            name=name,
            scalar=None,
            tables={_DEFAULT_KEY: {str(preset): entry for preset, entry in value.items()}},
        )

    def resolve(self, *, phase: str | None, preset: str | None) -> T:
        """Resolve the value for one (phase, preset) pair.

        Looks in the phase's own table first, then the ``default`` table; within each, an exact
        preset match wins over the ``*`` wildcard.

        Raises:
            KeyError: If no entry covers ``preset``.
        """
        if self.scalar is not None:
            return self.scalar
        for table in (self.tables.get(phase or ""), self.tables.get(_DEFAULT_KEY)):
            if table is None:
                continue
            if preset is not None and preset in table:
                return table[preset]
            if _ANY_KEY in table:
                return table[_ANY_KEY]
        raise KeyError(f"{self.name} has no entry for phase={phase!r} preset={preset!r}")


def _require(raw: Mapping[str, Any], key: str) -> Any:
    if key not in raw:
        raise ValueError(f"slurm config is missing required field {key!r}")
    return raw[key]


def _positive_int(raw: Mapping[str, Any], key: str, default: int | None = None) -> int:
    value = raw.get(key, default)
    if value is None:
        raise ValueError(f"slurm config is missing required field {key!r}")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"slurm config {key} must be a positive int, got {value!r}")
    return value


@dataclass(frozen=True)
class SlurmConfig:
    """Validated ``slurm:`` block."""

    partition: str
    time_limit: PhaseTable[str]
    nodes: PhaseTable[int]
    max_batch_size: dict[str, int]
    log_dir: Path
    env_file: Path
    container_image: Path
    container_mounts: tuple[str, ...]
    install: str
    gpus_per_node: int = 8
    cpus_per_task: int = 128
    mem: str = "0"
    exclusive: bool = True
    max_concurrent: int = 4
    account: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SlurmConfig:
        """Validate and build from the raw YAML mapping.

        Raises:
            ValueError: On a missing required field or an out-of-range value.
        """
        max_batch = {
            str(preset): int(value) for preset, value in dict(raw.get("max_batch_size", {})).items()
        }
        bad = sorted(preset for preset, value in max_batch.items() if value < 1)
        if bad:
            raise ValueError(f"slurm config max_batch_size must be >= 1; bad presets: {bad}")
        return cls(
            partition=str(_require(raw, "partition")),
            time_limit=PhaseTable.from_value(_require(raw, "time_limit"), name="time_limit"),
            nodes=PhaseTable.from_value(_require(raw, "nodes"), name="nodes"),
            max_batch_size=max_batch,
            log_dir=Path(str(_require(raw, "log_dir"))),
            env_file=Path(str(_require(raw, "env_file"))),
            container_image=Path(str(_require(raw, "container_image"))),
            container_mounts=tuple(str(mount) for mount in _require(raw, "container_mounts")),
            install=str(_require(raw, "install")),
            gpus_per_node=_positive_int(raw, "gpus_per_node", 8),
            cpus_per_task=_positive_int(raw, "cpus_per_task", 128),
            mem=str(raw.get("mem", "0")),
            exclusive=bool(raw.get("exclusive", True)),
            max_concurrent=_positive_int(raw, "max_concurrent", 4),
            account=str(raw["account"]) if raw.get("account") is not None else None,
        )


def load_slurm_config(config_path: Path) -> SlurmConfig:
    """Read the ``slurm:`` block out of an oplm training config.

    Raises:
        ValueError: If the config has no ``slurm:`` block.
    """
    from omegaconf import OmegaConf

    raw = OmegaConf.load(config_path)
    block = OmegaConf.select(raw, "slurm")
    if block is None:
        raise ValueError(f"{config_path} has no `slurm:` block")
    return SlurmConfig.from_mapping(
        dict(OmegaConf.to_container(block, resolve=True))  # ty: ignore[invalid-argument-type]
    )
```

`src/oplm/slurm/__init__.py`:

```python
"""Turn an oplm training config into Slurm job scripts. Knows nothing about μP."""

from __future__ import annotations

from oplm.slurm.config import SlurmConfig, load_slurm_config

__all__ = ["SlurmConfig", "load_slurm_config"]
```

Create an empty `tests/slurm/__init__.py`.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/slurm/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff check src/ && ruff format --check src/ && ty check src/
git add -A src/oplm/slurm tests/slurm
git commit -m "feat: add slurm config schema and phase/preset resolution"
```

### Task 6: Batch planning

**Files:**
- Modify: `src/oplm/slurm/config.py`
- Test: `tests/slurm/test_config.py`

**Interfaces:**
- Produces: `BatchPlan` frozen dataclass with `per_device_batch: int`,
  `gradient_accumulation_steps: int`, `world_size: int`.
- Produces: `resolve_batch_plan(*, global_examples: int, nodes: int, gpus_per_node: int,
  max_batch_size: int) -> BatchPlan`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/slurm/test_config.py`:

```python
from oplm.slurm.config import BatchPlan, resolve_batch_plan


@pytest.mark.parametrize(
    ("global_examples", "nodes", "cap", "expected"),
    [
        # The spec's node table: every row resolves to accum == 1.
        (2048, 1, 256, BatchPlan(per_device_batch=256, gradient_accumulation_steps=1, world_size=8)),
        (2048, 4, 256, BatchPlan(per_device_batch=64, gradient_accumulation_steps=1, world_size=32)),
        (2048, 8, 256, BatchPlan(per_device_batch=32, gradient_accumulation_steps=1, world_size=64)),
        (2048, 8, 128, BatchPlan(per_device_batch=32, gradient_accumulation_steps=1, world_size=64)),
        (8192, 4, 256, BatchPlan(per_device_batch=256, gradient_accumulation_steps=1, world_size=32)),
        (8192, 8, 256, BatchPlan(per_device_batch=128, gradient_accumulation_steps=1, world_size=64)),
    ],
)
def test_batch_plan_matches_the_spec_table(
    global_examples: int, nodes: int, cap: int, expected: BatchPlan
) -> None:
    assert (
        resolve_batch_plan(
            global_examples=global_examples, nodes=nodes, gpus_per_node=8, max_batch_size=cap
        )
        == expected
    )


def test_cap_forces_accumulation() -> None:
    # 2048 / 8 = 256 per device at accum 1, over a cap of 128, so accum must rise to 2.
    plan = resolve_batch_plan(global_examples=2048, nodes=1, gpus_per_node=8, max_batch_size=128)
    assert plan == BatchPlan(per_device_batch=128, gradient_accumulation_steps=2, world_size=8)


def test_indivisible_global_batch_raises() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        resolve_batch_plan(global_examples=2048, nodes=3, gpus_per_node=8, max_batch_size=256)


def test_global_batch_smaller_than_world_raises() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        resolve_batch_plan(global_examples=4, nodes=1, gpus_per_node=8, max_batch_size=256)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/slurm/test_config.py -q -k batch`
Expected: FAIL — `ImportError: cannot import name 'BatchPlan'`.

- [ ] **Step 3: Implement**

Append to `src/oplm/slurm/config.py`:

```python
@dataclass(frozen=True)
class BatchPlan:
    """Per-device batch and accumulation for one cell."""

    per_device_batch: int
    gradient_accumulation_steps: int
    world_size: int


def resolve_batch_plan(
    *, global_examples: int, nodes: int, gpus_per_node: int, max_batch_size: int
) -> BatchPlan:
    """Derive per-device batch and accumulation from the global batch and node count.

    Picks the smallest accumulation that yields an integer per-device batch no larger than
    ``max_batch_size``. Node counts are chosen for wall time, so per-device batch is derived
    rather than configured; an infeasible combination is an error, never a silently adjusted
    global batch.

    Args:
        global_examples: Target global batch in examples per optimizer step.
        nodes: Node count for this cell.
        gpus_per_node: Processes per node.
        max_batch_size: Memory cap on per-device batch for this preset.

    Returns:
        The resolved plan.

    Raises:
        ValueError: If the global batch is not divisible by the world size, or no accumulation
            brings the per-device batch within ``max_batch_size``.
    """
    if global_examples < 1 or nodes < 1 or gpus_per_node < 1 or max_batch_size < 1:
        raise ValueError(
            "global_examples, nodes, gpus_per_node, and max_batch_size must all be >= 1; got "
            f"{global_examples=}, {nodes=}, {gpus_per_node=}, {max_batch_size=}"
        )
    world_size = nodes * gpus_per_node
    if global_examples % world_size != 0:
        raise ValueError(
            f"global batch {global_examples} is not divisible by world size {world_size} "
            f"({nodes} nodes x {gpus_per_node} GPUs)"
        )
    base = global_examples // world_size
    for accum in range(1, base + 1):
        if base % accum != 0:
            continue
        per_device = base // accum
        if per_device <= max_batch_size:
            return BatchPlan(
                per_device_batch=per_device,
                gradient_accumulation_steps=accum,
                world_size=world_size,
            )
    raise ValueError(
        f"no accumulation brings per-device batch within {max_batch_size} for "
        f"global batch {global_examples} on {nodes} nodes"
    )
```

Add `BatchPlan` and `resolve_batch_plan` to `src/oplm/slurm/__init__.py`'s imports and `__all__`.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/slurm/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff check src/ && ruff format --check src/ && ty check src/
git add -A src/oplm/slurm tests/slurm
git commit -m "feat: derive per-device batch and accumulation from node count"
```

---

## Phase 4 — `oplm.slurm` rendering

### Task 7: Render single and array job scripts

**Files:**
- Create: `src/oplm/slurm/render.py`
- Test: `tests/slurm/test_render.py`

**Interfaces:**
- Produces: `JobSpec` frozen dataclass: `name: str`, `nodes: int`, `time_limit: str`,
  `command: str`, `array_size: int | None = None`, `array_cells_file: Path | None = None`,
  `gres: bool = True`, `phase_dir: Path | None = None`.
- Produces: `render_job(spec: JobSpec, slurm: SlurmConfig) -> str`.
- Produces: `accelerate_command(*, module: str, gpus_per_node: int, args: str,
  mixed_precision: str = "bf16") -> str`.
- Produces: `SubmitEntry` frozen dataclass (`var: str`, `script: Path`,
  `depends_on: tuple[str, ...] = ()`) and `render_submit_script(entries: list[SubmitEntry]) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/slurm/test_render.py`:

```python
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from oplm.slurm.config import SlurmConfig
from oplm.slurm.render import JobSpec, SubmitEntry, render_job, render_submit_script
from tests.slurm.test_config import RAW

SLURM = SlurmConfig.from_mapping(RAW)


def _array_spec(tmp_path: Path) -> JobSpec:
    return JobSpec(
        name="oplm-coarse-170M",
        nodes=1,
        time_limit="168:00:00",
        command='python -m oplm.sweep.run --config "$RUN_DIR/run.yaml"',
        array_size=7,
        array_cells_file=tmp_path / "jobs" / "170M.cells",
        phase_dir=tmp_path,
    )


def test_array_header_includes_throttle(tmp_path: Path) -> None:
    text = render_job(_array_spec(tmp_path), SLURM)
    assert "#SBATCH --array=0-6%4" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks-per-node=1" in text
    assert "#SBATCH --gres=gpu:8" in text
    assert "#SBATCH --time=168:00:00" in text
    assert "#SBATCH --requeue" in text
    assert "#SBATCH --exclusive" in text


def test_array_logs_use_array_placeholders(tmp_path: Path) -> None:
    text = render_job(_array_spec(tmp_path), SLURM)
    assert "%x_%A_%a.out" in text
    assert "%x_%A_%a.err" in text


def test_single_job_logs_use_job_id() -> None:
    spec = JobSpec(
        name="oplm-scale-400M",
        nodes=8,
        time_limit="168:00:00",
        command="python -m oplm.train --config cfg.yaml",
    )
    text = render_job(spec, SLURM)
    assert "%x_%j.out" in text
    assert "--array" not in text
    assert "SLURM_ARRAY_TASK_ID" not in text


def test_pyxis_flags_are_on_srun_not_sbatch(tmp_path: Path) -> None:
    text = render_job(_array_spec(tmp_path), SLURM)
    for line in text.splitlines():
        if line.startswith("#SBATCH"):
            assert "--container-" not in line
    assert "--container-image=/mnt/data/containers/dl.sqsh" in text
    assert (
        "--container-mounts=/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},"
        "/mnt/data:/mnt/data,/tmp:/tmp" in text
    )
    assert "--no-container-mount-home" in text


def test_workdir_is_created_on_every_node(tmp_path: Path) -> None:
    """JOB_WORK_DIR is node-local /tmp and .env only creates it on the batch node."""
    text = render_job(_array_spec(tmp_path), SLURM)
    assert 'srun --nodes=$SLURM_NNODES --ntasks-per-node=1 mkdir -p "$JOB_WORK_DIR"' in text
    # The mkdir fanout must precede the training srun.
    assert text.index("mkdir -p") < text.index("--container-image=")


def test_rendezvous_variables(tmp_path: Path) -> None:
    text = render_job(_array_spec(tmp_path), SLURM)
    assert "export MASTER_ADDR=$(hostname --ip-address)" in text
    # SLURM_JOB_ID is unique per array task, so concurrent cells cannot collide.
    assert "export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))" in text
    assert "export OMP_NUM_THREADS=1" in text


def test_slurm_vars_expand_inside_the_container(tmp_path: Path) -> None:
    """The inner command is single-quoted so $SLURM_PROCID expands in the container shell."""
    spec = JobSpec(
        name="oplm-coarse-170M",
        nodes=1,
        time_limit="168:00:00",
        command=accelerate_command(module="oplm.sweep.run", gpus_per_node=8, args="--config x"),
        array_size=7,
        array_cells_file=tmp_path / "jobs" / "170M.cells",
        phase_dir=tmp_path,
    )
    text = render_job(spec, SLURM)
    inner = text.split("bash -c '", 1)[1]
    assert "--machine_rank $SLURM_PROCID" in inner
    assert "--num_machines $SLURM_NNODES" in inner
    # gpus_per_node is a render-time constant, so the arithmetic is already substituted.
    assert "--num_processes $((SLURM_NNODES * 8))" in inner
    assert "pip install oplm[train]" in inner


def test_array_index_maps_through_the_cells_file(tmp_path: Path) -> None:
    text = render_job(_array_spec(tmp_path), SLURM)
    assert "170M.cells" in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "export RUN_DIR" in text


def test_gres_omitted_for_cpu_only_jobs() -> None:
    spec = JobSpec(
        name="oplm-coarse-analyze",
        nodes=1,
        time_limit="01:00:00",
        command="oplm sweep analyze phase.json",
        gres=False,
    )
    text = render_job(spec, SLURM)
    assert "--gres" not in text


def test_submit_script_wires_afterany_across_arrays() -> None:
    text = render_submit_script(
        [
            SubmitEntry(var="A_400M", script=Path("jobs/400M.sbatch")),
            SubmitEntry(var="A_800M", script=Path("jobs/800M.sbatch")),
            SubmitEntry(var="A_1B", script=Path("jobs/1B.sbatch")),
            SubmitEntry(
                var="ANALYZE",
                script=Path("jobs/analyze.sbatch"),
                depends_on=("A_400M", "A_800M", "A_1B"),
            ),
        ]
    )
    assert "A_400M=$(sbatch --parsable jobs/400M.sbatch)" in text
    assert (
        "ANALYZE=$(sbatch --parsable --dependency=afterany:$A_400M:$A_800M:$A_1B "
        "jobs/analyze.sbatch)" in text
    )
    # afterok would wedge the analyze job the first time a cell diverges.
    assert "afterok" not in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_rendered_scripts_are_valid_bash(tmp_path: Path) -> None:
    for index, text in enumerate(
        (
            render_job(_array_spec(tmp_path), SLURM),
            render_submit_script([SubmitEntry(var="A", script=Path("jobs/a.sbatch"))]),
        )
    ):
        script = tmp_path / f"candidate{index}.sh"
        script.write_text(text)
        subprocess.run(["bash", "-n", str(script)], check=True)
```

Add `accelerate_command` to the import line at the top of the test file.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/slurm/test_render.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'oplm.slurm.render'`.

- [ ] **Step 3: Implement the renderer**

`src/oplm/slurm/render.py`:

```python
"""Render Slurm job scripts from a :class:`~oplm.slurm.config.SlurmConfig`.

One launcher form is used for every job. The multi-node ``srun`` form degrades correctly at
``SLURM_NNODES=1``, so there is no separate single-node path to maintain or test.

Quoting matters here. The inner training command is wrapped in ``bash -c '...'`` with *single*
quotes, so ``$SLURM_NNODES`` / ``$SLURM_PROCID`` / ``$MASTER_ADDR`` expand in the container's
shell (they reach it via ``--export=ALL``), not at render time. Anything that must be substituted
at render time is interpolated into the template directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from oplm.slurm.config import SlurmConfig

# The user's home mount is user-specific, so it is added at render time rather than configured.
_HOME_MOUNT = "/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER}"


@dataclass(frozen=True)
class JobSpec:
    """One Slurm job: a single run, or an array over homogeneous cells."""

    name: str
    nodes: int
    time_limit: str
    command: str
    array_size: int | None = None
    array_cells_file: Path | None = None
    gres: bool = True
    phase_dir: Path | None = None


@dataclass(frozen=True)
class SubmitEntry:
    """One ``sbatch`` invocation in a generated ``submit.sh``."""

    var: str
    script: Path
    depends_on: tuple[str, ...] = ()


def _header(spec: JobSpec, slurm: SlurmConfig) -> list[str]:
    suffix = "%A_%a" if spec.array_size is not None else "%j"
    lines = [
        "#!/bin/bash",
        "",
        "# --- job ---",
        f"#SBATCH --job-name={spec.name}",
        f"#SBATCH --partition={slurm.partition}",
    ]
    if slurm.account is not None:
        lines.append(f"#SBATCH --account={slurm.account}")
    lines += [
        "",
        "# --- nodes & resources ---",
        f"#SBATCH --nodes={spec.nodes}",
        "#SBATCH --ntasks-per-node=1",
    ]
    if spec.gres:
        lines.append(f"#SBATCH --gres=gpu:{slurm.gpus_per_node}")
    lines.append(f"#SBATCH --cpus-per-task={slurm.cpus_per_task}")
    lines.append(f"#SBATCH --mem={slurm.mem}")
    if slurm.exclusive:
        lines.append("#SBATCH --exclusive")
    lines += [
        f"#SBATCH --time={spec.time_limit}",
        "#SBATCH --requeue",
        "",
        "# --- logs ---",
        f"#SBATCH --output={slurm.log_dir}/%x_{suffix}.out",
        f"#SBATCH --error={slurm.log_dir}/%x_{suffix}.err",
    ]
    if spec.array_size is not None:
        lines += [
            "",
            "# --- array ---",
            f"#SBATCH --array=0-{spec.array_size - 1}%{slurm.max_concurrent}",
        ]
    return lines


def _array_lookup(spec: JobSpec) -> list[str]:
    if spec.array_cells_file is None or spec.phase_dir is None:
        return []
    return [
        "",
        "# Map this array index to its cell. One run id per line; index = line number - 1.",
        f'PHASE_DIR="{spec.phase_dir}"',
        f'CELLS_FILE="{spec.array_cells_file}"',
        'RUN_ID=$(awk "NR==$((SLURM_ARRAY_TASK_ID + 1))" "$CELLS_FILE")',
        'if [ -z "$RUN_ID" ]; then',
        '  echo "no cell for array index $SLURM_ARRAY_TASK_ID" >&2',
        "  exit 1",
        "fi",
        'export RUN_DIR="$PHASE_DIR/runs/$RUN_ID"',
    ]


def render_job(spec: JobSpec, slurm: SlurmConfig) -> str:
    """Render one sbatch script.

    Args:
        spec: What to run and at what size.
        slurm: Cluster settings.

    Returns:
        Complete script text, ending in a newline.
    """
    mounts = ",".join((_HOME_MOUNT, *slurm.container_mounts))
    lines = _header(spec, slurm)
    lines += [
        "",
        "set -euo pipefail",
        "",
        "# Creates JOB_WORK_DIR and exports object-storage / W&B credentials.",
        f"source {slurm.env_file}",
    ]
    lines += _array_lookup(spec)
    lines += [
        "",
        "# JOB_WORK_DIR is node-local /tmp; .env only created it on the batch node.",
        'srun --nodes=$SLURM_NNODES --ntasks-per-node=1 mkdir -p "$JOB_WORK_DIR"',
        "",
        "# Distributed rendezvous. The sbatch body runs on the rank-0 node, and SLURM_JOB_ID is",
        "# unique per array task, so concurrent cells cannot collide on a port.",
        "export MASTER_ADDR=$(hostname --ip-address)",
        "export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))",
        "export NCCL_DEBUG=INFO",
        "export OMP_NUM_THREADS=1",
        "",
        "srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \\",
        "  --export=ALL \\",
        f"  --container-image={slurm.container_image} \\",
        f"  --container-mounts={mounts} \\",
        '  --container-workdir="$JOB_WORK_DIR" \\',
        "  --no-container-mount-home \\",
        f"  bash -c '{slurm.install} && \\",
        f"    {spec.command}'",
        "",
    ]
    return "\n".join(lines)


def accelerate_command(
    *, module: str, gpus_per_node: int, args: str, mixed_precision: str = "bf16"
) -> str:
    """Build the inner ``accelerate launch`` command for a container shell.

    ``$SLURM_NNODES`` and ``$SLURM_PROCID`` are left unexpanded on purpose: the caller embeds this
    inside single quotes so the container's shell resolves them.
    """
    return (
        "accelerate launch \\\n"
        "    --multi_gpu \\\n"
        f"    --mixed_precision {mixed_precision} \\\n"
        "    --num_machines $SLURM_NNODES \\\n"
        f"    --num_processes $((SLURM_NNODES * {gpus_per_node})) \\\n"
        "    --machine_rank $SLURM_PROCID \\\n"
        '    --main_process_ip "$MASTER_ADDR" \\\n'
        "    --main_process_port $MASTER_PORT \\\n"
        f"    -m {module} \\\n"
        f"    {args}"
    )


def render_submit_script(entries: list[SubmitEntry]) -> str:
    """Render a ``submit.sh`` that submits every job and wires dependencies.

    Dependencies use ``afterany``, not ``afterok``: a divergent cell is expected data, and ranking
    already treats a missing or non-finite result as ineligible. Under ``afterok`` the first
    blown-up cell would leave the analyze job in ``DependencyNeverSatisfied`` forever.
    """
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        "# Run from the phase directory regardless of the caller's cwd.",
        'cd "$(dirname "$0")/.."',
        "",
    ]
    for entry in entries:
        if entry.depends_on:
            deps = ":".join(f"${var}" for var in entry.depends_on)
            call = f"sbatch --parsable --dependency=afterany:{deps} {entry.script}"
        else:
            call = f"sbatch --parsable {entry.script}"
        lines.append(f"{entry.var}=$({call})")
        lines.append(f'echo "submitted {entry.script}: ${entry.var}"')
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/slurm/test_render.py -q`
Expected: PASS, including the `bash -n` syntax check.

- [ ] **Step 5: Commit**

```bash
ruff check src/ && ruff format --check src/ && ty check src/
git add -A src/oplm/slurm tests/slurm
git commit -m "feat: render single and array slurm job scripts"
```

---

## Phase 5 — `oplm.slurm` submission and status

### Task 8: Submit via `sbatch --parsable` and report status

**Files:**
- Create: `src/oplm/slurm/submit.py`
- Test: `tests/slurm/test_submit.py`

**Interfaces:**
- Produces: `submit_job(script: Path, *, depends_on: Sequence[str] = ()) -> str` returning the job
  id.
- Produces: `submit_all(entries: Sequence[SubmitEntry], *, base_dir: Path) -> dict[str, str]`
  mapping each entry's `var` to its job id.
- Produces: `running_job_ids(job_ids: Sequence[str]) -> set[str]`.

- [ ] **Step 1: Write the failing tests**

`tests/slurm/test_submit.py`:

```python
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from oplm.slurm.render import SubmitEntry
from oplm.slurm.submit import running_job_ids, submit_all, submit_job


def _install_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, body: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / name
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


@pytest.fixture
def fake_sbatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a recording `sbatch` stub on PATH; no cluster required."""
    log = tmp_path / "sbatch.log"
    _install_stub(
        tmp_path,
        monkeypatch,
        "sbatch",
        "#!/bin/bash\n"
        f'echo "$@" >> "{log}"\n'
        f'count=$(wc -l < "{log}")\n'
        "echo $((812344 + count))\n",
    )
    return log


def test_submit_job_returns_parsed_id(fake_sbatch: Path, tmp_path: Path) -> None:
    script = tmp_path / "job.sbatch"
    script.write_text("#!/bin/bash\n")
    assert submit_job(script) == "812345"
    assert fake_sbatch.read_text().strip() == f"--parsable {script}"


def test_submit_job_builds_afterany_dependency(fake_sbatch: Path, tmp_path: Path) -> None:
    script = tmp_path / "analyze.sbatch"
    script.write_text("#!/bin/bash\n")
    submit_job(script, depends_on=["100", "200", "300"])
    assert "--dependency=afterany:100:200:300" in fake_sbatch.read_text()


def test_submit_all_threads_ids_into_dependencies(fake_sbatch: Path, tmp_path: Path) -> None:
    for name in ("400M.sbatch", "800M.sbatch", "analyze.sbatch"):
        (tmp_path / name).write_text("#!/bin/bash\n")
    ids = submit_all(
        [
            SubmitEntry(var="A_400M", script=Path("400M.sbatch")),
            SubmitEntry(var="A_800M", script=Path("800M.sbatch")),
            SubmitEntry(
                var="ANALYZE", script=Path("analyze.sbatch"), depends_on=("A_400M", "A_800M")
            ),
        ],
        base_dir=tmp_path,
    )
    assert ids == {"A_400M": "812345", "A_800M": "812346", "ANALYZE": "812347"}
    assert "--dependency=afterany:812345:812346" in fake_sbatch.read_text()


def test_submit_job_raises_on_sbatch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_stub(
        tmp_path, monkeypatch, "sbatch", '#!/bin/bash\necho "queue full" >&2\nexit 1\n'
    )
    job = tmp_path / "job.sbatch"
    job.write_text("#!/bin/bash\n")
    with pytest.raises(RuntimeError, match="queue full"):
        submit_job(job)


def test_running_job_ids_parses_squeue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub(
        tmp_path, monkeypatch, "squeue", "#!/bin/bash\nprintf '812345_3\\n812347\\n'\n"
    )
    assert running_job_ids(["812345", "812346", "812347"]) == {"812345", "812347"}


def test_running_job_ids_empty_when_squeue_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    assert running_job_ids(["812345"]) == set()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/slurm/test_submit.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'oplm.slurm.submit'`.

- [ ] **Step 3: Implement**

`src/oplm/slurm/submit.py`:

```python
"""Submit generated job scripts and query their state."""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from oplm.slurm.render import SubmitEntry

logger = logging.getLogger(__name__)


def submit_job(script: Path, *, depends_on: Sequence[str] = ()) -> str:
    """Submit one script with ``sbatch --parsable`` and return its job id.

    Dependencies use ``afterany`` so a diverged cell cannot wedge a downstream job.

    Args:
        script: Path to the sbatch script.
        depends_on: Job ids this submission waits on.

    Returns:
        The Slurm job id.

    Raises:
        RuntimeError: If ``sbatch`` exits non-zero.
    """
    argv = ["sbatch", "--parsable"]
    if depends_on:
        argv.append(f"--dependency=afterany:{':'.join(depends_on)}")
    argv.append(str(script))
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed for {script} (exit {result.returncode}): {result.stderr.strip()}"
        )
    # --parsable prints "<jobid>" or "<jobid>;<cluster>".
    return result.stdout.strip().split(";", 1)[0]


def submit_all(entries: Sequence[SubmitEntry], *, base_dir: Path) -> dict[str, str]:
    """Submit every entry in order, threading earlier job ids into later dependencies.

    Args:
        entries: Jobs to submit, in dependency order.
        base_dir: Directory the entries' script paths are relative to.

    Returns:
        Mapping of each entry's ``var`` to its job id.
    """
    ids: dict[str, str] = {}
    for entry in entries:
        depends = [ids[var] for var in entry.depends_on]
        job_id = submit_job(base_dir / entry.script, depends_on=depends)
        ids[entry.var] = job_id
        logger.info("submitted %s as %s", entry.script, job_id)
    return ids


def running_job_ids(job_ids: Sequence[str]) -> set[str]:
    """Return the subset of ``job_ids`` still known to the scheduler.

    Returns an empty set when ``squeue`` is unavailable (e.g. off-cluster), so callers degrade to
    filesystem-only status rather than failing.
    """
    if not job_ids or shutil.which("squeue") is None:
        return set()
    result = subprocess.run(
        ["squeue", "--noheader", "--format=%i", "--jobs", ",".join(job_ids)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return set()
    # Array elements report as "<arrayjobid>_<index>"; match on the base id.
    return {line.strip().split("_", 1)[0] for line in result.stdout.splitlines() if line.strip()}
```

Add `submit_job`, `submit_all`, `running_job_ids`, `JobSpec`, `SubmitEntry`, `render_job`,
`render_submit_script`, `accelerate_command` to `src/oplm/slurm/__init__.py`.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/slurm/test_submit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff check src/ && ruff format --check src/ && ty check src/
git add -A src/oplm/slurm tests/slurm
git commit -m "feat: submit slurm jobs and query scheduler state"
```

---

## Phase 6 — The general layer stands alone

### Task 9: `oplm slurm` CLI, `configs/scaling.yaml`, and the layering guard

**Files:**
- Create: `src/oplm/slurm/cli.py`, `configs/scaling.yaml`
- Modify: `src/oplm/cli.py`
- Test: `tests/slurm/test_cli.py`, `tests/slurm/test_layering.py`

**Interfaces:**
- Produces: `oplm slurm generate --config <cfg> [--preset P] [--nodes N] --out <dir>` writing
  `<dir>/<name>.sbatch`; `oplm slurm submit <dir>`; `oplm slurm status <dir>`.

- [ ] **Step 1: Write the failing tests**

`tests/slurm/test_layering.py`:

```python
from __future__ import annotations

import ast
from pathlib import Path

import oplm.slurm


def test_slurm_layer_does_not_import_sweep() -> None:
    """oplm.slurm is general-purpose; oplm.sweep depends on it, never the reverse."""
    package = Path(oplm.slurm.__file__).parent
    offenders = []
    for module in sorted(package.glob("*.py")):
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.startswith("oplm.sweep") for name in names):
                offenders.append(f"{module.name}:{node.lineno}")
    assert offenders == [], f"oplm.slurm must not import oplm.sweep: {offenders}"
```

`tests/slurm/test_cli.py`:

```python
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from oplm.cli import app

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_slurm_subcommand_is_registered() -> None:
    result = runner.invoke(app, ["slurm", "--help"])
    assert result.exit_code == 0
    for command in ("generate", "submit", "status"):
        assert command in result.stdout


def test_generate_from_the_committed_scaling_config(tmp_path: Path) -> None:
    """The general layer works from a plain training config, with no sweep artifacts."""
    result = runner.invoke(
        app,
        [
            "slurm",
            "generate",
            "--config",
            str(REPO_ROOT / "configs" / "scaling.yaml"),
            "--preset",
            "400M",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    script = tmp_path / "oplm-400M.sbatch"
    assert script.exists()
    text = script.read_text()
    assert "#SBATCH --nodes=8" in text
    assert "--preset 400M" in text
    assert "sweep" not in text
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/slurm/test_cli.py tests/slurm/test_layering.py -q`
Expected: FAIL — no `slurm` subcommand and no `configs/scaling.yaml`.

- [ ] **Step 3: Write `configs/scaling.yaml`**

A complete, runnable production config. Dataset paths are the CoreWeave locations from the
production job scripts. `container_mounts` omits the user's home mount because `render.py` adds
`/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER}` automatically.

```yaml
# Production scaling runs. Usable on its own:
#   oplm slurm generate --config configs/scaling.yaml --preset 400M --out jobs/400M
# `oplm sweep scale` merges the confirmed muP winner into this file before generating.

model:
  max_position_embeddings: 512
  gradient_checkpointing: true
  gradient_checkpointing_mode: full
  norm_strategy: sandwich
  canon_enabled: true
  canon_positions: [A, B, C, D]
  residual_gate: channel
  mup_enable: true
  mup_base_width: 768

train:
  optimizer: muon
  muon_adjust_lr_fn: original
  weight_decay: 0.01
  mup_depth_reference_layers: 24
  scheduler: wsd_linear
  max_steps: 100000
  stable_steps: 95000
  warmup_steps: 5000
  save_every: 10000
  save_total_limit: 3
  eval_every: {steps: 5000}
  compile: true
  wandb_project: oplm-scaling
  output_dir: /mnt/home/briney/projects/oplm/scaling

data:
  train:
    uniref70:
      path: /mnt/data/datasets/plm-training-data/uniref70_omg70/uniref70/train
      fraction: 0.35
    omg70:
      path: /mnt/data/datasets/plm-training-data/uniref70_omg70/omg70/train
      fraction: 0.45
    deepclust30:
      path: /mnt/data/datasets/plm-training-data/deepclust30-bigg2/train
      fraction: 0.20
  eval:
    uniref70:
      path: /mnt/data/datasets/plm-training-data/uniref70_omg70/uniref70/eval.parquet
      type: sequence
    omg70:
      path: /mnt/data/datasets/plm-training-data/uniref70_omg70/omg70/eval.parquet
      type: sequence
    deepclust30:
      path: /mnt/data/datasets/plm-training-data/deepclust30-bigg2/eval.parquet
      type: sequence
    casp14:
      path: /mnt/data/datasets/plm-training-data/casp/casp14
      type: structure
    proteingym_clinical:
      path: /mnt/data/datasets/plm-training-data/ProteinGym/eval/clinical_substitutions
      type: proteingym_clinical
      scoring: wt_marginals
      top_k_fraction: 0.1
    proteingym_dms:
      path: /mnt/data/datasets/plm-training-data/ProteinGym/eval/dms_substitutions
      type: proteingym
      scoring: wt_marginals
      top_k_fraction: 0.1

slurm:
  partition: hpc-mid
  time_limit:
    default: "168:00:00"
  cpus_per_task: 128
  gpus_per_node: 8
  exclusive: true
  mem: "0"
  log_dir: /mnt/home/briney/logs
  env_file: /mnt/home/briney/.env
  container_image: /mnt/data/containers/deeplearning_v2026-05-26.sqsh
  container_mounts:
    - /mnt/data:/mnt/data
    - /tmp:/tmp
  install: pip install oplm[train]
  max_concurrent: 4
  nodes: {170M: 4, 400M: 8, 800M: 16, 1B: 16}
  max_batch_size: {170M: 256, 400M: 256, 800M: 256, 1B: 128}
```

- [ ] **Step 4: Implement the CLI**

`src/oplm/slurm/cli.py`:

```python
"""`oplm slurm` command surface: turn a training config into job scripts."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003  # Typer resolves annotations at runtime.
from typing import Annotated

import typer
from rich.console import Console

from oplm.slurm.config import load_slurm_config
from oplm.slurm.render import JobSpec, SubmitEntry, accelerate_command, render_job
from oplm.slurm.submit import running_job_ids, submit_all

app = typer.Typer(name="slurm", help="Generate and submit Slurm jobs", add_completion=False)
console = Console()

_MANIFEST = "jobs.json"


@app.command()
def generate(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)],
    preset: Annotated[str | None, typer.Option("--preset")] = None,
    nodes: Annotated[int | None, typer.Option("--nodes")] = None,
    name: Annotated[str | None, typer.Option("--name")] = None,
    time_limit: Annotated[str | None, typer.Option("--time-limit")] = None,
) -> None:
    """Write one sbatch script for a training config."""
    slurm = load_slurm_config(config)
    job_name = name or f"oplm-{preset or 'run'}"
    resolved_nodes = nodes if nodes is not None else slurm.nodes.resolve(phase=None, preset=preset)
    resolved_time = time_limit or slurm.time_limit.resolve(phase=None, preset=preset)
    args = f"--config {config}"
    if preset is not None:
        args += f" --preset {preset}"
    spec = JobSpec(
        name=job_name,
        nodes=resolved_nodes,
        time_limit=resolved_time,
        command=accelerate_command(
            module="oplm.train", gpus_per_node=slurm.gpus_per_node, args=args
        ),
    )
    out.mkdir(parents=True, exist_ok=True)
    script = out / f"{job_name}.sbatch"
    script.write_text(render_job(spec, slurm))
    (out / _MANIFEST).write_text(
        json.dumps({"scripts": [script.name], "job_ids": {}}, indent=2) + "\n"
    )
    console.print(f"wrote {script} ({resolved_nodes} nodes, {resolved_time})")


@app.command()
def submit(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Submit every script a previous `generate` wrote into DIRECTORY."""
    manifest_path = directory / _MANIFEST
    manifest = json.loads(manifest_path.read_text())
    entries = [
        SubmitEntry(var=f"JOB_{index}", script=Path(script))
        for index, script in enumerate(manifest["scripts"])
    ]
    ids = submit_all(entries, base_dir=directory)
    manifest["job_ids"] = ids
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for var, job_id in ids.items():
        console.print(f"{var}: {job_id}")


@app.command()
def status(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Report which submitted jobs are still known to the scheduler."""
    manifest = json.loads((directory / _MANIFEST).read_text())
    ids: dict[str, str] = manifest.get("job_ids", {})
    if not ids:
        console.print("no jobs submitted yet")
        raise typer.Exit()
    live = running_job_ids(list(ids.values()))
    for var, job_id in ids.items():
        state = "active" if job_id in live else "finished or unknown"
        console.print(f"{var} ({job_id}): {state}")
```

In `src/oplm/cli.py`, alongside the sweep mount:

```python
app.add_typer(slurm_app, name="slurm", help="Generate and submit Slurm jobs")
```

with `from oplm.slurm.cli import app as slurm_app` at the top of the file.

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/slurm/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
ruff check src/ && ruff format --check src/ && ty check src/ && python -m pytest tests/slurm -q
git add -A src/oplm configs/scaling.yaml tests/slurm
git commit -m "feat: add oplm slurm CLI and a standalone scaling config"
```

---

## Phase 7 — Sweep phases emit Slurm jobs

### Task 10: Pin per-cell overrides from the batch plan

**Files:**
- Modify: `src/oplm/sweep/phases.py` (`_write_run_config`, `_generate_phase`)
- Test: `tests/sweep/test_phases.py`

**Interfaces:**
- Consumes: `resolve_batch_plan`, `BatchPlan`, `SlurmConfig`, `load_slurm_config` from
  `oplm.slurm.config`.
- Produces: `_write_run_config(base_config, run_dir, params, *, plan: BatchPlan,
  diagnostics: bool = False) -> Path` — the `num_processes` parameter is replaced by `plan`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/sweep/test_phases.py`:

```python
def test_cells_pin_full_gradient_checkpointing(tmp_path: Path) -> None:
    run_yaml = _generate_one_cell(tmp_path)
    cfg = load_config(["--config", str(run_yaml)])
    assert cfg.model.gradient_checkpointing is True
    assert cfg.model.gradient_checkpointing_mode == "full"


def test_cells_pin_batch_from_the_plan(tmp_path: Path) -> None:
    """Per-device batch comes from the node plan, not from the base config."""
    run_yaml = _generate_one_cell(tmp_path)
    cfg = load_config(["--config", str(run_yaml)])
    assert cfg.train.batch_size == 256
    assert cfg.train.gradient_accumulation_steps == 1


def test_cells_checkpoint_often_enough_to_resume(tmp_path: Path) -> None:
    """save_every defaults to 10_000, which would checkpoint a 10k cell only at the end."""
    run_yaml = _generate_one_cell(tmp_path, max_steps=10_000)
    cfg = load_config(["--config", str(run_yaml)])
    assert cfg.train.save_every == 1250
    assert cfg.train.save_total_limit == 1
```

Write `_generate_one_cell` as a helper in the same file that builds a minimal base config carrying
the `slurm:` block from `tests/slurm/test_config.py::RAW`, invokes the `coarse` generator with a
single LR at `--global-examples 2048`, and returns the resulting `runs/<id>/run.yaml`. Reuse the
file's existing base-config fixture for the μP requirements (`optimizer: muon`,
`muon_adjust_lr_fn: original`, `weight_decay: 0.01`, exactly one eval task).

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/sweep/test_phases.py -q -k "pin or checkpoint_often"`
Expected: FAIL — checkpointing is not pinned and `save_every` is the default.

- [ ] **Step 3: Rewrite `_write_run_config`**

Replace the `grad_accum` computation and the `overrides` list in `_write_run_config`:

```python
def _write_run_config(
    base_config: Path,
    run_dir: Path,
    params: Params,
    *,
    plan: BatchPlan,
    diagnostics: bool = False,
) -> Path:
    max_steps = int(params["max_steps"])
    warmup_steps = int(params["warmup_steps"])
    if warmup_steps < 0 or warmup_steps >= max_steps:
        raise ValueError(
            "warmup_steps must satisfy 0 <= warmup_steps < max_steps, "
            f"got {warmup_steps} and {max_steps}"
        )

    preset = str(params["preset"])
    preset_model = get_preset_config(preset).model
    base = load_config(["--config", str(base_config)])
    if base.train.optimizer != "muon":
        raise ValueError(f"μP sweep requires train.optimizer=muon, got {base.train.optimizer}")
    if base.train.muon_adjust_lr_fn != "original":
        raise ValueError(
            "μP sweep requires train.muon_adjust_lr_fn=original, "
            f"got {base.train.muon_adjust_lr_fn}"
        )
    if base.train.weight_decay != 0.01:
        raise ValueError(
            f"μP sweep requires train.weight_decay=0.01, got {base.train.weight_decay}"
        )
    run_name = run_dir.name
    # Checkpoint eight times per cell so a requeue loses at most ~12% of the run, and keep only
    # the newest: these checkpoints exist solely to make requeue cheap.
    save_every = max(1, max_steps // 8)
    overrides = [
        f"model.hidden_size={preset_model.hidden_size}",
        f"model.num_hidden_layers={preset_model.num_hidden_layers}",
        f"model.num_attention_heads={preset_model.num_attention_heads}",
        "model.head_dim=64",
        "model.mup_enable=true",
        "model.mup_base_width=768",
        f"model.mup_output_mult={params['output_mult']}",
        # Full checkpointing is what makes the derived per-device batches fit at 400M+. Both keys
        # are pinned so a change to packaged defaults cannot alter the memory profile mid-sweep.
        "model.gradient_checkpointing=true",
        "model.gradient_checkpointing_mode=full",
        f"train.lr={params['lr']}",
        f"train.mup_depth_lr_exponent={params['depth_exponent']}",
        "train.mup_depth_reference_layers=24",
        f"train.seed={params['seed']}",
        f"train.batch_size={plan.per_device_batch}",
        f"train.gradient_accumulation_steps={plan.gradient_accumulation_steps}",
        f"train.max_steps={params['max_steps']}",
        "train.max_epochs=null",
        f"train.warmup_steps={params['warmup_steps']}",
        f"train.stable_steps={max_steps - warmup_steps}",
        "train.scheduler=wsd_linear",
        f"train.save_every={save_every}",
        "train.save_total_limit=1",
        f"train.output_dir={run_dir / 'output'}",
        f"train.wandb_run_name={run_name}",
        # Explicitly pin diagnostics so the flag overrides whatever the base config
        # carries (off by default keeps the sweep robust; see docs/LR_SWEEP.md).
        f"train.stability_diagnostics={str(diagnostics).lower()}",
    ]
    cfg = load_config(["--config", str(base_config), *overrides])
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.yaml"
    path.write_text(serialize_config(cfg))
    return path
```

Add to the imports in `phases.py`:

```python
from oplm.slurm.config import BatchPlan, SlurmConfig, load_slurm_config, resolve_batch_plan
```

In `_generate_phase`, load the slurm block once and compute the plan per cell before writing its
config:

```python
    slurm = load_slurm_config(base_config)
    ...
        preset = str(params["preset"])
        nodes = slurm.nodes.resolve(phase=name, preset=preset)
        plan = resolve_batch_plan(
            global_examples=int(params["global_examples"]),
            nodes=nodes,
            gpus_per_node=slurm.gpus_per_node,
            max_batch_size=slurm.max_batch_size[preset],
        )
```

Pass `plan=plan` to `_write_run_config`, and record `nodes`, `plan.per_device_batch`, and
`plan.gradient_accumulation_steps` into each `RunSpec.params`.

`gradient_accumulation_steps` in `oplm/sweep/common.py` is now unused by `phases.py`. Leave the
function in place — `test_common.py` covers it and it documents the divisibility contract — but do
not call it from the new path.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/sweep/test_phases.py -q`
Expected: PASS. Existing tests that asserted on `num_processes`-derived accumulation need updating
to the plan-derived values.

- [ ] **Step 5: Commit**

```bash
ruff check src/ && ruff format --check src/ && ty check src/ && python -m pytest tests/sweep -q
git add -A src/oplm/sweep tests/sweep
git commit -m "feat: pin per-cell batch and checkpointing from the node plan"
```

### Task 11: Emit `jobs/` per phase

**Files:**
- Modify: `src/oplm/sweep/phases.py` (`_generate_phase`, every phase command),
  `src/oplm/sweep/common.py` (`PhaseManifest`)
- Create: `tests/sweep/conftest.py`, `tests/sweep/test_jobs.py`

**Interfaces:**
- Produces: `_write_jobs(out: Path, phase: str, runs: list[RunSpec], slurm: SlurmConfig,
  phase_json: Path) -> list[SubmitEntry]`.
- Produces: a `--submit/--no-submit` flag on every phase command except `scale`, defaulting to
  `--no-submit`.
- Produces: `PhaseManifest` gains `oplm_version: str | None`, `generated_at: str | None`,
  `job_ids: dict[str, str] | None`, all defaulting to `None`.

- [ ] **Step 1: Write the failing tests**

`tests/sweep/test_jobs.py`:

```python
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_single_preset_phase_emits_one_array(coarse_phase: Path) -> None:
    jobs = coarse_phase.parent / "jobs"
    assert (jobs / "170M.sbatch").exists()
    assert (jobs / "170M.cells").exists()
    assert (jobs / "analyze.sbatch").exists()
    assert (jobs / "submit.sh").exists()
    cells = (jobs / "170M.cells").read_text().splitlines()
    assert len(cells) == 7
    assert "#SBATCH --array=0-6%4" in (jobs / "170M.sbatch").read_text()


def test_transfer_emits_one_array_per_preset(transfer_phase: Path) -> None:
    """Slurm arrays need homogeneous resources, and node count varies by preset."""
    jobs = transfer_phase.parent / "jobs"
    for preset, nodes in (("400M", 4), ("800M", 8), ("1B", 8)):
        text = (jobs / f"{preset}.sbatch").read_text()
        assert f"#SBATCH --nodes={nodes}" in text
    submit = (jobs / "submit.sh").read_text()
    assert "--dependency=afterany:$A_400M:$A_800M:$A_1B" in submit


def test_analyze_job_is_cpu_only(coarse_phase: Path) -> None:
    text = (coarse_phase.parent / "jobs" / "analyze.sbatch").read_text()
    assert "--gres" not in text
    assert "oplm sweep analyze" in text
    assert "#SBATCH --time=01:00:00" in text


def test_cells_file_order_matches_manifest(coarse_phase: Path) -> None:
    manifest = json.loads(coarse_phase.read_text())
    cells = (coarse_phase.parent / "jobs" / "170M.cells").read_text().split()
    assert cells == [run["id"] for run in manifest["runs"]]


def test_manifest_records_provenance(coarse_phase: Path) -> None:
    from importlib.metadata import version

    manifest = json.loads(coarse_phase.read_text())
    assert manifest["oplm_version"] == version("oplm")
    assert manifest["generated_at"]


def test_generated_scripts_are_valid_bash(coarse_phase: Path) -> None:
    for script in sorted((coarse_phase.parent / "jobs").glob("*.sbatch")):
        subprocess.run(["bash", "-n", str(script)], check=True)
    subprocess.run(["bash", "-n", str(coarse_phase.parent / "jobs" / "submit.sh")], check=True)
```

Create `tests/sweep/conftest.py` with `coarse_phase`, `transfer_phase`, and `confirm_phase`
fixtures. Each writes a temp base config carrying the `slurm:` block from
`tests/slurm/test_config.py::RAW` plus the μP requirements, invokes the matching phase generator,
and returns the resulting `phase.json` path. `confirm_phase` additionally writes `selected` with
one winner dict holding `lr`, `output_mult`, `depth_exponent`, and `batch_mult`.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/sweep/test_jobs.py -q`
Expected: FAIL — no `jobs/` directory is written.

- [ ] **Step 3: Extend `PhaseManifest`**

In `src/oplm/sweep/common.py`:

```python
@dataclass
class PhaseManifest:
    version: int
    phase: str
    metric: str
    source: str | None
    runs: list[RunSpec]
    ranking: list[dict[str, object]]
    selected: list[Params]
    oplm_version: str | None = None
    generated_at: str | None = None
    job_ids: dict[str, str] | None = None
```

Update `load_phase` to read the three new fields with `raw.get(...)` defaults so manifests written
before this change still load:

```python
def load_phase(path: Path) -> PhaseManifest:
    raw = json.loads(path.read_text())
    return PhaseManifest(
        version=int(raw["version"]),
        phase=str(raw["phase"]),
        metric=str(raw["metric"]),
        source=raw["source"],
        runs=[RunSpec(**run) for run in raw["runs"]],
        ranking=list(raw["ranking"]),
        selected=list(raw["selected"]),
        oplm_version=raw.get("oplm_version"),
        generated_at=raw.get("generated_at"),
        job_ids=raw.get("job_ids"),
    )
```

- [ ] **Step 4: Implement `_write_jobs`**

Add to `src/oplm/sweep/phases.py`:

```python
def _preset_sort_key(preset: str) -> tuple[float, str]:
    """Order presets by size so `400M` precedes `1B`."""
    scale = {"M": 1e6, "B": 1e9}.get(preset[-1].upper(), 1.0)
    try:
        return (float(preset[:-1]) * scale, preset)
    except ValueError:
        return (float("inf"), preset)


def _write_jobs(
    out: Path,
    phase: str,
    runs: list[RunSpec],
    slurm: SlurmConfig,
    phase_json: Path,
) -> list[SubmitEntry]:
    """Write one array job per preset plus a dependent analyze job.

    Slurm job arrays require homogeneous resources and node count varies by preset, so a phase
    spanning several presets (``transfer``) emits one array each, with a single analyze job
    depending on all of them.

    Returns:
        The submission entries, in dependency order.
    """
    jobs = out / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    presets = sorted({str(run.params["preset"]) for run in runs}, key=_preset_sort_key)
    entries: list[SubmitEntry] = []
    for preset in presets:
        group = [run for run in runs if str(run.params["preset"]) == preset]
        cells_file = jobs / f"{preset}.cells"
        cells_file.write_text("\n".join(run.id for run in group) + "\n")
        spec = JobSpec(
            name=f"oplm-{phase}-{preset}",
            nodes=slurm.nodes.resolve(phase=phase, preset=preset),
            time_limit=slurm.time_limit.resolve(phase=phase, preset=preset),
            command=accelerate_command(
                module="oplm.sweep.run",
                gpus_per_node=slurm.gpus_per_node,
                args='--config "$RUN_DIR/run.yaml" --result "$RUN_DIR/result.json"',
            ),
            array_size=len(group),
            array_cells_file=cells_file,
            phase_dir=out,
        )
        script = jobs / f"{preset}.sbatch"
        script.write_text(render_job(spec, slurm))
        entries.append(SubmitEntry(var=f"A_{preset}", script=Path("jobs") / script.name))

    analyze_spec = JobSpec(
        name=f"oplm-{phase}-analyze",
        nodes=1,
        time_limit=slurm.time_limit.resolve(phase="analyze", preset=None),
        command=f"oplm sweep analyze {phase_json}",
        gres=False,
    )
    analyze_script = jobs / "analyze.sbatch"
    analyze_script.write_text(render_job(analyze_spec, slurm))
    entries.append(
        SubmitEntry(
            var="ANALYZE",
            script=Path("jobs") / analyze_script.name,
            depends_on=tuple(entry.var for entry in entries),
        )
    )

    submit_path = jobs / "submit.sh"
    submit_path.write_text(render_submit_script(entries))
    submit_path.chmod(0o755)
    return entries
```

Add to the imports in `phases.py`:

```python
from oplm.slurm.render import (
    JobSpec,
    SubmitEntry,
    accelerate_command,
    render_job,
    render_submit_script,
)
from oplm.slurm.submit import submit_all
```

- [ ] **Step 5: Wire it into `_generate_phase` and the commands**

In `_generate_phase`, set `oplm_version` and `generated_at` on the manifest before `write_phase`:

```python
    from datetime import UTC, datetime
    from importlib.metadata import version

    manifest = PhaseManifest(
        version=1,
        phase=name,
        metric=_resolve_metric(base_config, metric),
        source=str(relative_path(source, out)) if source is not None else None,
        runs=runs,
        ranking=[],
        selected=[],
        oplm_version=version("oplm"),
        generated_at=datetime.now(tz=UTC).isoformat(),
    )
```

After `write_phase`, call `_write_jobs(out, name, runs, slurm, phase_path)`. Add a
`submit: bool = False` parameter to `_generate_phase`; when true, call
`submit_all(entries, base_dir=out)`, store the returned ids on `manifest.job_ids`, and
`write_phase` again.

Add to every phase command signature except `scale`:

```python
    submit: Annotated[bool, typer.Option("--submit/--no-submit")] = False,
```

Thread it into `_generate_phase`. Keep `--local` working unchanged: when `--local` is passed, run
the commands sequentially and analyze as today. Passing `--local` and `--submit` together is an
error:

```python
        if local and submit:
            raise ValueError("--local and --submit are mutually exclusive")
```

- [ ] **Step 6: Run to verify pass**

Run: `python -m pytest tests/sweep -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
ruff check src/ && ruff format --check src/ && ty check src/ && python -m pytest tests -q -m "not slow"
git add -A src/oplm tests
git commit -m "feat: emit per-preset slurm arrays and a dependent analyze job"
```

### Task 12: `oplm sweep status`

**Files:**
- Modify: `src/oplm/sweep/phases.py`
- Test: `tests/sweep/test_status.py`

**Interfaces:**
- Produces: `oplm sweep status <phase.json>` printing per-cell state and a resubmit line.

- [ ] **Step 1: Write the failing test**

`tests/sweep/test_status.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from oplm.cli import app

runner = CliRunner()


def test_status_reports_cell_states_and_resubmit_line(coarse_phase: Path) -> None:
    manifest = json.loads(coarse_phase.read_text())
    runs = manifest["runs"]
    # One complete, one non-finite, the rest missing.
    for index, value in ((0, 3.0), (1, float("nan"))):
        result_path = coarse_phase.parent / runs[index]["result"]
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({"eval": {"eval/heldout/loss": value}}))

    result = runner.invoke(app, ["sweep", "status", str(coarse_phase)])
    assert result.exit_code == 0
    assert "complete" in result.stdout
    assert "non-finite" in result.stdout
    assert "missing" in result.stdout
    # Indices 1..6 need rerunning: the non-finite one and the five with no result.
    assert "--array=1,2,3,4,5,6" in result.stdout
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/sweep/test_status.py -q`
Expected: FAIL — no `status` command.

- [ ] **Step 3: Implement**

Add to `src/oplm/sweep/phases.py`:

```python
@app.command()
def status(
    phase_json: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Report per-cell state for one phase and print the resubmit line for incomplete cells."""
    from rich.console import Console
    from rich.table import Table

    from oplm.slurm.submit import running_job_ids

    console = Console()
    path = phase_json.resolve()
    phase = load_phase(path)
    live = running_job_ids(list((phase.job_ids or {}).values()))

    table = Table(title=f"{phase.phase} ({phase.metric})")
    table.add_column("idx", justify="right")
    table.add_column("cell")
    table.add_column("state")
    table.add_column(phase.metric, justify="right")

    seen: dict[str, int] = {}
    incomplete: dict[str, list[int]] = {}
    for run in phase.runs:
        preset = str(run.params["preset"])
        index = seen.get(preset, 0)
        seen[preset] = index + 1
        value = result_metric(path.parent, run, phase.metric)
        if value is not None:
            state, shown = "complete", f"{value:.4f}"
        else:
            if (path.parent / run.result).exists():
                state = "non-finite"
            elif live:
                state = "running"
            else:
                state = "missing"
            shown = "-"
            incomplete.setdefault(preset, []).append(index)
        table.add_row(str(index), run.id, state, shown)
    console.print(table)

    for preset, indices in sorted(incomplete.items()):
        listed = ",".join(str(index) for index in indices)
        console.print(f"resubmit {preset}: sbatch --array={listed} jobs/{preset}.sbatch")
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/sweep/test_status.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff check src/ && ruff format --check src/ && ty check src/
git add -A src/oplm/sweep tests/sweep
git commit -m "feat: add oplm sweep status"
```

---

## Phase 8 — Resilience

### Task 13: Auto-resume and version recording

**Files:**
- Modify: `src/oplm/sweep/run.py`, `src/oplm/training/mup.py`
- Test: `tests/sweep/test_run.py`

**Interfaces:**
- Produces: `latest_checkpoint(output_dir: Path) -> Path | None` in `oplm/sweep/run.py`.
- Produces: `result.json` gains an `oplm_version` key.

- [ ] **Step 1: Write the failing tests**

Add to `tests/sweep/test_run.py`:

```python
from oplm.sweep.run import latest_checkpoint


def test_latest_checkpoint_none_when_empty(tmp_path: Path) -> None:
    assert latest_checkpoint(tmp_path) is None


def test_latest_checkpoint_missing_dir(tmp_path: Path) -> None:
    assert latest_checkpoint(tmp_path / "nope") is None


def test_latest_checkpoint_picks_numeric_max(tmp_path: Path) -> None:
    """checkpoint-9000 must lose to checkpoint-10000: numeric, not lexicographic."""
    for step in (1000, 9000, 10000):
        (tmp_path / f"checkpoint-{step}").mkdir()
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-10000"


def test_latest_checkpoint_ignores_malformed_names(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-1000").mkdir()
    (tmp_path / "checkpoint-final").mkdir()
    (tmp_path / "checkpoint-").mkdir()
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-1000"
```

Add to `tests/sweep/test_common.py` (or wherever `SweepMetricsCallback` is already covered):

```python
def test_result_json_records_installed_version(tmp_path: Path) -> None:
    """oplm.__version__ is stale (0.0.1); the installed distribution version is what ran."""
    import json
    from importlib.metadata import version

    from oplm.training.mup import SweepMetricsCallback

    callback = SweepMetricsCallback(tmp_path / "result.json")
    callback.on_train_end(_FakeTrainer())
    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["oplm_version"] == version("oplm")
```

Write `_FakeTrainer` in the test file as a minimal stand-in exposing exactly the attributes
`on_train_end` reads at `src/oplm/training/mup.py:278` — at minimum `cfg.train.batch_size`,
`cfg.train.gradient_accumulation_steps`, `cfg.train.lr`, `cfg.model.hidden_size`,
`accelerator.num_processes`, and `global_step`. Read that method first and match it exactly.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/sweep/ -q -k "checkpoint or installed_version"`
Expected: FAIL — `ImportError: cannot import name 'latest_checkpoint'`.

- [ ] **Step 3: Implement checkpoint discovery and auto-resume**

Rewrite `src/oplm/sweep/run.py`:

```python
"""Run one fully resolved μP sweep cell under Accelerate."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003  # Typer resolves annotations at runtime.
from typing import Annotated

import typer

app = typer.Typer(name="mup-run", help=__doc__, add_completion=False)

_CHECKPOINT_PREFIX = "checkpoint-"


def latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the highest-step checkpoint under ``output_dir``, or ``None``.

    Checkpoints are named ``checkpoint-<global_step>`` (see
    ``oplm.training.trainer.Trainer._save_checkpoint``), so ordering is numeric on the suffix —
    lexicographic ordering would rank ``checkpoint-9000`` above ``checkpoint-10000``.
    """
    if not output_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob(f"{_CHECKPOINT_PREFIX}*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix(_CHECKPOINT_PREFIX)
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        return None
    return max(candidates)[1]


@app.command()
def main(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    result: Annotated[Path, typer.Option("--result", dir_okay=False)],
) -> None:
    """Train one resolved config and write its sweep result.

    Idempotent under Slurm ``--requeue``: if the cell's output directory already holds a
    checkpoint and the config does not pin ``resume_from``, training resumes from the newest one
    rather than restarting at step 0.
    """
    from oplm.train import _bootstrap_training_environment

    _bootstrap_training_environment()

    from oplm.config import load_config
    from oplm.training.mup import SweepMetricsCallback
    from oplm.training.trainer import Trainer

    cfg = load_config(["--config", str(config)])
    if cfg.train.resume_from is None:
        checkpoint = latest_checkpoint(Path(cfg.train.output_dir))
        if checkpoint is not None:
            cfg.train.resume_from = str(checkpoint)
    Trainer(cfg, callbacks=[SweepMetricsCallback(result)]).train()


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Record the installed version in `result.json`**

In `src/oplm/training/mup.py`, inside `SweepMetricsCallback.on_train_end`, before building
`payload`:

```python
        from importlib.metadata import PackageNotFoundError, version

        try:
            oplm_version: str | None = version("oplm")
        except PackageNotFoundError:  # pragma: no cover - only from a bare checkout
            oplm_version = None
```

and add `"oplm_version": oplm_version` to the `payload` dict. Generated scripts install oplm
unpinned, so this is what makes version drift across a multi-week sweep detectable after the fact.

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/sweep -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
ruff check src/ && ruff format --check src/ && ty check src/ && python -m pytest tests -q -m "not slow"
git add -A src/oplm tests
git commit -m "feat: resume sweep cells from checkpoint and record oplm version"
```

---

## Phase 9 — `scale` as a thin caller

### Task 14: Rewrite `scale` to delegate to `oplm.slurm`

**Files:**
- Modify: `src/oplm/sweep/phases.py` (`scale` command; delete `_scale_cells`)
- Test: `tests/sweep/test_scale.py`

**Interfaces:**
- Produces: `oplm sweep scale --from <confirm/phase.json> --config configs/scaling.yaml
  --presets 170M,400M,800M,1B --out <dir>` writing `<dir>/<preset>/run.yaml` and
  `<dir>/jobs/oplm-<preset>-scale.sbatch`, and submitting nothing.

- [ ] **Step 1: Write the failing tests**

`tests/sweep/test_scale.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from oplm.cli import app
from oplm.config import load_config

runner = CliRunner()
SCALING = str(Path(__file__).resolve().parents[2] / "configs" / "scaling.yaml")


def _run_scale(confirm_phase: Path, out: Path, presets: str) -> None:
    result = runner.invoke(
        app,
        [
            "sweep", "scale",
            "--from", str(confirm_phase),
            "--config", SCALING,
            "--presets", presets,
            "--out", str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout


def test_scale_writes_one_script_per_preset(confirm_phase: Path, tmp_path: Path) -> None:
    out = tmp_path / "scale"
    _run_scale(confirm_phase, out, "170M,400M")
    for preset, nodes in (("170M", 4), ("400M", 8)):
        script = out / "jobs" / f"oplm-{preset}-scale.sbatch"
        assert script.exists()
        text = script.read_text()
        assert f"#SBATCH --nodes={nodes}" in text
        # Not an array: one long production run per preset.
        assert "--array" not in text


def test_scale_carries_the_confirmed_winner(confirm_phase: Path, tmp_path: Path) -> None:
    out = tmp_path / "scale"
    _run_scale(confirm_phase, out, "170M")
    cfg = load_config(["--config", str(out / "170M" / "run.yaml")])
    winner = json.loads(confirm_phase.read_text())["selected"][0]
    assert cfg.train.lr == winner["lr"]
    assert cfg.train.mup_depth_lr_exponent == winner["depth_exponent"]
    assert cfg.model.mup_output_mult == winner["output_mult"]


def test_scale_keeps_the_full_eval_suite(confirm_phase: Path, tmp_path: Path) -> None:
    """Sweep cells run one eval; scale runs the production suite."""
    out = tmp_path / "scale"
    _run_scale(confirm_phase, out, "170M")
    cfg = load_config(["--config", str(out / "170M" / "run.yaml")])
    assert set(cfg.data.eval) >= {"uniref70", "omg70", "deepclust30", "casp14"}


def test_scale_has_no_submit_flag(confirm_phase: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["sweep", "scale", "--from", str(confirm_phase), "--config", SCALING,
         "--presets", "170M", "--out", str(tmp_path / "scale"), "--submit"],
    )
    assert result.exit_code != 0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/sweep/test_scale.py -q`
Expected: FAIL — `scale` still generates sweep cells and accepts `--submit`.

- [ ] **Step 3: Rewrite the `scale` command**

Delete `_scale_cells` and replace the `scale` command:

```python
@app.command()
def scale(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    source: Annotated[Path, typer.Option("--from", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)],
    presets: Annotated[str, typer.Option("--presets")] = "170M,400M,800M,1B",
) -> None:
    """Write per-preset scaling job scripts carrying the confirmed winner.

    Generates only. These runs are 100k steps at production batch — days to weeks each, and
    multi-node — so the operator reviews and submits them on their own schedule. ``--config`` is
    an ordinary training config (see ``configs/scaling.yaml``) with no μP sweep concepts in it;
    this command's only job is to carry four hyperparameters across without hand-transcription.
    """
    try:
        selected = load_phase(source).selected
        if not selected:
            raise ValueError(f"source phase {source} has no selected winner; analyze it first")
        winner = selected[0]
        slurm = load_slurm_config(config)
        out = out.resolve()
        jobs = out / "jobs"
        jobs.mkdir(parents=True, exist_ok=True)
        for preset in _parse_strings(presets, name="--presets"):
            run_dir = out / preset
            run_dir.mkdir(parents=True, exist_ok=True)
            overrides = [
                f"model.mup_output_mult={winner['output_mult']}",
                f"train.lr={winner['lr']}",
                f"train.mup_depth_lr_exponent={winner['depth_exponent']}",
                f"train.output_dir={run_dir / 'output'}",
                f"train.wandb_run_name=oplm-{preset}-scale",
            ]
            cfg = load_config(["--config", str(config), "--preset", preset, *overrides])
            run_yaml = run_dir / "run.yaml"
            run_yaml.write_text(serialize_config(cfg))
            name = f"oplm-{preset}-scale"
            spec = JobSpec(
                name=name,
                nodes=slurm.nodes.resolve(phase="scale", preset=preset),
                time_limit=slurm.time_limit.resolve(phase="scale", preset=preset),
                command=accelerate_command(
                    module="oplm.train",
                    gpus_per_node=slurm.gpus_per_node,
                    args=f"--config {run_yaml}",
                ),
            )
            (jobs / f"{name}.sbatch").write_text(render_job(spec, slurm))
        typer.echo(f"wrote {len(list(jobs.glob('*.sbatch')))} scaling scripts to {jobs}")
        typer.echo("review and submit them yourself; `scale` does not submit")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/sweep/test_scale.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff check src/ && ruff format --check src/ && ty check src/ && python -m pytest tests -q -m "not slow"
git add -A src/oplm tests
git commit -m "feat: make scale a generate-only caller over oplm.slurm"
```

---

## Phase 10 — Documentation and verification

### Task 15: Write `docs/SLURM.md` and update the μP docs

**Files:**
- Create: `docs/SLURM.md`
- Modify: `docs/LR_SWEEP.md`, `docs/MUP.md:221`, `docs/TRAIN.md`, `AGENTS.md`

- [ ] **Step 1: Write `docs/SLURM.md`**

Operator documentation for the general layer, containing **no μP content**. Cover, in this order:
the `slurm:` block schema with every field and its default; the three accepted forms for `nodes`
and `time_limit`; how per-device batch and accumulation are derived and why an indivisible
configuration is an error; the single launcher form and why it degrades to one node; job arrays
and the homogeneous-resource constraint; `afterany` versus `afterok` and why the former is
correct; `--requeue` plus checkpoint resume; and `oplm slurm generate|submit|status` with worked
examples against `configs/scaling.yaml`.

- [ ] **Step 2: Rewrite the execution sections of `docs/LR_SWEEP.md`**

Replace "Local eight-GPU run" and "SUNK/Slurm use" with a Slurm workflow that references
`SLURM.md` rather than restating it. Include:

- the per-phase command sequence, with `--submit` on `smoke`/`coarse`/`refine`/`replicate` and
  review-then-`submit.sh` on `transfer`/`bridge`/`confirm`;
- the node and derived-batch table from the spec;
- `oplm sweep status` and the resubmit line;
- the new grid, with a sentence on why it moved (the 170M coarse winner sat on the old grid's
  lower boundary, and `analyze` only selects a refinement region when the winner is interior);
- a correction to the "keep exactly one named sequence eval task" instruction: that holds for the
  proxy phases, while `scale` uses the full suite from `configs/scaling.yaml`;
- the updated phase table, with `scale` marked generate-only;
- a note that wall-time estimates derive from a single 170M measurement and that cells now pin
  full gradient checkpointing, so they are order-of-magnitude guidance.

Keep the existing "Parameterization gates", "Depth-LR exponent grid", "Stability diagnostics",
"Deep stability probe", "Head-count control", "Phase artifacts", "Selection protocol", and
"Production WSD schedule" sections, updating command names from `python -m scripts.mup_*` to
`oplm sweep *`.

- [ ] **Step 3: Update the remaining docs**

- `docs/MUP.md:221`: point at `oplm sweep` and `docs/SLURM.md`.
- `docs/TRAIN.md`: add a link to `SLURM.md` from the distributed-launch section.
- `AGENTS.md`: record that job generation lives in `src/oplm/slurm/`, sweep tooling in
  `src/oplm/sweep/`, and that `scripts/` no longer exists.

- [ ] **Step 4: Verify every documented command exists**

```bash
oplm slurm generate --help && oplm slurm submit --help && oplm slurm status --help
oplm sweep smoke --help && oplm sweep coarse --help && oplm sweep refine --help
oplm sweep replicate --help && oplm sweep transfer --help && oplm sweep bridge --help
oplm sweep confirm --help && oplm sweep scale --help && oplm sweep analyze --help
oplm sweep status --help && oplm sweep coord-check --help
```
Expected: all exit 0.

- [ ] **Step 5: Verify no stale references remain**

Run: `grep -rn "scripts/mup_\|scripts\.mup_\|python -m scripts" docs/*.md AGENTS.md README.md`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add -A docs AGENTS.md
git commit -m "docs: document the slurm layer and the recentered sweep"
```

### Task 16: Full-suite verification

- [ ] **Step 1: Run every CI gate**

```bash
ruff check src/
ruff format --check src/
ty check src/
python -m pytest -m "not slow" --cov=oplm
```
Expected: all four pass. Do not claim completion on a partial run.

- [ ] **Step 2: Confirm the acceptance criteria from the spec**

Walk the nine acceptance criteria in
`docs/superpowers/specs/2026-07-31-mup-sweep-slurm-design.md` and verify each against the built
code. Any that cannot be demonstrated is unfinished work, not a documentation gap.

- [ ] **Step 3: Commit any remaining fixes**

```bash
git add -A
git commit -m "test: verify slurm sweep acceptance criteria"
```

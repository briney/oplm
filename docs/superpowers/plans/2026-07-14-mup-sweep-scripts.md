# Production μP Sweep Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy one-GPU μP sweep with a lean, phase-oriented workflow that calibrates production LR, output multiplier, depth correction, and batch correction from one production OPLM YAML.

**Architecture:** Keep the standard OPLM config loader, `Trainer`, and Accelerate launch path. A small one-cell runner consumes resolved YAML; a Typer phase CLI generates run YAML, commands, and a versioned JSON handoff, then ranks finite validation losses. Add only the depth-dependent optimizer grouping needed to test one exponent across the production width/depth ray.

**Tech Stack:** Python 3.11, Typer, OmegaConf-backed OPLM configuration, PyTorch Muon/AdamW, Hugging Face Accelerate, pytest, Ruff, ty.

## Global Constraints

- Use one user-provided OPLM YAML for data, evaluation, microbatch, architecture, optimizer, precision, compilation, and checkpoint settings.
- Generated cells may override only model preset geometry, μP fields, LR/depth fields, seed, global-batch-derived accumulation, duration, WSD schedule, run name, and output directory.
- Every sweep cell must set `model.mup_enable=true`, `model.mup_base_width=768`, `train.scheduler=wsd_linear`, and `train.stable_steps=train.max_steps-train.warmup_steps`.
- Keep `train.weight_decay=0.01`; reject a base config with any other value instead of silently changing it.
- Require the base config to use `train.optimizer=muon` and
  `train.muon_adjust_lr_fn=original`; preserve its remaining optimizer settings.
- Use 32:1 hidden-size/depth scaling and 64-dimensional heads; the 800M preset is 1280 hidden, 40 layers, and 20 heads.
- Treat global examples as authoritative: `global_examples = train.batch_size * gradient_accumulation_steps * world_size` must divide exactly.
- Default to 2,048 global examples for proxy cells, 8,192 for production-batch cells, and eight Accelerate processes.
- Local mode runs commands sequentially, using all requested processes for every cell, and stops at the first nonzero exit.
- Remote mode writes ordinary `accelerate launch` commands only; do not add Slurm submission or polling.
- Selection requires one finite `eval/*/loss` metric; never fall back to training loss.
- Keep `mup_output_mult` in the local LR search; do not add automated weight-decay tuning.
- Do not add token-accounting, gradient/update/activation instrumentation, retries, automatic grid expansion, concurrency, databases, dashboards, or workflow frameworks.
- Preserve unrelated untracked workspace files and directories.

## File Map

- Create `scripts/mup_run.py`: run one resolved cell under Accelerate.
- Create `tests/scripts/test_mup_common.py`: artifact, parsing, batch, and command helpers.
- Create `tests/scripts/test_mup_sweep.py`: config generation, phase grids, handoff, ranking, and local execution.
- Create `tests/scripts/test_mup_run.py`: one tiny real-data runner integration test.
- Modify `scripts/_mup_common.py`: shared geometry, JSON dataclasses, parsing, paths, metrics, and Accelerate command construction.
- Replace `scripts/mup_sweep.py`: phase generation, analysis, and sequential local execution.
- Modify `scripts/mup_coord_check.py`: resolve models from the production base YAML.
- Delete `scripts/mup_pilot_run.py`: superseded by `mup_run.py`.
- Delete `tests/scripts/test_mup_grad_accum.py`: tests the removed pilot CLI.
- Delete `tests/training/test_mup_sweep.py`: tests legacy train-loss sweep summarization.
- Modify `src/oplm/config.py` and `src/oplm/configs/train/base.yaml`: depth-LR fields and validation.
- Modify `src/oplm/configs/model/presets/800M.yaml`: correct attention geometry.
- Modify `src/oplm/training/optim.py`: depth-aware Muon and AdamW groups.
- Modify `src/oplm/training/mup.py`: retain the callback but remove dead legacy summary/argmin utilities.
- Modify `tests/training/test_config.py`, `tests/training/test_mup.py`, and `tests/training/test_optim.py`: geometry/config/optimizer coverage.
- Modify `tests/scripts/test_mup_coord_check_data.py`: production-config coordinate-check coverage.
- Modify `docs/LR_SWEEP.md`, `docs/MUP.md`, and `docs/CONFIG.md`: executable workflow and new fields.

---

### Task 1: Correct Model Geometry and Add Depth-LR Configuration

**Files:**
- Modify: `src/oplm/configs/model/presets/800M.yaml:1-7`
- Modify: `src/oplm/config.py:6-65,120-165`
- Modify: `src/oplm/configs/train/base.yaml:16-41`
- Modify: `tests/training/test_config.py:261-275,327-360`

**Interfaces:**
- Consumes: existing `load_config(argv: list[str]) -> OplmConfig`.
- Produces: `TrainConfig.mup_depth_lr_exponent: float` and `TrainConfig.mup_depth_reference_layers: int` for optimizer construction and generated run YAML.

- [ ] **Step 1: Write failing geometry and validation tests**

Add these tests to `tests/training/test_config.py`:

```python
import math


def test_800m_preset_uses_64_dimensional_heads() -> None:
    cfg = load_config(["--preset", "800M"])
    assert cfg.model.hidden_size == 1280
    assert cfg.model.num_hidden_layers == 40
    assert cfg.model.num_attention_heads == 20
    assert cfg.model.head_dim == 64


def test_depth_lr_defaults_are_noop_at_170m_reference() -> None:
    cfg = load_config([])
    assert cfg.train.mup_depth_lr_exponent == 0.0
    assert cfg.train.mup_depth_reference_layers == 24


@pytest.mark.parametrize("value", [-0.1, math.inf, -math.inf, math.nan])
def test_depth_lr_exponent_must_be_finite_and_nonnegative(value: float) -> None:
    with pytest.raises(ValueError, match="mup_depth_lr_exponent"):
        TrainConfig(mup_depth_lr_exponent=value)


@pytest.mark.parametrize("value", [0, -1])
def test_depth_lr_reference_layers_must_be_positive(value: int) -> None:
    with pytest.raises(ValueError, match="mup_depth_reference_layers"):
        TrainConfig(mup_depth_reference_layers=value)
```

- [ ] **Step 2: Run the focused tests and verify the intended failures**

Run:

```bash
pytest tests/training/test_config.py::test_800m_preset_uses_64_dimensional_heads tests/training/test_config.py::test_depth_lr_defaults_are_noop_at_170m_reference tests/training/test_config.py::test_depth_lr_exponent_must_be_finite_and_nonnegative tests/training/test_config.py::test_depth_lr_reference_layers_must_be_positive -v
```

Expected: failures showing 16 heads and missing `TrainConfig` depth fields.

- [ ] **Step 3: Implement the minimal schema and preset correction**

In `src/oplm/configs/model/presets/800M.yaml`, change only:

```yaml
  num_attention_heads: 20
```

In `src/oplm/config.py`, import `math` and add these fields immediately after
`muon_ns_steps`:

```python
import math

mup_depth_lr_exponent: float = 0.0
mup_depth_reference_layers: int = 24
```

Insert these checks at the end of `TrainConfig.__post_init__`:

```python
if not math.isfinite(self.mup_depth_lr_exponent) or self.mup_depth_lr_exponent < 0:
    raise ValueError(
        "mup_depth_lr_exponent must be finite and >= 0, "
        f"got {self.mup_depth_lr_exponent}"
    )
if self.mup_depth_reference_layers < 1:
    raise ValueError(
        "mup_depth_reference_layers must be >= 1, "
        f"got {self.mup_depth_reference_layers}"
    )
```

Add matching production defaults to `src/oplm/configs/train/base.yaml`:

```yaml
  # Empirical repeated-block LR correction for the production scaling ray.
  mup_depth_lr_exponent: 0.0
  mup_depth_reference_layers: 24
```

- [ ] **Step 4: Run configuration tests**

Run:

```bash
pytest tests/training/test_config.py -v
```

Expected: all configuration tests pass, including the packaged-YAML drift guard.

- [ ] **Step 5: Commit the configuration slice**

```bash
git add src/oplm/config.py src/oplm/configs/train/base.yaml src/oplm/configs/model/presets/800M.yaml tests/training/test_config.py
git commit -m "fix: align production muP depth geometry"
```

---

### Task 2: Apply the Depth Multiplier to Optimizer Groups

**Files:**
- Modify: `src/oplm/training/optim.py:23-200`
- Modify: `tests/training/test_mup.py:217-303`
- Modify: `tests/training/test_optim.py:50-170`

**Interfaces:**
- Consumes: `TrainConfig.mup_depth_lr_exponent`, `TrainConfig.mup_depth_reference_layers`, `model.config.num_hidden_layers`, and `mup_lr_multiplier(name, param, model_config)`.
- Produces: `MuonParamGroup(params: list[Parameter], lr_mult: float)`, `OptimizerParamGroups.muon_groups`, and effective block LR `base_lr * width_mult * (24 / L) ** alpha` where width scaling applies.

- [ ] **Step 1: Write failing effective-LR tests for both optimizers**

Append to `tests/training/test_mup.py`:

```python
def test_depth_lr_multiplier_composes_with_muon_groups() -> None:
    model = _mlm_model()
    m, _ = _mults(model)
    lr = 1e-2
    depth_mult = (1 / _LAYERS) ** 0.5
    cfg = TrainConfig(
        optimizer="muon",
        muon_adjust_lr_fn="original",
        lr=lr,
        weight_decay=0.01,
        mup_depth_lr_exponent=0.5,
        mup_depth_reference_layers=1,
    )
    optimizers = build_optimizers(model, cfg)

    assert _group_lr(
        "oplm.backbone.layers.0.attention.q_proj.weight", model, optimizers
    ) == ("Muon", pytest.approx(lr * depth_mult))
    assert _group_lr("lm_head.dense.weight", model, optimizers) == (
        "AdamW",
        pytest.approx(lr / m),
    )


def test_depth_lr_multiplier_composes_with_adamw_width_groups() -> None:
    model = _mlm_model()
    m, _ = _mults(model)
    lr = 1e-2
    depth_mult = (1 / _LAYERS) ** 0.5
    cfg = TrainConfig(
        optimizer="adamw",
        lr=lr,
        weight_decay=0.01,
        mup_depth_lr_exponent=0.5,
        mup_depth_reference_layers=1,
    )
    optimizers = build_optimizers(model, cfg)

    assert _group_lr(
        "oplm.backbone.layers.0.attention.q_proj.weight", model, optimizers
    ) == ("AdamW", pytest.approx(lr * depth_mult / m))
    assert _group_lr("lm_head.dense.weight", model, optimizers) == (
        "AdamW",
        pytest.approx(lr / m),
    )
    assert _group_lr(
        "oplm.backbone.embed_tokens.embed_tokens.weight", model, optimizers
    ) == ("AdamW", pytest.approx(lr))
```

Update the optimizer import in `tests/training/test_optim.py` to include
`OptimizerParamGroups`, then flatten `groups.muon_groups` wherever the old flat
`muon_params` field was used:

```python
def _muon_params(groups: OptimizerParamGroups) -> list[torch.nn.Parameter]:
    return [param for group in groups.muon_groups for param in group.params]
```

Use `_muon_params(groups)` in the ownership and coverage assertions. Add this no-op grouping test:

```python
def test_zero_depth_exponent_keeps_one_muon_lr_group() -> None:
    model = _model()
    groups = partition_optimizer_params(
        model,
        TrainConfig(
            optimizer="muon",
            mup_depth_lr_exponent=0.0,
            mup_depth_reference_layers=24,
        ),
    )
    assert len(groups.muon_groups) == 1
    assert groups.muon_groups[0].lr_mult == 1.0
```

- [ ] **Step 2: Run the focused optimizer tests and verify they fail**

Run:

```bash
pytest tests/training/test_mup.py::test_depth_lr_multiplier_composes_with_muon_groups tests/training/test_mup.py::test_depth_lr_multiplier_composes_with_adamw_width_groups tests/training/test_optim.py::test_zero_depth_exponent_keeps_one_muon_lr_group -v
```

Expected: failures because Muon has one flat base-LR list and AdamW lacks a depth multiplier.

- [ ] **Step 3: Implement block/non-block grouping without changing alpha-zero behavior**

In `src/oplm/training/optim.py`, add:

```python
@dataclass(frozen=True)
class MuonParamGroup:
    """Muon parameters sharing one depth LR multiplier."""

    params: list[torch.nn.Parameter]
    lr_mult: float


@dataclass(frozen=True)
class OptimizerParamGroups:
    muon_groups: list[MuonParamGroup]
    adamw_groups: list[AdamwParamGroup]


def _depth_lr_multiplier(name: str, model_config: OplmConfig, cfg: TrainConfig) -> float:
    """Return the repeated-block depth correction for one named parameter."""
    if not name.startswith("oplm.backbone.layers."):
        return 1.0
    return (
        cfg.mup_depth_reference_layers / model_config.num_hidden_layers
    ) ** cfg.mup_depth_lr_exponent
```

Replace the flat Muon list in `partition_optimizer_params` with a multiplier bucket and compose the AdamW key:

```python
muon_buckets: dict[float, list[torch.nn.Parameter]] = {}
adamw_buckets: dict[tuple[float, float], list[torch.nn.Parameter]] = {}

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue

    depth_mult = _depth_lr_multiplier(name, model_config, cfg)
    if _uses_no_weight_decay(name, param):
        weight_decay = 0.0
    elif cfg.optimizer == "muon" and param.ndim == 2 and not name.startswith("lm_head."):
        muon_buckets.setdefault(depth_mult, []).append(param)
        continue
    else:
        weight_decay = cfg.weight_decay

    lr_mult = mup_lr_multiplier(name, param, model_config) * depth_mult
    adamw_buckets.setdefault((weight_decay, lr_mult), []).append(param)

muon_groups = [
    MuonParamGroup(params=params, lr_mult=mult)
    for mult, params in muon_buckets.items()
]
adamw_groups = [
    AdamwParamGroup(params=params, weight_decay=wd, lr_mult=mult)
    for (wd, mult), params in adamw_buckets.items()
]
```

Perform coverage checks over flattened Muon groups and return both group lists:

```python
grouped = [
    *(param for group in muon_groups for param in group.params),
    *(param for group in adamw_groups for param in group.params),
]
grouped_ids = [id(param) for param in grouped]
model_ids = [id(param) for param in model.parameters() if param.requires_grad]
if len(grouped_ids) != len(set(grouped_ids)):
    raise RuntimeError("Optimizer parameter partition duplicated one or more parameters")
if set(grouped_ids) != set(model_ids):
    raise RuntimeError("Optimizer parameter partition did not cover all trainable parameters")
if cfg.optimizer == "muon" and not muon_groups:
    raise ValueError("Muon optimizer requires at least one eligible 2D hidden weight")
return OptimizerParamGroups(muon_groups=muon_groups, adamw_groups=adamw_groups)
```

Construct Muon with explicit per-group LRs:

```python
muon_optimizer = torch.optim.Muon(
    [
        {"params": list(group.params), "lr": cfg.lr * group.lr_mult}
        for group in param_groups.muon_groups
    ],
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
    momentum=cfg.muon_momentum,
    nesterov=cfg.muon_nesterov,
    ns_steps=cfg.muon_ns_steps,
    adjust_lr_fn=cfg.muon_adjust_lr_fn,
)
```

Update the `AdamwParamGroup`, `OptimizerParamGroups`,
`partition_optimizer_params`, and `_build_adamw_optimizer` docstrings so
`lr_mult` is described as the composed width/depth multiplier and Muon ownership
is described as a list of depth-LR groups.

- [ ] **Step 4: Run all optimizer and μP tests**

Run:

```bash
pytest tests/training/test_mup.py tests/training/test_optim.py tests/training/test_e2e_optim.py -v
```

Expected: all tests pass; alpha zero retains the current effective LRs and every trainable parameter is covered once.

- [ ] **Step 5: Commit the optimizer slice**

```bash
git add src/oplm/training/optim.py tests/training/test_mup.py tests/training/test_optim.py
git commit -m "feat: add muP depth learning-rate groups"
```

---

### Task 3: Add Lean Phase Artifacts and Launcher Helpers

**Files:**
- Modify: `scripts/_mup_common.py:1-71`
- Create: `tests/scripts/test_mup_common.py`

**Interfaces:**
- Consumes: phase directories and result JSON written by `SweepMetricsCallback`.
- Produces: `RunSpec`, `PhaseManifest`, `gradient_accumulation_steps`, `parse_candidates`, `relative_path`, `load_phase`, `write_phase`, `result_metric`, and `accelerate_argv`.

- [ ] **Step 1: Write failing unit tests for the shared contract**

Create `tests/scripts/test_mup_common.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts._mup_common import (
    PhaseManifest,
    RunSpec,
    accelerate_argv,
    gradient_accumulation_steps,
    load_phase,
    parse_candidates,
    relative_path,
    result_metric,
    write_phase,
)


def test_gradient_accumulation_uses_global_examples() -> None:
    assert gradient_accumulation_steps(2048, per_device_batch=32, world_size=8) == 8
    assert gradient_accumulation_steps(8192, per_device_batch=32, world_size=8) == 32


@pytest.mark.parametrize("global_examples", [0, 2050])
def test_gradient_accumulation_rejects_invalid_global_batch(global_examples: int) -> None:
    with pytest.raises(ValueError, match="global examples"):
        gradient_accumulation_steps(global_examples, per_device_batch=32, world_size=8)


def test_phase_manifest_roundtrips(tmp_path: Path) -> None:
    phase = PhaseManifest(
        version=1,
        phase="coarse",
        metric="eval/heldout/loss",
        source=None,
        runs=[
            RunSpec(
                id="170M-lr0.01-om1-a0-s42",
                config="runs/170M-lr0.01-om1-a0-s42/run.yaml",
                result="runs/170M-lr0.01-om1-a0-s42/result.json",
                params={"preset": "170M", "lr": 0.01, "seed": 42},
            )
        ],
        ranking=[],
        selected=[],
    )
    path = tmp_path / "phase.json"
    write_phase(path, phase)
    assert load_phase(path) == phase


def test_result_metric_requires_finite_validation_loss(tmp_path: Path) -> None:
    run = RunSpec("run", "runs/run/run.yaml", "runs/run/result.json", {"lr": 0.01})
    result = tmp_path / run.result
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"eval": {"eval/heldout/loss": 1.25}}))
    assert result_metric(tmp_path, run, "eval/heldout/loss") == 1.25

    result.write_text(json.dumps({"eval": {"eval/heldout/loss": float("nan")}}))
    assert result_metric(tmp_path, run, "eval/heldout/loss") is None
    result.unlink()
    assert result_metric(tmp_path, run, "eval/heldout/loss") is None


def test_accelerate_command_targets_one_resolved_cell(tmp_path: Path) -> None:
    argv = accelerate_argv(
        config=tmp_path / "run.yaml",
        result=tmp_path / "result.json",
        num_processes=8,
        accelerate_config=tmp_path / "accelerate.yaml",
    )
    assert argv == [
        "accelerate",
        "launch",
        "--config_file",
        str(tmp_path / "accelerate.yaml"),
        "--num_processes",
        "8",
        "-m",
        "scripts.mup_run",
        "--config",
        str(tmp_path / "run.yaml"),
        "--result",
        str(tmp_path / "result.json"),
    ]


def test_candidate_and_relative_path_parsing(tmp_path: Path) -> None:
    assert parse_candidates("0.01:1,0.016:0.5", ("lr", "output_mult")) == [
        {"lr": 0.01, "output_mult": 1.0},
        {"lr": 0.016, "output_mult": 0.5},
    ]
    root = tmp_path / "sweep"
    source = root / "coarse" / "phase.json"
    target = root / "refine"
    assert relative_path(source, target) == Path("../coarse/phase.json")
```

- [ ] **Step 2: Run the new unit tests and verify import failures**

Run:

```bash
pytest tests/scripts/test_mup_common.py -v
```

Expected: collection fails because the new interfaces do not exist.

- [ ] **Step 3: Implement only the shared dataclasses and helpers**

Replace the module docstring so it describes shared coordinate-check geometry
and phase artifacts without naming `mup_pilot_run`. Keep `HEAD_DIM`,
`PRESET_ASPECT_RATIO`, `Scaling`, `Optimizer`, `parse_widths`, `parse_floats`, and
`num_layers_for` for the coordinate checker. Add these types and functions to
`scripts/_mup_common.py`:

```python
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

JsonScalar = str | int | float | bool | None
Params = dict[str, JsonScalar]


@dataclass(frozen=True)
class RunSpec:
    id: str
    config: str
    result: str
    params: Params


@dataclass
class PhaseManifest:
    version: int
    phase: str
    metric: str
    source: str | None
    runs: list[RunSpec]
    ranking: list[dict[str, object]]
    selected: list[Params]


def gradient_accumulation_steps(
    global_examples: int, *, per_device_batch: int, world_size: int
) -> int:
    denominator = per_device_batch * world_size
    if global_examples < 1 or denominator < 1 or global_examples % denominator != 0:
        raise ValueError(
            "global examples must be a positive multiple of "
            f"per_device_batch * world_size ({denominator}), got {global_examples}"
        )
    return global_examples // denominator


def parse_candidates(raw: str, names: tuple[str, ...]) -> list[Params]:
    candidates: list[Params] = []
    for item in raw.split(","):
        values = item.split(":")
        if len(values) != len(names):
            raise typer.BadParameter(
                f"each candidate must contain {len(names)} colon-separated values: {item!r}"
            )
        try:
            candidates.append(dict(zip(names, (float(value) for value in values), strict=True)))
        except ValueError as exc:
            raise typer.BadParameter(f"candidate values must be numeric: {item!r}") from exc
    if not candidates:
        raise typer.BadParameter("candidates must not be empty")
    return candidates


def relative_path(path: Path, start: Path) -> Path:
    path_parts = path.resolve().parts
    start_parts = start.resolve().parts
    common = 0
    for left, right in zip(path_parts, start_parts, strict=False):
        if left != right:
            break
        common += 1
    return Path(*([".."] * (len(start_parts) - common)), *path_parts[common:])


def write_phase(path: Path, phase: PhaseManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(phase), indent=2, sort_keys=True) + "\n")


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
    )


def result_metric(phase_dir: Path, run: RunSpec, metric: str) -> float | None:
    path = phase_dir / run.result
    if not path.exists():
        return None
    value = (json.loads(path.read_text()).get("eval") or {}).get(metric)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def accelerate_argv(
    *, config: Path, result: Path, num_processes: int, accelerate_config: Path | None
) -> list[str]:
    if num_processes < 1:
        raise ValueError(f"num_processes must be >= 1, got {num_processes}")
    argv = ["accelerate", "launch"]
    if accelerate_config is not None:
        argv.extend(["--config_file", str(accelerate_config)])
    argv.extend(
        [
            "--num_processes",
            str(num_processes),
            "-m",
            "scripts.mup_run",
            "--config",
            str(config),
            "--result",
            str(result),
        ]
    )
    return argv
```

- [ ] **Step 4: Run helper tests and lint the shared module**

Run:

```bash
pytest tests/scripts/test_mup_common.py -v
ruff check scripts/_mup_common.py tests/scripts/test_mup_common.py
```

Expected: all tests pass and Ruff reports no violations.

- [ ] **Step 5: Commit the shared artifact contract**

```bash
git add scripts/_mup_common.py tests/scripts/test_mup_common.py
git commit -m "feat: add muP phase artifact helpers"
```

---

### Task 4: Add the Distributed One-Cell Runner

**Files:**
- Create: `scripts/mup_run.py`
- Create: `tests/scripts/test_mup_run.py`
- Modify: `src/oplm/training/mup.py:249-297`
- Modify: `tests/training/test_trainer.py`

**Interfaces:**
- Consumes: `--config RUN.yaml` and `--result RESULT.json`.
- Produces: a normal `Trainer` run and one callback result JSON containing final EMA train loss plus the last evaluation metrics.

- [ ] **Step 1: Write the failing tiny real-data runner test**

Create `tests/scripts/test_mup_run.py`:

```python
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from oplm.config import serialize_config
from scripts import mup_run
from tests.training.conftest import tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow


def test_mup_run_writes_validation_result(training_parquet: Path, tmp_path: Path) -> None:
    cfg = tiny_train_cfg(
        tmp_path / "trainer-output",
        training_parquet,
        max_steps=2,
        batch_size=4,
        log_every=1,
    )
    cfg.data.eval = {
        "heldout": {
            "path": str(training_parquet),
            "type": "sequence",
            "every": {"steps": 1},
        }
    }
    run_yaml = tmp_path / "run.yaml"
    result_json = tmp_path / "result.json"
    run_yaml.write_text(serialize_config(cfg))

    mup_run.main(config=run_yaml, result=result_json)

    payload = json.loads(result_json.read_text())
    assert payload["steps"] == 2
    assert payload["global_batch"] == 4
    assert payload["eval"]["eval/heldout/loss"] > 0
```

Add a focused distributed-write contract test to `tests/training/test_trainer.py`:

```python
def test_train_end_callback_is_not_emitted_on_nonmain_process() -> None:
    from oplm.training.trainer import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.accelerator = SimpleNamespace(is_main_process=False)
    calls: list[Trainer] = []

    class Callback:
        def on_train_end(self, callback_trainer: Trainer) -> None:
            calls.append(callback_trainer)

    trainer.callbacks = [Callback()]
    trainer._emit_train_end()
    assert calls == []
```

Import `SimpleNamespace` from `types` in that test module. This test should pass
before implementation because the trainer already owns the main-process guard;
the runner integration remains the red test for this task.

- [ ] **Step 2: Run the integration test and verify the missing-module failure**

Run:

```bash
pytest tests/scripts/test_mup_run.py -v
```

Expected: collection fails because `scripts.mup_run` does not exist.

- [ ] **Step 3: Implement the two-argument runner**

Create `scripts/mup_run.py` with no model or optimizer construction of its own:

```python
"""Run one fully resolved μP sweep cell under Accelerate."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(name="mup-run", help=__doc__, add_completion=False)


@app.command()
def main(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    result: Annotated[Path, typer.Option("--result", dir_okay=False)],
) -> None:
    """Train one resolved config and write its sweep result."""
    from oplm.train import _bootstrap_training_environment

    _bootstrap_training_environment()

    from oplm.config import load_config
    from oplm.training.mup import SweepMetricsCallback
    from oplm.training.trainer import Trainer

    cfg = load_config(["--config", str(config)])
    Trainer(cfg, callbacks=[SweepMetricsCallback(result)]).train()


if __name__ == "__main__":
    app()
```

Do not add a callback-side process check: `Trainer._emit_train_end` already returns on non-main processes before invoking callbacks. Update the callback docstring in `src/oplm/training/mup.py` to say `result.json` rather than prescribing the obsolete `metrics.json` filename.

- [ ] **Step 4: Run the runner integration and callback tests**

Run:

```bash
pytest tests/scripts/test_mup_run.py tests/training/test_trainer.py -v
```

Expected: all tests pass and only one result file is written.

- [ ] **Step 5: Commit the one-cell runner**

```bash
git add scripts/mup_run.py src/oplm/training/mup.py tests/scripts/test_mup_run.py tests/training/test_trainer.py
git commit -m "feat: add distributed muP sweep cell runner"
```

---

### Task 5: Generate Smoke, Coarse, and Refine Phases from Production YAML

**Files:**
- Replace: `scripts/mup_sweep.py:1-305`
- Delete: `scripts/mup_pilot_run.py`
- Delete: `tests/scripts/test_mup_grad_accum.py`
- Create: `tests/scripts/test_mup_sweep.py`

**Interfaces:**
- Consumes: `--config BASE.yaml`, optional/required `--from PHASE.json`, phase-specific grid overrides, and common launcher options.
- Produces: `phase.json`, `commands.txt`, and `runs/<id>/run.yaml` for `smoke`, `coarse`, and `refine`.

- [ ] **Step 1: Write failing generation tests**

Create `tests/scripts/test_mup_sweep.py` with a minimal production-style base config fixture:

```python
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from oplm.config import load_config
from scripts import mup_sweep
from scripts._mup_common import PhaseManifest, RunSpec, load_phase, write_phase

runner = CliRunner()


@pytest.fixture
def base_config(tmp_path: Path) -> Path:
    path = tmp_path / "base.yaml"
    path.write_text(
        """
model:
  norm_strategy: sandwich
  canon_enabled: true
  canon_positions: [A, B, C, D]
  residual_gate: channel
train:
  batch_size: 32
  optimizer: muon
  weight_decay: 0.01
  muon_adjust_lr_fn: original
  wandb_enabled: false
  save_final: false
data:
  train: /data/train
  eval:
    heldout:
      path: /data/heldout.parquet
      type: sequence
      every: {steps: 500}
""".lstrip()
    )
    return path


def test_smoke_generates_three_production_cells(base_config: Path, tmp_path: Path) -> None:
    out = tmp_path / "smoke"
    result = runner.invoke(
        mup_sweep.app,
        ["smoke", "--config", str(base_config), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output

    phase = load_phase(out / "phase.json")
    assert phase.phase == "smoke"
    assert phase.metric == "eval/heldout/loss"
    assert [run.params["lr"] for run in phase.runs] == [0.0025, 0.01, 0.04]
    assert len((out / "commands.txt").read_text().splitlines()) == 3

    cfg = load_config(["--config", str(out / phase.runs[1].config)])
    assert cfg.model.hidden_size == 768
    assert cfg.model.num_hidden_layers == 24
    assert cfg.model.num_attention_heads == 12
    assert cfg.model.norm_strategy == "sandwich"
    assert cfg.model.canon_positions == ["A", "B", "C", "D"]
    assert cfg.model.residual_gate == "channel"
    assert cfg.model.mup_enable is True
    assert cfg.model.mup_base_width == 768
    assert cfg.model.mup_output_mult == 1.0
    assert cfg.train.optimizer == "muon"
    assert cfg.train.weight_decay == 0.01
    assert cfg.train.lr == 0.01
    assert cfg.train.scheduler == "wsd_linear"
    assert cfg.train.max_steps == 1000
    assert cfg.train.max_epochs is None
    assert cfg.train.warmup_steps == 100
    assert cfg.train.stable_steps == 900
    assert cfg.train.gradient_accumulation_steps == 8


def test_generation_rejects_nonproduction_weight_decay(
    base_config: Path, tmp_path: Path
) -> None:
    text = base_config.read_text().replace("weight_decay: 0.01", "weight_decay: 0.0")
    base_config.write_text(text)
    result = runner.invoke(
        mup_sweep.app,
        ["smoke", "--config", str(base_config), "--out", str(tmp_path / "smoke")],
    )
    assert result.exit_code != 0
    assert "weight_decay=0.01" in result.output


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("optimizer: muon", "optimizer: adamw", "optimizer=muon"),
        (
            "muon_adjust_lr_fn: original",
            "muon_adjust_lr_fn: match_rms_adamw",
            "muon_adjust_lr_fn=original",
        ),
    ],
)
def test_generation_rejects_nonproduction_optimizer(
    base_config: Path, tmp_path: Path, old: str, new: str, message: str
) -> None:
    base_config.write_text(base_config.read_text().replace(old, new))
    result = runner.invoke(
        mup_sweep.app,
        ["smoke", "--config", str(base_config), "--out", str(tmp_path / "smoke")],
    )
    assert result.exit_code != 0
    assert message in result.output


def test_generation_rejects_nonintegral_accumulation(
    base_config: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        mup_sweep.app,
        [
            "smoke",
            "--config",
            str(base_config),
            "--out",
            str(tmp_path / "smoke"),
            "--global-examples",
            "2050",
        ],
    )
    assert result.exit_code != 0
    assert "global examples" in result.output


def test_refine_uses_coarse_selected_lrs(base_config: Path, tmp_path: Path) -> None:
    coarse_dir = tmp_path / "coarse"
    coarse_dir.mkdir()
    write_phase(
        coarse_dir / "phase.json",
        PhaseManifest(
            1,
            "coarse",
            "eval/heldout/loss",
            None,
            [],
            [],
            [{"lr": 0.0063, "output_mult": 1.0}, {"lr": 0.01, "output_mult": 1.0}, {"lr": 0.016, "output_mult": 1.0}],
        ),
    )
    out = tmp_path / "refine"
    result = runner.invoke(
        mup_sweep.app,
        [
            "refine",
            "--config",
            str(base_config),
            "--from",
            str(coarse_dir / "phase.json"),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    phase = load_phase(out / "phase.json")
    assert len(phase.runs) == 9
    assert {
        (run.params["lr"], run.params["output_mult"]) for run in phase.runs
    } == {
        (lr, output_mult)
        for lr in (0.0063, 0.01, 0.016)
        for output_mult in (0.5, 1.0, 2.0)
    }
```

- [ ] **Step 2: Run generation tests and verify the legacy CLI fails them**

Run:

```bash
pytest tests/scripts/test_mup_sweep.py -v
```

Expected: failures because the current script has no phase subcommands or base-YAML generation.

- [ ] **Step 3: Implement resolved run generation**

Replace the legacy concurrency orchestrator in `scripts/mup_sweep.py`. Define one standard run-parameter shape:

```python
def _cell(
    *,
    preset: str,
    lr: float,
    output_mult: float,
    depth_exponent: float,
    seed: int,
    global_examples: int,
    max_steps: int,
    warmup_steps: int,
    batch_mult: float | None = None,
) -> Params:
    params: Params = {
        "preset": preset,
        "lr": lr,
        "output_mult": output_mult,
        "depth_exponent": depth_exponent,
        "seed": seed,
        "global_examples": global_examples,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
    }
    if batch_mult is not None:
        params["batch_mult"] = batch_mult
    return params
```

Use the preset YAML as the single source for the three geometry overrides, then load the user YAML through the normal loader:

```python
def _write_run_config(
    base_config: Path,
    run_dir: Path,
    params: Params,
    *,
    num_processes: int,
) -> Path:
    preset = str(params["preset"])
    preset_model = get_preset_config(preset).model
    base = load_config(["--config", str(base_config)])
    if base.train.optimizer != "muon":
        raise ValueError(
            f"μP sweep requires train.optimizer=muon, got {base.train.optimizer}"
        )
    if base.train.muon_adjust_lr_fn != "original":
        raise ValueError(
            "μP sweep requires train.muon_adjust_lr_fn=original, "
            f"got {base.train.muon_adjust_lr_fn}"
        )
    if base.train.weight_decay != 0.01:
        raise ValueError(
            f"μP sweep requires train.weight_decay=0.01, got {base.train.weight_decay}"
        )
    grad_accum = gradient_accumulation_steps(
        int(params["global_examples"]),
        per_device_batch=base.train.batch_size,
        world_size=num_processes,
    )
    run_name = run_dir.name
    overrides = [
        f"model.hidden_size={preset_model.hidden_size}",
        f"model.num_hidden_layers={preset_model.num_hidden_layers}",
        f"model.num_attention_heads={preset_model.num_attention_heads}",
        "model.head_dim=64",
        "model.mup_enable=true",
        "model.mup_base_width=768",
        f"model.mup_output_mult={params['output_mult']}",
        f"train.lr={params['lr']}",
        f"train.mup_depth_lr_exponent={params['depth_exponent']}",
        "train.mup_depth_reference_layers=24",
        f"train.seed={params['seed']}",
        f"train.gradient_accumulation_steps={grad_accum}",
        f"train.max_steps={params['max_steps']}",
        "train.max_epochs=null",
        f"train.warmup_steps={params['warmup_steps']}",
        f"train.stable_steps={int(params['max_steps']) - int(params['warmup_steps'])}",
        "train.scheduler=wsd_linear",
        f"train.output_dir={run_dir / 'output'}",
        f"train.wandb_run_name={run_name}",
    ]
    cfg = load_config(["--config", str(base_config), *overrides])
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.yaml"
    path.write_text(serialize_config(cfg))
    return path
```

At the start of `_write_run_config`, validate the pilot duration:

```python
max_steps = int(params["max_steps"])
warmup_steps = int(params["warmup_steps"])
if warmup_steps < 0 or warmup_steps >= max_steps:
    raise ValueError(
        f"warmup_steps must satisfy 0 <= warmup_steps < max_steps, "
        f"got {warmup_steps} and {max_steps}"
    )
```

Infer the metric only when exactly one non-null `data.eval` entry exists;
otherwise require `--metric`:

```python
def _resolve_metric(base_config: Path, metric: str | None) -> str:
    if metric is not None:
        return metric
    cfg = load_config(["--config", str(base_config)])
    eval_names = [name for name, value in (cfg.data.eval or {}).items() if value is not None]
    if len(eval_names) != 1:
        raise ValueError("--metric is required unless the base config has exactly one eval task")
    return f"eval/{eval_names[0]}/loss"


def _input_candidates(
    source: Path, raw: str | None, names: tuple[str, ...]
) -> list[Params]:
    if raw is not None:
        return parse_candidates(raw, names)
    selected = load_phase(source).selected
    if not selected:
        raise ValueError(
            f"source phase {source} has no selected candidates; analyze it or pass --candidates"
        )
    return [dict(candidate) for candidate in selected]
```

Generate deterministic IDs from `preset`, LR, output multiplier, depth exponent, and seed. Store config/result paths relative to the phase directory. Write commands with `shlex.join` and a trailing newline:

```python
def _run_id(params: Params) -> str:
    return (
        f"{params['preset']}-lr{float(params['lr']):g}"
        f"-om{float(params['output_mult']):g}"
        f"-a{float(params['depth_exponent']):g}-s{int(params['seed'])}"
    )


def _generate_phase(
    *,
    name: str,
    base_config: Path,
    out: Path,
    metric: str | None,
    source: Path | None,
    cells: list[Params],
    num_processes: int,
    accelerate_config: Path | None,
) -> tuple[Path, list[list[str]]]:
    base_config = base_config.resolve()
    out = out.resolve()
    source = source.resolve() if source is not None else None
    out.mkdir(parents=True, exist_ok=True)
    runs: list[RunSpec] = []
    commands: list[list[str]] = []
    for params in cells:
        run_id = _run_id(params)
        run_dir = out / "runs" / run_id
        config = _write_run_config(base_config, run_dir, params, num_processes=num_processes)
        result = run_dir / "result.json"
        runs.append(
            RunSpec(
                run_id,
                str(relative_path(config, out)),
                str(relative_path(result, out)),
                params,
            )
        )
        commands.append(
            accelerate_argv(
                config=config,
                result=result,
                num_processes=num_processes,
                accelerate_config=accelerate_config,
            )
        )
    phase_path = out / "phase.json"
    write_phase(
        phase_path,
        PhaseManifest(
            version=1,
            phase=name,
            metric=_resolve_metric(base_config, metric),
            source=str(relative_path(source, out)) if source is not None else None,
            runs=runs,
            ranking=[],
            selected=[],
        ),
    )
    (out / "commands.txt").write_text(
        "\n".join(shlex.join(command) for command in commands) + "\n"
    )
    return phase_path, commands
```

- [ ] **Step 4: Register the three early phase commands with exact defaults**

Use these cell definitions and expose each listed value as a Typer option:

```python
SMOKE_LRS = (0.0025, 0.01, 0.04)
COARSE_LRS = (0.0025, 0.004, 0.0063, 0.01, 0.016, 0.025, 0.04)
OUTPUT_MULTS = (0.5, 1.0, 2.0)


def _smoke_cells(lrs: list[float], global_examples: int, seed: int, steps: int, warmup: int) -> list[Params]:
    return [
        _cell(
            preset="170M",
            lr=lr,
            output_mult=1.0,
            depth_exponent=0.0,
            seed=seed,
            global_examples=global_examples,
            max_steps=steps,
            warmup_steps=warmup,
        )
        for lr in lrs
    ]


def _coarse_cells(lrs: list[float], global_examples: int, seed: int, steps: int, warmup: int) -> list[Params]:
    return _smoke_cells(lrs, global_examples, seed, steps, warmup)


def _refine_cells(
    candidates: list[Params],
    output_mults: list[float],
    global_examples: int,
    seed: int,
    steps: int,
    warmup: int,
) -> list[Params]:
    return [
        _cell(
            preset="170M",
            lr=float(candidate["lr"]),
            output_mult=output_mult,
            depth_exponent=0.0,
            seed=seed,
            global_examples=global_examples,
            max_steps=steps,
            warmup_steps=warmup,
        )
        for candidate in candidates
        for output_mult in output_mults
    ]
```

Command defaults:

| Command | Source | LR/candidate default | Global examples | Seed | Steps | Warmup |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `smoke` | none | `0.0025,0.01,0.04` | 2048 | 42 | 1000 | 100 |
| `coarse` | optional smoke | seven-point coarse grid | 2048 | 42 | 10000 | 5000 |
| `refine` | required coarse | `source.selected` or `--candidates LR:OUTPUT,...`; output mults `0.5,1,2` | 2048 | 42 | 20000 | 5000 |

For `coarse`, inspect a supplied smoke result and remove exactly `0.04` only when that smoke cell lacks a finite configured validation metric. Refuse coarse generation when smoke `0.0025` or `0.01` is non-finite. Do not otherwise rank in this task.

All commands take `--config`, `--out`, optional `--metric`, `--num-processes` default 8, and optional `--accelerate-config`. `refine` takes `--from`; `coarse` accepts it optionally.

- [ ] **Step 5: Delete legacy pilot surfaces and run generation tests**

Delete `scripts/mup_pilot_run.py` and `tests/scripts/test_mup_grad_accum.py`. Run:

```bash
pytest tests/scripts/test_mup_common.py tests/scripts/test_mup_sweep.py -v
ruff check scripts/_mup_common.py scripts/mup_run.py scripts/mup_sweep.py tests/scripts
```

Expected: helper and early-phase generation tests pass; no code imports `mup_pilot_run`.

- [ ] **Step 6: Commit early phase generation**

```bash
git add scripts/mup_sweep.py scripts/mup_pilot_run.py tests/scripts/test_mup_grad_accum.py tests/scripts/test_mup_sweep.py
git commit -m "feat: generate production muP sweep phases"
```

---

### Task 6: Generate Replication, Transfer, Bridge, Confirmation, and Scaling Phases

**Files:**
- Modify: `scripts/mup_sweep.py`
- Modify: `tests/scripts/test_mup_sweep.py`

**Interfaces:**
- Consumes: selected parameter dictionaries in a prior `phase.json`, with optional `--candidates` override.
- Produces: all later protocol grids while preserving paired LR/output configurations.

- [ ] **Step 1: Write failing default-grid and handoff tests**

Add helpers to `tests/scripts/test_mup_sweep.py` that write source manifests with selected candidates, then add:

```python
def test_later_phase_default_cell_counts() -> None:
    candidates = [
        {"lr": 0.01, "output_mult": 1.0},
        {"lr": 0.016, "output_mult": 0.5},
    ]
    source_runs = [
        RunSpec(
            id=f"refine-{index}",
            config=f"runs/refine-{index}/run.yaml",
            result=f"runs/refine-{index}/result.json",
            params={
                "preset": "170M",
                **candidate,
                "depth_exponent": 0.0,
                "seed": 42,
                "global_examples": 2048,
                "max_steps": 20000,
                "warmup_steps": 5000,
            },
        )
        for index, candidate in enumerate(candidates)
    ]

    assert len(mup_sweep._replicate_cells(source_runs, candidates, [42, 43, 44])) == 4
    assert len(
        mup_sweep._transfer_cells(
            candidates,
            presets=["400M", "800M", "1B"],
            steps=[10000, 20000, 10000],
            exponents=[0.0, 0.25, 0.5],
            global_examples=2048,
            seed=42,
            warmup=5000,
        )
    ) == 18

    transferred = [{"lr": 0.01, "output_mult": 1.0, "depth_exponent": 0.25}]
    bridge = mup_sweep._bridge_cells(
        transferred,
        multipliers=[0.7, 1.0, 1.4, 2.0],
        global_examples=8192,
        seed=42,
        steps=10000,
        warmup=5000,
    )
    assert len(bridge) == 4
    assert [cell["lr"] for cell in bridge] == pytest.approx([0.007, 0.01, 0.014, 0.02])

    bridge_finalists = [
        {"lr": 0.01, "output_mult": 1.0, "depth_exponent": 0.25, "batch_mult": 1.0},
        {"lr": 0.014, "output_mult": 1.0, "depth_exponent": 0.25, "batch_mult": 1.4},
    ]
    assert len(
        mup_sweep._confirm_cells(
            bridge_finalists,
            global_examples=8192,
            seed=42,
            steps=10000,
            warmup=5000,
        )
    ) == 2
    assert len(
        mup_sweep._scale_cells(
            bridge_finalists[:1],
            presets=["50M", "170M", "400M", "800M", "1B"],
            global_examples=8192,
            seed=42,
            steps=100000,
            warmup=5000,
        )
    ) == 5
```

- [ ] **Step 2: Run the new test and verify missing builder failures**

Run:

```bash
pytest tests/scripts/test_mup_sweep.py::test_later_phase_default_cell_counts -v
```

Expected: failure because later-phase builders do not exist.

- [ ] **Step 3: Implement later phase builders with paired candidates**

Add strict list parsers to `scripts/mup_sweep.py` for the aligned model/step and
seed options:

```python
def _parse_ints(raw: str, *, name: str) -> list[int]:
    try:
        values = [int(token) for token in raw.split(",") if token.strip()]
    except ValueError as exc:
        raise typer.BadParameter(f"{name} must be comma-separated ints, got {raw!r}") from exc
    if not values:
        raise typer.BadParameter(f"{name} must list at least one value")
    return values


def _parse_strings(raw: str, *, name: str) -> list[str]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise typer.BadParameter(f"{name} must list at least one value")
    return values
```

Add these pure builders:

```python
def _replicate_cells(
    source_runs: list[RunSpec], candidates: list[Params], seeds: list[int]
) -> list[Params]:
    cells: list[Params] = []
    if 42 not in seeds:
        raise ValueError("replicate seeds must include source seed 42")
    for candidate in candidates:
        source = next(
            run
            for run in source_runs
            if run.params.get("lr") == candidate["lr"]
            and run.params.get("output_mult") == candidate["output_mult"]
            and run.params.get("seed") == 42
        )
        for seed in seeds:
            if seed == 42:
                continue
            cells.append({**source.params, "seed": seed})
    return cells


def _transfer_cells(
    candidates: list[Params],
    *,
    presets: list[str],
    steps: list[int],
    exponents: list[float],
    global_examples: int,
    seed: int,
    warmup: int,
) -> list[Params]:
    if len(presets) != len(steps):
        raise ValueError("--presets and --steps must contain the same number of values")
    return [
        _cell(
            preset=preset,
            lr=float(candidate["lr"]),
            output_mult=float(candidate["output_mult"]),
            depth_exponent=exponent,
            seed=seed,
            global_examples=global_examples,
            max_steps=model_steps,
            warmup_steps=warmup,
        )
        for candidate in candidates
        for preset, model_steps in zip(presets, steps, strict=True)
        for exponent in exponents
    ]


def _bridge_cells(
    candidates: list[Params],
    *,
    multipliers: list[float],
    global_examples: int,
    seed: int,
    steps: int,
    warmup: int,
) -> list[Params]:
    candidate = candidates[0]
    return [
        _cell(
            preset="170M",
            lr=float(candidate["lr"]) * multiplier,
            output_mult=float(candidate["output_mult"]),
            depth_exponent=float(candidate["depth_exponent"]),
            seed=seed,
            global_examples=global_examples,
            max_steps=steps,
            warmup_steps=warmup,
            batch_mult=multiplier,
        )
        for multiplier in multipliers
    ]


def _confirm_cells(
    candidates: list[Params],
    *,
    global_examples: int,
    seed: int,
    steps: int,
    warmup: int,
) -> list[Params]:
    return [
        _cell(
            preset="800M",
            lr=float(candidate["lr"]),
            output_mult=float(candidate["output_mult"]),
            depth_exponent=float(candidate["depth_exponent"]),
            seed=seed,
            global_examples=global_examples,
            max_steps=steps,
            warmup_steps=warmup,
            batch_mult=float(candidate["batch_mult"]),
        )
        for candidate in candidates
    ]


def _scale_cells(
    candidates: list[Params],
    *,
    presets: list[str],
    global_examples: int,
    seed: int,
    steps: int,
    warmup: int,
) -> list[Params]:
    candidate = candidates[0]
    return [
        _cell(
            preset=preset,
            lr=float(candidate["lr"]),
            output_mult=float(candidate["output_mult"]),
            depth_exponent=float(candidate["depth_exponent"]),
            seed=seed,
            global_examples=global_examples,
            max_steps=steps,
            warmup_steps=warmup,
            batch_mult=float(candidate["batch_mult"]),
        )
        for preset in presets
    ]
```

- [ ] **Step 4: Register later commands and exact defaults**

Every command takes the common `--config`, `--from`, `--out`, `--metric`, `--num-processes=8`, and optional `--accelerate-config` options. A supplied `--candidates` string replaces `source.selected` using these formats:

- `replicate`, `transfer`: `LR:OUTPUT,LR:OUTPUT`.
- `bridge`, `confirm`, `scale`: `LR:OUTPUT:ALPHA` for bridge and `LR:OUTPUT:ALPHA:BATCH_MULT` for confirm/scale.

Reuse `_input_candidates` from Task 5 in every later command so an unanalyzed
or blocked source cannot silently generate an empty phase.

Use these exact defaults:

| Command | Models | Grid | Batch | Seed(s) | Steps | Warmup |
| --- | --- | --- | ---: | --- | --- | ---: |
| `replicate` | inherited | selected candidates | inherited | `42,43,44` | inherited | inherited |
| `transfer` | `400M,800M,1B` | candidate pairs × `0,0.25,0.5` | 2048 | 42 | `10000,20000,10000` | 5000 |
| `bridge` | 170M | top transfer candidate × `0.7,1,1.4,2` | 8192 | 42 | 10000 | 5000 |
| `confirm` | 800M | top one or two replicated bridge candidates | 8192 | 42 | 10000 | 5000 |
| `scale` | `50M,170M,400M,800M,1B` | confirmed winner only | 8192 | 42 | 100000 | 5000 |

Parse comma-separated preset, integer, float, and seed lists with the same strict
nonempty behavior as `parse_floats`. Validate aligned transfer preset/step list
lengths and require replicate seed lists to contain 42. With defaults,
`replicate` schedules only seeds 43 and 44; analysis reuses seed 42 from its
source phase.

- [ ] **Step 5: Run all generation tests**

Run:

```bash
pytest tests/scripts/test_mup_common.py tests/scripts/test_mup_sweep.py -v
ruff check scripts/mup_sweep.py tests/scripts/test_mup_sweep.py
```

Expected: every default grid has the approved cell count and handoff values.

- [ ] **Step 6: Commit later phase generation**

```bash
git add scripts/mup_sweep.py tests/scripts/test_mup_sweep.py
git commit -m "feat: generate muP transfer and bridge phases"
```

---

### Task 7: Analyze Validation Results and Run Locally in Sequence

**Files:**
- Modify: `scripts/mup_sweep.py`
- Modify: `scripts/_mup_common.py`
- Modify: `tests/scripts/test_mup_sweep.py`
- Modify: `src/oplm/training/mup.py:244-362`
- Delete: `tests/training/test_mup_sweep.py`

**Interfaces:**
- Consumes: completed or partial `phase.json` run lists and callback result JSON.
- Produces: deterministic `ranking` and `selected` entries in the same manifest; `analyze PHASE_JSON`; `--local` sequential execution for every generation command.

- [ ] **Step 1: Write failing ranking, failure, handoff, and local-execution tests**

Add a result writer to `tests/scripts/test_mup_sweep.py`:

```python
def _write_result(phase_dir: Path, run: RunSpec, loss: float) -> None:
    path = phase_dir / run.result
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"eval": {"eval/heldout/loss": loss}}))
```

Add direct/coarse tests:

```python
def test_coarse_selects_interior_winner_and_neighbors(tmp_path: Path) -> None:
    lrs = [0.0025, 0.004, 0.0063, 0.01, 0.016, 0.025, 0.04]
    runs = [
        RunSpec(
            f"run-{index}",
            f"runs/run-{index}/run.yaml",
            f"runs/run-{index}/result.json",
            {"lr": lr, "output_mult": 1.0, "seed": 42},
        )
        for index, lr in enumerate(lrs)
    ]
    phase = PhaseManifest(1, "coarse", "eval/heldout/loss", None, runs, [], [])
    path = tmp_path / "phase.json"
    write_phase(path, phase)
    for run in runs:
        _write_result(tmp_path, run, abs(float(run.params["lr"]) - 0.01) + 1.0)

    mup_sweep.analyze_phase(path)

    analyzed = load_phase(path)
    assert [candidate["lr"] for candidate in analyzed.selected] == [0.0063, 0.01, 0.016]


def test_coarse_edge_winner_blocks_refinement(tmp_path: Path) -> None:
    runs = [
        RunSpec(
            f"run-{index}",
            f"runs/run-{index}/run.yaml",
            f"runs/run-{index}/result.json",
            {"lr": lr, "output_mult": 1.0, "seed": 42},
        )
        for index, lr in enumerate([0.0025, 0.004, 0.0063])
    ]
    path = tmp_path / "phase.json"
    write_phase(path, PhaseManifest(1, "coarse", "eval/heldout/loss", None, runs, [], []))
    for index, run in enumerate(runs):
        _write_result(tmp_path, run, 1.0 + index)
    mup_sweep.analyze_phase(path)
    assert load_phase(path).selected == []


def test_missing_and_nonfinite_results_are_ineligible(tmp_path: Path) -> None:
    runs = [
        RunSpec("ok", "runs/ok/run.yaml", "runs/ok/result.json", {"lr": 0.01}),
        RunSpec("missing", "runs/missing/run.yaml", "runs/missing/result.json", {"lr": 0.016}),
        RunSpec("nan", "runs/nan/run.yaml", "runs/nan/result.json", {"lr": 0.025}),
    ]
    path = tmp_path / "phase.json"
    write_phase(path, PhaseManifest(1, "confirm", "eval/heldout/loss", None, runs, [], []))
    _write_result(tmp_path, runs[0], 1.0)
    _write_result(tmp_path, runs[2], float("nan"))
    mup_sweep.analyze_phase(path)
    analyzed = load_phase(path)
    scores = {entry["id"]: entry["score"] for entry in analyzed.ranking}
    assert scores == {"ok": 1.0, "missing": None, "nan": None}
    assert analyzed.selected == [{"lr": 0.01}]
```

Add aggregate ranking tests using small synthetic manifests:

```python
def _replicate_fixture(
    tmp_path: Path, *, losses: dict[float, list[float]]
) -> tuple[Path, Path]:
    source_dir = tmp_path / "refine"
    replicate_dir = tmp_path / "replicate"
    source_dir.mkdir()
    replicate_dir.mkdir()
    candidates = [
        {"lr": lr, "output_mult": 1.0, "depth_exponent": 0.0}
        for lr in losses
    ]
    source_runs = [
        RunSpec(
            f"lr-{lr:g}-seed-42",
            f"runs/lr-{lr:g}-seed-42/run.yaml",
            f"runs/lr-{lr:g}-seed-42/result.json",
            {**candidate, "seed": 42},
        )
        for lr, candidate in zip(losses, candidates, strict=True)
    ]
    replicate_runs = [
        RunSpec(
            f"lr-{lr:g}-seed-{seed}",
            f"runs/lr-{lr:g}-seed-{seed}/run.yaml",
            f"runs/lr-{lr:g}-seed-{seed}/result.json",
            {**candidate, "seed": seed},
        )
        for lr, candidate in zip(losses, candidates, strict=True)
        for seed in (43, 44)
    ]
    source_path = source_dir / "phase.json"
    replicate_path = replicate_dir / "phase.json"
    write_phase(
        source_path,
        PhaseManifest(
            1, "refine", "eval/heldout/loss", None, source_runs, [], candidates
        ),
    )
    write_phase(
        replicate_path,
        PhaseManifest(
            1,
            "replicate",
            "eval/heldout/loss",
            "../refine/phase.json",
            replicate_runs,
            [],
            [],
        ),
    )
    for run in source_runs:
        lr = float(run.params["lr"])
        _write_result(source_dir, run, losses[lr][0])
    for run in replicate_runs:
        lr = float(run.params["lr"])
        seed_index = int(run.params["seed"]) - 42
        _write_result(replicate_dir, run, losses[lr][seed_index])
    return source_path, replicate_path


def _transfer_fixture(tmp_path: Path) -> Path:
    phase_dir = tmp_path / "transfer"
    phase_dir.mkdir()
    runs: list[RunSpec] = []
    for lr in (0.01, 0.016):
        for exponent in (0.0, 0.25, 0.5):
            for preset in ("400M", "800M", "1B"):
                run_id = f"{preset}-lr{lr:g}-a{exponent:g}"
                runs.append(
                    RunSpec(
                        run_id,
                        f"runs/{run_id}/run.yaml",
                        f"runs/{run_id}/result.json",
                        {
                            "preset": preset,
                            "lr": lr,
                            "output_mult": 1.0,
                            "depth_exponent": exponent,
                            "seed": 42,
                        },
                    )
                )
    path = phase_dir / "phase.json"
    write_phase(
        path,
        PhaseManifest(1, "transfer", "eval/heldout/loss", None, runs, [], []),
    )
    model_offset = {"400M": 0.0, "800M": 0.01, "1B": 0.02}
    for run in runs:
        lr = float(run.params["lr"])
        exponent = float(run.params["depth_exponent"])
        preset = str(run.params["preset"])
        if lr == 0.016 and exponent == 0.5 and preset == "1B":
            continue
        loss = 1.0 + model_offset[preset] + abs(exponent - 0.25) + (lr - 0.01) * 10
        _write_result(phase_dir, run, loss)
    return path


def test_replicate_ranks_three_seed_mean_and_reuses_source_seed(tmp_path: Path) -> None:
    # Source refine has seed 42 for two candidates; replicate has seeds 43 and 44.
    # Candidate 0.01 losses 1.0, 1.1, 1.2; candidate 0.016 losses 0.9, 1.5, 1.6.
    # Mean therefore selects 0.01 despite 0.016 winning seed 42.
    source_path, replicate_path = _replicate_fixture(
        tmp_path,
        losses={0.01: [1.0, 1.1, 1.2], 0.016: [0.9, 1.5, 1.6]},
    )
    assert source_path.exists()
    mup_sweep.analyze_phase(replicate_path)
    analyzed = load_phase(replicate_path)
    assert analyzed.selected[0]["lr"] == 0.01
    assert analyzed.ranking[0]["score"] == pytest.approx(1.1)


def test_transfer_sums_per_model_ranks_and_requires_all_models(tmp_path: Path) -> None:
    path = _transfer_fixture(tmp_path)
    mup_sweep.analyze_phase(path)
    analyzed = load_phase(path)
    assert analyzed.selected[0] == {
        "lr": 0.01,
        "output_mult": 1.0,
        "depth_exponent": 0.25,
    }
    incomplete = next(
        entry
        for entry in analyzed.ranking
        if entry["params"]["lr"] == 0.016
        and entry["params"]["depth_exponent"] == 0.5
    )
    assert incomplete["score"] is None
```

Add the local stop test:

```python
def test_local_execution_is_sequential_and_stops_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(argv: list[str], *, check: bool) -> None:
        seen.append(argv)
        if len(seen) == 2:
            raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(subprocess, "run", fake_run)
    commands = [["run", "one"], ["run", "two"], ["run", "three"]]
    with pytest.raises(subprocess.CalledProcessError):
        mup_sweep._run_local(commands)
    assert seen == commands[:2]
```

- [ ] **Step 2: Run analysis tests and verify missing behavior**

Run:

```bash
pytest tests/scripts/test_mup_sweep.py -v
```

Expected: new analysis and local-execution tests fail because the functions do not exist.

- [ ] **Step 3: Implement direct and aggregate ranking**

Use one candidate-key helper so transient fields never leak into `selected`:

```python
def _candidate(params: Params, keys: tuple[str, ...]) -> Params:
    return {key: params[key] for key in keys if key in params}


def _direct_ranking(phase_dir: Path, phase: PhaseManifest) -> list[dict[str, object]]:
    entries = [
        {
            "id": run.id,
            "params": run.params,
            "score": result_metric(phase_dir, run, phase.metric),
        }
        for run in phase.runs
    ]
    return sorted(
        entries,
        key=lambda entry: (
            entry["score"] is None,
            float(entry["score"]) if entry["score"] is not None else math.inf,
            float(entry["params"].get("lr", math.inf)),
        ),
    )
```

Resolve a source manifest and normalize aggregate candidate keys with:

```python
def _source_phase(phase_path: Path, phase: PhaseManifest) -> tuple[Path, PhaseManifest]:
    if phase.source is None:
        raise ValueError(f"{phase.phase} requires a source phase")
    source_path = (phase_path.parent / phase.source).resolve()
    return source_path, load_phase(source_path)


def _aggregate_key(params: Params) -> tuple[float, float, float, float | None]:
    batch_mult = params.get("batch_mult")
    return (
        float(params["lr"]),
        float(params["output_mult"]),
        float(params.get("depth_exponent", 0.0)),
        float(batch_mult) if batch_mult is not None else None,
    )


def _aggregate_params(key: tuple[float, float, float, float | None]) -> Params:
    lr, output_mult, depth_exponent, batch_mult = key
    params: Params = {
        "lr": lr,
        "output_mult": output_mult,
        "depth_exponent": depth_exponent,
    }
    if batch_mult is not None:
        params["batch_mult"] = batch_mult
    return params


def _sort_aggregate(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    def key(entry: dict[str, object]) -> tuple[bool, float, float, float, float, float]:
        params = entry["params"]
        batch_mult = params.get("batch_mult")
        return (
            entry["score"] is None,
            float(entry["score"]) if entry["score"] is not None else math.inf,
            float(params["lr"]),
            float(params["output_mult"]),
            float(params["depth_exponent"]),
            float(batch_mult) if batch_mult is not None else 0.0,
        )

    return sorted(
        entries,
        key=key,
    )


def _select_count(ranking: list[dict[str, object]], count: int) -> list[Params]:
    eligible = [entry for entry in ranking if entry["score"] is not None]
    if len(eligible) < count:
        return []
    return [dict(entry["params"]) for entry in eligible[:count]]
```

Implement the three-seed mean and cross-model rank sum directly from planned
runs, so missing files remain represented by `score: null`:

```python
def _replicate_ranking(
    phase_path: Path, phase: PhaseManifest
) -> list[dict[str, object]]:
    source_path, source = _source_phase(phase_path, phase)
    locations = [
        *((source_path.parent, run) for run in source.runs),
        *((phase_path.parent, run) for run in phase.runs),
    ]
    keys = {_aggregate_key(run.params) for run in phase.runs}
    entries: list[dict[str, object]] = []
    for key in keys:
        matching = [
            (phase_dir, run)
            for phase_dir, run in locations
            if _aggregate_key(run.params) == key
        ]
        seeds = {int(run.params["seed"]) for _, run in matching}
        values = [result_metric(phase_dir, run, phase.metric) for phase_dir, run in matching]
        score = (
            sum(float(value) for value in values) / len(values)
            if 42 in seeds
            and len(seeds) == len(matching)
            and len(seeds) >= 2
            and all(value is not None for value in values)
            else None
        )
        entries.append({"params": _aggregate_params(key), "score": score})
    return _sort_aggregate(entries)


def _transfer_ranking(phase_path: Path, phase: PhaseManifest) -> list[dict[str, object]]:
    keys = {_aggregate_key(run.params) for run in phase.runs}
    presets = {str(run.params["preset"]) for run in phase.runs}
    per_model_rank: dict[tuple[str, tuple[float, float, float, float | None]], int] = {}
    for preset in presets:
        rows = [
            (result_metric(phase_path.parent, run, phase.metric), run)
            for run in phase.runs
            if run.params["preset"] == preset
        ]
        valid = [(loss, run) for loss, run in rows if loss is not None]
        valid.sort(key=lambda row: (float(row[0]), float(row[1].params["lr"])))
        for rank, (_, run) in enumerate(valid, start=1):
            per_model_rank[(preset, _aggregate_key(run.params))] = rank

    entries: list[dict[str, object]] = []
    for key in keys:
        ranks = [per_model_rank.get((preset, key)) for preset in presets]
        score = sum(int(rank) for rank in ranks) if all(rank is not None for rank in ranks) else None
        params = _aggregate_params(key)
        params.pop("batch_mult", None)
        entries.append({"params": params, "score": score})
    return _sort_aggregate(entries)
```

Dispatch the phase rules in `analyze_phase`:

```python
def analyze_phase(path: Path) -> PhaseManifest:
    path = path.resolve()
    phase = load_phase(path)
    phase.selected = []

    if phase.phase == "replicate":
        phase.ranking = _replicate_ranking(path, phase)
        phase.selected = _select_count(phase.ranking, 2)
    elif phase.phase == "transfer":
        phase.ranking = _transfer_ranking(path, phase)
        phase.selected = _select_count(phase.ranking, 2)
    else:
        phase.ranking = _direct_ranking(path.parent, phase)
        eligible = [entry for entry in phase.ranking if entry["score"] is not None]
        if phase.phase == "smoke":
            scores = {
                float(entry["params"]["lr"]): entry["score"] for entry in phase.ranking
            }
            if scores.get(0.0025) is None or scores.get(0.01) is None:
                raise ValueError("smoke requires finite validation loss at LR 0.0025 and 0.01")
        elif phase.phase == "coarse":
            if eligible:
                by_lr = sorted(eligible, key=lambda entry: float(entry["params"]["lr"]))
                winner_lr = float(phase.ranking[0]["params"]["lr"])
                winner_index = next(
                    index
                    for index, entry in enumerate(by_lr)
                    if float(entry["params"]["lr"]) == winner_lr
                )
                if 0 < winner_index < len(by_lr) - 1:
                    phase.selected = [
                        _candidate(entry["params"], ("lr", "output_mult"))
                        for entry in by_lr[winner_index - 1 : winner_index + 2]
                    ]
        elif phase.phase == "refine":
            ranked = [
                {**entry, "params": _candidate(entry["params"], ("lr", "output_mult"))}
                for entry in phase.ranking
            ]
            phase.selected = _select_count(ranked, 2)
        elif phase.phase == "bridge":
            keys = ("lr", "output_mult", "depth_exponent", "batch_mult")
            ranked = [
                {**entry, "params": _candidate(entry["params"], keys)}
                for entry in phase.ranking
            ]
            phase.selected = _select_count(ranked, 2)
        elif phase.phase == "confirm":
            keys = ("lr", "output_mult", "depth_exponent", "batch_mult")
            ranked = [
                {**entry, "params": _candidate(entry["params"], keys)}
                for entry in phase.ranking
            ]
            phase.selected = _select_count(ranked, 1)
        elif phase.phase == "scale":
            pass
        else:
            raise ValueError(f"unknown μP phase {phase.phase!r}")

    write_phase(path, phase)
    return phase
```

Every ranking entry has `params` and `score`; direct entries also have `id`.
Missing/non-finite results use `score: null`, and exact ties prefer lower LR. A
phase with too few eligible finalists leaves `selected=[]`; the next phase then
refuses to generate without an explicit candidate override. A coarse winner at
either edge of the finite LR region likewise leaves refinement blocked.

- [ ] **Step 4: Add generic analysis and local execution to the CLI**

Register:

```python
@app.command()
def analyze(
    phase_json: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Rank completed cells and update one phase manifest in place."""
    analyze_phase(phase_json)


def _run_local(commands: list[list[str]]) -> None:
    for command in commands:
        subprocess.run(command, check=True)
```

Add `--local/--no-local` defaulting to false to every phase command. After `_generate_phase` returns, local mode must call `_run_local(commands)` and then `analyze_phase(phase_path)`. Remote mode stops after writing the files. Do not capture output, retry, run concurrently, or invoke a shell.

- [ ] **Step 5: Remove dead legacy summary code and tests**

Delete `summarize_sweep`, `MupTransferResult`, and `best_lr_per_width` from
`src/oplm/training/mup.py`; the new analyzer reads callback JSON directly and no
production code should retain the old exact-argmin transfer verdict. Remove the
now-unused `dataclass` import, delete those three names from `__all__`, and make
the module overview list `SweepMetricsCallback` as the sole LR-cell metric
utility. Delete `tests/training/test_mup_sweep.py`. Keep the callback payload
unchanged.

- [ ] **Step 6: Run script, callback, and trainer tests**

Run:

```bash
pytest tests/scripts/test_mup_common.py tests/scripts/test_mup_sweep.py tests/scripts/test_mup_run.py tests/training/test_trainer.py -v
ruff check scripts src/oplm/training/mup.py tests/scripts
```

Expected: all tests pass; missing results remain visible and cannot be selected; local execution stops at the first failure.

- [ ] **Step 7: Commit analysis and local execution**

```bash
git add scripts/_mup_common.py scripts/mup_sweep.py src/oplm/training/mup.py tests/scripts/test_mup_sweep.py tests/training/test_mup_sweep.py
git commit -m "feat: rank and run muP phases sequentially"
```

---

### Task 8: Resolve Coordinate Checks from the Production YAML

**Files:**
- Modify: `scripts/mup_coord_check.py:89-117,269-344`
- Modify: `tests/scripts/test_mup_coord_check_data.py:1-45`

**Interfaces:**
- Consumes: required `--config BASE.yaml`, existing geometry/scaling options, and optional `--data`.
- Produces: coordinate-check models with the base YAML's production architecture and optimizer-independent μP geometry.

- [ ] **Step 1: Write the failing production-feature preservation test**

Append to `tests/scripts/test_mup_coord_check_data.py`:

```python
from scripts._mup_common import Scaling
from scripts.mup_coord_check import _build_cfg_fn


def test_coord_check_resolves_production_model_features(tmp_path: Path) -> None:
    config = tmp_path / "base.yaml"
    config.write_text(
        """
model:
  norm_strategy: sandwich
  canon_enabled: true
  canon_positions: [A, B, C, D]
  residual_gate: channel
  value_residual: learnable
""".lstrip()
    )
    build = _build_cfg_fn(
        config=config,
        depth=24,
        mup=True,
        scaling=Scaling.preset_ray,
        base_width=768,
        output_mult=1.0,
    )

    cfg = build(1280)

    assert cfg.hidden_size == 1280
    assert cfg.num_hidden_layers == 40
    assert cfg.num_attention_heads == 20
    assert cfg.head_dim == 64
    assert cfg.norm_strategy == "sandwich"
    assert cfg.canon_positions == ["A", "B", "C", "D"]
    assert cfg.residual_gate == "channel"
    assert cfg.value_residual == "learnable"
    assert cfg.mup_base_width == 768
```

- [ ] **Step 2: Run the focused test and verify the signature failure**

Run:

```bash
pytest tests/scripts/test_mup_coord_check_data.py::test_coord_check_resolves_production_model_features -v
```

Expected: failure because `_build_cfg_fn` does not accept or load a base config.

- [ ] **Step 3: Load each coordinate-check geometry through `load_config`**

Change `_build_cfg_fn` to accept `config: Path` and build each width with normal YAML resolution:

```python
def _build_cfg_fn(
    *,
    config: Path,
    depth: int,
    mup: bool,
    scaling: Scaling,
    base_width: int,
    output_mult: float,
) -> Callable[[int], OplmConfig]:
    """Make a production-configured model builder for the requested scaling."""
    from oplm.config import load_config

    def build(width: int) -> OplmConfig:
        layers = num_layers_for(width, depth, scaling)
        return load_config(
            [
                "--config",
                str(config),
                f"model.hidden_size={width}",
                f"model.num_hidden_layers={layers}",
                f"model.num_attention_heads={width // HEAD_DIM}",
                f"model.head_dim={HEAD_DIM}",
                f"model.mup_enable={str(mup).lower()}",
                f"model.mup_base_width={base_width}",
                f"model.mup_output_mult={output_mult}",
            ]
        ).model

    return build
```

Make `--config` required in `main`, pass it to `_build_cfg_fn`, and change the CLI `--base-width` default from 512 to 768. Keep the built-in/fixed-sequence path, CSV/PNG outputs, fixed-depth mode, preset-ray mode, and non-μP control unchanged.
Update the module's three command examples to include
`--config configs/mup-production.yaml` so every documented invocation remains
runnable.

- [ ] **Step 4: Run coordinate-check tests**

Run:

```bash
pytest tests/scripts/test_mup_coord_check_data.py tests/training/test_mup_coordcheck.py -v
```

Expected: all coordinate-check tests pass with production features preserved.

- [ ] **Step 5: Commit the coordinate-check update**

```bash
git add scripts/mup_coord_check.py tests/scripts/test_mup_coord_check_data.py
git commit -m "fix: coordinate check production muP configs"
```

---

### Task 9: Align the Runbook and Configuration Documentation

**Files:**
- Modify: `docs/LR_SWEEP.md:44-240,250-510,543-654`
- Modify: `docs/MUP.md:35-55,75-83,160-235`
- Modify: `docs/CONFIG.md:75-85,193-214,237-275`

**Interfaces:**
- Consumes: final CLI help and implemented config fields.
- Produces: one executable operator runbook for local Accelerate and SUNK/Slurm use.

- [ ] **Step 1: Replace obsolete tooling and scope statements in `LR_SWEEP.md`**

Make these exact content changes:

- State that the 800M preset is corrected to 20 heads; remove the warning that it is currently unsafe.
- Keep weight decay fixed at 0.01 and remove the automated `0.003,0.01,0.03` weight-decay stage.
- Remove requirements for padded/masked-token accounting, gradient/update/activation metrics, exact four-batch unions, custom rejection heuristics, retries, and provenance bundles.
- Replace the artifact section with the implemented `phase.json`, `commands.txt`, `runs/<id>/run.yaml`, `result.json`, and `output/` layout.
- Replace exact-argmin transfer language with validation-loss ranking, three-seed means, and summed per-model ranks.
- Keep the production WSD description: warmup, 1–2M stable steps, then 0.5–1M linear decay with optimizer state preserved through the small mixture change.

Add a concise command sequence using one `configs/mup-production.yaml`:

```bash
# Parameterization gates (run μP-on and --no-mup control).
python -m scripts.mup_coord_check --config configs/mup-production.yaml \
  --scaling width --widths 384,768,1536 --depth 24 --base-width 768 \
  --out sweeps/coord-width
python -m scripts.mup_coord_check --config configs/mup-production.yaml \
  --no-mup --scaling width --widths 384,768,1536 --depth 24 \
  --base-width 768 --out sweeps/coord-width-control
python -m scripts.mup_coord_check --config configs/mup-production.yaml \
  --scaling preset_ray --widths 512,768,1024,1280 --base-width 768 \
  --out sweeps/coord-ray

# Generate jobs; add --local to run each phase sequentially on all eight GPUs.
python -m scripts.mup_sweep smoke --config configs/mup-production.yaml \
  --out sweeps/smoke --num-processes 8 --local
python -m scripts.mup_sweep coarse --config configs/mup-production.yaml \
  --from sweeps/smoke/phase.json --out sweeps/coarse --num-processes 8 --local
python -m scripts.mup_sweep refine --config configs/mup-production.yaml \
  --from sweeps/coarse/phase.json --out sweeps/refine --num-processes 8 --local
python -m scripts.mup_sweep replicate --config configs/mup-production.yaml \
  --from sweeps/refine/phase.json --out sweeps/refine-replicate --num-processes 8 --local
python -m scripts.mup_sweep transfer --config configs/mup-production.yaml \
  --from sweeps/refine-replicate/phase.json --out sweeps/transfer --num-processes 8 --local
python -m scripts.mup_sweep bridge --config configs/mup-production.yaml \
  --from sweeps/transfer/phase.json --out sweeps/bridge --num-processes 8 --local
python -m scripts.mup_sweep replicate --config configs/mup-production.yaml \
  --from sweeps/bridge/phase.json --out sweeps/bridge-replicate --num-processes 8 --local
python -m scripts.mup_sweep confirm --config configs/mup-production.yaml \
  --from sweeps/bridge-replicate/phase.json --out sweeps/confirm --num-processes 8 --local
python -m scripts.mup_sweep scale --config configs/mup-production.yaml \
  --from sweeps/confirm/phase.json --out sweeps/scale --num-processes 8 --local
```

Document remote use separately and minimally:

```bash
# Generate without --local, queue each commands.txt line inside its SUNK allocation,
# then rank completed results before generating the next phase.
python -m scripts.mup_sweep analyze sweeps/coarse/phase.json
```

- [ ] **Step 2: Update `MUP.md` and `CONFIG.md`**

In `docs/MUP.md`:

- Retain the width-μP recipe and coordinate-check interpretation.
- Replace old `mup_sweep --widths` and `mup_pilot_run` examples with a link to `LR_SWEEP.md`.
- Distinguish the model-config fallback/default `mup_base_width=512` from the production sweep's forced anchor `768`.
- Document the optional repeated-block formula:

```text
effective_block_lr(L) = width_aware_lr * (mup_depth_reference_layers / L) ** mup_depth_lr_exponent
```

In `docs/CONFIG.md`:

- Change the 800M row to 20 heads and head dimension 64.
- Add these two rows to the training table:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `train.mup_depth_lr_exponent` | `float` | `0.0` | Repeated-block LR exponent; finite and `>= 0`. Zero is a no-op. |
| `train.mup_depth_reference_layers` | `int` | `24` | Reference layer count in `(reference/L)^exponent`; must be `>= 1`. |

- [ ] **Step 3: Check documentation references and formatting**

Run:

```bash
rg -n "mup_pilot_run|--gpus|widths.*lrs|16 heads|head dimension of 80|weight_decay = 0.003|optimizer_groups.json|batch_accounting.json" docs/LR_SWEEP.md docs/MUP.md docs/CONFIG.md scripts tests
git diff --check
```

Expected: no obsolete pilot/concurrency/tooling references remain; any deliberate historical mention is rewritten to avoid presenting it as a runnable command; `git diff --check` is clean.

- [ ] **Step 4: Commit the documentation**

```bash
git add docs/LR_SWEEP.md docs/MUP.md docs/CONFIG.md
git commit -m "docs: document phased production muP sweep"
```

---

### Task 10: Run Full Verification

**Files:**
- Verify only; modify a file only to fix a failure caused by Tasks 1–9.

**Interfaces:**
- Consumes: the complete implementation.
- Produces: fresh evidence that the workflow is formatted, lint-clean, type-clean, and test-clean.

- [ ] **Step 1: Verify the focused μP and script suite**

Run:

```bash
pytest tests/training/test_config.py tests/training/test_mup.py tests/training/test_optim.py tests/training/test_mup_coordcheck.py tests/scripts/test_mup_common.py tests/scripts/test_mup_sweep.py tests/scripts/test_mup_coord_check_data.py -v
```

Expected: all focused tests pass.

- [ ] **Step 2: Verify the real-data one-cell integration**

Run:

```bash
pytest tests/scripts/test_mup_run.py -v
```

Expected: the tiny training run passes and writes finite `eval/heldout/loss`.

- [ ] **Step 3: Run formatting, linting, and type checking**

Run:

```bash
ruff format --check src scripts tests
ruff check src scripts tests
ty check src/
```

Expected: all three commands exit zero with no diagnostics.

- [ ] **Step 4: Run the complete test suite**

Run:

```bash
pytest
```

Expected: all tests pass with zero failures.

- [ ] **Step 5: Verify the final diff and commit history**

Run:

```bash
git diff --check
git status --short
git log --oneline --decorate -10
```

Expected: no whitespace errors; only intentional user-owned untracked paths remain; the implementation is split across the reviewable commits above.

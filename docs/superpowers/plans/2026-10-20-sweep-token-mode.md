# Sweep Token Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the μP sweep (`oplm sweep …` / `oplm.sweep.phases`) generate cells and production
scaling jobs in token mode: a global token budget per optimizer step instead of a global example
count, a per-preset micro-batch token cap, and `train.tokens_per_step` / `train.batch_tokens` in
each run config.

**Architecture:** A new `resolve_token_plan` in `src/oplm/slurm/config.py` mirrors
`resolve_batch_plan`. It turns a global token budget, world size, per-preset cap
(`slurm.max_batch_tokens`) and row width into a per-device `batch_tokens` and an accumulation.
Each cell-generating phase command gains `--global-tokens`. A token cell carries `global_tokens`
in place of `global_examples`, and `_generate_phase` / `_write_run_config` pin the token-mode keys
into `run.yaml`. `scale` gives each preset its own `batch_tokens` when the training config is in
token mode. Sequence-mode sweeps are unchanged.

**Tech Stack:** Python 3.11, Typer, OmegaConf configs, pytest.

**Spec:** No separate spec document. The design was agreed in conversation on 2026-10-20 and is
recorded in **Design** below. It builds on token mode as specified in
`docs/superpowers/specs/2026-10-19-token-batching-flex-attention-design.md` (§5.1:
`train.tokens_per_step`, `train.batch_tokens`, derived accumulation, `data.pack_sequences`
required).

## Design

1. **Why.** μP transfers the learning rate across width only at a fixed batch, sequence length and
   horizon (`docs/MUP.md` §7). The production series now uses one global token budget per step
   (about 4M tokens) at 2048 context. Proxy sweeps must therefore run at that token budget, and
   the harness only knows example counts (`global_examples` → `train.batch_size` + accumulation).
2. **Phase flags.** `smoke`, `coarse`, `refine`, `transfer`, `bridge` and `confirm` gain
   `--global-tokens INT`.
   - `--global-examples` keeps its meaning, but its default moves into the command body, so the
     two flags can be checked for mutual exclusion.
   - Passing both is an error. Passing neither keeps today's default (2048, or 8192 for
     `bridge`/`confirm`).
   - `replicate` copies its source cells and so inherits whichever budget they carry.
3. **Cells.** A cell has exactly one of `global_examples` or `global_tokens`. Run ids are
   unchanged, since a phase uses one budget throughout.
4. **Plan.** `resolve_token_plan(*, global_tokens, world_size, max_batch_tokens, width)`:
   - the budget must be a whole number of `width`-token rows per replica;
   - a device's micro-batch is a whole number of rows, at most `max_batch_tokens`;
   - the smallest accumulation that fits wins;
   - `width` is the base config's `model.max_position_embeddings`.

   Infeasible inputs raise `ValueError`, which every phase command turns into a clean
   `typer.BadParameter`.
5. **Cap table.** The `slurm:` block gains an optional `max_batch_tokens: {preset: int}` table,
   required only when a token cell or token-mode `scale` needs that preset. Sweep cells pin full
   activation checkpointing (`model.gradient_checkpointing_mode=full`), so size the caps under
   full checkpointing.
6. **Run config of a token cell.** It pins:
   - `train.tokens_per_step=<global_tokens>`;
   - `train.batch_tokens=<plan.batch_tokens>`;
   - `train.gradient_accumulation_steps=<plan accumulation>` — the derived value, which the
     trainer accepts; a launch at another world size then fails at startup instead of silently
     training on another budget;
   - `data.pack_sequences=true`.

   `train.batch_size` keeps the base config's value, which in token mode sizes eval batches only.
7. **Audit trail.** `phase.json` records `batch_tokens` and `gradient_accumulation_steps` for token
   cells, and `per_device_batch` and `gradient_accumulation_steps` for sequence cells, as today.
8. **`scale`.** When the training config has `train.tokens_per_step`, each preset gets
   `train.batch_tokens` and `train.gradient_accumulation_steps` from `resolve_token_plan`, using
   that preset's node count, `gpus_per_node`, and `max_batch_tokens` entry. Sequence-mode `scale`
   is unchanged.
9. **Out of scope:**
   - the full-checkpointing pin in sweep cells (it affects speed, not the math);
   - per-cell wall-time estimates in `docs/LR_SWEEP.md`, which stay sequence-based;
   - `coord_check`, which does not use batches.

## Global Constraints

- Sequence-mode sweeps are unchanged: same `run.yaml` keys and values, same `phase.json` params, same `commands.txt`.
- `--global-tokens` and `--global-examples` are mutually exclusive; with neither, a phase uses its existing default example count (2048; 8192 for `bridge` and `confirm`).
- A token cell's `run.yaml` pins `train.tokens_per_step`, `train.batch_tokens`, `train.gradient_accumulation_steps` (the derived value) and `data.pack_sequences=true`, and leaves `train.batch_size` at the base config's value.
- A device's token micro-batch is a whole number of `model.max_position_embeddings`-token rows, at most `slurm.max_batch_tokens[preset]`, with the smallest accumulation that fits.
- Every validation failure is a `ValueError` that the phase commands surface as a clean `typer.BadParameter` (never a raw `KeyError` or traceback).
- Run everything through the repo venv: `.venv/bin/python -m pytest`, `.venv/bin/ruff`, `VIRTUAL_ENV=$PWD/.venv .venv/bin/ty check src/`. `ruff format` only the files you touched; `main` has unrelated drift.
- Commit messages: `type(scope): summary`, ending with a `Co-Authored-By:` trailer naming the model that wrote the commit.

## Review Focus

1. **Both `--global-tokens` and `--global-examples` passed** → a clean CLI error naming both flags. Pinned by `test_global_tokens_and_examples_are_exclusive` (Task 2).
2. **A token budget that isn't whole rows per replica** (for example `--local` with an odd process count) → a clean error naming the budget, width and world size. Pinned by `test_token_budget_must_split_into_whole_rows_per_replica` (Task 2).
3. **A preset missing from `max_batch_tokens` in token mode** → a clean error naming the table and the defined presets, in phases and in `scale`. Pinned by `test_token_cell_without_a_cap_is_a_clean_error` (Task 2) and `test_scale_token_mode_needs_a_cap_for_every_preset` (Task 4).
4. **A cap smaller than one row** → a clean error. Pinned by `test_token_plan_cap_below_one_row_raises` (Task 1).
5. **`replicate` from a token-mode source** → the new seeds keep the token budget. Pinned by `test_replicate_keeps_the_token_budget` (Task 3).

---

### Task 1: `TokenPlan`, `resolve_token_plan` and the `max_batch_tokens` table

**Files:**
- Modify: `src/oplm/slurm/config.py` (imports; `SlurmConfig` fields ~line 134 and `from_mapping` ~line 158; new code after `resolve_batch_plan` ~line 258)
- Modify: `src/oplm/slurm/__init__.py` (export `TokenPlan`, `resolve_token_plan`)
- Test: `tests/slurm/test_config.py`

**Interfaces:**
- Produces:
  - `TokenPlan(batch_tokens: int, gradient_accumulation_steps: int, world_size: int)` (frozen dataclass);
  - `resolve_token_plan(*, global_tokens: int, world_size: int, max_batch_tokens: int, width: int) -> TokenPlan`;
  - `SlurmConfig.max_batch_tokens: dict[str, int]` (default `{}`).

- [ ] **Step 1: Add the cap table to the shared fixture.** In `tests/slurm/test_config.py`, add to `RAW` after `"max_batch_size"`:

```python
    "max_batch_tokens": {"170M": 65536, "400M": 32768, "800M": 16384, "1B": 16384},
```

- [ ] **Step 2: Write the failing tests** (append to `tests/slurm/test_config.py`; add `TokenPlan, resolve_token_plan` to the `oplm.slurm.config` import)

```python
def test_slurm_config_parses_max_batch_tokens() -> None:
    cfg = SlurmConfig.from_mapping(RAW)
    assert cfg.max_batch_tokens == {"170M": 65536, "400M": 32768, "800M": 16384, "1B": 16384}
    without = {key: value for key, value in RAW.items() if key != "max_batch_tokens"}
    assert SlurmConfig.from_mapping(without).max_batch_tokens == {}


def test_slurm_config_rejects_a_bad_max_batch_tokens_entry() -> None:
    with pytest.raises(ValueError, match=r"max_batch_tokens\['170M'\]"):
        SlurmConfig.from_mapping({**RAW, "max_batch_tokens": {"170M": 0}})


@pytest.mark.parametrize(
    ("global_tokens", "world_size", "cap", "width", "expected"),
    [
        # 1M tokens over 8 GPUs at width 1024: 128 rows per replica; a 64-row cap forces accum 2.
        (1_048_576, 8, 65_536, 1024, TokenPlan(65_536, 2, 8)),
        # Over 32 GPUs: 32 rows per replica, under the 32-row cap at accum 1.
        (1_048_576, 32, 32_768, 1024, TokenPlan(32_768, 1, 32)),
        # The production shape: 4M tokens, 2048-token rows, 32 GPUs, a 64k cap -> 64k x 2.
        (4_194_304, 32, 65_536, 2048, TokenPlan(65_536, 2, 32)),
        # A cap that is not a whole number of rows rounds down to whole rows (3.5 rows -> 3; 6 rows
        # per replica split as 3 x 2).
        (98_304, 8, 7_168, 2048, TokenPlan(6_144, 2, 8)),
    ],
)
def test_token_plan_examples(
    global_tokens: int, world_size: int, cap: int, width: int, expected: TokenPlan
) -> None:
    plan = resolve_token_plan(
        global_tokens=global_tokens, world_size=world_size, max_batch_tokens=cap, width=width
    )
    assert plan == expected
    assert plan.batch_tokens * plan.gradient_accumulation_steps * plan.world_size == global_tokens


def test_token_plan_needs_whole_rows_per_replica() -> None:
    with pytest.raises(ValueError, match="whole number of 1024-token rows"):
        resolve_token_plan(global_tokens=1_000_000, world_size=8, max_batch_tokens=65_536, width=1024)
    with pytest.raises(ValueError, match="world size 6"):
        resolve_token_plan(global_tokens=1_048_576, world_size=6, max_batch_tokens=65_536, width=1024)


def test_token_plan_cap_below_one_row_raises() -> None:
    with pytest.raises(ValueError, match="smaller than one 2048-token row"):
        resolve_token_plan(global_tokens=4_194_304, world_size=32, max_batch_tokens=1024, width=2048)


@pytest.mark.parametrize("bad", [0, -1, True])
def test_token_plan_rejects_non_positive_and_bool(bad: int) -> None:
    with pytest.raises(ValueError):
        resolve_token_plan(global_tokens=bad, world_size=8, max_batch_tokens=65_536, width=1024)
```

Check the fourth example by hand before relying on it: 98,304 / (8 × 2048) = 6 rows per replica; the cap 7,168 // 2048 = 3 rows; the smallest accumulation dividing 6 with 6/accum ≤ 3 is 2 → 3 rows = 6,144 tokens. If your arithmetic differs, fix the example, not the function.

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/slurm/test_config.py -k "token or max_batch_tokens" -v`
Expected: FAIL (`ImportError: cannot import name 'TokenPlan'`).

- [ ] **Step 4: Implement** in `src/oplm/slurm/config.py`

Import `field` alongside `dataclass` (`from dataclasses import dataclass, field`).

Factor the per-preset cap parsing out of `from_mapping` so both tables share it:

```python
def _preset_caps(raw: Mapping[str, Any], key: str) -> dict[str, int]:
    """Parse an optional ``{preset: positive int}`` table, naming every bad entry at once."""
    caps: dict[str, int] = {}
    bad: list[str] = []
    for preset, value in dict(raw.get(key, {})).items():
        try:
            caps[str(preset)] = _coerce_positive_int(value, f"slurm config {key}[{preset!r}]")
        except ValueError as exc:
            bad.append(str(exc))
    if bad:
        raise ValueError("; ".join(bad))
    return caps
```

In `SlurmConfig`, add after `nccl_debug: str = "WARN"`:

```python
    # Per-preset cap on a device's token micro-batch, for token-mode sweeps and scaling jobs
    # (`resolve_token_plan`). Optional: only token-mode generation reads it.
    max_batch_tokens: dict[str, int] = field(default_factory=dict)
```

In `from_mapping`, replace the inline `max_batch` loop with `max_batch = _preset_caps(raw, "max_batch_size")` (same messages as before), and add `max_batch_tokens=_preset_caps(raw, "max_batch_tokens"),` to the `cls(...)` call.

After `resolve_batch_plan`:

```python
@dataclass(frozen=True)
class TokenPlan:
    """Per-device token micro-batch and accumulation for one token-mode training job."""

    batch_tokens: int
    gradient_accumulation_steps: int
    world_size: int


def resolve_token_plan(
    *, global_tokens: int, world_size: int, max_batch_tokens: int, width: int
) -> TokenPlan:
    """Derive each device's token micro-batch and the accumulation for a global token budget.

    The token-mode counterpart of :func:`resolve_batch_plan`. A device's micro-batch is a whole
    number of ``width``-token rows (``model.max_position_embeddings``), so the budget must split
    into whole rows per replica. The smallest accumulation that brings a device to at most
    ``max_batch_tokens`` wins. An infeasible combination is an error, never a silently adjusted
    budget.

    Args:
        global_tokens: Token capacity per optimizer step (``train.tokens_per_step``).
        world_size: Total training processes.
        max_batch_tokens: Memory cap on one device's micro-batch, in tokens, for this preset.
        width: Tokens per row (``model.max_position_embeddings``).

    Returns:
        The resolved plan; ``batch_tokens * gradient_accumulation_steps * world_size`` equals
        ``global_tokens``.

    Raises:
        ValueError: A non-positive or ``bool`` argument, a cap smaller than one row, or a budget
            that is not a whole number of rows per replica.
    """
    named = {
        "global_tokens": global_tokens,
        "world_size": world_size,
        "max_batch_tokens": max_batch_tokens,
        "width": width,
    }
    for name, value in named.items():
        _reject_bool(value, f"resolve_token_plan {name}")
    if min(named.values()) < 1:
        raise ValueError(f"resolve_token_plan arguments must all be >= 1; got {named}")
    if max_batch_tokens < width:
        raise ValueError(
            f"max_batch_tokens {max_batch_tokens} is smaller than one {width}-token row"
        )
    if global_tokens % (world_size * width):
        raise ValueError(
            f"global token budget {global_tokens} is not a whole number of {width}-token rows "
            f"per replica at world size {world_size} (it must be a multiple of "
            f"{world_size * width})"
        )
    rows = global_tokens // (world_size * width)
    max_rows = max_batch_tokens // width
    # accum == rows (one row per device) always divides and always fits, since max_rows >= 1.
    accum = next(a for a in range(1, rows + 1) if rows % a == 0 and rows // a <= max_rows)
    return TokenPlan(
        batch_tokens=rows // accum * width,
        gradient_accumulation_steps=accum,
        world_size=world_size,
    )
```

Export both from `src/oplm/slurm/__init__.py` next to `BatchPlan` / `resolve_batch_plan` (imports and `__all__`).

- [ ] **Step 5: Run the slurm and sweep suites**

Run: `.venv/bin/python -m pytest tests/slurm tests/sweep -q`
Expected: all PASS (the new optional table changes nothing for existing configs).

- [ ] **Step 6: Commit**

```bash
.venv/bin/ruff check src/oplm/slurm tests/slurm/test_config.py
.venv/bin/ruff format src/oplm/slurm/config.py src/oplm/slurm/__init__.py tests/slurm/test_config.py
git add src/oplm/slurm/config.py src/oplm/slurm/__init__.py tests/slurm/test_config.py
git commit -m "feat(slurm): resolve_token_plan and a per-preset max_batch_tokens table"
```

---

### Task 2: Token cells in phase generation (`_cell`, `_generate_phase`, `_write_run_config`, `coarse`)

**Files:**
- Modify: `src/oplm/sweep/phases.py` (imports ~line 18; `_cell` ~108; `_write_run_config` ~146; `_resolve_max_batch_size` ~293; `_generate_phase` ~415; `_smoke_cells`/`_coarse_cells` ~504-525; `coarse` command ~1141)
- Test: `tests/sweep/test_phases.py`

**Interfaces:**
- Consumes: `TokenPlan`, `resolve_token_plan`, `SlurmConfig.max_batch_tokens` (Task 1).
- Produces (Task 3 and Task 4 rely on these):
  - `_cell(..., global_examples: int | None = None, global_tokens: int | None = None, ...)` — exactly one must be given;
  - `_batch_budget(global_examples: int | None, global_tokens: int | None, *, default_examples: int) -> dict[str, int]`, returning `{"global_tokens": n}` or `{"global_examples": n}`;
  - `_resolve_max_batch_tokens(slurm: SlurmConfig, preset: str) -> int`;
  - `_smoke_cells(lrs, *, seed, steps, warmup, global_examples=None, global_tokens=None)`, and `_coarse_cells` with the same signature.

- [ ] **Step 1: Write the failing tests** (append to `tests/sweep/test_phases.py`)

```python
def _generate_token_cell(tmp_path: Path, *, extra: list[str] | None = None) -> Path:
    """One coarse cell at a 1M-token budget (170M: 1 node x 8 GPUs, width 1024 by default)."""
    config_path = tmp_path / "base.yaml"
    config_path.write_text(_base_config_text())
    out = tmp_path / "coarse"
    result = runner.invoke(
        phases.app,
        [
            "coarse", "--config", str(config_path), "--out", str(out), "--lrs", "0.0016",
            "--global-tokens", "1048576", "--steps", "1000", "--warmup", "100", *(extra or []),
        ],
    )
    assert result.exit_code == 0, result.output
    return out


def test_token_cell_pins_token_mode_in_run_yaml(tmp_path: Path) -> None:
    out = _generate_token_cell(tmp_path)
    phase = load_phase(out / "phase.json")
    cfg = load_config(["--config", str(out / phase.runs[0].config)])
    # 1,048,576 / (8 GPUs x 1024) = 128 rows per replica; the 170M cap of 65,536 = 64 rows.
    assert cfg.train.tokens_per_step == 1_048_576
    assert cfg.train.batch_tokens == 65_536
    assert cfg.train.gradient_accumulation_steps == 2
    assert cfg.data.pack_sequences is True
    assert cfg.train.batch_size == 32  # the base config's eval batch size, untouched


def test_token_cell_records_its_plan_in_phase_json(tmp_path: Path) -> None:
    run = load_phase(_generate_token_cell(tmp_path) / "phase.json").runs[0]
    assert run.params["global_tokens"] == 1_048_576
    assert "global_examples" not in run.params and "per_device_batch" not in run.params
    assert run.params["batch_tokens"] == 65_536
    assert run.params["gradient_accumulation_steps"] == 2


def test_sequence_cells_carry_no_token_keys(tmp_path: Path) -> None:
    run_yaml = _generate_one_cell(tmp_path)
    cfg = load_config(["--config", str(run_yaml)])
    assert cfg.train.tokens_per_step is None and cfg.train.batch_tokens is None


def test_global_tokens_and_examples_are_exclusive(tmp_path: Path) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text(_base_config_text())
    result = runner.invoke(
        phases.app,
        [
            "coarse", "--config", str(config_path), "--out", str(tmp_path / "c"), "--lrs", "0.0016",
            "--global-tokens", "1048576", "--global-examples", "2048",
        ],
    )
    assert result.exit_code != 0
    assert "--global-examples" in plain(result.output) and "--global-tokens" in plain(result.output)


def test_token_budget_must_split_into_whole_rows_per_replica(base_config: Path, tmp_path: Path) -> None:
    cell = phases._cell(
        preset="170M", lr=0.01, output_mult=1.0, depth_exponent=0.0, seed=42,
        global_tokens=1_048_576, max_steps=1000, warmup_steps=100,
    )
    with pytest.raises(ValueError, match="world size 6"):
        phases._generate_phase(
            name="grid", base_config=base_config, out=tmp_path / "grid", metric=None, source=None,
            cells=[cell], num_processes=6, local=True, accelerate_config=None,
        )


def test_token_cell_without_a_cap_is_a_clean_error(tmp_path: Path) -> None:
    slurm_raw = {key: value for key, value in SLURM_RAW.items() if key != "max_batch_tokens"}
    config_path = tmp_path / "base.yaml"
    config_path.write_text(_base_config_text(slurm_raw))
    result = runner.invoke(
        phases.app,
        [
            "coarse", "--config", str(config_path), "--out", str(tmp_path / "c"), "--lrs", "0.0016",
            "--global-tokens", "1048576",
        ],
    )
    assert result.exit_code != 0
    assert not isinstance(result.exception, KeyError)
    assert "max_batch_tokens" in plain(result.output) and "170M" in plain(result.output)


def test_multinode_token_cell_matches_commands_txt(base_config: Path, tmp_path: Path) -> None:
    cell = phases._cell(
        preset="400M", lr=0.01, output_mult=1.0, depth_exponent=0.0, seed=42,
        global_tokens=1_048_576, max_steps=1000, warmup_steps=100,
    )
    out = tmp_path / "grid"
    phases._generate_phase(
        name="grid", base_config=base_config, out=out, metric=None, source=None,
        cells=[cell], num_processes=8, local=False, accelerate_config=None,
    )
    run = load_phase(out / "phase.json").runs[0]
    cfg = load_config(["--config", str(out / run.config)])
    command = shlex.split((out / "commands.txt").read_text().splitlines()[0])
    num_processes = int(command[command.index("--num_processes") + 1])
    assert num_processes == 32  # 400M: 4 nodes x 8 GPUs
    assert cfg.train.batch_tokens * cfg.train.gradient_accumulation_steps * num_processes == 1_048_576
```

`_base_config_text(slurm_raw)` and `_generate_one_cell` already exist in this file; check the `base_config` fixture name and `plain` import at the top of the file, and use them as the existing tests do. If ruff wants the long argument lists wrapped, let `ruff format` wrap them.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/sweep/test_phases.py -k "token or exclusive" -v`
Expected: FAIL (no such option `--global-tokens`; `_cell()` got an unexpected keyword `global_tokens`).

- [ ] **Step 3: Implement** in `src/oplm/sweep/phases.py`

Imports: `from oplm.slurm.config import BatchPlan, SlurmConfig, TokenPlan, load_slurm_config, resolve_batch_plan, resolve_token_plan`.

`_cell`: make the budget keyword-only and exactly-one.

```python
def _cell(
    *,
    preset: str,
    lr: float,
    output_mult: float,
    depth_exponent: float,
    seed: int,
    max_steps: int,
    warmup_steps: int,
    global_examples: int | None = None,
    global_tokens: int | None = None,
    batch_mult: float | None = None,
    layers: int | None = None,
    lr_mult: float | None = None,
    base_lr: float | None = None,
) -> Params:
    if (global_examples is None) == (global_tokens is None):
        raise ValueError("a sweep cell needs exactly one of global_examples or global_tokens")
    params: Params = {
        "preset": preset,
        "lr": lr,
        "output_mult": output_mult,
        "depth_exponent": depth_exponent,
        "seed": seed,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
    }
    # One budget per cell: global examples (sequence batches) or a token budget (token mode).
    if global_tokens is not None:
        params["global_tokens"] = global_tokens
    else:
        params["global_examples"] = global_examples
    # ... the existing batch_mult / layers / lr_mult / base_lr handling, unchanged ...
    return params
```

A budget helper next to `_cell`:

```python
def _batch_budget(
    global_examples: int | None, global_tokens: int | None, *, default_examples: int
) -> dict[str, int]:
    """A phase's cell budget: ``--global-tokens`` (token mode), else ``--global-examples``.

    The example count's default lives here rather than in the Typer option so the two flags can
    be checked for mutual exclusion.
    """
    if global_examples is not None and global_tokens is not None:
        raise ValueError("pass --global-examples or --global-tokens, not both")
    if global_tokens is not None:
        return {"global_tokens": global_tokens}
    return {"global_examples": default_examples if global_examples is None else global_examples}
```

Generalize the cap lookup so both tables share one helper. Replace `_resolve_max_batch_size` with:

```python
def _resolve_preset_cap(table: dict[str, int], *, table_name: str, preset: str) -> int:
    """Look up a per-preset cap, with an actionable error if the preset is missing.

    Mirrors `_resolve_nodes`: a bare `KeyError` would surface as an uncaught traceback instead of
    the clean `typer.BadParameter` every other validation failure produces.
    """
    try:
        return table[preset]
    except KeyError as exc:
        defined = sorted(table)
        raise ValueError(
            f"preset {preset!r} has no entry in the slurm `{table_name}` table; "
            f"defined presets: {', '.join(defined) if defined else '(none)'}"
        ) from exc


def _resolve_max_batch_size(slurm: SlurmConfig, preset: str) -> int:
    return _resolve_preset_cap(slurm.max_batch_size, table_name="max_batch_size", preset=preset)


def _resolve_max_batch_tokens(slurm: SlurmConfig, preset: str) -> int:
    return _resolve_preset_cap(
        slurm.max_batch_tokens, table_name="max_batch_tokens", preset=preset
    )
```

Keep `_resolve_max_batch_size`'s exact error text: the test `test_missing_preset_in_slurm_tables_is_a_clean_cli_error` and others match on it.

The batch overrides helper, used by `_write_run_config`:

```python
def _batch_overrides(params: Params, plan: BatchPlan | TokenPlan) -> list[str]:
    """Config overrides that pin a cell's batch: sequences per device, or token mode."""
    if isinstance(plan, TokenPlan):
        return [
            f"train.tokens_per_step={_as_int(params['global_tokens'])}",
            f"train.batch_tokens={plan.batch_tokens}",
            # The derived value, which the trainer re-derives and accepts; a launch at another
            # world size then fails at startup instead of training on a different budget.
            f"train.gradient_accumulation_steps={plan.gradient_accumulation_steps}",
            # Token mode packs whole sequences into fixed rows. train.batch_size keeps the base
            # config's value, which in token mode sizes eval batches only.
            "data.pack_sequences=true",
        ]
    return [
        f"train.batch_size={plan.per_device_batch}",
        f"train.gradient_accumulation_steps={plan.gradient_accumulation_steps}",
    ]
```

In `_write_run_config`: change the `plan: BatchPlan` parameter to `plan: BatchPlan | TokenPlan`, and replace the two lines
`f"train.batch_size={plan.per_device_batch}",` / `f"train.gradient_accumulation_steps={plan.gradient_accumulation_steps}",`
with `*_batch_overrides(params, plan),`.

In `_generate_phase`: before the loop, read the row width once,

```python
    width = int(load_config(["--config", str(base_config)]).model.max_position_embeddings)
```

and replace the plan resolution and the `RunSpec` params with:

```python
        plan: BatchPlan | TokenPlan
        if "global_tokens" in params:
            plan = resolve_token_plan(
                global_tokens=_as_int(params["global_tokens"]),
                world_size=world_size,
                max_batch_tokens=_resolve_max_batch_tokens(slurm, preset),
                width=width,
            )
            audit: Params = {"batch_tokens": plan.batch_tokens}
        else:
            plan = resolve_batch_plan(
                global_examples=_as_int(params["global_examples"]),
                world_size=world_size,
                max_batch_size=_resolve_max_batch_size(slurm, preset),
            )
            audit = {"per_device_batch": plan.per_device_batch}
```

with the `RunSpec` params becoming
`{**params, "nodes": nodes, **audit, "gradient_accumulation_steps": plan.gradient_accumulation_steps}`.
Update the `commands.txt` comment there to say "agrees with run.yaml's batch keys". (Keep `nodes`
first and `gradient_accumulation_steps` last so the sequence-mode audit dict is unchanged.)

`_smoke_cells` / `_coarse_cells`: keyword-only budget passed through to `_cell`.

```python
def _smoke_cells(
    lrs: list[float],
    *,
    seed: int,
    steps: int,
    warmup: int,
    global_examples: int | None = None,
    global_tokens: int | None = None,
) -> list[Params]:
    return [
        _cell(
            preset="170M",
            lr=lr,
            output_mult=1.0,
            depth_exponent=0.0,
            seed=seed,
            global_examples=global_examples,
            global_tokens=global_tokens,
            max_steps=steps,
            warmup_steps=warmup,
        )
        for lr in lrs
    ]


def _coarse_cells(
    lrs: list[float],
    *,
    seed: int,
    steps: int,
    warmup: int,
    global_examples: int | None = None,
    global_tokens: int | None = None,
) -> list[Params]:
    return _smoke_cells(
        lrs, seed=seed, steps=steps, warmup=warmup,
        global_examples=global_examples, global_tokens=global_tokens,
    )
```

Update the `smoke` command's call accordingly (keep its `--global-examples 2048` option unchanged
in this task:
`_smoke_cells(parse_floats(lrs, name="--lrs"), global_examples=global_examples, seed=seed, steps=steps, warmup=warmup)`).

`coarse` command: change the option to
`global_examples: Annotated[int | None, typer.Option("--global-examples")] = None,` and add
`global_tokens: Annotated[int | None, typer.Option("--global-tokens")] = None,` right after it.
Build cells with
`_coarse_cells(lr_values, seed=seed, steps=steps, warmup=warmup, **_batch_budget(global_examples, global_tokens, default_examples=2048))`.
`_batch_budget` raises `ValueError`, which the command's existing `except ValueError` turns into
`typer.BadParameter`.

- [ ] **Step 4: Run the sweep suite**

Run: `.venv/bin/python -m pytest tests/sweep tests/slurm -q`
Expected: all PASS (existing sequence-mode tests unchanged).

- [ ] **Step 5: Commit**

```bash
VIRTUAL_ENV=$PWD/.venv .venv/bin/ty check src/oplm/sweep src/oplm/slurm
.venv/bin/ruff check src/oplm/sweep tests/sweep/test_phases.py
.venv/bin/ruff format src/oplm/sweep/phases.py tests/sweep/test_phases.py
git add src/oplm/sweep/phases.py tests/sweep/test_phases.py
git commit -m "feat(sweep): token-mode cells (--global-tokens) for the coarse phase"
```

---

### Task 3: `--global-tokens` on the remaining phases, and `replicate`

**Files:**
- Modify: `src/oplm/sweep/phases.py` (`_refine_cells` ~528, `_transfer_cells` ~573, `_bridge_cells` ~618, `_confirm_cells` ~644; commands `smoke` ~1099, `refine` ~1185, `transfer` ~1279, `bridge` ~1339, `confirm` ~1393)
- Test: `tests/sweep/test_phases.py`

**Interfaces:**
- Consumes: `_cell(..., global_examples=None, global_tokens=None)`, `_batch_budget(...)` and `_smoke_cells(lrs, *, seed, steps, warmup, global_examples=None, global_tokens=None)` (Task 2).
- Produces: every cell-generating command accepts `--global-tokens`; every builder takes keyword-only `global_examples: int | None = None, global_tokens: int | None = None`.

- [ ] **Step 1: Write the failing tests** (append to `tests/sweep/test_phases.py`)

```python
@pytest.mark.parametrize(
    ("command", "extra"),
    [
        ("smoke", []),
        ("refine", ["--candidates", "0.0016:1"]),
        ("transfer", ["--candidates", "0.0016:1"]),
        ("bridge", ["--candidates", "0.0016:1:0"]),
        ("confirm", ["--candidates", "0.0016:1:0:1"]),
    ],
)
def test_every_phase_accepts_a_token_budget(
    tmp_path: Path, command: str, extra: list[str]
) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text(_base_config_text())
    out = tmp_path / command
    args = [command, "--config", str(config_path), "--out", str(out), "--global-tokens", "1048576"]
    if command != "smoke":
        source = _write_selected_phase(tmp_path, "source", [])  # --candidates replaces selection
        args += ["--from", str(source)]
    result = runner.invoke(phases.app, [*args, *extra])
    assert result.exit_code == 0, result.output
    for run in load_phase(out / "phase.json").runs:
        assert run.params["global_tokens"] == 1_048_576
        cfg = load_config(["--config", str(out / run.config)])
        assert cfg.train.tokens_per_step == 1_048_576


def test_replicate_keeps_the_token_budget(tmp_path: Path) -> None:
    coarse_out = _generate_token_cell(tmp_path)
    phase = load_phase(coarse_out / "phase.json")
    candidate = {"lr": phase.runs[0].params["lr"], "output_mult": phase.runs[0].params["output_mult"]}
    write_phase(coarse_out / "phase.json", PhaseManifest(**{**phase.__dict__, "selected": [candidate]}))
    out = tmp_path / "replicate"
    result = runner.invoke(
        phases.app,
        ["replicate", "--config", str(tmp_path / "base.yaml"), "--from", str(coarse_out / "phase.json"),
         "--out", str(out), "--seeds", "42,43"],
    )
    assert result.exit_code == 0, result.output
    run = load_phase(out / "phase.json").runs[0]
    assert run.params["global_tokens"] == 1_048_576 and run.params["seed"] == 43
    cfg = load_config(["--config", str(out / run.config)])
    assert cfg.train.tokens_per_step == 1_048_576 and cfg.train.batch_tokens == 65_536
```

Check `_write_selected_phase`'s signature in this file before using it (the controller saw `_write_selected_phase(tmp_path, "confirm", [...])` used by `test_missing_preset_in_slurm_tables_is_a_clean_cli_error`), and how existing tests pass `--candidates` for each phase (the candidate tuples are `lr:output_mult` for refine/transfer/replicate, `lr:output_mult:depth_exponent` for bridge, and `lr:output_mult:depth_exponent:batch_mult` for confirm). If an existing helper fabricates a selected phase more simply than the `PhaseManifest(**phase.__dict__)` line above, use it.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/sweep/test_phases.py -k "token_budget or replicate_keeps" -v`
Expected: FAIL (no such option `--global-tokens` for smoke/refine/transfer/bridge/confirm). `test_replicate_keeps_the_token_budget` may already pass, since `replicate` copies source params. That's fine: it pins the behaviour.

- [ ] **Step 3: Implement**

The builders `_refine_cells`, `_transfer_cells`, `_bridge_cells` and `_confirm_cells` each take the budget keyword-only and pass both keywords to `_cell`:
- In each signature, replace `global_examples: int` with `global_examples: int | None = None, global_tokens: int | None = None`. Keep them keyword-only: `_refine_cells` is positional today, so give it `*` before `seed`, as in `_smoke_cells`.
- In each `_cell(...)` call, pass `global_examples=global_examples, global_tokens=global_tokens`.
- Existing tests call `_transfer_cells(..., global_examples=2048, ...)`, `_bridge_cells(..., global_examples=8192, ...)` and `_confirm_cells(..., global_examples=8192, ...)` by keyword, so they keep working unchanged.

Commands `smoke`, `refine`, `transfer`, `bridge` and `confirm` each get the same two changes:
- the options become `global_examples: Annotated[int | None, typer.Option("--global-examples")] = None` and `global_tokens: Annotated[int | None, typer.Option("--global-tokens")] = None`;
- the builder receives `**_batch_budget(global_examples, global_tokens, default_examples=D)`, with `D` the command's old default: 2048 for smoke, refine and transfer; 8192 for bridge and confirm.

Example for `refine`:

```python
        cells = _refine_cells(
            _input_candidates(source, candidates, ("lr", "output_mult")),
            parse_floats(output_mults, name="--output-mults"),
            seed=seed,
            steps=steps,
            warmup=warmup,
            **_batch_budget(global_examples, global_tokens, default_examples=2048),
        )
```

The `_batch_budget` call sits inside each command's existing `try:`, so a both-flags error becomes `typer.BadParameter`.

- [ ] **Step 4: Run the sweep suite**

Run: `.venv/bin/python -m pytest tests/sweep tests/slurm -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check src/oplm/sweep tests/sweep/test_phases.py
.venv/bin/ruff format src/oplm/sweep/phases.py tests/sweep/test_phases.py
git add src/oplm/sweep/phases.py tests/sweep/test_phases.py
git commit -m "feat(sweep): --global-tokens on every cell-generating phase"
```

---

### Task 4: Token-mode `scale`

**Files:**
- Modify: `src/oplm/sweep/phases.py` (`scale` command ~1445)
- Test: `tests/sweep/test_phases.py` (or `tests/sweep/test_scale.py`, wherever existing `scale` tests live; check both)

**Interfaces:**
- Consumes: `resolve_token_plan` (Task 1); `_resolve_max_batch_tokens`, `_resolve_nodes` (Task 2 / existing).
- Produces: `scale` writes per-preset `train.batch_tokens` and `train.gradient_accumulation_steps` when the training config has `train.tokens_per_step`.

- [ ] **Step 1: Write the failing tests**

```python
def _token_scale_config(tmp_path: Path, slurm_raw: dict | None = None) -> Path:  # type: ignore[type-arg]
    """A token-mode training config (1M tokens/step, width 1024) with the shared slurm block."""
    path = tmp_path / "scaling.yaml"
    text = _base_config_text(slurm_raw if slurm_raw is not None else SLURM_RAW)
    text = text.replace(
        "train:\n", "train:\n  tokens_per_step: 1048576\n  batch_tokens: 65536\n", 1
    ).replace("data:\n", "data:\n  pack_sequences: true\n", 1)
    path.write_text(text)
    return path


def test_scale_token_mode_resolves_batch_tokens_per_preset(tmp_path: Path) -> None:
    source = _write_selected_phase(
        tmp_path, "confirm",
        [{"lr": 0.004, "output_mult": 1.0, "depth_exponent": 0.5, "batch_mult": 1.0}],
    )
    out = tmp_path / "scale"
    result = runner.invoke(
        phases.app,
        ["scale", "--config", str(_token_scale_config(tmp_path)), "--from", str(source),
         "--out", str(out), "--presets", "170M,400M,1B"],
    )
    assert result.exit_code == 0, result.output
    expected = {  # nodes x 8 GPUs at width 1024 under the shared caps
        "170M": (65_536, 2),  # 8 GPUs: 128 rows per replica, cap 64 rows
        "400M": (32_768, 1),  # 32 GPUs: 32 rows, cap 32 rows
        "1B": (16_384, 1),  # 64 GPUs: 16 rows, cap 16 rows
    }
    for preset, (batch_tokens, accumulation) in expected.items():
        cfg = load_config(["--config", str(out / preset / "run.yaml")])
        assert cfg.train.tokens_per_step == 1_048_576
        assert (cfg.train.batch_tokens, cfg.train.gradient_accumulation_steps) == (
            batch_tokens, accumulation,
        )


def test_scale_token_mode_needs_a_cap_for_every_preset(tmp_path: Path) -> None:
    slurm_raw = {**SLURM_RAW, "max_batch_tokens": {"170M": 65536}}
    source = _write_selected_phase(
        tmp_path, "confirm",
        [{"lr": 0.004, "output_mult": 1.0, "depth_exponent": 0.5, "batch_mult": 1.0}],
    )
    result = runner.invoke(
        phases.app,
        ["scale", "--config", str(_token_scale_config(tmp_path, slurm_raw)), "--from", str(source),
         "--out", str(tmp_path / "scale"), "--presets", "170M,400M"],
    )
    assert result.exit_code != 0
    assert not isinstance(result.exception, KeyError)
    assert "max_batch_tokens" in plain(result.output) and "400M" in plain(result.output)
```

The node counts come from `SLURM_RAW["nodes"]["default"]` (`170M: 1, 400M: 4, 1B: 8`); `scale` resolves nodes with `phase="scale"`, which falls back to `default`. If the `_base_config_text` body does not contain `train:\n` and `data:\n` exactly as the `replace` calls expect, build the YAML the way the existing `scale` tests do instead.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/sweep -k "scale_token" -v`
Expected: FAIL. Without the change, every preset keeps the config's `batch_tokens=65536` and accumulation 1. That loads, but it is wrong for 400M and 1B: at 32 GPUs, 65,536 × 32 is more than 1,048,576, so the trainer would refuse the run at startup.

- [ ] **Step 3: Implement** in `scale`'s per-preset loop

```python
        base = load_config(["--config", str(config)])
        token_mode = base.train.tokens_per_step is not None
        for preset in _parse_strings(presets, name="--presets"):
            run_dir = out / preset
            run_dir.mkdir(parents=True, exist_ok=True)
            nodes = _resolve_nodes(slurm, phase="scale", preset=preset)
            overrides = [
                # ... the existing overrides, unchanged ...
            ]
            if token_mode:
                # One token budget for every preset; each preset's micro-batch and accumulation
                # follow from its node count and memory cap.
                width = int(
                    load_config(["--config", str(config), "--preset", preset]).model.max_position_embeddings
                )
                plan = resolve_token_plan(
                    global_tokens=int(base.train.tokens_per_step),
                    world_size=nodes * slurm.gpus_per_node,
                    max_batch_tokens=_resolve_max_batch_tokens(slurm, preset),
                    width=width,
                )
                overrides += [
                    f"train.batch_tokens={plan.batch_tokens}",
                    f"train.gradient_accumulation_steps={plan.gradient_accumulation_steps}",
                ]
            cfg = load_config(["--config", str(config), "--preset", preset, *overrides])
            # ... unchanged: write run.yaml; JobSpec(nodes=nodes, ...) reusing `nodes` ...
```

Reuse the `nodes` variable in the `JobSpec(...)` call instead of calling `_resolve_nodes` a second time. Add one sentence to the `scale` docstring: in token mode it sets each preset's `train.batch_tokens` and accumulation from the `max_batch_tokens` table.

- [ ] **Step 4: Run the sweep suite**

Run: `.venv/bin/python -m pytest tests/sweep tests/slurm -q`
Expected: all PASS (sequence-mode `scale` tests unchanged).

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check src/oplm/sweep tests/sweep
.venv/bin/ruff format src/oplm/sweep/phases.py <the test file you edited>
git add src/oplm/sweep/phases.py tests/sweep
git commit -m "feat(sweep): token-mode scale sets batch_tokens per preset"
```

---

### Task 5: Docs

**Files:**
- Modify: `docs/LR_SWEEP.md` (new section after "Node counts and derived per-device batch", ~line 340; the funnel paragraph ~line 405; the config example ~line 75)
- Modify: `docs/SLURM.md` (fields table ~line 49; §4 plan derivation ~line 135)

- [ ] **Step 1: `docs/SLURM.md`.** Add a fields-table row after `max_batch_size`:

```markdown
| `max_batch_tokens` | map of preset → int | `{}` | Per-preset memory cap on a device's token micro-batch, consumed by the token-mode plan (`resolve_token_plan`, §4). Only token-mode sweeps and token-mode `scale` read it. |
```

In §4, after the `resolve_batch_plan` description, add:

```markdown
`oplm.slurm.config.resolve_token_plan(*, global_tokens, world_size, max_batch_tokens, width)` is
the token-mode counterpart. A device's micro-batch is a whole number of `width`-token rows
(`model.max_position_embeddings`). The budget must split into whole rows per replica, and the
smallest accumulation that keeps a device at or under `max_batch_tokens` wins. For example,
4,194,304 tokens over 32 GPUs at width 2048 with a 65,536 cap gives `batch_tokens=65536`,
accumulation 2.
```

- [ ] **Step 2: `docs/LR_SWEEP.md`, new section "Token mode"** after "Node counts and derived per-device batch":

```markdown
### Token mode

The production series fixes one token budget per optimizer step for every model size (about 4M
tokens at 2048 context; see TRAIN.md "Token-budget batching"). μP transfers the learning rate only
at a fixed batch and sequence length, so the proxy phases should run at that budget:

- Pass `--global-tokens N` (for example `4194304`) to `smoke`, `coarse`, `refine`, `transfer`,
  `bridge` and `confirm` in place of `--global-examples`. Passing both is an error. `replicate`
  inherits its source cells' budget.
- **Config requirements:**
  - The production YAML must carry the production context (`model.max_position_embeddings`, for
    example 2048), and preferably `model.packed_attention: flex` and `train.compile_dynamic:
    false`.
  - The `slurm:` block needs a `max_batch_tokens` entry for every preset you generate, sized
    under full activation checkpointing (sweep cells pin it). `docs/perf/attention_width_bench.py`
    measures peak memory per shape.
- **What each cell's `run.yaml` pins:**
  - `train.tokens_per_step` and `train.batch_tokens`;
  - the derived `train.gradient_accumulation_steps`;
  - `data.pack_sequences: true`.

  `train.batch_size` keeps the base value, which sizes eval batches only. `phase.json` records
  each cell's `batch_tokens` and accumulation.
- **The bridge phase is optional.** When the proxy phases already run at the production token
  budget, `bridge` has nothing to correct. Skip it and pass the transfer winner straight to
  `confirm` with an explicit batch multiplier of 1:
  `--candidates <lr>:<output_mult>:<depth_exponent>:1`.
- **`scale`:** with a token-mode training config (`train.tokens_per_step` set), it gives each
  preset its own `train.batch_tokens` and accumulation from that preset's node count and
  `max_batch_tokens` entry.
- The wall-time table above is for the sequence-mode defaults.
```

In the "Phase funnel" paragraph that begins "The proxy phases default to 2,048 global examples", append: "In token mode (`--global-tokens`, see "Token mode"), every phase runs at one token budget and the bridge is optional."

In the config example near line 75, add a commented line under `max_batch_size`:

```yaml
  # max_batch_tokens: {170M: 65536, 400M: 32768, 800M: 16384, 1B: 16384}  # token mode only
```

- [ ] **Step 3: Commit**

```bash
git add docs/LR_SWEEP.md docs/SLURM.md
git commit -m "docs: token mode for the μP sweep and the token plan"
```

- [ ] **Step 4: Full fast suite**

Run: `.venv/bin/python -m pytest -m "not slow" -q`
Expected: all PASS.

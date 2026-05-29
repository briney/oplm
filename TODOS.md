# Eval Harness Implementation Plan

Bring `src/oplm/eval/` (and the trainer's eval integration) into alignment with the
design in [`docs/EVAL_HARNESS.md`](docs/EVAL_HARNESS.md). This file is **standalone**:
every change below is specified with the exact file, the current state, and the target
code. You do not need to read any other document to execute it.

## What changes, in one paragraph

Today the trainer passes a bare `global_step: int` to `Evaluator.__call__`, which runs
a task when `global_step % task.eval_every == 0`. We replace this with: a frozen,
**rank-synchronized** `EvalContext` carrying cumulative counters and per-step deltas; a
`Schedule` strategy object per task that fires via a **crossing test** supporting
**steps and tokens** (epochs deferred); an `Evaluator.run_due(ctx, model, accelerator)`
that unwraps the model only when something is due; a `every: {unit: N}` config grammar
replacing `eval_every`; deletion of the duplicated `src/oplm/eval/data/` loaders in
favor of `oplm.data`; and a trainer-side change so `tokens_seen` is the rank-reduced
global token count (required for token cadence not to deadlock). As a prerequisite, we
also rename the rewritten model class everywhere (`OplmForMLM` → `OplmForMaskedLM`,
imported from `oplm.model`) and bring the structure task onto the new forward API
(`output_attentions` / `.attentions`), since the old model and its `eval/` references no
longer exist (Phase 0).

## Ground rules (respect throughout)

- **`oplm.data` must never import `oplm.eval`.** The cadence *spec* type (`ScheduleSpec`)
  and its parser (`parse_schedule_block`) live in `oplm.config` so both the data/config
  layer and the eval layer can use them with no cycle. The cadence *behavior*
  (`Schedule`, `build_schedule`) lives in `oplm.eval`.
- **The rank-sync invariant is load-bearing.** Every `EvalContext` field must be
  identical on every distributed rank. Tasks call collectives (`accelerator.reduce`,
  `dist.all_gather_object`) inside `evaluate`; if ranks disagree on "is this task due,"
  the run deadlocks at the collective. This is why token counts must be rank-reduced.
- **Schedules hold no state** — they are pure functions of `EvalContext`, which makes
  resume work with nothing to persist.
- Style: `from __future__ import annotations` in every file; type hints on all
  signatures; Google-style docstrings; line length 100; `ruff` + `mypy` clean.

## Files touched

| Action | Path |
| --- | --- |
| modify | `src/oplm/config.py` (add `ScheduleSpec`, `parse_schedule_block`; change `EvalDatasetEntry`, `TrainConfig`; reject removed `eval_every`) |
| modify | `src/oplm/data/config.py` (`parse_eval_configs` → `every:` grammar) |
| create | `src/oplm/eval/context.py` (`EvalContext`) |
| create | `src/oplm/eval/schedule.py` (`Schedule`, `_crossed`, `EveryNSteps`, `EveryNTokens`, `build_schedule`) |
| modify | `src/oplm/eval/tasks/base.py` (`EvalTask` holds a `Schedule`; rename `OplmForMLM`) |
| modify | `src/oplm/eval/evaluator.py` (`run_due`, unwrap-behind-due-check, `needs_token_count`; rename `OplmForMLM`) |
| modify | `src/oplm/eval/tasks/sequence.py` (import loader from `oplm.data`; rename `OplmForMLM`) |
| modify | `src/oplm/eval/tasks/structure.py` (import from `oplm.data`; canonical tokenizer; `StructureTaskConfig`; new forward API; rename `OplmForMLM`) |
| modify | `src/oplm/eval/tasks/{proteingym,tape,everest,proteinglue}.py`, `src/oplm/eval/metrics/mlm.py` (rename `OplmForMLM` → `OplmForMaskedLM`, fix import path — Phase 0) |
| delete | `src/oplm/eval/data/` (`__init__.py`, `sequence_loader.py`, `structure_loader.py`) |
| modify | `src/oplm/eval/__init__.py` (public API) |
| modify | `src/oplm/training/trainer.py` (build `EvalContext`; rank-reduce tokens; call `run_due`; rename `OplmForMLM` import/hint — Phase 0) |
| modify | `src/oplm/inference.py`, `src/oplm/cli.py` (rename `OplmForMLM` import/hint — Phase 0) |
| modify | `configs/README.md` (document `every:` + `eval_default_every`) |
| create | `tests/eval/test_*.py` (the harness is currently untested) |

---

## Phase 0 — Rename the model class: `OplmForMLM` → `OplmForMaskedLM`

The model was rewritten: `oplm.model.transformer.OplmForMLM` **no longer exists**. The
masked-LM class is now `OplmForMaskedLM`, exported from `oplm.model` (defined in
`oplm.model.modeling_oplm`). Every `from oplm.model.transformer import OplmForMLM` is a
dead import and every `OplmForMLM` annotation is an undefined name — under `mypy --strict`
(this repo's setting) both fail, so `mypy src/` cannot pass until they are renamed. Do this
first; later phases assume the new name.

- [x] Replace the import everywhere it appears. The class is **not** in
  `oplm.model.transformer` anymore (that module now holds only `OplmBlock` / `OplmStack`);
  import it from the package root:

  ```python
  # was: from oplm.model.transformer import OplmForMLM
  from oplm.model import OplmForMaskedLM
  ```

- [x] Rename every `OplmForMLM` reference (imports **and** type hints) to `OplmForMaskedLM`
  across all of these files (all currently carry the stale name):

  ```
  src/oplm/eval/evaluator.py            (TYPE_CHECKING import + run_due hint)
  src/oplm/eval/tasks/base.py           (TYPE_CHECKING import + evaluate hint)
  src/oplm/eval/tasks/sequence.py       (TYPE_CHECKING import + evaluate hint)
  src/oplm/eval/tasks/structure.py      (TYPE_CHECKING import + two evaluate/_eval hints)
  src/oplm/eval/tasks/proteingym.py     (TYPE_CHECKING import + evaluate hint)
  src/oplm/eval/tasks/tape.py           (TYPE_CHECKING import + evaluate hint)
  src/oplm/eval/tasks/everest.py        (TYPE_CHECKING import + evaluate hint)
  src/oplm/eval/tasks/proteinglue.py    (TYPE_CHECKING import + evaluate hint)
  src/oplm/eval/metrics/mlm.py          (TYPE_CHECKING import + compute_mlm_metrics hint)
  src/oplm/training/trainer.py:40       (local import in __init__)
  src/oplm/inference.py:11              (module-level import + return-type hint at :62)
  src/oplm/cli.py:113                   (local import in inspect command)
  ```

  The code blocks in the phases below already use the new name; this step covers the files
  those phases don't otherwise rewrite (the stub tasks, `metrics/mlm.py`, `inference.py`,
  `cli.py`).

- [x] Confirm none remain: `grep -rn "OplmForMLM" src/` → empty.

- [x] **Scope caveat — model construction is owned by the trainer refactor, not this plan.**
  The *constructor call sites* `OplmForMLM(cfg.model)` (`trainer.py:90`, `inference.py:65`,
  `cli.py:119`) pass `cfg.model`, which is an `oplm.config.ModelConfig` dataclass — **not**
  the `oplm.model.OplmConfig` (`PretrainedConfig`) that `OplmForMaskedLM.__init__` requires.
  The two schemas diverge substantially (e.g. `hidden_dim`/`num_layers`/`num_kv_heads`/
  `conv_*`/`value_residual` vs. `hidden_size`/`num_hidden_layers`/`norm_strategy`/`canon_*`),
  so there is no clean rename and **no conversion helper is to be built** (it would be
  obsoleted the moment the trainer is updated). Renaming the class on these three lines is
  still correct and unbreaks the import, but the `ModelConfig → OplmConfig` mismatch on the
  construction argument is resolved by the **separate trainer-refactor effort**, which will
  converge the trainer/inference/cli onto the HF `OplmConfig`. Consequently:
  - `mypy src/oplm/eval` is a firm gate for **this** plan and must be clean.
  - Full `mypy src/` green (and the live-`Trainer` test, Phase 8.6) is **gated on the
    trainer refactor** updating those construction sites; see Phase 9.

---

## Phase 1 — Config foundation

The cadence spec and parser are the shared contract. Do this first; everything depends
on it.

### 1.1 Add `ScheduleSpec` and `parse_schedule_block` to `src/oplm/config.py`

- [x] Add a module-level constant and the spec dataclass (place near the other
  `_VALID_*` tuples, before `EvalDatasetEntry`):

  ```python
  _VALID_SCHEDULE_UNITS = ("steps", "tokens")  # "epochs" deferred — EVAL_HARNESS.md §4.6
  _SCHEDULE_KEYS = frozenset({*_VALID_SCHEDULE_UNITS, "at_start", "at_end"})


  @dataclass(frozen=True)
  class ScheduleSpec:
      """Parsed, behavior-free eval cadence built from an ``every: {unit: N}`` block.

      Carries no behavior so it can live in ``oplm.config`` without importing
      ``oplm.eval``. The eval harness turns it into a concrete ``Schedule`` via
      ``oplm.eval.schedule.build_schedule``.
      """

      unit: str  # one of _VALID_SCHEDULE_UNITS
      n: int  # positive
      at_start: bool = False
      at_end: bool = True
  ```

- [x] Add the parser (used by both the per-entry `every` and the global default). The
  `epochs` check must precede the generic unknown-key check so its message wins:

  ```python
  def parse_schedule_block(raw: Any, label: str) -> ScheduleSpec:
      """Parse an ``every: {unit: N, at_start?, at_end?}`` mapping into a ScheduleSpec.

      Args:
          raw: The cadence mapping (a dataset's ``every`` or ``train.eval_default_every``).
          label: Human-readable source used in error messages.

      Raises:
          ValueError: If ``raw`` is not a mapping, names ``epochs`` (deferred), does not
              name exactly one valid unit, has unknown keys, the unit value is not a
              positive int, or ``at_start`` / ``at_end`` are not actual bools.
      """
      if not isinstance(raw, dict):
          raise ValueError(
              f"{label}: cadence must be a mapping like {{steps: N}} or {{tokens: N}}, "
              f"got {type(raw).__name__}"
          )
      if "epochs" in raw:
          raise ValueError(
              f"{label}: epoch cadence is not yet supported (see docs/EVAL_HARNESS.md "
              f"§4.6); use {{steps: N}} or {{tokens: N}}"
          )
      unknown = [k for k in raw if k not in _SCHEDULE_KEYS]
      if unknown:
          raise ValueError(f"{label}: unknown keys in cadence block: {sorted(unknown)}")
      unit_keys = [k for k in raw if k in _VALID_SCHEDULE_UNITS]
      if len(unit_keys) != 1:
          raise ValueError(
              f"{label}: cadence must name exactly one of {list(_VALID_SCHEDULE_UNITS)}, "
              f"got {sorted(unit_keys)}"
          )
      unit = unit_keys[0]
      n = raw[unit]
      if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
          raise ValueError(f"{label}: cadence {unit!r} must be a positive int, got {n!r}")
      # The schema says bools; parse them strictly. bool(raw.get(...)) would coerce the
      # YAML string "false" to True, silently inverting the flag — validate the type.
      at_start = raw.get("at_start", False)
      at_end = raw.get("at_end", True)
      for flag_name, flag_value in (("at_start", at_start), ("at_end", at_end)):
          if not isinstance(flag_value, bool):
              raise ValueError(
                  f"{label}: {flag_name!r} must be a bool, got {flag_value!r} "
                  f"({type(flag_value).__name__})"
              )
      return ScheduleSpec(unit=unit, n=n, at_start=at_start, at_end=at_end)
  ```

### 1.2 Change `EvalDatasetEntry` (in `src/oplm/config.py`)

- [x] Replace the `eval_every: int | None = None` field with `schedule: ScheduleSpec`.
  Field order must keep non-default fields before defaulted ones:

  ```python
  @dataclass
  class EvalDatasetEntry:
      """Parsed configuration for a single evaluation dataset.

      Populated by :func:`oplm.data.config.parse_eval_configs`, not directly from YAML.
      """

      name: str
      path: str
      type: str  # registry key: "sequence", "structure", ...
      schedule: ScheduleSpec  # was: eval_every: int | None
      metrics: list[str] | None = None
      extra: dict[str, Any] = field(default_factory=dict)
  ```

### 1.3 Change `TrainConfig` (in `src/oplm/config.py`)

- [x] Remove `eval_every: int = 10_000`.
- [x] Add a structured default mirroring the per-entry grammar (place where
  `eval_every` was, in the logging block):

  ```python
  # Default eval cadence for datasets that omit `every`. Same grammar as a
  # data.eval.<name>.every block: exactly one of {steps, tokens}. Parsed into a
  # ScheduleSpec by the Evaluator via oplm.config.parse_schedule_block.
  eval_default_every: Any = field(default_factory=lambda: {"steps": 10_000})
  ```

  (`Any` + `default_factory` dict matches the existing `DataConfig.eval: Any = None`
  pattern; OmegaConf merges it as a plain mapping.)

### 1.4 Reject the removed `train.eval_every` (in `src/oplm/config.py`)

- [x] Add a rejector next to the existing `_reject_removed_sequence_length_alias`,
  reusing `_lookup_nested_mapping_value` / `_NESTED_VALUE_MISSING`:

  ```python
  def _reject_removed_eval_every_alias(override_dicts: list[Any]) -> None:
      """Reject the removed steps-only ``train.eval_every`` override."""
      present = any(
          _lookup_nested_mapping_value(ov, ("train", "eval_every")) is not _NESTED_VALUE_MISSING
          for ov in override_dicts
      )
      if present:
          raise ValueError(
              "`train.eval_every` has been removed. Use "
              "`train.eval_default_every: {steps: N}` (or {tokens: N}) for the global "
              "default eval cadence."
          )
  ```

- [x] In `load_config`, call it right after the existing sequence-length rejection:

  ```python
  _reject_removed_sequence_length_alias(override_dicts)
  _reject_removed_eval_every_alias(override_dicts)
  ```

### 1.5 Update `parse_eval_configs` (in `src/oplm/data/config.py`)

- [x] Update the import to pull the spec and parser from `oplm.config`:

  ```python
  from oplm.config import EvalDatasetEntry, ScheduleSpec, TrainDatasetEntry, parse_schedule_block
  ```

- [x] Replace `eval_every` with `every` in the known-keys set:

  ```python
  _KNOWN_EVAL_KEYS = frozenset({"path", "type", "every", "metrics"})
  ```

- [x] Change the signature and the per-entry body. New signature:

  ```python
  def parse_eval_configs(raw: Any, default_schedule: ScheduleSpec) -> list[EvalDatasetEntry]:
  ```

  Inside the per-entry loop, **after** the existing `path`/`type` checks and the
  existing nested-`extra` rejection, replace the old `eval_every` handling with:

  ```python
  if "eval_every" in value:
      raise ValueError(
          f"Eval dataset {name!r} uses the removed `eval_every` key. "
          f"Use `every: {{steps: N}}` (or {{tokens: N}})."
      )

  raw_every = value.get("every")
  schedule = (
      parse_schedule_block(raw_every, f"data.eval.{name}.every")
      if raw_every is not None
      else default_schedule
  )

  raw_metrics = value.get("metrics")
  if raw_metrics is None:
      metrics = None
  elif isinstance(raw_metrics, (list, tuple)):
      metrics = [str(m) for m in raw_metrics]
  else:
      # A bare string is iterable, so the naive `[str(m) for m in "loss"]` yields
      # ["l", "o", "s", "s"]. Require an actual list/tuple of metric names; reject
      # strings (and anything else) explicitly. (OmegaConf is already resolved to
      # plain containers here — see the isinstance(value, dict) checks above.)
      raise ValueError(
          f"Eval dataset {name!r}: `metrics` must be a list of names "
          f"(e.g. [loss, accuracy]), got {raw_metrics!r}"
      )
  extra = {k: v for k, v in value.items() if k not in _KNOWN_EVAL_KEYS}

  entries.append(
      EvalDatasetEntry(
          name=str(name),
          path=str(path),
          type=str(eval_type),
          schedule=schedule,
          metrics=metrics,
          extra=extra,
      )
  )
  ```

  Update the function docstring to describe `every` instead of `eval_every`.

---

## Phase 2 — Scheduling primitives

### 2.1 Create `src/oplm/eval/context.py`

- [x] New file:

  ```python
  """Immutable training-state snapshot handed across the trainer↔eval boundary."""

  from __future__ import annotations

  from dataclasses import dataclass


  @dataclass(frozen=True)
  class EvalContext:
      """Synchronized training state for one optimizer step.

      INVARIANT: every field MUST be identical across all distributed ranks, so each
      rank independently computes the same ``Schedule.is_due`` without communicating.
      Tasks run collectives inside ``evaluate``; a rank disagreement would deadlock.
      See docs/EVAL_HARNESS.md §3.2.
      """

      global_step: int  # cumulative optimizer steps completed
      epoch: int  # cumulative epochs (carried for future epoch cadence; unused now)
      tokens_seen: int  # cumulative GLOBAL tokens — rank-reduced, not per-rank
      steps_delta: int  # optimizer steps since the previous context (== 1)
      tokens_delta: int  # GLOBAL tokens processed in this optimizer step (rank-reduced)
      epoch_delta: int  # epochs crossed since the previous context (0/1; future use)
      is_final: bool  # True on the last optimizer step (global_step >= total_steps)
  ```

### 2.2 Create `src/oplm/eval/schedule.py`

- [x] New file. The crossing test reduces to `step % n == 0` for steps (delta 1) and
  fires once when a counter passes a multiple for tokens:

  ```python
  """Eval cadence strategies — pure functions of EvalContext. See EVAL_HARNESS.md §4."""

  from __future__ import annotations

  from dataclasses import dataclass
  from typing import Protocol, runtime_checkable

  from oplm.config import ScheduleSpec
  from oplm.eval.context import EvalContext


  def _crossed(curr: int, delta: int, n: int) -> bool:
      """True iff the half-open interval ``(curr - delta, curr]`` contains a multiple of n."""
      return curr // n > (curr - delta) // n


  @runtime_checkable
  class Schedule(Protocol):
      """Decides, from a synchronized EvalContext, whether a task runs this step."""

      def is_due(self, ctx: EvalContext) -> bool: ...


  @dataclass(frozen=True)
  class EveryNSteps:
      """Fire every ``n`` optimizer steps."""

      n: int
      at_start: bool = False
      at_end: bool = True

      def is_due(self, ctx: EvalContext) -> bool:
          return (
              (self.at_start and ctx.global_step - ctx.steps_delta == 0)
              or (self.at_end and ctx.is_final)
              or _crossed(ctx.global_step, ctx.steps_delta, self.n)
          )


  @dataclass(frozen=True)
  class EveryNTokens:
      """Fire each time cumulative global tokens cross a multiple of ``n``."""

      n: int
      at_start: bool = False
      at_end: bool = True

      def is_due(self, ctx: EvalContext) -> bool:
          return (
              (self.at_start and ctx.tokens_seen - ctx.tokens_delta == 0)
              or (self.at_end and ctx.is_final)
              or _crossed(ctx.tokens_seen, ctx.tokens_delta, self.n)
          )


  _SCHEDULE_BY_UNIT: dict[str, type] = {"steps": EveryNSteps, "tokens": EveryNTokens}


  def build_schedule(spec: ScheduleSpec) -> Schedule:
      """Turn a parsed :class:`ScheduleSpec` into a concrete :class:`Schedule`."""
      cls = _SCHEDULE_BY_UNIT.get(spec.unit)
      if cls is None:
          raise ValueError(
              f"Unsupported schedule unit {spec.unit!r}; supported: {sorted(_SCHEDULE_BY_UNIT)}"
          )
      return cls(n=spec.n, at_start=spec.at_start, at_end=spec.at_end)
  ```

---

## Phase 3 — Evaluator & EvalTask base

### 3.1 Update `src/oplm/eval/tasks/base.py`

- [x] `EvalTask.__init__` builds a `Schedule` from the entry instead of holding
  `eval_every`. Add the import and rewrite the constructor:

  ```python
  from oplm.eval.schedule import build_schedule
  ```

  ```python
  def __init__(self, entry: EvalDatasetEntry, cfg: OplmConfig) -> None:
      self.name = entry.name
      self.path = entry.path
      self.metrics = entry.metrics or self.default_metrics
      self.schedule = build_schedule(entry.schedule)
      self.cfg = cfg
  ```

  Remove the `self.eval_every` line. The abstract `evaluate(self, model, accelerator)`
  signature is unchanged. (Add `Schedule` to the `TYPE_CHECKING` block if you annotate
  `self.schedule`.)

### 3.2 Rewrite `src/oplm/eval/evaluator.py`

- [x] Replace the class body. Construction resolves the global default schedule, parses
  entries, builds tasks, and warns on a provably-unreachable step schedule. `run_due`
  takes an `EvalContext`, returns `{}` cheaply when nothing is due, and unwraps the
  model only when at least one task runs:

  ```python
  """Evaluator orchestrator — the single integration point between Trainer and tasks."""

  from __future__ import annotations

  import logging
  from typing import TYPE_CHECKING

  from oplm.config import parse_schedule_block
  from oplm.data.config import parse_eval_configs
  from oplm.eval.registry import get_eval_task_class
  from oplm.eval.schedule import EveryNSteps, EveryNTokens

  if TYPE_CHECKING:
      from accelerate import Accelerator

      from oplm.config import OplmConfig
      from oplm.eval.context import EvalContext
      from oplm.eval.tasks.base import EvalTask
      from oplm.model import OplmForMaskedLM

  logger = logging.getLogger(__name__)


  class Evaluator:
      """Builds eval tasks from config and runs the ones due at the current step."""

      def __init__(self, cfg: OplmConfig) -> None:
          import oplm.eval.tasks  # noqa: F401  -- triggers task registration

          default_schedule = parse_schedule_block(
              cfg.train.eval_default_every, "train.eval_default_every"
          )
          entries = parse_eval_configs(cfg.data.eval, default_schedule)
          self.tasks: list[EvalTask] = []
          for entry in entries:
              cls = get_eval_task_class(entry.type)
              self.tasks.append(cls(entry, cfg))
              logger.info(
                  "Registered eval task %r (type=%s, schedule=%r)",
                  entry.name,
                  entry.type,
                  entry.schedule,
              )
          self._warn_unreachable(cfg)

      def _warn_unreachable(self, cfg: OplmConfig) -> None:
          """Warn about step schedules that can provably never fire (step-bounded runs)."""
          if cfg.train.max_epochs is not None:
              return  # total steps not known here for epoch-bounded runs
          total_steps = cfg.train.max_steps
          for task in self.tasks:
              sched = task.schedule
              # A schedule with at_start=True fires on the first eval call regardless of
              # cadence, and at_end=True fires on the final step — so it is only provably
              # unreachable when BOTH are off and the cadence exceeds the run length.
              if (
                  isinstance(sched, EveryNSteps)
                  and not sched.at_start
                  and not sched.at_end
                  and sched.n > total_steps
              ):
                  logger.warning(
                      "Eval task %r: step cadence n=%d exceeds max_steps=%d with at_start "
                      "and at_end both false; it will never run.",
                      task.name,
                      sched.n,
                      total_steps,
                  )

      def run_due(
          self, ctx: EvalContext, model: OplmForMaskedLM, accelerator: Accelerator
      ) -> dict[str, float]:
          """Run every task due at ``ctx`` and return merged ``eval/<name>/<metric>`` metrics.

          Returns an empty dict (and does no unwrap / no eval-mode toggle) when nothing
          is due. ``model`` is the WRAPPED model; it is unwrapped here only when needed.
          """
          due = [t for t in self.tasks if t.schedule.is_due(ctx)]
          if not due:
              return {}
          unwrapped = accelerator.unwrap_model(model)
          unwrapped.eval()
          metrics: dict[str, float] = {}
          try:
              for task in due:
                  for key, value in task.evaluate(unwrapped, accelerator).items():
                      metrics[f"eval/{task.name}/{key}"] = value
          finally:
              unwrapped.train()
          return metrics

      @property
      def has_tasks(self) -> bool:
          """Whether any eval tasks are configured."""
          return len(self.tasks) > 0

      @property
      def needs_token_count(self) -> bool:
          """Whether any task uses a token schedule (so the trainer must reduce tokens)."""
          return any(isinstance(t.schedule, EveryNTokens) for t in self.tasks)
  ```

---

## Phase 4 — Migrate tasks to `oplm.data`; delete `eval/data/`

### 4.1 `src/oplm/eval/tasks/sequence.py`

- [ ] Change the loader import from the (to-be-deleted) eval copy to `oplm.data`:

  ```python
  from oplm.data import build_sequence_eval_dataloader  # was: oplm.eval.data.sequence_loader
  ```

- [ ] Remove `_reset_dataloader_state` and its call. `oplm.data`'s
  `build_sequence_eval_dataloader` returns a `_ResettingDataLoader` that rewinds the
  deterministic collator on every `__iter__`, so the manual reset is redundant.
  `evaluate` becomes:

  ```python
  def evaluate(self, model: OplmForMaskedLM, accelerator: Accelerator) -> dict[str, float]:
      if self._dataloader is None:
          self._dataloader = build_sequence_eval_dataloader(self.path, self.cfg)
      all_metrics = compute_mlm_metrics(model, self._dataloader, accelerator)
      return {k: v for k, v in all_metrics.items() if k in self.metrics}
  ```

  Keep the `compute_mlm_metrics` import from `oplm.eval.metrics.mlm` (metrics stay in
  eval). Keep `self._dataloader` lazy init.

### 4.2 `src/oplm/eval/tasks/structure.py`

- [ ] Replace the two top imports:

  ```python
  # remove:
  #   from oplm.data.tokenizer import ProteinTokenizer
  #   from oplm.eval.data.structure_loader import StructureData, load_structures
  # add:
  from oplm.data import StructureData, get_tokenizer, load_structures
  ```

  Keep the `metrics.contact` / `metrics.categorical_jacobian` imports (eval-owned).

- [ ] Add a typed task config and parse `entry.extra` once. Add this dataclass near the
  top of the module (it carries the exact current defaults and the two current
  validations):

  ```python
  @dataclass(frozen=True)
  class StructureTaskConfig:
      """Typed structure-eval knobs, parsed from EvalDatasetEntry.extra."""

      contact_threshold: float = 8.0
      min_seq_sep: int = 6
      l_divisor: int = 1
      use_cbeta: bool = True
      use_logistic_regression: bool = True
      logreg_n_train: int = 20
      logreg_n_iterations: int = 5
      logreg_c: float = 0.15
      use_categorical_jacobian: bool = False
      categorical_jacobian_sample_size: int | None = None
      categorical_jacobian_sample_seed: int = 42
      categorical_jacobian_mutation_batch_size: int = 20
      max_structures: int | None = None

      @classmethod
      def from_extra(cls, extra: dict[str, Any]) -> StructureTaskConfig:
          def _opt_int(key: str) -> int | None:
              return int(extra[key]) if key in extra else None

          def _strict_bool(key: str, default: bool) -> bool:
              # Same rationale as parse_schedule_block: bool("false") is True, so a YAML
              # string would silently invert the flag. Require an actual bool.
              value = extra.get(key, default)
              if not isinstance(value, bool):
                  raise ValueError(
                      f"structure-eval {key!r} must be a bool, got {value!r} "
                      f"({type(value).__name__})"
                  )
              return value

          cfg = cls(
              contact_threshold=float(extra.get("contact_threshold", 8.0)),
              min_seq_sep=int(extra.get("min_seq_sep", 6)),
              l_divisor=int(extra.get("l_divisor", 1)),
              use_cbeta=_strict_bool("use_cbeta", True),
              use_logistic_regression=_strict_bool("use_logistic_regression", True),
              logreg_n_train=int(extra.get("logreg_n_train", 20)),
              logreg_n_iterations=int(extra.get("logreg_n_iterations", 5)),
              logreg_c=float(extra.get("logreg_c", 0.15)),
              use_categorical_jacobian=_strict_bool("use_categorical_jacobian", False),
              categorical_jacobian_sample_size=_opt_int("categorical_jacobian_sample_size"),
              categorical_jacobian_sample_seed=int(
                  extra.get("categorical_jacobian_sample_seed", 42)
              ),
              categorical_jacobian_mutation_batch_size=int(
                  extra.get("categorical_jacobian_mutation_batch_size", 20)
              ),
              max_structures=_opt_int("max_structures"),
          )
          if cfg.categorical_jacobian_sample_size is not None and (
              cfg.categorical_jacobian_sample_size < 1
          ):
              raise ValueError("categorical_jacobian_sample_size must be >= 1 when provided")
          if not 1 <= cfg.categorical_jacobian_mutation_batch_size <= 20:
              raise ValueError("categorical_jacobian_mutation_batch_size must be in [1, 20]")
          return cfg
  ```

- [ ] In `__init__`, replace the block of `self.X = ... extra.get(...)` assignments
  (and the two inline `if` validations) with one line, and update the tokenizer types:

  ```python
  def __init__(self, entry: EvalDatasetEntry, cfg: OplmConfig) -> None:
      super().__init__(entry, cfg)
      self.tcfg = StructureTaskConfig.from_extra(entry.extra)
      if entry.metrics is None and self.tcfg.use_categorical_jacobian:
          self.metrics = ["precision_at_L", "categorical_jacobian_precision_at_L"]
      self._structures: list[StructureData] | None = None
      self._tokenizer: OplmTokenizerFast | None = None
      self._canonical_aa_token_ids: torch.Tensor | None = None
  ```

  Add `from oplm.model import OplmTokenizerFast` under `TYPE_CHECKING` (used only for the
  annotation).

- [ ] Replace every remaining `self.<knob>` reference in the file with `self.tcfg.<knob>`
  (`self.contact_threshold` → `self.tcfg.contact_threshold`, `self.min_seq_sep`,
  `self.l_divisor`, `self.use_cbeta`, `self.use_logistic_regression`,
  `self.logreg_n_train`, `self.logreg_n_iterations`, `self.logreg_c`,
  `self.use_categorical_jacobian`, `self.categorical_jacobian_*`, `self.max_structures`).
  Grep to confirm none remain: `grep -n "self\.\(contact_threshold\|min_seq_sep\|l_divisor\|use_cbeta\|use_logistic_regression\|logreg_\|use_categorical_jacobian\|categorical_jacobian_\|max_structures\)" src/oplm/eval/tasks/structure.py`.

- [ ] In the lazy-init inside `evaluate`, swap the tokenizer constructor:

  ```python
  if self._tokenizer is None:
      self._tokenizer = get_tokenizer()  # was: ProteinTokenizer()
  ```

  Leave `get_canonical_amino_acid_token_ids(self._tokenizer)` and
  `self._tokenizer.encode(struct.sequence)` as-is — `OplmTokenizerFast` supports both
  (`.encode(aa, add_special_tokens=False)` and `.encode(seq)` with default special
  tokens). **Verification (do in Phase 8):** the structure-task test must produce a
  finite `precision_at_L` in `[0, 1]`; if it does not, the special-token handling of
  `OplmTokenizerFast.encode(seq)` differs from the legacy tokenizer — pass
  `add_special_tokens=True` explicitly and confirm the contact extraction still strips
  the `<cls>`/`<eos>` positions.

- [ ] **Update the forward call to the rewritten model's API.** The old model took
  `need_weights=` and returned attention under the `"attention_weights"` key; the rewritten
  `OplmForMaskedLM` takes `output_attentions=` and returns a HuggingFace `MaskedLMOutput`
  whose attention tuple is `.attentions` (key `"attentions"`). Phase 4.2 otherwise only
  swaps the loader/tokenizer/config, so without this the structure task still crashes at the
  forward call. Change both sites:

  ```python
  # in the per-structure forward (was: need_weights=need_attention):
  with torch.no_grad():
      outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          output_attentions=need_attention,
      )

  # in the attention branch (was: raw_attn = outputs["attention_weights"]):
  if need_attention:
      raw_attn = outputs.attentions
      if raw_attn is None:
          raise ValueError("Model did not return attention weights for structure evaluation")
      ...
  ```

  The `outputs["logits"]` accesses elsewhere in the file stay as-is — `MaskedLMOutput` is a
  `ModelOutput`, so both `outputs["logits"]` and `outputs.logits` work. Confirm no
  `need_weights` / `attention_weights` references remain:
  `grep -n "need_weights\|attention_weights" src/oplm/eval/tasks/structure.py` → empty.
  Note the rewritten model emits attentions only via its manual fallback kernel
  (`output_attentions=True` forces it; see docs/MODEL_ARCHITECTURE.md §6.5), so no extra
  config flag is needed at the call site.

- [ ] Optionally update `get_canonical_amino_acid_token_ids`'s parameter annotation in
  `src/oplm/eval/metrics/categorical_jacobian.py` from `ProteinTokenizer` to
  `OplmTokenizerFast` (it only calls `.encode`, so behavior is unchanged).

### 4.3 Delete the duplicated loaders

- [ ] Remove the directory and its three files:

  ```bash
  git rm src/oplm/eval/data/__init__.py \
         src/oplm/eval/data/sequence_loader.py \
         src/oplm/eval/data/structure_loader.py
  ```

- [ ] Confirm nothing else imports them:

  ```bash
  grep -rn "oplm.eval.data" src/ tests/   # expect no results
  ```

---

## Phase 5 — Public API

### 5.1 `src/oplm/eval/__init__.py`

- [ ] Export the new types:

  ```python
  """OPLM evaluation harness. See docs/EVAL_HARNESS.md for the design."""

  from __future__ import annotations

  from oplm.eval.context import EvalContext
  from oplm.eval.evaluator import Evaluator
  from oplm.eval.registry import register_eval_task
  from oplm.eval.schedule import EveryNSteps, EveryNTokens, Schedule
  from oplm.eval.tasks.base import EvalTask

  __all__ = [
      "EvalContext",
      "EvalTask",
      "Evaluator",
      "EveryNSteps",
      "EveryNTokens",
      "Schedule",
      "register_eval_task",
  ]
  ```

---

## Phase 6 — Trainer integration

All edits are in `src/oplm/training/trainer.py`. The contract: build one rank-identical
`EvalContext` per optimizer step and call `run_due` with the WRAPPED model.

### 6.1 Imports and state

- [ ] Add `import torch` to the module-level imports (needed for the token reduction).
- [ ] In `__init__`, add two state fields next to the other training-state inits
  (`self.global_step = 0`, etc.):

  ```python
  self._epoch_at_last_opt_step = 0
  self._step_local_tokens = 0  # local tokens accumulated across the current opt step
  ```

### 6.2 Rank-reduce tokens and build the context in the loop

- [ ] In `train()`, find the per-micro-batch accounting:

  ```python
  tokens_in_batch = batch["attention_mask"].sum().item()
  self.tokens_seen += int(tokens_in_batch) * self.accelerator.num_processes
  self._samples_seen += len(batch["input_ids"]) * self.accelerator.num_processes
  ```

  Replace the first two lines (per-rank token estimate) with a local accumulation; leave
  the samples line:

  ```python
  self._step_local_tokens += int(batch["attention_mask"].sum().item())
  self._samples_seen += len(batch["input_ids"]) * self.accelerator.num_processes
  ```

- [ ] Immediately after `self.global_step += 1` (inside the `sync_gradients` branch),
  reduce the step's tokens across ranks so `tokens_seen` is rank-identical:

  ```python
  tokens_tensor = torch.tensor(
      self._step_local_tokens, device=self.accelerator.device, dtype=torch.long
  )
  tokens_delta = int(self.accelerator.reduce(tokens_tensor, reduction="sum").item())
  self.tokens_seen += tokens_delta
  self._step_local_tokens = 0
  ```

  **Reduce unconditionally — do not gate on `needs_token_count`.** It is tempting to skip the
  all-reduce when no token-cadence task is configured and fall back to
  `self._step_local_tokens * num_processes`, but that per-rank estimate diverges across ranks
  under variable-length / ragged batches and would make `tokens_seen` / `tokens_delta`
  rank-dependent — violating the load-bearing "every `EvalContext` field is rank-identical"
  invariant (design §3.2) on whichever fields the context carries. The reduction is a single
  tiny all-reduce per optimizer step (negligible beside the backward pass), so always perform
  it: `tokens_seen` is then the true global count and rank-identical by construction, and the
  logged `train/tokens` is correct too. (`Evaluator.needs_token_count` is still exported and
  exercised by tests, but the trainer does **not** use it to gate this reduction.)

### 6.3 Build the context and call `run_due`

- [ ] Add the builder method:

  ```python
  def _build_eval_context(self, tokens_delta: int) -> EvalContext:
      """Build a rank-identical EvalContext for the current optimizer step."""
      epoch_delta = self.epoch - self._epoch_at_last_opt_step
      self._epoch_at_last_opt_step = self.epoch
      return EvalContext(
          global_step=self.global_step,
          epoch=self.epoch,
          tokens_seen=self.tokens_seen,
          steps_delta=1,
          tokens_delta=tokens_delta,
          epoch_delta=epoch_delta,
          is_final=(self.global_step >= self.total_steps),
      )
  ```

  Add `from oplm.eval.context import EvalContext` under `TYPE_CHECKING` (the body uses it
  only via the trainer; import it lazily inside the method if you prefer to avoid a
  top-level eval import — `from oplm.eval.context import EvalContext` at the top of
  `_build_eval_context`).

- [ ] Change `_run_eval` to take the step's `tokens_delta`, build the context, and pass
  the WRAPPED model (the evaluator unwraps behind the due-check):

  ```python
  def _run_eval(self, tokens_delta: int) -> dict[str, float]:
      if self.evaluator is None:
          return {}
      ctx = self._build_eval_context(tokens_delta)
      return self.evaluator.run_due(ctx, self.model, self.accelerator)
  ```

  Update its call site in `train()` from `self._run_eval()` to
  `self._run_eval(tokens_delta)`. (Remove the old `unwrap_model` in `_run_eval`.)

### 6.4 Resume

- [ ] In `_resume_from_checkpoint`, after `self.epoch` is restored, reset the snapshot
  markers so the first post-resume step computes correct deltas:

  ```python
  self._epoch_at_last_opt_step = self.epoch
  self._step_local_tokens = 0
  ```

  `tokens_seen` is already restored from `trainer_state.json`; with the half-open
  crossing test (§4.6 of the design) step/token schedules neither re-fire at the resumed
  step nor skip the next multiple.

---

## Phase 7 — Config surface docs

### 7.1 `configs/README.md`

- [ ] Replace the `train.eval_every` config-table row with:

  ```
  | `train.eval_default_every` | `dict` | `{steps: 10000}` | Default eval cadence (`{steps: N}` or `{tokens: N}`) for datasets that omit `every`. | active |
  ```

- [ ] In the Eval Datasets section, replace the per-entry `eval_every` documentation
  with the `every:` grammar and refer to the design doc:

  ```
  Per-dataset cadence (`every`): exactly one of `{steps: N}` or `{tokens: N}`, with
  optional `at_start` (default false) and `at_end` (default true) keys. Datasets that
  omit `every` use `train.eval_default_every`. See docs/EVAL_HARNESS.md §9.

  data:
    eval:
      heldout:
        path: /data/eval_sequences.parquet
        type: sequence
        every: { tokens: 20_000_000 }
      structures:
        path: /data/pdb
        type: structure
        every: { steps: 20_000 }
        use_categorical_jacobian: true
  ```

  Update any other `eval_every` mentions in this file to the new grammar.

- [ ] (Optional) Add a commented `eval:` example block to
  `src/oplm/configs/data/base.yaml` (currently it has no `eval` key; `DataConfig.eval`
  defaults to `null`). Not required for correctness.

---

## Phase 8 — Tests

The eval harness currently has **no tests** (`tests/eval/` holds only `__init__.py`). Add
the suite below. Reuse the existing session-scoped fixtures from `tests/conftest.py`:
`training_parquet` (small real sequences), `structure_fixtures_dir`,
`structure_logreg_fixtures_dir` (skip when absent). Pure scheduling tests need no model
or GPU; task/trainer tests build a tiny model on CPU — mark the model-building ones
`@pytest.mark.slow`.

**Building a model in a test (issue #3 — config types).** `OplmForMaskedLM.__init__`
takes the HuggingFace `oplm.model.OplmConfig` (a `PretrainedConfig`), **not** the
`oplm.config.ModelConfig` dataclass that lives at `cfg.model`. The two schemas diverge
(`hidden_dim`/`num_layers`/`num_kv_heads` vs. `hidden_size`/`num_hidden_layers`/…) and
there is **no conversion helper** (it would be obsoleted by the pending trainer refactor —
Phase 0 caveat). So construct the HF config **directly** with its native field names, as the
existing model tests already do (e.g. `tests/data/test_e2e.py`, `tests/model/test_pilot_train.py`).
Alias the two `OplmConfig` names to avoid the clash:

```python
from oplm.config import OplmConfig          # root run config (data.eval, train, …)
from oplm.model import OplmConfig as OplmModelConfig, OplmForMaskedLM

model = OplmForMaskedLM(
    OplmModelConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
    )
).eval()
```

The eval task / metrics code only *annotates* `model: OplmForMaskedLM` and calls
`model(...)`; it never constructs the model, so it stays mypy-clean. Only the tests (and the
trainer, owned by the separate refactor) construct one.

### 8.1 `tests/eval/test_schedule.py` (pure, no model)

- [ ] Add a small `ctx(...)` helper that builds an `EvalContext` with sensible defaults
  (`steps_delta=1`, others 0/False) so cases read clearly. Cover, per design §4.5:
  - `EveryNSteps(1000)`: due at step 1000 (`delta=1`); not at 999 or 1001.
  - `at_start`: `EveryNSteps(1000, at_start=True)` due at `ctx(1, steps_delta=1)`
    (because `global_step - steps_delta == 0`); the default `at_start=False` is not.
  - `at_end`: `EveryNSteps(1000)` due at `ctx(1500, is_final=True)` though off-cadence.
  - `EveryNTokens(1_000_000)`: due at `ctx(step=10, tokens_seen=1_000_030,
    tokens_delta=80)`; not due the next step `tokens_seen=1_000_110, tokens_delta=80`.
  - `delta > n`: `EveryNTokens(1_000_000)` due exactly once at
    `tokens_seen=2_500_000, tokens_delta=2_500_000`.
  - resume no-refire: a step schedule that fired at 1000 is not due at
    `ctx(1001, steps_delta=1)`.
  - `n > total`: `EveryNSteps(100_000, at_end=False)` never due across steps 1..50_000;
    with `at_end=True` it is due only at `is_final`.
  - `build_schedule(ScheduleSpec("steps", 5))` is `EveryNSteps(5)`; `"tokens"` →
    `EveryNTokens`; an unsupported unit raises `ValueError`.

### 8.2 `tests/eval/test_config.py` (pure, no model)

- [ ] Test `parse_eval_configs` / `parse_schedule_block`:
  - `every: {steps: 2000}` → `ScheduleSpec("steps", 2000, at_start=False, at_end=True)`.
  - `every: {tokens: 1_000_000, at_start: true, at_end: false}` parsed correctly.
  - omitted `every` → the supplied `default_schedule`.
  - two unit keys (`{steps: 1, tokens: 2}`) → `ValueError`.
  - `every: {epochs: 1}` → `ValueError` mentioning "not yet supported".
  - non-positive / bool `n` → `ValueError`.
  - non-bool `at_start` / `at_end` (e.g. `{steps: 10, at_start: "false"}`) → `ValueError`
    mentioning the flag name (issue #5: the string `"false"` must not coerce to `True`).
  - `metrics` as a bare string (e.g. `metrics: "loss"`) → `ValueError` (issue #8: must be a
    list, not split into `["l", "o", "s", "s"]`); a real list `[loss, accuracy]` parses to
    `["loss", "accuracy"]`.
  - removed per-entry `eval_every: 500` → `ValueError` mentioning `every`.
  - unknown task key (`contact_threshold: 8.0`) folds into `EvalDatasetEntry.extra`.
- [ ] Test `load_config(["train.eval_every=500"])` raises `ValueError` mentioning
  `eval_default_every`; and `load_config(["train.eval_default_every={steps: 7}"])`
  resolves (no error).

### 8.3 `tests/eval/test_evaluator.py` (no model — use a dummy task)

- [ ] Register a dummy task type for the test (in the test module, via
  `@register_eval_task("dummy")`) whose `evaluate` returns a fixed dict like
  `{"score": 1.0}`. Build an `OplmConfig` (defaults are fine) with
  `data.eval = {"d": {"path": "x", "type": "dummy", "every": {"steps": 10}}}` and a
  dummy `train.eval_default_every`. Assert:
  - `run_due(ctx(step=10, steps_delta=1), model=None, accelerator=None)` is **not**
    reached for `model=None` only if due — so instead assert the not-due path returns
    `{}` without touching the model: `run_due(ctx(step=9), None, None) == {}`.
  - For the due path, pass a tiny stub object with `.eval()/.train()` and a fake
    `accelerator` whose `unwrap_model(m)` returns `m`; assert the returned dict is
    `{"eval/d/score": 1.0}` (namespacing works).
  - `Evaluator(cfg).needs_token_count` is `True` when an entry uses `every: {tokens: N}`
    and `False` otherwise.
  - Construct with `every: {steps: 10**9, at_end: false}` and `max_steps` small; assert a
    warning is logged (use `caplog`).

### 8.4 `tests/eval/test_sequence_task.py`

- [ ] `@pytest.mark.slow`. Build **two** configs (see the Phase 8 intro): a root
  `oplm.config.OplmConfig` for the task (`cfg`, providing `data`/`train` so
  `build_sequence_eval_dataloader(self.path, cfg)` can tokenize/mask), and the model from the
  HF `OplmModelConfig` directly — **not** `OplmForMaskedLM(cfg.model)`:

  ```python
  model = OplmForMaskedLM(
      OplmModelConfig(
          hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
          max_position_embeddings=64,  # >= the data cfg's max_seq_len
          # vocab_size defaults to the tokenizer's 33 — keep it consistent with the data cfg
      )
  ).eval()
  ```

  Construct a `SequenceEvalTask` from an `EvalDatasetEntry(name="hd",
  path=str(training_parquet), type="sequence", schedule=ScheduleSpec("steps", 1))` with the
  root `cfg`. Build a single-process `Accelerator` (use the `_reset_accelerator_state` autouse
  fixture already in `tests/conftest.py`). Assert `evaluate` returns `loss`, `accuracy`,
  `perplexity`, all finite, with `0 <= accuracy <= 1`. Keep `max_position_embeddings` and
  `vocab_size` consistent with the data cfg's `max_seq_len` / tokenizer so the dataloader's
  token ids index the model embedding correctly.

### 8.5 `tests/eval/test_structure_task.py`

- [ ] `@pytest.mark.slow`. Use `structure_fixtures_dir` (skips if absent). Build the model
  from `OplmModelConfig` directly as in 8.4, but with `max_position_embeddings` large enough
  for the fixtures. Construct a `StructureEvalTask` with `type="structure"`,
  `schedule=ScheduleSpec("steps", 1)`, and `extra={"max_structures": 2,
  "use_logistic_regression": False}` (a real Python bool — `from_extra` now rejects the
  string `"false"`; use the mean-attention path so it runs with few structures). Assert
  `precision_at_L` is present and in `[0, 1]`. This exercises **both** migration gates: the
  tokenizer swap (§4.2) **and** the new `output_attentions` / `.attentions` forward API — if
  the forward call were left on the old `need_weights` / `"attention_weights"` API the task
  would raise here.
- [ ] Add a **non-slow** unit test (no model/fixtures) for `StructureTaskConfig.from_extra`
  (issue #5): a string boolean such as `{"use_logistic_regression": "false"}` raises
  `ValueError` naming the key (must not coerce to `True`), real bools parse, and the existing
  numeric validations (`categorical_jacobian_sample_size >= 1`,
  `categorical_jacobian_mutation_batch_size` in `[1, 20]`) still raise on bad values.

### 8.6 `tests/eval/test_trainer_integration.py`

> **Dependency — gated on the trainer refactor (Phase 0 caveat).** This is the only test
> that runs the live `Trainer`, which builds the model via `OplmForMaskedLM(cfg.model)` at
> `trainer.py:90`. That call is broken until the **separate trainer-refactor effort** converges
> the trainer onto the HF `OplmConfig` (this plan deliberately does not build a
> `ModelConfig → OplmConfig` converter). So write this test now, but mark it
> `@pytest.mark.skip(reason="enable once the trainer constructs OplmForMaskedLM from the HF
> OplmConfig — see Phase 0")` (or `xfail`), and flip it on when the trainer refactor lands.
> The eval-integration code in Phase 6 (`_build_eval_context`, rank-reduced tokens,
> `run_due`) is still exercised in isolation by 8.3 (evaluator) and 8.8 (token accounting),
> so this gating does not leave Phase 6 untested — it only defers the full end-to-end run.

- [ ] `@pytest.mark.slow`. End-to-end: tiny `OplmConfig` with `train.max_steps=4`,
  `train.wandb_enabled=False`, `train.batch_size` small, `data.train=str(training_parquet)`,
  and
  ```python
  data.eval = {"hd": {"path": str(training_parquet), "type": "sequence",
                       "every": {"steps": 2}}}
  ```
  Attach a `TrainerCallback` subclass that records `on_eval_end(trainer, metrics, step)`
  calls. Run `Trainer(cfg).train()`. Assert it completes without error and that at least
  one recorded metrics dict contains a key starting with `eval/hd/` (i.e. eval actually
  fired on the step cadence). Add a second case with `every: {tokens: 1}` and assert eval
  fires every step (token cadence path + the rank-reduction code runs).

### 8.7 `tests/eval/test_registry.py` (pure, no model)

The design (§10) calls for explicit registry coverage; add it (the current plan only used the
registry implicitly via the dummy task in 8.3).

- [ ] Duplicate registration raises: calling `@register_eval_task("sequence")` a second time
  (e.g. on a throwaway subclass) raises `ValueError`/`KeyError` naming the duplicate type.
- [ ] Unknown type raises with the known list: `get_eval_task_class("does_not_exist")` raises
  and the message includes the registered type names (so the error is actionable).
- [ ] Sanity: each real registered type (`sequence`, `structure`, `proteingym`, `tape`,
  `proteinglue`, `everest`) resolves via `get_eval_task_class` to an `EvalTask` subclass after
  `import oplm.eval.tasks`.
- [ ] Use an isolated registry where possible (or register/clean up a uniquely-named dummy in a
  fixture) so the test does not mutate the shared `EVAL_TASK_REGISTRY` for other tests.

### 8.8 `tests/eval/test_token_accounting.py`

The design (§10) calls for token-accounting and rank-sync coverage; this is the source-of-truth
guarantee behind token cadence (no deadlock). Add both a focused single-process check and, if
the environment supports spawning, a small ragged distributed smoke test.

- [ ] **Focused token accounting (single process, no GPU).** Drive the trainer's token
  reduction (or a small extracted helper mirroring Phase 6.2) over a few synthetic batches with
  *known, varying* `attention_mask` sums; assert the accumulated `tokens_seen` equals the exact
  sum of all `attention_mask` ones (the true global count) — i.e. **not** a
  `local_tokens × num_processes` estimate. With `num_processes == 1` the reduction is identity,
  so this nails the per-step accumulation + reset logic and that `_step_local_tokens` zeroes
  each optimizer step.
- [ ] **Rank-sync purity (no model).** Given one `EvalContext`, every task's `is_due` is a pure
  function of it, so all ranks agree by construction. Assert this directly: build a context and
  check `task.schedule.is_due(ctx)` is deterministic and that two independently-built contexts
  with identical fields yield identical `is_due` for an `EveryNTokens` schedule.
- [ ] **Ragged distributed smoke (mark `@pytest.mark.slow`; skip if multi-proc launch is
  unavailable).** Launch ≥2 ranks (e.g. `accelerate`/`torch.distributed` with `gloo`) feeding
  **ragged** per-rank token counts, run several optimizer steps with a token-cadence eval
  configured, and assert (a) every rank computes the *same* `tokens_seen`/`tokens_delta` after
  the reduction, and (b) the token-cadence eval does **not** hang (all ranks reach the same
  `is_due` and the collective inside `evaluate` completes). This is the direct regression test
  for the §3.2 invariant and the reason the reduction is unconditional (issue #6). If the CI
  environment cannot spawn workers, document the skip clearly (do not silently drop coverage).

---

## Phase 9 — Verification & quality gates

- [ ] No references to the deleted module remain:
  `grep -rn "oplm.eval.data" src/ tests/` → empty.
- [ ] No `OplmForMLM` references remain (Phase 0):
  `grep -rn "OplmForMLM" src/ tests/` → empty (all renamed to `OplmForMaskedLM`).
- [ ] Structure task is off the old forward API:
  `grep -rn "need_weights\|attention_weights" src/oplm/eval/` → empty.
- [ ] No stray `eval_every` remains except the two rejection messages:
  `grep -rn "eval_every" src/` → only `_reject_removed_eval_every_alias`, the
  `parse_eval_configs` rejection, and `eval_default_every`.
- [ ] `python -c "import oplm.eval; print(oplm.eval.__all__)"` lists the new exports.
- [ ] `python -c "from oplm.config import load_config; load_config([])"` works (default
  `eval_default_every` parses), and `load_config(['train.eval_every=1'])` raises.
- [ ] Lint/format: `ruff check src/ tests/`, `ruff format src/ tests/`.
- [ ] Type — **firm gate for this plan:** `mypy src/oplm/eval` clean. **Broader gate, partial:**
  `mypy src/` will still report the `OplmForMaskedLM(cfg.model)` argument-type mismatch in
  `trainer.py` / `inference.py` / `cli.py` (`ModelConfig` vs HF `OplmConfig`) — that is the
  **trainer-refactor's** responsibility (Phase 0 caveat), not this plan's. After Phase 0 the
  only remaining `mypy src/` errors should be those construction-site mismatches; confirm there
  are no *new* eval-introduced errors and no surviving stale-import errors.
- [ ] Fast tests: `pytest -m "not slow"` green (this includes the new pure tests: schedule,
  config, evaluator, registry, and the single-process token-accounting/rank-sync checks).
- [ ] Full tests (needs the structure/sequence fixtures present):
  `pytest tests/eval` green; the structure test confirms **both** the tokenizer migration and
  the new `output_attentions` forward API. The live-`Trainer` test (8.6) stays skipped until
  the trainer refactor lands.
- [ ] Sanity-check the design alignment: steps + tokens cadence only (epochs rejected);
  `at_end` defaults true; `at_start`/`at_end` parsed as strict bools; `metrics` rejects bare
  strings; `_warn_unreachable` honors `at_start`; `run_due` unwraps only when due;
  `tokens_seen` reduced **unconditionally** (rank-identical); model class is `OplmForMaskedLM`;
  structure task uses `output_attentions` / `.attentions`; `eval/data/` gone.

---

## Notes / known follow-ups (not blocking)

- **Epoch cadence is deferred** (design §4.6). `EvalContext` already carries `epoch` /
  `epoch_delta`, and `parse_schedule_block` rejects `epochs` with a clear message. Adding
  it later is: a new `EveryNEpochs` class
  (`_crossed(ctx.epoch, ctx.epoch_delta, n)`), `"epochs"` in `_VALID_SCHEDULE_UNITS` and
  `_SCHEDULE_BY_UNIT`, and removing the rejection — no interface change.
- **`parse_eval_configs` second arg changed** from `default_eval_every: int` to
  `default_schedule: ScheduleSpec`. The only caller is the `Evaluator`; update it (done
  in Phase 3.2). `parse_eval_configs` is re-exported from `oplm.data` — keep the export.
- **Stub tasks** (`proteingym`, `tape`, `proteinglue`, `everest`) inherit the new
  `EvalTask.__init__`, so they need no *behavioral* changes and still raise
  `NotImplementedError` — but they **do** need the Phase 0 rename (each has a stale
  `from oplm.model.transformer import OplmForMLM` import and an `OplmForMLM` hint on its
  `evaluate` stub), or `mypy src/oplm/eval` fails on the undefined name.

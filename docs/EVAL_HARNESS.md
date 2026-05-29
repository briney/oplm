# OPLM Evaluation Harness

> Founding reference for the OPLM evaluation harness — how a model is scored
> during and after pretraining across sequence, structure, variant, and
> downstream tasks. This document specifies the **target** design: the task
> registry, the per-task interface, the trainer↔eval scheduling contract, metric
> ownership, and the `data.eval:` configuration schema, in enough detail to
> implement directly. It is the sole source of truth for the eval harness design.
> The data loaders themselves are out of scope and live in
> [`DATA_TOOLING.md`](DATA_TOOLING.md); the model and its forward pass live in
> [`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md).

---

## 1. Scope and design philosophy

### 1.1 What this document covers

- The modules under `src/oplm/eval/` — the task registry, the `EvalTask`
  interface, the `Evaluator` orchestrator, the `Schedule` strategy types, and the
  eval-specific metric code.
- The **trainer↔eval contract**: the `EvalContext` passed across the boundary, the
  `run_due` call, and the synchronized-state invariant that keeps distributed eval
  from deadlocking.
- **Scheduling**: how each eval dataset runs on its own cadence (every N steps or
  every N tokens), including the exact firing semantics and their guarantees on
  resume.
- The `data.eval:` block of `OplmConfig` — the contract between YAML and the
  harness — and the `every: {…}` cadence grammar.
- Concise definitions of the metrics the harness computes (MLM perplexity, contact
  precision@L via the categorical Jacobian, and the ranking metrics for variant
  tasks).
- The implemented task types (sequence, structure) and the registered stubs
  (ProteinGym, TAPE, ProteinGlue, EVEREST).

### 1.2 What this document does not cover

- The data loaders, on-disk formats, tokenizer, and masking strategy. →
  [`DATA_TOOLING.md`](DATA_TOOLING.md) (the harness *consumes* `oplm.data`; it does
  not define loaders).
- The model, its forward signature, and how logits are exposed (the categorical
  Jacobian probes them for contact prediction). →
  [`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md).
- The trainer loop, optimizer, schedules, FLOPs accounting, and checkpoint format.
  → future `docs/TRAINER.md`. This document specifies only the *interface* the
  trainer must satisfy to drive eval (§6).
- The CLI. → future `docs/CLI.md`.

### 1.3 Design principles

1. **The trainer owns the clock; the harness owns the policy.** The trainer
   advances training state and announces "we are at step N / token M"; the harness
   decides what that means — which datasets are due, what they compute. Per-task
   cadence never leaks into the training loop.

2. **Scheduling is a strategy object, not a branch.** "When to run" and "what to
   compute" are orthogonal. A task owns a `Schedule`; the `Evaluator` never grows a
   conditional per cadence type. New cadence units are new `Schedule` classes, not
   edits to the orchestrator.

3. **Schedules are pure functions of synchronized state.** A `Schedule` decides
   from an immutable `EvalContext` and nothing else — no internal counters. This
   makes scheduling **resume-safe with no state to persist** and is the foundation
   of the rank-sync invariant (§3.2).

4. **The rank-sync invariant is load-bearing, not cosmetic.** Every field of
   `EvalContext` is identical across distributed ranks, so all ranks reach the same
   "is due" decision without communicating. Tasks run collective ops
   (`accelerator.reduce`, `all_gather_object`) inside `evaluate`; a rank
   disagreement would hang at the collective. See §3.2.

5. **Eval owns scoring; data owns loading.** The harness imports `oplm.data` for
   every byte it reads and never re-implements a loader. Metric computation
   (perplexity, P@L, APC, categorical Jacobian) lives in `eval/metrics/` because it
   is eval-specific scoring, not shared data machinery. This mirrors the train/eval
   boundary in [`DATA_TOOLING.md`](DATA_TOOLING.md) §2.

6. **Typed config over loose dicts.** Task-specific knobs arrive as a `dict` at the
   YAML boundary, but each task converts them to an internal frozen dataclass via a
   `from_extra()` classmethod — one place to cast and validate, no scattered
   `dict.get` calls in `evaluate`.

### 1.4 Status: target design vs. today's code

This is a **forward-looking design**. The current contents of `src/oplm/eval/` are
an earlier implementation — clean and conforming, but written before the model,
config system, and data tooling were rewritten — retained as reference and slated
for a rewrite to match this document. §11 records the mapping. The load-bearing
divergences that motivate the rewrite:

- The interface is a bare `global_step: int`; cadence is `global_step % eval_every`
  on optimizer steps only. → generalized to an `EvalContext` and unit-agnostic
  `Schedule`s (steps **and tokens**) in §3–§4.
- `src/oplm/eval/data/` re-implements sequence and structure loaders that now live
  in `oplm.data`. → deleted; tasks import `oplm.data` (§2.1, §11).
- The structure task instantiates the removed `ProteinTokenizer`. → uses the
  canonical tokenizer accessor `oplm.data.get_tokenizer()` (§7.3, §11).
- The trainer accumulates `tokens_seen` per-rank, so the value differs across ranks
  — incompatible with token-based scheduling. → replaced by a rank-reduced count
  (§6.2).

**First cut: steps and tokens.** Both cadences are *exact* and resume-safe. **Epoch
cadence is deferred** to a later cut: under unbalanced shards and mid-epoch
checkpoints it cannot give the same exactness guarantees (§4.6). The `Schedule`
abstraction and `EvalContext` are designed so `EveryNEpochs` slots in later with no
interface change.

---

## 2. Architecture overview

The trainer builds one immutable `EvalContext` per optimizer step and hands it to a
single `Evaluator`. The `Evaluator` asks each task's `Schedule` whether it is due,
runs the due tasks, and returns a flat metrics dict. Tasks read their data through
`oplm.data` and score it with `eval/metrics/`.

```
┌────────────────────────────────────────────────────────────────────────┐
│ Trainer  (owns the clock)                                                │
│   builds EvalContext{step, tokens, …}  ──►  evaluator.run_due(ctx, …)    │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │  EvalContext (frozen, rank-identical)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Evaluator  (owns the policy)                                             │
│   due = [t for t in tasks if t.schedule.is_due(ctx)]                     │
│   if none: return {}     ── else unwrap model, model.eval(), run, train()│
└───────────────┬───────────────────────────────────┬──────────────────────┘
                │ Schedule (strategy)                │ EvalTask (registry)
                ▼                                     ▼
   EveryNSteps / EveryNTokens            SequenceEvalTask  StructureEvalTask  …
   is_due(ctx) -> bool  (pure)           evaluate(model, accelerator) -> dict
                                                      │
                                                      ▼ scores with        ▼ reads via
                                            src/oplm/eval/metrics/   oplm.data.*
```

### 2.1 Target module tree

There is **no** `src/oplm/eval/data/` directory; the harness imports `oplm.data`.

```
src/oplm/eval/
  __init__.py        # public API: Evaluator, EvalTask, register_eval_task,
                     #             EvalContext, Schedule, EveryNSteps, EveryNTokens
  context.py         # EvalContext (frozen dataclass)
  schedule.py        # Schedule protocol + EveryNSteps, EveryNTokens
                     #             (_crossed helper; EveryNEpochs deferred — §4.6)
  evaluator.py       # Evaluator.run_due orchestrator
  registry.py        # @register_eval_task + get_eval_task_class + EVAL_TASK_REGISTRY
  tasks/
    __init__.py      # imports task modules to trigger registration
    base.py          # EvalTask ABC (holds a Schedule + a typed task config)
    sequence.py      # SequenceEvalTask   → oplm.data.build_sequence_eval_dataloader
    structure.py     # StructureEvalTask  → oplm.data.load_structures
    proteingym.py    # stub → oplm.data.load_variant_assays
    tape.py          # stub
    proteinglue.py   # stub
    everest.py       # stub → oplm.data.load_variant_assays
  metrics/
    __init__.py
    mlm.py                  # MLM loss / accuracy / perplexity
    contact.py              # contact map, APC, precision@L primitives
    categorical_jacobian.py # categorical-Jacobian coupling → precision@L
```

| Component | Responsibility |
| --- | --- |
| `EvalContext` | Immutable, rank-identical snapshot of training state for one optimizer step (§3). |
| `Schedule` | Pure `is_due(ctx) -> bool` policy; one per task (§4). |
| `Evaluator` | Builds tasks from config, runs the due ones, namespaces metrics (§5). |
| `EvalTask` | One per configured dataset: loads (via `oplm.data`), computes metrics, returns bare metric names (§7). |
| registry | Maps a `type` string to an `EvalTask` subclass (§7.2). |
| `eval/metrics/` | Eval-specific scoring math (§8). |

---

## 3. The trainer↔eval contract: `EvalContext`

The single value crossing the boundary is a frozen `EvalContext`. It carries
**cumulative counters** (for absolute-position decisions) and **per-step deltas**
(for crossing detection, §4.2), plus lifecycle flags.

```python
@dataclass(frozen=True)
class EvalContext:
    """Synchronized training state for one optimizer step.

    Every field MUST be identical across all distributed ranks (see §3.2).
    Schedules are pure functions of this object and nothing else.
    """
    global_step: int    # cumulative optimizer steps completed
    epoch: int          # cumulative epochs (carried for future epoch cadence; §4.6)
    tokens_seen: int    # cumulative GLOBAL tokens — rank-reduced, not per-rank (§6.2)
    steps_delta: int    # optimizer steps since the previous EvalContext (== 1)
    tokens_delta: int   # GLOBAL tokens processed in this optimizer step (rank-reduced)
    epoch_delta: int    # epochs crossed since the previous EvalContext (0 or 1; future use)
    is_final: bool      # True on the last optimizer step (global_step >= total_steps)
```

| Field | Kind | Rank-identical? | Meaning |
| --- | --- | --- | --- |
| `global_step` | cumulative | yes (synced counter) | optimizer steps completed |
| `epoch` | cumulative | yes under balanced shards (§4.6) | epochs completed |
| `tokens_seen` | cumulative | **only after the §6.2 reduction** | global tokens consumed |
| `steps_delta` | delta | yes (always 1) | steps since last context |
| `tokens_delta` | delta | **only after the §6.2 reduction** | global tokens this step |
| `epoch_delta` | delta | yes under balanced shards | epoch boundaries crossed this step |
| `is_final` | flag | yes (derived from `total_steps`, §6.3) | last optimizer step |

`epoch` and `epoch_delta` are present now so adding epoch cadence later requires no
change to this contract; no first-cut `Schedule` reads them.

### 3.1 How the trainer builds it

The trainer constructs exactly one `EvalContext` per optimizer step, at the eval
call site, from state it already tracks (`global_step`, `epoch`, `tokens_seen`):

```python
def _build_eval_context(self, tokens_delta: int) -> EvalContext:
    epoch_delta = self.epoch - self._epoch_at_last_opt_step
    self._epoch_at_last_opt_step = self.epoch
    return EvalContext(
        global_step=self.global_step,
        epoch=self.epoch,
        tokens_seen=self.tokens_seen,     # rank-reduced — see §6.2
        steps_delta=1,                    # one optimizer step per context
        tokens_delta=tokens_delta,        # rank-reduced — see §6.2
        epoch_delta=epoch_delta,          # snapshot per opt-step, not per micro-batch
        is_final=(self.global_step >= self.total_steps),
    )
```

`epoch_delta` is snapshotted per optimizer step (not per micro-batch) so gradient
accumulation cannot double-count an epoch boundary that falls between micro-batches.

### 3.2 The rank-sync invariant (and the hang it prevents)

Eval tasks run **collective operations** inside `evaluate`: the sequence task calls
`accelerator.reduce(...)` to sum loss across ranks (`metrics/mlm.py`); the structure
task shards structures across ranks and `dist.all_gather_object(...)`s the results
(`tasks/structure.py`). A collective only completes when *every* rank enters it.

Therefore: if rank 0 decides a task is due and rank 1 does not, rank 0 enters the
collective alone and all ranks block until the NCCL timeout. The invariant that
prevents this is:

> **Every field of `EvalContext` is identical on every rank, so each rank
> independently computes the same `is_due` for every task — no communication, no
> disagreement.**

This is why `tokens_seen`/`tokens_delta` must be rank-reduced (§6.2) rather than the
current per-rank estimate, and why schedules must be pure functions of the context
(no rank-local state). The invariant is tested directly (§10).

---

## 4. Scheduling

### 4.1 The `Schedule` protocol

```python
class Schedule(Protocol):
    def is_due(self, ctx: EvalContext) -> bool: ...
```

A `Schedule` is pure and stateless. A task holds exactly one, built from its config
(§9). The `Evaluator` calls `is_due` on every optimizer step and never inspects the
cadence itself.

### 4.2 The crossing test (why not modulo)

For a counter `c` that advances by `delta` each step, "fire once each time `c`
crosses a multiple of `n`" is:

```python
def _crossed(curr: int, delta: int, n: int) -> bool:
    """True iff the half-open interval (curr - delta, curr] contains a multiple of n."""
    return curr // n > (curr - delta) // n
```

Modulo equality (`c % n == 0`) works for *steps*, where `delta == 1` and the counter
lands on every integer — and indeed `_crossed(step, 1, n)` reduces exactly to
`step % n == 0` for `step > 0`, so step behavior is bit-for-bit identical to today.
But it **fails for tokens**: `tokens_seen` jumps by a whole batch each step and
almost never lands exactly on a multiple of `n`. The crossing test fires on the
first step *past* each multiple, which is the correct and only realizable behavior
(you cannot evaluate at a token count that never materialized as a model state).

| `curr` | `delta` | `n` | `_crossed` | why |
| --- | --- | --- | --- | --- |
| 2000 | 1 | 1000 | True | step lands on a multiple (= `2000 % 1000 == 0`) |
| 2001 | 1 | 1000 | False | no multiple in (2000, 2001] |
| 1_000_030 | 80 | 1_000_000 | True | crossed 1e6 inside (999950, 1000030] |
| 2_500_000 | 2_500_000 | 1_000_000 | True | fires once though it spans two multiples |

### 4.3 Concrete schedules

```python
@dataclass(frozen=True)
class EveryNSteps:
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
    n: int
    at_start: bool = False
    at_end: bool = True
    def is_due(self, ctx: EvalContext) -> bool:
        return (
            (self.at_start and ctx.tokens_seen - ctx.tokens_delta == 0)
            or (self.at_end and ctx.is_final)
            or _crossed(ctx.tokens_seen, ctx.tokens_delta, self.n)
        )
```

`at_start` and `at_end` are combined with cadence by a boolean OR, which **inherently
dedupes**: when the final step also lands on a cadence multiple, `is_due` is still a
single `True`, so the task runs once. `at_start` is defined as "the counter
*before* this step was zero" (i.e. the first eval call), not `step == 0` — the
trainer's first eval call happens after the first optimizer step, so `global_step`
is never 0 at an eval call.

### 4.4 `at_start` / `at_end` defaults and meaning

| Flag | Default | Meaning |
| --- | --- | --- |
| `at_start` | `False` | Also evaluate on the very first eval call (a baseline before training has moved the weights; also fails fast on broken eval data/config). |
| `at_end` | `True` | Always evaluate once on the final model, even if the last step is not a cadence multiple. |

`at_end=True` is the default for **all** tasks, including expensive ones (e.g.
structure P@L): a run should never finish without metrics on its final checkpoint.
Opt a costly task out with `at_end: false` (§9).

### 4.5 Normative edge-case guarantees

| Case | Behavior |
| --- | --- |
| First eval call | `global_step == 1`; cadence fires only if `n == 1`; baseline requires `at_start: true`. |
| Final step | `at_end` (default) fires one eval at the last step even off-cadence; metrics are logged at `global_step`, which tolerates an off-cadence step. |
| `delta > n` (e.g. tokens-per-step exceeds the token cadence) | Fires **at most once per optimizer step**; `n` smaller than a batch degenerates to "every step." |
| `n > total` with `at_end=True` | Never fires on cadence; fires once at the end. |
| `n > total` with `at_end=False` | Never fires. The `Evaluator` logs a warning at construction when a schedule is provably unreachable (`total_steps` is known; token totals are estimated). |

### 4.6 Resume safety

Schedules persist **no** state, so there is nothing to checkpoint or restore.
Resume-safety follows from the crossing test using the *restored cumulative counter*
plus a *freshly computed delta* over the **half-open** interval `(curr - delta,
curr]`:

- **Steps and tokens are exact.** A checkpoint taken at `global_step = K` already
  fired (and logged) any eval due at `K`. On resume the first step makes
  `global_step = K + 1`, `steps_delta = 1`, and `_crossed(K+1, 1, n)` excludes `K`
  (it is the open end of the interval) — so no re-fire, and the next multiple after
  `K` still fires normally. The identical argument holds for `tokens_seen` because
  the trainer restores it verbatim and computes `tokens_delta` from the post-resume
  batch.
- **Epochs are deferred for exactly this reason** (§1.4). `epoch` is restored, but a
  mid-epoch checkpoint restarts the dataloader at the *beginning* of the restored
  epoch, so the first post-resume context carries no `epoch_delta` even though
  intra-epoch position was lost. Epoch cadence would be exact only for
  epoch-boundary checkpoints under balanced shards; that guarantee is weaker than
  steps/tokens, so epoch cadence ships in a later cut. When it does, `EveryNEpochs`
  is `_crossed(ctx.epoch, ctx.epoch_delta, n)` — no other change.

---

## 5. The `Evaluator`

```python
class Evaluator:
    def __init__(self, cfg: OplmConfig) -> None: ...
    def run_due(self, ctx: EvalContext, model, accelerator) -> dict[str, float]: ...
    @property
    def has_tasks(self) -> bool: ...
    @property
    def needs_token_count(self) -> bool: ...   # any task on a token schedule? (§6.2)
```

**Construction.** Import `oplm.eval.tasks` to trigger registration; call
`oplm.data.parse_eval_configs(cfg.data.eval, default_schedule)` to get typed
entries; for each, look up the class via the registry and instantiate it
(`cls(entry, cfg)`), which builds the task's `Schedule` from the entry's
`ScheduleSpec`. Warn on any provably unreachable schedule (§4.5).

**`run_due`.** The hot path, called every optimizer step:

```python
def run_due(self, ctx, model, accelerator) -> dict[str, float]:
    due = [t for t in self.tasks if t.schedule.is_due(ctx)]
    if not due:
        return {}                                   # cheap: no unwrap, no toggle
    unwrapped = accelerator.unwrap_model(model)     # only when ≥1 task is due
    unwrapped.eval()
    metrics: dict[str, float] = {}
    try:
        for task in due:
            for key, value in task.evaluate(unwrapped, accelerator).items():
                metrics[f"eval/{task.name}/{key}"] = value
    finally:
        unwrapped.train()
    return metrics
```

Two deliberate changes from today's `Evaluator.__call__(model, accelerator,
global_step)`: the bare `global_step` becomes a full `EvalContext`, and
`unwrap_model` moves **behind** the due-check so the common "nothing due" path costs
only a list comprehension over pure `is_due` calls. Tasks return **bare** metric
names (`"loss"`, `"precision_at_L"`); the `Evaluator` is the sole place that applies
the `eval/{name}/{metric}` namespace.

---

## 6. Trainer integration & distributed token accounting

This section specifies only the interface the trainer must satisfy; the trainer
loop itself is → future `docs/TRAINER.md`.

### 6.1 Call site

The trainer builds one `EvalContext` per optimizer step (§3.1) and calls
`evaluator.run_due(ctx, self.model, self.accelerator)` — passing the **wrapped**
model (the evaluator unwraps behind the due-check). Returned metrics are logged at
`ctx.global_step` and forwarded to callbacks. This replaces today's `_run_eval`,
which unwrapped every step and passed only `global_step`.

### 6.2 Distributed token accounting (required for token cadence)

Today the trainer accumulates `tokens_seen += local_batch_tokens * num_processes`,
where `local_batch_tokens` is the calling rank's own batch. With variable-length
sequences and ragged final shards, each rank's local count differs, so `tokens_seen`
**differs across ranks** — which violates the §3.2 invariant and would desync token
schedules. Replace the per-rank estimate with a true reduction:

```python
local_tokens = batch["attention_mask"].sum()                       # device tensor
tokens_delta = int(accelerator.reduce(local_tokens, reduction="sum").item())
self.tokens_seen += tokens_delta                                   # rank-identical
```

This yields the *true* global token count and makes `tokens_seen`/`tokens_delta`
rank-identical by construction. Under gradient accumulation, sum micro-batch tokens
locally across the accumulation window and reduce once at the optimizer step, so
`tokens_delta` is "global tokens in this optimizer step." The reduction is one small
all-reduce per step (negligible beside the backward pass) and is performed
**unconditionally** — not gated on `evaluator.needs_token_count`. Gating it off would
leave the per-rank estimate (`local_tokens × num_processes`), which diverges across ranks
on ragged batches and violates the §3.2 rank-identical invariant; the cost is too small to
justify that risk. (`needs_token_count` remains a useful introspection property, but the
trainer does not use it to skip the reduction.)

### 6.3 `is_final` and termination

`is_final` is `global_step >= total_steps`, where `total_steps` is computed
identically on all ranks at construction. The training loop must terminate on
`global_step` (step-bounded), which keeps `is_final` rank-identical. Epoch-bounded
runs with unbalanced shards have the same fragility as epoch cadence (§4.6) and are
out of the harness's rank-safety guarantee for the first cut.

### 6.4 Callbacks

Eval output is first-class training data — it drives the progress bar, checkpoint
selection, and (later) early stopping — so the evaluator is an explicit collaborator
the loop calls and gets metrics back from, **not** a fire-and-forget callback. The
existing `TrainerCallback.on_eval_end(trainer, metrics, step)` hook fires after a
non-empty eval, on the main process, with the namespaced metrics dict.

---

## 7. Eval tasks

### 7.1 The `EvalTask` base class

```python
class EvalTask(ABC):
    default_metrics: ClassVar[list[str]] = []

    def __init__(self, entry: EvalDatasetEntry, cfg: OplmConfig) -> None:
        self.name = entry.name
        self.path = entry.path
        self.metrics = entry.metrics or self.default_metrics
        self.schedule = build_schedule(entry.schedule)   # ScheduleSpec -> Schedule
        self.cfg = cfg

    @abstractmethod
    def evaluate(self, model, accelerator) -> dict[str, float]:
        """Return bare metric names; the Evaluator namespaces them. The model is
        already unwrapped and in eval mode."""
```

The `evaluate` signature is unchanged from today; the only structural change is that
a task now holds a `Schedule` (built from `entry.schedule`) instead of an
`eval_every: int`. Data loaders are initialized **lazily** on the first `evaluate`
call so configuring a rare eval costs nothing until it first runs (and a step-0
`at_start` eval surfaces broken data immediately).

### 7.2 The registry

```python
@register_eval_task("sequence")
class SequenceEvalTask(EvalTask): ...
```

`@register_eval_task(type_str)` adds the class to `EVAL_TASK_REGISTRY` (raising on a
duplicate type); `get_eval_task_class(type_str)` looks it up (raising with the list
of known types on a miss). `tasks/__init__.py` imports every task module so
registration happens on `import oplm.eval.tasks`. Adding a benchmark is: implement a
subclass, register it, point a `data.eval` entry's `type` at it — no change to the
`Evaluator` or trainer.

### 7.3 Implemented tasks

**`sequence` — `SequenceEvalTask`.** Masked-language-model metrics on held-out
sequences.
- Data: a parquet file or sharded directory with `(sequence_id, sequence)` columns
  (the training schema), loaded via `oplm.data.build_sequence_eval_dataloader(path,
  cfg)` — deterministic, frozen MLM masking, no shuffling, so the metric is
  comparable across checkpoints.
- Metrics: `loss`, `accuracy`, `perplexity` (§8.1). Distributed via
  `accelerator.reduce` over summed loss / correct / masked counts.

**`structure` — `StructureEvalTask`.** Contact-prediction precision@L.
- Data: a directory of PDB/CIF files via `oplm.data.load_structures(path,
  max_structures)` → `list[StructureData]`. Tokenization uses
  `oplm.data.get_tokenizer()` (the canonical tokenizer; the legacy `ProteinTokenizer`
  is gone).
- Metrics: `precision_at_L`, `precision_at_L_2`, `precision_at_L_5` computed from the
  categorical Jacobian (§8.3); `precision_at_L` is the default.
- Distributed: structures are sharded across ranks
  (`structures[rank::world_size]`), each rank processes its shard one structure at a
  time (the Jacobian and intermediates offloaded to CPU after each forward for
  memory), and per-structure results are gathered with `dist.all_gather_object`. Token
  cadence affects only *when* this task runs; the sharding is internal and independent
  of scheduling.

### 7.4 Stub tasks

Registered but not implemented; each raises `NotImplementedError` (no silent
no-op). Each maps to an existing `oplm.data` loader so implementation is metric work,
not loader work:

| Type | Data loader (`oplm.data`) | Planned metrics |
| --- | --- | --- |
| `proteingym` | `load_variant_assays` | `spearman`, `ndcg` (zero-shot variant effect via masked-marginal scoring) |
| `everest` | `load_variant_assays` | `spearman`, `auroc` (viral variant effect) |
| `tape` | downstream loader | `ss3_accuracy`, `contact_precision`, `fluorescence_spearman`, `stability_spearman`, … |
| `proteinglue` | downstream loader | `fold_accuracy`, `enzyme_accuracy`, `go_fmax` |

### 7.5 Per-task typed config

Shared keys (`path`, `type`, `every`, `metrics`) are consumed by the parser; every
other key on a `data.eval` entry is folded into `EvalDatasetEntry.extra: dict`
(§9). A task converts `extra` to a frozen, validated dataclass **once**, in its
constructor, rather than scattering `extra.get(...)` casts through `evaluate`:

```python
@dataclass(frozen=True)
class StructureTaskConfig:
    contact_threshold: float = 8.0
    min_seq_sep: int = 6
    l_divisor: int = 1
    use_cbeta: bool = True
    categorical_jacobian_sample_size: int | None = None
    categorical_jacobian_sample_seed: int = 42
    categorical_jacobian_mutation_batch_size: int = 20
    max_structures: int | None = None

    @classmethod
    def from_extra(cls, extra: dict[str, Any]) -> "StructureTaskConfig":
        """Cast and validate task knobs; raise ValueError on bad values."""
        ...
```

This keeps `extra: dict` at the YAML boundary (no global typed-config registry to
maintain) while giving each task a single, typed, validated config object.

---

## 8. Metrics

`eval/metrics/` holds eval-specific scoring math — it stays in the harness (not
`oplm.data`) because it is scoring, not loading. Definitions below are concise;
exact implementations and references live in the code docstrings.

### 8.1 MLM metrics (`metrics/mlm.py`)

Over masked positions only:
- **loss** — mean cross-entropy of the model's logits against the true tokens at
  masked positions, summed locally then reduced across ranks.
- **accuracy** — fraction of masked positions whose argmax prediction is correct.
- **perplexity** — `exp(loss)`, capped at 1000 to keep early-training values finite.

`compute_mlm_metrics(model, dataloader, accelerator) -> {loss, accuracy,
perplexity}`.

### 8.2 Contact-map and precision@L primitives (`metrics/contact.py`)

Shared scoring math used by the categorical-Jacobian metric:

- **Contact map** — binary `(L, L)`; residues `i, j` are in contact iff their Cβ–Cβ
  distance (virtual Cβ inferred from N, CA, C when absent) is below
  `contact_threshold` (8.0 Å) and `|i − j| ≥ min_seq_sep` (6).
- **APC (Average Product Correction)** — removes per-position background from a
  coupling matrix `F`: `F_apc[i,j] = F[i,j] − (rowmean_i · colmean_j) / mean(F)`.
- **Precision@L** — rank candidate long-range pairs (`|i − j| ≥ min_seq_sep`) by
  score, take the top `L / l_divisor`, and report the fraction that are true
  contacts.

### 8.3 Categorical-Jacobian coupling (`metrics/categorical_jacobian.py`)

The single contact-prediction signal. The categorical Jacobian
`J[i,a,j,b] = ∂ logit(x_j = b) / ∂(x_i = a)` is estimated by finite differences
(mutating each position to each canonical amino acid and measuring the logit shift).
Center over all four axes, symmetrize (`(J + Jᵀ)/2`), APC-correct, and reduce to an
`(L, L)` coupling map → precision@L using the §8.2 primitives. The result is reported
as `precision_at_L{,_2,_5}`. `categorical_jacobian_sample_size` optionally restricts
the (expensive, `L × 20` forwards per structure) computation to a deterministic subset
of structures.

### 8.4 Variant-task ranking metrics

For zero-shot variant effect (ProteinGym, EVEREST): score each variant by a
masked-marginal / pseudo-likelihood and compare to experimental labels with
**Spearman** rank correlation, **NDCG** (ranking quality emphasizing top variants),
and **AUROC** (for binary viral-effect labels). These are specified here for
completeness; the corresponding tasks are stubs (§7.4).

---

## 9. Configuration schema

The `data.eval` block is a mapping of dataset name → entry. The parser
(`oplm.data.parse_eval_configs`) produces typed `EvalDatasetEntry` objects; the
harness turns each entry's `ScheduleSpec` into a `Schedule`. The data layer parses
raw → typed; the eval layer interprets typed → behavior.

### 9.1 Dataclasses

```python
@dataclass(frozen=True)
class ScheduleSpec:                 # lives in oplm.config — no dependency on oplm.eval
    unit: str                       # "steps" | "tokens"   ("epochs" reserved — §4.6)
    n: int                          # positive
    at_start: bool = False
    at_end: bool = True

@dataclass
class EvalDatasetEntry:
    name: str
    path: str
    type: str                       # registry key: "sequence", "structure", …
    schedule: ScheduleSpec          # replaces the old eval_every: int | None
    metrics: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
```

`ScheduleSpec` is behavior-free and importable without pulling in `oplm.eval`, so
the data/config layer can produce it without a layering cycle.

### 9.2 The `every:` grammar

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `every.steps` \| `every.tokens` | positive `int` | — | **exactly one** unit key required |
| `every.at_start` | `bool` | `false` | also eval on the first call (baseline) |
| `every.at_end` | `bool` | `true` | always eval on the final model |

```yaml
every: { tokens: 50_000_000 }            # cadence by global tokens
every: { steps: 2000, at_end: false }    # cadence by optimizer steps, skip final-step eval
```

Rules enforced by the parser: `every` must be a mapping; exactly one of
`{steps, tokens}` present; the value a positive int; no unknown keys inside `every`.

### 9.3 Global default and resolution

A single global default applies to entries that omit `every`:

```yaml
train:
  eval_default_every: { steps: 10_000 }   # same grammar as per-entry `every`
```

Resolution: an entry's `every` wins; otherwise `train.eval_default_every`. One shared
`_parse_schedule_block` helper parses both the global default and each per-entry
`every` into a `ScheduleSpec`.

### 9.4 Validation and removed-alias errors

Following the project's clean-break convention (cf. `_reject_removed_sequence_length_alias`
in `config.py`, which rejects `data.max_length`), the steps-only `eval_every` is
**removed** and rejected with an explicit message rather than silently honored:

- `train.eval_every` (global): rejected in `load_config` —
  *"`train.eval_every` has been removed. Use `train.eval_default_every: {steps: N}`
  (or `{tokens: N}`)."*
- per-entry `eval_every`: rejected in `parse_eval_configs` before the `extra` fold —
  *"Eval dataset {name!r} uses the removed `eval_every` key. Use `every: {steps: N}`
  (or `{tokens: N}`)."*

### 9.5 Example

```yaml
train:
  eval_default_every: { steps: 5000 }

data:
  eval:
    heldout:                                  # cheap MLM eval, frequent
      path: /data/eval_sequences.parquet
      type: sequence
      every: { tokens: 20_000_000 }
    structures:                               # expensive contact eval, infrequent
      path: /data/pdb
      type: structure
      every: { steps: 20_000 }
      # task-specific knobs → EvalDatasetEntry.extra → StructureTaskConfig.from_extra
      contact_threshold: 8.0
      categorical_jacobian_sample_size: 12
```

---

## 10. Testing strategy

Scheduling is a pure function, so the bulk of the harness is testable without a GPU.

- **Crossing semantics** (no model): table-drive `_crossed` and each `Schedule` over
  the §4.5 cases — first call, final step, `delta > n`, `n > total` with/without
  `at_end`, `at_start` baseline.
- **Resume** (no model): build contexts up to step `K`, "restore" the counters,
  resume, and assert no re-fire at `K` and a correct next-fire — for both steps and
  tokens.
- **Rank-sync** (no model): given one `EvalContext`, every task's `is_due` is a pure
  function of it, so all ranks agree by construction; assert it, and add a multi-rank
  smoke test that a token-cadence eval does **not** hang on ragged batches.
- **Token accounting**: the reduced `tokens_seen` is rank-identical and equals the
  true global sum (not the per-rank `* num_processes` estimate).
- **Registry**: duplicate registration raises; unknown type raises with the known
  list.
- **Per-task evaluate** (small/real data, mark `@pytest.mark.slow` where heavy):
  sequence metrics on a tiny parquet fixture; structure P@L on one real PDB.
- **End-to-end**: a pilot-scale model trains a few steps with both a token-cadence
  and a step-cadence eval configured, and produces namespaced `eval/...` metrics
  without shape or scheduling errors.

Prefer real data over synthetic; session-scope fixtures that load files.

---

## 11. Migration from the current implementation

| Current (`src/oplm/eval/`, `training/`) | Target |
| --- | --- |
| `Evaluator.__call__(model, accelerator, global_step)` | `Evaluator.run_due(ctx: EvalContext, model, accelerator)` |
| `global_step % task.eval_every == 0` inside `Evaluator` | `task.schedule.is_due(ctx)` via `_crossed` (§4) |
| `EvalTask.eval_every: int` | `EvalTask.schedule: Schedule` |
| `EvalDatasetEntry.eval_every: int \| None` | `EvalDatasetEntry.schedule: ScheduleSpec` |
| `TrainConfig.eval_every: int` | `TrainConfig.eval_default_every: {unit: N}` + rejection of the old key (§9.4) |
| `eval/data/sequence_loader.py`, `eval/data/structure_loader.py` | **deleted**; tasks import `oplm.data` (`build_sequence_eval_dataloader`, `load_structures`/`StructureData`) |
| `StructureEvalTask` uses `ProteinTokenizer` | uses `oplm.data.get_tokenizer()` |
| `tokens_seen += local_batch_tokens * num_processes` (per-rank) | `accelerator.reduce` of per-step tokens (rank-identical, §6.2) |
| `unwrap_model` every step in `_run_eval` | unwrap behind the due-check in `run_due` (§5) |
| scattered `extra.get(...)` in `StructureEvalTask.__init__` | `StructureTaskConfig.from_extra(extra)` (§7.5) |

New capabilities with no legacy equivalent: **token cadence**, the `at_start` /
`at_end` flags, and the `EvalContext` deltas that make unit-agnostic, resume-safe
scheduling possible. Epoch cadence is designed (§4.6) but ships in a later cut.

---

## See also

- [`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md) — the model, its forward
  signature, and how logits are exposed (probed by the categorical Jacobian for
  contact prediction).
- [`DATA_TOOLING.md`](DATA_TOOLING.md) — the eval data loaders the harness consumes
  (`build_sequence_eval_dataloader`, `load_structures`, `load_variant_assays`) and
  the train/eval data boundary.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — high-level module map and extension points.
- [`../src/oplm/config.py`](../src/oplm/config.py) — `OplmConfig`, `EvalDatasetEntry`,
  `ScheduleSpec`, and config merge/validation.
- [`../src/oplm/configs/data/base.yaml`](../src/oplm/configs/data/base.yaml) — the
  packaged data/eval config defaults.

# YAGNI assessment

Assessed on 2026-09-22 against commit `795e73760e71c9ea9a881ee609c3816ef356dbce`.
This report proposes changes; none have been applied.

## Assessment

OPLM's strongest simplification opportunities are unfinished benchmark scaffolding,
an unused dependency, and interfaces that no longer have production callers. The
evidence does not support removing the distributed training infrastructure or
research options merely because they are large or disabled by default.

The largest candidate is the downstream labeled-data module. It works in isolation
and has tests, but no implemented application path consumes it. Its removal is
conditional: it is documented as a Python API, and repository searches cannot tell
whether external notebooks use it.

## Scope and method

The static inventory covered all 82 Python files under `src/oplm/` (18,300 physical
lines), dependency declarations, packaged recipes, and relevant tests and docs.
Candidate findings were checked through symbol/import searches and focused reads
of definitions, callers, configuration, and tests. This was not an exhaustive
line-by-line correctness review or a runtime coverage study.

The principal flows examined were CLI/configuration into training and inference;
training into data loaders, optimizers, evaluation and checkpointing; evaluation
registration into concrete tasks; and sweep/Slurm configuration into job generation.
Absence of an internal caller is evidence of an internal simplification opportunity,
not proof that a public API has no users. Tests that only exercise a helper show
that it works; they do not establish that an application needs it.

The scope is unnecessary functionality and maintenance cost. Correctness, security,
and performance findings are outside this assessment. No training, GPU, cluster, or
benchmark runs were performed for the analysis.

## Ranked findings

Ranked by potential source reduction first; dependency-only changes follow. Line
counts include comments and docstrings, exclude test/documentation cleanup, and are
estimates of a future change, not measured savings from an implemented patch.

| Rank | Finding | Confidence in evidence | Potential reduction | Decision |
| --- | --- | --- | --- | --- |
| 1 | `yagni:` downstream loader without an application consumer | High internally; external use unknown | 310 source lines | Remove only if standalone API support is unnecessary |
| 2 | `delete:` registered benchmark stubs | High | About 129 source lines | Defer executable registration until implementation |
| 3 | `delete:` singular optimizer factory without callers | High internally; external use unknown | 15 source lines | Remove or deprecate the unused entry point |
| 4 | `yagni:` epoch metadata for an unsupported eval cadence | High | About 8 source lines | Remove eval-only bookkeeping |
| 5 | `delete:` unused token-count capability property | High | About 5 source lines | Remove property and its dedicated assertions |
| 6 | `delete:` unused HuggingFace `datasets` requirement | High | One direct dependency | Remove from the training extra |
| 7 | `yagni:` DeepSpeed installed for every training user | High about packaging; usage unknown | One dependency moved out of the default training extra | Preserve explicit opt-in support |

### 1. Defer the downstream loader until it has a consumer

**Evidence:** [loader.py](../src/oplm/data/downstream/loader.py), particularly
`load_downstream_dataset` (line 196) and `collate_downstream_labels` (line 250),
implements three label families, CSV/parquet parsing, validation and collation.
Searches for these functions and their types found no consumers in other source
modules. Calls occur in its [335-line test module](../tests/data/downstream/test_loader.py).
The intended TAPE and ProteinGlue evaluation tasks are still stubs and do not
import the loader.

**Smallest action:** If standalone labeled-data loading is not a supported use
case, remove the 309-line loader and its one-line package initializer. Keep the
benchmark requirements in documentation until an actual evaluation implementation
needs them. Do not replace the module with another abstraction or expand the stubs
just to give it a caller.

**Limit:** [DATA_TOOLING.md](DATA_TOOLING.md) advertises this modality for
probing/fine-tuning. External callers must be considered before removal. If those
users exist, keep the implementation and its validation/tests; “no internal calls”
alone is insufficient grounds to break their API. The 335 test lines are not
included in the estimated source savings.

**Add back when:** A concrete benchmark or supported standalone workflow needs
labeled-sequence loading. Implement the formats and label types that workflow uses.

### 2. Remove executable registrations for unimplemented benchmarks

**Evidence:** [tape.py](../src/oplm/eval/tasks/tape.py),
[proteinglue.py](../src/oplm/eval/tasks/proteinglue.py), and
[everest.py](../src/oplm/eval/tasks/everest.py) total 126 lines. Each registers an
`EvalTask`, supplies default metrics, and implements `evaluate` solely by raising
`NotImplementedError`. [tasks/__init__.py](../src/oplm/eval/tasks/__init__.py)
imports all three, and [registry tests](../tests/eval/test_registry.py) include them
in `_REAL_TYPES`.

**Smallest action:** Remove the three modules and their three registration imports.
Retain their intended protocols as roadmap text. The existing registry already
rejects unavailable types and lists available ones; no replacement framework is
needed. Adjust registry expectations and the available-task documentation together.

**Effect:** Selecting these types would fail when constructing the evaluator,
instead of being accepted until the first due evaluation. This changes error timing
and removes importable class names, but removes no implemented benchmark behavior.

**Add back when:** A task can load its data and produce a meaningful metric, with
an integration check covering that path. Keep the registry and `EvalTask` base:
sequence, structure, ProteinGym DMS and ProteinGym clinical are real implementations.

### 3. Retire the unused singular optimizer factory

**Evidence:** [optim.py](../src/oplm/training/optim.py), lines 176–190, defines
`build_optimizer` as `build_optimizers(model, cfg)[0]`. A repository-wide symbol
search found no callers in source or tests. The trainer and coordinate-check code
use `build_optimizers`, which preserves the full Muon-plus-AdamW optimizer set.

**Smallest action:** Remove the singular wrapper if it is not externally supported.
An explicit caller that needs the first optimizer can select it from the existing
list. If compatibility matters, deprecate it first; that postpones the line savings.

**Limit:** Do not remove `build_scheduler` by analogy: `build_schedulers` calls it.
Do not replace the optimizer list with one optimizer; Muon training needs both.

**Add back when:** A supported caller needs a separately defined primary-optimizer
API with semantics beyond indexing an existing list.

### 4. Remove epoch bookkeeping carried only for future eval scheduling

**Evidence:** [context.py](../src/oplm/eval/context.py), lines 19 and 23, explicitly
labels `epoch` and `epoch_delta` as future/unused fields. The only implemented
[schedules](../src/oplm/eval/schedule.py) use steps or tokens, and
`parse_schedule_block` in [config.py](../src/oplm/config.py) rejects epoch cadence.
[Trainer](../src/oplm/training/trainer.py) nevertheless initializes/restores
`_epoch_at_last_opt_step` and updates it in `_build_eval_context` to populate these
fields. Neither schedule reads them.

**Smallest action:** Remove the two context fields, their constructor arguments,
and `_epoch_at_last_opt_step` bookkeeping. Update context fixtures and the docs that
promise future epoch support without a contract change.

**Limit:** Preserve `Trainer.epoch`, epoch-bounded training, data-cursor restoration,
and epoch logging. Those are real consumers of epoch state. `EvalContext` is
exported, so changing its constructor also requires an API compatibility decision.
Keep `steps_delta` and `tokens_delta`: current schedules actually read them.

**Add back when:** Epoch-based evaluation is implemented with defined semantics and
tests for accumulation, resume, and agreement across ranks.

### 5. Remove the unused `needs_token_count` property

**Evidence:** [evaluator.py](../src/oplm/eval/evaluator.py), lines 99–102, computes
whether any task uses `EveryNTokens`. Its only executable consumers are two
[evaluator tests](../tests/eval/test_evaluator.py). The trainer unconditionally
counts and reduces tokens, including when evaluation is absent. Its reduction also
coordinates drain, non-finite loss, checkpoint timing, and async-save completion.

**Smallest action:** Remove the property and the now-unused `EveryNTokens` import
from the evaluator, plus tests and documentation solely describing this property.

**Limit:** Do not make token reduction conditional to justify this interface. Token
accounting and control coordination already have independent consumers. As with
other public methods, check compatibility expectations before removal.

**Add back when:** A real caller needs this query. Do not build optional token
accounting infrastructure in anticipation of one.

### 6. Remove `datasets` from the training dependency group

**Evidence:** [pyproject.toml](../pyproject.toml), line 58, declares
`datasets>=2.18`. AST import inventory and text searches found no HuggingFace
`datasets` imports in source or tests. [DATA_TOOLING.md](DATA_TOOLING.md) explicitly
says there is no HuggingFace `datasets` integration. The live path uses PyArrow,
`ShardedProteinDataset`/`InterleavedDataset`, and PyTorch `DataLoader`.

**Smallest action:** Delete that requirement and update the training-install
descriptions in [README.md](../README.md) and [TRAIN.md](TRAIN.md). No code or
replacement dependency is needed.

**Limit:** This saves one declared dependency; the resulting transitive package
count and installation size have not been measured. A clean installation should
verify that no tooling relies accidentally on packages supplied transitively by it.

**Add back when:** An implemented dataset integration imports and uses it.

### 7. Keep DeepSpeed installation opt-in, as its runtime behavior already is

**Evidence:** [pyproject.toml](../pyproject.toml), line 56, includes DeepSpeed in
every `oplm[train]` install. [train.py](../src/oplm/train.py) explicitly disables its
Accelerate environment settings unless `OPLM_ENABLE_DEEPSPEED` is enabled.
[TRAIN.md](TRAIN.md) documents that opt-in and external Accelerate configuration.
There are no direct DeepSpeed imports in source/tests, but this is **not** evidence
that the opt-in is unused: Accelerate provides the indirect integration.

**Smallest action:** Remove DeepSpeed from the default training extra and document
an explicit installation command for its users. A separate extra is optional if
the project wants to maintain that installation combination.

**Limit:** Preserve bootstrap guards and the opt-in behavior. This is dependency
placement, not a recommendation to delete DeepSpeed support or a claim that the
backend has been validated. Users currently relying on `oplm[train]` to install it
would need the revised installation instructions. The package is not counted as
eliminated project-wide.

**Broaden installation again when:** DeepSpeed becomes part of the ordinary,
supported training recipe.

## Complexity worth keeping

| Area | Evidence of a present requirement |
| --- | --- |
| Checkpoint commits, RNG/scaler state, remote mirroring, drain/requeue and HSDP | Concrete recovery requirements in the [fault-tolerance design](superpowers/specs/2026-08-12-fault-tolerant-training-design.md), implemented trainer paths, and dedicated checkpoint/resume/distributed tests. Local rename and object-store manifest commits have different constraints. Replacing these with a plain save call would discard behavior. |
| `DeviceDataLoader` and data-exact cursor handling | [loaders.py](../src/oplm/data/sequence/loaders.py) explains why normal Accelerate preparation would shard an already-sharded iterable again. [Double-sharding tests](../tests/data/test_double_sharding.py) and [resume-cursor tests](../tests/data/test_resume_cursor.py) cover concrete requirements. The wrapper is not redundant delegation. |
| Research toggles, model factories, and looping | Norm and FFN factories have multiple implementations; [toggle tests](../tests/model/test_toggles.py) exercise combinations. [Looping requirements](superpowers/specs/2026-09-21-looped-oplm-design.md) explicitly cover inference and staged training. Disabled-by-default does not establish lack of need in a research codebase. Retiring an experiment needs a research/support decision. |
| Trainer callbacks | `SweepMetricsCallback` and `StabilityDiagnosticsCallback` are production implementations in [mup.py](../src/oplm/training/mup.py). This is not an interface with only hypothetical consumers. |
| Shared Slurm infrastructure and sweep phases | Both general Slurm commands and sweeps use the config/render/submit modules. The [sweep guide](LR_SWEEP.md) defines real phase-specific ranking and selection behavior. A large phase file alone is not evidence that those phases should disappear. |
| Existing numerical and storage dependencies | NumPy, PyArrow, BioPython, fsspec, pandas and matplotlib have actual source imports. Several imports are lazy or only needed by particular tools; those facts do not make the dependencies unused. |
| HuggingFace integration and inference formats | Model exports, Auto-class registration, tokenizer compatibility and checkpoint/HF loading have documented consumers and tests. Internal reference counting cannot establish that these public entry points are dead. |

Some duplication is cheaper than a new framework. The DMS and clinical tasks already
share variant scoring, while differing in label meaning, score sign and aggregation.
The assessment does not recommend a generic benchmark hierarchy to merge those
small differences. Likewise, it does not recommend merging modules just because
they export one concept, removing input validation, or deleting regression tests
to improve a line-count score.

## Suggested order and validation for a later change

1. Remove the unused `datasets` requirement and correct installation documentation.
   Validate a clean `dev,train` installation, import/CLI smoke checks, and the normal
   CPU test suite without installing `datasets` explicitly.
2. Remove benchmark stub registrations and, if compatibility permits, the unused
   optimizer wrapper and eval metadata/query. Update registry/context tests;
   exercise schedule, token-accounting, eval integration, optimizer and resume tests.
3. Decide whether the downstream loader is a supported standalone API. Delete it
   only if that answer is no; retire its dedicated tests and documentation in the
   same change. Otherwise keep it and exclude its 310 lines from the savings.
4. Separate DeepSpeed installation while preserving its opt-in documentation and
   bootstrap tests. Validate that installation path independently if it remains
   officially supported.

Any later source changes should run the repository's Ruff, `ty check src/`, and
appropriate pytest checks. This report does not establish that those proposed
changes pass them. No new abstraction or replacement subsystem is proposed.

If every source-removal candidate is accepted without compatibility shims, the
estimated reduction is about 470 physical source/metadata lines, plus removal of
one direct dependency (`datasets`). Roughly 310 of those lines depend on retiring
the standalone downstream API. Test/doc cleanup is excluded; DeepSpeed relocation
is not counted as deleting a dependency from the project.

**Potential net: approximately -470 lines, -1 dependency; conditional, not applied.**

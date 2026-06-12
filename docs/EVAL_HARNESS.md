# OPLM Evaluation Harness

How OPLM evaluates models during training. This is the practical reference for
the eval tasks, the during-training scheduling contract, and the metrics each
task returns. For the full narrative — the trainer↔eval contract, distributed
token accounting, and the `Evaluator` internals — see
[OVERVIEW.md §21–27](OVERVIEW.md). For the `data.eval` field reference, see
[CONFIG.md](CONFIG.md).

Code lives under `src/oplm/eval/`.

## Tasks

Eval tasks are registered by a string `type` via the `@register_eval_task(...)`
decorator (`src/oplm/eval/registry.py`) and resolved at config-parse time. Every
task subclasses `EvalTask` (`src/oplm/eval/tasks/base.py`) and implements
`evaluate(model, accelerator) -> dict[str, float]`.

| `type` | Status | Default metrics | What it computes |
| --- | --- | --- | --- |
| `sequence` | **implemented** | `loss`, `accuracy`, `perplexity` | Masked-LM metrics over held-out parquet (same schema as training data). |
| `structure` | **implemented** | `precision_at_L` | Unsupervised contact prediction from the model's categorical Jacobian over PDB/CIF structures. Also supports `precision_at_L_2`, `precision_at_L_5`. |
| `tape` | stub | SS3/SS8, contact, homology, fluorescence, stability | TAPE benchmark — raises `NotImplementedError`. |
| `proteingym` | stub | `spearman`, `ndcg` | ProteinGym DMS variant-effect prediction — raises `NotImplementedError`. |
| `proteingym_clinical` | **implemented** | `auroc` | ProteinGym clinical-substitution pathogenicity (Pathogenic vs Benign) via per-position log-likelihood ratios; mean AUROC across per-protein assays. |
| `proteinglue` | stub | fold/enzyme/GO | ProteinGlue benchmark — raises `NotImplementedError`. |
| `everest` | stub | `spearman`, `auroc` | EVEREST clinical-variant benchmark — raises `NotImplementedError`. |

The stubs are registered (so configs referencing them parse) but raise when run;
they document the planned task surface.

## Running eval

Evaluation is **training-integrated only** — there is no standalone `oplm eval`
CLI command. Eval runs are configured under `data.eval` and fire on a schedule
during training. The CLI exposes `train`, `encode`, and `info`
(`src/oplm/cli.py`); none of these runs eval in isolation.

### During-training integration

When `cfg.data.eval` is non-empty, the trainer builds an `Evaluator(cfg)`
(`src/oplm/eval/evaluator.py`). At each optimizer step the trainer calls
`evaluator.run_due(ctx, model, accelerator)` with an `EvalContext` carrying the
rank-reduced step/token counters. The evaluator:

1. checks each task's schedule and returns early if none are due;
2. unwraps the distributed model and switches it to `.eval()`;
3. runs every due task and namespaces its metrics as
   `eval/<task_name>/<metric>`;
4. restores `.train()` in a `finally` block.

Returned metrics are logged to W&B via `accelerator.log(..., step=global_step)`
and forwarded to callbacks through `on_eval_end`. The progress bar surfaces the
eval loss (`eval/<name>/loss`).

### Cadence

Each eval dataset may set a per-dataset `every` block with exactly one of
`{steps: N}` or `{tokens: N}`, plus optional `at_start` (default `false`) and
`at_end` (default `true`). Datasets that omit `every` inherit `train.eval_every`,
which itself defaults to `{steps: 10_000}`. The two schedule types
(`EveryNSteps`, `EveryNTokens` in `src/oplm/eval/schedule.py`) fire on half-open
interval crossings, so a cadence fires once per crossing regardless of step
granularity. Multiple datasets on different cadences fire independently and merge
on shared steps.

## Structure eval: the categorical Jacobian

The `structure` task (`src/oplm/eval/tasks/structure.py` +
`src/oplm/eval/metrics/categorical_jacobian.py`) predicts residue–residue
contacts with no supervised contact head, by probing how each position's logits
respond to point mutations elsewhere:

1. **Wildtype pass** — run the model once on the wildtype sequence to get
   reference logits `(L, A)` restricted to the canonical amino acids.
2. **Mutation sweep** — for every (position, amino-acid) substitution, run a
   forward pass and subtract the wildtype logits, building a coupling tensor
   `(L, A, L, A)`. Mutations are batched (`categorical_jacobian_mutation_batch_size`,
   default 20, clamped to `[1, 20]`).
3. **Reduce to a contact map** — center across all four axes, symmetrize over the
   input/output positions, reduce `(L, A, L, A) → (L, L)` by the Frobenius norm
   over the amino-acid axes, zero the diagonal, apply Average Product Correction
   (APC) to remove background bias, and symmetrize.
4. **Score** — build the ground-truth contact map from backbone coordinates
   (virtual Cβ via ideal backbone geometry, works for glycine; contact iff Cβ–Cβ
   distance < `contact_threshold`, default 8.0 Å), restrict to long-range pairs
   (`|i − j| ≥ min_seq_sep`, default 6), and compute precision over the top
   `L / l_divisor` predicted pairs.

Structures are sharded across ranks and gathered with `all_gather_object`. A
sequence is eligible only if `len + 2 ≤ max_position_embeddings`. NaN backbone
coordinates are treated as infinite distance (no contact); with no valid pairs or
no loaded structures the precision is `0.0` (a finite value in `[0, 1]`, not an
error).

> The cost of the Jacobian is `L × 20` forward passes per structure. Use
> `categorical_jacobian_sample_size` to evaluate a deterministic subset and
> `max_structures` to cap how many are loaded.

## ProteinGym clinical eval: variant scoring

The `proteingym_clinical` task reads a directory of one-protein clinical CSVs
(`protein_sequence` = the constant wild-type, `mutant`, `DMS_bin_score` =
`Pathogenic`/`Benign`) via `load_clinical_variant_assays`. The wild-type comes
straight from the `protein_sequence` column — no reconstruction. Two scoring
methods trade accuracy for cost:

- `masked_marginals` (default) — mask each unique mutated position and read its
  log-probabilities from that forward pass (the ESM-1v protocol; one batched
  forward per unique position). Most accurate.
- `wt_marginals` — a single wild-type forward pass; read every mutation's
  log-probabilities from it. Cheapest, useful for frequent during-training eval.

Assays whose wild-type exceeds the model context are skipped; so are assays
without both classes (AUROC needs a positive and a negative). With no scorable
assay the metric is `0.0` (finite, not an error).

## Metrics

Each task returns a `dict[str, float]` of bare metric names; the evaluator
namespaces them as `eval/<task_name>/<metric>`:

- **sequence** — `loss` (cross-entropy over masked positions), `accuracy`
  (fraction of masked positions predicted correctly), `perplexity` (`exp(loss)`,
  capped at 1000) — see `src/oplm/eval/metrics/mlm.py`.
- **structure** — `precision_at_L`, `precision_at_L_2`, `precision_at_L_5` (top
  `L`, `L/2`, `L/5` long-range contacts).
- **proteingym_clinical** — `auroc`: per-protein ROC-AUC of Pathogenic-vs-Benign
  discrimination, macro-averaged across proteins. Each variant is scored by the
  summed per-position log-likelihood ratio
  `LLR = log P(mut | ctx) − log P(wt | ctx)`; since a higher LLR means a more
  tolerated (Benign-like) substitution, the **pathogenicity score is `−LLR`** so a
  good model yields AUROC > 0.5. Scoring uses `masked_marginals` (default) or
  `wt_marginals`. Assays without both classes are skipped. AUROC is implemented in
  NumPy in `src/oplm/eval/metrics/classification.py`.

Metrics are logged to W&B and passed to callbacks; there are no separate eval
output files.

## Config surface (`data.eval`)

`data.eval` maps a task name to a spec; the parsed form is `EvalDatasetEntry`
(`src/oplm/data/config.py`):

```yaml
data:
  eval:
    heldout:
      path: /data/eval.parquet
      type: sequence
      every: { tokens: 20_000_000 }
    structures:
      path: /data/pdb
      type: structure
      every: { steps: 20_000 }
      categorical_jacobian_sample_size: 12   # task-specific key, same level as path/type
    clinical_variants:
      path: /data/proteingym_clinical    # dir of one-protein CSVs
      type: proteingym_clinical
      every: { steps: 20_000 }
      scoring: masked_marginals          # task-specific keys, same level as path/type
      mask_batch_size: 64
```

- `path` — file or directory consumed by the task.
- `type` — one of the registry keys above.
- `every` — optional cadence (see [Cadence](#cadence)).
- `metrics` — optional list of metric names to keep (defaults to the task's
  defaults).
- Task-specific keys (e.g. the structure task's `contact_threshold`,
  `min_seq_sep`, `l_divisor`, `use_cbeta`, `categorical_jacobian_*`,
  `max_structures`) sit at the **same level** as `path`/`type`, not nested under
  an `extra:` key. The full structure-task key table is in
  [CONFIG.md](CONFIG.md#structure).

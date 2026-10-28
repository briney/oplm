# Evaluation

How OPLM evaluates a model during training: the eval tasks and their metrics, when
evals run, the `data.eval` schema, and the published structure eval sets. Code lives
under `src/oplm/eval/`. For every `data.eval` key and default, see
[config.md](config.md), *Eval datasets*. For how the eval data was built, see
[data.md](data.md), *Evaluation data*.

Evaluation is training-integrated only. There is no standalone `oplm eval` command;
the CLI exposes `train`, `encode`, `info` and `fold` (`src/oplm/cli.py`).

## Tasks and metrics

Eval tasks are registered by a string `type` with `@register_eval_task(...)`
(`src/oplm/eval/registry.py`) and resolved at config-parse time. Every task subclasses
`EvalTask` (`src/oplm/eval/tasks/base.py`) and implements
`evaluate(model, accelerator) -> dict[str, float]` returning bare metric names. The
evaluator namespaces them as `eval/<task_name>/<metric>`. Metrics go to W&B and to
callbacks (`on_eval_end`); there are no eval output files.

| `type` | Status | Default metrics | What it computes |
| --- | --- | --- | --- |
| `sequence` | implemented | `loss`, `accuracy`, `perplexity` | Masked-LM metrics over held-out parquet (same schema as training data). |
| `structure` | implemented | `precision_at_L` | Unsupervised contact prediction from the categorical Jacobian over PDB/CIF structures. Also `precision_at_L_2`, `precision_at_L_5`. |
| `proteingym` | implemented | `spearman`, `auroc`, `top_k_precision` | ProteinGym DMS variant-effect prediction by per-position log-likelihood ratio. |
| `proteingym_clinical` | implemented | `auroc` | ProteinGym clinical pathogenicity (Pathogenic vs Benign) by per-position log-likelihood ratio. |
| `tape`, `proteinglue`, `everest` | stub | | Registered so configs parse; they raise `NotImplementedError` when run. |

### sequence

`loss` is cross-entropy over masked positions, `accuracy` the fraction of masked
positions predicted correctly, `perplexity` is `exp(loss)` capped at 1000
(`src/oplm/eval/metrics/mlm.py`). Masking is deterministic and frozen, with no shuffle,
so values are comparable across checkpoints (see [data.md](data.md), *Deterministic
eval masking*).

### structure

The task (`src/oplm/eval/tasks/structure.py`,
`src/oplm/eval/metrics/categorical_jacobian.py`) predicts residue-residue contacts
with no supervised head, by probing how each position's logits respond to point
mutations elsewhere:

1. Run the model on the wild-type sequence for reference logits `(L, A)` over the
   canonical amino acids.
2. For every (position, amino acid) substitution run a forward pass and subtract the
   wild-type logits, giving a coupling tensor `(L, A, L, A)`. Mutations are batched
   (`categorical_jacobian_mutation_batch_size`, clamped to `[1, 20]`).
3. Center across all four axes, symmetrize, reduce to `(L, L)` by the Frobenius norm
   over the amino-acid axes, zero the diagonal, apply Average Product Correction
   (APC), and symmetrize.
4. Score against the ground-truth contact map from backbone coordinates (virtual Cβ,
   so glycine works; contact iff Cβ-Cβ distance is below `contact_threshold`),
   restricted to pairs with `|i - j| >= min_seq_sep`. Pairs involving a residue
   without coordinates are dropped (unknown, not non-contacts). Precision is taken
   over the top `L / l_divisor` predicted pairs, where L is the number of residues
   with coordinates.

`precision_at_L_2` and `precision_at_L_5` are the same score over the top `L/2` and
`L/5` pairs. Defaults for the keys above are in [config.md](config.md), *structure*.

The model reads the full sequence. For a PDB file whose numbering fits its SEQRES that
includes unmodeled residues, which load as NaN rows (the ESM-C convention). A
sequence is eligible only if `len + 2 <= model.max_position_embeddings`. A structure
with no full-backbone residue is skipped. With no valid pairs or no loaded structures
the precision is `0.0` (finite, not an error). Structures are sharded across ranks
(`structures[rank::world_size]`) and gathered with `all_gather_object`.

> The Jacobian costs `L x 20` forward passes per structure. Use
> `categorical_jacobian_sample_size` for a deterministic subset and `max_structures`
> to cap how many load.

### proteingym and proteingym_clinical

Both share one scoring core (`src/oplm/eval/tasks/variant_scoring.py`). A variant's
score is the summed per-position log-likelihood ratio
`LLR = log P(mut | ctx) - log P(wt | ctx)`; a higher LLR means the substitution is
more tolerated. The `scoring` key picks the cost:

- `masked_marginals` (default): mask each unique mutated position and read its
  log-probabilities from that forward pass (the ESM-1v protocol). Most accurate.
- `wt_marginals`: one wild-type forward pass serves every mutation. Cheapest, for
  frequent in-training eval.

`proteingym` reads DMS CSVs (`mutant`, `DMS_score`, `mutated_sequence`, optional
`DMS_score_bin`; the wild-type is rebuilt from `mutated_sequence`). Metrics:
`spearman` (rank correlation of LLR with `DMS_score`), `auroc` (against
`DMS_score_bin`, when present) and `top_k_precision` (overlap of the top
`top_k_fraction` of variants by `DMS_score` and by model score). Ranking metrics are
in `src/oplm/eval/metrics/ranking.py`.

`proteingym_clinical` reads one-protein CSVs (`protein_sequence`, `mutant`,
`DMS_bin_score` of `Pathogenic` or `Benign`). A high LLR is Benign-like, so the
pathogenicity score is `-LLR` and a good model gets `auroc` above 0.5. AUROC is in
`src/oplm/eval/metrics/classification.py`.

Per-assay metrics are macro-averaged across assays. Assays whose wild-type exceeds the
model context are skipped, as are assays degenerate for a metric (one AUROC class, a
constant Spearman input). With no scorable assay the metric is `0.0`.

## Scheduling during training

When `data.eval` is non-empty the trainer builds an `Evaluator(cfg)`
(`src/oplm/eval/evaluator.py`). Each optimizer step it builds an `EvalContext`
(`src/oplm/eval/context.py`) and calls `evaluator.run_due(ctx, model, accelerator)`.
The evaluator:

1. asks each task's schedule whether it is due, returning early if none are;
2. unwraps the distributed model and switches it to `.eval()`;
3. runs every due task and namespaces its metrics;
4. restores `.train()` in a `finally` block.

Metrics are logged at `ctx.global_step`, and the progress bar shows
`eval/<name>/loss`.

**Cadence.** Each `data.eval` entry may set `every` with exactly one of `{steps: N}`
or `{tokens: N}`, plus `at_start` (default `false`) and `at_end` (default `true`).
Entries that omit `every` inherit `train.eval_every`
(`src/oplm/configs/train/base.yaml`). The schedules (`EveryNSteps`, `EveryNTokens` in
`src/oplm/eval/schedule.py`) are pure functions of the context. They fire when the
counter crosses a multiple of N: `curr // n > (curr - delta) // n`. A modulo test would
miss token cadences, because tokens advance by a whole batch. A schedule fires at most
once per optimizer step. Entries on different cadences fire independently and merge on
shared steps.

- `at_start` fires on the first eval call (the counter before the step was zero), a
  pre-training baseline that also fails fast on broken eval data.
- `at_end` guarantees metrics on the final step, for every task including expensive
  ones. Set `at_end: false` to opt a costly task out.
- Resume is safe with nothing to persist: a checkpoint at step K already fired any eval
  due at K, and the crossing test excludes K on resume.
- Epoch cadence is not supported (a mid-epoch checkpoint restarts the dataloader at
  the epoch start, so it cannot be exact). `every: {epochs: N}` is rejected at parse
  time.

**Rank-sync invariant.** Eval tasks run collectives (`accelerator.reduce`,
`all_gather_object`), which complete only when every rank enters them. Every
`EvalContext` field is therefore identical on every rank, so each rank computes the
same `is_due` with no communication. For that, the trainer all-reduces the per-step
token count (`tokens_delta`) every step, rather than estimating it as
`local_tokens x num_processes`, which diverges across ranks on ragged batches. Under
gradient accumulation it sums micro-batch tokens locally and reduces once per
optimizer step. `is_final` is `global_step >= total_steps`.

## `data.eval` schema

`data.eval` maps an entry name you choose to a spec, parsed into `EvalDatasetEntry`
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
      categorical_jacobian_sample_size: 12   # task key, same level as path/type
```

- `path`: file or directory consumed by the task.
- `type`: a registry key from the table above.
- `every`: optional cadence (see *Scheduling during training*).
- `metrics`: optional list of metric names to keep (defaults to the task's).
- Task-specific keys sit at the same level as `path` and `type`, not under `extra:`.
  Each task validates them once in its constructor.

The per-task key tables (names, types, defaults) live in [config.md](config.md),
*Eval datasets*. The recipe's eval entries are in `data.eval` of
[`configs/scaling.yaml`](../configs/scaling.yaml).

## Published structure eval sets

The structure sets are published as the Hugging Face dataset
[`brineylab/oplm-structure-evals`](https://huggingface.co/datasets/brineylab/oplm-structure-evals),
tag `2026-10-12`. Provenance, curation rules and counts are in [data.md](data.md),
*Structure evaluation sets*. To use them as `type: structure` entries, download the tag
and point `path` at a set's folder:

| set | use | needs `model.max_position_embeddings` |
| --- | --- | --- |
| `casp15_16` | CASP15 and CASP16 domains | at least 820 |
| `cameo`, `cameo_256` | CAMEO chains; seed-42 256-chain subset | `cameo_256` at least 514 |
| `pdb40`, `pdb40_2048`, `pdb40_256` | PDB40 representatives; seed-42 subsets | full `pdb40` at least 1026; `_256` subsets at least 514 |

Sequences too long for the context are skipped, not errors, so a too-small
`model.max_position_embeddings` silently shrinks the set. The `2026-10-12` folder
distributes full `pdb40` as `pdb40.tar.gz` (the Hub caps a directory at 10,000
files); extract it to get `pdb40/`. Full `pdb40` is expensive (29,429 structures at
`L x 20` forwards each), so the recipe evaluates the 256 and 2048 subsets. See
`data.eval` in [`configs/scaling.yaml`](../configs/scaling.yaml).

## Offline evaluation

In-loop eval cannot run under `train.parallelism: hsdp`. `fully_shard` leaves the
evaluator holding the sharded module, tasks stripe work across ranks so ranks run
different numbers of forwards, and the early finishers never issue the matching
all-gathers. The group hangs until `dist_timeout_minutes`. `load_config` refuses the
combination up front (`validate_parallelism_compat` in `src/oplm/config.py`), so an
HSDP run must have an empty `data.eval`.

> **Pending:** standalone eval of a saved checkpoint, to score HSDP runs offline (waiting on #57).

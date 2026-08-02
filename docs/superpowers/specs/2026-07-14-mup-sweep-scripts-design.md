# μP Sweep Scripts Design

## Goal

Replace the current one-GPU μP pilot/sweep scripts with a small phase-oriented
workflow that executes the production LR-optimization protocol in
`docs/LR_SWEEP.md`. The same generated training cells must run through
`accelerate launch` on one local 8×B200 node or inside SUNK/Slurm jobs.

The workflow optimizes:

- the μP base learning rate;
- `mup_output_mult` in a local joint search with LR;
- one depth-LR exponent for the production scaling ray; and
- the empirical LR correction from 2,048 to 8,192 global examples.

Production weight decay remains fixed at `0.01`.

## Design principles

- Use the normal OPLM configuration loader and `Trainer`; do not build a second
  training stack.
- Require one user-provided base OPLM YAML. Phase code changes only the
  experimental axes.
- Generate fully resolved YAML for every training cell.
- Keep resource scheduling outside the μP workflow. The scripts generate normal
  `accelerate launch` commands that can run locally or be wrapped by Slurm.
- Local mode runs cells sequentially and gives each cell all eight GPUs.
- Prefer explicit phase artifacts over hidden state or a generalized workflow
  engine.
- Add core training behavior only where a faithful LR experiment requires it.

## Out of scope

- Automated weight-decay optimization.
- Padded-token, masked-target, gradient-norm, update-RMS, activation, attention,
  Canon, or optimizer-state instrumentation.
- Custom batching or a guarantee that each 4M batch is the exact union of four
  1M batches.
- Slurm submission, polling, retry, or scheduler simulation.
- Concurrent local experiment cells or GPU partitioning.
- A new experiment configuration language, database, dashboard, or schema
  migration framework.
- Automatic LR-grid expansion, generalized early stopping, or elaborate failure
  recovery.

## Files and responsibilities

### `scripts/mup_coord_check.py`

Retain this as the independent parameterization gate. It must accept the base
OPLM YAML, resolve production model features, and support both fixed-depth and
32:1 production-ray checks. Its existing CSV/PNG outputs and one-sided RMS
oracle remain sufficient.

### `scripts/mup_run.py`

New distributed-capable one-cell entry point. It accepts a fully resolved run
YAML and result path, loads the config, constructs the normal `Trainer`, adds
`SweepMetricsCallback`, and trains. It is invoked under `accelerate launch`.

The callback writes `result.json` only from the main process. The result uses the
existing final EMA training loss and last evaluation metrics; selection requires
a finite validation-loss metric.

### `scripts/mup_sweep.py`

Replace the current width×LR local orchestrator with a Typer CLI whose
subcommands correspond to protocol phases. Each phase:

1. reads a base YAML or preceding `phase.json`;
2. generates a list of run configurations;
3. writes a fully resolved YAML for every run;
4. writes one `accelerate launch` command per run to `commands.txt`;
5. optionally executes those commands sequentially under `--local`; and
6. ranks results and records its recommendation.

The CLI also exposes a generic `analyze PHASE_JSON` command for jobs executed by
Slurm.

### `scripts/_mup_common.py`

Keep only shared constants, run/phase dataclasses, JSON load/write helpers,
float-list parsing, command construction, and ranking helpers. Phase definitions
remain in `mup_sweep.py` so the workflow can be understood in one place.

### `scripts/mup_pilot_run.py`

Remove it. `mup_run.py` replaces the one-cell responsibility without retaining
a compatibility wrapper.

## Minimal production-code changes

### Correct the 800M geometry

Change `src/oplm/configs/model/presets/800M.yaml` from 16 to 20 attention heads.
The resolved model must have `hidden_size=1280`, 40 layers, 20 heads, and
`head_dim=64`.

### Add depth-aware LR groups

Add two `TrainConfig` fields:

```yaml
train:
  mup_depth_lr_exponent: 0.0
  mup_depth_reference_layers: 24
```

Validate that the exponent is finite and nonnegative and that the reference
depth is at least one.

For a model with `L` layers, parameters whose names begin with
`oplm.backbone.layers.` receive:

```text
depth_lr_mult = (mup_depth_reference_layers / L) ** mup_depth_lr_exponent
```

All other parameters receive `1.0`. Compose this multiplier with the existing
width-aware μP multiplier.

Muon parameter construction may create a block and non-block group, each with
its effective LR. AdamW adds the depth multiplier to its existing
`(weight_decay, lr_mult)` bucket key. When the exponent is zero, group structure
and effective LRs remain equivalent to the current behavior.

No other trainer or model instrumentation is added.

## Base configuration contract

Every phase takes `--config BASE.yaml` for its first invocation. The YAML owns:

- training and validation datasets;
- production architecture features;
- per-process microbatch size;
- optimizer settings, including `weight_decay=0.01`;
- precision, compilation, and activation checkpointing;
- evaluation cadence and validation dataset; and
- ordinary logging/checkpoint settings.

Generated runs force only the experiment-owned values:

- model preset dimensions;
- `model.mup_enable=true`;
- `model.mup_base_width=768`;
- `model.mup_output_mult`;
- LR and depth-LR fields;
- seed;
- global-batch-derived accumulation;
- maximum, warmup, and stable steps;
- `scheduler=wsd_linear`; and
- a unique output directory and run name.

For every pilot, set:

```text
stable_steps = max_steps - warmup_steps
```

This produces linear warmup followed by a constant LR with no decay.

The base config must yield exactly one `eval/*/loss` metric or the phase command
must receive `--metric`. There is no training-loss fallback for selection.

## Batch and launcher contract

The authoritative batch input is global examples, not nominal tokens:

```text
gradient_accumulation_steps =
    global_examples / (train.batch_size * world_size)
```

The generator validates that this value is a positive integer. Defaults are:

- 2,048 examples for approximately 1M total tokens;
- 8,192 examples for approximately 4M total tokens; and
- eight processes for local execution.

`--num-processes` denotes the total data-parallel world size used to construct
the run YAML and command. `--accelerate-config` optionally supplies the
Accelerate config file used by both local commands and emitted commands.

Local mode uses `subprocess.run(..., check=True)` for each command in order. It
stops on the first failed command and preserves completed run artifacts. Remote
mode only writes `commands.txt`; SUNK/Slurm owns allocation and submission.

## Artifacts and handoff

Each phase directory has this shape:

```text
phase.json
commands.txt
runs/
  <run-id>/
    run.yaml
    result.json
    output/
```

`run.yaml` is a fully resolved standard OPLM config. `output/` is normal Trainer
output. No additional provenance bundle is created.

`phase.json` uses one versioned shape:

```json
{
  "version": 1,
  "phase": "coarse",
  "metric": "eval/heldout/loss",
  "source": null,
  "runs": [
    {
      "id": "lr-0.01",
      "config": "runs/lr-0.01/run.yaml",
      "result": "runs/lr-0.01/result.json",
      "params": {"lr": 0.01, "seed": 42}
    }
  ],
  "ranking": [],
  "selected": []
}
```

Paths are relative to the phase directory. Generation fills `runs`; analysis
fills `ranking` and `selected` in the same file. A subsequent phase accepts
`--from PREVIOUS/phase.json` and uses the selected parameter dictionaries.
Phase-specific CLI overrides may replace automatic selections.

## Phase definitions

### `smoke`

- Model: 170M.
- Global examples: 2,048.
- Seed: 42.
- LR: `0.0025,0.01,0.04`.
- Output multiplier: `1.0`.
- Schedule: 1,000 total steps, 100 warmup, 900 stable.

The low and middle LR must produce finite validation loss. A failed `0.04` run
is recorded and that exact value is excluded from the default coarse grid. This
phase does not otherwise rank candidates.

### `coarse`

- Model: 170M.
- Global examples: 2,048.
- Seed: 42.
- LR grid: `0.0025,0.004,0.0063,0.01,0.016,0.025,0.04`, minus an upper smoke
  failure.
- Output multiplier: `1.0`.
- Schedule: 10,000 total steps, 5,000 warmup, 5,000 stable.

Rank directly by validation loss. Select the winner and its available immediate
LR neighbors. If the winner is at the grid boundary, record that expansion is
required and do not select candidates for refinement.

### `refine`

- Model: 170M.
- Global examples: 2,048.
- Seed: 42.
- LR: the three values selected by `coarse`.
- Output multiplier: `0.5,1.0,2.0`.
- Schedule: 20,000 total steps, 5,000 warmup, 15,000 stable.

Run the local 3×3 product grid and select the best two LR/output pairs.

### `replicate`

This reusable phase accepts a prior `refine` or `bridge` result. For each
selected candidate it reuses the prior seed-42 result and schedules missing
seeds 43 and 44 with otherwise identical settings. Rank candidates by mean
validation loss over the three seeds and select the best two finite candidates.

### `transfer`

- Inputs: top two replicated LR/output candidates.
- Global examples: 2,048.
- Models and default total steps: 400M/10,000, 800M/20,000, 1B/10,000.
- Warmup: 5,000 steps for each model; the remainder is stable.
- Depth exponent: `0.0,0.25,0.5`.
- Seed: 42.

Rank every LR/output/exponent combination within each model size and sum its
three model-specific ranks. A combination missing any model result is
ineligible. Select the best two complete combinations. Ties prefer lower base
LR.

### `bridge`

- Model: 170M.
- Global examples: 8,192.
- Input: the top-ranked transferred base configuration by default; an explicit
  candidate override may select the runner-up instead.
- LR multiplier: `0.7,1.0,1.4,2.0`.
- Seed: 42.
- Schedule: 10,000 total steps, 5,000 warmup, 5,000 stable.

Rank directly by validation loss and select the top two candidates. Run
`replicate` on this phase before confirmation.

### `confirm`

- Model: 800M.
- Global examples: 8,192.
- Input: top one or two replicated bridge candidates, including the selected
  depth exponent.
- Seed: 42.
- Schedule: 10,000 total steps, 5,000 warmup, 5,000 stable.

Rank directly by validation loss. The best finite result is the final production
configuration.

### `scale`

- Input: confirmed production configuration.
- Default presets: `50M,170M,400M,800M,1B`.
- Global examples: 8,192.
- Seed: 42.
- Schedule: 100,000 total steps, 5,000 warmup, 95,000 stable.

This phase generates or executes winner-only scaling runs. It reports results
but performs no additional hyperparameter selection.

All grids, model lists, seeds, step counts, and warmup counts have direct CLI
overrides. Defaults implement the approved protocol.

## Selection and failure behavior

- A result is eligible only when the command completed and the configured
  validation loss exists and is finite.
- Missing result files and non-finite metrics are listed as failed cells.
- Local execution stops on the first nonzero command; remote analysis may still
  summarize completed cells.
- A selection phase refuses to populate `selected` when it lacks the required
  candidates or complete cross-model results.
- Single-run phases rank raw validation loss.
- Replication ranks mean validation loss across seeds.
- Transfer sums within-model ranks across model sizes.
- Exact ties prefer the lower base LR.
- There is no automatic retry or grid expansion.

## Testing strategy

Tests cover only behavior that could invalidate the LR protocol:

1. The 800M preset resolves to 20 heads and head dimension 64.
2. Generated run YAML retains the production architecture and optimizer from the
   base config while applying only phase-owned overrides.
3. Pilot schedules resolve to warmup followed by a stable plateau with no decay.
4. Global-example/world-size calculations produce exact accumulation or reject
   a non-integral combination.
5. Depth scaling changes block LRs but not non-block LRs under Muon and AdamW.
6. `mup_depth_lr_exponent=0` preserves current effective LRs and group coverage.
7. Each phase produces the expected parameter grid and command count.
8. Analysis rejects missing/non-finite validation metrics and implements each
   phase's ranking rule.
9. `--from` carries selected values into the next phase and explicit overrides
   replace them.
10. Local mode executes commands sequentially and stops on failure.
11. One tiny real-data training run writes a main-process result through
    `mup_run.py`.

Existing μP initialization, coordinate-check, optimizer, scheduler, and trainer
tests remain in place and are adjusted only for renamed/removed script surfaces.

## Documentation changes

- Update `docs/LR_SWEEP.md` command examples to the phase CLI and remove
  requirements intentionally excluded by this minimal design.
- Update `docs/MUP.md` to point production LR tuning to `LR_SWEEP.md` and remove
  obsolete `mup_pilot_run.py` examples.
- Update `docs/CONFIG.md` for the two depth-LR training fields and corrected
  800M head count.

## Acceptance criteria

- A user can generate every phase from one base YAML and a preceding
  `phase.json`.
- `--local --num-processes 8` runs all cells in a phase sequentially through
  `accelerate launch` and writes a recommendation.
- The emitted `commands.txt` can be queued unchanged inside a Slurm allocation.
- Generated cells use the full production architecture, Muon configuration,
  fixed weight decay, correct global batch, and warmup-stable schedule.
- LR/output, depth, and batch corrections flow into the final 100k scaling
  configs without manual transcription.
- No out-of-scope instrumentation or workflow infrastructure is introduced.

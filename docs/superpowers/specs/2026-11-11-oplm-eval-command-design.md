# `oplm eval` command — design

Issue: #57. Date: 2026-11-11.

## Intent

Score a checkpoint's `hf/` export on any configured eval task outside training, with one
command. Today this is done with one-off cluster scripts that call task classes directly
(`mup_redo/ab_cdr3_eval/run_eval.py`, `mup_redo/coevo_eval/run_eval.py`); both are the same
loop with a different task class. The command replaces both.

## Goals

- `oplm eval <ckpt>... [--config <yaml>]` runs every task in the config's `data.eval` block
  once per checkpoint and prints/writes per-task metrics.
- Numbers match the in-training eval for the same checkpoint and config.
- Covers every registered MLM-model task: `sequence`, `structure` (including `coevolution:`),
  `antibody_cdr3`, and the variant-scoring tasks, with no per-task code in the command.
- One config can score checkpoints of different architectures (OPLM, stack2, ESM-C).

## Non-goals

- `fold_monomer` (needs `OplmForFolding`, a different loader).
- Slurm submission (`oplm slurm eval`); a shell loop over `sbatch` is enough for now.
- W&B logging and cross-checkpoint summary tables; JSONL plus `jq` covers it.

## Anti-goals

- A second eval config format. The command reads the same `data.eval` block training does.
- Any change to task classes or to in-training eval behavior.

## Interface

```
oplm eval <ckpt>... [--config <yaml>] [--only name,name] [--output results.jsonl]
                    [--precision no|bf16|fp16] [--override KEY=VALUE]...
```

- `ckpt`: one or more checkpoint directories (containing `hf/`), or anything
  `load_model_for_inference` accepts.
- `--config`: a training config or an eval-only YAML; only `data.*` is read. When omitted,
  falls back to the checkpoint's own `config.yaml` (same rule as `oplm encode`, via
  `resolve_inference_config`), resolved per checkpoint.
- `--only`: comma-separated task names to run. An unknown name is an error.
- `--output`: append JSON lines to this file (in addition to stdout).
- `--precision`: accelerate mixed-precision mode. Defaults to the resolved config's
  `train.mixed_precision` (itself `bf16` by default), so in-loop precision is matched.
- `--override KEY=VALUE` (repeatable, as in `oplm encode`): config overrides, e.g. to repoint
  a `path`. Not positional, because the positional slot is the checkpoint list.

Errors, before any model load: no `data.eval` entries after filtering; unknown `--only` name.

## Output

One JSON line per (checkpoint, task), main process only:

```json
{"ckpt": "<path>", "task": "<entry name>", "seconds": 12.3, "<metric>": 0.123, ...}
```

Metric keys are the task's bare names (no `eval/<name>/` prefix). Lines are flushed as each
task finishes so a killed job keeps partial results.

## How it works

In `src/oplm/cli.py`, roughly:

```python
for ck in ckpts:
    cfg = resolve_inference_config(ck, config_path=config, overrides=overrides)
    entries = parse_eval_configs(cfg.data.eval, ScheduleSpec("steps", 1))  # `every:` ignored
    model = load_model_for_inference(ck, cfg, device=acc.device)
    cfg.model = model.config          # export's config wins over the YAML's model block
    for entry in entries:
        task = get_eval_task_class(entry.type)(entry, cfg)
        with acc.autocast():
            metrics = task.evaluate(model, acc)
```

- Bypasses `Evaluator`, which exists for scheduling against `train.eval_every`.
- Tasks are rebuilt per checkpoint because they read `cfg.model.max_position_embeddings`.
  Known ceiling: structure sets and coevolution maps reload per checkpoint; cache if slow.
- Tasks are already rank-aware, so `accelerate launch oplm eval ...` is expected to work
  multi-GPU with no extra code. Verify once; not a blocker for single-GPU use.

## Testing

- One fast test: tiny model + existing eval fixtures; the CLI's emitted metrics equal
  `task.evaluate` called directly, and `--only` with an unknown name exits non-zero.
- Acceptance (manual, cluster): one checkpoint's `oplm eval` numbers match its in-training
  eval numbers at the same step.

## Docs

Add a short "Evaluating a checkpoint outside training" section to `docs/evaluation.md`, and
the command to the CLI list in `AGENTS.md`.

## Decisions made without explicit sign-off (Claude's calls)

- JSONL only, no W&B write-back.
- Config falls back to the checkpoint's `config.yaml`, per checkpoint.
- Bypass `Evaluator`; rebuild tasks per checkpoint.
- Unknown `--only` names error.
- `seconds` included in each record.

## Left to the builder

Exact typer option declarations, the emit helper, and where the fixture model for the test
comes from (reuse `tests/eval/conftest.py`).

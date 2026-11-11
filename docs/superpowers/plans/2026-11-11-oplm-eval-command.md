# `oplm eval` Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `oplm eval <ckpt>... [--config <yaml>]` scores checkpoints' `hf/` exports on every task in the config's `data.eval` block and emits one JSON line per (checkpoint, task).

**Architecture:** One typer command in `src/oplm/cli.py`. It reuses `resolve_inference_config`, `load_model_for_inference`, `parse_eval_configs` and the eval task registry; it bypasses `Evaluator` (scheduling only). No new modules, no task changes.

**Tech Stack:** typer, accelerate, existing `oplm.eval` tasks.

**Spec:** `docs/superpowers/specs/2026-11-11-oplm-eval-command-design.md`

## Global Constraints

- Reads the existing `data.eval` schema; no second config format.
- No change to task classes or in-training eval behavior.
- Output: one JSON object per line with keys `ckpt`, `task`, `seconds`, then bare metric names; main process only; flushed per task.
- Precision defaults to the resolved config's `train.mixed_precision`; `--precision` overrides.
- `ruff check src/`, `ruff format` on touched files only, `VIRTUAL_ENV=$PWD/.venv ty check src/` clean. Run everything through `.venv/bin/`.

## Review Focus

- `--only` naming a task that is not in `data.eval`: exits non-zero and names the unknown task, before any model load.
- Config with empty/absent `data.eval`: exits non-zero with a clear message, before any model load.
- No `--config` and the checkpoint has no `config.yaml`: the existing `FileNotFoundError` text from `resolve_inference_config` surfaces as a clean CLI error, not a traceback.
- `--output` to an existing file: appends, never truncates.
- Multiple checkpoints with different `max_position_embeddings`: each task is built against that checkpoint's model config (tasks are rebuilt per checkpoint).

---

### Task 1: `oplm eval` command

**Files:**
- Modify: `src/oplm/cli.py`
- Test: `tests/test_cli_eval.py` (create)

**Interfaces:**
- Consumes: `oplm.inference.resolve_inference_config(model_path, *, config_path, overrides) -> OplmConfig`; `oplm.inference.load_model_for_inference(model_path, cfg, *, device)`; `oplm.data.config.parse_eval_configs(raw, default_schedule)`; `oplm.eval.registry.get_eval_task_class(type_name)`; `oplm.config.ScheduleSpec("steps", 1)`.
- Produces: CLI `oplm eval CKPT... [-c YAML] [--only a,b] [--output/-o FILE] [--precision no|bf16|fp16] [--override KEY=VALUE]...`. Overrides use `--override` (as `encode` does) because the positional slot is the checkpoint list.

- [ ] **Step 1: Write the failing tests** in `tests/test_cli_eval.py`. Fixture: a tiny `OplmForMaskedLM` (`max_position_embeddings=256`) saved with `save_pretrained(tmp/ckpt/hf)`, and an eval YAML whose `data.eval` has one `antibody_cdr3` entry `ab` pointing at `tests/data/fixtures/antibody/eval_gte2_head.parquet`.
  - `test_eval_matches_direct_task_call`: `oplm eval ckpt -c eval.yaml --precision no -o out.jsonl` exits 0; `out.jsonl` has one line with `ckpt`, `task == "ab"`, `seconds`; its metrics equal `AntibodyCdr3EvalTask(entry, cfg).evaluate(model, Accelerator(cpu=True))` (`pytest.approx`).
  - `test_eval_appends_output`: running twice leaves two lines.
  - `test_eval_falls_back_to_checkpoint_config`: with the YAML copied to `ckpt/config.yaml` and no `-c`, exits 0 and emits the `ab` line.
  - `test_eval_unknown_only_errors`: `--only nope` exits non-zero, output mentions `nope`.
  - `test_eval_no_tasks_errors`: a YAML with no `data.eval` exits non-zero, output mentions `data.eval`.
  - `test_eval_no_config_errors`: no `-c`, no `ckpt/config.yaml` exits non-zero with no traceback (`result.exception` is `SystemExit`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_cli_eval.py -q`. Expected: FAIL (no such command `eval`).
- [ ] **Step 3: Implement** `eval_cmd` (`@app.command(name="eval")`) in `src/oplm/cli.py` per the spec's "How it works" loop. Validation (config resolution, `--only`, empty entries) runs for all checkpoints before the first model load.
- [ ] **Step 4: Run** the test file, then `.venv/bin/python -m pytest -m "not slow" -q`, ruff and ty. Expected: all pass/clean.
- [ ] **Step 5: Commit** `feat(cli): oplm eval scores checkpoints on configured eval tasks (#57)`.

### Task 2: Docs

**Files:**
- Modify: `docs/evaluation.md` (*Offline evaluation*: replace the `> **Pending:**` marker with the command, its options, the output record, and the HSDP use)
- Modify: `AGENTS.md` only if it lists CLI commands (it does not today; skip otherwise)

- [ ] **Step 1:** Edit `docs/evaluation.md`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_docs.py -q`. Expected: PASS (the documented `oplm eval` resolves; Pending-marker guards still pass).
- [ ] **Step 3: Commit** `docs(evaluation): oplm eval for offline checkpoint scoring (#57)`.

### Acceptance (manual, cluster; not part of this branch)

One checkpoint's `oplm eval <ckpt>` numbers match its in-training eval numbers at the same step, and `accelerate launch` multi-GPU runs complete. This closes #57.

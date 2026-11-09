# `wsd_sqrt` Decay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `wsd_sqrt` LR scheduler (decay multiplier `1 - sqrt(progress)`), run one paired sqrt-vs-linear decay branch on the #58 overfit proxy, and record the production decay shape.

**Architecture:** One new branch in `get_schedule_fn`'s decay expression plus the name in `_VALID_SCHEDULERS`. The experiment reuses the cluster's `overfit.sh decay` path with a new `SCHED` env var. Nothing else in the trainer or checkpointing changes.

**Tech Stack:** Python 3.11, PyTorch `LambdaLR`, pytest, slurm via `oplm slurm`, W&B.

**Spec:** `docs/superpowers/specs/2026-11-07-wsd-sqrt-decay-design.md`

## Global Constraints

- `wsd_linear`, `wsd_cosine`, `warmup_linear` and `warmup_cosine` must stay bit-identical.
- The only intended difference between the sqrt arm and `decay140k-u25` (W&B `01wo8hzi`) is `train.scheduler`: same `branch_from` checkpoint-140000, 140k → 175k, uniref70 0.11 rows + `sample_members`, 4 nodes, LR 0.003, `min_lr` 0.0.
- The sqrt arm runs at the PR head commit from Task 1, built on `628d7bf`. Don't use a merge commit.
- Use `.venv/bin/python`, `.venv/bin/ruff` and `.venv/bin/python -m pytest`. Format only the files you touched.
- Branch: `feat/wsd-sqrt` (the spec is already committed there as `e682c1e`).

## Review Focus

1. Steps past the end of the decay (a resume with a larger `max_steps`, or an off-by-one) must hold at the floor, not go negative. The existing `min(progress, 1.0)` clamp covers it; Task 1 pins it.
2. A sqrt change must not move the other shapes. Task 1's parametrized test pins linear and cosine at the same point.
3. The decay must be monotone non-increasing. sqrt's steep start makes a sign slip easy. Task 1 pins it.
4. If `overfit.sh` is edited wrong, the "sqrt" run can silently train `wsd_linear`. Task 2 checks the generated config and the logged `train/lr` at a known step.
5. A misspelled `train.scheduler=wsd_sqr` must fail at config load, not mid-job. That's existing `_VALID_SCHEDULERS` validation, so no new test.

---

### Task 1: `wsd_sqrt` scheduler, tests, docs; open PR

**Files:**
- Modify: `src/oplm/training/optim.py:291-339` (`get_schedule_fn` and its docstring)
- Modify: `src/oplm/config.py:20` (`_VALID_SCHEDULERS`)
- Modify: `src/oplm/configs/train/base.yaml:61` (scheduler comment)
- Modify: `docs/config.md:296`, `docs/training.md:275` and `:281` (*Decay* bullet: "linear, cosine or 1-sqrt"), `docs/testing.md:96` (G7 row)
- Test: `tests/training/test_optim.py` (fast), `tests/training/test_e2e_optim.py:28` (slow G7)

**Interfaces:**
- Produces: `get_schedule_fn("wsd_sqrt", warmup_steps, total_steps, min_ratio, stable_steps)`, the existing signature. Also the config value `train.scheduler: wsd_sqrt`.

- [ ] **Step 1: Write the failing tests** in `tests/training/test_optim.py` (import `get_schedule_fn`):

```python
@pytest.mark.parametrize(
    ("scheduler", "quarter"),
    [("wsd_linear", 0.75), ("wsd_cosine", 0.5 * (1 + math.cos(math.pi / 4))), ("wsd_sqrt", 0.5)],
)
def test_wsd_decay_shapes(scheduler: str, quarter: float) -> None:
    """Each WSD shape: plateau at 1, its own value a quarter into the decay, floor at the end."""
    r = 0.1
    fn = get_schedule_fn(scheduler, warmup_steps=0, total_steps=500, min_ratio=r, stable_steps=100)
    assert fn(0) == fn(99) == 1.0
    assert fn(200) == pytest.approx(r + (1 - r) * quarter)  # decay_steps=400, 100/400 = 0.25
    assert fn(500) == pytest.approx(r)
    assert fn(520) == pytest.approx(r)  # past the end: held at the floor
    decay = [fn(s) for s in range(100, 521)]
    assert all(a >= b for a, b in zip(decay, decay[1:]))
```

Also add `"wsd_sqrt"` to `_SCHEDULERS` in `tests/training/test_e2e_optim.py:28`.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/training/test_optim.py -k wsd_decay_shapes -v`
Expected: the linear and cosine cases PASS; `wsd_sqrt` FAILS (it falls through to linear, giving 0.775 instead of 0.55).

- [ ] **Step 3: Implement.** In `get_schedule_fn`, add `is_sqrt = scheduler_name.endswith("_sqrt")` and a `decay = 1.0 - math.sqrt(progress)` branch between cosine and linear (spec, *Part 1*). Add `wsd_sqrt` to the docstring's name list and to `_VALID_SCHEDULERS`. Make the four doc and comment edits listed under Files.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/training/test_optim.py -v && .venv/bin/python -m pytest tests/training/test_e2e_optim.py -k lr_trajectory -v`
Expected: all PASS, including `test_lr_trajectory_matches_schedule[wsd_sqrt]`.
Then: `.venv/bin/python -m pytest -q -m "not slow"`. Expected: no new failures.
Then: `.venv/bin/ruff check` and `.venv/bin/ruff format --check` on the touched `.py` files. Expected: clean.

- [ ] **Step 5: Commit, push, open the PR**

```bash
git add src/oplm/training/optim.py src/oplm/config.py src/oplm/configs/train/base.yaml \
  docs/config.md docs/training.md docs/testing.md tests/training/test_optim.py tests/training/test_e2e_optim.py
git commit -m "feat(optim): wsd_sqrt scheduler (1-sqrt decay) (#81)"
git push -u origin feat/wsd-sqrt
```

Open the PR against `main` with `Refs #81` (not `Closes`, since the experiment is still pending). Record `git rev-parse HEAD`: that is the run `REF` for Task 2.

---

### Task 2: Launch `decay140k-u25-sqrt` on the cluster

The script is outside the repo, at `~/projects/oplm/mup_redo/overfit.sh` on the login node (`ssh briney@sunk.6af0ac-eu-south-04a-briney-b200.coreweave.app`).

**Interfaces:**
- Consumes: the Task 1 head SHA (pushed to GitHub; the job pip-installs `oplm @ git+...@$REF`).
- Produces: run dir `runs/overfit-1of35/decay140k-u25-sqrt`, W&B run `overfit-decay140k-u25-sqrt` in `thebrineylab/oplm-170M-overfit`.

- [ ] **Step 1: Add `SCHED` to `overfit.sh`'s `decay` case.** Use `SCHED=${SCHED:-wsd_linear}`. When it isn't `wsd_linear`, append `-${SCHED#wsd_}` to `name` (so `dir` follows), and add `train.scheduler=$SCHED` to the `oplm slurm generate` overrides. With the default, the generated output must not change. Add one usage line to the header comment.

- [ ] **Step 2: Dry run and check**

Run: `cd ~/projects/oplm/mup_redo && SCHED=wsd_sqrt REF=<task1 sha> DRY=1 bash overfit.sh decay 140000 25`
Expected: dir `runs/overfit-1of35/decay140k-u25-sqrt/` is created. The generated job's overrides include `train.scheduler=wsd_sqrt`, `train.branch_from=.../stable/output/checkpoint-140000` and `train.max_steps=175000`. The install line pins `<task1 sha>`. Diff its `config.yaml` against `runs/overfit-1of35/decay140k-u25/config.yaml`: the only difference should be the install ref. Also run `DRY=1 bash overfit.sh decay 140000 25` with no `SCHED` and confirm it prints `skip ...: already submitted`, which shows the linear name is unchanged.

- [ ] **Step 3: Submit.** Re-run without `DRY`, with `NODES` at its default of 4. Tell the user the job ID: they set `admin_override` QOS themselves if it has to overlap the running branches.

- [ ] **Step 4: Verify the schedule on the live run.** Once step 148750 is logged (a quarter of the way into the decay), W&B `train/lr` should be about `0.003 × 0.5 = 0.0015`. Linear would give 0.00225 there. If it reads linear, cancel the job and fix Step 1.

---

### Task 3: Read out, record, close

**Interfaces:**
- Consumes: final (step 175k) evals of `01wo8hzi` (linear) and the Task 2 run.

- [ ] **Step 1: Pull both runs' 175k metrics from W&B.** Pull held-out CE (`uniref70_singletons`, `uniref70_gte2`, `deepclust70_*`), the train-probe evals, and P@L (pdb40, cameo, casp, pdb40_2048). Build the table: linear, sqrt, delta per metric, mean P@L delta, and u70/dc70 train-probe vs held-out gaps. Estimate P@L noise from the linear branch's last few structure evals, or from the stable run's late constant-LR evals (spec, *Noise*).

- [ ] **Step 2: Apply the spec's decision rule.** Use *Reading the result → Decision rule*. If the result is ambiguous, stop and propose the 252k pair to the user (`SCHED=wsd_sqrt ... decay 252000 25`). Don't launch it unasked.

- [ ] **Step 3: Post the table to #81** with W&B links: `board comment 81 --body - <<'EOF' ... EOF`.

- [ ] **Step 4: Update `docs/mup-lr.md` *Decay recipe*.** Settle the shape: one or two sentences of evidence plus a link to #81. Keep length fraction and LR floor as "Pending" (#58). If sqrt wins, add a line saying production decays use `train.scheduler: wsd_sqrt`.

- [ ] **Step 5: Commit to the same PR and finish**

```bash
git add docs/mup-lr.md
git commit -m "docs(mup-lr): production decay shape from the 140k sqrt-vs-linear branch (#81)"
git push
```

Change the PR body's `Refs #81` to `Closes #81`. The PR head now differs from the run `REF`; that's expected, since the change is docs only.

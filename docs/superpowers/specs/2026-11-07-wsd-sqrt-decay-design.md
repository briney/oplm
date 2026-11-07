# `wsd_sqrt`: 1-sqrt LR decay vs linear — design

Board card: #81. Date: 2026-11-07.

## Intent

The production decay shape is still open (`docs/mup-lr.md`, *Decay recipe*: "Pending: the
production decay shape, length fraction and LR floor"). Every OPLM decay so far is `wsd_linear`
to `min_lr: 0.0`, so there is no in-house evidence on shape. The literature says shape matters a
little:

- Hägele et al. 2024 (*Scaling Laws and Compute-Optimal Training Beyond Fixed Training
  Durations*): a 1-sqrt cooldown slightly beats linear in WSD.
- Bergsma et al. 2025 (*Straight to Zero*): linear-to-zero beats cosine, more so at high
  tokens/param.

Both are AdamW results; OPLM trains matrices with Muon. This work adds a `wsd_sqrt` scheduler
and runs one controlled sqrt-vs-linear branch on the #58 overfit proxy, so the production shape
is chosen on evidence from this model and this optimizer.

## Goals

1. `train.scheduler: wsd_sqrt` exists: warmup → stable → decay with multiplier
   `1 - sqrt(progress)`, floored at `min_lr / lr` like the other schedules.
2. One sqrt decay branch, paired with the existing linear branch `decay140k-u25`, with
   everything else identical.
3. The comparison (held-out CE and structure P@L) is recorded on #81, and the production decay
   **shape** is written into `docs/mup-lr.md`.

## Non-goals

- Decay **length fraction** and **LR floor**. Both stay as they are in the baseline (25% of the
  branch step, `min_lr: 0.0`). They are separate questions on the same "Pending" line.
- Other shapes (cosine, exponential, a general `decay_shape` knob). Cosine already exists and
  the literature puts it behind linear.
- `warmup_sqrt` (no stable phase). Only WSD runs decay this way.
- Multiple seeds. See *Noise* below for why a single pair is enough to start.

## Anti-goals

- **A confounded pair.** If the sqrt branch runs a different oplm commit, data mix, data
  position, node count or eval set from the linear one, the result is worthless at an expected
  effect of about 0.01 CE. The only intended difference is `train.scheduler`.
- **Ranking mid-decay.** sqrt sits below linear in LR for the whole decay, so its loss leads
  early by construction. Compare only at the branch end (step 175k).
- **Changing what existing names do.** `wsd_linear`/`wsd_cosine` stay bit-identical, so every
  past run and checkpoint stays reproducible.

## Part 1: code

All in the repo; one small PR.

**`src/oplm/training/optim.py`, `get_schedule_fn`.** Add the sqrt branch to the decay
expression:

```python
is_sqrt = scheduler_name.endswith("_sqrt")
...
if is_cosine:
    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
elif is_sqrt:
    decay = 1.0 - math.sqrt(progress)
else:
    decay = 1.0 - progress
```

`is_wsd` already keys on the `wsd_` prefix, so `wsd_sqrt` gets the stable phase for free. Update
the docstring's list of names.

**`src/oplm/config.py`.** Add `"wsd_sqrt"` to `_VALID_SCHEDULERS`.

**Docs/comments** that list the scheduler names:

- `src/oplm/configs/train/base.yaml:61` (the `scheduler` comment)
- `docs/config.md:296` (`train.scheduler` row)
- `docs/training.md:275` and the *Decay* bullet below it ("linear, cosine or 1-sqrt")
- `docs/testing.md:96` (G7 row)

**Tests.**

- `tests/training/test_optim.py` (fast): `wsd_sqrt` with warmup 0, stable S, decay D,
  `min_ratio` r: multiplier is 1.0 through the plateau, `r + (1-r)·0.5` at a quarter of the way
  through the decay (sqrt(0.25) = 0.5), and `r` at the final step. Use a nonzero `r` so the
  floor is checked too.
- `tests/training/test_e2e_optim.py` (slow, G7): add `"wsd_sqrt"` to `_SCHEDULERS`; the existing
  test asserts the logged `train/lr` matches `get_schedule_fn`.

Nothing in checkpointing changes: `LambdaLR` stores only step counters, and `branch_from` already
takes the schedule from the branch's config (`src/oplm/config.py`, `branch_from` comment).

## Part 2: experiment

**Test bed.** The #58 1/35 overfit proxy: 170M, LR 0.003, stable run
`runs/overfit-1of35/stable` (W&B `thebrineylab/oplm-170M-overfit/taqdsivi`), driven by
`~/projects/oplm/mup_redo/overfit.sh` on the cluster login node.

**Baseline (exists).** `decay140k-u25`, W&B `01wo8hzi`: `branch_from` stable
`checkpoint-140000`, 140k → 175k (25% decay), `wsd_linear` to 0, uniref70 0.11 rows
(~25% tokens) with `sample_members` on, deepclust70 the rest, oplm `REF=628d7bf`, 4 nodes.
Final at 175k: P@L pdb40 .375 / cameo .401 / casp .327 / pdb40_2048 .373; held-out CE
u70 2.048 / 2.036 (singletons / gte2), dc70 2.141.

**New arm.** `decay140k-u25-sqrt`: identical except `train.scheduler=wsd_sqrt`.

- `overfit.sh` gets a `SCHED` env var (default `wsd_linear`). When set to anything else it adds
  `train.scheduler=$SCHED` to the `decay` command and appends `-${SCHED#wsd_}` to the run
  name/dir, e.g. `SCHED=wsd_sqrt REF=<sha> bash overfit.sh decay 140000 25`. The linear runs'
  names and dirs don't change.
- **Commit (`REF`).** The cluster pip-installs oplm at `REF`. The linear arm ran `628d7bf`, which
  is also current `main`. Branch the PR from `628d7bf` and run the sqrt arm at the PR's head
  commit, so the oplm diff between arms is only this change. If `main` moves before the run,
  still use the PR head, not the merge commit. Merge after the run or in parallel; it doesn't
  matter, since the run doesn't depend on the merge.
- **Same node count (4).** Accumulation is derived from the node count; keep it equal so the
  per-microbatch data split matches the baseline.
- `branch_from` restores the data position, so both arms see the same batches in the same order.
  The comparison is paired: same start weights, same optimizer state, same data, different LR
  path.

**Cost.** 35k steps on 4 nodes, the same as each 140k anneal-mix branch (u50/u65 are running
now). It needs `admin_override` QOS like the others if it is to overlap them.

## Reading the result

**Primary: held-out CE at step 175k** on `uniref70_singletons`, `uniref70_gte2` and
`deepclust70_*`. Fixed eval sets plus a paired run make this the low-noise metric. Expected
effect from the literature: about 0.01 lower for sqrt.

**Secondary: structure P@L at 175k** (pdb40, cameo, casp, pdb40_2048). This is what the model
is judged on (see the looping probe and #58 mix decision), but it is noisier. Report the deltas
per set and their mean.

**Also record:** the train-probe vs held-out gap per source at 175k, since a lower final LR
could change how much the decay widens the u70 gap (+.213/+.370 under linear).

**Noise.** There's one seed per arm. Because the pair shares start point and data, CE differences
of a few thousandths are meaningful; P@L noise is unmeasured. A reasonable proxy: the
spread of P@L across the last few structure evals of the linear branch, if more than one was
logged near the end, or else across the stable run's late evals at constant LR.

**Decision rule (my call; adjust before running if you disagree).**

- sqrt lower held-out CE on both sources **and** mean P@L delta ≥ −0.005 → production shape is
  `wsd_sqrt`.
- Otherwise → stay `wsd_linear` (the status quo; no reason to move without a win).
- If CE improves but P@L drops by more than 0.005, or the signs disagree between sources,
  call it ambiguous. Then run the second pair at 252k (`decay252k-u25` linear is running now; add
  `SCHED=wsd_sqrt ... decay 252000 25`) before deciding. Don't run it pre-emptively.

## Done when

- `wsd_sqrt` merged (fast + G7 tests passing).
- `decay140k-u25-sqrt` finished; a #81 comment has the per-metric table (linear, sqrt, delta)
  with W&B links.
- `docs/mup-lr.md` *Decay recipe*: the "Pending" line names the decay shape as settled (with the
  evidence in one or two sentences and a link to #81), and keeps length fraction and floor
  pending.

## Left to the builder

- Exact W&B metric keys for the eval losses and P@L.
- Wording of the docs edits.
- How to estimate P@L noise from what's already logged.

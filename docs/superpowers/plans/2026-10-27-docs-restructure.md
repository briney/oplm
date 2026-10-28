# Documentation Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace OPLM's sprawling, partly stale docs with a README for people (pLM and OPLMFold as
equals), an ordered reproduce path, and one current doc per topic, written so an agent can act from
them.

**Architecture:** Each topic doc absorbs its predecessors plus the matching `OVERVIEW.md` sections,
then the old files are deleted and every reference to them is rewritten in the same task. One small
test file (`tests/test_docs.py`) is the guard: links and anchors resolve, retired doc names are
gone, `oplm` commands in the docs exist, and config fields in the docs are real. Each task adds the
names it retires, so the suite stays green task by task.

**Tech Stack:** Markdown, pytest, typer `CliRunner`, OmegaConf (config loading).

**Spec:** `docs/superpowers/specs/2026-10-27-docs-restructure-design.md`. Read it first. Its
rules, readers and "what counts as final" section are requirements of every task.

## Global Constraints

- Rule 1: one fact, one home. Other docs link to that home and don't restate it.
- Rule 2: docs in the tree describe `main` today. Deleted docs are cited at
  `https://github.com/briney/oplm/blob/7db9c1bf0155fc9b4c05c8b358b3cf4cd6648f2c/<path>`
  (main at the time of the design). Use exactly this SHA.
- Rule 3: docs point to configs instead of copying numbers. Name the YAML file and the field
  (`train.lr` in `src/oplm/configs/train/base.yaml`). Derived numbers state their formula.
- Rule 4: method docs use this section order: Recipe → Commands → What to check → How to change it
  → Why and what was ruled out.
- New doc filenames are lowercase: `docs/data.md`, not `docs/DATA.md`. `docs/FOLD.md` and
  `docs/fold/` keep their names. Only link fixes are allowed inside `FOLD.md`.
- Don't touch `docs/superpowers/**` (other than adding this plan), and don't touch `docs/fold/**`.
- New docs don't number their sections. References cite a heading by name ("training.md,
  *Checkpointing and resume*"), never "§14".
- Don't hardcode the package version anywhere in the README or docs.
- Pending content uses exactly this blockquote form, so readers and graders can find it:
  `> **Pending:** <what is missing> (waiting on <issue or run>).`
- Readers in priority order: embeddings users, OPLMFold users (stub only for now), trainers,
  contributors. No μP, FSDP, Accelerate or Slurm vocabulary in the README before the embeddings
  quickstart.
- Python commands use `.venv/bin/python`. The fast suite is `.venv/bin/python -m pytest -m "not slow" -q`.

## Review Focus

1. **A reader copies the embeddings quickstart and it errors** (API drift, or no published
   checkpoint yet). Expected: the snippet runs against `main` when given a locally saved tiny model.
   Pinned by Task 9 Step 4.
2. **An agent follows a doc link or `#anchor` that doesn't exist** (headings change in every task).
   Expected: every relative link and anchor in the tree docs resolves. Pinned by
   `test_relative_links_resolve` (Task 1).
3. **An agent runs an `oplm …` command from the docs that doesn't exist.** Expected: every `oplm`
   command line in a fenced block parses with `--help`. Pinned by `test_documented_cli_commands_exist`
   (Task 1).
4. **An agent sets a config field named in the docs that isn't real** (a renamed or removed knob).
   Expected: every backticked or `key=value` `model.*` / `train.*` / `data.*` / `fold.*` field
   exists. Pinned by `test_documented_config_fields_exist` (Task 1).
5. **Pending content reads as done.** Expected: each pending item listed in the spec's "Pending
   until the long runs finish" section appears with the exact Pending marker. Pinned by
   `test_pending_markers_present` (Task 11).

---

### Task 1: Docs guard test

**Files:**
- Create: `tests/test_docs.py`

**Interfaces:**
- Produces: `RETIRED: list[str]` at module top. Later tasks append the names they retire. Starts
  empty. Each entry is a literal substring (for example `"TRAIN.md"` or `"docs/perf/"`).
- Produces: `TREE_DOCS: list[Path]`, which is `README.md`, `AGENTS.md`, `docs/FOLD.md`, and every
  top-level `docs/*.md` whose name is lowercase. New docs are lowercase, so they're picked up
  automatically, and the old uppercase docs that are about to be deleted are never checked.

- [ ] **Step 1: Write the four tests**

```python
REPO = Path(__file__).resolve().parents[1]
RETIRED: list[str] = []  # tasks append retired doc names as they delete them
TREE_DOCS = [REPO / "README.md", REPO / "AGENTS.md", REPO / "docs" / "FOLD.md",
             *sorted(p for p in (REPO / "docs").glob("*.md") if p.name == p.name.lower())]
_SCAN = [*TREE_DOCS, *REPO.glob("src/**/*.py"), *REPO.glob("src/**/*.yaml"),
         *REPO.glob("tests/**/*.py"), *REPO.glob("configs/**/*.yaml"), REPO / "pyproject.toml"]

def _slug(heading: str) -> str:  # GitHub anchor slug
    s = re.sub(r"[^\w\- ]", "", heading.strip().lower().replace("`", ""))
    return s.replace(" ", "-")

def test_relative_links_resolve(): ...
    # every [text](target) in TREE_DOCS whose target is not http(s)/mailto:
    # path part resolves (relative to the doc's dir) to an existing file or dir;
    # if an #anchor is present and the target is .md, _slug(some heading) == anchor.
    # Collect all failures, assert not failures (message lists doc: target).

def test_no_references_to_retired_docs(): ...
    # for each file in _SCAN except this test file, each line not containing "/blob/":
    # assert no RETIRED name occurs. Report file:line.

def test_documented_cli_commands_exist(): ...
    # lines inside ``` fences in TREE_DOCS that start with "oplm " (optionally after "$ "):
    # argv = first token after oplm, plus a second token when the first is sweep|slurm|fold
    # and the second doesn't start with "-"; CliRunner().invoke(oplm.cli.app, [*argv, "--help"])
    # exit_code == 0.

def test_documented_config_fields_exist(): ...
    # regex r"(?:`|\s)(model|train|data|fold)\.([a-z_][a-z0-9_]*)(?=[`=.\[])" over TREE_DOCS,
    # skipping field names {"py", "yaml", "md", "json"};
    # train/data/fold → name in {f.name for f in dataclasses.fields(TrainConfig|DataConfig|FoldRunConfig)};
    # model → hasattr(oplm.model.OplmConfig(), name).
```

Write the bodies as the comments describe. Fill in the imports.

- [ ] **Step 2: Run, and fix what it finds in the current docs**

Run: `.venv/bin/python -m pytest tests/test_docs.py -v`
Expected: all four tests pass, or failures that show real rot that already exists today (for
example a broken anchor, or a documented field that was renamed). Fix those in the docs they name.
If a hit is a false positive of the regex, tighten the regex. Don't add exclusions for real rot.
The retired-names test passes trivially, since `RETIRED` is empty.

- [ ] **Step 3: Commit**

```bash
git add tests/test_docs.py docs README.md AGENTS.md
git commit -m "test: docs guard — links, retired names, documented commands and config fields"
```

### Task 2: Ship the production recipe config

`configs/scaling.yaml` already plays this role (`oplm sweep scale` merges into it, and
`tests/test_scaling_config.py` and `tests/sweep/test_scale.py` load it), but its content is stale
(`residual_gate: channel`, context 512, batch mode, old data mix). Fix it in place. Don't add a
second file. *(This is the spec's "production.yaml or similar"; keeping the name avoids rewiring
sweep code.)*

**Files:**
- Modify: `configs/scaling.yaml`
- Modify: `tests/test_scaling_config.py`, plus `tests/sweep/test_scale.py` if batch-mode
  expectations break

**Interfaces:**
- Produces: `configs/scaling.yaml`, the canonical recipe config that `reproduce.md`,
  `training.md` and `mup-lr.md` point at.

- [ ] **Step 1: Pull the long run's resolved config**

Source of truth: the W&B config of run `170M-lr0.0063-long` in project `oplm-170M-long` (entity
`thebrineylab`, host `forge.coreweave.com`; `wandb.Api()` works from the laptop). Save it to the
scratchpad. If W&B isn't reachable, stop and ask the user for that run's `base.yaml` from the
cluster.

- [ ] **Step 2: Write the failing test**

```python
def test_scaling_config_is_the_current_recipe():
    cfg = load_config(["--config", str(_SCALING_CONFIG), "--preset", "170M"])
    assert cfg.train.tokens_per_step == 4_194_304
    assert cfg.data.pack_sequences is True
    assert cfg.model.max_position_embeddings == 2048
    assert cfg.model.residual_gate == "none"
    assert cfg.train.weight_decay_to_init == 0.1
```

Run: `.venv/bin/python -m pytest tests/test_scaling_config.py -v`. Expected: FAIL.

- [ ] **Step 3: Rewrite `configs/scaling.yaml`**

Keep only the keys whose values in the W&B config differ from `load_config(["--preset", "170M"])`.
Drop preset dimension keys, `output_dir`, `wandb_run_name`, `config_path` and `seed`. Keep the
`slurm:` block, rewritten for token mode (`nodes`, `max_batch_tokens` or whatever the token-mode
schema uses; see `src/oplm/slurm/config.py`). Data paths become placeholders under
`/path/to/oplm-training-data/…`. Add these comments, verbatim in meaning:
- on `data.train`: `# PENDING(#61): paths become the published training-set layout.`
- on the mix fractions: `# Rows, not tokens (0.05 rows ≈ 11% tokens). PENDING(#58): production mix.`
- on `max_steps`/`stable_steps`/`warmup_steps`: `# 170M long-run horizon. PENDING(#55): production horizon.`
- on `lr` (state it explicitly even if it equals the default): `# Tuned at 4M tok/step for 24k steps. PENDING(#55): best LR falls with training length; production value comes from the LR-vs-length extrapolation (docs/mup-lr.md).`
Update the header comment so it no longer calls a batch-mode recipe "the reference".

- [ ] **Step 4: Run the config tests**

Run: `.venv/bin/python -m pytest tests/test_scaling_config.py tests/sweep/test_scale.py -v`
Expected: PASS. Where a `test_scale.py` assertion pinned old batch-mode values, update it to the
token-mode equivalent. Don't delete coverage.

- [ ] **Step 5: Commit**

```bash
git add configs/scaling.yaml tests/test_scaling_config.py tests/sweep/test_scale.py
git commit -m "config: scaling.yaml is the current production recipe (token mode, no gate, wdi 0.1)"
```

---

**Shared steps for Tasks 3–8 (the doc migrations).** Each of these tasks:

1. Writes the new doc from the listed sources. Keep every fact that is still true of `main`. Drop
   superseded material. Content from the listed `OVERVIEW.md` sections goes in only if the new doc
   doesn't already say it.
2. Deletes the retired files and appends their names to `RETIRED` in `tests/test_docs.py`.
3. Rewrites every reference to the retired names. Find them with
   `grep -rn '<NAME>' src tests configs pyproject.toml README.md AGENTS.md docs/*.md`. Map section
   numbers to the new heading names. These are comment- and string-only edits.
4. Runs `.venv/bin/python -m pytest tests/test_docs.py -v && .venv/bin/ruff check src tests`.
   Expected: PASS.
5. Commits with `docs: <new doc> replaces <old docs>`.

---

### Task 3: `docs/data.md`

**Sources:** `docs/DATA_TOOLING.md`, `docs/DATA_MANIFEST.md`, `OVERVIEW.md` §14–20.
**Retire:** `DATA_TOOLING.md`, `DATA_MANIFEST.md`.

**Must contain, in this order:**
- *Training data*: what the training set is (uniref70 + deepclust70), with the mix **as rows**
  (point at `configs/scaling.yaml`), and the rows→tokens conversion formula with its inputs
  (uniref70 129M rows ≈ 40B tokens; deepclust70 5.35B rows ≈ 730B tokens; one natural-mix epoch
  ≈ 183k steps at 4M tokens/step).
- *Download the exact training set*: Pending marker (waiting on #61).
- *How the training set was built*: a skeleton with these headings: Primary sources (link +
  version/date each), Filtering, Clustering, Shuffling (deepclust70 was shuffled; the published copy
  is the shuffled one), Final counts. Each heading gets a Pending marker (waiting on #61) unless the
  facts are already in the repo.
- *Evaluation data*: held-out sequence sets, plus the structure eval sets with their full current
  provenance (all of `DATA_MANIFEST.md`'s content, verbatim where it's measurements).
- *Dataset formats*, *Tokenizer*, *Masking*, *Truncation, padding, packing*, *Other modalities*:
  from `DATA_TOOLING.md`.
- *Why*: the late-loss finding, in one paragraph: cluster-ordered shards flatter train loss at
  epoch boundaries, so training data must be globally shuffled. Cite
  `LATE_LOSS_INVESTIGATION.md` at the pinned SHA.

### Task 4: `docs/architecture.md` and `docs/config.md`

**Sources:** `docs/MODEL_ARCHITECTURE.md`, `docs/CONFIG.md`, `OVERVIEW.md` §0–13 (architecture)
and §28 (config assembly).
**Retire:** `MODEL_ARCHITECTURE.md`, `CONFIG.md`.

- `architecture.md`: opens with the default architecture as the shipped defaults produce it
  (`src/oplm/configs/model/base.yaml`). Then the semantics contract (Canon, norm strategies, RoPE,
  residual scaling, μP hooks). Research toggles that are off by default are described as available,
  not recommended.
- `config.md`: CONFIG.md's field reference, carried over complete, including the `fold.*` fields.
  Its *Size presets* table must list all eight presets in `src/oplm/configs/model/presets/`.

### Task 5: `docs/evaluation.md`

**Sources:** `docs/EVAL_HARNESS.md`, `OVERVIEW.md` §21–27.
**Retire:** `EVAL_HARNESS.md`.

**Must contain:** tasks and metrics, during-training scheduling, the `data.eval` schema, and the
published structure eval sets (HF `brineylab/oplm-structure-evals`, tag `2026-10-12`; full pdb40 as
`pdb40.tar.gz`; `max_position_embeddings` needs: casp15_16 ≥ 820, full pdb40 ≥ 1026, `_256`
subsets ≥ 514). Note that HSDP runs can't do in-loop eval; link the standalone eval card (#57) with
a Pending marker for offline eval.

### Task 6: `docs/training.md` and `docs/slurm.md`

**Sources:** `docs/TRAIN.md`, `docs/SLURM.md`, `OVERVIEW.md` §29–32.
**Retire:** `TRAIN.md`, `SLURM.md`. (The Slurm material stays its own doc: combined with TRAIN.md
it would be about 1,500 lines.)

**`training.md` must contain:**
- Everything in TRAIN.md that's still true, with the WSD schedule, `branch_from` vs `init_from`,
  and fault tolerance kept intact.
- *Published checkpoints*: what each one contains (weights + HF export, optimizer state,
  scheduler, data cursor, RNG, resolved config), which dataset tag the cursors refer to, cadence and
  locations. Pending marker (waiting on #62).
- *Branching a decay off a checkpoint*: the `branch_from` recipe with both of the user's examples
  (50k decay off the 2.5M checkpoint; 20k off 1.5M), using the hypothetical 2.5M + 500k recipe and
  labeled as an example. The schedule overrides are `train.max_steps`, `train.stable_steps` and
  `train.branch_from`. Show the arithmetic.
- *Resuming on different hardware*: data-exact resume requires the same world size, worker count,
  per-rank batch size and seed (`train.resume_data_position` validates this and raises). On other
  hardware, `train.resume_data_position=false` restarts the epoch from row 0, so some rows are seen
  again. Link #63 with a Pending marker.
- The looped-stage section stays, as reference material that isn't on the reproduce path.

### Task 7: `docs/mup-lr.md`

**Sources:** `docs/MUP.md`, `docs/LR_SWEEP.md`, and the spec's "What counts as final" paragraph
(authoritative wherever LR_SWEEP.md disagrees).
**Retire:** `MUP.md`, `LR_SWEEP.md`, `LATE_LOSS_INVESTIGATION.md`, `docs/late-loss/`.

**Must contain (rule 4 order):**
- *Recipe*: μP + Muon (`muon_adjust_lr_fn=original` and why it's mandatory), base width 768, LR
  tuned once at the 170M proxy, then reused across width.
- *Commands*: the refresh funnel. Token mode (`--global-tokens 4194304`, context 2048, packed,
  flex). Smoke → coarse → refine at 170M. Branch every refine cell into a 20% linear decay tail and
  rank on **decayed** eval loss (`train.branch_from`; the harness's own replicate ranking is
  undecayed, so don't use it). Replicate the top candidates and decay them too. Confirm at 800M.
  Transfer and bridge phases exist in `oplm sweep` but are not part of the recipe; mention them in
  one line.
- *What to check*: the coordinate check as the correctness gate; decaying flipped the refine
  ranking (0.0063 beats 0.004 after decay); the optimum is bracketed (0.0063 vs 0.01 ≈ 0.002 apart).
- *LR vs training length*: the best LR falls as training gets longer (est. ~0.008 at 24k, ~0.0043
  at 60k, ~0.0033 at 120k, decayed). The production LR comes from extrapolating that trend.
  Pending marker for the long-run results and the production LR (waiting on #55).
- *Decay recipe*: Pending marker (waiting on #55).
- *Why and what was ruled out*: the residual gate was dropped (paired decayed runs: noise-level
  difference), and the AdamW comparison recipe lives in `esm-c.yaml`. Cite deleted docs at the
  pinned SHA.

### Task 8: `docs/testing.md` and repo-root cleanup

**Sources:** `docs/TESTING_E2E.md`, `docs/RECOVERY_TESTS.md`, `OVERVIEW.md` §34–35.
**Retire:** `TESTING_E2E.md`, `RECOVERY_TESTS.md`, `NOTES.md`, `yagni_assessment.md`,
`docs/perf/`, `TODOS.md`.

- `testing.md` gets the suite layout, how to run it, the E2E gallery, and the cluster recovery
  drills, plus the Build & Test commands moved out of the README's Development section (Task 11
  removes them from the README).
- `git mv TODOS.md docs/superpowers/plans/2026-08-12-fault-tolerant-training.md`. Rewrite the
  `TODOS.md` references in `tests/eval/conftest.py`, `tests/eval/test_structure_task.py`,
  `tests/training/test_fsdp2_muon_spike.py` and `tests/training/_fsdp2_muon_spike_worker.py` to
  that path.
- Delete `docs/perf/` (including `__pycache__`), `NOTES.md` and `docs/yagni_assessment.md`. The
  reference in `tests/model/test_packing.py` to `docs/perf/packing_check.py` becomes a pinned-SHA
  link.

### Task 9: `docs/inference.md`

**Sources:** the README's current Quick start, *Coming from ESM-C?*, *Command line*; `OVERVIEW.md`
§2 (public API) and §32 (CLI & inference).

**Files:**
- Create: `docs/inference.md`

- [ ] **Step 1: Write `docs/inference.md`**: loading (`from_pretrained`, Auto* classes), per-residue
  logits and embeddings, per-protein (pooled) embeddings, `oplm encode` to a file, `oplm info`, the
  ESM-C migration notes, batching and device tips. Checkpoint names get a Pending marker (waiting
  on #62). Snippets keep `brineylab/oplm-170M`-style names, labeled as placeholders until then.
  Add an *OPLMFold* line: structure prediction is coming, so link `FOLD.md` for developers.
- [ ] **Step 2: Run the docs guard.** `.venv/bin/python -m pytest tests/test_docs.py -v`. Expected: PASS.
- [ ] **Step 3: Write the snippet check** in the scratchpad (not committed): save a tiny
  `OplmForMaskedLM` (hidden 64, 2 layers, 2 heads) with its tokenizer via `save_pretrained` to a
  temp dir. Then run every Python snippet in `docs/inference.md` with the checkpoint name replaced
  by that dir.
- [ ] **Step 4: Run it.** Expected: every snippet runs, and the shapes in the comments match.
  Fix the docs, not the check.
- [ ] **Step 5: Commit** with `docs: inference.md`.

### Task 10: `docs/reproduce.md` and `docs/README.md`

**Files:**
- Create: `docs/reproduce.md`, `docs/README.md`

- [ ] **Step 1: Write `reproduce.md`** per the spec's "reproduce.md: the main path": five ordered
  steps (data → μP LR → stable → decay → eval). Each step has a few sentences, the command, and a
  link to its home doc. The commands use `configs/scaling.yaml`. Then *Starting from a published
  checkpoint*, a main path entering at step 4, which links to `training.md` *Branching a decay off
  a checkpoint* and *Resuming on different hardware* rather than restating them. Add one line:
  OPLMFold training gets its own path later. Add a Pending marker (waiting on #55) for the decay
  step and for the expected loss and P@L ranges that the long runs will supply.
- [ ] **Step 2: Write `docs/README.md`**: the index, grouped as Use the model / Reproduce or modify
  training / Reference / Contribute, one line per doc. Include `FOLD.md` under Reference as
  "developer architecture doc (fold work in progress)".
- [ ] **Step 3: Run the docs guard.** Expected: PASS.
- [ ] **Step 4: Commit** with `docs: reproduce path and docs index`.

### Task 11: README, AGENTS.md, delete OVERVIEW

**Files:**
- Modify: `README.md`, `AGENTS.md`, `tests/test_docs.py`
- Delete: `docs/OVERVIEW.md`

- [ ] **Step 1: Write the failing test** in `tests/test_docs.py`:

```python
PENDING = "> **Pending:**"

def test_pending_markers_present():
    expect = {  # doc → issue numbers whose Pending markers must appear in it
        "docs/data.md": ["#61"],
        "docs/training.md": ["#62", "#63"],
        "docs/mup-lr.md": ["#55"],
        "docs/inference.md": ["#62"],
        "docs/evaluation.md": ["#57"],
        "docs/reproduce.md": ["#55"],  # decay recipe + expected loss / P@L ranges
    }
    for doc, issues in expect.items():
        lines = [l for l in (REPO / doc).read_text().splitlines() if l.startswith(PENDING)]
        for issue in issues:
            assert any(issue in l for l in lines), f"{doc}: no Pending marker for {issue}"
```

Also append `"OVERVIEW.md"` to `RETIRED`. Run: expected FAIL, because `OVERVIEW.md` is still
referenced.

- [ ] **Step 2: Delete `docs/OVERVIEW.md`.** First confirm that every section went somewhere:
  §0–13 and §28 to Task 4, §14–20 to Task 3, §21–27 to Task 5, §29–32 to Tasks 6 and 9, §34–35 to
  Task 8. §1 Philosophy is condensed into `architecture.md`'s opening. §33 (bugs the rewrite fixed)
  and the appendix are history, so drop them. Rewrite the remaining references.

- [ ] **Step 3: Rewrite `README.md`** per the spec's README section, in this order:
  1. What this is: OPLM and OPLMFold, one plain sentence each, plus the "everything open" line.
  2. Install.
  3. Two quickstarts given equal weight: *Embed sequences* (≤ 15 lines, taken from
     `docs/inference.md` Step 1, placeholder name labeled) and *Predict a structure* (coming-soon
     stub).
  4. Models: two sections, each "coming soon".
  5. What you can do today, and what's coming: embeddings and logits, `oplm encode`; coming:
     standalone eval (#57) and structure prediction.
  6. Training and reproducing: one paragraph linking `docs/reproduce.md`, covering the compute
     reality and branching off published checkpoints.
  7. Docs (→ `docs/README.md`), citation, license.

  Remove Development, Releases and versions, and the long Configuration and Architecture
  sections. Architecture becomes one line with a link.

- [ ] **Step 4: Update `AGENTS.md`.** Add a `## Documentation` section with a pointer to
  `docs/README.md`, the four rules (one line each), "update the home doc in the same PR as the
  behavior change", and "superseded material is deleted and cited at a pinned commit; no archive
  folder". Fix the doc names in *Project Structure* and *Testing* (`TESTING_E2E.md` →
  `testing.md`, and so on).

- [ ] **Step 5: Full verification**

Run: `.venv/bin/python -m pytest tests/test_docs.py -v && .venv/bin/python -m pytest -m "not slow" -q && .venv/bin/ruff check src tests`
Expected: all pass. Then
`grep -rnE 'OVERVIEW|TRAIN\.md|SLURM\.md|MUP\.md|LR_SWEEP|DATA_TOOLING|DATA_MANIFEST|EVAL_HARNESS|CONFIG\.md|MODEL_ARCHITECTURE|TESTING_E2E|RECOVERY_TESTS|LATE_LOSS|TODOS\.md|NOTES\.md|docs/perf' src tests configs README.md AGENTS.md docs/*.md | grep -v /blob/`
should print nothing. Then rerun the Task 9 snippet check against the README quickstart.

- [ ] **Step 6: Commit** with `docs: README for embeddings and OPLMFold users; AGENTS.md documentation rules; drop OVERVIEW`.

# Documentation restructure — design

Date: 2026-10-27. Status: design, not yet planned.

## Intent

OPLM's docs are out of date and hard to navigate. `docs/` has about 7,400 lines that mix current
guides with dated investigations, raw evidence and finished plans. Several facts live in two places:
`OVERVIEW.md` repeats `CONFIG.md`, `DATA_TOOLING.md` and `EVAL_HARNESS.md`, and they point back at
it. The README says "v0.0.1" and "five sizes", though there are eight presets.

The rewrite should give people a friendly way in and give agents enough exact detail to act. An
agent working for a user should be able to run inference, retrain OPLM from scratch, or start from a
published checkpoint and try a different phase-2 (decay) strategy, using only the docs and the
shipped configs.

The restructure is happening now, while the 300k-step 170M runs finish, because the pLM training
setup must be ready for a large compute allocation that arrives soon.

## Readers, in priority order

1. **Embeddings users.** The likely majority, since production-scale training needs compute almost
   nobody has. They want sequences in and vectors out.
2. **OPLMFold users.** The structure predictor will live in this repo, and its audience will be
   roughly as large as the pLM's. It is not built yet, so it appears only as a first-class stub for
   now (see README below).
3. **Trainers.** People who reproduce OPLM from scratch, or start from a published checkpoint and
   change something, such as the decay phase. They must not feel lost, but they don't lead.
4. **Contributors**, through `AGENTS.md`.

Agents can act for any of these readers. Every doc past the README is written so that an agent can
follow it without guessing: exact commands, real config paths, and expected outputs to check
against.

## Goals

- A README that treats the pLM and OPLMFold as equals, with embeddings first in reading order and a
  clear path for trainers.
- One ordered "reproduce OPLM" path from data to eval, linking to one detailed doc per step.
- Method docs (μP and LR determination, data mix, schedule) that describe **only the final
  approach**, in enough detail to recreate it.
- Every fact has exactly one home. Superseded material is gone from the working tree.

## Non-goals

- Building any missing tooling (inference CLI, standalone eval, fold). The docs describe what
  exists on `main` and mark what is coming.
- Rewriting or moving any fold documentation (see "Untouched").
- Changing code, apart from updating comments that point at moved or deleted docs.

## Anti-goals: failure even if technically done

- Method docs that tell the story of superseded attempts, or present them as options.
- A README that mentions Accelerate, FSDP, μP or Slurm before the reader has embedded a sequence.
- Numbers copied out of configs that will drift away from the YAML.
- Docs that present unbuilt features as if they work.
- Keeping deleted material in an archive folder in the tree. Version control is the archive.

## Target layout

```
README.md            people first: what it is, install, two quickstarts, models, where next
AGENTS.md            contributor rules (role unchanged) + a new "Documentation" section
docs/
  README.md          index, organized by "I want to…"
  inference.md       load a checkpoint, logits, per-residue and pooled embeddings, `oplm encode`
  reproduce.md       ★ main path: data → μP LR → stable phase → decay → eval
  data.md            datasets, building them, mix weights (rows vs tokens), provenance
  mup-lr.md          final μP + LR method, including LR vs training length
  training.md        launching, checkpoints, resume, Slurm, continuing from a checkpoint
  evaluation.md      eval harness + structure eval sets
  config.md          field-by-field reference
  architecture.md    model semantics contract
  testing.md         dev-only: suite layout, E2E, cluster recovery drills
  FOLD.md, fold/     unchanged (fold work is paused; see "Untouched")
  superpowers/       specs and plans, unchanged
```

Filenames become lowercase, apart from the untouched fold files. The builder may split or merge
docs if a section grows past what one doc can hold. The rules below matter more than the exact
file list.

## Rules

1. **One fact, one home.** Other docs link to that home and don't restate it. Break `OVERVIEW.md`
   up into the topic docs, then delete it.
2. **Docs in the tree describe `main` today.** Superseded work is deleted. When a method doc's
   "why" section needs evidence, it links to the file on GitHub at a fixed commit
   (`https://github.com/briney/oplm/blob/<sha>/docs/…`), so the link survives the deletion.
3. **Docs point to configs instead of copying numbers.** Recipe docs name the shipped YAML
   (`src/oplm/configs/train/base.yaml`, `configs/model/presets/*.yaml`, `configs/data/base.yaml`)
   and the fields that matter. They don't transcribe values. Derived quantities (tokens per step,
   epochs, decay fraction) can be stated together with the formula and its inputs.
4. **Method docs use one section order:** recipe → exact commands → what to check along the way
   (expected loss or P@L ranges, sanity checks) → how to change it → why, and what was ruled out.
   An agent should be able to act on the first three sections alone.

## README.md

1. **What this is.** Two models from one open project: OPLM (protein language model: embeddings
   and logits) and OPLMFold (structure prediction on top of it), one plain sentence each. Plus one
   line on what sets it apart: weights, data and recipe are all open.
2. **Install.** One `pip` line. OPLMFold as an extra once it ships.
3. **Two quickstarts, given equal space.** "Embed sequences" (about 10 copy-and-paste lines: load,
   embed, get per-residue and pooled vectors), followed by a short "Coming from ESM-C?" note.
   "Predict a structure" is a clearly marked coming-soon stub until OPLMFold ships.
4. **Models.** One table with two sections (OPLM, OPLMFold). Until checkpoints are published, each
   section says "coming soon" rather than listing presets nobody can download.
5. **What you can do today, and what's coming.** Embeddings and logits work now, through both the
   Python API and `oplm encode` (per-residue embeddings to a file). Standalone eval (#57) and
   structure prediction are listed as coming, each linking to its doc once it exists. Don't promise
   a separate "inference CLI"; none is planned beyond `encode`. *(Design call.)*
6. **Training and reproducing.** One paragraph: everything is open, the path is
   `docs/reproduce.md`, full-scale training needs serious compute, and continuing from a published
   checkpoint (for example, trying a different decay) is much cheaper. One line saying OPLMFold
   training will get its own path later.
7. **Docs, citation, license.** Links to `docs/README.md`.

Development, versioning and release details move to `AGENTS.md` / `testing.md`. Architecture
shrinks to one line with a link. Don't hardcode a version number.

## docs/README.md (index)

Grouped by what the reader wants to do: use the model, reproduce or modify training, look up a
config field, contribute. One line per doc.

## reproduce.md: the main path

An ordered pipeline. Each step gets a few sentences, the command, and a link to the detailed doc:

1. Get or build the training and eval data (`data.md`).
2. Determine the LR with μP: proxy sweep, transfer, LR vs training length (`mup-lr.md`).
3. Run the stable phase at production scale (`training.md`, Slurm).
4. Run the decay phase (`training.md`, `mup-lr.md`).
5. Evaluate (`evaluation.md`).

It also has a "Starting from a published checkpoint" section. This is the same path, entered at
step 4: load a stable-phase checkpoint and run a different decay. It explains what must match the
original run (data stream position, optimizer state, schedule) and what can change.

## Where current content goes

| Current | Destination | Notes |
|---|---|---|
| `README.md` | `README.md` | Rewritten per the README section above |
| `docs/OVERVIEW.md` | split into topic docs, then deleted | Keep only what isn't already in its home doc |
| `docs/TRAIN.md` | `training.md`, `reproduce.md` | Includes the fault-tolerance and looped-stage sections, as reference |
| `docs/MUP.md` + `docs/LR_SWEEP.md` | `mup-lr.md` | Final method only; the coordinate check stays as a correctness gate |
| `docs/DATA_TOOLING.md` + `docs/DATA_MANIFEST.md` | `data.md` | Provenance stays exact; it supports the methods section |
| `docs/EVAL_HARNESS.md` | `evaluation.md` | Add the published structure eval sets |
| `docs/CONFIG.md` | `config.md` | `fold.*` fields stay |
| `docs/MODEL_ARCHITECTURE.md` | `architecture.md` | |
| `docs/SLURM.md` | `training.md` (or its own doc if too long) | Builder's call |
| `docs/TESTING_E2E.md` + `docs/RECOVERY_TESTS.md` | `testing.md` | |
| `docs/LATE_LOSS_INVESTIGATION.md`, `docs/late-loss/` | deleted | Cite at a fixed commit from `data.md` / `mup-lr.md` where relevant |
| `docs/perf/` | deleted | Benchmark scripts and sbatch files; results are already reflected in defaults |
| `docs/yagni_assessment.md` | deleted | |
| `NOTES.md` | deleted | Contains only "pending fixtures: none" |
| `TODOS.md` | moved to `docs/superpowers/plans/2026-08-12-fault-tolerant-training.md` | It's the August plan; four test comments reference it |

References to update: about 105 lines in 66 files under `src/`, `tests/` and `configs/` point at
old doc names, and many cite section numbers ("TRAIN.md §14", "DATA_TOOLING.md §4.5"). Update all
of them to the new file and section (or to a fixed-commit link for deleted docs). These are
comment- and string-only changes and are in scope. Treat this as its own mechanical step and grep
for every old filename before finishing. *(Design call.)*

Link fixes are allowed in `docs/FOLD.md`, since it is a current doc, but nothing else in it changes.
`docs/superpowers/` records are exempt: they're dated and keep their original links. *(Design call.)*

## What counts as "final"

The final approach is what the current defaults on `main` produce: `train/base.yaml` (μP + Muon,
the promoted recipe), the presets, and `data/base.yaml`.

The final LR method is the **October 2026 μP refresh**, not the batch-mode funnel `LR_SWEEP.md`
describes today. The refresh ran in token mode at 4M tokens/step (context 2048, packed, flex
attention). It went smoke → coarse → refine at the 170M proxy. Every refine cell was then branched
into a 20% linear decay tail and the cells were ranked on **decayed** eval loss (decaying flipped
the ranking to 0.0063). The top candidates were replicated and also decayed, and the result was
confirmed at 800M. Transfer and bridge were skipped. The harness's own replicate ranking is
undecayed and is not used. `mup-lr.md` documents this funnel as the recipe, including the
decay-branch ranking step. The coordinate check stays as the correctness gate. Source: the refresh
records summarized in project history; check details against W&B `thebrineylab/oplm-mup-4M`.

Features that exist but aren't part of the recipe (the `esm-c.yaml` AdamW
comparison, the looped stage, the research architecture toggles) stay documented in the reference
docs (`config.md`, `architecture.md`, `training.md`) and are kept off the reproduce path. When it's
unclear whether something is final or superseded, ask the user rather than guess.

## Pending until the long runs finish (briney/oplm#55, #58)

These sections get written with an explicit "pending" marker that says what result they're
waiting for:

- the decay recipe (shape, length fraction, LR) in `reproduce.md` / `mup-lr.md`;
- extrapolating the best LR against training length to production horizons in `mup-lr.md`;
- the production data mix for the stable and decay phases in `data.md` (#58);
- published checkpoint names (README models table, `inference.md`);
- the expected loss and P@L ranges in each "what to check" section. The reference runs are the
  170M long runs (#55). W&B links go in only if the project is made public; otherwise the docs
  state the numbers. *(Design call.)*

## Untouched

- All fold material: `docs/FOLD.md`, `docs/fold/`, and the fold specs and plans. Fold work is
  paused until the large pLM pre-trains are running, then resumes. `config.md` keeps its `fold.*`
  section, and the README carries only the OPLMFold stub. A user-facing `docs/fold.md` comes later.
- `docs/superpowers/` specs and plans (dated design records).

## AGENTS.md

Add a short "Documentation" section: a map of the docs (or a pointer to `docs/README.md`), the four
rules above, and "update the home doc in the same PR as the behavior change." Fix the doc links in
"Project Structure" to the new filenames.

## Left to the builder

- Wording, ordering inside each doc, and whether `SLURM.md` gets folded in or stays separate.
- How to split the work into PRs.
- Checking that commands and snippets run against `main`: `oplm --help` output, and the embeddings
  quickstart on CPU with a tiny randomly initialized model until real checkpoints exist.

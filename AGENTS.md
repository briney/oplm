# Agent Instructions for oplm

Open protein language model.

This file is the single source of truth for agent and contributor instructions
in this repository. Make all future updates here, not in `CLAUDE.md` (which
simply points back to this file).

## Build & Test Commands

```bash
pip install -e ".[dev]"                        # install (editable, with dev dependencies)
.venv/bin/python -m pytest -m "not slow" -q    # fast suite
ruff check src/                                # lint
ruff format src/                               # format
ty check src/                                  # type check (Astral's ty; [tool.ty] in pyproject.toml)
```

Run tests through `python -m pytest`, not bare `pytest`. The full suite, coverage and the E2E and
cluster drills are in [docs/testing.md](docs/testing.md).

> **Type checking uses `ty`, not mypy.** `ty` is pre-1.0/beta but handles this
> HuggingFace/torch-heavy codebase far better than mypy (which drowned in
> untyped-`transformers` stub noise). Framework-boundary diagnostics are
> suppressed inline with documented `# ty: ignore[<rule>]` comments. `ty check
> src/` must be clean.

## Architecture

- **src layout**: all package code lives under `src/oplm/`.
- **Build system**: hatchling (`pyproject.toml` only, no `setup.py`).
- **Testing**: pytest with tests in `tests/`.

## Project Structure

```
src/oplm/                      # main package (src layout)
├── model/                     # transformer model package (see docs/architecture.md)
│   ├── outputs.py             # LogitsConfig / LogitsOutput dataclasses (ESM-C-style API)
│   ├── norm.py                # LayerNorm / RMSNorm (fp32 internals) + make_norm factory
│   ├── masking.py             # pad-mask helpers, conv-input zeroing
│   ├── rope.py                # RoPE / partial RoPE applied to Q and K
│   ├── embedding.py           # token embedding + mean / CLS pooling
│   ├── ffn.py                 # SwiGLU / GEGLU / squared-ReLU feed-forward + make_ffn factory
│   ├── conv.py                # Canon depthwise 1D convolution (A/C/D in block; B on Q/K/V in attention)
│   ├── attention.py           # attention (SDPA + manual softmax for weights); hosts Canon-B convs on Q/K/V
│   ├── transformer.py         # OplmBlock + OplmStack
│   ├── configuration_oplm.py  # OplmConfig (PretrainedConfig)
│   ├── tokenization_oplm.py   # OplmTokenizerFast (33-token ESM-C-compatible)
│   └── modeling_oplm.py       # all public Oplm* model classes
├── fold/                       # structure prediction head: kernels, model, data, training, predict, eval/ (held-out metrics, `fold_monomer` task), evalsets/ (`oplm fold eval-sets`: CASP/CAMEO/PDB40 contact-eval sets); parity tests need OPLM_FOLD_FIXTURES (docs/FOLD.md, Milestone 1 through Milestone 2b)
├── slurm/                      # general Slurm job generation/submission (docs/slurm.md)
│   ├── config.py               # `slurm:` block schema, PhaseTable, resolve_batch_plan
│   ├── render.py                # JobSpec -> sbatch script text, submit.sh, job arrays
│   ├── submit.py                # sbatch/squeue subprocess calls
│   └── cli.py                   # `oplm slurm generate|submit|status`
└── sweep/                      # μP learning-rate sweep phases (docs/mup-lr.md), built on slurm/
    ├── phases.py                # phase generators (smoke/coarse/.../scale), ranking, analyze/status
    ├── coord_check.py           # μP coordinate-check CLI (`oplm sweep coord-check`)
    ├── run.py                   # trains one resolved sweep cell (`oplm.sweep.run`)
    ├── common.py                # shared phase-manifest / candidate-parsing helpers
    └── cli.py                   # `oplm sweep <phase>` command surface
tests/                         # pytest tests, mirrors src/ structure
```

There is no unpackaged top-level `scripts/` directory anymore. Its former `mup_sweep.py`,
`mup_run.py`, and `mup_coord_check.py` modules now live in `src/oplm/sweep/` as `phases.py`,
`run.py`, and `coord_check.py`, reached through the installed `oplm sweep <phase>` /
`oplm sweep coord-check` commands instead of an unpackaged module invocation.

## Code Style

- Python 3.11+, modern typing syntax (`X | Y`, `Self`, etc.).
- `from __future__ import annotations` in every file.
- Type hints on all function signatures.
- Google-style docstrings on public APIs.
- Ruff for linting and formatting (configured in `pyproject.toml`).
- Line length: 100 characters.
- No wildcard imports, no bare `except:`.

## Testing

- Mirror source layout: `src/oplm/foo.py` -> `tests/test_foo.py`.
- Use `pytest.fixture` for setup, `@pytest.mark.parametrize` for input variation.
- Mark slow tests with `@pytest.mark.slow`.
- Prefer real data over synthetic data in tests.
- End-to-end training-run tests (the slow, multi-step `Trainer` suite) and the cluster
  recovery drills are described in [docs/testing.md](docs/testing.md).

## Releases and versions

Git tags are the version source: `hatch-vcs` builds a clean checkout tagged `vX.Y.Z` as package
version `X.Y.Z`. To release, publish a GitHub release with a new `vX.Y.Z` tag on the desired
commit; the publishing workflow builds it and uploads it to PyPI. There is no version constant to
update in `pyproject.toml` or Python source, and docs never hardcode a version.

`oplm.__version__` reads the installed distribution metadata. Untagged commits get a development
version; after changing commits or tags, reinstall an editable checkout to refresh it. Build from a
Git checkout with tags available, or from a published sdist. GitHub's auto-generated source
archives are not sdists.

## Documentation

[docs/README.md](docs/README.md) is the map of the docs, grouped by what the reader wants to do.
The README is for people using the models; the docs under `docs/` hold the detail.

1. **One fact, one home.** Other docs link to that home and don't restate it.
2. **Docs describe `main` today.** Superseded material is deleted, not archived.
3. **Docs point to configs instead of copying numbers.** Name the YAML file and field; state the
   formula for derived numbers.
4. **Method docs use one section order:** Recipe, Commands, What to check, How to change it, Why
   and what was ruled out.

- Update the home doc in the same PR as the behavior change.
- Superseded material is deleted and, where evidence is needed, cited at a pinned commit
  (`https://github.com/briney/oplm/blob/<sha>/<path>`). There is no archive folder.
- Cite other docs by heading name ("training.md, *Checkpointing and resume*"), never by section
  number. Mark missing content with `> **Pending:** <what is missing> (waiting on <issue or run>).`
- `tests/test_docs.py` guards links, retired doc names, documented `oplm` commands, config fields
  and Pending markers.

## What Not To Do

- Don't add `# type: ignore` without a specific error code.
- Don't use `os.path` — use `pathlib.Path`.
- Don't put logic in `__init__.py`.
- Don't commit notebooks with output cells.

<!-- codebox:board:start -->
## Board

Deferred work for this repo lives on its GitHub Projects board (Triage, Planned, In Progress, Done). Use the `board` command (`board --help`) to keep it current:

1. **Work starts on a card:** move it to In Progress immediately: `board move N in-progress`.
2. **A PR is opened:** items it completes get `Closes #N` in the PR body (GitHub moves them to Done on merge). Items it only partly advances get `Refs #N`, are moved to In Progress, and get a short comment on what changed and what's left: `board move N in-progress --comment - <<'EOF'` (text, then `EOF`).
3. **Non-PR work** (cluster runs, investigations, dataset publishes): when a card's "Done when:" is met, close it with a comment linking the evidence (W&B run, HF tag, commit): `board move N done --comment - <<'EOF'` (text, then `EOF`).

Make these moves without asking, and report them in one line, e.g. `Board: #12 → Done (PR #49), #15 → In Progress`.

New ideas go in Triage (the `codebox-defer` skill does this). Check `board search TERMS` first; on an open match, `board comment N --body - <<'EOF'` instead of filing a duplicate. A card must stand on its own weeks later:

```
board add --title 'Short, specific title' --body - <<'EOF'
**Why:** what prompted it and what problem it solves.
**Context:** relevant files, commits, PRs, runs (W&B/HF links), constraints already known.
**Done when:** one concrete, checkable line.
EOF
```

`--body -` and `--comment -` read the text from stdin; otherwise the text is the flag's value.

Quote titles with single quotes and send bodies/comments through a quoted heredoc so backticks and $ aren't expanded.
<!-- codebox:board:end -->

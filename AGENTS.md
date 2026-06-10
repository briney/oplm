# Agent Instructions for oplm

Open protein language model.

This file is the single source of truth for agent and contributor instructions
in this repository. Make all future updates here, not in `CLAUDE.md` (which
simply points back to this file).

## Build & Test Commands

```bash
# install (editable, with dev dependencies)
pip install -e ".[dev]"

# run all tests
pytest

# run tests with coverage
pytest --cov=oplm

# skip slow tests
pytest -m "not slow"

# lint
ruff check src/

# format
ruff format src/

# type check (ty — Astral's checker; configured under [tool.ty] in pyproject.toml)
ty check src/
```

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
└── model/                     # transformer model package (see docs/MODEL_ARCHITECTURE.md)
    ├── outputs.py             # LogitsConfig / LogitsOutput dataclasses (ESM-C-style API)
    ├── norm.py                # LayerNorm / RMSNorm (fp32 internals) + make_norm factory
    ├── masking.py             # pad-mask helpers, conv-input zeroing
    ├── rope.py                # RoPE / partial RoPE applied to Q and K
    ├── embedding.py           # token embedding + mean / CLS pooling
    ├── ffn.py                 # SwiGLU / GEGLU / squared-ReLU feed-forward + make_ffn factory
    ├── conv.py                # Canon depthwise 1D convolution (A/C/D in block; B on Q/K/V in attention)
    ├── attention.py           # attention (SDPA + manual softmax for weights); hosts Canon-B convs on Q/K/V
    ├── transformer.py         # OplmBlock + OplmStack
    ├── configuration_oplm.py  # OplmConfig (PretrainedConfig)
    ├── tokenization_oplm.py   # OplmTokenizerFast (33-token ESM-C-compatible)
    └── modeling_oplm.py       # all public Oplm* model classes
tests/                         # pytest tests, mirrors src/ structure
```

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
- End-to-end training-run tests (the slow, multi-step `Trainer` suite) follow the
  plan in [docs/TESTING_E2E.md](docs/TESTING_E2E.md).

## What Not To Do

- Don't add `# type: ignore` without a specific error code.
- Don't use `os.path` — use `pathlib.Path`.
- Don't put logic in `__init__.py`.
- Don't commit notebooks with output cells.

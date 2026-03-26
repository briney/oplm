# Agent Instructions for oplm

## Architecture

- **src layout**: all package code lives under `src/oplm/`
- **Build system**: hatchling (pyproject.toml only, no setup.py)
- **Testing**: pytest with tests in `tests/`

## Conventions

- Type hints on all function signatures
- `from __future__ import annotations` in every file
- Google-style docstrings on public APIs
- Ruff for linting and formatting
- No wildcard imports, no bare `except:`

## Testing

- Mirror source layout: `src/oplm/foo.py` -> `tests/test_foo.py`
- Use `pytest.fixture` for setup, `@pytest.mark.parametrize` for input variation
- Mark slow tests with `@pytest.mark.slow`
- Prefer real data over synthetic data in tests

## What Not To Do

- Don't add `# type: ignore` without a specific error code
- Don't use `os.path` — use `pathlib.Path`
- Don't put logic in `__init__.py`
- Don't commit notebooks with output cells

# oplm

Open protein language model.

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

# type check
mypy src/
```

## Project Structure

```
src/oplm/         # main package (src layout)
tests/            # pytest tests, mirrors src/ structure
```

## Code Style

- Python 3.11+, modern typing syntax (`X | Y`, `Self`, etc.)
- `from __future__ import annotations` in every file
- Ruff for linting and formatting (configured in pyproject.toml)
- Line length: 100 characters
- Google-style docstrings

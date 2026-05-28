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
src/oplm/                      # main package (src layout)
└── model/                     # transformer model package (see docs/MODEL_ARCHITECTURE.md)
    ├── outputs.py             # LogitsConfig / LogitsOutput dataclasses (ESM-C-style API)
    ├── norm.py                # LayerNorm / RMSNorm (fp32 internals) + make_norm factory
    ├── masking.py             # pad-mask helpers, flex_attention mask_mod, conv-input zeroing
    ├── rope.py                # RoPE / partial RoPE applied to Q and K
    ├── embedding.py           # token embedding + mean / CLS pooling
    ├── ffn.py                 # SwiGLU feed-forward + make_ffn factory
    ├── conv.py                # Canon depthwise 1D convolution
    ├── attention.py           # dual-path attention (flex_attention + manual fallback)
    ├── transformer.py         # OplmBlock + OplmStack
    ├── configuration_oplm.py  # OplmConfig (PretrainedConfig)
    ├── tokenization_oplm.py   # OplmTokenizerFast (33-token ESM-C-compatible)
    └── modeling_oplm.py       # all public Oplm* model classes
tests/                         # pytest tests, mirrors src/ structure
```

## Code Style

- Python 3.11+, modern typing syntax (`X | Y`, `Self`, etc.)
- `from __future__ import annotations` in every file
- Ruff for linting and formatting (configured in pyproject.toml)
- Line length: 100 characters
- Google-style docstrings

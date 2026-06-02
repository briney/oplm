"""Training entry point.

Launch with torchrun (single- or multi-GPU)::

    torchrun --nproc_per_node=1 -m oplm.train --config configs/my_run.yaml
    torchrun --nproc_per_node=8 -m oplm.train --config configs/my_run.yaml

Append dotlist overrides to opt into features::

    train.precision=fp8     FP8 training (Blackwell / sm90+ only)
    train.compile=true      torch.compile the model
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from oplm.config import OplmConfig

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _ensure_triton_cache_dir(
    env: MutableMapping[str, str],
    *,
    home_dir: Path | None = None,
    tmp_dir: Path | None = None,
) -> Path:
    """Resolve (or create) a writable Triton autotune cache dir and export it in ``env``.

    Honors an already-set ``TRITON_CACHE_DIR``; otherwise tries a home-cache
    location and falls back to a tmp location. Returns the resolved directory.
    """
    existing = env.get("TRITON_CACHE_DIR")
    if existing:
        return Path(existing)

    home_root = home_dir if home_dir is not None else Path.home()
    tmp_root = tmp_dir if tmp_dir is not None else Path(tempfile.gettempdir())
    candidates = (
        home_root / ".cache" / "oplm" / "triton" / "autotune",
        tmp_root / "oplm-triton-cache" / "autotune",
    )
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        env["TRITON_CACHE_DIR"] = str(candidate)
        return candidate
    raise RuntimeError("Unable to create a Triton cache directory for training.")


def _setup_triton_cache(
    env: MutableMapping[str, str] | None = None,
    *,
    home_dir: Path | None = None,
    tmp_dir: Path | None = None,
) -> Path:
    """Ensure ``torch.compile``'s Triton backend has a writable autotune cache.

    The Triton backend writes autotune artifacts to ``TRITON_CACHE_DIR``; this
    guarantees the variable points at a directory that exists. Defaults to the
    live process environment when ``env`` is omitted.

    Args:
        env: Environment mapping to read/update. Defaults to ``os.environ``.
        home_dir: Override for the home-cache root (testing).
        tmp_dir: Override for the tmp fallback root (testing).

    Returns:
        The resolved Triton cache directory.
    """
    runtime_env = os.environ if env is None else env
    return _ensure_triton_cache_dir(runtime_env, home_dir=home_dir, tmp_dir=tmp_dir)


def main(cfg: OplmConfig | None = None) -> None:
    """Run training.

    Args:
        cfg: Pre-loaded config. If None, parses from sys.argv.
    """
    _setup_triton_cache()

    if cfg is None:
        from oplm.config import load_config

        cfg = load_config(sys.argv[1:])

    from oplm.training import Trainer

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()

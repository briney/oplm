"""Training entry point.

Run directly:     python -m oplm.train --config configs/my_run.yaml
Run distributed:  accelerate launch -m oplm.train --config configs/my_run.yaml model.num_layers=32
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

_DEEPSPEED_OPT_IN_ENV = "OPLM_ENABLE_DEEPSPEED"
_DEEPSPEED_ENV_VARS = (
    "ACCELERATE_USE_DEEPSPEED",
    "ACCELERATE_DEEPSPEED_CONFIG_FILE",
    "ACCELERATE_DEEPSPEED_MOE_LAYER_CLS_NAMES",
    "ACCELERATE_CONFIG_DS_FIELDS",
)


def _env_flag_is_enabled(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def _ensure_triton_cache_dir(
    env: MutableMapping[str, str],
    *,
    home_dir: Path | None = None,
    tmp_dir: Path | None = None,
) -> Path:
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


def _bootstrap_training_environment(
    env: MutableMapping[str, str] | None = None,
    *,
    home_dir: Path | None = None,
    tmp_dir: Path | None = None,
) -> Path:
    runtime_env = os.environ if env is None else env

    if not _env_flag_is_enabled(runtime_env.get(_DEEPSPEED_OPT_IN_ENV)):
        runtime_env["ACCELERATE_USE_DEEPSPEED"] = "false"
        for key in _DEEPSPEED_ENV_VARS[1:]:
            runtime_env.pop(key, None)

    return _ensure_triton_cache_dir(runtime_env, home_dir=home_dir, tmp_dir=tmp_dir)


def main(cfg: OplmConfig | None = None) -> None:
    """Run training.

    Args:
        cfg: Pre-loaded config. If None, parses from sys.argv.
    """
    _bootstrap_training_environment()

    if cfg is None:
        from oplm.config import load_config

        cfg = load_config(sys.argv[1:])

    from oplm.training import Trainer

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()

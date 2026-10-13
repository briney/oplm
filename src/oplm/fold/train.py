"""Fold training entry point.

Run directly:     python -m oplm.fold.train --config configs/fold/proof-w128.yaml train.max_steps=10
Run distributed:  accelerate launch -m oplm.fold.train --config configs/fold/proof-w128.yaml

Parses the run config exactly as ``oplm.train`` does and hands ``FoldTask`` to the Trainer.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oplm.config import OplmConfig


def main(cfg: OplmConfig | None = None) -> None:
    """Train the structure head from ``cfg`` (parsed from ``sys.argv`` when ``None``).

    Raises:
        ValueError: The config has no ``fold:`` block.
    """
    if cfg is None:
        from oplm.config import load_config

        cfg = load_config(sys.argv[1:])
    from oplm.fold.training import FoldTask, require_fold
    from oplm.train import main as train_main

    require_fold(cfg)
    train_main(cfg, task=FoldTask())


if __name__ == "__main__":
    main()

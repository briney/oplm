"""Training entry point.

Run directly:     python -m oplm.train --config configs/my_run.yaml
Run distributed:  accelerate launch -m oplm.train --config configs/my_run.yaml model.num_layers=32
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oplm.config import OplmConfig


def main(cfg: OplmConfig | None = None) -> None:
    """Run training.

    Args:
        cfg: Pre-loaded config. If None, parses from sys.argv.
    """
    if cfg is None:
        from oplm.config import load_config

        cfg = load_config(sys.argv[1:])

    # TODO: instantiate model, optimizer, dataloader, training loop
    raise NotImplementedError(
        "Training loop not yet implemented. "
        "Model and config are ready — see docs/ARCHITECTURE.md for next steps."
    )


if __name__ == "__main__":
    main()

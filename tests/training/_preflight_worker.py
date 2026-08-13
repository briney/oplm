"""Worker for the preflight rank-attribution test (Task 1.8 fix round). Run under
``torch.distributed.run``.

Rank 1's local alloc+matmul check is patched to fail when ``OPLM_TEST_FAIL_RANK1=1`` is
set, so the parent test can assert that BOTH ranks abort quickly with an attributable
error naming rank 1 -- not a generic collective-timeout naming nobody. Each rank writes
its outcome to ``out_dir/rank<i>.json`` and then re-raises on failure, so the process
exits nonzero and torchrun's own aggregated stderr also carries the message.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path


def main(out_dir: str) -> None:
    """Run `run_preflight` on this rank (rank 1 forced to fail locally) and record it.

    Args:
        out_dir: Directory to write ``rank<i>.json`` into (must already exist).
    """
    from accelerate import Accelerator

    from oplm.training import preflight as preflight_module

    accelerator = Accelerator()
    rank = accelerator.process_index

    if rank == 1 and os.environ.get("OPLM_TEST_FAIL_RANK1") == "1":

        def _fail_local_check(device: object) -> str:
            return "synthetic preflight failure injected for the Task 1.8 fix-round test"

        preflight_module._run_local_checks = _fail_local_check  # type: ignore[method-assign]

    error_message: str | None = None
    try:
        preflight_module.run_preflight(accelerator)
    except RuntimeError as exc:
        error_message = str(exc)

    Path(out_dir, f"rank{rank}.json").write_text(
        json.dumps({"rank": rank, "host": socket.gethostname(), "error": error_message})
    )

    if error_message is not None:
        raise RuntimeError(error_message)


if __name__ == "__main__":
    main(sys.argv[1])

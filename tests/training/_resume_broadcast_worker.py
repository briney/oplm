"""Worker for the resume-target rank-agreement test (Task 1.7 fix round). Run under
``torch.distributed.run``.

Constructs the real ``Trainer`` against an ``output_dir`` that already holds a committed
checkpoint, with ``auto_resume=True`` and no explicit ``resume_from`` -- the exact requeue
scenario ``oplm.training.trainer._resolve_resume_target`` exists to make rank-identical.
Each rank writes its own resolved resume target and post-construction ``global_step`` to
``out_dir/rank<i>.json`` so the parent test process can assert every rank agrees.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(output_dir: str, train_data: str, out_dir: str, max_steps: int) -> None:
    """Construct the Trainer on this rank and dump its resolved resume state.

    Args:
        output_dir: Training output dir (already holds a committed checkpoint).
        train_data: Parquet fixture path for the (unused, construction-only) dataloader.
        out_dir: Directory to write ``rank<i>.json`` into (must already exist).
        max_steps: ``cfg.train.max_steps`` for the constructed config.
    """
    from oplm.training.trainer import Trainer
    from tests.training.conftest import tiny_train_cfg

    cfg = tiny_train_cfg(
        Path(output_dir),
        Path(train_data),
        max_steps=max_steps,
        save_every=max_steps,
        auto_resume=True,
        log_every=1,
        # Phase 1 (the parent test process) trains at world_size=1; this worker resumes
        # at world_size=2 -- a genuine world-size change across the resume, which Task
        # 3.3's data-cursor layout guard makes a hard error by default (world_size is
        # part of the validated layout). This test's purpose is rank-agreement on the
        # *resume target*, not data-cursor exactness across a world-size change, so it
        # opts into the documented escape hatch (mirrors test_e2e_dcp.py's reshard test).
        resume_data_position=False,
    )
    trainer = Trainer(cfg, callbacks=[])
    rank = trainer.accelerator.process_index
    payload = {
        "resolved_resume_target": trainer._resolved_resume_target,
        "global_step": trainer.global_step,
    }
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(payload))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from oplm.config import serialize_config
from oplm.sweep import run
from tests.training.conftest import tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow


def test_mup_run_writes_validation_result(training_parquet: Path, tmp_path: Path) -> None:
    cfg = tiny_train_cfg(
        tmp_path / "trainer-output",
        training_parquet,
        max_steps=2,
        batch_size=4,
        log_every=1,
    )
    cfg.data.eval = {
        "heldout": {
            "path": str(training_parquet),
            "type": "sequence",
            "every": {"steps": 1},
        }
    }
    run_yaml = tmp_path / "run.yaml"
    result_json = tmp_path / "result.json"
    run_yaml.write_text(serialize_config(cfg))

    run.main(config=run_yaml, result=result_json)

    payload = json.loads(result_json.read_text())
    assert payload["steps"] == 2
    assert payload["global_batch"] == 4
    assert payload["eval"]["eval/heldout/loss"] > 0

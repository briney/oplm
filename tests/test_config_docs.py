"""Documentation checks for the canonical config reference."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from oplm.config import DataConfig, ModelConfig, TrainConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DOC = REPO_ROOT / "configs" / "README.md"
README_DOC = REPO_ROOT / "README.md"
ARCHITECTURE_DOC = REPO_ROOT / "docs" / "ARCHITECTURE.md"


def _extract_override_rows(text: str, heading: str) -> list[str]:
    """Extract override-path rows from a markdown table section."""
    marker = f"## {heading}\n"
    section = text.split(marker, 1)[1].split("\n## ", 1)[0]
    return [
        line.split("|")[1].strip().strip("`")
        for line in section.splitlines()
        if line.startswith("| `")
    ]


def test_config_reference_covers_all_model_fields() -> None:
    text = CONFIG_DOC.read_text()
    overrides = _extract_override_rows(text, "Model Fields")
    expected = [f"model.{field.name}" for field in fields(ModelConfig)]
    assert overrides == expected


def test_config_reference_covers_all_train_fields() -> None:
    text = CONFIG_DOC.read_text()
    overrides = _extract_override_rows(text, "Train Fields")
    expected = [f"train.{field.name}" for field in fields(TrainConfig)]
    assert overrides == expected


def test_config_reference_covers_all_data_fields() -> None:
    text = CONFIG_DOC.read_text()
    overrides = _extract_override_rows(text, "Data Fields")
    expected = [f"data.{field.name}" for field in fields(DataConfig)]
    assert overrides == expected


def test_config_reference_mentions_all_eval_task_sections() -> None:
    text = CONFIG_DOC.read_text()
    for heading in (
        "### Sequence",
        "### Structure",
        "### ProteinGym",
        "### TAPE",
        "### ProteinGlue",
        "### EVEREST",
    ):
        assert heading in text


def test_readme_points_to_config_reference_and_flat_eval_keys() -> None:
    text = README_DOC.read_text()
    assert "[configs/README.md](configs/README.md)" in text
    assert "--override model.num_layers=16" in text
    assert "extra:" not in text
    assert "/path/to/checkpoint.pt" not in text


def test_config_reference_documents_cli_override_flags() -> None:
    text = CONFIG_DOC.read_text()
    assert "--override train.max_steps=1000" in text
    assert "`oplm train`, `oplm info`, and `oplm encode` all use repeated `--override" in text


def test_architecture_doc_stays_high_level_and_points_to_live_sources() -> None:
    text = ARCHITECTURE_DOC.read_text()
    assert "This document is intentionally a high-level map" in text
    assert "[`configs/README.md`](../configs/README.md)" in text
    assert "[`src/oplm/config.py`](../src/oplm/config.py)" in text
    assert "[`TrainerCallback`](../src/oplm/training/callbacks.py)" in text

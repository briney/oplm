"""Fast unit tests for ``_bootstrap_training_environment`` (docs/TESTING_E2E.md §5, G11).

The bootstrap step scrubs DeepSpeed env vars (unless explicitly opted in) and
ensures a writable Triton autotune cache directory exists. These tests inject an
isolated ``env`` dict plus ``home_dir`` / ``tmp_dir`` so nothing touches the real
process environment or filesystem home. Not marked slow — pure, sub-millisecond.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from oplm.train import _bootstrap_training_environment

if TYPE_CHECKING:
    from pathlib import Path

_DEEPSPEED_VARS = (
    "ACCELERATE_USE_DEEPSPEED",
    "ACCELERATE_DEEPSPEED_CONFIG_FILE",
    "ACCELERATE_DEEPSPEED_MOE_LAYER_CLS_NAMES",
    "ACCELERATE_CONFIG_DS_FIELDS",
)


def test_deepspeed_scrubbed_by_default(tmp_path: Path) -> None:
    """Without the opt-in flag, DeepSpeed is disabled and its config vars are removed."""
    env = {
        "ACCELERATE_USE_DEEPSPEED": "true",
        "ACCELERATE_DEEPSPEED_CONFIG_FILE": "/some/ds.json",
        "ACCELERATE_DEEPSPEED_MOE_LAYER_CLS_NAMES": "Block",
        "ACCELERATE_CONFIG_DS_FIELDS": "x,y",
    }
    _bootstrap_training_environment(env, home_dir=tmp_path)

    assert env["ACCELERATE_USE_DEEPSPEED"] == "false"
    for key in _DEEPSPEED_VARS[1:]:
        assert key not in env


def test_deepspeed_opt_in_preserves_vars(tmp_path: Path) -> None:
    """The OPLM_ENABLE_DEEPSPEED opt-in leaves the DeepSpeed env untouched."""
    env = {
        "OPLM_ENABLE_DEEPSPEED": "1",
        "ACCELERATE_USE_DEEPSPEED": "true",
        "ACCELERATE_DEEPSPEED_CONFIG_FILE": "/some/ds.json",
    }
    _bootstrap_training_environment(env, home_dir=tmp_path)

    assert env["ACCELERATE_USE_DEEPSPEED"] == "true"
    assert env["ACCELERATE_DEEPSPEED_CONFIG_FILE"] == "/some/ds.json"


def test_triton_cache_dir_created_under_home(tmp_path: Path) -> None:
    """A missing TRITON_CACHE_DIR is created under the home cache and exported."""
    env: dict[str, str] = {}
    result = _bootstrap_training_environment(env, home_dir=tmp_path)

    expected = tmp_path / ".cache" / "oplm" / "triton" / "autotune"
    assert result == expected
    assert expected.is_dir()
    assert env["TRITON_CACHE_DIR"] == str(expected)


def test_triton_cache_dir_respects_existing(tmp_path: Path) -> None:
    """An already-set TRITON_CACHE_DIR is honored and returned unchanged."""
    preset = tmp_path / "preset-cache"
    env = {"TRITON_CACHE_DIR": str(preset)}
    result = _bootstrap_training_environment(env, home_dir=tmp_path)

    assert result == preset
    assert env["TRITON_CACHE_DIR"] == str(preset)


def test_triton_cache_dir_falls_back_to_tmp(tmp_path: Path) -> None:
    """An unwritable home cache falls back to the tmp candidate."""
    home_file = tmp_path / "home-as-file"
    home_file.write_text("not a directory")
    tmp_root = tmp_path / "tmp"
    tmp_root.mkdir()

    result = _bootstrap_training_environment({}, home_dir=home_file, tmp_dir=tmp_root)

    expected = tmp_root / "oplm-triton-cache" / "autotune"
    assert result == expected
    assert expected.is_dir()

"""Fast unit tests for ``_setup_triton_cache`` (docs/TESTING_E2E.md §5, G11).

The setup step ensures a writable Triton autotune cache directory exists for
``torch.compile``'s Triton backend. These tests inject an isolated ``env`` dict
plus ``home_dir`` / ``tmp_dir`` so nothing touches the real process environment
or filesystem home. Not marked slow — pure, sub-millisecond.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from oplm.train import _setup_triton_cache

if TYPE_CHECKING:
    from pathlib import Path


def test_triton_cache_dir_created_under_home(tmp_path: Path) -> None:
    """A missing TRITON_CACHE_DIR is created under the home cache and exported."""
    env: dict[str, str] = {}
    result = _setup_triton_cache(env, home_dir=tmp_path)

    expected = tmp_path / ".cache" / "oplm" / "triton" / "autotune"
    assert result == expected
    assert expected.is_dir()
    assert env["TRITON_CACHE_DIR"] == str(expected)


def test_triton_cache_dir_respects_existing(tmp_path: Path) -> None:
    """An already-set TRITON_CACHE_DIR is honored and returned unchanged."""
    preset = tmp_path / "preset-cache"
    env = {"TRITON_CACHE_DIR": str(preset)}
    result = _setup_triton_cache(env, home_dir=tmp_path)

    assert result == preset
    assert env["TRITON_CACHE_DIR"] == str(preset)


def test_triton_cache_dir_falls_back_to_tmp(tmp_path: Path) -> None:
    """An unwritable home cache falls back to the tmp candidate."""
    home_file = tmp_path / "home-as-file"
    home_file.write_text("not a directory")
    tmp_root = tmp_path / "tmp"
    tmp_root.mkdir()

    result = _setup_triton_cache({}, home_dir=home_file, tmp_dir=tmp_root)

    expected = tmp_root / "oplm-triton-cache" / "autotune"
    assert result == expected
    assert expected.is_dir()

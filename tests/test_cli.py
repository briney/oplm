"""CLI-layer tests for positional ``key=value`` config overrides.

These exercise the Typer wiring in ``oplm.cli`` (how override tokens are
collected), not the merge/coercion logic in ``load_config`` — that is covered by
``tests/training/test_config.py``. The side-effect-free ``info`` command is used
for parsing assertions so no training is launched.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from oplm.cli import app

if TYPE_CHECKING:
    import pytest

    from oplm.config import OplmConfig

runner = CliRunner()

# Typer renders ``--help`` via rich; under ``FORCE_COLOR`` (e.g. on CI) option
# names and metavars are wrapped in ANSI SGR escapes, so a raw
# ``"--override" in output`` substring check fails even when the flag is present.
# Strip styling before asserting so the checks are color-/environment-independent.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _plain(output: str) -> str:
    """Return CLI ``output`` with ANSI styling removed."""
    return _ANSI_RE.sub("", output)


def test_train_help_uses_positional_overrides() -> None:
    """``train --help`` advertises a KEY=VALUE positional, not ``--override``."""
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "KEY=VALUE" in _plain(result.output)
    assert "--override" not in _plain(result.output)


def test_info_help_uses_positional_overrides() -> None:
    """``info --help`` no longer offers the ``--override`` option."""
    result = runner.invoke(app, ["info", "--help"])
    assert result.exit_code == 0
    assert "--override" not in _plain(result.output)


def test_encode_help_keeps_override_flag() -> None:
    """``encode`` keeps ``--override`` (its positional slot holds sequences)."""
    result = runner.invoke(app, ["encode", "--help"])
    assert result.exit_code == 0
    assert "--override" in _plain(result.output)


def test_info_positional_override_applies() -> None:
    """A bare ``model.num_hidden_layers=4`` token flows through to the config."""
    result = runner.invoke(app, ["info", "--preset", "small", "model.num_hidden_layers=4"])
    assert result.exit_code == 0, result.output
    # The "Layers" row of the architecture table should report the override.
    assert re.search(r"Layers[^\n]*\b4\b", _plain(result.output)), result.output


def test_info_rejects_legacy_override_flag() -> None:
    """``--override`` is gone from ``info`` and Typer reports it as unknown."""
    result = runner.invoke(app, ["info", "--override", "model.num_hidden_layers=4"])
    assert result.exit_code == 2


def test_info_rejects_non_keyvalue_positional() -> None:
    """A positional token without ``=`` is rejected before config parsing."""
    result = runner.invoke(app, ["info", "notanoverride"])
    assert result.exit_code == 2
    assert "KEY=VALUE" in _plain(result.output)


def test_train_positional_override_reaches_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """A positional override on ``train`` lands in the cfg handed to training."""
    captured: dict[str, OplmConfig] = {}

    def fake_main(cfg: OplmConfig) -> None:
        captured["cfg"] = cfg

    # `train` does `from oplm.train import main as train_main` at call time,
    # so patching the module attribute intercepts the real training launch.
    monkeypatch.setattr("oplm.train.main", fake_main)

    result = runner.invoke(app, ["train", "--preset", "small", "train.max_steps=123"])
    assert result.exit_code == 0, result.output
    assert captured["cfg"].train.max_steps == 123

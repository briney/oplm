"""Custom-code copy + remote-loading round-trip for the OPLM auto-class hook.

These exercise the ``register_for_auto_class`` file-copy step (§13.4): saving a
model must drop the source ``.py`` files next to ``config.json`` so that a fresh
interpreter can reload it via ``trust_remote_code=True`` without ``oplm``
installed in the registry.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
import torch

import oplm  # noqa: F401  (import triggers auto-class + custom-code registration)
from oplm import OplmConfig, OplmForMaskedLM, OplmTokenizerFast

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module", params=["stack", "interleave"])
def saved_model_dir(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> Path:
    """Save a tiny model + tokenizer and return the directory."""
    tmpdir = tmp_path_factory.mktemp("oplm_remote")
    config = OplmConfig(
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_loops=2,
        loop_start=1,
        loop_strategy=request.param,
        tie_word_embeddings=True,
    )
    model = OplmForMaskedLM(config).eval()
    tokenizer = OplmTokenizerFast()
    model.save_pretrained(tmpdir)
    tokenizer.save_pretrained(tmpdir)
    with torch.no_grad():
        torch.save(model(**tokenizer("MEEPQ", return_tensors="pt")).logits, tmpdir / "expected.pt")
    return tmpdir


def test_custom_code_files_copied(saved_model_dir: Path) -> None:
    """The public source files plus their helper imports land beside config.json."""
    names = {p.name for p in saved_model_dir.iterdir()}

    # Public custom-code modules HF copies via register_for_auto_class.
    assert "modeling_oplm.py" in names
    assert "configuration_oplm.py" in names
    assert "tokenization_oplm.py" in names

    # Helper modules imported by modeling_oplm.py must come along too, or the
    # remote import fails.
    for helper in ("attention.py", "transformer.py", "norm.py", "rope.py", "ffn.py", "looping.py"):
        assert helper in names, f"missing helper module {helper}"

    # auto_map entries must be present in the written config.
    config_text = (saved_model_dir / "config.json").read_text()
    assert "auto_map" in config_text
    assert "modeling_oplm.OplmForMaskedLM" in config_text


@pytest.mark.slow
def test_remote_reload_in_subprocess(saved_model_dir: Path) -> None:
    """A fresh interpreter reloads the model via trust_remote_code (no `import oplm`).

    Run in a subprocess so the in-process auto-class registration from
    ``import oplm`` cannot mask a regression in the custom-code copy step.
    """
    script = (
        "import torch; from transformers import AutoModelForMaskedLM, AutoTokenizer; "
        f"m = AutoModelForMaskedLM.from_pretrained({str(saved_model_dir)!r}, "
        "trust_remote_code=True); "
        f"t = AutoTokenizer.from_pretrained({str(saved_model_dir)!r}, "
        "trust_remote_code=True); "
        "assert type(m).__name__ == 'OplmForMaskedLM'; "
        "assert t('MEEPQ').input_ids == [0, 20, 9, 9, 14, 16, 2]; "
        "expected_order = (0, 1, 2, 1, 2) if m.config.loop_strategy == 'stack' "
        "else (0, 1, 1, 2, 2); "
        "assert m.oplm.backbone.layer_execution_order == expected_order; "
        f"expected = torch.load({str(saved_model_dir / 'expected.pt')!r}, weights_only=True); "
        "torch.testing.assert_close(m(**t('MEEPQ', return_tensors='pt')).logits, "
        "expected, rtol=0, atol=0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

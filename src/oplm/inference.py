"""Inference helpers for loading configs and model weights."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from oplm.config import OplmConfig, load_config
from oplm.model.transformer import OplmForMLM

_MODEL_STATE_FILENAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model.pt",
    "model.bin",
    "model.pt",
)


def resolve_inference_config(
    model_path: str | Path,
    *,
    config_path: str | None = None,
    preset: str | None = None,
    overrides: list[str] | None = None,
) -> OplmConfig:
    """Resolve config for inference from CLI args or checkpoint metadata.

    Explicit ``config_path`` or ``preset`` takes priority. Otherwise, this
    attempts to load ``config.yaml`` from a training checkpoint directory or
    from the sibling directory of a model state-dict file.
    """
    argv: list[str] = []
    if preset is not None:
        argv.extend(["--preset", preset])
    if config_path is not None:
        argv.extend(["--config", config_path])

    inferred_config = _find_associated_config(Path(model_path))
    if not argv and inferred_config is not None:
        argv.extend(["--config", str(inferred_config)])

    if overrides:
        argv.extend(overrides)

    if not argv:
        raise FileNotFoundError(
            "Could not infer a config for inference. Pass --config/--preset or use a "
            "checkpoint directory that contains config.yaml."
        )

    return load_config(argv)


def load_model_for_inference(
    model_path: str | Path,
    cfg: OplmConfig,
    *,
    device: torch.device | str = "cpu",
) -> OplmForMLM:
    """Load an inference-ready model from a checkpoint directory or state-dict file."""
    state_dict = load_model_state_dict(model_path)
    model = OplmForMLM(cfg.model)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_model_state_dict(model_path: str | Path) -> dict[str, Tensor]:
    """Load a model state dict from a checkpoint directory or weights file."""
    path = _resolve_state_path(Path(model_path))

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected a state dict at {path}, got {type(checkpoint).__name__}")

    return checkpoint


def _find_associated_config(model_path: Path) -> Path | None:
    """Find a config file associated with a checkpoint path."""
    candidates = (
        [model_path / "config.yaml"] if model_path.is_dir() else [model_path.parent / "config.yaml"]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_state_path(model_path: Path) -> Path:
    """Resolve a checkpoint directory to its model weights file."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if model_path.is_dir():
        for filename in _MODEL_STATE_FILENAMES:
            candidate = model_path / filename
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not find model weights in checkpoint directory: {model_path}"
        )

    return model_path

"""Strict weights-only initialization from a local OPLM pretrained export."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from oplm.model import OplmConfig, OplmForMaskedLM

# These fields change execution order, runtime reporting, or initial values that
# are overwritten by a complete checkpoint. Classifier fields do not affect MLM.
_IGNORED_INITIALIZATION_FIELDS = frozenset(
    {
        "num_loops",
        "loop_strategy",
        "loop_start",
        "loop_end",
        "gradient_checkpointing",
        "gradient_checkpointing_mode",
        "initializer_range",
        "init_scale_output_projections",
        "residual_gate_init",
        "qk_norm_l2_scale_init",
        "value_residual_lambda_init",
        "classifier_pool",
        "classifier_dropout",
        "num_labels",
        "pre_head_norm",
    }
)


def resolve_initialization_source(source: str) -> Path:
    """Resolve a local HF export or a training checkpoint's ``hf`` child.

    Args:
        source: Local directory, optionally using a home-directory abbreviation.

    Returns:
        Absolute path to the directory containing ``config.json``.

    Raises:
        FileNotFoundError: Neither local export exists. No Hub lookup is attempted.
    """
    path = Path(source).expanduser().resolve()
    for candidate in (path, path / "hf"):
        if candidate.is_dir() and (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError(f"No local OPLM pretrained export at {path} or {path / 'hf'}.")


def _semantic_config(config: OplmConfig) -> dict[str, Any]:
    # Reconstruct to resolve derived geometry and Canon schedules, and validate
    # configurations edited by library callers, without mutating their objects.
    normalized = OplmConfig.from_dict(config.to_dict())
    signature = {
        name: getattr(normalized, name)
        for name, parameter in inspect.signature(OplmConfig.__init__).parameters.items()
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY
        and name not in _IGNORED_INITIALIZATION_FIELDS
    }
    if not normalized.canon_enabled:
        for field in (
            "canon_positions",
            "canon_kernel_sizes",
            "canon_activation",
            "canon_residual",
        ):
            signature.pop(field)
    else:
        signature["canon_positions"] = tuple(sorted(normalized.canon_positions))
        signature["canon_kernel_sizes"] = tuple(normalized.canon_kernel_sizes)
    if not normalized.mup_enable:
        signature.pop("mup_base_width")
        signature.pop("mup_output_mult")
    if not normalized.qk_norm:
        signature.pop("qk_norm_mode")
    if not normalized.mask_dropout:
        signature.pop("mask_dropout_reference_ratio")
    return signature


def validate_initialization_config(source: OplmConfig, target: OplmConfig) -> None:
    """Reject changes to forward semantics other than loop execution.

    Args:
        source: Saved model configuration.
        target: Resolved model configuration for the new training stage.

    Raises:
        ValueError: Configurations differ in model shape or active forward settings.
    """
    before, after = _semantic_config(source), _semantic_config(target)
    differences = [
        f"{name} (source={before.get(name)!r}, target={after.get(name)!r})"
        for name in sorted(before.keys() | after.keys())
        if before.get(name) != after.get(name)
    ]
    if differences:
        raise ValueError("Incompatible pretrained initialization: " + "; ".join(differences))


def load_initial_model(source: Path, target: OplmConfig) -> OplmForMaskedLM:
    """Load every model tensor without transferring any training state.

    Args:
        source: Local HF export directory from ``resolve_initialization_source``.
        target: Resolved model configuration, including desired loop settings.

    Returns:
        Initialized MLM model. The caller controls its training/evaluation mode.

    Raises:
        ValueError: Forward settings or checkpoint tensors are incompatible/incomplete.
        OSError: The local configuration or weight files cannot be read.
    """
    source_config = OplmConfig.from_pretrained(source, local_files_only=True)
    validate_initialization_config(source_config, target)
    try:
        model, info = OplmForMaskedLM.from_pretrained(
            source, config=target, local_files_only=True, output_loading_info=True
        )
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Cannot initialize pretrained weights from {source}: {exc}") from exc
    problems = {
        name: info[name]
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if info.get(name)
    }
    if problems:
        raise ValueError(f"Incomplete pretrained initialization from {source}: {problems}")
    return model

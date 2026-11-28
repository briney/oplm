"""Strict weights-only initialization from a local OPLM pretrained export."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch

from oplm.model import OplmConfig, OplmForMaskedLM

# These fields change execution order, runtime reporting, or initial values that
# are overwritten by a complete checkpoint. Classifier fields do not affect MLM.
_IGNORED_INITIALIZATION_FIELDS = frozenset(
    {
        "num_loops",
        "loop_strategy",
        "loop_start",
        "loop_end",
        "loop_input_injection",
        "loop_gate",
        "loop_gate_init",
        "loop_untie_norms",
        "loop_step_conditioning",
        "gradient_checkpointing",
        "gradient_checkpointing_mode",
        "packed_attention",
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

_UNTIED_LOOP_KEY_MARKERS = (
    ".loop_attn_norms.",
    ".loop_ffn_norms.",
    ".loop_attn_post_norms.",
    ".loop_ffn_post_norms.",
    ".loop_value_residual_lambdas.",
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


def _validate_loop_extension_transition(source: OplmConfig, target: OplmConfig) -> None:
    """Permit enabling Step 2 loop extensions on unlooped/vanilla sources; reject swaps/drops."""
    src = OplmConfig.from_dict(source.to_dict())
    tgt = OplmConfig.from_dict(target.to_dict())
    errors: list[str] = []

    if src.loop_input_injection != "none" and tgt.loop_input_injection != src.loop_input_injection:
        errors.append(
            f"loop_input_injection (source={src.loop_input_injection!r}, "
            f"target={tgt.loop_input_injection!r})"
        )
    if src.loop_gate != "none" and tgt.loop_gate != src.loop_gate:
        errors.append(f"loop_gate (source={src.loop_gate!r}, target={tgt.loop_gate!r})")
    if src.loop_untie_norms and not tgt.loop_untie_norms:
        errors.append(
            f"loop_untie_norms (source={src.loop_untie_norms!r}, target={tgt.loop_untie_norms!r})"
        )
    if (
        src.loop_step_conditioning != "none"
        and tgt.loop_step_conditioning != src.loop_step_conditioning
    ):
        errors.append(
            f"loop_step_conditioning (source={src.loop_step_conditioning!r}, "
            f"target={tgt.loop_step_conditioning!r})"
        )

    if src.loop_gate != "none" or src.loop_untie_norms:
        src_end = src.num_hidden_layers if src.loop_end is None else src.loop_end
        tgt_end = tgt.num_hidden_layers if tgt.loop_end is None else tgt.loop_end
        for field, s_val, t_val in (
            ("num_loops", src.num_loops, tgt.num_loops),
            ("loop_strategy", src.loop_strategy, tgt.loop_strategy),
            ("loop_start", src.loop_start, tgt.loop_start),
            ("loop_end", src_end, tgt_end),
        ):
            if s_val != t_val:
                errors.append(
                    f"{field} (source={s_val!r}, target={t_val!r}) cannot change when source "
                    "already carries per-pass loop modules"
                )

    if errors:
        raise ValueError("Incompatible pretrained initialization: " + "; ".join(errors))


def _allowed_new_loop_keys(
    source: OplmConfig, target: OplmConfig, model: OplmForMaskedLM
) -> set[str]:
    """Return exact state-dict keys belonging to Step 2 extensions newly enabled on ``target``."""
    src = OplmConfig.from_dict(source.to_dict())
    tgt = OplmConfig.from_dict(target.to_dict())
    keys = model.state_dict().keys()
    allowed: set[str] = set()
    if src.loop_input_injection == "none" and tgt.loop_input_injection != "none":
        allowed.update(k for k in keys if k.startswith("oplm.backbone.loop_injector."))
    if src.loop_gate == "none" and tgt.loop_gate != "none":
        allowed.update(k for k in keys if k.startswith("oplm.backbone.loop_gate_module."))
    if src.loop_step_conditioning == "none" and tgt.loop_step_conditioning != "none":
        allowed.update(k for k in keys if k.startswith("oplm.backbone.loop_step_cond."))
    if not src.loop_untie_norms and tgt.loop_untie_norms:
        allowed.update(k for k in keys if any(m in k for m in _UNTIED_LOOP_KEY_MARKERS))
    return allowed


def _initialize_new_loop_modules(
    source: OplmConfig, target: OplmConfig, model: OplmForMaskedLM
) -> None:
    """Initialize custom/per-pass parameters of newly enabled Step 2 loop extensions."""
    src = OplmConfig.from_dict(source.to_dict())
    tgt = OplmConfig.from_dict(target.to_dict())
    backbone = model.oplm.backbone

    if (
        src.loop_input_injection == "none"
        and tgt.loop_input_injection != "none"
        and backbone.loop_injector is not None
    ):
        backbone.loop_injector.reset_parameters()

    if (
        src.loop_gate == "none"
        and tgt.loop_gate != "none"
        and backbone.loop_gate_module is not None
    ):
        with torch.no_grad():
            for gate in backbone.loop_gate_module.gates:
                gate.fill_(float(tgt.loop_gate_init))

    if not src.loop_untie_norms and tgt.loop_untie_norms:
        with torch.no_grad():
            for block in backbone.layers:
                if not getattr(block, "loop_untie_norms", False):
                    continue
                for base_attr, loop_attr in (
                    ("attn_norm", "loop_attn_norms"),
                    ("ffn_norm", "loop_ffn_norms"),
                    ("attn_post_norm", "loop_attn_post_norms"),
                    ("ffn_post_norm", "loop_ffn_post_norms"),
                ):
                    if hasattr(block, loop_attr):
                        base_sd = getattr(block, base_attr).state_dict()
                        for copy_norm in getattr(block, loop_attr):
                            copy_norm.load_state_dict(base_sd)
                attention = block.attention
                if hasattr(attention, "loop_value_residual_lambdas"):
                    for lam in attention.loop_value_residual_lambdas:
                        lam.copy_(attention.value_residual_lambda)


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
    _validate_loop_extension_transition(source, target)


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
    allowed_missing = _allowed_new_loop_keys(source_config, target, model)
    missing_keys = [k for k in info.get("missing_keys", []) if k not in allowed_missing]
    problems: dict[str, Any] = {}
    if missing_keys:
        problems["missing_keys"] = missing_keys
    for name in ("unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(name):
            problems[name] = info[name]
    if problems:
        raise ValueError(f"Incomplete pretrained initialization from {source}: {problems}")
    _initialize_new_loop_modules(source_config, target, model)
    if target.tie_word_embeddings:
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        if input_weight is not output_weight:
            # Some exports store both aliases. Transformers interprets this as
            # untied, even when the config requires tying. Preserve the declared
            # architecture only when both saved values agree; never discard data.
            if not torch.equal(input_weight, output_weight):
                raise ValueError(f"Conflicting tied embedding/head weights in {source}.")
            model.tie_weights()
    return model

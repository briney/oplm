"""FP8 training support via torchao.

Standalone module — it has no dependency on Accelerate or the Trainer. It
encapsulates all torchao FP8 logic so the rest of the training stack only sees
three small, hardware-aware entry points:

- :func:`is_fp8_supported` — capability gate (sm90+ / Blackwell, H100+).
- :func:`apply_fp8_training` — in-place ``nn.Linear`` -> ``Float8Linear`` conversion,
  called BEFORE ``fully_shard``.
- :func:`sync_fp8_history` — precompute dynamic FP8 weight scales for FSDP2, called
  AFTER ``optimizer.step()``.

``torchao`` is imported lazily inside the functions that need it so this module
stays importable on machines where torchao is absent (e.g. CPU-only CI that only
ever calls :func:`is_fp8_supported`).
"""

from __future__ import annotations

import torch
from torch import nn


def is_fp8_supported() -> bool:
    """Return True if the current CUDA device supports FP8 (sm90+).

    FP8 tensor-core matmuls require compute capability sm90 or newer (Blackwell,
    H100+). Returns False when CUDA is unavailable so callers can gate FP8
    conversion and fail fast with a clear error instead of letting torchao raise
    deep inside the training loop.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def apply_fp8_training(model: nn.Module) -> None:
    """Convert all ``nn.Linear`` layers to ``Float8Linear`` with rowwise scaling.

    Must be called BEFORE ``fully_shard()`` — torchao swaps the linear modules in
    place, and FSDP2 must wrap the already-converted modules. Norms, RoPE, Conv1d,
    and embedding tables are left untouched: the ``module_filter_fn`` selects only
    ``nn.Linear`` instances, so non-Linear modules are skipped.

    Args:
        model: The model to convert in place. The forward API is unchanged.
    """
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    config = Float8LinearConfig.from_recipe_name("rowwise")
    convert_to_float8_training(
        model,
        config=config,
        module_filter_fn=lambda m, fqn: isinstance(m, nn.Linear),
    )


def sync_fp8_history(model: nn.Module) -> None:
    """Precompute dynamic FP8 weight scales for FSDP2.

    The ``rowwise`` recipe uses *dynamic* scaling: scales are derived from the
    current tensor values on every forward, so there is no cross-iteration amax
    history to synchronize (the older ``sync_float8_amax_and_scale_history`` was a
    delayed-scaling primitive and no longer exists). Under FSDP2 this instead
    precomputes each sharded weight's scale in a single all-reduce, which overlaps
    with the next iteration's parameter all-gather.

    Call AFTER ``optimizer.step()`` (the weights must be updated first). It is a
    no-op when the model has no FSDP2-sharded ``Float8Linear`` weights — including
    the ``fsdp_sharding_strategy="none"`` debug path, where weights are not
    ``DTensor`` — so it is safe to call unconditionally whenever
    ``precision == "fp8"``.

    Args:
        model: The FP8-converted, FSDP2-sharded model.
    """
    from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

    precompute_float8_dynamic_scale_for_fsdp(model)

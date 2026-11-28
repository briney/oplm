"""Pure validation and execution ordering for shared transformer layers."""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_LOOP_INPUT_INJECTIONS = ("none", "concat_proj", "decayed", "orthogonal")
_VALID_LOOP_GATES = ("none", "scalar", "channel")
_VALID_LOOP_STEP_CONDITIONINGS = ("none", "sinusoidal")


@dataclass(frozen=True)
class LoopSchedule:
    """Explicit prelude / core / coda partition for shared-layer execution."""

    prelude: tuple[int, ...]
    core: tuple[int, ...]
    coda: tuple[int, ...]
    num_loops: int
    loop_strategy: str
    execution_order: tuple[int, ...]


def resolve_layer_execution_order(
    num_hidden_layers: int,
    *,
    num_loops: int = 1,
    loop_strategy: str = "stack",
    loop_start: int = 0,
    loop_end: int | None = None,
) -> tuple[int, ...]:
    """Validate recurrence settings and return physical block indices in execution order.

    Args:
        num_hidden_layers: Number of unique transformer blocks.
        num_loops: Total executions of each selected block.
        loop_strategy: Repeat the selected stack or interleave repeated blocks.
        loop_start: Inclusive, zero-based beginning of the repeated region.
        loop_end: Exclusive end, or None for the physical depth.

    Returns:
        Physical block indices, including the once-only prefix and suffix.

    Raises:
        ValueError: A count, range, or strategy is invalid.
    """
    return resolve_loop_schedule(
        num_hidden_layers,
        num_loops=num_loops,
        loop_strategy=loop_strategy,
        loop_start=loop_start,
        loop_end=loop_end,
    ).execution_order


def resolve_loop_schedule(
    num_hidden_layers: int,
    *,
    num_loops: int = 1,
    loop_strategy: str = "stack",
    loop_start: int = 0,
    loop_end: int | None = None,
) -> LoopSchedule:
    """Validate recurrence settings and return explicit prelude/core/coda partitions."""
    values = {
        "num_hidden_layers": num_hidden_layers,
        "num_loops": num_loops,
        "loop_start": loop_start,
    }
    if loop_end is not None:
        values["loop_end"] = loop_end
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer; got {value!r}.")
    end = num_hidden_layers if loop_end is None else loop_end
    if num_loops < 1:
        raise ValueError("num_loops must be >= 1.")
    if not 0 <= loop_start < end <= num_hidden_layers:
        raise ValueError("Require 0 <= loop_start < loop_end <= num_hidden_layers.")
    if loop_strategy not in ("stack", "interleave"):
        raise ValueError("loop_strategy must be 'stack' or 'interleave'.")
    prelude = tuple(range(loop_start))
    core = tuple(range(loop_start, end))
    coda = tuple(range(end, num_hidden_layers))
    repeated = (
        core * num_loops
        if loop_strategy == "stack"
        else tuple(index for index in core for _ in range(num_loops))
    )
    return LoopSchedule(
        prelude=prelude,
        core=core,
        coda=coda,
        num_loops=num_loops,
        loop_strategy=loop_strategy,
        execution_order=prelude + repeated + coda,
    )


def validate_loop_extensions(
    *,
    num_loops: int,
    loop_strategy: str,
    loop_input_injection: str = "none",
    loop_gate: str = "none",
    loop_gate_init: float = 1.0,
    loop_untie_norms: bool = False,
    loop_step_conditioning: str = "none",
) -> bool:
    """Validate Step 2 loop-extension settings and return whether any extension is active."""
    if loop_input_injection not in _VALID_LOOP_INPUT_INJECTIONS:
        raise ValueError(
            f"loop_input_injection must be one of {_VALID_LOOP_INPUT_INJECTIONS}; "
            f"got {loop_input_injection!r}."
        )
    if loop_gate not in _VALID_LOOP_GATES:
        raise ValueError(f"loop_gate must be one of {_VALID_LOOP_GATES}; got {loop_gate!r}.")
    if isinstance(loop_gate_init, bool) or not isinstance(loop_gate_init, (int, float)):
        raise ValueError(f"loop_gate_init must be a finite float; got {loop_gate_init!r}.")
    if not math.isfinite(float(loop_gate_init)):
        raise ValueError(f"loop_gate_init must be finite; got {loop_gate_init!r}.")
    if not isinstance(loop_untie_norms, bool):
        raise ValueError(f"loop_untie_norms must be a bool; got {loop_untie_norms!r}.")
    if loop_step_conditioning not in _VALID_LOOP_STEP_CONDITIONINGS:
        raise ValueError(
            f"loop_step_conditioning must be one of {_VALID_LOOP_STEP_CONDITIONINGS}; "
            f"got {loop_step_conditioning!r}."
        )
    active = (
        loop_input_injection != "none"
        or loop_gate != "none"
        or loop_untie_norms
        or loop_step_conditioning != "none"
    )
    if active:
        if loop_strategy != "stack":
            raise ValueError(
                f"Non-default loop extensions require loop_strategy='stack'; got {loop_strategy!r}."
            )
        if num_loops < 2:
            raise ValueError(
                f"Non-default loop extensions require num_loops >= 2; got num_loops={num_loops}."
            )
    return active

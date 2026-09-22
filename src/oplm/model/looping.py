"""Pure validation and execution ordering for shared transformer layers."""

from __future__ import annotations


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
    region = tuple(range(loop_start, end))
    repeated = (
        region * num_loops
        if loop_strategy == "stack"
        else tuple(index for index in region for _ in range(num_loops))
    )
    return tuple(range(loop_start)) + repeated + tuple(range(end, num_hidden_layers))

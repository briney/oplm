"""Eval cadence strategies — pure functions of EvalContext. See EVAL_HARNESS.md §4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.config import ScheduleSpec
    from oplm.eval.context import EvalContext


def _crossed(curr: int, delta: int, n: int) -> bool:
    """True iff the half-open interval ``(curr - delta, curr]`` contains a multiple of n."""
    return curr // n > (curr - delta) // n


@runtime_checkable
class Schedule(Protocol):
    """Decides, from a synchronized EvalContext, whether a task runs this step."""

    def is_due(self, ctx: EvalContext) -> bool: ...


@dataclass(frozen=True)
class EveryNSteps:
    """Fire every ``n`` optimizer steps."""

    n: int
    at_start: bool = False
    at_end: bool = True

    def is_due(self, ctx: EvalContext) -> bool:
        return (
            (self.at_start and ctx.global_step - ctx.steps_delta == 0)
            or (self.at_end and ctx.is_final)
            or _crossed(ctx.global_step, ctx.steps_delta, self.n)
        )


@dataclass(frozen=True)
class EveryNTokens:
    """Fire each time cumulative global tokens cross a multiple of ``n``."""

    n: int
    at_start: bool = False
    at_end: bool = True

    def is_due(self, ctx: EvalContext) -> bool:
        return (
            (self.at_start and ctx.tokens_seen - ctx.tokens_delta == 0)
            or (self.at_end and ctx.is_final)
            or _crossed(ctx.tokens_seen, ctx.tokens_delta, self.n)
        )


_SCHEDULE_BY_UNIT: dict[str, Callable[..., Schedule]] = {
    "steps": EveryNSteps,
    "tokens": EveryNTokens,
}


def build_schedule(spec: ScheduleSpec) -> Schedule:
    """Turn a parsed :class:`ScheduleSpec` into a concrete :class:`Schedule`."""
    cls = _SCHEDULE_BY_UNIT.get(spec.unit)
    if cls is None:
        raise ValueError(
            f"Unsupported schedule unit {spec.unit!r}; supported: {sorted(_SCHEDULE_BY_UNIT)}"
        )
    return cls(n=spec.n, at_start=spec.at_start, at_end=spec.at_end)

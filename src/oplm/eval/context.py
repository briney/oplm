"""Immutable training-state snapshot handed across the trainer↔eval boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalContext:
    """Synchronized training state for one optimizer step.

    INVARIANT: every field MUST be identical across all distributed ranks, so each
    rank independently computes the same ``Schedule.is_due`` without communicating.
    Tasks run collectives inside ``evaluate``; a rank disagreement would deadlock.
    See docs/EVAL_HARNESS.md §3.2.
    """

    global_step: int  # cumulative optimizer steps completed
    epoch: int  # cumulative epochs (carried for future epoch cadence; unused now)
    tokens_seen: int  # cumulative GLOBAL tokens — rank-reduced, not per-rank
    steps_delta: int  # optimizer steps since the previous context (== 1)
    tokens_delta: int  # GLOBAL tokens processed in this optimizer step (rank-reduced)
    epoch_delta: int  # epochs crossed since the previous context (0/1; future use)
    is_final: bool  # True on the last optimizer step (global_step >= total_steps)

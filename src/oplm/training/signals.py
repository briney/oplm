"""Drain trigger for graceful preemption: signals plus the Slurm wall-clock margin.

A drain request tells the trainer "stop after the current optimizer step, checkpoint,
and exit" instead of letting the scheduler kill the process mid-step. Two independent
sources can raise it: an operator/scheduler signal (SIGUSR1 or SIGTERM) and a wall-clock
margin computed from Slurm's ``SLURM_JOB_END_TIME``. Neither is CoreWeave- or
Slurm-cluster-specific beyond that one generic env var, and the env clock is inert on a
plain workstation where the var is unset.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import FrameType
    from typing import TypeAlias

    # Mirrors signal.signal's own handler type (typeshed's private _HANDLER alias):
    # either a real Python callback, or one of the two non-callable sentinels
    # (signal.Handlers.SIG_DFL / SIG_IGN, exposed via signal.getsignal as plain ints
    # for handlers this process didn't install itself), or None (signal not set up
    # for this interpreter, per the stdlib docs).
    _SignalHandler: TypeAlias = (
        Callable[[int, "FrameType | None"], Any] | int | signal.Handlers | None
    )

logger = logging.getLogger(__name__)

#: Reserved for "drained cleanly, resume expected" -- distinct from 0 (reached
#: max_steps) and any other nonzero exit (crash).
DRAIN_EXIT_CODE = 85

_DRAIN_SIGNALS = (signal.SIGUSR1, signal.SIGTERM)


def seconds_until_job_end(env: Mapping[str, str] | None = None) -> float | None:
    """Seconds of wall clock left before ``SLURM_JOB_END_TIME``, or ``None``.

    The single place this codebase reads the scheduler's end-time clock: used by
    :class:`DrainSignal` to decide when to raise a drain, and by the trainer to size
    the budgets that must fit *inside* the drain margin (see
    ``Trainer._drain_remote_uploads``).

    Args:
        env: Environment mapping to read ``SLURM_JOB_END_TIME`` (unix seconds) from.
            Defaults to the real process environment.

    Returns:
        Remaining seconds (negative once the end time has passed), or ``None`` when
        the variable is unset or unparseable -- i.e. "there is no wall clock here",
        which every caller must treat as inert rather than as zero time left.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    end_time_raw = source.get("SLURM_JOB_END_TIME")
    if end_time_raw is None:
        return None
    try:
        end_time = float(end_time_raw)
    except ValueError:
        logger.warning(
            "SLURM_JOB_END_TIME=%r is not a number; ignoring the env drain clock",
            end_time_raw,
        )
        return None
    return end_time - time.time()


class DrainSignal:
    """Set-a-flag drain trigger: SIGUSR1/SIGTERM handlers plus the Slurm end-time clock.

    ``install()`` registers handlers (idempotent; chains any previous non-default handler).
    ``requested`` is True once a signal arrived OR ``SLURM_JOB_END_TIME`` (unix seconds, exported
    by Slurm and forwarded by ``--export=ALL``) minus ``margin_seconds`` has passed. Handlers only
    set a bool -- the only async-signal-safe action.
    """

    def __init__(self, *, margin_seconds: int = 600, env: Mapping[str, str] | None = None) -> None:
        """Initialize the drain trigger.

        Args:
            margin_seconds: How long before ``SLURM_JOB_END_TIME`` the env clock should
                start reporting ``requested``. Ignored when the env var is unset.
            env: Environment mapping to read ``SLURM_JOB_END_TIME`` from. Defaults to the
                real process environment; tests inject a plain dict instead.
        """
        self._margin_seconds = margin_seconds
        self._env: Mapping[str, str] = os.environ if env is None else env
        self._signal_received = False
        self._installed = False

    def install(self) -> None:
        """Register SIGUSR1/SIGTERM handlers. Idempotent; safe to call more than once.

        Any previous handler that is not one of the default sentinels
        (``signal.SIG_DFL`` / ``signal.SIG_IGN``) is chained -- called after the flag is
        set -- so an already-installed application handler keeps running.
        """
        if self._installed:
            return
        for sig in _DRAIN_SIGNALS:
            previous_handler = signal.getsignal(sig)
            signal.signal(sig, self._make_handler(previous_handler))
        self._installed = True

    def _make_handler(
        self, previous_handler: _SignalHandler
    ) -> Callable[[int, FrameType | None], None]:
        """Build a handler that sets the flag, then chains a real previous handler."""

        def _handler(signum: int, frame: FrameType | None) -> None:
            self._signal_received = True
            # previous_handler is signal.getsignal's own return type: a real Python
            # callback, or one of the non-callable sentinels (int/signal.Handlers is
            # SIG_DFL/SIG_IGN) or None. Only the callback case gets chained.
            if previous_handler is None or isinstance(previous_handler, int):
                return
            previous_handler(signum, frame)

        return _handler

    @property
    def requested(self) -> bool:
        """True once a drain signal arrived or the env-clock margin has elapsed."""
        return self._signal_received or self._env_clock_expired()

    def _env_clock_expired(self) -> bool:
        """Check the Slurm wall-clock margin. Inert (always False) if unset/malformed."""
        remaining = seconds_until_job_end(self._env)
        return remaining is not None and remaining <= self._margin_seconds

"""Unit tests for :class:`~oplm.training.signals.DrainSignal` (Task 1.5).

Signal handlers are process-global, so every test that installs one must restore the
prior handler afterward -- the autouse ``_restore_signal_handlers`` fixture below does
that unconditionally, even on assertion failure.
"""

from __future__ import annotations

import os
import signal
import time
from typing import TYPE_CHECKING

import pytest

from oplm.training.signals import DRAIN_EXIT_CODE, DrainSignal

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _restore_signal_handlers() -> Generator[None, None, None]:
    """Snapshot SIGUSR1/SIGTERM handlers and restore them after each test."""
    previous = {
        signal.SIGUSR1: signal.getsignal(signal.SIGUSR1),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def test_drain_exit_code_is_85() -> None:
    assert DRAIN_EXIT_CODE == 85


def test_requested_false_with_no_signal_and_no_env() -> None:
    drain = DrainSignal(env={})
    assert drain.requested is False


def test_flag_flips_on_sigusr1() -> None:
    drain = DrainSignal(env={})
    drain.install()

    assert drain.requested is False
    os.kill(os.getpid(), signal.SIGUSR1)
    assert drain.requested is True


def test_flag_flips_on_sigterm() -> None:
    drain = DrainSignal(env={})
    drain.install()

    assert drain.requested is False
    os.kill(os.getpid(), signal.SIGTERM)
    assert drain.requested is True


def test_install_is_idempotent() -> None:
    drain = DrainSignal(env={})
    drain.install()
    first = signal.getsignal(signal.SIGUSR1)
    drain.install()
    second = signal.getsignal(signal.SIGUSR1)
    assert first is second


def test_install_chains_previous_non_default_handler() -> None:
    calls: list[int] = []

    def _previous_handler(signum: int, frame: object) -> None:
        calls.append(signum)

    signal.signal(signal.SIGUSR1, _previous_handler)

    drain = DrainSignal(env={})
    drain.install()
    os.kill(os.getpid(), signal.SIGUSR1)

    assert drain.requested is True
    assert calls == [signal.SIGUSR1]


def test_env_clock_triggers_within_margin() -> None:
    now = time.time()
    env = {"SLURM_JOB_END_TIME": str(int(now + 300))}
    drain = DrainSignal(margin_seconds=600, env=env)
    assert drain.requested is True


def test_env_clock_false_before_margin() -> None:
    now = time.time()
    env = {"SLURM_JOB_END_TIME": str(int(now + 3600))}
    drain = DrainSignal(margin_seconds=600, env=env)
    assert drain.requested is False


def test_env_clock_inert_without_slurm_env() -> None:
    drain = DrainSignal(margin_seconds=600, env={})
    assert drain.requested is False


def test_env_clock_inert_on_malformed_value() -> None:
    drain = DrainSignal(margin_seconds=600, env={"SLURM_JOB_END_TIME": "not-a-number"})
    assert drain.requested is False


def test_default_env_reads_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no ``env=`` override, the real ``SLURM_JOB_END_TIME`` env var is honored."""
    now = time.time()
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(now + 100)))
    drain = DrainSignal(margin_seconds=600)
    assert drain.requested is True

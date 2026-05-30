"""G2 — eval cadence through the real Trainer (docs/TESTING_E2E.md §5).

Asserts that ``on_eval_end`` fires on exactly the steps the ``_crossed`` schedule
semantics predict — for both step and token cadences — by reconstructing the
expected firing set from the per-step token counts the same run logs. A second
test runs two eval datasets on different cadences and asserts their
``eval/<name>/*`` namespaces appear independently and merge on shared steps.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow

_MAX_STEPS = 6
_BATCH_SIZE = 4


def _expected_step_cadence(n: int, max_steps: int) -> list[int]:
    """Steps an ``EveryNSteps(n)`` (at_end default) schedule is due, by ``_crossed``."""
    due = {s for s in range(1, max_steps + 1) if s // n > (s - 1) // n}
    due.add(max_steps)  # at_end fires on the final step
    return sorted(due)


def _expected_token_cadence(n: int, tokens_by_step: dict[int, int], max_steps: int) -> list[int]:
    """Steps an ``EveryNTokens(n)`` (at_end default) schedule is due, by ``_crossed``."""
    due = set()
    for step in range(1, max_steps + 1):
        prev = tokens_by_step.get(step - 1, 0)
        curr = tokens_by_step[step]
        if curr // n > prev // n:
            due.add(step)
    due.add(max_steps)  # at_end fires on the final step
    return sorted(due)


@pytest.mark.parametrize("unit", ["steps", "tokens"])
def test_eval_fires_on_cadence(
    unit: str,
    training_parquet: Path,
    tmp_path: Path,
) -> None:
    """``on_eval_end`` firings match the half-open ``_crossed`` schedule exactly."""
    from oplm.training.trainer import Trainer

    cadence_n = 2 if unit == "steps" else 256
    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=_MAX_STEPS,
        batch_size=_BATCH_SIZE,
        log_every=1,  # a train log on every step gives us per-step token counts
        eval={
            "hd": {"path": str(training_parquet), "type": "sequence", "every": {unit: cadence_n}}
        },
    )
    callback = FullRecordingCallback()
    trainer = Trainer(cfg, callbacks=[callback])
    trainer.train()

    # Reconstruct the per-step cumulative token counts from the train logs.
    tokens_by_step = {step: m["train/tokens"] for step, m in callback.train_logs}
    assert len(tokens_by_step) == _MAX_STEPS

    if unit == "steps":
        expected = _expected_step_cadence(cadence_n, _MAX_STEPS)
    else:
        expected = _expected_token_cadence(cadence_n, tokens_by_step, _MAX_STEPS)

    assert callback.eval_steps == expected
    assert callback.eval_steps  # non-vacuous

    # Every eval payload carries the hd namespace with finite values.
    for _, metrics in callback.evals:
        hd_values = [v for k, v in metrics.items() if k.startswith("eval/hd/")]
        assert hd_values
        assert all(math.isfinite(v) for v in hd_values)

    # The progress-bar eval loss was extracted and is finite.
    assert trainer._last_eval_loss is not None
    assert math.isfinite(trainer._last_eval_loss)


def test_two_eval_datasets_merge_per_cadence(
    training_parquet: Path,
    second_eval_parquet: Path,
    tmp_path: Path,
) -> None:
    """Two datasets on different step cadences fire independently and merge on shared steps."""
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=_MAX_STEPS,
        batch_size=_BATCH_SIZE,
        log_every=1,
        eval={
            "hd": {"path": str(training_parquet), "type": "sequence", "every": {"steps": 2}},
            "second": {
                "path": str(second_eval_parquet),
                "type": "sequence",
                "every": {"steps": 3},
            },
        },
    )
    callback = FullRecordingCallback()
    trainer = Trainer(cfg, callbacks=[callback])
    trainer.train()

    evals_by_step = {step: metrics for step, metrics in callback.evals}

    # hd at {2,4,6}, second at {3,6}; union of firing steps is {2,3,4,6}.
    assert sorted(evals_by_step) == [2, 3, 4, 6]

    def has_namespace(step: int, prefix: str) -> bool:
        return any(k.startswith(prefix) for k in evals_by_step[step])

    for step in (2, 4, 6):
        assert has_namespace(step, "eval/hd/")
    for step in (3, 6):
        assert has_namespace(step, "eval/second/")

    # Off-cadence isolation: each namespace appears only when its own schedule is due.
    assert not has_namespace(2, "eval/second/")
    assert not has_namespace(3, "eval/hd/")

    # The shared final step merges both namespaces into one payload.
    assert has_namespace(6, "eval/hd/")
    assert has_namespace(6, "eval/second/")

    # _extract_eval_loss averages the two */loss metrics into a finite value.
    assert trainer._last_eval_loss is not None
    assert math.isfinite(trainer._last_eval_loss)

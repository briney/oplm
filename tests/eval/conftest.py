"""Shared fixtures for the eval-harness test suite.

Provides a lightweight :class:`~oplm.eval.context.EvalContext` factory (``ctx``)
used by the pure scheduling/accounting tests, a register/cleanup fixture for a
throwaway dummy :class:`~oplm.eval.tasks.base.EvalTask` (so tests do not leak
into the shared registry), and a tiny-model factory for the slow task tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from oplm.eval import EvalContext, EvalTask

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from accelerate import Accelerator

    from oplm.model import OplmForMaskedLM


def make_eval_context(
    step: int = 0,
    *,
    epoch: int = 0,
    tokens_seen: int = 0,
    steps_delta: int = 1,
    tokens_delta: int = 0,
    epoch_delta: int = 0,
    is_final: bool = False,
) -> EvalContext:
    """Build an :class:`EvalContext` with cadence-test-friendly defaults.

    The first positional argument is ``global_step`` (the field most cases vary);
    everything else is keyword-only so call sites read clearly.
    """
    return EvalContext(
        global_step=step,
        epoch=epoch,
        tokens_seen=tokens_seen,
        steps_delta=steps_delta,
        tokens_delta=tokens_delta,
        epoch_delta=epoch_delta,
        is_final=is_final,
    )


@pytest.fixture
def ctx() -> Callable[..., EvalContext]:
    """Return the :func:`make_eval_context` factory for terse in-test construction."""
    return make_eval_context


class DummyScoreTask(EvalTask):
    """Minimal eval task that returns a fixed score without touching the model."""

    default_metrics = ["score"]

    def evaluate(self, model: OplmForMaskedLM, accelerator: Accelerator) -> dict[str, float]:
        """Return a constant metric dict; ``model`` / ``accelerator`` are unused."""
        return {"score": 1.0}


@pytest.fixture
def dummy_task_type() -> Iterator[str]:
    """Register :class:`DummyScoreTask` under a unique type and clean up afterward.

    Yields the registered type string. Mutating ``EVAL_TASK_REGISTRY`` directly
    (rather than via the ``@register_eval_task`` decorator) keeps the insertion
    idempotent across the session and guarantees removal even if the test fails.
    """
    from oplm.eval.registry import EVAL_TASK_REGISTRY

    type_name = "dummy_score"
    EVAL_TASK_REGISTRY[type_name] = DummyScoreTask
    try:
        yield type_name
    finally:
        EVAL_TASK_REGISTRY.pop(type_name, None)


@pytest.fixture
def make_model() -> Callable[..., OplmForMaskedLM]:
    """Return a factory building a tiny ``OplmForMaskedLM`` on CPU.

    ``OplmForMaskedLM.__init__`` takes the HuggingFace ``oplm.model.OplmConfig``
    (a ``PretrainedConfig``), not the ``oplm.config.ModelConfig`` dataclass — the
    two schemas diverge and there is no converter (see TODOS.md Phase 0). So the
    factory constructs the HF config directly with its native field names, keeping
    ``vocab_size`` at the tokenizer's 33-token default.
    """
    from oplm.model import OplmConfig as OplmModelConfig
    from oplm.model import OplmForMaskedLM

    def _build(
        max_position_embeddings: int = 64,
        *,
        hidden_size: int = 32,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 2,
    ) -> OplmForMaskedLM:
        return OplmForMaskedLM(
            OplmModelConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                max_position_embeddings=max_position_embeddings,
            )
        ).eval()

    return _build
